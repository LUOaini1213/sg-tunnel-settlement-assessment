"""Drift tests: the committed outputs must be what the code produces from the committed inputs.

They are marked ``golden`` and left out of mutation testing, where a kill has to come from a test
with an independent expectation (a published value, a hand calculation, a closed form). Floats are
compared to 1e-5 relative so that a rerun on another platform does not fail on the last digit.
"""
import csv
import json
import sys
from pathlib import Path

import pytest

from tunnelscreen import pipeline
from tunnelscreen import uncertainty as un

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

pytestmark = pytest.mark.golden


def committed(name: str):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def as_written(obj, nd: int = 6):
    """The object as it would be written to a results file."""
    return json.loads(pipeline.to_json(pipeline._round(obj, nd)))


def assert_same(a, b, where: str = "") -> None:
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), where
        for k in a:
            assert_same(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), where
        for k, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{where}[{k}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert a == pytest.approx(b, rel=1e-5, abs=1e-12), where
    else:
        assert a == b, where


@pytest.fixture(scope="module")
def inputs(scn, buildings):
    return scn, buildings


@pytest.fixture(scope="module")
def core(inputs):
    return pipeline.core_results(*inputs)


def test_generated_documents_are_up_to_date():
    import build_docs

    for path, text in build_docs.outputs().items():
        assert path.read_text(encoding="utf-8").replace("\r\n", "\n") == text, f"{path.name} is stale: run scripts/build_docs.py"


def test_technical_note_matches_a_fresh_build(tmp_path):
    import build_note

    fresh = tmp_path / "note.pdf"
    assert build_note.build(fresh) == 4
    assert build_note.page_texts(fresh) == build_note.page_texts(ROOT / "docs" / "technical_note.pdf"), "run scripts/build_note.py"


@pytest.mark.parametrize("name,key,nd", [("verification.json", "verification", 8), ("screening.json", "screening", 6),
                                         ("existing_tunnel.json", "existing", 6), ("monitoring.json", "plan", 6),
                                         ("backanalysis.json", "back", 6)])
def test_results_match_a_fresh_run(core, name, key, nd):
    assert_same(committed(name), as_written(core[key], nd), name)


def test_meta_matches_the_inputs(scn):
    meta = committed("meta.json")
    with (ROOT / "data" / "facts.csv").open(encoding="utf-8", newline="") as f:
        facts = list(csv.DictReader(f))
    assert meta["facts"]["total"] == len(facts)
    assert meta["facts"]["verified"] == sum(1 for r in facts if r["status"] == "VERIFIED")
    assert meta["facts"]["unverified"] == sum(1 for r in facts if r["status"] == "UNVERIFIED")
    assert (meta["seed"], meta["workbook_building"]) == (scn["assumptions"]["random_seed"]["value"], pipeline.WORKBOOK_BUILDING)


def test_csv_tables_match_a_fresh_run(core):
    for name, rows in (("instruments.csv", pipeline.instrument_rows(core["instruments"])),
                       ("screening_design_basis.csv", pipeline.screening_rows(core["screening"]["design"]))):
        with (ROOT / "results" / name).open(encoding="utf-8", newline="") as f:
            assert list(csv.reader(f)) == [[str(c) for c in r] for r in rows], name


def test_monte_carlo_results_match_a_fresh_run(inputs):
    """The full sample count and seed of the scenario; the zones, the buildings that are not
    screened at Stage 2, and one shallow building are recomputed (the rest takes minutes)."""
    scn, buildings = inputs
    a = scn["assumptions"]
    n, seed = a["monte_carlo_samples"]["value"], a["random_seed"]["value"]
    stored = committed("uncertainty.json")
    assert committed("meta.json")["monte_carlo_samples"] == n and stored["base"]["n"] == n == stored["independent_drives"]["n"]
    some = [b for b in buildings if b.foundation != "shallow" or b.id == "B09"]
    for key, corr in (("base", a["vl_correlation_between_drives"]["value"]), ("independent_drives", 0.0)):
        fresh = as_written(un.monte_carlo(scn, some, n, seed, correlation=corr))
        assert_same(stored[key]["zones"], fresh["zones"], f"{key}.zones")
        for b in some:
            assert_same(stored[key]["buildings"][b.id], fresh["buildings"][b.id], f"{key}.{b.id}")


def test_tornado_matches_a_fresh_run(inputs, core):
    scn, buildings = inputs
    fresh = as_written(pipeline.tornado_results(scn, buildings, core["screening"]["design"]))
    assert_same(committed("uncertainty.json")["tornado"], fresh, "tornado")
