"""The Excel workbook is recalculated from its formulas and compared with the Python engine."""
import math
from pathlib import Path

import numpy as np
import openpyxl
import pytest

from tunnelscreen import building as bld
from tunnelscreen import greenfield as gf
from tunnelscreen import pipeline
from tunnelscreen import screening as sc
from tunnelscreen import workbook as wbk

PATH = pipeline.WORKBOOK_PATH


@pytest.fixture(scope="module")
def book():
    return openpyxl.load_workbook(PATH)  # formulas as text


@pytest.fixture(scope="module")
def calc():
    """Every cell value after recalculating the workbook with the `formulas` package."""
    import formulas

    sol = formulas.ExcelModel().loads(str(PATH)).finish().calculate()
    values = {}
    for key, rng in sol.items():
        sheet, _, cell = key.rpartition("!")
        sheet = sheet.strip("'").split("]")[-1].upper()
        v = rng.value
        if ":" in cell:
            continue
        values[(sheet, cell)] = v[0, 0] if hasattr(v, "shape") else v
    return lambda sheet, cell: values[(sheet.upper(), cell)]


def _inputs(book):
    ws, s2 = book["Inputs"], book["Stage2"]
    return dict(D=ws["B4"].value, z0=ws["B5"].value, K=ws["B6"].value, VL=ws["B7"].value, y1=ws["B8"].value, y2=ws["B9"].value,
                factor=ws["B10"].value, a=s2["B4"].value, b=s2["B5"].value, H=s2["B6"].value, eg=s2["B7"].value, nu=s2["B8"].value)


def test_workbook_inputs_are_the_scenario_values(book, scn, by_id):
    i = _inputs(book)
    b = by_id[pipeline.WORKBOOK_BUILDING]
    ts = sc.tunnels_at(scn, b.chainage)
    assert (i["D"], i["z0"], i["K"], i["VL"], i["y1"], i["y2"]) == (ts[0].D, ts[0].z0, ts[0].K, ts[0].VL, ts[0].y0, ts[1].y0)
    assert (i["a"], i["b"], i["H"], i["eg"], i["nu"]) == (b.y_start, b.y_end, b.height, b.e_over_g, 0.3)


def test_every_result_cell_is_a_formula(book):
    ws = book["Inputs"]
    assert all(str(ws[f"B{r}"].value).startswith("=") for r in range(13, 19))
    wt = book["Trough"]
    for row in wt.iter_rows(min_row=2, max_row=len(wbk.TROUGH_Y) + 1, min_col=2, max_col=9):
        assert all(str(c.value).startswith("=") for c in row)
    s2 = book["Stage2"]
    for col in "BCD":
        assert all(str(s2[f"{col}{r}"].value).startswith("=") for r in range(13, 29))
    assert all(str(s2[f"H{r}"].value).startswith("=") for r in (5, 6, 7, 8))
    assert "not a design" in book["About"]["A2"].value.lower() and "SYNTHETIC" in book["Inputs"]["A2"].value


def test_derived_inputs(book, calc):
    i = _inputs(book)
    assert calc("Inputs", "B13") == pytest.approx(gf.trough_width(i["K"], i["z0"]), rel=1e-12)
    assert calc("Inputs", "B14") == pytest.approx(gf.trough_volume(i["VL"], i["D"]), rel=1e-12)
    assert calc("Inputs", "B15") == pytest.approx(gf.s_max(i["VL"], i["D"], i["K"], i["z0"]), rel=1e-12)
    assert calc("Inputs", "B16") == pytest.approx(calc("Inputs", "B17"), rel=0.011)  # LTA rounded coefficient


def test_trough_sheet_matches_python(book, calc):
    i = _inputs(book)
    ts = [gf.Tunnel(i["y1"], i["z0"], i["D"], i["VL"], i["K"]), gf.Tunnel(i["y2"], i["z0"], i["D"], i["VL"] * i["factor"], i["K"])]
    ys = np.array(wbk.TROUGH_Y)
    expect = {
        "B": gf.settlement_one(ys, ts[0]) * 1000, "C": gf.settlement_one(ys, ts[1]) * 1000, "D": gf.settlement(ys, ts) * 1000,
        "F": gf.slope(ys, ts), "G": gf.curvature(ys, ts), "H": gf.horizontal_displacement(ys, ts) * 1000,
        "I": gf.horizontal_strain(ys, ts) * 100,
    }
    for col, arr in expect.items():
        got = np.array([calc("Trough", f"{col}{r}") for r in range(2, len(ys) + 2)], dtype=float)
        assert np.allclose(got, arr, rtol=1e-9, atol=1e-15), col
    assert calc("Trough", "L1") == pytest.approx(float(expect["D"].max()), rel=1e-12)
    # trapezium-rule volume on the sheet against the two volume losses
    assert calc("Trough", "L2") == pytest.approx(calc("Trough", "L3"), rel=1e-4)
    assert calc("Trough", "L3") == pytest.approx(gf.trough_volume(i["VL"], i["D"]) * (1 + i["factor"]), rel=1e-12)


def test_stage2_sheet_matches_python(book, calc):
    i = _inputs(book)
    t1 = gf.Tunnel(i["y1"], i["z0"], i["D"], i["VL"], i["K"])
    r = bld.assess_facade(sc.transverse_profile([t1]), i["a"], i["b"], i["H"], i["eg"], n_grid=wbk.N_GRID, nu=i["nu"])
    spans = {col: calc("Stage2", f"{col}15") for col in "BCD"}
    present = [col for col in "BCD" if spans[col] > 0]
    assert len(present) == len(r.segments) >= 1
    for col, sg in zip(present, r.segments):
        assert sg.mode == ("sagging" if col == "C" else "hogging")
        for row, value in ((13, sg.a), (14, sg.b), (15, sg.L), (16, sg.delta), (17, sg.deflection_ratio), (20, sg.eps_h),
                           (25, sg.eps_b), (26, sg.eps_d), (27, sg.eps_bt), (28, sg.eps_dt)):
            assert calc("Stage2", f"{col}{row}") == pytest.approx(value, rel=1e-6, abs=1e-15), (col, row)
        assert calc("Stage2", f"{col}23") == pytest.approx(bld.bending_factor(sg.L, i["H"], i["eg"], sg.mode), rel=1e-9)
        assert calc("Stage2", f"{col}24") == pytest.approx(bld.diagonal_factor(sg.L, i["H"], i["eg"], sg.mode), rel=1e-9)
    assert calc("Stage2", "H5") == pytest.approx(r.eps_max, rel=1e-6)
    assert calc("Stage2", "H6") == pytest.approx(r.eps_max * 100, rel=1e-6)
    assert int(calc("Stage2", "H7")) == r.category
    assert calc("Stage2", "H8") == bld.CATEGORY_NAMES[r.category]


def test_stage2_result_matches_the_pipeline_result(calc, scn, by_id):
    b = by_id[pipeline.WORKBOOK_BUILDING]
    cases = sc.stage2_cases(b, sc.tunnels_at(scn, b.chainage), include_longitudinal=False, n_grid=wbk.N_GRID)
    assert calc("Stage2", "H5") == pytest.approx(cases["transverse, first drive only"][0], rel=1e-6)


def test_committed_workbook_is_what_the_generator_writes(book, tmp_path, scn, by_id):
    b = by_id[pipeline.WORKBOOK_BUILDING]
    ts = sc.tunnels_at(scn, b.chainage)
    fresh_path = wbk.build(tmp_path / "fresh.xlsx", D=ts[0].D, z0=ts[0].z0, K=ts[0].K, VL=ts[0].VL, y0_first=ts[0].y0,
                           y0_second=ts[1].y0, second_factor=scn["alignment"]["second_tunnel_vl_factor"], facade_start=b.y_start,
                           facade_end=b.y_end, H=b.height, e_over_g=b.e_over_g, nu=scn["assumptions"]["poisson_ratio"]["value"],
                           building_label=f"building {b.id} (SYNTHETIC)")
    fresh = openpyxl.load_workbook(fresh_path)
    assert fresh.sheetnames == book.sheetnames == ["Inputs", "Trough", "Stage2", "About"]
    for name in book.sheetnames:
        a = {c.coordinate: c.value for row in book[name].iter_rows() for c in row if c.value is not None}
        f = {c.coordinate: c.value for row in fresh[name].iter_rows() for c in row if c.value is not None}
        assert a == f, name


def test_excel_minus_before_power_is_bracketed(book):
    """In Excel -x^2 means (-x)^2, so the exponent of the Gaussian must be written -((x)^2)."""
    import re

    bad = re.compile(r"EXP\(-\([^()]+\)\^")  # a minus applied straight to a bracket that is then squared
    n = 0
    for name in ("Trough", "Stage2"):
        for row in book[name].iter_rows():
            for c in row:
                if isinstance(c.value, str) and "EXP(" in c.value:
                    n += 1
                    assert not bad.search(c.value), c.coordinate
                    assert "EXP(-((" in c.value, c.coordinate
    assert n > 300
    assert bad.search("=B1*EXP(-(A2-B8)^2/(2*B13^2))")  # the faulty form is recognised
