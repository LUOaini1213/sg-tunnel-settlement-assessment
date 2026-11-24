"""Generate README.md, DATA_SOURCES.md and docs/reference_sheet.md from results and data files.

Usage: python scripts/build_docs.py [--check]
With --check nothing is written; the exit code is 1 if a generated file differs from the one on disk.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tunnelscreen import building as bld  # noqa: E402


def load(name: str):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def read_csv(rel: str) -> list[dict]:
    with (ROOT / rel).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def fmt(x, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def pct(p: float, nd: int = 2) -> str:
    return f"{100 * p:.{nd}f}"


FINAL_NAMES = ("Negligible", "Very Slight", "Slight", "Moderate", "Severe", "Very Severe")


def cat_name(c, final: bool = False) -> str:
    """Category with its name. A Stage 2 category of 4 means "4 to 5"; a final category is a single number."""
    if c is None:
        return "not screened"
    return f"{c} {FINAL_NAMES[c]}" if final else f"{c} {bld.CATEGORY_NAMES[c]}"


def published(v) -> str:
    return ", ".join(f"{x:g}" for x in v) if isinstance(v, list) else f"{v:.6g}"


def face_text(nd: float) -> str:
    """Face position relative to an array, in words."""
    if nd < 0:
        return f"when the face is still {abs(nd):g} tunnel diameters short of the array"
    if nd == 0:
        return "when the face is under the array"
    return f"when the face is {nd:g} tunnel diameters past the array"


def context() -> dict:
    scn = json.loads((ROOT / "inputs" / "scenario.json").read_text(encoding="utf-8"))
    facts, sources = read_csv("data/facts.csv"), read_csv("data/sources.csv")
    ver, scr, unc = load("verification.json"), load("screening.json"), load("uncertainty.json")
    mon, back, ext, meta = load("monitoring.json"), load("backanalysis.json"), load("existing_tunnel.json"), load("meta.json")
    mut_path = ROOT / "results" / "mutation.json"
    mut = json.loads(mut_path.read_text(encoding="utf-8")) if mut_path.exists() else None
    design = scr["design"]
    by_id = {r["id"]: r for r in design}
    shallow = [r for r in design if r["stage2_applicable"]]
    a = scn["assumptions"]
    zones = {z["id"]: z for z in scn["ground_zones"]}
    c = {}
    c["n_facts"], c["n_sources"] = len(facts), len(sources)
    c["n_verified"] = sum(1 for f in facts if f["status"] == "VERIFIED")
    c["n_unverified"] = sum(1 for f in facts if f["status"] == "UNVERIFIED")
    c["access_date"] = sources[0]["access_date"]
    c["n_checks"] = len(ver["checks"])
    c["n_buildings"] = len(design)
    c["n_stage1"] = sum(1 for r in design if r["action"] == "None: negligible at Stage 1")
    c["n_low"] = sum(1 for r in design if r["action"] == "None: low risk at Stage 2")
    c["n_more"] = sum(1 for r in design if not r["action"].startswith("None"))
    c["n_stage3_shallow"] = sum(1 for r in shallow if r["action"].startswith("Stage 3"))
    c["n_mixed"] = sum(1 for r in design if r["foundation"] == "mixed")
    c["n_piled"] = sum(1 for r in design if r["foundation"] == "piled")
    top = max(shallow, key=lambda r: r["eps_max_pct"])
    c["max_eps"], c["max_eps_id"] = fmt(top["eps_max_pct"], 3), top["id"]
    c["vl_mean_kf"], c["vl_design_kf"] = zones["Z3"]["observed_vl"]["mean"], zones["Z3"]["design"]["VL"]
    mcb = unc["base"]["buildings"]
    p1 = {k: v["p_stage2_category_at_least"]["1"]["p"] for k, v in mcb.items() if "p_stage2_category_at_least" in v}
    worst = max(p1, key=p1.get)
    c["max_p_cat1"], c["max_p_cat1_id"] = pct(p1[worst], 1), worst
    ind = unc["independent_drives"]["buildings"]
    p1i = {k: v["p_stage2_category_at_least"]["1"]["p"] for k, v in ind.items() if "p_stage2_category_at_least" in v}
    worst_i = max(p1i, key=p1i.get)
    c["max_p_cat1_indep"], c["max_p_cat1_indep_id"] = pct(p1i[worst_i], 1), worst_i
    pz = [z["p_vl_exceeds_design"]["p"] for z in unc["base"]["zones"].values()]
    c["p_exceed_design_min"], c["p_exceed_design_max"] = pct(min(pz), 2), pct(max(pz), 2)
    t0 = unc["tornado"][0]
    vlbar = next(b for b in t0["bars"] if b["input"].startswith("Volume loss (5th"))
    c["tornado_id"], c["tornado_vl_low"], c["tornado_vl_high"] = t0["building"], fmt(vlbar["low_pct"], 3), fmt(vlbar["high_pct"], 3)
    c["et_design"], c["et_mean"] = fmt(ext["design"]["max_subsurface_settlement_mm"], 1), fmt(ext["mean"]["max_subsurface_settlement_mm"], 1)
    c["et_allow"] = f"{ext['design']['allowable_total_movement_mm']:g}"
    k = mon["counts"]
    c["n_d"], c["n_e"], c["n_e_points"], c["n_f"], c["n_bldg_points"] = k["type_d_points"], k["type_e_arrays"], k["type_e_points"], k["type_f_arrays"], k["building_points"]
    c["alert_factor"], c["min_susp"] = a["alert_factor"]["value"], f"{a['min_suspension_level_mm']['value']:g}"
    cb = back["cases"][1]
    e = cb["estimate"]
    c["ba_n"], c["ba_sigma"] = len(cb["readings_mm"]), f"{back['sigma_mm']:g}"
    c["ba_vl"], c["ba_vl_lo"], c["ba_vl_hi"] = fmt(e["VL"]), fmt(e["VL_ci95"][0]), fmt(e["VL_ci95"][1])
    c["ba_k"], c["ba_k_lo"], c["ba_k_hi"] = fmt(e["K"], 3), fmt(e["K_ci95"][0], 3), fmt(e["K_ci95"][1], 3)
    c["ba_true_vl"], c["ba_true_k"] = cb["true"]["VL"], cb["true"]["K"]
    c["ba_slope_fit"], c["ba_slope_design"] = f"{cb['max_slope_fitted_1_in']:.0f}", f"{cb['max_slope_design_1_in']:.0f}"
    cm = cb["centre_marker"]
    c["ba_first_alert_text"] = face_text(cm["first_review_state_at_D"])
    c["ba_first_proj_text"] = face_text(cm["first_projection_reaching_alert_at_D"])
    c["ba_lead"] = f"{cm['first_review_state_at_D'] - cm['first_projection_reaching_alert_at_D']:g}"
    ec = back["estimator_comparison"]
    tv = ec["true"]["VL"]
    rows = {r["estimator"]: r for r in ec["rows"]}
    nls = next(v for kk, v in rows.items() if kk.startswith("least squares"))
    l05 = next(v for kk, v in rows.items() if "0.05" in kk)
    l20 = next(v for kk, v in rows.items() if "0.2 " in kk)
    c["ll_bias_005"], c["ll_bias_02"] = f"{100 * (l05['mean_VL'] / tv - 1):.0f}", f"{100 * (l20['mean_VL'] / tv - 1):.0f}"
    c["nls_bias"] = f"{100 * (nls['mean_VL'] / tv - 1):+.1f}"
    c["ec_trials"] = ec["trials"]
    c["n_mc"] = f"{unc['base']['n']:,}"
    y = sorted(t["y0_m"] for t in scn["alignment"]["tunnels"])
    c["diameter"] = scn["alignment"]["excavated_diameter_m"]
    c["clear_sep"] = f"{(y[1] - y[0]) - c['diameter']:g}"
    mut = mut or {}
    c["n_tests"] = mut.get("tests_collected", "(not yet run)")
    for run in ("first", "final"):
        rec = mut.get(run, {})
        c[f"mut_{run}_killed"] = rec.get("killed", "(not yet run)")
        c[f"mut_{run}_total"] = rec.get("total", "(not yet run)")

    # ---- tables
    c["verification_table"] = table(
        ["ID", "Quantity", "Source", "Published", "This code", "Tolerance"],
        [[v["id"], v["what"], v["source"], published(v["published"]), published(v["computed"]), v["tolerance"]] for v in ver["checks"]])
    c["design_values"] = "; ".join(f"{z['formation']}: K {z['design']['K']}, V {z['design']['VL']}%" for z in scn["ground_zones"])
    c["screening_table"] = table(
        ["Rank", "Building", "Ground", "Foundation", "Max settlement (mm)", "Max slope", "Stage 1", "Max tensile strain (%)",
         "Governing case", "Stage 2 category", "Index sum", "Final category", "Next step"],
        [[r["rank"], f"{r['id']} {r['description']}", r["zone"], r["foundation"], fmt(r["max_settlement_mm"], 1), f"1:{r['max_slope_1_in']:.0f}",
          "negligible" if r["stage1_negligible"] else "go on", "-" if r["eps_max_pct"] is None else fmt(r["eps_max_pct"], 3),
          "-" if r["governing_case"] is None else f"{r['governing_case']} ({r['governing_mode']})",
          "-" if r["stage2_category"] is None else cat_name(r["stage2_category"]), r["sensitivity_index_sum"],
          "-" if r["final_category"] is None else cat_name(r["final_category"], final=True), r["action"]] for r in design])
    by_basis = {b: {r["id"]: r for r in scr[b]} for b in ("design", "characteristic", "mean")}
    c["basis_table"] = table(
        ["Building", "V design / characteristic / mean (%)", "Max settlement (mm)", "Max tensile strain (%)", "Stage 2 category"],
        [[r["id"], " / ".join(f"{by_basis[b][r['id']]['VL_pct']:g}" for b in by_basis),
          " / ".join(fmt(by_basis[b][r["id"]]["max_settlement_mm"], 1) for b in by_basis),
          " / ".join(fmt(by_basis[b][r["id"]]["eps_max_pct"], 3) for b in by_basis),
          " / ".join(str(by_basis[b][r["id"]]["stage2_category"]) for b in by_basis)] for r in sorted(shallow, key=lambda r: r["id"])])

    def pcell(p):
        return f"{pct(p['p'])} ({pct(p['lo95'])} to {pct(p['hi95'])})"

    c["mc_table"] = table(
        ["Building", "Zone", "Settlement 5th / 50th / 95th percentile (mm)", "Not negligible at Stage 1 (%)",
         "Stage 2 category 1 or higher (%)", "Stage 2 category 2 or higher (%)", "Final category 2 or higher (%)"],
        [[bid, r["zone"], " / ".join(fmt(r["settlement_mm"][q], 1) for q in ("p05", "p50", "p95")), pcell(r["p_stage1_not_negligible"]),
          pcell(r["p_stage2_category_at_least"]["1"]) if "p_stage2_category_at_least" in r else "not screened",
          pcell(r["p_stage2_category_at_least"]["2"]) if "p_stage2_category_at_least" in r else "not screened",
          pcell(r["p_final_category_at_least"]["2"]) if "p_final_category_at_least" in r else "not screened"]
         for bid, r in sorted(mcb.items())])
    c["tornado_table"] = "\n\n".join(
        f"Building {t['building']}: base maximum tensile strain {fmt(t['base_eps_max_pct'], 4)}% at V {fmt(t['base_VL_pct'])}%, K {t['base_K']:g}.\n\n"
        + table(["Input", "Range", "Strain at low end (%)", "Strain at high end (%)", "Swing (%)"],
                [[b["input"], b["range"], fmt(b["low_pct"], 4), fmt(b["high_pct"], 4), fmt(b["swing_pct"], 4)] for b in t["bars"]])
        for t in unc["tornado"][:1])
    bp = {b["id"]: b for b in mon["buildings"]}
    c["monitoring_table"] = table(
        ["Item", "Number", "Basis"],
        [["Type D ground settlement points", k["type_d_points"], "CDC Cl. 13.4.2.1 a): every 25 m of each tunnel"],
         ["Type E arrays (lines of settlement points across the tunnels)", f"{k['type_e_arrays']} arrays, {k['type_e_points']} points",
          "CDC Cl. 13.4.2.1 b): every 200 m; Section 13.4.9: about 25 m and 50 m before each building in use with a tunnel under it"],
         ["Type F arrays (with subsurface instruments)", k["type_f_arrays"], "CDC Cl. 13.4.2.1 c): every 1000 m"],
         ["Building settlement points", k["building_points"], "CDC Cl. 13.4.7.1; points at facade ends, predicted inflexion points and the deepest point (ASSUMPTION)"],
         ["Buildings monitored in real time", f"{k['buildings_real_time']} ({', '.join(b for b in sorted(bp) if bp[b]['real_time'])})",
          "CDC Section 13.4.9: in use, tunnel directly underneath"],
         ["Buildings with tensile strain monitoring", k["buildings_tensile_strain"], "CDC Cl. 13.4.7.2: predicted tensile strain above 0.1%"]])
    brows = []
    for cs in back["cases"]:
        es, cmk = cs["estimate"], cs["centre_marker"]
        brows.append([
            f"{cs['id']}: {cs['title']}", f"{cs['true']['VL']:g}% / {cs['true']['K']:g}",
            f"{fmt(es['VL'])}% ({fmt(es['VL_ci95'][0])} to {fmt(es['VL_ci95'][1])})", f"{fmt(es['K'], 3)} ({fmt(es['K_ci95'][0], 3)} to {fmt(es['K_ci95'][1], 3)})",
            f"{fmt(cs['bootstrap_ci95']['VL'][0])} to {fmt(cs['bootstrap_ci95']['VL'][1])}",
            f"{cs['design']['VL']:g}% / {cs['design']['K']:g}",
            ("flagged" if cs["departure_from_expected"]["flagged"] else "not flagged") + f" (largest z {fmt(cs['departure_from_expected']['max_abs_z'], 1)})",
            f"{cs['review_states']['normal']} normal, {cs['review_states']['alert']} alert, {cs['review_states']['work_suspension']} work suspension",
            f"1:{cs['max_slope_fitted_1_in']:.0f} against 1:{cs['max_slope_design_1_in']:.0f}"])
    c["backanalysis_table"] = table(
        ["Case (SYNTHETIC)", "True V / K", "Estimated V (95% interval)", "Estimated K (95% interval)", "Bootstrap 95% interval for V (%)",
         "Design V / K", "Departure from expected", "Review state of the markers", "Max slope, fitted against design"], brows)
    c["estimator_table"] = table(["Estimator", "Mean V (%)", "Standard deviation of V (%)", "Mean K", "Standard deviation of K"],
                                 [[r["estimator"], fmt(r["mean_VL"], 3), fmt(r["sd_VL"], 3), fmt(r["mean_K"], 3), fmt(r["sd_K"], 3)] for r in ec["rows"]])

    prov = [
        ["Trough equations, horizontal movement, beam equations, damage categories, Stage 1 limits", "REAL", "S1, S3, S4, S9 (see docs/formula_audit.md)"],
        ["Design K and V by ground type: " + c["design_values"], "REAL", "S1 Table 14.1"],
        ["Observed volume loss by ground type (mean, 95% characteristic value, upper bound) and best-fit K", "REAL", "S5 Table 2"],
        ["Array types and spacings, tensile strain monitoring threshold 0.1%, definitions of the review levels", "REAL", "S1 Chapter 13"],
        ["Allowable total movement of an existing bored tunnel, 15 mm", "REAL", "S2 Table 2(b)"],
        ["E/G 2.6 (masonry) and 12.5 (framed), Poisson's ratio 0.3", "REAL", "S3 sections 4.2, 4.3 and 4.5"],
        ["Lognormal parameters of volume loss", "DERIVED", "From the S5 mean and 95% value for each ground type"],
        ["Review levels of each instrument", "DERIVED", "Design-basis prediction and the alert factor"],
        ["Excavated diameter 6.6 m", "ILLUSTRATIVE", scn["alignment"]["diameter_basis"].replace("ILLUSTRATIVE: ", "")],
        ["Alignment (900 m, twin tunnels 15 m apart, axis 20 to 26 m deep), three ground zones along it", "SYNTHETIC", "inputs/scenario.json"],
        ["Thirteen buildings: position, size, structure, foundation, sensitivity indices", "SYNTHETIC", "inputs/buildings.csv"],
        ["Existing tunnel crossing above the drive", "SYNTHETIC", "inputs/scenario.json"],
        ["Settlement-marker readings in the back-analysis", "SYNTHETIC", "Generated from the Gaussian trough plus noise, seed fixed"],
    ]
    labels = {"poisson_ratio": "Poisson's ratio in Eq. 10", "alert_factor": "Alert level as a fraction of the work suspension level",
              "min_suspension_level_mm": "Smallest work suspension level (mm)", "heave_alert_mm": "Heave alert level (mm)",
              "heave_suspension_mm": "Heave work suspension level (mm)", "marker_noise_sd_mm": "Noise of one marker reading, standard deviation (mm)",
              "vl_distribution": "Distribution of volume loss", "k_distribution": "Distribution of K",
              "vl_correlation_between_drives": "Correlation of volume loss between the two drives",
              "depth_tolerance_m": "Range of axis depth in the sensitivity study (m)",
              "second_tunnel_factor_range": "Range of the volume-loss factor on the second drive",
              "monte_carlo_samples": "Monte Carlo samples", "random_seed": "Random seed",
              "stabilised_distance_diameters": "Distance behind the face at which settlement is taken as final (diameters)"}
    for key, text in labels.items():
        item = a[key]
        if key == "poisson_ratio":
            continue
        val = item["value"]
        val = " to ".join(f"{v:g}" for v in val) if isinstance(val, list) else val
        prov.append([f"{text}: {val}", item["label"], item["basis"]])
    prov.append(["K ranges for sampling: " + "; ".join(f"{z['id']} {z['k_range'][0]:g} to {z['k_range'][1]:g}" for z in scn["ground_zones"]),
                 "ASSUMPTION", "Chosen to bracket the S1 design value and the S5 best-fit value for each zone"])
    prov.append(["Thresholds of the departure test (3 standard deviations; chi-square p below 0.01), 2 m merge distance of array points, "
                 "0.2 Smax cut-off of the straight-line estimator, placement rule for building markers, ranking order for instrumentation",
                 "ASSUMPTION", "Stated in the code where each is used"])
    prov.append(["Scaling of the subsurface trough width for a surface K other than 0.5", "ASSUMPTION", "The published relation (S4 Eq. 8) is for clay with K = 0.5 at the surface"])
    c["provenance_table"] = table(["Input", "Label", "Source or basis"], prov)
    return c


def render(template: str, ctx: dict) -> str:
    def sub(m):
        key = m.group(1)
        if key not in ctx:
            raise KeyError(f"no value for placeholder {key}")
        return str(ctx[key])

    out = re.sub(r"\{\{\s*([a-z0-9_]+)\s*\}\}", sub, template)
    return out.rstrip("\n") + "\n"


def data_sources() -> str:
    sources = read_csv("data/sources.csv")
    head = ("# Data sources\n\n> Personal portfolio study built with AI assistance; not affiliated with or endorsed by LTA. Student-level\n"
            "> screening study, not a design. This page lists every external document used, when it was opened and on what terms.\n\n"
            "<!-- Generated by scripts/build_docs.py from data/sources.csv. -->\n\n")
    body = table(["ID", "Document", "Publisher or host", "URL", "Accessed", "Terms and how it is used"],
                 [[s["source_id"], s["citation"], s["publisher"], f"<{s['url']}>", s["access_date"], s["terms"]] for s in sources])
    tail = ("\n\n## How the sources are handled\n\n"
            "- Each document was opened over plain HTTP from the URL above on the access date and read as text; scanned pages and figures\n"
            "  were read as page images where the table says so.\n"
            "- No paper, standard or figure is committed to this repository. Statements are paraphrased, with short quotations in\n"
            "  `data/facts.csv` so that each can be checked against the source.\n"
            "- Quotations from scanned documents were matched after removing spaces and punctuation, because the text layer of a scan\n"
            "  splits and joins words unpredictably. Quotations are otherwise as printed.\n"
            "- Numbers in `inputs/scenario.json` that come from a source carry the source ID and table or clause.\n"
            "- Primary papers that could not be opened are listed as UNVERIFIED in `data/facts.csv`; their content is used only as\n"
            "  reported by a document that was opened.\n")
    return head + body + tail


def reference_sheet() -> str:
    facts = read_csv("data/facts.csv")
    n_v = sum(1 for f in facts if f["status"] == "VERIFIED")
    out = ["# Sourced reference sheet", "",
           "> Personal portfolio study built with AI assistance; not affiliated with or endorsed by LTA. Student-level screening study,",
           "> not a design. Each row is one statement, where it was found, and the words in the source that support it.", "",
           "<!-- Generated by scripts/build_docs.py from data/facts.csv. -->", "",
           f"{len(facts)} statements: {n_v} VERIFIED, {len(facts) - n_v} UNVERIFIED. VERIFIED means the quotation was matched against the text",
           "of the document opened on the access date. Where a value was read from a figure or a typeset equation, the locator says so and",
           "the quotation is the caption or the sentence beside it. Source IDs are listed in [DATA_SOURCES.md](../DATA_SOURCES.md).", ""]
    topics = []
    for f in facts:
        if f["topic"] not in topics:
            topics.append(f["topic"])
    for topic in topics:
        rows = [f for f in facts if f["topic"] == topic]
        out.append(f"## {topic}")
        out.append("")
        if topic == "Primary source not opened":
            out.append(table(["ID", "Statement", "Status"], [[f["id"], f["claim"], f["status"]] for f in rows]))
        else:
            out.append(table(["ID", "Statement", "Source", "Where", "Supporting quotation", "Status"],
                             [[f["id"], f["claim"], f["source_id"], f["locator"], "\"" + f["supporting_quote"].replace("|", "/") + "\"", f["status"]] for f in rows]))
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def outputs() -> dict[Path, str]:
    template = (ROOT / "docs" / "templates" / "README.md").read_text(encoding="utf-8")
    return {ROOT / "README.md": render(template, context()), ROOT / "DATA_SOURCES.md": data_sources(),
            ROOT / "docs" / "reference_sheet.md": reference_sheet()}


def main() -> int:
    check = "--check" in sys.argv
    stale = []
    for path, text in outputs().items():
        if check:
            if not path.exists() or path.read_text(encoding="utf-8").replace("\r\n", "\n") != text:
                stale.append(path.relative_to(ROOT).as_posix())
        else:
            path.write_text(text, encoding="utf-8", newline="\n")
    if check and stale:
        print("stale generated documents: " + ", ".join(stale))
        return 1
    print("documents are up to date" if check else "documents written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
