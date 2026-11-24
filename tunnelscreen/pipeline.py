"""Run the whole study offline from the committed inputs and write results/.

Usage: python -m tunnelscreen.pipeline [--samples N] [--no-figures]
Everything is deterministic: the random seeds are inputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from pathlib import Path

import numpy as np

from . import backanalysis as ba
from . import building as bld
from . import greenfield as gf
from . import monitoring as mon
from . import screening as sc
from . import uncertainty as un
from . import workbook as wbk

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
WORKBOOK_PATH = ROOT / "workbook" / "trough_and_stage2.xlsx"
WORKBOOK_BUILDING = "B07"


def to_json(obj) -> str:
    """The exact text written to a results file."""
    return json.dumps(obj, indent=1, sort_keys=True, default=_default) + "\n"


def _dump(name: str, obj, directory: Path = RESULTS) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(to_json(obj), encoding="utf-8", newline="\n")


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _round(obj, nd: int = 6):
    """Round floats so that results are stable across platforms and readable in diffs."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(f"{obj:.{nd}g}")
    if isinstance(obj, (np.floating,)):
        return _round(float(obj), nd)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, nd) for v in obj]
    return obj


# ----------------------------------------------------------------------------- verification
def verification() -> dict:
    """Published values and closed-form identities reproduced by the code (see tests/test_published.py)."""
    t = gf.Tunnel(0.0, 20.0, 6.0, 1.5, 0.5)
    i = gf.trough_width(t.K, t.z0)
    smax = gf.s_max(t.VL, t.D, t.K, t.z0)
    ys = np.linspace(-12 * i, 12 * i, 200001)
    area = float(np.trapezoid(gf.settlement_one(ys, t), ys))
    hog = [min(lim * bld.bending_factor(1, 1, 2.6, "hogging"), lim * bld.diagonal_factor(1, 1, 2.6, "hogging"))
           for lim in (0.05, 0.075, 0.15, 0.3)]
    return {
        "example_tunnel": {"z0_m": t.z0, "D_m": t.D, "VL_pct": t.VL, "K": t.K},
        "checks": [
            {"id": "V1", "what": "LTA coefficient: Smax i / (V D^2) with V in %", "published": 0.0031, "source": "S1 Cl. 14.2.1.2",
             "computed": smax * i / (t.VL * t.D ** 2), "tolerance": "1.1% (the published coefficient is rounded to two figures)"},
            {"id": "V2", "what": "Mair et al. coefficient: Smax K z0 / (Vl D^2) with Vl as a fraction", "published": 0.31, "source": "S3 Eq. 5",
             "computed": smax * i / ((t.VL / 100) * t.D ** 2), "tolerance": "1.1% (rounded to two figures)"},
            {"id": "V3", "what": "Trough volume / (i Smax)", "published": 2.5, "source": "S9 Fig. 8",
             "computed": area / (i * smax), "tolerance": "0.3% (published as 2.5; exact value sqrt(2 pi))"},
            {"id": "V4", "what": "Numerical trough volume / volume loss", "published": 1.0, "source": "S3 Eq. 3-4",
             "computed": area / gf.trough_volume(t.VL, t.D), "tolerance": "1e-9"},
            {"id": "V5", "what": "Settlement at the point of inflexion / Smax", "published": 0.61, "source": "S9 section 1.4.6",
             "computed": float(gf.settlement_one(i, t)) / smax, "tolerance": "0.005"},
            {"id": "V6", "what": "Settlement at the point of maximum hogging curvature (sqrt(3) i) / Smax", "published": 0.22, "source": "S9 Fig. 8",
             "computed": float(gf.settlement_one(math.sqrt(3) * i, t)) / smax, "tolerance": "0.005"},
            {"id": "V7", "what": "Maximum horizontal movement / (K Smax)", "published": 0.61, "source": "S4 section 5.5",
             "computed": float(np.max(np.abs(gf.horizontal_displacement_one(ys, t)))) / (t.K * smax), "tolerance": "0.005"},
            {"id": "V8", "what": "Settlement above the face / Smax", "published": 0.5, "source": "S3 section 2.3",
             "computed": float(gf.settlement_advancing(0.0, 0.0, t, 0.0)) / smax, "tolerance": "1e-12"},
            {"id": "V9", "what": "Longitudinal displacement at the face on the centre line x 8 z0 / (VL D^2)", "published": 1.0, "source": "S6 Eq. 2.17",
             "computed": abs(float(gf.long_displacement_advancing(0.0, 0.0, t, 0.0))) * 8 * t.z0 / ((t.VL / 100) * t.D ** 2), "tolerance": "1e-12"},
            {"id": "V10", "what": "Surface K from the subsurface relation at z = 0", "published": 0.5, "source": "S4 Eq. 8",
             "computed": gf.subsurface_trough_width(0.0, t.z0) / t.z0, "tolerance": "1e-12"},
            {"id": "V11", "what": "Damage chart, L/H = 1 hogging: Delta/L (%) on the category boundaries at zero horizontal strain",
             "published": [0.05, 0.08, 0.16, 0.32], "source": "S3 Fig. 4 (values read from the chart)", "computed": hog,
             "tolerance": "0.01 (chart-reading accuracy)"},
            {"id": "V12", "what": "Diagonal strain for eps_h = eps_d = 0.1%: 0.35 x 0.1 + sqrt(0.065^2 + 0.1^2)", "published": 0.035 + math.hypot(0.065, 0.1),
             "source": "S3 Eq. 10", "computed": bld.total_diagonal_strain(0.001, 0.001) * 100, "tolerance": "1e-12"},
            {"id": "V13", "what": "Characteristic volume loss from mean 0.30% and standard deviation 0.24% (%)", "published": 0.7,
             "source": "S5 section 4.5.1", "computed": un.characteristic_value(0.30, 0.24), "tolerance": "0.05 (published to one decimal)"},
        ],
    }


# ----------------------------------------------------------------------------- back-analysis demo
def backanalysis_demo(scn: dict, plan: dict) -> dict:
    a = scn["assumptions"]
    sigma = a["marker_noise_sd_mm"]["value"]
    rng = np.random.default_rng(a["random_seed"]["value"] + 1)
    D = scn["alignment"]["excavated_diameter_m"]
    y_first = scn["alignment"]["tunnels"][0]["y0_m"]
    cases = [
        {"id": "A", "title": "Old Alluvium array, drive performing as experience suggests", "chainage": 100.0, "true_VL": 0.30, "true_K": 0.48},
        {"id": "B", "title": "Kallang Formation array, larger volume loss and a narrower trough than assumed", "chainage": 700.0, "true_VL": 2.2, "true_K": 0.42},
    ]
    out = {"sigma_mm": sigma, "label": "SYNTHETIC readings: Gaussian trough of the first drive plus normal noise", "cases": []}
    for c in cases:
        ch = c["chainage"]
        zone = sc.zone_at(scn, ch)
        z0 = sc.axis_depth(scn, ch)
        ys_abs = np.array(mon.type_e_offsets(z0, [t["y0_m"] for t in scn["alignment"]["tunnels"]]))
        y = ys_abs - y_first  # offsets from the centre line of the first drive
        readings = ba.synthetic_readings(y, c["true_VL"], c["true_K"], z0, D, sigma, rng)
        fit = ba.fit_gaussian(y, readings, z0, D, sigma_mm=sigma)
        ci = fit.interval()
        boot = []
        for _ in range(400):
            rb = ba.model_mm(y, fit.VL, fit.K, z0, D) + rng.normal(0.0, sigma, y.size)
            fb = ba.fit_gaussian(y, rb, z0, D, sigma_mm=sigma, start=(fit.VL, fit.K))
            boot.append((fb.VL, fb.K))
        boot = np.array(boot)
        try:
            ll = ba.fit_loglinear(y, readings, z0, D)
        except ValueError as exc:  # too few usable points
            ll = {"error": str(exc)}
        design_vl, design_k = zone["design"]["VL"], zone["design"]["K"]
        exp_vl, exp_k = zone["observed_vl"]["mean"], zone["observed_k_best_fit"]
        pred_design = ba.model_mm(y, design_vl, design_k, z0, D)
        pred_expected = ba.model_mm(y, exp_vl, exp_k, z0, D)
        dep_expected = ba.departure_test(readings, pred_expected, sigma)
        lv = [mon.review_levels(float(p), a["alert_factor"]["value"], a["min_suspension_level_mm"]["value"]) for p in pred_design]
        states = [mon.classify_reading(float(r), l["alert_mm"], l["suspension_mm"], a["heave_alert_mm"]["value"],
                                       a["heave_suspension_mm"]["value"]) for r, l in zip(readings, lv)]
        t_fit = gf.Tunnel(0.0, z0, D, fit.VL, fit.K)
        t_des = gf.Tunnel(0.0, z0, D, design_vl, design_k)
        # sequential readings of the centre-line marker as the face advances
        i_true = c["true_K"] * z0
        seq = []
        lv0 = mon.review_levels(float(ba.model_mm(np.array([0.0]), design_vl, design_k, z0, D)[0]),
                                a["alert_factor"]["value"], a["min_suspension_level_mm"]["value"])
        first_alert = first_susp = first_projected = None
        for nd in (-3, -2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 5):
            xf = ch + nd * D
            true = float(ba.model_mm(np.array([0.0]), c["true_VL"], c["true_K"], z0, D)[0]) * float(gf.face_factor(ch, xf, i_true))
            r = true + float(rng.normal(0.0, sigma))
            stage_pred = float(ba.model_mm(np.array([0.0]), exp_vl, exp_k, z0, D)[0]) * float(gf.face_factor(ch, xf, exp_k * z0))
            state = mon.classify_reading(r, lv0["alert_mm"], lv0["suspension_mm"], a["heave_alert_mm"]["value"], a["heave_suspension_mm"]["value"])
            proj = ba.project_final(r, ch, xf, design_k * z0)
            above_stage = mon.exceeds_stage_prediction(r, stage_pred, 3.0 * sigma)
            seq.append({"face_minus_array_D": nd, "reading_mm": r, "stage_prediction_mm": stage_pred, "state": state,
                        "exceeds_stage_prediction": above_stage, "projected_final_mm": proj})
            if first_alert is None and state != "normal":
                first_alert = nd
            if first_susp is None and state == "work_suspension":
                first_susp = nd
            if first_projected is None and proj is not None and proj >= lv0["alert_mm"]:
                first_projected = nd
        out["cases"].append({
            "id": c["id"], "title": c["title"], "chainage_m": ch, "zone": zone["id"], "axis_depth_m": z0,
            "true": {"VL": c["true_VL"], "K": c["true_K"]},
            "offsets_m": y.tolist(), "readings_mm": readings.tolist(),
            "estimate": {"VL": fit.VL, "K": fit.K, "VL_ci95": list(ci["VL"]), "K_ci95": list(ci["K"]),
                         "se_VL": ci["se_VL"], "se_K": ci["se_K"], "rms_residual_mm": fit.rms_residual_mm},
            "bootstrap_ci95": {"VL": [float(np.percentile(boot[:, 0], 2.5)), float(np.percentile(boot[:, 0], 97.5))],
                               "K": [float(np.percentile(boot[:, 1], 2.5)), float(np.percentile(boot[:, 1], 97.5))], "n": int(boot.shape[0])},
            "loglinear": ll,
            "design": {"VL": design_vl, "K": design_k}, "expected": {"VL": exp_vl, "K": exp_k},
            "p_vl_exceeds_design": fit.prob_vl_exceeds(design_vl),
            "p_vl_exceeds_observed_upper_bound": fit.prob_vl_exceeds(zone["observed_vl"]["upper_bound"]),
            "departure_from_expected": {k: dep_expected[k] for k in ("max_abs_z", "chi2", "dof", "p_value", "flagged", "direction")},
            "review_states": {s: states.count(s) for s in ("normal", "alert", "work_suspension")},
            "max_slope_fitted_1_in": 1.0 / gf.max_slope_single(t_fit), "max_slope_design_1_in": 1.0 / gf.max_slope_single(t_des),
            "slope_steeper_than_design": gf.max_slope_single(t_fit) > gf.max_slope_single(t_des),
            "centre_marker": {"alert_mm": lv0["alert_mm"], "suspension_mm": lv0["suspension_mm"], "sequence": seq,
                              "first_review_state_at_D": first_alert, "first_suspension_at_D": first_susp,
                              "first_projection_reaching_alert_at_D": first_projected},
        })
    out["estimator_comparison"] = estimator_comparison(scn, rng)
    return out


def estimator_comparison(scn: dict, rng: np.random.Generator, trials: int = 300) -> dict:
    """Bias and scatter of the two estimators on repeated SYNTHETIC arrays (case B geometry)."""
    sigma = scn["assumptions"]["marker_noise_sd_mm"]["value"]
    D = scn["alignment"]["excavated_diameter_m"]
    ch, vl, k = 700.0, 2.2, 0.42
    z0 = sc.axis_depth(scn, ch)
    y0s = [t["y0_m"] for t in scn["alignment"]["tunnels"]]
    y = np.array(mon.type_e_offsets(z0, y0s)) - y0s[0]
    res = {"true": {"VL": vl, "K": k}, "trials": trials, "sigma_mm": sigma, "rows": []}
    sets = [ba.synthetic_readings(y, vl, k, z0, D, sigma, rng) for _ in range(trials)]
    nls = np.array([(f.VL, f.K) for f in (ba.fit_gaussian(y, r, z0, D, sigma_mm=sigma) for r in sets)])
    res["rows"].append({"estimator": "least squares on the Gaussian curve", "mean_VL": float(nls[:, 0].mean()), "sd_VL": float(nls[:, 0].std()),
                        "mean_K": float(nls[:, 1].mean()), "sd_K": float(nls[:, 1].std())})
    for mf in (0.05, 0.2):
        ll = np.array([(d["VL"], d["K"]) for d in (ba.fit_loglinear(y, r, z0, D, min_fraction=mf) for r in sets)])
        res["rows"].append({"estimator": f"straight-line method (S5), readings above {mf:g} Smax", "mean_VL": float(ll[:, 0].mean()),
                            "sd_VL": float(ll[:, 0].std()), "mean_K": float(ll[:, 1].mean()), "sd_K": float(ll[:, 1].std())})
    return res


# ----------------------------------------------------------------------------- workbook
def build_workbook(path: Path, scn: dict | None = None, buildings=None) -> Path:
    """Write the Excel workbook for building WORKBOOK_BUILDING with the design-basis inputs."""
    scn = scn or sc.load_scenario()
    by_id = {b.id: b for b in (buildings or sc.load_buildings())}
    b = by_id[WORKBOOK_BUILDING]
    ts = sc.tunnels_at(scn, b.chainage)
    return wbk.build(path, D=ts[0].D, z0=ts[0].z0, K=ts[0].K, VL=ts[0].VL, y0_first=ts[0].y0, y0_second=ts[1].y0,
                     second_factor=scn["alignment"]["second_tunnel_vl_factor"], facade_start=b.y_start, facade_end=b.y_end,
                     H=b.height, e_over_g=b.e_over_g, nu=scn["assumptions"]["poisson_ratio"]["value"],
                     building_label=f"building {b.id} (SYNTHETIC)")


def sagging_compression_variant(base_rows: list[dict], variant_rows: list[dict]) -> list[dict]:
    """Design-basis Stage 2 results with and without the relief from compressive average strain in
    sagging spans (see building.assess_segment)."""
    var = {r["id"]: r for r in variant_rows}
    out = []
    for r in sorted(base_rows, key=lambda r: r["id"]):
        if not r["stage2_applicable"]:
            continue
        v = var[r["id"]]
        out.append({"id": r["id"], "eps_max_pct": r["eps_max_pct"], "eps_max_pct_variant": v["eps_max_pct"],
                    "stage2_category": r["stage2_category"], "stage2_category_variant": v["stage2_category"],
                    "governing_mode_variant": v["governing_mode"], "action": r["action"], "action_variant": v["action"]})
    return out


# ----------------------------------------------------------------------------- main
def existing_tunnel_results(scn: dict) -> dict:
    """Free-field check of the SYNTHETIC existing tunnel on the three volume-loss bases."""
    zone = sc.zone_at(scn, scn["existing_tunnel"]["chainage_m"])
    return {"design": sc.existing_tunnel_check(scn),
            "characteristic": sc.existing_tunnel_check(scn, zone["observed_vl"]["characteristic_95"]),
            "mean": sc.existing_tunnel_check(scn, zone["observed_vl"]["mean"])}


def core_results(scn: dict, buildings) -> dict:
    """Every result except the Monte Carlo runs and the tornado. Deterministic and quick; the golden
    tests recompute it and compare it with the committed files."""
    screening = {basis: sc.screen_all(scn, buildings, basis) for basis in ("design", "characteristic", "mean")}
    screening["sagging_compression_variant"] = sagging_compression_variant(
        screening["design"], sc.screen_all(scn, buildings, "design", compression_relief=False))
    plan = mon.instrumentation_plan(scn, buildings, screening["design"])
    back = backanalysis_demo(scn, plan)
    instruments = plan.pop("instruments")
    return {"verification": verification(), "screening": screening, "existing": existing_tunnel_results(scn),
            "plan": plan, "instruments": instruments, "back": back}


def tornado_results(scn: dict, buildings, design_rows) -> list[dict]:
    """Sensitivity study for the three highest-ranked buildings on shallow foundations."""
    by_id = {b.id: b for b in buildings}
    return [un.tornado(scn, by_id[r["id"]]) for r in design_rows if r["stage2_applicable"]][:3]


INSTRUMENT_COLUMNS = ["id", "type", "chainage_m", "offset_m", "predicted_expected_mm", "predicted_design_mm", "alert_mm",
                      "suspension_mm", "heave_alert_mm", "heave_suspension_mm", "reason"]
SCREENING_COLUMNS = ["rank", "id", "description", "chainage_m", "formation", "foundation", "structure", "max_settlement_mm",
                     "max_slope_1_in", "stage1_negligible", "eps_max_pct", "governing_case", "governing_mode", "stage2_category",
                     "sensitivity_index_sum", "final_category", "action"]


def instrument_rows(instruments) -> list[list]:
    return [INSTRUMENT_COLUMNS] + [[f"{i[c]:.2f}" if isinstance(i[c], float) else i[c] for c in INSTRUMENT_COLUMNS] for i in instruments]


def screening_rows(rows) -> list[list]:
    return [SCREENING_COLUMNS] + [[("" if r[c] is None else (f"{r[c]:.4g}" if isinstance(r[c], float) else r[c]))
                                   for c in SCREENING_COLUMNS] for r in rows]


def _write_csv(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)


def run(samples: int | None = None, figures: bool = True, out_dir: Path | None = None) -> dict:
    """Rebuild results/, the figures and the workbook. ``out_dir`` (default: the repository) lets a
    smoke run write somewhere else."""
    base = Path(out_dir) if out_dir is not None else ROOT
    results_dir = base / "results"
    scn = sc.load_scenario()
    buildings = sc.load_buildings()
    a = scn["assumptions"]
    n = samples or a["monte_carlo_samples"]["value"]
    seed = a["random_seed"]["value"]

    core = core_results(scn, buildings)
    mc = un.monte_carlo(scn, buildings, n, seed, correlation=a["vl_correlation_between_drives"]["value"])
    mc_indep = un.monte_carlo(scn, buildings, n, seed, correlation=0.0)
    tornados = tornado_results(scn, buildings, core["screening"]["design"])
    build_workbook(base / "workbook" / WORKBOOK_PATH.name, scn, buildings)

    facts = list(csv.DictReader((ROOT / "data" / "facts.csv").open(encoding="utf-8")))
    meta = {
        "facts": {"total": len(facts), "verified": sum(1 for r in facts if r["status"] == "VERIFIED"),
                  "unverified": sum(1 for r in facts if r["status"] == "UNVERIFIED"),
                  "sources": sum(1 for _ in csv.DictReader((ROOT / "data" / "sources.csv").open(encoding="utf-8")))},
        "monte_carlo_samples": n, "seed": seed, "python": platform.python_version().rsplit(".", 1)[0],
        "workbook_building": WORKBOOK_BUILDING,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(results_dir / "instruments.csv", instrument_rows(core["instruments"]))
    _write_csv(results_dir / "screening_design_basis.csv", screening_rows(core["screening"]["design"]))
    for name, obj in (("verification.json", _round(core["verification"], 8)), ("screening.json", _round(core["screening"])),
                      ("existing_tunnel.json", _round(core["existing"])),
                      ("uncertainty.json", _round({"base": mc, "independent_drives": mc_indep, "tornado": tornados})),
                      ("monitoring.json", _round(core["plan"])), ("backanalysis.json", _round(core["back"])), ("meta.json", meta)):
        _dump(name, obj, results_dir)
    out = dict(core, mc=mc, mc_indep=mc_indep, tornado=tornados, meta=meta)
    if figures:
        from . import figures as fig

        fig.make_all(scn, buildings, out, base / "docs" / "figures")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=None, help="Monte Carlo samples (default: inputs/scenario.json)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out", default=None, help="write results/, docs/figures/ and workbook/ under this folder instead")
    args = ap.parse_args()
    out = run(args.samples, not args.no_figures, args.out)
    d = out["screening"]["design"]
    print(f"screened {len(d)} buildings; design-basis Stage 3 or pile assessment needed for "
          f"{sum(1 for r in d if not r['action'].startswith('None'))}")


if __name__ == "__main__":
    main()
