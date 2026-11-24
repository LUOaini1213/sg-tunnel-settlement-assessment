"""Figures for the README and the technical note (matplotlib, non-interactive backend)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import building as bld  # noqa: E402
from . import greenfield as gf  # noqa: E402
from . import screening as sc  # noqa: E402

FIG = Path(__file__).resolve().parent.parent / "docs" / "figures"
CAT_COLOURS = {None: "#9e9e9e", 0: "#2e7d32", 1: "#9ccc65", 2: "#fbc02d", 3: "#ef6c00", 4: "#c62828", 5: "#6a1b1a"}
TAG = "SYNTHETIC alignment and buildings. Student screening study, not a design."


def _save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=130, metadata={"Software": None})
    plt.close(fig)


def fig_trough(scn: dict) -> None:
    ch = 700.0
    ts = sc.tunnels_at(scn, ch)
    ys = np.linspace(-60, 60, 601)
    fig, ax = plt.subplots(3, 1, figsize=(8, 8.2), sharex=True)
    ax[0].plot(ys, -gf.settlement(ys, ts[:1]) * 1000, "--", color="#1565c0", label="first drive only")
    ax[0].plot(ys, -gf.settlement(ys, ts[1:]) * 1000, ":", color="#6a1b9a", label="second drive only")
    ax[0].plot(ys, -gf.settlement(ys, ts) * 1000, color="k", lw=2, label="both drives (superposition)")
    for t in ts:
        ax[0].axvline(t.y0, color="#bdbdbd", lw=0.8)
    ax[0].set_ylabel("settlement (mm)")
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].set_title(f"Greenfield trough at chainage {ch:.0f} m (Kallang Formation zone, design K {ts[0].K}, V {ts[0].VL}%, "
                    f"axis depth {ts[0].z0:.1f} m)", fontsize=9)
    ax[1].plot(ys, gf.slope(ys, ts) * 1000, color="k")
    ax[1].axhline(2, color="#c62828", lw=0.8, ls="--")
    ax[1].axhline(-2, color="#c62828", lw=0.8, ls="--")
    ax[1].text(-59, 2.15, "1:500 (Stage 1 slope limit)", fontsize=7, color="#c62828")
    ax[1].set_ylabel("slope (mm/m)")
    eh = gf.horizontal_strain(ys, ts) * 100
    ax[2].plot(ys, eh, color="k")
    ax[2].fill_between(ys, 0, eh, where=eh > 0, color="#ef9a9a", label="tension (hogging zones)")
    ax[2].fill_between(ys, 0, eh, where=eh < 0, color="#90caf9", label="compression (sagging zone)")
    ax[2].set_ylabel("horizontal strain (%)")
    ax[2].set_xlabel("offset from the centre of the alignment (m)")
    ax[2].legend(fontsize=8)
    fig.text(0.01, 0.005, TAG, fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    _save(fig, "fig1_trough.png")


def fig_longitudinal(scn: dict) -> None:
    ch = 700.0
    t = sc.tunnels_at(scn, ch)[0]
    i = gf.trough_width(t.K, t.z0)
    xs = np.linspace(-60, 60, 481)
    fig, ax = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True)
    ax[0].plot(xs, -gf.settlement_advancing(xs, t.y0, t, 0.0) * 1000, color="k")
    ax[0].axvline(0, color="#c62828", lw=0.8)
    ax[0].text(1, -2, "face", color="#c62828", fontsize=8)
    ax[0].set_ylabel("settlement on the centre line (mm)")
    ax[0].set_title("Longitudinal trough above the first drive (cumulative normal curve): tunnel advancing to the right", fontsize=9)
    ex = gf.long_strain_advancing(xs, t.y0, t, 0.0) * 100
    ax[1].plot(xs, ex, color="k")
    ax[1].fill_between(xs, 0, ex, where=ex > 0, color="#ef9a9a", label="tension ahead of the face")
    ax[1].fill_between(xs, 0, ex, where=ex < 0, color="#90caf9", label="compression behind the face")
    ax[1].set_ylabel("longitudinal strain (%)")
    ax[1].set_xlabel(f"distance from the face (m); i = {i:.1f} m")
    ax[1].legend(fontsize=8)
    fig.text(0.01, 0.005, TAG + " The 0.5 Smax at the face applies to open-face tunnelling.", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, "fig2_longitudinal.png")


def fig_plan(scn: dict, buildings, rows) -> None:
    length = scn["alignment"]["length_m"]
    xs = np.linspace(0, length, 181)
    ys = np.linspace(-75, 75, 151)
    S = np.zeros((ys.size, xs.size))
    for j, x in enumerate(xs):
        S[:, j] = gf.settlement(ys, sc.tunnels_at(scn, float(x))) * 1000
    fig, ax = plt.subplots(figsize=(10, 4.6))
    cs = ax.contourf(xs, ys, S, levels=[1, 5, 10, 20, 30, 40, 50, 60, 80], cmap="Greys", alpha=0.75, extend="max")
    ax.contour(xs, ys, S, levels=[10], colors="#1565c0", linewidths=1.2)
    for t in scn["alignment"]["tunnels"]:
        ax.plot([0, length], [t["y0_m"]] * 2, color="k", lw=0.8, ls="--")
    by_id = {r["id"]: r for r in rows}
    for b in buildings:
        r = by_id[b.id]
        col = CAT_COLOURS[r["final_category"]]
        ax.add_patch(plt.Rectangle((b.chainage - b.along_length / 2, b.y_start), b.along_length, b.y_end - b.y_start,
                                   facecolor=col, edgecolor="k", lw=0.8, hatch=None if r["stage2_applicable"] else "///"))
        ax.text(b.chainage, b.y_end + 2, f"{b.id} (#{r['rank']})", ha="center", fontsize=7)
    for z in scn["ground_zones"]:
        ax.axvline(z["to_m"], color="#6d4c41", lw=0.8)
        ax.text((z["from_m"] + z["to_m"]) / 2, -72, z["formation"], ha="center", fontsize=7, color="#6d4c41")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CAT_COLOURS[c], edgecolor="k") for c in (0, 1, 2, 3, 4)]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=CAT_COLOURS[None], edgecolor="k", hatch="///"))
    ax.legend(handles, ["0 Negligible", "1 Very Slight", "2 Slight", "3 Moderate", "4+ Severe", "piled or mixed: not screened"],
              fontsize=7, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.13), title="Final category (Stage 2 + sensitivity indices)", title_fontsize=7)
    fig.colorbar(cs, ax=ax, label="design-basis greenfield settlement (mm)", pad=0.01)
    ax.set_xlabel("chainage (m)")
    ax.set_ylabel("offset (m)")
    ax.set_xlim(0, length)
    ax.set_ylim(-75, 78)
    fig.text(0.01, 0.005, TAG + " Blue line: 10 mm contour. Labels: building (instrumentation rank).", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, "fig3_plan.png")


def fig_probabilities(mc: dict, design_rows) -> None:
    ids = [r["id"] for r in sorted(design_rows, key=lambda r: r["id"]) if r["stage2_applicable"]]
    d_eps = {r["id"]: r["eps_max_pct"] for r in design_rows}
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(ids))
    p05 = [mc["buildings"][i]["eps_max_pct"]["p05"] for i in ids]
    p50 = [mc["buildings"][i]["eps_max_pct"]["p50"] for i in ids]
    p95 = [mc["buildings"][i]["eps_max_pct"]["p95"] for i in ids]
    ax.vlines(x, p05, p95, color="#1565c0", lw=5, alpha=0.5, label="Monte Carlo 5th to 95th percentile")
    ax.plot(x, p50, "o", color="#0d47a1", label="Monte Carlo median")
    ax.plot(x, [d_eps[i] for i in ids], "D", color="#c62828", label="design-basis value (LTA typical K and V)")
    for lim, name in zip(bld.CATEGORY_LIMITS, ("0|1", "1|2", "2|3", "3|4")):
        ax.axhline(lim * 100, color="#757575", lw=0.7, ls="--")
        ax.text(len(ids) - 0.45, lim * 100, name, fontsize=7, va="bottom", color="#757575")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 0.6)
    for k, i in enumerate(ids):
        if p95[k] < 1e-3:
            ax.annotate("below scale", (k, 1.15e-3), ha="center", fontsize=7, color="#0d47a1")
    ax.set_xticks(x)
    ax.set_xticklabels(ids)
    ax.set_ylabel("maximum tensile strain (%)")
    ax.set_title("Stage 2 tensile strain: design basis against observed volume-loss statistics", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    fig.text(0.01, 0.005, TAG + " Dashed lines: category boundaries of LTA Table 14.2.", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, "fig4_uncertainty.png")


def fig_tornado(tornados) -> None:
    t = tornados[0]
    bars = t["bars"][::-1]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for k, b in enumerate(bars):
        lo, hi = sorted((b["low_pct"], b["high_pct"]))
        ax.barh(k, hi - lo, left=lo, color="#1565c0", alpha=0.75)
        ax.text(hi, k, f"  {b['range']}", va="center", fontsize=7)
    ax.axvline(t["base_eps_max_pct"], color="k", lw=1)
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels([b["input"] for b in bars], fontsize=8)
    ax.set_xlabel("maximum tensile strain (%)")
    ax.set_title(f"One-at-a-time sensitivity for building {t['building']} (base: V {t['base_VL_pct']:.2f}%, K {t['base_K']:.3f})", fontsize=9)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.35)
    fig.text(0.01, 0.005, TAG, fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, "fig5_tornado.png")


def fig_backanalysis(scn: dict, back: dict) -> None:
    from . import backanalysis as ba

    D = scn["alignment"]["excavated_diameter_m"]
    c = back["cases"][1]
    y = np.array(c["offsets_m"])
    r = np.array(c["readings_mm"])
    z0 = c["axis_depth_m"]
    yy = np.linspace(y.min(), y.max(), 300)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax[0].errorbar(y, -r, yerr=2 * back["sigma_mm"], fmt="o", color="k", ms=3, lw=0.8, label="SYNTHETIC readings (bars: 2 standard deviations)")
    e = c["estimate"]
    ax[0].plot(yy, -ba.model_mm(yy, e["VL"], e["K"], z0, D), color="#c62828",
               label=f"fitted: V {e['VL']:.2f}% [{e['VL_ci95'][0]:.2f}, {e['VL_ci95'][1]:.2f}], K {e['K']:.3f} [{e['K_ci95'][0]:.3f}, {e['K_ci95'][1]:.3f}]")
    ax[0].plot(yy, -ba.model_mm(yy, c["design"]["VL"], c["design"]["K"], z0, D), "--", color="#1565c0",
               label=f"design basis: V {c['design']['VL']}%, K {c['design']['K']} (work suspension level)")
    ax[0].plot(yy, -ba.model_mm(yy, c["expected"]["VL"], c["expected"]["K"], z0, D), ":", color="#2e7d32",
               label=f"expected: V {c['expected']['VL']}%, K {c['expected']['K']}")
    ax[0].set_xlabel("offset from the first drive (m)")
    ax[0].set_ylabel("settlement (mm)")
    ax[0].legend(fontsize=6.5, loc="lower right")
    ax[0].set_title("Back-analysis of one Type E array (case B)", fontsize=9)
    seq = c["centre_marker"]["sequence"]
    nd = [s["face_minus_array_D"] for s in seq]
    ax[1].plot(nd, [s["reading_mm"] for s in seq], "o-", color="k", label="centre-line marker reading")
    ax[1].plot(nd, [s["stage_prediction_mm"] for s in seq], ":", color="#2e7d32", label="stage prediction (expected)")
    pj = [(s["face_minus_array_D"], s["projected_final_mm"]) for s in seq if s["projected_final_mm"] is not None]
    ax[1].plot([p[0] for p in pj], [p[1] for p in pj], "s", mfc="none", color="#c62828", label="projected final settlement")
    ax[1].axhline(c["centre_marker"]["alert_mm"], color="#ef6c00", ls="--", lw=1, label="alert level")
    ax[1].axhline(c["centre_marker"]["suspension_mm"], color="#c62828", ls="--", lw=1, label="work suspension level")
    ax[1].set_xlabel("face position relative to the array (tunnel diameters)")
    ax[1].set_ylabel("settlement (mm)")
    ax[1].legend(fontsize=6.5, loc="upper left")
    ax[1].set_title("Review levels as the face advances", fontsize=9)
    fig.text(0.01, 0.005, "SYNTHETIC readings. Alert factor 0.7 is an ASSUMPTION. Student screening study, not a design.", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, "fig6_backanalysis.png")


def fig_chart_check() -> None:
    """The published damage chart for L/H = 1 in hogging, redrawn from the equations."""
    bf = bld.bending_factor(1, 1, 2.6, "hogging")
    df = bld.diagonal_factor(1, 1, 2.6, "hogging")
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for lim in (0.05, 0.075, 0.15, 0.3):
        eh = np.linspace(0, lim, 400)
        dr_b = (lim - eh) * bf
        rem = lim - 0.35 * eh
        dr_d = np.sqrt(np.clip(rem ** 2 - (0.65 * eh) ** 2, 0, None)) * df
        ax.plot(eh, np.minimum(dr_b, dr_d), color="k")
    for yv in (0.05, 0.08, 0.16, 0.32):
        ax.plot(0, yv, "o", mfc="none", color="#c62828")
    ax.plot([], [], "o", mfc="none", color="#c62828", label="intercepts read from the published chart")
    ax.plot([], [], color="k", label="this code (S3 Eq. 7 to 10)")
    ax.set_xlim(0, 0.32)
    ax.set_ylim(0, 0.4)
    ax.set_xlabel("horizontal strain (%)")
    ax.set_ylabel("deflection ratio (%)")
    ax.set_title("Check against S3 Fig. 4: L/H = 1, hogging, E/G = 2.6", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "fig7_chart_check.png")


def make_all(scn: dict, buildings, out: dict) -> None:
    fig_trough(scn)
    fig_longitudinal(scn)
    fig_plan(scn, buildings, out["screening"]["design"])
    fig_probabilities(out["mc"], out["screening"]["design"])
    fig_tornado(out["tornado"])
    fig_backanalysis(scn, out["back"])
    fig_chart_check()
