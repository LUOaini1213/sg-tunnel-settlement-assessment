"""Scenario handling and the staged building screening along the SYNTHETIC alignment."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import building as bld
from . import greenfield as gf
from .greenfield import Tunnel

ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class Building:
    id: str
    description: str
    chainage: float
    y_start: float
    y_end: float
    along_length: float
    height: float
    structure: str
    foundation: str
    in_use: bool
    idx_form: int
    idx_heritage: int
    idx_condition: int

    @property
    def indices(self) -> tuple[int, int, int]:
        return (self.idx_form, self.idx_heritage, self.idx_condition)

    @property
    def e_over_g(self) -> float:
        return bld.E_OVER_G[self.structure]


def load_scenario(path: Path | None = None) -> dict:
    return json.loads((path or ROOT / "inputs" / "scenario.json").read_text(encoding="utf-8"))


def load_buildings(path: Path | None = None) -> list[Building]:
    out = []
    with (path or ROOT / "inputs" / "buildings.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out.append(Building(r["id"], r["description"], float(r["chainage_m"]), float(r["y_start_m"]),
                                float(r["y_end_m"]), float(r["along_length_m"]), float(r["height_m"]),
                                r["structure"], r["foundation"], r["in_use"].strip().lower() == "yes",
                                int(r["idx_form"]), int(r["idx_heritage"]), int(r["idx_condition"])))
    return out


def axis_depth(scn: dict, chainage: float) -> float:
    """Depth to tunnel axis by linear interpolation between the control points."""
    pts = scn["alignment"]["axis_depth_control_points"]
    return float(np.interp(chainage, [p[0] for p in pts], [p[1] for p in pts]))


def zone_at(scn: dict, chainage: float) -> dict:
    """Ground zone containing the chainage; a zone owns its lower bound, the last zone both bounds."""
    zones = scn["ground_zones"]
    for z in zones:
        if z["from_m"] <= chainage < z["to_m"]:
            return z
    if chainage == zones[-1]["to_m"]:
        return zones[-1]
    raise ValueError(f"chainage {chainage} is outside the alignment")


def tunnels_at(scn: dict, chainage: float, K: float | None = None, VL: Sequence[float] | float | None = None,
               depth_offset: float = 0.0, second_factor: float | None = None) -> list[Tunnel]:
    """The two tunnels at a chainage. Defaults are the design-basis K and VL of the ground zone."""
    zone = zone_at(scn, chainage)
    al = scn["alignment"]
    k = zone["design"]["K"] if K is None else K
    if VL is None:
        VL = zone["design"]["VL"]
    vls = [float(VL)] * len(al["tunnels"]) if np.isscalar(VL) else [float(v) for v in VL]
    factor = al.get("second_tunnel_vl_factor", 1.0) if second_factor is None else second_factor
    z0 = axis_depth(scn, chainage) + depth_offset
    order = al["drive_order"]
    out = []
    for t, vl in zip(al["tunnels"], vls):
        if t["id"] != order[0]:
            vl = vl * factor
        out.append(Tunnel(t["y0_m"], z0, al["excavated_diameter_m"], vl, k))
    return out


# ----------------------------------------------------------------------------- profiles
def transverse_profile(tunnels: Sequence[Tunnel]) -> bld.Profile:
    """Profile along a facade at right angles to the tunnels, after the listed tunnels have passed."""
    ts = list(tunnels)
    return bld.Profile(s=lambda y: gf.settlement(y, ts), u=lambda y: gf.horizontal_displacement(y, ts),
                       kappa=lambda y: gf.curvature(y, ts), limits=gf.trough_limits(ts))


def longitudinal_profile(t: Tunnel, y: float, x_face: float) -> bld.Profile:
    """Profile along a facade parallel to tunnel ``t`` at offset ``y`` while its face is at x_face.

    Tunnels already completed add the same settlement at every point of such a facade, so they add
    no relative deflection and no strain along it and are left out.
    """
    half = gf.TROUGH_LIMIT_FACTOR * gf.trough_width(t.K, t.z0)
    return bld.Profile(s=lambda x: gf.settlement_advancing(x, y, t, x_face),
                       u=lambda x: gf.long_displacement_advancing(x, y, t, x_face),
                       kappa=lambda x: gf.long_curvature_advancing(x, y, t, x_face),
                       limits=(x_face - half, x_face + half))


def longitudinal_offsets(b: Building, t: Tunnel) -> list[float]:
    """Offsets at which the facade parallel to the tunnel is checked: both edges, and the tunnel
    centre line when it passes under the building (where the settlement is largest)."""
    ys = [b.y_start, b.y_end]
    if b.y_start < t.y0 < b.y_end:
        ys.append(t.y0)
    return ys


def worst_longitudinal(b: Building, t: Tunnel, n_face: int = 25, n_grid: int = 41,
                       hogging_neutral_axis: str = "bottom") -> tuple[float, str]:
    """Largest transient tensile strain in a facade parallel to tunnel ``t`` as its face advances.

    The face is stepped from 2.5 i before the building to 2.5 i beyond it.
    """
    i = gf.trough_width(t.K, t.z0)
    x0, x1 = b.chainage - b.along_length / 2.0, b.chainage + b.along_length / 2.0
    best, gov = 0.0, "none"
    for y in longitudinal_offsets(b, t):
        for xf in np.linspace(x0 - gf.TROUGH_LIMIT_FACTOR * i, x1 + gf.TROUGH_LIMIT_FACTOR * i, n_face):
            r = bld.assess_facade(longitudinal_profile(t, y, float(xf)), x0, x1, b.height, b.e_over_g,
                                  n_grid=n_grid, hogging_neutral_axis=hogging_neutral_axis)
            if r.eps_max > best:
                best, gov = r.eps_max, r.governing
    return best, gov


# ----------------------------------------------------------------------------- staged assessment
def footprint_extremes(b: Building, tunnels: Sequence[Tunnel], n: int = 201) -> tuple[float, float]:
    """Largest settlement and largest slope under the building.

    Settlement is taken after all drives. Slope is the largest of: the transverse slope after the
    first drive alone, the transverse slope after all drives, and the transient slope along the
    tunnel as each face passes, which is S_trans(y) / (i sqrt(2 pi)) at the face.
    """
    ys = np.linspace(b.y_start, b.y_end, n)
    s_max = float(np.max(gf.settlement(ys, tunnels)))
    slopes = [float(np.max(np.abs(gf.slope(ys, tunnels)))), float(np.max(np.abs(gf.slope(ys, tunnels[:1]))))]
    for t in tunnels:
        i = gf.trough_width(t.K, t.z0)
        slopes.append(float(np.max(gf.settlement_one(ys, t))) / (i * gf.SQRT_2PI))
    return s_max, max(slopes)


def stage2_cases(b: Building, tunnels: Sequence[Tunnel], include_longitudinal: bool = True,
                 n_grid: int = 81, hogging_neutral_axis: str = "bottom") -> dict:
    """Maximum tensile strain for each stage of construction considered.

    Cases: transverse facade after the first drive alone and after both drives; facade parallel to
    the tunnels as the face of each drive passes. S3 section 2.4 notes that a single tunnel can be
    worse for a building than the later combination, which is why the first-drive case is kept.
    """
    cases = {}
    first = [tunnels[0]]
    for name, ts in (("transverse, first drive only", first), ("transverse, both drives", list(tunnels))):
        r = bld.assess_facade(transverse_profile(ts), b.y_start, b.y_end, b.height, b.e_over_g,
                              n_grid=n_grid, hogging_neutral_axis=hogging_neutral_axis)
        cases[name] = (r.eps_max, r.governing, r.max_deflection_ratio)
    if include_longitudinal:
        for k, t in enumerate(tunnels):
            eps, gov = worst_longitudinal(b, t, hogging_neutral_axis=hogging_neutral_axis)
            cases[f"longitudinal, drive {k + 1} passing"] = (eps, gov, float("nan"))
    return cases


def action_for(foundation: str, stage1_pass: bool, final_cat: int | None, indices: Sequence[int] = (0, 0, 0)) -> str:
    """Next step for a building, following S1 Sections 14.3.6 to 14.3.10.

    A building that passes Stage 1 but carries a sensitivity index is still taken through Stage 2
    (ASSUMPTION, the cautious reading of S1 Section 14.3.10).
    """
    if foundation == "piled":
        return "Pile assessment (S1 Section 14.3.7); outside this study"
    if foundation == "mixed":
        return "Stage 3 detailed evaluation (S1 Section 14.3.8, mixed foundations); outside this study"
    if indices[2] >= 1:
        return "Stage 3 detailed evaluation (S1 Section 14.3.9, poor existing condition); outside this study"
    if stage1_pass and sum(indices) == 0:
        return "None: negligible at Stage 1"
    if final_cat is not None and final_cat <= 1:
        return "None: low risk at Stage 2"
    return "Stage 3 detailed evaluation (S1 Section 14.3.6); outside this study"


def assess_building(b: Building, tunnels: Sequence[Tunnel], include_longitudinal: bool = True) -> dict:
    """Stage 1 and, for buildings on shallow foundations, Stage 2 with sensitivity indices."""
    smax, slope_max = footprint_extremes(b, tunnels)
    s1 = bld.stage1_negligible(smax, slope_max)
    above_tunnel = any(b.y_start < t.y0 < b.y_end for t in tunnels)
    row = {
        "id": b.id, "description": b.description, "chainage_m": b.chainage, "foundation": b.foundation,
        "structure": b.structure, "above_tunnel": above_tunnel,
        "max_settlement_mm": smax * 1000.0, "max_slope_1_in": (1.0 / slope_max) if slope_max > 0 else float("inf"),
        "stage1_negligible": s1, "stage2_applicable": b.foundation == "shallow",
        "eps_max_pct": None, "governing_case": None, "governing_mode": None,
        "stage2_category": None, "sensitivity_index_sum": int(sum(b.indices)), "final_category": None,
        "cases": {},
    }
    if b.foundation == "shallow":
        cases = stage2_cases(b, tunnels, include_longitudinal)
        name = max(cases, key=lambda c: cases[c][0])
        eps = cases[name][0]
        cat = bld.damage_category(eps)
        row.update(eps_max_pct=eps * 100.0, governing_case=name, governing_mode=cases[name][1],
                   stage2_category=cat, final_category=bld.final_category(cat, b.indices),
                   cases={k: {"eps_max_pct": v[0] * 100.0, "mode": v[1]} for k, v in cases.items()})
    row["action"] = action_for(b.foundation, s1, row["final_category"], b.indices)
    return row


def priority_key(row: dict) -> tuple:
    """Ranking for instrumentation, most demanding first.

    Order of keys (ASSUMPTION of this study): 1) buildings whose category could not be screened
    here (piled or mixed foundations) and those needing Stage 3; 2) final damage category;
    3) maximum tensile strain; 4) maximum settlement.
    """
    needs_more = 0 if row["action"].startswith("None") else 1
    cat = row["final_category"] if row["final_category"] is not None else -1
    eps = row["eps_max_pct"] if row["eps_max_pct"] is not None else -1.0
    return (-needs_more, -cat, -eps, -row["max_settlement_mm"])


def screen_all(scn: dict, buildings: Sequence[Building], basis: str = "design") -> list[dict]:
    """Run the staged screening for every building.

    ``basis`` is "design" (S1 Table 14.1 values), "mean" or "characteristic" (S5 observed volume
    loss with the design K).
    """
    rows = []
    for b in buildings:
        zone = zone_at(scn, b.chainage)
        if basis == "design":
            vl = zone["design"]["VL"]
        elif basis == "mean":
            vl = zone["observed_vl"]["mean"]
        elif basis == "characteristic":
            vl = zone["observed_vl"]["characteristic_95"]
        else:
            raise ValueError(basis)
        row = assess_building(b, tunnels_at(scn, b.chainage, VL=vl))
        row.update(zone=zone["id"], formation=zone["formation"], K=zone["design"]["K"], VL_pct=vl,
                   axis_depth_m=axis_depth(scn, b.chainage))
        rows.append(row)
    ranked = sorted(rows, key=priority_key)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


# ----------------------------------------------------------------------------- existing tunnel
def existing_tunnel_check(scn: dict, basis_vl: float | None = None) -> dict:
    """Greenfield settlement at the invert level of the SYNTHETIC existing tunnel.

    A screening proxy only: it compares free-field ground settlement at that depth with the
    allowable total movement of the structure in S2 Table 2(b); the structure itself is not
    modelled. The subsurface relation is the one published for tunnels in clay (see
    greenfield.subsurface_trough_width).
    """
    et = scn["existing_tunnel"]
    ch = et["chainage_m"]
    tunnels = tunnels_at(scn, ch, VL=basis_vl)
    z_invert = et["axis_depth_m"] + et["diameter_m"] / 2.0
    ys = np.linspace(-60.0, 60.0, 1201)
    s = gf.subsurface_settlement(ys, z_invert, tunnels)
    grad = np.gradient(s, ys)
    smax_mm = float(s.max() * 1000.0)
    return {
        "chainage_m": ch, "level_checked_m": z_invert, "new_axis_depth_m": tunnels[0].z0,
        "clear_distance_m": tunnels[0].z0 - tunnels[0].D / 2.0 - z_invert,
        "VL_pct": tunnels[0].VL, "K": tunnels[0].K,
        "max_subsurface_settlement_mm": smax_mm,
        "surface_settlement_mm": float(gf.settlement(ys, tunnels).max() * 1000.0),
        "max_slope_1_in": float(1.0 / np.max(np.abs(grad))),
        "allowable_total_movement_mm": et["allowable_total_movement_mm"],
        "exceeds_allowable": smax_mm > et["allowable_total_movement_mm"],
    }
