"""Instrumentation layout and review levels for the SYNTHETIC alignment.

Array types, spacings and the two review levels follow the LTA Civil Design Criteria (source S1,
Chapter 13). Every numerical factor that S1 does not give is an ASSUMPTION held in
inputs/scenario.json and repeated in the provenance table.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from . import greenfield as gf
from . import screening as sc

#: Offsets of the ground settlement points of a Type E array as multiples of Z, the depth to the
#: tunnel axis: spacings 0.5 Z, eight of 0.25 Z, 0.5 Z (S1 Figures 13.5 and 13.6).
TYPE_E_OFFSETS = (-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
TYPE_D_SPACING_M = 25.0  # S1 Cl. 13.4.2.1 a)
TYPE_E_SPACING_M = 200.0  # S1 Cl. 13.4.2.1 b)
TYPE_F_SPACING_M = 1000.0  # S1 Cl. 13.4.2.1 c)
APPROACH_ARRAY_DISTANCES_M = (25.0, 50.0)  # S1 Section 13.4.9
TENSILE_STRAIN_MONITORING_PCT = 0.1  # S1 Cl. 13.4.7.2
MONITORING_ZONE_FACTOR = 1.5  # S1 Figure 13.3: 1.5 Z each side of a tunnel


def zone_half_width(z0: float, tunnel_offsets: Sequence[float]) -> float:
    """Half-width of the building monitoring zone about the alignment datum: 1.5 Z beyond the
    centre line of the outer tunnel on each side (S1 Figure 13.3, one zone per tunnel, joined)."""
    return max(abs(y) for y in tunnel_offsets) + MONITORING_ZONE_FACTOR * z0


def array_count(length_m: float, spacing_m: float) -> int:
    """One array "for every [spacing] (and part thereof)" of tunnel (S1 Cl. 13.4.2.1)."""
    return int(math.ceil(length_m / spacing_m - 1e-9))


def type_e_offsets(z0: float, tunnel_offsets: Sequence[float], merge_within_m: float = 2.0) -> list[float]:
    """Transverse positions of the settlement points of a Type E array over one or more tunnels.

    Each tunnel gets the Figure 13.5 pattern about its own centre line. Where points from different
    tunnels fall closer together than ``merge_within_m`` only one is kept, and a point on a tunnel
    centre line is always the one kept (S1 Cl. 13.4.2.4 allows overlapping arrays of twin tunnels
    to be reduced; the 2 m merge distance is an ASSUMPTION).
    """
    pts = sorted((round(y0 + f * z0, 3), f == 0.0) for y0 in tunnel_offsets for f in TYPE_E_OFFSETS)
    kept: list[tuple[float, bool]] = []
    for pos, centre in pts:
        if kept and pos - kept[-1][0] < merge_within_m:
            if centre and not kept[-1][1]:
                kept[-1] = (pos, centre)
            continue
        kept.append((pos, centre))
    return [p for p, _ in kept]


def review_levels(predicted_design_mm: float, alert_factor: float, min_suspension_mm: float) -> dict:
    """Alert and Work Suspension levels for settlement at one instrument.

    Work Suspension = the settlement predicted with the design-basis K and V, i.e. "the highest
    reading anticipated based on the design" (S1 Cl. 13.7.4), but not less than a floor.
    Alert = alert_factor x Work Suspension. The factor and the floor are ASSUMPTIONS.
    """
    suspension = max(predicted_design_mm, min_suspension_mm)
    return {"alert_mm": alert_factor * suspension, "suspension_mm": suspension}


def classify_reading(reading_mm: float, alert_mm: float, suspension_mm: float,
                     heave_alert_mm: float, heave_suspension_mm: float) -> str:
    """Review state of one reading. Settlement is positive, heave negative (S1 Cl. 13.7.6).

    A reading equal to a level counts as reaching it (S1 Cl. 13.7.7: "If the Alert level is
    reached"); the Work Suspension level is treated the same way (ASSUMPTION, the cautious reading
    of "exceeds" in S1 Cl. 13.7.8).
    """
    if reading_mm >= suspension_mm or -reading_mm >= heave_suspension_mm:
        return "work_suspension"
    if reading_mm >= alert_mm or -reading_mm >= heave_alert_mm:
        return "alert"
    return "normal"


def exceeds_stage_prediction(reading_mm: float, predicted_stage_mm: float, tolerance_mm: float) -> bool:
    """True when a reading is above the level predicted for the current stage by more than the
    tolerance (S1 Cl. 13.7.5 calls for a design review when a predicted level is exceeded)."""
    return reading_mm > predicted_stage_mm + tolerance_mm


# ----------------------------------------------------------------------------- plan
def _design_settlement_mm(scn: dict, chainage: float, y: float) -> float:
    return float(gf.settlement(np.array([y]), sc.tunnels_at(scn, chainage))[0] * 1000.0)


def _expected_settlement_mm(scn: dict, chainage: float, y: float) -> float:
    zone = sc.zone_at(scn, chainage)
    ts = sc.tunnels_at(scn, chainage, VL=zone["observed_vl"]["mean"])
    return float(gf.settlement(np.array([y]), ts)[0] * 1000.0)


def _instrument(scn: dict, inst_id: str, kind: str, chainage: float, y: float, reason: str) -> dict:
    a = scn["assumptions"]
    chainage, y = round(chainage, 1), round(y, 2)  # the tabulated position is the one the levels are computed for
    design = _design_settlement_mm(scn, chainage, y)
    lv = review_levels(design, a["alert_factor"]["value"], a["min_suspension_level_mm"]["value"])
    return {"id": inst_id, "type": kind, "chainage_m": chainage, "offset_m": y,
            "predicted_design_mm": design, "predicted_expected_mm": _expected_settlement_mm(scn, chainage, y),
            "alert_mm": lv["alert_mm"], "suspension_mm": lv["suspension_mm"],
            "heave_alert_mm": a["heave_alert_mm"]["value"], "heave_suspension_mm": a["heave_suspension_mm"]["value"],
            "reason": reason}


def building_marker_offsets(b: sc.Building, tunnels: Sequence[gf.Tunnel]) -> list[float]:
    """Settlement points on a building facade: both ends, each predicted point of inflexion and
    the point of largest predicted settlement, so that every hogging or sagging span of the Stage 2
    model can be measured (placement rule is an ASSUMPTION of this study)."""
    from . import building as bld

    prof = sc.transverse_profile(tunnels)
    lo, hi = max(b.y_start, prof.limits[0]), min(b.y_end, prof.limits[1])
    pts = {round(b.y_start, 2), round(b.y_end, 2)}
    if hi > lo:
        pts.update(round(v, 2) for v in bld.inflexion_points(prof, lo, hi))
        ys = np.linspace(b.y_start, b.y_end, 401)
        pts.add(round(float(ys[int(np.argmax(prof.s(ys)))]), 2))
    return sorted(pts)


def instrumentation_plan(scn: dict, buildings: Sequence[sc.Building], screening_rows: Sequence[dict]) -> dict:
    """Notional surface instrumentation with review levels for every point."""
    al = scn["alignment"]
    length = al["length_m"]
    y0s = [t["y0_m"] for t in al["tunnels"]]
    rows_by_id = {r["id"]: r for r in screening_rows}
    instruments, arrays = [], []

    # Type D: one settlement point above each tunnel centre line every 25 m
    n_d = array_count(length, TYPE_D_SPACING_M)
    for k in range(n_d + 1):
        ch = min(k * TYPE_D_SPACING_M, length)
        for j, y0 in enumerate(y0s, start=1):
            instruments.append(_instrument(scn, f"D{k:02d}-T{j}", "ground settlement point (Type D)", ch, y0,
                                           "S1 Cl. 13.4.2.1 a): every 25 m of each tunnel"))
    # Type E: every 200 m, placed at the middle of each 200 m length
    e_chainages = {}
    n_e = array_count(length, TYPE_E_SPACING_M)
    for k in range(n_e):
        ch = min((k + 0.5) * TYPE_E_SPACING_M, length)
        e_chainages[round(ch, 1)] = "S1 Cl. 13.4.2.1 b): every 200 m of tunnel"
    # additional Type E arrays 25 m and 50 m before each building in use with any part of a tunnel under it
    for b in buildings:
        row = rows_by_id[b.id]
        if b.in_use and row["above_tunnel"]:
            front = b.chainage - b.along_length / 2.0
            for d in APPROACH_ARRAY_DISTANCES_M:
                ch = front - d
                if 0.0 <= ch <= length:
                    e_chainages[round(ch, 1)] = f"S1 Section 13.4.9: about {d:g} m before building {b.id}"
    for n, (ch, reason) in enumerate(sorted(e_chainages.items()), start=1):
        z0 = sc.axis_depth(scn, ch)
        offs = type_e_offsets(z0, y0s)
        arrays.append({"id": f"E{n:02d}", "type": "E", "chainage_m": ch, "axis_depth_m": z0,
                       "n_points": len(offs), "reason": reason})
        for m, y in enumerate(offs, start=1):
            instruments.append(_instrument(scn, f"E{n:02d}-{m:02d}", "ground settlement point (Type E array)", ch, y, reason))
    n_f = array_count(length, TYPE_F_SPACING_M)

    # buildings
    building_plan = []
    for b in buildings:
        row = rows_by_id[b.id]
        half_zone = zone_half_width(sc.axis_depth(scn, b.chainage), y0s)
        in_zone = b.y_start < half_zone and b.y_end > -half_zone
        if not in_zone:
            building_plan.append({"id": b.id, "in_monitoring_zone": False, "markers": 0, "real_time": False,
                                  "tensile_strain_monitoring": False})
            continue
        ts = sc.tunnels_at(scn, b.chainage)
        offs = building_marker_offsets(b, ts)
        for m, y in enumerate(offs, start=1):
            instruments.append(_instrument(scn, f"{b.id}-BS{m:02d}", "building settlement point", b.chainage, y,
                                           "S1 Cl. 13.4.7.1: building within the monitoring zone of Figure 13.3"))
        eps = row["eps_max_pct"]
        building_plan.append({
            "id": b.id, "in_monitoring_zone": True, "markers": len(offs), "marker_offsets_m": offs,
            # S1 Section 13.4.9 a): buildings in use with a tunnel directly or partially underneath are monitored in real time
            "real_time": bool(b.in_use and row["above_tunnel"]),
            # S1 Cl. 13.4.7.2: tensile strain monitoring where the predicted tensile strain exceeds 0.1%
            "tensile_strain_monitoring": bool(eps is not None and eps > TENSILE_STRAIN_MONITORING_PCT),
            "rank": row["rank"],
        })
    return {
        "counts": {"type_d_points": sum(1 for i in instruments if "Type D" in i["type"]),
                   "type_e_arrays": len(arrays), "type_e_points": sum(1 for i in instruments if "Type E" in i["type"]),
                   "type_f_arrays": n_f,
                   "building_points": sum(1 for i in instruments if i["type"] == "building settlement point"),
                   "buildings_real_time": sum(1 for b in building_plan if b["real_time"]),
                   "buildings_tensile_strain": sum(1 for b in building_plan if b["tensile_strain_monitoring"])},
        "arrays": arrays, "buildings": building_plan, "instruments": instruments,
        "note": "Type F arrays add subsurface instruments (inclinometers, extensometers, piezometers); only their number is given here.",
    }
