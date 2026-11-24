"""Instrumentation layout and review-level logic."""
import numpy as np
import pytest

from tunnelscreen import greenfield as gf
from tunnelscreen import monitoring as mon
from tunnelscreen import screening as sc


def test_constants_are_the_cited_lta_values():
    assert (mon.TYPE_D_SPACING_M, mon.TYPE_E_SPACING_M, mon.TYPE_F_SPACING_M) == (25.0, 200.0, 1000.0)
    assert mon.APPROACH_ARRAY_DISTANCES_M == (25.0, 50.0)
    assert mon.TENSILE_STRAIN_MONITORING_PCT == 0.1 and mon.MONITORING_ZONE_FACTOR == 1.5
    # Figure 13.5: 0.5 Z, eight spacings of 0.25 Z, 0.5 Z
    assert list(np.round(np.diff(mon.TYPE_E_OFFSETS), 6)) == [0.5] + [0.25] * 8 + [0.5]
    assert mon.TYPE_E_OFFSETS[0] == -1.5 and mon.TYPE_E_OFFSETS[-1] == 1.5 and len(mon.TYPE_E_OFFSETS) == 11


@pytest.mark.parametrize("length,spacing,n", [(900, 25, 36), (900, 200, 5), (900, 1000, 1), (1000, 1000, 1),
                                               (1001, 1000, 2), (25, 25, 1), (26, 25, 2), (200, 200, 1)])
def test_array_count_every_spacing_and_part_thereof(length, spacing, n):
    assert mon.array_count(length, spacing) == n


def test_type_e_offsets_single_and_twin():
    single = mon.type_e_offsets(20.0, [0.0])
    assert single == [-30.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 30.0]
    # twin tunnels 15 m apart with Z = 20 m: the two 5 m patterns coincide, duplicates are dropped
    twin = mon.type_e_offsets(20.0, [-7.5, 7.5])
    assert twin == [-37.5, -27.5, -22.5, -17.5, -12.5, -7.5, -2.5, 2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 37.5]
    # Z = 22 m: the patterns interleave 1.5 m apart; one of each close pair is kept and the
    # points over the two centre lines always survive
    inter = mon.type_e_offsets(22.0, [-7.5, 7.5])
    assert -7.5 in inter and 7.5 in inter and min(np.diff(inter)) >= 2.0
    assert inter[0] == -40.5 and inter[-1] == 40.5 and len(inter) == 14
    assert len(mon.type_e_offsets(22.0, [-7.5, 7.5], merge_within_m=0.0)) == 22
    assert -9.0 in mon.type_e_offsets(22.0, [-7.5, 7.5], merge_within_m=0.0) and -9.0 not in inter


def test_review_levels():
    assert mon.review_levels(30.0, 0.7, 5.0) == {"alert_mm": pytest.approx(21.0), "suspension_mm": 30.0}
    # the floor applies to instruments with a small prediction
    assert mon.review_levels(2.0, 0.7, 5.0) == {"alert_mm": pytest.approx(3.5), "suspension_mm": 5.0}
    assert mon.review_levels(5.0, 0.5, 5.0) == {"alert_mm": 2.5, "suspension_mm": 5.0}


@pytest.mark.parametrize("reading,state", [(0.0, "normal"), (20.99, "normal"), (21.0, "alert"), (29.99, "alert"),
                                            (30.0, "work_suspension"), (45.0, "work_suspension"),
                                            (-3.49, "normal"), (-3.5, "alert"), (-4.99, "alert"), (-5.0, "work_suspension")])
def test_classify_reading_settlement_and_heave(reading, state):
    assert mon.classify_reading(reading, 21.0, 30.0, 3.5, 5.0) == state


def test_classify_reading_uses_the_right_level_for_each_direction():
    # a large settlement must not be judged against the (smaller) heave levels, nor the reverse
    assert mon.classify_reading(4.0, 21.0, 30.0, 3.5, 5.0) == "normal"
    assert mon.classify_reading(-25.0, 21.0, 30.0, 3.5, 40.0) == "alert"


def test_exceeds_stage_prediction():
    assert mon.exceeds_stage_prediction(8.1, 5.0, 3.0) is True
    assert mon.exceeds_stage_prediction(8.0, 5.0, 3.0) is False
    assert mon.exceeds_stage_prediction(2.0, 5.0, 3.0) is False


@pytest.fixture(scope="module")
def plan(scn, buildings):
    rows = sc.screen_all(scn, buildings, "design")
    return mon.instrumentation_plan(scn, buildings, rows), {r["id"]: r for r in rows}


def test_plan_counts(plan):
    p, _ = plan
    c = p["counts"]
    assert c["type_d_points"] == 2 * 37  # chainage 0 to 900 at 25 m, one point over each tunnel
    assert c["type_f_arrays"] == 1
    regular = [a for a in p["arrays"] if "every 200 m" in a["reason"]]
    assert [a["chainage_m"] for a in regular] == [100.0, 300.0, 500.0, 700.0, 900.0]
    assert c["type_e_arrays"] == len(p["arrays"]) and c["type_e_points"] == sum(a["n_points"] for a in p["arrays"])
    assert len({i["id"] for i in p["instruments"]}) == len(p["instruments"])


def test_approach_arrays_before_buildings_over_a_tunnel(plan, by_id):
    p, rows = plan
    over = [bid for bid, r in rows.items() if r["above_tunnel"]]
    # B02 and B06 have only the edge of a tunnel under them; S1 Section 13.4.9 still applies
    assert sorted(over) == ["B01", "B02", "B04", "B06", "B07", "B11"]
    for bid in over:
        front = by_id[bid].chainage - by_id[bid].along_length / 2
        got = sorted(a["chainage_m"] for a in p["arrays"] if a["reason"].endswith(f"before building {bid}"))
        want = sorted(round(front - d, 1) for d in (25.0, 50.0) if front - d >= 0)
        assert got == want
    assert not any("B09" in a["reason"] for a in p["arrays"])


def test_every_instrument_has_consistent_review_levels(plan, scn):
    p, _ = plan
    a = scn["assumptions"]
    for inst in p["instruments"]:
        design = float(gf.settlement(np.array([inst["offset_m"]]), sc.tunnels_at(scn, inst["chainage_m"]))[0] * 1000)
        assert inst["predicted_design_mm"] == pytest.approx(design, rel=1e-9)
        assert inst["suspension_mm"] == pytest.approx(max(design, a["min_suspension_level_mm"]["value"]))
        assert inst["alert_mm"] == pytest.approx(a["alert_factor"]["value"] * inst["suspension_mm"])
        assert inst["predicted_expected_mm"] < inst["predicted_design_mm"] or design == 0.0
        assert (inst["heave_alert_mm"], inst["heave_suspension_mm"]) == (3.5, 5.0)
    # a point over the first drive in the Kallang zone: suspension level is the design prediction
    d = next(i for i in p["instruments"] if i["id"] == "D28-T1")
    assert d["chainage_m"] == 700.0 and d["offset_m"] == -7.5 and d["suspension_mm"] == pytest.approx(d["predicted_design_mm"])
    assert d["predicted_expected_mm"] == pytest.approx(d["predicted_design_mm"] * 0.55 / 3.0)


def test_building_plan_follows_the_cited_rules(plan, by_id, scn):
    p, rows = plan
    bp = {b["id"]: b for b in p["buildings"]}
    assert all(b["in_monitoring_zone"] for b in bp.values() if b["id"] != "B13")
    # B13 starts 48 m out; the zone edge is 7.5 + 1.5 x 22.67 = 41.5 m
    assert bp["B13"]["in_monitoring_zone"] is False and bp["B13"]["markers"] == 0
    for bid, b in bp.items():
        assert b["real_time"] is bool(rows[bid]["above_tunnel"] and by_id[bid].in_use and b["in_monitoring_zone"])
        eps = rows[bid]["eps_max_pct"]
        assert b["tensile_strain_monitoring"] is bool(b["in_monitoring_zone"] and eps is not None and eps > 0.1)
    assert p["counts"]["buildings_real_time"] == 6


def test_building_markers_sit_on_ends_inflexions_and_deepest_point(scn, by_id):
    b = by_id["B07"]  # from -12 m to 18 m, across both tunnels
    ts = sc.tunnels_at(scn, b.chainage)
    offs = mon.building_marker_offsets(b, ts)
    assert offs[0] == -12.0 and offs[-1] == 18.0 and offs == sorted(offs)
    ys = np.linspace(-12, 18, 3001)
    deepest = float(ys[int(np.argmax(gf.settlement(ys, ts)))])
    assert any(abs(o - deepest) < 0.05 for o in offs)
    for o in offs[1:-1]:
        if abs(o - deepest) >= 0.05:
            assert abs(float(gf.curvature(np.array([o]), ts)[0])) < 1e-7  # a point of inflexion
    # and every predicted point of inflexion on the facade has a marker
    from tunnelscreen import building as bld

    roots = bld.inflexion_points(sc.transverse_profile(ts), -12.0, 18.0)
    assert len(roots) >= 1 and all(any(abs(o - r) < 0.01 for o in offs) for r in roots)
    far = by_id["B03"]  # entirely in one hogging zone: only the two ends
    assert mon.building_marker_offsets(far, sc.tunnels_at(scn, far.chainage)) == [-48.0, -30.0]


def test_monitoring_zone_reaches_one_and_a_half_z_beyond_the_outer_tunnel(scn, by_id):
    assert mon.zone_half_width(20.0, [0.0]) == pytest.approx(30.0)
    assert mon.zone_half_width(22.0, [-7.5, 7.5]) == pytest.approx(7.5 + 33.0)
    # a building starting 36 m out at chainage 660 m (Z = 23.2 m): outside 1.5 Z of the alignment
    # datum but inside 1.5 Z of the second drive, so it is in the zone
    from dataclasses import replace

    b = replace(by_id["B13"], id="BX", y_start=36.0, y_end=50.0)
    z0 = sc.axis_depth(scn, b.chainage)
    assert 1.5 * z0 < 36.0 < 7.5 + 1.5 * z0
    rows = sc.screen_all(scn, [b], "design")
    assert mon.instrumentation_plan(scn, [b], rows)["buildings"][0]["in_monitoring_zone"] is True
