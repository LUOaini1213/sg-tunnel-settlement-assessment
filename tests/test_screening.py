"""Scenario handling, staged screening rules and ranking."""
import math

import numpy as np
import pytest

from tunnelscreen import building as bld
from tunnelscreen import greenfield as gf
from tunnelscreen import screening as sc


def test_inputs_are_labelled_synthetic(scn, buildings):
    assert scn["label"] == "SYNTHETIC" and "not a real line" in scn["note"]
    assert scn["existing_tunnel"]["label"] == "SYNTHETIC"
    assert len(buildings) == 13 and len({b.id for b in buildings}) == 13
    assert {b.foundation for b in buildings} == {"shallow", "piled", "mixed"}
    assert {b.structure for b in buildings} == {"masonry", "framed"}


def test_axis_depth_interpolates_between_control_points(scn):
    assert sc.axis_depth(scn, 0.0) == 22.0
    assert sc.axis_depth(scn, 200.0) == pytest.approx(24.0)
    assert sc.axis_depth(scn, 400.0) == 26.0
    assert sc.axis_depth(scn, 750.0) == pytest.approx(22.0)
    assert sc.axis_depth(scn, 900.0) == 20.0


def test_zone_lookup_and_boundaries(scn):
    assert sc.zone_at(scn, 0.0)["id"] == "Z1"
    assert sc.zone_at(scn, 379.9)["id"] == "Z1"
    assert sc.zone_at(scn, 380.0)["id"] == "Z2"  # a zone owns its lower bound
    assert sc.zone_at(scn, 560.0)["id"] == "Z3"
    assert sc.zone_at(scn, 900.0)["id"] == "Z3"
    with pytest.raises(ValueError):
        sc.zone_at(scn, 900.1)
    with pytest.raises(ValueError):
        sc.zone_at(scn, -1.0)


def test_design_values_are_the_lta_table_14_1_values(scn):
    z = {x["id"]: x for x in scn["ground_zones"]}
    assert (z["Z1"]["design"]["K"], z["Z1"]["design"]["VL"]) == (0.45, 1.0)  # Old Alluvium
    assert (z["Z2"]["design"]["K"], z["Z2"]["design"]["VL"]) == (0.45, 3.0)  # mixed face with Kallang
    assert (z["Z3"]["design"]["K"], z["Z3"]["design"]["VL"]) == (0.5, 3.0)  # Kallang Formation
    # observed statistics of S5 Table 2
    assert [z[k]["observed_vl"]["mean"] for k in ("Z1", "Z2", "Z3")] == [0.23, 0.70, 0.55]
    assert [z[k]["observed_vl"]["characteristic_95"] for k in ("Z1", "Z2", "Z3")] == [0.5, 1.5, 1.2]
    for zone in scn["ground_zones"]:
        lo, hi = zone["k_range"]
        assert lo <= zone["design"]["K"] <= hi and lo <= zone["observed_k_best_fit"] <= hi


def test_tunnels_at_uses_design_values_and_options(scn):
    ts = sc.tunnels_at(scn, 100.0)
    assert [(t.y0, t.D, t.K, t.VL) for t in ts] == [(-7.5, 6.6, 0.45, 1.0), (7.5, 6.6, 0.45, 1.0)]
    assert ts[0].z0 == pytest.approx(23.0)
    ts = sc.tunnels_at(scn, 700.0, K=0.6, VL=2.0, depth_offset=1.5, second_factor=1.25)
    assert [(t.K, t.VL) for t in ts] == [(0.6, 2.0), (0.6, 2.5)]  # the factor applies to the second drive only
    assert ts[0].z0 == pytest.approx(sc.axis_depth(scn, 700.0) + 1.5)
    ts = sc.tunnels_at(scn, 700.0, VL=[0.4, 0.9])
    assert [t.VL for t in ts] == [0.4, 0.9]


def test_clear_separation_of_the_twin_tunnels_exceeds_one_diameter(scn):
    # S3 section 2.4: superposition may be unconservative when the clear separation is under 1 D
    al = scn["alignment"]
    y = sorted(t["y0_m"] for t in al["tunnels"])
    assert (y[1] - y[0]) - al["excavated_diameter_m"] > al["excavated_diameter_m"]


def test_footprint_extremes_cover_all_three_slopes(scn, by_id):
    b = by_id["B11"]  # straddles the first drive
    ts = sc.tunnels_at(scn, b.chainage)
    smax, slope = sc.footprint_extremes(b, ts)
    ys = np.linspace(b.y_start, b.y_end, 201)
    assert smax == pytest.approx(float(gf.settlement(ys, ts).max()))
    transverse = float(np.abs(gf.slope(ys, ts)).max())
    first_only = float(np.abs(gf.slope(ys, ts[:1])).max())
    i = gf.trough_width(ts[0].K, ts[0].z0)
    transient = max(float(gf.settlement_one(ys, t).max()) for t in ts) / (i * math.sqrt(2 * math.pi))
    assert slope == pytest.approx(max(transverse, first_only, transient))
    assert slope >= transverse


def test_stage2_cases_first_drive_can_govern(scn, by_id):
    """S3 section 2.4: a single tunnel can be worse for a building than the later combination."""
    b = by_id["B07"]
    cases = sc.stage2_cases(b, sc.tunnels_at(scn, b.chainage))
    assert set(cases) == {"transverse, first drive only", "transverse, both drives",
                          "longitudinal, drive 1 passing", "longitudinal, drive 2 passing"}
    assert cases["transverse, first drive only"][0] > cases["transverse, both drives"][0]
    direct = bld.assess_facade(sc.transverse_profile(sc.tunnels_at(scn, b.chainage)[:1]), b.y_start, b.y_end, b.height, 12.5)
    assert cases["transverse, first drive only"][0] == pytest.approx(direct.eps_max)


def test_longitudinal_offsets_include_the_centre_line_only_when_under_the_building(scn, by_id):
    ts = sc.tunnels_at(scn, 700.0)
    assert sc.longitudinal_offsets(by_id["B11"], ts[0]) == [-20.0, 5.0, -7.5]
    assert sc.longitudinal_offsets(by_id["B11"], ts[1]) == [-20.0, 5.0]


def test_worst_longitudinal_is_proportional_to_volume_loss(scn, by_id):
    b = by_id["B11"]
    t = sc.tunnels_at(scn, b.chainage)[0]
    e1, _ = sc.worst_longitudinal(b, t)
    e2, _ = sc.worst_longitudinal(b, t.with_(VL=2 * t.VL))
    assert e1 > 0 and e2 == pytest.approx(2 * e1, rel=1e-9)


@pytest.mark.parametrize("foundation,s1,cat,idx,expected", [
    ("piled", True, None, (0, 0, 0), "Pile assessment"),
    ("mixed", False, None, (0, 0, 0), "Stage 3 detailed evaluation (S1 Section 14.3.8"),
    ("shallow", False, 0, (0, 0, 1), "Stage 3 detailed evaluation (S1 Section 14.3.9"),
    ("shallow", True, 0, (0, 0, 0), "None: negligible at Stage 1"),
    ("shallow", True, 1, (0, 1, 0), "None: low risk at Stage 2"),
    ("shallow", True, 2, (2, 0, 0), "Stage 3 detailed evaluation (S1 Section 14.3.6"),
    ("shallow", False, 1, (0, 0, 0), "None: low risk at Stage 2"),
    ("shallow", False, 2, (0, 0, 0), "Stage 3 detailed evaluation (S1 Section 14.3.6"),
    ("shallow", False, 0, (0, 0, 0), "None: low risk at Stage 2"),
])
def test_action_rules(foundation, s1, cat, idx, expected):
    assert sc.action_for(foundation, s1, cat, idx).startswith(expected)


def test_assess_building_shallow_and_piled(scn, by_id):
    b = by_id["B09"]
    row = sc.assess_building(b, sc.tunnels_at(scn, b.chainage))
    assert row["stage2_applicable"] and row["stage2_category"] == bld.damage_category(row["eps_max_pct"] / 100)
    assert row["eps_max_pct"] == pytest.approx(max(c["eps_max_pct"] for c in row["cases"].values()))
    assert row["final_category"] == row["stage2_category"] + row["sensitivity_index_sum"]
    assert row["above_tunnel"] is False
    p = by_id["B04"]
    prow = sc.assess_building(p, sc.tunnels_at(scn, p.chainage))
    assert prow["stage2_applicable"] is False and prow["eps_max_pct"] is None and prow["final_category"] is None
    assert prow["above_tunnel"] is True and prow["action"].startswith("Pile assessment")
    assert prow["max_settlement_mm"] > 0


def test_sensitivity_indices_raise_the_final_category(scn):
    rows = {r["id"]: r for r in sc.screen_all(scn, sc.load_buildings(), "design")}
    assert rows["B06"]["final_category"] == rows["B06"]["stage2_category"] + 2
    assert rows["B05"]["final_category"] == rows["B05"]["stage2_category"] + 1
    assert rows["B11"]["action"].startswith("Stage 3 detailed evaluation (S1 Section 14.3.9")


def test_design_basis_screening_headline(scn, buildings):
    rows = sc.screen_all(scn, buildings, "design")
    assert [r["rank"] for r in rows] == list(range(1, 14))
    by = {r["id"]: r for r in rows}
    assert by["B13"]["stage1_negligible"] and by["B03"]["stage1_negligible"]
    assert not by["B09"]["stage1_negligible"]
    # buildings needing more than this study can give come first
    needs_more = [not r["action"].startswith("None") for r in rows]
    assert needs_more == sorted(needs_more, reverse=True)
    assert sum(needs_more) == 6
    assert rows[0]["id"] == "B06"


def test_basis_changes_volume_loss_only(scn, buildings):
    d = {r["id"]: r for r in sc.screen_all(scn, buildings, "design")}
    m = {r["id"]: r for r in sc.screen_all(scn, buildings, "mean")}
    c = {r["id"]: r for r in sc.screen_all(scn, buildings, "characteristic")}
    for bid in ("B01", "B09"):
        ratio = m[bid]["VL_pct"] / d[bid]["VL_pct"]
        assert m[bid]["max_settlement_mm"] == pytest.approx(d[bid]["max_settlement_mm"] * ratio)
        assert m[bid]["eps_max_pct"] == pytest.approx(d[bid]["eps_max_pct"] * ratio, rel=1e-6)
        assert m[bid]["max_settlement_mm"] < c[bid]["max_settlement_mm"] < d[bid]["max_settlement_mm"]
    with pytest.raises(ValueError):
        sc.screen_all(scn, buildings, "guess")


def test_priority_key_orders_by_need_then_category_then_strain_then_settlement():
    base = {"action": "None: low risk at Stage 2", "final_category": 1, "eps_max_pct": 0.06, "max_settlement_mm": 20.0}
    stage3 = dict(base, action="Stage 3 detailed evaluation", final_category=0, eps_max_pct=0.01, max_settlement_mm=1.0)
    higher_cat = dict(base, final_category=2, eps_max_pct=0.01)
    higher_eps = dict(base, eps_max_pct=0.07, max_settlement_mm=1.0)
    higher_s = dict(base, max_settlement_mm=25.0)
    piled = {"action": "Pile assessment", "final_category": None, "eps_max_pct": None, "max_settlement_mm": 5.0}
    order = sorted([base, higher_s, higher_eps, higher_cat, stage3, piled], key=sc.priority_key)
    assert order == [stage3, piled, higher_cat, higher_eps, higher_s, base]


def test_existing_tunnel_check(scn):
    r = sc.existing_tunnel_check(scn)
    assert r["level_checked_m"] == 13.0 and r["clear_distance_m"] == pytest.approx(24.5 - 3.3 - 13.0)
    # hand check of one tunnel's contribution: i(z) = (0.45/0.5) x 24.5 x (0.175 + 0.325 (1 - 13/24.5))
    i_z = 0.9 * 24.5 * (0.175 + 0.325 * (1 - 13.0 / 24.5))
    one = gf.trough_volume(1.0, 6.6) / (math.sqrt(2 * math.pi) * i_z) * 1000
    both_at_centre = 2 * one * math.exp(-7.5 ** 2 / (2 * i_z ** 2))
    assert r["max_subsurface_settlement_mm"] >= both_at_centre - 1e-9
    assert r["max_subsurface_settlement_mm"] == pytest.approx(22.14, abs=0.01)
    assert r["max_subsurface_settlement_mm"] > r["surface_settlement_mm"]  # narrower and deeper at depth
    assert r["exceeds_allowable"] is True and r["allowable_total_movement_mm"] == 15.0
    mean = sc.existing_tunnel_check(scn, 0.23)
    assert mean["max_subsurface_settlement_mm"] == pytest.approx(r["max_subsurface_settlement_mm"] * 0.23)
    assert mean["exceeds_allowable"] is False
