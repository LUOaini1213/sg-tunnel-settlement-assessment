"""Monte Carlo inputs, probabilities and the tornado."""
import math

import numpy as np
import pytest

from tunnelscreen import building as bld
from tunnelscreen import screening as sc
from tunnelscreen import uncertainty as un


def test_lognormal_matches_mean_and_95th_percentile():
    for mean, q95 in ((0.23, 0.5), (0.55, 1.2), (0.70, 1.5)):
        mu, sigma = un.lognormal_params(mean, q95)
        assert math.exp(mu + sigma ** 2 / 2) == pytest.approx(mean, rel=1e-12)
        assert math.exp(mu + un.Z95 * sigma) == pytest.approx(q95, rel=1e-12)
        assert 0 < sigma < un.Z95  # the smaller root
    mu, sigma = un.lognormal_params(0.23, 0.5)
    assert (mu, sigma) == (pytest.approx(-1.632876, abs=1e-5), pytest.approx(0.571315, abs=1e-5))


def test_lognormal_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        un.lognormal_params(0.5, 0.5)
    with pytest.raises(ValueError):
        un.lognormal_params(0.0, 0.5)
    with pytest.raises(ValueError):
        un.lognormal_params(0.1, 10.0)  # ratio too large for any lognormal


def test_characteristic_value():
    assert un.characteristic_value(0.30, 0.24) == pytest.approx(0.694764)
    assert un.characteristic_value(1.0, 0.5, t=2.0) == 2.0


def test_wilson_interval():
    lo, hi = un.wilson_interval(0, 100)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.0369935, abs=1e-6)
    lo, hi = un.wilson_interval(50, 100)
    assert (lo, hi) == (pytest.approx(0.4038315, abs=1e-6), pytest.approx(0.5961685, abs=1e-6))
    lo, hi = un.wilson_interval(100, 100)
    assert hi == pytest.approx(1.0, abs=1e-12) and lo == pytest.approx(1 - 0.0369935, abs=1e-6)
    assert un.wilson_interval(0, 0) == (0.0, 1.0)


def test_sample_zone_statistics_and_correlation(scn):
    zone = scn["ground_zones"][2]
    v1, v2, k = un.sample_zone(zone, 200_000, np.random.default_rng(1), correlation=1.0)
    assert np.array_equal(v1, v2)  # one draw shared by both drives
    assert v1.mean() == pytest.approx(0.55, rel=0.01)
    assert np.percentile(v1, 95) == pytest.approx(1.2, rel=0.015)
    assert k.min() >= 0.45 and k.max() <= 0.65 and k.mean() == pytest.approx(0.55, abs=0.002)
    assert k.std() == pytest.approx(0.2 / math.sqrt(12), rel=0.01)  # uniform over the whole range
    v1, v2, _ = un.sample_zone(zone, 200_000, np.random.default_rng(1), correlation=0.0)
    assert abs(np.corrcoef(np.log(v1), np.log(v2))[0, 1]) < 0.01
    assert v2.mean() == pytest.approx(0.55, rel=0.01)
    v1, v2, _ = un.sample_zone(zone, 200_000, np.random.default_rng(1), correlation=0.5)
    assert np.corrcoef(np.log(v1), np.log(v2))[0, 1] == pytest.approx(0.5, abs=0.01)


def test_longitudinal_unit_strain_scales_with_volume_loss(scn, by_id):
    b = by_id["B11"]
    unit = un.longitudinal_unit_strain(scn, b, [0.5])
    assert unit.shape == (2, 1) and (unit > 0).all()
    t = sc.tunnels_at(scn, b.chainage, K=0.5, VL=2.0)[0]
    direct, _ = sc.worst_longitudinal(b, t)
    assert direct == pytest.approx(2.0 * unit[0, 0], rel=1e-9)


@pytest.fixture(scope="module")
def mc(scn, buildings):
    return un.monte_carlo(scn, buildings, 300, seed=7)


def test_monte_carlo_is_reproducible(scn, by_id):
    some = [by_id["B09"], by_id["B04"]]
    first = un.monte_carlo(scn, some, 120, seed=7)
    assert un.monte_carlo(scn, some, 120, seed=7) == first
    other = un.monte_carlo(scn, some, 120, seed=8)
    assert other["buildings"]["B09"]["eps_max_pct"] != first["buildings"]["B09"]["eps_max_pct"]


def test_monte_carlo_structure_and_monotonic_probabilities(mc, buildings):
    assert mc["n"] == 300 and set(mc["buildings"]) == {b.id for b in buildings}
    for b in buildings:
        row = mc["buildings"][b.id]
        s = row["settlement_mm"]
        assert s["p05"] <= s["p50"] <= s["p95"]
        if b.foundation == "shallow":
            ps = [row["p_stage2_category_at_least"][c]["p"] for c in "1234"]
            pf = [row["p_final_category_at_least"][c]["p"] for c in "1234"]
            assert ps == sorted(ps, reverse=True) and pf == sorted(pf, reverse=True)
            assert all(f >= s2 for f, s2 in zip(pf, ps))  # indices can only raise the category
            for c in "1234":
                p = row["p_stage2_category_at_least"][c]
                assert p["lo95"] <= p["p"] <= p["hi95"] and p["count"] == round(p["p"] * 300)
        else:
            assert "p_stage2_category_at_least" not in row


def test_monte_carlo_sensitivity_indices_shift_final_category(mc):
    # B06 carries a structural-form index of 2, so its final category is always at least 2
    assert mc["buildings"]["B06"]["p_final_category_at_least"]["2"]["p"] == 1.0
    assert mc["buildings"]["B06"]["p_stage2_category_at_least"]["2"]["p"] < 0.2
    # B05 carries a heritage index of 1
    assert mc["buildings"]["B05"]["p_final_category_at_least"]["1"]["p"] == 1.0


def test_monte_carlo_matches_a_direct_recalculation_of_one_sample(scn, by_id):
    """Re-run the Stage 2 cases for the first random draw and compare with a one-sample run."""
    b = by_id["B09"]
    out = un.monte_carlo(scn, [b], 1, seed=11, n_k_grid=41)
    rng = np.random.default_rng(11)
    draws = {z["id"]: un.sample_zone(z, 1, rng, 1.0) for z in scn["ground_zones"]}
    v1, v2, k = (float(a[0]) for a in draws["Z3"])
    ts = sc.tunnels_at(scn, b.chainage, K=k, VL=[v1, v2], second_factor=1.0)
    cases = sc.stage2_cases(b, ts, n_grid=41)
    eps = max(c[0] for c in cases.values())
    assert out["buildings"]["B09"]["eps_max_pct"]["p50"] == pytest.approx(eps * 100, rel=2e-3)
    smax, _ = sc.footprint_extremes(b, ts, n=101)
    assert out["buildings"]["B09"]["settlement_mm"]["p50"] == pytest.approx(smax * 1000, rel=1e-9)


def test_zone_exceedance_probabilities(scn, by_id):
    out = un.monte_carlo(scn, [by_id["B12"]], 20_000, seed=3)  # a piled building: no Stage 2 loop, so this is quick
    for z in scn["ground_zones"]:
        row = out["zones"][z["id"]]
        mu, sigma = un.lognormal_params(z["observed_vl"]["mean"], z["observed_vl"]["characteristic_95"])
        from scipy.stats import norm

        exact = norm.sf((math.log(z["design"]["VL"]) - mu) / sigma)
        assert row["p_vl_exceeds_design"]["lo95"] <= exact <= row["p_vl_exceeds_design"]["hi95"] + 1e-4
        assert row["median_vl_pct"] == pytest.approx(math.exp(mu))
        assert row["p_vl_exceeds_upper_bound"]["p"] >= row["p_vl_exceeds_design"]["p"]


def test_stage1_probability_uses_both_criteria(scn, by_id):
    out = un.monte_carlo(scn, [by_id["B13"], by_id["B07"]], 200, seed=5)
    assert out["buildings"]["B13"]["p_stage1_not_negligible"]["p"] == 0.0  # far from the tunnels
    assert 0.2 < out["buildings"]["B07"]["p_stage1_not_negligible"]["p"] < 0.9


def test_tornado(scn, by_id):
    t = un.tornado(scn, by_id["B06"])
    names = [b["input"] for b in t["bars"]]
    assert len(names) == 7 and names[0].startswith("Volume loss")
    swings = [b["swing_pct"] for b in t["bars"]]
    assert swings == sorted(swings, reverse=True)
    for b in t["bars"]:
        assert b["swing_pct"] == pytest.approx(abs(b["high_pct"] - b["low_pct"]))
    vl = next(b for b in t["bars"] if b["input"].startswith("Volume loss"))
    assert vl["low_pct"] < t["base_eps_max_pct"] < vl["high_pct"]
    k = next(b for b in t["bars"] if b["input"].startswith("Trough width"))
    assert k["low_pct"] > k["high_pct"]  # a narrower trough (small K) strains the building more
    z = next(b for b in t["bars"] if b["input"].startswith("Depth"))
    assert z["low_pct"] > z["high_pct"]  # a shallower tunnel strains the building more
    f = next(b for b in t["bars"] if "second drive" in b["input"])
    assert f["low_pct"] == pytest.approx(t["base_eps_max_pct"]) and f["high_pct"] > f["low_pct"]
    na = next(b for b in t["bars"] if b["input"].startswith("Neutral axis"))
    assert na["low_pct"] == pytest.approx(t["base_eps_max_pct"])
    mu, sigma = un.lognormal_params(0.70, 1.5)
    assert t["base_VL_pct"] == pytest.approx(math.exp(mu)) and t["base_K"] == pytest.approx(0.525)
    # the base value is the Stage 2 result for the base inputs
    b6 = by_id["B06"]
    cases = sc.stage2_cases(b6, sc.tunnels_at(scn, b6.chainage, K=0.525, VL=math.exp(mu)), n_grid=81)
    assert t["base_eps_max_pct"] == pytest.approx(max(c[0] for c in cases.values()) * 100, rel=1e-3)
    assert bld.damage_category(t["base_eps_max_pct"] / 100) == 0


def _scenario_copy(scn, **zone3_vl):
    import copy

    s = copy.deepcopy(scn)
    s["ground_zones"][2]["observed_vl"].update(zone3_vl)
    return s


def test_probabilities_are_recounted_from_the_samples(scn, by_id):
    """Every probability is the fraction of samples meeting the criterion (counted again here)."""
    s = _scenario_copy(scn, mean=1.5, characteristic_95=3.0)  # larger losses so that categories 1 to 3 occur
    b = by_id["B09"]
    row = un.monte_carlo(s, [b], 80, seed=4, keep_samples=True)["buildings"]["B09"]
    sm = row["samples"]
    cats = np.array([bld.damage_category(e) for e in sm["eps"]])
    assert len(set(cats)) >= 3
    for c in (1, 2, 3, 4):
        assert row["p_stage2_category_at_least"][str(c)]["count"] == int(np.sum(cats >= c))
        assert row["p_final_category_at_least"][str(c)]["count"] == int(np.sum(np.minimum(5, cats + sum(b.indices)) >= c))
    fail = (sm["smax"] >= bld.STAGE1_SETTLEMENT_M) | (sm["slope"] >= bld.STAGE1_SLOPE)
    assert row["p_stage1_not_negligible"]["count"] == int(fail.sum())
    assert row["settlement_mm"]["p95"] == pytest.approx(float(np.percentile(sm["smax"], 95)) * 1000)
    assert row["eps_max_pct"]["p50"] == pytest.approx(float(np.median(sm["eps"])) * 100)
    # the sampled volume losses follow the zone's lognormal; K stays in its range
    assert sm["k"].min() >= 0.45 and sm["k"].max() <= 0.65 and np.array_equal(sm["vl1"], sm["vl2"])


def test_second_drive_factor_is_applied_in_the_monte_carlo(scn, by_id):
    import copy

    s = copy.deepcopy(scn)
    s["alignment"]["second_tunnel_vl_factor"] = 1.5
    b = by_id["B10"]
    one = un.monte_carlo(scn, [b], 30, seed=9, keep_samples=True)["buildings"]["B10"]["samples"]
    more = un.monte_carlo(s, [b], 30, seed=9, keep_samples=True)["buildings"]["B10"]["samples"]
    assert np.allclose(more["vl2"], 1.5 * one["vl2"], rtol=1e-12) and np.array_equal(more["vl1"], one["vl1"])
    assert np.all(more["smax"] > one["smax"])
    ts = sc.tunnels_at(s, b.chainage, K=float(more["k"][0]), VL=[float(more["vl1"][0]), float(one["vl2"][0])])
    assert ts[1].VL == pytest.approx(1.5 * float(one["vl2"][0]))  # same factor as the deterministic engine


def test_monte_carlo_includes_the_longitudinal_case(scn, by_id):
    """B11 is governed by the face passing (design basis); the sampled strain must include that case."""
    b = by_id["B11"]
    out = un.monte_carlo(scn, [b], 1, seed=11, n_k_grid=41, keep_samples=True)
    sm = out["buildings"]["B11"]["samples"]
    ts = sc.tunnels_at(scn, b.chainage, K=float(sm["k"][0]), VL=[float(sm["vl1"][0]), float(sm["vl2"][0])], second_factor=1.0)
    cases = sc.stage2_cases(b, ts, n_grid=41)
    longitudinal = max(v[0] for k, v in cases.items() if k.startswith("longitudinal"))
    transverse = max(v[0] for k, v in cases.items() if k.startswith("transverse"))
    assert longitudinal > 1.05 * transverse  # so that leaving the longitudinal case out would be seen
    assert float(sm["eps"][0]) == pytest.approx(longitudinal, rel=5e-3)


def test_tornado_volume_loss_range_is_the_5th_to_95th_percentile(scn, by_id):
    t = un.tornado(scn, by_id["B06"])
    mu, sigma = un.lognormal_params(0.70, 1.5)
    lo, hi = t["VL_range_pct"]
    assert hi == pytest.approx(1.5, rel=1e-12)  # the 95th percentile is the characteristic value of S5
    assert lo == pytest.approx(math.exp(mu - un.Z95 * sigma), rel=1e-12)
    vl = next(b for b in t["bars"] if b["input"].startswith("Volume loss"))
    assert vl["range"] == f"{lo:.2f}% to {hi:.2f}%"


def test_sampled_slope_is_the_deterministic_footprint_slope(scn, by_id):
    """Each sampled slope and settlement is the Stage 1 value of footprint_extremes for the sampled inputs
    (that function is checked case by case in test_screening.py)."""
    for bid in ("B01", "B07", "B11", "B04"):
        b = by_id[bid]
        sm = un.monte_carlo(scn, [b], 3, seed=21, keep_samples=True)["buildings"][bid]["samples"]
        for j in range(3):
            ts = sc.tunnels_at(scn, b.chainage, K=float(sm["k"][j]), VL=[float(sm["vl1"][j]), float(sm["vl2"][j])], second_factor=1.0)
            smax, slope = sc.footprint_extremes(b, ts, n=101)
            assert float(sm["slope"][j]) == pytest.approx(slope, rel=1e-9) and float(sm["smax"][j]) == pytest.approx(smax, rel=1e-9)
