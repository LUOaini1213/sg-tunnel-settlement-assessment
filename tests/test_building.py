"""Deep-beam equations, strain combination, damage categories and the facade partition."""
import math

import numpy as np
import pytest

from tunnelscreen import building as bld
from tunnelscreen import greenfield as gf
from tunnelscreen import screening as sc

T = gf.Tunnel(0.0, 20.0, 6.0, 1.5, 0.5)
I = 10.0
SMAX = gf.s_max(1.5, 6.0, 0.5, 20.0)


# ----------------------------------------------------------------------------- beam theory
def test_section_properties():
    assert bld.section_properties(6.0, "sagging") == (6.0 ** 3 / 12, 3.0)
    assert bld.section_properties(6.0, "hogging") == (6.0 ** 3 / 3, 6.0)
    assert bld.section_properties(6.0, "hogging", "middle") == (6.0 ** 3 / 12, 3.0)
    with pytest.raises(ValueError):
        bld.section_properties(6.0, "twisting")


def test_hogging_second_moment_is_about_the_bottom_edge():
    # parallel-axis theorem: H^3/12 + H (H/2)^2 = H^3/3
    H = 7.0
    assert bld.section_properties(H, "hogging")[0] == pytest.approx(H ** 3 / 12 + H * (H / 2) ** 2)


@pytest.mark.parametrize("mode", ["sagging", "hogging"])
@pytest.mark.parametrize("L,H,eg", [(12.0, 8.0, 2.6), (30.0, 6.0, 12.5), (5.0, 15.0, 2.6)])
def test_beam_factors_from_first_principles(mode, L, H, eg):
    """Centrally loaded Timoshenko beam of unit thickness (the model behind S3 Eq. 7-8).

    Mid-span deflection = P L^3 / (48 E I) + 3 P L / (8 G H) (shear area H / 1.5).
    Extreme-fibre bending strain = (P L / 4) t / (E I). Diagonal strain = half the largest shear
    strain = (1.5 (P/2) / H) / (2 G).
    """
    P, E = 1000.0, 2.0e6
    G = E / eg
    Isec, t = bld.section_properties(H, mode)
    delta = P * L ** 3 / (48 * E * Isec) + 3 * P * L / (8 * G * H)
    eps_b = (P * L / 4) * t / (E * Isec)
    eps_d = (1.5 * (P / 2) / H) / G / 2
    assert bld.bending_factor(L, H, eg, mode) == pytest.approx((delta / L) / eps_b, rel=1e-12)
    assert bld.diagonal_factor(L, H, eg, mode) == pytest.approx((delta / L) / eps_d, rel=1e-12)


def test_beam_factor_hand_values_for_l_over_h_one():
    # hogging, L = H = 1, E/G = 2.6: 1/12 + 3 (1/3) 2.6 / 2 = 1.38333; 1 + 1/(18 (1/3) 2.6) = 1.064103
    assert bld.bending_factor(1.0, 1.0, 2.6, "hogging") == pytest.approx(1.383333, abs=1e-6)
    assert bld.diagonal_factor(1.0, 1.0, 2.6, "hogging") == pytest.approx(1.064103, abs=1e-6)
    # sagging, L = H = 1: 1/6 + 3 (1/12) 2.6 / 1 = 0.816667; 1 + 1/(18 (1/12) 2.6) = 1.256410
    assert bld.bending_factor(1.0, 1.0, 2.6, "sagging") == pytest.approx(0.816667, abs=1e-6)
    assert bld.diagonal_factor(1.0, 1.0, 2.6, "sagging") == pytest.approx(1.256410, abs=1e-6)


def _strain_per_unit_deflection_ratio(L, H, mode):
    return max(1.0 / bld.bending_factor(L, H, 2.6, mode), 1.0 / bld.diagonal_factor(L, H, 2.6, mode))


def test_hogging_against_sagging_for_the_same_deflection_ratio():
    """S3 section 4.2 says hogging "is likely to be more damaging" for a given Delta/L.

    From Eq. 7-8 alone (no horizontal strain) that holds for long spans, L/H of about 3 and more.
    For short spans (L/H of 1 or 2) the sagging beam, with its neutral axis at mid-height, has the
    larger bending strain. The expected values below were worked by hand from Eq. 7-8.
    """
    assert _strain_per_unit_deflection_ratio(10.0, 10.0, "sagging") == pytest.approx(1 / 0.816667, rel=1e-5)
    assert _strain_per_unit_deflection_ratio(10.0, 10.0, "hogging") == pytest.approx(1 / 1.064103, rel=1e-5)
    assert _strain_per_unit_deflection_ratio(30.0, 10.0, "sagging") == pytest.approx(1 / 0.716667, rel=1e-5)
    assert _strain_per_unit_deflection_ratio(30.0, 10.0, "hogging") == pytest.approx(1 / 0.683333, rel=1e-5)
    for ratio in (3.0, 5.0, 8.0):
        assert _strain_per_unit_deflection_ratio(ratio * 8.0, 8.0, "hogging") > _strain_per_unit_deflection_ratio(ratio * 8.0, 8.0, "sagging")
    for ratio in (1.0, 2.0):
        assert _strain_per_unit_deflection_ratio(ratio * 8.0, 8.0, "hogging") < _strain_per_unit_deflection_ratio(ratio * 8.0, 8.0, "sagging")


def test_framed_buildings_are_less_strained_in_shear():
    # larger E/G (more flexible in shear) lowers the bending strain for a given deflection ratio
    assert bld.bending_factor(10.0, 10.0, 12.5, "hogging") > bld.bending_factor(10.0, 10.0, 2.6, "hogging")
    assert bld.E_OVER_G == {"masonry": 2.6, "framed": 12.5}


# ----------------------------------------------------------------------------- strain combination
def test_total_bending_strain_adds_horizontal_strain():
    assert bld.total_bending_strain(0.0004, 0.0003) == pytest.approx(0.0007)
    assert bld.total_bending_strain(0.0004, -0.0003) == pytest.approx(0.0001)  # compression relieves it


@pytest.mark.parametrize("eps_d,eps_h,nu", [(0.0005, 0.0003, 0.3), (0.0002, -0.0004, 0.3), (0.0010, 0.0, 0.3), (0.0003, 0.0006, 0.2)])
def test_total_diagonal_strain_is_the_major_principal_strain(eps_d, eps_h, nu):
    """Mohr's circle: plane strain state with eps_x = eps_h, eps_y = -nu eps_h, shear gamma = 2 eps_d."""
    tensor = np.array([[eps_h, eps_d], [eps_d, -nu * eps_h]])
    assert bld.total_diagonal_strain(eps_d, eps_h, nu) == pytest.approx(float(np.linalg.eigvalsh(tensor).max()), rel=1e-12)


def test_total_diagonal_strain_special_cases():
    assert bld.total_diagonal_strain(0.0006, 0.0) == pytest.approx(0.0006)
    assert bld.total_diagonal_strain(0.0, 0.0006) == pytest.approx(0.0006)  # 0.35 + 0.65
    assert bld.total_diagonal_strain(0.0, -0.0006) == pytest.approx(0.3 * 0.0006)  # lateral expansion


# ----------------------------------------------------------------------------- categories
@pytest.mark.parametrize("eps_pct,cat", [(0.0, 0), (0.0499, 0), (0.05, 1), (0.0749, 1), (0.075, 2), (0.1499, 2),
                                          (0.15, 3), (0.2999, 3), (0.3, 4), (1.0, 4)])
def test_damage_category_boundaries(eps_pct, cat):
    assert bld.damage_category(eps_pct / 100.0) == cat


def test_category_names_and_limits_match_lta_table_14_2():
    assert bld.CATEGORY_LIMITS == (0.0005, 0.00075, 0.0015, 0.003)
    assert bld.CATEGORY_NAMES == ("Negligible", "Very Slight", "Slight", "Moderate", "Severe to Very Severe")


def test_final_category_adds_sensitivity_indices_and_caps_at_five():
    assert bld.final_category(1, (0, 0, 0)) == 1
    assert bld.final_category(1, (2, 1, 0)) == 4
    assert bld.final_category(0, (0, 0, 1)) == 1
    assert bld.final_category(4, (2, 2, 2)) == 5


@pytest.mark.parametrize("s_mm,slope,ok", [(9.99, 1 / 501, True), (10.0, 1 / 1000, False), (5.0, 1 / 500, False),
                                            (5.0, 1 / 499, False), (12.0, 1 / 2000, False), (0.0, 0.0, True)])
def test_stage1_needs_both_settlement_and_slope_below_the_limits(s_mm, slope, ok):
    assert bld.stage1_negligible(s_mm / 1000.0, slope) is ok


# ----------------------------------------------------------------------------- facade partition
def test_inflexion_points_of_a_single_trough():
    prof = sc.transverse_profile([T])
    roots = bld.inflexion_points(prof, -30.0, 30.0)
    assert roots == pytest.approx([-I, I], abs=1e-7)
    assert bld.inflexion_points(prof, 12.0, 24.0) == []


def test_hand_worked_hogging_span():
    """Facade from the point of inflexion (10 m) to the trough limit (25 m), H = 6 m, masonry."""
    r = bld.assess_facade(sc.transverse_profile([T]), 10.0, 25.0, 6.0, 2.6, n_grid=2001)
    assert len(r.segments) == 1
    sg = r.segments[0]
    assert (sg.mode, sg.L) == ("hogging", pytest.approx(15.0))
    s10, s25 = SMAX * math.exp(-0.5), SMAX * math.exp(-3.125)
    # relative deflection: largest gap between the profile and the chord, found here by brute force
    ys = np.linspace(10.0, 25.0, 1_000_001)
    gap = np.abs(SMAX * np.exp(-ys ** 2 / 200.0) - (s10 + (s25 - s10) * (ys - 10.0) / 15.0))
    assert sg.delta == pytest.approx(float(gap.max()), rel=1e-6)
    # average horizontal strain from the end movements: u = -(y/z0) S
    eps_h = ((-25.0 / 20.0) * s25 - (-10.0 / 20.0) * s10) / 15.0
    assert sg.eps_h == pytest.approx(eps_h, rel=1e-12)
    assert sg.eps_h > 0  # tension in the hogging zone
    Isec, t = 6.0 ** 3 / 3, 6.0
    eps_b = (sg.delta / 15.0) / (15.0 / (12 * t) + 3 * Isec * 2.6 / (2 * t * 15.0 * 6.0))
    eps_d = (sg.delta / 15.0) / (1 + 6.0 * 15.0 ** 2 / (18 * Isec * 2.6))
    assert sg.eps_b == pytest.approx(eps_b, rel=1e-12)
    assert sg.eps_d == pytest.approx(eps_d, rel=1e-12)
    assert sg.eps_bt == pytest.approx(eps_h + eps_b, rel=1e-12)
    assert sg.eps_dt == pytest.approx(0.35 * eps_h + math.hypot(0.65 * eps_h, eps_d), rel=1e-12)
    assert r.eps_max == pytest.approx(max(sg.eps_bt, sg.eps_dt))
    assert r.category == bld.damage_category(r.eps_max)


def test_facade_is_cut_at_inflexion_points_and_at_the_trough_limit():
    r = bld.assess_facade(sc.transverse_profile([T]), -40.0, 40.0, 8.0, 2.6)
    assert [s.mode for s in r.segments] == ["hogging", "sagging", "hogging"]
    assert [(s.a, s.b) for s in r.segments] == [pytest.approx((-25.0, -10.0)), pytest.approx((-10.0, 10.0)), pytest.approx((10.0, 25.0))]
    assert r.segments[1].eps_h < 0 < r.segments[0].eps_h  # compression in sagging, tension in hogging
    assert r.segments[0].eps_max == pytest.approx(r.segments[2].eps_max, rel=1e-9)  # symmetry
    assert r.governing.startswith("hogging")


def test_sagging_span_values():
    r = bld.assess_facade(sc.transverse_profile([T]), -10.0, 10.0, 8.0, 2.6, n_grid=4001)
    sg = r.segments[0]
    assert sg.mode == "sagging"
    assert sg.delta == pytest.approx(SMAX * (1 - math.exp(-0.5)), rel=1e-9)  # centre minus the ends
    # average compression over the whole sagging zone: (u(i) - u(-i)) / 2i = -exp(-1/2) Smax / z0
    assert sg.eps_h == pytest.approx(-math.exp(-0.5) * SMAX / 20.0, rel=1e-12)
    assert sg.deflection_ratio == pytest.approx(sg.delta / 20.0)


def test_building_outside_the_trough_has_no_strain():
    r = bld.assess_facade(sc.transverse_profile([T]), 26.0, 60.0, 8.0, 2.6)
    assert (r.eps_max, r.category, r.governing, r.segments) == (0.0, 0, "outside trough", [])


def test_mirrored_building_gives_the_same_result():
    a = bld.assess_facade(sc.transverse_profile([T]), 4.0, 22.0, 9.0, 2.6)
    b = bld.assess_facade(sc.transverse_profile([T]), -22.0, -4.0, 9.0, 2.6)
    assert a.eps_max == pytest.approx(b.eps_max, rel=1e-9)
    assert [s.mode for s in a.segments] == ["sagging", "hogging"] and [s.mode for s in b.segments] == ["hogging", "sagging"]


def test_strain_is_proportional_to_volume_loss_for_one_tunnel():
    a = bld.assess_facade(sc.transverse_profile([T]), 2.0, 24.0, 9.0, 2.6).eps_max
    b = bld.assess_facade(sc.transverse_profile([T.with_(VL=3.0)]), 2.0, 24.0, 9.0, 2.6).eps_max
    assert b == pytest.approx(2 * a, rel=1e-9)


def test_twin_tunnel_profile_is_partitioned_at_every_inflexion_point():
    wide = [T.with_(y0=-25.0), T.with_(y0=25.0)]  # two separate troughs: four points of inflexion
    prof = sc.transverse_profile(wide)
    roots = bld.inflexion_points(prof, -60.0, 60.0)
    assert len(roots) == 4
    assert np.allclose(gf.curvature(np.array(roots), wide), 0.0, atol=1e-12)
    r = bld.assess_facade(prof, -60.0, 60.0, 8.0, 2.6)
    assert [s.mode for s in r.segments] == ["hogging", "sagging", "hogging", "sagging", "hogging"]
    close = [T.with_(y0=-7.5), T.with_(y0=7.5)]  # troughs merge: two points of inflexion
    assert len(bld.inflexion_points(sc.transverse_profile(close), -60.0, 60.0)) == 2


def test_neutral_axis_option_changes_only_hogging():
    prof = sc.transverse_profile([T])
    hog_b = bld.assess_facade(prof, 10.0, 25.0, 6.0, 2.6).eps_max
    hog_m = bld.assess_facade(prof, 10.0, 25.0, 6.0, 2.6, hogging_neutral_axis="middle").eps_max
    assert hog_b != pytest.approx(hog_m, rel=1e-3)
    sag_b = bld.assess_facade(prof, -10.0, 10.0, 6.0, 2.6).segments[0]
    sag_m = bld.assess_facade(prof, -10.0, 10.0, 6.0, 2.6, hogging_neutral_axis="middle").segments[0]
    assert sag_b.eps_bt == sag_m.eps_bt


def test_longitudinal_facade_ahead_of_the_face_is_hogging_and_in_tension():
    prof = sc.longitudinal_profile(T, 0.0, x_face=0.0)
    r = bld.assess_facade(prof, 0.0, 20.0, 8.0, 2.6)
    assert [s.mode for s in r.segments] == ["hogging"]
    assert r.segments[0].eps_h > 0
    behind = bld.assess_facade(prof, -20.0, 0.0, 8.0, 2.6)
    assert [s.mode for s in behind.segments] == ["sagging"] and behind.segments[0].eps_h < 0
    assert prof.limits == pytest.approx((-25.0, 25.0))
