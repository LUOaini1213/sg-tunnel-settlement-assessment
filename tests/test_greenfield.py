"""Closed-form identities and sign conventions of the greenfield engine."""
import math

import numpy as np
import pytest

from tunnelscreen import greenfield as gf

T = gf.Tunnel(y0=0.0, z0=20.0, D=6.0, VL=1.5, K=0.5)
I = 10.0  # K z0
VS = 0.015 * math.pi * 36.0 / 4.0  # 0.424115 m3 per m, worked by hand
SMAX = VS / (math.sqrt(2 * math.pi) * I)  # 16.92 mm


def test_hand_worked_example():
    assert gf.trough_width(0.5, 20.0) == 10.0
    assert gf.trough_volume(1.5, 6.0) == pytest.approx(0.4241150, abs=1e-6)
    assert gf.s_max(1.5, 6.0, 0.5, 20.0) * 1000 == pytest.approx(16.9197, abs=1e-3)
    assert float(gf.settlement_one(0.0, T)) == pytest.approx(SMAX)
    assert float(gf.settlement_one(10.0, T)) * 1000 == pytest.approx(16.9197 * math.exp(-0.5), abs=1e-3)
    assert float(gf.settlement_one(25.0, T)) * 1000 == pytest.approx(16.9197 * math.exp(-3.125), abs=1e-3)


def test_volume_loss_is_percent_of_face_area():
    # doubling VL doubles the trough; VL = 100% would give the whole face area
    assert gf.trough_volume(100.0, 6.0) == pytest.approx(math.pi * 9.0)
    assert gf.s_max(3.0, 6.0, 0.5, 20.0) == pytest.approx(2 * SMAX)


def test_trough_volume_equals_volume_loss_numerically():
    ys = np.linspace(-150, 150, 300001)
    assert float(np.trapezoid(gf.settlement_one(ys, T), ys)) == pytest.approx(VS, rel=1e-9)


def test_trough_is_centred_on_the_tunnel_and_symmetric():
    t = T.with_(y0=7.5)
    ys = np.linspace(-40, 60, 2001)
    s = gf.settlement_one(ys, t)
    assert ys[int(np.argmax(s))] == pytest.approx(7.5)
    assert float(gf.settlement_one(7.5 + 4.0, t)) == pytest.approx(float(gf.settlement_one(7.5 - 4.0, t)))


def test_smax_inversely_proportional_to_depth_and_k():
    assert gf.s_max(1.5, 6.0, 0.5, 40.0) == pytest.approx(SMAX / 2)
    assert gf.s_max(1.5, 6.0, 0.25, 20.0) == pytest.approx(SMAX * 2)
    assert gf.s_max(1.5, 12.0, 0.5, 20.0) == pytest.approx(SMAX * 4)


def test_slope_is_derivative_of_settlement():
    ys = np.linspace(-35, 35, 141)
    h = 1e-5
    num = (gf.settlement_one(ys + h, T) - gf.settlement_one(ys - h, T)) / (2 * h)
    assert np.allclose(gf.slope_one(ys, T), num, atol=1e-9)
    # settlement decreases away from the tunnel on both sides
    assert float(gf.slope_one(5.0, T)) < 0 < float(gf.slope_one(-5.0, T))


def test_max_slope_is_at_the_point_of_inflexion():
    ys = np.linspace(0, 40, 400001)
    sl = np.abs(gf.slope_one(ys, T))
    assert ys[int(np.argmax(sl))] == pytest.approx(I, abs=1e-3)
    assert float(sl.max()) == pytest.approx(math.exp(-0.5) * SMAX / I, rel=1e-9)
    assert gf.max_slope_single(T) == pytest.approx(math.exp(-0.5) * SMAX / I)


def test_curvature_is_second_derivative_and_changes_sign_at_i():
    ys = np.linspace(-35, 35, 141)
    h = 1e-4
    num = (gf.settlement_one(ys + h, T) - 2 * gf.settlement_one(ys, T) + gf.settlement_one(ys - h, T)) / h ** 2
    assert np.allclose(gf.curvature_one(ys, T), num, atol=1e-8)
    assert float(gf.curvature_one(0.0, T)) == pytest.approx(-SMAX / I ** 2)  # sagging at the centre
    assert float(gf.curvature_one(I, T)) == pytest.approx(0.0, abs=1e-15)
    assert float(gf.curvature_one(1.5 * I, T)) > 0  # hogging outside the point of inflexion
    # largest hogging curvature at sqrt(3) i, equal to 2 exp(-3/2) Smax / i^2
    yy = np.linspace(I, 4 * I, 300001)
    k = gf.curvature_one(yy, T)
    assert yy[int(np.argmax(k))] == pytest.approx(math.sqrt(3) * I, abs=1e-3)
    assert float(k.max()) == pytest.approx(2 * math.exp(-1.5) * SMAX / I ** 2, rel=1e-8)


def test_horizontal_displacement_points_towards_the_tunnel():
    assert float(gf.horizontal_displacement_one(8.0, T)) < 0  # ground on the +y side moves in -y
    assert float(gf.horizontal_displacement_one(-8.0, T)) > 0
    assert float(gf.horizontal_displacement_one(0.0, T)) == 0.0
    # vector directed at the tunnel axis: |u| / S = |y| / z0
    assert abs(float(gf.horizontal_displacement_one(8.0, T))) / float(gf.settlement_one(8.0, T)) == pytest.approx(8.0 / 20.0)
    t = T.with_(y0=7.5)
    assert float(gf.horizontal_displacement_one(7.5, t)) == 0.0
    assert float(gf.horizontal_displacement_one(9.5, t)) == pytest.approx(-(2.0 / 20.0) * float(gf.settlement_one(9.5, t)))


def test_horizontal_strain_sign_zero_and_extremes():
    ys = np.linspace(-45, 45, 181)
    h = 1e-5
    num = (gf.horizontal_displacement_one(ys + h, T) - gf.horizontal_displacement_one(ys - h, T)) / (2 * h)
    assert np.allclose(gf.horizontal_strain_one(ys, T), num, atol=1e-10)
    assert float(gf.horizontal_strain_one(0.0, T)) == pytest.approx(-SMAX / 20.0)  # compressive above the tunnel
    assert float(gf.horizontal_strain_one(I, T)) == pytest.approx(0.0, abs=1e-15)
    assert float(gf.horizontal_strain_one(0.5 * I, T)) < 0 < float(gf.horizontal_strain_one(2 * I, T))
    yy = np.linspace(I, 4 * I, 300001)
    e = gf.horizontal_strain_one(yy, T)
    assert yy[int(np.argmax(e))] == pytest.approx(math.sqrt(3) * I, abs=1e-3)
    assert float(e.max()) == pytest.approx(gf.max_tensile_strain_single(T), rel=1e-8)
    assert gf.max_tensile_strain_single(T) == pytest.approx(2 * math.exp(-1.5) * SMAX / 20.0)


def test_superposition_of_twin_tunnels():
    a, b = T.with_(y0=-7.5), T.with_(y0=7.5, VL=3.0)
    ys = np.linspace(-150, 150, 300001)
    s = gf.settlement(ys, [a, b])
    assert np.allclose(s, gf.settlement_one(ys, a) + gf.settlement_one(ys, b))
    assert float(np.trapezoid(s, ys)) == pytest.approx(gf.trough_volume(1.5, 6.0) + gf.trough_volume(3.0, 6.0), rel=1e-9)
    y = np.array([-20.0, 0.0, 3.0, 25.0])
    for total, one in ((gf.slope, gf.slope_one), (gf.curvature, gf.curvature_one),
                       (gf.horizontal_displacement, gf.horizontal_displacement_one), (gf.horizontal_strain, gf.horizontal_strain_one)):
        assert np.allclose(total(y, [a, b]), one(y, a) + one(y, b))
    # the larger volume loss pulls the deepest point towards the second tunnel
    assert ys[int(np.argmax(s))] > 0


def test_trough_limits_are_two_and_a_half_i_outside_the_outer_tunnels():
    a, b = T.with_(y0=-7.5), T.with_(y0=7.5)
    assert gf.trough_limits([a, b]) == pytest.approx((-7.5 - 25.0, 7.5 + 25.0))
    assert float(gf.settlement_one(25.0, T)) / SMAX == pytest.approx(0.044, abs=5e-4)  # S3 section 4.1


def test_longitudinal_trough_development():
    assert float(gf.face_factor(0.0, 0.0, I)) == pytest.approx(0.5)
    assert float(gf.face_factor(0.0, -40.0, I)) < 1e-4  # face still far behind the point
    assert float(gf.face_factor(0.0, 40.0, I)) > 1 - 1e-4  # face well past: fully developed
    xs = np.linspace(-30, 30, 13)
    f = gf.face_factor(0.0, xs, I)
    assert np.all(np.diff(f) > 0)  # settlement grows as the face advances
    assert float(gf.settlement_advancing(0.0, 4.0, T, 60.0)) == pytest.approx(float(gf.settlement_one(4.0, T)), rel=1e-8)
    # one trough-width behind the face 84% has developed, one ahead 16%
    assert float(gf.face_factor(-I, 0.0, I)) == pytest.approx(0.8413447, abs=1e-6)
    assert float(gf.face_factor(I, 0.0, I)) == pytest.approx(0.1586553, abs=1e-6)


def test_longitudinal_derivatives_and_signs():
    xs = np.linspace(-30, 30, 121)
    h = 1e-5
    y, xf = 3.0, 2.0
    num1 = (gf.settlement_advancing(xs + h, y, T, xf) - gf.settlement_advancing(xs - h, y, T, xf)) / (2 * h)
    assert np.allclose(gf.long_slope_advancing(xs, y, T, xf), num1, atol=1e-9)
    num2 = (gf.long_slope_advancing(xs + h, y, T, xf) - gf.long_slope_advancing(xs - h, y, T, xf)) / (2 * h)
    assert np.allclose(gf.long_curvature_advancing(xs, y, T, xf), num2, atol=1e-9)
    num3 = (gf.long_displacement_advancing(xs + h, y, T, xf) - gf.long_displacement_advancing(xs - h, y, T, xf)) / (2 * h)
    assert np.allclose(gf.long_strain_advancing(xs, y, T, xf), num3, atol=1e-10)
    # ahead of the face: ground moves back towards the face, strain tensile, profile hogging
    assert float(gf.long_displacement_advancing(xf + 8, y, T, xf)) < 0
    assert float(gf.long_strain_advancing(xf + 8, y, T, xf)) > 0
    assert float(gf.long_curvature_advancing(xf + 8, y, T, xf)) > 0
    assert float(gf.long_strain_advancing(xf - 8, y, T, xf)) < 0
    assert float(gf.long_curvature_advancing(xf - 8, y, T, xf)) < 0
    # steepest transient slope is at the face: S_trans / (i sqrt(2 pi))
    assert abs(float(gf.long_slope_advancing(xf, y, T, xf))) == pytest.approx(float(gf.settlement_one(y, T)) / (I * math.sqrt(2 * math.pi)))


def test_displacement_rule_is_the_same_in_both_directions():
    # u = (i^2 / z0) grad(S): transverse (S3 Eq. 6) and longitudinal (S6 Eq. 2.17)
    y = 6.0
    assert float(gf.horizontal_displacement_one(y, T)) == pytest.approx(I ** 2 / 20.0 * float(gf.slope_one(y, T)))
    assert float(gf.long_displacement_advancing(4.0, y, T, 1.0)) == pytest.approx(I ** 2 / 20.0 * float(gf.long_slope_advancing(4.0, y, T, 1.0)))


def test_subsurface_trough_width_and_volume():
    assert gf.subsurface_trough_width(0.0, 20.0) == pytest.approx(10.0)
    # i = 0.175 z0 + 0.325 (z0 - z): at z = 10 m, 3.5 + 3.25 = 6.75 m
    assert gf.subsurface_trough_width(10.0, 20.0) == pytest.approx(6.75)
    # K = i / (z0 - z) rises with depth
    ks = [gf.subsurface_trough_width(z, 20.0) / (20.0 - z) for z in (0.0, 5.0, 10.0, 15.0)]
    assert ks == sorted(ks) and ks[0] == pytest.approx(0.5) and ks[2] == pytest.approx(0.675)
    assert gf.subsurface_trough_width(10.0, 20.0, K_surface=0.45) == pytest.approx(6.75 * 0.9)
    ys = np.linspace(-150, 150, 300001)
    s = gf.subsurface_settlement(ys, 10.0, [T])
    assert float(np.trapezoid(s, ys)) == pytest.approx(VS, rel=1e-9)  # same volume at depth
    assert float(s.max()) == pytest.approx(SMAX * 10.0 / 6.75)  # narrower, so deeper
    with pytest.raises(ValueError):
        gf.subsurface_trough_width(20.0, 20.0)
    with pytest.raises(ValueError):
        gf.subsurface_trough_width(-1.0, 20.0)
