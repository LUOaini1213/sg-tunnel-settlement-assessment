"""Published values reproduced by the code, each with its stated tolerance.

Source IDs are those of data/sources.csv. The same checks are written to results/verification.json.
"""
import math

import numpy as np
import pytest

from tunnelscreen import building as bld
from tunnelscreen import greenfield as gf
from tunnelscreen import uncertainty as un

T = gf.Tunnel(0.0, 20.0, 6.0, 1.5, 0.5)
I = 10.0
SMAX = gf.s_max(1.5, 6.0, 0.5, 20.0)


def test_lta_formula_coefficient_0_0031():
    """S1 Cl. 14.2.1.2: Smax = 0.0031 V D^2 / i with V in per cent. Exact value sqrt(pi/2)/400."""
    coeff = SMAX * I / (1.5 * 6.0 ** 2)
    assert coeff == pytest.approx(math.sqrt(math.pi / 2) / 400)
    assert coeff == pytest.approx(0.0031, rel=0.011)
    assert round(coeff, 4) == 0.0031
    # and for the SYNTHETIC tunnel size, the LTA formula agrees within its rounding
    for vl, d, k, z0 in ((1.0, 6.6, 0.45, 22.0), (3.0, 6.6, 0.5, 20.0)):
        assert gf.s_max(vl, d, k, z0) == pytest.approx(0.0031 * vl * d ** 2 / (k * z0), rel=0.011)


def test_mair_taylor_burland_eq5_coefficient_0_31():
    """S3 Eq. 5: Smax = 0.31 Vl D^2 / (K z0) with Vl as a fraction."""
    assert SMAX * (0.5 * 20.0) / (0.015 * 36.0) == pytest.approx(0.31, rel=0.011)


def test_peck_fig8_properties_of_the_error_function():
    """S9 Fig. 8 and section 1.4.6: 0.61 at i, 0.22 at sqrt(3) i, volume about 2.5 i Smax."""
    assert float(gf.settlement_one(I, T)) / SMAX == pytest.approx(0.61, abs=0.005)
    assert float(gf.settlement_one(math.sqrt(3) * I, T)) / SMAX == pytest.approx(0.22, abs=0.005)
    ys = np.linspace(-120, 120, 240001)
    assert float(np.trapezoid(gf.settlement_one(ys, T), ys)) / (I * SMAX) == pytest.approx(2.5, rel=0.003)


def test_max_horizontal_movement_0_61_k_smax():
    """S4 section 5.5: Sh,max = 0.61 K Smax at the point of inflexion."""
    ys = np.linspace(0, 50, 500001)
    u = np.abs(gf.horizontal_displacement_one(ys, T))
    assert ys[int(np.argmax(u))] == pytest.approx(I, abs=1e-3)
    assert float(u.max()) / (0.5 * SMAX) == pytest.approx(0.61, abs=0.005)


def test_half_smax_above_the_face():
    """S3 section 2.3."""
    assert float(gf.settlement_advancing(0.0, 0.0, T, 0.0)) == pytest.approx(0.5 * SMAX, rel=1e-12)


def test_longitudinal_displacement_at_face_matches_s6_eq_2_17():
    """S6 Eq. 2.17 at the face on the centre line: VL D^2 / (8 z0)."""
    assert abs(float(gf.long_displacement_advancing(0.0, 0.0, T, 0.0))) == pytest.approx(0.015 * 36.0 / (8 * 20.0), rel=1e-12)


def test_subsurface_relation_gives_k_half_at_surface():
    """S4 Eq. 8 at z = 0."""
    assert gf.subsurface_trough_width(0.0, 23.0) / 23.0 == pytest.approx(0.5)


@pytest.mark.parametrize("limit,read_from_chart", [(0.05, 0.05), (0.075, 0.08), (0.15, 0.16), (0.3, 0.32)])
def test_damage_chart_intercepts_on_the_deflection_axis(limit, read_from_chart):
    """S3 Fig. 4 (Burland 1995), L/H = 1, hogging, E/G = 2.6: Delta/L at zero horizontal strain.

    Tolerance 0.01 (% deflection ratio) is the accuracy of reading the printed chart.
    """
    dr_bending = limit * bld.bending_factor(1.0, 1.0, 2.6, "hogging")
    dr_diagonal = limit * bld.diagonal_factor(1.0, 1.0, 2.6, "hogging")
    assert min(dr_bending, dr_diagonal) == pytest.approx(read_from_chart, abs=0.01)
    assert dr_diagonal < dr_bending  # diagonal strain governs at L/H = 1


@pytest.mark.parametrize("limit", [0.05, 0.075, 0.15, 0.3])
def test_damage_chart_intercepts_on_the_strain_axis(limit):
    """S3 Fig. 4: with no deflection the boundary is at eps_h equal to the limiting strain."""
    assert bld.total_bending_strain(0.0, limit) == pytest.approx(limit)
    assert bld.total_diagonal_strain(0.0, limit) == pytest.approx(limit)


def test_damage_chart_kink_of_the_category_3_to_4_boundary():
    """S3 Fig. 4: the 0.3% boundary changes from the diagonal to the bending branch near
    eps_h = 0.10%, Delta/L = 0.27% (read from the chart, tolerance 0.01)."""
    from scipy.optimize import brentq

    bf = bld.bending_factor(1.0, 1.0, 2.6, "hogging")
    df = bld.diagonal_factor(1.0, 1.0, 2.6, "hogging")

    def gap(eh):
        dr = (0.3 - eh) * bf  # deflection ratio on the bending branch
        return bld.total_diagonal_strain(dr / df, eh) - 0.3

    eh = brentq(gap, 0.01, 0.29)
    assert eh == pytest.approx(0.10, abs=0.01)
    assert (0.3 - eh) * bf == pytest.approx(0.27, abs=0.01)


def test_eq10_coefficients_0_35_and_0_65():
    """S3 Eq. 10 for Poisson's ratio 0.3."""
    eh, ed = 0.0010, 0.0010
    assert bld.total_diagonal_strain(ed, eh) == pytest.approx(0.35 * eh + math.sqrt((0.65 * eh) ** 2 + ed ** 2), rel=1e-12)
    # the misprinted form in S6 Eq. 2.25 ((1 - nu)/2 inside the root) gives a different number
    assert bld.total_diagonal_strain(ed, eh) != pytest.approx(0.35 * eh + math.sqrt((0.35 * eh) ** 2 + ed ** 2), rel=1e-3)


def test_general_form_of_s8_eq52_reduces_to_eq10():
    """S8 Eq. 52: eps_h (1 - E/4G) + sqrt(eps_h^2 (E/G)^2 / 16 + eps_d^2), with E/G = 2.6."""
    eh, ed, eg = 0.0007, 0.0004, 2.6
    assert bld.total_diagonal_strain(ed, eh, nu=0.3) == pytest.approx(eh * (1 - eg / 4) + math.sqrt(eh ** 2 * eg ** 2 / 16 + ed ** 2), rel=1e-12)


def test_characteristic_volume_loss_worked_example():
    """S5 section 4.5.1: mean 0.30%, standard deviation 0.24%, t = 1.64485 gives 0.7%."""
    assert un.characteristic_value(0.30, 0.24, 1.64485) == pytest.approx(0.7, abs=0.05)
    assert round(un.characteristic_value(0.30, 0.24, 1.64485), 1) == 0.7


def test_singapore_observations_are_consistent_with_the_engine():
    """S7 section 3.1: in Old Alluvium a single 6.6 m tunnel gave 2 to 8 mm above the centre line
    and volume losses of 0.2% to 1.2% at axis depths of 16 to 44 m. The Gaussian formula with
    K = 0.45 must be able to produce that settlement range from that volume-loss range."""
    lo = gf.s_max(0.2, 6.6, 0.45, 44.0) * 1000
    hi = gf.s_max(1.2, 6.6, 0.45, 16.0) * 1000
    assert lo < 2.0 and hi > 8.0
