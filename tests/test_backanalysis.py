"""Back-analysis: recovery of known parameters, confidence intervals and departure flags."""
import math

import numpy as np
import pytest

from tunnelscreen import backanalysis as ba
from tunnelscreen import greenfield as gf

Z0, D = 22.0, 6.6
Y = np.array([-33.0, -22.0, -16.5, -11.0, -5.5, 0.0, 5.5, 11.0, 16.5, 22.0, 33.0])  # Type E pattern for Z = 22 m


def test_model_is_the_gaussian_trough_in_millimetres():
    assert float(ba.model_mm(np.array([0.0]), 1.0, 0.45, Z0, D)[0]) == pytest.approx(gf.s_max(1.0, D, 0.45, Z0) * 1000)
    assert float(ba.model_mm(np.array([3.0]), 1.0, 0.45, Z0, D, y0=3.0)[0]) == pytest.approx(gf.s_max(1.0, D, 0.45, Z0) * 1000)


@pytest.mark.parametrize("vl,k", [(0.3, 0.48), (2.2, 0.42), (1.0, 0.6)])
def test_noise_free_readings_recover_the_parameters_exactly(vl, k):
    s = ba.model_mm(Y, vl, k, Z0, D)
    fit = ba.fit_gaussian(Y, s, Z0, D, sigma_mm=1.0)
    assert fit.converged and fit.n == 11
    assert (fit.VL, fit.K) == (pytest.approx(vl, rel=1e-6), pytest.approx(k, rel=1e-6))
    assert fit.rms_residual_mm < 1e-6
    ll = ba.fit_loglinear(Y, s, Z0, D)
    assert (ll["VL"], ll["K"]) == (pytest.approx(vl, rel=1e-9), pytest.approx(k, rel=1e-9))
    assert ll["Smax_mm"] == pytest.approx(gf.s_max(vl, D, k, Z0) * 1000)


def test_fit_needs_three_readings():
    with pytest.raises(ValueError):
        ba.fit_gaussian([0.0, 5.0], [10.0, 8.0], Z0, D)


def test_covariance_is_sigma_squared_times_inverse_normal_matrix():
    rng = np.random.default_rng(0)
    s = ba.synthetic_readings(Y, 2.0, 0.45, Z0, D, 1.0, rng)
    fit = ba.fit_gaussian(Y, s, Z0, D, sigma_mm=1.0)
    # numerical Jacobian at the solution
    h = 1e-6
    J = np.column_stack([(ba.model_mm(Y, fit.VL + h, fit.K, Z0, D) - ba.model_mm(Y, fit.VL - h, fit.K, Z0, D)) / (2 * h),
                         (ba.model_mm(Y, fit.VL, fit.K + h, Z0, D) - ba.model_mm(Y, fit.VL, fit.K - h, Z0, D)) / (2 * h)])
    assert np.allclose(fit.cov, np.linalg.inv(J.T @ J), rtol=1e-4)
    # doubling the reading noise doubles the standard errors
    fit2 = ba.fit_gaussian(Y, s, Z0, D, sigma_mm=2.0)
    assert fit2.interval()["se_VL"] == pytest.approx(2 * fit.interval()["se_VL"], rel=1e-9)
    ci = fit.interval()
    assert ci["VL"][0] < fit.VL < ci["VL"][1]
    assert ci["VL"][1] - ci["VL"][0] == pytest.approx(2 * 1.959964 * ci["se_VL"], rel=1e-6)
    assert fit.interval(0.5)["VL"][1] - fit.interval(0.5)["VL"][0] < ci["VL"][1] - ci["VL"][0]


def test_sigma_is_estimated_from_residuals_when_not_given():
    rng = np.random.default_rng(3)
    s = ba.synthetic_readings(Y, 2.0, 0.45, Z0, D, 1.5, rng)
    fit = ba.fit_gaussian(Y, s, Z0, D)
    assert fit.sigma_used == pytest.approx(fit.rms_residual_mm * math.sqrt(11 / 9), rel=1e-9)  # n - 2 degrees of freedom
    assert 0.6 < fit.sigma_used < 3.0


def test_confidence_interval_coverage():
    """About 95% of the 95% intervals should contain the true values."""
    rng = np.random.default_rng(42)
    n, hit_v, hit_k = 400, 0, 0
    for _ in range(n):
        s = ba.synthetic_readings(Y, 2.2, 0.42, Z0, D, 1.0, rng)
        ci = ba.fit_gaussian(Y, s, Z0, D, sigma_mm=1.0).interval()
        hit_v += ci["VL"][0] <= 2.2 <= ci["VL"][1]
        hit_k += ci["K"][0] <= 0.42 <= ci["K"][1]
    assert 0.92 <= hit_v / n <= 0.98
    assert 0.92 <= hit_k / n <= 0.98


def test_estimates_are_unbiased_and_tighter_than_the_straight_line_method():
    rng = np.random.default_rng(7)
    nls, ll = [], []
    for _ in range(300):
        s = ba.synthetic_readings(Y, 2.2, 0.42, Z0, D, 1.0, rng)
        f = ba.fit_gaussian(Y, s, Z0, D, sigma_mm=1.0)
        nls.append((f.VL, f.K))
        d = ba.fit_loglinear(Y, s, Z0, D)
        ll.append((d["VL"], d["K"]))
    nls, ll = np.array(nls), np.array(ll)
    assert nls[:, 0].mean() == pytest.approx(2.2, rel=0.01) and nls[:, 1].mean() == pytest.approx(0.42, rel=0.01)
    assert ll[:, 0].mean() == pytest.approx(2.2, rel=0.06) and ll[:, 1].mean() == pytest.approx(0.42, rel=0.06)
    assert nls[:, 0].std() < ll[:, 0].std()


def test_prob_vl_exceeds():
    fit = ba.Fit(VL=2.0, K=0.45, cov=np.array([[0.04, 0.0], [0.0, 0.0001]]), sigma_used=1.0, rms_residual_mm=1.0, n=11, converged=True)
    assert fit.prob_vl_exceeds(2.0) == pytest.approx(0.5)
    assert fit.prob_vl_exceeds(2.2) == pytest.approx(0.158655, abs=1e-6)  # one standard error above
    assert fit.prob_vl_exceeds(1.8) == pytest.approx(0.841345, abs=1e-6)
    assert fit.prob_vl_exceeds(3.0) < 1e-6


def test_loglinear_method_details():
    s = ba.model_mm(Y, 1.0, 0.5, Z0, D)
    ll = ba.fit_loglinear(Y, s, Z0, D)
    assert ll["slope"] == pytest.approx(-1 / (2 * 0.5 ** 2))  # ln(S/Smax) = -(x/z0)^2 / (2 K^2)
    assert ll["n_used"] == 6  # centre point and readings below 0.2 Smax are left out
    assert ba.fit_loglinear(Y, s, Z0, D, min_fraction=0.0)["n_used"] == 10
    with pytest.raises(ValueError):
        ba.fit_loglinear(Y + 100.0, s, Z0, D)  # nothing within 3 m of the centre line
    with pytest.raises(ValueError):
        ba.fit_loglinear(Y, np.where(Y == 0, 5.0, 9.0), Z0, D)  # settlement not decreasing outwards
    # Smax is taken only from readings within the centre window
    shifted = Y + 2.0
    s2 = ba.model_mm(shifted, 1.0, 0.5, Z0, D)
    assert ba.fit_loglinear(shifted, s2, Z0, D)["Smax_mm"] == pytest.approx(float(s2[5]))


def test_departure_test():
    pred = ba.model_mm(Y, 1.0, 0.45, Z0, D)
    same = ba.departure_test(pred, pred, 1.0)
    assert same["flagged"] is False and same["chi2"] == 0.0 and same["p_value"] == 1.0 and same["dof"] == 11
    noisy = ba.departure_test(pred + np.array([1, -1, 0.5, -0.5, 1, -1, 0.5, 0, 1, -1, 0.5]), pred, 1.0)
    assert noisy["flagged"] is False and noisy["max_abs_z"] == 1.0
    one_bad = pred.copy()
    one_bad[5] += 3.5
    r = ba.departure_test(one_bad, pred, 1.0)
    assert r["flagged"] is True and r["max_abs_z"] == pytest.approx(3.5) and r["direction"] == "above"
    exactly3 = pred.copy()
    exactly3[5] += 3.0
    assert ba.departure_test(exactly3, pred, 1.0)["flagged"] is False  # needs to exceed 3 standard deviations
    assert ba.departure_test(exactly3, pred, 0.5)["flagged"] is True  # same gap, tighter survey accuracy
    bias = ba.departure_test(pred + 2.0, pred, 1.0)  # every marker 2 sigma high: no single outlier, chi-square catches it
    assert bias["max_abs_z"] == pytest.approx(2.0) and bias["chi2"] == pytest.approx(44.0) and bias["flagged"] is True
    low = ba.departure_test(pred - 2.0, pred, 1.0)
    assert low["direction"] == "below" and low["flagged"] is True
    assert ba.departure_test(pred + 2.0, pred, 1.0, p_limit=1e-9)["flagged"] is False


def test_project_final():
    i = 10.0
    assert ba.project_final(10.0, 100.0, 100.0, i) == pytest.approx(20.0)  # half developed at the face
    assert ba.project_final(8.413447, 100.0, 110.0, i) == pytest.approx(10.0, rel=1e-6)
    assert ba.project_final(1.0, 100.0, 85.0, i) is None  # only 7% expected: too early to project
    assert ba.project_final(1.0, 100.0, 85.0, i, min_fraction=0.05) == pytest.approx(1.0 / 0.0668072, rel=1e-5)
    assert ba.project_final(30.0, 100.0, 160.0, i) == pytest.approx(30.0, rel=1e-6)


def test_synthetic_readings_are_reproducible_and_noisy():
    a = ba.synthetic_readings(Y, 2.0, 0.45, Z0, D, 1.0, np.random.default_rng(5))
    b = ba.synthetic_readings(Y, 2.0, 0.45, Z0, D, 1.0, np.random.default_rng(5))
    assert np.array_equal(a, b)
    clean = ba.model_mm(Y, 2.0, 0.45, Z0, D)
    assert not np.allclose(a, clean) and np.abs(a - clean).max() < 5.0
    assert np.array_equal(ba.synthetic_readings(Y, 2.0, 0.45, Z0, D, 0.0, np.random.default_rng(5)), clean)
