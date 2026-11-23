"""Back-analysis of settlement-marker readings: estimate volume loss and trough width.

Two estimators are provided for a single tunnel:

* ``fit_gaussian``: non-linear least squares on (VL, K) with confidence intervals;
* ``fit_loglinear``: the published straight-line method used on Singapore data (source S5,
  section 4.1.1), kept as an independent cross-check.

``departure_test`` flags when readings depart from a prediction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2, norm

from . import greenfield as gf


@dataclass
class Fit:
    VL: float
    K: float
    cov: np.ndarray
    sigma_used: float
    rms_residual_mm: float
    n: int
    converged: bool

    def interval(self, level: float = 0.95) -> dict:
        """Normal-approximation confidence intervals from the covariance of the estimates."""
        z = norm.ppf(0.5 + level / 2.0)
        se = np.sqrt(np.diag(self.cov))
        return {"VL": (self.VL - z * se[0], self.VL + z * se[0]), "K": (self.K - z * se[1], self.K + z * se[1]),
                "se_VL": float(se[0]), "se_K": float(se[1])}

    def prob_vl_exceeds(self, vl_limit: float) -> float:
        """Probability that the true volume loss is above ``vl_limit`` under the normal approximation."""
        se = math.sqrt(self.cov[0, 0])
        return float(norm.sf((vl_limit - self.VL) / se))


def model_mm(y, VL: float, K: float, z0: float, D: float, y0: float = 0.0):
    """Predicted settlement in millimetres at offsets y for a single tunnel."""
    return gf.settlement_one(y, gf.Tunnel(y0, z0, D, VL, K)) * 1000.0


def fit_gaussian(y, s_mm, z0: float, D: float, sigma_mm: float | None = None, y0: float = 0.0,
                 start: tuple[float, float] = (1.0, 0.5)) -> Fit:
    """Least-squares estimate of (VL, K) from a transverse array of readings.

    The covariance is sigma^2 (J^T J)^-1 with J the Jacobian at the solution. If ``sigma_mm`` is
    not given, sigma is estimated from the residuals with n - 2 degrees of freedom.
    """
    y = np.asarray(y, dtype=float)
    s = np.asarray(s_mm, dtype=float)
    if y.size < 3:
        raise ValueError("at least three readings are needed to estimate two parameters")

    def resid(p):
        return model_mm(y, p[0], p[1], z0, D, y0) - s

    sol = least_squares(resid, x0=np.array(start, dtype=float), bounds=([1e-4, 0.05], [20.0, 1.5]), x_scale=[1.0, 0.1])
    r = sol.fun
    dof = max(1, y.size - 2)
    sigma = sigma_mm if sigma_mm is not None else math.sqrt(float(r @ r) / dof)
    J = sol.jac
    cov = sigma ** 2 * np.linalg.inv(J.T @ J)
    return Fit(float(sol.x[0]), float(sol.x[1]), cov, float(sigma), math.sqrt(float(r @ r) / y.size), int(y.size), bool(sol.success))


def fit_loglinear(y, s_mm, z0: float, D: float, centre_window_m: float = 3.0, min_fraction: float = 0.2) -> dict:
    """Straight-line method of S5 section 4.1.1.

    ln(S/Smax) = -(x/z0)^2 / (2 K^2), so the slope m of the best-fit line through the origin on a
    plot of ln(S/Smax) against (x/z0)^2 gives K = sqrt(-1 / (2 m)). Smax is the largest reading
    within ``centre_window_m`` of the centre line (S5 uses 3 m). Volume loss then follows from
    Vs = sqrt(2 pi) K z0 Smax (S5 Eq. 6-7). Readings below ``min_fraction`` of Smax are left out
    because the logarithm magnifies their scatter; the 0.2 default is an ASSUMPTION chosen from the
    comparison stored in results/backanalysis.json (with 0.05 the method over-estimates K and VL).
    """
    y = np.asarray(y, dtype=float)
    s = np.asarray(s_mm, dtype=float)
    near = np.abs(y) <= centre_window_m
    if not near.any():
        raise ValueError("no reading within the centre window")
    smax = float(s[near].max())
    keep = (s > min_fraction * smax) & ~np.isclose(y, 0.0)
    X = (y[keep] / z0) ** 2
    Y = np.log(s[keep] / smax)
    slope = float((X @ Y) / (X @ X))
    if slope >= 0.0:
        raise ValueError("readings do not decrease away from the centre line")
    K = math.sqrt(-1.0 / (2.0 * slope))
    vs = gf.SQRT_2PI * K * z0 * smax / 1000.0
    VL = 100.0 * vs / (math.pi * D ** 2 / 4.0)
    return {"K": K, "VL": VL, "Smax_mm": smax, "n_used": int(keep.sum()), "slope": slope}


def departure_test(readings_mm, predicted_mm, sigma_mm: float, z_limit: float = 3.0, p_limit: float = 0.01) -> dict:
    """Do the readings depart from the prediction by more than survey scatter explains?

    z_i = (reading_i - predicted_i) / sigma. The set is flagged when any |z_i| exceeds ``z_limit``
    or when the chi-square statistic sum(z_i^2) has a p-value below ``p_limit`` (both thresholds
    are ASSUMPTIONS). ``direction`` says whether the readings are mostly above or below.
    """
    r = np.asarray(readings_mm, dtype=float)
    p = np.asarray(predicted_mm, dtype=float)
    z = (r - p) / sigma_mm
    stat = float(z @ z)
    pval = float(chi2.sf(stat, df=z.size))
    flagged = bool(np.max(np.abs(z)) > z_limit or pval < p_limit)
    return {"z": z.tolist(), "max_abs_z": float(np.max(np.abs(z))), "chi2": stat, "dof": int(z.size),
            "p_value": pval, "flagged": flagged, "direction": "above" if float(z.sum()) > 0 else "below"}


def project_final(reading_mm: float, x_marker: float, x_face: float, i: float, min_fraction: float = 0.2) -> float | None:
    """Final settlement implied by a reading taken before the trough has fully developed.

    final = reading / Phi((x_face - x_marker) / i). Returned as None while less than
    ``min_fraction`` of the final settlement is expected, because the projection is then too
    sensitive to noise (ASSUMPTION).
    """
    frac = float(gf.face_factor(x_marker, x_face, i))
    if frac < min_fraction:
        return None
    return reading_mm / frac


def synthetic_readings(y, VL: float, K: float, z0: float, D: float, sigma_mm: float, rng: np.random.Generator,
                       y0: float = 0.0) -> np.ndarray:
    """SYNTHETIC marker readings: the Gaussian trough plus independent normal noise."""
    y = np.asarray(y, dtype=float)
    return model_mm(y, VL, K, z0, D, y0) + rng.normal(0.0, sigma_mm, y.size)
