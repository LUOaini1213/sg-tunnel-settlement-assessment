"""Greenfield ground movements above bored tunnels (empirical Gaussian method).

Every function cites the published equation it implements. Source IDs (S1, S2, ...) are listed
in data/sources.csv; the supporting quotations are in data/facts.csv.

Conventions used throughout the package
---------------------------------------
* Lengths are in metres, volume loss ``VL`` is in per cent of the excavated face area.
* ``y`` is the horizontal offset transverse to the tunnel; ``x`` is the distance along it.
* Settlement ``S`` is positive downwards.
* Horizontal displacement ``u`` is positive in the +y (or +x) direction, so ground on the +y side
  of a tunnel, which moves towards the centre line, has a negative ``u``.
* Horizontal strain is positive in tension.
* "Sagging" is where the settled ground surface is concave upwards (between the points of
  inflexion); "hogging" is where it is convex upwards (outside them).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np
from scipy.special import ndtr

SQRT_2PI = math.sqrt(2.0 * math.pi)
#: Practical half-width of the trough as a multiple of i (S3 section 4.1: "2.5i (where Sv/Smax = 0.044)").
TROUGH_LIMIT_FACTOR = 2.5


@dataclass(frozen=True)
class Tunnel:
    """One bored tunnel. ``y0`` is the offset of its centre line from the alignment datum."""

    y0: float
    z0: float  # depth from ground surface to tunnel axis, m
    D: float  # excavated diameter, m
    VL: float  # volume loss, % of excavated face area
    K: float  # trough width parameter, dimensionless

    def with_(self, **kw) -> "Tunnel":
        return replace(self, **kw)


def trough_width(K: float, z0: float) -> float:
    """i = K * z0, distance from centre line to the point of inflexion (S3 Eq. 2; S1 Cl. 14.2.1.2)."""
    return K * z0


def trough_volume(VL: float, D: float) -> float:
    """Vs = VL * pi * D^2 / 4, volume of the trough per metre of tunnel (S3 Eq. 4). VL in %."""
    return (VL / 100.0) * math.pi * D ** 2 / 4.0


def s_max(VL: float, D: float, K: float, z0: float) -> float:
    """Smax = Vs / (sqrt(2 pi) * i) (S3 Eq. 3 rearranged; equals S3 Eq. 5 and S1 Cl. 14.2.1.2)."""
    return trough_volume(VL, D) / (SQRT_2PI * trough_width(K, z0))


def _i(t: Tunnel) -> float:
    return trough_width(t.K, t.z0)


def _smax(t: Tunnel) -> float:
    return s_max(t.VL, t.D, t.K, t.z0)


# ----------------------------------------------------------------------------- single tunnel
def settlement_one(y, t: Tunnel):
    """Sv = Smax * exp(-y^2 / (2 i^2)) (Peck 1969; S3 Eq. 1; S1 Cl. 14.2.1.2)."""
    d = np.asarray(y, dtype=float) - t.y0
    i = _i(t)
    return _smax(t) * np.exp(-d ** 2 / (2.0 * i ** 2))


def slope_one(y, t: Tunnel):
    """dSv/dy = -(y / i^2) * Sv (derivative of S3 Eq. 1)."""
    d = np.asarray(y, dtype=float) - t.y0
    return -(d / _i(t) ** 2) * settlement_one(y, t)


def curvature_one(y, t: Tunnel):
    """d2Sv/dy2 = (Sv / i^2) * (y^2 / i^2 - 1); negative (sagging) inside |y| < i."""
    d = np.asarray(y, dtype=float) - t.y0
    i = _i(t)
    return settlement_one(y, t) / i ** 2 * (d ** 2 / i ** 2 - 1.0)


def horizontal_displacement_one(y, t: Tunnel):
    """u = -(y / z0) * Sv: vectors directed towards the tunnel axis (S3 Eq. 6; S1 Cl. 14.2.1.8)."""
    d = np.asarray(y, dtype=float) - t.y0
    return -(d / t.z0) * settlement_one(y, t)


def horizontal_strain_one(y, t: Tunnel):
    """eps_h = du/dy = (Sv / z0) * (y^2 / i^2 - 1), tension positive (S3 section 2.2; S6 Eq. 2.16)."""
    d = np.asarray(y, dtype=float) - t.y0
    i = _i(t)
    return settlement_one(y, t) / t.z0 * (d ** 2 / i ** 2 - 1.0)


# ----------------------------------------------------------------------------- superposition
def _sum(fn, y, tunnels: Iterable[Tunnel]):
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y, dtype=float)
    for t in tunnels:
        out = out + fn(y, t)
    return out


def settlement(y, tunnels: Sequence[Tunnel]):
    """Superposition of independent single-tunnel troughs (S3 section 2.4, with its stated limits)."""
    return _sum(settlement_one, y, tunnels)


def slope(y, tunnels: Sequence[Tunnel]):
    return _sum(slope_one, y, tunnels)


def curvature(y, tunnels: Sequence[Tunnel]):
    return _sum(curvature_one, y, tunnels)


def horizontal_displacement(y, tunnels: Sequence[Tunnel]):
    return _sum(horizontal_displacement_one, y, tunnels)


def horizontal_strain(y, tunnels: Sequence[Tunnel]):
    return _sum(horizontal_strain_one, y, tunnels)


def trough_limits(tunnels: Sequence[Tunnel]) -> tuple[float, float]:
    """Practical transverse extent: union of y0 +/- 2.5 i over the tunnels (S3 section 4.1)."""
    lo = min(t.y0 - TROUGH_LIMIT_FACTOR * _i(t) for t in tunnels)
    hi = max(t.y0 + TROUGH_LIMIT_FACTOR * _i(t) for t in tunnels)
    return lo, hi


# ----------------------------------------------------------------------------- longitudinal
def face_factor(x, x_face: float, i: float):
    """Fraction of the final settlement developed at chainage x when the face is at x_face.

    Cumulative normal curve, Phi((x_face - x) / i), after Attewell and Woodman (1982) as given in
    S6 Eq. 2.9-2.10 with i_x = i_y; it gives 0.5 Smax above the face (S3 section 2.3).
    """
    return ndtr((x_face - np.asarray(x, dtype=float)) / i)


def _pdf(xi):
    return np.exp(-0.5 * xi ** 2) / SQRT_2PI


def settlement_advancing(x, y: float, t: Tunnel, x_face: float):
    """Settlement at (x, y) while the face of tunnel ``t`` is at chainage ``x_face``."""
    return float(settlement_one(y, t)) * face_factor(x, x_face, _i(t))


def long_slope_advancing(x, y: float, t: Tunnel, x_face: float):
    """dS/dx = -S_trans(y) * phi(xi) / i, xi = (x - x_face) / i."""
    i = _i(t)
    xi = (np.asarray(x, dtype=float) - x_face) / i
    return -float(settlement_one(y, t)) * _pdf(xi) / i


def long_curvature_advancing(x, y: float, t: Tunnel, x_face: float):
    """d2S/dx2 = S_trans(y) * xi * phi(xi) / i^2: hogging ahead of the face, sagging behind it."""
    i = _i(t)
    xi = (np.asarray(x, dtype=float) - x_face) / i
    return float(settlement_one(y, t)) * xi * _pdf(xi) / i ** 2


def long_displacement_advancing(x, y: float, t: Tunnel, x_face: float):
    """u_x = (i^2 / z0) * dS/dx = -(i / z0) * S_trans(y) * phi(xi).

    Ground ahead of the face moves back towards it. On the centre line at the face the magnitude is
    VL * D^2 / (8 z0), which is S6 Eq. 2.17. The transverse relation u_y = -(y / z0) S (S3 Eq. 6)
    is the same rule, u = (i^2 / z0) * grad(S), applied in the other direction.
    """
    i = _i(t)
    return (i ** 2 / t.z0) * long_slope_advancing(x, y, t, x_face)


def long_strain_advancing(x, y: float, t: Tunnel, x_face: float):
    """eps_x = du_x/dx = S_trans(y) * xi * phi(xi) / z0: tensile ahead of the face (S6 Eq. 2.18)."""
    i = _i(t)
    xi = (np.asarray(x, dtype=float) - x_face) / i
    return float(settlement_one(y, t)) * xi * _pdf(xi) / t.z0


# ----------------------------------------------------------------------------- subsurface
def subsurface_trough_width(z: float, z0: float, K_surface: float = 0.5) -> float:
    """i(z) at depth z below the surface, for tunnels in clay (Mair et al. 1993, as S4 Eq. 7-8).

    i = K(z) * (z0 - z) with K(z) = [0.175 + 0.325 (1 - z/z0)] / (1 - z/z0), i.e.
    i = z0 * [0.175 + 0.325 (1 - z/z0)], which has K = 0.5 at the surface. For a surface value of
    K other than 0.5 the profile is scaled by K_surface / 0.5; that scaling is an ASSUMPTION of
    this study, not part of the published relation.
    """
    if not 0.0 <= z < z0:
        raise ValueError("z must lie between the surface and the tunnel axis")
    return (K_surface / 0.5) * z0 * (0.175 + 0.325 * (1.0 - z / z0))


def subsurface_settlement(y, z: float, tunnels: Sequence[Tunnel]):
    """Gaussian settlement at depth z with the same trough volume as at the surface (S4 section 5.4)."""
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y, dtype=float)
    for t in tunnels:
        i_z = subsurface_trough_width(z, t.z0, t.K)
        smax_z = trough_volume(t.VL, t.D) / (SQRT_2PI * i_z)
        out = out + smax_z * np.exp(-(y - t.y0) ** 2 / (2.0 * i_z ** 2))
    return out


# ----------------------------------------------------------------------------- closed forms
def max_slope_single(t: Tunnel) -> float:
    """Largest slope of a single trough, at y = i: exp(-1/2) * Smax / i."""
    return math.exp(-0.5) * _smax(t) / _i(t)


def max_tensile_strain_single(t: Tunnel) -> float:
    """Largest tensile horizontal strain of a single trough, at y = sqrt(3) i: 2 exp(-3/2) Smax / z0."""
    return 2.0 * math.exp(-1.5) * _smax(t) / t.z0
