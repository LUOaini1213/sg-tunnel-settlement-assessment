"""Stage 1 and Stage 2 building screening (limiting tensile strain method).

Stage 2 follows Mair, Taylor and Burland (1996) (source S3), which is the method named in the LTA
Civil Design Criteria (source S1, Section 14.3.6). The building is idealised as a weightless elastic
beam that follows the greenfield ground profile. Section and equation numbers below are those of
S3 unless another source is named.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import brentq

#: E/G for masonry (isotropic, Poisson's ratio 0.3) and for framed buildings (S3 sections 4.2, 4.5).
E_OVER_G = {"masonry": 2.6, "framed": 12.5}

#: Upper limits of limiting tensile strain for categories 0 to 3, as strain (S1 Table 14.2; S3 Table 2).
CATEGORY_LIMITS = (0.0005, 0.00075, 0.0015, 0.003)
CATEGORY_NAMES = ("Negligible", "Very Slight", "Slight", "Moderate", "Severe to Very Severe")

#: Stage 1 criteria (S1 Section 14.3.6; S3 section 5.2).
STAGE1_SETTLEMENT_M = 0.010
STAGE1_SLOPE = 1.0 / 500.0


# ----------------------------------------------------------------------------- beam equations
def section_properties(H: float, mode: str, hogging_neutral_axis: str = "bottom") -> tuple[float, float]:
    """Second moment of area I (per unit thickness) and extreme-fibre distance t.

    Sagging: neutral axis at mid-height, I = H^3/12, t = H/2.
    Hogging: neutral axis at the bottom edge, I = H^3/3, t = H (S3 section 4.2).
    ``hogging_neutral_axis="middle"`` is offered only for the sensitivity study.
    """
    if mode not in ("sagging", "hogging"):
        raise ValueError(mode)
    if mode == "hogging" and hogging_neutral_axis == "bottom":
        return H ** 3 / 3.0, H
    return H ** 3 / 12.0, H / 2.0


def bending_factor(L: float, H: float, e_over_g: float, mode: str, hogging_neutral_axis: str = "bottom") -> float:
    """(Delta/L) / eps_b = L/(12 t) + 3 I (E/G) / (2 t L H) (S3 Eq. 7)."""
    I, t = section_properties(H, mode, hogging_neutral_axis)
    return L / (12.0 * t) + 3.0 * I * e_over_g / (2.0 * t * L * H)


def diagonal_factor(L: float, H: float, e_over_g: float, mode: str, hogging_neutral_axis: str = "bottom") -> float:
    """(Delta/L) / eps_d = 1 + H L^2 / (18 I (E/G)) (S3 Eq. 8)."""
    I, _ = section_properties(H, mode, hogging_neutral_axis)
    return 1.0 + H * L ** 2 / (18.0 * I * e_over_g)


def total_bending_strain(eps_b: float, eps_h: float) -> float:
    """eps_bt = eps_h + eps_b (S3 Eq. 9)."""
    return eps_h + eps_b


def total_diagonal_strain(eps_d: float, eps_h: float, nu: float = 0.3) -> float:
    """eps_dt = eps_h (1 - nu)/2 + sqrt((eps_h (1 + nu)/2)^2 + eps_d^2), Mohr's circle of strain.

    With nu = 0.3 this is S3 Eq. 10: eps_dt = 0.35 eps_h + sqrt((0.65 eps_h)^2 + eps_d^2).
    """
    return eps_h * (1.0 - nu) / 2.0 + math.sqrt((eps_h * (1.0 + nu) / 2.0) ** 2 + eps_d ** 2)


def damage_category(eps_max: float) -> int:
    """Category 0 to 4 from the maximum tensile strain (S1 Table 14.2; "4" stands for "4 to 5").

    The table gives ranges that share their end values. A strain exactly on a boundary is put in
    the higher category here (ASSUMPTION, the cautious reading).
    """
    cat = 0
    for limit in CATEGORY_LIMITS:
        if eps_max >= limit:
            cat += 1
    return cat


# ----------------------------------------------------------------------------- profiles
@dataclass(frozen=True)
class Profile:
    """Ground movement along one facade: settlement, displacement along the facade, curvature."""

    s: Callable  # settlement (m, down positive) at positions along the facade
    u: Callable  # horizontal displacement along the facade (m, positive in +direction)
    kappa: Callable  # second derivative of settlement along the facade (1/m)
    limits: tuple[float, float]  # practical extent of the trough along the facade


@dataclass
class Segment:
    a: float
    b: float
    mode: str
    L: float
    delta: float
    deflection_ratio: float
    eps_h: float
    eps_b: float
    eps_d: float
    eps_bt: float
    eps_dt: float
    eps_max: float


@dataclass
class FacadeResult:
    eps_max: float
    category: int
    governing: str  # e.g. "hogging/diagonal"
    segments: list = field(default_factory=list)

    @property
    def max_deflection_ratio(self) -> float:
        return max((s.deflection_ratio for s in self.segments), default=0.0)


def inflexion_points(profile: Profile, a: float, b: float, n_scan: int = 241) -> list[float]:
    """Points in (a, b) where the curvature of the settlement profile changes sign."""
    xs = np.linspace(a, b, n_scan)
    k = np.asarray(profile.kappa(xs), dtype=float)
    roots = [float(xs[j]) for j in np.nonzero(k[1:-1] == 0.0)[0] + 1]
    for j in np.nonzero(k[:-1] * k[1:] < 0.0)[0]:
        roots.append(float(brentq(lambda v: float(profile.kappa(np.array([v]))[0]), xs[j], xs[j + 1], xtol=1e-9)))
    return sorted(roots)


def assess_segment(profile: Profile, a: float, b: float, H: float, e_over_g: float, n_grid: int = 81,
                   nu: float = 0.3, hogging_neutral_axis: str = "bottom", compression_relief: bool = True) -> Segment:
    """Strains in one hogging or sagging span [a, b] (S3 sections 4.2 and 4.3).

    With ``compression_relief=True`` (the default, S3 Eq. 9 and 10 as written) a compressive average
    horizontal strain in a sagging span lowers the total strain. ``False`` sets a compressive
    average strain to zero in sagging spans; it is offered as a cautious variant for a robustness
    check, not as a published rule.
    """
    L = b - a
    xs = np.linspace(a, b, n_grid)
    s = np.asarray(profile.s(xs), dtype=float)
    chord = s[0] + (s[-1] - s[0]) * (xs - a) / L
    # relative deflection: largest departure of the profile from the chord joining the span ends
    delta = float(np.max(np.abs(s - chord)))
    mid_kappa = float(profile.kappa(np.array([0.5 * (a + b)]))[0])
    mode = "sagging" if mid_kappa < 0.0 else "hogging"
    dr = delta / L
    eps_b = dr / bending_factor(L, H, e_over_g, mode, hogging_neutral_axis)
    eps_d = dr / diagonal_factor(L, H, e_over_g, mode, hogging_neutral_axis)
    # average horizontal strain: difference of the end displacements over the span (S3 section 4.3)
    u = np.asarray(profile.u(np.array([a, b])), dtype=float)
    eps_h = float(u[1] - u[0]) / L
    if not compression_relief and mode == "sagging":
        eps_h = max(eps_h, 0.0)
    eps_bt = total_bending_strain(eps_b, eps_h)
    eps_dt = total_diagonal_strain(eps_d, eps_h, nu)
    return Segment(a, b, mode, L, delta, dr, eps_h, eps_b, eps_d, eps_bt, eps_dt, max(eps_bt, eps_dt))


def assess_facade(profile: Profile, a: float, b: float, H: float, e_over_g: float, n_grid: int = 81,
                  nu: float = 0.3, hogging_neutral_axis: str = "bottom", min_span: float = 1e-6,
                  compression_relief: bool = True) -> FacadeResult:
    """Stage 2 for one facade spanning [a, b].

    The facade is cut at the practical limit of the trough and at each point of inflexion, and each
    span is assessed on its own (S3 section 4.1). The largest combined tensile strain governs.
    """
    lo, hi = profile.limits
    a2, b2 = max(a, lo), min(b, hi)
    if b2 - a2 <= min_span:
        return FacadeResult(0.0, 0, "outside trough", [])
    cuts = [a2] + inflexion_points(profile, a2, b2) + [b2]
    segments = []
    for p, q in zip(cuts[:-1], cuts[1:]):
        if q - p > min_span:
            segments.append(assess_segment(profile, p, q, H, e_over_g, n_grid, nu, hogging_neutral_axis, compression_relief))
    if not segments:
        return FacadeResult(0.0, 0, "outside trough", [])
    worst = max(segments, key=lambda sg: sg.eps_max)
    eps_max = max(worst.eps_max, 0.0)
    kind = "bending" if worst.eps_bt >= worst.eps_dt else "diagonal"
    return FacadeResult(eps_max, damage_category(eps_max), f"{worst.mode}/{kind}", segments)


# ----------------------------------------------------------------------------- Stage 1
def stage1_negligible(max_settlement_m: float, max_slope: float) -> bool:
    """True when settlement is below 10 mm and slope is flatter than 1:500 (S1 Section 14.3.6)."""
    return max_settlement_m < STAGE1_SETTLEMENT_M and max_slope < STAGE1_SLOPE


def final_category(stage2_category: int, sensitivity_indices: Sequence[int]) -> int:
    """Stage 2 category plus the sensitivity indices for structural form, heritage features and
    existing condition (S1 Section 14.3.10, after Goh and Mair 2014), capped at category 5."""
    return min(5, stage2_category + int(sum(sensitivity_indices)))
