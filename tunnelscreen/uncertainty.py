"""Monte Carlo over volume loss and trough width, and one-at-a-time sensitivity (tornado)."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from . import building as bld
from . import greenfield as gf
from . import screening as sc

Z95 = 1.6448536269514722  # 95th percentile of the standard normal


def lognormal_params(mean: float, q95: float) -> tuple[float, float]:
    """mu and sigma of a lognormal with the given mean and 95th percentile.

    mean = exp(mu + sigma^2/2) and q95 = exp(mu + z95 sigma) give
    sigma = z95 - sqrt(z95^2 - 2 ln(q95/mean)), the smaller root.
    """
    if not q95 > mean > 0.0:
        raise ValueError("need 0 < mean < q95")
    disc = Z95 ** 2 - 2.0 * math.log(q95 / mean)
    if disc < 0.0:
        raise ValueError("no lognormal has this mean and 95th percentile")
    sigma = Z95 - math.sqrt(disc)
    mu = math.log(mean) - 0.5 * sigma ** 2
    return mu, sigma


def characteristic_value(mean: float, sd: float, t: float = 1.64485) -> float:
    """Characteristic volume loss as mean + t * sd (the worked example in S5 section 4.5.1)."""
    return mean + t * sd


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1.0 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (0.0 if k == 0 else max(0.0, centre - half), 1.0 if k == n else min(1.0, centre + half))


def sample_zone(zone: dict, n: int, rng: np.random.Generator, correlation: float = 1.0):
    """Draws of (VL first drive, VL second drive, K) for a ground zone.

    Volume loss is lognormal with the mean and 95th percentile of the zone's observed values; the
    two drives are linked by a Gaussian copula with the given correlation. K is uniform on the
    zone's range and shared by both drives.
    """
    mu, sigma = lognormal_params(zone["observed_vl"]["mean"], zone["observed_vl"]["characteristic_95"])
    z1 = rng.standard_normal(n)
    z2 = correlation * z1 + math.sqrt(max(0.0, 1.0 - correlation ** 2)) * rng.standard_normal(n)
    k_lo, k_hi = zone["k_range"]
    return np.exp(mu + sigma * z1), np.exp(mu + sigma * z2), rng.uniform(k_lo, k_hi, n)


def longitudinal_unit_strain(scn: dict, b: sc.Building, k_grid: Sequence[float]) -> np.ndarray:
    """Transient longitudinal tensile strain per 1% volume loss, for each drive and each K.

    For one advancing tunnel every strain is proportional to its volume loss, so the Monte Carlo
    run scales these values instead of repeating the face sweep.
    """
    out = np.zeros((2, len(k_grid)))
    for j, k in enumerate(k_grid):
        for d, t in enumerate(sc.tunnels_at(scn, b.chainage, K=float(k), VL=1.0, second_factor=1.0)):
            out[d, j] = sc.worst_longitudinal(b, t)[0]
    return out


def monte_carlo(scn: dict, buildings: Sequence[sc.Building], n: int, seed: int, correlation: float = 1.0,
                n_k_grid: int = 9, keep_samples: bool = False) -> dict:
    """Probability of each damage category for every building on shallow foundations.

    ``keep_samples=True`` adds the per-sample arrays (volume losses, K, settlement, slope, strain)
    to each building's row, so that tests can recount the probabilities independently.
    """
    rng = np.random.default_rng(seed)
    factor = scn["alignment"].get("second_tunnel_vl_factor", 1.0)
    zone_draws = {z["id"]: sample_zone(z, n, rng, correlation) for z in scn["ground_zones"]}
    results = {}
    for b in buildings:
        zone = sc.zone_at(scn, b.chainage)
        vl1, vl2, ks = zone_draws[zone["id"]]
        vl2 = vl2 * factor
        z0 = sc.axis_depth(scn, b.chainage)
        D = scn["alignment"]["excavated_diameter_m"]
        y0 = [t["y0_m"] for t in scn["alignment"]["tunnels"]]
        ys = np.linspace(b.y_start, b.y_end, 101)
        smax = np.empty(n)
        slope = np.empty(n)
        eps = np.full(n, np.nan)
        shallow = b.foundation == "shallow"
        if shallow:
            k_grid = np.linspace(zone["k_range"][0], zone["k_range"][1], n_k_grid)
            unit = longitudinal_unit_strain(scn, b, k_grid)
        for j in range(n):
            ts = [gf.Tunnel(y0[0], z0, D, float(vl1[j]), float(ks[j])), gf.Tunnel(y0[1], z0, D, float(vl2[j]), float(ks[j]))]
            smax[j] = float(np.max(gf.settlement(ys, ts)))
            i = ks[j] * z0
            slope[j] = max(float(np.max(np.abs(gf.slope(ys, ts)))), float(np.max(np.abs(gf.slope(ys, ts[:1])))),
                           max(float(np.max(gf.settlement_one(ys, t))) for t in ts) / (i * gf.SQRT_2PI))
            if shallow:
                e_both = bld.assess_facade(sc.transverse_profile(ts), b.y_start, b.y_end, b.height, b.e_over_g, n_grid=41).eps_max
                e_first = bld.assess_facade(sc.transverse_profile(ts[:1]), b.y_start, b.y_end, b.height, b.e_over_g, n_grid=41).eps_max
                e_long = max(vl1[j] * np.interp(ks[j], k_grid, unit[0]), vl2[j] * np.interp(ks[j], k_grid, unit[1]))
                eps[j] = max(e_both, e_first, e_long)
        stage1_fail = ~((smax < bld.STAGE1_SETTLEMENT_M) & (slope < bld.STAGE1_SLOPE))
        row = {
            "id": b.id, "zone": zone["id"], "foundation": b.foundation, "n": n,
            "settlement_mm": {q: float(np.percentile(smax, p) * 1000.0) for q, p in (("p05", 5), ("p50", 50), ("p95", 95))},
            "p_stage1_not_negligible": _prob(stage1_fail),
        }
        if shallow:
            cats = np.array([bld.damage_category(e) for e in eps])
            final = np.minimum(5, cats + sum(b.indices))
            row["eps_max_pct"] = {q: float(np.percentile(eps, p) * 100.0) for q, p in (("p05", 5), ("p50", 50), ("p95", 95))}
            row["p_stage2_category_at_least"] = {str(c): _prob(cats >= c) for c in (1, 2, 3, 4)}
            row["p_final_category_at_least"] = {str(c): _prob(final >= c) for c in (1, 2, 3, 4)}
        if keep_samples:
            row["samples"] = {"vl1": vl1, "vl2": vl2, "k": ks, "smax": smax, "slope": slope, "eps": eps}
        results[b.id] = row
    zones = {}
    for z in scn["ground_zones"]:
        vl1 = zone_draws[z["id"]][0]
        mu, sigma = lognormal_params(z["observed_vl"]["mean"], z["observed_vl"]["characteristic_95"])
        zones[z["id"]] = {"mu": mu, "sigma": sigma, "median_vl_pct": math.exp(mu),
                          "p_vl_exceeds_design": _prob(vl1 > z["design"]["VL"]),
                          "p_vl_exceeds_upper_bound": _prob(vl1 > z["observed_vl"]["upper_bound"])}
    return {"n": n, "seed": seed, "vl_correlation": correlation, "buildings": results, "zones": zones}


def _prob(mask: np.ndarray) -> dict:
    k, n = int(np.count_nonzero(mask)), int(mask.size)
    lo, hi = wilson_interval(k, n)
    return {"p": k / n, "lo95": lo, "hi95": hi, "count": k}


# ----------------------------------------------------------------------------- tornado
def _eps_for(scn: dict, b: sc.Building, K: float, vl: float, depth_offset: float = 0.0, second_factor: float = 1.0,
             height: float | None = None, structure: str | None = None, hogging_neutral_axis: str = "bottom") -> float:
    """Largest Stage 2 tensile strain (as strain) over all construction cases for one set of inputs."""
    from dataclasses import replace

    bb = replace(b, height=height if height is not None else b.height, structure=structure or b.structure)
    ts = sc.tunnels_at(scn, b.chainage, K=K, VL=vl, depth_offset=depth_offset, second_factor=second_factor)
    return max(c[0] for c in sc.stage2_cases(bb, ts, hogging_neutral_axis=hogging_neutral_axis).values())


def tornado(scn: dict, b: sc.Building) -> dict:
    """One-at-a-time sensitivity of the maximum tensile strain for one building.

    Base case: median volume loss and mid-range K of the building's ground zone. Each input is then
    moved to the two ends of its range with the others held at the base.
    """
    zone = sc.zone_at(scn, b.chainage)
    a = scn["assumptions"]
    mu, sigma = lognormal_params(zone["observed_vl"]["mean"], zone["observed_vl"]["characteristic_95"])
    vl_lo, vl_med, vl_hi = (math.exp(mu + q * sigma) for q in (-Z95, 0.0, Z95))
    k_lo, k_hi = zone["k_range"]
    k_mid = 0.5 * (k_lo + k_hi)
    dz = a["depth_tolerance_m"]["value"]
    f_lo, f_hi = a["second_tunnel_factor_range"]["value"]
    base = _eps_for(scn, b, k_mid, vl_med)
    bars = [
        ("Volume loss (5th to 95th percentile)", f"{vl_lo:.2f}% to {vl_hi:.2f}%",
         _eps_for(scn, b, k_mid, vl_lo), _eps_for(scn, b, k_mid, vl_hi)),
        ("Trough width parameter K", f"{k_lo:.2f} to {k_hi:.2f}",
         _eps_for(scn, b, k_lo, vl_med), _eps_for(scn, b, k_hi, vl_med)),
        ("Depth to tunnel axis", f"-{dz:g} m to +{dz:g} m",
         _eps_for(scn, b, k_mid, vl_med, depth_offset=-dz), _eps_for(scn, b, k_mid, vl_med, depth_offset=dz)),
        ("Volume loss factor on second drive", f"{f_lo:g} to {f_hi:g}",
         _eps_for(scn, b, k_mid, vl_med, second_factor=f_lo), _eps_for(scn, b, k_mid, vl_med, second_factor=f_hi)),
        ("Building height H", "-20% to +20%",
         _eps_for(scn, b, k_mid, vl_med, height=0.8 * b.height), _eps_for(scn, b, k_mid, vl_med, height=1.2 * b.height)),
        ("E/G of the equivalent beam", "2.6 (masonry) to 12.5 (framed)",
         _eps_for(scn, b, k_mid, vl_med, structure="masonry"), _eps_for(scn, b, k_mid, vl_med, structure="framed")),
        ("Neutral axis in hogging", "bottom edge to mid-height",
         _eps_for(scn, b, k_mid, vl_med, hogging_neutral_axis="bottom"), _eps_for(scn, b, k_mid, vl_med, hogging_neutral_axis="middle")),
    ]
    rows = [{"input": name, "range": rng, "low_pct": lo * 100.0, "high_pct": hi * 100.0,
             "swing_pct": abs(hi - lo) * 100.0} for name, rng, lo, hi in bars]
    rows.sort(key=lambda r: -r["swing_pct"])
    return {"building": b.id, "base_eps_max_pct": base * 100.0, "base_VL_pct": vl_med, "base_K": k_mid,
            "VL_range_pct": [vl_lo, vl_hi], "bars": rows}
