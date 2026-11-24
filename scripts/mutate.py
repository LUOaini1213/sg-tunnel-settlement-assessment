"""Plant errors in the engineering code one at a time and check that the test suite notices each one.

Usage: python scripts/mutate.py --record first|added|final|ci [--require-all] [--only N[,N...]] [--from N]

Each mutant is a hand-written, plausible mistake: a wrong sign, i in place of z0, the neutral axis in
the wrong place, a threshold off by one category, and so on. The golden tests (committed results
against a fresh run, committed workbook against the generator) are excluded, so a kill has to come
from a test with an independent expectation. Every file is restored after its mutant has run, and
again on exit. Results go to results/mutation.json under the given key. Mutants judged equivalent
(no input can tell them apart) are listed in EQUIVALENT with the reason.

History: the first 117 mutants were written and run before the review ("first"). The review added
the mutants from index FIRST_SET onwards and ran them once against the suite as it then stood
("added"), before writing tests for the ones that survived; "final" is the whole list afterwards.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "mutation.json"
PYTEST = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "-m", "not golden"]
G, B, S = "tunnelscreen/greenfield.py", "tunnelscreen/building.py", "tunnelscreen/screening.py"
U, M, A, W = "tunnelscreen/uncertainty.py", "tunnelscreen/monitoring.py", "tunnelscreen/backanalysis.py", "tunnelscreen/workbook.py"
P = "tunnelscreen/pipeline.py"

MUTANTS = [
    # ---- greenfield: trough geometry and volume
    (G, "    return K * z0", "    return K + z0"),
    (G, "    return (VL / 100.0) * math.pi * D ** 2 / 4.0", "    return (VL / 100.0) * math.pi * D ** 2 / 2.0"),
    (G, "    return (VL / 100.0) * math.pi * D ** 2 / 4.0", "    return VL * math.pi * D ** 2 / 4.0"),
    (G, "    return trough_volume(VL, D) / (SQRT_2PI * trough_width(K, z0))", "    return trough_volume(VL, D) / (2.5 * trough_width(K, z0))"),
    (G, "    return _smax(t) * np.exp(-d ** 2 / (2.0 * i ** 2))", "    return _smax(t) * np.exp(-d ** 2 / (i ** 2))"),
    (G, "    return -(d / _i(t) ** 2) * settlement_one(y, t)", "    return (d / _i(t) ** 2) * settlement_one(y, t)"),
    (G, "    return settlement_one(y, t) / i ** 2 * (d ** 2 / i ** 2 - 1.0)", "    return settlement_one(y, t) / i ** 2 * (1.0 - d ** 2 / i ** 2)"),
    # ---- greenfield: horizontal movement and strain
    (G, "    return -(d / t.z0) * settlement_one(y, t)", "    return (d / t.z0) * settlement_one(y, t)"),
    (G, "    return -(d / t.z0) * settlement_one(y, t)", "    return -(d / _i(t)) * settlement_one(y, t)"),
    (G, "    return settlement_one(y, t) / t.z0 * (d ** 2 / i ** 2 - 1.0)", "    return settlement_one(y, t) / t.z0 * (1.0 - d ** 2 / i ** 2)"),
    (G, "    return settlement_one(y, t) / t.z0 * (d ** 2 / i ** 2 - 1.0)", "    return settlement_one(y, t) / i * (d ** 2 / i ** 2 - 1.0)"),
    (G, "TROUGH_LIMIT_FACTOR = 2.5", "TROUGH_LIMIT_FACTOR = 3.0"),
    (G, "        out = out + fn(y, t)", "        out = np.maximum(out, fn(y, t))"),
    # ---- greenfield: longitudinal
    (G, "    return ndtr((x_face - np.asarray(x, dtype=float)) / i)", "    return ndtr((np.asarray(x, dtype=float) - x_face) / i)"),
    (G, "    return -float(settlement_one(y, t)) * _pdf(xi) / i", "    return float(settlement_one(y, t)) * _pdf(xi) / i"),
    (G, "    return float(settlement_one(y, t)) * xi * _pdf(xi) / i ** 2", "    return -float(settlement_one(y, t)) * xi * _pdf(xi) / i ** 2"),
    (G, "    return (i ** 2 / t.z0) * long_slope_advancing(x, y, t, x_face)", "    return (i / t.z0) * long_slope_advancing(x, y, t, x_face)"),
    (G, "    return float(settlement_one(y, t)) * xi * _pdf(xi) / t.z0", "    return -float(settlement_one(y, t)) * xi * _pdf(xi) / t.z0"),
    # ---- greenfield: subsurface and closed forms
    (G, "    return (K_surface / 0.5) * z0 * (0.175 + 0.325 * (1.0 - z / z0))", "    return (K_surface / 0.5) * z0 * (0.325 + 0.175 * (1.0 - z / z0))"),
    (G, "    return (K_surface / 0.5) * z0 * (0.175 + 0.325 * (1.0 - z / z0))", "    return K_surface * z0 * (0.175 + 0.325 * (1.0 - z / z0))"),
    (G, "        smax_z = trough_volume(t.VL, t.D) / (SQRT_2PI * i_z)", "        smax_z = _smax(t)"),
    (G, "    return math.exp(-0.5) * _smax(t) / _i(t)", "    return math.exp(-1.0) * _smax(t) / _i(t)"),
    (G, "    return 2.0 * math.exp(-1.5) * _smax(t) / t.z0", "    return math.exp(-1.5) * _smax(t) / t.z0"),
    # ---- building: section and beam equations
    (B, "        return H ** 3 / 3.0, H", "        return H ** 3 / 12.0, H"),
    (B, "        return H ** 3 / 3.0, H", "        return H ** 3 / 3.0, H / 2.0"),
    (B, "    return H ** 3 / 12.0, H / 2.0", "    return H ** 3 / 12.0, H"),
    (B, "    return L / (12.0 * t) + 3.0 * I * e_over_g / (2.0 * t * L * H)", "    return L / (12.0 * t) + 3.0 * I / (2.0 * t * L * H * e_over_g)"),
    (B, "    return L / (12.0 * t) + 3.0 * I * e_over_g / (2.0 * t * L * H)", "    return L / (6.0 * t) + 3.0 * I * e_over_g / (2.0 * t * L * H)"),
    (B, "    return 1.0 + H * L ** 2 / (18.0 * I * e_over_g)", "    return 1.0 + H * L ** 2 * e_over_g / (18.0 * I)"),
    (B, "    return 1.0 + H * L ** 2 / (18.0 * I * e_over_g)", "    return 1.0 + H * L ** 2 / (12.0 * I * e_over_g)"),
    # ---- building: strain combination
    (B, "    return eps_h + eps_b", "    return eps_b"),
    (B, "math.sqrt((eps_h * (1.0 + nu) / 2.0) ** 2 + eps_d ** 2)", "math.sqrt((eps_h * (1.0 - nu) / 2.0) ** 2 + eps_d ** 2)"),
    (B, "    return eps_h * (1.0 - nu) / 2.0 + math.sqrt(", "    return eps_h * (1.0 + nu) / 2.0 + math.sqrt("),
    # ---- building: damage categories and Stage 1
    (B, "        if eps_max >= limit:", "        if eps_max > limit:"),
    (B, "CATEGORY_LIMITS = (0.0005, 0.00075, 0.0015, 0.003)", "CATEGORY_LIMITS = (0.0005, 0.001, 0.0015, 0.003)"),
    (B, "STAGE1_SETTLEMENT_M = 0.010", "STAGE1_SETTLEMENT_M = 0.015"),
    (B, "STAGE1_SLOPE = 1.0 / 500.0", "STAGE1_SLOPE = 1.0 / 300.0"),
    (B, "    return max_settlement_m < STAGE1_SETTLEMENT_M and max_slope < STAGE1_SLOPE", "    return max_settlement_m < STAGE1_SETTLEMENT_M or max_slope < STAGE1_SLOPE"),
    (B, "    return max_settlement_m < STAGE1_SETTLEMENT_M and max_slope < STAGE1_SLOPE", "    return max_settlement_m <= STAGE1_SETTLEMENT_M and max_slope < STAGE1_SLOPE"),
    (B, "    return min(5, stage2_category + int(sum(sensitivity_indices)))", "    return min(5, stage2_category + int(max(sensitivity_indices)))"),
    (B, "    return min(5, stage2_category + int(sum(sensitivity_indices)))", "    return min(4, stage2_category + int(sum(sensitivity_indices)))"),
    # ---- building: facade partition and span strains
    (B, "    delta = float(np.max(np.abs(s - chord)))", "    delta = float(np.max(s) - np.min(s))"),
    (B, '    mode = "sagging" if mid_kappa < 0.0 else "hogging"', '    mode = "sagging" if mid_kappa > 0.0 else "hogging"'),
    (B, "    eps_h = float(u[1] - u[0]) / L", "    eps_h = float(u[0] - u[1]) / L"),
    (B, "    eps_b = dr / bending_factor(", "    eps_b = dr * bending_factor("),
    (B, "eps_bt, eps_dt, max(eps_bt, eps_dt))", "eps_bt, eps_dt, min(eps_bt, eps_dt))"),
    (B, "    a2, b2 = max(a, lo), min(b, hi)", "    a2, b2 = a, b"),
    (B, "    cuts = [a2] + inflexion_points(profile, a2, b2) + [b2]", "    cuts = [a2, b2]"),
    (B, "    worst = max(segments, key=lambda sg: sg.eps_max)", "    worst = min(segments, key=lambda sg: sg.eps_max)"),
    # ---- screening: scenario and staged rules
    (S, '        if z["from_m"] <= chainage < z["to_m"]:', '        if z["from_m"] < chainage <= z["to_m"]:'),
    (S, '        if t["id"] != order[0]:', '        if t["id"] == order[0]:'),
    (S, "    z0 = axis_depth(scn, chainage) + depth_offset", "    z0 = axis_depth(scn, chainage) - depth_offset"),
    (S, "    return s_max, max(slopes)", "    return s_max, slopes[0]"),
    (S, "        slopes.append(float(np.max(gf.settlement_one(ys, t))) / (i * gf.SQRT_2PI))", "        slopes.append(float(np.max(gf.settlement_one(ys, t))) / i)"),
    (S, "    first = [tunnels[0]]", "    first = [tunnels[-1]]"),
    (S, "    if b.y_start < t.y0 < b.y_end:", "    if False:"),
    (S, "            if r.eps_max > best:", "            if r.eps_max < best:"),
    (S, "limits=(x_face - half, x_face + half))", "limits=(x_face, x_face + half))"),
    (S, "    if indices[2] >= 1:", "    if indices[0] >= 1:"),
    (S, "    if final_cat is not None and final_cat <= 1:", "    if final_cat is not None and final_cat <= 2:"),
    (S, "    if stage1_pass and sum(indices) == 0:", "    if stage1_pass:"),
    (S, "        name = max(cases, key=lambda c: cases[c][0])", "        name = min(cases, key=lambda c: cases[c][0])"),
    (S, '    return (-needs_more, -cat, -eps, -row["max_settlement_mm"])', '    return (-needs_more, -eps, -cat, -row["max_settlement_mm"])'),
    (S, '    z_invert = et["axis_depth_m"] + et["diameter_m"] / 2.0', '    z_invert = et["axis_depth_m"] - et["diameter_m"] / 2.0'),
    (S, '"exceeds_allowable": smax_mm > et["allowable_total_movement_mm"],', '"exceeds_allowable": smax_mm < et["allowable_total_movement_mm"],'),
    # ---- uncertainty
    (U, "    sigma = Z95 - math.sqrt(disc)", "    sigma = Z95 + math.sqrt(disc)"),
    (U, "    mu = math.log(mean) - 0.5 * sigma ** 2", "    mu = math.log(mean)"),
    (U, "    return mean + t * sd", "    return mean + sd"),
    (U, "    centre = (p + z ** 2 / (2 * n)) / den", "    centre = p"),
    (U, "    z2 = correlation * z1 + math.sqrt(max(0.0, 1.0 - correlation ** 2)) * rng.standard_normal(n)", "    z2 = rng.standard_normal(n)"),
    (U, "        stage1_fail = ~((smax < bld.STAGE1_SETTLEMENT_M) & (slope < bld.STAGE1_SLOPE))", "        stage1_fail = ~((smax < bld.STAGE1_SETTLEMENT_M) | (slope < bld.STAGE1_SLOPE))"),
    (U, "            final = np.minimum(5, cats + sum(b.indices))", "            final = cats"),
    (U, "                eps[j] = max(e_both, e_first, e_long)", "                eps[j] = max(e_both, e_first)"),
    (U, '{str(c): _prob(cats >= c) for c in (1, 2, 3, 4)}', '{str(c): _prob(cats > c) for c in (1, 2, 3, 4)}'),
    (U, "        vl2 = vl2 * factor", "        vl2 = vl2 * 1.0"),
    (U, '"p_vl_exceeds_design": _prob(vl1 > z["design"]["VL"]),', '"p_vl_exceeds_design": _prob(vl1 < z["design"]["VL"]),'),
    (U, '    rows.sort(key=lambda r: -r["swing_pct"])', '    rows.sort(key=lambda r: r["swing_pct"])'),
    (U, "for q in (-Z95, 0.0, Z95))", "for q in (-1.0, 0.0, 1.0))"),
    (U, "sc.tunnels_at(scn, b.chainage, K=float(k), VL=1.0, second_factor=1.0)", "sc.tunnels_at(scn, b.chainage, K=float(k), VL=2.0, second_factor=1.0)"),
    # ---- monitoring: layout
    (M, "    return int(math.ceil(length_m / spacing_m - 1e-9))", "    return int(math.floor(length_m / spacing_m))"),
    (M, "TYPE_E_OFFSETS = (-1.5, -1.0, -0.75,", "TYPE_E_OFFSETS = (-1.5, -1.25, -0.75,"),
    (M, "            if centre and not kept[-1][1]:", "            if False:"),
    (M, '        if b.in_use and row["above_tunnel"]:', "        if b.in_use:"),
    (M, "                ch = front - d", "                ch = front + d"),
    (M, "    return max(abs(y) for y in tunnel_offsets) + MONITORING_ZONE_FACTOR * z0", "    return MONITORING_ZONE_FACTOR * z0"),
    (M, "TENSILE_STRAIN_MONITORING_PCT = 0.1", "TENSILE_STRAIN_MONITORING_PCT = 0.05"),
    (M, '    ts = sc.tunnels_at(scn, chainage, VL=zone["observed_vl"]["mean"])', '    ts = sc.tunnels_at(scn, chainage, VL=zone["observed_vl"]["characteristic_95"])'),
    # ---- monitoring: review levels and trigger logic
    (M, "    suspension = max(predicted_design_mm, min_suspension_mm)", "    suspension = predicted_design_mm"),
    (M, '    return {"alert_mm": alert_factor * suspension, "suspension_mm": suspension}', '    return {"alert_mm": alert_factor * predicted_design_mm, "suspension_mm": suspension}'),
    (M, "    if reading_mm >= suspension_mm or -reading_mm >= heave_suspension_mm:", "    if reading_mm > suspension_mm or -reading_mm >= heave_suspension_mm:"),
    (M, "    if reading_mm >= suspension_mm or -reading_mm >= heave_suspension_mm:", "    if reading_mm >= suspension_mm or -reading_mm >= suspension_mm:"),
    (M, "    if reading_mm >= suspension_mm or -reading_mm >= heave_suspension_mm:", "    if reading_mm >= suspension_mm or -reading_mm > heave_suspension_mm:"),
    (M, "    if reading_mm >= alert_mm or -reading_mm >= heave_alert_mm:", "    if reading_mm > alert_mm or -reading_mm >= heave_alert_mm:"),
    (M, "    if reading_mm >= alert_mm or -reading_mm >= heave_alert_mm:", "    if reading_mm >= alert_mm or reading_mm >= heave_alert_mm:"),
    (M, "    if reading_mm >= alert_mm or -reading_mm >= heave_alert_mm:", "    if reading_mm >= alert_mm and -reading_mm >= heave_alert_mm:"),
    (M, "    return reading_mm > predicted_stage_mm + tolerance_mm", "    return reading_mm >= predicted_stage_mm + tolerance_mm"),
    (M, "    return reading_mm > predicted_stage_mm + tolerance_mm", "    return reading_mm > predicted_stage_mm - tolerance_mm"),
    # ---- back-analysis
    (A, "        z = norm.ppf(0.5 + level / 2.0)", "        z = norm.ppf(level)"),
    (A, "        return float(norm.sf((vl_limit - self.VL) / se))", "        return float(norm.cdf((vl_limit - self.VL) / se))"),
    (A, "    cov = sigma ** 2 * np.linalg.inv(J.T @ J)", "    cov = sigma * np.linalg.inv(J.T @ J)"),
    (A, "    dof = max(1, y.size - 2)", "    dof = max(1, y.size)"),
    (A, "        return model_mm(y, p[0], p[1], z0, D, y0) - s", "        return model_mm(y, p[0], p[1], z0, D) - s"),
    (A, "    K = math.sqrt(-1.0 / (2.0 * slope))", "    K = math.sqrt(-1.0 / slope)"),
    (A, "    X = (y[keep] / z0) ** 2", "    X = np.abs(y[keep] / z0)"),
    (A, "    vs = gf.SQRT_2PI * K * z0 * smax / 1000.0", "    vs = 2.5 * K * z0 * smax / 1000.0"),
    (A, "    near = np.abs(y) <= centre_window_m", "    near = np.abs(y) <= 10.0 * centre_window_m"),
    (A, "    z = (r - p) / sigma_mm", "    z = (p - r) / sigma_mm"),
    (A, "    flagged = bool(np.max(np.abs(z)) > z_limit or pval < p_limit)", "    flagged = bool(np.max(np.abs(z)) > z_limit and pval < p_limit)"),
    (A, "    flagged = bool(np.max(np.abs(z)) > z_limit or pval < p_limit)", "    flagged = bool(np.max(np.abs(z)) >= z_limit or pval < p_limit)"),
    (A, "    pval = float(chi2.sf(stat, df=z.size))", "    pval = float(chi2.sf(stat, df=z.size - 2))"),
    (A, "    return reading_mm / frac", "    return reading_mm * frac"),
    (A, "    if frac < min_fraction:", "    if frac > min_fraction:"),
    # ---- workbook formulas
    (W, '(13, "i = K z0 (m)", "=B6*B5", "S3 Eq. 2")', '(13, "i = K z0 (m)", "=B6+B5", "S3 Eq. 2")'),
    (W, '            s2ws[f"{c}21"] = "=$B$6^3/3"', '            s2ws[f"{c}21"] = "=$B$6^3/12"'),
    (W, 's2ws[f"{c}28"] = f"={c}20*(1-$B$8)/2+SQRT(({c}20*(1+$B$8)/2)^2+{c}26^2)"', 's2ws[f"{c}28"] = f"={c}20*(1-$B$8)/2+SQRT(({c}20*(1-$B$8)/2)^2+{c}26^2)"'),
    (W, 'IF(H6>=0.075,2,IF(H6>=0.05,1,0))))"', 'IF(H6>=0.075,2,IF(H6>=0.06,1,0))))"'),
    (W, '        return f"{smax}*EXP(-(({x}-{cen})^2)/(2*{ii}^2))"', '        return f"{smax}*EXP(-({x}-{cen})^2/(2*{ii}^2))"'),
]
FIRST_SET = len(MUTANTS)  # 117: the list as it stood for the first run

MUTANTS += [
    # ---- added in the review: greenfield
    (G, "SQRT_2PI = math.sqrt(2.0 * math.pi)", "SQRT_2PI = math.sqrt(math.pi)"),
    (G, "    return np.exp(-0.5 * xi ** 2) / SQRT_2PI", "    return np.exp(-xi ** 2) / SQRT_2PI"),
    (G, "    lo = min(t.y0 - TROUGH_LIMIT_FACTOR * _i(t) for t in tunnels)", "    lo = min(t.y0 - TROUGH_LIMIT_FACTOR * t.z0 for t in tunnels)"),
    (G, "    return float(settlement_one(y, t)) * face_factor(x, x_face, _i(t))", "    return float(settlement_one(y, t)) * face_factor(x, x_face, t.z0)"),
    (G, "        out = out + smax_z * np.exp(-(y - t.y0) ** 2 / (2.0 * i_z ** 2))", "        out = out + smax_z * np.exp(-(y - t.y0) ** 2 / (2.0 * _i(t) ** 2))"),
    # ---- added: building
    (B, 'E_OVER_G = {"masonry": 2.6, "framed": 12.5}', 'E_OVER_G = {"masonry": 12.5, "framed": 2.6}'),
    (B, "CATEGORY_LIMITS = (0.0005, 0.00075, 0.0015, 0.003)", "CATEGORY_LIMITS = (0.0005, 0.00075, 0.002, 0.003)"),
    (B, "CATEGORY_LIMITS = (0.0005, 0.00075, 0.0015, 0.003)", "CATEGORY_LIMITS = (0.0005, 0.00075, 0.0015, 0.0025)"),
    (B, '    if mode == "hogging" and hogging_neutral_axis == "bottom":', '    if mode == "sagging" and hogging_neutral_axis == "bottom":'),
    (B, "    eps_d = dr / diagonal_factor(L, H, e_over_g, mode, hogging_neutral_axis)", "    eps_d = dr / bending_factor(L, H, e_over_g, mode, hogging_neutral_axis)"),
    (B, "    chord = s[0] + (s[-1] - s[0]) * (xs - a) / L", "    chord = s[0] + 0.0 * xs"),
    (B, "    for j in np.nonzero(k[:-1] * k[1:] < 0.0)[0]:", "    for j in np.nonzero(k[:-1] * k[1:] < 0.0)[0][:1]:"),
    (B, '    if not compression_relief and mode == "sagging":', '    if compression_relief and mode == "sagging":'),
    # ---- added: screening
    (S, "    return b.y_start < t.y0 + t.D / 2.0 and b.y_end > t.y0 - t.D / 2.0", "    return b.y_start < t.y0 < b.y_end"),
    (S, "    return float(np.interp(chainage, [p[0] for p in pts], [p[1] for p in pts]))", "    return float(pts[0][1])"),
    (S, "    half = gf.TROUGH_LIMIT_FACTOR * gf.trough_width(t.K, t.z0)", "    half = gf.trough_width(t.K, t.z0)"),
    (S, '            vl = zone["observed_vl"]["mean"]', '            vl = zone["observed_vl"]["characteristic_95"]'),
    (S, '"exceeds_differential_gradient": grad > 1.0 / limit_1_in,', '"exceeds_differential_gradient": grad > limit_1_in,'),
    (S, "    grad = float(np.max(np.abs(np.gradient(s, ys))))", "    grad = float(np.max(np.abs(np.gradient(gf.settlement(ys, tunnels), ys))))"),
    (S, '    if foundation == "piled":', '    if foundation == "mixed":'),
    # ---- added: uncertainty
    (U, "rng.uniform(k_lo, k_hi, n)", "np.full(n, 0.5 * (k_lo + k_hi))"),
    (U, "    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den", "    half = z * math.sqrt(p * (1 - p) / n) / den"),
    (U, '    return {"p": k / n, "lo95": lo, "hi95": hi, "count": k}', '    return {"p": k / (n + 1), "lo95": lo, "hi95": hi, "count": k}'),
    (U, "            smax[j] = float(np.max(gf.settlement(ys, ts)))", "            smax[j] = float(np.max(gf.settlement(ys, ts[:1])))"),
    (U, "max(float(np.max(gf.settlement_one(ys, t))) for t in ts) / (i * gf.SQRT_2PI))", "max(float(np.max(gf.settlement_one(ys, t))) for t in ts) / i)"),
    # ---- added: monitoring
    (M, "TYPE_D_SPACING_M = 25.0", "TYPE_D_SPACING_M = 50.0"),
    (M, "TYPE_E_SPACING_M = 200.0", "TYPE_E_SPACING_M = 250.0"),
    (M, "APPROACH_ARRAY_DISTANCES_M = (25.0, 50.0)", "APPROACH_ARRAY_DISTANCES_M = (25.0, 75.0)"),
    (M, "        pts.update(round(v, 2) for v in bld.inflexion_points(prof, lo, hi))", "        pts.update([])"),
    (M, '"tensile_strain_monitoring": bool(eps is not None and eps > TENSILE_STRAIN_MONITORING_PCT),', '"tensile_strain_monitoring": bool(eps is not None),'),
    # ---- added: back-analysis
    (A, "    keep = (s > min_fraction * smax) & ~np.isclose(y, 0.0)", "    keep = ~np.isclose(y, 0.0)"),
    (A, "    VL = 100.0 * vs / (math.pi * D ** 2 / 4.0)", "    VL = vs / (math.pi * D ** 2 / 4.0)"),
    (A, "    return model_mm(y, VL, K, z0, D, y0) + rng.normal(0.0, sigma_mm, y.size)", "    return model_mm(y, VL, K, z0, D, y0) + rng.normal(0.0, 2.0 * sigma_mm, y.size)"),
    # ---- added: workbook formulas
    (W, '"=B14/(SQRT(2*PI())*B13)"', '"=B14/(2.5*B13)"'),
    (W, '        return f"(-({x}-{cen})/{z0c}*{s_of(x)})"', '        return f"(({x}-{cen})/{z0c}*{s_of(x)})"'),
    (W, "{c}15/(12*{c}22)", "{c}15/(6*{c}22)"),
    (W, "1+$B$6*{c}15^2/(18*{c}21*$B$7)", "1+$B$6*{c}15^2*$B$7/(18*{c}21)"),
    (W, '"=MIN($B$5,$B$9+2.5*$B$10)"', '"=MIN($B$5,$B$9+3*$B$10)"'),
    # ---- added: trigger logic of the back-analysis demonstration
    (P, '            if first_alert is None and state != "normal":', '            if first_alert is None and state == "work_suspension":'),
    (P, '            if first_projected is None and proj is not None and proj >= lv0["alert_mm"]:', '            if first_projected is None and proj is not None and proj >= lv0["suspension_mm"]:'),
    (P, "            proj = ba.project_final(r, ch, xf, design_k * z0)", '            proj = ba.project_final(r, ch, xf, c["true_K"] * z0)'),
    (P, "        dep_expected = ba.departure_test(readings, pred_expected, sigma)", "        dep_expected = ba.departure_test(readings, pred_design, sigma)"),
]

# (index in MUTANTS) -> reason no test can tell the mutant from the original
EQUIVALENT: dict[int, str] = {}


def run_one(index: int) -> bool:
    rel, old, new = MUTANTS[index]
    path = ROOT / rel
    src = path.read_text(encoding="utf-8")
    if src.count(old) != 1:
        raise SystemExit(f"mutant {index}: expected exactly one occurrence in {rel} of: {old!r} (found {src.count(old)})")
    try:
        path.write_text(src.replace(old, new), encoding="utf-8", newline="\n")
        res = subprocess.run(PYTEST, cwd=ROOT, capture_output=True, text=True)
        return res.returncode != 0
    finally:
        path.write_text(src, encoding="utf-8", newline="\n")


def collected() -> int:
    res = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
    for line in res.stdout.splitlines()[::-1]:
        if "tests collected" in line or "test collected" in line:
            return int(line.split()[0])
    raise SystemExit("could not count the tests:\n" + res.stdout[-500:])


def main() -> int:
    args = sys.argv[1:]
    key = args[args.index("--record") + 1] if "--record" in args else "ci"
    only = [int(v) for v in args[args.index("--only") + 1].split(",")] if "--only" in args else None
    start = int(args[args.index("--from") + 1]) if "--from" in args else 0
    for i, (rel, old, _) in enumerate(MUTANTS):  # every mutant must still apply to exactly one place
        n = (ROOT / rel).read_text(encoding="utf-8").count(old)
        if n != 1:
            raise SystemExit(f"mutant {i}: expected exactly one occurrence in {rel} (found {n}): {old!r}")
    base = subprocess.run(PYTEST, cwd=ROOT, capture_output=True, text=True)
    if base.returncode != 0:
        print(base.stdout[-2000:])
        raise SystemExit("the unmutated suite must pass before mutation testing")
    survivors, killed, total = [], 0, 0
    for i in (only if only is not None else range(start, len(MUTANTS))):
        if i in EQUIVALENT:
            continue
        total += 1
        if run_one(i):
            killed += 1
        else:
            rel, old, new = MUTANTS[i]
            survivors.append({"id": i, "file": rel, "original": old.strip(), "mutation": new.strip()})
            print(f"SURVIVED {i}: {rel}: {old.strip()}  ->  {new.strip()}", flush=True)
    print(f"{key}: killed {killed} of {total}", flush=True)
    if only is None:
        record = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
        record[key] = {"total": total, "killed": killed, "mutants": f"{start} to {len(MUTANTS) - 1}", "survivors": survivors}
        record["n_mutants"] = len(MUTANTS)
        record["tests_collected"] = collected()
        record["equivalent"] = [{"id": i, "file": MUTANTS[i][0], "original": MUTANTS[i][1].strip(), "mutation": MUTANTS[i][2].strip(), "reason": why}
                                for i, why in sorted(EQUIVALENT.items())]
        record["files"] = sorted({m[0] for m in MUTANTS})
        OUT.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 1 if ("--require-all" in args and survivors) else 0


if __name__ == "__main__":
    sys.exit(main())
