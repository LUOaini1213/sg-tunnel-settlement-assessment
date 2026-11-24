# Formula audit

> Personal portfolio study built with AI assistance; not affiliated with or endorsed by LTA. Student-level screening
> study, not a design. This page is the final self-review: every implemented equation is set against the equation
> in its source, with the test that would fail if the code departed from it.

Source IDs are those of [DATA_SOURCES.md](../DATA_SOURCES.md). "Read from the page" means the equation is typeset
in a scanned document and was read from the page image, not from extracted text. Sign conventions used in the
code: settlement positive downwards; horizontal displacement positive in the +y (or +x) direction; strain positive
in tension; sagging = ground surface concave upwards.

## Ground movement (`tunnelscreen/greenfield.py`)

| # | Function | Equation as implemented | Source equation | Verified by (tests) | Review note |
|---|---|---|---|---|---|
| 1 | `trough_width` | i = K z0 | S3 Eq. 2; S1 Cl. 14.2.1.2 | `test_hand_worked_example` | z0 is the depth to the tunnel axis, not to the crown. |
| 2 | `trough_volume` | Vs = (V/100) pi D^2 / 4 | S3 Eq. 4 | `test_volume_loss_is_percent_of_face_area` | V is in per cent, as in S1; S3 writes it as a fraction. D is the excavated diameter. |
| 3 | `s_max` | Smax = Vs / (sqrt(2 pi) i) | S3 Eq. 3 and 5; S1 Cl. 14.2.1.2 | `test_lta_formula_coefficient_0_0031`, `test_mair_taylor_burland_eq5_coefficient_0_31`, `test_trough_volume_equals_volume_loss_numerically` | The printed coefficients 0.0031 and 0.31 are sqrt(pi/2)/4 rounded; the code keeps the exact value (1.1% larger). |
| 4 | `settlement_one` | S = Smax exp(-y^2 / (2 i^2)) | S9 section 1.4.6; S3 Eq. 1 | `test_peck_fig8_properties_of_the_error_function` | The 2 in the denominator is the usual slip; a mutant without it is killed. |
| 5 | `slope_one` | dS/dy = -(y / i^2) S | derivative of 4 | `test_slope_is_derivative_of_settlement`, `test_max_slope_is_at_the_point_of_inflexion` | Largest slope at y = i: 0.607 Smax / i. |
| 6 | `curvature_one` | d2S/dy2 = (S / i^2)(y^2/i^2 - 1) | derivative of 5 | `test_curvature_is_second_derivative_and_changes_sign_at_i` | Negative (sagging) for \|y\| < i. The point of inflexion is at i, not at 2.5 i (the practical edge of the trough). |
| 7 | `horizontal_displacement_one` | u = -(y / z0) S | S3 Eq. 6; S1 Cl. 14.2.1.8 | `test_horizontal_displacement_points_towards_the_tunnel`, `test_max_horizontal_movement_0_61_k_smax` | The minus sign makes ground on the +y side move towards the centre line. Divisor is z0, not i. |
| 8 | `horizontal_strain_one` | eps_h = (S / z0)(y^2/i^2 - 1) | S3 section 2.2; S6 Eq. 2.16 | `test_horizontal_strain_sign_zero_and_extremes` | Compressive inside the points of inflexion, zero at them, tensile outside: checked against the wording of S3. |
| 9 | `settlement`, `slope`, ... | sum over tunnels | S3 section 2.4 | `test_superposition_of_twin_tunnels` | See the limits of superposition below. |
| 10 | `trough_limits` | y0 +/- 2.5 i, union over tunnels | S3 section 4.1 (single tunnel) | `test_trough_limits_are_two_and_a_half_i_outside_the_outer_tunnels` | The union for two tunnels is this study's extension of the single-tunnel rule. |
| 11 | `face_factor` | Phi((x_face - x) / i) | S6 Eq. 2.9-2.10 (after Attewell and Woodman 1982); S3 section 2.3 | `test_longitudinal_trough_development`, `test_half_smax_above_the_face` | Longitudinal i taken equal to the transverse i, as S6 reports. Applies to open-face tunnelling (S4, S6). |
| 12 | `long_slope_advancing`, `long_curvature_advancing` | derivatives of 11 | derivative | `test_longitudinal_derivatives_and_signs` | Hogging ahead of the face, sagging behind. |
| 13 | `long_displacement_advancing` | u_x = (i^2 / z0) dS/dx | S6 Eq. 2.17 on the centre line (read from the page) | `test_longitudinal_displacement_at_face_matches_s6_eq_2_17`, `test_displacement_rule_is_the_same_in_both_directions` | Off the centre line the transverse Gaussian factor is applied; S6 gives only the centre line, so that part is DERIVED. |
| 14 | `long_strain_advancing` | eps_x = S_trans xi phi(xi) / z0 | S6 Eq. 2.18 (read from the page) | `test_longitudinal_derivatives_and_signs` | Tensile ahead of the face, as S6 states. |
| 15 | `subsurface_trough_width` | i(z) = z0 [0.175 + 0.325 (1 - z/z0)] | S4 Eq. 7-8 (after Mair et al. 1993) | `test_subsurface_trough_width_and_volume`, `test_subsurface_relation_gives_k_half_at_surface` | Published for clay with K = 0.5 at the surface. The K_surface / 0.5 scaling is an ASSUMPTION. |
| 16 | `subsurface_settlement` | Gaussian with i(z) and the same Vs | S4 section 5.4 | `test_subsurface_trough_width_and_volume`, `test_existing_tunnel_check` | Constant trough volume with depth is assumed (undrained movement). |

## Building strains (`tunnelscreen/building.py`)

| # | Function | Equation as implemented | Source equation | Verified by (tests) | Review note |
|---|---|---|---|---|---|
| 17 | `section_properties` | sagging: I = H^3/12, t = H/2; hogging: I = H^3/3, t = H | S3 section 4.2 | `test_section_properties`, `test_hogging_second_moment_is_about_the_bottom_edge` | Hogging: neutral axis at the bottom edge, so I is taken about that edge. S8 uses H^3/12 for both modes; S3, the primary paper, is followed. |
| 18 | `bending_factor` | Delta/L = eps_b [L/(12t) + 3 I (E/G) / (2 t L H)] | S3 Eq. 7 (read from the page); S6 Eq. 2.22; S8 Eq. 53 | `test_beam_factors_from_first_principles`, `test_beam_factor_hand_values_for_l_over_h_one` | Re-derived in the test from a centrally loaded beam with shear area H/1.5. E/G multiplies the shear term. |
| 19 | `diagonal_factor` | Delta/L = eps_d [1 + H L^2 / (18 I (E/G))] | S3 Eq. 8 (read from the page); S6 Eq. 2.23; S8 Eq. 54 | same two tests; `test_damage_chart_intercepts_on_the_deflection_axis` | E/G divides here. |
| 20 | `total_bending_strain` | eps_bt = eps_h + eps_b | S3 Eq. 9 | `test_total_bending_strain_adds_horizontal_strain` | Compressive eps_h (sagging zone) reduces the total, as the equation is written. |
| 21 | `total_diagonal_strain` | eps_dt = eps_h (1-nu)/2 + sqrt((eps_h (1+nu)/2)^2 + eps_d^2) | S3 Eq. 10 (0.35 and 0.65 for nu = 0.3); S8 Eq. 52 | `test_total_diagonal_strain_is_the_major_principal_strain`, `test_eq10_coefficients_0_35_and_0_65`, `test_general_form_of_s8_eq52_reduces_to_eq10` | S6 Eq. 2.25 prints (1 - nu)/2 inside the root. That contradicts S3, S8 and Mohr's circle, and is treated as a misprint. |
| 22 | `damage_category` | limits 0.05, 0.075, 0.15, 0.3 % | S1 Table 14.2; S3 Table 2 | `test_damage_category_boundaries`, `test_damage_chart_intercepts_on_the_strain_axis` | A strain exactly on a boundary goes to the higher category (ASSUMPTION). Category 4 stands for "4 to 5". |
| 23 | `stage1_negligible` | settlement < 10 mm and slope < 1:500 | S1 Section 14.3.6; S3 section 5.2 | `test_stage1_needs_both_settlement_and_slope_below_the_limits` | Both conditions are needed. |
| 24 | `assess_segment` | Delta = largest gap between profile and chord; eps_h = (u_b - u_a) / L | S3 sections 4.2 and 4.3 | `test_hand_worked_hogging_span`, `test_sagging_span_values` | S3 says the maximum of Delta is unlikely to be at mid-span and should be searched numerically; that is what is done. |
| 25 | `assess_facade` | cut at points of inflexion and at the trough limit; largest total strain governs | S3 sections 4.1 and 5.3 | `test_facade_is_cut_at_inflexion_points_and_at_the_trough_limit`, `test_twin_tunnel_profile_is_partitioned_at_every_inflexion_point` | With two tunnels the profile can have two or four points of inflexion; they are found numerically. |
| 26 | `final_category` | Stage 2 category + three indices, capped at 5 | S1 Section 14.3.10 | `test_final_category_adds_sensitivity_indices_and_caps_at_five` | The cap at 5 is this study's choice; the table has no category above 5. |

## Screening rules, uncertainty, monitoring, back-analysis

| # | Function | Rule or equation | Source | Verified by (tests) | Review note |
|---|---|---|---|---|---|
| 27 | `screening.action_for` | low risk only for categories 0 and 1; piled, mixed and poor-condition buildings routed out | S1 Sections 14.3.6 to 14.3.9 | `test_action_rules` | Revision A3 is one category stricter than Revision A2; a mutant with the A2 rule is killed. |
| 28 | `screening.footprint_extremes` | slope = largest of transverse (first drive, both drives) and transient longitudinal S_trans / (i sqrt(2 pi)) | derivative of 11 at the face | `test_footprint_extremes_cover_all_three_slopes` | The transient longitudinal slope is included because a building over the centre line sees almost no transverse slope. |
| 29 | `screening.stage2_cases` | first drive alone, both drives, each face passing | S3 section 2.4 | `test_stage2_cases_first_drive_can_govern` | S3 notes one tunnel can be worse than the pair; building B07 is such a case. |
| 30 | `uncertainty.lognormal_params` | sigma = z95 - sqrt(z95^2 - 2 ln(q95/mean)) | algebra | `test_lognormal_matches_mean_and_95th_percentile` | Shape is an ASSUMPTION; the two moments are from S5 Table 2. |
| 31 | `uncertainty.characteristic_value` | mean + t sd | S5 section 4.5.1 | `test_characteristic_volume_loss_worked_example` | Reproduces the worked example in S5 (0.7%). |
| 32 | `monitoring.review_levels` | work suspension = design-basis prediction (with a floor); alert = factor x that | S1 Cl. 13.7.3-13.7.4 for the definitions | `test_review_levels`, `test_every_instrument_has_consistent_review_levels` | S1 gives no ratio; the factor 0.7, the floor and the heave levels are ASSUMPTIONS. |
| 33 | `monitoring.classify_reading` | settlement and heave judged against their own levels; reaching a level counts | S1 Cl. 13.7.6 to 13.7.8 | `test_classify_reading_settlement_and_heave`, `test_classify_reading_uses_the_right_level_for_each_direction` | |
| 34 | `monitoring.TYPE_E_OFFSETS`, `array_count` | 0.5 Z, 8 x 0.25 Z, 0.5 Z; one array per spacing and part thereof | S1 Figures 13.5 and 13.6 (read from the figure); Cl. 13.4.2.1 | `test_constants_are_the_cited_lta_values`, `test_array_count_every_spacing_and_part_thereof` | |
| 35 | `backanalysis.fit_gaussian` | least squares on (V, K); covariance sigma^2 (J'J)^-1 | standard | `test_noise_free_readings_recover_the_parameters_exactly`, `test_covariance_is_sigma_squared_times_inverse_normal_matrix`, `test_confidence_interval_coverage` | 95% intervals cover the true values in about 95% of repeated synthetic arrays. |
| 36 | `backanalysis.fit_loglinear` | ln(S/Smax) = -(x/z0)^2 / (2 K^2); K = sqrt(-1 / (2 slope)) | S5 section 4.1.1 | `test_loglinear_method_details` | The equations in S5 are images; the relation follows from equations 1 and 2 of that paper. Biased when small readings are kept (see README). |
| 37 | `backanalysis.departure_test` | z = (reading - prediction) / sigma; chi-square on the sum of squares | standard | `test_departure_test` | Thresholds are ASSUMPTIONS. |
| 38 | `backanalysis.project_final` | final = reading / Phi((x_face - x) / i) | inverse of 11 | `test_project_final` | Inherits the open-face limitation of 11. |

## What the review changed

1. **Edition.** The 2019 revision of the Civil Design Criteria is the one most often cited, including in the
   2023 paper used for the volume-loss statistics. The LTA page lists Revision A3 (December 2025) as current, with
   new clause numbers and a stricter Stage 2 rule. Everything was re-read against A3.
2. **Diagonal strain.** One secondary source has a misprint in the square root (row 21). The primary paper was
   then found in an open archive and the equation read from the page.
3. **Hogging against sagging.** A test first written from the sentence "the hogging mode is likely to be more
   damaging" failed. The beam equations give that result only for long spans; the test now records both regimes
   with hand-worked values.
4. **Excel operator precedence.** `-x^2` is `(-x)^2` in Excel. The recalculation test caught a trough with a
   positive exponent; every square is now bracketed and a test forbids the faulty form.
5. **Centre-line marker dropped by array merging.** Merging the overlapping arrays of the two tunnels at first
   discarded the point above one centre line. The merge now always keeps centre-line points.
6. **Straight-line estimator.** With small readings included it over-estimates K and V on synthetic data; the
   default cut-off was raised to 0.2 Smax and the comparison is reported.

## Known limits of superposition and of the longitudinal curve

- S3 section 2.4: superposition "may be unconservative" when the clear separation is less than one diameter, and
  a larger volume loss can be taken for the second tunnel. S10 reports a measured case at about three diameters
  spacing where the first tunnel changed the shape of the second trough. S7 reports superposition as applicable
  on a Singapore line. The code offers a factor on the second drive; the base case uses 1.0.
- S4 and S6: half the final settlement above the face is observed for open-face tunnelling; with face support the
  value is lower (0.25 to 0.40 is quoted). The curve used here therefore develops settlement earlier than a
  closed-face drive would. This affects the timing in the review-level example, not the final settlements.
