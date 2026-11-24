"""The back-analysis demonstration in the pipeline: every reported state, trigger and estimate is
recomputed here from the readings it stores, with the functions of the package."""
import numpy as np
import pytest

from tunnelscreen import backanalysis as ba
from tunnelscreen import greenfield as gf
from tunnelscreen import monitoring as mon
from tunnelscreen import pipeline
from tunnelscreen import screening as sc


@pytest.fixture(scope="module")
def demo(scn, buildings):
    rows = sc.screen_all(scn, buildings, "design")
    return pipeline.backanalysis_demo(scn, mon.instrumentation_plan(scn, buildings, rows))


def test_estimates_and_departure_flags_follow_from_the_readings(scn, demo):
    D, sigma = scn["alignment"]["excavated_diameter_m"], demo["sigma_mm"]
    for c in demo["cases"]:
        y, r, z0 = np.array(c["offsets_m"]), np.array(c["readings_mm"]), c["axis_depth_m"]
        fit = ba.fit_gaussian(y, r, z0, D, sigma_mm=sigma)
        assert (c["estimate"]["VL"], c["estimate"]["K"]) == (pytest.approx(fit.VL, rel=1e-9), pytest.approx(fit.K, rel=1e-9))
        expected = ba.model_mm(y, c["expected"]["VL"], c["expected"]["K"], z0, D)
        dep = ba.departure_test(r, expected, sigma)
        assert c["departure_from_expected"]["flagged"] is dep["flagged"]
        assert c["departure_from_expected"]["chi2"] == pytest.approx(dep["chi2"])
        a = scn["assumptions"]
        design = ba.model_mm(y, c["design"]["VL"], c["design"]["K"], z0, D)
        states = []
        for reading, pred in zip(r, design):
            lv = mon.review_levels(float(pred), a["alert_factor"]["value"], a["min_suspension_level_mm"]["value"])
            states.append(mon.classify_reading(float(reading), lv["alert_mm"], lv["suspension_mm"],
                                               a["heave_alert_mm"]["value"], a["heave_suspension_mm"]["value"]))
        assert c["review_states"] == {s: states.count(s) for s in ("normal", "alert", "work_suspension")}
        assert c["max_slope_fitted_1_in"] == pytest.approx(1 / gf.max_slope_single(gf.Tunnel(0.0, z0, D, fit.VL, fit.K)), rel=1e-9)


def test_sequence_of_readings_and_triggers(scn, demo):
    D, sigma = scn["alignment"]["excavated_diameter_m"], demo["sigma_mm"]
    a = scn["assumptions"]
    for c in demo["cases"]:
        cm, z0 = c["centre_marker"], c["axis_depth_m"]
        first_alert = first_susp = first_proj = None
        for s in cm["sequence"]:
            state = mon.classify_reading(s["reading_mm"], cm["alert_mm"], cm["suspension_mm"],
                                         a["heave_alert_mm"]["value"], a["heave_suspension_mm"]["value"])
            assert s["state"] == state
            proj = ba.project_final(s["reading_mm"], c["chainage_m"], c["chainage_m"] + s["face_minus_array_D"] * D,
                                    c["design"]["K"] * z0)
            assert s["projected_final_mm"] == (None if proj is None else pytest.approx(proj, rel=1e-12))
            assert s["exceeds_stage_prediction"] is mon.exceeds_stage_prediction(s["reading_mm"], s["stage_prediction_mm"], 3 * sigma)
            if first_alert is None and state != "normal":
                first_alert = s["face_minus_array_D"]
            if first_susp is None and state == "work_suspension":
                first_susp = s["face_minus_array_D"]
            if first_proj is None and proj is not None and proj >= cm["alert_mm"]:
                first_proj = s["face_minus_array_D"]
        assert (cm["first_review_state_at_D"], cm["first_suspension_at_D"], cm["first_projection_reaching_alert_at_D"]) == \
            (first_alert, first_susp, first_proj)
    # case B: the projection gives warning before the reading itself reaches the alert level
    cb = demo["cases"][1]["centre_marker"]
    assert cb["first_projection_reaching_alert_at_D"] < cb["first_review_state_at_D"]
