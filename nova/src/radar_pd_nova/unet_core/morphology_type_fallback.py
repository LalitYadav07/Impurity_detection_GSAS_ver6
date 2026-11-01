"""Conservative morphology tags for local boxes left unnamed by type v2.

This experimental rule only changes local feature-type names. It does not
alter attention decisions, event coordinates, source support, or broad boxes.
All thresholds were explored on inspected development data; the rule needs a
new prospective evaluation before release.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import numpy as np

try:
    from .lag_morphology import lag_signature
except ImportError:
    from lag_morphology import lag_signature


POLICY = {
    "version": "focus-v2-experimental-unnamed-morphology-type/1",
    "strong_native_residual_z": 2.,
    "shift_probability_min": .4,
    "shift_strong_bins_each_sign_min": 2,
    "shift_minority_sign_fraction_min": .2,
    "shift_abs_best_lag_bins_min": 1,
    "shift_corr_gain_min": .01,
    "one_sided_shift_probability_min": .6,
    "one_sided_shift_best_corr_min": .7,
    "one_sided_shift_corr_gain_min": .1,
    "relative_intensity_probability_min": .7,
    "relative_intensity_strong_bins_one_sign_min": 3,
    "relative_intensity_minority_sign_fraction_max": .2,
    "relative_intensity_zero_lag_corr_min": .9,
    "relative_intensity_corr_gain_max": .03,
    "appearance_disappearance_fallback": "abstain; development labels too sparse",
    "broad_event": "preserve",
}
POLICY_SHA256 = hashlib.sha256(json.dumps(
    POLICY, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def signed_evidence(event: dict, matched: dict) -> dict:
    x = np.asarray(matched["x"], dtype=float)
    z = np.asarray(matched["direct_z"], dtype=float)
    good = np.asarray(matched["good"], dtype=bool)
    if x.ndim != 1 or z.shape != x.shape or good.shape != x.shape:
        raise ValueError("Native signed arrays differ")
    lo, hi = float(event["x0"]), float(event["x1"])
    if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
        raise ValueError("Invalid local event interval")
    values = z[good & np.isfinite(z) & (x >= lo) & (x <= hi)]
    positive = int(np.sum(values >= POLICY["strong_native_residual_z"]))
    negative = int(np.sum(values <= -POLICY["strong_native_residual_z"]))
    strong = positive+negative
    return {
        "strong_positive_native_bins": positive,
        "strong_negative_native_bins": negative,
        "minority_sign_fraction": min(positive, negative)/strong if strong else 0.,
    }


def tag_unresolved_event(event: dict, matched: dict) -> tuple[dict, dict | None]:
    """Name one unresolved local event when a specific shape test passes."""
    tagged = deepcopy(event)
    if tagged.get("feature_types") or not tagged.get("feature_type_scores"):
        return tagged, None
    scores = tagged["feature_type_scores"]
    if not isinstance(scores, dict) or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or
            not 0 <= value <= 1 for value in scores.values()):
        raise ValueError("Invalid type score")
    signed = signed_evidence(tagged, matched)
    lag = lag_signature(tagged, matched)
    detail = {"signed": signed, "lag": lag, "selected_type": None}
    if lag["status"] != "valid":
        return tagged, detail
    positive = signed["strong_positive_native_bins"]
    negative = signed["strong_negative_native_bins"]
    minority = signed["minority_sign_fraction"]
    shift_probability = scores.get("peak_shift", 0)
    bipolar = (positive >= POLICY["shift_strong_bins_each_sign_min"] and
               negative >= POLICY["shift_strong_bins_each_sign_min"] and
               minority >= POLICY["shift_minority_sign_fraction_min"])
    displaced = (abs(lag["best_lag_bins"]) >= POLICY["shift_abs_best_lag_bins_min"] and
                 lag["corr_gain"] >= POLICY["shift_corr_gain_min"])
    one_sided_shift = (
        shift_probability >= POLICY["one_sided_shift_probability_min"] and
        lag["best_corr"] >= POLICY["one_sided_shift_best_corr_min"] and
        lag["corr_gain"] >= POLICY["one_sided_shift_corr_gain_min"])
    if (shift_probability >= POLICY["shift_probability_min"] and displaced and
            (bipolar or one_sided_shift)):
        tagged["feature_types"] = ["peak_shift"]
        tagged["type_status"] = "named"
        detail["selected_type"] = "peak_shift"
        detail["shift_basis"] = "bipolar" if bipolar else "one_sided_high_shape"
        return tagged, detail
    if (scores.get("relative_intensity_change", 0) >=
            POLICY["relative_intensity_probability_min"] and
            max(positive, negative) >=
            POLICY["relative_intensity_strong_bins_one_sign_min"] and
            minority <= POLICY["relative_intensity_minority_sign_fraction_max"] and
            lag["zero_corr"] >= POLICY["relative_intensity_zero_lag_corr_min"] and
            lag["corr_gain"] <= POLICY["relative_intensity_corr_gain_max"]):
        tagged["feature_types"] = ["relative_intensity_change"]
        tagged["type_status"] = "named"
        detail["selected_type"] = "relative_intensity_change"
    return tagged, detail


def retag_prediction(prediction_row: dict, matched: dict) -> tuple[dict, list[dict]]:
    """Apply after native type policy v2, preserving boxes and pair attention."""
    result = deepcopy(prediction_row)
    evidence = []
    before = [(event["x0"], event["x1"]) for event in
              result["prediction"]["events"]]
    for index, event in enumerate(result["prediction"]["events"]):
        tagged, detail = tag_unresolved_event(event, matched)
        result["prediction"]["events"][index] = tagged
        if detail is not None:
            evidence.append({"event_index": index, "x0": event["x0"],
                             "x1": event["x1"], **detail})
    after = [(event["x0"], event["x1"]) for event in
             result["prediction"]["events"]]
    if before != after:
        raise AssertionError("Type fallback changed box geometry")
    result["experimental_morphology_type_fallback_sha256"] = POLICY_SHA256
    result["experimental_morphology_type_evidence"] = evidence
    return result, evidence
