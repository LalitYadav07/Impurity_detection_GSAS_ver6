"""Experimental candidate-5 type retagging on directly matched native bins.

Only the existing local event's type tags change. Attention, event coordinates,
source support, and broad-region proposals remain exactly as provided.
This policy was explored on inspected development data and needs a different
prospective group before any release claim.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import numpy as np


POLICY = {
    "version": "focus-v2-experimental-native-type-policy/1",
    "local_default_probability": .9,
    "peak_shift_probability": .7,
    "relative_intensity_probability": .8,
    "peak_appearance_probability": .8,
    "strong_native_residual_z": 2.,
    "strong_bins_per_required_sign": 2,
    "strong_bins_single_sign": 3,
    "peak_shift_min_minority_sign_fraction": .1,
    "one_sign_max_minority_sign_fraction": .2,
    "local_background_change": "suppress; broad native-baseline branch owns background",
    "broad_event": "leave unchanged",
}
POLICY_SHA256 = hashlib.sha256(
    json.dumps(POLICY, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def signed_evidence(event: dict, matched: dict) -> dict:
    """Count strong signs from exact measured before/after native-bin matches."""
    x = np.asarray(matched["x"], dtype=float)
    z = np.asarray(matched["direct_z"], dtype=float)
    valid = np.asarray(matched["good"], dtype=bool)
    if x.ndim != 1 or z.shape != x.shape or valid.shape != x.shape:
        raise ValueError("Native signed source arrays differ")
    lo, hi = float(event["x0"]), float(event["x1"])
    if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
        raise ValueError("Invalid local event box")
    values = z[valid & np.isfinite(z) & (x >= lo) & (x <= hi)]
    positive = int(np.sum(values >= POLICY["strong_native_residual_z"]))
    negative = int(np.sum(values <= -POLICY["strong_native_residual_z"]))
    strong = positive+negative
    return {
        "strong_positive_native_bins": positive,
        "strong_negative_native_bins": negative,
        "minority_sign_fraction": min(positive, negative)/strong if strong else 0.,
        "policy_sha256": POLICY_SHA256,
    }


def tag_event(event: dict, matched: dict) -> tuple[dict, dict | None]:
    tagged = deepcopy(event)
    scores = tagged.get("feature_type_scores")
    if scores is None:
        return tagged, None
    if not isinstance(scores, dict) or not scores or any(
            not isinstance(key, str) or not key or
            not isinstance(value, (int, float)) or not math.isfinite(value) or
            not 0 <= value <= 1 for key, value in scores.items()):
        raise ValueError("Invalid feature type scores")
    evidence = signed_evidence(tagged, matched)
    positive = evidence["strong_positive_native_bins"]
    negative = evidence["strong_negative_native_bins"]
    ratio = evidence["minority_sign_fraction"]
    excluded = {"peak_shift", "relative_intensity_change", "peak_emergence",
                "peak_disappearance", "background_change"}
    tags = {name for name, probability in scores.items()
            if name not in excluded and
            probability >= POLICY["local_default_probability"]}
    if (scores.get("peak_shift", 0) >= POLICY["peak_shift_probability"] and
            positive >= POLICY["strong_bins_per_required_sign"] and
            negative >= POLICY["strong_bins_per_required_sign"] and
            ratio >= POLICY["peak_shift_min_minority_sign_fraction"]):
        tags.add("peak_shift")
    if (scores.get("relative_intensity_change", 0) >=
            POLICY["relative_intensity_probability"] and
            max(positive, negative) >= POLICY["strong_bins_single_sign"] and
            ratio <= POLICY["one_sign_max_minority_sign_fraction"]):
        tags.add("relative_intensity_change")
    if ratio <= POLICY["one_sign_max_minority_sign_fraction"]:
        if (scores.get("peak_emergence", 0) >=
                POLICY["peak_appearance_probability"] and
                positive >= POLICY["strong_bins_single_sign"]):
            tags.add("peak_emergence")
        if (scores.get("peak_disappearance", 0) >=
                POLICY["peak_appearance_probability"] and
                negative >= POLICY["strong_bins_single_sign"]):
            tags.add("peak_disappearance")
    tagged["feature_types"] = sorted(tags)
    tagged["type_status"] = "named" if tags else "uncertain"
    return tagged, evidence


def retag_prediction(prediction_row: dict, matched: dict) -> tuple[dict, list[dict]]:
    result = deepcopy(prediction_row)
    evidence = []
    for index, event in enumerate(result["prediction"]["events"]):
        tagged, detail = tag_event(event, matched)
        result["prediction"]["events"][index] = tagged
        if detail is not None:
            evidence.append({"event_index": index, "x0": event["x0"],
                             "x1": event["x1"], **detail})
    result["experimental_native_type_policy_sha256"] = POLICY_SHA256
    result["experimental_native_type_evidence"] = evidence
    return result, evidence
