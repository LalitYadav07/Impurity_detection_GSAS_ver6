"""Conservative native peak-fate retagging after candidate-5 localization.

H35685 informed this rule and is development data. A different preregistered
group is needed for independent evaluation. Attention and boxes never change.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import numpy as np


POLICY = {
    "version": "focus-v2-post-h35685-peak-fate-conservative/2",
    "local_flank_widths_each_side": 1.0,
    "survival_search_widths_each_side": 1.0,
    "survival_local_radius_fraction_of_box_width": .75,
    "flank_baseline_quantile": .20,
    "min_center_native_bins": 3,
    "min_flank_native_bins_each_side": 6,
    "min_survival_valley_native_bins_each_side": 3,
    "min_local_good_fraction": .90,
    "max_native_spacing_factor": 3.0,
    "before_apex_edge_margin_fraction_of_box_width": .10,
    "max_flank_baseline_difference_fraction_of_before_prominence": .75,
    "max_before_halfheight_width_fraction_of_box": .65,
    "halfheight_median_filter_bins": 3,
    "strong_negative_z": -2.,
    "negative_bin_count_min": 3,
    "negative_to_positive_count_min": 3.0,
    "strong_positive_z": 2.,
    "before_peak_prominence_sigma_min": 10.,
    "after_surviving_peak_prominence_sigma_min": 4.,
    "survival_prominence_fraction_of_before_min": .25,
    "survival_new_peak_prominence_factor_min": 1.8,
    "positive_residual_lobe_z_min": 2.,
    "positive_residual_lobe_run_bins_min": 3,
    "positive_residual_lobe_prominence_fraction_of_before_min": .20,
    "positive_residual_lobe_prominence_sigma_min": 3.,
    "positive_residual_lobe_new_peak_factor_min": 1.5,
    "relative_tag_to_disappearance_ratio_max_exclusive": .18,
    "unnamed_to_disappearance_ratio_max_exclusive": .12,
    "unnamed_before_peak_prominence_sigma_min": 20.,
    "eligible_named_types": ["relative_intensity_change", "peak_disappearance"],
    "broad_events": "unchanged",
}
POLICY_SHA256 = hashlib.sha256(json.dumps(
    POLICY, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _native_arrays(matched: dict) -> dict[str, np.ndarray]:
    fields = ("x", "good", "before_y", "after_y", "before_sigma",
              "after_sigma", "direct_z")
    arrays = {name: np.asarray(matched[name]) for name in fields}
    x = arrays["x"]
    if x.ndim != 1 or any(value.shape != x.shape for value in arrays.values()) or \
            len(x) < 2 or not np.all(np.isfinite(x)) or not np.all(np.diff(x) > 0):
        raise ValueError("Native pair arrays are misaligned")
    if arrays["good"].dtype != np.bool_:
        raise ValueError("Native pair support must be boolean")
    return arrays


def _local_prominence(a: dict[str, np.ndarray], index: int, radius: float,
                      name: str) -> float | None:
    """Two-sided local prominence at one exact native bin, if measured."""
    x, good = a["x"], a["good"]
    left = good & (x >= x[index]-radius) & (x < x[index])
    right = good & (x > x[index]) & (x <= x[index]+radius)
    minimum = POLICY["min_survival_valley_native_bins_each_side"]
    if int(np.sum(left)) < minimum or int(np.sum(right)) < minimum:
        return None
    y = a[name]
    q = POLICY["flank_baseline_quantile"]
    return float(y[index] - max(np.quantile(y[left], q),
                                np.quantile(y[right], q)))


def _halfheight_width_fraction(a: dict[str, np.ndarray], center: np.ndarray,
                               apex: int, baseline: float,
                               prominence: float, width: float) -> float | None:
    """Measured half-height span inside the box; None means no two crossings."""
    x, y, good = a["x"], a["before_y"], a["good"]
    first, last = np.flatnonzero(center)[[0, -1]]
    half = baseline + .5*prominence
    filter_bins = POLICY["halfheight_median_filter_bins"]
    if not isinstance(filter_bins, int) or filter_bins < 1 or filter_bins % 2 != 1:
        raise ValueError("Half-height median width must be a positive odd integer")
    radius = filter_bins//2

    def level(index: int) -> float | None:
        if (index < radius or index >= len(x)-radius or
                not np.all(good[index-radius:index+radius+1])):
            return None
        return float(np.median(y[index-radius:index+radius+1]))

    if level(apex) is None or level(apex) <= half:
        return None
    left, right = apex, apex
    while left >= first and level(left) is not None and level(left) > half:
        left -= 1
    while right <= last and level(right) is not None and level(right) > half:
        right += 1
    if (left < first or right > last or level(left) is None or
            level(right) is None):
        return None
    return float((x[right]-x[left])/width)


def _nearby_survivor(a: dict[str, np.ndarray], center: np.ndarray,
                     search: np.ndarray, width: float,
                     before_prominence: float) -> dict | None:
    """Find a newly prominent after maximum near the boxed before peak.

    A pre-existing neighboring peak is insufficient: its after local contrast
    must rise substantially over its own before local contrast. This also
    catches a shifted/merged peak when its net z is not positive.
    """
    x, good, after, z = a["x"], a["good"], a["after_y"], a["direct_z"]
    radius = POLICY["survival_local_radius_fraction_of_box_width"] * width
    for index in np.flatnonzero(search & ~center):
        if (index == 0 or index == len(x)-1 or not good[index-1] or
                not good[index+1]):
            continue
        after_max = (after[index] >= after[index-1] and
                     after[index] > after[index+1])
        positive_residual_max = (z[index] >= POLICY["positive_residual_lobe_z_min"] and
                                 z[index] >= z[index-1] and z[index] > z[index+1])
        if not after_max and not positive_residual_max:
            continue
        after_prom = _local_prominence(a, index, radius, "after_y")
        before_prom = _local_prominence(a, index, radius, "before_y")
        if after_prom is None or before_prom is None:
            continue
        strength = after_prom / float(a["after_sigma"][index])
        nearby = good & (x >= x[index]-radius) & (x <= x[index]+radius)
        positive_indices = np.flatnonzero(
            nearby & (a["direct_z"] >= POLICY["positive_residual_lobe_z_min"]))
        run = 0
        if len(positive_indices):
            run = int(max(map(len, np.split(
                positive_indices, np.flatnonzero(np.diff(positive_indices) > 1)+1))))
        surviving_peak = (
            after_max and
            strength >= POLICY["after_surviving_peak_prominence_sigma_min"] and
                after_prom >= POLICY["survival_prominence_fraction_of_before_min"] *
                before_prominence and
                after_prom >= POLICY["survival_new_peak_prominence_factor_min"] *
                max(0., before_prom))
        positive_lobe = (
            positive_residual_max and
            run >= POLICY["positive_residual_lobe_run_bins_min"] and
            strength >= POLICY["positive_residual_lobe_prominence_sigma_min"] and
            after_prom >= POLICY["positive_residual_lobe_prominence_fraction_of_before_min"] *
            before_prominence and
            after_prom >= POLICY["positive_residual_lobe_new_peak_factor_min"] *
            max(0., before_prom))
        if surviving_peak or positive_lobe:
            return {"survivor_x": float(x[index]),
                    "survivor_after_prominence": after_prom,
                    "survivor_before_prominence_at_x": before_prom,
                    "survivor_after_prominence_sigma": strength,
                    "survivor_positive_residual_run_bins": run,
                    "survivor_direct_z": float(a["direct_z"][index])}
    return None


def evidence(event: dict, matched: dict) -> dict:
    """Assess disappearance only for an isolated before peak on exact bins."""
    a = _native_arrays(matched)
    lo, hi = float(event["x0"]), float(event["x1"])
    if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
        raise ValueError("Invalid local event box")
    width = hi-lo
    flank_width = POLICY["local_flank_widths_each_side"] * width
    search_width = POLICY["survival_search_widths_each_side"] * width
    x, good = a["x"], a["good"]
    center = good & (x >= lo) & (x <= hi)
    left = good & (x >= lo-flank_width) & (x < lo)
    right = good & (x > hi) & (x <= hi+flank_width)
    search_all = (x >= lo-search_width) & (x <= hi+search_width)
    search = good & search_all
    counts = {"center_bins": int(np.sum(center)),
              "left_flank_bins": int(np.sum(left)),
              "right_flank_bins": int(np.sum(right))}
    if (counts["center_bins"] < POLICY["min_center_native_bins"] or
            min(counts["left_flank_bins"], counts["right_flank_bins"]) <
            POLICY["min_flank_native_bins_each_side"]):
        return {"status": "insufficient_two_sided_native_bins", **counts}
    local_all = search_all | left | right
    if (int(np.sum(good & local_all)) <
            POLICY["min_local_good_fraction"] * int(np.sum(local_all))):
        return {"status": "incomplete_local_support", **counts}
    measured_x = x[good & local_all]
    if len(measured_x) >= 3 and np.max(np.diff(measured_x)) > \
            POLICY["max_native_spacing_factor"] * np.median(np.diff(measured_x)):
        return {"status": "native_support_gap", **counts}
    for name in ("before_y", "after_y", "before_sigma", "after_sigma", "direct_z"):
        if not np.all(np.isfinite(a[name][good & local_all])):
            raise ValueError("Matched native intensities or errors are invalid")
    if (np.any(a["before_sigma"][good & local_all] <= 0) or
            np.any(a["after_sigma"][good & local_all] <= 0)):
        raise ValueError("Matched native errors must be positive")
    before, after = a["before_y"], a["after_y"]
    apex = np.flatnonzero(center)[int(np.argmax(before[center]))]
    apex_fraction = float((x[apex]-lo)/width)
    margin = POLICY["before_apex_edge_margin_fraction_of_box_width"]
    if not margin <= apex_fraction <= 1.-margin:
        return {"status": "nonisolated_before_peak", **counts,
                "before_apex_x": float(x[apex]),
                "before_apex_fraction_of_box": apex_fraction}
    q = POLICY["flank_baseline_quantile"]
    before_left = float(np.quantile(before[left], q))
    before_right = float(np.quantile(before[right], q))
    after_left = float(np.quantile(after[left], q))
    after_right = float(np.quantile(after[right], q))
    bp = float(before[apex] - max(before_left, before_right))
    ap = float(np.max(after[center]) - max(after_left, after_right))
    if (bp <= 0 or
            abs(before_left-before_right) >
            POLICY["max_flank_baseline_difference_fraction_of_before_prominence"] * bp):
        return {"status": "nonisolated_before_peak", **counts,
                "before_apex_x": float(x[apex]), "before_prominence": bp,
                "before_left_valley": before_left,
                "before_right_valley": before_right}
    halfwidth = _halfheight_width_fraction(
        a, center, apex, max(before_left, before_right), bp, width)
    if (halfwidth is None or
            halfwidth > POLICY["max_before_halfheight_width_fraction_of_box"]):
        return {"status": "broad_before_profile", **counts,
                "before_apex_x": float(x[apex]),
                "before_halfheight_width_fraction_of_box": halfwidth}
    before_strength = bp / float(np.median(a["before_sigma"][center]))
    after_strength = ap / float(np.median(a["after_sigma"][center]))
    negative = int(np.sum(a["direct_z"][center] <= POLICY["strong_negative_z"]))
    positive = int(np.sum(a["direct_z"][center] >= POLICY["strong_positive_z"]))
    metrics = {"center_bins": counts["center_bins"],
               "left_flank_bins": counts["left_flank_bins"],
               "right_flank_bins": counts["right_flank_bins"],
               "before_apex_x": float(x[apex]),
               "before_halfheight_width_fraction_of_box": halfwidth,
               "before_prominence": bp, "after_prominence": ap,
               "before_left_valley": before_left,
               "before_right_valley": before_right,
               "after_to_before_prominence_ratio": ap/bp,
               "before_prominence_sigma": before_strength,
               "after_prominence_sigma": after_strength,
               "strong_negative_bins": negative,
               "strong_positive_bins": positive}
    if after_strength >= POLICY["after_surviving_peak_prominence_sigma_min"]:
        return {"status": "surviving_center_peak", **metrics}
    survivor = _nearby_survivor(a, center, search, width, bp)
    if survivor is not None:
        return {"status": "nearby_surviving_after_peak", **metrics, **survivor}
    return {"status": "valid", **metrics}


def retag_prediction(row: dict, matched: dict) -> tuple[dict, list[dict]]:
    """Retag supported local boxes while preserving candidate-5 geometry."""
    result = deepcopy(row)
    original = [(e["x0"], e["x1"]) for e in row["prediction"]["events"]]
    details = []
    eligible = set(POLICY["eligible_named_types"])
    for index, event in enumerate(result["prediction"]["events"]):
        if event.get("proposal_source") == "native_valley_baseline" or \
                event.get("feature_type_scores") is None:
            continue
        tags = set(event.get("feature_types", []))
        before = sorted(tags)
        if tags - eligible or (tags and "relative_intensity_change" not in tags):
            continue
        measured = evidence(event, matched)
        if measured["status"] != "valid":
            continue
        negative = measured["strong_negative_bins"]
        positive = measured["strong_positive_bins"]
        ratio = measured["after_to_before_prominence_ratio"]
        strength = measured["before_prominence_sigma"]
        one_sided = (negative >= POLICY["negative_bin_count_min"] and
                     negative >= POLICY["negative_to_positive_count_min"] *
                     max(1, positive))
        if not one_sided or strength < POLICY["before_peak_prominence_sigma_min"]:
            continue
        if ("relative_intensity_change" in tags and
                ratio < POLICY["relative_tag_to_disappearance_ratio_max_exclusive"]):
            tags.discard("relative_intensity_change")
            tags.add("peak_disappearance")
        elif (not tags and
              strength >= POLICY["unnamed_before_peak_prominence_sigma_min"] and
              ratio < POLICY["unnamed_to_disappearance_ratio_max_exclusive"]):
            tags.add("peak_disappearance")
        if sorted(tags) != before:
            event["feature_types"] = sorted(tags)
            event["type_status"] = "named" if tags else "uncertain"
            details.append({"event_index": index, "x0": event["x0"],
                            "x1": event["x1"], "previous_types": before,
                            "new_types": sorted(tags), **measured})
    after = [(e["x0"], e["x1"]) for e in result["prediction"]["events"]]
    if after != original:
        raise AssertionError("Peak-fate rule changed box geometry")
    result["peak_fate_policy_sha256"] = POLICY_SHA256
    result["peak_fate_evidence"] = details
    return result, details
