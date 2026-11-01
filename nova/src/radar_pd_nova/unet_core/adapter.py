"""Standalone, additive adapter for local diffraction feature predictions.

No transition-state assignment is made here. The input pair must already have
been reduced and scoped by the caller; the adapter verifies the scope before
mapping 156-position model masks back to the native coordinate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import math
from statistics import median

SCHEMA = "radar-pd-local-feature-prediction/v1"
LENGTH = 156
SCOPE_FIELDS = (
    "specimen", "instrument", "bank", "coordinate", "geometry", "reduction",
    "normalization", "intensity_semantics",
)
COORDINATES = {"tof_us", "two_theta_deg", "d_angstrom", "q_inv_angstrom"}


def pattern_from_observed(observed):
    """Convert a RadarPD-like ObservedPattern to the standalone input shape."""
    scope = observed.scope
    scope = asdict(scope) if is_dataclass(scope) else dict(scope)
    result = {"id": str(getattr(observed, "observation_id", "")),
              "scope": scope, "x": observed.x, "intensity": observed.intensity}
    if getattr(observed, "valid", None) is not None:
        result["valid"] = [bool(value) for value in observed.valid]
    return result


def _number(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _interval(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain two bounds")
    lo, hi = _number(value[0], name), _number(value[1], name)
    if lo >= hi:
        raise ValueError(f"{name} must have increasing bounds")
    return lo, hi


def _unavailable(before, after, model, reason, *, scope=None):
    return {
        "schema": SCHEMA,
        "status": "unsupported",
        "attention": "uncertain",
        "reasons": [reason],
        "before_id": str(before.get("id", "")),
        "after_id": str(after.get("id", "")),
        "scope": dict(scope) if scope else None,
        "coordinate": scope.get("coordinate") if scope else None,
        "target_interval": None,
        "coverage_fraction": 0.0,
        "model": dict(model),
        "events": [],
    }


def _validated_pattern(pattern):
    if not isinstance(pattern, Mapping):
        raise ValueError("pattern must be an object")
    scope = pattern.get("scope")
    if not isinstance(scope, Mapping) or any(
        not isinstance(scope.get(field), str) or not scope[field].strip()
        for field in SCOPE_FIELDS
    ):
        raise ValueError("pattern scope must declare every compatibility field")
    if scope["coordinate"] not in COORDINATES:
        raise ValueError("unsupported coordinate")
    x = tuple(_number(v, "x") for v in pattern.get("x", ()))
    raw_y = pattern.get("intensity", ())
    if not (32 <= len(x) <= 100_000) or len(raw_y) != len(x):
        raise ValueError("pattern needs 32–100000 matching x/intensity values")
    if x[0] <= 0 or any(b <= a for a, b in zip(x, x[1:])):
        raise ValueError("pattern x must be positive and increase strictly")
    valid = pattern.get("valid")
    if valid is None:
        valid = (True,) * len(x)
    elif len(valid) != len(x) or any(
        type(v) is not bool and not (
            type(v).__module__.startswith("numpy") and
            type(v).__name__ in ("bool", "bool_")
        ) for v in valid
    ):
        raise ValueError("valid mask must contain one boolean per x value")
    for value, keep in zip(raw_y, valid):
        if keep:
            _number(value, "intensity")
    valid_x = tuple(xi for xi, keep in zip(x, valid) if keep)
    if len(valid_x) < 8:
        raise ValueError("fewer than eight valid measured points")
    return dict(scope), valid_x


def _source_quality(x, lo, hi):
    local = [v for v in x if lo <= v <= hi]
    if len(local) < 8:
        return False
    spacings = [b-a for a, b in zip(local, local[1:])]
    return bool(spacings) and max(spacings) <= 4 * median(spacings)


def _probabilities(value, name):
    if value is None or isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != LENGTH:
        raise ValueError(f"{name} must contain {LENGTH} probabilities")
    result = [_number(v, name) for v in value]
    if any(v < 0 or v > 1 for v in result):
        raise ValueError(f"{name} probabilities must be in [0, 1]")
    return result


def _components(flags):
    start = None
    for index, flag in enumerate([*flags, False]):
        if flag and start is None:
            start = index
        if not flag and start is not None:
            yield start, index
            start = None


def _smooth(values, width):
    if width <= 1:
        return values
    radius = width // 2
    # Zero padding matches the existing prototype's same-length smoothing.
    return [sum(values[max(0, i-radius):min(LENGTH, i+radius+1)]) / width
            for i in range(LENGTH)]


def _mask_events(window, threshold, type_threshold, smooth_bins, target):
    wid = str(window.get("id", ""))
    lo, hi = _interval(window.get("focus_interval"), "focus_interval")
    scores = _probabilities(window.get("attention"), "attention")
    types = window.get("type_probabilities", {})
    if not isinstance(types, Mapping):
        raise ValueError("type_probabilities must be an object")
    types = {str(name): _probabilities(value, f"type {name}")
             for name, value in types.items()}
    smoothed = _smooth(scores, smooth_bins)
    events = []
    for start, stop in _components(v >= threshold for v in smoothed):
        x0 = max(target[0], lo + (hi-lo) * start / LENGTH)
        x1 = min(target[1], lo + (hi-lo) * stop / LENGTH)
        if x1 <= x0:
            continue
        type_scores = {name: sum(values[start:stop]) / (stop-start)
                       for name, values in types.items()}
        events.append({"x0": x0, "x1": x1,
                       "attention_score": max(smoothed[start:stop]),
                       "type_scores": type_scores,
                       "source_windows": [wid]})
    return (lo, hi), events, max(smoothed)


def _box_events(window, target):
    wid = str(window.get("id", ""))
    lo, hi = _interval(window.get("focus_interval"), "focus_interval")
    boxes = window.get("boxes")
    if not isinstance(boxes, list):
        raise ValueError("boxes must be a list")
    events = []
    for box in boxes:
        x0, x1 = _interval((box.get("x0"), box.get("x1")), "box")
        if x0 < lo - 1e-8 or x1 > hi + 1e-8:
            raise ValueError("box extends outside its focus interval")
        score = _number(box.get("attention_score", 1), "attention_score")
        if not 0 <= score <= 1:
            raise ValueError("attention_score must be in [0, 1]")
        raw_types = box.get("feature_type_scores") or {}
        if not raw_types and box.get("feature_types"):
            raw_types = {name: 1.0 for name in box["feature_types"]}
        type_scores = {str(name): _number(value, f"type score {name}")
                       for name, value in raw_types.items()}
        if any(value < 0 or value > 1 for value in type_scores.values()):
            raise ValueError("type scores must be in [0, 1]")
        x0, x1 = max(x0, target[0]), min(x1, target[1])
        if x1 > x0:
            events.append({"x0": x0, "x1": x1, "attention_score": score,
                           "type_scores": type_scores,
                           "source_windows": [wid]})
    return (lo, hi), events, max((e["attention_score"] for e in events), default=0.0)


def _union_length(intervals):
    length = 0.0
    end = -math.inf
    for lo, hi in sorted(intervals):
        length += max(0.0, hi - max(lo, end))
        end = max(end, hi)
    return length


def _merge_events(events, type_threshold):
    if not events:
        return []
    parent = list(range(len(events)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, a in enumerate(events):
        for j in range(i+1, len(events)):
            b = events[j]
            overlap = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
            if overlap <= 0:
                continue
            width_a, width_b = a["x1"]-a["x0"], b["x1"]-b["x0"]
            iou = overlap / (width_a + width_b - overlap)
            if iou >= .25 or overlap / min(width_a, width_b) >= .5:
                parent[find(j)] = find(i)
    groups = {}
    for i, event in enumerate(events):
        groups.setdefault(find(i), []).append(event)
    merged = []
    for group in groups.values():
        scores = {}
        for event in group:
            for name, value in event["type_scores"].items():
                scores[name] = max(scores.get(name, 0), value)
        merged.append({
            "x0": round(min(e["x0"] for e in group), 8),
            "x1": round(max(e["x1"] for e in group), 8),
            "attention_score": round(max(e["attention_score"] for e in group), 6),
            "feature_type_scores": {k: round(v, 6) for k, v in sorted(scores.items())},
            "feature_types": sorted(k for k, v in scores.items() if v >= type_threshold),
            "type_status": "named" if any(v >= type_threshold for v in scores.values()) else "uncertain",
            "source_windows": sorted({wid for e in group for wid in e["source_windows"]}),
        })
    return sorted(merged, key=lambda e: (e["x0"], e["x1"]))


def adapt_pair(before, after, windows, model, *, target_interval=None,
               attention_threshold=.5, type_threshold=.5, uncertainty_floor=.35,
               min_coverage=.98, smooth_bins=5):
    """Return versioned local-feature evidence for one compatible scan pair.

    A no-attention decision applies only to ``target_interval``. Callers must
    pass the full common range when they want a whole-pair decision. ``windows``
    can contain 156-position probability masks or frozen native-coordinate
    box fixtures; both produce the same output contract.
    """
    if not isinstance(model, Mapping) or not model.get("name") or not model.get("version"):
        raise ValueError("model must declare name and version")
    if not 0 <= uncertainty_floor < attention_threshold <= 1:
        raise ValueError("invalid attention thresholds")
    if not 0 < type_threshold <= 1 or not 0 < min_coverage <= 1:
        raise ValueError("invalid type threshold or coverage")
    if smooth_bins < 1 or smooth_bins % 2 != 1:
        raise ValueError("smooth_bins must be a positive odd integer")
    windows = list(windows)
    try:
        scope0, x0 = _validated_pattern(before)
        scope1, x1 = _validated_pattern(after)
    except ValueError as error:
        return _unavailable(before, after, model, f"invalid_pattern:{error}")
    if scope0 != scope1:
        return _unavailable(before, after, model, "incompatible_scope", scope=scope0)
    common = (max(x0[0], x1[0]), min(x0[-1], x1[-1]))
    if common[1] <= common[0]:
        return _unavailable(before, after, model, "no_common_measured_interval", scope=scope0)
    target = _interval(target_interval, "target_interval") if target_interval is not None else None
    if target is None:
        intervals = [_interval(w.get("focus_interval"), "focus_interval") for w in windows]
        if not intervals:
            raise ValueError("target_interval is required when windows is empty")
        target = (min(lo for lo, _ in intervals), max(hi for _, hi in intervals))
    if target[0] < common[0] - 1e-8 or target[1] > common[1] + 1e-8:
        return _unavailable(before, after, model, "target_outside_common_measured_interval", scope=scope0)
    intervals, events, maximum = [], [], 0.0
    reasons = []
    for window in windows:
        lo, hi = _interval(window.get("focus_interval"), "focus_interval")
        if lo < target[0] - 1e-8 or hi > target[1] + 1e-8:
            raise ValueError("focus interval extends outside target interval")
        if not _source_quality(x0, lo, hi) or not _source_quality(x1, lo, hi):
            reasons.append(f"unsupported_window:{window.get('id', '')}")
            continue
        if "attention" in window:
            interval, local_events, local_max = _mask_events(
                window, attention_threshold, type_threshold, smooth_bins, target)
        elif "boxes" in window:
            interval, local_events, local_max = _box_events(window, target)
        else:
            raise ValueError("window needs attention mask or frozen boxes")
        intervals.append(interval)
        events.extend(local_events)
        maximum = max(maximum, local_max)
    coverage = min(1.0, _union_length(intervals) / (target[1]-target[0]))
    merged = _merge_events(events, type_threshold)
    if coverage < min_coverage:
        reasons.append("incomplete_window_coverage")
    if not merged and maximum >= uncertainty_floor:
        reasons.append("borderline_attention_score")
    status = "uncertain" if reasons else "valid"
    attention = "yes" if merged else "uncertain" if reasons else "no"
    return {
        "schema": SCHEMA,
        "status": status,
        "attention": attention,
        "reasons": reasons,
        "before_id": str(before.get("id", "")),
        "after_id": str(after.get("id", "")),
        "scope": scope0,
        "coordinate": scope0["coordinate"],
        "target_interval": [target[0], target[1]],
        "coverage_fraction": round(coverage, 6),
        "model": {**dict(model), "attention_threshold": attention_threshold,
                  "type_threshold": type_threshold, "uncertainty_floor": uncertainty_floor,
                  "smooth_bins": smooth_bins},
        "events": merged,
    }
