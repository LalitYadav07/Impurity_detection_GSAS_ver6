"""Full-pair, local-only 2 x 156 inference prototype.

The caller supplies a batch predictor returning probabilities. This module
never sends the full curve to that predictor and never assigns a transition
state. The default tile spans 151 valid native samples of the before curve,
matching the training-window construction across nonuniform TOF grids.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil, isfinite
from statistics import median
from time import perf_counter

import numpy as np

try:
    from .adapter import (LENGTH, _unavailable, _validated_pattern,
                          adapt_pair, pattern_from_observed)
except ImportError:  # Also support direct execution from this prototype directory.
    from adapter import (LENGTH, _unavailable, _validated_pattern,
                         adapt_pair, pattern_from_observed)


def shared_percentile_v1(before, after):
    """Match the original focus_156 pooled piecewise-percentile transform."""
    pooled = np.concatenate((before, after))
    knots = np.percentile(pooled, [0, 10, 40, 80, 95, 99.5, 100])
    target = np.array([0.0, .10, .28, .52, .76, .93, 1.0])
    if knots[-1] <= knots[0]:
        return np.zeros((2, LENGTH), dtype=np.float32)
    keep = np.r_[True, np.diff(knots) > max(float(np.ptp(knots)), 1.0) * 1e-12]
    knots, target = knots[keep], target[keep]
    return np.stack((np.interp(before, knots, target),
                     np.interp(after, knots, target))).astype(np.float32)


def _pattern(value):
    return value if isinstance(value, Mapping) else pattern_from_observed(value)


def _measured_values(pattern):
    x = np.asarray(pattern["x"], dtype=np.float64)
    y = np.asarray(pattern["intensity"], dtype=np.float64)
    valid = pattern.get("valid")
    mask = np.ones(len(x), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    return x[mask], y[mask]


def _tiles(target, width, overlap_fraction, max_windows):
    lo, hi = target
    span = hi - lo
    if width >= span:
        return [(lo, hi)]
    stride = width * (1 - overlap_fraction)
    count = ceil((span - width) / stride) + 1
    if count > max_windows:
        return None
    starts = [lo + i * stride for i in range(count - 1)]
    starts.append(hi - width)
    return [(start, start + width) for start in starts]


def _native_point_tiles(target, before_x, window_points, overlap_fraction,
                        max_windows):
    """Cover the common interval with locally sized, overlapping native tiles.

    The training pool selects consecutive aligned before-scan x values. Its
    normal 151-point focus is reproduced here before resampling to 156 model
    positions. Endpoint extension covers the exact common measured interval
    when the after scan starts or ends between before-scan samples.
    """
    lo, hi = target
    local = before_x[(before_x >= lo) & (before_x <= hi)]
    stride = max(1, round((window_points - 1) * (1 - overlap_fraction)))
    if len(local) <= window_points:
        return [(lo, hi)], stride
    final = len(local) - window_points
    starts = list(range(0, final + 1, stride))
    if starts[-1] != final:
        starts.append(final)
    if len(starts) > max_windows:
        return None, stride
    tiles = [(float(local[start]), float(local[start + window_points - 1]))
             for start in starts]
    tiles[0] = (lo, tiles[0][1])
    tiles[-1] = (tiles[-1][0], hi)
    return tiles, stride


def _quality(x, interval, min_native_points):
    lo, hi = interval
    left = int(np.searchsorted(x, lo, side="left"))
    right = int(np.searchsorted(x, hi, side="right"))
    local = x[left:right]
    if len(local) < min_native_points:
        return False
    spacings = np.diff(local)
    spacing = median(spacings)
    # The adapter's internal gap check covers the window interior. These edge
    # checks prevent interpolation across a masked gap just outside a tile.
    return (local[0] - lo <= 4 * spacing and
            hi - local[-1] <= 4 * spacing and
            float(np.max(spacings)) <= 4 * spacing)


def _batch_probabilities(response, batch_size):
    if not isinstance(response, Mapping) or "attention" not in response:
        raise ValueError("predict_batch must return attention probabilities")
    attention = np.asarray(response["attention"])
    if attention.shape != (batch_size, LENGTH):
        raise ValueError(f"attention output must have shape ({batch_size}, {LENGTH})")
    raw_types = response.get("type_probabilities", {})
    if not isinstance(raw_types, Mapping):
        raise ValueError("type_probabilities output must be a mapping")
    types = {}
    for name, value in raw_types.items():
        scores = np.asarray(value)
        if scores.shape != (batch_size, LENGTH):
            raise ValueError(f"type {name} output must have shape ({batch_size}, {LENGTH})")
        types[str(name)] = scores
    for name, values in [("attention", attention), *types.items()]:
        if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} output must contain finite probabilities")
        if np.any((values < 0) | (values > 1)):
            raise ValueError(f"{name} output probabilities must lie in [0, 1]")
    return attention, types


def run_pair(before, after, predict_batch, model, *, window_width=None,
             window_points=151,
             overlap_fraction=.5, batch_size=64, preprocess=shared_percentile_v1,
             min_native_points=32, max_windows=2048,
             attention_threshold=.5, type_threshold=.5, uncertainty_floor=.35,
             min_coverage=.98, smooth_bins=5):
    """Predict local features over a full compatible scan pair.

    ``predict_batch`` receives only a float32 array shaped ``(B, 2, 156)``.
    It returns ``{"attention": (B, 156), "type_probabilities": {tag:
    (B, 156)}}`` with probabilities, not logits. ``preprocess`` receives
    two arrays already interpolated onto one local native-x interval and must
    return ``(2, 156)``. Use the transform frozen with the checkpoint.

    By default each tile spans 151 consecutive valid before-scan x samples;
    ``overlap_fraction`` determines its stride in sample indices. A caller
    can instead set ``window_width`` in the scope's native x unit to request
    the earlier fixed-width policy. Every window must
    contain enough measured points in both scans and no gap above four times
    its local median spacing. Skipped windows are never passed to the model.
    """
    started = perf_counter()
    before, after = _pattern(before), _pattern(after)
    if not callable(predict_batch) or not callable(preprocess):
        raise TypeError("predict_batch and preprocess must be callables")
    if not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction must lie in [0, 1)")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(min_native_points) is not int or min_native_points < 8:
        raise ValueError("min_native_points must be an integer of at least eight")
    if type(max_windows) is not int or max_windows < 1:
        raise ValueError("max_windows must be a positive integer")
    if type(window_points) is not int or window_points < 8:
        raise ValueError("window_points must be an integer of at least eight")
    if window_width is not None:
        try:
            window_width = float(window_width)
        except (TypeError, ValueError) as error:
            raise ValueError("window_width must be positive and finite") from error
        if not isfinite(window_width) or window_width <= 0:
            raise ValueError("window_width must be positive and finite")
    if not isinstance(model, Mapping) or not model.get("name") or not model.get("version"):
        raise ValueError("model must declare name and version")

    stats = {"latency_ms": 0.0, "window_count": 0, "inferred_window_count": 0,
             "skipped_window_count": 0, "batch_count": 0, "window_width": None,
             "window_policy": ("fixed_native_width_v1" if window_width is not None
                               else "native_before_points_v1"),
             "window_points": None if window_width is not None else window_points,
             "window_stride_points": None,
             "overlap_fraction": overlap_fraction, "min_native_points": min_native_points,
             "source_points_before": 0, "source_points_after": 0,
             "skipped_windows": [], "inferred_windows": []}

    def finish(result):
        stats["latency_ms"] = round((perf_counter() - started) * 1000, 3)
        result["inference"] = stats
        return result

    try:
        scope0, valid_x0 = _validated_pattern(before)
        scope1, valid_x1 = _validated_pattern(after)
    except ValueError as error:
        return finish(_unavailable(before, after, model, f"invalid_pattern:{error}"))
    stats["source_points_before"] = len(valid_x0)
    stats["source_points_after"] = len(valid_x1)
    if scope0 != scope1:
        return finish(_unavailable(before, after, model, "incompatible_scope", scope=scope0))
    target = (max(valid_x0[0], valid_x1[0]), min(valid_x0[-1], valid_x1[-1]))
    if target[1] <= target[0]:
        return finish(_unavailable(before, after, model,
                                   "no_common_measured_interval", scope=scope0))

    native_x0, native_y0 = _measured_values(before)
    native_x1, native_y1 = _measured_values(after)
    if window_width is None:
        tiles, stride_points = _native_point_tiles(
            target, native_x0, window_points, overlap_fraction, max_windows)
        stats["window_stride_points"] = stride_points
    else:
        stats["window_width"] = float(window_width)
        tiles = _tiles(target, float(window_width), overlap_fraction, max_windows)
    if tiles is None:
        return finish(_unavailable(before, after, model,
                                   "window_limit_exceeded", scope=scope0))
    stats["window_count"] = len(tiles)

    inputs, pending = [], []
    for index, interval in enumerate(tiles):
        wid = f"tile-{index:04d}"
        if not _quality(native_x0, interval, min_native_points) or not _quality(
                native_x1, interval, min_native_points):
            stats["skipped_windows"].append({"id": wid,
                                              "focus_interval": list(interval),
                                              "reason": "insufficient_or_gapped_native_points"})
            continue
        grid = np.linspace(interval[0], interval[1], LENGTH)
        raw0 = np.interp(grid, native_x0, native_y0)
        raw1 = np.interp(grid, native_x1, native_y1)
        local = np.asarray(preprocess(raw0, raw1), dtype=np.float32)
        if local.shape != (2, LENGTH) or not np.all(np.isfinite(local)):
            raise ValueError("preprocess must return finite 2 x 156 values")
        inputs.append(local)
        pending.append({"id": wid, "focus_interval": list(interval)})
    stats["skipped_window_count"] = len(stats["skipped_windows"])
    stats["inferred_window_count"] = len(pending)
    stats["inferred_windows"] = pending

    if not pending:
        result = adapt_pair(before, after, [], model, target_interval=target,
                            attention_threshold=attention_threshold,
                            type_threshold=type_threshold,
                            uncertainty_floor=uncertainty_floor,
                            min_coverage=min_coverage, smooth_bins=smooth_bins)
        result["status"] = "unsupported"
        result["reasons"] = ["no_eligible_windows"]
        return finish(result)

    windows = []
    for start in range(0, len(inputs), batch_size):
        batch = np.stack(inputs[start:start + batch_size]).astype(np.float32)
        attention, types = _batch_probabilities(predict_batch(batch), len(batch))
        stats["batch_count"] += 1
        for offset, row in enumerate(pending[start:start + len(batch)]):
            windows.append({**row, "attention": attention[offset].tolist(),
                            "type_probabilities": {name: values[offset].tolist()
                                                   for name, values in types.items()}})

    result = adapt_pair(before, after, windows, model, target_interval=target,
                        attention_threshold=attention_threshold,
                        type_threshold=type_threshold,
                        uncertainty_floor=uncertainty_floor,
                        min_coverage=min_coverage, smooth_bins=smooth_bins)
    return finish(result)
