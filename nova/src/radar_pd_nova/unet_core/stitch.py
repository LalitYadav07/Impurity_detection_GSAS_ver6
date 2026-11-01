"""Exploratory whole-curve fusion of overlapping fixed-model tile probabilities.

All coordinates remain native. This module only changes how already-computed
tile probabilities become boxes; it does not alter the U-Net or its inputs.
"""

from __future__ import annotations

import numpy as np


LENGTH = 156
EDGE_FLOOR = 0.25
TAPER_FRACTION = 0.20
SMOOTH_BINS = 5
ATTENTION_THRESHOLD = 0.80


def native_grid(before: dict, target: tuple[float, float]) -> np.ndarray:
    x = np.asarray(before["x"], dtype=np.float64)
    valid = np.asarray(before.get("valid", np.ones(len(x), dtype=bool)), dtype=bool)
    lo, hi = target
    inside = x[valid & (x >= lo) & (x <= hi)]
    result = np.unique(np.r_[lo, inside, hi])
    if len(result) < 2 or np.any(np.diff(result) <= 0):
        raise ValueError("No valid common native grid")
    return result


def stitch(windows: list[dict], x: np.ndarray, type_names: tuple[str, ...]) -> list[dict]:
    """Center-weighted mean on native x, followed by one global threshold.

    Missing tile support is not interpolated. The caller preserves the original
    runner's incomplete-coverage status and source-support checks.
    """
    if not windows or len(type_names) == 0:
        return []
    x = np.asarray(x, dtype=np.float64)
    sums = np.zeros((1 + len(type_names), len(x)), dtype=np.float64)
    weights = np.zeros(len(x), dtype=np.float64)
    for row in windows:
        lo, hi = map(float, row["focus_interval"])
        if hi <= lo:
            raise ValueError("Invalid tile interval")
        indices = np.flatnonzero((x >= lo) & (x <= hi))
        if not len(indices):
            continue
        local_x = x[indices]
        u = (local_x - lo) / (hi - lo)
        centrality = np.minimum(u, 1 - u) / TAPER_FRACTION
        weight = EDGE_FLOOR + (1 - EDGE_FLOOR) * np.clip(centrality, 0, 1)
        tile_x = np.linspace(lo, hi, LENGTH)
        maps = (row["attention"], *(row["type_probabilities"][name]
                                   for name in type_names))
        for channel, values in enumerate(maps):
            values = np.asarray(values, dtype=np.float64)
            if values.shape != (LENGTH,) or not np.all(np.isfinite(values)) or \
                    np.any((values < 0) | (values > 1)):
                raise ValueError("Invalid tile probability map")
            sums[channel, indices] += weight * np.interp(local_x, tile_x, values)
        weights[indices] += weight
    covered = weights > 0
    fused = np.zeros_like(sums)
    fused[:, covered] = sums[:, covered] / weights[covered]
    # Match the existing five-bin zero-padded smoothing policy on this grid.
    smoothed = np.convolve(fused[0], np.ones(SMOOTH_BINS) / SMOOTH_BINS,
                           mode="same")
    active = covered & (smoothed >= ATTENTION_THRESHOLD)
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    stops = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
    events = []
    for start, stop in zip(starts, stops):
        x0 = float(x[0] if start == 0 else (x[start - 1] + x[start]) / 2)
        x1 = float(x[-1] if stop == len(x) else (x[stop - 1] + x[stop]) / 2)
        if x1 <= x0:
            continue
        contributing = []
        for row in windows:
            lo, hi = map(float, row["focus_interval"])
            if min(x1, hi) <= max(x0, lo):
                continue
            tile_x = np.linspace(lo, hi, LENGTH)
            positions = x[start:stop]
            within = positions[(positions >= lo) & (positions <= hi)]
            if len(within) and np.max(np.interp(
                    within, tile_x, row["attention"])) >= ATTENTION_THRESHOLD:
                contributing.append(str(row["id"]))
        events.append({
            "x0": round(x0, 8), "x1": round(x1, 8),
            "attention_score": round(float(np.max(smoothed[start:stop])), 6),
            "feature_type_scores": {
                name: round(float(np.mean(fused[channel + 1, start:stop])), 6)
                for channel, name in enumerate(type_names)},
            "feature_types": [], "type_status": "uncertain",
            "source_windows": sorted(set(contributing)),
        })
    return events
