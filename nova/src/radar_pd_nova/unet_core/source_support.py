"""Measured native support for a pair of curves, in the source coordinate.

This module is deliberately independent of model predictions and reference
labels. It closes a gap where overlapping inference tiles had been mistaken
for measured source support.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _source_gaps(curve: dict, target: tuple[float, float]) -> list[tuple[float, float]]:
    x = np.asarray(curve["x"])
    valid = np.asarray(curve["valid"])
    if x.ndim != 1 or valid.shape != x.shape or valid.dtype != bool:
        raise ValueError("Source x and valid must be one-dimensional matching arrays")
    if len(x) < 2 or not np.isfinite(x).all() or not np.all(np.diff(x) > 0):
        raise ValueError("Source x must be finite and strictly increasing")
    lo, hi = target
    if not np.isfinite((lo, hi)).all() or lo >= hi:
        raise ValueError("Invalid target interval")
    ids = np.flatnonzero(valid & (x >= lo) & (x <= hi))
    if len(ids) < 2:
        return [(lo, hi)]
    width = curve.get("bin_width")
    if width is None and curve.get("native_x") is not None:
        edges = np.asarray(curve["native_x"], dtype=float)
        if edges.shape == (len(x)+1,):
            width = np.diff(edges)
    if width is not None:
        width = np.asarray(width, dtype=float)
        if width.shape != x.shape or not np.isfinite(width).all() or np.any(width <= 0):
            raise ValueError("Invalid native bin widths")

    def local_step(a: int, b: int) -> float:
        if width is not None:
            return float((width[a]+width[b])/2)
        neighbors = []
        if a > 0:
            neighbors.append(float(x[a]-x[a-1]))
        if b+1 < len(x):
            neighbors.append(float(x[b+1]-x[b]))
        return max(neighbors) if neighbors else float(x[b]-x[a])

    # TOF widths can grow smoothly by an order of magnitude. Only a missing
    # valid native index or a jump relative to its immediate local width is a
    # source gap. A global median step falsely removes valid high-TOF tails.
    gaps = []
    if x[ids[0]] > lo:
        gaps.append((lo, float(x[ids[0]])))
    for a, b in zip(ids[:-1], ids[1:]):
        if b != a + 1 or x[b] - x[a] > 4*local_step(int(a), int(b)):
            gaps.append((float(x[a]), float(x[b])))
    if x[ids[-1]] < hi:
        gaps.append((float(x[ids[-1]]), hi))
    return gaps


def measured_support(before: dict, after: dict,
                     target: tuple[float, float]) -> dict:
    """Intersect valid native spans; never bridge missing bins or x jumps."""
    lo, hi = target
    gaps = sorted(_source_gaps(before, target) + _source_gaps(after, target))
    merged: list[list[float]] = []
    for a, b in gaps:
        a, b = max(lo, a), min(hi, b)
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    support = []
    cursor = lo
    for a, b in merged:
        if a > cursor:
            support.append([cursor, a])
        cursor = max(cursor, b)
    if cursor < hi:
        support.append([cursor, hi])
    covered = sum(b-a for a, b in support)
    return {"supported_intervals": support, "source_gaps": merged,
            "coverage_fraction": min(1.0, covered/(hi-lo))}


def intersect_intervals(first: list, second: list) -> list[list[float]]:
    """Intersect sorted interval unions without inventing support."""
    a = sorted((float(lo), float(hi)) for lo, hi in first)
    b = sorted((float(lo), float(hi)) for lo, hi in second)
    result: list[list[float]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            if result and lo <= result[-1][1]:
                result[-1][1] = max(hi, result[-1][1])
            else:
                result.append([lo, hi])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return result


def load_curves(selection: dict, sources_root: Path, source_path,
                sha256) -> tuple[dict, dict]:
    curves = []
    for side in ("before", "after"):
        path = source_path(sources_root, selection["pair_id"],
                           selection[f"{side}_file"])
        if sha256(path) != selection[f"{side}_file_sha256"]:
            raise ValueError(f"Source SHA-256 mismatch for {selection['pair_id']} {side}")
        with np.load(path) as data:
            curve = {"x": np.asarray(data["x"]),
                     "valid": np.asarray(data["valid"])}
            for key in ("y", "sigma"):
                if key in data:
                    curve[key] = np.asarray(data[key])
            if "bin_width" in data:
                curve["bin_width"] = np.asarray(data["bin_width"])
            if "native_x" in data:
                curve["native_x"] = np.asarray(data["native_x"])
            curves.append(curve)
    return tuple(curves)
