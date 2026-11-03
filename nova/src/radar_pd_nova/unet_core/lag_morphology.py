"""Experimental local curve alignment signature in a native event interval."""

from __future__ import annotations

import math
import numpy as np


def _smooth(values: np.ndarray) -> np.ndarray:
    if len(values) < 5:
        return values
    return np.convolve(np.pad(values, (1, 1), mode="edge"),
                       np.array([.25, .5, .25]), mode="valid")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    vectors = []
    for values in (a, b):
        values = np.asarray(values, dtype=float)
        if not len(values) or not np.isfinite(values).all():
            return 0.
        scale = float(np.max(np.abs(values)))
        if scale == 0:
            return 0.
        # Normalize before centering or squaring, so intensity units cannot
        # underflow/overflow the norm or determine whether a curve is flat.
        centered = values / scale
        centered -= np.mean(centered)
        norm = float(np.linalg.norm(centered))
        if norm <= np.finfo(float).eps * np.sqrt(len(centered)):
            return 0.
        vectors.append(centered / norm)
    return float(np.clip(np.dot(*vectors), -1., 1.))


def lag_signature(event: dict, matched: dict) -> dict:
    """Measure how much a small x shift improves normalized curve similarity.

    We compare the two native curves inside the event with 10% box padding.
    Linear edge baselines and a three-bin smooth reduce offset/noise effects.
    The result is diagnostic and has no calibrated physical interpretation.
    """
    lo, hi = float(event["x0"]), float(event["x1"])
    if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
        raise ValueError("Invalid event interval")
    x = np.asarray(matched["x"], dtype=float)
    good = np.asarray(matched["good"], dtype=bool)
    a = np.asarray(matched["before_y"], dtype=float)
    b = np.asarray(matched["after_y"], dtype=float)
    if not (x.ndim == good.ndim == a.ndim == b.ndim == 1 and
            len(x) == len(good) == len(a) == len(b)):
        raise ValueError("Native matched arrays differ")
    pad = .1*(hi-lo)
    selected = good & np.isfinite(a) & np.isfinite(b) & (x >= lo-pad) & (x <= hi+pad)
    xx, aa, bb = x[selected], a[selected], b[selected]
    if len(xx) < 16 or np.any(np.diff(xx) <= 0):
        return {"status": "insufficient", "points": len(xx)}
    n = min(256, len(xx))
    grid = np.linspace(float(xx[0]), float(xx[-1]), n)
    aa = np.interp(grid, xx, aa)
    bb = np.interp(grid, xx, bb)
    edge = max(2, int(.1*n))
    for curve in (aa, bb):
        left = float(np.median(curve[:edge]))
        right = float(np.median(curve[-edge:]))
        curve -= np.linspace(left, right, n)
    aa, bb = _smooth(aa), _smooth(bb)
    max_lag = min(20, max(1, n//8))
    values = []
    for lag in range(-max_lag, max_lag+1):
        if lag < 0:
            left, right = aa[-lag:], bb[:n+lag]
        elif lag > 0:
            left, right = aa[:n-lag], bb[lag:]
        else:
            left, right = aa, bb
        values.append((lag, _corr(left, right)))
    best_lag, best_corr = max(values, key=lambda item: (item[1], -abs(item[0])))
    zero_corr = dict(values)[0]
    delta_x = (grid[-1]-grid[0])/(n-1)
    return {
        "status": "valid",
        "points": len(xx),
        "best_lag_bins": best_lag,
        "best_lag_native_x": best_lag*delta_x,
        "best_corr": best_corr,
        "zero_corr": zero_corr,
        "corr_gain": best_corr-zero_corr,
        "max_lag_bins": max_lag,
    }
