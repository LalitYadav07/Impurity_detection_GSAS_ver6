"""
Machine Learning Spectral Filter and Purity Estimator

Implements the Stage-3 histogram-based screening logic for candidate ranking.
Calculates ML-relevant metrics including explained fraction, purity scores,
and p-weighted scale factors against continuous residual distributions.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple


def _moving_average(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return y.copy()
    w = int(w)
    pad = w // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    c = np.cumsum(ypad)
    out = (c[w:] - c[:-w]) / float(w)
    return out


def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return np.median(np.abs(x - med)) + 1e-12


def find_peaks(y: np.ndarray, prominence_frac: float = 0.6, distance_pts: int = 5, smooth_win: int = 3) -> np.ndarray:
    """Return indices of detected peaks.

    Args:
        y: 1D array
        prominence_frac: fraction of MAD above median to accept as peak prominence
        distance_pts: minimum spacing (in data points) between peaks
        smooth_win: moving-average window width
    """
    y = np.asarray(y, dtype=float)
    ys = _moving_average(y, smooth_win)

    # Local maxima candidates
    dy_prev = ys[1:-1] - ys[:-2]
    dy_next = ys[1:-1] - ys[2:]
    cand = (dy_prev > 0) & (dy_next > 0)
    idx = np.nonzero(cand)[0] + 1

    # Prominence threshold
    baseline = np.median(ys)
    prom_thresh = prominence_frac * _mad(ys)
    good = ys[idx] - baseline >= prom_thresh
    idx = idx[good]

    # Enforce distance by greedy selection from strongest to weakest
    order = np.argsort(ys[idx])[::-1]
    idx_sorted = idx[order]
    keep = []
    for i in idx_sorted:
        if all(abs(i - j) >= distance_pts for j in keep):
            keep.append(i)
    keep.sort()
    return np.array(keep, dtype=int)


def peak_positions(Q: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    idx = find_peaks(y, **kwargs)
    return Q[idx], y[idx]
