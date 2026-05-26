#!/usr/bin/env python3
"""
Automatic low-envelope background-point selection for GSAS-II.

This module estimates a smooth lower envelope from the observed pattern and
converts it into fixed background points that GSAS-II can fit its own
background function against. The observed intensities and uncertainties remain
unchanged; only the background support points are generated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import math
import numpy as np


@dataclass(frozen=True)
class AutoBackgroundParameters:
    method: str = "snip_low_envelope"
    snip_iterations: Optional[int] = None
    snip_window_fraction: float = 0.025
    envelope_quantile: float = 0.10
    envelope_window_fraction: float = 0.035
    smooth_window_fraction: float = 0.0042
    point_spacing_fraction: float = 0.0053
    endpoint_count: int = 2
    max_points: int = 180
    min_points: int = 18


def coerce_auto_background_params(raw: Optional[dict[str, Any]] = None) -> AutoBackgroundParameters:
    """Build validated auto-background parameters from an optional config dict."""
    raw = dict(raw or {})
    defaults = AutoBackgroundParameters()

    def _maybe_int(key: str, default: Optional[int]) -> Optional[int]:
        value = raw.get(key, default)
        if value in (None, "", "auto"):
            return None
        return int(value)

    def _maybe_float(key: str, default: float) -> float:
        return float(raw.get(key, default))

    return AutoBackgroundParameters(
        method=str(raw.get("method", defaults.method)),
        snip_iterations=_maybe_int("snip_iterations", defaults.snip_iterations),
        snip_window_fraction=_maybe_float("snip_window_fraction", defaults.snip_window_fraction),
        envelope_quantile=_maybe_float("envelope_quantile", defaults.envelope_quantile),
        envelope_window_fraction=_maybe_float("envelope_window_fraction", defaults.envelope_window_fraction),
        smooth_window_fraction=_maybe_float("smooth_window_fraction", defaults.smooth_window_fraction),
        point_spacing_fraction=_maybe_float("point_spacing_fraction", defaults.point_spacing_fraction),
        endpoint_count=int(raw.get("endpoint_count", defaults.endpoint_count)),
        max_points=int(raw.get("max_points", defaults.max_points)),
        min_points=int(raw.get("min_points", defaults.min_points)),
    )


def _odd_window(n: int, fraction: float, *, minimum: int = 5, maximum: Optional[int] = None) -> int:
    width = int(round(max(minimum, n * float(fraction))))
    if maximum is not None:
        width = min(width, maximum)
    width = max(3, width)
    if width % 2 == 0:
        width += 1
    return min(width, n if n % 2 == 1 else max(3, n - 1))


def _moving_average(y: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or y.size < 3:
        return y.astype(float, copy=True)
    width = min(width, y.size if y.size % 2 == 1 else y.size - 1)
    if width <= 1:
        return y.astype(float, copy=True)
    pad = width // 2
    padded = np.pad(y, pad, mode="edge")
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(padded, kernel, mode="valid")


def _running_quantile(y: np.ndarray, width: int, quantile: float) -> np.ndarray:
    if y.size == 0:
        return y.astype(float, copy=True)
    width = max(3, min(width, y.size if y.size % 2 == 1 else max(3, y.size - 1)))
    if width % 2 == 0:
        width -= 1
    pad = width // 2
    padded = np.pad(y, pad, mode="edge")
    out = np.empty_like(y, dtype=float)
    q = float(np.clip(quantile, 0.0, 1.0))
    for i in range(y.size):
        out[i] = float(np.nanquantile(padded[i : i + width], q))
    return out


def _running_positive_quantile(y: np.ndarray, width: int, quantile: float) -> np.ndarray:
    """Return a running quantile using only strictly positive samples when possible."""
    if y.size == 0:
        return y.astype(float, copy=True)
    width = max(3, min(width, y.size if y.size % 2 == 1 else max(3, y.size - 1)))
    if width % 2 == 0:
        width -= 1
    pad = width // 2
    padded = np.pad(y, pad, mode="edge")
    out = np.empty_like(y, dtype=float)
    q = float(np.clip(quantile, 0.0, 1.0))
    for i in range(y.size):
        window = padded[i : i + width]
        positive = window[window > 0.0]
        if positive.size:
            out[i] = float(np.nanquantile(positive, q))
        else:
            out[i] = 0.0
    return out


def _running_indicator_fraction(mask: np.ndarray, width: int) -> np.ndarray:
    """Return the local fraction of True values in a boolean mask."""
    if mask.size == 0:
        return np.asarray(mask, dtype=float)
    width = max(3, min(width, mask.size if mask.size % 2 == 1 else max(3, mask.size - 1)))
    if width % 2 == 0:
        width -= 1
    pad = width // 2
    padded = np.pad(mask.astype(float), pad, mode="edge")
    out = np.empty(mask.size, dtype=float)
    for i in range(mask.size):
        out[i] = float(np.mean(padded[i : i + width]))
    return out


def _snip_background(y: np.ndarray, iterations: int) -> np.ndarray:
    """SNIP-style peak clipping in a stabilized asinh intensity space."""
    values = np.asarray(y, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=float)
    floor = float(np.nanpercentile(values[finite], 0.5))
    shifted = np.maximum(values - floor, 0.0)
    scale = float(np.nanpercentile(shifted[finite], 65.0))
    if not math.isfinite(scale) or scale <= 0.0:
        scale = max(1.0, float(np.nanmax(shifted[finite])))
    clipped = np.arcsinh(shifted / scale)
    n = clipped.size
    max_iter = max(1, min(int(iterations), max(1, (n - 1) // 2)))
    for k in range(1, max_iter + 1):
        if 2 * k >= n:
            break
        mid = slice(k, n - k)
        clipped[mid] = np.minimum(clipped[mid], 0.5 * (clipped[:-2 * k] + clipped[2 * k :]))
    return np.sinh(clipped) * scale + floor


def _sanitize_xys(
    x: Iterable[float],
    y: Iterable[float],
    sigma: Optional[Iterable[float]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx = np.asarray(list(x), dtype=float)
    yy = np.asarray(list(y), dtype=float)
    if sigma is None:
        ss = np.sqrt(np.maximum(np.abs(yy), 1.0))
    else:
        ss = np.asarray(list(sigma), dtype=float)
    if xx.size != yy.size or xx.size != ss.size:
        raise ValueError("x, y, and sigma arrays must have the same length")
    mask = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(ss)
    xx, yy, ss = xx[mask], yy[mask], ss[mask]
    if xx.size < 10:
        raise ValueError("At least 10 finite powder points are required")
    order = np.argsort(xx)
    return xx[order], yy[order], np.maximum(ss[order], 1.0e-12)


def select_fixed_points(
    x: np.ndarray,
    y: np.ndarray,
    background: np.ndarray,
    *,
    params: Optional[AutoBackgroundParameters] = None,
) -> np.ndarray:
    params = params or AutoBackgroundParameters()
    n = int(x.size)
    target = max(params.min_points, min(params.max_points, max(params.min_points, int(round(1.0 / max(params.point_spacing_fraction, 1.0e-6))))))
    target = min(target, n)
    edges = np.linspace(0, n, target + 1, dtype=int)
    chosen: list[int] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if hi <= lo:
            continue
        segment = slice(lo, hi)
        y_seg = y[segment]
        zero_fraction = float(np.mean(y_seg <= 0.0))
        if zero_fraction > 0.45:
            idx = lo + int((hi - lo - 1) // 2)
        else:
            residual = np.abs(y_seg - background[segment])
            idx = lo + int(np.nanargmin(residual))
        chosen.append(idx)
    chosen.extend(range(min(params.endpoint_count, n)))
    chosen.extend(range(max(0, n - params.endpoint_count), n))
    chosen = sorted(set(max(0, min(n - 1, idx)) for idx in chosen))
    return np.column_stack([x[chosen], background[chosen]])


def estimate_background(
    x: Iterable[float],
    y: Iterable[float],
    sigma: Optional[Iterable[float]] = None,
    *,
    params: Optional[AutoBackgroundParameters] = None,
) -> tuple[np.ndarray, np.ndarray, AutoBackgroundParameters]:
    """Return (background_curve, fixed_points, resolved_params)."""
    params = params or AutoBackgroundParameters()
    xx, yy, _ss = _sanitize_xys(x, y, sigma)
    n = xx.size

    snip_iterations = params.snip_iterations
    if snip_iterations is None:
        snip_iterations = max(8, min(120, int(round(n * params.snip_window_fraction))))

    snip = _snip_background(yy, snip_iterations)
    envelope_width = _odd_window(n, params.envelope_window_fraction, minimum=9)
    low_envelope = _running_quantile(yy, envelope_width, params.envelope_quantile)
    smooth_width = _odd_window(n, params.smooth_window_fraction, minimum=5)
    background = _moving_average(np.minimum(snip, low_envelope), smooth_width)

    guard = _running_quantile(yy, envelope_width, min(0.25, params.envelope_quantile + 0.12))
    background = np.minimum(background, guard)

    # TOF tails can contain many exact zeros from sparse detector coverage.
    # If we let the lower envelope collapse to those zeros, the chosen fixed
    # points become degenerate and GSAS fits a poor oscillatory tail.
    span = float(xx[-1] - xx[0]) if n > 1 else 0.0
    sparse_width = _odd_window(
        n,
        max(params.envelope_window_fraction, 0.08),
        minimum=31,
        maximum=max(31, n // 2),
    )
    positive_floor = _running_positive_quantile(
        yy,
        sparse_width,
        min(0.12, params.envelope_quantile + 0.02),
    )
    positive_floor = _moving_average(
        positive_floor,
        _odd_window(n, max(params.smooth_window_fraction, 0.015), minimum=9),
    )
    nonpositive_fraction = _running_indicator_fraction(yy <= 0.0, sparse_width)
    sparse_weight = np.clip((nonpositive_fraction - 0.30) / 0.35, 0.0, 1.0)
    tail_weight = np.clip(
        (xx - (float(xx[0]) + 0.65 * span)) / max(1.0, 0.35 * span),
        0.0,
        1.0,
    )
    background = np.maximum(background, 0.85 * positive_floor * sparse_weight * tail_weight)
    background = np.maximum(background, min(float(np.nanmin(yy)), 0.0))
    points = select_fixed_points(xx, yy, background, params=params)

    resolved = AutoBackgroundParameters(
        method=params.method,
        snip_iterations=int(snip_iterations),
        snip_window_fraction=params.snip_window_fraction,
        envelope_quantile=params.envelope_quantile,
        envelope_window_fraction=params.envelope_window_fraction,
        smooth_window_fraction=params.smooth_window_fraction,
        point_spacing_fraction=params.point_spacing_fraction,
        endpoint_count=params.endpoint_count,
        max_points=params.max_points,
        min_points=params.min_points,
    )
    return background, points, resolved
