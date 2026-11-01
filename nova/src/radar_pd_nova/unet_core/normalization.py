"""Shared, local 2×156 intensity transforms for a controlled input ablation.

These functions transform the two curves together. They never independently
normalize before and after, cap a tall peak, or infer a physical change label.
The old percentile map is copied exactly from local_unet/build_dataset.py so
the baseline remains byte-for-byte comparable on the same interpolated inputs.
"""

from __future__ import annotations

import numpy as np

LENGTH = 156
MODES = ("old_percentile", "robust_linear", "smooth_power")
SMOOTH_GAMMA = 0.70


def _pair(before, after):
    a = np.asarray(before, dtype=np.float64)
    b = np.asarray(after, dtype=np.float64)
    if a.ndim != 1 or b.shape != a.shape or len(a) < 2:
        raise ValueError("Expected two one-dimensional curves of the same length")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("Curves must contain only finite values")
    return a, b


def old_percentile(before, after):
    """The current local U-Net input map, retained as the A/B/C control."""
    a, b = _pair(before, after)
    pooled = np.concatenate((a, b))
    knots = np.percentile(pooled, [0, 10, 40, 80, 95, 99.5, 100])
    target = np.array([0.0, .10, .28, .52, .76, .93, 1.0])
    if knots[-1] <= knots[0]:
        return np.zeros((2, len(a)), dtype=np.float32)
    keep = np.r_[True, np.diff(knots) > max(float(np.ptp(knots)), 1.0) * 1e-12]
    knots, target = knots[keep], target[keep]
    return np.stack((np.interp(a, knots, target),
                     np.interp(b, knots, target))).astype(np.float32)


def shared_affine_parameters(before, after):
    """A pooled baseline and scale with a range floor for isolated tall bins."""
    a, b = _pair(before, after)
    pooled = np.concatenate((a, b))
    q01, q99 = np.percentile(pooled, [1, 99])
    full_range = float(np.ptp(pooled))
    floor = max(.05 * full_range, np.finfo(np.float64).eps *
                max(1.0, float(np.max(np.abs(pooled)))))
    scale = max(float(q99 - q01), floor)
    return float(q01), float(scale)


def robust_linear(before, after):
    """Shared affine map. Peak shape and all raw intensity ratios stay linear."""
    a, b = _pair(before, after)
    offset, scale = shared_affine_parameters(a, b)
    return np.stack(((a - offset) / scale,
                     (b - offset) / scale)).astype(np.float32)


def smooth_power(before, after, gamma=SMOOTH_GAMMA):
    """Shared C1 monotone compression, unbounded and without a flat ceiling.

    At normalized intensity z>=0, f(z)=((1+z)^gamma-1)/(2^gamma-1).
    Negative z uses the tangent at zero. Thus f(0)=0, f(1)=1 and the first
    derivative is continuous. The current gamma is a candidate to test, not a
    calibrated choice.
    """
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    z = robust_linear(before, after).astype(np.float64)
    denominator = np.expm1(gamma * np.log(2.0))
    result = np.empty_like(z)
    nonnegative = z >= 0
    result[nonnegative] = np.expm1(gamma * np.log1p(z[nonnegative])) / denominator
    result[~nonnegative] = (gamma / denominator) * z[~nonnegative]
    return result.astype(np.float32)


def transform_pair(before, after, mode):
    if mode == "old_percentile":
        return old_percentile(before, after)
    if mode == "robust_linear":
        return robust_linear(before, after)
    if mode == "smooth_power":
        return smooth_power(before, after)
    raise ValueError(f"Unknown transform {mode!r}; expected one of {MODES}")
