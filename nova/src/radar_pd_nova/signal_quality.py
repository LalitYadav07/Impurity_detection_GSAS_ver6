"""Conservative signal-resolvability evidence using the supplied uncertainties."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def supplied_error_signal_quality(y: Any, sigma: Any, valid: Any = None) -> dict[str, Any]:
    """Flag a signal whose total squared SNR is below one, before any background fit.

    The fixed zero-model bound can establish that the entire measured signal
    is unresolved under its stated errors. Failing that bound says nothing
    about whether the signal contains Bragg peaks or identifies any phase.
    Inputs, errors and masks are never adjusted by this diagnostic.
    """
    intensity = np.asarray(y, dtype=float)
    uncertainty = np.asarray(sigma, dtype=float)
    if intensity.ndim != 1 or uncertainty.shape != intensity.shape:
        raise ValueError("Signal intensities and uncertainties must be equal-length one-dimensional arrays")
    mask = np.ones(intensity.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if mask.shape != intensity.shape:
        raise ValueError("Signal validity mask must match the intensity array")
    finite = np.isfinite(intensity) & np.isfinite(uncertainty) & (uncertainty > 0)
    use = mask & finite
    count = int(np.count_nonzero(use))
    result: dict[str, Any] = {
        "method": "supplied-error-zero-model-bound/v1",
        "point_count": count,
        "masked_point_count": int(np.count_nonzero(~mask)),
        "invalid_point_count": int(np.count_nonzero(mask & ~finite)),
        "total_norm_squared": None,
        "max_abs_snr": None,
        "norm_overflow": False,
        "criterion": "sum((y/sigma)^2) < 1",
        "unresolved": None,
        "needs_review": True,
        "status": "unavailable",
        "message": (
            "No finite valid intensity/positive-uncertainty pairs are available. "
            "Signal resolvability cannot be assessed under the supplied errors; "
            "no phase absence or identity is established."
        ),
    }
    if not count:
        return result

    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        ratios = np.abs(intensity[use] / uncertainty[use])
    maximum = float(np.max(ratios))
    try:
        total = math.fsum(float(value) * float(value) for value in ratios)
    except OverflowError:
        total = math.inf
    overflow = not math.isfinite(total) or not math.isfinite(maximum)
    unresolved = total < 1.0
    result.update(
        total_norm_squared=total if math.isfinite(total) else None,
        max_abs_snr=maximum if math.isfinite(maximum) else None,
        norm_overflow=overflow,
        unresolved=unresolved,
        needs_review=unresolved,
        status="unresolved" if unresolved else "not_flagged",
        message=(
            "The entire measured signal in the valid unmasked support is unresolved under the supplied errors "
            "(sum((y/sigma)^2) < 1); small Rwp does not establish phase identification. "
            "This does not establish phase absence."
            if unresolved else
            "The conservative unresolved-signal bound is not met under the supplied errors. "
            "This does not establish phase identity, Bragg-signal significance, or valid uncertainty estimates."
        ),
    )
    return result
