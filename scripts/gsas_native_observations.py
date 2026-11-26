"""Install already-native observations without reinterpreting centers as edges.

The generic GSAS-II TOF text importer expects bin edges and replaces N supplied
coordinates by N-1 midpoints. Internal residuals already use native centers.
Their text files remain diagnostics; these arrays are the refinement contract.
"""
import numpy as np


def restore_native_observations(histogram, x, y, *, weights=None, shift_positive=False):
    """Replace only observations/derived arrays and reset their full native range.

    Preserve supplied center coordinates, y values, weights and masks. Unit
    weights are the existing residual-screening default. The optional constant
    y shift retains the previous positive-residual screening convention; it
    does not alter the original measured histogram or its published data.
    """
    original = [np.ma.asarray(value) for value in (x, y)]
    if (any(value.ndim != 1 for value in original) or
            original[0].shape != original[1].shape or original[0].size < 2):
        raise ValueError('Native observations require matching one-dimensional arrays')
    n = original[0].size
    original.append(np.ma.asarray(np.ones(n) if weights is None else weights))
    if original[2].ndim != 1 or original[2].shape != original[0].shape:
        raise ValueError('Native observation weights must align with their coordinates')
    values = [np.array(np.ma.getdata(value), dtype=float, copy=True) for value in original]
    masks = [np.ma.getmaskarray(value).copy() for value in original]
    centers, observed, weight = values
    if not np.all(np.isfinite(centers)) or np.any(np.diff(centers) <= 0):
        raise ValueError('Native center coordinates must be finite and strictly increasing')
    if np.any(np.isfinite(weight) & (weight < 0)):
        raise ValueError('Native observation weights cannot be negative')
    inactive = masks[0] | masks[1] | masks[2] | ~np.isfinite(weight) | (weight == 0)
    if np.any(~np.isfinite(observed) & ~inactive):
        raise ValueError('Native positive-weight observations must be finite')
    # Any masked source channel must stay outside the native fit. Propagating
    # the union to weight masks retains the original values beneath each mask.
    weight_mask = inactive | ~np.isfinite(observed)
    if shift_positive:
        active = ~weight_mask
        minimum = float(np.min(observed[active])) if active.any() else 0.
        if minimum < 0:
            observed[active] -= minimum - 1.
    arrays = []
    for value, mask in zip(values, (masks[0], masks[1], weight_mask)):
        arrays.append(np.ma.array(value, mask=mask) if mask.any() else value)
    data = histogram.data
    if not isinstance(data.get('data'), list) or len(data['data']) < 2:
        raise ValueError('Native histogram observation storage is unavailable')
    # These are calculated/background/difference caches, never observations.
    arrays.extend(np.zeros(n) for _ in range(3))
    data['data'][1] = arrays
    domain = [float(centers[0]), float(centers[-1])]
    # GSAS's refinement setup converts masked arrays to plain arrays. Encode
    # unavailable native samples as exact inclusive native exclusions too, so
    # a subsequent refinement/reload cannot silently reactivate them. These
    # ranges contain only existing invalid samples, never neighboring bins.
    bad = np.flatnonzero(weight_mask)
    runs = np.split(bad, np.flatnonzero(np.diff(bad) > 1)+1) if bad.size else []
    data['Limits'] = [domain.copy(), domain.copy(),
                      *[[float(centers[run[0]]), float(centers[run[-1]])] for run in runs]]
    return histogram
