"""Peak-preserving, mask-aware Q channels for a structural ranker.

These are machine-learning features, never replacement observations, weights
or likelihoods. Native refinement retains the original measurements.
"""
import numpy as np


def pattern_features(q, intensity, edges, *, background=None, valid=None, contrast=30.):
    """Return mean, peak, valid-coverage and Q-coordinate channels.

    Integrals are exact for the piecewise-linear input representation. A peak
    between output bin centers remains in its bin's maximum channel. Missing
    intervals are not interpolated across and have an explicit coverage mask.
    No fitted phases, scan number or future observation is used.
    """
    q = np.asarray(q, float); edges = np.asarray(edges, float)
    y = np.asarray(np.ma.getdata(intensity), float)
    if (q.ndim != 1 or len(q) < 2 or y.shape != q.shape
            or not np.isfinite(q).all() or np.any(np.diff(q) <= 0)
            or edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0)):
        raise ValueError('Finite increasing Q coordinates and bin edges with one intensity per point required')
    mask = ~np.ma.getmaskarray(intensity)
    if valid is not None:
        valid = np.asarray(valid)
        if valid.shape != q.shape or valid.dtype != bool:
            raise ValueError('One explicit boolean validity flag per measured point required')
        mask = mask & valid
    if not np.isfinite(y[mask]).all():
        raise ValueError('Unmasked nonfinite intensity cannot become a ranker feature')
    if background is not None:
        background = np.asarray(background, float)
        if background.shape != q.shape or not np.isfinite(background[mask]).all():
            raise ValueError('One finite supplied background value per valid pattern point required')
        y = y-background
    if type(contrast) not in (int, float) or not np.isfinite(contrast) or not 1 <= contrast <= 1000:
        raise ValueError('A finite feature contrast between one and one thousand is required')
    if np.count_nonzero(mask) < 2:
        raise ValueError('At least two valid pattern points are required')
    values = np.where(mask, y, 0.)
    nodes = np.unique(np.r_[q, edges]); intervals = np.diff(nodes)
    interpolated = np.interp(nodes, q, values)
    midpoint = nodes[:-1]+intervals*.5
    left = np.searchsorted(q, midpoint, side='right')-1
    active = (left >= 0) & (left < len(q)-1)
    indices = np.flatnonzero(active)
    active[indices] &= mask[left[indices]] & mask[left[indices]+1]
    area = np.r_[0., np.cumsum((interpolated[:-1]+interpolated[1:])*.5*intervals*active)]
    length = np.r_[0., np.cumsum(intervals*active)]
    locations = np.searchsorted(nodes, edges)
    bin_length = np.diff(length[locations]); width = np.diff(edges)
    mean = np.divide(np.diff(area[locations]), bin_length, out=np.zeros(len(width)), where=bin_length > 0)
    coverage = np.clip(bin_length/width, 0., 1.)
    endpoints = np.r_[active, False] | np.r_[False, active]
    bins = np.searchsorted(edges, nodes, side='right')-1
    bins[nodes == edges[-1]] = len(width)-1
    usable = endpoints & (bins >= 0) & (bins < len(width))
    peak = np.full(len(width), -np.inf)
    np.maximum.at(peak, bins[usable], interpolated[usable])
    peak[~np.isfinite(peak)] = 0.
    mean[coverage == 0] = 0.; peak[coverage == 0] = 0.
    # Peaks outside the model's Q range, or isolated points surrounded by
    # missing intervals, must not suppress every feature inside its range.
    scale = max(float(np.max(abs(interpolated[usable]), initial=0.)), np.finfo(float).tiny)
    transform = lambda v: np.arcsinh(contrast*(v/scale))/np.arcsinh(contrast)
    coordinate = ((edges[:-1]+edges[1:])*.5-edges[0])/(edges[-1]-edges[0])
    features = np.asarray([transform(mean), transform(peak), coverage, coordinate], np.float32)
    if not np.isfinite(features).all():
        raise ValueError('Structural pattern normalization is not numerically finite')
    return features


def with_parent_reference(observed, reference):
    """Pair a pattern with its parent, retaining the size of weak changes.

    Difference channels use a fixed contrast, never normalization by the
    difference's own maximum: a tiny change or noise cannot become a unit
    transition signal merely because it is the largest residual present.
    Both inputs must have been constructed on the same Q edges. This is a
    representation for ranking, not subtraction of measured observations.
    """
    observed, reference = np.asarray(observed), np.asarray(reference)
    if (observed.ndim != 2 or observed.shape[0] != 4 or reference.shape != observed.shape
            or observed.dtype != np.float32 or reference.dtype != np.float32
            or not np.isfinite(observed).all() or not np.isfinite(reference).all()
            or np.any(observed[2] < 0) or np.any(observed[2] > 1)
            or np.any(reference[2] < 0) or np.any(reference[2] > 1)
            or not np.array_equal(observed[3], reference[3])):
        raise ValueError('Finite pattern and parent features on the same Q grid are required')
    coverage = np.minimum(observed[2], reference[2])
    difference = np.arcsinh(30.*(observed[:2]-reference[:2]))/np.arcsinh(30.)
    result = np.concatenate([observed.copy(), reference[:2], difference]).astype(np.float32)
    result[2] = coverage
    result[np.array([0, 1, 4, 5, 6, 7])[:, None], np.flatnonzero(coverage == 0)] = 0.
    return result
