"""Peak-preserving, mask-aware Q channels for a structural ranker.

These are machine-learning features, never replacement observations, weights
or likelihoods. Native refinement retains the original measurements.
"""
import numpy as np


def _valid_intervals(q, mask, nodes):
    midpoint = nodes[:-1]+np.diff(nodes)*.5
    left = np.searchsorted(q, midpoint, side='right')-1
    active = (left >= 0) & (left < len(q)-1)
    indices = np.flatnonzero(active)
    active[indices] &= mask[left[indices]] & mask[left[indices]+1]
    return active


class _PatternProjection:
    """One immutable grid/mask plan; intensity arithmetic stays unchanged."""
    def __init__(self, q, edges, mask, *, nodes=None, interval_valid=None):
        self.q = q; self.edges = edges; self.mask = mask.copy()
        nodes = np.unique(np.r_[q, edges]) if nodes is None else nodes
        intervals = np.diff(nodes); active = _valid_intervals(q, mask, nodes)
        if interval_valid is not None:
            active &= interval_valid
        length = np.r_[0., np.cumsum(intervals*active)]
        locations = np.searchsorted(nodes, edges)
        bin_length = np.diff(length[locations]); width = np.diff(edges)
        endpoints = np.r_[active, False] | np.r_[False, active]
        bins = np.searchsorted(edges, nodes, side='right')-1
        bins[nodes == edges[-1]] = len(width)-1
        usable = endpoints & (bins >= 0) & (bins < len(width))
        self.nodes = nodes; self.intervals = intervals; self.active = active
        self.locations = locations; self.bin_length = bin_length
        self.coverage = np.clip(bin_length/width, 0., 1.)
        self.bins = bins; self.usable = usable
        self.coordinate = ((edges[:-1]+edges[1:])*.5-edges[0])/(edges[-1]-edges[0])

    def project(self, intensity, background, *, include_peak=True):
        y = np.asarray(np.ma.getdata(intensity), float); mask = self.mask
        if y.shape != self.q.shape:
            raise ValueError('One intensity per native Q point required')
        if not np.isfinite(y[mask]).all():
            raise ValueError('Unmasked nonfinite intensity cannot become a ranker feature')
        if background is not None:
            background = np.asarray(background, float)
            if background.shape != self.q.shape or not np.isfinite(background[mask]).all():
                raise ValueError('One finite supplied background value per valid pattern point required')
            y = y-background
        values = np.where(mask, y, 0.)
        interpolated = np.interp(self.nodes, self.q, values)
        area = np.r_[0., np.cumsum((interpolated[:-1]+interpolated[1:])*.5*self.intervals*self.active)]
        mean = np.divide(np.diff(area[self.locations]), self.bin_length,
            out=np.zeros(len(self.bin_length)), where=self.bin_length > 0)
        peak = None
        if include_peak:
            peak = np.full(len(self.bin_length), -np.inf)
            np.maximum.at(peak, self.bins[self.usable], interpolated[self.usable])
            peak[~np.isfinite(peak)] = 0.
            peak[self.coverage == 0] = 0.
        mean[self.coverage == 0] = 0.
        scale = max(float(np.max(abs(interpolated[self.usable]), initial=0.)), np.finfo(float).tiny)
        return mean, peak, self.coverage, self.coordinate, scale


def _grid(q, edges):
    q = np.asarray(q, float); edges = np.asarray(edges, float)
    if (q.ndim != 1 or len(q) < 2 or not np.isfinite(q).all() or np.any(np.diff(q) <= 0)
            or edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0)):
        raise ValueError('Finite increasing Q coordinates and bin edges with one intensity per point required')
    return q, edges


def _mask(q, intensity, valid):
    if np.shape(intensity) != q.shape:
        raise ValueError('One intensity per native Q point required')
    mask = ~np.ma.getmaskarray(intensity)
    if valid is not None:
        valid = np.asarray(valid)
        if valid.shape != q.shape or valid.dtype != bool:
            raise ValueError('One explicit boolean validity flag per measured point required')
        mask &= valid
    if np.count_nonzero(mask) < 2:
        raise ValueError('At least two valid pattern points are required')
    return mask


def _pattern_bins(q, intensity, edges, *, background=None, valid=None):
    """Piecewise-linear projection; masked gaps are never interpolated across."""
    q, edges = _grid(q, edges)
    return _PatternProjection(q, edges, _mask(q, intensity, valid)).project(intensity, background)


def pattern_features(q, intensity, edges, *, background=None, valid=None, contrast=30.):
    """Return legacy mean, peak, valid-coverage and Q-coordinate channels."""
    if type(contrast) not in (int, float) or not np.isfinite(contrast) or not 1 <= contrast <= 1000:
        raise ValueError('A finite feature contrast between one and one thousand is required')
    mean, peak, coverage, coordinate, scale = _pattern_bins(q, intensity, edges,
        background=background, valid=valid)
    transform = lambda v: np.arcsinh(contrast*(v/scale))/np.arcsinh(contrast)
    features = np.asarray([transform(mean), transform(peak), coverage, coordinate], np.float32)
    if not np.isfinite(features).all():
        raise ValueError('Structural pattern normalization is not numerically finite')
    return features


class ThreeCurveProjector:
    """Reuse Q-bin geometry for native pairs and their observation siblings.

    One exact parent projection is retained for observation siblings. Its key
    owns the converted intensity/background bytes and its mask plan; in-place
    changes cannot reuse stale values. Current intensity, shared normalization
    and signed residuals are recalculated on every call. The cache is bounded
    to one parent and one mask plan.
    """
    def __init__(self, q, edges):
        q, edges = _grid(q, edges)
        self.q = np.frombuffer(q.tobytes(), dtype=q.dtype)
        self.edges = np.frombuffer(edges.tobytes(), dtype=edges.dtype)
        self._projection = None
        self._reference_key = None
        self._reference_projection = None
        self._reference = None

    def features(self, current, reference, *, background=None, reference_background=None,
                 valid=None, contrast=30.):
        if type(contrast) not in (int, float) or not np.isfinite(contrast) or not 1 <= contrast <= 1000:
            raise ValueError('A finite feature contrast between one and one thousand is required')
        if np.shape(current) != self.q.shape or np.shape(reference) != self.q.shape:
            raise ValueError('Both curves must share their native Q coordinates')
        common = ~np.ma.getmaskarray(current) & ~np.ma.getmaskarray(reference)
        if valid is not None:
            valid = np.asarray(valid)
            if valid.shape != self.q.shape or valid.dtype != bool:
                raise ValueError('One explicit boolean validity flag per native point required')
            common &= valid
        if np.count_nonzero(common) < 2:
            raise ValueError('At least two valid pattern points are required')
        if self._projection is None or not np.array_equal(common, self._projection.mask):
            self._projection = _PatternProjection(self.q, self.edges, common)
        reference_values = np.asarray(np.ma.getdata(reference), float)
        reference_bg = None if reference_background is None else np.asarray(reference_background, float)
        if reference_bg is not None and reference_bg.shape != self.q.shape:
            raise ValueError('One finite supplied background value per valid pattern point required')
        key = (reference_values.tobytes(), None if reference_bg is None else reference_bg.tobytes())
        if self._reference_projection is not self._projection or self._reference_key != key:
            # The three-curve representation uses means, coverage and scale,
            # so computing the legacy peak-envelope channel is unnecessary.
            before = self._projection.project(reference_values, reference_bg, include_peak=False)
            self._reference = before
            self._reference_key = key
            self._reference_projection = self._projection
        before = self._reference
        after = self._projection.project(current, background, include_peak=False)
        coverage = np.minimum(before[2], after[2]); scale = max(before[4], after[4])
        transform = lambda v: np.arcsinh(contrast*(v/scale))/np.arcsinh(contrast)
        values = np.asarray([transform(before[0]), transform(after[0]), transform(after[0]-before[0])], np.float32)
        values[:, coverage == 0] = 0.
        if not np.isfinite(values).all():
            raise ValueError('Three-curve structural features are not finite')
        return dict(features=values, coverage=coverage.astype(np.float32),
            q_edges=self.edges.copy(), intensity_scale=scale,
            channels=['reference', 'current', 'current_minus_reference'],
            difference='background-subtracted native intensity difference before shared asinh compression')


def three_curve_features(q, current, reference, edges, *, background=None,
                         reference_background=None, valid=None, contrast=30.):
    """Exactly reference/current/signed-difference channels on a shared Q scale.

    Difference is computed before compression. Coverage and Q coordinates
    remain preprocessing metadata, not additional learned channels. Native
    refinement retains its original measurements and statistical weights.
    """
    return ThreeCurveProjector(q, edges).features(current, reference, background=background,
        reference_background=reference_background, valid=valid, contrast=contrast)


class IndependentCurveProjector:
    """Reference/current/difference on two independently calibrated Q grids.

    Integrate both piecewise-linear curves over the exact same valid native
    intervals. Neither curve is aligned to the other's peaks, extrapolated,
    independently normalized, smoothed or edited. A masked gap in either
    input excludes that interval from both integrals. This is a feature
    builder, not permission to reuse a model trained on calculated references
    with a measured reference.
    """
    def __init__(self, q, reference_q, edges):
        q, edges = _grid(q, edges)
        reference_q, _ = _grid(reference_q, edges)
        self.q = np.frombuffer(q.tobytes(), dtype=q.dtype)
        self.reference_q = np.frombuffer(reference_q.tobytes(), dtype=reference_q.dtype)
        self.edges = np.frombuffer(edges.tobytes(), dtype=edges.dtype)
        nodes = np.unique(np.r_[q, reference_q, edges])
        self.nodes = np.frombuffer(nodes.tobytes(), dtype=nodes.dtype)
        self._plans = None
        self._reference_key = self._reference = None

    def features(self, current, reference, *, background=None, reference_background=None,
                 valid=None, reference_valid=None, contrast=30.):
        if type(contrast) not in (int, float) or not np.isfinite(contrast) or not 1 <= contrast <= 1000:
            raise ValueError('A finite feature contrast between one and one thousand is required')
        mask = _mask(self.q, current, valid)
        reference_mask = _mask(self.reference_q, reference, reference_valid)
        if (self._plans is None or not np.array_equal(mask, self._plans[0].mask)
                or not np.array_equal(reference_mask, self._plans[1].mask)):
            common = (_valid_intervals(self.q, mask, self.nodes)
                      & _valid_intervals(self.reference_q, reference_mask, self.nodes))
            self._plans = (
                _PatternProjection(self.q, self.edges, mask, nodes=self.nodes, interval_valid=common),
                _PatternProjection(self.reference_q, self.edges, reference_mask,
                                   nodes=self.nodes, interval_valid=common))
            self._reference_key = self._reference = None
        after_plan, before_plan = self._plans
        reference_values = np.asarray(np.ma.getdata(reference), float)
        reference_bg = None if reference_background is None else np.asarray(reference_background, float)
        if reference_bg is not None and reference_bg.shape != self.reference_q.shape:
            raise ValueError('One finite supplied background value per valid pattern point required')
        key = (reference_values.tobytes(), None if reference_bg is None else reference_bg.tobytes())
        if key != self._reference_key:
            self._reference = before_plan.project(reference_values, reference_bg, include_peak=False)
            self._reference_key = key
        before = self._reference
        after = after_plan.project(current, background, include_peak=False)
        coverage = after[2]
        if not np.any(coverage > 0):
            raise ValueError('The reference and current patterns have no shared valid Q interval')
        scale = max(before[4], after[4])
        transform = lambda value: np.arcsinh(contrast*(value/scale))/np.arcsinh(contrast)
        features = np.asarray([transform(before[0]), transform(after[0]),
                               transform(after[0]-before[0])], np.float32)
        features[:, coverage == 0] = 0.
        if not np.isfinite(features).all():
            raise ValueError('Three-curve structural features are not finite')
        return dict(features=features, coverage=coverage.astype(np.float32),
            q_edges=self.edges.copy(), intensity_scale=scale,
            channels=['reference', 'current', 'current_minus_reference'],
            difference='background-subtracted native intensity difference before shared asinh compression')


def independent_curve_features(q, current, reference_q, reference, edges, *, background=None,
                               reference_background=None, valid=None, reference_valid=None, contrast=30.):
    """Three channels from separate native Q grids, with common valid support."""
    return IndependentCurveProjector(q, reference_q, edges).features(current, reference,
        background=background, reference_background=reference_background,
        valid=valid, reference_valid=reference_valid, contrast=contrast)


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
