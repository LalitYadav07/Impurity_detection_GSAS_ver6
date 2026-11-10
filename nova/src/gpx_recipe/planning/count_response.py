"""Positive count-reference inference and an explicitly elastic transfer operator.

The observations must be unweighted integer ON/OFF counts, with known exposure.
The model is n~Poisson(q_on*(s+b)), m~Poisson(q_off*b), s,b>=0, with independent
flat priors on s and b. This is a declared statistical model, not a test that a
facility stream is Poisson or stationary. Rates in distinct supplied cells are
independent under this model. No subtraction, clipping or Gaussian error floor
is used. Physical source drift, metrology and scattering-law uncertainty remain
separate inputs.

Elastic transfer is conditional on supplied differential cross sections, path
weights and a constant effective fluence density within each reference cell.
It does not invert an inelastic vanadium measurement into an incident spectrum.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy import sparse
from scipy.stats import nbinom

VERSION = 'powgen-positive-count-reference/1'
_MAX_CELLS = 100000
_MAX_COUNT = 1000000000
_TAIL_TOL = 1e-13


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(name + ' must be a finite positive scalar')
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(name + ' must be a finite positive scalar')
    return result


def _counts(values):
    array = np.asarray(values)
    if (array.dtype.kind not in 'iuf' or array.ndim < 1 or not 1 <= array.size <= _MAX_CELLS
            or not np.isfinite(array).all() or np.any(array < 0) or np.any(array > _MAX_COUNT)
            or not np.array_equal(array, np.rint(array))):
        raise ValueError('Observations must be bounded unweighted nonnegative integer count arrays')
    return np.array(array, dtype=np.int64, copy=True)


def _latent_window(n, m, ratio, *, max_terms=200000):
    """Stable truncated NB weights, with a log-concave geometric tail bound.

    The latent number k of ON background events has probability proportional
    to Gamma(m+k+1)/k! * [q_on/(q_on+q_off)]**k, 0<=k<=n.
    Starting at its truncated mode avoids tiny CDFs and overflow. The omitted
    probability bound is <=2e-13, including in an arbitrarily rare lower tail.
    """
    q = ratio / (1 + ratio)
    mode = min(n, int(math.floor(m * ratio)))
    indices, weights = [mode], [1.]
    total = 1.
    tails = []
    for direction in (-1, 1):
        k, weight = mode, 1.
        while (k > 0 if direction < 0 else k < n):
            step = k / ((m + k) * q) if direction < 0 else q * (m + k + 1) / (k + 1)
            tail = weight * step / (1 - step) if step < 1 else math.inf
            if tail <= _TAIL_TOL * total:
                tails.append(tail)
                break
            weight *= step
            k += direction
            indices.append(k)
            weights.append(weight)
            total += weight
            if len(indices) > max_terms or not math.isfinite(total):
                raise ValueError('Latent count summation exceeds its numerical resource budget')
        else:
            tails.append(0.)
    order = np.argsort(indices)
    indices = np.array(indices, dtype=np.int64)[order]
    probability = np.array(weights)[order] / total
    return indices, probability, float(sum(tails) / total)


@dataclass(frozen=True)
class OnOffPosterior:
    on_counts: np.ndarray
    off_counts: np.ndarray
    on_exposure: float
    off_exposure: float
    source_mean: np.ndarray
    background_mean: np.ndarray
    source_variance: np.ndarray
    background_variance: np.ndarray
    source_background_covariance: np.ndarray
    _cdf: np.ndarray
    _windows: dict
    metadata: dict

    def sample_rates(self, draws, seed=0, *, max_bytes=256 * 1024**2):
        """Joint positive rates; reuse a draw across times/specimens in a campaign."""
        if type(draws) is not int or draws < 1 or type(max_bytes) is not int or max_bytes < 1024:
            raise ValueError('Explicit positive draw and memory budgets are required')
        # Include latent integers, uniform/PPF working arrays and both rate outputs.
        if draws * self.on_counts.size * 64 > max_bytes:
            raise ValueError('Posterior draws exceed their declared memory budget')
        rng = np.random.default_rng(seed)
        n, m = self.on_counts.ravel(), self.off_counts.ravel()
        k = np.empty((draws, len(n)), dtype=np.int64)
        normal = np.ones(len(n), dtype=bool)
        normal[list(self._windows)] = False
        if normal.any():
            cdf = self._cdf.ravel()[normal]
            u = np.maximum(rng.random((draws, normal.sum())) * cdf, np.nextafter(0., 1.))
            u = np.minimum(u, np.nextafter(cdf, 0.))
            p = self.off_exposure / (self.on_exposure + self.off_exposure)
            values = nbinom.ppf(u, m[normal] + 1, p)
            if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > n[normal]):
                raise ValueError('Latent count quantile failed its support check')
            k[:, normal] = values.astype(np.int64)
        for index, (support, probability, _) in self._windows.items():
            k[:, index] = rng.choice(support, size=draws, p=probability)
        source = rng.gamma(n - k + 1, 1 / self.on_exposure)
        background = rng.gamma(m + k + 1, 1 / (self.on_exposure + self.off_exposure))
        return source.reshape((draws,) + self.on_counts.shape), background.reshape((draws,) + self.on_counts.shape)

    def _weights(self, source_weights, background_weights=None):
        a = sparse.csr_matrix(source_weights, dtype=float)
        b = sparse.csr_matrix(a.shape) if background_weights is None else sparse.csr_matrix(background_weights, dtype=float)
        if a.shape != b.shape or a.shape[1] != self.on_counts.size or not 1 <= a.shape[0] <= 100000:
            raise ValueError('Transfer matrices must match the flattened reference cells')
        for matrix in (a, b):
            if matrix.nnz > 2000000 or not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
                raise ValueError('Count transfer matrices must be bounded, finite and nonnegative')
        return a, b

    def linear_moments(self, source_weights, background_weights=None):
        """Mean and diagonal calibration variance of explicitly declared rate maps.

        To get new integer-count predictive variance at exposure q, use
        q*mean + q**2*variance. Background is included only via its supplied map.
        """
        a, b = self._weights(source_weights, background_weights)
        mean = a @ self.source_mean.ravel() + b @ self.background_mean.ravel()
        variance = (a.multiply(a) @ self.source_variance.ravel()
                    + b.multiply(b) @ self.background_variance.ravel()
                    + 2 * a.multiply(b) @ self.source_background_covariance.ravel())
        if np.any(variance < -1e-12 * np.maximum(1., np.abs(variance))):
            raise ValueError('Invalid projected calibration variance')
        return np.asarray(mean), np.maximum(np.asarray(variance), 0.)

    def linear_covariance(self, source_weights, background_weights=None):
        """Small projected covariance, retaining source/blank anticorrelation."""
        a, b = self._weights(source_weights, background_weights)
        if a.shape[0] > 256:
            raise ValueError('Project onto at most 256 declared functionals; do not allocate a full pattern covariance')
        covariance = ((a.multiply(self.source_variance.ravel()) @ a.T)
                      + (b.multiply(self.background_variance.ravel()) @ b.T)
                      + (a.multiply(self.source_background_covariance.ravel()) @ b.T)
                      + (b.multiply(self.source_background_covariance.ravel()) @ a.T))
        return covariance.toarray()


def on_off_posterior(on_counts, off_counts, on_exposure, off_exposure):
    """Infer nonnegative source and blank rates without clipping signed differences.

    A latent truncated negative-binomial mixture gives independent conditional
    Gamma(s;n-k+1,q_on) and Gamma(b;m+k+1,q_on+q_off). This preserves the joint
    covariance, zero-count support and the uncertainty of a finite blank.
    The prior is flat per supplied rate cell, so changing cells changes the model.
    """
    n, m = _counts(on_counts), _counts(off_counts)
    if n.shape != m.shape:
        raise ValueError('ON and OFF count arrays must have identical shapes')
    qon, qoff = _positive(on_exposure, 'ON exposure'), _positive(off_exposure, 'OFF exposure')
    ratio = qon / qoff
    if (not 1e-8 <= ratio <= 1e8 or not math.isfinite(qon + qoff)
            or min(qon, qoff) < math.sqrt(np.finfo(float).tiny)
            or qon + qoff > math.sqrt(np.finfo(float).max)):
        raise ValueError('Exposure ratio is outside the bounded inference domain')
    shape = n.shape
    nf, mf = n.ravel(), m.ravel()
    r = mf.astype(float) + 1
    p = qoff / (qon + qoff)
    cdf = nbinom.cdf(nf, r, p)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        hazard = np.exp(nbinom.logpmf(nf, r, p) - np.log(cdf))
        delta = ratio * (nf + r) * hazard
        ek = r * ratio - delta
        vk = r * ratio * (1 + ratio) - delta * (nf + 1 + ratio - ek)
    unstable = ((cdf < 1e-8) | ~np.isfinite(ek + vk) | (ek < 0) | (ek > nf)
                | (vk < 0) | (vk < 1e-5 * r * ratio * (1 + ratio)) | (nf == 0))
    windows = {}
    for index in np.flatnonzero(unstable):
        support, probability, bound = _latent_window(int(nf[index]), int(mf[index]), ratio)
        mean = float(probability @ support)
        ek[index], vk[index] = mean, float(probability @ (support - mean)**2)
        windows[int(index)] = (support, probability, bound)
    total_q = qon + qoff
    source_mean = (nf + 1 - ek) / qon
    background_mean = (mf + 1 + ek) / total_q
    source_var = (nf + 1 - ek + vk) / qon**2
    background_var = (mf + 1 + ek + vk) / total_q**2
    covariance = -vk / (qon * total_q)
    output = [n, m, source_mean, background_mean, source_var, background_var, covariance, cdf]
    output = [np.asarray(value).reshape(shape) for value in output]
    if not all(np.isfinite(v).all() for v in output):
        raise ValueError('Posterior moments exceed their finite numerical domain')
    for value in output:
        value.setflags(write=False)
    metadata = dict(version=VERSION, likelihood='independent Poisson ON/OFF with known exposure',
        prior='independent flat nonnegative source and background rate in each supplied cell',
        cells=n.size, signed_difference_not_clipped=True, latent_tail_tolerance=_TAIL_TOL,
        largest_omitted_latent_probability_bound=max((v[2] for v in windows.values()), default=0.),
        physical_Poisson_independence_or_stationarity_verified=False,
        metrology_or_scattering_law_uncertainty_included=False, experimental_accuracy_certified=False)
    return OnOffPosterior(output[0], output[1], qon, qoff, *output[2:7], output[7], windows, metadata)


def _edges(values, name):
    array = np.asarray(values, float)
    if (array.ndim != 1 or not 2 <= len(array) <= 100001 or not np.isfinite(array).all()
            or array[0] <= 0 or np.any(np.diff(array) <= 0)):
        raise ValueError(name + ' must be finite positive increasing wavelength edges')
    return array


def _integrals(edges, cross_section, path_weight, order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    half = np.diff(edges) / 2
    wave = (edges[:-1] + half)[:, None] + half[:, None] * nodes
    xs = np.broadcast_to(np.asarray(cross_section(wave.ravel()), float), (wave.size,)).reshape(wave.shape)
    path = np.broadcast_to(np.asarray(path_weight(wave.ravel()), float), (wave.size,)).reshape(wave.shape)
    if (not np.isfinite(xs).all() or np.any(xs < 0) or not np.isfinite(path).all()
            or np.any(path < 0) or np.any(path > 1)):
        raise ValueError('Cross sections must be nonnegative; whole-powder path weights must lie in [0,1]')
    return half * ((xs * path) @ weights)


def elastic_transfer(reference_wavelength_edges_a, target_wavelength_edges_a, *,
                     reference_cross_section, sample_cross_section, reference_path_weight,
                     sample_path_weight, reference_cells, sample_cells, energy_model,
                     integration_breakpoints_a=(), quadrature_order=8, max_bytes=128 * 1024**2):
    """Map measured net reference rates to sample rates, with finite-bin integrals.

    Cross sections are differential barn/sr per declared cell at one physical
    direction; path weights include the illuminated fraction of the WHOLE
    powder once. Both cell counts are for that whole powder. Effective incident
    fluence times elastic detector acceptance is constant per reference cell.
    W_ij=(Ns/Nr)*integral_overlap(Ts*dSigma_s/dOmega dlambda)
                    /integral_reference_cell(Tr*dSigma_r/dOmega dlambda).

    This ratio of integrals is not the integral of a pointwise ratio. Reference
    and target can have different bins, and summing target bins conserves the
    modeled counts. Inelastic scattering needs an energy-redistributing operator
    and is deliberately not accepted by this route. No model is qualified here.
    Supply integration breakpoints resolving every sharp profile/Bragg edge and
    check quadrature convergence. An empty list asserts smooth supplied laws;
    no finite set of black-box quadrature nodes can detect arbitrary hidden peaks.
    """
    if energy_model != 'elastic-only':
        raise ValueError('This transfer requires the explicitly declared elastic-only energy model')
    ref = _edges(reference_wavelength_edges_a, 'Reference')
    target = _edges(target_wavelength_edges_a, 'Target')
    breaks = np.asarray(integration_breakpoints_a, float)
    if (breaks.ndim != 1 or len(breaks) > 200000 or not np.isfinite(breaks).all()
            or np.any(np.diff(breaks) <= 0) or np.any(breaks < ref[0]) or np.any(breaks > ref[-1])):
        raise ValueError('Integration breakpoints must increase within reference coverage')
    if target[0] < ref[0] or target[-1] > ref[-1]:
        raise ValueError('Target bins extend beyond measured reference coverage')
    if type(quadrature_order) is not int or not 2 <= quadrature_order <= 32:
        raise ValueError('Quadrature order must be in [2,32]')
    if type(max_bytes) is not int or max_bytes < 1024 or (len(ref) + len(target) + 2 * len(breaks)) * quadrature_order * 128 > max_bytes:
        raise ValueError('Elastic transfer exceeds its declared numerical memory budget')
    scale = _positive(sample_cells, 'Whole sample cells') / _positive(reference_cells, 'Whole reference cells')
    reference_pieces = np.unique(np.concatenate((ref, breaks)))
    reference_index = np.searchsorted(ref, (reference_pieces[1:] + reference_pieces[:-1]) / 2, side='right') - 1
    denominator = np.bincount(reference_index,
        weights=_integrals(reference_pieces, reference_cross_section, reference_path_weight, quadrature_order),
        minlength=len(ref) - 1)
    segments = np.unique(np.concatenate((ref[(ref > target[0]) & (ref < target[-1])], target,
        breaks[(breaks > target[0]) & (breaks < target[-1])])))
    middle = (segments[1:] + segments[:-1]) / 2
    ri = np.searchsorted(ref, middle, side='right') - 1
    ti = np.searchsorted(target, middle, side='right') - 1
    if np.any(denominator[ri] <= 0):
        raise ValueError('A target bin intersects zero modeled reference sensitivity')
    numerator = _integrals(segments, sample_cross_section, sample_path_weight, quadrature_order)
    result = sparse.coo_matrix((scale * numerator / denominator[ri], (ti, ri)),
                               shape=(len(target) - 1, len(ref) - 1)).tocsr()
    result.eliminate_zeros()
    if not np.isfinite(result.data).all():
        raise ValueError('Elastic count transfer exceeds its finite numerical domain')
    return result
