"""Explicit cumulative likelihood targets; normalized intensities are not counts."""
from dataclasses import dataclass, asdict
import hashlib
import json

import numpy as np
from scipy.special import gammaln, xlogy


class NewProtocolRequired(ValueError):
    """A data or preparation change prevents a silent posterior continuation."""


@dataclass(frozen=True)
class Acquisition:
    specimen: str
    run: str
    bank: str
    reduction: str
    epoch: str
    profile_sha256: str
    axis: str
    normalization: str
    exposure_unit: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in asdict(self).values()):
            raise ValueError('Complete acquisition identity is required')
        if self.axis not in {'tof_us', 'two_theta_deg', 'd_angstrom', 'q_inverse_angstrom'}:
            raise ValueError('Undeclared diffraction coordinate')


def array_digest(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True, eq=False)
class CumulativeObservation:
    acquisition: Acquisition
    protocol_sha256: str
    revision: int
    exposure: float
    x: np.ndarray
    y: np.ndarray
    variance: np.ndarray
    mask: np.ndarray

    def __post_init__(self):
        if not self.protocol_sha256 or type(self.revision) is not int or self.revision < 0:
            raise ValueError('Protocol and nonnegative integer revision are required')
        if not np.isfinite(self.exposure) or self.exposure <= 0:
            raise ValueError('Positive finite cumulative exposure is required')
        for name in ['x', 'y', 'variance', 'mask']:
            a = np.array(getattr(self, name), dtype=bool if name == 'mask' else float, copy=True)
            a.flags.writeable = False; object.__setattr__(self, name, a)
        n = len(self.x)
        if n < 2 or any(a.shape != (n,) for a in [self.x,self.y,self.variance,self.mask]):
            raise ValueError('Observation arrays must share one nonempty grid')
        if not np.isfinite(self.x).all() or not np.all(np.diff(self.x) > 0) or not self.mask.any():
            raise ValueError('Ordered finite axis and usable bins required')
        if not np.isfinite(self.y[self.mask]).all() or not np.isfinite(self.variance[self.mask]).all() or np.any(self.variance[self.mask] <= 0):
            raise ValueError('Usable bins need finite observations and positive variances')

    @property
    def observation_sha256(self):
        body = json.dumps(dict(acquisition=asdict(self.acquisition), protocol=self.protocol_sha256,
                               revision=self.revision, exposure=float(self.exposure)), sort_keys=True)
        return hashlib.sha256((body+array_digest(self.x,self.y,self.variance,self.mask)).encode()).hexdigest()

    def require_continuation(self, previous):
        if type(self) is not type(previous):
            raise NewProtocolRequired('Observation likelihood representation changed')
        if self.acquisition != previous.acquisition or self.protocol_sha256 != previous.protocol_sha256:
            raise NewProtocolRequired('Acquisition or reviewed protocol changed')
        if not np.array_equal(self.x, previous.x) or not np.array_equal(self.mask, previous.mask):
            raise NewProtocolRequired('Observation grid or mask changed')
        if self.revision == previous.revision:
            if self.observation_sha256 == previous.observation_sha256: return 'duplicate'
            raise NewProtocolRequired('Producer reused a revision for different observations')
        if self.revision < previous.revision: raise ValueError('Stale acquisition revision')
        if self.exposure < previous.exposure:
            raise NewProtocolRequired('Cumulative exposure reset')
        # Reprocessing at identical exposure is a replacement likelihood, not
        # new independent information, provided reduction identity is unchanged.
        return 'replace_likelihood'


@dataclass(frozen=True, eq=False)
class CumulativeCounts(CumulativeObservation):
    """Independent raw bin counts with an explicitly supplied rate conversion.

    ``E * count_rate_scale * prediction`` is the expected count. Never construct
    this object by estimating counts from a normalized or rebinned reduced Y.
    The inherited variance is a plotting/Rwp diagnostic, not the likelihood.
    Background subtraction, weighted events and correlated rebinning require
    another observation model and are not covered by this class.
    """
    counts: np.ndarray
    count_rate_scale: np.ndarray

    def __post_init__(self):
        super().__post_init__()
        counts = np.asarray(self.counts)
        if (counts.shape != self.x.shape or not np.isfinite(counts).all() or
                np.any(counts < 0) or np.any(counts != np.floor(counts)) or
                np.any(counts >= np.iinfo(np.int64).max)):
            raise ValueError('Exact nonnegative integer bin counts are required')
        counts = np.array(counts, dtype=np.int64, copy=True)
        scale = np.broadcast_to(np.asarray(self.count_rate_scale, float), self.x.shape).copy()
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError('Positive finite count-rate conversion is required')
        if not np.allclose(self.y, counts / (self.exposure * scale), rtol=1e-12, atol=0):
            raise ValueError('Intensity does not match the supplied counts and exposure conversion')
        for name, value in [('counts', counts), ('count_rate_scale', scale)]:
            value.flags.writeable = False; object.__setattr__(self, name, value)

    @classmethod
    def from_counts(cls, acquisition, protocol_sha256, revision, exposure, x, counts, count_rate_scale, mask):
        denominator = exposure * np.asarray(count_rate_scale, float)
        counts = np.asarray(counts)
        return cls(acquisition, protocol_sha256, revision, exposure, x,
                   counts / denominator, np.maximum(counts, 1) / denominator**2,
                   mask, counts, count_rate_scale)

    @property
    def observation_sha256(self):
        return hashlib.sha256((super().observation_sha256 + 'independent-poisson-counts-v1' +
                               array_digest(self.counts, self.count_rate_scale)).encode()).hexdigest()

    def require_continuation(self, previous):
        state = super().require_continuation(previous)
        if not np.array_equal(self.count_rate_scale, previous.count_rate_scale):
            raise NewProtocolRequired('Count-rate conversion changed')
        if np.any(self.counts < previous.counts):
            raise NewProtocolRequired('Independent cumulative raw counts cannot decrease')
        if self.exposure == previous.exposure and not np.array_equal(self.counts, previous.counts):
            raise NewProtocolRequired('Raw counts changed without additional exposure')
        return state


def poisson_means(observation, prediction):
    if not isinstance(observation, CumulativeCounts):
        raise TypeError('A verified raw-count observation is required for a Poisson likelihood')
    prediction = np.asarray(prediction, float)
    if prediction.shape != observation.y.shape:
        raise ValueError('Prediction does not match observed bins')
    use = observation.mask
    means = observation.exposure * observation.count_rate_scale[use] * prediction[use]
    if not np.isfinite(means).all() or np.any(means < 0):
        raise ValueError('Poisson expected counts must be finite and nonnegative')
    return means



def count_rwp_percent(observation, prediction):
    """Poisson-weighted Rwp diagnostic in verified count units.

    The conventional weight is 1/max(observed_count, 1). Convert to floating
    point before squaring: int64 counts overflow above about 3e9 per bin.
    This descriptive fit statistic is not a model-selection probability.
    """
    expected = poisson_means(observation, prediction)
    counts = observation.counts[observation.mask].astype(float)
    denominator = float(np.sum(counts))  # k**2/max(k,1) == k for integer k>=0
    if denominator == 0:
        return None
    residual = (counts - expected) / np.sqrt(np.maximum(counts, 1.))
    return float(100 * np.linalg.norm(residual) / np.sqrt(denominator))


def count_curve_difference_sigma(observation, reference, approximation):
    """Maximum curve disagreement relative to predicted Poisson noise.

    Uses the acquisition's exposure AND bin-dependent rate conversion.
    Exact agreement at zero expected count is zero; disagreement is infinite.
    """
    expected = poisson_means(observation, reference)
    other = poisson_means(observation, approximation)
    difference = np.abs(other - expected)
    scaled = np.zeros_like(expected)
    np.divide(difference, np.sqrt(expected), out=scaled, where=expected>0)
    scaled[(expected == 0) & (difference > 0)] = np.inf
    return float(np.max(scaled))


def poisson_log_likelihood(observation, prediction):
    means = poisson_means(observation, prediction)
    counts = observation.counts[observation.mask]
    return float(np.sum(xlogy(counts, means) - means - gammaln(counts + 1.)))


def poisson_deviance_residuals(observation, prediction):
    """Signed square-root likelihood-ratio residuals, stable near large counts."""
    means = poisson_means(observation, prediction)
    counts = observation.counts[observation.mask]
    half = means.copy()
    positive = counts > 0
    k=counts[positive]; mu=means[positive]
    relative = (mu-k)/k
    with np.errstate(divide='ignore'):
        # log1p avoids cancellation near equality. For very small means the
        # relative difference rounds to -1; direct log ratios remain finite.
        near = np.abs(relative) < .5
        value=np.empty_like(mu)
        value[near]=k[near]*(relative[near]-np.log1p(relative[near]))
        value[~near]=mu[~near]-k[~near]+k[~near]*(np.log(k[~near])-np.log(mu[~near]))
        half[positive] = value
    return np.sign(means - counts) * np.sqrt(2 * np.maximum(half, 0.))


def gaussian_log_likelihood(observation, prediction):
    """Diagonal reduced-data likelihood with explicit supplied variances.

    This is a conditional likelihood, not a qualification of independent bins
    or shared-reference uncertainty. Correlated/reduction terms must be modeled
    before a stopping policy is certified.
    """
    prediction = np.asarray(prediction, float)
    if prediction.shape != observation.y.shape or not np.isfinite(prediction[observation.mask]).all():
        raise ValueError('Prediction does not match observed usable bins')
    use = observation.mask; variance = observation.variance[use]
    residual = observation.y[use] - prediction[use]
    return float(-.5 * np.sum(residual**2 / variance + np.log(2*np.pi*variance)))


def observation_log_likelihood(observation, prediction):
    if isinstance(observation, CumulativeCounts):
        return poisson_log_likelihood(observation, prediction)
    return gaussian_log_likelihood(observation, prediction)


def cumulative_log_weight_change(previous, current, old_prediction, new_prediction):
    """Importance bridge between cumulative-summary posteriors.

    Use this difference once. The old posterior must already target the old
    likelihood. This does not invent independent increments from reduced Y.
    """
    state = current.require_continuation(previous)
    if state == 'duplicate':
        if not np.array_equal(old_prediction, new_prediction):
            raise ValueError('Duplicate snapshot changed its model prediction')
        return 0.
    old=observation_log_likelihood(previous,old_prediction)
    new=observation_log_likelihood(current,new_prediction)
    if not np.isfinite(old):
        raise ValueError('Old particle is outside finite likelihood support; discard it before importance bridging')
    return new-old
