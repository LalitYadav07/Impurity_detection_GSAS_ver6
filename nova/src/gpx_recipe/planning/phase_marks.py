"""Positive conditional mark calibration with coupled channel uncertainty.

The model describes channel probabilities given observed row totals and phase
coordinates. It does not infer absolute beam flux, separate source/background
components, or make fitted detector response transferable between specimens.
The first explicitly named channel fixes the otherwise arbitrary logit gauge.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import logsumexp

VERSION = 'harmonic-conditional-mark-calibration/1'


def _design(phases_rad, degree):
    phase = np.asarray(phases_rad)
    if (phase.ndim != 1 or not 1 <= len(phase) <= 2**20 or phase.dtype.kind not in 'iuf'
            or not np.isfinite(phase).all() or np.any(abs(phase) > 1e8)):
        raise ValueError('Bounded finite phase coordinates in radians are required')
    phase = np.remainder(phase.astype(float), 2*np.pi)
    out = np.ones((len(phase), 1+2*degree))
    for h in range(1, degree+1):
        out[:, 2*h-1], out[:, 2*h] = np.cos(h*phase), np.sin(h*phase)
    return out


def _readonly(value):
    value = np.ascontiguousarray(value, dtype=float)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _probability(design, parameter):
    logits = np.column_stack([np.zeros(len(design)), design@parameter.T])
    logp = logits-logsumexp(logits, axis=1, keepdims=True)
    p = np.exp(logp)
    if np.any(p == 0) or not np.isfinite(p).all():
        raise ValueError('A positive calibrated probability exceeds floating-point range')
    return p, logp


def _objective(parameter, design, counts):
    theta = parameter.reshape(counts.shape[1]-1, design.shape[1])
    p, logp = _probability(design, theta)
    total = counts.sum(1); N = total.sum()
    score = (total[:, None]*p[:, 1:]-counts[:, 1:]).T@design
    return float(-np.sum(counts*logp)/N), score.ravel()/N


def _information(design, counts, parameter):
    p, _ = _probability(design, parameter)
    total = counts.sum(1); d = design.shape[1]; channels = p.shape[1]-1
    common = (np.sqrt(total)[:, None, None]*p[:, 1:, None]*design[:, None, :]).reshape(len(design), -1)
    info = -common.T@common
    for b in range(channels):
        sl = slice(b*d, (b+1)*d)
        info[sl, sl] += (design.T*(total*p[:, b+1]))@design
    return (info+info.T)/2


def _inverse_information(info):
    scale = np.sqrt(np.diag(info))
    if np.any(scale <= 0): raise ValueError('Conditional mark calibration is not identified')
    factor = cho_factor(info/scale[:, None]/scale[None, :], lower=True)
    cov = cho_solve(factor, np.eye(len(info)))/scale[:, None]/scale[None, :]
    return (cov+cov.T)/2


@dataclass(frozen=True)
class HarmonicMarkFit:
    channel_ids: tuple[str, ...]
    degree: int
    coefficients: np.ndarray
    covariance: np.ndarray
    provenance: str
    calibration_events: int
    likelihood_score_decrement: float

    def __post_init__(self):
        if isinstance(self.channel_ids, (str, bytes)):
            raise ValueError('An explicit sequence of channel IDs is required')
        ids = tuple(self.channel_ids); c, cov = np.asarray(self.coefficients), np.asarray(self.covariance)
        if (not 2 <= len(ids) <= 512 or any(not isinstance(s, str) or not s or len(s) > 256 for s in ids)
                or len(set(ids)) != len(ids) or type(self.degree) is not int or not 0 <= self.degree <= 16
                or not isinstance(self.provenance, str) or not self.provenance.strip() or len(self.provenance) > 4096
                or c.shape != (len(ids)-1, 1+2*self.degree) or c.dtype.kind not in 'iuf'
                or cov.shape != (c.size, c.size) or cov.dtype.kind not in 'iuf'
                or not np.isfinite(c).all() or not np.isfinite(cov).all()
                or type(self.calibration_events) is not int or not 0 < self.calibration_events <= 2**40
                or not np.isfinite(self.likelihood_score_decrement) or self.likelihood_score_decrement < 0):
            raise ValueError('Matching identified named mark coefficients, covariance and calibration provenance are required')
        if not np.allclose(cov, cov.T, rtol=2e-13, atol=0.):
            raise ValueError('A symmetric conditional calibration covariance is required')
        np.linalg.cholesky(cov)
        object.__setattr__(self, 'channel_ids', ids)
        object.__setattr__(self, 'coefficients', _readonly(c))
        object.__setattr__(self, 'covariance', _readonly(cov))

    def probabilities(self, phases_rad, *, max_bytes=256*1024**2):
        phase = np.asarray(phases_rad)
        if phase.ndim != 1: raise ValueError('A vector of phase coordinates is required')
        if (type(max_bytes) is not int or max_bytes < 1024
                or len(phase)*(64*len(self.channel_ids)+8*(1+2*self.degree)) > max_bytes):
            raise ValueError('Conditional mark probabilities exceed the working-memory budget')
        x = _design(phase, self.degree)
        return _readonly(_probability(x, self.coefficients)[0])

    @property
    def metadata(self):
        return dict(version=VERSION, provenance=self.provenance, reference_channel=self.channel_ids[0],
            channel_ids=list(self.channel_ids), degree=self.degree, calibration_events=self.calibration_events,
            likelihood_score_decrement=self.likelihood_score_decrement,
            observed_row_totals_are_conditioning_inputs=True,
            covariance='Conditional local Gaussian likelihood information; bank coefficients coupled',
            coefficient_order='Nonreference channel, then intercept,c1,s1,c2,s2,...',
            absolute_source_rate_or_background_decomposition_not_inferred=True,
            transfer_to_other_specimens_or_configurations_not_established=True,
            experimental_accuracy_certified=False)


def fit_harmonic_marks(counts, phases_rad, channel_ids, *, degree, provenance, max_bytes=256*1024**2):
    """Fit a multinomial harmonic logit on fixed, explicitly supplied phases.

    Zero-total rows carry no likelihood information. Every retained channel
    must have observed counts; empty channels require an explicit outside
    mask/contract rather than pseudocounts. Calibration covariance is local
    and conditional; it does not include phase-estimation/process uncertainty.
    """
    if isinstance(channel_ids, (str, bytes)): raise ValueError('An explicit sequence of channel IDs is required')
    y = np.asarray(counts); ids = tuple(channel_ids)
    if (type(degree) is not int or not 0 <= degree <= 16 or y.ndim != 2
            or y.dtype.kind not in 'iu' or np.any(y < 0) or not 2 <= y.shape[1] <= 512
            or len(ids) != y.shape[1] or any(not isinstance(s, str) or not s or len(s) > 256 for s in ids)
            or len(set(ids)) != len(ids)
            or not isinstance(provenance, str) or not provenance.strip() or len(provenance) > 4096
            or type(max_bytes) is not int or max_bytes < 1024):
        raise ValueError('Named distinct channels, integer counts, bounded degree and provenance are required')
    if np.any(y > 2**40) or not 0 < np.sum(y, dtype=float) <= 2**40:
        raise ValueError('Finite exactly represented bounded calibration counts are required')
    parameters = (y.shape[1]-1)*(1+2*degree)
    if 48*len(y)*parameters+64*parameters**2+y.nbytes+8*len(y)*(1+2*degree) > max_bytes:
        raise ValueError('Conditional mark fit exceeds the working-memory budget')
    x = _design(phases_rad, degree)
    if len(x) != len(y): raise ValueError('Every count row requires its supplied phase')
    positive = y.sum(1) > 0; y, x = y[positive].astype(float), x[positive]
    totals = y.sum(0)
    if np.any(totals == 0): raise ValueError('An empty channel has no finite logit calibration; no pseudocount is inserted')
    if np.linalg.matrix_rank(x) != x.shape[1]: raise ValueError('Supplied phase coordinates do not identify the requested harmonics')
    initial = np.zeros((len(ids)-1, x.shape[1])); initial[:, 0] = np.log(totals[1:]/totals[0])
    result = minimize(_objective, initial.ravel(), args=(x, y), jac=True, method='L-BFGS-B',
        options={'maxiter':1000, 'maxls':40, 'ftol':2e-15, 'gtol':1e-10})
    if not result.success: raise RuntimeError('Conditional harmonic mark likelihood did not converge')
    parameter = result.x.copy(); N = y.sum()
    # Newton refinement is measured in likelihood-information units, rather
    # than accepting a visually plausible optimizer message alone.
    for _ in range(12):
        value, gradient = _objective(parameter, x, y)
        info = _information(x, y, parameter.reshape(initial.shape)); cov = _inverse_information(info)
        g = gradient*N; decrement = float(np.sqrt(g@cov@g))
        if decrement < 1e-4: break
        step = cov@g; length = 1.
        while length > 2**-20:
            proposal = parameter-length*step
            if _objective(proposal, x, y)[0] <= value+2e-15:
                parameter = proposal; break
            length /= 2
        else: raise RuntimeError('Conditional mark likelihood Newton refinement failed')
    else: raise RuntimeError('Conditional mark likelihood failed its information-scaled score check')
    return HarmonicMarkFit(ids, degree, _readonly(parameter.reshape(initial.shape)), _readonly(cov),
        provenance, int(N), decrement)
