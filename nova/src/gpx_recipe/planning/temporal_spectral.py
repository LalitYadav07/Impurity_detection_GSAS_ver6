"""Nonnegative finite periodic intensities, without the sufficient L1 bound.

Each independent phase contributes a sum of squared complex polynomials.
Per-channel stationary mean is one. This extends the mathematical count
process; it neither infers a physical cause nor calibrates a POWGEN response.
The original PhaseRateLaw remains available with its original identities.
"""
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from .temporal_counts import PhaseRateLaw, _budget, _readonly

VERSION = 'spectral-factor-phase-pulse-counts/1'
_FACTOR_LIMIT = 128 * 1024**2
_HALF_PI = np.pi / 2


def _immutable(value):
    """An array backed by immutable bytes, including through setflags()."""
    a = np.ascontiguousarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _shifted(factor):
    """Taylor coefficients at the four exact unit-circle quadrants.

    Positive half-angle coordinates for z/centre-1 avoid subtracting cos(phi)
    from one near a multiple root. Accurate coefficient sums retain exact
    cancellations in representable factors such as (1-z)**2.
    """
    result = np.empty((4,) + factor.shape, complex)
    degree = factor.shape[-1] - 1
    for quadrant in range(4):
        phase = np.array([1, 1j, -1, -1j], complex)[(quadrant*np.arange(degree+1)) % 4]
        rotated = factor * phase
        for index in np.ndindex(factor.shape[:-1]):
            for h in range(degree+1):
                terms = [math.comb(k, h)*rotated[index+(k,)] for k in range(h, degree+1)]
                result[(quadrant,)+index+(h,)] = complex(
                    math.fsum(float(v.real) for v in terms),
                    math.fsum(float(v.imag) for v in terms))
    return result


def _evaluate(factor, shifted, phases):
    """Bounded-degree Horner, with local coordinates near exact quadrants."""
    z = np.cos(phases) + 1j*np.sin(phases)
    value = np.broadcast_to(factor[..., -1], (len(phases),)+factor.shape[:-1]).copy()
    for k in range(factor.shape[-1]-2, -1, -1):
        value *= z[:, None, None]
        value += factor[..., k]
    nearest = np.rint(phases/_HALF_PI).astype(int)
    residual = phases-nearest*_HALF_PI
    for quadrant in range(4):
        ix = np.flatnonzero((nearest % 4 == quadrant) & (abs(residual) <= .125))
        if not len(ix):
            continue
        a = residual[ix]
        delta = -2*np.sin(a/2)**2 + 1j*np.sin(a)
        coefficients = shifted[quadrant]
        local = np.broadcast_to(coefficients[..., -1], (len(ix),)+factor.shape[:-1]).copy()
        for k in range(factor.shape[-1]-2, -1, -1):
            local *= delta[:, None, None]
            local += coefficients[..., k]
        value[ix] = local
    return value


def _norm_over_pulses_and_rows(amplitude):
    """Scaled positive norm; no unscaled squaring of tiny or large values."""
    scale = np.max(amplitude, axis=(0, 2))
    relative = np.divide(amplitude, scale[None, :, None],
                         out=np.zeros_like(amplitude), where=scale[None, :, None] > 0)
    return scale*np.sqrt(np.sum(relative**2, axis=(0, 2)))


def _squared(norm):
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        result = norm*norm
    if not np.isfinite(result).all():
        raise ValueError('Requested positive intensity or mean exceeds floating-point range')
    if np.any((norm > 0) & (result == 0)):
        raise ValueError('A positive intensity or mean is below floating-point range; integrate with explicit rates and dose')
    return result


@dataclass(frozen=True, init=False)
class SpectralPhaseRateLaw(PhaseRateLaw):
    """g_j = sum_(m,r) |sum_k F[m,j,r,k] exp(i*k*phi_m)|**2.

    Supply factors with shape (phases, distinct channels, rows, degree+1).
    Total phase-averaged energy is normalized to one per channel. One phase
    covers every nonnegative finite scalar trigonometric polynomial; sums of
    independent phase contributions do not imply arbitrary multivariate
    coupling. Factors, derived coefficients and phase parameters are immutable.
    """
    normalized_factors: np.ndarray

    def __init__(self, frequency_hz, phase_variance_rate_rad2_s, factors, provenance):
        a = np.asarray(factors)
        if (a.ndim != 4 or a.dtype.kind not in 'iufc' or a.real.dtype.itemsize > 8
                or not 1 <= a.shape[0] <= 8 or not 1 <= a.shape[1] <= 100000
                or not 1 <= a.shape[2] <= 32 or not 1 <= a.shape[3] <= 17
                or a.size*16 > _FACTOR_LIMIT or not np.isfinite(a).all()):
            raise ValueError('Bounded finite polynomial factors with phase/channel/row/coefficient axes are required')
        degree = max(1, a.shape[3]-1)
        shape = (a.shape[0], a.shape[1], degree)
        # Reuse the original phase, frequency, diffusion and provenance domain
        # validation. Its sufficient-amplitude evaluator is replaced below.
        PhaseRateLaw.__init__(self, frequency_hz, phase_variance_rate_rad2_s,
                              np.zeros(shape), np.zeros(shape), provenance)
        real, imag = np.array(a.real, float, copy=True), np.array(a.imag, float, copy=True)
        scale = np.maximum(np.max(abs(real), axis=(0, 2, 3)), np.max(abs(imag), axis=(0, 2, 3)))
        if np.any(scale == 0):
            raise ValueError('Every channel requires positive stationary factor energy')
        real /= scale[None, :, None, None]
        imag /= scale[None, :, None, None]
        root_energy = np.sqrt(np.sum(real**2 + imag**2, axis=(0, 2, 3)))
        normalized = real/root_energy[None, :, None, None] + 1j*(imag/root_energy[None, :, None, None])
        cosine, sine = np.zeros(shape), np.zeros(shape)
        for h in range(1, a.shape[3]):
            left, right = normalized[..., h:], normalized[..., :-h]
            with np.errstate(under='ignore'):
                products = left*np.conj(right)
            if np.any((np.abs(left) > 0) & (np.abs(right) > 0) & (products == 0)):
                raise ValueError('A harmonic factor product is below floating-point range')
            coefficient = np.empty(shape[:2], complex)
            for m in range(shape[0]):
                for j in range(shape[1]):
                    terms = products[m, j].ravel()
                    coefficient[m, j] = complex(math.fsum(float(v.real) for v in terms),
                                                 math.fsum(float(v.imag) for v in terms))
            cosine[..., h-1] = 2*coefficient.real
            sine[..., h-1] = -2*coefficient.imag
        for name, value in [('normalized_factors', normalized), ('cosine', cosine), ('sine', sine),
                            ('_amplitude', np.hypot(cosine, sine)),
                            ('frequency_hz', self.frequency_hz),
                            ('phase_variance_rate_rad2_s', self.phase_variance_rate_rad2_s)]:
            object.__setattr__(self, name, _immutable(value))
        # These private coordinates belong only to the old L1 evaluator.
        for name in ['_baseline', '_half_cosine', '_half_sine']:
            object.__delattr__(self, name)

    @property
    def _resident_bytes(self):
        return sum(a.nbytes for a in [self.normalized_factors, self.cosine, self.sine,
                                     self._amplitude, self.frequency_hz, self.phase_variance_rate_rad2_s])

    def _metadata(self, dose_unit):
        meta = super()._metadata(dose_unit)
        digest = hashlib.sha256(VERSION.encode())
        digest.update(meta['parameter_sha256'].encode())
        digest.update(json.dumps(self.normalized_factors.shape).encode())
        flat = self.normalized_factors.ravel()
        for first in range(0, len(flat), 65536):
            block = flat[first:first+65536].astype('<c16', copy=False)
            digest.update(memoryview(block).cast('B'))
        meta.update(version=VERSION, parameter_sha256=digest.hexdigest(),
                    positivity_representation='Sum of squared complex polynomials; unit stationary mean',
                    factor_rows=self.normalized_factors.shape[2], degree=self.normalized_factors.shape[3]-1,
                    independent_phase_contributions_not_arbitrary_multivariate_coupling=True,
                    no_harmonic_L1_sufficient_bound=True,
                    no_count_or_source_floor=True,
                    quadrant_centred_Horner_near_exact_quadrant_roots=True)
        return meta

    def _accumulate(self, phases, *, rate=None, dose=None, max_bytes):
        n = len(phases)
        history = rate is None
        output_bytes = 24*n*self.channels if history else 64*self.channels
        fixed = self._resident_bytes + 96*n + output_bytes
        available = _budget(max_bytes)-fixed
        if available < 4096:
            raise ValueError('Spectral-factor evaluation exceeds the working-memory budget')
        result = np.zeros((n, self.channels) if history else self.channels)
        # Bounds include shifted coefficients, complex Horner temporaries,
        # selection copies, positive norms and scaled-sum work arrays.
        width = min(self.channels, 64)
        rows = min(self.normalized_factors.shape[2], 8)
        while 128*width*rows*self.normalized_factors.shape[3]+256*width*rows > available:
            if width > 1:
                width = max(1, width//2)
            elif rows > 1:
                rows = max(1, rows//2)
            else:
                raise ValueError('Spectral-factor polynomial workspace exceeds the working-memory budget')
        coefficient_bytes = 128*width*rows*self.normalized_factors.shape[3]
        pulses = min(8192, max(1, (available-coefficient_bytes)//(256*width*rows)))
        root_dose = None if history else np.sqrt(dose)
        root_rate = None if history else np.sqrt(rate)
        for first_channel in range(0, self.channels, width):
            channels = slice(first_channel, min(self.channels, first_channel+width))
            for mode in range(len(self.frequency_hz)):
                for first_row in range(0, self.normalized_factors.shape[2], rows):
                    factor = self.normalized_factors[mode, channels, first_row:first_row+rows]
                    shifted = _shifted(factor)
                    for first in range(0, n, pulses):
                        ix = slice(first, min(n, first+pulses))
                        value = _evaluate(factor, shifted, phases[ix, mode])
                        if not history:
                            # Scale each real/imaginary amplitude BEFORE taking
                            # a norm or square. Tiny positive means can survive
                            # large rates at a strong component's exact zero.
                            scale = root_dose[ix, None]*root_rate[None, channels]
                            value.real *= scale[..., None]
                            value.imag *= scale[..., None]
                        amplitude = np.hypot(value.real, value.imag)
                        if history:
                            component = np.hypot.reduce(amplitude, axis=2)
                            result[ix, channels] = np.hypot(result[ix, channels], component)
                        else:
                            component = _norm_over_pulses_and_rows(amplitude)
                            result[channels] = np.hypot(result[channels], component)
        return _readonly(_squared(result))

    def multipliers(self, phases_rad, *, max_bytes=256*1024**2):
        return self._accumulate(self._phases(phases_rad), max_bytes=max_bytes)

    def integrated_mean(self, rate, dose, phases_rad, *, max_bytes=256*1024**2):
        phases = self._phases(phases_rad)
        rate, dose, _ = self._rate_dose(rate, dose, len(phases))
        return self._accumulate(phases, rate=rate, dose=dose, max_bytes=max_bytes)

    def moments(self, rate, dose, times_s, *, dose_unit, max_bytes=256*1024**2):
        # The autocorrelation coefficients give the same exact stationary
        # harmonic covariance kernel, including signed cross-channel loadings.
        return super().moments(rate, dose, times_s, dose_unit=dose_unit,
                               max_bytes=_budget(max_bytes)-self._resident_bytes)

    def sample_phases(self, times_s, *, seed, max_bytes=256*1024**2):
        return super().sample_phases(times_s, seed=seed,
                                     max_bytes=_budget(max_bytes)-self._resident_bytes)

    def draw_nested(self, rate, dose, times_s, stop_indices, *, seed, dose_unit, max_bytes=256*1024**2):
        return super().draw_nested(rate, dose, times_s, stop_indices, seed=seed, dose_unit=dose_unit,
                                   max_bytes=_budget(max_bytes)-self._resident_bytes)
