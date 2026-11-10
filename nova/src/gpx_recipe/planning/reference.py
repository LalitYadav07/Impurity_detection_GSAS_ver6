"""Linear reference smoothing with retained covariance.

This is a numerical counterpart of the declared Mantid 6.16.1 FFTSmooth v2
Butterworth operation, including its legacy boundary extension. The native
operation discards errors; these methods propagate independent input variances
through the same linear map instead. A reference reused between measurements
remains the same random variable. Peak stripping, pulse drift and uncertainty
in the input means require separate models and experimental verification.

Mathematical contract checked against the matched official runtime; see the
synthetic fixture and tests. Source definition: mantidproject/mantid commit
3c606e5cfb5bc90daa9d8a4a577811190671b33d, FFTSmooth2 and Kernel/Smoothing.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np


def _integer(value, label, lower, upper):
    if isinstance(value, bool) or not isinstance(value, Integral) or not lower <= value <= upper:
        raise ValueError(f'{label} must be an integer in [{lower}, {upper}]')
    return int(value)


@dataclass(frozen=True)
class Butterworth:
    """A bin-index smoother; it does not assume equal physical bin widths."""

    bins: int
    truncation: int
    order: int = 2

    def __post_init__(self):
        _integer(self.bins, 'Bin count', 2, 250000)
        _integer(self.truncation, 'Truncation', 2, 1000000)
        _integer(self.order, 'Filter order', 1, 16)

    def _array(self, values, label):
        array = np.asarray(values, dtype=float)
        if array.ndim < 1 or array.shape[-1] != self.bins or not np.isfinite(array).all():
            raise ValueError(f'{label} must be finite with {self.bins} bins on the final axis')
        if array.size > 8000000:
            raise ValueError('Reference array budget exceeded')
        return array

    def _variance(self, variance):
        array = self._array(variance, 'Source variance')
        if array.ndim != 1 or np.any(array < 0):
            raise ValueError('Source variance must be a nonnegative vector')
        return array

    def _transfer(self):
        cutoff = max(1, (self.bins + 1) // self.truncation)
        frequency = np.arange(self.bins + 1, dtype=float)
        return 1. / (1. + (frequency / cutoff) ** (2 * self.order))

    def apply(self, values):
        """Smooth vectors, including both endpoints and the declared extension."""
        array = self._array(values, 'Reference')
        extended = np.concatenate((array[..., -1:], array[..., :0:-1], array), axis=-1)
        transformed = np.fft.rfft(extended, axis=-1) * self._transfer()
        return np.fft.irfft(transformed, n=2 * self.bins, axis=-1)[..., self.bins:]

    def adjoint(self, weights):
        """Pull a reduced-bin linear measurement back to independent inputs."""
        array = self._array(weights, 'Linear measurement weights')
        padded = np.zeros(array.shape[:-1] + (2 * self.bins,))
        padded[..., self.bins:] = array
        filtered = np.fft.irfft(np.fft.rfft(padded, axis=-1) * self._transfer(),
                               n=2 * self.bins, axis=-1)
        result = filtered[..., self.bins:].copy()
        result[..., 1:] += filtered[..., self.bins - 1:0:-1]
        result[..., -1] += filtered[..., 0]
        return result

    def rows(self, indices):
        """Selected rows of the map, with an explicit memory bound."""
        index = np.asarray(indices)
        if (index.ndim != 1 or index.dtype.kind not in 'iu' or
                np.any(index < 0) or np.any(index >= self.bins)):
            raise ValueError('Select valid integer reference-bin indices')
        if len(index) * self.bins > 2000000:
            raise ValueError('Reference matrix row budget exceeded')
        impulse = np.fft.irfft(self._transfer(), n=2 * self.bins)
        row = index[:, None]
        column = np.arange(self.bins)[None, :]
        result = impulse[(row - column) % (2 * self.bins)] + impulse[(row + column) % (2 * self.bins)]
        result[:, 0] -= impulse[index]
        result[:, -1] += impulse[(self.bins + index) % (2 * self.bins)]
        return result

    def marginal_variance(self, source_variance):
        """Exact diagonal for independent inputs; correlations are not zero."""
        variance = self._variance(source_variance)
        if self.bins ** 2 > 100000000:
            raise ValueError('Exact marginal-variance work budget exceeded; use projected_covariance')
        block = min(128, max(1, 2000000 // self.bins))
        result = np.empty(self.bins)
        for start in range(0, self.bins, block):
            stop = min(self.bins, start + block)
            rows = self.rows(np.arange(start, stop))
            result[start:stop] = rows ** 2 @ variance
        return result

    def projected_covariance(self, source_variance, weights, other_weights=None):
        """Covariance of linear measurements sharing one finite reference.

        Rows of weights may describe separate regions or measurement snapshots.
        The caller supplies any ratio derivatives and exposure factors. Reuse
        of a reference is represented by this covariance, not an independent
        redraw or an extra 1/exposure rescaling.
        """
        variance = self._variance(source_variance)
        left = np.atleast_2d(self._array(weights, 'Measurement weights'))
        right = left if other_weights is None else np.atleast_2d(
            self._array(other_weights, 'Other measurement weights'))
        if left.ndim != 2 or right.ndim != 2 or left.shape[0] * right.shape[0] * self.bins > 100000000:
            raise ValueError('Projected covariance work budget exceeded')
        gradient_left = self.adjoint(left)
        gradient_right = gradient_left if other_weights is None else self.adjoint(right)
        return (gradient_left * variance) @ gradient_right.T


def ratio_reference_weights(numerator, denominator, weights, *, scale=1.):
    """First-order derivatives for scale * numerator / finite denominator.

    Return numerator and denominator weights to pass to the relevant covariance
    operators. This is a local delta-method calculation, not an exact ratio
    likelihood or a validated low-count approximation.
    """
    top, bottom = np.asarray(numerator, float), np.asarray(denominator, float)
    if (top.ndim != 1 or top.shape != bottom.shape or not np.isfinite(top).all() or
            not np.isfinite(bottom).all() or np.any(bottom <= 0)):
        raise ValueError('Ratio means must be matching finite vectors with positive denominator')
    measurement = np.asarray(weights, float)
    if measurement.ndim not in (1, 2) or measurement.shape[-1] != len(top) or not np.isfinite(measurement).all():
        raise ValueError('Ratio weights must be finite vectors or rows')
    if isinstance(scale, bool) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Ratio scale must be finite and positive')
    return measurement * (scale / bottom), measurement * (-scale * top / bottom ** 2)
