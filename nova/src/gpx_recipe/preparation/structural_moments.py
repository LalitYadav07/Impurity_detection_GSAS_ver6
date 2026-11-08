"""Owned row summaries that can certify the original normalization bytes.

This is not an uploaded statistics format. The assembler captures these values
from each actual float32 row before its final trajectory partition is known.
Only selected training rows enter a query. If the error enclosure straddles a
float32 rounding boundary, the caller runs the unchanged two-pass calculation.
"""
import numpy as np

MAX_BYTES = 80*1024**2
BYTES_PER_ROW = 80  # nine float64 moments and the original float32 norm
_INF = np.float64(np.inf)


def _down(x):
    return np.nextafter(x, -_INF)


def _up(x):
    return np.nextafter(x, _INF)


def _gamma(count):
    # Use epsilon (twice unit roundoff), with outward-rounded arithmetic.
    # This also conservatively allows fused or unfused multiply/add reductions.
    value = _up(np.float64(count)*np.finfo(np.float64).eps)
    return _up(value/_down(1.-value))


def _add(a, b):
    return _down(a[0]+b[0]), _up(a[1]+b[1])


def _mul(a, b):
    products = np.array([a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]])
    return _down(products.min(axis=0)), _up(products.max(axis=0))


def _square(a):
    low = np.minimum(a[0]*a[0], a[1]*a[1])
    low = np.where((a[0] <= 0)&(a[1] >= 0), 0., _down(low))
    return low, _up(np.maximum(a[0]*a[0], a[1]*a[1]))


def _rounding_supported():
    """Require binary64, gradual underflow and round-to-nearest/ties-even.

    Inputs are float32. Their sums, differences and squares stay in binary64's
    normal range at our bounded row/grid sizes, including float32 subnormals.
    Test the running arithmetic, not an assumed machine or BLAS vendor.
    """
    one = np.array([1.], np.float64)
    half = np.array([2.**-53], np.float64)
    tiny = np.array([np.finfo(np.float64).tiny], np.float64)
    tiny32 = np.array([np.finfo(np.float32).tiny], np.float32)
    return (np.finfo(np.float64).nmant == 52
            and (one+half)[0] == 1.
            and (one+3.*half)[0] == 1.+2.**-51
            and (tiny*.5)[0] > 0. and (tiny32*np.float32(.5))[0] > 0.)


class Moments:
    """An internal bounded sink, sealed before any training query."""
    def __init__(self, count, bins):
        if (type(count) is not int or not 1 <= count or count*BYTES_PER_ROW > MAX_BYTES
                or type(bins) is not int or not 64 <= bins <= 8192):
            raise ValueError('Owned structural moments exceed their bounded inventory')
        self.bins = bins; self._rows = 0; self._sealed = False
        self._values = np.empty((count, 10), np.float64)
        self._supported = _rounding_supported()

    def append(self, feature):
        if (self._sealed or self._rows >= len(self._values) or feature.dtype != np.float32
                or feature.shape != (3, self.bins) or not np.isfinite(feature).all()):
            raise ValueError('Capture complete finite owned three-curve summaries')
        value = feature.astype(np.float64)
        row = self._values[self._rows]
        row[:3] = value.sum(axis=1)
        row[3:6] = np.einsum('ij,ij->i', value, value)
        row[6:9] = np.abs(value).sum(axis=1)
        # Exactly the same contiguous 2D axis reduction as _representatives.
        with np.errstate(over='ignore'):
            row[9] = np.linalg.norm(feature[None, 2], axis=1)[0]
        self._rows += 1

    def seal(self):
        if self._sealed:
            raise ValueError('Owned structural moments are already sealed')
        # Immutable backing bytes; sealing has at most twice MAX_BYTES peak
        # storage, and no summaries are exported with the prepared library.
        self._values = np.frombuffer(self._values[:self._rows].tobytes(), np.float64).reshape(-1, 10)
        self._sealed = True

    @property
    def nbytes(self):
        return self._values.nbytes

    def _indices(self, indices):
        indices = np.asarray(indices)
        if (not self._sealed or indices.ndim != 1 or indices.dtype.kind not in 'iu'
                or not len(indices) or len(indices)*BYTES_PER_ROW > MAX_BYTES
                or np.any(indices < 0) or np.any(indices >= len(self._values))):
            raise ValueError('Select bounded owned structural summary rows')
        return indices

    def strengths(self, indices):
        if not self._supported or not _rounding_supported():
            return None
        return self._values[self._indices(indices), 9]

    def statistics(self, indices, *, minimum_scale, center):
        selected = self._values[self._indices(indices)]
        if not self._supported or not _rounding_supported():
            return None
        count = len(selected)*self.bins
        batches = (len(selected)+63)//64
        # A summary has one within-row and one across-row reduction. Allow
        # twice their scalar operation counts for all sums and dot products.
        summary_error = _gamma(2*self.bins+2*len(selected)+16)
        total, square, absolute = np.split(selected[:, :9].sum(axis=0), 3)
        absolute = _up(absolute/_down(1.-summary_error))
        s_error = _up(summary_error*absolute)
        s = _down(total-s_error), _up(total+s_error)
        r = (_down(square/_up(1.+summary_error)),
             _up(square/_down(1.-summary_error)))
        # Enclose the old first pass: arbitrary within-batch reduction order,
        # then ordered additions of batch totals, then one division.
        old_sum_error = _gamma(64*self.bins+batches+16)
        e = _up(old_sum_error*absolute)
        mean = _down(_down(s[0]-e)/count), _up(_up(s[1]+e)/count)
        mean = _down(mean[0]), _up(mean[1])  # division's last rounding

        # For that same old mean, the exact centered sum is R-2*m*S+N*m*m.
        # Interval products retain cancellation uncertainty instead of calling
        # nearly constant channels precise. Those cases use the original path.
        cross = _mul((-2.*mean[1], -2.*mean[0]), s)
        centered = _add(_add(r, cross), _mul((np.float64(count),)*2, _square(mean)))
        # Subtraction, squaring, einsum and batch additions in the original
        # second pass are enclosed by this conservative operation bound.
        old_square_error = _gamma(4*64*self.bins+4*batches+32)
        low = np.maximum(0., _down(centered[0]*_down(1.-old_square_error)))
        high = np.maximum(0., _up(centered[1]*_up(1.+old_square_error)))
        low = np.maximum(0., _down(low/count)); high = _up(high/count)
        low = np.maximum(_down(np.sqrt(low)), minimum_scale)
        high = np.maximum(_up(np.sqrt(high)), minimum_scale)
        scale_low, scale_high = low.astype(np.float32), high.astype(np.float32)
        mean_low, mean_high = mean[0].astype(np.float32), mean[1].astype(np.float32)
        if (scale_low.tobytes() != scale_high.tobytes()
                or (center and mean_low.tobytes() != mean_high.tobytes())):
            return None
        result_mean = mean_low if center else np.zeros(3, np.float32)
        return result_mean[:, None], scale_low[:, None]
