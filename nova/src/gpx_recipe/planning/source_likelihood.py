"""Positive warm coordinates for a detector ON/OFF spectral calibration.

The likelihood profiles a nonnegative background separately in every cell.
Source coefficients and relative group gains share a gauge, fixed here by one
unit gain. These coordinates initialize an original nonnegative-domain solve;
their log bounds must not be interpreted as a final positive source floor.
No detector efficiency, physical source, stationarity or coverage is qualified.
"""
import numpy as np
from scipy import sparse

from .joint_calibration import _profile_background_mean, _count_loss

VERSION = 'powgen-detector-source-log-likelihood/1'


class DetectorSourceLogLikelihood:
    def __init__(self, on_counts, off_counts, design, on_charge, off_charge,
                 groups, initial_coefficients, initial_gains):
        n, m = np.asarray(on_counts), np.asarray(off_counts)
        group = np.asarray(groups)
        c, gain = np.asarray(initial_coefficients), np.asarray(initial_gains)
        a = sparse.csr_matrix(design, dtype=float, copy=True)
        if (n.ndim != 1 or not 1 <= len(n) <= 1000000 or m.shape != n.shape
                or group.shape != n.shape or group.dtype.kind not in 'iu'
                or c.ndim != 1 or not 1 <= len(c) <= 4096
                or gain.ndim != 1 or not 1 <= len(gain) <= 1024
                or a.shape != (len(n), len(c)) or a.nnz > 20000000
                or any(v.dtype.kind not in 'iuf' or not np.isfinite(v).all() for v in (n, m, c, gain))
                or np.any(n < 0) or np.any(m < 0) or np.any(n > 1e9) or np.any(m > 1e9)
                or not np.array_equal(n, np.rint(n)) or not np.array_equal(m, np.rint(m))
                or np.any(c <= 0) or np.any(gain <= 0) or np.any(group >= len(gain))
                or np.any(group < 0) or not np.isfinite(a.data).all() or np.any(a.data < 0)):
            raise ValueError('Bounded integer ON/OFF observations, positive starts and nonnegative source design required')
        doses = np.asarray([on_charge, off_charge], float)
        if not np.isfinite(doses).all() or np.any(doses <= 0):
            raise ValueError('Positive finite ON/OFF charge required')
        self.n, self.m = np.array(n, float, copy=True), np.array(m, float, copy=True)
        self.groups = np.array(group, int, copy=True)
        self.design = a * doses[0]
        self.ratio = doses[1] / doses[0]
        self.normalizer = float(self.n.sum() + self.m.sum())
        self.source_size, self.gain_size = len(c), len(gain)
        self.anchor = int(np.argmax(gain))
        self.free_gain = np.arange(len(gain)) != self.anchor
        adjusted_c = np.asarray(c, float) * float(gain[self.anchor])
        adjusted_gain = np.asarray(gain, float) / float(gain[self.anchor])
        self.scale = float(np.median(adjusted_c))
        if (self.normalizer <= 0 or not np.isfinite(self.ratio) or self.ratio <= 0
                or not np.isfinite(self.design.data).all()
                or not np.isfinite(adjusted_c).all() or not np.isfinite(self.scale) or self.scale <= 0
                or np.any(adjusted_gain <= 0)):
            raise ValueError('Counts or scaled initial coordinates exceed their finite domain')
        self.initial = np.r_[np.log(adjusted_c / self.scale), np.log(adjusted_gain[self.free_gain])]
        if not np.isfinite(self.initial).all():
            raise ValueError('Initial log coordinates are not finite')

    def decode(self, coordinates):
        x = np.asarray(coordinates, float)
        if (x.shape != (self.source_size + self.gain_size - 1,)
                or not np.isfinite(x).all() or np.any(abs(x) > 100)):
            raise ValueError('Finite bounded log-source/gain coordinates required')
        gain = np.ones(self.gain_size)
        gain[self.free_gain] = np.exp(x[self.source_size:])
        coefficients = self.scale * np.exp(x[:self.source_size])
        if not np.isfinite(coefficients).all():
            raise ValueError('Source coordinates overflow')
        return coefficients, gain

    def value_gradient(self, coordinates):
        c, gain = self.decode(coordinates)
        source = gain[self.groups] * (self.design @ c)
        background = _profile_background_mean(self.n, self.m, source, self.ratio)
        mean = source + background
        value = (_count_loss(self.n, mean) + _count_loss(self.m, self.ratio * background)) / self.normalizer
        residual = 1 - np.divide(self.n, mean, out=np.zeros_like(mean), where=mean > 0)
        dc = c * np.asarray(self.design.T @ (gain[self.groups] * residual)) / self.normalizer
        dg = np.bincount(self.groups, weights=source * residual, minlength=self.gain_size) / self.normalizer
        gradient = np.r_[dc, dg[self.free_gain]]
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise ValueError('Profiled count likelihood exceeds its finite domain')
        return float(value), gradient
