"""Measured ON/OFF reference -> joint cumulative, integer-count predictions.

The reference observations are fixed. One joint draw of their uncertain true
source/blank rates is reused across the campaign. Independent Poisson increments
are conditional on that draw. Predictive covariance retains the reference-cell
correlations, including source/blank anticorrelation; it is not N independent
Gaussian perturbations of a reduced curve. Physical stationarity is an explicit
model assumption, not established by the count arithmetic.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse


@dataclass
class CountLook:
    exposure: float
    counts: np.ndarray
    increment: np.ndarray
    expected_counts: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    counting_variance: np.ndarray
    reference_variance: np.ndarray


class ReferenceCountCampaign:
    def __init__(self, posterior, source_map, background_map, normalization, seed=0, *,
                 independent_background_counts=None, independent_background_charge=None,
                 independent_background_map=None, reduction_background=None):
        self.posterior = posterior
        self.source_map, self.background_map = posterior._weights(source_map, background_map)
        self.normalization = np.asarray(normalization, float)
        self.rate, self.rate_variance = posterior.linear_moments(self.source_map, self.background_map)
        if (self.normalization.shape != self.rate.shape or not np.isfinite(self.normalization).all()
                or np.any(self.normalization <= 0)):
            raise ValueError('Requested bins need measured positive reference coverage')
        self.background = self.background_map @ posterior.background_mean.ravel()
        seeds = np.random.SeedSequence(seed).spawn(3)
        source, blank = posterior.sample_rates(1, seed=seeds[0])
        self.sampled_rate = self.source_map @ source.ravel() + self.background_map @ blank.ravel()
        self.extra_counts = None
        if independent_background_counts is not None:
            n=np.asarray(independent_background_counts)
            q=float(independent_background_charge)
            matrix=sparse.csr_matrix(independent_background_map,dtype=float)
            if (n.ndim!=1 or n.dtype.kind not in 'iuf' or n.size>100000 or not np.isfinite(n).all()
                    or np.any(n<0) or np.any(n>1e9) or not np.array_equal(n,np.rint(n))
                    or not np.isfinite(q) or q<=0 or matrix.shape!=(len(self.rate),len(n))
                    or matrix.nnz>2000000 or not np.isfinite(matrix.data).all() or np.any(matrix.data<0)):
                raise ValueError('Independent background needs bounded integer counts, charge and a positive map')
            self.extra_counts,self.extra_charge,self.extra_map=n.astype(np.int64),q,matrix
            self.extra_mean,self.extra_variance=(n+1)/q,(n+1)/q**2
            self.background += matrix@self.extra_mean
            self.rate += matrix@self.extra_mean
            self.rate_variance += matrix.multiply(matrix)@self.extra_variance
            self.sampled_rate += matrix@np.random.default_rng(seeds[2]).gamma(n+1,1/q)
        # The latent raw background generates counts. The fixed, observed
        # subtraction can be a different estimate (e.g. facility FFT smoothing).
        # It is not an additional independent draw of calibration noise.
        self.subtraction = np.asarray(self.background if reduction_background is None else reduction_background, float).copy()
        if self.subtraction.shape != self.rate.shape or not np.isfinite(self.subtraction).all():
            raise ValueError('Reduced background must be a finite rate on the count grid')
        self.rng = np.random.default_rng(seeds[1])
        self.counts = np.zeros(self.rate.shape, dtype=np.int64)
        self.exposure = 0.

    @property
    def expected(self):
        return (self.rate - self.subtraction) / self.normalization

    def advance(self, charge_uah):
        q = float(charge_uah)
        if not np.isfinite(q) or q <= self.exposure:
            raise ValueError('Cumulative charge must increase and be finite')
        increment = self.rng.poisson((q - self.exposure) * self.sampled_rate)
        self.counts += increment
        self.exposure = q
        # Forecast moments conditional on the measured reference, with no
        # low-count pseudo-events or variance floor in the likelihood.
        counting = self.rate / q / self.normalization**2
        calibration = self.rate_variance / self.normalization**2
        y = (self.counts / q - self.subtraction) / self.normalization
        return CountLook(q, self.counts.copy(), increment, q*self.rate, y,
                         np.sqrt(counting + calibration), counting, calibration)

    def projected_covariance(self, weights, first_charge, second_charge=None):
        """Covariance of declared reduced-pattern functionals, including looks."""
        w = sparse.csr_matrix(weights, dtype=float)
        if w.shape[1] != len(self.rate) or w.shape[0] > 256 or not np.isfinite(w.data).all():
            raise ValueError('Supply at most 256 finite pattern functionals')
        q1, q2 = float(first_charge), float(first_charge if second_charge is None else second_charge)
        if not np.isfinite(q1+q2) or min(q1, q2) <= 0:
            raise ValueError('Positive finite charge required')
        z = w.multiply(1/self.normalization)
        # The cross-look Poisson term is min(q1,q2)/(q1*q2).
        result = (z.multiply(self.rate / max(q1, q2)) @ z.T).toarray()
        # Signed contrast functionals are allowed here; the physical count maps
        # themselves remain nonnegative.
        a, b = z @ self.source_map, z @ self.background_map
        p = self.posterior
        result += (a.multiply(p.source_variance.ravel()) @ a.T
                   + b.multiply(p.background_variance.ravel()) @ b.T
                   + a.multiply(p.source_background_covariance.ravel()) @ b.T
                   + b.multiply(p.source_background_covariance.ravel()) @ a.T).toarray()
        if self.extra_counts is not None:
            extra=z@self.extra_map
            result+=(extra.multiply(self.extra_variance)@extra.T).toarray()
        return result

    def reference_fields(self):
        p = self.posterior
        fields = dict(model_type='linear_on_off_reference',on_counts=p.on_counts, off_counts=p.off_counts,
            on_charge_uah=p.on_exposure, off_charge_uah=p.off_exposure,
            source_mean=p.source_mean, background_mean=p.background_mean,
            source_variance=p.source_variance, background_variance=p.background_variance,
            source_background_covariance=p.source_background_covariance,
            normalization=self.normalization, predicted_rate=self.rate,
            raw_background_mean=self.background, reduction_background=self.subtraction,
            predicted_rate_variance=self.rate_variance, sampled_rate=self.sampled_rate)
        for name, matrix in [('source_map', self.source_map), ('background_map', self.background_map)]:
            fields.update({name+'_data': matrix.data, name+'_indices': matrix.indices,
                           name+'_indptr': matrix.indptr, name+'_shape': matrix.shape})
        if self.extra_counts is not None:
            fields.update(independent_background_counts=self.extra_counts,
                independent_background_charge_uah=self.extra_charge,
                independent_background_mean=self.extra_mean,independent_background_variance=self.extra_variance,
                independent_background_map_data=self.extra_map.data,independent_background_map_indices=self.extra_map.indices,
                independent_background_map_indptr=self.extra_map.indptr,independent_background_map_shape=self.extra_map.shape)
        return fields

    def save_reference(self, path):
        np.savez_compressed(path, **self.reference_fields())
