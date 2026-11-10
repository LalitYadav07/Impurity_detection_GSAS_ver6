"""Cumulative Poisson acquisition and uncertainty-aware reduced outputs.

Counts, finite blank and finite response-reference counts are different random
variables. Reference noise is drawn once per campaign and stays shared across
all snapshots. Low-count exported errors use a documented +0.5 variance floor;
the integer sufficient statistics remain available for exact likelihoods.
"""
from dataclasses import dataclass
import numpy as np
from .physics import positive


@dataclass
class Snapshot:
    exposure: float
    counts: np.ndarray
    increment: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    diagonal_variance: np.ndarray
    covariance_factors: np.ndarray
    reference_variance: np.ndarray
    monitor_counts: int | None


class Accumulator:
    def __init__(self, rate, response, seed=0, *, background_rate=0.,
                 blank_exposure=None, reference_counts=None, monitor_rate=None,
                 scale_uncertainty_fraction=0., systematic_modes=None):
        self.rate, self.response = np.broadcast_arrays(np.asarray(rate, float), np.asarray(response, float))
        if self.rate.ndim != 1 or not np.isfinite(self.rate).all() or np.any(self.rate < 0):
            raise ValueError('Rate must be a finite nonnegative vector')
        if np.any(self.response <= 0) or not np.isfinite(self.response).all():
            raise ValueError('Response must be finite and positive (mask blind bins first)')
        self.background = np.broadcast_to(np.asarray(background_rate, float), self.rate.shape)
        if np.any(self.background < 0) or not np.isfinite(self.background).all():
            raise ValueError('Background rates must be finite and nonnegative')
        # Independent streams keep sample counts unchanged when adding diagnostics.
        seeds = np.random.SeedSequence(seed).spawn(4)
        self.rng, blank_rng, reference_rng, self.monitor_rng = [np.random.default_rng(s) for s in seeds]
        self.exposure = 0.
        self.counts = np.zeros(self.rate.shape, dtype=np.int64)
        self.monitor = 0
        self.monitor_rate = None if monitor_rate is None else positive(monitor_rate, 'monitor rate')
        self.blank_exposure = None if blank_exposure is None else positive(blank_exposure, 'blank exposure')
        self.blank_counts = None if blank_exposure is None else blank_rng.poisson(self.background*self.blank_exposure)
        self.reference_expected = None
        self.reference_counts = None
        self.response_estimate = self.response.copy()
        if reference_counts is not None:
            self.reference_expected = np.broadcast_to(np.asarray(reference_counts, float), self.rate.shape)
            if np.any(self.reference_expected < 25) or not np.isfinite(self.reference_expected).all():
                raise ValueError('Finite response reference needs >=25 expected counts/bin; rebin or mask otherwise')
            self.reference_counts = reference_rng.poisson(self.reference_expected)
            if np.any(self.reference_counts == 0):
                raise ValueError('Zero response-reference count: mask/rebin this channel')
            self.response_estimate *= self.reference_counts/self.reference_expected
        self.scale_uncertainty_fraction = float(scale_uncertainty_fraction)
        if not 0 <= self.scale_uncertainty_fraction < 1:
            raise ValueError('Scale uncertainty must lie in [0,1)')
        self.systematic_modes = np.zeros((0, len(self.rate))) if systematic_modes is None else np.asarray(systematic_modes, float)
        if self.systematic_modes.ndim != 2 or self.systematic_modes.shape[1] != len(self.rate) or not np.isfinite(self.systematic_modes).all():
            raise ValueError('Systematic covariance modes must be a finite (rank, bins) array')

    def advance(self, cumulative_exposure):
        exposure = positive(cumulative_exposure, 'cumulative exposure')
        if exposure <= self.exposure:
            raise ValueError('Cumulative exposure must increase strictly')
        delta = exposure-self.exposure
        increment = self.rng.poisson((self.rate+self.background)*delta)
        self.counts += increment
        self.exposure = exposure
        norm_exposure = exposure
        if self.monitor_rate is not None:
            self.monitor += int(self.monitor_rng.poisson(self.monitor_rate*delta))
            if self.monitor <= 0:
                raise ValueError('No monitor counts; no normalized snapshot is valid')
            norm_exposure = self.monitor/self.monitor_rate
        divisor = norm_exposure*self.response_estimate
        signal = self.counts.astype(float)
        # Do not subtract blank unless it was actually measured in this scenario.
        blank_var = np.zeros_like(signal)
        if self.blank_counts is not None:
            alpha = norm_exposure/self.blank_exposure
            signal -= alpha*self.blank_counts
            blank_var = alpha**2*(self.blank_counts+.5)/divisor**2
        y = signal/divisor
        diagonal = (self.counts+.5)/divisor**2 + blank_var
        reference_var = blank_var.copy()
        if self.reference_counts is not None:
            ref_var = y**2/self.reference_counts
            diagonal += ref_var
            reference_var += ref_var
        factors = list(self.systematic_modes)
        if self.monitor_rate is not None:
            # Shared denominator induces bin-to-bin covariance.
            # With blank subtraction, only the sample-run term depends on monitor.
            factors.append(self.counts/divisor/np.sqrt(self.monitor))
        if self.scale_uncertainty_fraction:
            factors.append(y*self.scale_uncertainty_fraction)
        factors = np.asarray(factors).reshape(-1, len(y))
        sigma = np.sqrt(diagonal+np.sum(factors**2, axis=0))
        return Snapshot(exposure, self.counts.copy(), increment, y, sigma, diagonal,
                        factors, reference_var, self.monitor if self.monitor_rate else None)

    def state(self):
        """JSON-serializable RNG states plus counts are stored with every look."""
        return dict(exposure=self.exposure, sample_rng=self.rng.bit_generator.state,
                    monitor_rng=self.monitor_rng.bit_generator.state, monitor=self.monitor)
