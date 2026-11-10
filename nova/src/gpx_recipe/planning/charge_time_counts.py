"""Pulse-resolved counts with separate recorded-charge and elapsed-time rates.

The source-linked component follows an explicitly supplied positive phase
law. The independent stationary time component retains its seconds unit.
Neither rate, phase parameters, nor physical cause is inferred by this module.
"""
from dataclasses import dataclass
import numpy as np

from .temporal_counts import (
    PhaseRateLaw, TemporalCountDraw, TemporalCountMoments,
    _budget, _clock, _readonly, _seed,
)

VERSION = 'positive-charge-and-time-counts/1'


@dataclass(frozen=True)
class ChargeTimeCountModel:
    signal_rate_per_uAh: np.ndarray
    time_rate_per_s: np.ndarray
    signal_law: PhaseRateLaw
    provenance: str

    def __post_init__(self):
        if not isinstance(self.signal_law, PhaseRateLaw):
            raise ValueError('An explicit positive source-rate phase law is required')
        if not isinstance(self.provenance, str) or not self.provenance.strip() or len(self.provenance) > 4096:
            raise ValueError('Explicit bounded rate and background provenance is required')
        for name in ['signal_rate_per_uAh', 'time_rate_per_s']:
            r = np.asarray(getattr(self, name))
            if (r.shape != (self.signal_law.channels,) or r.dtype.kind not in 'iuf'
                    or not np.isfinite(r).all() or np.any(r < 0)):
                raise ValueError('Matching finite nonnegative channel rates are required')
            object.__setattr__(self, name, _readonly(r.astype(float)))

    def _exposures(self, charge_uAh, elapsed_s, times_s):
        t = _clock(times_s)
        _, q, _ = self.signal_law._rate_dose(self.signal_rate_per_uAh, charge_uAh, len(t))
        _, dt, _ = self.signal_law._rate_dose(self.time_rate_per_s, elapsed_s, len(t))
        if len(t) > 1 and np.any(dt[:-1] > np.diff(t)*(1+2e-12)):
            raise ValueError('Assigned elapsed exposures must not overlap successive pulse intervals')
        return q, dt, t

    def _metadata(self):
        return dict(version=VERSION, provenance=self.provenance,
            signal_rate_unit='counts/uAh', charge_unit='uAh',
            time_rate_unit='counts/second', elapsed_unit='second',
            time_component_stationary_independent_of_source_phase=True,
            zero_charge_pulses_can_have_time_component_counts=True,
            physical_origin_of_time_component_not_identified=True,
            reference_and_parameter_uncertainty_not_included=True,
            experimental_accuracy_certified=False)

    def conditional_mean(self, charge_uAh, elapsed_s, phases_rad, *, max_bytes=256*1024**2):
        budget = _budget(max_bytes)
        phases = self.signal_law._phases(phases_rad)
        _, q, _ = self.signal_law._rate_dose(self.signal_rate_per_uAh, charge_uAh, len(phases))
        _, dt, elapsed = self.signal_law._rate_dose(self.time_rate_per_s, elapsed_s, len(phases))
        source = self.signal_law.integrated_mean(self.signal_rate_per_uAh, q, phases, max_bytes=budget)
        mean = source+self.time_rate_per_s*elapsed
        if not np.isfinite(mean).all():
            raise ValueError('Combined positive count mean exceeds floating-point range')
        return _readonly(mean)

    def moments(self, charge_uAh, elapsed_s, times_s, *, max_bytes=256*1024**2):
        q, dt, t = self._exposures(charge_uAh, elapsed_s, times_s)
        source = self.signal_law.moments(self.signal_rate_per_uAh, q, t,
            dose_unit='uAh', max_bytes=_budget(max_bytes))
        _, _, elapsed = self.signal_law._rate_dose(self.time_rate_per_s, dt, len(t))
        mean = source.mean+self.time_rate_per_s*elapsed
        if not np.isfinite(mean).all():
            raise ValueError('Combined positive count mean exceeds floating-point range')
        norm = np.hypot.reduce(source.covariance_factors, axis=0) if len(source.covariance_factors) else np.zeros_like(mean)
        if np.any(norm**2 > np.finfo(float).max-mean):
            raise ValueError('Combined count variance exceeds floating-point range')
        metadata = {**source.metadata, **self._metadata(),
            'covariance_factors_source_phase_only': True,
            'time_component_Poisson_variance_included': True}
        return TemporalCountMoments(_readonly(mean), _readonly(mean), source.covariance_factors, metadata)

    def draw_nested(self, charge_uAh, elapsed_s, times_s, stop_indices, *, seed, max_bytes=256*1024**2):
        budget = _budget(max_bytes)
        q, dt, t = self._exposures(charge_uAh, elapsed_s, times_s)
        stops = np.asarray(stop_indices)
        if (stops.ndim != 1 or stops.dtype.kind not in 'iu' or not 1 <= len(stops) <= 4096
                or stops[0] <= 0 or np.any(stops > len(t)) or np.any(np.diff(stops.astype(np.int64)) <= 0)):
            raise ValueError('Strictly increasing positive pulse stop indices are required')
        if 96*len(t)+64*len(stops)*self.signal_law.channels > budget:
            raise ValueError('Nested charge/time acquisition exceeds its working-memory budget')
        phase_stream, count_stream = np.random.SeedSequence(_seed(seed)).spawn(2)
        phase_seed = int(phase_stream.generate_state(1, dtype=np.uint64)[0])
        phases = self.signal_law.sample_phases(t, seed=phase_seed, max_bytes=budget)
        rng = np.random.default_rng(count_stream)
        means, increments, cumulative = [], [], []
        total = np.zeros(self.signal_law.channels, np.int64)
        first = 0
        for last in stops:
            mean = self.conditional_mean(q[first:last], dt[first:last], phases[first:last], max_bytes=budget)
            delta = rng.poisson(mean)
            if np.any(delta > np.iinfo(np.int64).max-total):
                raise ValueError('Cumulative counts exceed int64 range')
            total = total+delta
            means.append(mean); increments.append(delta); cumulative.append(total.copy())
            first = int(last)
        metadata = {**self._metadata(), 'seed': seed, 'phase_seed': phase_seed,
            'one_phase_history_for_all_nested_looks': True,
            'disjoint_conditional_Poisson_increments': True}
        return TemporalCountDraw(_readonly(stops.astype(np.int64)), phases, _readonly(means),
            _readonly(increments), _readonly(cumulative), metadata)
