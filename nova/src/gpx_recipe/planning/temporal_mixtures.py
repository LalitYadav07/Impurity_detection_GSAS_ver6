"""Positive component counts and joint parameter ensembles.

Component intensities share an explicitly declared phase basis. They remain
separate during positive pulse integration: collapsing a tiny stationary
component into normalized harmonic coefficients can erase its intensity near
an exact zero of a much larger modulated component.

An ensemble represents caller-supplied joint parameter uncertainty. It does
not estimate a posterior, establish its completeness, or qualify POWGEN data.
One parameter state and one shared phase history persist across nested looks.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np

from .temporal_counts import (
    PhaseRateLaw, TemporalCountDraw, TemporalCountMoments,
    _budget, _clock, _readonly, _seed,
)

VERSION = 'component-parameter-count-mixtures/1'


def _provenance(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError('Explicit bounded parameter/component provenance is required')
    return value


def _sum_vectors(values):
    result = np.array([math.fsum(float(v[j]) for v in values) for j in range(len(values[0]))])
    if not np.isfinite(result).all():
        raise ValueError('Combined count means exceed floating-point range')
    return result


def _factor_keys(law):
    return [(m, h, trig) for m in range(len(law.frequency_hz))
            for h in range(law.cosine.shape[2])
            if np.any(law._amplitude[m, :, h]) for trig in ['cosine', 'sine']]


def _valid_moments(mean, factors):
    norm = np.hypot.reduce(factors, axis=0) if len(factors) else np.zeros_like(mean)
    if (not np.isfinite(mean).all() or not np.isfinite(factors).all()
            or np.any(norm > math.sqrt(np.finfo(float).max))
            or np.any(norm ** 2 > np.finfo(float).max - mean)):
        raise ValueError('Combined count moments exceed floating-point range')


@dataclass(frozen=True)
class CountRateState:
    rate: np.ndarray
    law: PhaseRateLaw
    dose_unit: str
    provenance: str

    def __post_init__(self):
        if not isinstance(self.law, PhaseRateLaw):
            raise ValueError('An explicit positive phase-rate law is required')
        r = np.asarray(self.rate)
        if (r.shape != (self.law.channels,) or r.dtype.kind not in 'iuf'
                or not np.isfinite(r).all() or np.any(r < 0)):
            raise ValueError('Matching finite nonnegative channel rates are required')
        self.law._metadata(self.dose_unit)
        _provenance(self.provenance)
        object.__setattr__(self, 'rate', _readonly(r.astype(float)))

    @property
    def channels(self):
        return self.law.channels

    @property
    def factor_rank(self):
        return len(_factor_keys(self.law))

    @property
    def parameter_sha256(self):
        digest = hashlib.sha256(VERSION.encode())
        digest.update(self.rate.astype('<f8').tobytes())
        digest.update(self.law._metadata(self.dose_unit)['parameter_sha256'].encode())
        digest.update(self.dose_unit.encode())
        return digest.hexdigest()

    def integrated_mean(self, dose, phases_rad, *, max_bytes=256 * 1024**2):
        return self.law.integrated_mean(self.rate, dose, phases_rad, max_bytes=max_bytes)

    def moments(self, dose, times_s, *, max_bytes=256 * 1024**2):
        value = self.law.moments(self.rate, dose, times_s, dose_unit=self.dose_unit, max_bytes=max_bytes)
        metadata = {**value.metadata, 'rate_state_provenance': self.provenance,
                    'rate_state_parameter_sha256': self.parameter_sha256}
        return TemporalCountMoments(value.mean, value.poisson_variance, value.covariance_factors, metadata)

    def draw_nested(self, dose, times_s, stop_indices, *, seed, max_bytes=256 * 1024**2):
        return self.law.draw_nested(self.rate, dose, times_s, stop_indices, seed=seed,
                                   dose_unit=self.dose_unit, max_bytes=max_bytes)


@dataclass(frozen=True)
class SharedPhaseComponents:
    components: tuple
    phase_basis_id: str
    provenance: str

    def __post_init__(self):
        components = tuple(self.components)
        if not 1 <= len(components) <= 32 or not all(isinstance(c, CountRateState) for c in components):
            raise ValueError('One to 32 explicit component rate states are required')
        _provenance(self.phase_basis_id)
        _provenance(self.provenance)
        first = components[0]
        for component in components[1:]:
            if (component.dose_unit != first.dose_unit or component.channels != first.channels
                    or component.law.cosine.shape != first.law.cosine.shape
                    or not np.array_equal(component.law.frequency_hz, first.law.frequency_hz)
                    or not np.array_equal(component.law.phase_variance_rate_rad2_s, first.law.phase_variance_rate_rad2_s)):
                raise ValueError('Shared phase components require the same units, channels and explicit phase basis')
        object.__setattr__(self, 'components', components)

    @property
    def channels(self):
        return self.components[0].channels

    @property
    def dose_unit(self):
        return self.components[0].dose_unit

    @property
    def factor_rank(self):
        return len(set(key for c in self.components for key in _factor_keys(c.law)))

    @property
    def parameter_sha256(self):
        value = [VERSION, self.phase_basis_id, [c.parameter_sha256 for c in self.components]]
        return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()

    def _metadata(self):
        return dict(version=VERSION, parameter_sha256=self.parameter_sha256,
                    provenance=self.provenance, component_provenance=[c.provenance for c in self.components],
                    phase_basis_id=self.phase_basis_id, components=len(self.components),
                    dose_unit=self.dose_unit, rate_unit='counts/' + self.dose_unit,
                    shared_phase_cross_component_covariance_retained=True,
                    component_intensities_not_collapsed_to_normalized_coefficients=True,
                    calibration_reference_and_parameter_uncertainty_not_included=True,
                    observed_block_total_conditioning_not_included=True,
                    distinct_count_channels_required_not_overlapping_event_views=True,
                    physical_component_calibration_and_noise_cause_not_established=True,
                    experimental_accuracy_certified=False)

    def integrated_mean(self, dose, phases_rad, *, max_bytes=256 * 1024**2):
        budget = _budget(max_bytes)
        phases = self.components[0].law._phases(phases_rad)
        if 128 * len(phases) + (32 * len(self.components) + 64) * self.channels > budget:
            raise ValueError('Component integration exceeds the working-memory budget')
        values = [c.integrated_mean(dose, phases, max_bytes=budget) for c in self.components]
        try:
            result = _sum_vectors(values)
        except OverflowError as exc:
            raise ValueError('Combined count means exceed floating-point range') from exc
        return _readonly(result)

    def moments(self, dose, times_s, *, max_bytes=256 * 1024**2):
        t = _clock(times_s)
        budget = _budget(max_bytes)
        keys = sorted(set(key for c in self.components for key in _factor_keys(c.law)))
        if 96 * len(t) + (80 * len(keys) + 48 * len(self.components) + 128) * self.channels > budget:
            raise ValueError('Shared component moments exceed the working-memory budget')
        mapping = {key: index for index, key in enumerate(keys)}
        factors = np.zeros((len(keys), self.channels))
        compensation = np.zeros_like(factors)
        means = []
        for component in self.components:
            value = component.moments(dose, t, max_bytes=budget)
            means.append(value.mean)
            for row, key in zip(value.covariance_factors, _factor_keys(component.law)):
                index = mapping[key]
                delta = row - compensation[index]
                combined = factors[index] + delta
                compensation[index] = (combined - factors[index]) - delta
                factors[index] = combined
        try:
            mean = _sum_vectors(means)
        except OverflowError as exc:
            raise ValueError('Combined count means exceed floating-point range') from exc
        _valid_moments(mean, factors)
        metadata = self._metadata()
        metadata.update(covariance_rank=len(factors), independent_conditional_Poisson_components=True,
                        aligned_phase_factors_summed_before_covariance=True)
        return TemporalCountMoments(_readonly(mean), _readonly(mean), _readonly(factors), metadata)

    def draw_nested(self, dose, times_s, stop_indices, *, seed, max_bytes=256 * 1024**2):
        t = _clock(times_s)
        stops = np.asarray(stop_indices)
        if (stops.ndim != 1 or stops.dtype.kind not in 'iu' or not 1 <= len(stops) <= 4096
                or stops[0] <= 0 or np.any(stops > len(t)) or np.any(np.diff(stops.astype(np.int64)) <= 0)):
            raise ValueError('Strictly increasing positive pulse stop indices are required')
        budget = _budget(max_bytes)
        if (64 * len(stops) * self.channels + len(t) * (48 * len(self.components[0].law.frequency_hz) + 128)
                + 32 * len(self.components) * self.channels > budget):
            raise ValueError('Nested component acquisition exceeds the working-memory budget')
        # Validate the complete supplied dose, including pulses beyond the last look.
        for component in self.components:
            component.law._rate_dose(component.rate, dose, len(t))
        q = np.asarray(dose, dtype=float)
        phase_stream, count_stream = np.random.SeedSequence(_seed(seed)).spawn(2)
        phase_seed = int(phase_stream.generate_state(1, dtype=np.uint64)[0])
        phases = self.components[0].law.sample_phases(t, seed=phase_seed, max_bytes=budget)
        rng = np.random.default_rng(count_stream)
        means, increments, cumulative = [], [], []
        total = np.zeros(self.channels, np.int64)
        first = 0
        for last in stops:
            mean = self.integrated_mean(q[first:last], phases[first:last], max_bytes=budget)
            delta = rng.poisson(mean)
            if np.any(delta > np.iinfo(np.int64).max - total):
                raise ValueError('Cumulative counts exceed int64 range')
            total = total + delta
            means.append(mean)
            increments.append(delta)
            cumulative.append(total.copy())
            first = int(last)
        metadata = self._metadata()
        metadata.update(seed=seed, phase_seed=phase_seed, shot_stream_spawn_index=1,
                        one_shared_phase_history_across_components_and_looks=True,
                        disjoint_conditional_Poisson_increments=True)
        return TemporalCountDraw(_readonly(stops.astype(np.int64)), phases, _readonly(means),
                                 _readonly(increments), _readonly(cumulative), metadata)


@dataclass(frozen=True)
class EnsembleCountDraw:
    parameter_state_index: int
    draw: TemporalCountDraw
    metadata: dict


@dataclass(frozen=True)
class CountParameterEnsemble:
    states: tuple
    weights: np.ndarray
    provenance: str

    def __post_init__(self):
        states = tuple(self.states)
        if (not 1 <= len(states) <= 4096
                or not all(isinstance(s, (CountRateState, SharedPhaseComponents)) for s in states)):
            raise ValueError('One to 4096 explicit joint count parameter states are required')
        first = states[0]
        if any(s.channels != first.channels or s.dose_unit != first.dose_unit for s in states):
            raise ValueError('Parameter states require matching distinct channels and dose units')
        weights = np.asarray(self.weights)
        if (weights.shape != (len(states),) or weights.dtype.kind not in 'iuf'
                or not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0)):
            raise ValueError('Finite nonnegative relative ensemble weights are required')
        positive = weights > 0
        scaled = weights.astype(float) / float(weights.max())
        weights = scaled / math.fsum(scaled)
        if np.any(positive & (weights == 0)):
            raise ValueError('A positive ensemble weight is below the finite normalized domain')
        _provenance(self.provenance)
        object.__setattr__(self, 'states', states)
        object.__setattr__(self, 'weights', _readonly(weights))

    @property
    def channels(self):
        return self.states[0].channels

    def _metadata(self):
        state_hashes = [s.parameter_sha256 for s in self.states]
        value = [VERSION, state_hashes, self.weights.tolist()]
        return dict(version=VERSION, provenance=self.provenance,
                    parameter_sha256=hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest(),
                    parameter_state_sha256=state_hashes, normalized_relative_weights=self.weights.tolist(),
                    dose_unit=self.states[0].dose_unit, rate_unit='counts/' + self.states[0].dose_unit,
                    caller_supplied_joint_parameter_ensemble_integrated=True,
                    supplied_ensemble_is_not_an_estimated_or_qualified_posterior=True,
                    completeness_of_reference_parameter_or_model_uncertainty_not_established=True,
                    shared_between_channel_parameter_covariance_retained=True,
                    observed_block_total_conditioning_not_included=True,
                    distinct_count_channels_required_not_overlapping_event_views=True,
                    experimental_accuracy_certified=False)

    def moments(self, dose, times_s, *, max_bytes=256 * 1024**2):
        t = _clock(times_s)
        budget = _budget(max_bytes)
        active = np.flatnonzero(self.weights > 0)
        rank = sum(self.states[i].factor_rank + 1 for i in active)
        if 128 * len(t) + (112 * rank + 48 * len(active) + 128) * self.channels > budget:
            raise ValueError('Joint parameter-ensemble moments exceed the working-memory budget')
        values = [self.states[i].moments(dose, t, max_bytes=budget) for i in active]
        weights = self.weights[active]
        weighted_means = [w * v.mean for w, v in zip(weights, values)]
        if any(np.any((v.mean > 0) & (weighted == 0)) for v, weighted in zip(values, weighted_means)):
            raise ValueError('A positive weighted count mean is below floating-point range')
        try:
            mean = _sum_vectors(weighted_means)
        except OverflowError as exc:
            raise ValueError('Combined count means exceed floating-point range') from exc
        factors = []
        for weight, value in zip(weights, values):
            root = math.sqrt(float(weight))
            factors.extend(root * value.covariance_factors)
            factors.append(root * (value.mean - mean))
        factors = np.asarray(factors).reshape(-1, self.channels)
        _valid_moments(mean, factors)
        metadata = self._metadata()
        metadata.update(covariance_rank=len(factors), total_covariance_identity_used=True,
                        within_state_phase_and_between_state_mean_factors_retained=True,
                        poisson_variance_equals_ensemble_mean=True)
        return TemporalCountMoments(_readonly(mean), _readonly(mean), _readonly(factors), metadata)

    def draw_nested(self, dose, times_s, stop_indices, *, seed, max_bytes=256 * 1024**2):
        _budget(max_bytes)
        parameter_stream, acquisition_stream = np.random.SeedSequence(_seed(seed)).spawn(2)
        rng = np.random.default_rng(parameter_stream)
        index = int(rng.choice(len(self.states), p=self.weights))
        acquisition_seed = int(acquisition_stream.generate_state(1, dtype=np.uint64)[0])
        draw = self.states[index].draw_nested(dose, times_s, stop_indices,
                                             seed=acquisition_seed, max_bytes=max_bytes)
        metadata = self._metadata()
        metadata.update(seed=seed, acquisition_seed=acquisition_seed,
                        parameter_state_index=index, one_joint_parameter_state_for_all_looks=True,
                        parameter_stream_spawn_index=0, acquisition_stream_spawn_index=1)
        return EnsembleCountDraw(index, draw, metadata)
