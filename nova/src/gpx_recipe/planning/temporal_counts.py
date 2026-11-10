"""Positive pulse-resolved correlated intensities, separate from calibration.

Each independent phase follows wrapped Brownian motion with a stated drift.
Finite harmonics share that phase. A zero-drift mode supplies bounded slow
correlation; a nonzero drift supplies a broadened periodic component. Uniform
initial phases make the process stationary. Counts are Poisson conditional on
the complete intensity history, and are correlated after marginalizing it.

Parameters here describe a proposed count process, not measured POWGEN beam
physics. No actual source, specimen, background or reference law is inferred.
The existing planning UI is not switched to this model by importing it.
Channels must be distinct count outcomes. Overlapping angle/panel/TOF views
of the same events must not be supplied as independent Poisson channels.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np

VERSION = 'positive-phase-pulse-counts/1'
_TWO_PI = 2 * np.pi


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _budget(value):
    if type(value) is not int or value < 1024:
        raise ValueError('An explicit positive working-memory budget is required')
    return value


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError('A reproducible unsigned64 integer seed is required')
    return value


def _clock(value):
    a = np.asarray(value)
    if (a.ndim != 1 or a.dtype.kind not in 'iuf' or not 1 <= len(a) <= 2**22
            or not np.isfinite(a).all() or a[0] < 0 or a[-1] > 1e6
            or np.any(np.diff(a.astype(float)) <= 0)):
        raise ValueError('A finite strictly increasing relative source clock in seconds is required')
    return a.astype(float)


@dataclass(frozen=True)
class TemporalCountMoments:
    mean: np.ndarray
    poisson_variance: np.ndarray
    covariance_factors: np.ndarray
    metadata: dict

    @property
    def variance(self):
        return self.poisson_variance + np.sum(self.covariance_factors**2, axis=0)

    def covariance(self, *, max_bytes=256 * 1024**2):
        if 24 * len(self.mean)**2 > _budget(max_bytes):
            raise ValueError('Full count covariance exceeds the working-memory budget; use its factors')
        return np.diag(self.poisson_variance) + self.covariance_factors.T @ self.covariance_factors


@dataclass(frozen=True)
class TemporalCountDraw:
    stop_indices: np.ndarray
    phases_rad: np.ndarray
    conditional_increment_means: np.ndarray
    increments: np.ndarray
    cumulative_counts: np.ndarray
    metadata: dict


@dataclass(frozen=True)
class PhaseRateLaw:
    frequency_hz: np.ndarray
    phase_variance_rate_rad2_s: np.ndarray
    cosine: np.ndarray
    sine: np.ndarray
    provenance: str

    def __post_init__(self):
        f, d = np.asarray(self.frequency_hz), np.asarray(self.phase_variance_rate_rad2_s)
        c, s = np.asarray(self.cosine), np.asarray(self.sine)
        if (f.ndim != 1 or not 1 <= len(f) <= 8 or d.shape != f.shape
                or c.ndim != 3 or s.shape != c.shape or c.shape[0] != len(f)
                or not 1 <= c.shape[1] <= 100000 or not 1 <= c.shape[2] <= 16
                or c.size * 48 > 128 * 1024**2
                or any(a.dtype.kind not in 'iuf' or not np.isfinite(a).all() for a in [f, d, c, s])
                or np.any(f < 0) or np.any(f > 1e5) or np.any(d < 0) or np.any(d > 1e6)
                or not isinstance(self.provenance, str) or not self.provenance.strip()
                or len(self.provenance) > 4096):
            raise ValueError('Explicit bounded phase rates, matching harmonic loadings and provenance are required')
        f, d, c, s = [np.array(a, float, copy=True) for a in [f, d, c, s]]
        if np.any(abs(c)>1) or np.any(abs(s)>1):
            raise ValueError('The sum of harmonic amplitudes in each channel must be <=1 for nonnegative rates')
        amplitude = np.hypot(c, s)
        sums = np.array([math.fsum(amplitude[:, i, :].ravel()) for i in range(c.shape[1])])
        if np.any(sums > 1):
            raise ValueError('The sum of harmonic amplitudes in each channel must be <=1 for nonnegative rates')
        # Stable half-angle coordinates preserve the exact zero at c=-a,s=0,
        # phi=0. Evaluating cos(atan2(0,-a)/2) would introduce a tiny positive
        # intensity and could make an impossible fixed-total allocation appear
        # normalizable. No small positive intensity is otherwise discarded.
        unit_c=np.divide(c,amplitude,out=np.ones_like(c),where=amplitude>0)
        unit_s=np.divide(s,amplitude,out=np.zeros_like(s),where=amplitude>0)
        positive=c>=0
        u=np.sqrt((1+unit_c)/2)
        v=np.copysign(np.sqrt((1-unit_c)/2),unit_s)
        u[~positive]=unit_s[~positive]/(2*v[~positive])
        v[positive]=unit_s[positive]/(2*u[positive])
        for name, value in [('frequency_hz', f), ('phase_variance_rate_rad2_s', d),
                            ('cosine', c), ('sine', s), ('_amplitude', amplitude),
                            ('_baseline', 1-sums), ('_half_cosine', u), ('_half_sine', v)]:
            object.__setattr__(self, name, _readonly(value))

    @property
    def channels(self):
        return self.cosine.shape[1]

    def _metadata(self, dose_unit):
        if not isinstance(dose_unit, str) or not dose_unit.strip() or len(dose_unit) > 64:
            raise ValueError('The supplied dose and rate units must be stated explicitly')
        digest = hashlib.sha256(VERSION.encode())
        for a in [self.frequency_hz, self.phase_variance_rate_rad2_s, self.cosine, self.sine]:
            digest.update(json.dumps(a.shape).encode())
            digest.update(a.astype('<f8').tobytes())
        return dict(version=VERSION, parameter_sha256=digest.hexdigest(), provenance=self.provenance,
                    dose_unit=dose_unit, rate_unit='counts/'+dose_unit,
                    stationary_initial_phase='Independent uniform phases on [0,2pi)',
                    phase_increment_variance='D*delta_seconds in radians squared',
                    covariance_conditional_on_supplied_rates_dose_and_process_parameters=True,
                    observed_block_total_conditioning_not_included=True,
                    distinct_count_channels_required_not_overlapping_event_views=True,
                    calibration_reference_and_parameter_uncertainty_not_included=True,
                    physical_source_background_and_acquisition_cause_not_identified=True,
                    experimental_accuracy_certified=False)

    def _rate_dose(self, rate, dose, n):
        r, q = np.asarray(rate), np.asarray(dose)
        if (r.shape != (self.channels,) or q.shape != (n,)
                or any(a.dtype.kind not in 'iuf' or not np.isfinite(a).all() or np.any(a < 0) for a in [r, q])):
            raise ValueError('Finite nonnegative channel rates and matching pulse doses are required')
        r, q = r.astype(float), q.astype(float)
        try:
            total = math.fsum(q)
        except OverflowError as exc:
            raise ValueError('Pulse dose sum exceeds floating-point range') from exc
        if not math.isfinite(total) or np.any(r > np.finfo(float).max/max(total, 1.)):
            raise ValueError('Requested count moments exceed floating-point range')
        if total>0 and np.any((r>0)&(r*total==0)):
            raise ValueError('A positive count mean is below floating-point range')
        return r, q, total

    def sample_phases(self, times_s, *, seed, max_bytes=256 * 1024**2):
        t = _clock(times_s)
        if len(t) * (48*len(self.frequency_hz)+64) > _budget(max_bytes):
            raise ValueError('Phase history exceeds the working-memory budget')
        rng = np.random.default_rng(_seed(seed))
        initial = rng.uniform(0, _TWO_PI, len(self.frequency_hz))
        drift = _TWO_PI*np.remainder((t-t[0])[:, None]*self.frequency_hz, 1.)
        phase = initial + drift
        if len(t) > 1 and np.any(self.phase_variance_rate_rad2_s > 0):
            steps = rng.standard_normal((len(t)-1, len(self.frequency_hz)))
            steps *= np.sqrt(np.diff(t)[:, None]*self.phase_variance_rate_rad2_s)
            phase[1:] += np.cumsum(steps, axis=0)
        return _readonly(np.remainder(phase, _TWO_PI))

    def _phases(self, value):
        p = np.asarray(value)
        if (p.ndim != 2 or p.shape[1] != len(self.frequency_hz) or not 1 <= len(p) <= 2**22
                or p.dtype.kind not in 'iuf' or not np.isfinite(p).all()
                or np.any(p < 0) or np.any(p >= _TWO_PI)):
            raise ValueError('A matching finite phase history in [0,2pi) is required')
        return p.astype(float)

    def multipliers(self, phases_rad, *, max_bytes=256 * 1024**2):
        p = self._phases(phases_rad)
        if 48*len(p)*self.channels > _budget(max_bytes):
            raise ValueError('Full intensity history exceeds the memory budget; integrate pulse doses instead')
        result = np.broadcast_to(self._baseline, (len(p), self.channels)).copy()
        for m in range(len(self.frequency_hz)):
            for k in range(self.cosine.shape[2]):
                angle = (k+1)*p[:, m, None]/2
                value=np.cos(angle)*self._half_cosine[m,:,k]+np.sin(angle)*self._half_sine[m,:,k]
                result += 2*self._amplitude[m, :, k]*value**2
        assert np.isfinite(result).all() and np.all(result >= 0)
        return _readonly(result)

    def integrated_mean(self, rate, dose, phases_rad, *, max_bytes=256 * 1024**2):
        """Sum conditional pulse means with nonnegative QR/squared-norm algebra.

        A trig-moment subtraction near a zero-rate phase can go negative from
        cancellation. Here each harmonic is a weighted squared cosine and is
        accumulated through a two-column QR. No rate or count floor is used.
        """
        p = self._phases(phases_rad)
        if 96*len(p)+64*self.channels > _budget(max_bytes):
            raise ValueError('Pulse integration exceeds its working-memory budget')
        r, q, total = self._rate_dose(rate, dose, len(p))
        integrated = self._baseline.copy()
        root = np.sqrt(q/total) if total else np.zeros_like(q)
        if total:
            for m in range(len(self.frequency_hz)):
                for k in range(self.cosine.shape[2]):
                    if not np.any(self._amplitude[m, :, k]):
                        continue
                    a = (k+1)*p[:, m]/2
                    design = root[:, None]*np.column_stack([np.cos(a), np.sin(a)])
                    R = np.linalg.qr(design, mode='r')
                    direction = np.vstack([self._half_cosine[m, :, k],self._half_sine[m, :, k]])
                    integrated += 2*self._amplitude[m, :, k]*np.sum((R@direction)**2, axis=0)
        mean = (r*total)*integrated
        if not np.isfinite(mean).all():
            raise ValueError('Conditional mean exceeds floating-point range')
        return _readonly(mean)

    def moments(self, rate, dose, times_s, *, dose_unit, max_bytes=256 * 1024**2):
        """Exact unconditional count moments on the supplied discrete pulse grid.

        For harmonic h, K(t)=exp(-h²D|t|/2)*cos(2pi*h*f*t).
        Its dose double sum is evaluated through nonnegative state-innovation
        squares. The resulting count covariance is diag(mean)+F.T@F.
        """
        t = _clock(times_s)
        if 96*len(t)+64*self.cosine.size > _budget(max_bytes):
            raise ValueError('Count-moment integration exceeds the working-memory budget')
        meta = self._metadata(dose_unit)
        r, q, total = self._rate_dose(rate, dose, len(t))
        mean = r*total
        q = q/total if total else q
        factors = []
        for m, (frequency, diffusion) in enumerate(zip(self.frequency_hz, self.phase_variance_rate_rad2_s)):
            for k in range(self.cosine.shape[2]):
                if not np.any(self._amplitude[m, :, k]):
                    continue
                h = k+1
                gamma = .5*h*h*diffusion
                omega = _TWO_PI*h*frequency
                if gamma == 0:
                    value = np.sum(q*np.exp(1j*omega*(t-t[0])))
                    root_weight = float(abs(value))
                else:
                    future = complex(q[-1]);innovation_norm = 0.
                    for i in range(len(t)-1, 0, -1):
                        dt = t[i]-t[i-1]
                        innovation_norm = np.hypot(innovation_norm,
                            math.sqrt(-np.expm1(-2*gamma*dt))*abs(future))
                        future = q[i-1]+np.exp(complex(-gamma, omega)*dt)*future
                    root_weight = float(np.hypot(abs(future),innovation_norm))
                assert math.isfinite(root_weight) and root_weight >= 0
                scale = (mean*root_weight)/math.sqrt(2.)
                factors.extend([scale*self.cosine[m, :, k], scale*self.sine[m, :, k]])
        F = np.asarray(factors).reshape(-1, self.channels)
        norm = np.hypot.reduce(F, axis=0) if len(F) else np.zeros(self.channels)
        if np.any(norm > math.sqrt(np.finfo(float).max)) or np.any(norm**2 > np.finfo(float).max-mean):
            raise ValueError('Count variance exceeds floating-point range')
        meta.update(discrete_pulse_dose_used_without_interpolation=True,
                    normalized_internal_dose_weights_and_root_norm_avoid_unit_scaling_underflow=True,
                    no_count_or_source_floor=True, covariance_rank=len(F),
                    omitted_inactive_harmonics_have_zero_loadings=True)
        return TemporalCountMoments(_readonly(mean), _readonly(mean), _readonly(F), meta)

    def draw_nested(self, rate, dose, times_s, stop_indices, *, seed, dose_unit,
                    max_bytes=256 * 1024**2):
        """One phase path and disjoint Poisson increments shared by all looks."""
        t = _clock(times_s);stops = np.asarray(stop_indices)
        if (stops.ndim != 1 or stops.dtype.kind not in 'iu' or not 1 <= len(stops) <= 4096
                or stops[0] <= 0 or np.any(stops > len(t)) or np.any(np.diff(stops.astype(np.int64)) <= 0)):
            raise ValueError('Strictly increasing positive pulse stop indices are required')
        if 64*len(stops)*self.channels+len(t)*(48*len(self.frequency_hz)+128) > _budget(max_bytes):
            raise ValueError('Nested acquisition exceeds its working-memory budget')
        meta = self._metadata(dose_unit)
        r, q, _ = self._rate_dose(rate, dose, len(t))
        phase_stream, count_stream = np.random.SeedSequence(_seed(seed)).spawn(2)
        phase_seed = int(phase_stream.generate_state(1, dtype=np.uint64)[0])
        phases = self.sample_phases(t, seed=phase_seed, max_bytes=max_bytes)
        rng = np.random.default_rng(count_stream)
        means=[];increments=[];cumulative=[];total=np.zeros(self.channels, np.int64);first=0
        for last in stops:
            mean=self.integrated_mean(r,q[first:last],phases[first:last],max_bytes=max_bytes)
            delta=rng.poisson(mean)
            if np.any(delta > np.iinfo(np.int64).max-total):
                raise ValueError('Cumulative counts exceed int64 range')
            total=total+delta;means.append(mean);increments.append(delta);cumulative.append(total.copy());first=int(last)
        meta.update(seed=seed,phase_seed=phase_seed,shot_stream_spawn_index=1,
                    one_stationary_phase_path_shared_across_looks=True,
                    disjoint_conditional_Poisson_increments=True,
                    pulse_dose_zero_keeps_phase_evolution_but_generates_no_counts=True,
                    future_look_schedule_not_guaranteed_bit_identical_shot_counts=True)
        return TemporalCountDraw(_readonly(stops.astype(np.int64)),phases,_readonly(means),
                                 _readonly(increments),_readonly(cumulative),meta)
