"""Traceable pulse dose and clock alignment, separate from neutron availability.

The optional native filter reproduces LoadEventNexus's centre-log, zero-
tolerance, unexpanded half-open bad-pulse ROI. A retained proton dose is not
proof of transmitted neutrons. No shutter, flux or live-time correction is
inferred from detector counts. Missing state coverage stays explicit.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from numbers import Real
import re
import numpy as np

VERSION = 'powgen-pulse-exposure/1'
PC_PER_UAH = 3.6e9
_MAX_RELATIVE_SECONDS = 1e6
_ISO = re.compile(r'^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,9}))?(Z|[+-]\d\d:\d\d)$')


def _origin(value):
    if type(value) is not int or not -(2**63) <= value < 2**63:
        raise ValueError('An explicit integer UTC nanosecond origin is required')
    return value


def iso8601_origin_ns(value):
    """Parse an explicit UTC offset and up to nine fractional digits exactly."""
    if isinstance(value, bytes):
        value = value.decode('ascii')
    if type(value) is not str or len(value) > 64:
        raise ValueError('A bounded ISO8601 timestamp with timezone is required')
    match = _ISO.fullmatch(value)
    if match is None:
        raise ValueError('An ISO8601 timestamp with an explicit timezone is required')
    main, fraction, zone = match.groups()
    date = datetime.fromisoformat(main + ('+00:00' if zone == 'Z' else zone))
    delta = date.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    seconds = delta.days * 86400 + delta.seconds
    return _origin(seconds * 10**9 + int((fraction or '').ljust(9, '0') or '0'))


def relative_clock_ns(times_s, origin_ns, reference_origin_ns, *, max_bytes=128 * 1024**2):
    """Align relative clocks without converting a large epoch to float.

    Relative floats are rounded to integer nanoseconds. The bounded relative
    span keeps float64 resolution finer than one nanosecond. This does not
    recover precision that the producer did not record.
    """
    origin, reference = _origin(origin_ns), _origin(reference_origin_ns)
    if type(max_bytes) is not int or max_bytes < 1024:
        raise ValueError('An explicit positive working-memory budget is required')
    t = np.asarray(times_s)
    if (t.ndim != 1 or t.dtype.kind not in 'iuf' or t.size * 32 > max_bytes
            or not np.isfinite(t).all() or (np.any(t < -_MAX_RELATIVE_SECONDS) or np.any(t > _MAX_RELATIVE_SECONDS))):
        raise ValueError('A bounded finite vector of relative seconds is required')
    delta = origin - reference
    if abs(delta) + int(_MAX_RELATIVE_SECONDS * 1e9) >= 2**63:
        raise ValueError('Aligned relative clock exceeds int64 nanoseconds')
    return np.rint(t.astype(float) * 1e9).astype(np.int64) + delta


def native_bad_pulse_mask(charge_pC, times_s, cutoff_percent=90.):
    """Exact direct-loader interval rule; None disables this filter.

    Every qualifying sequence contributes [first_time,last_time); its endpoint
    is excluded and isolated qualifying samples contribute no interval. An
    explicit zero cutoff still requests this rule, matching the direct loader.
    This is a software-compatibility selection, not a physical pulse diagnosis.
    """
    q, t = np.asarray(charge_pC), np.asarray(times_s)
    if q.dtype.kind not in 'iuf' or t.dtype.kind not in 'iuf':
        raise ValueError('Charge and pulse-clock vectors must be real numeric values')
    q, t = q.astype(float), t.astype(float)
    if (q.ndim != 1 or t.shape != q.shape or not len(q) or not np.isfinite(q).all()
            or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)):
        raise ValueError('Finite charge and a strictly increasing pulse clock are required')
    if cutoff_percent is None:
        return np.ones(len(q), bool), dict(filter_enabled=False, cutoff_percent=None)
    if (isinstance(cutoff_percent, (bool, str, bytes, np.bool_)) or not isinstance(cutoff_percent, Real)
            or not math.isfinite(cutoff_percent) or not 0 <= cutoff_percent <= 100 or q.max() <= 0):
        raise ValueError('A cutoff in [0,100] and positive maximum charge are required')
    lower, upper = float(q.mean()) * float(cutoff_percent) * .01, float(q.max()) * 1.1
    if lower >= upper:
        raise ValueError('The native pulse window is empty')
    qualified = (q >= lower) & (q <= upper)
    keep = qualified & np.r_[qualified[1:], False]
    return keep, dict(filter_enabled=True, cutoff_percent=float(cutoff_percent),
                      lower_pC=lower, upper_pC=upper, qualifying_pulses=int(qualified.sum()),
                      excluded_qualifying_endpoints=int((qualified & ~keep).sum()),
                      interval_rule='centre=true; tolerance0; expand=false; half-open')


def _frozen(value):
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class BinnedDose:
    edges_s: np.ndarray
    charge_uah: np.ndarray
    retained_pulses: np.ndarray
    outside_charge_uah: float
    metadata: dict


@dataclass(frozen=True)
class PulseExposure:
    times_s: np.ndarray
    times_ns: np.ndarray
    charge_pC: np.ndarray
    keep: np.ndarray
    origin_ns: int
    metadata: dict

    @property
    def total_charge_uah(self):
        return float(self.charge_pC[self.keep].sum() / PC_PER_UAH)

    def bin_charge(self, edges_s):
        """Half-open bins with pairwise sums; all outside dose is retained in metadata."""
        edges = np.asarray(edges_s)
        if (edges.ndim != 1 or not 2 <= len(edges) <= 1000001 or edges.dtype.kind not in 'iuf'
                or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0)):
            raise ValueError('Finite strictly increasing time-bin edges are required')
        edges = np.array(edges, float, copy=True)
        ens = relative_clock_ns(edges, self.origin_ns, self.origin_ns)
        if np.any(np.diff(ens) <= 0):
            raise ValueError('Time-bin edges must remain distinct at nanosecond precision')
        boundary = np.searchsorted(self.times_ns, ens, side='left')
        dose, pulses = [], []
        for first, last in zip(boundary[:-1], boundary[1:]):
            mask = self.keep[first:last]
            dose.append(self.charge_pC[first:last][mask].sum() / PC_PER_UAH)
            pulses.append(int(mask.sum()))
        outside = ((self.charge_pC[:boundary[0]][self.keep[:boundary[0]]].sum()
                    + self.charge_pC[boundary[-1]:][self.keep[boundary[-1]:]].sum()) / PC_PER_UAH)
        charge, number = np.array(dose), np.array(pulses, np.int64)
        total = self.total_charge_uah
        if not np.isclose(charge.sum() + outside, total, rtol=2e-12, atol=1e-15):
            raise ValueError('Dose partition failed conservation')
        return BinnedDose(_frozen(edges), _frozen(charge), _frozen(number), float(outside),
                         dict(version=VERSION, whole_retained_charge_uah=total,
                              bin_rule='[left,right)', outside_dose_not_renormalized=True,
                              neutron_availability_inferred=False, experimental_accuracy_certified=False))


def pulse_exposure(charge_pC, times_s, origin_ns, *, cutoff_percent=90., max_bytes=128 * 1024**2):
    """Build a nonnegative dose ledger from a stated clock and native filter."""
    if type(max_bytes) is not int or max_bytes < 1024:
        raise ValueError('An explicit positive working-memory budget is required')
    t, q = np.asarray(times_s), np.asarray(charge_pC)
    if (t.ndim != 1 or q.shape != t.shape or t.size * 64 > max_bytes
            or t.dtype.kind not in 'iuf' or q.dtype.kind not in 'iuf'):
        raise ValueError('Pulse vectors must fit the explicit working-memory budget')
    ns = relative_clock_ns(t, origin_ns, origin_ns, max_bytes=max_bytes)
    if np.any(np.diff(ns) <= 0):
        raise ValueError('Pulse times must remain strictly increasing at nanosecond precision')
    mask, meta = native_bad_pulse_mask(q, t, cutoff_percent)
    if np.any(q[mask] < 0):
        raise ValueError('Selected physical proton charge cannot be negative')
    return PulseExposure(_frozen(np.array(t, float, copy=True)), _frozen(ns),
                         _frozen(np.array(q, float, copy=True)), _frozen(mask), _origin(origin_ns),
                         dict(meta, version=VERSION, charge_unit='pC', dose_unit='uAh',
                              transmitted_neutron_exposure_verified=False,
                              experimental_accuracy_certified=False))


@dataclass(frozen=True)
class StepState:
    index: np.ndarray
    values: np.ndarray
    known: np.ndarray
    metadata: dict

    def classify(self, accepted_values):
        """Return accepted/rejected/unknown masks without imputing missing states."""
        accepted = np.zeros(len(self.index), bool)
        accepted[self.known] = np.isin(self.values[self.index[self.known]], accepted_values)
        return accepted, self.known & ~accepted, ~self.known


def align_step_state(query_times_s, query_origin_ns, log_times_s, log_origin_ns, values,
                     *, coverage_stop_s, max_bytes=128 * 1024**2):
    """Right-continuous recorded state; last duplicate update wins.

    Coverage ends at an explicitly supplied log-clock boundary. Before the
    first update and at/after that boundary, state is unknown. Holding between
    updates is a declared step-log interpretation, not independent hardware proof.
    """
    query = relative_clock_ns(query_times_s, query_origin_ns, query_origin_ns, max_bytes=max_bytes)
    clock = relative_clock_ns(log_times_s, log_origin_ns, query_origin_ns, max_bytes=max_bytes)
    v = np.asarray(values)
    if (not len(clock) or np.any(np.diff(clock) < 0) or v.ndim != 1 or len(v) != len(clock)
            or v.dtype.kind not in 'biufSU' or v.nbytes + 24 * len(query) > max_bytes
            or (v.dtype.kind in 'f' and not np.isfinite(v).all())):
        raise ValueError('A bounded nondecreasing scalar state log is required')
    stop = relative_clock_ns([coverage_stop_s], log_origin_ns, query_origin_ns, max_bytes=max_bytes)[0]
    if stop <= clock[-1]:
        raise ValueError('Explicit state coverage must end after its last update')
    index = np.searchsorted(clock, query, side='right') - 1
    known = (index >= 0) & (query < stop)
    index[~known] = -1
    return StepState(_frozen(index), _frozen(np.array(v, copy=True)), _frozen(known),
                     dict(version=VERSION, interpretation='right-continuous step; final duplicate wins',
                          coverage_stop_relative_ns=int(stop), missing_state_imputed=False,
                          hardware_state_independently_verified=False, experimental_accuracy_certified=False))
