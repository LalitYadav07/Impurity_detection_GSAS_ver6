"""Restrict a sampled identity log only when its acquisition clock is proved.

NXlog time values are relative to time@start, in time@units. A preceding
observation is the value at acquisition start; subsequent changes are kept.
Malformed or incomplete clocks retain every observation, so ambiguity cannot
silently become a resolved sample ID.
"""
from __future__ import annotations

import math

from .facility_metadata import parse_nexus_timestamp


_SECONDS = {
    "s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
    "ms": 1e-3, "millisecond": 1e-3, "milliseconds": 1e-3,
    "us": 1e-6, "µs": 1e-6, "microsecond": 1e-6, "microseconds": 1e-6,
    "ns": 1e-9, "nanosecond": 1e-9, "nanoseconds": 1e-9,
    "min": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hour": 3600.0, "hours": 3600.0,
}


def scope_identity_log(values, times, attributes, run_start, run_end):
    """Return kept indices and a small clock/provenance receipt.

    The one-microsecond boundary tolerance only accommodates truncating
    nanosecond ISO timestamps to Python's datetime precision. Both identities
    at a duplicate timestamp are retained; no tie is resolved by array order.
    """
    kept = list(range(len(values)))
    receipt = {"status": "unqualified", "recorded_observations": list(values)}
    try:
        if len(values) != len(times) or not values:
            raise ValueError("identity values and timestamps do not have matching lengths")
        start = parse_nexus_timestamp(run_start)
        end = parse_nexus_timestamp(run_end)
        origin = parse_nexus_timestamp(attributes.get("start"))
        if any(t.tzinfo is None for t in (start, end, origin)):
            raise ValueError("identity clock or acquisition timestamps lack a timezone")
        duration = (end - start).total_seconds()
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("acquisition end precedes its start")
        unit = str(attributes.get("units", "")).strip().casefold()
        factor = _SECONDS.get(unit)
        if factor is None:
            raise ValueError("identity time units are not supported")
        scale = attributes.get("scaling_factor", 1.0)
        if isinstance(scale, bool):
            raise ValueError("identity clock scaling is invalid")
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("identity clock scaling is invalid")
        offset = (origin - start).total_seconds()
        seconds = []
        for value in times:
            if isinstance(value, bool):
                raise ValueError("identity timestamps are invalid")
            value = float(value) * factor * scale + offset
            if not math.isfinite(value):
                raise ValueError("identity timestamps are nonfinite")
            seconds.append(value)
        if any(right < left for left, right in zip(seconds, seconds[1:])):
            raise ValueError("identity timestamps are not ordered")
        epsilon = 1e-6
        # The tolerance may include an uncertain boundary observation, but a
        # post-start value must not replace the true predecessor at t=0.
        before = [i for i, value in enumerate(seconds) if value <= 0]
        active = []
        if before:
            last = seconds[before[-1]]
            active = [i for i in before if seconds[i] == last]
        within = [i for i, value in enumerate(seconds) if -epsilon <= value <= duration + epsilon]
        kept = sorted(set(active + within))
        receipt.update(
            status="acquisition_window", start=run_start, end=run_end,
            log_start=attributes["start"], time_units=unit,
            scaling_factor=scale, seconds_from_acquisition_start=seconds,
            kept_indices=kept,
            ignored_before_count=sum(i not in kept and t < 0 for i, t in enumerate(seconds)),
            ignored_after_count=sum(t > duration + epsilon for t in seconds),
        )
    except (TypeError, ValueError, OverflowError) as error:
        receipt["reason"] = str(error)
    return kept, receipt
