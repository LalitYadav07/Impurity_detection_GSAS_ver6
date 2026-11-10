"""Periodic prompt cuts using the same absolute boundaries as interval kernels."""
import math
from fractions import Fraction
import numpy as np


def periodic_prompt_acceptance(time_us, period_us, width_us, *, upper_inclusive=False,
                               nominal_frequency_hz=None):
    """Reject n*period <= t < n*period+width, with an optional closed upper end.

    Floating remainder is inconsistent with rounded absolute boundaries at
    exactly representable TOFs such as50000us for a nominal60Hz period.
    Check the neighboring absolute intervals instead. Integer-frame products
    define the same boundaries used by the source/monitor partition kernels.
    A width of zero disables the cut. Input float32 clocks are promoted before
    comparisons so a Python scalar cannot silently round a boundary to float32.
    If an integer nominal frequency is explicitly declared, use exact rational
    period boundaries, rounding comparisons outward by one representable float
    only when required by an exact Fraction comparison. No event tolerance or
    inferred snapping of a measured period is applied.
    """
    if (isinstance(period_us, (bool, np.bool_)) or not np.isscalar(period_us)
            or not math.isfinite(float(period_us)) or period_us <= 0):
        raise ValueError("A finite positive prompt period is required")
    if (isinstance(width_us, (bool, np.bool_)) or not np.isscalar(width_us)
            or not math.isfinite(float(width_us)) or not 0 <= width_us < period_us):
        raise ValueError("A finite prompt width within the period is required")
    if not isinstance(upper_inclusive, (bool, np.bool_)):
        raise ValueError("An explicit prompt upper-boundary convention is required")
    if nominal_frequency_hz is not None:
        if (isinstance(nominal_frequency_hz, (bool, np.bool_))
                or not isinstance(nominal_frequency_hz, (int, np.integer))
                or not 1 <= nominal_frequency_hz <= 1000000
                or float(period_us) != 1000000/int(nominal_frequency_hz)):
            raise ValueError("An explicit consistent integer nominal frequency is required")
    t = np.asarray(time_us, dtype=np.float64)
    if not np.isfinite(t).all():
        raise ValueError("Finite recorded event clocks are required")
    if width_us == 0 or t.size == 0:
        return np.ones(t.shape, dtype=bool)
    with np.errstate(over="ignore"):
        frame = np.floor(t/float(period_us))
    if not np.isfinite(frame).all() or np.any(np.abs(frame) > 2**50):
        raise ValueError("Prompt frame indexing exceeds its finite precision domain")
    accepted = np.ones(t.shape, dtype=bool)
    if nominal_frequency_hz is not None:
        first, last = int(frame.min())-1, int(frame.max())+1
        if last-first+1 > 100000:
            raise ValueError("Exact nominal prompt partition exceeds its interval budget")
        def directed(value, upward):
            rounded = float(value)
            exact_rounded = Fraction.from_float(rounded)
            if upward and exact_rounded < value:
                return np.nextafter(rounded, np.inf)
            if not upward and exact_rounded > value:
                return np.nextafter(rounded, -np.inf)
            return rounded
        width = Fraction.from_float(float(width_us))
        for n in range(first, last+1):
            lower = Fraction(n*1000000, int(nominal_frequency_hz))
            low = directed(lower, True)
            high = directed(lower+width, not upper_inclusive)
            inside = (t >= low) & ((t <= high) if upper_inclusive else (t < high))
            accepted &= ~inside
        return accepted
    for neighbor in (-1., 0., 1.):
        low = (frame+neighbor)*float(period_us)
        high = low+float(width_us)
        inside = (t >= low) & ((t <= high) if upper_inclusive else (t < high))
        accepted &= ~inside
    return accepted
