"""Validate TOF Gaussian variance over the actual fitted coordinate interval.

Only an invalid low-d tail can be repaired by a lower cutoff. Invalid upper
regions or holes require a reviewed profile/range, not a skipped safety check.
"""
import copy
import math
import numpy as np

_D_BUFFER = 0.05


def _value(parameters, key, default=0.0):
    value = parameters.get(key, default)
    if isinstance(value, (list, tuple, np.ndarray)) and np.ndim(value):
        if not len(value):
            raise ValueError(f'Empty TOF parameter {key}')
        # GSAS-II stores [original, current, refine]. Check the actual model.
        value = value[1] if len(value) > 1 else value[0]
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'Nonfinite TOF parameter {key}')
    return value


def _coefficients(parameters):
    return np.array([_value(parameters, 'sig-2'), 0.0,
                     _value(parameters, 'sig-1'), _value(parameters, 'sig-q'),
                     _value(parameters, 'sig-0')])


def _positive_roots(coefficients):
    coefficients = np.trim_zeros(coefficients, 'f')
    if len(coefficients) <= 1:
        return []
    roots = sorted(float(r.real) for r in np.roots(coefficients)
                   if r.real > 0 and abs(r.imag) < 1e-8 * max(1.0, abs(r.real)))
    unique = []
    for root in roots:
        if not unique or not math.isclose(root, unique[-1], rel_tol=1e-8, abs_tol=1e-10):
            unique.append(root)
    return unique


def _minimum(coefficients, lower, upper):
    points = [lower, upper] + [r for r in _positive_roots(np.polyder(coefficients))
                               if lower < r < upper]
    return min(float(np.polyval(coefficients, d)) for d in points)


def _positive_intervals(coefficients, lower, upper):
    points = [lower] + [r for r in _positive_roots(coefficients) if lower < r < upper] + [upper]
    return [(lo, hi) for lo, hi in zip(points, points[1:])
            if np.polyval(coefficients, (lo + hi) / 2) > 0]


def calculate_safe_d_min(inst_params, *, d_max=None):
    """Find a cutoff whose entire remaining fitted interval has positive variance.

    Without an upper bound this requires a positive asymptotic tail. A negative
    quartic profile has no globally safe lower cutoff.
    """
    coefficients = _coefficients(inst_params)
    nonzero = np.trim_zeros(coefficients, 'f')
    if not len(nonzero):
        raise ValueError('TOF instrument Gaussian variance is zero everywhere')
    roots = _positive_roots(coefficients)
    if d_max is None:
        if nonzero[0] <= 0:
            raise ValueError('TOF Gaussian variance has an invalid high-d tail; a lower cutoff cannot fix it')
    else:
        if not math.isfinite(d_max) or d_max <= 0 or np.polyval(coefficients, d_max) <= 0:
            raise ValueError('TOF Gaussian variance is non-positive at the fitted upper d limit')
        roots = [r for r in roots if r < d_max]
    return max(0.1, max(roots) + _D_BUFFER) if roots else 0.1


def _coordinate(parameters):
    dc, da, db, zero = [_value(parameters, k) for k in ('difC', 'difA', 'difB', 'Zero')]
    if dc <= 0:
        raise ValueError('TOF difC must be positive')

    def position(d):
        return dc * d + da * d * d + db / d + zero

    def slope(d):
        return dc + 2 * da * d - db / (d * d)

    def inverse(tof):
        # Multiplying T(d)-tof by d gives a cubic. Ignore the spurious d=0.
        roots = [r for r in _positive_roots([da, dc, zero - tof, db]) if slope(r) > 0]
        if len(roots) != 1:
            raise ValueError('TOF fit limit has no unique increasing positive-d calibration branch')
        return roots[0]

    def increasing(lower, upper):
        points = [lower, upper]
        if da and -db / da > 0:
            critical = (-db / da) ** (1 / 3)
            if lower < critical < upper:
                points.append(critical)
        return min(slope(d) for d in points) > 0

    return position, inverse, increasing


def apply_safe_limits(proj):
    """Keep valid ranges; trim only an isolated invalid low-d tail.

    Reject invalid upper/interior regions and excessive truncation. Validate all
    banks before changing any. CW histograms are unaffected; errors propagate.
    """
    changes = []
    for hist in proj.histograms():
        if 'Instrument Parameters' not in hist.data:
            continue
        inst = hist.data['Instrument Parameters'][0]
        token = inst.get('Type', '')
        if isinstance(token, (list, tuple, np.ndarray)):
            token = token[0] if len(token) else ''
        if 'T' not in str(token):
            continue
        try:
            limits = hist.data['Limits']
            current_min, current_max = map(float, limits[1])
            if not all(math.isfinite(v) for v in (current_min, current_max)) or current_max <= current_min:
                raise ValueError('Invalid native fitted limits')
            position, inverse, increasing = _coordinate(inst)
            lower, upper = inverse(current_min), inverse(current_max)
            if upper <= lower or not increasing(lower, upper):
                raise ValueError('TOF calibration is not increasing across the fitted range')
            coefficients = _coefficients(inst)
            if _minimum(coefficients, lower, upper) > 0:
                continue
            intervals = _positive_intervals(coefficients, lower, upper)
            if (len(intervals) != 1 or not math.isclose(intervals[0][1], upper)
                    or np.polyval(coefficients, upper) <= 0):
                support = ', '.join(f'{lo:.5g}–{hi:.5g}' for lo, hi in intervals) or 'none'
                raise ValueError(
                    f'Instrument Gaussian variance is non-positive within the fitted d range '
                    f'{lower:.5g}–{upper:.5g} Å (positive intervals: {support} Å). '
                    'A lower cutoff cannot safely repair this range. Supply a reviewed '
                    'instrument profile or explicitly select a supported fitted range.'
                )
            safe_d = calculate_safe_d_min(inst, d_max=upper)
            if safe_d >= upper or _minimum(coefficients, max(lower, safe_d), upper) <= 0:
                raise ValueError('No usable contiguous positive-variance range after the low-d safety buffer')
            safe_native = position(safe_d)
            if safe_native > (current_min + current_max) / 2:
                raise ValueError('A safe lower cutoff would remove more than half the fitted TOF span; review the profile/range explicitly')
            if safe_native > current_min:
                changes.append((hist, limits, current_min, safe_native, safe_d))
        except (ValueError, TypeError, KeyError, IndexError, FloatingPointError) as exc:
            raise ValueError(f'[SafeLimits] Histogram {hist.name}: {exc}') from exc
    for hist, limits, old, new, safe_d in changes:
        updated = copy.deepcopy(limits)
        updated[1] = list(updated[1])
        updated[1][0] = float(new)
        hist.data['Limits'] = updated
        print(f'[SafeLimits] Histogram {hist.name}: Raising min limit {old:.2f} -> {new:.2f} (d_min={safe_d:.5g} Å)')
    return bool(changes)
