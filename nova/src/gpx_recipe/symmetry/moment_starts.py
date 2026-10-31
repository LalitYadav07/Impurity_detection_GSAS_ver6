"""Reproducible independent-orbit starts inside exact site-symmetry bases."""
import hashlib

import numpy as np


def independent_orbit_start(ids, coefficients, current, metric, *, magnitude, seed, orbit):
    """Break accidental equal-orbit starts without changing any allowed DOF.

    Each repeated component ID is one coupled native parameter. Its components
    receive one shared multiplier, preserving the exact coefficient ratios.
    The finite portfolio is deliberately not described as exhaustive.
    """
    ids = np.asarray(ids, dtype=int)
    coefficients = np.asarray(coefficients, dtype=float)
    current = np.asarray(current, dtype=float)
    metric = np.asarray(metric, dtype=float)
    if (ids.shape != (3,) or coefficients.shape != (3,) or current.shape != (3,)
            or metric.shape != (3, 3) or type(seed) is not int or not 1 <= seed <= 2000
            or not np.isfinite([*coefficients, *current, *metric.ravel(), magnitude]).all()
            or magnitude <= 0):
        raise ValueError('Invalid independent magnetic start')

    def draw(key):
        raw = hashlib.sha256(f'{seed}:{orbit}:{key}'.encode()).digest()
        return int.from_bytes(raw[:8], 'big') / 2**64

    result = np.zeros(3)
    for mode in sorted(set(ids) - {0}):
        group = np.flatnonzero(ids == mode)
        pivot = next((i for i in group if abs(coefficients[i]) > 1e-12), None)
        if pivot is None:
            raise ValueError('An allowed magnetic basis has no nonzero coefficient')
        amplitude = current[pivot] / coefficients[pivot]
        if abs(amplitude) < 1e-10:
            amplitude = magnitude * .02 / coefficients[pivot]
        sign = -1 if draw(f'{mode}:sign') < .5 else 1
        result[group] = amplitude * sign * (.55 + .9 * draw(f'{mode}:amplitude')) * coefficients[group]
    norm_squared = float(result @ metric @ result)
    if not np.isfinite(norm_squared) or norm_squared <= 0:
        raise ValueError('An independent magnetic start has no physical ordered moment')
    # Keep both amplitude levels and variation between independent child orbits;
    # normalizing every child to one common magnitude would erase that diversity.
    radius = magnitude * (.65 + .35 * draw('orbit_amplitude'))
    return result * radius / np.sqrt(norm_squared)
