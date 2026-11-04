"""Reproducible independent-orbit starts inside exact site-symmetry bases."""
import hashlib

import numpy as np


def site_moment_start(ids, coefficients, basis_ids, refined_axes, *, magnitude,
                      axis, direction, site_sign, symmetry_allowed,
                      independent_seed=None, orbit=None, metric=None):
    """Construct the same initial site moment for native and grouped workers."""
    indices = {'x': 0, 'y': 1, 'z': 2}
    initial = np.array([magnitude if name == axis else 0. for name in 'xyz'])
    moment = np.zeros(3, dtype=float)
    for basis_id in sorted(basis_ids):
        group = [i for i, value in enumerate(ids) if int(value) == basis_id]
        requested = [i for i in group if 'xyz'[i] in refined_axes]
        pivot = requested[0] if requested else group[0]
        coefficient = float(coefficients[pivot])
        if abs(coefficient) < 1e-12:
            raise ValueError('GSAS-II returned a zero magnetic-basis coefficient')
        amplitude = float(initial[pivot])
        if symmetry_allowed and abs(amplitude) < 1e-8:
            amplitude = float(magnitude)*.02
        if direction is not None:
            component = float(direction[pivot])
            if abs(component) < 1e-8:
                component = .02
            amplitude = float(np.linalg.norm(initial))*component
        for i in group:
            moment[i] = amplitude*float(coefficients[i])/coefficient
    if not any(abs(moment[indices[name]]) >= 1e-8 for name in refined_axes):
        raise ValueError('A refined magnetic component would start at zero')
    moment *= site_sign
    if symmetry_allowed and independent_seed is not None:
        moment = independent_orbit_start(ids, coefficients, moment, metric,
            magnitude=magnitude, seed=independent_seed, orbit=orbit)
    return moment


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
