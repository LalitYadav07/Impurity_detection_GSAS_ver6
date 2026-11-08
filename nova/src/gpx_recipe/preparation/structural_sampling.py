"""Physical, trajectory-grouped examples for a compound-specific ranker.

These points are hypotheses to simulate, not a surrogate or fitted result.
The sampler retains every coordinate in a candidate. It balances displacement
RMS and relative reciprocal-metric norms instead of giving more complicated
children systematically larger distortions. Native symmetry and feasibility
remain the responsibilities of the supplied physical geometry and callback.
"""
from dataclasses import dataclass
import hashlib
import numpy as np

from ..structural_live.optimization.physical_prior_geometry import (
    displacement_gram, native_reciprocal_matrix, orthonormal_lift,
    reciprocal_metric_gram,
)
from ..structural_live.optimization.chemical_geometry import direct_metric_from_reciprocal


@dataclass(frozen=True)
class TrajectorySamples:
    points: np.ndarray
    trajectory: np.ndarray
    displacement_rms: np.ndarray
    relative_metric_norm: np.ndarray
    rejected_directions: int


def _interval(value, name):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(v) not in (int, float) for v in value)
            or not np.isfinite(value).all() or not 0 < value[0] <= value[1]):
        raise ValueError(f'Positive finite {name} interval required')
    return tuple(map(float, value))


def _direction(rng, dimensions, sparse):
    if not dimensions:
        return np.empty(0)
    direction = np.zeros(dimensions)
    if sparse:
        active = rng.choice(dimensions, size=min(sparse, dimensions), replace=False)
        direction[active] = rng.normal(size=len(active))
    else:
        direction[:] = rng.normal(size=dimensions)
    return direction / np.linalg.norm(direction)


def draw_trajectories(mode_gram, cell_gram, bounds, *, seed, trajectories=64,
                      steps=4, displacement_range=(.0005, .04),
                      relative_metric_range=(.0001, .01), feasible=None,
                      trajectory_feasible=None, maximum_attempts=1000):
    """Draw independent directions, with nested amplitudes kept in one group.

    Intervals describe maximum trajectory amplitudes. Each trajectory contains
    ``steps`` amplitudes from one tenth of that maximum to the maximum. An
    explicit parent/null example is generated separately by the dataset owner;
    it must not be duplicated under every child's compulsory class label.
    Relative metric norm is NOT the exact finite real-space strain.
    An indexed feasibility callback can additionally check a trajectory in
    its already-declared shared parent context. Rejections redraw the whole
    child trajectory; they cannot change that parent context or its identity.
    """
    for value, limit, name in [(seed, 2**32-1, 'seed'),
                               (trajectories, 4096, 'trajectory count'),
                               (steps, 32, 'steps'),
                               (maximum_attempts, 100000, 'attempt count')]:
        minimum = 0 if name == 'seed' else 1
        if type(value) is not int or not minimum <= value <= limit:
            raise ValueError(f'Bounded integer {name} required')
    if trajectories*steps > 32768:
        raise ValueError('Generate the example trajectories in smaller chunks')
    low_mode, high_mode = _interval(displacement_range, 'RMS displacement')
    low_cell, high_cell = _interval(relative_metric_range, 'relative metric')
    mg = np.asarray(mode_gram, float); cg = np.asarray(cell_gram, float)
    if (mg.ndim != 2 or mg.shape[0] != mg.shape[1]
            or cg.ndim != 2 or cg.shape[0] != cg.shape[1] or not len(cg)):
        raise ValueError('Square displacement and nonempty cell Gram matrices required')
    # This existing transformation rejects unresolved rank; it never silently
    # drops a physical coordinate to make the classification problem easier.
    mode_lift = orthonormal_lift(mg); cell_lift = orthonormal_lift(cg)
    nm, nc = len(mg), len(cg)
    bounds = np.asarray(bounds, float)
    if (bounds.shape != (nm+nc,) or not np.isfinite(bounds).all()
            or np.any(bounds <= 0)):
        raise ValueError('Positive finite per-coordinate bounds required')
    if trajectories*steps*(nm+nc)*8 > 64*1024**2:
        raise ValueError('Generate the example trajectories in smaller chunks')
    if any(check is not None and not callable(check) for check in (feasible,trajectory_feasible)):
        raise TypeError('The physical feasibility checks must be callable')
    rng = np.random.default_rng(seed); fractions = np.geomspace(.1, 1., steps)
    points = []; labels = []; mode_norms = []; cell_norms = []; rejected = 0
    for index in range(trajectories):
        # Pure displacement, pure lattice, and mixed trajectories. Each block
        # cycles through sparse and collective directions in physical norm.
        kind = index % 3 if nm else 1
        sparse = (1, 2, 0)[(index//3) % 3]
        for _ in range(maximum_attempts):
            amplitude = np.exp(rng.uniform(np.log(low_mode), np.log(high_mode)))
            metric = np.exp(rng.uniform(np.log(low_cell), np.log(high_cell)))
            modes = (mode_lift@_direction(rng, nm, sparse)*amplitude
                     if nm and kind != 1 else np.zeros(nm))
            cells = (cell_lift@_direction(rng, nc, sparse)*metric
                     if kind != 0 else np.zeros(nc))
            block = fractions[:, None]*np.r_[modes, cells]
            allowed = np.all(abs(block) <= bounds)
            if allowed and feasible is not None:
                mask = np.asarray(feasible(block))
                if mask.shape != (steps,) or mask.dtype != bool:
                    raise ValueError('Feasibility must return one boolean per configuration')
                allowed = bool(mask.all())
            if allowed and trajectory_feasible is not None:
                mask = np.asarray(trajectory_feasible(index,block))
                if mask.shape != (steps,) or mask.dtype != bool:
                    raise ValueError('Trajectory feasibility must return one boolean per configuration')
                allowed = bool(mask.all())
            if allowed:
                points.extend(block); labels.extend([index]*steps)
                mode_norms.extend(np.sqrt(np.maximum(np.einsum('ni,ij,nj->n', block[:, :nm], mg, block[:, :nm]), 0.)))
                cell_norms.extend(np.sqrt(np.maximum(np.einsum('ni,ij,nj->n', block[:, nm:], cg, block[:, nm:]), 0.)))
                break
            rejected += 1
        else:
            raise ValueError('Could not complete a physical trajectory inside the declared bounds; revise its scope, not its label')
    arrays = [np.asarray(points, float), np.asarray(labels, np.int64),
              np.asarray(mode_norms), np.asarray(cell_norms)]
    for array in arrays:
        array.flags.writeable = False
    return TrajectorySamples(*arrays, rejected)


def from_source(source, **settings):
    """Use the same complete, symmetry-constrained native geometry as fitting."""
    geometry = source.fast.empirical_constraint.geometry
    nm = source.mode_dimensions
    reciprocal = native_reciprocal_matrix(geometry.a0)
    direct = direct_metric_from_reciprocal(reciprocal)
    mapping = np.einsum('aim,mk->aik', geometry.mapping, source.lift[:nm, :nm])
    mg = displacement_gram(mapping, direct, geometry.occupancy)
    directions = geometry.cell_mapping@source.lift[nm:, nm:]
    cg = reciprocal_metric_gram(reciprocal, np.asarray([native_reciprocal_matrix(v) for v in directions.T]))
    return draw_trajectories(mg, cg, source.radii,
        feasible=lambda rows: np.isfinite(source.chemistry.log_kernel(rows)), **settings)


def partition_trajectories(groups, *, seed=3917):
    """Assign balanced, nonempty holdouts after the complete cohort is known.

    Exact duplicate configurations must first connect their whole trajectory
    groups. This method uses group identities, never spectra or class labels,
    to select the holdout. No label or amplitude step can change its split.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('A bounded integer split seed is required')
    groups = list(groups)
    if (len(groups) > 1000000 or any(not isinstance(g, str) or not g or len(g) > 500 for g in groups)):
        raise ValueError('Bounded nonempty physical-trajectory identities required')
    unique = sorted(set(groups), key=lambda g: hashlib.sha256(f'{seed}:{g}'.encode()).digest())
    if len(unique) < 3:
        raise ValueError('At least three independent trajectory groups are required for holdouts')
    holdout = max(1, int(np.floor(.15*len(unique)+.5)))
    return {g: ('test' if i < holdout else 'validation' if i < 2*holdout else 'train')
            for i, g in enumerate(unique)}
