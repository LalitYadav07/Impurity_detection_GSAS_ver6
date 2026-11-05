"""Remove only demonstrated uniform-origin gauges, not weak scattering modes."""
import numpy as np
from scipy.linalg import null_space


def origin_gauges(displacements, rotations, *, tolerance=1e-9):
    """Return orthonormal native-mode bases for gauges and physical coordinates.

    displacements[atom, fractional_axis, native_mode] is obtained from native
    ApplyModeDisp. A gauge must translate every asymmetric site equally AND
    remain a uniform translation after every child point-group operation.
    No diffraction sensitivity or numerical Hessian is used to discard modes.
    """
    b = np.asarray(displacements, dtype=float)
    r = np.asarray(rotations, dtype=float)
    if b.ndim != 3 or b.shape[1] != 3 or not len(b) or not np.isfinite(b).all():
        raise ValueError('Expected finite atom-by-axis-by-mode displacement mapping')
    if r.ndim != 3 or r.shape[1:] != (3, 3) or not len(r) or not np.isfinite(r).all():
        raise ValueError('Expected finite child point-group rotations')
    n = b.shape[2]
    if not n:
        return dict(gauge_basis=np.zeros((0, 0)), physical_basis=np.zeros((0, 0)),
                    mapping_rank=0, gauge_rank=0, physical_rank=0, residual=0.)
    if np.linalg.matrix_rank(b.reshape(-1, n), tol=tolerance) != n:
        raise ValueError('Native coordinate map has an unexplained rank deficiency')
    equations = np.concatenate([(b - b[:1]).reshape(-1, n),
                                np.einsum('rij,jk->rik', r - np.eye(3), b[0]).reshape(-1, n)])
    # Absolute precision relative to the original coordinate map matters when
    # the whole equation matrix should be zero (e.g. one free P1 atom).
    u, singular, vh = np.linalg.svd(equations, full_matrices=True)
    threshold = tolerance * max(np.linalg.norm(b.reshape(-1, n), 2), 1.)
    rank = int(np.count_nonzero(singular > threshold))
    gauge = vh[rank:].T.copy()
    physical = null_space(gauge.T) if gauge.shape[1] else np.eye(n)
    residual = float(np.max(np.abs(equations @ gauge), initial=0.))
    if residual > threshold * 10:
        raise ValueError('Unstable origin-gauge decomposition')
    return dict(gauge_basis=gauge, physical_basis=physical, mapping_rank=n,
                gauge_rank=gauge.shape[1], physical_rank=physical.shape[1], residual=residual)
