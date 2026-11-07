"""Empirical synthetic-domain support; never a confidence guarantee.

The compact index uses training examples only. Its admission limit is the
largest validation distance. Exchangeability, experimental uncertainty and
unknown-phase coverage are not established by that empirical limit.
"""
import numpy as np

MAX_BYTES = 16*1024**2


def distances(prototypes, values):
    """Nearest prototype RMS distance in the model's normalized feature space."""
    p, x = np.asarray(prototypes), np.asarray(values)
    if (p.ndim != 2 or x.ndim != 2 or not len(p) or not p.shape[1] or p.shape[1] != x.shape[1]
            or p.dtype != np.float32 or x.dtype != np.float32
            or p.nbytes > MAX_BYTES or not np.isfinite(p).all() or not np.isfinite(x).all()):
        raise ValueError('Finite bounded support prototypes and matching features are required')
    # Float64 accumulation avoids subtracting nearly equal float32 norms when
    # a query is itself a prototype. Working memory is bounded by row blocks.
    p = p.astype(np.float64); squared = np.einsum('ij,ij->i', p, p)
    result = []
    for start in range(0, len(x), 32):
        block = x[start:start+32].astype(np.float64)
        delta = np.maximum(np.einsum('ij,ij->i', block, block)[:, None]+squared[None, :]-2*block@p.T, 0.)
        result.extend(np.sqrt(delta.min(1)/p.shape[1]))
    return np.asarray(result)


def fit(values, training, validation, *, maximum_anchors=128):
    """Farthest-first training support, with no labels or test examples used."""
    x = np.asarray(values); training = np.asarray(training); validation = np.asarray(validation)
    if (x.ndim != 3 or not x.shape[1] or not x.shape[2] or x.dtype != np.float32 or not np.isfinite(x).all()
            or x.nbytes > 384*1024**2 or training.dtype != bool or validation.dtype != bool
            or training.shape != (len(x),) or validation.shape != training.shape
            or np.any(training & validation) or not training.any() or not validation.any()
            or type(maximum_anchors) is not int or not 1 <= maximum_anchors <= 256):
        raise ValueError('Independent training and validation support rows are required')
    matrix = x[training].reshape(np.count_nonzero(training), -1)
    count = min(maximum_anchors, len(matrix), MAX_BYTES//(matrix.shape[1]*4))
    if not count:
        raise ValueError('One support prototype must fit the numerical budget')
    wide = matrix.astype(np.float64); norms = np.einsum('ij,ij->i', wide, wide)
    center = wide.mean(0)
    index = int(np.argmin(norms-2*wide@center))
    minimum = np.full(len(matrix), np.inf); chosen = []
    for _ in range(count):
        chosen.append(index)
        delta = np.maximum(norms+norms[index]-2*wide@wide[index], 0.)
        minimum = np.minimum(minimum, delta); minimum[chosen] = -np.inf
        index = int(np.argmax(minimum))
    prototypes = matrix[chosen].copy()
    held_out = distances(prototypes, x[validation].reshape(np.count_nonzero(validation), -1))
    return dict(prototypes=prototypes, limit=float(held_out.max()),
        method='training-k-center/validation-maximum-rms/v1',
        validation_rows=len(held_out), validation_distances=held_out,
        statistical_coverage_qualified=False)
