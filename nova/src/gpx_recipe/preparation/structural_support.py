"""Empirical synthetic-domain support; never a confidence guarantee.

The compact index uses training examples only. Its admission limit is the
largest validation distance. Exchangeability, experimental uncertainty and
unknown-phase coverage are not established by that empirical limit.
"""
import numpy as np

MAX_BYTES = 16*1024**2
STREAM_METHOD = 'training-stratified-pool-k-center/validation-maximum-rms/v1'


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


def _prototypes(matrix, maximum_anchors):
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
    return matrix[chosen].copy()


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
    prototypes = _prototypes(matrix, maximum_anchors)
    held_out = distances(prototypes, x[validation].reshape(np.count_nonzero(validation), -1))
    return dict(prototypes=prototypes, limit=float(held_out.max()),
        method='training-k-center/validation-maximum-rms/v1',
        validation_rows=len(held_out), validation_distances=held_out,
        statistical_coverage_qualified=False)


def fit_batches(rows, training, validation, targets, mean, scale, *, seed=89113):
    """Bounded training-only anchor pool, with every validation row inspected.

    At least one example of each compatible training label enters the pool.
    Remaining members use a deterministic random training subset; neither
    validation nor test examples select anchors. This empirical envelope is
    not an experimental-coverage or confidence certificate.
    """
    from .structural_training_data import batches
    training = np.asarray(training); validation = np.asarray(validation)
    if (training.ndim != 1 or validation.ndim != 1 or not len(training) or not len(validation)
            or training.dtype.kind not in 'iu' or validation.dtype.kind not in 'iu'
            or np.intersect1d(training, validation).size):
        raise ValueError('Independent training and validation support rows are required')
    capacity = min(2048, 128*1024**2//(rows.shape[1]*rows.shape[2]*4))
    rng = np.random.default_rng(seed); order = rng.permutation(training)
    chosen = []; covered = np.zeros(targets.shape[1], bool)
    for index in order:
        if np.any(targets[index] & ~covered):
            chosen.append(int(index)); covered |= targets[index]
    if not covered.all() or len(chosen) > capacity:
        raise ValueError('Training-support pool cannot cover the declared classes within its budget')
    selected = set(chosen)
    for index in order:
        if len(chosen) >= min(capacity, len(training)):
            break
        if int(index) not in selected:
            chosen.append(int(index)); selected.add(int(index))
    pool = np.empty((len(chosen), rows.shape[1]*rows.shape[2]), np.float32)
    position = 0
    for _, values in batches(rows, np.asarray(chosen, np.int64)):
        pool[position:position+len(values)] = ((values-mean)/scale).reshape(len(values), -1)
        position += len(values)
    prototypes = _prototypes(pool, 128)
    limit = 0.
    for _, values in batches(rows, validation):
        result = distances(prototypes, ((values-mean)/scale).reshape(len(values), -1))
        limit = max(limit, float(result.max()))
    return dict(prototypes=prototypes, limit=limit, method=STREAM_METHOD,
                validation_rows=len(validation), training_pool_rows=len(pool),
                statistical_coverage_qualified=False)
