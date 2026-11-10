"""Bounded numerical sensitivity of a fixed linear response.

Column scaling changes only this diagnostic's coordinates. It does not change
the response, fitted coefficients, counts or missing-domain rates. Full rank
conditional on a response does not identify source fluence separately from an
unknown monitor area, detector gain, specimen amount or physical discrepancy.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy import sparse

VERSION = 'powgen-fixed-response-design-sensitivity/1'


@dataclass(frozen=True)
class ResponseDesignSensitivity:
    singular_values: np.ndarray
    column_l2_scales: np.ndarray
    metadata: dict


def analyze_response_design(matrix, *, relative_rank_tolerance=None,
                            max_bytes=128 * 1024**2):
    """Report column-normalized SVD rank for a supplied nonnegative response.

The tolerance is relative to the largest singular value; its default is
max(rows, columns)*float64 epsilon. Rank is a numerical diagnostic at that
declared tolerance, not an exact algebraic proof, a statistical confidence
interval or identification of a nonlinear calibration. A nonnegative solution
on a boundary can have different identifiability from this unrestricted linear
map. No singular modes are removed and no regularization is applied.

The explicit working-storage estimate is conservative for the arrays used
here; native LAPACK/runtime overhead is separate. Large matrices raise before
densification. Supply an ndarray or a scipy sparse matrix, not nested lists.
    """
    if (not isinstance(matrix, np.ndarray) and not sparse.issparse(matrix)):
        raise ValueError('Supply an explicit ndarray or scipy sparse response')
    if (matrix.ndim != 2 or not 1 <= matrix.shape[0] <= 100000
            or not 1 <= matrix.shape[1] <= 4097 or matrix.dtype.kind not in 'iuf'):
        raise ValueError('A bounded real two-dimensional response is required')
    if type(max_bytes) is not int or not 1024 <= max_bytes <= 2 * 1024**3:
        raise ValueError('An explicit bounded working-memory budget is required')
    rows, columns = matrix.shape
    small = min(rows, columns)
    estimate = 8 * (4 * rows * columns + 12 * small**2 + 32 * max(rows, columns))
    if sparse.issparse(matrix):
        estimate += 32 * matrix.nnz
    if estimate > max_bytes:
        raise ValueError('Response sensitivity exceeds its working-memory budget')
    if relative_rank_tolerance is None:
        tolerance = max(rows, columns) * np.finfo(float).eps
    else:
        if (isinstance(relative_rank_tolerance, (bool, np.bool_, str, bytes, complex, np.complexfloating))
                or not np.isscalar(relative_rank_tolerance)):
            raise ValueError('Rank tolerance must be a finite relative scalar')
        tolerance = float(relative_rank_tolerance)
        if not math.isfinite(tolerance) or not 0 < tolerance < 1:
            raise ValueError('Rank tolerance must lie strictly between zero and one')
    if sparse.issparse(matrix):
        # COO exposes numeric entries consistently for CSR/CSC/BSR and the
        # dictionary/list formats, while preserving stored duplicates.
        values = matrix.tocoo(copy=False).data
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Response entries must be finite and nonnegative')
        dense = matrix.astype(float).toarray()
    else:
        dense = np.array(matrix, dtype=float, copy=True)
    if not np.isfinite(dense).all() or np.any(dense < 0):
        raise ValueError('Response entries must be finite and nonnegative')
    # A maximum first avoids overflowing squares for a large finite column.
    maxima = dense.max(axis=0)
    detected = maxima > 0
    np.divide(dense, maxima, out=dense, where=detected[None, :])
    lengths = np.linalg.norm(dense, axis=0)
    np.divide(dense, lengths, out=dense, where=detected[None, :])
    with np.errstate(over='ignore'):
        scales = maxima * lengths
    # The diagnostic is still well-defined when the dimensional L2 norm is
    # larger than float64. Do not return an unrepresentable scale as infinity.
    if not np.isfinite(scales).all():
        raise ValueError('Dimensional response column scales exceed float64')
    singular = np.linalg.svd(dense, compute_uv=False)
    threshold = float(tolerance * singular[0])
    rank = int(np.count_nonzero(singular > threshold))
    condition = float(singular[0] / singular[rank-1]) if rank else None
    for value in (singular, scales):
        value.setflags(write=False)
    return ResponseDesignSensitivity(singular, scales, dict(
        version=VERSION, rows=rows, source_parameters=columns,
        numerical_rank=rank, source_nullity=columns-rank,
        full_column_rank=rank == columns,
        zero_response_source_indices=np.flatnonzero(~detected).tolist(),
        column_scaling='positive L2; diagnostic coordinates only',
        relative_rank_tolerance=tolerance,
        singular_value_threshold=threshold,
        condition_number_on_detectable_subspace=condition,
        estimated_working_array_bytes=estimate,
        working_memory_budget_bytes=max_bytes,
        fixed_response_linear_map_only=True,
        physical_source_response_separation_established=False,
        parameter_uncertainty_or_predictive_coverage_established=False,
        regularization_or_mode_removal_applied=False,
        experimental_accuracy_certified=False))
