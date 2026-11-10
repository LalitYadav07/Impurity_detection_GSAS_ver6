"""Positive reference-count calibration with explicit numerical convergence.

This is conditional maximum likelihood for a supplied response matrix and known
ON/OFF exposures. It does not establish physical Poisson statistics, metrology,
an absolute source/detector separation, predictive coverage or global optimality.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy import sparse
from scipy.optimize import brentq
from scipy.special import xlogy

VERSION = 'powgen-positive-reference-coordinate-polish/2'


def _background(n, m, source, qon, qoff):
    total = qon + qoff
    a = n + m - total * source
    root = np.hypot(a, 2 * np.sqrt(total * m * source))
    result = np.zeros_like(source)
    positive = a >= 0
    result[positive] = (a[positive] + root[positive]) / (2 * total)
    np.divide(2 * m * source, root - a, out=result, where=~positive)
    return result


def _poisson_loss(counts, mean):
    result = mean.copy()
    positive = counts > 0
    if np.any(positive & (mean <= 0)):
        return np.inf
    relative = np.divide(mean-counts, counts, out=np.zeros_like(mean), where=positive)
    close = positive & (abs(relative) < .5)
    result[close] = counts[close] * (relative[close] - np.log1p(relative[close]))
    far = positive & ~close
    result[far] = mean[far] - counts[far] + xlogy(counts[far], counts[far]/mean[far])
    return float(result.sum())


def _projected_step(value, gradient, root, upper=None):
    fill = np.zeros_like(gradient)
    fill[gradient > 0] = np.inf;fill[gradient < 0] = -np.inf
    scaled = np.divide(gradient, root, out=fill, where=root > 0)
    downward = np.minimum(np.maximum(scaled, 0.), value*root)
    upward = np.maximum(-scaled, 0.)
    if upper is not None:
        upward = np.minimum(upward, (upper-value)*root)
    result = np.where(gradient >= 0, downward, upward)
    kkt = (gradient == 0) | ((value == 0) & (gradient >= 0))
    if upper is not None:
        kkt |= (value == upper) & (gradient <= 0)
    return np.where(root > 0, result, np.where(kkt, 0., np.inf))


@dataclass(frozen=True)
class ReferenceCalibration:
    coefficients: np.ndarray
    angular_gain: np.ndarray
    source_rate: np.ndarray
    background_rate: np.ndarray
    metadata: dict


def polish_on_off_response(on_counts, off_counts, on_exposure, off_exposure, design,
                           observed_group_index, initial_coefficients, initial_angular,
                           *, tolerance=.05, max_iterations=100, max_bytes=512*1024**2):
    """Polish a supplied fit by exact convex one-coordinate profile minimizations.

    n~Poisson(q_on*(g*A*c+b)), m~Poisson(q_off*b). Each nonnegative source
    coefficient and angular gain is minimized with a bracketed derivative root.
    A global scale convention sets max(g)=1 without changing expected counts.
    Convergence uses feasible Newton-step distances in expected-curvature units,
    including the actual distance to a bound. No count or signal floor is used.
    Multiple initial solutions and physical model checks remain the caller's work.
    """
    n, m = np.asarray(on_counts, float).ravel(), np.asarray(off_counts, float).ravel()
    if (n.shape != m.shape or not 1 <= len(n) <= 1000000
            or any(not np.isfinite(v).all() or np.any(v < 0) or np.any(v > 2**53)
                   or not np.array_equal(v, np.rint(v)) for v in (n, m))):
        raise ValueError('Matching finite nonnegative integer ON/OFF counts are required')
    exposures = (on_exposure, off_exposure, tolerance)
    if any(isinstance(v, (bool, np.bool_)) or not np.isscalar(v) or not math.isfinite(float(v)) or v <= 0 for v in exposures):
        raise ValueError('Known positive exposures and a positive convergence tolerance are required')
    qon, qoff = float(on_exposure), float(off_exposure)
    matrix = sparse.csr_matrix(design, dtype=float)
    matrix.sum_duplicates();matrix.sort_indices()
    if (matrix.shape[0] != len(n) or not 1 <= matrix.shape[1] <= 4097
            or not np.isfinite(matrix.data).all() or np.any(matrix.data < 0)):
        raise ValueError('The physical response design must be finite, nonnegative and match counts')
    groups = np.asarray(observed_group_index)
    if groups.shape != n.shape or groups.dtype.kind not in 'iu' or np.any(groups < 0) or groups.max() >= 2048:
        raise ValueError('An explicit bounded observed-group index is required for each row')
    ngroup = int(groups.max()) + 1
    coefficients, angular = np.array(initial_coefficients, float, copy=True), np.array(initial_angular, float, copy=True)
    if (coefficients.shape != (matrix.shape[1],) or angular.shape != (ngroup,)
            or any(not np.isfinite(v).all() or np.any(v < 0) for v in (coefficients, angular))
            or angular.max() <= 0
            or np.any((np.bincount(groups, weights=n, minlength=ngroup) > 0) & (angular == 0))):
        raise ValueError('Matching nonnegative starts and positive gains in observed groups are required')
    if (type(max_iterations) is not int or not 1 <= max_iterations <= 1000 or type(max_bytes) is not int
            or matrix.nnz*64 + len(n)*512 + matrix.shape[1]*128 > max_bytes):
        raise ValueError('The calibration exceeds its explicit iteration or working memory budget')
    scale = angular.max();angular /= scale;coefficients *= scale
    columns = matrix.tocsc();group_rows = [np.flatnonzero(groups == group) for group in range(ngroup)]
    trace = [];maximum_roundoff = 0.

    def coordinate(rows, values, other):
        total = values.sum()
        if total <= 0:
            raise ValueError('A coordinate root needs a positive design direction')
        normalized = values/total
        def derivative(rate):
            source = other + values*rate
            background = _background(n[rows], m[rows], source, qon, qoff)
            ratio = np.divide(n[rows], source+background, out=np.zeros_like(source), where=(source+background) > 0)
            return 1 - float(normalized@ratio)/qon
        if derivative(0.) >= 0:
            return 0.
        # Since other and background are nonnegative, this upper endpoint has
        # normalized derivative >=1/2. It avoids weak-column expansion heuristics.
        upper = 2*float(n[rows].sum())/(qon*total)
        if not math.isfinite(upper) or upper <= 0 or derivative(upper) < 0:
            raise ValueError('Coordinate root is outside the finite calibration domain')
        return upper*brentq(lambda unit: derivative(upper*unit), 0., 1., xtol=1e-13, rtol=1e-13)

    for iteration in range(max_iterations):
        gain = angular[groups]
        source = gain*np.asarray(matrix@coefficients)
        if not np.isfinite(source).all():
            raise ValueError('Calibration scaling exceeds its finite domain')
        for column in range(matrix.shape[1]):
            first, last = columns.indptr[column:column+2]
            rows = columns.indices[first:last];values = columns.data[first:last]*gain[rows]
            active = values > 0;rows, values = rows[active], values[active]
            if not len(rows):
                continue  # No data direction: retain its explicitly supplied value.
            contribution = values*coefficients[column]
            other = source[rows]-contribution
            error = max(0., -float(other.min()))
            allowed = 128*np.finfo(float).eps*max(float(abs(source[rows]).max()), float(abs(contribution).max()))
            if error > allowed:
                raise ValueError('Coordinate subtraction exceeds the declared floating-roundoff bound')
            maximum_roundoff = max(maximum_roundoff, error)
            other = np.maximum(other, 0.)  # Only the checked subtraction roundoff.
            coefficients[column] = coordinate(rows, values, other)
            source[rows] = other + values*coefficients[column]
        base = np.asarray(matrix@coefficients)
        for group, rows in enumerate(group_rows):
            if len(rows) and base[rows].sum() > 0:
                angular[group] = coordinate(rows, base[rows], np.zeros(len(rows)))
        scale = angular.max()
        if scale <= 0:
            raise ValueError('Calibration has no defined positive angular scale')
        angular /= scale;coefficients *= scale
        gain = angular[groups];source = gain*np.asarray(matrix@coefficients)
        background = _background(n, m, source, qon, qoff)
        loss = _poisson_loss(n, qon*(source+background)) + _poisson_loss(m, qoff*background)
        residual = qon-np.divide(n, source+background, out=np.zeros_like(source), where=(source+background) > 0)
        variance = (source+background)/qon + background/qoff
        weight = np.divide(1., variance, out=np.zeros_like(variance), where=variance > 0)
        gradient = np.asarray(matrix.T@(gain*residual))
        curvature = np.asarray(matrix.multiply(gain[:, None]).power(2).T@weight)
        source_score = float(_projected_step(coefficients, gradient, np.sqrt(curvature)).max())
        base = np.asarray(matrix@coefficients)
        ag = np.bincount(groups, weights=base*residual, minlength=ngroup)
        ac = np.bincount(groups, weights=base**2*weight, minlength=ngroup)
        angular_score = float(_projected_step(angular, ag, np.sqrt(ac), 1.).max())
        trace.append(dict(iteration=iteration,profile_deviance_half=loss,
            maximum_source_projected_step=source_score,maximum_angular_projected_step=angular_score))
        if max(source_score, angular_score) <= tolerance:
            break
    for value in (coefficients, angular, source, background):value.setflags(write=False)
    zero_columns = np.flatnonzero(np.asarray(matrix.sum(axis=0)).ravel() == 0)
    zero_effective_columns = np.flatnonzero(np.asarray(matrix.multiply(angular[groups, None]).sum(axis=0)).ravel() == 0)
    return ReferenceCalibration(coefficients, angular, source, background, dict(version=VERSION,
        converged_numerically=max(source_score, angular_score) <= tolerance,iterations=len(trace),
        maximum_source_projected_step=source_score,maximum_angular_projected_step=angular_score,
        tolerance=float(tolerance),profile_deviance_half=loss,trace=trace,
        maximum_checked_roundoff_clipping_in_source_rate=maximum_roundoff,
        source_count_floor_added=False,gauge='maximum angular gain=1',global_optimality_established=False,
        zero_response_source_indices=zero_columns.tolist(),
        zero_final_gain_response_source_indices=zero_effective_columns.tolist(),
        zero_response_coefficients_role='Not inferred; supplied values carried through count-preserving gain-gauge changes',
        source_parameter_identifiability_established=False,
        physical_Poisson_stationarity_verified=False,experimental_accuracy_certified=False))
