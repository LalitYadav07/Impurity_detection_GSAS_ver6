"""Joint positive detector/reference/monitor count likelihood.

Every block is a beam-ON count stream with its own signal exposure and either
an explicit known-background mean hypothesis or a separately exposed OFF
reference. Signal and background exposures need not use the same units. A
blank specimen's upstream monitor is another beam-ON block, never detector-OFF
data. Conditional Poisson likelihood does not establish physical independence,
stationarity, specimen metrology, complete uncertainty or experimental accuracy.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy import sparse
from scipy.optimize import brentq
from .reference_calibration import _projected_step

VERSION = 'powgen-joint-positive-count-calibration/5'


@dataclass(frozen=True)
class CountBlock:
    name: str
    role: str
    counts: np.ndarray
    design: sparse.csr_matrix
    source_exposure: float
    source_exposure_unit: str
    gain_index: np.ndarray
    background_role: str
    known_background_mean: np.ndarray | None = None
    off_counts: np.ndarray | None = None
    background_on_exposure: float | None = None
    background_off_exposure: float | None = None
    background_exposure_unit: str | None = None
    off_role: str | None = None


@dataclass(frozen=True)
class JointCountCalibration:
    coefficients: np.ndarray
    gains: np.ndarray
    block_source_means: tuple
    block_background_means: tuple
    block_background_rates: tuple
    metadata: dict


def _positive(value, name):
    if isinstance(value, (bool, np.bool_, str, bytes)) or not np.isscalar(value):
        raise ValueError(name+' must be a finite positive scalar')
    value=float(value)
    if not math.isfinite(value) or value<=0:
        raise ValueError(name+' must be a finite positive scalar')
    return value


def _label(value, name):
    if type(value) is not str or not value.strip() or len(value) > 512:
        raise ValueError(name+' requires an explicit bounded label')
    return value


def _counts(value, name):
    result = np.array(value, float, copy=True)
    if (result.ndim != 1 or not 1 <= len(result) <= 1000000 or not np.isfinite(result).all()
            or np.any(result < 0) or np.any(result > 2**53)
            or not np.array_equal(result, np.rint(result))):
        raise ValueError(name+' must be a bounded vector of nonnegative integer counts')
    return result


def _profile_background_mean(n, m, source_mean, off_on_exposure_ratio):
    """ON background mean for n~Pois(s+v), m~Pois(r*v), r>0.

    Scaling the quadratic before its root avoids products r*s and m*s. Its
    negative-root branch uses dimensionless ratios to avoid cancellation.
    """
    total = 1 + off_on_exposure_ratio
    a = (n+m)/total-source_mean
    p = m/total
    root = np.hypot(a, 2*np.sqrt(p)*np.sqrt(source_mean))
    result = np.empty_like(source_mean)
    positive = a >= 0
    result[positive] = (a[positive]+root[positive])/2
    negative = ~positive
    result[negative] = p[negative] * (2/(root[negative]/source_mean[negative]
                                                    - a[negative]/source_mean[negative]))
    return result


def _count_loss(counts, mean):
    """Poisson deviance/2, with no count floor or unsafe n/mean logarithm."""
    if not np.isfinite(mean).all() or np.any(mean < 0):
        raise ValueError('Expected counts must be finite and nonnegative')
    positive = counts > 0
    if np.any(positive & (mean == 0)):
        return math.inf
    result = mean.copy()
    relative = np.divide(mean-counts, counts, out=np.zeros_like(mean), where=positive)
    close = positive & (abs(relative) < .5)
    result[close] = counts[close]*(relative[close]-np.log1p(relative[close]))
    far = positive & ~close
    result[far] = mean[far]-counts[far]+counts[far]*(np.log(counts[far])-np.log(mean[far]))
    return float(result.sum())


def _ratio(n, mean):
    result = np.where(n > 0, np.inf, 0.)
    # A derivative may tend to infinity as a positive-count mean tends to zero.
    # Overflow here means that boundary limit; it is not a floor on any mean.
    with np.errstate(over='ignore'):
        np.divide(n, mean, out=result, where=mean > 0)
    return result


def _root_squared_sum(values):
    """Euclidean norm without squaring an extreme finite tail or reciprocal."""
    scale=float(np.max(abs(values),initial=0.))
    if not math.isfinite(scale):raise ValueError('Expected-curvature root exceeds its finite numerical domain')
    if scale==0:return 0.
    result=scale*math.sqrt(float(np.sum((values/scale)**2)))
    if not math.isfinite(result):raise ValueError('Expected-curvature root exceeds its finite numerical domain')
    return result


def polish_joint_counts(blocks, initial_coefficients, initial_gains, *,
        gain_upper_bounds=None, fixed_gains=None, gauge_gain_indices=None,
        tolerance=.05, max_iterations=400, max_bytes=512*1024**2):
    """Coordinate maximum likelihood for supplied shared source responses.

    Source mean in block b is q_b * gain[row] * design_b * coefficients.
    A profiled OFF reference uses its own background ON/OFF exposures, in the
    explicitly named common background unit, independently of signal exposure.
    A known-background block instead supplies count means, not pseudo-data.

    Gains may be fixed or bounded. An all-free relative calibration requires
    explicit detector gain indices with upper bounds one; their maximum sets a
    count-preserving gauge, including inverse rescaling of free monitor gains.
    Fixed positive gain responses can anchor the supplied absolute model, with
    no gauge normalization. Their physical calibration is not proved here.

    Each coordinate is minimized by a bounded monotone derivative root. The
    stopping score is a feasible projected Newton displacement in conditional
    expected-curvature units. It proves numerical stationarity only, not global
    optimality, parameter identifiability or posterior/predictive coverage.
    """
    if not isinstance(blocks, (list, tuple)) or not 1 <= len(blocks) <= 256:
        raise ValueError('Supply bounded explicit count blocks')
    if (type(max_iterations) is not int or not 1 <= max_iterations <= 1000
            or type(max_bytes) is not int or max_bytes < 1024):
        raise ValueError('Explicit bounded iteration and working memory budgets are required')
    tol = _positive(tolerance, 'Convergence tolerance')
    coefficients = np.array(initial_coefficients, float, copy=True)
    gains = np.array(initial_gains, float, copy=True)
    if (coefficients.ndim != 1 or not 1 <= len(coefficients) <= 4097 or gains.ndim != 1
            or not 1 <= len(gains) <= 2048
            or any(not np.isfinite(v).all() or np.any(v < 0) for v in (coefficients, gains))):
        raise ValueError('Bounded matching finite nonnegative source/gain starts are required')
    fixed = np.zeros(len(gains), bool) if fixed_gains is None else np.asarray(fixed_gains)
    upper = np.full(len(gains), np.inf) if gain_upper_bounds is None else np.asarray(gain_upper_bounds, float)
    if (fixed.shape != gains.shape or fixed.dtype.kind != 'b' or upper.shape != gains.shape
            or np.isnan(upper).any() or np.any(upper <= 0) or np.any(gains > upper)
            or np.any((gains == 0) & ~fixed)):
        raise ValueError('Gains need valid bounds, an explicit boolean fixed mask and positive free starts')
    matrices=[];all_n=[];all_m=[];all_groups=[];all_known=[];all_profile=[];all_ratio=[]
    slices=[];records=[];total_rows=0;total_nnz=0;names=set();detector_groups=set()
    for block in blocks:
        if not isinstance(block, CountBlock):
            raise ValueError('Each stream must be an explicit CountBlock')
        name = _label(block.name, 'Count-block name')
        if name in names or block.role not in ('detector', 'incident_monitor_beam_on'):
            raise ValueError('Count-block names must be unique and roles explicit')
        names.add(name)
        _label(block.source_exposure_unit, 'Source-exposure unit')
        _label(block.background_role, 'Background model/provenance role')
        n = _counts(block.counts, name+' ON counts')
        q = _positive(block.source_exposure, name+' signal exposure')
        matrix = sparse.csr_matrix(block.design, dtype=float, copy=True)
        matrix.sum_duplicates();matrix.sort_indices();matrix.eliminate_zeros()
        groups = np.asarray(block.gain_index)
        if (matrix.shape != (len(n), len(coefficients)) or not np.isfinite(matrix.data).all()
                or np.any(matrix.data < 0) or groups.shape != n.shape or groups.dtype.kind not in 'iu'
                or np.any(groups < 0) or np.any(groups >= len(gains))):
            raise ValueError('Each count block needs a matching positive response and explicit valid gain indices')
        if block.role == 'detector':detector_groups.update(groups.tolist())
        matrix.data *= q
        matrix.eliminate_zeros()
        if not np.isfinite(matrix.data).all():
            raise ValueError('Exposure-scaled response exceeds its finite count domain')
        known_mode = block.known_background_mean is not None
        if known_mode:
            if any(x is not None for x in (block.off_counts, block.background_on_exposure,
                    block.background_off_exposure, block.background_exposure_unit, block.off_role)):
                raise ValueError('A known background hypothesis must not contain artificial OFF data')
            known = np.array(block.known_background_mean, float, copy=True)
            if known.shape != n.shape or not np.isfinite(known).all() or np.any(known < 0):
                raise ValueError('Known background hypotheses must be matching nonnegative count means')
            m = np.zeros_like(n);profile=np.zeros(len(n), bool);ratio=np.ones(len(n))
            background_exposure = None
        else:
            m = _counts(block.off_counts, name+' OFF counts')
            if m.shape != n.shape:
                raise ValueError('Background reference and ON counts must have matching bins')
            ton = _positive(block.background_on_exposure, name+' background ON exposure')
            toff = _positive(block.background_off_exposure, name+' background OFF exposure')
            background_exposure = _label(block.background_exposure_unit, 'Background-exposure unit')
            if block.off_role not in ('beam_off','dark','empty_instrument','empty_container'):
                raise ValueError('An OFF reference needs an explicit physical role')
            if block.role == 'incident_monitor_beam_on' and block.off_role not in ('beam_off','dark'):
                raise ValueError('A blank specimen monitor still sees beam; it cannot be monitor-OFF data')
            r = toff/ton
            if not math.isfinite(r) or r <= 0:
                raise ValueError('Background exposure ratio exceeds its finite numerical domain')
            known=np.zeros_like(n);profile=np.ones(len(n), bool);ratio=np.full(len(n),r)
        capacity=total_rows+len(n);nnz=total_nnz+matrix.nnz
        if capacity>1000000 or nnz*96+capacity*512+len(coefficients)*256+len(gains)*128>max_bytes:
            raise ValueError('Joint count calibration exceeds its working memory/row budget')
        slices.append(slice(total_rows,capacity));total_rows=capacity;total_nnz=nnz
        matrices.append(matrix);all_n.append(n);all_m.append(m);all_groups.append(groups)
        all_known.append(known);all_profile.append(profile);all_ratio.append(ratio)
        records.append(dict(name=name,role=block.role,source_exposure=q,
            source_exposure_unit=block.source_exposure_unit,background_role=block.background_role,
            background_mode='known_count_mean_hypothesis' if known_mode else 'profiled_OFF_counts',
            background_on_exposure=block.background_on_exposure,background_off_exposure=block.background_off_exposure,
            background_exposure_unit=background_exposure,off_role=block.off_role,
            rows=len(n),source_nnz=matrix.nnz))
    matrix=sparse.vstack(matrices,format='csr');n=np.concatenate(all_n);m=np.concatenate(all_m)
    groups=np.concatenate(all_groups);known=np.concatenate(all_known);profile=np.concatenate(all_profile)
    ratio=np.concatenate(all_ratio)
    if np.any(np.bincount(groups,minlength=len(gains)) == 0):
        raise ValueError('Every gain index must have an observed count row')
    gauge=None if gauge_gain_indices is None else np.asarray(gauge_gain_indices)
    if gauge is not None:
        if (gauge.ndim != 1 or not len(gauge) or gauge.dtype.kind not in 'iu'
                or len(np.unique(gauge)) != len(gauge) or np.any(gauge < 0) or np.any(gauge >= len(gains))
                or not set(gauge.tolist()).issubset(detector_groups) or np.any(fixed)
                or np.any(upper[gauge] != 1) or np.any(np.isfinite(np.delete(upper,gauge)))):
            raise ValueError('A relative gauge requires unit-bounded detector gains, unbounded other gains and no fixed gains')
    elif not np.any(fixed & (gains > 0) & (np.bincount(groups,weights=np.asarray(matrix.sum(axis=1)).ravel(),minlength=len(gains)) > 0)):
        raise ValueError('Supply a relative detector gauge or a fixed positive response to anchor the count model')
    impossible=(~profile)&(known==0)&(n>0)&((np.asarray(matrix.sum(axis=1)).ravel()==0)|(gains[groups]==0))
    if impossible.any():
        raise ValueError('Positive counts have no possible source or background response')
    columns=matrix.tocsc();group_rows=[np.flatnonzero(groups==g) for g in range(len(gains))]
    trace=[];maximum_roundoff=0.
    recomputed_subtractions=0;recomputed_rows=0
    maximum_discarded_negative=0.;maximum_recomputed_difference=0.

    def background(rows, source):
        value=known[rows].copy();active=profile[rows]
        value[active]=_profile_background_mean(n[rows][active],m[rows][active],source[active],ratio[rows][active])
        return value

    def coordinate(rows, values, other, bound=math.inf):
        total=float(values.sum())
        if total <= 0 or not math.isfinite(total):
            raise ValueError('Coordinate response must have finite positive total weight')
        normalized=values/total
        def derivative(rate):
            source=other+values*rate
            return 1-float(normalized@_ratio(n[rows],source+background(rows,source)))
        if derivative(0.) >= 0:return 0.
        high=2*float(n[rows].sum())/total
        if not math.isfinite(high) or high <= 0:
            raise ValueError('Coordinate root is outside its finite count domain')
        high=min(high,bound)
        if derivative(high) <= 0:
            if math.isfinite(bound) and high==bound:return high
            raise ValueError('The guaranteed coordinate derivative bracket failed')
        return high*brentq(lambda unit:derivative(high*unit),0.,1.,xtol=1e-15,rtol=1e-14)

    def normalize():
        if gauge is not None:
            scale=float(gains[gauge].max())
            if scale <= 0:raise ValueError('The relative detector scale is undefined')
            gains[:] /= scale;coefficients[:] *= scale

    normalize()
    for iteration in range(max_iterations):
        gain=gains[groups];source=gain*np.asarray(matrix@coefficients)
        if not np.isfinite(source).all():raise ValueError('Source counts exceed the finite numerical domain')
        for column in range(matrix.shape[1]):
            start,end=columns.indptr[column:column+2]
            rows=columns.indices[start:end];values=columns.data[start:end]*gain[rows]
            active=values>0;rows,values=rows[active],values[active]
            if not len(rows):continue
            contribution=values*coefficients[column];other=source[rows]-contribution
            error=max(0.,-float(other.min()))
            allowed=128*np.finfo(float).eps*np.maximum(abs(source[rows]),abs(contribution))
            uncertain=(other<0)|((contribution>0)&(abs(other)<=allowed))
            if np.any(uncertain):
                # Earlier coordinate updates may remove a large source term,
                # leaving cancellation error larger than today's tiny local
                # tail. Evaluate the remaining positive terms directly; do not
                # enlarge a clipping bound or replace a resolved tail by zero.
                remaining=coefficients.copy();remaining[column]=0.
                direct=gain[rows[uncertain]]*np.asarray(matrix[rows[uncertain]]@remaining)
                if not np.isfinite(direct).all() or np.any(direct<0):
                    raise ValueError('Direct positive coordinate sum is outside its finite numerical domain')
                maximum_recomputed_difference=max(maximum_recomputed_difference,
                    float(abs(direct-other[uncertain]).max()))
                maximum_discarded_negative=max(maximum_discarded_negative,error)
                other[uncertain]=direct
                recomputed_subtractions+=1;recomputed_rows+=int(uncertain.sum())
            if np.any(other<0):raise ValueError('Direct coordinate evaluation remained negative')
            coefficients[column]=coordinate(rows,values,other)
            source[rows]=other+values*coefficients[column]
        base=np.asarray(matrix@coefficients)
        for group,rows in enumerate(group_rows):
            if fixed[group]:continue
            active=base[rows]>0;selected=rows[active]
            if len(selected):gains[group]=coordinate(selected,base[selected],np.zeros(len(selected)),upper[group])
        normalize()
        gain=gains[groups];source=gain*np.asarray(matrix@coefficients)
        if not np.isfinite(source).all():raise ValueError('Source counts exceed the finite numerical domain')
        all_rows=np.arange(len(n));bg=background(all_rows,source);mean=source+bg
        if np.any((mean==0)&(n>0)):raise ValueError('The fitted model has impossible positive-count bins')
        loss=_count_loss(n,mean)+_count_loss(m[profile],ratio[profile]*bg[profile])
        residual=1-_ratio(n,mean)
        variance=mean.copy();variance[profile]+=bg[profile]/ratio[profile]
        root_variance=np.sqrt(variance)
        weighted_residual=np.multiply(gain,residual,out=np.zeros_like(residual),where=gain>0)
        gradient=np.asarray(matrix.T@weighted_residual)
        root_curvature=np.zeros(matrix.shape[1])
        for column in range(matrix.shape[1]):
            start,end=columns.indptr[column:column+2]
            rows=columns.indices[start:end];values=columns.data[start:end]*gain[rows]
            # Divide by sqrt(variance) before taking a scaled norm. Forming
            # 1/variance or values**2 first can overflow/underflow even when
            # this final root is finite. Zero variance follows the original
            # empty-bin convention and contributes no expected-curvature term.
            normalized=np.divide(values,root_variance[rows],out=np.zeros_like(values),where=root_variance[rows]>0)
            root_curvature[column]=_root_squared_sum(normalized)
        source_score=float(_projected_step(coefficients,gradient,root_curvature).max())
        base=np.asarray(matrix@coefficients)
        gain_residual=np.multiply(base,residual,out=np.zeros_like(residual),where=base>0)
        ag=np.bincount(groups,weights=gain_residual,minlength=len(gains))
        normalized_base=np.divide(base,root_variance,out=np.zeros_like(base),where=root_variance>0)
        gain_roots=np.array([_root_squared_sum(normalized_base[rows]) for rows in group_rows])
        scores=_projected_step(gains,ag,gain_roots)
        bounded=np.isfinite(upper)
        scores[bounded]=_projected_step(gains[bounded],ag[bounded],gain_roots[bounded],upper[bounded])
        scores[fixed]=0
        gain_score=float(scores.max())
        trace.append(dict(iteration=iteration,poisson_deviance_half=loss,
            maximum_source_projected_step=source_score,maximum_gain_projected_step=gain_score))
        if max(source_score,gain_score)<=tol:break
    sources=[];backgrounds=[];rates=[]
    for block,section,record in zip(blocks,slices,records):
        s=source[section].copy();b=bg[section].copy()
        sources.append(s);backgrounds.append(b)
        rate=None if block.known_background_mean is not None else b/float(block.background_on_exposure)
        rates.append(rate)
        record['poisson_deviance_half']=_count_loss(n[section],s+b)
        if block.known_background_mean is None:
            record['poisson_deviance_half']+=_count_loss(m[section],ratio[section]*b)
        for array in (s,b,rate):
            if array is not None:array.setflags(write=False)
    zero=np.flatnonzero(np.asarray(matrix.multiply(gains[groups,None]).sum(axis=0)).ravel()==0)
    for array in (coefficients,gains):array.setflags(write=False)
    return JointCountCalibration(coefficients,gains,tuple(sources),tuple(backgrounds),tuple(rates),dict(
        version=VERSION,converged_numerically=max(source_score,gain_score)<=tol,iterations=len(trace),
        tolerance=tol,maximum_source_projected_step=source_score,maximum_gain_projected_step=gain_score,
        poisson_deviance_half=loss,trace=trace,blocks=records,
        gauge='explicit detector max gain=1' if gauge is not None else 'supplied fixed positive gain response',
        gauge_gain_indices=None if gauge is None else gauge.tolist(),fixed_gain_indices=np.flatnonzero(fixed).tolist(),
        gain_upper_bounds=[None if math.isinf(v) else float(v) for v in upper],
        zero_final_response_source_indices=zero.tolist(),source_parameter_identifiability_established=False,
        maximum_checked_source_mean_roundoff_clipping=maximum_roundoff,
        positive_source_subtraction_recomputations=recomputed_subtractions,
        positive_source_subtraction_recomputed_rows=recomputed_rows,
        maximum_discarded_source_subtraction_negative_error=maximum_discarded_negative,
        maximum_direct_positive_sum_difference=maximum_recomputed_difference,
        normalized_coordinate_root_xtol=1e-15,normalized_coordinate_root_rtol=1e-14,
        source_or_count_floor_added=False,known_backgrounds_are_conditional_hypotheses=True,
        monitor_blank_misused_as_OFF=False,global_optimality_established=False,
        physical_count_independence_or_stationarity_verified=False,
        full_posterior_or_predictive_coverage_established=False,experimental_accuracy_certified=False))
