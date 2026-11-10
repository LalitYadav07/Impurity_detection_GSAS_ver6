"""Guard normalization of a nonnegative response with a supplied error ledger.

A tiny positive FFT tail must not become a precisely known unit-area peak by
normalizing it. The caller supplies an absolute L1 bound for the exact bins
being normalized. This helper neither estimates that bound nor establishes
physical accuracy, continuous within-bin structure, or parameter uncertainty.
"""
from dataclasses import dataclass
import math
import numpy as np

VERSION = 'powgen-conditioned-template-normalization/1'


@dataclass(frozen=True)
class ConditionedTemplate:
    total_mean: float
    absolute_l1_error_bound: float
    relative_count_error_bound: float | None
    normalized_l1_error_bound: float | None
    declared_normalization_roundoff_allowance: float
    requested_normalized_l1_tolerance: float
    status: str
    probability: np.ndarray | None


def conditioned_template(mean, absolute_l1_error_bound, *, normalized_l1_tolerance=.001,
                         max_bins=2000000):
    """Return a unit-area template only when the supplied ledger resolves it.

    For nonnegative approximate a and true b with ||a-b||_1 <= E, A=sum(a)>E
    implies B=sum(b)>0 and ||a/A-b/B||_1 <= 2E/A. This follows by inserting
    b/A and using |A-B|<=E. The count error relative to the unknown true total
    is bounded by E/(A-E). These are different bounds.

    A positive global-bin L1 ledger also bounds any subset; its relative
    precision can be inadequate on that subset. Unresolved/zero templates and
    templates above the requested tolerance return probability=None. No data
    or response mass is discarded, repaired, or silently renormalized.

    Eight double epsilons plus a per-bin subnormal allowance are declared for
    fsum/division roundoff under ordinary IEEE double arithmetic. This is a
    floating-point allowance, not a formally verified arithmetic bound.
    """
    if type(max_bins) is not int or not 1 <= max_bins <= 2000000:
        raise ValueError('A bounded positive integer bin budget is required')
    for value in [absolute_l1_error_bound, normalized_l1_tolerance]:
        if (isinstance(value,(bool,np.bool_,str,bytes)) or not np.isscalar(value)
                or not np.isrealobj(value) or not math.isfinite(float(value))):
            raise ValueError('Finite real scalar error and tolerance are required')
    error = float(absolute_l1_error_bound)
    tolerance = float(normalized_l1_tolerance)
    if error < 0 or not 0 < tolerance <= 2:
        raise ValueError('Nonnegative error and a positive L1 tolerance at most two are required')
    raw = np.asarray(mean)
    if raw.ndim != 1 or not 1 <= raw.size <= max_bins or not np.isrealobj(raw):
        raise ValueError('A bounded real response vector is required')
    try:
        a = np.array(raw,dtype=float,copy=True)
    except (ValueError,TypeError) as e:
        raise ValueError('A numeric response vector is required') from e
    if not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('A bounded finite nonnegative response vector is required')
    try:
        total = math.fsum(a)
    except OverflowError as e:
        raise ValueError('The total response exceeds finite double arithmetic') from e
    if not math.isfinite(total) or (0 < total < np.finfo(float).tiny):
        raise ValueError('A nonzero response total must be a finite normal double')
    roundoff = float(8*np.finfo(float).eps+a.size*np.nextafter(0.,1.))
    count_bound = shape_bound = None
    probability = None
    if total == 0:
        status = 'zero_template' if error == 0 else 'unresolved_mass'
    elif total <= error:
        status = 'unresolved_mass'
    else:
        count_bound = error/(total-error)
        shape_bound = min(2.,2*(error/total)+roundoff)
        status = 'resolved' if shape_bound <= tolerance else 'insufficient_precision'
        if status == 'resolved':
            probability = a/total
            probability.setflags(write=False)
    return ConditionedTemplate(total,error,count_bound,shape_bound,roundoff,
                               tolerance,status,probability)
