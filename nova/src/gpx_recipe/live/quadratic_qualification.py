"""Check every coefficient direction against a separate native Jacobian.

This applies only to an already scoped homogeneous quadratic magnetic model.
It does not establish that an arbitrary native correction is quadratic, that a
search is globally complete, or that a physical magnetic structure is unique.
Independent full-profile checks remain required.
"""
import numpy as np


def check_certified_factor_context(factors,forward,original,origin):
    """Reuse a complete authenticated proof, with fresh current-context probes.

    These probes are not a new complete coefficient certificate. Authority
    for all coefficients comes from the exact originating native preparation
    plus the caller's exact current scattering-identity comparison. Without
    that provenance the ordinary complete coefficient check remains required.
    """
    a=np.asarray(factors,float);dimension=a.shape[-1] if a.ndim==3 else 0
    if (a.ndim!=3 or not len(a) or not 1<=dimension<=24 or not np.isfinite(a).all()
            or not isinstance(origin,dict) or not origin
            or not isinstance(original,dict)
            or original.get('method')!='complete native reflection-intensity polarization'
            or original.get('all_coefficient_directions_checked') is not True
            or original.get('dimensions')!=dimension or original.get('reflections')!=len(a)
            or original.get('symmetric_coefficients')!=dimension*(dimension+1)//2
            or original.get('native_structure_factor_calls')!=2*dimension*dimension+1
            or original.get('relative_tolerance')!=1e-8
            or type(original.get('absolute_floor')) not in (float,int)
            or not np.isfinite(original['absolute_floor']) or original['absolute_floor']<=0
            or type(original.get('maximum_tolerance_ratio')) not in (float,int)
            or not np.isfinite(original['maximum_tolerance_ratio'])
            or not 0<=original['maximum_tolerance_ratio']<=1):
        raise ValueError('An authenticated complete native coefficient proof is required')
    floor=max(float(np.max(np.sum(a*a,axis=1)))*1e-12,np.finfo(float).tiny)
    ratios=[]
    for point in np.random.default_rng(629173).normal(size=(3,dimension)):
        expected=np.sum((a@point)**2,axis=1)
        actual=np.asarray(forward(point),float)
        if actual.shape!=expected.shape or not np.isfinite(actual).all():
            raise ValueError('Certified amplitude native probe changed its reflection inventory')
        ratio=float(np.max(abs(actual-expected)/(1e-8*np.maximum(abs(actual),abs(expected))+floor)))
        if not np.isfinite(ratio) or ratio>1:
            raise ValueError('Certified amplitudes disagree with current native intensities')
        ratios.append(ratio)
    return dict(method='authenticated complete coefficients with current-context native probes',
        dimensions=dimension,reflections=len(a),symmetric_coefficients=dimension*(dimension+1)//2,
        all_coefficient_directions_checked=True,complete_check_scope='authenticated originating native preparation',
        original_native_validation=original,native_origin=origin,current_probe_tolerance_ratios=ratios,
        native_structure_factor_calls=3,native_jacobian_calls=0,
        scope='Exact current scattering identity and amplitude bytes; independent full-profile and fitted-state checks remain required')


def compare_coefficients(quadratic,reference,*,relative_tolerance=1e-8,absolute_fraction=1e-12):
    """Compare complete independently obtained coefficient arrays."""
    q=np.asarray(quadratic,float);other=np.asarray(reference,float)
    if q.ndim!=3 or q.shape!=other.shape or q.shape[1]!=q.shape[2] or not np.isfinite(q).all() or not np.isfinite(other).all():
        raise ValueError('Complete finite coefficient arrays in the same basis are required')
    diagonals=np.max(abs(np.diagonal(other,axis1=1,axis2=2)),axis=0)
    floor=max(float(np.max(diagonals))*absolute_fraction,np.finfo(float).tiny)
    maximum=0.
    for i in range(q.shape[1]):
        scales=np.sqrt(diagonals)*np.sqrt(diagonals[i])
        ratio=np.max(abs(q[:,:,i]-other[:,:,i]),axis=0)/(relative_tolerance*scales+floor)
        maximum=max(maximum,float(np.max(ratio)))
    if not np.isfinite(maximum) or maximum>1.:
        raise ValueError('Prepared quadratic coefficients disagree with independently rebuilt native coefficients')
    return dict(maximum_tolerance_ratio=maximum,relative_tolerance=relative_tolerance,absolute_floor=floor)


def check_factor_coefficients(factors,forward,*,relative_tolerance=1e-8,absolute_fraction=1e-12):
    """Verify every Gram coefficient using black-box native reflection F².

    Native peak shapes are checked separately. This skips neither mixed
    coefficients nor reflections, and does not differentiate a cached tensor.
    """
    a=np.asarray(factors,float)
    if a.ndim!=3 or not len(a) or not 1<=a.shape[2]<=24 or not np.isfinite(a).all():
        raise ValueError('A bounded finite reflection amplitude basis is required')
    dimension=a.shape[2];diagonals=np.sum(a*a,axis=1)
    floor=max(float(np.max(diagonals))*absolute_fraction,np.finfo(float).tiny)
    calls=0;maximum=0.
    def evaluate(point):
        nonlocal calls
        value=np.asarray(forward(point),float);calls+=1
        if value.shape!=(len(a),) or not np.isfinite(value).all():
            raise ValueError('Native reflection intensity inventory changed')
        return value.copy()
    def check(actual,expected,i,j):
        nonlocal maximum
        scale=np.sqrt(diagonals[:,i])*np.sqrt(diagonals[:,j])
        ratio=float(np.max(abs(actual-expected)/(relative_tolerance*scale+floor)))
        maximum=max(maximum,ratio)
        if not np.isfinite(ratio) or ratio>1.:
            raise ValueError('Captured magnetic amplitudes disagree with complete native reflection coefficients')
    zero=evaluate(np.zeros(dimension));eye=np.eye(dimension)
    for i in range(dimension):
        check((evaluate(eye[i])+evaluate(-eye[i]))/2-zero,diagonals[:,i],i,i)
        for j in range(i):
            native=(evaluate(eye[i]+eye[j])+evaluate(-eye[i]-eye[j])-
                evaluate(eye[i]-eye[j])-evaluate(-eye[i]+eye[j]))/8.
            check(native,np.sum(a[:,:,i]*a[:,:,j],axis=1),i,j)
    return dict(method='complete native reflection-intensity polarization',dimensions=dimension,
        reflections=len(a),symmetric_coefficients=dimension*(dimension+1)//2,
        native_structure_factor_calls=calls,native_jacobian_calls=0,
        relative_tolerance=relative_tolerance,absolute_floor=floor,maximum_tolerance_ratio=maximum,
        all_coefficient_directions_checked=True,
        scope='Every native reflection coefficient at this fixed context; independent full-profile checks remain required')


def check_coefficients(quadratic,derivatives,*,relative_tolerance=1e-8,
                       absolute_fraction=1e-12):
    """Stream coefficient columns using J(+e_i) - J(-e_i) = 4 Q[:, :, i].

    ``derivatives`` returns moment-variable by bin derivatives of an independent
    native forward profile, not derivatives of the tensor under test. Central
    differences of the analytic Jacobian cancel native exact-zero guards.
    The diagonal scales are invariant under independent variable rescaling.
    """
    q=np.asarray(quadratic,float)
    if (q.ndim!=3 or q.shape[1]!=q.shape[2] or not 1<=q.shape[1]<=24
            or not len(q) or not np.isfinite(q).all()
            or not np.isfinite([relative_tolerance,absolute_fraction]).all()
            or relative_tolerance<=0 or absolute_fraction<0):
        raise ValueError('A finite bounded quadratic response and positive coefficient tolerances are required')
    dimension=q.shape[1];diagonals=np.max(abs(np.diagonal(q,axis1=1,axis2=2)),axis=0)
    absolute_floor=max(float(np.max(diagonals))*absolute_fraction,np.finfo(float).tiny)
    ratios=[];differences=[]
    for i in range(dimension):
        point=np.eye(dimension)[i]
        plus=np.asarray(derivatives(point),float).copy()
        minus=np.asarray(derivatives(-point),float)
        if (plus.shape!=(dimension,len(q)) or minus.shape!=plus.shape
                or not np.isfinite(plus).all() or not np.isfinite(minus).all()):
            raise ValueError('Independent native moment derivatives have a different basis or nonfinite values')
        native=(plus-minus).T/4.
        difference=np.max(abs(native-q[:,:,i]),axis=0)
        scales=np.sqrt(diagonals)*np.sqrt(diagonals[i])
        allowed=relative_tolerance*scales+absolute_floor
        ratio=difference/allowed
        if not np.isfinite(ratio).all() or np.max(ratio)>1.:
            raise ValueError('Prepared quadratic coefficients disagree with independent native derivatives')
        ratios.append(float(np.max(ratio)));differences.append(float(np.max(difference)))
    return dict(method='central native Jacobian coefficient directions',dimensions=dimension,
        symmetric_coefficients=dimension*(dimension+1)//2,native_jacobian_calls=2*dimension,
        relative_tolerance=float(relative_tolerance),absolute_floor=absolute_floor,
        maximum_tolerance_ratios=ratios,maximum_absolute_errors=differences,
        all_coefficient_directions_checked=True,
        scope='Fixed-context homogeneous quadratic coefficient agreement; separate forward and fitted-result checks remain required')
