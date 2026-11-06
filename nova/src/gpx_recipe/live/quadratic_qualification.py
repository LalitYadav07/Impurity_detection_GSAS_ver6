"""Check every coefficient direction against a separate native Jacobian.

This applies only to an already scoped homogeneous quadratic magnetic model.
It does not establish that an arbitrary native correction is quadratic, that a
search is globally complete, or that a physical magnetic structure is unique.
Independent full-profile checks remain required.
"""
import numpy as np


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
