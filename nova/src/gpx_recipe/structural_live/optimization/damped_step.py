"""Reuse one SVD for the information score and damped least-squares step."""
import numpy as np


def damped_step(jacobian,residual,damping):
    j=np.asarray(jacobian,float);r=np.asarray(residual,float)
    if j.ndim!=2 or r.shape!=(j.shape[0],) or not np.isfinite(j).all() or not np.isfinite(r).all():
        raise ValueError('Finite matching Jacobian and residual required')
    if not np.isfinite(damping) or damping<=0:raise ValueError('Positive finite damping required')
    norms=np.linalg.norm(j,axis=0);scale=np.maximum(norms,1e-10);scaled=j/scale
    u,singular,vh=np.linalg.svd(scaled,full_matrices=False)
    cutoff=max(scaled.shape)*np.finfo(float).eps*(singular[0] if len(singular) else 0.)
    rank=int(np.count_nonzero(singular>cutoff));projection=u.T@r
    step=-(vh.T@((singular/(singular**2+damping))*projection))/scale
    step[norms==0.]=0.  # the unique damped solution has zero in exact null columns
    return step,dict(rank=rank,information_score=float(np.linalg.norm(projection[:rank])),
        largest_scaled_singular_value=float(singular[0]) if len(singular) else 0.)


def bounded_damped_step(jacobian,residual,damping,lower,upper,radius=.5):
    """Solve the bounded LM subproblem instead of shrinking every coordinate.

An unconstrained step in a weakly identified coordinate can be enormous.
Uniformly shrinking that step also suppresses well-determined coordinates.
The bounded linear solve preserves their useful motion.
"""
    from scipy.optimize import lsq_linear
    j=np.asarray(jacobian,float);r=np.asarray(residual,float)
    if not np.isfinite(radius) or radius<=0:raise ValueError('Finite positive trust region required')
    step,info=damped_step(j,r,damping)
    lo=np.maximum(np.asarray(lower,float),-radius);hi=np.minimum(np.asarray(upper,float),radius)
    if lo.shape!=step.shape or hi.shape!=step.shape or np.any(lo>=hi) or np.any(lo>0) or np.any(hi<0):
        raise ValueError('Finite step intervals containing zero are required')
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or radius<=0:raise ValueError('Finite positive trust region required')
    if np.all(step>=lo) and np.all(step<=hi):
        return step,dict(info,bounded_subproblem=False)
    scale=np.maximum(np.linalg.norm(j,axis=0),1e-10)
    matrix=np.vstack([j/scale,np.sqrt(damping)*np.eye(j.shape[1])])
    target=np.r_[-r,np.zeros(j.shape[1])]
    solved=lsq_linear(matrix,target,bounds=(lo*scale,hi*scale),method='bvls',tol=1e-9,max_iter=100)
    result=np.clip(solved.x/scale,lo,hi)
    result[np.linalg.norm(j,axis=0)==0.]=0.
    return result,dict(info,bounded_subproblem=True,bounded_success=bool(solved.success),
        bounded_iterations=int(solved.nit),bounded_optimality=float(solved.optimality))
