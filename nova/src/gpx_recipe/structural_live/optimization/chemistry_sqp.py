"""Geometry-only contact derivatives and a chemistry-constrained GN proposal."""
import numpy as np
from scipy.optimize import minimize,Bounds,LinearConstraint

def contacts(geometry,z,radii):
    z=np.asarray(z,float);radii=np.asarray(radii,float);u=z*radii
    if z.ndim!=1 or radii.shape!=z.shape or not np.isfinite(z).all() or not np.isfinite(radii).all() or np.any(radii<=0):
        raise ValueError('Finite contact coordinates and positive scales required')
    reciprocal=geometry.baseline+np.einsum('k,kij->ij',u,geometry.cell_directions)
    v=geometry.delta+np.einsum('cik,k->ci',geometry.directions,u)
    w=np.linalg.solve(reciprocal,v.T).T
    squared=np.einsum('ci,ci->c',v,w);threshold2=geometry.threshold**2
    value=squared/threshold2-1
    jac=(2*np.einsum('cik,ci->ck',geometry.directions,w)-np.einsum('ci,kij,cj->ck',w,geometry.cell_directions,w))
    jac=jac*radii[None,:]/threshold2[:,None]
    for a,b,r in zip(geometry.offsets[:-1],geometry.offsets[1:],geometry.radii):
        if b-a<=1:continue # Exact coordinate boxes already enter the step.
        val=1-float(u[a:b]@u[a:b])/(r*r);gradient=np.zeros(len(z));gradient[a:b]=-2*u[a:b]*radii[a:b]/(r*r)
        value=np.r_[value,val];jac=np.vstack([jac,gradient])
    if not np.isfinite(value).all() or not np.isfinite(jac).all():raise ValueError('Nonfinite contact derivatives')
    return value,jac

def step(jacobian,residual,point,values,contact_jacobian,*,damping=.001,radius=.3):
    j=np.asarray(jacobian,float);r=np.asarray(residual,float);z=np.asarray(point,float)
    values=np.asarray(values,float);a=np.asarray(contact_jacobian,float)
    if j.shape!=(len(r),len(z)) or a.shape!=(len(values),len(z)):raise ValueError('Incompatible constrained GN dimensions')
    if (not all(np.isfinite(v).all() for v in [j,r,z,values,a]) or np.any(abs(z)>1)
            or not np.isfinite([damping,radius]).all() or min(damping,radius)<=0):
        raise ValueError('Finite constrained GN inputs and positive controls required')
    scale=np.maximum(np.linalg.norm(j,axis=0),1e-6);js=j/scale;h=js.T@js+damping*np.eye(len(z));g=js.T@r
    norm=max(float(r@r),1.);lower=np.maximum(-1-z,-radius)*scale;upper=np.minimum(1-z,radius)*scale
    # A tiny inward reserve avoids following a curved boundary with a purely
    # tangent linear step. This tightens proposals; it does not relax chemistry.
    reserve=np.where(np.linalg.norm(a,axis=1)>1e-12,1e-9,0.)
    constraints=[] if not len(values) else [LinearConstraint(a/scale,-values+reserve,np.inf)]
    solved=minimize(lambda y:float(g@y+.5*y@h@y)/norm,np.zeros(len(z)),
        jac=lambda y:(g+h@y)/norm,bounds=Bounds(lower,upper),constraints=constraints,method='SLSQP',
        options=dict(maxiter=100,ftol=1e-12))
    p=solved.x/scale
    feasible=bool(np.all(p>=lower/scale-1e-10) and np.all(p<=upper/scale+1e-10)
                  and np.all(values+a@p>=-1e-9))
    descent=float((j.T@r)@p)
    if not feasible or not np.isfinite(p).all() or descent>=0:p=np.zeros(len(z))
    return p,dict(subproblem_success=bool(solved.success),message=str(solved.message),iterations=int(solved.nit),
        linear_constraints_satisfied=feasible,predicted_directional_derivative=descent,contact_constraints=len(values))
