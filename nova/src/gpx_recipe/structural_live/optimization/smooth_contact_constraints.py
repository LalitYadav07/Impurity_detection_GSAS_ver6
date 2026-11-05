"""Exact smooth inequalities for a certified bounded periodic contact list.

This differentiates geometry, not scattering. For fractional separation v and
reciprocal metric B, d² = v.T B^-1 v. Both v and B are affine in the native
parameter coordinates. BoundedChemicalGeometry proves which periodic images
can reach the empirical floor anywhere in the entire declared domain.
"""
import numpy as np
from scipy.optimize import NonlinearConstraint
from .bounded_chemical_prior import BoundedChemicalGeometry


class SmoothContactConstraints:
    def __init__(self, geometry):
        if not isinstance(geometry, BoundedChemicalGeometry):
            raise ValueError('A certified bounded periodic contact geometry is required')
        self.geometry=geometry
        # MP/COD source screening can explicitly impose no minimum on a
        # pair. Older bounded-domain archives may retain its zero threshold.
        # It contributes no inequality and must never be divided by zero.
        self.active=geometry.threshold>0
        self.threshold=geometry.threshold[self.active]
        self.delta=geometry.delta[self.active]
        self.directions=geometry.directions[self.active]
        self.count=len(self.threshold)

    def _terms(self, point):
        g=self.geometry;z=np.asarray(point,float)
        if z.shape!=(g.dimension,) or not np.isfinite(z).all():raise ValueError('Finite bounded coordinates required')
        if any(np.linalg.norm(z[a:b])>r*(1+1e-12) for a,b,r in zip(g.offsets[:-1],g.offsets[1:],g.radii)):
            raise ValueError('Contact completeness is certified only within the declared domain')
        reciprocal=g.baseline+np.einsum('k,kij->ij',z,g.cell_directions)
        factor=np.linalg.cholesky(reciprocal)
        inverse_l=np.linalg.solve(factor,np.eye(3));direct=inverse_l.T@inverse_l
        fractional=self.delta+np.einsum('cik,k->ci',self.directions,z)
        covector=fractional@direct
        return fractional,covector,direct

    def values(self, point):
        fractional,covector,_=self._terms(point)
        return np.einsum('ci,ci->c',fractional,covector)/self.threshold**2-1

    def jacobian(self, point):
        _,covector,_=self._terms(point);g=self.geometry
        return (2*np.einsum('cik,ci->ck',self.directions,covector)
                -np.einsum('ci,kij,cj->ck',covector,g.cell_directions,covector))/self.threshold[:,None]**2

    def hessian(self, point, multipliers):
        _,covector,direct=self._terms(point);g=self.geometry
        weights=np.asarray(multipliers,float)
        if weights.shape!=(self.count,) or not np.isfinite(weights).all():raise ValueError('One finite multiplier per contact required')
        tangent=self.directions-np.einsum('kij,cj->cik',g.cell_directions,covector)
        result=2*np.einsum('cik,ij,cjl,c->kl',tangent,direct,tangent,weights/self.threshold**2)
        return (result+result.T)/2

    def scipy_constraint(self):
        if not self.count:raise ValueError('No retained contacts need an optimizer constraint')
        return NonlinearConstraint(self.values,0.,np.inf,jac=self.jacobian,hess=self.hessian,keep_feasible=True)


def audit_local_kkt(result,gradient,contacts,lower,upper,*,tolerance=1e-6):
    """Independently reconstruct first-order conditions; not global optimality.

    The supplied objective gradient should be independently checked for step
    stability. Multipliers follow SciPy's lower-negative/upper-positive rule.
    """
    x=np.asarray(result.x,float);grad=np.asarray(gradient,float)
    lower=np.asarray(lower,float);upper=np.asarray(upper,float)
    if grad.shape!=x.shape or lower.shape!=x.shape or upper.shape!=x.shape:
        raise ValueError('Consistent KKT coordinate dimensions required')
    expected=2 if contacts is not None and contacts.count else 1
    if len(result.v)!=expected:raise ValueError('Unexpected constraint multiplier order')
    bound=np.asarray(result.v[-1],float)
    if bound.shape!=x.shape:raise ValueError('Missing bound multipliers')
    stationarity=grad+bound
    violations=[np.max(lower-x),np.max(x-upper),0.]
    complementarity=[float(np.max(abs(bound*np.where(bound>=0,upper-x,x-lower)),initial=0.))]
    dual=0.;minimum_margin=None;active=[]
    if contacts is not None and contacts.count:
        values=contacts.values(x);jac=contacts.jacobian(x);multipliers=np.asarray(result.v[0],float)
        if multipliers.shape!=values.shape:raise ValueError('Missing contact multipliers')
        stationarity+=jac.T@multipliers
        violations.append(float(np.max(-values)))
        dual=max(0.,float(np.max(multipliers)))
        complementarity.append(float(np.max(abs(values*multipliers))))
        minimum_margin=float(np.min(values));active=np.flatnonzero(values<1e-5).tolist()
    norms=dict(primal_violation=float(max(violations)),dual_sign_violation=dual,
               complementarity_residual=max(complementarity),stationarity_infinity_norm=float(np.max(abs(stationarity))))
    finite=bool(np.isfinite(list(norms.values())).all())
    return dict(**norms,finite=finite,first_order_conditions_passed=finite and max(norms.values())<=tolerance,
                tolerance=tolerance,minimum_squared_distance_relative_margin=minimum_margin,
                active_contact_indices=active,global_optimality_proved=False,
                interpretation='Local first-order numerical conditions; objective derivative reliability is a separate gate.')
