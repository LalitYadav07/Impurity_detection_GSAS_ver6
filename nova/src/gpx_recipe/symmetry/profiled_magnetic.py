"""Fixed-geometry magnetic starting points with profiled linear nuisances.

This module does not declare GSAS-II convergence or supply a covariance matrix.
Its inputs are an independently checked native quadratic profile in the *full*
independent moment basis, unchanged weighted observations, and physical moment
maps for every magnetic atom. It is deliberately separate from the legacy
native executor until native-oracle and campaign integration checks pass.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize,nnls


@dataclass(frozen=True)
class StartingPoint:
    coefficients: np.ndarray
    scale: float
    background: np.ndarray
    weighted_cost: float
    success: bool
    message: str
    iterations: int
    evaluations: int
    maximum_moment: float
    scaled_gradient_norm: float
    scaled_stationarity_norm: float
    local_descent_found: bool
    optimizer_status: int
    optimizer_success: bool
    objective_ftol: float
    curvature_eigenvalues: np.ndarray
    bound_active: bool
    optimizer_objective_scale: float


class ProfiledMagneticObjective:
    """Eliminate only explicitly free linear background and common scale.

    Weighted residual = offset + scale * (nuclear + m.T Q m) + B @ b.
    Fixed background terms belong in ``offset``. B contains exactly the
    declared free background columns, with no new basis or fitted peak terms.
    ``moment_maps[a] @ m`` is atom a's Cartesian moment in mu_B, including all
    native symmetry/linear constraints. A fixed scale is a positive number;
    ``None`` requests a common positive nuclear/magnetic histogram scale.
    """

    def __init__(self, *, offset, nuclear, quadratic, background, moment_maps,
                 maximum_moment, normalization, fixed_scale=None):
        self.offset=np.asarray(offset,dtype=float).copy()
        self.nuclear=np.asarray(nuclear,dtype=float).copy()
        self.quadratic=np.asarray(quadratic,dtype=float).copy()
        self.background=np.asarray(background,dtype=float).copy()
        self.moment_maps=np.asarray(moment_maps,dtype=float).copy()
        self.maximum_moment=float(maximum_moment)
        self.normalization=float(normalization)
        self.fixed_scale=None if fixed_scale is None else float(fixed_scale)
        n=len(self.offset) if self.offset.ndim==1 else -1
        q=self.quadratic
        if q.ndim!=3 or q.shape[0]!=n or q.shape[1]!=q.shape[2] or not 1<=q.shape[1]<=24:
            raise ValueError('A bounded independent magnetic basis is required')
        self.dimension=q.shape[1]
        if self.nuclear.shape!=(n,) or self.background.ndim!=2 or self.background.shape[0]!=n:
            raise ValueError('Native residual and nuisance dimensions differ')
        if self.moment_maps.ndim!=3 or self.moment_maps.shape[1:]!=(3,self.dimension) or not len(self.moment_maps):
            raise ValueError('Every magnetic atom needs a Cartesian moment map')
        arrays=[self.offset,self.nuclear,q,self.background,self.moment_maps]
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError('Native profile inputs must be finite')
        if not np.allclose(q,q.swapaxes(1,2),rtol=1e-10,atol=1e-12):
            raise ValueError('The native magnetic quadratic form is not symmetric')
        if not np.isfinite(self.normalization) or self.normalization<=0 or not np.isfinite(self.maximum_moment) or self.maximum_moment<=0:
            raise ValueError('Positive finite normalization and moment ceiling are required')
        if self.fixed_scale is not None and (not np.isfinite(self.fixed_scale) or self.fixed_scale<=0):
            raise ValueError('A fixed common histogram scale must be positive')
        # Whiten the *physical* independent moment metric, not Jacobian
        # columns that vanish when one magnetic component tends to zero.
        metric=np.einsum('aki,akj->ij',self.moment_maps,self.moment_maps)/len(self.moment_maps)
        eigenvalues,eigenvectors=np.linalg.eigh(metric)
        if eigenvalues[0]<=1e-12*eigenvalues[-1]:
            raise ValueError('The independent magnetic moment map is rank deficient')
        self.to_native=(eigenvectors/np.sqrt(eigenvalues))@eigenvectors.T
        self.to_physical=np.linalg.inv(self.to_native)
        self.physical_maps=self.moment_maps@self.to_native
        self.qphysical=np.einsum('ia,nij,jb->nab',self.to_native,q,self.to_native)
        if self.background.shape[1]:
            if np.linalg.matrix_rank(self.background)!=self.background.shape[1]:
                raise ValueError('The existing background basis is rank deficient')
            self.background_q,_=np.linalg.qr(self.background,mode='reduced')
        else:
            self.background_q=np.empty((n,0))
        self.projected_offset=self._project(self.offset)
        self.projected_nuclear=self._project(self.nuclear)
        self.projected_quadratic=self._project(self.qphysical.reshape(n,-1)).reshape(self.qphysical.shape)

    def _project(self,array):
        return array-self.background_q@(self.background_q.T@array)

    def _evaluate(self,physical):
        z=np.asarray(physical,dtype=float)
        curve=self.projected_nuclear+np.einsum('i,nij,j->n',z,self.projected_quadratic,z)
        denominator=float(curve@curve)
        if self.fixed_scale is None:
            if denominator<=np.finfo(float).tiny:
                raise ValueError('No identifiable common nuclear/magnetic scale')
            scale=max(0.,-float(curve@self.projected_offset)/denominator)
        else:
            scale=self.fixed_scale
        residual=self.projected_offset+scale*curve
        gradient=2*scale*np.einsum('nij,j->ni',self.projected_quadratic,z).T@residual
        return float(.5*(residual@residual)),gradient,scale

    def value_gradient(self,physical):
        cost,gradient,_=self._evaluate(physical)
        return cost/self.normalization,gradient/self.normalization

    def _constraint(self,z):
        moments=np.einsum('aki,i->ak',self.physical_maps,z)
        return self.maximum_moment**2-np.einsum('ak,ak->a',moments,moments)

    def _constraint_jacobian(self,z):
        return -2*np.einsum('aki,ak->ai',self.physical_maps,
            np.einsum('aki,i->ak',self.physical_maps,z))

    def _hessian(self, z):
        step=1e-4*max(1.,float(np.linalg.norm(z)))
        hessian=np.column_stack([(self.value_gradient(z+np.eye(self.dimension)[j]*step)[1]-
                                  self.value_gradient(z-np.eye(self.dimension)[j]*step)[1])/(2*step)
                                 for j in range(self.dimension)])
        return (hessian+hessian.T)/2

    def fit(self,seed,*,max_iterations=200,objective_ftol=1e-14,callback=None):
        seed=np.asarray(seed,dtype=float)
        if seed.shape!=(self.dimension,) or not np.isfinite(seed).all():
            raise ValueError('One finite seed in the complete independent basis is required')
        if isinstance(max_iterations,bool) or not isinstance(max_iterations,int) or max_iterations<1:
            raise ValueError('The numerical budget must be a positive integer')
        if isinstance(objective_ftol,bool) or not np.isfinite(objective_ftol) or not 0<objective_ftol<=1e-12:
            raise ValueError('The normalized objective tolerance must be positive and no looser than 1e-12')
        z=self.to_physical@seed
        maximum=float(np.max(np.linalg.norm(self.moment_maps@seed,axis=1)))
        if maximum>self.maximum_moment:
            z*=self.maximum_moment/maximum
        # Nuclear intensity can dominate the normalization by many orders of
        # magnitude. SLSQP otherwise accepts a tiny first step while a magnetic
        # descent direction remains. Rescale only its numerical objective; all
        # physical checks below use the original weighted objective/tolerances.
        curvature_scale=float(np.max(np.abs(np.linalg.eigvalsh(self._hessian(z)))))
        optimizer_scale=max(1.,1./max(curvature_scale,1e-12))
        def optimizer_objective(point):
            value,gradient=self.value_gradient(point)
            return value*optimizer_scale,gradient*optimizer_scale
        result=minimize(optimizer_objective,z,method='SLSQP',jac=True,
            constraints=[dict(type='ineq',fun=self._constraint,jac=self._constraint_jacobian)],
            bounds=[(-self.maximum_moment,self.maximum_moment)]*self.dimension,
            options=dict(maxiter=max_iterations,ftol=objective_ftol),callback=callback)
        cost,gradient,scale=self._evaluate(result.x)
        coefficients=self.to_native@result.x
        full=self.offset+scale*(self.nuclear+np.einsum('i,nij,j->n',coefficients,self.quadratic,coefficients))
        background=np.linalg.lstsq(self.background,-full,rcond=None)[0] if self.background.shape[1] else np.empty(0)
        maximum=float(np.max(np.linalg.norm(self.moment_maps@coefficients,axis=1)))
        physical=maximum<=self.maximum_moment*(1+1e-8)
        active=self._constraint(result.x)<=1e-7*self.maximum_moment**2
        scaled_gradient=gradient/self.normalization
        if active.any():
            constraint_jacobian=self._constraint_jacobian(result.x)[active].T
            multipliers,_=nnls(constraint_jacobian,scaled_gradient)
            stationary=scaled_gradient-constraint_jacobian@multipliers
        else:
            stationary=scaled_gradient
        stationarity=float(np.linalg.norm(stationary))
        # Quadratic intensities have zero first derivatives at zero moments.
        # A native or SciPy "success" at a maximum is not a useful start.
        step=1e-4*max(1.,float(np.linalg.norm(result.x)))
        curvature,directions=np.linalg.eigh(self._hessian(result.x))
        local_descent=False
        for direction in directions.T:
            for sign in [-1.,1.]:
                probe=result.x+sign*step*direction
                probe_max=float(np.max(np.linalg.norm(self.physical_maps@probe,axis=1)))
                if probe_max>self.maximum_moment:probe*=self.maximum_moment/probe_max
                if self.value_gradient(probe)[0]<cost/self.normalization-1e-12:
                    local_descent=True
        return StartingPoint(coefficients=coefficients,scale=float(scale),background=background,
            weighted_cost=cost,success=bool(result.success and physical and scale>0 and stationarity<=1e-7 and not local_descent),
            message=str(result.message),iterations=int(result.nit),evaluations=int(result.nfev),
            maximum_moment=maximum,scaled_gradient_norm=float(np.linalg.norm(gradient)/self.normalization),
            scaled_stationarity_norm=stationarity,local_descent_found=local_descent,
            optimizer_status=int(result.status),optimizer_success=bool(result.success),objective_ftol=float(objective_ftol),
            curvature_eigenvalues=curvature,bound_active=bool(active.any()),
            optimizer_objective_scale=optimizer_scale)

    def heldout_error(self,coefficients,scale,background,native_residual):
        """Compare with a separate native evaluation, never its interpolants."""
        m=np.asarray(coefficients,dtype=float)
        residual=self.offset+scale*(self.nuclear+np.einsum('i,nij,j->n',m,self.quadratic,m))+self.background@background
        expected=np.asarray(native_residual,dtype=float)
        if expected.shape!=residual.shape or not np.isfinite(expected).all():
            raise ValueError('The independent native residual is unavailable')
        return float(np.max(np.abs(residual-expected))/max(1.,float(np.max(np.abs(expected)))))
