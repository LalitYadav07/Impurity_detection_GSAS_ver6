"""Fixed-geometry magnetic starting points with profiled linear nuisances.

This module does not declare GSAS-II convergence or supply a covariance matrix.
Its inputs are an independently checked native quadratic profile in the *full*
independent moment basis, unchanged weighted observations, and physical moment
maps for every magnetic atom. It is deliberately separate from the legacy
native executor until native-oracle and campaign integration checks pass.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
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
    descent_restarts: int = 0
    descent_probe_evaluations: int = 0


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
        from .profiled_reduction import MagneticResidualReduction
        self.reduction,self.reduction_diagnostics=MagneticResidualReduction.build(
            self.projected_offset,self.projected_nuclear,self.projected_quadratic)
        self.reduction_evaluations=0
        self.full_evaluations=0

    def _project(self,array):
        return array-self.background_q@(self.background_q.T@array)

    def _evaluate(self,physical):
        z=np.asarray(physical,dtype=float)
        if self.reduction is not None:
            result=self.reduction.evaluate(z,self.fixed_scale)
            if result is not None:
                self.reduction_evaluations+=1
                return result
        return self._evaluate_full(z)

    def _evaluate_full(self,physical):
        """Original weighted arrays; independent of the QR representation."""
        self.full_evaluations+=1
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

    def _full_value_gradient(self,physical):
        cost,gradient,_=self._evaluate_full(physical)
        return cost/self.normalization,gradient/self.normalization

    def _constraint(self,z):
        moments=np.einsum('aki,i->ak',self.physical_maps,z)
        return self.maximum_moment**2-np.einsum('ak,ak->a',moments,moments)

    def _constraint_jacobian(self,z):
        return -2*np.einsum('aki,ak->ai',self.physical_maps,
            np.einsum('aki,i->ak',self.physical_maps,z))

    def _hessian(self, z, *, direct=False):
        step=1e-4*max(1.,float(np.linalg.norm(z)))
        evaluate=self._full_value_gradient if direct else self.value_gradient
        hessian=np.column_stack([(evaluate(z+np.eye(self.dimension)[j]*step)[1]-
                                  evaluate(z-np.eye(self.dimension)[j]*step)[1])/(2*step)
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
        iterations=evaluations=restarts=probe_evaluations=0
        while True:
            result=minimize(optimizer_objective,z,method='SLSQP',jac=True,
                constraints=[dict(type='ineq',fun=self._constraint,jac=self._constraint_jacobian)],
                bounds=[(-self.maximum_moment,self.maximum_moment)]*self.dimension,
                options=dict(maxiter=max_iterations-iterations,ftol=objective_ftol),callback=callback)
            # A restart uses the remaining original budget, not a fresh copy
            # of it. Count zero-iteration terminations as one admission.
            iterations+=max(1,int(result.nit))
            evaluations+=int(result.nfev)
            fit=self._assess(result.x,optimizer_success=bool(result.success),
                optimizer_status=int(result.status),message=str(result.message),
                iterations=iterations,evaluations=evaluations,
                objective_ftol=objective_ftol,optimizer_scale=optimizer_scale)
            fit=replace(fit,descent_restarts=restarts,descent_probe_evaluations=probe_evaluations)
            if (fit.success or not fit.optimizer_success or not fit.local_descent_found
                    or iterations>=max_iterations or restarts>=3):
                return fit
            if callback is not None:
                callback(result.x)
            proposal,checked=self._descent_restart(result.x)
            probe_evaluations+=checked
            if proposal is None:
                return replace(fit,descent_probe_evaluations=probe_evaluations)
            z=proposal
            restarts+=1

    def _descent_restart(self,point):
        """A deterministic feasible escape only after measured local descent.

        Every proposed step is tested against the unchanged weighted cost.
        This does not replace the declared multistart portfolio or certify a
        global minimum. If no better feasible point is found, keep unresolved.
        """
        cost=self._evaluate_full(point)[0]
        _,directions=np.linalg.eigh(self._hessian(point,direct=True))
        length=max(1.,float(np.linalg.norm(point)))
        best=None;best_cost=cost;checked=0
        for relative in (1e-3,1e-2,5e-2):
            for direction in directions.T:
                for sign in (-1.,1.):
                    probe=point+sign*relative*length*direction
                    maximum=float(np.max(np.linalg.norm(self.physical_maps@probe,axis=1)))
                    if maximum>self.maximum_moment:
                        probe*=self.maximum_moment/maximum
                    probe_cost=self._evaluate_full(probe)[0];checked+=1
                    roundoff=64*np.finfo(float).eps*max(abs(cost),abs(probe_cost),np.finfo(float).tiny)
                    if probe_cost<best_cost-roundoff:
                        best,best_cost=probe.copy(),probe_cost
        return best,checked

    def reassess(self,fit):
        """Recheck a screened solution on this independently loaded objective.

        This does not trust saved costs, gradients, bounds or curvature, and
        does not run a second local search. The native stage subsequently
        checks the serialized GPX against its independent forward model.
        """
        coefficients=np.asarray(fit.coefficients,float)
        if coefficients.shape!=(self.dimension,) or not np.isfinite(coefficients).all():
            raise ValueError('Screened solution has the wrong independent moment basis')
        result=self._assess(self.to_physical@coefficients,
            optimizer_success=fit.optimizer_success,optimizer_status=fit.optimizer_status,
            message=fit.message,iterations=fit.iterations,evaluations=fit.evaluations,
            objective_ftol=fit.objective_ftol,optimizer_scale=fit.optimizer_objective_scale)
        return replace(result,descent_restarts=getattr(fit,'descent_restarts',0),
            descent_probe_evaluations=getattr(fit,'descent_probe_evaluations',0))

    def _assess(self,point,*,optimizer_success,optimizer_status,message,
                iterations,evaluations,objective_ftol,optimizer_scale):
        # Accept a state only on the original complete weighted profile.
        # Numerical reduction changes neither convergence nor descent guards.
        cost,gradient,scale=self._evaluate_full(point)
        coefficients=self.to_native@point
        full=self.offset+scale*(self.nuclear+np.einsum('i,nij,j->n',coefficients,self.quadratic,coefficients))
        background=np.linalg.lstsq(self.background,-full,rcond=None)[0] if self.background.shape[1] else np.empty(0)
        maximum=float(np.max(np.linalg.norm(self.moment_maps@coefficients,axis=1)))
        physical=maximum<=self.maximum_moment*(1+1e-8)
        active=self._constraint(point)<=1e-7*self.maximum_moment**2
        scaled_gradient=gradient/self.normalization
        if active.any():
            constraint_jacobian=self._constraint_jacobian(point)[active].T
            multipliers,_=nnls(constraint_jacobian,scaled_gradient)
            stationary=scaled_gradient-constraint_jacobian@multipliers
        else:
            stationary=scaled_gradient
        stationarity=float(np.linalg.norm(stationary))
        # Quadratic intensities have zero first derivatives at zero moments.
        # A native or SciPy "success" at a maximum is not a useful start.
        step=1e-4*max(1.,float(np.linalg.norm(point)))
        curvature,directions=np.linalg.eigh(self._hessian(point,direct=True))
        local_descent=False
        for direction in directions.T:
            for sign in [-1.,1.]:
                probe=point+sign*step*direction
                probe_max=float(np.max(np.linalg.norm(self.physical_maps@probe,axis=1)))
                if probe_max>self.maximum_moment:probe*=self.maximum_moment/probe_max
                # Compare the weighted residual costs, not a decrement in
                # Rwp-normalized cost. An arbitrarily large nuclear signal
                # can otherwise hide real magnetic descent at zero moments.
                probe_cost=self._evaluate_full(probe)[0]
                roundoff=64*np.finfo(float).eps*max(abs(cost),abs(probe_cost),np.finfo(float).tiny)
                if probe_cost<cost-roundoff:
                    local_descent=True
        return StartingPoint(coefficients=coefficients,scale=float(scale),background=background,
            weighted_cost=cost,success=bool(optimizer_success and physical and scale>0 and stationarity<=1e-7 and not local_descent),
            message=message,iterations=iterations,evaluations=evaluations,
            maximum_moment=maximum,scaled_gradient_norm=float(np.linalg.norm(gradient)/self.normalization),
            scaled_stationarity_norm=stationarity,local_descent_found=local_descent,
            optimizer_status=optimizer_status,optimizer_success=optimizer_success,objective_ftol=float(objective_ftol),
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
