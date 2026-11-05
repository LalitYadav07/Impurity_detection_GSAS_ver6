"""Native nonlinear multistart fitting baseline for structural inference.

This supplies optimizer and local-curvature diagnostics. It does NOT turn fit
ranks or Hessian errors into calibrated model probabilities or stopping advice.
"""
from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import least_squares

from .observations import CumulativeCounts, poisson_deviance_residuals


@dataclass(frozen=True)
class FitScope:
    mode_bound: float = .8
    cell_bound: float = .01
    log_scale_bound: float = .3
    background0_bound: float = 1.
    background0_lower: float | None = None
    max_evaluations: int = 80
    starts: int = 4
    seed: int = 8712
    gradient_tolerance: float = 1e-4

    def __post_init__(self):
        if any(not np.isfinite(x) or x <= 0 for x in
               [self.mode_bound,self.cell_bound,self.log_scale_bound,self.background0_bound,self.gradient_tolerance]):
            raise ValueError('Explicit finite positive fit bounds required')
        if type(self.max_evaluations) is not int or self.max_evaluations < 1 or type(self.starts) is not int or self.starts < 1:
            raise ValueError('Positive fitting budgets required')
        if self.background0_lower is not None and (not np.isfinite(self.background0_lower) or
                not -self.background0_bound <= self.background0_lower < self.background0_bound):
            raise ValueError('Background lower bound must lie inside the declared fit scope')


def fit(candidate, observation, scope=FitScope(), *, warm_starts=()):
    candidate.bind_observation(observation)
    nm = candidate.meta['inventory']['physical_rank']; nc = candidate.meta['inventory']['cell_rank']
    dimension = nm+nc+2
    bounds = np.array([scope.mode_bound]*nm+[scope.cell_bound]*nc+
                      [scope.log_scale_bound,scope.background0_bound])
    lower = -bounds.copy()
    if scope.background0_lower is not None: lower[-1] = scope.background0_lower
    scaling = np.array([max(scope.mode_bound*.25,1e-4)]*nm+[scope.cell_bound*.1]*nc+
                       [scope.log_scale_bound*.1,scope.background0_bound*.1])
    mask = observation.mask; sd = np.sqrt(observation.variance[mask])
    if mask.sum() <= dimension: raise ValueError('Insufficient observed bins for the declared fit scope')
    rng=np.random.default_rng(scope.seed); starts=[]
    if len(warm_starts) >= scope.starts:
        raise ValueError('Warm starts must leave room for an independent start within the budget')
    for value in warm_starts:
        value=np.asarray(value,float)
        if (value.shape != (dimension,) or not np.isfinite(value).all() or
                np.any(value <= lower) or np.any(value >= bounds)):
            raise ValueError('Warm start is outside this candidate fit scope')
        starts.append(value.copy())
    # Starting exactly on a bound makes SciPy's tiny feasibility nudge set
    # an equally tiny trust region; ftol can then stop before a useful step.
    origin=np.clip(np.zeros(dimension),lower+.05*(bounds-lower),bounds-.05*(bounds-lower))
    starts.append(origin.copy())
    while len(starts)<scope.starts:
        start=origin.copy()
        start[:nm]=rng.uniform(-.25,.25,nm)*scope.mode_bound
        start[nm:nm+nc]=rng.uniform(-.1,.1,nc)*scope.cell_bound
        starts.append(start)
    calls=0;started=time.monotonic();records=[];solutions=[]
    def forward(theta):
        nonlocal calls
        calls+=1
        return candidate.evaluate(theta[:nm], theta[nm:nm+nc],
                                  log_scale=theta[-2], background0_delta=theta[-1])
    def residual(theta):
        predicted = forward(theta)
        if isinstance(observation, CumulativeCounts):
            return poisson_deviance_residuals(observation, predicted)
        return (predicted[mask]-observation.y[mask])/sd
    objective = 'poisson_deviance' if isinstance(observation, CumulativeCounts) else 'supplied_variance_chi2'
    for initial in starts:
        try:
            solution=least_squares(residual,initial,bounds=(lower,bounds),x_scale=scaling,
                                    max_nfev=scope.max_evaluations,ftol=1e-9,xtol=1e-9,gtol=1e-8)
            cost=float(2*solution.cost)
            records.append(dict(success=bool(solution.success),objective=objective,objective_value=cost,message=solution.message,
                                optimality=float(solution.optimality),nfev=int(solution.nfev),parameters=solution.x.tolist()))
            if np.isfinite(cost) and np.isfinite(solution.x).all(): solutions.append(solution)
        except (ValueError,FloatingPointError) as exc:
            records.append(dict(success=False,error=str(exc)))
    if not solutions:
        return dict(state='optimization_unresolved',starts=records,seconds=time.monotonic()-started,native_calls=calls),None
    best=min(solutions,key=lambda x:x.cost)
    predicted=forward(best.x)
    _,s,vh=np.linalg.svd(best.jac,full_matrices=False)
    threshold=max(best.jac.shape)*np.finfo(float).eps*max(float(s[0]),1.)
    rank=int(np.count_nonzero(s>threshold))
    at_boundary=bool(np.any(best.x <= lower+(bounds-lower)*1e-5) or
                     np.any(best.x >= bounds-(bounds-lower)*1e-5))
    covariance=(vh.T/(s*s))@vh if rank==dimension else None
    stationary=bool(best.success and np.isfinite(best.optimality) and best.optimality <= scope.gradient_tolerance)
    weighted_data=float(np.sum(observation.y[mask]**2/observation.variance[mask]))
    chi2=float(np.sum((predicted[mask]-observation.y[mask])**2/observation.variance[mask]))
    rwp=100*np.sqrt(chi2/weighted_data) if weighted_data>0 else None
    output=dict(state='fitted' if stationary else 'optimization_unresolved',
                optimizer_reported_success=bool(best.success),stationarity_verified=stationary,
                projected_gradient=float(best.optimality),gradient_tolerance=scope.gradient_tolerance,
                observation_sha256=observation.observation_sha256,protocol_sha256=observation.protocol_sha256,
                revision=observation.revision,exposure=float(observation.exposure),
                parameters=best.x.tolist(),physical_modes=best.x[:nm].tolist(),
                cell_offsets=best.x[nm:nm+nc].tolist(),log_scale=float(best.x[-2]),
                background0_delta=float(best.x[-1]),chi2=chi2,rwp_percent=rwp,
                objective=objective,objective_value=float(2*best.cost),
                local_covariance=None if covariance is None else covariance.tolist(),jacobian_rank=rank,
                parameters_count=dimension,at_boundary=at_boundary,
                local_curvature_regular=bool(stationary and rank==dimension and not at_boundary),
                starts=records,native_calls=calls,seconds=time.monotonic()-started,
                likelihood_scope=('independent exact raw counts with known exposure conversion; reference conditional'
                                  if isinstance(observation, CumulativeCounts) else
                                  'diagonal supplied reduced-data variances; reference conditional'),
                calibrated_intervals=False,model_probabilities_available=False,counting_advice_available=False)
    output['export']=candidate.save_verified_prediction(predicted)
    output['seconds']=time.monotonic()-started
    return output,predicted
