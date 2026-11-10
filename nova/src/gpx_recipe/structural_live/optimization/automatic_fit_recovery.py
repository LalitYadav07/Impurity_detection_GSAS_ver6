"""Budgeted recovery of native structural fits; no model-exclusion authority.

Every stage is a proposal mechanism. Only the original-bin native objective
can replace the incumbent. The finite search box and chemical policy remain
unchanged. Run in an owned subprocess for a hard wall-clock deadline: a single
GSAS-II/linear-algebra call cannot be interrupted cooperatively.
"""
from dataclasses import asdict, dataclass
import time
import numpy as np
from .feasible_gauss_newton import fit_feasible_gn, observation_rwp_percent
from .fit_fast_native import residual_and_factor
from .multiaxis_fit import background_components, background_reference


@dataclass(frozen=True)
class RecoveryPolicy:
    local_seconds: float = 15.
    proposal_seconds: float = 30.
    joint_seconds: float = 40.
    starts: int = 5
    seed: int = 18453
    chemistry_seconds: float = 0.

    def __post_init__(self):
        times=[self.local_seconds,self.proposal_seconds,self.joint_seconds]
        if not np.isfinite(times).all() or min(times)<=0:
            raise ValueError('Finite positive recovery budgets required')
        if type(self.starts) is not int or not 1<=self.starts<=16:
            raise ValueError('Bounded integer restart count required')
        if type(self.seed) is not int or not 0<=self.seed<2**31:
            raise ValueError('Bounded integer seed required')
        if not np.isfinite(self.chemistry_seconds) or self.chemistry_seconds<0:
            raise ValueError('Optional chemistry recovery budget must be finite and nonnegative')


def select_recovery_candidates(rows, top=3, include_flexible=True, include_pareto=True):
    """Use the common observation-only recovery allocation policy."""
    from .catalog_recovery_policy import select_recovery_candidates as select
    return select(rows,top=top,include_flexible=include_flexible,include_pareto=include_pareto)


def improve(source,table,observation,initial,*,bank=None,native_pool=None,
            policy=None,checkpoint=None,force_native_mode_derivatives=False,evaluator=None):
    if policy is None:policy=RecoveryPolicy()
    if not isinstance(policy,RecoveryPolicy):raise TypeError('Explicit RecoveryPolicy required')
    if policy.chemistry_seconds and bank is None:
        raise ValueError('Chemistry-constrained recovery requires a prepared native start bank')
    from .native_derivative_source import derivative_source,NativeDerivativeSource
    source=derivative_source(source,evaluator=evaluator,force_native=force_native_mode_derivatives)
    started=time.monotonic()
    d=len(source.radii);z=np.asarray(initial,float)
    lower=np.r_[-np.ones(d),-.3,np.log(.25)]
    upper=np.r_[np.ones(d),.3,np.log(4.)]
    if z.shape!=(d+2,) or not np.isfinite(z).all() or np.any(z<lower) or np.any(z>upper):
        raise ValueError('Recovery requires a complete incumbent inside the original domain')
    source.check(z[:d]*source.radii)
    fixed,response=background_components(table)
    def score(point):
        signal=source.native(point[:d]*source.radii)-table.renderer.background
        rate=np.exp(point[-2])*signal+fixed+background_reference(table)*np.exp(point[-1])*response
        residual,_=residual_and_factor(observation,rate)
        return float(residual@residual),rate
    cost,rate=score(z);best=dict(parameters=z.tolist(),deviance=cost)
    records=[]
    def publish():
        receipt=dict(best=best,stages=records.copy(),seconds=time.monotonic()-started,
            policy=asdict(policy),global_optimum_certified=False,model_exclusion_authorized=False,
            rwp_percent=observation_rwp_percent(observation,rate),original_objective_preserved=True)
        if isinstance(source,NativeDerivativeSource):
            receipt['native_derivative_execution']=source.diagnostics()
        if checkpoint is not None:checkpoint(receipt,rate)
        return receipt
    publish()
    def fit_stage(name,seconds,warm,seed):
        nonlocal best,rate
        before=time.monotonic()
        try:
            fitted,prediction=fit_feasible_gn(source,table,observation,warm_pool=warm,
                seconds=seconds,starts=policy.starts,maxiter=250,adaptive_cells=True,
                multiscale=True,offline_bank=bank,precomputed_proposal_steps=bank is not None,
                balanced_restarts=True,stationarity_polls=2,screening_seed=seed,
                force_native_mode_derivatives=force_native_mode_derivatives,native_pool=native_pool)
        except TimeoutError as exc:
            # Only a budget/deadline failure is optional. Integrity, physics
            # and objective disagreements must still stop publication.
            records.append(dict(stage=name,state='budget_expired',accepted=False,
                incumbent_retained=True,reason=str(exc),seconds=time.monotonic()-before))
            publish()
            return
        candidate=np.asarray(fitted['best']['parameters']);exact,checked=score(candidate)
        if not np.isclose(exact,fitted['best']['deviance'],rtol=1e-8,atol=1e-8):
            raise ValueError('Recovery objective disagrees with a fresh native calculation')
        accepted=exact<best['deviance']
        if accepted:best=dict(parameters=candidate.tolist(),deviance=exact);rate=checked
        records.append(dict(stage=name,seconds=time.monotonic()-before,accepted=accepted,
            candidate_deviance=exact,incumbent_deviance=best['deviance'],fit=fitted))
        publish()
    fit_stage('local_prepared_cell_proposals' if bank is not None else 'local_recalibrated_cell_derivatives',policy.local_seconds,[z],policy.seed)
    warm=[np.asarray(best['parameters'])]
    if 1<=source.mode_dimensions<=32:
        from .quadratic_start_proposals import explore,ProposalUnavailable
        from .empirical_catalog import EmpiricalConfigurationRejected
        before=time.monotonic()
        try:
            guesses,details=explore(source,table,observation,warm[0],seed=policy.seed,
                seconds=policy.proposal_seconds,draw_count=180,polish_starts=8)
            warm.extend(guesses[:16]);state='completed'
        except TimeoutError as exc:
            details=dict(reason=str(exc));state='budget_expired'
        except (EmpiricalConfigurationRejected,ProposalUnavailable) as exc:
            details=dict(reason=str(exc),incumbent_retained=True)
            state='proposal_unavailable'
        records.append(dict(stage='native_checked_quadratic_starts',state=state,
            seconds=time.monotonic()-before,details=details,likelihood_replaced=False))
    else:
        records.append(dict(stage='native_checked_quadratic_starts',state='unsupported_dimension',
            mode_dimensions=source.mode_dimensions,dimensions_truncated=False))
    publish()
    fit_stage('joint_native_multistart',policy.joint_seconds,warm,policy.seed+1)
    if policy.chemistry_seconds:
        if bank is None:raise ValueError('Chemistry-constrained recovery requires a prepared native start bank')
        from .chemistry_sqp_native_fit import fit
        before=time.monotonic()
        try:
            fitted,prediction=fit(source,table,observation,bank,np.asarray(best['parameters']),
                seconds=policy.chemistry_seconds,force_native_mode_derivatives=force_native_mode_derivatives)
        except TimeoutError as exc:
            records.append(dict(stage='chemistry_constrained_native_gn',state='budget_expired',accepted=False,
                incumbent_retained=True,reason=str(exc),seconds=time.monotonic()-before))
            return publish(),rate
        candidate=np.asarray(fitted['best']['parameters']);exact,checked=score(candidate)
        if not np.isclose(exact,fitted['best']['deviance'],rtol=1e-8,atol=1e-8):
            raise ValueError('Chemistry-constrained recovery differs from the native objective')
        accepted=exact<best['deviance']
        if accepted:best=dict(parameters=candidate.tolist(),deviance=exact);rate=checked
        records.append(dict(stage='chemistry_constrained_native_gn',seconds=time.monotonic()-before,accepted=accepted,
            candidate_deviance=exact,incumbent_deviance=best['deviance'],fit=fitted))
        publish()
    return publish(),rate
