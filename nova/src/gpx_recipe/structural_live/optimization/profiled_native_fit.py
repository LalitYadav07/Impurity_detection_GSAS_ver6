"""Candidate-local alternating native optimization with weak-mode restarts.

All allowed modes remain jointly free. Mode blocks reuse GSAS-II scattering
and GSAS-II profile responses only at one exact cell. Every accepted move is
independently checked against the full native objective. Weak-mode moves may
seed a worse local trajectory; the published incumbent never becomes worse.
No truth values, custom broadening, probability or global certificate.
"""
import time
import numpy as np
from .native_cell_derivative import NativeCellDerivative
from .damped_step import bounded_damped_step
from .fit_fast_native import residual_and_factor
from .feasible_gauss_newton import observation_rwp_percent,select_ensemble_starts
from .multiaxis_fit import background_components,background_reference
from .reference_secants import feasible_secant,FeasibleDirectionUnavailable
from .profiled_nuisance import ConditionalNuisance


class CellDirectionUnavailable(RuntimeError):
    """A constrained coordinate has no feasible finite-difference stencil."""


def fit(source,table,observation,bank,*,seconds=120.,starts=4,seed=90231,warm=None,warm_pool=None,
        mode_iterations=1,cell_iterations=1,kicks=False,evaluator=None,force_native_mode_derivatives=False,screening_exclusions=None):
    if not np.isfinite(seconds) or seconds<=0 or type(starts)is not int or starts<1:
        raise ValueError('Positive finite time and integer start count required')
    if type(mode_iterations)is not int or mode_iterations<1 or type(cell_iterations)is not int or cell_iterations<1:
        raise ValueError('Positive block iteration counts required')
    if kicks:raise ValueError('Weak-mode restarts are not qualified for profiled nuisance fitting')
    began=time.monotonic();deadline=began+seconds;s=source;m=s.mode_dimensions;d=len(s.radii);mask=observation.mask
    lo=np.r_[-np.ones(d),-.3,np.log(.25)];hi=np.r_[np.ones(d),.3,np.log(4.)]
    cache=NativeCellDerivative(s,max_cells=2,force_native_derivatives=force_native_mode_derivatives) if evaluator is None else evaluator
    fixed,response=background_components(table);background=background_reference(table)
    best=None;trace=[];blocks=[];restarts=[];blocked_starts=[];attempts=[];native_calls=0;rejected=0;cache_checks=0;max_error=0.;max_sigma=0.
    rng=np.random.default_rng(seed)
    def check_time():
        if time.monotonic()>=deadline:raise TimeoutError('Native alternating fit allocation exhausted')
    def allowed(z):
        return z.shape==(d+2,) and np.isfinite(z).all() and np.all(z>=lo) and np.all(z<=hi) and bool(np.isfinite(s.chemistry.log_kernel((z[:d]*s.radii)[None,:])[0]))
    def rate_for(z,curve):return np.exp(z[-2])*(curve-table.renderer.background)+fixed+background*np.exp(z[-1])*response
    def score(z,cached=False,remember=True):
        nonlocal best,native_calls
        check_time()
        if not allowed(z):raise ValueError('Infeasible structural geometry reached native optimizer')
        curve=cache.evaluate(z[:d]*s.radii) if cached else s.native(z[:d]*s.radii)
        native_calls+=int(not cached)
        conditional=ConditionalNuisance(observation,curve-table.renderer.background,fixed,background*response)
        cost,rate,residual,factor=conditional.solve(np.exp(z[-2:]))
        adjusted=z.copy();adjusted[-2:]=np.clip(np.log(conditional.coefficients),lo[-2:],hi[-2:])
        if not cached and remember and (best is None or cost<best['cost']):
            best=dict(z=adjusted.copy(),cost=cost,rate=rate.copy())
            trace.append(dict(seconds=time.monotonic()-began,deviance=cost,parameters=adjusted.tolist(),rwp_percent=observation_rwp_percent(observation,rate)))
        return cost,rate,residual,factor,adjusted
    def verify(z,predicted):
        nonlocal cache_checks,max_error,max_sigma
        exact=score(z);cache_checks+=1
        error=float(np.linalg.norm((predicted[1]-exact[1])[mask])/max(np.linalg.norm(exact[1][mask]),1e-30))
        sigma=float(np.max(abs((predicted[1]-exact[1])[mask])/np.sqrt(observation.variance[mask])))
        max_error=max(max_error,error);max_sigma=max(max_sigma,sigma)
        if error>1e-8 or sigma>.01:raise ValueError('Native fixed-cell response disagrees with the full native profile')
        return exact
    def jacobian(z,mode):
        check_time()
        if mode:
            curve,j=cache.mode_jacobian(z[:d]*s.radii)
            conditional=ConditionalNuisance(observation,curve-table.renderer.background,fixed,background*response)
            _,rate,residual,factor=conditional.solve(np.exp(z[-2:]))
            raw=conditional.rate_jacobian(j*s.radii[:m,None])
            return (raw[:,mask]*factor).T,residual
        cost,rate,residual,factor,adjusted=score(z)
        def profile(point):
            trial=z.copy();trial[:d]=point;return score(trial,remember=False)[1]
        raw=[]
        for i in range(m,d):
            check_time()
            try:
                column,_=feasible_secant(profile,z[:d],i,bank.proposal_cell_steps[i],
                    allowed=lambda p:allowed(np.r_[p,z[-2:]]))
            except FeasibleDirectionUnavailable as exc:
                raise CellDirectionUnavailable(f'No feasible lattice stencil for coordinate {i}') from exc
            raw.append(column)
        return (np.asarray(raw)[:,mask]*factor).T,residual
    def block(z,mode,iterations):
        nonlocal rejected
        columns=np.arange(m) if mode else np.arange(m,d)
        z=z.copy();current=score(z);z=current[4];damping=1e-3;accepted=[];start=time.monotonic();reason='iteration_limit'
        # A forced native derivative already evaluates the complete profile.
        # Building a fixed-cell reflection-response matrix for one interleaved
        # step costs more than these few direct profiles, especially in TOF.
        cached_mode=mode and not force_native_mode_derivatives
        if cached_mode:current=verify(z,score(z,cached=True,remember=False))
        for iteration in range(iterations):
            check_time();j,r=jacobian(z,mode)
            step,info=bounded_damped_step(j,r,damping,lo[columns]-z[columns],hi[columns]-z[columns],radius=.4)
            if info['information_score']<1e-4:reason='small_local_score';break
            descent=float((j.T@r)@step)
            if descent>=0:damping*=10.;continue
            found=False
            for power in range(12):
                trial=z.copy();trial[columns]+=step*(.5**power)
                if not allowed(trial):rejected+=1;continue
                proposed=score(trial,cached=cached_mode)
                if proposed[0]<=current[0]+2e-4*(.5**power)*descent:
                    exact=verify(trial,proposed) if cached_mode else proposed
                    if exact[0]<=current[0]+2e-4*(.5**power)*descent:
                        z=exact[4];current=exact;accepted.append(current[0]);damping=max(damping*.3,1e-10);found=True;break
            if not found:damping*=10.
            if damping>1e7:reason='native_descent_stalled';break
            if found and np.max(abs(step*(.5**power)))<1e-8:reason='small_step';break
        blocks.append(dict(kind='modes' if mode else 'cell',seconds=time.monotonic()-start,accepted_costs=accepted,reason=reason))
        return z,current[0]
    warm_ranked=[];prior_ranked=[];selected=[]
    exclusions=[] if screening_exclusions is None else [np.asarray(v,float) for v in screening_exclusions]
    if any(v.shape!=(d+2,) or not np.isfinite(v).all() for v in exclusions):raise ValueError('Complete finite excluded starts required')
    for point in bank.points:
        z=np.r_[point,0.,0.]
        if allowed(z):score(z);break
    try:
        retained=[] if warm_pool is None else list(warm_pool)
        if warm is not None:retained.insert(0,warm)
        for item in retained:
            point=np.asarray(item,float)
            warm_ranked.append((score(point)[0],point))
        weights=1/np.sqrt(observation.variance[mask])
        for point,curve in zip(bank.points,bank.curves):
            check_time();signal=curve-table.renderer.background
            conditional=ConditionalNuisance(observation,signal,fixed,background*response)
            cost,rate,residual,factor=conditional.solve()
            z=np.r_[point,np.clip(np.log(conditional.coefficients),lo[-2:],hi[-2:])]
            if not allowed(z):rejected+=1;continue
            if any(np.linalg.norm(z[:d]-previous[:d])<=.2 for previous in exclusions):continue
            prior_ranked.append((cost,z))
        selected=select_ensemble_starts(warm_ranked,prior_ranked,starts,d)
        if not selected:raise ValueError('No feasible native starting point')
    except TimeoutError:
        selected=[]
    whole_deadline=deadline
    for index,initial in enumerate(selected):
        remaining=whole_deadline-time.monotonic()
        if remaining<=0:break
        deadline=time.monotonic()+remaining/(len(selected)-index)
        attempt=dict(start=index,initial_parameters=initial.tolist(),iterations=0,reason='running')
        attempts.append(attempt)
        try:
            z=initial.copy();cost=score(z)[0];cycles=0;restart_points=[]
            while True:
                attempt['iterations']+=1
                before=cost
                # Peak-position errors must not spend an entire short start
                # being absorbed into atomic intensities. Interleave one cell
                # and one displacement step, beginning with the lattice.
                if d>m:z,cost=block(z,False,cell_iterations)
                if m:z,cost=block(z,True,mode_iterations)
                cycles+=1
                if before-cost<1e-9*max(cost,1.) and (not kicks or not restart_points):
                    attempt['reason']='local_block_progress_small';break
        except CellDirectionUnavailable as exc:
            attempt.update(reason='proposal_unavailable',error=str(exc))
            blocked_starts.append(dict(start=index,reason=str(exc),incumbent_retained=best is not None,
                state='proposal_unavailable',convergence_claimed=False,seconds=time.monotonic()-began))
        except TimeoutError:attempt['reason']='per_start_time_budget'
        finally:
            attempt['final_parameters']=z.tolist()
            deadline=whole_deadline
    if best is None:raise ValueError('No complete independent native evaluation')
    z=best['z'];s.check(z[:d]*s.radii)
    return dict(best=dict(parameters=z.tolist(),deviance=best['cost']),rwp_percent=observation_rwp_percent(observation,best['rate']),
        seconds=time.monotonic()-began,method='profiled_native_alternating',exact_native_deviance=best['cost'],
        weak_mode_restarts_enabled=kicks,force_native_mode_derivatives=force_native_mode_derivatives,starts=len(selected),mode_dimensions=m,cell_dimensions=d-m,
        blocks=blocks,restarts=restarts,blocked_starts=blocked_starts,attempts=attempts,trace=trace,full_native_calls=native_calls,chemistry_rejections=rejected,
        independent_cache_checks=cache_checks,maximum_native_relative_error=max_error,maximum_native_sigma_error=max_sigma,
        native_response_cache=cache.diagnostics(),all_allowed_modes_retained=True,truth_used=False,
        global_optimum_certified=False,model_exclusion_authorized=False),best['rate']
