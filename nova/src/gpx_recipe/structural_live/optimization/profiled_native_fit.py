"""Candidate-local alternating native optimization with weak-mode restarts.

All allowed modes remain jointly free. Mode blocks reuse GSAS-II scattering
and GSAS-II profile responses only at one exact cell. Every accepted move is
independently checked against the full native objective. Weak-mode moves may
seed a worse local trajectory; the published incumbent never becomes worse.
No truth values, custom broadening, probability or global certificate.
"""
import time
from collections import OrderedDict
import numpy as np
from .native_cell_derivative import NativeCellDerivative
from .damped_step import bounded_damped_step
from .fit_fast_native import residual_and_factor
from .feasible_gauss_newton import observation_rwp_percent,select_ensemble_starts
from .multiaxis_fit import background_components,background_reference
from .reference_secants import feasible_secant,FeasibleDirectionUnavailable
from .profiled_nuisance import ConditionalNuisance
from .cell_derivative_model import CellDerivativeModel
from .native_cell_proposals import for_source as cell_proposals_for_source,ProposalUnavailable


class CellDirectionUnavailable(RuntimeError):
    """A constrained coordinate has no feasible finite-difference stencil."""


def fit(source,table,observation,bank,*,seconds=120.,starts=4,seed=90231,warm=None,warm_pool=None,
        mode_iterations=1,cell_iterations=1,kicks=False,evaluator=None,force_native_mode_derivatives=False,
        screening_exclusions=None,force_native_cell_directions=False,joint_steps=False,
        qualification_iterations=None):
    """Use the same complete native objective for either update schedule.

    ``joint_steps`` is an internal qualification control, not a user accuracy
    choice. Coupled columns retain native acceptance and fresh-cell retries.
    The existing schedule stays the default until full-fit comparisons qualify
    the coupled update; a low point-fit Rwp alone is insufficient. Internal
    ``qualification_iterations`` bounds each start's complete outer updates
    for matched-work benchmarks. Reaching it is explicitly unresolved and
    never a convergence criterion; production callers leave it unset.
    """
    if not np.isfinite(seconds) or seconds<=0 or type(starts)is not int or starts<1:
        raise ValueError('Positive finite time and integer start count required')
    if type(mode_iterations)is not int or mode_iterations<1 or type(cell_iterations)is not int or cell_iterations<1:
        raise ValueError('Positive block iteration counts required')
    if type(joint_steps)is not bool:raise ValueError('Joint native update selection must be boolean')
    if qualification_iterations is not None and (type(qualification_iterations)is not int
            or not 1<=qualification_iterations<=1000):
        raise ValueError('Qualification iterations must be an integer from one to one thousand')
    if kicks:raise ValueError('Weak-mode restarts are not qualified for profiled nuisance fitting')
    began=time.monotonic();deadline=began+seconds;s=source;m=s.mode_dimensions;d=len(s.radii);mask=observation.mask
    lo=np.r_[-np.ones(d),-.3,np.log(.25)];hi=np.r_[np.ones(d),.3,np.log(4.)]
    cache=NativeCellDerivative(s,max_cells=2,force_native_derivatives=force_native_mode_derivatives) if evaluator is None else evaluator
    fixed,response=background_components(table);background=background_reference(table)
    best=None;trace=[];blocks=[];restarts=[];blocked_starts=[];attempts=[];native_calls=0;rejected=0;cache_checks=0;max_error=0.;max_sigma=0.
    rng=np.random.default_rng(seed)
    cell_model=CellDerivativeModel(m,d)
    cell_proposals=None if force_native_cell_directions else cell_proposals_for_source(s)
    pending_native_cell_refresh=False
    native_profiles=OrderedDict();profile_hits=0
    native_probe_calls=0
    # Alternation divides one local solve into short blocks. Preserve each
    # block's trust adaptation until this start ends, rather than restarting
    # LM at its initial damping after every accepted or unsuccessful step.
    block_damping={True:1e-3,False:1e-3,None:1e-3}
    def check_time():
        if time.monotonic()>=deadline:raise TimeoutError('Native alternating fit allocation exhausted')
    def allowed(z):
        return z.shape==(d+2,) and np.isfinite(z).all() and np.all(z>=lo) and np.all(z<=hi) and bool(np.isfinite(s.chemistry.log_kernel((z[:d]*s.radii)[None,:])[0]))
    def rate_for(z,curve):return np.exp(z[-2])*(curve-table.renderer.background)+fixed+background*np.exp(z[-1])*response
    def native_curve(point,*,prepare_response=True):
        nonlocal native_calls,profile_hits,native_probe_calls
        point=np.asarray(point,float);key=tuple(point)
        needs=getattr(cache,'needs_capture',None)
        prepare=prepare_response and callable(needs) and needs(point)
        if key in native_profiles and not prepare:
            owner=getattr(getattr(s,'fast',None),'native',None)
            if owner is not None:owner._check()
            profile_hits+=1;native_profiles.move_to_end(key)
            return native_profiles[key].copy()
        native=getattr(cache,'native_evaluate',s.native) if prepare_response else s.native
        curve=native(point);native_calls+=1
        native_probe_calls+=int(not prepare_response)
        remember_curve(point,curve)
        return curve
    def remember_curve(point,curve):
        curve=np.asarray(curve,float)
        if curve.shape!=observation.y.shape or not np.isfinite(curve).all():
            raise ValueError('Complete finite native structural profile required')
        key=tuple(point);native_profiles[key]=curve.copy();native_profiles.move_to_end(key)
        while len(native_profiles)>2:native_profiles.popitem(last=False)
    def score(z,cached=False,remember=True,prepare_response=True):
        nonlocal best,native_calls
        check_time()
        if not allowed(z):raise ValueError('Infeasible structural geometry reached native optimizer')
        curve=cache.evaluate(z[:d]*s.radii) if cached else native_curve(z[:d]*s.radii,prepare_response=prepare_response)
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
    def jacobian(z,mode,*,native_cells=False):
        nonlocal native_calls
        check_time()
        if mode is None:
            modes,mode_residual=jacobian(z,True)
            cells,native_residual=jacobian(z,False,native_cells=native_cells)
            if mode_residual.shape!=native_residual.shape or modes.shape[0]!=cells.shape[0]:
                raise ValueError('Joint native derivative columns disagree on their observation')
            return np.column_stack([modes,cells]),native_residual
        if mode:
            curve,j=cache.mode_jacobian(z[:d]*s.radii)
            # A full-native derivative call already evaluated this same curve.
            # Factored curves enter the exact memo only after native verify().
            if force_native_mode_derivatives:remember_curve(z[:d]*s.radii,curve)
            conditional=ConditionalNuisance(observation,curve-table.renderer.background,fixed,background*response)
            _,rate,residual,factor=conditional.solve(np.exp(z[-2:]))
            raw=conditional.rate_jacobian(j*s.radii[:m,None])
            cell_model.observe_modes(z[:d],rate,raw)
            return (raw[:,mask]*factor).T,residual
        cost,rate,residual,factor,adjusted=score(z)
        def profile(point):
            trial=z.copy();trial[:d]=point;return score(trial,remember=False,prepare_response=False)[1]
        def build(columns=None):
            raw=[]
            for i in range(m,d) if columns is None else columns:
                check_time()
                try:
                    column,_=feasible_secant(profile,z[:d],i,bank.proposal_cell_steps[i],
                        allowed=lambda p:allowed(np.r_[p,z[-2:]]))
                except FeasibleDirectionUnavailable as exc:
                    raise CellDirectionUnavailable(f'No feasible lattice stencil for coordinate {i}') from exc
                raw.append(column)
            return np.asarray(raw).reshape(-1,len(rate))
        def propose():
            nonlocal native_calls
            before=cell_proposals.anchor_calls
            kernel_for=getattr(cache,'kernel_for',None)
            prepared_kernel=kernel_for(z[:d]*s.radii) if callable(kernel_for) else None
            kernel_options={} if prepared_kernel is None else dict(prepared_kernel=prepared_kernel)
            if getattr(cell_proposals,'supports_native_anchor',False):
                kernel_options['native_anchor']=native_profiles.get(tuple(z[:d]*s.radii))
            try:
                curve,raw,usable=cell_proposals.columns(z[:d],bank.proposal_cell_steps,
                    allowed=lambda p:allowed(np.r_[p,z[-2:]]),check_time=check_time,**kernel_options)
            finally:
                native_calls+=cell_proposals.anchor_calls-before
            remember_curve(z[:d]*s.radii,curve)
            if not np.any(usable):
                raise ProposalUnavailable('All cell columns changed native reflection membership')
            conditional=ConditionalNuisance(observation,curve-table.renderer.background,fixed,background*response)
            _,anchor_rate,_,_=conditional.solve(np.exp(z[-2:]))
            if np.linalg.norm(anchor_rate-rate)>1e-10*max(np.linalg.norm(rate),1e-30):
                raise ValueError('Native cell proposal anchor changed the original objective')
            result=conditional.rate_jacobian(raw)
            if not np.all(usable):
                result[~usable]=build(np.arange(m,d)[~usable])
            return result
        weights=np.zeros(len(rate));weights[mask]=abs(factor)
        if cell_proposals is not None:
            raw=cell_model.columns(z[:d],rate,weights,build,propose=propose,force_native=native_cells)
        else:
            raw=cell_model.columns(z[:d],rate,weights,build)
        return (raw[:,mask]*factor).T,residual
    def block(z,mode,iterations):
        nonlocal rejected,pending_native_cell_refresh
        columns=np.arange(d) if mode is None else np.arange(m) if mode else np.arange(m,d)
        has_cells=mode is not True and d>m
        z=z.copy();current=score(z);z=current[4];damping=block_damping[mode];initial_damping=damping
        accepted=[];start=time.monotonic();reason='iteration_limit'
        # Captured responses add no second profile construction. A context
        # that cannot capture uses the original native acceptance path, rather
        # than evaluating a native fallback twice under a cached-mode label.
        checked_modes=mode is not False and not force_native_mode_derivatives
        if checked_modes and hasattr(cache,'has_response'):
            checked_modes=cache.has_response(z[:d]*s.radii)
        if checked_modes:current=verify(z,score(z,cached=True,remember=False))
        # A joint trial changes the cell, so it must use the current native
        # profile rather than the old cell's displacement response matrix.
        cached_mode=mode is True and checked_modes
        for iteration in range(iterations):
            found=False;stationary=False
            # Neither a recycled direction nor a fresh peak proposal can
            # declare stationarity or end a stalled trajectory. Retry this
            # SAME iterate once with original native cell differences.
            for retry in range(2):
                force_cells=has_cells and (retry>0 or pending_native_cell_refresh)
                check_time();j,r=jacobian(z,mode,native_cells=force_cells)
                if force_cells:pending_native_cell_refresh=False
                recycled=has_cells and (cell_model.last_reused or cell_model.last_requires_native_retry)
                step,info=bounded_damped_step(j,r,damping,lo[columns]-z[columns],hi[columns]-z[columns],radius=.4)
                if info['information_score']<1e-4:
                    if recycled:cell_model.invalidate();continue
                    reason='small_local_score';stationary=True;break
                descent=float((j.T@r)@step)
                if descent>=0:
                    if recycled:cell_model.invalidate();continue
                    damping*=10.;break
                for power in range(12):
                    trial=z.copy();trial[columns]+=step*(.5**power)
                    if not allowed(trial):rejected+=1;continue
                    proposed=score(trial,cached=cached_mode)
                    if proposed[0]<=current[0]+2e-4*(.5**power)*descent:
                        exact=verify(trial,proposed) if cached_mode else proposed
                        if exact[0]<=current[0]+2e-4*(.5**power)*descent:
                            z=exact[4];current=exact;accepted.append(current[0]);damping=max(damping*.3,1e-10);found=True;break
                if found:break
                if recycled:cell_model.invalidate();continue
                damping*=10.;break
            if stationary:break
            if damping>1e7:reason='native_descent_stalled';break
            reason='iteration_limit' if found else 'damping_increased'
            if found and np.max(abs(step*(.5**power)))<1e-8:reason='small_step';break
        block_damping[mode]=damping
        blocks.append(dict(start=index,kind='joint' if mode is None else 'modes' if mode else 'cell',seconds=time.monotonic()-start,
            accepted_costs=accepted,reason=reason,initial_damping=initial_damping,final_damping=damping))
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
        selected=select_ensemble_starts(warm_ranked,prior_ranked,starts,d,mode_dimensions=m)
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
        cell_model.invalidate()
        block_damping.update({True:1e-3,False:1e-3,None:1e-3})
        pending_native_cell_refresh=False
        try:
            z=initial.copy();cost=score(z)[0];cycles=0;restart_points=[]
            while True:
                attempt['iterations']+=1
                before=cost
                block_begin=len(blocks)
                # Peak-position errors must not spend an entire short start
                # being absorbed into atomic intensities. Interleave one cell
                # and one displacement step, beginning with the lattice.
                if joint_steps and m and d>m:
                    z,cost=block(z,None,max(mode_iterations,cell_iterations))
                else:
                    if d>m:z,cost=block(z,False,cell_iterations)
                    if m:z,cost=block(z,True,mode_iterations)
                cycles+=1
                if qualification_iterations is not None and cycles>=qualification_iterations:
                    attempt['reason']='qualification_iteration_limit';break
                if before-cost<1e-9*max(cost,1.) and (not kicks or not restart_points):
                    reasons={row['reason'] for row in blocks[block_begin:]}
                    if 'native_descent_stalled' in reasons:
                        attempt['reason']='native_descent_stalled';break
                    if 'damping_increased' in reasons:
                        continue
                    if cell_model.last_requires_native_retry or (cell_model.model is not None and cell_model.model['age']):
                        cell_model.invalidate()
                        pending_native_cell_refresh=True
                        continue
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
        seconds=time.monotonic()-began,method='profiled_native_joint' if joint_steps else 'profiled_native_alternating',exact_native_deviance=best['cost'],
        weak_mode_restarts_enabled=kicks,force_native_mode_derivatives=force_native_mode_derivatives,starts=len(selected),mode_dimensions=m,cell_dimensions=d-m,
        qualification_iterations_per_start=qualification_iterations,
        blocks=blocks,restarts=restarts,blocked_starts=blocked_starts,attempts=attempts,trace=trace,full_native_calls=native_calls,chemistry_rejections=rejected,
        independent_cache_checks=cache_checks,maximum_native_relative_error=max_error,maximum_native_sigma_error=max_sigma,
        native_response_cache=cache.diagnostics(),all_allowed_modes_retained=True,truth_used=False,
        cell_derivative_proposals=cell_model.diagnostics(),exact_profile_memo=dict(hits=profile_hits,entries=len(native_profiles),maximum_entries=2),
        native_cell_probe_calls=native_probe_calls,
        native_cell_proposals=cell_proposals.diagnostics() if cell_proposals is not None else None,
        global_optimum_certified=False,model_exclusion_authorized=False),best['rate']
