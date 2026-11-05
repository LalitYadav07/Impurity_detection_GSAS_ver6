"""Feasible damped Gauss–Newton proposals, accepted by the exact native cost.

Contact checks precede every trial profile. Damping and line search propose
steps only; neither replaces the likelihood or certifies a global optimum.
"""
import time
import numpy as np
from gpx_recipe.structural_live.observations import CumulativeCounts,count_rwp_percent
from scipy.stats import qmc
from .constrained_native_fit import FeasibleNativeObjective
from .multiaxis_fit import background_reference,background_components
from .fit_fast_native import residual_and_factor
from .damped_step import damped_step,bounded_damped_step
from .reference_secants import FeasibleDirectionUnavailable

def observation_rwp_percent(observation,prediction):
    """Keep Gaussian diagnostics distinct from verified raw-count likelihoods."""
    if isinstance(observation,CumulativeCounts):return count_rwp_percent(observation,prediction)
    value=np.asarray(prediction,float);mask=observation.mask
    if value.shape!=observation.y.shape or not np.isfinite(value[mask]).all():
        raise ValueError('Finite predictions on the observation grid are required')
    denominator=float(np.sum(observation.y[mask]**2/observation.variance[mask]))
    if denominator==0:return None
    return float(100*np.sqrt(np.sum((value[mask]-observation.y[mask])**2/observation.variance[mask])/denominator))

def screen_curve(source,cache,point,anchor,native):
    """Amortize a response cache only at the repeatedly visited exact cell."""
    if cache is not None and np.array_equal(np.asarray(point)[source.mode_dimensions:],anchor):
        return cache.evaluate(np.asarray(point)*source.radii),True
    return native(point),False

def select_ensemble_starts(warm,prior,starts,dimensions):
    """Reserve independent restarts; rank the reserved set by current data."""
    if starts==1:
        ranked=sorted(warm+prior,key=lambda row:row[0])
        return [ranked[0][1]] if ranked else []
    selected=[]
    def add(rows,maximum):
        added=0
        for cost,point in sorted(rows,key=lambda row:row[0]):
            if all(np.linalg.norm(point[:dimensions]-p[:dimensions])>.2 for _,p in selected):
                selected.append((cost,point));added+=1
            if added>=maximum:break
    if starts>1 and warm:add(warm,min(2,starts-1))
    add(prior,max(starts-len(selected),1))
    if len(selected)<starts:add(warm+prior,starts-len(selected))
    return [point for _,point in sorted(selected,key=lambda row:row[0])[:starts]]

def fit_feasible_gn(source,table,observation,*,warm=None,warm_pool=None,seconds=90.,starts=4,maxiter=70,adaptive_cells=False,multiscale=False,recycle_cells=False,box_steps=False,offline_bank=None,balanced_restarts=False,precomputed_proposal_steps=False,aggregation_width=1,mode_substeps=0,force_native_mode_derivatives=False,stationarity_polls=0,screening_seed=82519,native_pool=None,screening_exclusions=None):
    if not np.isfinite(seconds) or seconds<=0 or type(starts)is not int or starts<1 or type(maxiter)is not int or maxiter<1:
        raise ValueError('Positive finite fit budget and integer start/iteration limits required')
    if precomputed_proposal_steps and offline_bank is None:raise ValueError('Prepared proposal steps require a bound native bank')
    if type(aggregation_width)is not int or aggregation_width<1:raise ValueError('Positive integer count-aggregation width required')
    if type(mode_substeps)is not int or not 0<=mode_substeps<=20:raise ValueError('Bounded integer mode-substep limit required')
    if mode_substeps and aggregation_width!=1:raise ValueError('Mode blocks currently require the original-bin objective')
    if type(stationarity_polls)is not int or not 0<=stationarity_polls<=3:raise ValueError('At most three native stationarity polls per start are supported')
    if type(screening_seed)is not int or not 0<=screening_seed<2**31:raise ValueError('Nonnegative bounded integer screening seed required')
    grouping=None
    if aggregation_width>1:
        from .multiresolution_native_fit import CountAggregation
        grouping=CountAggregation(observation,aggregation_width)
    begin=time.monotonic();s=source;d=len(s.radii);mask=observation.mask
    if recycle_cells:
        from .recycled_cell_objective import RecycledCellObjective
        obj=RecycledCellObjective(s,table,observation,deadline=begin+seconds,native_pool=native_pool)
    else:obj=FeasibleNativeObjective(s,table,observation,deadline=begin+seconds,native_pool=native_pool)
    lo=np.r_[-np.ones(d),-.3,np.log(.25)];hi=np.r_[np.ones(d),.3,np.log(4.)]
    exclusions=[] if screening_exclusions is None else [np.asarray(v,float) for v in screening_exclusions]
    if any(v.shape!=(d+2,) or not np.isfinite(v).all() for v in exclusions):
        raise ValueError('Screening exclusions require complete finite previous starting coordinates')
    excluded_prior_starts=0
    fixed,response=background_components(table);background=background_reference(table)
    best=None;best_full=None;rejected=0;trials=0;trace=[];linear_algebra_seconds=0.;active=None
    direct_screening_calls=0;cached_screening_calls=0;screening_seconds=None;proposal_selection_seconds=0.
    bounded_subproblems=0;bounded_subproblem_failures=0
    mode_blocks=[];mode_evaluator=None
    if mode_substeps:
        from .native_cell_derivative import NativeCellDerivative
        mode_evaluator=NativeCellDerivative(s,max_cells=2,force_native_derivatives=force_native_mode_derivatives)
    def evaluate(z):
        nonlocal best,best_full,trials
        if not obj.allowed(z[:d]) or np.any(z<lo) or np.any(z>hi):raise ValueError('Internal infeasible proposal guard')
        signal=obj.native(z[:d])-table.renderer.background;trials+=1
        rate=np.exp(z[-2])*signal+fixed+background*np.exp(z[-1])*response
        original,_=residual_and_factor(observation,rate);full_cost=float(original@original)
        residual=original if grouping is None else grouping.evaluate(rate)[0];cost=float(residual@residual)
        if best is None or cost<best['cost']:best=dict(z=z.copy(),cost=cost,rate=rate.copy())
        if best_full is None or full_cost<best_full['cost']:best_full=dict(z=z.copy(),cost=full_cost,rate=rate.copy())
        return cost,rate
    ranked=[];warm_ranked=[];prior_ranked=[];cache=None;anchor=None
    if offline_bank is not None:
        if not multiscale:raise ValueError('Offline starts require the multiscale start policy')
        points=offline_bank.points
    elif multiscale:
        from .structural_start_bank import multiscale_starts
        from .native_cell_intensity import ExactCellIntensity
        anchor=np.zeros(source.cell_dimensions) if warm is None or warm_pool is not None else np.asarray(warm)[source.mode_dimensions:d]
        points=multiscale_starts(source.mode_dimensions,source.cell_dimensions,cell_anchor=anchor,seed=screening_seed)
        cache=ExactCellIntensity(source)
    else:points=[np.zeros(d),*(qmc.Sobol(d,scramble=True,seed=screening_seed).random_base2(6)*1.6-.8)]
    weights=1/np.sqrt(observation.variance[mask])
    try:
        # Always retain one independently native feasible curve, even if a
        # subsequent screening or derivative budget expires.
        evaluate(np.zeros(d+2))
        if warm is not None:
            warm=np.clip(np.asarray(warm,float),lo,hi)
            if obj.allowed(warm[:d]):warm_ranked.append((evaluate(warm)[0],warm))
        for point in ([] if warm_pool is None else warm_pool):
            point=np.clip(np.asarray(point,float),lo,hi)
            if point.shape!=(d+2,):raise ValueError('Retained start has wrong dimension')
            if obj.allowed(point[:d]) and all(np.linalg.norm(point-z)>1e-7 for _,z in warm_ranked):
                warm_ranked.append((evaluate(point)[0],point))
        for point_index,point in enumerate(points):
            if not obj.allowed(point):rejected+=1;continue
            if time.monotonic()>begin+seconds:raise TimeoutError('Screening budget exhausted')
            if offline_bank is None:
                curve,cached=screen_curve(s,cache,point,anchor,obj.native)
                cached_screening_calls+=int(cached);direct_screening_calls+=int(not cached)
            else:curve=offline_bank.curves[point_index]
            signal=curve-table.renderer.background
            design=np.column_stack([signal[mask],response[mask]])
            a,b=np.linalg.lstsq(design*weights[:,None],(observation.y[mask]-fixed[mask])*weights,rcond=None)[0]
            z=np.clip(np.r_[point,np.log(np.clip(a,np.exp(-.3),np.exp(.3))),np.log(np.clip(b/background,.25,4.))],lo,hi)
            rate=np.exp(z[-2])*signal+fixed+background*np.exp(z[-1])*response
            residual=residual_and_factor(observation,rate)[0] if grouping is None else grouping.evaluate(rate)[0]
            prior_ranked.append((float(residual@residual),z))
        if exclusions:
            retained=[(cost,point) for cost,point in prior_ranked
                      if all(np.linalg.norm(point[:d]-old[:d])>.2 for old in exclusions)]
            excluded_prior_starts=len(prior_ranked)-len(retained)
            prior_ranked=retained
        ranked=sorted(warm_ranked+prior_ranked,key=lambda r:r[0]);selected=[]
        if warm_pool is not None:selected=select_ensemble_starts(warm_ranked,prior_ranked,starts,d)
        else:
            for _,z in ranked:
                if all(np.linalg.norm(z[:d]-v[:d])>.2 for v in selected):selected.append(z)
                if len(selected)>=starts:break
        screening_seconds=time.monotonic()-begin
        for start_index,z in enumerate(selected):
            if balanced_restarts:
                remaining=begin+seconds-time.monotonic()
                if remaining<=0:raise TimeoutError('Total restart budget exhausted')
                obj.deadline=time.monotonic()+remaining/(len(selected)-start_index)
            try:
                if cache is not None and np.array_equal(z[source.mode_dimensions:d],anchor):cache.verify(z[:d]*s.radii)
                evaluate(z)
                if offline_bank is not None and start_index==0:
                    # The selected starting point is recomputed natively before
                    # optimization; cached curves never become final predictions.
                    matches=np.flatnonzero(np.all(offline_bank.points==z[:d],axis=1))
                    if len(matches):
                        independent=obj.native(z[:d]);stored=offline_bank.curves[matches[0]]
                        if np.linalg.norm(independent-stored)>1e-8*max(np.linalg.norm(independent),1e-30):
                            raise ValueError('Offline starting curve disagrees with the current native forward calculation')
                damping=1e-3;accepted=[];reason='iteration_budget';last_score=None
                if recycle_cells:obj.invalidate_cell_approximation()
                active=dict(start=start_index,initial_parameters=z.tolist(),reason='running',iterations=0,information_score=None,accepted_costs=accepted,proposal_secants=[],stationarity_polls=[])
                trace.append(active);selection_begin=time.monotonic()
                active['proposal_selection_state']='running'
                try:
                    if precomputed_proposal_steps:
                        obj.cell_steps=offline_bank.proposal_cell_steps.copy()
                        secants=dict(source='prepared_parent_geometry',steps=obj.cell_steps.tolist(),derivative_qualified=False)
                    else:secants=obj.select_proposal_steps(z) if adaptive_cells else []
                    active['proposal_secants']=secants
                    active['proposal_selection_state']='prepared' if precomputed_proposal_steps else 'completed'
                except TimeoutError:
                    active['proposal_selection_state']='budget_expired'
                    raise
                finally:
                    elapsed=time.monotonic()-selection_begin
                    proposal_selection_seconds+=elapsed
                    active['proposal_selection_seconds']=elapsed
                def rescue(current,current_cost,jacobian,trigger):
                    if len(active['stationarity_polls'])>=stationarity_polls:return current,False
                    from .native_stationarity_poll import stationarity_poll
                    candidate,detail=stationarity_poll(current,current_cost,jacobian,s.mode_dimensions,s.cell_dimensions,obj.cell_steps,
                        lambda point:evaluate(point)[0],lambda point:np.all(point>=lo) and np.all(point<=hi) and obj.allowed(point[:d]))
                    detail['trigger']=trigger;active['stationarity_polls'].append(detail)
                    if detail['improved']:accepted.append(detail['best_cost']);obj.last=None
                    return candidate,detail['improved']
                for iteration in range(maxiter):
                    active['iterations']=iteration+1
                    state=obj.calculate(z,True);residual=state['residual'];jac=state['jac'];cost=float(residual@residual)
                    if grouping is not None:
                        residual,jac=grouping.evaluate(state['rate'],jac/state['factor'][:,None]);cost=float(residual@residual)
                    linear_begin=time.monotonic()
                    step,step_info=(bounded_damped_step(jac,residual,damping,lo-z,hi-z) if box_steps else damped_step(jac,residual,damping))
                    bounded_subproblems+=int(step_info.get('bounded_subproblem',False))
                    bounded_subproblem_failures+=int(step_info.get('bounded_success',True) is False)
                    linear_algebra_seconds+=time.monotonic()-linear_begin;last_score=step_info['information_score']
                    active['information_score']=last_score
                    if last_score<1e-3:
                        if recycle_cells and obj.last_reused:obj.invalidate_cell_approximation();continue
                        z,improved=rescue(z,cost,jac,'small_local_information_score')
                        if improved:damping=1e-3;continue
                        reason='small_local_information_score';break
                    norm=max(np.max(abs(step)),1e-30)
                    if not box_steps and norm>.5:step*=.5/norm
                    step=np.clip(step,lo-z,hi-z);gradient=jac.T@residual
                    if box_steps and np.max(abs(step))<1e-9:reason='bounded_local_step_small';break
                    descent=float(gradient@step)
                    if descent>=0:damping*=10.;continue
                    found=False
                    for reduction in range(16):
                        alpha=.5**reduction;proposal=np.clip(z+alpha*step,lo,hi)
                        if not obj.allowed(proposal[:d]):rejected+=1;continue
                        value,_=evaluate(proposal)
                        if value<=cost+2e-4*alpha*descent:
                            accepted.append(value);z=proposal;damping=max(damping*.3,1e-10);found=True;break
                    if not found:
                        if recycle_cells:obj.invalidate_cell_approximation()
                        damping*=10.
                        if damping>1e10:
                            z,improved=rescue(z,cost,jac,'native_descent_stalled')
                            if improved:damping=1e-3;continue
                            reason='native_descent_stalled';break
                    elif np.max(abs(alpha*step))<1e-7 and not mode_substeps:
                        if recycle_cells and obj.last_reused:obj.invalidate_cell_approximation();continue
                        z,improved=rescue(z,value,jac,'small_accepted_step')
                        if improved:damping=1e-3;continue
                        reason='small_accepted_step';break
                    if found and mode_substeps and s.mode_dimensions:
                        from .feasible_mode_block import improve_modes
                        z,block=improve_modes(s,table,observation,z,mode_evaluator,obj.allowed,evaluate,deadline=obj.deadline,iterations=mode_substeps)
                        block.update(start=start_index,joint_iteration=iteration+1);mode_blocks.append(block)
                        accepted.extend(block['accepted_costs']);rejected+=block['rejected_geometries'];obj.last=None
                        if block['reason']=='time_budget':raise TimeoutError('Mode-block budget expired')
                active.update(reason=reason,final_parameters=z.tolist());active=None
            except FeasibleDirectionUnavailable as exc:
                if active is None:
                    trace.append(dict(start=start_index,initial_parameters=z.tolist(),reason='proposal_unavailable',iterations=0,error=str(exc)))
                else:active.update(reason='proposal_unavailable',final_parameters=z.tolist(),error=str(exc))
                active=None
            except TimeoutError:
                if not balanced_restarts:raise
                if active is None:trace.append(dict(start=start_index,initial_parameters=z.tolist(),reason='per_start_time_budget',stage='before_iterations',iterations=0))
                else:active.update(reason='per_start_time_budget',final_parameters=z.tolist())
                active=None
            finally:
                obj.deadline=begin+seconds
    except TimeoutError:
        if active is None:trace.append(dict(reason='time_budget',stage='screening_or_between_starts'))
        else:active.update(reason='time_budget',final_parameters=z.tolist())
    if screening_seconds is None:screening_seconds=time.monotonic()-begin
    if best is None:raise ValueError('No feasible native evaluation completed')
    coarse_endpoint=None if grouping is None else dict(parameters=best['z'].tolist(),deviance=best['cost'])
    best=best_full
    z=best['z'];s.check(z[:d]*s.radii)
    rwp=observation_rwp_percent(observation,best['rate'])
    return dict(best=dict(parameters=z.tolist(),deviance=best['cost']),exact_native_deviance=best['cost'],rwp_percent=rwp,
        requested_budget=dict(seconds=seconds,starts=starts,iterations=maxiter),stationarity_poll_limit=stationarity_polls,screening_seed=screening_seed,
        method='feasible_native_damped_Gauss_Newton',adaptive_cell_proposals=adaptive_cells,
        native_profile_pool=None if native_pool is None else native_pool.diagnostics(),
        multiscale_start_bank=multiscale,screening_cache=None if cache is None else cache.diagnostics(),
        proposal_derivatives_qualified=False,seconds=time.monotonic()-begin,native_calls=obj.calls,trials=trials,
        linear_algebra_seconds=linear_algebra_seconds,linear_algebra='one SVD reused for rank, information score and damped step',
        screening_seconds=screening_seconds,proposal_selection_seconds=proposal_selection_seconds,
        cached_screening_calls=cached_screening_calls,direct_screening_calls=direct_screening_calls,
        cell_derivative_recycling=None if not recycle_cells else obj.diagnostics(),
        retained_start_count=len(warm_ranked),independent_restart_reservation=warm_pool is not None,
        previous_starts_excluded=excluded_prior_starts,screening_exclusion_count=len(exclusions),
        offline_start_bank=None if offline_bank is None else offline_bank.diagnostics(),
        offline_screening_calls=0 if offline_bank is None else len(points),
        precomputed_proposal_steps=precomputed_proposal_steps,balanced_restart_budget=balanced_restarts,box_constrained_steps=box_steps,bounded_subproblems=bounded_subproblems,bounded_subproblem_failures=bounded_subproblem_failures,
        aggregation_width=aggregation_width,coarse_optimization_endpoint=coarse_endpoint,
        fixed_cell_mode_substeps=mode_substeps,mode_blocks=mode_blocks,mode_response_cache=None if mode_evaluator is None else mode_evaluator.diagnostics(),
        rejected_geometries=rejected,attempts=trace,bounded_optimum=bool(np.any(abs(z[:d])>.995)),
        global_optimum_certified=False),best['rate']
