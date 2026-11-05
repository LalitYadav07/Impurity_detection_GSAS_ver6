"""Experimental native profile fitting with explicit chemistry tangents."""
import time
import numpy as np
from .native_cell_derivative import NativeCellDerivative
from .profiled_nuisance import ConditionalNuisance
from .multiaxis_fit import background_components,background_reference
from .reference_secants import feasible_secant,FeasibleDirectionUnavailable
from .feasible_gauss_newton import observation_rwp_percent
from .chemistry_sqp import contacts,step

def fit(source,table,observation,bank,point,*,seconds=45,force_native_mode_derivatives=False,evaluator=None):
    s=source;m=s.mode_dimensions;d=len(s.radii);z=np.asarray(point,float);begin=time.monotonic();deadline=begin+seconds
    if z.shape!=(d+2,) or not np.isfinite(z).all() or not np.isfinite(seconds) or seconds<=0:raise ValueError('Complete finite start and budget required')
    cache=NativeCellDerivative(s,max_cells=1,force_native_derivatives=force_native_mode_derivatives) if evaluator is None else evaluator
    fixed,response=background_components(table);bg=background_reference(table);mask=observation.mask;trace=[];iterations=[];best=None;restores=[];reason='budget';damping=.001
    def checktime():
        if time.monotonic()>=deadline:raise TimeoutError('Native SQP allocation exhausted')
    def allowed(p):return np.all(abs(p[:d])<=1) and np.isfinite(s.chemistry.log_kernel((p[:d]*s.radii)[None,:])[0])
    def conditional(raw,p):
        c=ConditionalNuisance(observation,raw-table.renderer.background,fixed,bg*response);return c,c.solve(np.exp(p[-2:]))
    def score(p,remember=True):
        nonlocal best
        if not allowed(p):raise ValueError('Infeasible native SQP trial')
        raw=s.native(p[:d]*s.radii);c,values=conditional(raw,p);cost,rate,r,factor=values
        q=p.copy();q[-2:]=np.clip(np.log(c.coefficients),[-.3,np.log(.25)],[.3,np.log(4.)])
        if remember and (best is None or cost<best['cost']):
            best=dict(point=q,cost=cost,rate=rate);trace.append(dict(seconds=time.monotonic()-begin,deviance=cost,parameters=q.tolist()))
        return cost,rate,r,factor,q
    current=score(z);z=current[-1]
    def jacobian(p):
        checktime();raw,j=cache.mode_jacobian(p[:d]*s.radii);c,values=conditional(raw,p);_,rate,r,factor=values
        exact=score(p)
        if np.linalg.norm((rate-exact[1])[mask])/max(np.linalg.norm(exact[1][mask]),1e-30)>1e-8:raise ValueError('SQP native cache mismatch')
        derivatives=list(c.rate_jacobian(j*s.radii[:m,None]))
        def evaluate(physical):
            checktime();q=p.copy();q[:d]=physical;return score(q,remember=False)[1]
        for i in range(m,d):
            checktime();column,_=feasible_secant(evaluate,p[:d],i,min(float(bank.proposal_cell_steps[i]),.002),
                allowed=lambda q:allowed(np.r_[q,p[-2:]]))
            derivatives.append(column)
        return (np.array(derivatives)[:,mask]*factor).T,r
    try:
        for iteration in range(1000):
            checktime()
            try:j,r=jacobian(z)
            except FeasibleDirectionUnavailable:
                restored=False
                for fraction in [1e-6,1e-5,1e-4,.001,.01]:
                    trial=z.copy();trial[:d]*=1-fraction
                    if not allowed(trial):continue
                    try:j,r=jacobian(trial)
                    except FeasibleDirectionUnavailable:continue
                    z=trial;current=score(z);restores.append(dict(fraction=fraction,cost=current[0]));restored=True;break
                if not restored:reason='no_feasible_stencil';break
            values,a=contacts(s.chemistry,z[:d],s.radii)
            move,diagnostic=step(j,r,z[:d],values,a,damping=damping)
            row=dict(iteration=iteration,initial_deviance=current[0],constraint_minimum=float(np.min(values)) if len(values) else None,
                step=diagnostic,accepted=False);iterations.append(row)
            if np.linalg.norm(move)<1e-9:reason='constrained_step_small';break
            slope=float((j.T@r)@move);accepted=False
            for power in range(20):
                checktime();trial=z.copy();trial[:d]+=(.5**power)*move
                if not allowed(trial):continue
                proposed=score(trial)
                if proposed[0]<=current[0]+2e-4*(.5**power)*slope:
                    z=proposed[-1];current=proposed;accepted=True;damping=max(damping*.3,1e-10)
                    row.update(accepted=True,alpha=.5**power,deviance=current[0]);break
            if not accepted:damping*=10
            if damping>1e7:reason='native_descent_stalled';break
    except TimeoutError:reason='allocation_exhausted_incumbent_retained'
    result=dict(best=dict(parameters=best['point'].tolist(),deviance=best['cost']),
        rwp_percent=observation_rwp_percent(observation,best['rate']),trace=trace,iterations=iterations,
        interior_restorations=restores,seconds=time.monotonic()-begin,state=reason,all_allowed_modes_retained=True,
        global_optimum_certified=False,model_exclusion_authorized=False,method='native_chemistry_constrained_gn')
    return result,best['rate']
