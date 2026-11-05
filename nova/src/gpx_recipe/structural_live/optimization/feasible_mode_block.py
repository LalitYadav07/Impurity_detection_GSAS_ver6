"""Short fixed-cell mode steps; every acceptance uses the full native curve.

The caller subsequently updates all cell coordinates. No coordinate is frozen
out of the overall fit, and no response is shared between candidate families.
"""
import time
import numpy as np
from .damped_step import damped_step
from .fit_fast_native import residual_and_factor
from .multiaxis_fit import background_reference,background_components


def improve_modes(source,table,observation,initial,evaluator,allowed,native_evaluate,*,deadline,iterations=3):
    if type(iterations)is not int or iterations<1:raise ValueError('Positive mode-block iteration limit required')
    s=source;m=s.mode_dimensions;d=len(s.radii);z=np.asarray(initial,float).copy()
    if z.shape!=(d+2,) or not allowed(z[:d]):raise ValueError('Feasible full-dimensional starting point required')
    columns=np.r_[np.arange(m),d,d+1];lower=np.r_[-np.ones(d),-.3,np.log(.25)];upper=np.r_[np.ones(d),.3,np.log(4.)]
    if not np.isfinite(z).all() or np.any(z<lower) or np.any(z>upper):raise ValueError('Finite bounded mode-block start required')
    fixed,response=background_components(table);background=background_reference(table);mask=observation.mask
    begin=time.monotonic();record=dict(iterations=0,accepted_costs=[],rejected_geometries=0,reason='iteration_budget',maximum_profile_relative_error=0.,maximum_profile_sigma_error=0.)
    damping=1e-3
    try:
        if time.monotonic()>=deadline:raise TimeoutError('Mode-block budget expired')
        current,native_rate=native_evaluate(z)
        for iteration in range(iterations):
            if time.monotonic()>=deadline:raise TimeoutError('Mode-block budget expired')
            record['iterations']=iteration+1
            curve,derivative=evaluator.mode_jacobian(z[:d]*s.radii)
            signal=curve-table.renderer.background;scale=np.exp(z[-2]);back=background*np.exp(z[-1]);rate=scale*signal+fixed+back*response
            relative=float(np.linalg.norm((rate-native_rate)[mask])/max(np.linalg.norm(native_rate[mask]),1e-30))
            sigma=float(np.max(abs((rate-native_rate)[mask])/np.sqrt(observation.variance[mask])))
            record['maximum_profile_relative_error']=max(record['maximum_profile_relative_error'],relative)
            record['maximum_profile_sigma_error']=max(record['maximum_profile_sigma_error'],sigma)
            if relative>1e-8 or sigma>.01:raise ValueError('Fixed-cell native response differs from the full native profile')
            residual,factor=residual_and_factor(observation,rate)
            jac=np.vstack([scale*derivative*s.radii[:m,None],scale*signal,back*response])
            jac=(jac[:,mask]*factor).T
            step,info=damped_step(jac,residual,damping)
            if info['information_score']<1e-3:record['reason']='small_local_information_score';break
            norm=max(np.max(abs(step)),1e-30)
            if norm>.5:step*=.5/norm
            step=np.clip(step,lower[columns]-z[columns],upper[columns]-z[columns]);descent=float((jac.T@residual)@step)
            if descent>=0:damping*=10.;continue
            accepted=False
            for reduction in range(16):
                if time.monotonic()>=deadline:raise TimeoutError('Mode-block budget expired')
                alpha=.5**reduction;trial=z.copy();trial[columns]=np.clip(z[columns]+alpha*step,lower[columns],upper[columns])
                if not allowed(trial[:d]):record['rejected_geometries']+=1;continue
                cost,prediction=native_evaluate(trial)
                if cost<=current+2e-4*alpha*descent:
                    z=trial;current=cost;native_rate=prediction;record['accepted_costs'].append(cost)
                    damping=max(damping*.3,1e-10);accepted=True;break
            if not accepted:damping*=10.
            if damping>1e10:record['reason']='native_descent_stalled';break
            if accepted and np.max(abs(alpha*step))<1e-7:record['reason']='small_accepted_step';break
    except TimeoutError:record['reason']='time_budget'
    record.update(seconds=time.monotonic()-begin,final_parameters=z.tolist(),cell_coordinates_unchanged=bool(np.array_equal(z[m:d],np.asarray(initial)[m:d])))
    return z,record
