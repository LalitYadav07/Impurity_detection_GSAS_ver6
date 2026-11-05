"""Feasible native fitting when empirical contact limits cut the search box.

Rejected geometries are neither evaluated by GSAS-II nor treated as rejected
subgroups. All physical coordinates remain free within the original domain.
"""
import time
import numpy as np
from gpx_recipe.structural_live.observations import count_rwp_percent
from scipy.optimize import Bounds, NonlinearConstraint, minimize
from scipy.stats import qmc
from .fit_fast_native import residual_and_factor
from .multiaxis_fit import background_reference,background_components
from .reference_secants import feasible_secant
from .smooth_contact_constraints import SmoothContactConstraints


class FeasibleNativeObjective:
    def __init__(self,source,table,observation,deadline=float('inf'),native_pool=None):
        self.s=source;self.table=table;self.obs=observation;self.deadline=deadline
        self.d=len(source.radii);self.m=source.mode_dimensions;self.calls=0;self.last=None;self.best=None
        self.background=background_reference(table)
        self.fixed,self.response=background_components(table)
        self.cell_steps=np.full(self.d,.002)
        self.native_pool=native_pool
        if native_pool is not None:native_pool.validate_source(source)

    def allowed(self,z):
        p=np.asarray(z,float)
        return p.shape==(self.d,) and np.isfinite(p).all() and np.all(abs(p)<=1.) and bool(np.isfinite(self.s.chemistry.log_kernel((p*self.s.radii)[None,:])[0]))

    def native(self,z):
        if time.monotonic()>self.deadline:raise TimeoutError('Constrained fit time budget exhausted')
        self.calls+=1;return self.s.native(z*self.s.radii)

    def cell_columns(self,z):
        if self.m==self.d:
            return np.empty((0,len(self.s.fast.x)))
        if self.native_pool is None:
            return np.array([feasible_secant(self.native,z[:self.d],i,self.cell_steps[i],allowed=self.allowed)[0]
                             for i in range(self.m,self.d)])
        from .reference_secants import feasible_stencil
        points=[];stencils=[];indices={}
        for i in range(self.m,self.d):
            locations,detail=feasible_stencil(z[:self.d],i,self.cell_steps[i],allowed=self.allowed)
            rows=[]
            for location in locations:
                key=tuple(location)
                if key not in indices:indices[key]=len(points);points.append(location)
                rows.append(indices[key])
            stencils.append((rows,detail))
        values=self.native_pool.evaluate_many(points,deadline=self.deadline)
        columns=np.array([(values[rows[0]]-values[rows[1]])/((2 if detail['scheme']=='central' else 1)*detail['effective_step']) for rows,detail in stencils])
        if not np.isfinite(columns).all():raise ValueError('Nonfinite parallel native cell secant')
        return columns

    def calculate(self,z,jacobian=False):
        z=np.asarray(z,float)
        if time.monotonic()>self.deadline:raise TimeoutError('Constrained fit time budget exhausted')
        if self.last is None or not np.array_equal(z,self.last['z']):
            y,j=self.s.native_mode_jacobian(z[:self.d]*self.s.radii);self.calls+=1
            signal=y-self.table.renderer.background;scale=np.exp(z[-2]);back=self.background*np.exp(z[-1])
            rate=scale*signal+self.fixed+back*self.response;r,factor=residual_and_factor(self.obs,rate)
            self.last=dict(z=z.copy(),rate=rate,residual=r,factor=factor,signal=signal,mode_jac=j,jac=None)
            cost=float(r@r)
            if self.best is None or cost<self.best['cost']:self.best=dict(z=z.copy(),cost=cost,rate=rate.copy())
        a=self.last
        if jacobian and a['jac'] is None:
            j=np.zeros((self.d+2,len(a['rate'])))
            j[:self.m]=a['mode_jac']*self.s.radii[:self.m,None]*np.exp(z[-2])
            j[self.m:self.d]=self.cell_columns(z)*np.exp(z[-2])
            j[-2]=a['signal']*np.exp(z[-2]);j[-1]=self.background*np.exp(z[-1])*self.response
            a['jac']=(j[:,self.obs.mask]*a['factor']).T
        return a

    def select_proposal_steps(self,z):
        from .reference_secants import select_secant_steps
        # Compare directions in observed-noise units. This affects proposals
        # only: acceptance and the final score retain the exact native bins.
        weight=1/np.sqrt(self.obs.variance[self.obs.mask])
        function=lambda point:self.native(point)[self.obs.mask]*weight
        self.cell_steps,receipt=select_secant_steps(function,np.asarray(z)[:self.d],
            range(self.m,self.d),allowed=self.allowed,scales=(.0005,.002,.008,.032,.128))
        if self.last is not None:self.last['jac']=None
        return receipt


def fit_feasible(source,table,observation,*,warm=None,seconds=90.,starts=3,maxiter=80):
    begin=time.monotonic();s=source;d=len(s.radii);m=s.mode_dimensions
    objective=FeasibleNativeObjective(s,table,observation,deadline=begin+seconds)
    contacts=SmoothContactConstraints(s.chemistry)
    lower=np.r_[-np.ones(d),-.3,np.log(.25)];upper=np.r_[np.ones(d),.3,np.log(4.)]
    def values(z):return contacts.values(z[:d]*s.radii)
    def jac(z):return np.column_stack([contacts.jacobian(z[:d]*s.radii)*s.radii,np.zeros((contacts.count,2))])
    def hess(z,v):
        out=np.zeros((d+2,d+2));out[:d,:d]=contacts.hessian(z[:d]*s.radii,v)*s.radii[:,None]*s.radii[None,:]
        return out
    constraints=[] if not contacts.count else [NonlinearConstraint(values,0.,np.inf,jac=jac,hess=hess,keep_feasible=True)]
    grid=qmc.Sobol(d,scramble=True,seed=82519).random_base2(5)*1.6-.8
    seeds=[np.zeros(d),*grid]
    if warm is not None:seeds.insert(0,np.asarray(warm)[:d])
    ranked=[];rejected=0;mask=observation.mask;bg=background_reference(table)
    fixed,response=background_components(table)
    for point in seeds:
        if not objective.allowed(point):rejected+=1;continue
        # Native structural derivative evaluation is deferred until optimization.
        signal=objective.native(point)-table.renderer.background
        design=np.column_stack([signal[mask],response[mask]]);w=1/np.sqrt(observation.variance[mask])
        a,b=np.linalg.lstsq(design*w[:,None],(observation.y[mask]-fixed[mask])*w,rcond=None)[0]
        a=np.clip(a,np.exp(-.3),np.exp(.3));b=np.clip(b,bg*.25,bg*4)
        z=np.clip(np.r_[point,np.log(a),np.log(b/bg)],lower,upper);r,_=residual_and_factor(observation,a*signal+fixed+b*response)
        ranked.append((float(r@r),z))
    if not ranked:raise ValueError('No feasible fitting start; no subgroup conclusion is possible')
    ranked.sort(key=lambda x:x[0]);records=[]
    for _,initial in ranked[:starts]:
        try:
            def fun(z):
                a=objective.calculate(z,True);return .5*float(a['residual']@a['residual']),a['jac'].T@a['residual']
            def gn(z):
                j=objective.calculate(z,True)['jac'];return j.T@j
            result=minimize(fun,initial,jac=True,hess=gn,method='trust-constr',
                bounds=Bounds(lower,upper,keep_feasible=True),constraints=constraints,
                options=dict(maxiter=maxiter,gtol=1e-6,xtol=1e-8,barrier_tol=1e-8,initial_barrier_parameter=.001,verbose=0))
            records.append(dict(success=bool(result.success),iterations=int(result.nit),
                cost=float(2*result.fun),optimality=float(result.optimality),constraint_violation=float(result.constr_violation)))
        except TimeoutError:
            records.append(dict(success=False,budget_exhausted=True));break
        except ValueError as exc:
            # Keep the best verified feasible iterate; expose the failed attempt.
            records.append(dict(success=False,error=str(exc)))
    if objective.best is None:raise ValueError('No evaluated feasible optimizer iterate')
    best=objective.best;z=best['z'];s.check(z[:d]*s.radii)
    rwp=count_rwp_percent(observation,best['rate'])
    return dict(best=dict(parameters=z.tolist(),deviance=best['cost']),exact_native_deviance=best['cost'],
        seconds=time.monotonic()-begin,native_calls=objective.calls,rwp_percent=rwp,
        method='native_feasible_trust_region',attempts=records,screening_rejections=rejected,
        contact_constraints=contacts.count,converged_best=any(r.get('success') and abs(r.get('cost',float('inf'))-best['cost'])<1e-5 for r in records),
        bounded_optimum=bool(np.any(abs(z[:d])>.995)),global_optimum_certified=False,
        minimum_contact_margin=float(np.min(values(z))) if contacts.count else None),best['rate']
