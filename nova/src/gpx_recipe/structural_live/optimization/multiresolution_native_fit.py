"""Count-aggregation continuation with unchanged native GSAS-II profiles.

Each coarse objective is the exact Poisson marginal for sums of acquired bins.
It is only an optimization aid. Final scores always use all original bins.
No instrumental parameter or calculated peak shape is broadened or replaced.
"""
import time
import numpy as np
from gpx_recipe.structural_live.observations import count_rwp_percent
from scipy.sparse import csr_matrix
from scipy.optimize import least_squares
from scipy.stats import qmc
from gpx_recipe.structural_live.observations import CumulativeCounts
from .fit_fast_native import residual_and_factor
from .multiaxis_fit import NativeObjective,background_reference,background_components

class CountAggregation:
    def __init__(self,observation,width):
        if not isinstance(observation,CumulativeCounts):raise ValueError('Independent integer Poisson counts are required')
        if type(width)is not int or width<1:raise ValueError('Positive integer aggregation width required')
        self.original=observation;indices=np.flatnonzero(observation.mask);groups=[];current=[]
        for index in indices:
            if current and (index!=current[-1]+1 or len(current)==width):groups.append(current);current=[]
            current.append(int(index))
        if current:groups.append(current)
        group_index=np.concatenate([np.full(len(g),i,int) for i,g in enumerate(groups)])
        self.matrix=csr_matrix((np.ones(len(indices)),(group_index,np.arange(len(indices)))),shape=(len(groups),len(indices)))
        self.rate_scale=observation.count_rate_scale[indices]
        counts=np.array(self.matrix@observation.counts[indices],dtype=np.int64)
        x=np.array([np.mean(observation.x[g]) for g in groups])
        self.observation=CumulativeCounts.from_counts(observation.acquisition,observation.protocol_sha256+'-aggregate-'+str(width),
            observation.revision,observation.exposure,x,counts,1.,np.ones(len(groups),bool))

    def evaluate(self,rate,rate_jacobian=None):
        prediction=self.matrix@(np.asarray(rate)[self.original.mask]*self.rate_scale)
        residual,factor=residual_and_factor(self.observation,prediction)
        jac=None if rate_jacobian is None else np.asarray(self.matrix@(rate_jacobian*self.rate_scale[:,None]))*factor[:,None]
        return residual,jac

class AggregatedNativeObjective:
    def __init__(self,source,table,observation,width):
        self.native=NativeObjective(source,table,observation);self.grouping=CountAggregation(observation,width)
    def calculate(self,z,jacobian=False):
        a=self.native.calculate(z,jacobian)
        rate_jac=a['jac']/a['factor'][:,None] if jacobian else None
        residual,jac=self.grouping.evaluate(a['rate'],rate_jac)
        return dict(rate=a['rate'],residual=residual,jac=jac)

def fit_multiresolution(source,table,observation,*,warm=None,widths=(16,4,1),starts=4,max_nfev=35):
    if not source.chemistry.certified_full_support:
        from .constrained_native_fit import fit_feasible
        result,rate=fit_feasible(source,table,observation,warm=warm)
        result['requested_method_fallback']='Count continuation currently requires a chemically certified full box'
        return result,rate
    begin=time.perf_counter();s=source;d=len(s.radii);bg=background_reference(table);fixed,response=background_components(table)
    lower=np.r_[-np.ones(d),-.3,np.log(.25)];upper=np.r_[np.ones(d),.3,np.log(4.)]
    points=[np.zeros(d),*(qmc.Sobol(d,scramble=True,seed=28471).random_base2(5)*1.6-.8)]
    guesses=[];calls=0;mask=observation.mask;weights=1/np.sqrt(observation.variance[mask])
    for point in points:
        signal=s.native(point*s.radii)-table.renderer.background;calls+=1
        design=np.column_stack([signal[mask],response[mask]])
        a,b=np.linalg.lstsq(design*weights[:,None],(observation.y[mask]-fixed[mask])*weights,rcond=None)[0]
        guesses.append(np.clip(np.r_[point,np.log(np.clip(a,np.exp(-.3),np.exp(.3))),np.log(np.clip(b/bg,.25,4.))],lower,upper))
    if warm is not None:guesses.insert(0,np.clip(warm,lower,upper))
    records=[];best=None
    for width in widths:
        objective=AggregatedNativeObjective(s,table,observation,width)
        ranked=sorted([(float(objective.calculate(z)['residual']@objective.calculate(z)['residual']),z) for z in guesses],key=lambda v:v[0])
        selected=[]
        for _,z in ranked:
            if all(np.linalg.norm(z[:d]-other[:d])>.15 for other in selected):selected.append(z)
            if len(selected)>=starts:break
        results=[]
        for z in selected:
            result=least_squares(lambda v:objective.calculate(v)['residual'],z,jac=lambda v:objective.calculate(v,True)['jac'],
                bounds=(lower,upper),x_scale='jac',max_nfev=max_nfev,ftol=1e-8,xtol=1e-7,gtol=1e-6)
            results.append((float(2*result.cost),result.x.copy()))
            records.append(dict(bin_width=width,cost=float(2*result.cost),success=bool(result.success),nfev=int(result.nfev)))
        guesses=[z for _,z in sorted(results,key=lambda v:v[0])];calls+=objective.native.calls;best=guesses[0]
    if widths[-1]!=1:raise ValueError('Final optimization must use original native bins')
    objective=NativeObjective(s,table,observation);final=objective.calculate(best);calls+=objective.calls
    rwp=count_rwp_percent(observation,final['rate'])
    return dict(best=dict(parameters=best.tolist(),deviance=float(final['residual']@final['residual'])),
        exact_native_deviance=float(final['residual']@final['residual']),rwp_percent=rwp,
        seconds=time.perf_counter()-begin,native_calls=calls,method='native_Poisson_count_aggregation_continuation',
        attempts=records,widths=list(widths),bounded_optimum=bool(np.any(abs(best[:d])>.995)),
        global_optimum_certified=False),final['rate']
