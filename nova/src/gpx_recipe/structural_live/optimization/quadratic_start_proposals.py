"""Candidate-local quadratic lifting for optimizer starts.

Inspired by matrix lifting in phase retrieval (arXiv:1109.4499). Its recovery
theorems do not apply to structured powder measurements or a Taylor model.
All coefficients come from native GSAS-II evaluations at chemically checked
geometries. The polynomial and relaxed matrices never supply final likelihoods,
uncertainty or exclusion decisions. Only feasible native-checked starts survive.
"""
import time
import numpy as np
from scipy.linalg import eigh
from .native_cell_derivative import NativeCellDerivative
from .feasible_mode_block import improve_modes
from .fit_fast_native import residual_and_factor
from .multiaxis_fit import background_reference,background_components

class ProposalUnavailable(ValueError):
    """No useful quadratic starting proposal; the native model remains valid."""


def svec(matrix):
    n=len(matrix);i,j=np.triu_indices(n);return matrix[i,j]*np.where(i==j,1.,np.sqrt(2.))

def smat(vector,n):
    i,j=np.triu_indices(n);out=np.zeros((n,n));out[i,j]=np.asarray(vector)/np.where(i==j,1.,np.sqrt(2.));out[j,i]=out[i,j];return out

def project_matrix(matrix,tolerance=1e-9,iterations=100):
    """Dykstra projection onto PSD matrices with X00=1 and 0<=diag(X)<=1."""
    x=(matrix+matrix.T)/2;p=np.zeros_like(x);q=np.zeros_like(x)
    for step in range(iterations):
        old=x.copy();w,v=eigh(x+p,check_finite=False);y=(v*np.maximum(w,0))@v.T;p=x+p-y
        z=y+q;x=z.copy();diag=np.clip(np.diag(x),0,1);diag[0]=1;np.fill_diagonal(x,diag);q=z-x
        if np.linalg.norm(x-old)<tolerance*max(np.linalg.norm(x),1) and np.linalg.eigvalsh(x)[0]>-tolerance:break
    return x

def relax(design,target,m,*,seconds=5.,trace_penalty=0.):
    """Projected FISTA; report numerical feasibility, never certify an optimum."""
    if not np.isfinite(seconds) or seconds<=0:raise TimeoutError('No relaxation budget remains')
    begin=time.monotonic();gram=design.T@design;rhs=design.T@target
    lipschitz=float(eigh(gram,subset_by_index=[len(gram)-1,len(gram)-1],eigvals_only=True,check_finite=False)[0])
    if lipschitz<=0:raise ProposalUnavailable('Quadratic response has no observed variation')
    x=np.eye(m+1)*.05;x[0,0]=1;y=svec(x);v=y.copy();momentum=1.;trace=np.r_[0.,np.ones(m)]
    penalty=svec(np.diag(trace))*trace_penalty;objective=[]
    for iteration in range(1500):
        previous=v.copy();candidate=y-(gram@y-rhs+penalty)/lipschitz
        x=project_matrix(smat(candidate,m+1));v=svec(x)
        new_momentum=(1+np.sqrt(1+4*momentum**2))/2
        y=v+(momentum-1)/new_momentum*(v-previous);momentum=new_momentum
        if iteration%20==0:objective.append(float(.5*v@gram@v-rhs@v+penalty@v))
        if time.monotonic()-begin>seconds:break
        if iteration>30 and np.linalg.norm(v-previous)<1e-8*max(np.linalg.norm(v),1):break
    return x,dict(seconds=time.monotonic()-begin,iterations=iteration+1,minimum_eigenvalue=float(np.linalg.eigvalsh(x)[0]),
        diagonal_minimum=float(np.diag(x).min()),diagonal_maximum=float(np.diag(x).max()),objective_trace=objective,global_optimum_certified=False)

def rank_completed(proposals,native_cost):
    """A deadline never discards native evaluations already completed."""
    ranked=[];exhausted=False
    for z in proposals:
        try:cost=native_cost(z)[0]
        except TimeoutError:exhausted=True;break
        ranked.append((cost,z))
    ranked.sort(key=lambda row:row[0])
    return ranked,exhausted

def explore(source,table,observation,initial,*,seed=483721,step=.05,seconds=120.,draw_count=180,polish_starts=8):
    if not np.isfinite([seconds,step]).all() or seconds<=0 or not 0<step<=.5:raise ValueError('Positive bounded step and time budget required')
    begin=time.monotonic();deadline=begin+seconds;s=source;m=s.mode_dimensions;d=len(s.radii);n=m+1
    if type(draw_count)is not int or not 1<=draw_count<=2000 or type(polish_starts)is not int or not 1<=polish_starts<=32:raise ValueError('Bounded proposal and polish counts required')
    if not 1<=m<=32:raise ValueError('This experimental initializer is explicitly bounded to 1–32 modes; no modes are truncated')
    z0=np.asarray(initial,float)
    if z0.shape!=(d+2,) or not np.isfinite(z0).all():raise ValueError('Complete finite starting point required')
    cell=z0[m:d].copy();evaluator=NativeCellDerivative(s,max_cells=1)
    fixed,response=background_components(table);back=background_reference(table);mask=observation.mask
    def allowed(point):return np.all(abs(point)<=1) and bool(np.isfinite(s.chemistry.log_kernel((point*s.radii)[None,:])[0]))
    def profile(q):
        if time.monotonic()-begin>seconds:raise TimeoutError('Quadratic proposal construction exceeded its declared budget')
        return evaluator.evaluate(np.r_[q,cell]*s.radii)-table.renderer.background
    def native_cost(z,cached=True):
        if time.monotonic()>=deadline:raise TimeoutError('Native proposal evaluation budget exhausted')
        signal=profile(z[:m]) if cached else s.native(z[:d]*s.radii)-table.renderer.background
        rate=np.exp(z[-2])*signal+fixed+back*np.exp(z[-1])*response
        r,_=residual_and_factor(observation,rate);return float(r@r),rate
    zero=np.zeros(m);f0=profile(zero);indices=list(zip(*np.triu_indices(n)));position={pair:i for i,pair in enumerate(indices)}
    coefficients=np.zeros((len(indices),len(f0)));diagonal=[];linear=[];calls=1
    for i in range(m):
        direction=np.eye(m)[i]*step;plus=profile(direction);minus=profile(-direction);calls+=2
        coefficients[position[(0,i+1)]]=(plus-minus)/(2*step*np.sqrt(2))
        coefficients[position[(i+1,i+1)]]=(plus+minus-2*f0)/(2*step**2)
    for i in range(m):
        for j in range(i+1,m):
            ei=np.eye(m)[i]*step;ej=np.eye(m)[j]*step
            cross=(profile(ei+ej)-profile(ei-ej)-profile(-ei+ej)+profile(-ei-ej))/(4*step**2);calls+=4
            coefficients[position[(i+1,j+1)]]=cross/np.sqrt(2)
    construction_seconds=time.monotonic()-begin;rng=np.random.default_rng(seed);validation=[]
    for _ in range(12):
        q=rng.uniform(-.5,.5,m)
        if not allowed(np.r_[q,cell]):continue
        exact=profile(q);approx=f0+svec(np.outer(np.r_[1,q],np.r_[1,q]))@coefficients
        validation.append(dict(relative_pattern_error=float(np.linalg.norm(exact-approx)/np.linalg.norm(exact)),
            relative_distortion_error=float(np.linalg.norm(exact-approx)/max(np.linalg.norm(exact-f0),1e-30))))
    scale=np.exp(z0[-2]);target=(observation.y-fixed-back*np.exp(z0[-1])*response-scale*f0)[mask]/np.sqrt(observation.variance[mask])
    design=(scale*coefficients[:,mask]/np.sqrt(observation.variance[mask])).T
    solutions=[];proposals=[z0.copy()];ranked=[];budget_exhausted=False;scored=0
    for penalty_fraction in [0.,1e-5,1e-3]:
        remaining=deadline-time.monotonic()
        if remaining<=.01:break
        penalty=penalty_fraction*np.max(abs(design.T@target))
        matrix,receipt=relax(design,target,m,seconds=min(4.,remaining/2),trace_penalty=penalty);receipt['trace_penalty_fraction']=penalty_fraction;solutions.append(receipt)
        mean=matrix[0,1:];cov=(matrix[1:,1:]-np.outer(mean,mean));w,v=eigh((cov+cov.T)/2,check_finite=False);root=v*np.sqrt(np.maximum(w,0))
        draws=[mean]
        for i in range(max(0,m-6),m):draws.extend([mean+root[:,i],mean-root[:,i]])
        draws.extend(mean+root@rng.normal(size=m) for _ in range(draw_count))
        for q in draws:
            z=z0.copy();z[:m]=q
            if allowed(z[:d]):proposals.append(z)
        completed,expired=rank_completed(proposals[scored:],native_cost)
        ranked.extend(completed);scored=len(proposals)
        if expired:budget_exhausted=True;break
    ranked.sort(key=lambda row:row[0])
    chosen=[]
    for cost,z in ranked:
        if all(np.linalg.norm(z-q)>.15 for _,q in chosen):chosen.append((cost,z))
        if len(chosen)>=polish_starts:break
    endpoints=[];records=[]
    for cost,z in chosen:
        if time.monotonic()-begin>seconds-1:break
        try:
            endpoint,detail=improve_modes(s,table,observation,z,evaluator,allowed,native_cost,
                deadline=min(begin+seconds,time.monotonic()+5),iterations=80)
            exact_cost,_=native_cost(endpoint,False);error=evaluator.verify(endpoint[:d]*s.radii)
        except TimeoutError:
            budget_exhausted=True;break
        records.append(dict(initial_cost=cost,native_deviance=exact_cost,relative_native_check=error,detail=detail))
        endpoints.append((exact_cost,endpoint))
    endpoints.extend(chosen);endpoints.sort(key=lambda x:x[0])
    return [z for _,z in endpoints],dict(seconds=time.monotonic()-begin,construction_seconds=construction_seconds,native_construction_calls=calls,
        budget_exhausted=budget_exhausted or time.monotonic()-begin>=seconds,completed_native_proposals=len(ranked),finite_difference_step=step,draw_count_per_relaxation=draw_count,polish_starts=polish_starts,validation=validation,relaxations=solutions,feasible_proposals=len(proposals),polished=records,
        scope='Experimental starting points only. Taylor accuracy is not a likelihood certificate; all final ranking/refinement remains native.',
        literature='https://arxiv.org/abs/1109.4499',recovery_theorem_applies=False,full_cell_refinement_required=True)
