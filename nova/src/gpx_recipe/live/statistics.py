"""Conditional reduced-Gaussian posterior; experimental, not a rejection policy.

Each cumulative snapshot is fitted as a new likelihood, never multiplied by the
previous cumulative likelihood. Independent per-candidate kernels only. The
supplied covariance is diagonal; shared nuclear uncertainty is NOT inferred by
this module. Gates must remain closed until that observation model is qualified.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.special import logsumexp

@dataclass
class QuadraticObservation:
    base: np.ndarray
    quadratic: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    forms: np.ndarray
    lower: float = 0.
    upper: float = 9.

    def __post_init__(self):
        self.base=np.asarray(self.base,float);self.quadratic=np.asarray(self.quadratic,float)
        self.y=np.asarray(self.y,float);self.sigma=np.asarray(self.sigma,float);self.forms=np.asarray(self.forms,float)
        if self.quadratic.ndim!=3:raise ValueError('A rank-three quadratic profile is required')
        self.dimension=len(self.quadratic)
        if not 1<=self.dimension<=64 or not len(self.y) or self.quadratic.size>40_000_000:raise ValueError('Model exceeds the validated shape/resource budget')
        if self.base.ndim!=1 or self.y.shape!=self.base.shape or self.sigma.shape!=self.y.shape or self.quadratic.shape!=(self.dimension,self.dimension,len(self.y)):
            raise ValueError('Matching reduced-observation arrays required')
        if not all(np.isfinite(v).all() for v in [self.base,self.quadratic,self.y,self.sigma,self.forms]) or np.any(self.sigma<=0) or not 0<=self.lower<self.upper:
            raise ValueError('Finite data and positive uncertainties/bounds required')
        if self.forms.ndim!=3 or self.forms.shape[1:]!=(self.dimension,self.dimension):raise ValueError('Native moment forms required')
        if not len(self.forms) or not np.allclose(self.forms,self.forms.transpose(0,2,1),rtol=1e-10,atol=1e-12):raise ValueError('Symmetric physical moment forms required')
        if np.any(np.linalg.eigvalsh(self.forms)<-1e-10):raise ValueError('Moment forms must be positive semidefinite')
        if np.linalg.eigvalsh(self.forms.sum(axis=0))[0]<=1e-12:raise ValueError('Unbounded or unidentifiable spin coordinates in moment constraints')
        if not np.allclose(self.quadratic,self.quadratic.transpose(1,0,2),rtol=1e-10,atol=1e-12):raise ValueError('Symmetric intensity quadratic required')
        self.i,self.j=np.triu_indices(self.dimension)
        self.design=self.quadratic[self.i,self.j]*np.where(self.i==self.j,1.,2.)[:,None]
        # QR compresses a whitened observation exactly, without forming a badly
        # conditioned squared normal matrix or discarding residual lack-of-fit.
        a=self.design.T/self.sigma[:,None];r=(self.y-self.base)/self.sigma
        u,self.R=np.linalg.qr(a,mode='reduced');self.z=u.T@r
        perpendicular=r-u@self.z
        self.offset=float(perpendicular@perpendicular)
        self.box=self.upper*np.sqrt(len(self.forms)*np.diag(np.linalg.inv(np.sum(self.forms,axis=0))))

    def features(self,p):
        p=np.atleast_2d(p);return p[:,self.i]*p[:,self.j]
    def predict(self,p):return self.base+self.features(p)@self.design
    def loglike(self,p):
        r=self.features(p)@self.R.T-self.z
        return -.5*(np.einsum('ij,ij->i',r,r)+self.offset)
    def moments(self,p):return np.sqrt(np.maximum(0.,np.einsum('ni,sij,nj->ns',np.atleast_2d(p),self.forms,np.atleast_2d(p))))
    def valid(self,p):
        p=np.atleast_2d(p);m=self.moments(p)
        return np.isfinite(p).all(axis=1)&(m>=self.lower-1e-10).all(axis=1)&(m<=self.upper+1e-10).all(axis=1)
    def prior(self,rng,n,deadline=None):
        # Factor exactly disjoint native moment constraints into independent
        # coordinate blocks. Sampling a single high-dimensional bounding box
        # wastes almost all proposals for many independent magnetic orbits.
        supports=[set(np.flatnonzero(np.any(f!=0,axis=0)|np.any(f!=0,axis=1))) for f in self.forms]
        groups=[]
        for support in supports:
            if not support:
                if self.lower>0:raise ValueError('A selected orbit has identically zero moment')
                continue
            touching=[g for g in groups if g & support]
            for g in touching:support=support|g;groups.remove(g)
            groups.append(support)
        output=np.zeros((n,self.dimension));rng_deadline=deadline or time.monotonic()+30
        for group in groups:
            cols=sorted(group);forms=np.array([f[np.ix_(cols,cols)] for f,s in zip(self.forms,supports) if s and s<=group])
            bound=self.upper*np.sqrt(len(forms)*np.diag(np.linalg.inv(forms.sum(axis=0))))
            count=0;accepted=[]
            while count<n:
                if time.monotonic()>rng_deadline:raise TimeoutError('Prior sampling budget exceeded; candidate remains unresolved')
                p=rng.uniform(-bound,bound,(max(512,n-count),len(cols)))
                mag=np.sqrt(np.maximum(0.,np.einsum('ni,sij,nj->ns',p,forms,p)))
                good=p[((mag>=self.lower)&(mag<=self.upper)).all(axis=1)]
                accepted.append(good);count+=len(good)
            output[:,cols]=np.concatenate(accepted)[:n]
        if not self.valid(output).all():raise ValueError('Factored prior failed full native moment constraints')
        return output

def sample_posterior(model,*,particles=256,seed=0,moves=12,max_seconds=60.):
    if not isinstance(particles,int) or isinstance(particles,bool) or not isinstance(moves,int) or particles<64 or moves<2 or not np.isfinite(max_seconds) or max_seconds<=0:raise ValueError('Invalid sampler budget')
    start=time.monotonic();rng=np.random.default_rng(seed);x=model.prior(rng,particles,deadline=start+max_seconds);ll=model.loglike(x)
    if not np.isfinite(ll).all():raise ValueError('Nonfinite prior likelihood; inference remains unresolved')
    prior=x.copy();beta=0.;history=[];logz=0.
    while beta<1.-1e-10:
        if time.monotonic()-start>max_seconds or len(history)>600:raise TimeoutError('Posterior incomplete; no rejection or precision claim')
        lo=0.;hi=1.-beta
        def weight(step):
            z=step*(ll-ll.max());z-=logsumexp(z);w=np.exp(z);return w,1./(w@w)
        if weight(hi)[1]<.8*particles:
            for _ in range(40):
                mid=(lo+hi)/2
                if weight(mid)[1]<.8*particles:hi=mid
                else:lo=mid
            step=lo
        else:step=hi
        if step<1e-14:raise RuntimeError('Tempering collapsed')
        w,ess=weight(step);logz+=float(logsumexp(step*ll)-np.log(particles));beta=min(1.,beta+step)
        cdf=np.cumsum(w);cdf[-1]=1.
        indices=np.searchsorted(cdf,(rng.random()+np.arange(particles))/particles)
        x=x[indices].copy();ll=ll[indices].copy()
        # Frozen within-sign proposal covariance avoids the global +/- mode
        # inflating step size. Symmetric mixture; no adaptation during a move.
        folded=x*np.where(x[:,:1]<0,-1,1)
        cov=np.atleast_2d(np.cov(folded,rowvar=False))+np.diag(model.box**2)*1e-8
        chol=np.linalg.cholesky(cov)*(2.38/np.sqrt(model.dimension));accepted=0;local_accepted=0;local_proposals=0;sign_accepted=0;sign_proposals=0
        for _ in range(moves):
            if time.monotonic()-start>max_seconds:raise TimeoutError('Posterior move budget exceeded; candidate remains unresolved')
            scale=rng.choice([.1,.35,1.],size=particles)
            trial=x+(rng.normal(size=x.shape)@chol.T)*scale[:,None]
            # Global sign is an exact symmetry of the declared quadratic model.
            flips=rng.random(particles)<.05;trial[flips]=-x[flips]
            valid=model.valid(trial);tl=np.full(particles,-np.inf);tl[valid]=model.loglike(trial[valid])
            ok=np.log(rng.random(particles))<beta*(tl-ll);x[ok]=trial[ok];ll[ok]=tl[ok];accepted+=int(ok.sum())
            local_accepted+=int((ok&~flips).sum());local_proposals+=int((~flips).sum());sign_accepted+=int((ok&flips).sum());sign_proposals+=int(flips.sum())
        history.append(dict(beta=beta,ess=ess,acceptance=accepted/(particles*moves),local_acceptance=local_accepted/max(1,local_proposals),sign_acceptance=sign_accepted/max(1,sign_proposals),local_proposals=local_proposals,sign_proposals=sign_proposals))
    return dict(samples=x,prior=prior,loglike=ll,log_evidence=logz,history=history,seconds=time.monotonic()-start,qualified=False)

def compare_replicas(model,a,b):
    qa=np.quantile(model.moments(a['samples']),[.025,.5,.975],axis=0);qb=np.quantile(model.moments(b['samples']),[.025,.5,.975],axis=0)
    width=np.maximum(np.maximum(qa[2]-qa[0],qb[2]-qb[0]),.05)
    discrepancy=float(np.max(abs(qa[1]-qb[1])/width))
    return dict(median_discrepancy_in_interval_widths=discrepancy,screen_passed=discrepancy<.5,global_convergence_proven=False)
