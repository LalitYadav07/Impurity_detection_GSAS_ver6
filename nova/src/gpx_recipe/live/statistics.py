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
        if (not all(np.isfinite(v).all() for v in [self.base,self.quadratic,self.y,self.sigma,self.forms])
                or np.any(self.sigma<=0) or not np.isfinite([self.lower,self.upper]).all()
                or not 0<=self.lower<self.upper):
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
        if type(n) is not int or n<1:raise ValueError('A positive finite prior sample count is required')
        output=np.zeros((n,self.dimension));rng_deadline=time.monotonic()+30 if deadline is None else float(deadline)
        if not np.isfinite(rng_deadline):raise ValueError('A finite prior deadline is required')
        self.prior_diagnostics=[]
        for group in groups:
            cols=sorted(group);forms=np.array([f[np.ix_(cols,cols)] for f,s in zip(self.forms,supports) if s and s<=group])
            # Every feasible point satisfies
            #   K*lower² <= p.T sum(forms) p <= K*upper².
            # A uniform ellipsoidal-shell proposal contains the full target.
            # Its density is constant in these coordinates (the linear map
            # has a constant Jacobian); rejection with ALL original forms
            # therefore preserves the exact uniform physical constraint prior.
            # This avoids an exponentially wasteful axis-aligned box after
            # an equivalent dense change of independent magnetic coordinates.
            metric=forms.sum(axis=0)
            transform=np.linalg.solve(np.linalg.cholesky(metric).T,np.eye(len(cols)))
            # Enlarge the proposal if roundoff makes its computed metric
            # differ from identity. A conservative spectral bound keeps the
            # original feasible region inside the proposal, including after
            # moderately ill-conditioned changes of coordinates.
            actual=transform.T@metric@transform
            margin=(np.max(np.sum(abs(actual-np.eye(len(cols))),axis=1))+
                128*np.finfo(float).eps*len(cols)*np.linalg.norm(transform,2)**2*np.linalg.norm(metric,2))
            if not np.isfinite(margin) or margin>=.25:
                raise ValueError('Prior coordinates cannot safely bound the physical constraint region')
            outer=self.upper*np.sqrt(len(forms))/np.sqrt(1-margin)
            inner=self.lower*np.sqrt(len(forms))/np.sqrt(1+margin)
            radial_floor=(inner/outer)**len(cols)
            count=proposed=0;accepted=[]
            while count<n:
                if time.monotonic()>rng_deadline:raise TimeoutError('Prior sampling budget exceeded; candidate remains unresolved')
                rate=count/proposed if proposed else 1.
                batch=min(4096,max(512,int((n-count)/max(rate,1/4096))))
                directions=rng.normal(size=(batch,len(cols)))
                lengths=np.linalg.norm(directions,axis=1)
                if np.any(lengths==0):continue
                radius=outer*(radial_floor+(1-radial_floor)*rng.random(batch))**(1/len(cols))
                p=(directions*(radius/lengths)[:,None])@transform.T
                mag=np.sqrt(np.maximum(0.,np.einsum('ni,sij,nj->ns',p,forms,p)))
                good=p[((mag>=self.lower)&(mag<=self.upper)).all(axis=1)]
                accepted.append(good);count+=len(good);proposed+=batch
            output[:,cols]=np.concatenate(accepted)[:n]
            self.prior_diagnostics.append(dict(method='uniform_constraint_ellipsoid_rejection',
                dimensions=len(cols),constraints=len(forms),proposals=proposed,accepted=count,
                complete_original_constraints_checked=True))
        if not self.valid(output).all():raise ValueError('Factored prior failed full native moment constraints')
        return output

def sample_posterior(model,*,particles=256,seed=0,moves=12,max_seconds=60.):
    if not isinstance(particles,int) or isinstance(particles,bool) or not isinstance(moves,int) or particles<64 or moves<2 or not np.isfinite(max_seconds) or max_seconds<=0:raise ValueError('Invalid sampler budget')
    start=time.monotonic();rng=np.random.default_rng(seed);x=model.prior(rng,particles,deadline=start+max_seconds);ll=model.loglike(x)
    if not np.isfinite(ll).all():raise ValueError('Nonfinite prior likelihood; inference remains unresolved')
    prior=x.copy();prior_diagnostics=[dict(v) for v in model.prior_diagnostics];beta=0.;history=[];logz=0.
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
    return dict(samples=x,prior=prior,prior_diagnostics=prior_diagnostics,loglike=ll,log_evidence=logz,history=history,seconds=time.monotonic()-start,qualified=False)

def compare_replicas(model,a,b):
    """Necessary agreement screens, never a proof of global convergence.

    Predictive coordinates are the exact whitened QR sufficient statistics;
    they do not require rendering every full pattern again. Physical second
    moments detect orientation/relative-sign disagreement while respecting
    the exact global +/- symmetry. These are per-model replica checks, not
    posterior model probabilities or proof of an identifiable structure.
    """
    samples=[np.asarray(v['samples'],float) for v in (a,b)]
    if any(x.ndim!=2 or x.shape[1]!=model.dimension or len(x)<2
            or not model.valid(x).all() for x in samples):
        raise ValueError('Two complete finite physical replica ensembles are required')
    qa=np.quantile(model.moments(samples[0]),[.025,.5,.975],axis=0);qb=np.quantile(model.moments(samples[1]),[.025,.5,.975],axis=0)
    width=np.maximum(np.maximum(qa[2]-qa[0],qb[2]-qb[0]),.05)
    discrepancy=float(np.max(abs(qa[1]-qb[1])/width))
    def compare(first,second,floor):
        va=float(np.sum(np.var(first,axis=0,ddof=1)));vb=float(np.sum(np.var(second,axis=0,ddof=1)))
        mean=float(np.linalg.norm(np.mean(first,axis=0)-np.mean(second,axis=0))/max(np.sqrt(va+vb),floor))
        spread=abs(va-vb)/max(va+vb,floor**2)
        return dict(mean_discrepancy_in_spread=mean,relative_spread_discrepancy=float(spread),
            passed=mean<.5 and spread<.5)
    predictions=[model.features(x)@model.R.T for x in samples]
    predictive=compare(*predictions,1.)
    metric=np.linalg.cholesky(model.forms.sum(axis=0))
    physical=[x@metric for x in samples]
    factors=np.where(model.i==model.j,1.,np.sqrt(2.))
    second_moments=[x[:,model.i]*x[:,model.j]*factors for x in physical]
    orientation=compare(*second_moments,1e-12*max(1.,len(model.forms)*model.upper**2))
    return dict(median_discrepancy_in_interval_widths=discrepancy,
        predictive_agreement=predictive,physical_second_moment_agreement=orientation,
        screen_passed=bool(discrepancy<.5 and predictive['passed'] and orientation['passed']),
        global_convergence_proven=False,
        scope='Necessary magnitude, predictive and physical second-moment replica agreement; structural identifiability is not established')
