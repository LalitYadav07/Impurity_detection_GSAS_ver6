"""Exact bounded two-coefficient profiling; all structural profiles stay native.

Conditional coefficients are linear intensity scale and background multiplier.
Poisson and Gaussian objectives are convex in these coefficients. Implicit
derivatives include the design-matrix derivative, not only a GN projection.
"""
import numpy as np
from scipy.optimize import lsq_linear
from .fit_fast_native import residual_and_factor
from gpx_recipe.structural_live.observations import CumulativeCounts

def quadratic_box_step(gradient,hessian,lower,upper):
    """Exact two-dimensional convex quadratic step, including every box face."""
    proposals=[np.zeros(2)]
    unconstrained=-np.linalg.lstsq(hessian,gradient,rcond=1e-14)[0]
    if np.all(unconstrained>=lower) and np.all(unconstrained<=upper):proposals.append(unconstrained)
    for fixed in range(2):
        free=1-fixed
        for bound in [lower[fixed],upper[fixed]]:
            point=np.zeros(2);point[fixed]=bound
            slope=gradient[free]+hessian[free,fixed]*bound
            point[free]=np.clip(-slope/hessian[free,free],lower[free],upper[free]) if hessian[free,free]>0 else (lower[free] if slope>=0 else upper[free])
            proposals.append(point)
    return min(proposals,key=lambda p:float(gradient@p+.5*p@hessian@p))

class ConditionalNuisance:
    def __init__(self,observation,signal,fixed,background):
        self.obs=observation;self.signal=np.asarray(signal,float);self.fixed=np.asarray(fixed,float);self.background=np.asarray(background,float)
        if any(x.shape!=observation.y.shape or not np.isfinite(x).all() for x in [self.signal,self.fixed,self.background]):raise ValueError('Finite native intensity arrays required')
        self.design=np.column_stack([self.signal,self.background]);self.mask=observation.mask
        self.poisson=isinstance(observation,CumulativeCounts)
        self.conversion=observation.exposure*observation.count_rate_scale[self.mask] if self.poisson else 1/np.sqrt(observation.variance[self.mask])
        self.x=self.design[self.mask]*self.conversion[:,None]
        self.fixed_units=self.fixed[self.mask]*self.conversion
        self.observed=observation.counts[self.mask] if self.poisson else observation.y[self.mask]*self.conversion
        self.lower=np.array([np.exp(-.3),.25]);self.upper=np.array([np.exp(.3),4.])
        self.coefficients=None;self.iterations=0;self.free=None

    def value(self,coefficients):
        rate=self.fixed+self.design@coefficients
        residual,factor=residual_and_factor(self.obs,rate)
        return float(residual@residual),rate,residual,factor

    def gradient_hessian(self,c):
        mu=self.fixed_units+self.x@c
        if self.poisson:
            if np.any(mu<=0):raise ValueError('Nonpositive Poisson expectation')
            relative=1-self.observed/mu;weight=self.observed/mu**2
        else:relative=mu-self.observed;weight=np.ones(len(mu))
        return self.x.T@relative,self.x.T@(weight[:,None]*self.x),relative,weight

    def solve(self,initial=None,maxiter=80):
        if initial is None:
            initial=lsq_linear(self.x,self.observed-self.fixed_units,bounds=(self.lower,self.upper),method='bvls',tol=1e-12).x
        c=np.clip(np.asarray(initial,float),self.lower,self.upper)
        if c.shape!=(2,) or not np.isfinite(c).all():raise ValueError('Two finite nuisance coefficients required')
        current=self.value(c)[0]
        for iteration in range(maxiter):
            self.iterations=iteration+1;gradient,hessian,_,_=self.gradient_hessian(c)
            active=((c<=self.lower+1e-11)&(gradient>=0))|((c>=self.upper-1e-11)&(gradient<=0))
            free=np.flatnonzero(~active)
            if not len(free):break
            score=float(np.max(abs(gradient[free])/np.sqrt(np.maximum(np.diag(hessian)[free],1e-30))))
            if score<1e-7:break
            step=quadratic_box_step(gradient,hessian,self.lower-c,self.upper-c)
            slope=float(gradient@step)
            if slope>=0:break
            accepted=False
            for power in range(40):
                candidate=np.clip(c+(.5**power)*step,self.lower,self.upper)
                if self.poisson and np.any(self.fixed_units+self.x@candidate<=0):continue
                cost=self.value(candidate)[0]
                mu=self.fixed_units+self.x@c;delta=self.x@(candidate-c)
                change=float(np.sum(delta-self.observed*np.log1p(delta/mu))) if self.poisson else float(delta@(mu-self.observed)+.5*delta@delta)
                if change<=1e-4*(.5**power)*slope:
                    c=candidate;current=cost;accepted=True;break
            if not accepted:break
        gradient,hessian,relative,weight=self.gradient_hessian(c)
        active=((c<=self.lower+1e-9)&(gradient>=-1e-7))|((c>=self.upper-1e-9)&(gradient<=1e-7))
        self.free=np.flatnonzero(~active);self.coefficients=c.copy();self.hessian=hessian
        self.relative=relative;self.weight=weight
        self.projected_score=0. if not len(self.free) else float(np.max(abs(gradient[self.free])/np.sqrt(np.maximum(np.diag(hessian)[self.free],1e-30))))
        if self.projected_score>1e-4:raise ValueError('Conditional nuisance optimization did not satisfy its local KKT tolerance')
        return self.value(c)

    def rate_jacobian(self,native_signal_jacobian):
        if self.coefficients is None:raise ValueError('Solve conditional nuisance coefficients first')
        jac=np.asarray(native_signal_jacobian,float)
        if jac.ndim!=2 or jac.shape[1]!=len(self.signal) or not np.isfinite(jac).all():raise ValueError('Finite native signal derivatives required')
        dx=jac[:,self.mask]*self.conversion
        partial=self.coefficients[0]*dx
        cross=self.x.T@(self.weight[:,None]*partial.T)
        cross[0]+=dx@self.relative
        dc=np.zeros((2,len(jac)))
        if len(self.free):
            dc[self.free]=-np.linalg.lstsq(self.hessian[np.ix_(self.free,self.free)],cross[self.free],rcond=1e-12)[0]
        return self.coefficients[0]*jac+(self.design@dc).T

    def diagnostics(self):
        return dict(coefficients=self.coefficients.tolist(),free_coefficients=self.free.tolist(),
            projected_kkt_score=self.projected_score,iterations=self.iterations,objective='poisson' if self.poisson else 'gaussian',
            structural_parameters_reduced=False,original_likelihood=True)
