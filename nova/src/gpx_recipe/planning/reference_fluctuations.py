"""Bounded reference-count fluctuation calibration on an actual minute clock.

This is a development API. The restricted Gaussian likelihood is a high-count
calibration approximation to positive lognormal rates with Poisson counts.
It does not identify beamline hardware, certify predictive coverage, or change
the planner's default count law. Caller-selected calibration data must exclude
future test outcomes. Distinct count channels, not overlapping event views,
are required. Fluctuation-parameter uncertainty is not inferred by this fit.
"""
import numpy as np
from scipy.linalg import null_space, cho_factor, cho_solve
from scipy.optimize import minimize

VERSION = 'powgen-reference-fluctuation-calibration/1'


def rate_covariance(fit, times_minutes):
    t = np.asarray(times_minutes, float)
    a = np.asarray(fit['log_rate_loadings'], float)
    if (t.ndim != 1 or not 1 <= len(t) <= 240 or not np.isfinite(t).all()
            or np.any(np.diff(t) <= 0) or a.shape != (4,)
            or not np.isfinite(a).all() or np.any(a < 0) or np.any(a > .05)):
        raise ValueError('Bounded increasing minute coordinates and four finite loadings required')
    kind = fit['chosen']
    if kind == 'poisson':
        if np.any(a != 0):
            raise ValueError('A Poisson calibration cannot carry fluctuation loadings')
        return np.zeros((len(t)*4, len(t)*4))
    if kind == 'white':
        kernel = np.eye(len(t))
    elif kind == 'oscillatory':
        damping, frequency = fit['damping_per_minute'], fit['frequency_cycles_per_minute']
        if (not np.isfinite([damping, frequency]).all() or not 0 <= damping <= 4
                or not 0 <= frequency <= .5):
            raise ValueError('Invalid damping or frequency')
        lag = abs(t[:, None]-t)
        kernel = np.exp(-damping*lag)*np.cos(2*np.pi*frequency*lag)
    else:
        raise ValueError('Unknown fluctuation family')
    return np.expm1(np.einsum('ts,gh->tgsh', kernel, np.outer(a, a))).reshape(len(t)*4, len(t)*4)


class RestrictedCountLikelihood:
    def __init__(self, counts, charge_uah, times_minutes, *, background_rate=None, background_fit=None):
        y, q, t = np.asarray(counts), np.asarray(charge_uah), np.asarray(times_minutes)
        if (y.ndim != 2 or y.shape[1] != 4 or not 8 <= len(y) <= 240
                or q.shape != (len(y),) or t.shape != q.shape
                or any(v.dtype.kind not in 'iuf' or not np.isfinite(v).all() for v in (y,q,t))
                or np.any(y < 0) or np.any(y > 1e9) or not np.array_equal(y, np.rint(y))
                or np.any(q <= 0) or np.any(np.diff(t) <= 0)):
            raise ValueError('Supply 8–240 positive-dose observations, four disjoint integer-count channels and increasing minute coordinates')
        self.y, self.q, self.times = [np.array(v, float, copy=True) for v in (y,q,t)]
        self.n, self.g = self.y.shape
        self.rate = self.y.sum(0)/self.q.sum()
        self.mu = self.q[:,None]*self.rate
        if not np.isfinite(self.mu).all() or np.any(self.mu < 10):
            raise ValueError('The Gaussian calibration approximation requires expected counts >=10 in every supplied cell')
        self.root = np.sqrt(self.mu.ravel())
        self.outer = np.outer(self.root, self.root)
        self.design = np.tile(np.eye(4),(self.n,1))*self.root[:,None]
        self.u = null_space(self.design.T)
        self.standardized_residual = (self.y-self.mu).ravel()/self.root
        self.residual = self.u.T@self.standardized_residual
        self.lag = abs(self.times[:,None]-self.times)
        self.identity = np.eye(self.n*4)
        self.small_identity = np.eye(self.n*4-4)
        self.extra_covariance = np.zeros_like(self.identity)
        if (background_rate is None) != (background_fit is None):
            raise ValueError('Background rate and its declared fluctuation law must be supplied together')
        if background_rate is not None:
            b = np.asarray(background_rate,float)
            if b.shape != (4,) or not np.isfinite(b).all() or np.any(b < 0) or np.any(b >= self.rate):
                raise ValueError('Each total reference rate must exceed its nonnegative background rate')
            fraction = b/self.rate
            root_b = self.root*np.tile(fraction,self.n)
            self.extra_covariance = np.outer(root_b,root_b)*rate_covariance(background_fit,self.times)
            root_s = self.root*np.tile(1-fraction,self.n)
            self.outer = np.outer(root_s,root_s)

    def covariance(self, parameters, kind, derivatives=False):
        p = np.asarray(parameters,float)
        expected = 4 if kind == 'white' else 6 if kind == 'oscillatory' else None
        if (expected is None or p.shape != (expected,) or not np.isfinite(p).all()
                or np.any(p[:4] < 0) or np.any(p[:4] > 5)
                or (kind == 'oscillatory' and (not 0 <= p[-2] <= 4 or not 0 <= p[-1] <= .5))):
            raise ValueError('Parameters exceed the declared fluctuation domain')
        amplitude = .01*p[:4]
        if kind == 'white':
            kernel = np.eye(self.n)
        else:
            damping,frequency = p[-2:]
            kernel = np.exp(-damping*self.lag)*np.cos(2*np.pi*frequency*self.lag)
        aa = np.outer(amplitude,amplitude)
        logcov = np.einsum('ts,gh->tgsh',kernel,aa)
        c = self.identity+self.outer*np.expm1(logcov).reshape(self.identity.shape)+self.extra_covariance
        if not derivatives:return c
        exponential=np.exp(logcov);derivative=[]
        for j in range(4):
            unit=np.zeros(4);unit[j]=.01
            da=np.outer(unit,amplitude)+np.outer(amplitude,unit)
            derivative.append(self.outer*(exponential*np.einsum('ts,gh->tgsh',kernel,da)).reshape(c.shape))
        if kind=='oscillatory':
            for dk in [-self.lag*kernel,-2*np.pi*self.lag*np.exp(-damping*self.lag)*np.sin(2*np.pi*frequency*self.lag)]:
                derivative.append(self.outer*(exponential*np.einsum('ts,gh->tgsh',dk,aa)).reshape(c.shape))
        return c,derivative

    def objective(self,parameters,kind):
        c,derivatives=self.covariance(parameters,kind,True)
        small=self.u.T@c@self.u
        factor=cho_factor(small,lower=True,check_finite=False)
        solved=cho_solve(factor,self.residual,check_finite=False)
        value=.5*(2*np.log(np.diag(factor[0])).sum()+self.residual@solved)
        adjoint=cho_solve(factor,self.small_identity,check_finite=False)-np.outer(solved,solved)
        full=self.u@adjoint@self.u.T
        return float(value),np.array([.5*np.sum(full*dc) for dc in derivatives])


def fit_fluctuations(counts,charge_uah,times_minutes,*,background_rate=None,background_fit=None):
    model=RestrictedCountLikelihood(counts,charge_uah,times_minutes,
        background_rate=background_rate,background_fit=background_fit)
    relative=(model.y-model.mu)/model.mu
    excess=np.maximum(np.var(relative,axis=0,ddof=1)-np.mean(1/model.mu,axis=0),0)
    initial=np.clip(np.sqrt(excess)/.01,.01,4.)
    null,_=model.objective(np.zeros(4),'white')
    candidates=[dict(kind='poisson',parameters=[],objective=null,criterion=2*null,starts=[])]
    starts=[np.r_[initial,d,f] for f in (.1,.25,.4,.48) for d in (.02,.2,1.)]
    for kind,initials in [('white',[initial]),('oscillatory',starts)]:
        bounds=[(0.,5.)]*4+([] if kind=='white' else [(0.,4.),(0.,.5)])
        rows=[]
        for first in initials:
            result=minimize(model.objective,first,args=(kind,),method='L-BFGS-B',jac=True,bounds=bounds,
                            options=dict(maxiter=500,ftol=1e-11,gtol=1e-6))
            rows.append(dict(initial=first.tolist(),parameters=result.x.tolist(),objective=float(result.fun),
                             success=bool(result.success),message=str(result.message),iterations=int(result.nit)))
        eligible=[r for r in rows if r['success'] and np.isfinite(r['objective'])]
        if not eligible:
            raise RuntimeError('No converged '+kind+' calibration; retain this failure and do not select another family silently')
        best=min(eligible,key=lambda r:r['objective'])
        candidates.append(dict(kind=kind,parameters=best['parameters'],objective=best['objective'],
                               criterion=2*best['objective']+len(bounds)*np.log(model.n),starts=rows))
    chosen=min(candidates,key=lambda r:r['criterion']);p=np.asarray(chosen['parameters'])
    return dict(version=VERSION,chosen=chosen['kind'],log_rate_loadings=(.01*p[:4] if len(p) else np.zeros(4)).tolist(),
        damping_per_minute=float(p[-2]) if chosen['kind']=='oscillatory' else None,
        frequency_cycles_per_minute=float(p[-1]) if chosen['kind']=='oscillatory' else None,
        candidates=candidates,calibration_observations=model.n,calibration_charge_uah=float(model.q.sum()),
        calibration_clock_minutes=model.times.tolist(),training_rate_per_uah=model.rate.tolist(),
        criterion='2*restricted_negative_log_likelihood + parameter_count*log(number_of_positive_dose_calibration_minutes)',
        background_included_in_reference_covariance=background_rate is not None,
        approximation='High-count Gaussian restricted calibration with exact lognormal-Poisson moment covariance.',
        fluctuation_parameter_uncertainty_included=False,independent_source_background_fields_assumed=background_rate is not None,
        physical_cause_identified=False,experimental_accuracy_certified=False)
