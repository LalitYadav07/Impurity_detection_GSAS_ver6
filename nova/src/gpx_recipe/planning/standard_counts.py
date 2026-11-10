"""Positive standard normalization with shared V and empty-can uncertainty.

The standard supplies an aggregate signal S in fixed count regions. Its ON/OFF
posterior is joint with the aggregate can rate B. Vanadium supplies a response
shape s. A new specimen's rate is S*A*s/(a*s), where a is the standard's frozen
transfer functional. This cancels uncertainty in V's overall amplitude rather
than double-counting it as an independent fitted-scale error.

Sampling uses the positive nonlinear ratio. Reported means/covariances use its
first-order expansion in s, retaining exact S/B and Dirichlet moments. The
denominator CV is bounded and recorded; physical adequacy is not established.
"""
import numpy as np
from scipy import sparse
from .count_response import on_off_posterior
from .empirical_counts import ReferenceCountCampaign


class StandardCountCampaign(ReferenceCountCampaign):
    def __init__(self, posterior, source_map, background_map, normalization, seed=0, *,
                 calibration_weights, standard_on_counts, standard_charge,
                 can_counts, can_charge, calibration_can_mask, can_map=None,
                 reduction_background=None):
        self.posterior=posterior
        self.source_map,self.background_map=posterior._weights(source_map,background_map)
        self.normalization=np.asarray(normalization,float)
        self.a=np.asarray(calibration_weights,float)
        n=np.asarray(can_counts);mask=np.asarray(calibration_can_mask)
        if (self.a.shape!=posterior.source_mean.shape or not np.isfinite(self.a).all() or np.any(self.a<0)
                or posterior.source_mean.ndim!=1 or n.ndim!=1 or n.size>100000 or n.dtype.kind not in 'iu' or np.any(n<0) or np.any(n>1e9)
                or mask.shape!=n.shape or mask.dtype.kind!='b' or not mask.any()):
            raise ValueError('Bounded standard weights, integer can counts and a Boolean region mask required')
        self.cal=on_off_posterior([standard_on_counts],[int(n[mask].sum())],standard_charge,can_charge)
        self.denominator=float(self.a@posterior.source_mean)
        self.denominator_variance=float((self.a**2)@posterior.source_variance)
        if not np.isfinite(self.denominator) or self.denominator<=0:
            raise ValueError('Standard normalization has no positive response coverage')
        self.denominator_cv=np.sqrt(self.denominator_variance)/self.denominator
        if self.denominator_cv>.02:
            raise ValueError('Standard response denominator is too uncertain for first-order predictive moments')
        self.signal=float(self.cal.source_mean[0]);self.signal_var=float(self.cal.source_variance[0])
        self.cal_blank=float(self.cal.background_mean[0]);self.cal_blank_var=float(self.cal.background_variance[0])
        self.cal_cross=float(self.cal.source_background_covariance[0])
        self.gain=self.signal/self.denominator
        self.unit_rate=np.asarray(self.source_map@posterior.source_mean)/self.denominator
        self.background=np.asarray(self.background_map@posterior.background_mean)
        self.extra_counts=n.copy();self.extra_charge=float(can_charge);self.cal_mask=mask.copy()
        self.extra_map=sparse.csr_matrix((len(self.unit_rate),len(n))) if can_map is None else sparse.csr_matrix(can_map,dtype=float)
        if (self.extra_map.shape!=(len(self.unit_rate),len(n)) or self.extra_map.nnz>2000000
                or not np.isfinite(self.extra_map.data).all() or np.any(self.extra_map.data<0)):
            raise ValueError('Can map must be bounded, finite and nonnegative')
        self.alpha=n[mask]+1.;self.fraction=self.alpha/self.alpha.sum()
        self.can_diag=(self.cal_blank_var+self.cal_blank**2)/(self.alpha.sum()+1)*self.fraction
        self.can_rank=self.cal_blank_var-(self.cal_blank_var+self.cal_blank**2)/(self.alpha.sum()+1)
        self.extra_mean=(n+1.)/self.extra_charge
        self.extra_mean[mask]=self.cal_blank*self.fraction
        self.extra_variance=(n+1.)/self.extra_charge**2
        self.extra_variance[mask]=self.can_diag+self.can_rank*self.fraction**2
        self.background+=self.extra_map@self.extra_mean
        self.rate=self.signal*self.unit_rate+self.background
        self.subtraction=np.asarray(self.background if reduction_background is None else reduction_background,float).copy()
        if (self.normalization.shape!=self.rate.shape or np.any(self.normalization<=0)
                or not np.isfinite(self.normalization).all() or self.subtraction.shape!=self.rate.shape
                or not np.isfinite(self.subtraction).all()):
            raise ValueError('Positive normalization and finite background must match the count grid')
        p=posterior;u=self.unit_rate;D=self.denominator;A=self.source_map;B=self.background_map
        vu=(A.multiply(A)@p.source_variance-2*u*(A@(self.a*p.source_variance))+u*u*self.denominator_variance)/D**2
        cross=(A.multiply(B)@p.source_background_covariance-u*(B@(self.a*p.source_background_covariance)))/D
        within=self.extra_map[:,mask];outside=self.extra_map[:,~mask]
        self.can_transfer=np.asarray(within@self.fraction)
        can_variance=within.multiply(within)@self.can_diag+self.can_rank*self.can_transfer**2
        can_variance+=outside.multiply(outside)@self.extra_variance[~mask]
        self.rate_variance=((self.signal_var+self.signal**2)*vu+self.signal_var*u*u
            +B.multiply(B)@p.background_variance+2*self.signal*cross+can_variance
            +2*self.cal_cross*u*self.can_transfer)
        if np.any(self.rate_variance < -1e-12*np.maximum(1.,self.rate**2)):
            raise ValueError('Standard-normalized predictive covariance has a negative diagonal')
        self.rate_variance=np.maximum(self.rate_variance,0.)
        seeds=np.random.SeedSequence(seed).spawn(5)
        s,b=p.sample_rates(1,seed=seeds[0]);s,b=s.ravel(),b.ravel()
        signal,blank=self.cal.sample_rates(1,seed=seeds[1])
        self.sampled_standard_signal=float(signal.ravel()[0]);self.sampled_standard_blank=float(blank.ravel()[0])
        self.sampled_gain=self.sampled_standard_signal/float(self.a@s)
        can_rng=np.random.default_rng(seeds[2]);can=can_rng.gamma(n+1.,1/self.extra_charge)
        can[mask]=self.sampled_standard_blank*can_rng.dirichlet(self.alpha)
        self.sampled_rate=self.sampled_gain*(A@s)+B@b+self.extra_map@can
        self.rng=np.random.default_rng(seeds[3]);self.counts=np.zeros(len(self.rate),dtype=np.int64);self.exposure=0.

    def projected_covariance(self, weights, first_charge, second_charge=None):
        w=sparse.csr_matrix(weights,dtype=float)
        if w.shape[1]!=len(self.rate) or w.shape[0]>256 or not np.isfinite(w.data).all():
            raise ValueError('Supply at most 256 finite pattern functionals')
        q1,q2=float(first_charge),float(first_charge if second_charge is None else second_charge)
        if not np.isfinite(q1+q2) or min(q1,q2)<=0:raise ValueError('Positive finite charge required')
        z=w.multiply(1/self.normalization);u=np.asarray(z@self.unit_rate)
        jac=(np.asarray((z@self.source_map).toarray())-u[:,None]*self.a)/self.denominator
        b=z@self.background_map;p=self.posterior
        result=(self.signal_var+self.signal**2)*((jac*p.source_variance)@jac.T)
        result+=self.signal_var*np.outer(u,u)+(b.multiply(p.background_variance)@b.T).toarray()
        cross=np.asarray((b@(jac*p.source_background_covariance).T).T)
        result+=self.signal*(cross+cross.T)
        extra=z@self.extra_map;within=extra[:,self.cal_mask];outside=extra[:,~self.cal_mask]
        t=np.asarray(within@self.fraction)
        result+=(within.multiply(self.can_diag)@within.T).toarray()+self.can_rank*np.outer(t,t)
        result+=(outside.multiply(self.extra_variance[~self.cal_mask])@outside.T).toarray()
        result+=self.cal_cross*(np.outer(u,t)+np.outer(t,u))
        result+=(z.multiply(self.rate/max(q1,q2))@z.T).toarray()
        return result

    def reference_fields(self):
        fields=super().reference_fields()
        fields.update(model_type='standard_normalized_on_off_ratio',
            moment_approximation='first_order_in_vanadium_response_ratio; exact_standard_on_off_and_can_dirichlet_moments',
            calibration_weights=self.a,calibration_denominator=self.denominator,
            calibration_denominator_cv=self.denominator_cv,calibration_can_mask=self.cal_mask,
            standard_on_counts=self.cal.on_counts,standard_off_counts=self.cal.off_counts,
            standard_on_charge_uah=self.cal.on_exposure,standard_off_charge_uah=self.cal.off_exposure,
            standard_signal_mean=self.signal,standard_signal_variance=self.signal_var,
            standard_background_mean=self.cal_blank,standard_background_variance=self.cal_blank_var,
            standard_source_background_covariance=self.cal_cross,standard_gain=self.gain,
            sampled_standard_gain=self.sampled_gain,can_dirichlet_alpha=self.alpha)
        return fields
