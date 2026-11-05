"""Shared reference uncertainty for explicitly supplied Gaussian reduced data.

The algebra is exact for a linear Gaussian nuisance model. It does not assert
that a native diffraction profile is linear in lattice/profile parameters or
that a supplied reduced-data covariance is calibrated. Those are separate
qualification gates. Physical-mode predictions remain candidate-specific.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve, solve_triangular

from .observations import array_digest


def readonly(value):
    result=np.array(value,dtype=float,copy=True)
    if not np.isfinite(result).all():raise ValueError('Finite Gaussian model arrays required')
    result.flags.writeable=False
    return result


def covariance_factor(covariance):
    """PSD factor after correlation scaling; retain small-unit parameters."""
    covariance=readonly(covariance)
    if covariance.ndim!=2 or covariance.shape[0]!=covariance.shape[1]:
        raise ValueError('Square nuisance covariance required')
    if not len(covariance):return np.empty((0,0))
    variance=np.diag(covariance)
    if np.any(variance<0):raise ValueError('Negative nuisance variance')
    active=variance>0
    if np.any(covariance[~active]!=0) or np.any(covariance[:,~active]!=0):
        raise ValueError('Zero-variance parameters must have zero covariance')
    result=np.zeros((len(covariance),int(active.sum())))
    if not active.any():return result
    sd=np.sqrt(variance[active]);corr=covariance[np.ix_(active,active)]/sd[:,None]/sd[None,:]
    tolerance=64*np.finfo(float).eps*len(corr)
    if not np.allclose(corr,corr.T,atol=tolerance,rtol=0):raise ValueError('Asymmetric nuisance covariance')
    eigen,vectors=np.linalg.eigh((corr+corr.T)/2)
    if eigen.min() < -tolerance:raise ValueError('Indefinite nuisance covariance')
    # Only correlation eigenvalues indistinguishable from roundoff are gauges.
    use=eigen>tolerance
    result=np.zeros((len(covariance),int(use.sum())))
    result[active]=sd[:,None]*vectors[:,use]*np.sqrt(eigen[use])
    return result


class GaussianNoise:
    """Known covariance diag(diagonal) + factors @ factors.T, without n² storage."""
    def __init__(self,diagonal,factors=None):
        self.diagonal=readonly(diagonal)
        if self.diagonal.ndim!=1 or not self.diagonal.size or np.any(self.diagonal<=0):
            raise ValueError('Positive finite variance in every analyzed bin required')
        n=len(self.diagonal)
        self.factors=readonly(np.empty((n,0)) if factors is None else factors)
        if self.factors.ndim!=2 or self.factors.shape[0]!=n:raise ValueError('Noise factor bins changed')
        self.whitened=self.factors/np.sqrt(self.diagonal[:,None])
        self.q,self.r=np.linalg.qr(self.whitened,mode='reduced')
        small=np.eye(self.q.shape[1])+self.r@self.r.T
        self.cholesky=np.linalg.cholesky(small)
        self.logdet=float(np.log(self.diagonal).sum()+2*np.log(np.diag(self.cholesky)).sum())

    @property
    def fingerprint(self):return array_digest(self.diagonal,self.factors)

    @property
    def marginal_variance(self):return self.diagonal+np.sum(self.factors**2,axis=1)

    def solve(self,value):
        value=np.asarray(value,float)
        if value.ndim not in [1,2] or value.shape[0]!=len(self.diagonal) or not np.isfinite(value).all():
            raise ValueError('Noise solve requires finite vector or columns on the analyzed bins')
        matrix=value[:,None] if value.ndim==1 else value
        whitened=matrix/np.sqrt(self.diagonal[:,None])
        projection=self.q.T@whitened
        orthogonal=whitened-self.q@projection
        scaled=(orthogonal+self.q@cho_solve((self.cholesky,True),projection))/np.sqrt(self.diagonal[:,None])
        return scaled[:,0] if value.ndim==1 else scaled

    def log_density(self,residual):
        residual=np.asarray(residual,float)
        if residual.shape!=self.diagonal.shape:raise ValueError('One residual vector on the analyzed bins required')
        if not np.isfinite(residual).all():raise ValueError('Finite residual required')
        whitened=residual/np.sqrt(self.diagonal);projection=self.q.T@whitened
        orthogonal=whitened-self.q@projection
        value=float(orthogonal@orthogonal+projection@cho_solve((self.cholesky,True),projection))
        return -.5*(value+self.logdet+len(residual)*np.log(2*np.pi))

    def augment(self,persistent_factors):
        extra=np.asarray(persistent_factors,float)
        if extra.ndim!=2 or extra.shape[0]!=len(self.diagonal):raise ValueError('Shared nuisance factor bins changed')
        return GaussianNoise(self.diagonal,np.column_stack([self.factors,extra]))

    def draw(self,rng,size=1):
        if type(size) is not int or size<1:raise ValueError('Positive number of Gaussian draws required')
        return (rng.normal(size=(size,len(self.diagonal)))*np.sqrt(self.diagonal)
                +rng.normal(size=(size,self.factors.shape[1]))@self.factors.T)


@dataclass(frozen=True)
class NuisancePosterior:
    names: tuple
    mean: np.ndarray
    factor: np.ndarray
    reference_fingerprint: str
    log_reference_evidence: float

    def __post_init__(self):
        object.__setattr__(self,'names',tuple(self.names))
        object.__setattr__(self,'mean',readonly(self.mean));object.__setattr__(self,'factor',readonly(self.factor))
        if (len(set(self.names))!=len(self.names) or any(not isinstance(n,str) or not n for n in self.names)
                or self.mean.shape!=(len(self.names),) or self.factor.ndim!=2
                or self.factor.shape[0]!=len(self.names) or not self.reference_fingerprint
                or not np.isfinite(self.log_reference_evidence)):
            raise ValueError('Named finite nuisance posterior with reference identity required')

    @property
    def covariance(self):return self.factor@self.factor.T


def _latent_update(residual,response,noise):
    """z~N(0,I), r=response z+noise. Return posterior mean and a square root."""
    rank=response.shape[1]
    if rank==0:return np.empty(0),np.empty((0,0))
    weighted=noise.solve(response)
    precision=np.eye(rank)+response.T@weighted
    cholesky=np.linalg.cholesky((precision+precision.T)/2)
    mean=cho_solve((cholesky,True),response.T@noise.solve(residual))
    factor=solve_triangular(cholesky.T,np.eye(rank),lower=False)
    return mean,factor


def condition_reference(*,names,observed,baseline,response,noise,prior_mean,prior_covariance,context):
    """Consume the parent measurement once; retain its common nuisance posterior."""
    observed,baseline,response,prior_mean=map(readonly,[observed,baseline,response,prior_mean])
    names=tuple(names)
    if (not context or observed.shape!=noise.diagonal.shape or baseline.shape!=observed.shape
            or response.shape!=(len(observed),len(names)) or prior_mean.shape!=(len(names),)):
        raise ValueError('Reference arrays, nuisance names and context must agree')
    prior_factor=covariance_factor(prior_covariance)
    if prior_factor.shape[0]!=len(names):raise ValueError('Prior nuisance dimensions changed')
    latent_response=response@prior_factor
    residual=observed-baseline-response@prior_mean
    latent_mean,latent_factor=_latent_update(residual,latent_response,noise)
    evidence=noise.augment(latent_response).log_density(residual)
    fingerprint=context+':'+array_digest(observed,baseline,response,prior_mean,np.asarray(prior_covariance))+':'+noise.fingerprint
    return NuisancePosterior(names,prior_mean+prior_factor@latent_mean,prior_factor@latent_factor,fingerprint,evidence)


@dataclass(frozen=True)
class GaussianPrediction:
    """Mean plus persistent latent uncertainty, common across cumulative scans.

    Statistical noise belongs to each measurement, not these latent factors.
    Future simulation must condition these latents on the current scan before
    generating new independent measurement noise.
    """
    mean: np.ndarray
    persistent_factors: np.ndarray
    reference_fingerprint: str

    def __post_init__(self):
        object.__setattr__(self,'mean',readonly(self.mean))
        object.__setattr__(self,'persistent_factors',readonly(self.persistent_factors))
        if (self.mean.ndim!=1 or self.persistent_factors.ndim!=2
                or self.persistent_factors.shape[0]!=len(self.mean) or not self.reference_fingerprint):
            raise ValueError('Prediction bins, shared factors and reference context required')

    def log_likelihood(self,observed,noise):
        observed=np.asarray(observed,float)
        if observed.shape!=self.mean.shape:raise ValueError('Target observation bins changed')
        return noise.augment(self.persistent_factors).log_density(observed-self.mean)

    def condition_target(self,observed,noise):
        observed=np.asarray(observed,float)
        if observed.shape!=self.mean.shape:raise ValueError('Target observation bins changed')
        mean,factor=_latent_update(observed-self.mean,self.persistent_factors,noise)
        return GaussianPrediction(self.mean+self.persistent_factors@mean,
            self.persistent_factors@factor,self.reference_fingerprint)

    def draw_latent_means(self,rng,size):
        if type(size) is not int or size<1:raise ValueError('Positive number of predictive latent draws required')
        return self.mean+rng.normal(size=(size,self.persistent_factors.shape[1]))@self.persistent_factors.T


def target_prediction(baseline,response,reference,*,names,target_factors=None,target_mean=None):
    """Candidate-native mean/response, shared reference, independent target effects.

    Target factors describe physically allowed thermal/background/calibration
    changes. They are not eliminated simply because the parent was precise.
    No scattering matrix is shared between competing structural candidates.
    """
    baseline,response=readonly(baseline),readonly(response)
    if tuple(names)!=reference.names:raise ValueError('Shared nuisance coordinate order or identity changed')
    if baseline.ndim!=1 or response.shape!=(len(baseline),len(reference.names)):
        raise ValueError('Candidate response dimensions changed')
    factors=response@reference.factor
    mean=baseline+response@reference.mean
    if target_factors is not None:
        extra=readonly(target_factors)
        if extra.ndim!=2 or extra.shape[0]!=len(baseline):raise ValueError('Target-specific factor dimensions changed')
        factors=np.column_stack([factors,extra])
    if target_mean is not None:
        shift=readonly(target_mean)
        if shift.shape!=baseline.shape:raise ValueError('Target-specific mean dimensions changed')
        mean+=shift
    return GaussianPrediction(mean,factors,reference.reference_fingerprint)
