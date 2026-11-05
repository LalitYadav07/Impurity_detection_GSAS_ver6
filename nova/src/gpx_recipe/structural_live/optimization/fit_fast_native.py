"""Residuals for exact counts or supplied reduced-data variances."""
import numpy as np
from gpx_recipe.structural_live.observations import CumulativeCounts, poisson_deviance_residuals, poisson_means

def residual_and_factor(observation,prediction):
    use=observation.mask
    if not isinstance(observation,CumulativeCounts):
        factor=1/np.sqrt(observation.variance[use])
        return (prediction[use]-observation.y[use])*factor,factor
    residual=poisson_deviance_residuals(observation,prediction)
    mu=poisson_means(observation,prediction);k=observation.counts[use]
    derivative=np.empty_like(mu);near=(k>0)&(abs(mu-k)<1e-5*np.maximum(k,1))
    derivative[near]=1/np.sqrt(k[near])
    derivative[~near]=(1-k[~near]/mu[~near])/residual[~near]
    return residual,derivative*observation.exposure*observation.count_rate_scale[use]
