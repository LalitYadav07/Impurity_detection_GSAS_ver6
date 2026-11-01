"""Experimental analytic marginalization of ONE global normalization factor.

s ~ N(1, tau^2), y | s,p ~ N(s*f(p), diag(sigma^2)). This is not
sample-specific absorption or full GPX covariance. tau requires independent
calibration; this research model is not enabled by the live worker.
"""
import numpy as np
from .statistics import QuadraticObservation


class MarginalNormalizationObservation(QuadraticObservation):
    def __init__(self,*args,normalization_sd,**kwargs):
        super().__init__(*args,**kwargs)
        if not np.isfinite(normalization_sd) or not 0<normalization_sd<.2:raise ValueError('An independently calibrated small normalization uncertainty is required')
        self.tau=float(normalization_sd)
        self.response=np.c_[self.base/self.sigma,self.design.T/self.sigma[:,None]]
        self.response_q,self.response_r=np.linalg.qr(self.response,mode='reduced')
        self.y_projection=self.response_q.T@(self.y/self.sigma)
    def loglike(self,p):
        features=self.features(p);compressed=np.c_[np.ones(len(features)),features]@self.response_r.T
        norm=np.einsum('ij,ij->i',compressed,compressed)
        fr=np.einsum('ij,ij->i',compressed,self.y_projection-compressed)
        den=1+self.tau**2*norm
        return super().loglike(p)+.5*self.tau**2*fr**2/den-.5*np.log(den)
