"""Finite-table thermal scattering density with NCrystal's SAB convention.

Alpha uses neutron-mass units: (Ei+Ef-2*sqrt(Ei*Ef)*cos(theta))/kT.
Beta=(Ef-Ei)/kT, positive for neutron energy gain. Interpolation is log-linear
in alpha, falling back to linear if either value is zero, then linear in beta.
This is the ALPHA_FIRST/LOGLIN convention of NCrystal's SABEval. No extrapolated
free-gas contribution is invented outside the supplied table.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SABDensity:
    alpha: np.ndarray
    beta: np.ndarray
    values: np.ndarray
    kT_ev: float
    bound_cross_section_b: float

    def __post_init__(self):
        alpha, beta, values = [np.array(v, dtype=float, copy=True)
            for v in (self.alpha, self.beta, self.values)]
        for grid in (alpha, beta):
            if (grid.ndim != 1 or not 2 <= len(grid) <= 10001
                    or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0)):
                raise ValueError('SAB grids must be finite and increasing')
        if (alpha[0] < 0 or values.shape != (len(beta),len(alpha))
                or values.size > 4_000_000 or not np.isfinite(values).all() or np.any(values < 0)):
            raise ValueError('SAB values need a matching nonnegative finite table')
        for v in (self.kT_ev, self.bound_cross_section_b):
            if isinstance(v,(bool,np.bool_)) or not np.isscalar(v) or not np.isfinite(v) or v <= 0:
                raise ValueError('Positive kT and bound cross section are required')
        for name, data in [('alpha',alpha),('beta',beta),('values',values)]:
            data.setflags(write=False)
            object.__setattr__(self,name,data)

    def evaluate(self, alpha, beta):
        a,b=np.broadcast_arrays(np.asarray(alpha,float),np.asarray(beta,float))
        if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a<0):
            raise ValueError('SAB queries require finite alpha>=0 and finite beta')
        shape=a.shape
        a,b=a.ravel(),b.ravel()
        valid=(a>=self.alpha[0])&(a<=self.alpha[-1])&(b>=self.beta[0])&(b<=self.beta[-1])
        result=np.zeros(len(a))
        av,bv=a[valid],b[valid]
        ia=np.clip(np.searchsorted(self.alpha,av,side='right')-1,0,len(self.alpha)-2)
        ib=np.clip(np.searchsorted(self.beta,bv,side='right')-1,0,len(self.beta)-2)
        wa=(av-self.alpha[ia])/(self.alpha[ia+1]-self.alpha[ia])
        wb=(bv-self.beta[ib])/(self.beta[ib+1]-self.beta[ib])
        rows=[]
        for offset in (0,1):
            lo,hi=self.values[ib+offset,ia],self.values[ib+offset,ia+1]
            both=(lo>0)&(hi>0)&(wa>0)&(wa<1)
            value=(1-wa)*lo+wa*hi
            value[both]=np.exp((1-wa[both])*np.log(lo[both])+wa[both]*np.log(hi[both]))
            rows.append(value)
        result[valid]=(1-wb)*rows[0]+wb*rows[1]
        return result.reshape(shape)

    def differential_b_per_sr_ev(self, initial_energy_ev, final_energy_ev, cos_two_theta):
        ei,ef,mu=np.broadcast_arrays(*[np.asarray(v,float)
            for v in (initial_energy_ev,final_energy_ev,cos_two_theta)])
        if (not all(np.isfinite(v).all() for v in (ei,ef,mu))
                or np.any(ei<=0) or np.any(ef<=0) or np.any(mu<-1) or np.any(mu>1)):
            raise ValueError('Differential scattering needs positive energies and a physical cosine')
        rooti,rootf=np.sqrt(ei),np.sqrt(ef)
        alpha=((rooti-rootf)**2+2*rooti*rootf*(1-mu))/self.kT_ev
        beta=(ef-ei)/self.kT_ev
        return (self.bound_cross_section_b/(4*np.pi*self.kT_ev)
                *rootf/rooti*self.evaluate(alpha,beta))
