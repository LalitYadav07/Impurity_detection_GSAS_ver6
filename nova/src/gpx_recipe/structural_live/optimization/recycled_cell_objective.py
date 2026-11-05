"""Short-lived secant proposals for cell columns; every score stays native.

Native displacement derivatives are recomputed at every iterate. Cell
columns are periodically rebuilt with feasible native finite differences.
No approximate derivative is used to report a fitted curve or uncertainty.
"""
import numpy as np
from .constrained_native_fit import FeasibleNativeObjective
from .reference_secants import feasible_secant


class RecycledCellObjective(FeasibleNativeObjective):
    def __init__(self,*args,max_reuses=3,mismatch_limit=.25,**kwargs):
        super().__init__(*args,**kwargs)
        self.max_reuses=max_reuses;self.mismatch_limit=mismatch_limit
        self.cell_model=None;self.cell_reuses=0;self.cell_rebuilds=0;self.last_reused=False

    def invalidate_cell_approximation(self):
        self.cell_model=None
        if self.last is not None:self.last['jac']=None

    def select_proposal_steps(self,z):
        self.invalidate_cell_approximation()
        return super().select_proposal_steps(z)

    def calculate(self,z,jacobian=False):
        a=super().calculate(z,False)
        if not jacobian or a['jac'] is not None:return a
        z=np.asarray(z,float);normalized_modes=a['mode_jac']*self.s.radii[:self.m,None]
        model=self.cell_model;reuse=False;columns=None;age=0
        if model is not None and model['age']<self.max_reuses:
            delta=z[:self.d]-model['point'];denominator=float(delta@delta)
            predicted=model['modes'].T@delta[:self.m]+model['cells'].T@delta[self.m:]
            change=a['signal']-model['signal'];error=change-predicted
            weights=1/np.sqrt(self.obs.variance[self.obs.mask])
            relative=float(np.linalg.norm(error[self.obs.mask]*weights)/max(np.linalg.norm(change[self.obs.mask]*weights),1e-12))
            if np.max(abs(delta))<=.2 and relative<=self.mismatch_limit:
                columns=model['cells'].copy();age=model['age']+1;reuse=True
                # One Broyden update of the structural secant model; only its
                # cell columns survive because mode columns are native again.
                if denominator>1e-20:columns+=np.outer(delta[self.m:]/denominator,error)
        if not reuse:
            columns=self.cell_columns(z)
            self.cell_rebuilds+=1
        else:self.cell_reuses+=1
        self.last_reused=reuse
        derivative=np.zeros((self.d+2,len(a['rate'])))
        derivative[:self.m]=normalized_modes*np.exp(z[-2])
        derivative[self.m:self.d]=columns*np.exp(z[-2])
        derivative[-2]=a['signal']*np.exp(z[-2]);derivative[-1]=self.background*np.exp(z[-1])*self.response
        a['jac']=(derivative[:,self.obs.mask]*a['factor']).T
        self.cell_model=dict(point=z[:self.d].copy(),signal=a['signal'].copy(),modes=normalized_modes.copy(),cells=columns.copy(),age=age)
        return a

    def diagnostics(self):
        return dict(cell_reuses=self.cell_reuses,cell_rebuilds=self.cell_rebuilds,max_reuses=self.max_reuses,
            mismatch_limit=self.mismatch_limit,scope='optimization proposals only; native acceptance and export',uncertainty_derivatives_qualified=False)
