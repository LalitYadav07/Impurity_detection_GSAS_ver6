"""Short-lived secant proposals for cell columns; every score stays native.

Native displacement derivatives are recomputed at every iterate. Cell
columns are periodically rebuilt with feasible native finite differences.
No approximate derivative is used to report a fitted curve or uncertainty.
"""
import numpy as np
from .constrained_native_fit import FeasibleNativeObjective
from .cell_derivative_model import CellDerivativeModel


class RecycledCellObjective(FeasibleNativeObjective):
    def __init__(self,*args,max_reuses=3,mismatch_limit=.25,**kwargs):
        super().__init__(*args,**kwargs)
        self.max_reuses=max_reuses;self.mismatch_limit=mismatch_limit
        self.cell_model=CellDerivativeModel(self.m,self.d,max_reuses=max_reuses,mismatch_limit=mismatch_limit)
        self.cell_reuses=0;self.cell_rebuilds=0;self.last_reused=False

    def invalidate_cell_approximation(self):
        self.cell_model.invalidate()
        if self.last is not None:self.last['jac']=None

    def select_proposal_steps(self,z):
        self.invalidate_cell_approximation()
        return super().select_proposal_steps(z)

    def calculate(self,z,jacobian=False):
        a=super().calculate(z,False)
        if not jacobian or a['jac'] is not None:return a
        z=np.asarray(z,float);normalized_modes=a['mode_jac']*self.s.radii[:self.m,None]
        self.cell_model.observe_modes(z[:self.d],a['signal'],normalized_modes)
        weights=np.zeros(len(a['signal']));weights[self.obs.mask]=1/np.sqrt(self.obs.variance[self.obs.mask])
        columns=self.cell_model.columns(z[:self.d],a['signal'],weights,lambda:self.cell_columns(z))
        self.cell_reuses=self.cell_model.reuses;self.cell_rebuilds=self.cell_model.builds
        self.last_reused=self.cell_model.last_reused
        derivative=np.zeros((self.d+2,len(a['rate'])))
        derivative[:self.m]=normalized_modes*np.exp(z[-2])
        derivative[self.m:self.d]=columns*np.exp(z[-2])
        derivative[-2]=a['signal']*np.exp(z[-2]);derivative[-1]=self.background*np.exp(z[-1])*self.response
        a['jac']=(derivative[:,self.obs.mask]*a['factor']).T
        return a

    def diagnostics(self):
        return dict(cell_reuses=self.cell_reuses,cell_rebuilds=self.cell_rebuilds,max_reuses=self.max_reuses,
            mismatch_limit=self.mismatch_limit,proposal_model=self.cell_model.diagnostics(),
            scope='optimization proposals only; native acceptance and export',uncertainty_derivatives_qualified=False)
