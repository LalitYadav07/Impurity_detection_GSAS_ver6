"""Candidate-local native F² derivatives through native mode constraints.

Experimental until compared against full-profile derivatives and independent
finite differences. No atom-orbit scattering or profile formula is duplicated.
"""
import numpy as np
from .native_cell_intensity import ExactCellIntensity


class NativeCellDerivative(ExactCellIntensity):
    def __init__(self,source,max_cells=8,force_native_derivatives=False):
        super().__init__(source,max_cells)
        self.force_native_derivatives=bool(force_native_derivatives)
        self.native_derivative_fallback_calls=0
        self.derivative_function=source.fast.native.math.StructureFactorDerv2
        self.derivative_calls=0

    def mode_jacobian(self,point):
        if self.force_native_derivatives:
            self.native_derivative_fallback_calls+=1
            return self.source.native_mode_jacobian(point)
        s=self.source;p=s.check(point);m=s.mode_dimensions;y=self.evaluate(p)
        if not m:return y,np.empty((0,len(y)))
        entry=self._entry(p[m:])
        if entry['fallback']:return super().mode_jacobian(p)
        n=s.fast.native;bank=entry['bank']
        if n.math.StructureFactorDerv2 is not self.derivative_function:
            raise ValueError('Native derivative implementation changed')
        _,values=s.fast._values(s.theta(p),chemistry_domain=(s.chemistry,p))
        parms=entry['parms'].copy();parms.update(zip(n.vary,values));n.maps.Dict2Map(parms)
        derivatives=self.derivative_function(entry['ref'],entry['g'],bank.hfx,bank.pfx,entry['sg'],n.controls,parms)
        self.derivative_calls+=1
        indices=np.flatnonzero(np.any(s.fast.lift[:,:s.fast.nm]!=0,axis=1))
        names=[n.vary[i] for i in indices]
        df=np.zeros((len(names),len(bank.hkls)))
        for i,name in enumerate(names):
            if name in derivatives:df[i]=derivatives[name]
        n.maps.Dict2Deriv(names,derivatives,df)
        lift=s.fast.lift[indices,:s.fast.nm]@s.lift[:s.fast.nm,:m]
        jac=np.zeros((m,len(y)));jac[:,s.fast.slice]=(entry['matrix']@(lift.T@df).T).T
        if not np.isfinite(jac).all():raise ValueError('Nonfinite native mode derivative')
        return y,jac

    def diagnostics(self):
        return dict(super().diagnostics(),derivative_calls=self.derivative_calls,
            force_native_derivatives=self.force_native_derivatives,native_derivative_fallback_calls=self.native_derivative_fallback_calls,
            derivative='GSASIIstrMath.StructureFactorDerv2 + GSASIImapvars.Dict2Deriv')
