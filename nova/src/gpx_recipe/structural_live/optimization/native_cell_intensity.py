"""Exact GSAS-II intensities with native profile reuse at an exact fixed cell.

Unlike a fixed-HKL surrogate, each cell owns the reflection inventory that
GSAS-II generates there. No reflection list is extrapolated across a boundary.
StructureFactor2 is the same native function used by getPowderProfile.
"""
from collections import OrderedDict
from types import SimpleNamespace
import copy,time
import numpy as np
from .profile_cache import NativeProfileBank,MAXIMUM_RESPONSE_COORDINATES


def native_scattering_entry(source,bank,matrix):
    """Bind exact-cell native amplitudes to an already-checked response."""
    n=source.fast.native
    hist=n.histo_phases[0][bank.histogram_name];phase=n.histo_phases[1][bank.phase_name]
    if n.parms.get(bank.pfx+'isMag') or phase['General'].get('Modulated'):
        raise ValueError('Nuclear commensurate StructureFactor2 path required')
    ref=copy.deepcopy(hist['Reflection Lists'][bank.phase_name])
    order=np.lexsort((ref['RefList'][:,2],ref['RefList'][:,1],ref['RefList'][:,0]))
    ref['RefList']=ref['RefList'][order].copy()
    if not np.array_equal(ref['RefList'][:,:3],bank.hkls):raise ValueError('Native response/HKL alignment changed')
    parms=n.parms.copy();sg=phase['General']['SGData']
    a=np.array([parms[bank.pfx+f'A{i}'] for i in range(6)])+n.math.GetDij(bank.phfx,sg,parms)
    g,_=n.lattice.A2Gmat(a)
    return dict(bank=bank,matrix=matrix,ref=ref,parms=parms,sg=sg,g=g,fallback=False)


class ExactCellIntensity:
    def __init__(self,source,max_cells=8):
        if type(max_cells)is not int or not 1<=max_cells<=64:raise ValueError('Bounded exact-cell cache required')
        self.source=source;self.cache=OrderedDict();self.max_cells=max_cells
        self.builds=0;self.hits=0;self.structure_factor_calls=0;self.native_fallback_calls=0
        self.build_seconds=0.;self.maximum_check_error=0.;self.inventory_sizes=set()
        self.function=source.fast.native.math.StructureFactor2

    def _entry(self,cells):
        s=self.source;m=s.mode_dimensions;cells=np.asarray(cells,float)
        # The response-cache dimension ceiling is an optimization limit,
        # not an admission limit on the complete structural model.
        if len(s.radii)>MAXIMUM_RESPONSE_COORDINATES:
            return dict(fallback=True,reason='response_cache_dimension_limit')
        point=np.r_[np.zeros(m),cells]
        # A valid distorted structure need not have a chemically valid
        # zero-distortion reference at the same cell. Use the full native
        # engine for that cell instead of rejecting the valid structure.
        from .empirical_catalog import EmpiricalConfigurationRejected
        try:s.check(point)
        except EmpiricalConfigurationRejected:return dict(fallback=True,reason='infeasible_zero_mode_reference')
        s.fast.native._check()
        if s.fast.native.math.StructureFactor2 is not self.function:raise ValueError('Native scattering implementation changed')
        key=tuple(cells)
        if key in self.cache:
            item=self.cache[key];item['bank'].check();self.cache.move_to_end(key);self.hits+=1;return item
        began=time.perf_counter()
        def check(p):
            p=np.asarray(p,float)
            if p.shape!=s.radii.shape or np.any(p[m:]!=0.):raise ValueError('This response belongs to one exact cell')
            return s.check_domain(p+point)
        shifted=SimpleNamespace(radii=s.radii,cell_dimensions=s.cell_dimensions,fast=s.fast,
            check_domain=check,snapshot=lambda p:s.snapshot(check(p)))
        bank=NativeProfileBank(shifted)
        if not bank.linear:return dict(bank=bank,fallback=True)
        matrix=bank.matrix(np.zeros(s.cell_dimensions))
        item=native_scattering_entry(s,bank,matrix)
        self.cache[key]=item;self.builds+=1;self.inventory_sizes.add(len(bank.hkls))
        while len(self.cache)>self.max_cells:self.cache.popitem(last=False)
        self.build_seconds+=time.perf_counter()-began
        return item

    def evaluate(self,point):
        s=self.source;p=s.check(point);m=s.mode_dimensions;entry=self._entry(p[m:])
        if entry['fallback']:
            self.native_fallback_calls+=1;return s.native(p)
        bank=entry['bank'];n=s.fast.native
        _,values=s.fast._values(s.theta(p),chemistry_domain=(s.chemistry,p))
        parms=entry['parms'].copy();parms.update(zip(n.vary,values));n.maps.Dict2Map(parms)
        self.function(entry['ref'],entry['g'],bank.hfx,bank.pfx,entry['sg'],n.controls,parms)
        self.structure_factor_calls+=1
        f2=entry['ref']['RefList'][:,9]
        if not np.isfinite(f2).all() or np.any(f2<0):raise ValueError('Invalid native structural intensities')
        y=bank.background.copy();y[s.fast.slice]+=entry['matrix']@f2
        return y

    def mode_jacobian(self,point):
        s=self.source;p=s.check(point);y=self.evaluate(p);jac=[]
        if not s.chemistry.certified_full_support:
            # Native analytic derivatives require no chemically invalid
            # neighboring structures at an active contact boundary.
            return s.native_mode_jacobian(p)
        # Numerical derivatives of native F² reuse the SAME native response.
        # This avoids assumptions about inter-orbit interference or a new
        # scattering implementation. The step is independently validated.
        for i in range(s.mode_dimensions):
            lo=p.copy();hi=p.copy();h=s.radii[i]*1e-4
            lo[i]=max(-s.radii[i],lo[i]-h);hi[i]=min(s.radii[i],hi[i]+h)
            jac.append((self.evaluate(hi)-self.evaluate(lo))/(hi[i]-lo[i]))
        return y,np.asarray(jac).reshape(s.mode_dimensions, len(y))

    def verify(self,point,tolerance=1e-9):
        fast=self.evaluate(point);native=self.source.native(point)
        error=float(np.linalg.norm(fast-native)/max(np.linalg.norm(native),1e-30))
        self.maximum_check_error=max(self.maximum_check_error,error)
        if error>tolerance:raise ValueError(f'Exact-cell native scattering/profile decomposition disagrees: {error}')
        return error

    def diagnostics(self):
        memo=getattr(getattr(getattr(self.source,'fast',None),'native',None),'peak_memo',None)
        axis=getattr(getattr(getattr(self.source,'fast',None),'native',None),'axis_view',None)
        return dict(builds=self.builds,hits=self.hits,entries=len(self.cache),max_cells=self.max_cells,
            build_seconds=self.build_seconds,structure_factor_calls=self.structure_factor_calls,
            native_fallback_calls=self.native_fallback_calls,inventory_sizes=sorted(self.inventory_sizes),
            maximum_check_error=self.maximum_check_error,
            scattering='GSASIIstrMath.StructureFactor2',profile='GSASIIstrMath.getPowderProfile',
            intensity_interpolation=False,custom_scattering_or_profile=False,
            exact_native_peak_reuse=memo.diagnostics() if memo is not None else None,
            native_axis_view=axis.diagnostics() if axis is not None else None)
