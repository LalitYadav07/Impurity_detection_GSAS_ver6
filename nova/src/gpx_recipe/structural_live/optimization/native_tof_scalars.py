"""Collect native TOF peak arguments without rendering a powder pattern.

Only original GSAS-II functions calculate scattering, positions, corrections
and profile parameters. Compatible scalar operations are batched; native
position and intensity corrections retain their per-reflection calls. These
records supply proposal directions or the separately guarded native TOF start
renderer. They are not predictions until the original native peak kernels and
background have rendered them. Each caller checks an original full native pass.
"""
import hashlib
import inspect
import types
import numpy as np
from ...live.native_peak_records import PackedPeakRecords


class ScalarBatchUnavailable(ValueError):
    """Use the original scalar/native path for this unsupported batch."""


FUNCTION_SOURCES={
    'GetReflPos':'acfd687d4b9991ebc8b7622939089cbe933c61d9c8d08a04973270dd39040afa',
    'GetHStrainShift':'4063256718111cb3037058b01042ac6ec760e268e9d08f03e877ee934cd0bb88',
    'GetSampleSigGam':'b9436e2acfcd03f79cf1840e2ba8d415c4c754fe9428db2482d2cf9e407e2ff9',
    'GetIntensityCorr':'605dcf95a29bf5fe5f0147b8dc753566c38285bcc45ac937dfdb660c6a188efb',
    'GetAbsorb':'0419ccfc877eca85248da49cdcc3b9aa9268f921b0282bc1a9329e36b9f340d1',
    'StructureFactor2':'adac1ed638a4a062ff606676b2aeb4e43522b7ffa4b44bb388333f49c3672309',
}
WIDTH_SOURCE='de20bc0b7f1dc5b4b97d8ce27302cf2fdc3e09c59f2e4a821a1e01ffe136101d'


def _hash(function):
    try:return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
    except (OSError,TypeError):return None


def _nested(driver,name,bindings):
    codes=[c for c in driver.__code__.co_consts if isinstance(c,types.CodeType) and c.co_name==name]
    if len(codes)!=1 or not set(codes[0].co_freevars)<=set(bindings):
        raise ScalarBatchUnavailable('Unrecognized native scalar closure')
    def cell(value):return (lambda:value).__closure__[0]
    closure=tuple(cell(bindings[k]) for k in codes[0].co_freevars)
    return types.FunctionType(codes[0],driver.__globals__,name,closure=closure or None)


def record_bytes(records):
    # Include owned arrays plus a conservative allowance for Python keys,
    # tuples, floats, integers and the dictionary's sparse storage.
    if isinstance(records,PackedPeakRecords):return records.numeric_bytes+4096
    return sum(row[0].nbytes+1024 for row in records.values())


class NativeTOFScalars:
    def __init__(self,source,driver):
        self.source=source;self.native=source.fast.native;self.driver=driver
        n=self.native;h=n.histo_phases[0][source.fast.slow.hist.name]
        self.phase=n.histo_phases[1][source.fast.slow.phase.name]
        self.hfx=f":{h['hId']}:";self.pfx=f"{self.phase['pId']}::";self.phfx=f"{self.phase['pId']}:{h['hId']}:"
        self.widths=_nested(driver,'GetReflSigGamTOF',{'SGData':self.phase['General']['SGData']})
        self.exponentials=_nested(driver,'GetReflAlpBet',{})
        self.dependencies=[(n.math,name,getattr(n.math,name)) for name in FUNCTION_SOURCES]
        self.dependencies.extend([(n.math.G2pwd,'getWidthsTOF',n.math.G2pwd.getWidthsTOF),
                                  (n.math,'getPowderProfile',driver)])
        self.calls=0;self.maximum_workspace_bytes=0;self.last_workspace_bytes=0
        self.geometry_calls=0

    def collect(self,point,*,maximum_bytes):
        return self.collect_physical(np.asarray(point,float)*self.source.radii,
                                     maximum_bytes=maximum_bytes)

    def collect_physical(self,point,*,maximum_bytes):
        """Retain physical coordinate bits without a normalization round trip."""
        return self._collect(point,maximum_bytes=maximum_bytes,geometry_only=False)

    def collect_packed_physical(self,point,*,maximum_bytes):
        """Same native scalars/order, without per-reflection Python objects."""
        return self._collect(point,maximum_bytes=maximum_bytes,geometry_only=False,packed=True)

    def collect_kernel_geometry(self,point,*,maximum_bytes):
        """Native shape/window arguments only, with no atomic feasibility.

        The complete bounded point supplies its cell coordinates; displacement
        coordinates do not affect these isotropic instrument kernels. Returned
        records deliberately contain no intensity or structure-factor values.
        """
        return self._collect(point,maximum_bytes=maximum_bytes,geometry_only=True)

    def _collect(self,point,*,maximum_bytes,geometry_only,packed=False):
        if type(maximum_bytes)is not int or maximum_bytes<1:
            raise ScalarBatchUnavailable('No remaining scalar collection workspace')
        n=self.native;s=self.source;n._check()
        if any(getattr(module,name,None) is not function for module,name,function in self.dependencies):
            raise ScalarBatchUnavailable('Native scalar functions changed during this proposal')
        if geometry_only:
            p=s.check_domain(np.asarray(point,float))
            theta=np.r_[s.lift@p,0.,0.];theta[:s.fast.nm]=0.
            _,values=s.fast._peak_geometry_values(theta)
        else:
            p=s.check(np.asarray(point,float))
            _,values=s.fast._values(s.theta(p),chemistry_domain=(s.chemistry,p))
        n._prepare_values(values)
        h=n.histo_phases[0][s.fast.slow.hist.name];phase=n.histo_phases[1][s.fast.slow.phase.name]
        controls=n.controls;parms=n.parms;sg=phase['General']['SGData']
        refs=h['Reflection Lists'][s.fast.slow.phase.name]
        rows=refs['RefList']
        # The native engine owns its reflection array. Bound the additional
        # scalar arrays and worst-case returned records before creating them.
        workspace=len(rows)*((96 if packed else 1024)+8*64)+(4096 if packed else 0)
        if rows.ndim!=2 or not len(rows) or rows.shape[1]<18 or workspace>maximum_bytes:
            raise ScalarBatchUnavailable('Native scalar collection exceeds its workspace budget')
        self.maximum_workspace_bytes=max(self.maximum_workspace_bytes,workspace)
        self.last_workspace_bytes=workspace
        a=np.array([parms[self.pfx+f'A{i}'] for i in range(6)])+n.math.GetDij(self.phfx,sg,parms)
        reciprocal,direct=n.lattice.A2Gmat(a);_,gb=n.lattice.Gmat2AB(reciprocal)
        if not geometry_only:
            n.math.StructureFactor2(refs,reciprocal,self.hfx,self.pfx,sg,controls,parms)
        r=rows.T
        # GetReflPos's native calc_rDsq sums all axes and is scalar-only.
        r[5]=[n.math.GetReflPos(r[:,i],0,0.,a,self.pfx,self.hfx,self.phfx,controls,parms)
              for i in range(r.shape[1])]
        r[5]+=n.math.GetHStrainShift(r,0,sg,self.phfx,self.hfx,controls,parms)
        r[6:8]=self.widths(r,0,reciprocal,gb,self.phfx,controls,parms,n.histdict,self.hfx)
        r[12:14]=self.exponentials(r,0,self.hfx,parms)
        if not geometry_only:
            operators=np.array([op[0].T for op in sg['SGOps']]);unique=np.inner(r[:3].T,operators)
            # Absorption couples a scalar detector angle and wavelength. Keep
            # the entire original intensity-correction function scalar too.
            correction=np.asarray([n.math.GetIntensityCorr(r[:,i],0,unique[i],reciprocal,direct,
                self.pfx,self.phfx,self.hfx,sg,controls,parms) for i in range(r.shape[1])]).T
            r[11],r[15],r[16],r[17]=correction
            volume=np.sqrt(np.linalg.det(reciprocal))
            angle=n.math.sind(abs(parms[self.hfx+'2-theta'])/2)
            # The original driver raises each np.float64 d spacing to the
            # fourth power in its scalar reflection loop. NumPy's array
            # power may select a different SIMD implementation on another
            # CPU, changing a final intensity bit and disabling exact native
            # reuse. Preserve the original scalar operation order here;
            # geometry, widths and structure factors remain batched.
            for i in range(r.shape[1]):
                lorenz=angle*r[4,i]**4
                r[11,i]*=volume*lorenz
        _,low,high=n.math.G2pwd.getWidthsTOF(r[5],r[12],r[13],r[6],r[7])
        axis=s.fast.x[s.fast.slice]
        begin=np.searchsorted(axis,r[5]-low);end=np.searchsorted(axis,r[5]+high)
        stops=np.flatnonzero((begin==end)&((begin+end)!=0))
        stop=int(stops[0]) if len(stops) else len(begin)
        active=np.flatnonzero((np.arange(len(begin))<stop)&(begin<end))
        parameters=np.column_stack([r[i] for i in (5,12,13,6,7)])
        if not np.isfinite(parameters).all():
            raise ScalarBatchUnavailable('Nonfinite batched native peak arguments')
        if geometry_only:
            records={tuple(map(int,r[:3,i])):(parameters[i].copy(),int(begin[i]),int(end[i]))
                     for i in active}
            self.geometry_calls+=1
        else:
            height=r[9]*r[11]
            if not np.isfinite(height).all():
                raise ScalarBatchUnavailable('Nonfinite batched native peak arguments')
            if packed:
                # Fancy indexing creates owned buffers. No native reflection
                # or current structure-factor array is exposed or retained.
                if not np.isfinite(r[:3]).all() or np.any(abs(r[:3])>1e9):
                    raise ScalarBatchUnavailable('Packed native reflection indices exceed their integer domain')
                try:
                    records=PackedPeakRecords(np.ascontiguousarray(parameters[active]),height[active].copy(),
                        np.column_stack((begin[active],end[active])).astype(np.int64,copy=False),
                        r[4,active].copy(),np.array(r[:3,active].T,dtype=np.int64,order='C',copy=True))
                except ValueError as error:
                    raise ScalarBatchUnavailable('Packed native records require the original path: '+str(error)) from error
            else:
                records={tuple(map(int,r[:3,i])):(parameters[i].copy(),float(height[i]),int(begin[i]),int(end[i]))
                         for i in active}
        if len(records)!=len(active):raise ScalarBatchUnavailable('Duplicate native reflection identity')
        self.calls+=1
        return records


def for_source(source,driver):
    """Automatic qualified TOF shortcut; unsupported contexts keep old work."""
    n=source.fast.native;n._check()
    h=n.histo_phases[0][source.fast.slow.hist.name];phase=n.histo_phases[1][source.fast.slow.phase.name]
    hfx=f":{h['hId']}:";phfx=f"{phase['pId']}:{h['hId']}:"
    if (n.controls[hfx+'histType']!='PNT' or not callable(getattr(n,'_prepare_values',None))
            or n.controls[phfx+'SizeType']!='isotropic' or n.controls[phfx+'MustrainType']!='isotropic'
            or driver.__globals__ is not n.math.__dict__):
        return None
    if (any(_hash(getattr(n.math,name,None))!=value for name,value in FUNCTION_SOURCES.items())
            or _hash(n.math.G2pwd.getWidthsTOF)!=WIDTH_SOURCE):
        return None
    try:return NativeTOFScalars(source,driver)
    except ScalarBatchUnavailable:return None
