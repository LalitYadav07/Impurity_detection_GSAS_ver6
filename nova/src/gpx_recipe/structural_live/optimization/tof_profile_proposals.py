"""Owned continuous TOF proposals, separate from native likelihood authority.

The native profile compiles its own numeric table; uploaded arrays or proof
flags are never inputs. Native geometry, F², corrections, reflections and
windows remain current. Fixed-cell reuse belongs to the caller. This renderer
is only for initial proposals; accepted fits retain the original objective.
"""
from collections import Counter
import ctypes
import time
import numpy as np
from scipy.ndimage import minimum_filter, spline_filter, map_coordinates

from .native_forward import NativeTOFForward, MAXIMUM_BYTES
from .native_cell_proposals import ProposalUnavailable
from .native_tof_scalars import record_bytes
from ...live.native_peak_batch import BatchUnavailable, pack_records
from ...preparation.kernel_geometry import _profile
from ...preparation.kernel_runtime import floating_state

MINIMUM_REFERENCE_BINS=1_000_000
ROUGH_NATIVE_RELATIVE=32*np.finfo(np.float32).eps


def _immutable(array):
    array=np.ascontiguousarray(array)
    return np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape)


class TOFProfileProposals:
    def __init__(self,source,*,exact=None):
        began=time.monotonic()
        self.source=source;self.exact=exact or NativeTOFForward(source)
        context=self.exact._context();n,_,powder,_,peak,_,axis,hist,_,_=context
        if n.histdict.get(self.exact.collector.hfx+'pdabc') is not None:
            raise ProposalUnavailable('Tabulated TOF profile needs its original native calculation')
        if self.exact.native_batch is None:
            raise ProposalUnavailable('Installed continuous TOF proposal helper is unavailable')
        self.batch=self.exact.native_batch
        try:self.function=self.batch.library.proposal_tof_profile_sum
        except AttributeError as error:
            raise ProposalUnavailable('Installed continuous TOF proposal helper needs an image update') from error
        self.profile_sha256=_profile(source);self.floating=floating_state()
        phase=n.histo_phases[1][source.fast.slow.phase.name]
        refs=hist['Reflection Lists'][source.fast.slow.phase.name]['RefList']
        if not len(refs) or not np.isfinite(refs[:,4]).all() or np.any(refs[:,4]<=0):
            raise ProposalUnavailable('A complete positive native reflection metric is required')
        collector=self.exact.collector
        a=np.array([n.parms[collector.pfx+f'A{i}'] for i in range(6)])+n.math.GetDij(collector.phfx,phase['General']['SGData'],n.parms)
        g,_=n.lattice.A2Gmat(a);_,gb=n.lattice.Gmat2AB(g)
        logd=np.linspace(np.log(float(refs[:,4].min())*.7),np.log(float(refs[:,4].max())*1.3),513)
        instrument={key:list(value) if isinstance(value,list) else value
                    for key,value in hist['Instrument Parameters'][0].items()}
        for key,value in instrument.items():
            if isinstance(value,list) and len(value)>1 and collector.hfx+key in n.parms:
                value[1]=n.parms[collector.hfx+key]
        def shapes(ds):
            positions=n.lattice.Dsp2pos(instrument,ds)
            r=np.tile(refs[0],(len(ds),1)).T.copy();r[4]=ds;r[5]=positions
            r[6:8]=collector.widths(r,0,g,gb,collector.phfx,n.controls,n.parms,n.histdict,collector.hfx)
            r[12:14]=collector.exponentials(r,0,collector.hfx,n.parms)
            return np.asarray([r[i] for i in (12,13,6,7)]).T,positions
        ds=np.exp(logd);parameters,positions=shapes(ds)
        if not np.isfinite(parameters).all() or np.any(parameters[:,2]<=0):
            raise ProposalUnavailable('Native profile is not positive over this proposal envelope')
        widths=np.sqrt(parameters[:,2])
        _,low,high=powder.getWidthsTOF(positions,*parameters.T)
        eta=np.linspace(float(np.arcsinh(-low/widths).min())-.5,
                        float(np.arcsinh(high/widths).max())+.5,8193)
        if not np.isfinite(eta).all() or eta[-1]<=eta[0]:
            raise ProposalUnavailable('Invalid native TOF proposal support')
        table=np.zeros((len(ds),len(eta)));valid=np.zeros(table.shape,dtype=np.uint8)
        for i,(p,width) in enumerate(zip(parameters,widths)):
            offsets=np.sinh(eta)*width
            used=np.flatnonzero((offsets>=-low[i])&(offsets<high[i]))
            if len(used)<2:continue
            values=np.asarray(peak(0.,*p,offsets[used])[0],float)*width
            finite=np.isfinite(values);table[i,used[finite]]=values[finite];valid[i,used[finite]]=1
        if not valid.any():raise ProposalUnavailable('Native profile has no usable proposal envelope')
        for i in range(len(table)):
            good=np.flatnonzero(valid[i])
            if len(good):table[i]=np.interp(np.arange(len(eta)),good,table[i,good])
        self.coefficients=_immutable(spline_filter(table,order=3,mode='nearest'))
        # The float32 native profile can be numerically rough even in a
        # physically smooth tail (notably strongly asymmetric profiles).
        # Probe every interpolation cell at its centre using original native
        # peaks. Bad neighbourhoods retain native rendering; do not smooth
        # those native artefacts or relax the whole-curve acceptance checks.
        mid_ds=np.exp((logd[:-1]+logd[1:])/2)
        mid_parameters,mid_positions=shapes(mid_ds)
        mid_widths=np.sqrt(mid_parameters[:,2])
        _,mid_low,mid_high=powder.getWidthsTOF(mid_positions,*mid_parameters.T)
        mid_eta=(eta[:-1]+eta[1:])/2
        qualified=valid.copy();self.midpoint_checks=0;self.rejected_cells=0;self.native_only_stripes=0
        for i,(p,width) in enumerate(zip(mid_parameters,mid_widths)):
            offsets=np.sinh(mid_eta)*width
            used=np.flatnonzero((offsets>=-mid_low[i])&(offsets<mid_high[i]))
            if len(used)<2:continue
            expected=np.asarray(peak(0.,*p,offsets[used])[0],float)*width
            coords=np.asarray([np.full(len(used),i+.5),used+.5])
            predicted=map_coordinates(self.coefficients,coords,order=3,prefilter=False,mode='nearest')
            scale=max(float(np.max(abs(expected),initial=0.)),1e-30)
            rejected=(~np.isfinite(expected))|(~np.isfinite(predicted))|(abs(expected-predicted)>scale*1e-10)
            bad=used[rejected];self.midpoint_checks+=len(used);self.rejected_cells+=len(bad)
            rough=(~np.isfinite(expected))|(~np.isfinite(predicted))|(abs(expected-predicted)>scale*ROUGH_NATIVE_RELATIVE)
            if np.any(rough&(abs(np.sinh(mid_eta[used]))>=np.sinh(6.))):
                # Isolated sampling points cannot certify a numerically rough
                # d interval. Preserve its entire original native response,
                # rather than interpolating between apparently good islands.
                qualified[i:i+2,:]=0;self.native_only_stripes+=1
            for row in (i,i+1):
                qualified[row,bad]=0;qualified[row,bad+1]=0
        self.halo=_immutable(minimum_filter(qualified,size=(33,33),mode='constant',cval=0))
        self.logd=_immutable(logd);self.eta=_immutable(eta)
        self.numeric_bytes=sum(v.nbytes for v in (self.coefficients,self.halo,self.logd,self.eta))
        if self.numeric_bytes>=MAXIMUM_BYTES:
            raise ProposalUnavailable('Continuous TOF proposal table exceeds its resident bound')
        f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        f32=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
        i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        u8=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
        self.function.argtypes=[ctypes.c_void_p,ctypes.c_ssize_t,ctypes.c_ssize_t,
            f64,f64,f64,i64,f64,ctypes.c_ssize_t,f64,u8,ctypes.c_ssize_t,ctypes.c_ssize_t,
            *[ctypes.c_double]*4,f32,f32,f64,i64,f64,i64]
        self.function.restype=ctypes.c_int
        self.calls=0;self.statistics=Counter();self.maximum_workspace=0;self.seconds=0.
        self.build_seconds=time.monotonic()-began
        self._context()

    def _context(self):
        context=self.exact._context();self.batch._check()
        if _profile(self.source)!=self.profile_sha256 or floating_state()!=self.floating:
            raise ProposalUnavailable('Current profile changed; regenerate TOF proposals')
        return context

    def evaluate(self,point):
        began=time.monotonic();n,_,_,_,_,_,axis,hist,_,_=self._context()
        source=self.source;physical=source.check(np.asarray(point,float))
        records=self.exact.collector.collect_physical(physical,maximum_bytes=MAXIMUM_BYTES)
        parameters,heights,bounds,maximum,required=pack_records(records,axis,MAXIMUM_BYTES)
        refs=hist['Reflection Lists'][source.fast.slow.phase.name]['RefList']
        ds={tuple(map(int,row[:3])):float(row[4]) for row in refs}
        d=np.asarray([ds[key] for key in records],float)
        footprint=(self.numeric_bytes+self.exact.fixed_workspace+record_bytes(records)+
                   required+16*maximum+d.nbytes)
        if footprint>MAXIMUM_BYTES:
            raise ProposalUnavailable('Continuous TOF proposals exceed the bounded forward workspace')
        scratch=[np.empty(maximum,np.float32),np.empty(maximum,np.float32),
                 np.empty(maximum),np.empty(maximum,np.int64)]
        signal=np.zeros(len(axis));statistics=np.zeros(4,np.int64)
        code=self.function(self.batch.address,len(heights),len(axis),axis,parameters,heights,bounds,d,
            maximum,self.coefficients,self.halo,*self.coefficients.shape,
            self.logd[0],self.logd[1]-self.logd[0],self.eta[0],self.eta[1]-self.eta[0],
            *scratch,signal,statistics)
        self._context()
        if code or not np.isfinite(signal).all():
            raise ProposalUnavailable('Continuous TOF proposal rendering was refused')
        hfx=self.exact.collector.hfx
        background,_=self.exact.background_function(hfx,n.parms,n.controls[hfx+'bakType'],
            n.controls[hfx+'histType'],axis,hist['Background'][1].get('fixback'))
        curve=np.zeros(len(source.fast.x));curve[source.fast.slice]=signal+background
        if not np.isfinite(curve).all():raise ValueError('Nonfinite native proposal background')
        n._check();self.calls+=1;self.seconds+=time.monotonic()-began
        self.maximum_workspace=max(self.maximum_workspace,footprint)
        self.statistics.update(dict(zip(['native_bins','interpolated_bins','native_fallback_rows',
            'unsupported_native_bins'],map(int,statistics))))
        return curve

    def diagnostics(self):
        return dict(method='Native-compiled continuous TOF starting proposals',
            calls=self.calls,seconds=self.seconds,build_seconds=self.build_seconds,
            statistics=dict(self.statistics),resident_numeric_bytes=self.numeric_bytes,
            maximum_accounted_workspace_bytes=self.maximum_workspace,
            profile_sha256=self.profile_sha256,interpolation=True,
            native_midpoint_checks=self.midpoint_checks,rejected_interpolation_cells=self.rejected_cells,
            native_only_d_stripes=self.native_only_stripes,
            roughness_relative_to_native_precision=ROUGH_NATIVE_RELATIVE,
            native_origin_authority=False,observations_used_for_compilation=False,
            table_import_supported=False,original_native_objective_required=True)


def for_source(source,*,exact=None):
    """Avoid table setup for workloads below the measured large-profile case."""
    exact=exact or NativeTOFForward(source)
    records=exact.collector.collect_physical(np.zeros(len(source.radii)),maximum_bytes=MAXIMUM_BYTES)
    if sum(high-low for _,_,low,high in records.values())<MINIMUM_REFERENCE_BINS:
        return None
    result=TOFProfileProposals(source,exact=exact)
    if not np.any(result.halo):
        raise ProposalUnavailable('Current native profile needs original peaks throughout the proposal envelope')
    return result
