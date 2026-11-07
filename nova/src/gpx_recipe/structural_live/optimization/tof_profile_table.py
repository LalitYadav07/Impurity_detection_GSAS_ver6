"""Compile a native TOF shape without a cell, candidate model or scan grid.

The numeric table supplies initial proposals only. Compilation and midpoint
checks do not grant native likelihood authority or certify a global error
bound. Unsupported cells of the table retain the original native peak kernel.
"""
import time
import os
import threading
import copy
import numpy as np
from scipy.ndimage import minimum_filter,spline_filter,map_coordinates

from .native_cell_proposals import ProposalUnavailable,_hash,KERNELS
from .native_tof_scalars import WIDTH_SOURCE
from .tof_shape import IsotropicTOFShape
from ...preparation.kernel_runtime import floating_state
from ...live.native_peak_batch import DEFAULT_FLOATING

MAXIMUM_BYTES=64*1024*1024
ROUGH_NATIVE_RELATIVE=32*np.finfo(np.float32).eps
_shared=None
_shared_pid=os.getpid()
_shared_lock=threading.Lock()


def _reset_fork():
    global _shared,_shared_pid,_shared_lock
    if _shared_pid!=os.getpid():
        _shared=None;_shared_pid=os.getpid();_shared_lock=threading.Lock()


def _remember(table):
    global _shared
    _reset_fork();table.check_runtime()
    if table.numeric_bytes>=MAXIMUM_BYTES:raise ValueError('Shared TOF table exceeds its resident bound')
    with _shared_lock:_shared=table


def shared_table(shape,powder,d_range):
    """One bounded owned table, automatically shared across native sources.

    Outside its d envelope the renderer uses original native peaks. A wholly
    disjoint request compiles a new table; partially covered requests preserve
    the useful prepared component without excluding their other reflections.
    No uploaded data or saved qualification flags enter this cache.
    """
    global _shared
    _reset_fork();bounds=_bounds(d_range,'A finite positive d-spacing envelope is required')
    with _shared_lock:table=_shared
    if table is not None:
        try:
            table.check_runtime()
            if (table.shape_identity==shape.identity and table._powder is powder
                    and bounds[0]<=np.exp(table.logd[-1]) and bounds[1]>=np.exp(table.logd[0])):
                return table,True
        except ProposalUnavailable:
            with _shared_lock:
                if _shared is table:_shared=None
    # Compilation is intentionally outside the registry lock. The native
    # worker retains its separate single-context process/thread ownership.
    table=CompiledTOFTable(shape,powder,bounds);_remember(table);return table,False


def _immutable(array):
    array=np.ascontiguousarray(array)
    return np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape)


def _bounds(value,message):
    try:
        if (not isinstance(value,(list,tuple,np.ndarray)) or len(value)!=2
                or isinstance(value,np.ndarray) and value.shape!=(2,)):
            raise ValueError(message)
        if any(isinstance(v,(bool,np.bool_))
                or not isinstance(v,(int,float,np.integer,np.floating)) for v in value):
            raise ValueError(message)
        bounds=np.asarray(value,float)
        if not np.isfinite(bounds).all() or not 0<bounds[0]<bounds[1]:raise ValueError(message)
        return bounds
    except (TypeError,ValueError,OverflowError):raise ValueError(message) from None


class CompiledTOFTable:
    @classmethod
    def from_reference(cls,native,histogram_name,phase_name,*,tof_range):
        """Compile from a planned TOF interval; no diffraction arrays are read."""
        native._check();bounds=_bounds(tof_range,'A finite positive planned TOF interval is required')
        histogram=native.histo_phases[0][histogram_name];hfx=f":{histogram['hId']}:"
        instrument={k:list(v) if isinstance(v,list) else v
                    for k,v in histogram['Instrument Parameters'][0].items()}
        for key,value in instrument.items():
            if isinstance(value,list) and len(value)>1 and hfx+key in native.parms:
                value[1]=native.parms[hfx+key]
        ds=np.asarray(native.lattice.Pos2dsp(instrument,bounds),float)
        if not np.isfinite(ds).all() or np.any(ds<=0) or ds[0]==ds[1]:
            raise ProposalUnavailable('Planned TOF interval has no finite positive native d envelope')
        shape=IsotropicTOFShape(native,histogram_name,phase_name)
        result=cls(shape,native.math.G2pwd,(float(ds.min()),float(ds.max())))
        native._check();_remember(result);return result

    def __init__(self,shape,powder,d_range):
        began=time.monotonic()
        self._pid=os.getpid();self._thread=threading.current_thread();self._floating=floating_state()
        if self._floating!=DEFAULT_FLOATING:
            raise ProposalUnavailable('Native TOF table requires reviewed arithmetic controls')
        if not isinstance(shape,IsotropicTOFShape):raise TypeError('An installed native isotropic shape is required')
        bounds=_bounds(d_range,'A finite positive d-spacing envelope is required')
        peak=powder.getEpsVoigt;width_function=powder.getWidthsTOF;compiled=powder.pyd.pyepsvoigt
        from ...live.native_peak_memo import _compiled_tof_float32
        if (_hash(peak)!=KERNELS['PNT'][2] or _hash(width_function)!=WIDTH_SOURCE
                or not _compiled_tof_float32(powder.pyd)):
            raise ProposalUnavailable('Native TOF table functions are not qualified')
        self._shape_specification=shape.specification;self.shape_identity=shape.identity
        self._shape=shape;self._powder=powder;self._functions=(peak,width_function,compiled)
        logd=np.linspace(np.log(bounds[0]*.7),np.log(bounds[1]*1.3),513)
        ds=np.exp(logd);parameters,low,high=shape.support(ds)
        if not np.isfinite(parameters).all() or np.any(parameters[:,2]<=0):
            raise ProposalUnavailable('Native profile is not positive over this proposal envelope')
        widths=np.sqrt(parameters[:,2])
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
        coefficients=spline_filter(table,order=3,mode='nearest')
        mid_ds=np.exp((logd[:-1]+logd[1:])/2)
        mid_parameters,mid_low,mid_high=shape.support(mid_ds)
        mid_widths=np.sqrt(mid_parameters[:,2]);mid_eta=(eta[:-1]+eta[1:])/2
        qualified=valid.copy();self.midpoint_checks=0;self.rejected_cells=0;self.native_only_stripes=0
        for i,(p,width) in enumerate(zip(mid_parameters,mid_widths)):
            offsets=np.sinh(mid_eta)*width
            used=np.flatnonzero((offsets>=-mid_low[i])&(offsets<mid_high[i]))
            if len(used)<2:continue
            expected=np.asarray(peak(0.,*p,offsets[used])[0],float)*width
            coords=np.asarray([np.full(len(used),i+.5),used+.5])
            predicted=map_coordinates(coefficients,coords,order=3,prefilter=False,mode='nearest')
            scale=max(float(np.max(abs(expected),initial=0.)),1e-30)
            rejected=(~np.isfinite(expected))|(~np.isfinite(predicted))|(abs(expected-predicted)>scale*1e-10)
            bad=used[rejected];self.midpoint_checks+=len(used);self.rejected_cells+=len(bad)
            rough=(~np.isfinite(expected))|(~np.isfinite(predicted))|(abs(expected-predicted)>scale*ROUGH_NATIVE_RELATIVE)
            if np.any(rough&(abs(np.sinh(mid_eta[used]))>=np.sinh(6.))):
                qualified[i:i+2,:]=0;self.native_only_stripes+=1
            for row in (i,i+1):
                qualified[row,bad]=0;qualified[row,bad+1]=0
        shape._check()
        if (powder.getEpsVoigt is not peak or powder.getWidthsTOF is not width_function
                or powder.pyd.pyepsvoigt is not compiled):
            raise ProposalUnavailable('Native TOF table functions changed during compilation')
        self.coefficients=_immutable(coefficients)
        self.halo=_immutable(minimum_filter(qualified,size=(33,33),mode='constant',cval=0))
        self.logd=_immutable(logd);self.eta=_immutable(eta)
        self.numeric_bytes=sum(v.nbytes for v in (self.coefficients,self.halo,self.logd,self.eta))
        if self.numeric_bytes>=MAXIMUM_BYTES:
            raise ProposalUnavailable('Continuous TOF proposal table exceeds its resident bound')
        self.build_seconds=time.monotonic()-began
        self.check_runtime()

    def check_runtime(self):
        """Owned native construction only; a saved array cannot assert this."""
        if (os.getpid()!=self._pid or threading.current_thread() is not self._thread
                or floating_state()!=self._floating):
            raise ProposalUnavailable('Native TOF table ownership or arithmetic controls changed')
        self._shape._check()
        if (self._powder.getEpsVoigt,self._powder.getWidthsTOF,
                self._powder.pyd.pyepsvoigt)!=self._functions:
            raise ProposalUnavailable('Native TOF table implementation changed')

    def arrays(self):
        """Immutable numeric data only; these bytes carry no origin authority."""
        return dict(coefficients=self.coefficients,halo=self.halo,logd=self.logd,eta=self.eta)

    @property
    def shape_specification(self):return copy.deepcopy(self._shape_specification)
