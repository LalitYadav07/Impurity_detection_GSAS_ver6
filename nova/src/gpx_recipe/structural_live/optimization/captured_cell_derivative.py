"""Native structural derivatives reusing an already-needed Rietveld pass.

No portable numerical assertion is trusted here. Each response is captured
from this owned native process at the exact current cell and fixed context.
Every original full-native acceptance/export remains available independently.
"""
from collections import Counter,OrderedDict
import time
import numpy as np

from .native_cell_derivative import NativeCellDerivative
from .native_cell_intensity import native_scattering_entry
from .native_profile_capture import CaptureUnavailable,capture_rietveld_response
from .profile_cache import NativeProfileBank,MAXIMUM_RESPONSE_COORDINATES


def _numeric_bytes(value):
    if isinstance(value,np.ndarray):return value.nbytes
    if isinstance(value,dict):return sum(_numeric_bytes(v) for v in value.values())
    if isinstance(value,(list,tuple)):return sum(_numeric_bytes(v) for v in value)
    return 0


class CapturedCellDerivative(NativeCellDerivative):
    def __init__(self,source,max_cells=2,maximum_bytes=64*1024*1024):
        super().__init__(source,max_cells=max_cells)
        if type(maximum_bytes)is not int or maximum_bytes<1:raise ValueError('Positive native capture byte budget required')
        self.maximum_bytes=maximum_bytes;self.numeric_bytes=0
        self.capture_attempts=0;self.captured_native_calls=0;self.direct_native_calls=0
        self.capture_fallbacks=Counter();self.maximum_capture_error=0.;self.capture_seconds=0.
        self.unavailable=OrderedDict()
        from .native_forward import for_source
        self.native_forward=(for_source(source)
            if len(source.radii)<=MAXIMUM_RESPONSE_COORDINATES else None)
        self.kernel_capture_fallbacks=Counter();self.kernel_native_evaluations=0

    def has_response(self,point):
        p=self.source.check(point)
        return not self._entry(p[self.source.mode_dimensions:])['fallback']

    def needs_capture(self,point):
        p=self.source.check(point);cells=p[self.source.mode_dimensions:]
        entry=self._entry(cells)
        return (tuple(cells) not in self.unavailable and entry['fallback']
            and entry.get('reason')!='response_cache_dimension_limit')

    def _unavailable(self,key,reason):
        self.capture_fallbacks[reason]+=1;self.unavailable[key]=reason;self.unavailable.move_to_end(key)
        while len(self.unavailable)>16:self.unavailable.popitem(last=False)

    def _entry(self,cells):
        self.source.fast.native._check()
        if self.source.fast.native.math.StructureFactor2 is not self.function:
            raise ValueError('Native scattering implementation changed')
        if len(self.source.radii)>MAXIMUM_RESPONSE_COORDINATES:
            return dict(fallback=True,reason='response_cache_dimension_limit')
        key=tuple(np.asarray(cells,float))
        if key not in self.cache:return dict(fallback=True,reason='current_native_response_not_captured')
        item=self.cache[key];item['bank'].check();self.cache.move_to_end(key);self.hits+=1
        return item

    def native_evaluate(self,point):
        """Always return a full native curve; opportunistically retain columns."""
        s=self.source;p=s.check(point);n=s.fast.native;n._check();m=s.mode_dimensions
        key=tuple(p[m:]);entry=self._entry(p[m:])
        if not entry['fallback']:
            self.direct_native_calls+=1
            if self.native_forward is not None and entry.get('kernel') is not None:
                self.kernel_native_evaluations+=1
                return self.native_forward.evaluate(p,kernel=entry['kernel'])
            return s.native(p)
        if entry.get('reason')=='response_cache_dimension_limit' and key not in self.unavailable:
            self._unavailable(key,entry['reason'])
        if key in self.unavailable:
            self.direct_native_calls+=1;self.native_fallback_calls+=1
            return s.native(p)
        histogram_name=s.fast.slow.hist.name;phase_name=s.fast.slow.phase.name
        hist=n.histo_phases[0][histogram_name];phases=n.histo_phases[1]
        driver=n.math.getPowderProfile;began=time.perf_counter();self.capture_attempts+=1
        from ...preparation.native_tof_kernels import recorder_for_source
        recorder=recorder_for_source(s) if self.native_forward is not None else None
        try:
            matrix,bg,proof,curve=capture_rietveld_response(driver,n.math.G2pwd,n.parms,n.histdict,
                np.asarray(s.fast.x[s.fast.slice],float),hist,phases,n.controls,n.pawley,
                histogram_name=histogram_name,phase_name=phase_name,evaluate=lambda:s.native(p),
                maximum_bytes=self.maximum_bytes,
                observe_peak=recorder.observe if recorder is not None else None)
        except CaptureUnavailable as exc:
            self._unavailable(key,str(exc));self.direct_native_calls+=1
            self.native_fallback_calls+=1
            return s.native(p)
        finally:self.capture_seconds+=time.perf_counter()-began
        self.captured_native_calls+=1
        curve=np.asarray(curve,float)
        if curve.shape!=s.fast.x.shape or not np.isfinite(curve).all():
            raise ValueError('Captured native profile is incomplete')
        refs=hist['Reflection Lists'][phase_name]['RefList']
        order=np.lexsort((refs[:,2],refs[:,1],refs[:,0]))
        matrix=matrix[:,order].tocsc();refs=refs[order].copy()
        background=np.zeros_like(curve);background[s.fast.slice]=bg
        bank=NativeProfileBank(s,maximum_bytes=self.maximum_bytes,
            _snapshot=(curve,refs,background,phases[phase_name],hist))
        if not bank.linear:raise ValueError('A captured native context lost its linearity contract')
        rebuilt=background.copy();rebuilt[s.fast.slice]+=matrix@refs[:,9]
        error=float(np.linalg.norm(rebuilt-curve)/max(np.linalg.norm(curve-background),1e-30))
        self.maximum_capture_error=max(self.maximum_capture_error,error)
        if not np.isfinite(error) or error>1e-9:raise ValueError('Captured native response disagrees with the full native curve')
        item=native_scattering_entry(s,bank,matrix)
        size=(matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes+
              _numeric_bytes(item['ref'])+_numeric_bytes(item['parms'])+background.nbytes+bank.hkls.nbytes)
        if recorder is not None:
            kernel=recorder.finish(p,maximum_bytes=max(0,self.maximum_bytes-size))
            if kernel is not None:
                item['kernel']=kernel
                size+=kernel.data.numeric_bytes+1024*kernel.data.metadata['reflection_count']
            elif recorder.reason is not None:self.kernel_capture_fallbacks[recorder.reason]+=1
        item['numeric_bytes']=size;item['capture']=proof
        if size<=self.maximum_bytes:
            while self.cache and (len(self.cache)>=self.max_cells or self.numeric_bytes+size>self.maximum_bytes):
                _,old=self.cache.popitem(last=False);self.numeric_bytes-=old['numeric_bytes']
            for values in (matrix.data,matrix.indices,matrix.indptr):values.flags.writeable=False
            self.cache[key]=item;self.numeric_bytes+=size;self.builds+=1;self.inventory_sizes.add(len(refs))
        else:self._unavailable(key,'complete_numeric_context_exceeds_byte_budget')
        return curve

    def evaluate(self,point):
        p=self.source.check(point)
        if self._entry(p[self.source.mode_dimensions:])['fallback']:
            # The first call is itself native, so no second profile is needed.
            return self.native_evaluate(p)
        return super().evaluate(p)

    def mode_jacobian(self,point):
        p=self.source.check(point)
        if self.needs_capture(p):
            self.native_evaluate(p)
        if self._entry(p[self.source.mode_dimensions:])['fallback']:
            self.native_derivative_fallback_calls+=1
            return self.source.native_mode_jacobian(p)
        return super().mode_jacobian(p)

    def diagnostics(self):
        return dict(super().diagnostics(),capture_attempts=self.capture_attempts,
            captured_native_calls=self.captured_native_calls,direct_native_calls=self.direct_native_calls,
            numeric_cache_bytes=self.numeric_bytes,maximum_numeric_cache_bytes=self.maximum_bytes,
            unavailable_cells=len(self.unavailable),maximum_unavailable_cells=16,
            capture_seconds=self.capture_seconds,capture_fallbacks=dict(self.capture_fallbacks),
            kernel_native_evaluations=self.kernel_native_evaluations,
            kernel_capture_fallbacks=dict(self.kernel_capture_fallbacks),
            native_forward=self.native_forward.diagnostics() if self.native_forward is not None else None,
            maximum_capture_relative_error=self.maximum_capture_error,
            response_source='Already-required unchanged GSAS-II Rietveld call',interpolation=False)
