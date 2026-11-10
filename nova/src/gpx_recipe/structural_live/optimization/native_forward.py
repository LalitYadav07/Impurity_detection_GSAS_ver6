"""Native TOF start curves with scalar batching and original peak kernels.

This helper changes the construction schedule, not the peak model. It never
interpolates or reuses a curve across geometries. The original full native
source remains available for independent checks and all unsupported contexts.
"""
from collections import Counter
import time
import numpy as np

from .native_cell_proposals import NativeCellProposals,ProposalUnavailable,_hash
from .native_tof_scalars import for_source as scalar_for_source,ScalarBatchUnavailable
from .native_tof_scalars import record_bytes
from ...preparation.kernel_runtime import floating_state
from ...live.native_peak_records import record_items


BACKGROUND_SOURCE='ba21fb1e4304debbf4cb43065524b728fa39695087102ffdb50371f4dca274b7'
MAXIMUM_BYTES=64*1024*1024


class NativeTOFForward:
    def __init__(self,source,*,maximum_bytes=MAXIMUM_BYTES):
        if type(maximum_bytes)is not int or not 0<maximum_bytes<=MAXIMUM_BYTES:
            raise ValueError('A positive bounded native forward workspace is required')
        self.source=source;self.maximum_bytes=maximum_bytes
        self.floating=floating_state()
        if self.floating is None:
            raise ProposalUnavailable('Native TOF forward arithmetic controls are unavailable')
        self.guard=NativeCellProposals(source,maximum_bytes=maximum_bytes)
        context=self.guard._context();n=context[0]
        self.collector=scalar_for_source(source,context[1])
        if self.collector is None or not np.array_equal(n.factor,np.ones_like(n.factor)):
            raise ProposalUnavailable('Batched forward requires one qualified native TOF context')
        self.background_function=context[2].getBackground
        if _hash(self.background_function)!=BACKGROUND_SOURCE:
            raise ProposalUnavailable('Unreviewed native background function')
        self._context()
        self.fixed_workspace=8*(3*len(source.fast.x)+10*len(context[6]))
        if self.fixed_workspace>=maximum_bytes:
            raise ScalarBatchUnavailable('Native forward output exceeds its workspace budget')
        self.qualified=False;self.disabled=False
        self.anchor_comparison=None
        self.requests=self.rendered=self.anchor_checks=self.fallback_calls=self.peak_calls=0
        self.kernel_hits=self.kernel_misses=0
        self.seconds=0.;self.bypasses=Counter()
        self.prepared_bank=None
        from ...live.native_peak_batch import for_powder
        self.native_batch=for_powder(context[2]);self.batch_fallbacks=Counter();self.native_batch_disabled=False

    def _context(self):
        context=self.guard._context()
        n,_,powder,_,_,_,axis,histogram,_,_=context
        hfx=self.collector.hfx;phfx=self.collector.phfx
        if (n.controls.get(hfx+'histType')!='PNT'
                or n.controls.get(phfx+'SizeType')!='isotropic'
                or n.controls.get(phfx+'MustrainType')!='isotropic'
                or powder.getBackground is not self.background_function
                or floating_state()!=self.floating
                or not np.array_equal(n.factor,np.ones_like(n.factor))
                or not np.array_equal(axis,np.asarray(histogram['Data'][0][n.parts[0][1]]))):
            raise ProposalUnavailable('Native TOF forward context changed')
        return context

    def evaluate(self,point,*,kernel=None):
        prepared=None
        if kernel is not None:
            from ...preparation.native_tof_kernels import QualifiedKernel
            from ...preparation.kernel_geometry import PreparedKernelBank
            if isinstance(kernel,PreparedKernelBank):
                if kernel.source is not self.source:
                    raise TypeError('Prepared kernels must bind this owned native source')
                prepared=kernel
            elif not isinstance(kernel,QualifiedKernel):
                raise TypeError('A natively qualified kernel chunk is required')
        began=time.monotonic();self.requests+=1;s=self.source
        try:
            if self.disabled:
                if self.native_batch is not None:self.native_batch.discard_cached_plane()
                self.fallback_calls+=1
                return s.native(point)
            try:
                n,driver,powder,_,peak,_,axis,histogram,_,_=self._context()
                physical=s.check(np.asarray(point,float))
                budget=self.maximum_bytes-self.fixed_workspace
                cached=0 if self.native_batch is None else self.native_batch.cached_plane_bytes
                if cached>=budget:
                    self.native_batch.discard_cached_plane();cached=0
                try:
                    records=self.collector.collect_packed_physical(physical,maximum_bytes=budget-cached)
                except ScalarBatchUnavailable:
                    if not cached:raise
                    # Cached unit shapes are disposable. They cannot reduce
                    # the admitted scientific geometry or scalar workspace.
                    self.native_batch.discard_cached_plane()
                    records=self.collector.collect_packed_physical(physical,maximum_bytes=budget)
            except (ProposalUnavailable,ScalarBatchUnavailable) as exc:
                if self.native_batch is not None:self.native_batch.discard_cached_plane()
                self.bypasses[str(exc)]+=1;self.fallback_calls+=1
                return s.native(point)
            if prepared is not None:
                kernel=prepared.resolve(physical)
            hfx=self.collector.hfx
            background,_=self.background_function(hfx,n.parms,n.controls[hfx+'bakType'],
                n.controls[hfx+'histType'],axis,histogram['Background'][1].get('fixback'))
            background=np.asarray(background,float)
            if background.shape!=axis.shape or not np.isfinite(background).all():
                raise ValueError('Native background returned an invalid forward profile')
            if kernel is None and self.native_batch is not None and not self.native_batch_disabled:
                from ...live.native_peak_batch import BatchUnavailable
                try:
                    remaining=self.maximum_bytes-self.fixed_workspace-record_bytes(records)
                    if remaining<1:raise BatchUnavailable('No remaining native batch workspace')
                    signal=self.native_batch.render(records,axis,maximum_bytes=remaining)
                    self.peak_calls+=len(records)
                except BatchUnavailable as error:
                    self.native_batch.discard_cached_plane()
                    self.batch_fallbacks[str(error)]+=1;self.fallback_calls+=1
                    self.native_batch_disabled=True;self.qualified=False
                    # A changed ABI/control state or an unusual window must
                    # use a fresh complete native pass, not partial output.
                    return s.native(physical)
            else:
                if self.native_batch is not None:self.native_batch.discard_cached_plane()
                signal=np.zeros_like(background)
                for hkl,(parameters,height,low,high) in record_items(records):
                    value=None if kernel is None else kernel.lookup(hkl,parameters,axis,low,high,function=peak)
                    if value is None:
                        value=peak(*parameters,axis[low:high])[0];self.peak_calls+=1
                        self.kernel_misses+=int(kernel is not None)
                    else:self.kernel_hits+=1
                    signal[low:high]+=np.float64(height)*value
            # The native helper releases the GIL. The scientific worker is
            # isolated, but never publish if foreign code replaced its GSAS
            # constraint owner while the pure peak calculation was running.
            n._check()
            curve=np.zeros(len(s.fast.x));curve[s.fast.slice]=signal+background
            if not np.isfinite(curve).all():
                raise ValueError('Native batched forward returned a nonfinite profile')
            self.rendered+=1
            if not self.qualified:
                native=s.native(physical);self.anchor_checks+=1
                identical=np.array_equal(curve,native)
                delta=curve-native
                self.anchor_comparison=dict(array_identical=identical,
                    differing_bins=int(np.count_nonzero(delta)),
                    maximum_absolute_error=float(np.max(abs(delta),initial=0.)),
                    relative_l2_error=float(np.linalg.norm(delta)/max(np.linalg.norm(native),1e-30)),
                    arithmetic_controls=list(self.floating))
                if not identical:
                    self.disabled=True
                    self.bypasses['native anchor is not array-identical']+=1
                else:self.qualified=True
                return native
            return curve
        finally:
            self.seconds+=time.monotonic()-began
            if prepared is not None:self.prepared_bank=prepared.diagnostics()

    def diagnostics(self):
        return dict(method='batched native scalars and original native TOF kernels',
            requests=self.requests,rendered=self.rendered,native_anchor_checks=self.anchor_checks,
            native_fallback_calls=self.fallback_calls,peak_calls=self.peak_calls,
            peak_call_count_scope='logical reflections; actual native evaluations are in native_peak_batch',
            seconds=self.seconds,qualified_anchor=self.qualified,disabled=self.disabled,
            kernel_hits=self.kernel_hits,kernel_misses=self.kernel_misses,
            anchor_comparison=self.anchor_comparison,
            bypasses=dict(self.bypasses),maximum_workspace_bytes=self.maximum_bytes,
            profile_interpolation=False,observations_reused=False,
            native_peak_batch=self.native_batch.diagnostics() if self.native_batch is not None else None,
            native_batch_fallbacks=dict(self.batch_fallbacks),
            native_peak_batch_disabled=self.native_batch_disabled,
            prepared_kernel_bank=self.prepared_bank)


def for_source(source,*,maximum_bytes=MAXIMUM_BYTES):
    try:return NativeTOFForward(source,maximum_bytes=maximum_bytes)
    except (ProposalUnavailable,ScalarBatchUnavailable):return None
