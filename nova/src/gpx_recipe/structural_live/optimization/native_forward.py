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


BACKGROUND_SOURCE='ba21fb1e4304debbf4cb43065524b728fa39695087102ffdb50371f4dca274b7'
MAXIMUM_BYTES=64*1024*1024


class NativeTOFForward:
    def __init__(self,source,*,maximum_bytes=MAXIMUM_BYTES):
        if type(maximum_bytes)is not int or not 0<maximum_bytes<=MAXIMUM_BYTES:
            raise ValueError('A positive bounded native forward workspace is required')
        self.source=source;self.maximum_bytes=maximum_bytes
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
        self.requests=self.rendered=self.anchor_checks=self.fallback_calls=self.peak_calls=0
        self.kernel_hits=self.kernel_misses=0
        self.seconds=0.;self.bypasses=Counter()
        self.prepared_bank=None

    def _context(self):
        context=self.guard._context()
        n,_,powder,_,_,_,axis,histogram,_,_=context
        hfx=self.collector.hfx;phfx=self.collector.phfx
        if (n.controls.get(hfx+'histType')!='PNT'
                or n.controls.get(phfx+'SizeType')!='isotropic'
                or n.controls.get(phfx+'MustrainType')!='isotropic'
                or powder.getBackground is not self.background_function
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
                self.fallback_calls+=1
                return s.native(point)
            try:
                n,driver,powder,_,peak,_,axis,histogram,_,_=self._context()
                physical=s.check(np.asarray(point,float))
                records=self.collector.collect_physical(physical,
                    maximum_bytes=self.maximum_bytes-self.fixed_workspace)
            except (ProposalUnavailable,ScalarBatchUnavailable) as exc:
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
            signal=np.zeros_like(background)
            for hkl,(parameters,height,low,high) in records.items():
                value=None if kernel is None else kernel.lookup(hkl,parameters,axis,low,high,function=peak)
                if value is None:
                    value=peak(*parameters,axis[low:high])[0];self.peak_calls+=1
                    self.kernel_misses+=int(kernel is not None)
                else:self.kernel_hits+=1
                signal[low:high]+=np.float64(height)*value
            curve=np.zeros(len(s.fast.x));curve[s.fast.slice]=signal+background
            if not np.isfinite(curve).all():
                raise ValueError('Native batched forward returned a nonfinite profile')
            self.rendered+=1
            if not self.qualified:
                native=s.native(physical);self.anchor_checks+=1
                if not np.array_equal(curve,native):
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
            seconds=self.seconds,qualified_anchor=self.qualified,disabled=self.disabled,
            kernel_hits=self.kernel_hits,kernel_misses=self.kernel_misses,
            bypasses=dict(self.bypasses),maximum_workspace_bytes=self.maximum_bytes,
            profile_interpolation=False,observations_reused=False,
            prepared_kernel_bank=self.prepared_bank)


def for_source(source,*,maximum_bytes=MAXIMUM_BYTES):
    try:return NativeTOFForward(source,maximum_bytes=maximum_bytes)
    except (ProposalUnavailable,ScalarBatchUnavailable):return None
