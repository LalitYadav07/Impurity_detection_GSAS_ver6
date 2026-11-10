"""Owned continuous TOF proposals, separate from native likelihood authority.

The native profile compiles its numeric table or binds proposal-only data
with verified creating-job origin and current native shape/runtime checks.
Uploaded proof flags never confer authority. Geometry, F², reflections and
windows remain current. Fixed-cell reuse belongs to the caller. This renderer
is only for initial proposals; accepted fits retain the original objective.
"""
from collections import Counter
import ctypes
import time
import numpy as np

from .native_forward import NativeTOFForward, MAXIMUM_BYTES
from .native_cell_proposals import ProposalUnavailable
from .native_tof_scalars import record_bytes
from ...live.native_peak_batch import BatchUnavailable, pack_records
from ...preparation.kernel_geometry import _profile
from ...preparation.kernel_runtime import floating_state
from .tof_profile_table import CompiledTOFTable,ROUGH_NATIVE_RELATIVE,shared_table

MINIMUM_REFERENCE_BINS=1_000_000


class TOFProfileProposals:
    def __init__(self,source,*,exact=None,table=None):
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
        refs=hist['Reflection Lists'][source.fast.slow.phase.name]['RefList']
        if not len(refs) or not np.isfinite(refs[:,4]).all() or np.any(refs[:,4]<=0):
            raise ProposalUnavailable('A complete positive native reflection metric is required')
        from .tof_shape import IsotropicTOFShape
        shape=IsotropicTOFShape(n,source.fast.slow.hist.name,source.fast.slow.phase.name)
        self.shape_identity=shape.identity
        self.table_reused=table is not None
        if table is None:
            table,self.table_reused=shared_table(shape,powder,
                (float(refs[:,4].min()),float(refs[:,4].max())))
        elif not isinstance(table,CompiledTOFTable):
            raise TypeError('A currently owned native TOF table is required')
        table.check_runtime()
        if table.shape_identity!=shape.identity:
            raise ProposalUnavailable('Prepared TOF shape differs from the current instrument or sample broadening')
        self.table=table
        for name in ('coefficients','halo','logd','eta','numeric_bytes','midpoint_checks',
                     'rejected_cells','native_only_stripes'):
            setattr(self,name,getattr(self.table,name))
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
        self.table.check_runtime()
        if _profile(self.source)!=self.profile_sha256 or floating_state()!=self.floating:
            raise ProposalUnavailable('Current profile changed; regenerate TOF proposals')
        return context

    def evaluate(self,point):
        began=time.monotonic();n,_,_,_,_,_,axis,hist,_,_=self._context()
        source=self.source;physical=source.check(np.asarray(point,float))
        remaining=MAXIMUM_BYTES-self.numeric_bytes-self.exact.fixed_workspace
        records=self.exact.collector.collect_packed_physical(physical,maximum_bytes=remaining)
        scratch_budget=remaining-record_bytes(records)
        if scratch_budget<1:raise ProposalUnavailable('Continuous TOF proposals exceed the bounded forward workspace')
        parameters,heights,bounds,maximum,required=pack_records(records,axis,scratch_budget)
        d=records.spacings
        footprint=self.numeric_bytes+self.exact.fixed_workspace+max(
            self.exact.collector.last_workspace_bytes,record_bytes(records)+required+16*maximum)
        if footprint>MAXIMUM_BYTES:
            raise ProposalUnavailable('Continuous TOF proposals exceed the bounded forward workspace')
        scratch=[np.empty(maximum,np.float32),np.empty(maximum,np.float32),
                 np.empty(maximum),np.empty(maximum,np.int64)]
        signal=np.zeros(len(axis));statistics=np.zeros(6,np.int64)
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
            'unsupported_native_bins','identical_adjacent_rows_reused','identical_adjacent_bins_reused'],map(int,statistics))))
        return curve

    def diagnostics(self):
        return dict(method='Native-compiled continuous TOF starting proposals',
            calls=self.calls,seconds=self.seconds,build_seconds=self.build_seconds,
            statistics=dict(self.statistics),resident_numeric_bytes=self.numeric_bytes,
            maximum_accounted_workspace_bytes=self.maximum_workspace,
            profile_sha256=self.profile_sha256,interpolation=True,
            shape_sha256=self.shape_identity,
            native_midpoint_checks=self.midpoint_checks,rejected_interpolation_cells=self.rejected_cells,
            native_only_d_stripes=self.native_only_stripes,
            roughness_relative_to_native_precision=float(ROUGH_NATIVE_RELATIVE),
            native_origin_authority=False,observations_used_for_compilation=False,
            owned_table_reused=self.table_reused,
            scalar_record_format='owned packed native columns; original reflection order',
            table_import_supported=True,portable_table_bound=self.table.portable_origin is not None,
            portable_table_reused=self.table.portable_origin is not None and self.statistics['interpolated_bins']>0,
            portable_origin=self.table.portable_origin,current_portable_probes=self.table.portable_probes,
            original_native_objective_required=True)


def for_source(source,*,exact=None,table=None):
    """Avoid table setup for workloads below the measured large-profile case."""
    exact=exact or NativeTOFForward(source)
    records=exact.collector.collect_packed_physical(np.zeros(len(source.radii)),maximum_bytes=MAXIMUM_BYTES)
    if int(np.sum(records.bounds[:,1]-records.bounds[:,0],dtype=np.int64))<MINIMUM_REFERENCE_BINS:
        # This threshold amortizes compilation, not numerical fidelity. A
        # compatible already-prepared table has paid that cost. Discarding it
        # here would make a valid cold-library handoff silently unused.
        if table is None:
            from .tof_profile_table import available_table
            from .tof_shape import IsotropicTOFShape
            n,_,powder,_,_,_,_,hist,_,_=exact._context()
            refs=hist['Reflection Lists'][source.fast.slow.phase.name]['RefList']
            if (not len(refs) or not np.isfinite(refs[:,4]).all() or np.any(refs[:,4]<=0)
                    or refs[:,4].min()>=refs[:,4].max()):return None
            try:
                shape=IsotropicTOFShape(n,source.fast.slow.hist.name,source.fast.slow.phase.name)
                table=available_table(shape,powder,(float(refs[:,4].min()),float(refs[:,4].max())))
            except ProposalUnavailable:return None
        if table is None:return None
    result=TOFProfileProposals(source,exact=exact,table=table)
    if not np.any(result.halo):
        raise ProposalUnavailable('Current native profile needs original peaks throughout the proposal envelope')
    return result
