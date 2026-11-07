"""Stream native peak derivatives into lattice *proposal* directions.

Native scalar probes inspect reflection geometry and intensities without
rendering peaks. Their zero arrays never become predictions or objectives.
One genuine full native pass restores the current scientific state and streams
the original compiled peak derivatives into all lattice columns. Neither the
columns nor native support-window derivatives are claimed exact. Acceptance,
stationarity retry, uncertainty and exports retain the full native engine.
"""
import hashlib
import inspect
import sys
import threading
import time

import numpy as np

from .native_profile_capture import _capture_lock, RIETVELD_DRIVERS
from .reference_secants import feasible_stencil
from .native_tof_scalars import (for_source as scalar_collector_for_source,
    record_bytes,ScalarBatchUnavailable)


class ProposalUnavailable(ValueError):
    """The native scalar/peak contract requires original native differences."""


KERNELS = {
    'PNC': ('getFCJVoigt3', 'getdFCJVoigt3',
            '263ea0a0e0a337dd89c74de570f924a23d35648c05f9f90c3e8f334c9e698c82',
            '2e039fb28a7e671b590bb0c7404271f44973d556fbe26a19d430c17d32516ecc'),
    'PNT': ('getEpsVoigt', 'getdEpsVoigt',
            'a299231c52b148b2be785c4fd0b6d6ad8899cf5df67925adc06dc2fa297f73f3',
            'cbc214d0890f950f36dd358f65332a5e208204ab5bf21c2c8395b5c992cdd07a'),
}


def _hash(function):
    try:
        return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
    except (OSError, TypeError):
        return None


class NativeCellProposals:
    supports_native_anchor=True

    def __init__(self, source, *, maximum_bytes=64*1024*1024):
        if type(maximum_bytes) is not int or maximum_bytes < 1:
            raise ValueError('A positive cell-proposal workspace budget is required')
        self.source=source; self.maximum_bytes=maximum_bytes
        self.requests=0; self.completed=0; self.fallbacks=0
        self.scalar_calls=0; self.anchor_calls=0; self.seconds=0.
        self.maximum_accounted_bytes=0; self.membership_fallback_columns=0
        self.support_window_changes=0; self.last_reason=None
        self.batched_probe_calls=0;self.batched_anchor_checks=0;self.batched_anchor_mismatches=0
        self.maximum_batched_workspace_bytes=0
        self.cached_native_peak_calls=0;self.original_native_peak_calls=0
        from ...live.native_peak_derivative_batch import for_powder
        self.derivative_batch=for_powder(source.fast.native.math.G2pwd)
        self.derivative_batches=0;self.reused_native_anchors=0;self.batch_fallbacks=0
        self.maximum_batch_anchor_error=0.;self.last_batch_fallback=None

    def _context(self):
        s=self.source; n=s.fast.native; n._check()
        driver=n.math.getPowderProfile
        histograms,phases=n.histo_phases[:2]
        name=s.fast.slow.hist.name; phase=s.fast.slow.phase.name
        histogram=histograms[name]; hfx=f":{histogram['hId']}:"
        kind=n.controls.get(hfx+'histType')
        if (threading.current_thread() is not threading.main_thread()
                or kind not in KERNELS or _hash(driver) not in RIETVELD_DRIVERS
                or driver.__globals__.get('getPowderProfile') is not driver
                or len(n.parts)!=1 or hfx+'Lam1' in n.parms
                or set(phases)!={phase} or phases[phase]['General'].get('Type')!='nuclear'
                or phases[phase]['General'].get('Modulated')
                or phases[phase]['General'].get('doPawley')
                or n.controls.get('PhasePartials') or n.controls.get(hfx+'Exclude')
                or any(np.ma.getmaskarray(a).any() for a in histogram['Data'][:3])):
            raise ProposalUnavailable('Cell proposals require the reviewed unmasked single-phase native driver')
        phfx=f"{phases[phase]['pId']}:{histogram['hId']}:"
        if n.parms.get(phfx+'Extinction')!=0. or n.parms.get(phfx+'LeBail',False) or n.math.G2mp.InitMP()[0]:
            raise ProposalUnavailable('Cell proposals require isolated linear native peak rendering')
        powder=n.math.G2pwd; kernel,derivative,kh,dh=KERNELS[kind]
        peak=getattr(powder,kernel); deriv=getattr(powder,derivative)
        if _hash(peak)!=kh or _hash(deriv)!=dh:
            raise ProposalUnavailable('Native peak or derivative implementation is not qualified')
        axis=np.asarray(s.fast.x[s.fast.slice],float)
        if axis.ndim!=1 or not len(axis) or not np.isfinite(axis).all() or np.any(np.diff(axis)<=0):
            raise ProposalUnavailable('Ordered finite native grid required')
        return n,driver,powder,kernel,peak,deriv,axis,histogram,phases,phase

    def _batched_columns(self,context,collector,point,steps,anchor,allowed,check_time):
        """Reuse an already evaluated native anchor; retain native partials.

        Scalar probes and the compiled partials supply proposals only. Their
        reconstructed anchor must agree with the actual native curve passed
        by the owned optimizer. No saved-library assertion substitutes for it.
        """
        from ...live.native_peak_batch import BatchUnavailable
        s=self.source;n=context[0];m=s.mode_dimensions;count=len(point)-m
        if (collector is None or self.derivative_batch is None or not 1<=count<=24
                or type(anchor)is not np.ndarray or anchor.shape!=s.fast.x.shape
                or anchor.dtype!=np.dtype('float64') or not np.isfinite(anchor).all()):
            raise BatchUnavailable('An existing complete native anchor is required')
        if not _capture_lock.acquire(blocking=False):
            raise BatchUnavailable('Another native capture owns this process')
        pairs=[];details=[];required=8*(count+4)*len(anchor);restored=False
        try:
            for column in range(m,len(point)):
                stencil,detail=feasible_stencil(point,column,steps[column],allowed=allowed)
                pair=[]
                for at in stencil:
                    check_time();records=collector.collect(at,maximum_bytes=self.maximum_bytes-required)
                    required+=record_bytes(records);pair.append(records);self.batched_probe_calls+=1
                pairs.append(pair);details.append(detail)
            check_time();records=collector.collect(point,maximum_bytes=self.maximum_bytes-required)
            restored=True
            required+=record_bytes(records)
            keys=list(records);nr=len(keys)
            slopes_bytes=8*nr*count*6
            if required+slopes_bytes>self.maximum_bytes:
                raise BatchUnavailable('Native derivative coefficients exceed the proposal workspace')
            dp=np.zeros((nr,count,5));dh=np.zeros((nr,count));usable=np.ones(count,dtype=bool)
            for index,(pair,detail) in enumerate(zip(pairs,details)):
                if set(pair[0])!=set(records) or set(pair[1])!=set(records):
                    usable[index]=False;continue
                denominator=(2 if detail['scheme']=='central' else 1)*detail['effective_step']
                for row,key in enumerate(keys):
                    dp[row,index]=(pair[0][key][0]-pair[1][key][0])/denominator
                    dh[row,index]=(pair[0][key][1]-pair[1][key][1])/denominator
            axis=np.asarray(s.fast.x[s.fast.slice],float)
            signal,raw=self.derivative_batch.render_columns(records,axis,dp,dh,
                maximum_bytes=self.maximum_bytes-required)
            histogram=n.histo_phases[0][s.fast.slow.hist.name]
            background=np.asarray(histogram['Data'][4][s.fast.slice],float)
            reference=anchor[s.fast.slice]-background
            error=float(np.linalg.norm(signal-reference)/max(np.linalg.norm(reference),1e-30))
            self.maximum_batch_anchor_error=max(self.maximum_batch_anchor_error,error)
            if not np.isfinite(error) or error>1e-10:
                raise BatchUnavailable('Native derivative-batch anchor is not qualified at this geometry')
            n._check()
            full=np.zeros((count,len(anchor)));full[:,s.fast.slice]=raw;full[~usable]=0.
            self.membership_fallback_columns+=int(np.sum(~usable))
            self.derivative_batches+=1;self.reused_native_anchors+=1;self.completed+=1
            self.maximum_accounted_bytes=max(self.maximum_accounted_bytes,
                required+self.derivative_batch.last_workspace)
            return anchor.copy(),full,usable
        finally:
            try:
                if not restored:
                    s.native(point*s.radii);self.anchor_calls+=1
            finally:_capture_lock.release()

    def columns(self, point, steps, *, allowed, check_time=None, prepared_kernel=None,native_anchor=None):
        """Return a genuine native anchor, raw proposal columns and usable mask.

        ``point`` and steps use the optimizer's normalized physical geometry.
        Membership-changing columns are explicitly unavailable; the caller
        computes those complete columns with original native differences.
        """
        self.requests+=1; began=time.monotonic()
        if prepared_kernel is not None:
            from ...preparation.native_tof_kernels import QualifiedKernel
            if not isinstance(prepared_kernel,QualifiedKernel):
                raise TypeError('Native proposals require an owned qualified primitive kernel')
        s=self.source; point=np.asarray(point,float); steps=np.asarray(steps,float)
        m=s.mode_dimensions; d=len(s.radii); count=d-m
        if point.shape!=(d,) or steps.shape!=(d,) or count<1 or not np.isfinite(point).all() or not np.isfinite(steps).all():
            raise ValueError('Complete geometry and lattice proposal steps required')
        if check_time is None:check_time=lambda:None
        check_time()
        try:
            context=self._context()
        except ProposalUnavailable as error:
            self.fallbacks+=1;self.last_reason=str(error);self.seconds+=time.monotonic()-began
            raise
        n,driver,powder,kernel,peak,deriv,axis,histogram,phases,phase=context
        collector=scalar_collector_for_source(s,driver)
        if self.derivative_batch is not None and native_anchor is not None and collector is not None:
            from ...live.native_peak_batch import BatchUnavailable
            try:
                result=self._batched_columns(context,collector,point,steps,native_anchor,allowed,check_time)
            except (BatchUnavailable,ScalarBatchUnavailable) as error:
                self.batch_fallbacks+=1;self.last_batch_fallback=str(error)
            else:
                self.seconds+=time.monotonic()-began
                return result
        if not _capture_lock.acquire(blocking=False):
            self.fallbacks+=1;self.last_reason='Another native capture owns this process';self.seconds+=time.monotonic()-began
            raise ProposalUnavailable('Another native capture owns this process')
        # Account numeric output, probe records and a per-record allowance for
        # Python keys/containers. Native engine workspace and process RSS are
        # separate; no per-reflection derivative arrays survive their callback.
        required=8*(count+10)*len(s.fast.x)+8*count*d*3
        probes=[]; scalars_started=False; restored=False; calls=0
        raw=None
        usable=np.ones(count,dtype=bool)
        def budget(extra=0):
            if required+extra>self.maximum_bytes:
                raise ProposalUnavailable('Cell-proposal workspace budget exceeded')
            self.maximum_accounted_bytes=max(self.maximum_accounted_bytes,required+extra)

        def run(at, callback):
            nonlocal calls
            calls=0
            def watched(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls!=1 or len(args)<7 or args[4] is not histogram or args[5] is not phases or not np.array_equal(args[2],axis):
                    raise ProposalUnavailable('Native cell probe changed its histogram or grid')
                return driver(*args,**kwargs)
            def observed(*args,**kwargs):
                frame=sys._getframe(1)
                try:
                    local=frame.f_locals
                    if frame.f_code is not driver.__code__ or local.get('phase')!=phase or local.get('useMP') or local.get('im')!=0:
                        raise ProposalUnavailable('Native cell peak caller changed')
                    ref=local['refl']; begin,end=int(local['iBeg']),int(local['iFin'])
                    if not 0<=begin<end<=len(axis) or not np.array_equal(args[-1],axis[begin:end]):
                        raise ProposalUnavailable('Native cell peak window changed unexpectedly')
                    key=tuple(map(int,ref[:3])); parameters=np.asarray(args[:-1],float)
                    height=float(ref[9]*ref[11])
                    if not np.isfinite(parameters).all() or not np.isfinite(height):
                        raise ProposalUnavailable('Nonfinite native reflection arguments')
                    return callback(key,parameters,height,begin,end,args,kwargs)
                finally:
                    del frame
            try:
                powder.__dict__[kernel]=observed
                driver.__globals__['getPowderProfile']=watched
                value=s.native(np.asarray(at)*s.radii)
                if calls!=1:
                    raise ProposalUnavailable('Native cell probe omitted its reviewed driver')
                return np.asarray(value,float)
            finally:
                powder.__dict__[kernel]=peak
                driver.__globals__['getPowderProfile']=driver

        try:
            budget()
            raw=np.zeros((count,len(s.fast.x)))
            scalar_anchor=None
            if collector is not None:
                check_time();scalars_started=True
                scalar_anchor=collector.collect(point,maximum_bytes=self.maximum_bytes-required)
                required+=record_bytes(scalar_anchor);budget()
            for column in range(m,d):
                check_time()
                points,detail=feasible_stencil(point,column,steps[column],allowed=allowed)
                pair=[]
                for at in points:
                    check_time()
                    records={}
                    def scalar(key,parameters,height,begin,end,args,kwargs):
                        nonlocal required
                        if key in records:
                            raise ProposalUnavailable('Duplicate native reflection identity')
                        required+=parameters.nbytes+576; budget()
                        records[key]=(parameters.copy(),height,begin,end)
                        return np.zeros(end-begin),1.
                    scalars_started=True
                    if collector is None:
                        run(at,scalar);self.scalar_calls+=1
                    else:
                        records=collector.collect(at,maximum_bytes=self.maximum_bytes-required)
                        required+=record_bytes(records);budget();self.batched_probe_calls+=1
                    pair.append(records)
                probes.append((pair,detail))
            seen=set()
            def stream(key,parameters,height,begin,end,args,kwargs):
                if key in seen:
                    raise ProposalUnavailable('Duplicate native anchor reflection')
                seen.add(key)
                if scalar_anchor is not None:
                    saved=scalar_anchor.get(key)
                    if (saved is None or saved[2:]!=(begin,end)
                            or not np.allclose(saved[0],parameters,rtol=1e-12,atol=1e-12)
                            or not np.isclose(saved[1],height,rtol=1e-12,atol=1e-12)):
                        self.batched_anchor_mismatches+=1
                        raise ProposalUnavailable('Batched native scalars disagree with the full native anchor')
                result=(None if prepared_kernel is None else prepared_kernel.native_result(
                    key,parameters,axis,begin,end,function=peak))
                if result is None:
                    result=peak(*args,**kwargs);self.original_native_peak_calls+=1
                else:self.cached_native_peak_calls+=1
                values=deriv(*args,**kwargs)
                profile=np.asarray(result[0],float); derivatives=np.asarray(values[1:],float)
                budget(derivatives.nbytes+profile.nbytes)
                if (len(values)!=len(parameters)+1 or derivatives.shape!=(len(parameters),end-begin)
                        or not np.isfinite(derivatives).all() or not np.isfinite(profile).all()
                        or not np.array_equal(values[0],result[0])):
                    raise ProposalUnavailable('Native derivative profile differs from the original peak')
                for index,(pair,detail) in enumerate(probes):
                    if key not in pair[0] or key not in pair[1]:
                        usable[index]=False; continue
                    upper,lower=pair[0][key],pair[1][key]
                    denominator=(2 if detail['scheme']=='central' else 1)*detail['effective_step']
                    dp=(upper[0]-lower[0])/denominator; dh=(upper[1]-lower[1])/denominator
                    self.support_window_changes+=int(upper[2:]!=(begin,end) or lower[2:]!=(begin,end))
                    offset=s.fast.slice.start or 0
                    raw[index,offset+begin:offset+end]+=dh*profile+height*(dp@derivatives)
                return result
            check_time()
            anchor=run(point,stream); restored=True; self.anchor_calls+=1
            if scalar_anchor is not None:
                if set(scalar_anchor)!=seen:
                    self.batched_anchor_mismatches+=1
                    raise ProposalUnavailable('Batched native anchor reflection inventory changed')
                self.batched_anchor_checks+=1
            for index,(pair,detail) in enumerate(probes):
                if set(pair[0])!=seen or set(pair[1])!=seen:
                    usable[index]=False
            self.membership_fallback_columns+=int(np.sum(~usable))
            raw[~usable]=0.
            if anchor.shape!=(len(s.fast.x),) or not np.isfinite(anchor).all() or not np.isfinite(raw).all():
                raise ProposalUnavailable('Incomplete native cell proposal')
            self.completed+=1
            return anchor,raw,usable
        except ProposalUnavailable as error:
            self.fallbacks+=1; self.last_reason=str(error); raise
        except ScalarBatchUnavailable as error:
            self.fallbacks+=1;self.last_reason=str(error)
            raise ProposalUnavailable(str(error)) from error
        finally:
            powder.__dict__[kernel]=peak
            driver.__globals__['getPowderProfile']=driver
            try:
                if collector is not None:
                    self.maximum_batched_workspace_bytes=max(self.maximum_batched_workspace_bytes,
                        collector.maximum_workspace_bytes)
                if scalars_started and not restored:
                    s.native(point*s.radii); self.anchor_calls+=1
            finally:
                self.seconds+=time.monotonic()-began
                _capture_lock.release()

    def diagnostics(self):
        return dict(requests=self.requests,completed=self.completed,fallbacks=self.fallbacks,
            scalar_driver_calls=self.scalar_calls,full_native_anchor_calls=self.anchor_calls,
            batched_scalar_probe_calls=self.batched_probe_calls,
            batched_scalar_anchor_checks=self.batched_anchor_checks,
            batched_scalar_anchor_mismatches=self.batched_anchor_mismatches,
            maximum_batched_workspace_bytes=self.maximum_batched_workspace_bytes,
            cached_native_peak_calls=self.cached_native_peak_calls,
            original_native_peak_calls=self.original_native_peak_calls,
            membership_fallback_columns=self.membership_fallback_columns,
            support_window_changes=self.support_window_changes,
            maximum_accounted_workspace_bytes=self.maximum_accounted_bytes,
            workspace_budget_bytes=self.maximum_bytes,seconds=self.seconds,last_fallback_reason=self.last_reason,
            derivative_batches=self.derivative_batches,reused_native_anchors=self.reused_native_anchors,
            derivative_batch_fallbacks=self.batch_fallbacks,last_derivative_batch_fallback=self.last_batch_fallback,
            maximum_derivative_batch_anchor_error=self.maximum_batch_anchor_error,
            derivative_batch=None if self.derivative_batch is None else self.derivative_batch.diagnostics(),
            scope='Proposal directions only; native acceptance and fresh-native stationarity retry required',
            uncertainty_derivatives_qualified=False,portable_response=False)


def for_source(source):
    from .full_mode_slice import FullModeSlice
    return NativeCellProposals(source) if isinstance(source,FullModeSlice) and source.cell_dimensions else None
