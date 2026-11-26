"""Process-isolated construction of the unchanged complete start portfolio."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

from ...symmetry.contracts import digest
from ...symmetry.resources import allocation

_source=None
_renderer=None
_identity=None
_native=None


def _initialize(prepared,work,radii,modes,native,memory_mb):
    global _source,_renderer,_identity,_native
    os.environ['GALAXY_SLOTS']='1'
    if memory_mb is not None:os.environ['GALAXY_MEMORY_MB']=str(memory_mb)
    from ..service import SearchSettings,make_source
    from .offline_native_start_bank import context
    from .native_bank_reuse import NativeBankSource
    settings=SearchSettings(mode_radius=float(radii[0]) if modes else .05,
                            cell_radius=float(radii[-1]))
    _source=make_source(prepared,Path(work)/str(os.getpid()),settings)
    if not np.array_equal(_source.radii,radii):raise ValueError('Parallel start-bank bounds changed')
    exact=NativeBankSource(_source);_renderer=exact.native;_native=native
    if not native:
        from .tof_profile_proposals import for_source
        from .starting_proposals import _Hybrid
        from .native_cell_proposals import ProposalUnavailable
        from .native_tof_scalars import ScalarBatchUnavailable
        from ...live.native_peak_batch import BatchUnavailable
        try:proposals=for_source(_source,exact=exact.forward) if exact.forward is not None else None
        except (ProposalUnavailable,ScalarBatchUnavailable,BatchUnavailable):proposals=None
        if proposals is not None:_renderer=_Hybrid(_source,exact,proposals).evaluate
    _identity=digest(context(_source))


def _evaluate(point):
    begin=time.monotonic();usage=resource.getrusage(resource.RUSAGE_SELF);cpu=usage.ru_utime+usage.ru_stime
    curve=np.asarray(_renderer(point),float)
    after=resource.getrusage(resource.RUSAGE_SELF)
    return curve,dict(pid=os.getpid(),started=begin,finished=time.monotonic(),
        cpu_seconds=after.ru_utime+after.ru_stime-cpu,context_sha256=_identity,
        peak_rss_bytes=int(after.ru_maxrss*(1 if sys.platform=='darwin' else 1024)))


class NativeBankProcessMap:
    """Lazy: a reused bank never starts workers or pays native setup cost."""
    def __init__(self,source,output):
        from .offline_native_start_bank import context
        self.source=source;self.output=Path(output);self.capacity=allocation(16)
        self.identity=digest(context(source));self.workers=self.capacity['workers']

    def __call__(self,points,*,native):
        points=np.asarray(points,float)
        if points.ndim!=2 or points.shape[1]!=len(self.source.radii) or not np.isfinite(points).all():
            raise ValueError('Complete finite start-bank coordinates required')
        count=min(self.workers,len(points))
        if count<2:raise ValueError('A parallel bank requires at least two admitted tasks')
        self.output.mkdir(parents=True,exist_ok=False)
        began=time.monotonic()
        # Only the already verified prepared context enters children. Uploaded
        # native-origin flags and parent-process GSAS-II objects never do.
        prepared=str(self.source.fast.slow.root)
        memory=None if self.capacity['memory_mb'] is None else max(1,(self.capacity['memory_mb']-self.capacity['reserve_mb'])//count)
        args=(prepared,str(self.output),self.source.radii,self.source.mode_dimensions,native,memory)
        with ProcessPoolExecutor(max_workers=count,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_initialize,initargs=args) as pool:
            values=list(pool.map(_evaluate,points,chunksize=1))
        curves=[];records=[]
        for curve,record in values:
            if record['context_sha256']!=self.identity:raise ValueError('Parallel bank native context changed')
            if curve.shape!=self.source.fast.x.shape or not np.isfinite(curve).all():
                raise ValueError('Incomplete parallel starting curve')
            curves.append(curve);records.append(record)
        events=sorted([(r['started'],1) for r in records]+[(r['finished'],-1) for r in records])
        active=0;peak=0
        for _,change in events:active+=change;peak=max(peak,active)
        return np.asarray(curves),dict(allocation=self.capacity,workers=count,
            worker_pids=sorted({r['pid'] for r in records}),peak_active_tasks=peak,
            completed_curves=len(curves),seconds=time.monotonic()-began,
            native_curve_role=native,aggregate_curve_cpu_seconds=sum(r['cpu_seconds'] for r in records),
            peak_worker_rss_bytes=max(r['peak_rss_bytes'] for r in records),
            every_context_verified=True,all_points_retained=True)
