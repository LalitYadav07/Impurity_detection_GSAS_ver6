"""Allocation-bounded independent structural starts, with native validation.

GSAS-II constraint maps are process-global. Each start gets a new process and
a separate native working directory; no mutable native objects are shared.
The existing per-model optimization budget is divided between the same starts,
not multiplied by the number of CPUs. Preparation/export remain separately
measured. A worker error is an error, never an implicitly discarded candidate.
"""
from dataclasses import asdict
import contextlib
import json
import multiprocessing
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

import numpy as np

from ...execution import write_json
from ...symmetry.resources import allocation


def fit(*args, **kwargs):
    """Resolve the shared optimizer at call time, as the serial service did."""
    from .profiled_native_fit import fit as current_fit
    return current_fit(*args, **kwargs)


def _cpu():
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime+usage.ru_stime


def _start_worker(directory,prepared,settings,observation,initial,steps,parent,seconds,iterations,memory_mb):
    """Only owned Python objects enter here, not uploaded executable state."""
    directory=Path(directory)
    import os
    os.environ['GALAXY_SLOTS']='1'
    if memory_mb is not None:os.environ['GALAXY_MEMORY_MB']=str(memory_mb)
    with (directory/'native.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        began=time.monotonic();cpu=_cpu()
        try:
            from ..service import SearchSettings,make_fit_context
            from .captured_cell_derivative import CapturedCellDerivative
            source,observed,table,_=make_fit_context(prepared,directory/'context',SearchSettings(**settings),observation)
            bank=SimpleNamespace(points=np.asarray([parent]),curves=np.empty((0,len(observed.x))),
                                 proposal_cell_steps=np.asarray(steps))
            result,rate=fit(source,table,observed,bank,seconds=seconds,starts=1,
                selected_starts=[initial],evaluator=CapturedCellDerivative(source,max_cells=2),
                qualification_iterations=iterations)
            if not np.isfinite(rate).all():raise ValueError('Nonfinite independent-start profile')
            np.save(directory/'prediction.npy',rate,allow_pickle=False)
            usage=resource.getrusage(resource.RUSAGE_SELF)
            write_json(directory/'result.json',dict(state='complete',result=result,
                worker=dict(pid=os.getpid(),wall_seconds=time.monotonic()-began,
                    allocated_slots=1,allocated_memory_mb=memory_mb,
                    cpu_seconds=_cpu()-cpu,peak_rss_bytes=int(usage.ru_maxrss*(1 if sys.platform=='darwin' else 1024)))))
        except BaseException as error:
            import traceback
            traceback.print_exc()
            write_json(directory/'result.json',dict(state='failed',error=type(error).__name__+': '+str(error)))
            raise


def _merge(screen,screen_rate,finished,elapsed,execution):
    """Keep every start and select the incumbent by its exact native objective."""
    ranked=[(-1,screen,screen_rate)]+[(i,r['result'],rate) for i,r,rate,_ in finished]
    index,best,rate=min(ranked,key=lambda row:(row[1]['best']['deviance'],row[0]))
    result=dict(finished[0][1]['result'] if finished else screen)
    result.update(best)
    blocks=[];attempts=[];blocked=[];trace=[]
    for i,item,_,offset in sorted(finished,key=lambda row:row[0]):
        row=item['result']
        blocks.extend(dict(b,start=i) for b in row['blocks'])
        attempts.extend(dict(a,start=i) for a in row['attempts'])
        blocked.extend(dict(b,start=i) for b in row['blocked_starts'])
        trace.extend(dict(t,seconds=t['seconds']+offset,start=i) for t in row['trace'])
    incumbent=float('inf');retained=[]
    for point in sorted(trace,key=lambda t:(t['seconds'],t['start'])):
        if point['deviance']<incumbent:retained.append(point);incumbent=point['deviance']
    result.update(seconds=elapsed,method='profiled_native_alternating_parallel',
        starts=len(finished),blocks=blocks,attempts=attempts,blocked_starts=blocked,
        trace=retained,restarts=[],parallel_execution=execution,
        selected_start=index,all_allowed_modes_retained=True,
        exact_native_deviance=best['best']['deviance'],
        full_native_calls=screen['full_native_calls']+sum(r['result']['full_native_calls'] for _,r,_,_ in finished),
        independent_cache_checks=sum(r['result']['independent_cache_checks'] for _,r,_,_ in finished),
        chemistry_rejections=sum(r['result']['chemistry_rejections'] for _,r,_,_ in finished),
        maximum_native_relative_error=max((r['result']['maximum_native_relative_error'] for _,r,_,_ in finished),default=0.),
        maximum_native_sigma_error=max((r['result']['maximum_native_sigma_error'] for _,r,_,_ in finished),default=0.),
        global_optimum_certified=False,model_exclusion_authorized=False)
    return result,rate


def fit_allocated(prepared,output,settings,source,table,observation,bank,*,warm_starts=(),evaluator=None,
                  qualification_iterations=None):
    capacity=allocation(settings.starts)
    workers=min(capacity['workers'],settings.starts)
    if workers==1:
        result,rate=fit(source,table,observation,bank,seconds=settings.seconds,starts=settings.starts,
            warm_pool=list(warm_starts) or None,evaluator=evaluator,
            qualification_iterations=qualification_iterations)
        result['parallel_execution']=dict(allocation=capacity,workers=1,peak_active_workers=1,
            budget_scope='total per-model optimization time; sequential fallback',processes=[])
        return result,rate
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    began=time.monotonic()
    screen,screen_rate=fit(source,table,observation,bank,seconds=settings.seconds,starts=settings.starts,
        warm_pool=list(warm_starts) or None,evaluator=evaluator,screening_only=True)
    selected=screen['selected_starts'];remaining=max(0.,settings.seconds-(time.monotonic()-began))
    if not selected or remaining<=0:
        execution=dict(allocation=capacity,workers=workers,peak_active_workers=0,
            budget_scope='total per-model optimization time; screening exhausted allocation',processes=[])
        return _merge(screen,screen_rate,[],time.monotonic()-began,execution)
    per_start=remaining/len(selected)
    pending=list(enumerate(selected));active={};finished=[];peak=0;processes=[]
    context=multiprocessing.get_context('spawn')
    try:
        while pending or active:
            while pending and len(active)<workers:
                index,point=pending.pop(0);directory=output/f'start-{index:02d}';directory.mkdir()
                args=(str(directory),str(prepared),asdict(settings),observation,point,
                      bank.proposal_cell_steps,bank.points[0],per_start,qualification_iterations,
                      None if capacity['memory_mb'] is None else max(1,(capacity['memory_mb']-capacity['reserve_mb'])//workers))
                process=context.Process(target=_start_worker,args=args)
                launched=time.monotonic();process.start()
                active[index]=(process,directory,launched)
                peak=max(peak,len(active))
            for index,(process,directory,launched) in list(active.items()):
                if process.is_alive():
                    # Startup and a final indivisible native call are not part
                    # of the optimizer allocation, but cannot run unbounded.
                    if time.monotonic()-launched>per_start+60:
                        raise TimeoutError('Independent structural start exceeded its cleanup allowance')
                    continue
                process.join();path=directory/'result.json'
                if process.exitcode or not path.is_file():
                    raise RuntimeError(f'Structural start {index} failed; inspect {directory}/native.log')
                if path.stat().st_size>16*1024*1024:raise ValueError('Structural start receipt exceeds its limit')
                item=json.loads(path.read_text())
                if item.get('state')!='complete':raise RuntimeError(item.get('error','Incomplete structural start'))
                rate=np.load(directory/'prediction.npy',allow_pickle=False)
                if rate.shape!=observation.x.shape or not np.isfinite(rate).all():
                    raise ValueError('Independent structural start returned an invalid profile')
                finished.append((index,item,rate,launched-began))
                processes.append(dict(start=index,allocated_seconds=per_start,**item['worker']))
                del active[index]
            if active:time.sleep(.02)
    finally:
        for process,_,_ in active.values():
            if process.is_alive():process.terminate()
        for process,_,_ in active.values():
            process.join(timeout=3)
            if process.is_alive():process.kill();process.join(timeout=3)
    if len(finished)!=len(selected):raise RuntimeError('Not every selected structural start completed')
    execution=dict(allocation=capacity,workers=workers,peak_active_workers=peak,
        budget_scope='unchanged aggregate per-model optimization time divided among the same starts',
        screening_seconds=screen['seconds'],total_start_allocation_seconds=per_start*len(selected),
        per_start_seconds=per_start,processes=sorted(processes,key=lambda row:row['start']),
        aggregate_start_cpu_seconds=sum(row['cpu_seconds'] for row in processes),
        all_starts_retained=True,native_workspaces_separate=True)
    return _merge(screen,screen_rate,finished,time.monotonic()-began,execution)
