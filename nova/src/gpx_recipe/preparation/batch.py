"""Finite, allocation-bounded preparation batches with isolated native tasks.

The mathematical operations remain preparation.worker.execute. A batch shares
transport and scheduling, not mutable GSAS-II process state or model physics.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import threading
import time

from ..execution import write_json
from ..process import run_scientific, ScientificCancelled
from ..symmetry.compiler import confined, file_sha, native_environment
from ..symmetry.contracts import digest, fields
from ..symmetry.resources import allocation

SCHEMA='radar-preparation-batch/v1'
RESULT_SCHEMA='radar-preparation-batch-result/v1'
MAX_TASKS=64
MAX_METADATA=8*1024*1024
RESULT_BUDGET=480*1024*1024
# Reserve the existing maximum scientific array/metadata bounds before
# admitting a native task, then release unused space at its checked completion.
RESERVES={'reference':8*1024*1024,'magnetic':144*1024*1024,'structural':400*1024*1024}


def relative(value):
    if (not isinstance(value,str) or not value or len(value)>500 or '\\' in value or '\x00' in value
            or ':' in value or PurePosixPath(value).is_absolute() or '..' in PurePosixPath(value).parts
            or str(PurePosixPath(value))!=value):
        raise ValueError('Preparation batch paths must be normalized and confined')
    return value


def validate_plan(value):
    from .worker import validate_request
    value=fields(value,['schema','tasks'])
    if (value['schema']!=SCHEMA or not isinstance(value['tasks'],list)
            or not 1<=len(value['tasks'])<=MAX_TASKS
            or len(json.dumps(value,allow_nan=False).encode())>MAX_METADATA):
        raise ValueError('Declare a bounded finite preparation batch')
    seen=set()
    for task in value['tasks']:
        fields(task,['entry','attempt','kind','request'])
        if (not isinstance(task['entry'],str) or not re.fullmatch('[a-f0-9]{24}',task['entry']) or type(task['attempt']) is not int
                or not 1<=task['attempt']<=1000000 or task['kind'] not in RESERVES
                or task['entry'] in seen):
            raise ValueError('Invalid or duplicate preparation batch entry')
        seen.add(task['entry']);validate_request(task['kind'],task['request'])
        request=task['request']
        for role in ['source','instrument']+(['child'] if task['kind']=='structural' else []):
            relative(request[role])
        expected='tasks/'+task['entry']+'/attempt-'+str(task['attempt'])+'/compiled'
        if request['destination']!=expected:
            raise ValueError('Preparation output must belong to its exact entry attempt')
    return value


def validate_request(value):
    value=fields(value,['source','plan','destination','workers','seconds'])
    if (type(value['workers']) is not int or not 1<=value['workers']<=16
            or type(value['seconds']) is not int or not 30<=value['seconds']<=540
            or any(not isinstance(value[k],str) or not value[k] for k in ('source','plan','destination'))):
        raise ValueError('Use bounded preparation workers and wall time')
    return value


def read_plan(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>MAX_METADATA:
        raise ValueError('Preparation plan is missing or oversized')
    return validate_plan(json.loads(path.read_text()))


def inputs(plan):
    validate_plan(plan)
    return sorted({task['request'][role] for task in plan['tasks']
        for role in ['source','instrument']+(['child'] if task['kind']=='structural' else [])})


def stage(library, entries, directory, *, workers=4, seconds=480):
    """Capture or resume the same immutable batch before its first dispatch."""
    from .operations import stage_entry
    directory=Path(directory)
    if directory.is_symlink():raise ValueError('Preparation batch directory is not owned')
    directory=directory.resolve();directory.mkdir(parents=True,exist_ok=True)
    tasks=[];parent=None
    for entry,attempt in entries:
        request=stage_entry(library,entry,attempt,directory/'tasks'/entry/('attempt-'+str(attempt)))
        kind=library.row(entry)['model']['kind'];captured=copy.deepcopy(request)
        for role in ['source','instrument']+(['child'] if kind=='structural' else []):
            path=Path(request[role]);checksum=file_sha(path)
            shared=directory/'shared'/(checksum+path.suffix.lower());shared.parent.mkdir(exist_ok=True)
            if not shared.exists():shutil.copyfile(path,shared)
            if file_sha(shared)!=checksum:raise ValueError('A shared preparation input changed')
            captured[role]=str(shared.relative_to(directory))
            if role=='source':
                if parent is not None and file_sha(parent)!=checksum:
                    raise ValueError('One preparation batch must share the captured parent CIF')
                parent=shared
        captured['destination']=str(Path(request['destination']).relative_to(directory))
        tasks.append(dict(entry=entry,attempt=attempt,kind=kind,request=captured))
    plan=validate_plan(dict(schema=SCHEMA,tasks=tasks));path=directory/'plan.json'
    if path.exists():
        if read_plan(path)!=plan:raise ValueError('An immutable preparation batch plan changed')
    else:write_json(path,plan)
    return validate_request(dict(source=str(parent),plan=str(path),destination=str(directory),
        workers=workers,seconds=seconds))


def run_task(task, request, scratch, *, cancel, timeout):
    """Fresh process per model; no scientific implementation is duplicated."""
    scratch=Path(scratch);scratch.mkdir(parents=True,exist_ok=False)
    message=scratch/'request.json';output=scratch/'response.json';write_json(message,request)
    began=time.monotonic()
    with (scratch/'native.log').open('w') as stream:
        try:
            result=run_scientific([sys.executable,'-m','gpx_recipe.preparation.worker',task['kind'],str(message),str(output)],
                cwd=scratch,env=native_environment(),cancel_event=cancel,timeout=timeout,
                stdout=stream,stderr=subprocess.STDOUT)
        except ScientificCancelled:
            return dict(state='cancelled',reason='Native preparation was cancelled',seconds=time.monotonic()-began)
        except subprocess.TimeoutExpired:
            return dict(state='failed',reason='Native preparation exceeded this batch deadline',seconds=time.monotonic()-began)
    if result.returncode or not output.is_file():
        return dict(state='failed',reason='Native preparation exited without a complete receipt',seconds=time.monotonic()-began)
    if output.stat().st_size>MAX_METADATA:raise ValueError('Native preparation receipt is oversized')
    return dict(state='prepared',result=json.loads(output.read_text()),reason='',seconds=time.monotonic()-began)


def verify_result(task, request, row):
    from .operations import PORTABLE
    result=row['result'];root=Path(request['destination']).resolve()
    names=PORTABLE[task['kind']]|{'entry.json'}
    if (not isinstance(result,dict) or result.get('schema')!='radar-experiment-preparation-worker/v1'
            or result.get('status')!='prepared' or result.get('kind')!=task['kind']
            or result.get('observations_included') is not False or result.get('scientific_result') is not False
            or result.get('output')!=str(root) or not isinstance(result.get('artifacts'),list)
            or len(result['artifacts'])!=len(names) or set(result['artifacts'])!=names):
        raise ValueError('Preparation task returned an unrelated or incomplete receipt')
    paths=[confined(root,name) for name in sorted(names)]
    if sum(p.stat().st_size for p in paths)>RESERVES[task['kind']]-65536:
        raise ValueError('Preparation task exceeded its declared artifact budget')
    manifest_path=root/'entry.json'
    if manifest_path.stat().st_size>MAX_METADATA or file_sha(manifest_path)!=result.get('entry_sha256'):
        raise ValueError('Preparation manifest changed')
    manifest=json.loads(manifest_path.read_text())
    expected=[file_sha(request[k]) for k in ('source','instrument')+
              (('child',) if task['kind']=='structural' else ())]
    if (manifest.get('schema')!='radar-experiment-preparation-entry/v1' or manifest.get('kind')!=task['kind']
            or manifest.get('input_sha256')!=expected or manifest.get('settings')!=request['settings']
            or manifest.get('grid')!=request['grid'] or manifest.get('observations_included') is not False
            or manifest.get('scientific_result') is not False
            or set(manifest.get('artifacts',{}))!=PORTABLE[task['kind']]):
        raise ValueError('Preparation manifest does not bind its captured model and bank')
    for name,checksum in manifest['artifacts'].items():
        path=confined(root,name)
        if file_sha(path)!=checksum:raise ValueError('Preparation array changed before batch publication')
        if path.suffix=='.npz':
            from ..live.array_limits import validate_npz
            validate_npz(path)
    return paths


def execute(value, *, cancel=None, runner=run_task, clock=time.monotonic, environ=None):
    request=validate_request(value);plan_path=Path(request['plan']).resolve();plan=read_plan(plan_path)
    parent=Path(request['source']).resolve();source_sha=file_sha(parent)
    captured={name:file_sha(confined(plan_path.parent,name)) for name in inputs(plan)}
    output=Path(request['destination']).resolve();output.mkdir(parents=True,exist_ok=False)
    budget=allocation(request['workers'],environ=environ);cancel=cancel or threading.Event()
    began=clock();deadline=began+request['seconds'];remaining=list(plan['tasks']);active={};finished={}
    paths=[];used=0;peak=0;stop_reason=''
    def progress():
        write_json(output/'progress.json',dict(total=len(plan['tasks']),terminal=sum(v['state']!='pending' for v in finished.values()),
            prepared=sum(v['state']=='prepared' for v in finished.values()),active=len(active),
            pending=len(remaining),allocation=budget,peak_admitted_tasks=peak))
    with ThreadPoolExecutor(max_workers=budget['workers']) as pool:
        while remaining or active:
            while remaining and len(active)<budget['workers'] and clock()<deadline and not cancel.is_set():
                reserved=sum(RESERVES[t['kind']] for t in active.values())
                index=next((i for i,t in enumerate(remaining) if used+reserved+RESERVES[t['kind']]<=RESULT_BUDGET),None)
                if index is None:
                    stop_reason='batch_artifact_budget';break
                task=remaining.pop(index);payload=copy.deepcopy(task['request'])
                for role in ['source','instrument']+(['child'] if task['kind']=='structural' else []):
                    payload[role]=str(confined(plan_path.parent,payload[role]))
                if file_sha(payload['source'])!=source_sha:raise ValueError('Batch parent CIF changed between entries')
                payload['destination']=str(confined(output,payload['destination'],exists=False))
                scratch=output/'.work'/task['entry']/('attempt-'+str(task['attempt']))
                future=pool.submit(runner,task,payload,scratch,cancel=cancel,timeout=max(1.,deadline-clock()))
                active[future]=dict(task,payload=payload,scratch=scratch);peak=max(peak,len(active))
            progress()
            if not active:break
            ready,_=wait(active,timeout=.1,return_when=FIRST_COMPLETED)
            for future in ready:
                task=active.pop(future)
                try:
                    row=future.result()
                    if not isinstance(row,dict) or row.get('state') not in {'prepared','failed','cancelled'}:
                        raise ValueError('Invalid native preparation task status')
                    if row['state']=='prepared':
                        assets=verify_result(task,task['payload'],row)
                        paths.extend(assets);used+=sum(p.stat().st_size for p in assets)
                except Exception as exc:
                    row=dict(state='failed',reason='Preparation task failed verification ('+type(exc).__name__+')')
                log=task['scratch']/'native.log'
                if log.is_file():
                    target=output/'logs'/(task['entry']+'-'+str(task['attempt'])+'.log');target.parent.mkdir(exist_ok=True)
                    with log.open('rb') as stream:
                        stream.seek(max(0,log.stat().st_size-65536));target.write_bytes(stream.read(65536))
                    paths.append(target);used+=target.stat().st_size
                finished[task['entry']]=dict(entry=task['entry'],attempt=task['attempt'],**row)
        if cancel.is_set():stop_reason='cancelled_before_start'
        elif clock()>=deadline:stop_reason='batch_time_budget'
    if captured!={name:file_sha(confined(plan_path.parent,name)) for name in captured}:
        raise ValueError('Preparation inputs changed during batch execution')
    for task in remaining:
        finished[task['entry']]=dict(entry=task['entry'],attempt=task['attempt'],state='pending',
            reason=stop_reason or 'not_started')
    counts={name:sum(row['state']==name for row in finished.values())
            for name in ('prepared','failed','cancelled','pending')}
    result=dict(schema=RESULT_SCHEMA,plan_sha256=digest(plan),status='partial' if remaining else 'complete',
        counts=counts,all_prepared=counts['prepared']==len(plan['tasks']),
        entries=[finished[t['entry']] for t in plan['tasks']],allocation=dict(budget,peak_admitted_tasks=peak,
            result_byte_budget=RESULT_BUDGET),output=str(output),observations_included=False,scientific_result=False,
        artifacts=sorted(str(p.relative_to(output)) for p in paths),seconds=clock()-began)
    write_json(output/'batch.json',result);result['artifacts']+=['batch.json','progress.json']
    progress()
    return result
