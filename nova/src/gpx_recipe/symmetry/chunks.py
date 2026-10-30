"""A bounded chunk of independent, fresh-process native trials.

Common inputs are captured once. No network, LLM or credential is needed.
Unstarted trials at the chunk's soft deadline remain pending, never failures.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

from ..execution import write_json
from ..jobs import scientific_environment
from ..process import run_scientific, ScientificCancelled
from .compiler import compile_shared_job,prepare_shared_inputs,package_signature
from .contracts import digest,iter_trials,trial_id,validate_protocol
from .campaign import verify_receipt
from .resources import allocation


def validate_request(request):
    required={'schema','protocol','protocol_sha256','trials','workers','dispatch_key','seconds'}
    optional={'template','source_identity','origin'}
    if not isinstance(request,dict) or set(request)-required-optional or required-set(request) or request['schema']!='radar-model-chunk/v1':
        raise ValueError('Invalid model chunk request')
    protocol=validate_protocol(request['protocol'])
    if digest(protocol)!=request['protocol_sha256'] or protocol['implementation_sha256']!=package_signature():
        raise ValueError('The chunk does not bind this approved protocol and worker image')
    if 'template_sha256' in protocol:
        from .templates import validate_template
        if optional-set(request) or digest(validate_template(request['template']))!=protocol['template_sha256']:
            raise ValueError('The chunk lost its reviewed acquisition template')
    if not isinstance(request['trials'],list) or not 1<=len(request['trials'])<=256 or len(set(request['trials']))!=len(request['trials']):
        raise ValueError('Declare one to 256 unique starts in a chunk')
    expected={trial_id(protocol,c['id'],s['id']):(c['id'],s['id']) for c,s in iter_trials(protocol)}
    if any(key not in expected for key in request['trials']):raise ValueError('A chunk trial is outside the approved finite search')
    import re
    if not isinstance(request['dispatch_key'],str) or not re.fullmatch('[a-f0-9]{24}',request['dispatch_key']):
        raise ValueError('Invalid chunk dispatch identity')
    if type(request['seconds']) is not int or not 30<=request['seconds']<=1800:
        raise ValueError('Declare a bounded chunk walltime')
    return protocol,expected


def run_trial(directory, *, cancel):
    directory=Path(directory)
    started=time.monotonic()
    job=json.loads((directory/'job.json').read_text())
    with (directory/'native.log').open('w') as log:
        try:
            process=run_scientific([sys.executable,str(directory/'worker.py')],
                timeout=job['protocol']['policy']['timeout_seconds'],cancel_event=cancel,
                cwd=directory,env=scientific_environment(),stdout=log,stderr=subprocess.STDOUT)
        except ScientificCancelled:return dict(state='cancelled')
        except subprocess.TimeoutExpired:
            process=None
        if not (directory/'result.json').exists():
            write_json(directory/'result.json',dict(schema='radar-symmetry-trial/v1',id=job['id'],
                protocol_sha256=job['protocol_sha256'],llm_calls=0,
                status='timed_out' if process is None else 'technical_failure',
                reason='Native start exceeded its declared deadline' if process is None else 'Native start exited without a terminal receipt'))
    result=verify_receipt(directory,dict(id=job['id'],protocol_sha256=job['protocol_sha256']))
    return dict(state='finished',status=result['status'],wall_seconds=time.monotonic()-started,
        rwp=result.get('rwp'))


def execute_chunk(request, input_root, directory, *, cancel=None, runner=run_trial, clock=time.monotonic, environ=None):
    protocol,expected=validate_request(request)
    budget=allocation(request['workers'],environ=environ)
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    captured=prepare_shared_inputs(protocol,input_root,directory)
    if 'template_sha256' in protocol:
        from .native_audit import inspect_source
        from .templates import rebind
        context=inspect_source(captured/protocol['source']['path'])
        if rebind(request['template'],context,request['source_identity'],request['origin'])!=protocol:
            raise ValueError('Native source compatibility differs from the approved enrollment')
    cancel=cancel or threading.Event()
    remaining=list(request['trials']);active={};done=[];cancelled=[]
    for key in remaining:
        candidate,seed=expected[key]
        compile_shared_job(protocol,directory/'trials'/key,candidate_id=candidate,seed_id=seed,approved_hash=digest(protocol))
    started=clock();deadline=started+request['seconds']
    write_json(directory/'progress.json',dict(completed=0,total=len(remaining),active=0,pending=len(remaining)))
    with ThreadPoolExecutor(max_workers=budget['workers']) as pool:
        while remaining or active:
            while remaining and len(active)<budget['workers'] and clock()<deadline and not cancel.is_set():
                key=remaining.pop(0);active[pool.submit(runner,directory/'trials'/key,cancel=cancel)]=key
            if not active:break
            ready,_=wait(active,timeout=.1,return_when=FIRST_COMPLETED)
            for future in ready:
                key=active.pop(future)
                try:result=future.result()
                except Exception as exc:
                    result=dict(state='needs_recovery',reason=type(exc).__name__+': '+str(exc))
                if result['state']=='cancelled':cancelled.append(key)
                done.append(dict(id=key,**result))
            write_json(directory/'progress.json',dict(completed=len(done),total=len(request['trials']),active=len(active),pending=len(remaining)))
    response=dict(schema='radar-model-chunk-result/v1',dispatch_key=request['dispatch_key'],
        protocol_sha256=digest(protocol),trials=done,pending=remaining,cancelled=cancelled,
        allocation=budget,wall_seconds=clock()-started,llm_calls=0)
    write_json(directory/'response.json',response)
    return response
