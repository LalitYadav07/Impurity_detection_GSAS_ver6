"""Durable named native operations using the existing Galaxy model transport.

Handles contain only scientific requests, immutable input hashes and Galaxy
identities. Polling an uncertain submission never creates another job.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import re
import shutil
import threading
import uuid

from gpx_recipe.execution import write_json
from gpx_recipe.symmetry.compiler import assets, confined, file_sha, package_signature
from gpx_recipe.symmetry.contracts import digest
from gpx_recipe.symmetry.native_protocol import validate_native_payload
from .model_checkpoint import job_record
from .models import UtilityActionRecord, RunStatus
from .refinement_runtime import RuntimeUnavailable

SCHEMA = 'radar-model-native-handle/v1'


def validate_handle(runtime, value):
    fields = {'schema','history_id','key','source','request','asset_id','job','submission_started','identity'}
    if (not isinstance(value, dict) or set(value) != fields or value['schema'] != SCHEMA
            or value['history_id'] != runtime.service.history_id
            or not isinstance(value['key'], str) or not re.fullmatch('[a-f0-9]{24}', value['key'])
            or type(value['submission_started']) is not bool):
        raise ValueError('Native operation handle belongs to another workspace or schema')
    source, request = value['source'], value['request']
    if (not isinstance(source, dict) or set(source)-{'id','source_sha256','dataset_id','record_uid'}
            or not {'id','source_sha256','dataset_id'} <= set(source)
            or not isinstance(source['id'], str) or not 1 <= len(source['id']) <= 300
            or not isinstance(source['source_sha256'], str)
            or not re.fullmatch('[a-f0-9]{64}', source['source_sha256'])
            or not isinstance(source['dataset_id'], str)
            or not re.fullmatch('[A-Za-z0-9_-]{1,128}', source['dataset_id'])
            or ('record_uid' in source and (not isinstance(source['record_uid'], str)
                or not 1 <= len(source['record_uid']) <= 300))
            or not isinstance(request, dict) or request.get('schema') != 'radar-model-native/v1'
            or request.get('dispatch_key') != value['key']
            or request.get('expected_source_sha256') != source['source_sha256']):
        raise ValueError('Native handle source or request identity changed')
    if (set(request) != {'schema','operation','payload','implementation_sha256',
            'expected_source_sha256','input_hashes','timeout','dispatch_key'}
            or not isinstance(request['input_hashes'], dict)
            or not re.fullmatch('[a-f0-9]{64}', str(request['implementation_sha256']))
            or isinstance(request['timeout'], bool) or not isinstance(request['timeout'], (int,float))
            or not math.isfinite(request['timeout']) or not 1 <= request['timeout'] <= 600):
        raise ValueError('Native handle contains an invalid scientific request')
    validate_native_payload(request.get('operation'), request.get('payload'))
    identity = digest(dict(history_id=value['history_id'], source=source, request=request))
    if identity != value['identity'] or len(json.dumps(value, allow_nan=False)) > 8*1024*1024:
        raise ValueError('Native operation handle changed or exceeds its metadata budget')
    if value['asset_id'] is not None and (not isinstance(value['asset_id'], str)
            or not re.fullmatch('[A-Za-z0-9_-]{1,128}', value['asset_id'])):
        raise ValueError('Invalid native scientific input packet identity')
    if value['job'] is not None:
        job = UtilityActionRecord.model_validate(value['job'])
        if (set(value['job']) != set(job_record(job))
                or job.tool_id != runtime.tool_id or not job.galaxy_job_id):
            raise ValueError('Native handle names another scientific tool')
    return value


def handle_path(root, key):
    if not isinstance(key, str) or not re.fullmatch('[a-f0-9]{24}', key):
        raise ValueError('Invalid native dispatch identity')
    root = Path(root).resolve();directory=root/'native-dispatches'
    if directory.is_symlink() or not directory.resolve().is_relative_to(root):
        raise ValueError('Native dispatch directory escaped its owned workspace')
    directory.mkdir(parents=True, exist_ok=True)
    return directory/(key+'.json')


@contextmanager
def intent_lock(path):
    import fcntl
    with path.with_suffix('.lock').open('a+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def capture(runtime, operation, payload, root, source, timeout, preparation_attempt=None):
    """Use the same confined packet and request rules for all native clients."""
    from .model_runtime import EMPTY_PACKET_NAME, EMPTY_PACKET_SHA256
    validate_native_payload(operation, payload)
    root=Path(root).resolve();root.mkdir(parents=True, exist_ok=True)
    payload=copy.deepcopy(payload);files=[]
    for field in ['source','child','instrument','plan','anchor_snapshot','snapshot']:
        if field in payload:
            path=Path(payload[field]).resolve();relative=str(path.relative_to(root))
            files.append(relative);payload[field]=relative
    if operation=='preparation-batch':
        from gpx_recipe.preparation.batch import read_plan,inputs
        plan_path=confined(root,payload['plan']);plan=read_plan(plan_path)
        for name in inputs(plan):
            path=confined(plan_path.parent,name)
            files.append(str(path.relative_to(root)))
        if float(timeout)<payload['seconds']+15:
            raise ValueError('Preparation batch timeout must include its declared work and cleanup allowance')
    if 'children' in payload:
        children=[str(Path(p).resolve().relative_to(root)) for p in payload['children']]
        payload['children']=children;files.extend(children)
    for field in ['destination','temporary']:
        if field in payload:payload[field]=str(Path(payload[field]).resolve().relative_to(root))
    directories=payload.get('directories') or ([str(Path(payload['directory']).resolve().relative_to(root))] if 'directory' in payload else [])
    if 'directory' in payload:payload['directory']=directories[0]
    for relative in directories:
        directory=confined(root,relative);job=json.loads((directory/'job.json').read_text())
        asset_root=directory
        if operation=='structural-snapshot':
            from gpx_recipe.structural_live.job import reviewed_job
            from gpx_recipe.symmetry.compiler import job_inputs
            _,protocol,_=reviewed_job(directory)
            asset_root=job_inputs(directory,job)
            if (file_sha(confined(root,payload['source']))!=protocol['source']['sha256']
                    or float(timeout)<protocol['policy']['structural_seconds']+
                        protocol['policy']['structural_recovery_seconds']+15):
                raise ValueError('Structural snapshot source or native cleanup allocation differs from its reviewed scope')
            if (file_sha(confined(root,payload['anchor_snapshot']))!=payload['mapping']['snapshot_sha256']
                    or file_sha(confined(root,payload['snapshot']))!=payload['receipt']['content_sha256']):
                raise ValueError('Structural snapshot bytes differ from their reviewed receipt')
            from gpx_recipe.preparation.structural_hints import read as read_preparation
            try:prepared=read_preparation(asset_root,protocol)
            except (OSError,ValueError,EOFError):prepared=None
            if 'prepared_scattering' in payload and payload['prepared_scattering']!=prepared:
                raise ValueError('Structural preparation metadata differs from its captured model inputs')
            if prepared is not None:payload['prepared_scattering']=prepared
        files.append(str((directory/'job.json').relative_to(root)))
        for asset in assets(job['protocol']):files.append(str(confined(asset_root,asset['path']).relative_to(root)))
        from gpx_recipe.preparation.hints import packet_paths
        try:extra=packet_paths(asset_root,job['protocol'])
        except (OSError,ValueError,EOFError):extra=[]
        files.extend(str(confined(asset_root,name).relative_to(root)) for name in extra)
    if not files:
        write_json(root/'packet.json',{'schema':'radar-native-input-marker/v1'})
        files=['packet.json']
    source=runtime.resolve(source)
    if operation=='structural-snapshot' and file_sha(confined(root,payload['source']))!=source['source_sha256']:
        raise ValueError('Structural snapshot belongs to another Galaxy source revision')
    request=dict(schema='radar-model-native/v1',operation=operation,payload=payload,
        implementation_sha256=package_signature(),expected_source_sha256=source['source_sha256'],
        input_hashes={p:file_sha(confined(root,p)) for p in files},timeout=min(600,max(1,float(timeout))))
    if all(Path(p).suffix.lower()=='.gpx' and sha==source['source_sha256']
           for p,sha in request['input_hashes'].items()):
        request['input_hashes'][EMPTY_PACKET_NAME]=EMPTY_PACKET_SHA256
    key=digest([runtime.service.history_id,source['id'],request])[:24]
    if preparation_attempt is not None:
        from .model_preparation_retry import inspection_request, validate_attempt
        if operation!='inspect' or request!=inspection_request(source):
            raise RuntimeUnavailable('Preparation retries only support the unchanged captured source inspection')
        validate_attempt(runtime,source,request,preparation_attempt)
        key=preparation_attempt['dispatch_key']
    request['dispatch_key']=key
    descriptor={k:source[k] for k in ('id','source_sha256')}
    descriptor['dataset_id']=str(runtime.dataset_id(source))
    if source.get('record_uid'):descriptor['record_uid']=source['record_uid']
    handle=dict(schema=SCHEMA, history_id=runtime.service.history_id,key=key,source=descriptor,
        request=request,asset_id=None,job=None,submission_started=False,
        identity=digest(dict(history_id=runtime.service.history_id,source=descriptor,request=request)))
    return validate_handle(runtime, handle), files


def submit(runtime, operation, payload, root, *, source, timeout=120, cancel=None,
           preparation_attempt=None, before_submit=None, preparation_guard=None, on_intent=None):
    event=cancel or threading.Event()
    root=Path(root).resolve()
    with runtime._event_lock:
        if str(root) in runtime._cancelled_roots:
            raise RuntimeUnavailable('This model preparation was cancelled')
    handle, files=capture(runtime,operation,payload,root,source,timeout,preparation_attempt)
    path=handle_path(root,handle['key'])
    with intent_lock(path):
        if path.exists():
            if path.stat().st_size>8*1024*1024:raise ValueError('Saved native operation exceeds its metadata budget')
            saved=validate_handle(runtime,json.loads(path.read_text()))
            if saved['identity']!=handle['identity']:raise ValueError('The saved native input scope changed')
            handle=saved
            if handle['job'] is not None or handle['submission_started']:
                # Find/poll the captured attempt; never infer absence from a
                # missing acknowledgement or from a new viewer's empty ledger.
                recover_job(runtime,handle)
                write_json(path,handle)
                return handle
        else:
            write_json(path,handle)
        if event.is_set():raise RuntimeUnavailable('Model preparation paused before dispatch')
        handle['asset_id']=runtime.packet(root,files,source_sha256=handle['source']['source_sha256'])
        handle['submission_started']=True
        write_json(path,handle)
        try:
            if on_intent is not None:on_intent(copy.deepcopy(handle))
        except Exception:
            # The POST has not been entered. This is a verified pre-submission
            # failure, unlike a lost job acknowledgement.
            handle['submission_started']=False;write_json(path,handle)
            raise
        try:
            if preparation_attempt is not None:
                from .model_preparation_retry import submit_attempt
                job=submit_attempt(runtime,handle['source'],handle['request'],handle['asset_id'],
                    preparation_attempt,before_submit,event,preparation_guard)
            else:
                job=runtime._submit('native',handle['source'],handle['request'],handle['asset_id'],
                    handle['key'],cancel=event)
        except Exception:
            # The existing dispatcher records pending=True before its POST.
            # Its explicit false value proves this attempt did not submit.
            entry=runtime.state.get('dispatches',{}).get(handle['key'])
            if entry is None or (entry.get('pending') is False and not entry.get('job')):
                handle['submission_started']=False;write_json(path,handle)
            raise
        handle['job']=job_record(job)
        write_json(path,handle)
        return handle


def recover_job(runtime, handle):
    validate_handle(runtime,handle)
    job=UtilityActionRecord.model_validate(handle['job']) if handle['job'] else None
    if job is None:
        existing=runtime.state.get('dispatches',{}).get(handle['key'],{}).get('job')
        job=UtilityActionRecord.model_validate(existing) if existing else runtime.find_dispatch(handle['key'])
    if job is not None:
        runtime.verify_job(job,handle['key'])
        handle['job']=job_record(job)
    return job


def poll(runtime, handle):
    handle=copy.deepcopy(validate_handle(runtime,handle))
    job=recover_job(runtime,handle)
    if job is None:
        return dict(state='needs_recovery' if handle['submission_started'] else 'not_submitted',
            handle=handle,message='No acknowledged job was found; no duplicate was submitted.')
    runtime.service.refresh_utility(job)
    handle['job']=job_record(job)
    state={RunStatus.OK:'finished',RunStatus.ERROR:'failed',RunStatus.CANCELLED:'cancelled'}.get(job.status,job.status.value)
    return dict(state=state,handle=handle,message=job.message or '')


def restore_inputs(runtime, handle, root):
    """Restore the exact captured packet, independent of current plan builders."""
    from gpx_recipe.symmetry.bundle import unpack, hydrate_source, MAX_BYTES
    handle=copy.deepcopy(validate_handle(runtime,handle))
    job=recover_job(runtime,handle)
    if job is None:raise RuntimeUnavailable('Recover the native job before restoring its captured inputs')
    if not handle['asset_id']:raise ValueError('Native handle lost its captured scientific input packet')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    owned=runtime.root/'native-inputs'/(handle['key']+'-'+uuid.uuid4().hex)
    owned.mkdir(parents=True)
    archive=owned/'inputs.zip'
    try:
        runtime.service._download_dataset(handle['asset_id'],archive,maximum_bytes=MAX_BYTES)
        received=unpack(archive,owned/'unpacked')
        hashes=handle['request']['input_hashes']
        missing=[name for name in hashes if not (received/name).is_file()]
        if missing:
            source=runtime.resolve(handle['source'])
            hydrate_source(received,source['source_path'],hashes,handle['source']['source_sha256'])
        actual={str(p.relative_to(received)):file_sha(p) for p in received.rglob('*') if p.is_file()}
        if actual!=hashes:raise ValueError('Recovered native inputs differ from the captured request')
        copies=[]
        for name,checksum in hashes.items():
            target=confined(root,name,exists=False)
            if target.exists() and file_sha(target)!=checksum:
                raise ValueError('A different input already occupies this immutable native attempt')
            copies.append((confined(received,name),target,checksum))
        for path,target,checksum in copies:
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copyfile(path,target)
            if file_sha(target)!=checksum:raise ValueError('Native input changed during recovery')
        return {name:confined(root,name) for name in hashes}
    finally:
        shutil.rmtree(owned)


def collect(runtime, handle, root, *, immutable=True):
    handle=copy.deepcopy(validate_handle(runtime,handle));job=recover_job(runtime,handle)
    if job is None:raise RuntimeUnavailable('This native operation still needs job recovery')
    runtime.service.refresh_utility(job)
    if job.status!=RunStatus.OK:
        raise RuntimeUnavailable('Collect native output only after its Galaxy job completes successfully')
    root=Path(root).resolve();request=handle['request'];key=handle['key']
    received=runtime.result(job,runtime.root/'native-results'/(key+'-'+uuid.uuid4().hex))
    response=json.loads((received/'response.json').read_text())
    if (response.get('schema')!='radar-model-native-result/v1' or response.get('dispatch_key')!=key
            or response.get('request_sha256')!=digest(request)
            or response.get('source_sha256')!=handle['source']['source_sha256']):
        raise ValueError('Native model result belongs to another request/source revision')
    copies={}
    for relative,sha in response['files'].items():
        path=confined(received,relative)
        if file_sha(path)!=sha:raise ValueError('A native model output changed')
        parts=Path(relative).parts
        if len(parts)<2 or parts[0] not in {'native','inputs-packet'}:raise ValueError('Unexpected native output location')
        target=confined(root,str(Path(*parts[1:])),exists=False)
        target.parent.mkdir(parents=True,exist_ok=True)
        if immutable and target.exists() and file_sha(target)!=sha:
            raise ValueError('A different native output already occupies this immutable attempt')
        if target in copies and copies[target][1]!=sha:
            raise ValueError('Native outputs collide after relocation')
        copies[target]=(path,sha)
    for target,(path,sha) in copies.items():
        shutil.copyfile(path,target)
        if file_sha(target)!=sha:raise ValueError('A native model output changed during collection')
    def relocate(value):
        if isinstance(value,list):return [relocate(v) for v in value]
        if isinstance(value,dict):return {k:relocate(v) for k,v in value.items()}
        if isinstance(value,str):
            for prefix in [response['worker_root'],response['input_root']]:
                if value==prefix or value.startswith(prefix+'/'):return str(root)+value[len(prefix):]
        return value
    seconds=float(response.get('native_seconds',0))
    if not math.isfinite(seconds) or seconds<0:raise ValueError('Invalid native operation timing')
    return relocate(response['result']),seconds
