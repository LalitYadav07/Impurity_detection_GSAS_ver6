"""Planner adapters for durable Galaxy native jobs and explicit local testing.

NOVA validates/captures text and verifies outputs; only the native worker loads
GPX files. A hash in an uploaded plan is never authority to execute that GPX.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid

from gpx_recipe.execution import write_json
from gpx_recipe.planning.managed import (SCHEMA, RESULT_SCHEMA, OPERATIONS, NATIVE,
    TEXT_SUFFIXES, MAX_INPUT_BYTES, file_references, read_plan, confined)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt_identity(value):
    """Compare science metadata while excluding relocated paths/JSON encoding."""
    if isinstance(value,list):return [receipt_identity(v) for v in value]
    if not isinstance(value,dict):return value
    result={}
    for key,item in value.items():
        if key in {'gpx','cif','path','profile','directory','output','source_gpx','captured'}:
            result[key]='<relocated path>' if isinstance(item,str) else receipt_identity(item)
        elif key=='artifacts':
            result[key]={p:sha for p,sha in item.items() if not p.endswith('.json')}
        else:result[key]=receipt_identity(item)
    return result


def relocate(value, old, new):
    if isinstance(value,dict):return {k:relocate(v,old,new) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [relocate(v,old,new) for v in value]
    if isinstance(value,str) and (value==str(old) or value.startswith(str(old)+'/')):
        return str(new)+value[len(str(old)):]
    return value


def capture(operation, request, directory, *, check_text, check_native):
    """Copy only authorized files, without native parsing or arbitrary paths."""
    if operation not in OPERATIONS:raise ValueError('Unsupported planning operation')
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    request=copy.deepcopy(request);files={};copies={}
    for container,key,kind in file_references(operation,request):
        source=Path((check_native if kind=='native' else check_text)(container[key])).resolve(strict=True)
        if source.is_symlink() or source.suffix.lower() not in (TEXT_SUFFIXES if kind=='text' else {'.gpx'}):
            raise ValueError('Unsupported planning input type')
        if not 0<source.stat().st_size<=MAX_INPUT_BYTES:raise ValueError('Planning input exceeds the file limit')
        if source not in copies:
            name=f'assets/{len(copies):02d}'+source.suffix.lower()
            target=directory/name;target.parent.mkdir(exist_ok=True)
            before=sha(source);shutil.copyfile(source,target)
            if sha(target)!=before or sha(source)!=before:raise ValueError('Planning input changed while being captured')
            copies[source]=name;files[name]=before
        container[key]=copies[source]
    if not files:raise ValueError('A planner operation requires a captured source')
    document=dict(schema=SCHEMA,operation=operation,request=request,files=files)
    write_json(directory/'plan.json',document)
    read_plan(directory/'plan.json',operation)
    # A text CIF is the transport anchor for simulation. Native operations use
    # their verified worker-produced GPX. The wrapper treats this input as bytes.
    source_name=request.get('cif') or request.get('source_gpx')
    return document,dict(source=str(directory/source_name),plan=str(directory/'plan.json'),
        children=[str(directory/name) for name in files if name!=source_name],
        destination=str(directory/'result'))


def publish_result(wrapper, directory, operation):
    if wrapper.get('schema')!=RESULT_SCHEMA or wrapper.get('operation')!=operation:
        raise ValueError('The planner worker returned a different operation')
    directory=Path(directory).resolve();origin=Path(wrapper['output']).resolve()
    if not origin.is_relative_to(directory):raise ValueError('Planner worker output escaped its collected job')
    result=wrapper['result']
    if not isinstance(result,dict):raise ValueError('Planner worker result must be a JSON object')
    # The outer native result has already checked every byte. Keep the local
    # receipt's paths coherent after relocation, without deserializing a GPX.
    write_json(origin/'result.json',result)
    return result,origin


class GalaxyPlanningRuntime:
    kind='galaxy'

    def __init__(self, native, root, *, check_text, check_native, progress=None, checkpoint=None):
        self.native=native;self.service=native.service;self.root=Path(root).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.check_text=check_text;self.check_native=check_native
        self.progress=progress or (lambda **_:None)
        self.checkpoint=checkpoint
        self.artifacts={}
        self.verified_results={}

    def _notice(self, handle, state, message='', *, cancel=None):
        job=handle.get('job') or {}
        self.progress(operation=handle['request']['operation'],state=state,
            job_id=job.get('galaxy_job_id',''),message=message,cancel=cancel)

    def execute(self, operation, request, directory, cancel, *, parent=False):
        from . import native_operations
        directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
        packet=directory/'packet'
        document,payload=capture(operation,request,packet,check_text=self.check_text,check_native=self.check_native)
        source_path=Path(payload['source']);checksum=sha(source_path)
        with self.native._lock:
            sources=self.native.state.setdefault('planner_sources',{})
            source_id=sources.get(checksum)
            if source_id is None:
                source_id=self.service.upload_document(source_path,label='Planner source '+checksum,wait_ready=True)
                sources[checksum]=source_id;self.native._save()
        source=dict(id='planner-'+checksum,source_path=str(source_path),source_sha256=checksum,dataset_id=source_id)
        pending=directory/'operation.json'
        def intent(handle):
            write_json(pending,dict(schema='radar-planning-operation/v1',handle=handle))
            if self.checkpoint:self.checkpoint(handle)
            self._notice(handle,'submitting',cancel=cancel)
        handle=native_operations.submit(self.native,operation,payload,packet,source=source,
            timeout=300 if operation in {'planning-parent','planning-simulate'} else 180,
            cancel=cancel,on_intent=intent)
        return self.recover(handle,directory,cancel)

    def recover(self, handle, directory, cancel):
        """Observe/collect an acknowledged or uncertain attempt; never submit."""
        from . import native_operations
        handle=copy.deepcopy(native_operations.validate_handle(self.native,handle))
        operation=handle['request']['operation']
        if operation not in OPERATIONS:raise ValueError('This handle is not a planner operation')
        directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
        pending=directory/'operation.json'
        write_json(pending,dict(schema='radar-planning-operation/v1',handle=handle))
        cancelled=False;started=time.monotonic()
        while True:
            if cancel.is_set() and not cancelled:
                job=native_operations.recover_job(self.native,handle)
                if job is not None:self.native.cancel_job(job,handle['key'])
                cancelled=True;self._notice(handle,'cancelling',cancel=cancel)
            if cancelled:raise RuntimeError('Planner calculation cancelled; its Galaxy job was asked to stop')
            state=native_operations.poll(self.native,handle);handle=state['handle']
            write_json(pending,dict(schema='radar-planning-operation/v1',handle=handle))
            self._notice(handle,state['state'],state.get('message',''),cancel=cancel)
            if state['state']=='finished':break
            if state['state'] in {'failed','cancelled','needs_recovery','not_submitted'}:
                raise RuntimeError(state.get('message') or 'The managed planner job '+state['state'])
            if time.monotonic()-started>self.native.timeout:
                raise RuntimeError('The Galaxy planner job remains pending. Recover its saved operation; no duplicate was submitted.')
            cancel.wait(self.native.poll_seconds)
        if cancel.is_set():raise RuntimeError('Planner calculation cancelled before collecting its output')
        collected=directory/('collected-'+uuid.uuid4().hex)
        wrapper,seconds=native_operations.collect(self.native,handle,collected)
        result,origin=publish_result(wrapper,collected,operation)
        target=directory/('native' if operation in NATIVE else 'simulation')
        if target.exists():
            for path in origin.rglob('*'):
                if path.is_file() and path.name!='result.json':
                    previous=target/path.relative_to(origin)
                    if not previous.is_file() or sha(path)!=sha(previous):
                        raise ValueError('A different planner output already occupies this operation')
        else:shutil.copytree(origin,target)
        result=relocate(result,origin,target)
        write_json(target/'result.json',result)
        for path in target.rglob('*'):
            if not path.is_file() or path.suffix.lower() not in {'.gpx','.npz'}:continue
            self.artifacts[str(path.resolve())]=dict(sha256=sha(path),handle=copy.deepcopy(handle),
                relative=str((origin/path.relative_to(target)).relative_to(collected)))
        self._notice(handle,'finished',f'Native calculation: {seconds:.2f} s',cancel=cancel)
        return result

    def verify_artifacts(self, records, root):
        """Recollect original owned jobs; uploaded hashes alone grant no trust."""
        from . import native_operations
        root=Path(root);cache={};trusted={}
        for filename,record in records.items():
            path=Path(filename).resolve(strict=True)
            if sha(path)!=record.get('sha256'):raise ValueError('Saved native planner artifact changed')
            handle=native_operations.validate_handle(self.native,record['handle'])
            if not handle['request']['operation'].startswith('planning-'):
                raise ValueError('A saved planner artifact names another operation')
            key=handle['key']
            if key not in cache:
                target=root/('verify-'+key+'-'+uuid.uuid4().hex)
                original,_=native_operations.collect(self.native,handle,target)
                cache[key]=target
                self.verified_results[key]=(target,original)
            original=confined(cache[key],record['relative'])
            if sha(original)!=record['sha256']:
                raise ValueError('Saved GPX differs from its original Galaxy planner job')
            trusted[str(path)]=copy.deepcopy(record)
        self.artifacts.update(trusted)
        return trusted

    def verify_result(self, root, records):
        root=Path(root)
        record=records.get(str((root/'curves.npz').resolve()))
        if not record:raise ValueError('The saved preview has no original worker array receipt')
        original_root,wrapper=self.verified_results[record['handle']['key']]
        actual=json.loads((root/'result.json').read_text())
        if receipt_identity(actual)!=receipt_identity(wrapper['result']):
            raise ValueError('Saved preview metadata differs from its original worker result')
        scenario_root=original_root/Path(record['relative']).parent
        return json.loads((scenario_root/'scenario.json').read_text()),scenario_root


class LocalPlanningRuntime:
    """Explicit developer route; never selected merely because Galaxy failed."""
    kind='local'

    def __init__(self, *, check_text, check_native):
        self.check_text=check_text;self.check_native=check_native;self.artifacts={}

    def execute(self, operation, request, directory, cancel, **_):
        from gpx_recipe.jobs import scientific_environment
        from gpx_recipe.process import run_scientific
        directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
        document,payload=capture(operation,request,directory/'packet',check_text=self.check_text,check_native=self.check_native)
        output=directory/('native' if operation in NATIVE else 'simulation');payload['destination']=str(output)
        request_path=directory/'request.json';response=directory/'response.json';write_json(request_path,payload)
        python=os.environ.get('RADAR_PD_PLANNING_PYTHON') or os.environ.get('RADAR_PD_SYMMETRY_PYTHON') or sys.executable
        environment=scientific_environment(os.environ.get('GSASII_ROOT'))
        if os.environ.get('RADAR_PD_LOCAL_EXPERIMENTAL_BACKEND')=='1':environment['RADAR_PD_LOCAL_EXPERIMENTAL_BACKEND']='1'
        environment['MPLCONFIGDIR']=str(directory/'matplotlib')
        with (directory/'worker.log').open('w') as log:
            status=run_scientific([python,'-m','gpx_recipe.planning.managed_worker',operation,str(request_path),str(response)],
                cwd=directory,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=300,cancel_event=cancel)
        if status.returncode or not response.is_file():raise ValueError('The local planner worker stopped; inspect its retained log')
        result,_=publish_result(json.loads(response.read_text()),directory,operation)
        for path in output.rglob('*'):
            if not path.is_file() or path.suffix.lower() not in {'.gpx','.npz'}:continue
            self.artifacts[str(path.resolve())]=dict(sha256=sha(path),local=True)
        return result
