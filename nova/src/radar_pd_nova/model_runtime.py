"""Galaxy model preparation and chunk transport; credentials stay in NOVA."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import uuid

from gpx_recipe.execution import write_json
from gpx_recipe.symmetry.bundle import pack,unpack
from gpx_recipe.symmetry.compiler import assets,confined,file_sha,job_inputs,package_signature
from gpx_recipe.symmetry.contracts import digest
from .refinement_runtime import GalaxyRefinementRuntime,RuntimeUnavailable
from .models import UtilityActionRecord,RunStatus

MODEL_TOOL_ID='neutrons_radar_pd_model_campaign_prototype'
CAMPAIGN_LABEL='RADAR-PD Model campaign '
MODEL_BATCH_LIMITS=(1,2,4,8)
EMPTY_PACKET_NAME='radar-model-packet.json'
EMPTY_PACKET_BYTES=b'{"schema":"radar-model-packet/v1","files":{}}\n'
EMPTY_PACKET_SHA256=hashlib.sha256(EMPTY_PACKET_BYTES).hexdigest()


class CollectOnlyTransport:
    """Poll existing work without admitting more starts into a freed slot."""
    max_active_jobs=0

    def __init__(self,transport):self.transport=transport
    def __getattr__(self,name):return getattr(self.transport,name)
    def submit_many(self,*args,**kwargs):raise RuntimeError('Collection-only pass cannot submit work')


class PreparationSlotTransport:
    """Keep one model-job slot for an acquisition being inspected."""
    def __init__(self,transport):
        self.transport=transport
    @property
    def max_active_jobs(self):return max(0,self.transport.max_active_jobs-1)
    def __getattr__(self,name):return getattr(self.transport,name)
    def submit_many(self,*args,**kwargs):
        if not self.max_active_jobs:raise RuntimeError('The only model-job slot is reserved for input preparation')
        return self.transport.submit_many(*args,**kwargs)


def identifier(value):
    import re
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',value):raise ValueError('Invalid saved Galaxy dataset/job identity')
    return value


def document_label(name):
    """GalaxyService wraps upload labels as RADAR-PD <label> | <filename>."""
    name=str(name or '')
    return name[len('RADAR-PD '):].split(' | ',1)[0] if name.startswith('RADAR-PD ') and ' | ' in name else name


def active_batch_count(trials):
    return len({json.loads(r['handle']).get('dispatch_key',r['id']) if r.get('handle') else r['id']
                for r in trials if r['state'] in {'running','submitting','cancelling'}})


class GalaxyModelRuntime(GalaxyRefinementRuntime):
    tool_id=MODEL_TOOL_ID
    workspace_schema='radar-model-campaign/v1'

    def __init__(self,root,service,*,poll_seconds=3,timeout=7200,max_active_jobs=1):
        super().__init__(root,service,timeout=timeout,poll_seconds=poll_seconds)
        self.max_active_jobs=max_active_jobs
        self._lock=threading.RLock();self._event_lock=threading.Lock();self._events=set();self._cancelled_roots=set();self._verified_jobs=set()
        self.state_path=self.root/'model-dispatches.json'
        self.state=json.loads(self.state_path.read_text()) if self.state_path.exists() else {'assets':{},'dispatches':{}}

    @property
    def max_active_jobs(self):return self._max_active_jobs

    @max_active_jobs.setter
    def max_active_jobs(self,value):
        if type(value) is not int or value not in MODEL_BATCH_LIMITS:raise ValueError('Choose 1, 2, 4 or 8 concurrent Galaxy model batches')
        self._max_active_jobs=value

    def _save(self):
        with self._lock:write_json(self.state_path,self.state)

    def dataset_id(self,source):return identifier(super().dataset_id(source))

    def stop_preparations(self):
        with self._event_lock:
            for event,root in self._events:event.set();self._cancelled_roots.add(root)

    def allow_preparations(self,root):
        with self._event_lock:self._cancelled_roots={v for v in self._cancelled_roots if not Path(v).is_relative_to(root)}

    def find_dispatch(self,key):
        """Recover through the exact result HDA, avoiding a scan of every job."""
        import requests
        from .galaxy_service import _parameter_value,normalize_status
        label='RADAR-PD model '+key+' results'
        response=requests.get(self.service.galaxy_url+'/api/histories/'+self.service.history_id+'/contents',
            headers=self.service._headers,params={'v':'dev','q':'name-contains','qv':label,'limit':3},timeout=30)
        response.raise_for_status();rows=response.json()
        if not isinstance(rows,list):raise RuntimeUnavailable('Galaxy did not report model dispatch outputs')
        matches={}
        for row in rows:
            if row.get('name')!=label or row.get('history_content_type','dataset')!='dataset':continue
            response=requests.get(self.service.galaxy_url+'/api/histories/'+self.service.history_id+
                '/contents/'+identifier(row['id'])+'/provenance',headers=self.service._headers,params={'follow':'false'},timeout=30)
            response.raise_for_status();provenance=response.json()
            if provenance.get('tool_id')!=self.tool_id:raise RuntimeUnavailable('Model result provenance names another tool')
            job_id=provenance.get('job_id')
            if not job_id:raise RuntimeUnavailable('The model result has no recoverable job identity yet')
            job=self.service._job_details(str(job_id))
            if job.get('tool_id')!=self.tool_id or str(_parameter_value(job.get('params') or job.get('parameters') or {},'dispatch_key') or '')!=key:
                raise RuntimeUnavailable('The recovered model dispatch differs from this request')
            matches[str(job_id)]=job
        if len(matches)>1:raise RuntimeUnavailable('Several Galaxy jobs share this model dispatch; review them before retrying')
        if not matches:return None
        job=next(iter(matches.values()))
        return UtilityActionRecord(uid='model-'+key,tool_id=self.tool_id,name='Recovered model '+key,
            galaxy_job_id=str(job['id']),status=normalize_status(job.get('state')),outputs=self.service._job_output_ids(job))

    def wait_capacity(self,cancel=None):
        """A bounded per-History queue; Galaxy still owns site-wide admission."""
        import requests
        started=time.monotonic()
        while True:
            if cancel and cancel.is_set():raise RuntimeUnavailable('Model submission paused before dispatch')
            response=requests.get(self.service.galaxy_url+'/api/jobs',headers=self.service._headers,
                params={'history_id':self.service.history_id,'tool_id':self.tool_id,
                    'state':['new','waiting','queued','running','resubmitted','paused'],'limit':self.max_active_jobs},timeout=30)
            response.raise_for_status()
            jobs=response.json()
            if not isinstance(jobs,list):raise RuntimeUnavailable('Galaxy did not report the model resource queue')
            if len(jobs)<self.max_active_jobs:return
            if time.monotonic()-started>self.timeout:raise RuntimeUnavailable('Waiting for the existing model job to release its resource slot')
            if cancel:cancel.wait(self.poll_seconds)
            else:time.sleep(self.poll_seconds)

    def _packet_available(self,key):
        if not hasattr(self.service,'_dataset_metadata'):return True
        import requests
        try:metadata=self.service._dataset_metadata(key)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code==404:return False
            raise
        return not metadata.get('deleted') and not metadata.get('purged') and metadata.get('state') not in {'error','discarded','deleted'}

    def packet(self,root,paths,*,source_sha256=None):
        root=Path(root).resolve()
        paths=[p for p in paths if not (source_sha256 and Path(p).suffix.lower()=='.gpx'
                                      and file_sha(confined(root,p))==source_sha256)]
        hashes={p:file_sha(confined(root,p)) for p in sorted(set(paths))}
        if not hashes:
            # Galaxy's archive sniffing rejects a zero-member ZIP. Parent-only
            # inspection still needs a valid packet, without duplicating GPX.
            hashes={EMPTY_PACKET_NAME:EMPTY_PACKET_SHA256}
        signature=digest(hashes)
        with self._lock:
            cached=self.state['assets'].get(signature)
            if cached and self._packet_available(cached):return cached
            self.state['assets'].pop(signature,None)
            label='RADAR-PD Model inputs '+signature
            found=[r for r in self.service.search_history_datasets(query=label,limit=10,include_generated=True)
                   if document_label(r.get('name'))==label and self._packet_available(r['id'])]
            if found:
                self.state['assets'][signature]=found[0]['id'];self._save();return found[0]['id']
            archive=self.root/(signature+'.zip');pack(root,archive,paths)
            if not paths:
                import zipfile
                with zipfile.ZipFile(archive,'a',compression=zipfile.ZIP_DEFLATED) as packet:
                    packet.writestr(EMPTY_PACKET_NAME,EMPTY_PACKET_BYTES)
            try:identifier=self.service.upload_document(archive,label='RADAR-PD Model inputs '+signature,wait_ready=False)
            finally:archive.unlink(missing_ok=True)
            self.state['assets'][signature]=identifier;self._save()
            return identifier

    def _submit(self,action,source,request,asset_id,key,*,cancel=None):
        with self._lock:
            entry=self.state['dispatches'].setdefault(key,{'pending':False})
            if entry.get('job'):return UtilityActionRecord.model_validate(entry['job'])
            entry['absence_verified']=False
            existing=self.find_dispatch(key)
            if existing:
                entry['job']=existing.model_dump(mode='json');self._save();return existing
            if entry.get('pending'):
                raise RuntimeUnavailable('This model submission has no acknowledged job yet. Recover the existing dispatch before retrying; no duplicate was submitted.')
            entry['absence_verified']=True
            self.wait_capacity(cancel)
            capability=getattr(self.service,'tool_has_input',None)
            inline=bool(capability and capability(self.tool_id,'model_request_b64'))
            request_id=None;encoded=None
            if inline:
                import base64
                raw=json.dumps(request,separators=(',',':'),allow_nan=False).encode('utf-8')
                if len(raw)>4*1024*1024:raise ValueError('Model request exceeds the 4 MiB transport bound')
                encoded=base64.b64encode(raw).decode('ascii')
            else:
                request_id=self.service.upload_json_document(request,name='model-request.json',
                    label='RADAR-PD Model request '+key,wait_ready=True)
            if cancel and cancel.is_set():raise RuntimeUnavailable('Model submission paused before dispatch')
            entry.update(pending=True,request_id=request_id,request_sha256=digest(request),
                         request_format='inline_base64' if inline else 'history_json');self._save()
            try:
                inputs={'action':action,'dispatch_key':key,'source_gpx':{'dataset_id':self.dataset_id(source)},
                        'model_assets':{'dataset_id':asset_id}}
                if inline:inputs['model_request_b64']=encoded
                else:inputs['model_request']={'dataset_id':request_id}
                job=self.service.submit_utility(tool_id=self.tool_id,name='RADAR-PD model '+key,
                    associated_run_uid=source.get('record_uid'),inputs=inputs)
            except Exception as exc:
                raise RuntimeUnavailable('Galaxy did not acknowledge the model job. Its stable dispatch key was retained for recovery.') from exc
            entry['job']=job.model_dump(mode='json');self._save()
            self._verified_jobs.add((str(job.galaxy_job_id),key))
            return job

    def verify_job(self,job,key):
        identity=(str(job.galaxy_job_id),key)
        if identity not in self._verified_jobs:
            recovered=self.find_dispatch(key)
            if not recovered or str(recovered.galaxy_job_id)!=identity[0]:raise RuntimeUnavailable('The saved model job does not belong to this History and dispatch')
            self._verified_jobs.add(identity)

    def cancel_job(self,job,key):
        import requests
        self.verify_job(job,key)
        response=requests.delete(self.service.galaxy_url+'/api/jobs/'+identifier(str(job.galaxy_job_id)),headers=self.service._headers,timeout=30)
        response.raise_for_status()

    def failed_request_proof(self,job,key,*,retiring=False):
        """Prove an owned, empty request failed before a runner was assigned.

        Scientific failures, manually paused work and uncertain acknowledgments
        never qualify. A deleted job qualifies only with a saved retirement
        intent, so an ordinary user cancellation cannot restart itself.
        """
        if not hasattr(self.service,'_job_details'):return None
        self.verify_job(job,key)
        details=self.service._job_details(str(job.galaxy_job_id))
        states={'paused','deleted','deleted_new'} if retiring else {'paused'}
        if (details.get('state') not in states or details.get('tool_id')!=self.tool_id
                or details.get('history_id')!=self.service.history_id):return None
        if any(details.get(k) not in (None,'') for k in
               ['external_id','job_runner_name','command_line','command_version','exit_code']):return None
        refs=details.get('inputs') or {}
        if not isinstance(refs,dict):return None
        ids={k:(refs.get(k) or {}).get('id') for k in ['source_gpx','model_assets','model_request']}
        if not all(isinstance(v,str) and v for v in ids.values()):return None
        data={k:self.service._dataset_metadata(identifier(v)) for k,v in ids.items()}
        if any(d.get('deleted') or d.get('purged') for d in data.values()):return None
        if any(data[k].get('state')!='ok' for k in ['source_gpx','model_assets']):return None
        request=data['model_request']
        if (request.get('state')!='error' or request.get('file_size')!=0
                or document_label(request.get('name'))!='RADAR-PD Model request '+key):return None
        return dict(job_id=str(job.galaxy_job_id),dispatch_key=key,inputs=ids,
                    reason='Owned empty request failed before runner assignment')

    def retire_failed_request(self,job,key,proof):
        # Recheck after the durable intent and again after cancellation. No new
        # scientific submission is possible until the old job is terminal.
        if self.failed_request_proof(job,key,retiring=True)!=proof:
            raise RuntimeUnavailable('The failed-input execution proof changed; review this batch')
        details=self.service._job_details(str(job.galaxy_job_id))
        if details.get('state')=='paused':self.cancel_job(job,key)
        details=self.service._job_details(str(job.galaxy_job_id))
        if details.get('state') not in {'deleted','deleted_new'}:
            raise RuntimeUnavailable('Waiting for cancellation acknowledgment before retrying input staging')
        if self.failed_request_proof(job,key,retiring=True)!=proof:
            raise RuntimeUnavailable('The retired batch execution proof changed; review it before retrying')

    def wait_job(self,job,*,key,cancel=None):
        started=time.monotonic()
        while job.status not in {RunStatus.OK,RunStatus.ERROR,RunStatus.CANCELLED}:
            if cancel and cancel.is_set():
                self.cancel_job(job,key);raise RuntimeUnavailable('Model preparation cancelled')
            if time.monotonic()-started>self.timeout:raise RuntimeUnavailable('Galaxy model job is still pending; recover its existing dispatch')
            if cancel:cancel.wait(self.poll_seconds)
            else:time.sleep(self.poll_seconds)
            self.service.refresh_utility(job)
        if job.status==RunStatus.ERROR:raise RuntimeError(job.message or 'The Galaxy model job failed; inspect its console')
        if job.status==RunStatus.CANCELLED:raise RuntimeUnavailable('The native model job was cancelled; no result was accepted')
        return job

    def result(self,job,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        dataset=job.outputs.get('result_bundle')
        if not dataset:raise RuntimeError('Galaxy did not publish a model result bundle')
        archive=directory/'result.zip'
        self.service._download_dataset(identifier(dataset),archive)
        unpack(archive,directory/'unpacked')
        return directory/'unpacked'

    def native(self,operation,payload,root,*,source,timeout=120,cancel=None):
        """Only named scientific files are staged, never backend environments."""
        started=time.monotonic();root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
        event=cancel or threading.Event();token=(event,str(root))
        with self._event_lock:
            if str(root) in self._cancelled_roots:raise RuntimeUnavailable('This model preparation was cancelled')
            self._events.add(token)
        try:
            payload=copy.deepcopy(payload);files=[]
            for field in ['source','child']:
                if field in payload:
                    path=Path(payload[field]).resolve();relative=str(path.relative_to(root))
                    files.append(relative);payload[field]=relative
            if 'children' in payload:
                children=[str(Path(p).resolve().relative_to(root)) for p in payload['children']]
                payload['children']=children;files.extend(children)
            for field in ['destination','temporary']:
                if field in payload:payload[field]=str(Path(payload[field]).resolve().relative_to(root))
            directories=payload.get('directories') or ([str(Path(payload['directory']).resolve().relative_to(root))] if 'directory' in payload else [])
            if 'directory' in payload:payload['directory']=directories[0]
            for relative in directories:
                directory=confined(root,relative);job=json.loads((directory/'job.json').read_text())
                files.append(str((directory/'job.json').relative_to(root)))
                for asset in assets(job['protocol']):files.append(str(confined(directory,asset['path']).relative_to(root)))
            if not files:
                # Data-only packet, including a marker for content-only k-form matching.
                write_json(root/'packet.json',{'schema':'radar-native-input-marker/v1'})
                files=['packet.json']
            source=self.resolve(source)
            request=dict(schema='radar-model-native/v1',operation=operation,payload=payload,
                implementation_sha256=package_signature(),expected_source_sha256=source['source_sha256'],
                input_hashes={p:file_sha(confined(root,p)) for p in files},
                timeout=min(600,max(1,float(timeout))))
            if all(Path(p).suffix.lower()=='.gpx' and sha==source['source_sha256']
                   for p,sha in request['input_hashes'].items()):
                # The marker is a real captured transport input. Hashing it in
                # the request also separates fixed inspections from failed
                # zero-member-packet dispatches, without retrying those jobs.
                request['input_hashes'][EMPTY_PACKET_NAME]=EMPTY_PACKET_SHA256
            key=digest([self.service.history_id,source['id'],request])[:24]
            request['dispatch_key']=key
            asset_id=self.packet(root,files,source_sha256=source['source_sha256'])
            if event.is_set():raise RuntimeUnavailable('Model preparation cancelled before dispatch')
            job=self._submit('native',source,request,asset_id,key,cancel=event)
            self.wait_job(job,key=key,cancel=event)
            received=self.result(job,self.root/'native-results'/(key+'-'+uuid.uuid4().hex))
            response=json.loads((received/'response.json').read_text())
            if response.get('schema')!='radar-model-native-result/v1' or response.get('dispatch_key')!=key or response.get('request_sha256')!=digest(request) or response.get('source_sha256')!=source['source_sha256']:
                raise ValueError('Native model result belongs to another request/source revision')
            for relative,sha in response['files'].items():
                path=confined(received,relative)
                if file_sha(path)!=sha:raise ValueError('A native model output changed')
                # Worker packet and output roots map back to the caller's confined root.
                parts=Path(relative).parts
                if parts[0] not in {'native','inputs-packet'}:raise ValueError('Unexpected native output location')
                target=confined(root,str(Path(*parts[1:])),exists=False)
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
            def relocate(value):
                if isinstance(value,list):return [relocate(v) for v in value]
                if isinstance(value,dict):return {k:relocate(v) for k,v in value.items()}
                if isinstance(value,str):
                    for prefix in [response['worker_root'],response['input_root']]:
                        if value==prefix or value.startswith(prefix+'/'):return str(root)+value[len(prefix):]
                return value
            result=relocate(response['result'])
            # A k-survey can exclude Galaxy queue/transfer time from its scientific budget.
            wait=max(0,time.monotonic()-started-float(response.get('native_seconds',0)))
            return result,wait
        finally:
            with self._event_lock:self._events.discard(token)

    def k_runner(self,source,cancel):return GalaxyKRunner(self,source,cancel)


class GalaxyKRunner:
    supports_portfolio=True
    def __init__(self,runtime,source,cancel):self.runtime=runtime;self.source=source;self.cancel=cancel;self.wait_seconds=0
    def clock(self):return time.monotonic()-self.wait_seconds
    def call(self,action,request,directory,*,timeout):
        result,wait=self.runtime.native('k-'+action,request,directory,source=self.source,timeout=timeout,cancel=self.cancel)
        self.wait_seconds+=wait
        return result


class GalaxyChunkTransport:
    """One acknowledged Galaxy job contains a finite group of scientific starts."""
    max_active_jobs=1
    chunk_size=128

    def __init__(self,runtime,campaign):
        self.runtime=runtime;self.campaign=campaign;self.path=campaign.root/'galaxy.json';self.cache={}
        self.max_active_jobs=runtime.max_active_jobs
        self.state=json.loads(self.path.read_text()) if self.path.exists() else {'sources':{},'inputs':{},'chunks':{},'members':{}}
        self._state_lock=threading.RLock();self._checkpoint_lock=threading.Lock()
        self._checkpoint_hash=None
        self.checkpointer=GalaxyRefinementRuntime(campaign.root/'checkpoints',runtime.service)
        self.checkpointer.workspace_label=CAMPAIGN_LABEL+campaign.root.name
        self._paused=threading.Event()
        self._dispatch_wave=None
        self._buffered_dispatches=set()
        self._wave_flushing=False
        self._attempted_dispatches=set()

    def pause(self):self._paused.set()
    def resume(self):self._paused.clear()

    def begin_dispatch_wave(self):
        """Collect a bounded scheduler wave before its single durable intent."""
        with self._state_lock:
            if self._dispatch_wave is not None:raise RuntimeError('A dispatch wave is already open')
            self._dispatch_wave=[]

    def abort_dispatch_wave(self):
        """Only buffered, never-attempted starts may return to the local queue."""
        with self._state_lock:
            if self._wave_flushing:return False
            for key in self._dispatch_wave or []:
                self.state['chunks'][key].update(not_dispatched=True,complete=True)
                self._buffered_dispatches.discard(key)
            self._dispatch_wave=None
            self.save()
            return True

    def flush_dispatch_wave(self):
        """Persist every finite member list before the first scientific POST.

        POSTs retain their individual stable keys and acknowledgment recovery.
        An interrupted flush never treats a missing acknowledgment as absence.
        """
        with self._state_lock:
            if self._dispatch_wave is None:raise RuntimeError('No dispatch wave is open')
            keys=list(self._dispatch_wave)
        if not keys:
            with self._state_lock:self._dispatch_wave=None
            return
        if self._paused.is_set() or self.campaign.setting('paused',False):
            self.abort_dispatch_wave();return
        with self._state_lock:self._wave_flushing=True
        try:
            self.checkpoint(force=True)
        except Exception:
            # This wave has made no scientific POST. Keep that proof locally;
            # a timed-out acknowledged checkpoint remains owned by its uploader.
            with self._state_lock:self._wave_flushing=False
            self.abort_dispatch_wave()
            raise
        attempted=set()
        try:
            for key in keys:
                if self._paused.is_set() or self.campaign.setting('paused',False):break
                if self.state['chunks'][key].get('not_dispatched'):continue
                attempted.add(key)
                with self._state_lock:self._attempted_dispatches.add(key)
                self._dispatch_chunk(self.state['chunks'][key])
        finally:
            with self._state_lock:
                for key in keys:
                    if key not in attempted:
                        self.state['chunks'][key].update(not_dispatched=True,complete=True)
                    self._buffered_dispatches.discard(key)
                    self._attempted_dispatches.discard(key)
                self._dispatch_wave=None
                self._wave_flushing=False
                self.save()

    def save(self):
        with self._state_lock:write_json(self.path,self.state)

    def bind_source(self,protocol,source):
        source=copy.deepcopy(source)
        for key in ['context','inspection','source_path']:source.pop(key,None)
        with self._state_lock:self.state['sources'][protocol]=source;self.save()

    def prepare_source(self,protocol,root,source):
        """Capture input-HDA references before new trials enter the ledger."""
        from gpx_recipe.symmetry.compiler import prepare_shared_inputs
        sha=digest(protocol)
        captured=prepare_shared_inputs(protocol,root,self.campaign.root)
        packet=self.runtime.packet(captured,['protocol.json']+[a['path'] for a in assets(protocol)],
                                   source_sha256=protocol['source']['sha256'])
        with self._state_lock:
            self.state['inputs'][sha]=packet
            self.bind_source(sha,source)

    def checkpoint(self,*,force=False):
        with self._checkpoint_lock:
            # A background enrollment commits protocols and their trials in one
            # transaction. Keep all snapshot SELECTs on the same side of that
            # commit, without nesting Campaign's file lock inside a tick.
            with self.campaign.connect() as guard:
                guard.execute('BEGIN IMMEDIATE')
                with self._state_lock:
                    document=dict(schema='radar-model-campaign/v1',history_id=self.runtime.service.history_id,
                        campaign=self.campaign.root.name,ledger=self.campaign.snapshot(),transport=copy.deepcopy(self.state))
            stable=copy.deepcopy(document)
            for chunk in stable['transport']['chunks'].values():
                if chunk.get('job'):chunk['job']={k:chunk['job'].get(k) for k in ['galaxy_job_id','outputs']}
            sha=digest(stable)
            if not force and sha==self._checkpoint_hash:return
            self.checkpointer.save_workspace(document);self._checkpoint_hash=sha

    def chunk_size_for(self,protocol):
        root=self.campaign.root/'inputs'/protocol
        p=json.loads((root/'protocol.json').read_text())
        size=confined(root,p['source']['path']).stat().st_size
        # A small diverse pilot measures native cost before the long campaign.
        results=[c['result'] for c in self.state['chunks'].values() if c.get('result')]
        size_limit=min(self.chunk_size,max(1,(128*1024*1024)//max(1,8*size)))
        if not results:return min(8,size_limit)
        import math,statistics
        durations=[float(row['wall_seconds']) for result in results for row in result['trials'] if row.get('wall_seconds',0)>0]
        if not durations:return size_limit
        durations.sort();p80=durations[min(len(durations)-1,math.ceil(.8*len(durations))-1)]
        workers=results[-1]['allocation']['workers']
        return min(size_limit,max(1,int(180*workers/max(p80,.1))))

    def submit_many(self,entries,*,workers):
        if self._paused.is_set() or self.campaign.setting('paused',False):raise RuntimeUnavailable('Model campaign is paused before dispatch')
        first=json.loads((entries[0][1]/'job.json').read_text());protocol=first['protocol'];sha=first['protocol_sha256']
        ids=[key for key,_ in entries]
        # Reuse an unresolved acknowledgment; a pending old job is never retried.
        previous=[self.state['members'].get(key) for key in ids]
        if previous and previous[0] and len(set(previous))==1:
            old=self.state['chunks'][previous[0]]
            if not old.get('complete') and not old.get('not_dispatched'):
                recovered=self._job(old)
                if recovered:return {key:dict(dispatch_key=previous[0],key=key) for key in ids}
                raise RuntimeUnavailable('The previous model chunk acknowledgment is still ambiguous')
        key=digest([self.campaign.root.name,sha,ids,len(self.state['chunks'])])[:24]
        input_root=job_inputs(entries[0][1],first)
        if sha not in self.state['inputs']:
            packet=self.runtime.packet(input_root,['protocol.json']+[a['path'] for a in assets(protocol)],source_sha256=protocol['source']['sha256'])
            with self._state_lock:self.state['inputs'][sha]=packet
        request=dict(schema='radar-model-chunk/v1',protocol=protocol,protocol_sha256=sha,trials=ids,
            workers=workers,dispatch_key=key,seconds=240)
        if protocol.get('template_sha256'):
            template=self.campaign.setting('template');source=self.state['sources'][sha]
            request.update(template=template,origin=template['origin'],source_identity={k:source.get(k)
                for k in ['id','experiment','sample_key','bank','source_sha256']})
        chunk=dict(key=key,members=ids,protocol=sha,request=request,job=None,pending=True)
        with self._state_lock:
            self.state['chunks'][key]=chunk
            self.state['members'].update({uid:key for uid in ids});self.save()
            if self._dispatch_wave is not None:
                self._dispatch_wave.append(key)
                self._buffered_dispatches.add(key)
                return {uid:dict(dispatch_key=key,key=uid) for uid in ids}
        try:
            self.checkpoint(force=True)  # durable intent precedes the remote submission
        except Exception:
            # No native POST has been attempted. Preserve that proof locally so
            # explicit recovery can safely return these starts to the queue.
            with self._state_lock:chunk.update(not_dispatched=True,complete=True);self.save()
            raise
        return self._dispatch_chunk(chunk)

    def _dispatch_chunk(self,chunk):
        key,sha=chunk['key'],chunk['protocol']
        try:job=self.runtime._submit('chunk',self.state['sources'][sha],chunk['request'],self.state['inputs'][sha],key,cancel=self._paused)
        except Exception:
            dispatch=self.runtime.state['dispatches'].get(key,{})
            if dispatch.get('absence_verified') and not dispatch.get('pending'):
                with self._state_lock:chunk['not_dispatched']=True;self.save()
            raise
        with self._state_lock:chunk['job']=job.model_dump(mode='json');self.save()
        # The preceding remote intent already names every member and its stable
        # dispatch key. A lost viewer recovers the acknowledged job by that key.
        # The driver backs up acknowledgements together after filling free
        # slots, avoiding a metadata-job round trip after every individual POST.
        return {uid:dict(dispatch_key=key,key=uid) for uid in chunk['members']}

    def _job(self,chunk):
        if chunk.get('job'):
            job=UtilityActionRecord.model_validate(chunk['job']);self.runtime.verify_job(job,chunk['key']);return job
        job=self.runtime.find_dispatch(chunk['key'])
        if job:
            with self._state_lock:chunk['job']=job.model_dump(mode='json');self.save()
        return job

    def recover(self,key,directory):
        dispatch=self.state['members'].get(key)
        if not dispatch:return dict(state='absent')
        if self.state['chunks'][dispatch].get('not_dispatched'):return dict(state='not_started')
        return self.poll(dict(dispatch_key=dispatch,key=key),directory)

    def poll(self,handle,directory):
        dispatch=handle['dispatch_key'];chunk=self.state['chunks'][dispatch]
        if chunk.get('not_dispatched'):return dict(state='not_started')
        with self._state_lock:
            if dispatch in self._buffered_dispatches:
                return dict(state='running',handle=handle)
        current=self.cache.get(dispatch)
        if current is None or (current[1]['state']!='complete' and time.monotonic()-current[0]>1):
            job=self._job(chunk)
            if job is None:return dict(state='unknown',reason='The model chunk has no acknowledged Galaxy job')
            self.runtime.service.refresh_utility(job)
            with self._state_lock:chunk['job']=job.model_dump(mode='json');self.save()
            intent=chunk.get('staging_retirement')
            prior=sum(bool(c.get('staging_retirement')) for c in self.state['chunks'].values())
            if not chunk.get('cancel_requested') and (intent or (job.status==RunStatus.ERROR and prior<2)):
                proof=self.runtime.failed_request_proof(job,dispatch,retiring=bool(intent))
                if proof is not None:
                    if intent and intent!=proof:raise RuntimeUnavailable('The saved staging retirement belongs to different inputs')
                    if not intent:
                        with self._state_lock:chunk['staging_retirement']=proof;self.save()
                        self.checkpoint(force=True)
                    self.runtime.retire_failed_request(job,dispatch,proof)
                    with self._state_lock:chunk.update(not_dispatched=True,complete=True);self.save()
                    self.checkpoint(force=True)
                    self.campaign.set_setting('preparation_message','Recovered a failed empty request upload; retrying the same finite starts after confirmed cancellation. No native calculation ran in the retired batch.')
                    self.cache.pop(dispatch,None)
                    return dict(state='not_started')
            if job.status==RunStatus.CANCELLED:
                self.cache[dispatch]=(time.monotonic(),dict(state='cancelled'))
            elif job.status==RunStatus.ERROR:
                state=dict(state='cancelled') if chunk.get('cancel_requested') else dict(state='unknown',reason=job.message or 'Galaxy model chunk failed; review its console')
                self.cache[dispatch]=(time.monotonic(),state)
            elif job.status!=RunStatus.OK:self.cache[dispatch]=(time.monotonic(),dict(state='running'))
            else:
                received=self.runtime.result(job,self.campaign.root/'remote-results'/(dispatch+'-'+uuid.uuid4().hex))
                response=json.loads((received/'response.json').read_text())
                if response.get('schema')!='radar-model-chunk-result/v1' or response.get('dispatch_key')!=dispatch or response.get('protocol_sha256')!=chunk['protocol']:
                    raise ValueError('Model chunk result differs from the submitted search')
                terminal={r['id']:r for r in response['trials']};pending=response['pending']
                if len(terminal)!=len(response['trials']) or len(set(pending))!=len(pending) or set(terminal)|set(pending)!=set(chunk['members']) or set(terminal)&set(pending):raise ValueError('Model chunk coverage changed')
                for uid,row in terminal.items():
                    if row['state']=='finished':
                        source=received/'trials'/uid;target=self.campaign.root/'trials'/uid
                        for p in source.rglob('*'):
                            if p.is_file() and p.name!='result.json':
                                relative=str(p.relative_to(source));destination=confined(target,relative,exists=False)
                                destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,destination)
                        # A receipt is the atomic commit marker, after all of its artifacts.
                        temporary=target/('.result-'+uuid.uuid4().hex+'.partial')
                        shutil.copyfile(source/'result.json',temporary);temporary.replace(target/'result.json')
                with self._state_lock:chunk.update(complete=True,result=response);self.save()
                self.campaign.set_setting('allocation',response['allocation'])
                self.cache[dispatch]=(time.monotonic(),dict(state='complete',terminal=terminal,pending=pending))
        state=self.cache[dispatch][1]
        if state['state']=='running':return dict(state='running',handle=handle)
        if state['state']=='complete':
            if handle['key'] in state['pending']:return dict(state='not_started')
            row=state['terminal'][handle['key']]
            return dict(state=row['state'],reason=row.get('reason'))
        return state

    def cancel(self,handle,directory):
        dispatch=handle['dispatch_key'];chunk=self.state['chunks'][dispatch]
        with self._state_lock:
            if dispatch in self._buffered_dispatches and dispatch not in self._attempted_dispatches:
                chunk.update(not_dispatched=True,complete=True);self.save()
                return True
        job=self._job(chunk)
        if not job:return False
        self.runtime.cancel_job(job,chunk['key'])
        with self._state_lock:chunk['cancel_requested']=True;self.save()
        self.checkpoint(force=True)
        self.cache.pop(handle['dispatch_key'],None)
        return False
