"""Exercise Galaxy dispatch, packet capture, recovery and cancellation contracts."""
import copy
import json
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace
import zipfile

import pytest

from radar_pd_nova.models import UtilityActionRecord,RunStatus
from radar_pd_nova.model_runtime import GalaxyModelRuntime,GalaxyChunkTransport,MODEL_TOOL_ID,active_batch_count
from radar_pd_nova.refinement_runtime import RuntimeUnavailable
from gpx_recipe.symmetry.compiler import file_sha,package_signature
from gpx_recipe.symmetry.contracts import digest
from gpx_recipe.symmetry.campaign import Campaign


class Service:
    history_id='history';galaxy_url='https://galaxy.invalid';_headers={'x-api-key':'PRIVATE-SENTINEL'}
    def __init__(self,root):self.root=root;self.data={};self.requests={};self.jobs={};self.labels={};self.submissions=[];self.lose_ack=False
    def _put(self,value,label):
        key='hda'+str(len(self.data));self.data[key]=value;self.labels[key]=label;return key
    def upload_document(self,path,*,label,**kwargs):
        from radar_pd_nova.galaxy_service import _upload_filename
        return self._put(Path(path).read_bytes(),_upload_filename(Path(path),label))
    def upload_json_document(self,value,*,name,label,**kwargs):
        from radar_pd_nova.galaxy_service import _upload_filename
        key=self._put(json.dumps(value).encode(),_upload_filename(Path(name),label));self.requests[key]=copy.deepcopy(value);return key
    def search_history_datasets(self,query,**kwargs):
        rows=[dict(id=k,name=v) for k,v in reversed(list(self.labels.items())) if query in v]
        return rows[kwargs.get('offset',0):kwargs.get('offset',0)+kwargs.get('limit',500)]
    def _download_dataset(self,key,target):Path(target).write_bytes(self.data[key])
    def submit_utility(self,**kwargs):
        self.submissions.append(kwargs);key=kwargs['inputs']['dispatch_key']
        job=UtilityActionRecord(uid=key,tool_id=MODEL_TOOL_ID,name='RADAR-PD model '+key,galaxy_job_id='job'+str(len(self.jobs)),status=RunStatus.OK)
        self.jobs[key]=job
        if self.lose_ack:raise TimeoutError('Lost acknowledgment after acceptance')
        return job
    def refresh_utility(self,job):
        actual=next(v for v in self.jobs.values() if v.galaxy_job_id==job.galaxy_job_id)
        job.status=actual.status;job.outputs=copy.deepcopy(actual.outputs);return job


@pytest.fixture
def remote(tmp_path,monkeypatch):
    service=Service(tmp_path);runtime=GalaxyModelRuntime(tmp_path/'runtime',service,poll_seconds=.001)
    monkeypatch.setattr(runtime,'find_dispatch',lambda key:service.jobs.get(key))
    monkeypatch.setattr(runtime,'wait_capacity',lambda cancel=None:None)
    p=tmp_path/'source.gpx';p.write_bytes(b'opaque-scientific-project')
    source=dict(id='scan1',dataset_id='original-data',source_path=str(p),source_sha256=file_sha(p))
    return runtime,service,source


def test_native_packet_result_roundtrip_and_credentials_stay_in_nova(remote,tmp_path,monkeypatch):
    runtime,service,source=remote;source_dir=tmp_path/'prepare';source_dir.mkdir();shutil.copyfile(source['source_path'],source_dir/'source.gpx')
    monkeypatch.setenv('AZURE_ILLUMINE_API_KEY','PRIVATE-MODEL-SENTINEL')
    original=service.submit_utility
    def submit(**kwargs):
        job=original(**kwargs);request=service.requests[kwargs['inputs']['model_request']['dataset_id']]
        archive=tmp_path/'response.zip'
        content=b'{"checked":true}'
        import hashlib
        response=dict(schema='radar-model-native-result/v1',dispatch_key=request['dispatch_key'],request_sha256=digest(request),
            source_sha256=source['source_sha256'],worker_root='/worker/native',input_root='/worker/inputs-packet',
            result=dict(path='/worker/native/nested/result.json'),files={'native/nested/result.json':hashlib.sha256(content).hexdigest()},native_seconds=0.)
        with zipfile.ZipFile(archive,'w') as z:z.writestr('response.json',json.dumps(response));z.writestr('native/nested/result.json',content)
        job.outputs={'result_bundle':service._put(archive.read_bytes(),'model results')};return job
    monkeypatch.setattr(service,'submit_utility',submit)
    result,_=runtime.native('inspect',dict(source=str(source_dir/'source.gpx')),source_dir,source=source)
    assert result['path']==str(source_dir/'nested/result.json')
    assert json.loads(Path(result['path']).read_text())=={'checked':True}
    assert len(service.submissions)==1
    assert service.submissions[0]['inputs']['source_gpx']=={'dataset_id':'original-data'}
    assert 'SENTINEL' not in json.dumps(service.requests)
    for value in service.data.values():
        assert b'PRIVATE-' not in value
    import io
    packet=service.data[service.submissions[0]['inputs']['model_assets']['dataset_id']]
    with zipfile.ZipFile(io.BytesIO(packet)) as archive:assert archive.namelist()==[]


def test_native_inspections_reuse_an_empty_asset_packet_without_copying_gpx(remote,tmp_path):
    runtime,service,source=remote
    first=runtime.packet(tmp_path,['source.gpx'],source_sha256=source['source_sha256'])
    second_file=tmp_path/'second.gpx';second_file.write_bytes(b'another exact native project')
    second=runtime.packet(tmp_path,['second.gpx'],source_sha256=file_sha(second_file))
    assert first==second and len(service.data)==1
    # A different GPX is an actual scientific asset, not the supplied source.
    third=runtime.packet(tmp_path,['second.gpx'],source_sha256=source['source_sha256'])
    assert third!=first


def test_failed_asset_upload_does_not_poison_future_native_inspections(remote,tmp_path):
    runtime,service,source=remote
    first=runtime.packet(tmp_path,['source.gpx'],source_sha256=source['source_sha256'])
    service._dataset_metadata=lambda key:dict(state='error' if key==first else 'ok')
    second=runtime.packet(tmp_path,['source.gpx'],source_sha256=source['source_sha256'])
    assert second!=first
    assert runtime.packet(tmp_path,['source.gpx'],source_sha256=source['source_sha256'])==second


def test_lost_ack_is_recovered_without_resubmission(remote):
    runtime,service,source=remote;service.lose_ack=True
    with pytest.raises(RuntimeUnavailable,match='did not acknowledge'):runtime._submit('chunk',source,{},'assets','a'*24)
    assert runtime._submit('chunk',source,{},'assets','a'*24).galaxy_job_id=='job0'
    assert len(service.submissions)==1


def test_failed_request_staging_cannot_create_a_scientific_job(remote,monkeypatch):
    runtime,service,source=remote
    def failed(value,**kwargs):
        assert kwargs['wait_ready'] is True
        raise RuntimeError('Owned request upload failed')
    monkeypatch.setattr(service,'upload_json_document',failed)
    with pytest.raises(RuntimeError,match='upload failed'):
        runtime._submit('chunk',source,{},'assets','1'*24)
    assert not service.submissions and not runtime.state['dispatches']['1'*24]['pending']


def paused_request(remote,tmp_path,monkeypatch):
    runtime,service,source=remote;key='2'*24
    job=service.submit_utility(inputs={'dispatch_key':key});job.status=RunStatus.ERROR
    details=dict(id=job.galaxy_job_id,history_id='history',tool_id=MODEL_TOOL_ID,state='paused',
        inputs={k:{'id':k} for k in ['source_gpx','model_assets','model_request']})
    metadata={k:dict(state='ok',file_size=100) for k in ['source_gpx','model_assets']}
    metadata['model_request']=dict(state='error',file_size=0,name='RADAR-PD RADAR-PD Model request '+key+' | request.json')
    service._job_details=lambda uid:copy.deepcopy(details)
    service._dataset_metadata=lambda uid:copy.deepcopy(metadata[uid])
    cancellations=[]
    def cancel(*args,**kwargs):
        cancellations.append(args);details['state']='deleted';job.status=RunStatus.CANCELLED
        return SimpleNamespace(raise_for_status=lambda:None)
    monkeypatch.setattr('requests.delete',cancel)
    campaign=Campaign(tmp_path/'failed-staging');transport=GalaxyChunkTransport(runtime,campaign)
    transport.state['chunks'][key]=dict(key=key,job=job.model_dump(mode='json'),members=['trial'])
    transport.state['members']['trial']=key
    return runtime,transport,job,details,metadata,cancellations,key


def test_owned_empty_request_is_retired_before_same_starts_can_retry(remote,tmp_path,monkeypatch):
    runtime,t,job,details,metadata,cancel,key=paused_request(remote,tmp_path,monkeypatch)
    assert t.poll(dict(dispatch_key=key,key='trial'),tmp_path)=={'state':'not_started'}
    assert len(cancel)==1 and details['state']=='deleted'
    assert t.state['chunks'][key]['not_dispatched']
    # Durable intent and terminal acknowledgment survive a viewer loss.
    restored=GalaxyChunkTransport(runtime,t.campaign)
    assert restored.recover('trial',tmp_path)=={'state':'not_started'}
    assert len(cancel)==1


@pytest.mark.parametrize('change',[
    ('job','state','running'),('job','external_id','runner-1'),('job','job_runner_name','pulsar'),
    ('job','command_line','already assigned'),('job','exit_code',0),('job','history_id','other-history'),
    ('request','file_size',1),('request','state','ok'),('request','name','unowned request'),
    ('source','state','error')])
def test_ambiguous_or_started_batches_are_never_automatically_retried(remote,tmp_path,monkeypatch,change):
    runtime,t,job,details,metadata,cancel,key=paused_request(remote,tmp_path,monkeypatch)
    kind,field,value=change
    target=details if kind=='job' else metadata['model_request' if kind=='request' else 'source_gpx']
    target[field]=value
    assert t.poll(dict(dispatch_key=key,key='trial'),tmp_path)['state']=='unknown'
    assert not cancel and not t.state['chunks'][key].get('not_dispatched')


def test_staging_recovery_is_bounded_and_does_not_repeat_an_ordinary_cancellation(remote,tmp_path,monkeypatch):
    runtime,t,job,details,metadata,cancel,key=paused_request(remote,tmp_path,monkeypatch)
    for n in range(2):t.state['chunks']['previous'+str(n)]={'staging_retirement':{'job_id':str(n)}}
    assert t.poll(dict(dispatch_key=key,key='trial'),tmp_path)['state']=='unknown'
    assert not cancel
    t.cache.clear();job.status=RunStatus.CANCELLED;details['state']='deleted'
    assert t.poll(dict(dispatch_key=key,key='trial'),tmp_path)['state']=='cancelled'
    assert not cancel


def test_retirement_resumes_after_cancellation_but_before_final_checkpoint(remote,tmp_path,monkeypatch):
    runtime,t,job,details,metadata,cancel,key=paused_request(remote,tmp_path,monkeypatch)
    real=runtime.retire_failed_request
    def interrupted(*args):
        real(*args)
        raise TimeoutError('Viewer interrupted after cancellation')
    monkeypatch.setattr(runtime,'retire_failed_request',interrupted)
    with pytest.raises(TimeoutError):t.poll(dict(dispatch_key=key,key='trial'),tmp_path)
    assert details['state']=='deleted' and t.state['chunks'][key]['staging_retirement']
    monkeypatch.setattr(runtime,'retire_failed_request',real)
    restored=GalaxyChunkTransport(runtime,t.campaign)
    assert restored.poll(dict(dispatch_key=key,key='trial'),tmp_path)['state']=='not_started'
    assert len(cancel)==1


def test_execution_proof_is_rechecked_after_the_saved_retirement_intent(remote,tmp_path,monkeypatch):
    runtime,t,job,details,metadata,cancel,key=paused_request(remote,tmp_path,monkeypatch)
    proof=runtime.failed_request_proof(job,key)
    assert proof
    details['job_runner_name']='pulsar'
    with pytest.raises(RuntimeUnavailable,match='proof changed'):
        runtime.retire_failed_request(job,key,proof)
    assert not cancel


@pytest.mark.parametrize('saved_intent',[False,True])
def test_user_cancellation_wins_over_automatic_staging_recovery(remote,tmp_path,monkeypatch,saved_intent):
    runtime,t,job,details,metadata,cancel,key=paused_request(remote,tmp_path,monkeypatch)
    if saved_intent:t.state['chunks'][key]['staging_retirement']=runtime.failed_request_proof(job,key)
    t.state['chunks'][key]['cancel_requested']=True
    assert t.poll(dict(dispatch_key=key,key='trial'),tmp_path)['state']=='cancelled'
    assert not cancel and not t.state['chunks'][key].get('not_dispatched')


@pytest.mark.parametrize('limit,occupancies,expected_calls',[(1,[1,0],2),(2,[1],1),(2,[2,1],2),(4,[4,3],2),(8,[8,7],2)])
def test_history_capacity_respects_the_reviewed_batch_limit(tmp_path,monkeypatch,limit,occupancies,expected_calls):
    runtime=GalaxyModelRuntime(tmp_path/'bounded',Service(tmp_path),poll_seconds=.001,max_active_jobs=limit)
    calls=[]
    def get(url,**kwargs):
        calls.append(kwargs);jobs=[{}]*occupancies.pop(0)
        return SimpleNamespace(raise_for_status=lambda:None,json=lambda:jobs)
    monkeypatch.setattr('requests.get',get)
    runtime.wait_capacity()
    assert len(calls)==expected_calls
    assert all(q['params']['history_id']=='history' and q['params']['tool_id']==MODEL_TOOL_ID and q['params']['limit']==limit for q in calls)


@pytest.mark.parametrize('limit',[0,3,16,True,'2',1.5])
def test_parallel_batch_budget_requires_a_supported_explicit_limit(tmp_path,limit):
    with pytest.raises(ValueError,match='1, 2, 4 or 8'):
        GalaxyModelRuntime(tmp_path/'invalid',Service(tmp_path),max_active_jobs=limit)


def test_enrollment_counts_dispatched_batches_rather_than_their_member_trials():
    rows=[dict(id=str(i),state='running',handle=json.dumps({'dispatch_key':'first'})) for i in range(49)]
    assert active_batch_count(rows)==1
    rows.append(dict(id='next',state='submitting',handle=json.dumps({'dispatch_key':'second'})))
    rows.append(dict(id='done',state='terminal',handle=json.dumps({'dispatch_key':'finished'})))
    assert active_batch_count(rows)==2


def test_unresolved_ack_never_retries_and_cancel_before_post_submits_nothing(remote,monkeypatch):
    runtime,service,source=remote;service.lose_ack=True
    with pytest.raises(RuntimeUnavailable):runtime._submit('chunk',source,{},'assets','b'*24)
    service.jobs.clear()
    with pytest.raises(RuntimeUnavailable,match='no duplicate'):runtime._submit('chunk',source,{},'assets','b'*24)
    event=threading.Event();event.set()
    with pytest.raises(RuntimeUnavailable,match='paused'):runtime._submit('chunk',source,{},'assets','c'*24,cancel=event)
    assert len(service.submissions)==1


def test_restored_job_identity_is_verified_before_cancellation(remote,monkeypatch):
    runtime,service,source=remote
    wrong=UtilityActionRecord(uid='wrong',tool_id=MODEL_TOOL_ID,name='wrong',galaxy_job_id='unrelated',status=RunStatus.RUNNING)
    calls=[];monkeypatch.setattr('requests.delete',lambda *a,**k:calls.append(a))
    with pytest.raises(RuntimeUnavailable,match='does not belong'):runtime.cancel_job(wrong,'d'*24)
    assert calls==[]


def test_external_cancellation_is_terminal_in_preparation_and_chunk_poll(remote,tmp_path):
    runtime,service,source=remote;key='f'*24
    job=service.submit_utility(inputs={'dispatch_key':key});job.status=RunStatus.CANCELLED
    with pytest.raises(RuntimeUnavailable,match='cancelled'):runtime.wait_job(job,key=key)
    c=Campaign(tmp_path/'campaign');t=GalaxyChunkTransport(runtime,c)
    t.state['chunks'][key]=dict(key=key,job=job.model_dump(mode='json'))
    assert t.poll(dict(dispatch_key=key,key='trial'),tmp_path)['state']=='cancelled'


def test_input_packet_is_reused_across_runtime_restart(remote,tmp_path,monkeypatch):
    runtime,service,source=remote;root=Path(source['source_path']).parent
    first=runtime.packet(root,['source.gpx']);count=len(service.data)
    second=GalaxyModelRuntime(tmp_path/'new-viewer',service)
    assert second.packet(root,['source.gpx'])==first and len(service.data)==count


def test_recovery_uses_exact_result_hda_and_provenance(tmp_path,monkeypatch):
    service=Service(tmp_path);runtime=GalaxyModelRuntime(tmp_path/'runtime',service);calls=[]
    key='e'*24
    def get(url,**kwargs):
        calls.append((url,kwargs))
        payload=({'job_id':'job1','tool_id':MODEL_TOOL_ID} if url.endswith('/provenance') else
            [dict(id='hda1',name='RADAR-PD model '+key+' results',history_content_type='dataset')])
        return SimpleNamespace(raise_for_status=lambda:None,json=lambda:payload)
    monkeypatch.setattr('requests.get',get)
    service._job_details=lambda uid:dict(id=uid,tool_id=MODEL_TOOL_ID,state='running',params={'dispatch_key':key},outputs={})
    service._job_output_ids=lambda job:{}
    job=runtime.find_dispatch(key)
    assert job.galaxy_job_id=='job1' and len(calls)==2
    assert '/histories/history/contents' in calls[0][0]
    assert calls[0][1]['params']['qv']=='RADAR-PD model '+key+' results'


def test_native_and_chunk_have_distinct_history_recovery_names(remote,tmp_path):
    runtime,service,source=remote
    campaigns=[Campaign(tmp_path/name,shared_inputs=True) for name in ['campaign-'+('a'*12),'campaign-'+('b'*12)]]
    transports=[GalaxyChunkTransport(runtime,c) for c in campaigns]
    for t in transports:t.checkpoint(force=True)
    labels=list(service.labels.values())
    assert len(labels)==2 and len(set(labels))==2
    assert all('Model campaign campaign-' in label for label in labels)


def test_lost_viewer_restores_the_existing_chunk_and_all_trial_receipts(remote,tmp_path,monkeypatch):
    from gpx_recipe.symmetry.contracts import validate_protocol
    from gpx_recipe.symmetry.templates import make_template
    from radar_pd_nova.model_campaigns import restore_history
    runtime,service,source=remote
    inputs=tmp_path/'captured';inputs.mkdir();shutil.copyfile(source['source_path'],inputs/'source.gpx')
    candidate=dict(id='mag:fixture',magnetic_space_group="Pn'ma'",transformation_matrix=[[1,0,0],[0,1,0],[0,0,1]],origin_shift=[0,0,0])
    (inputs/'catalog.json').write_text(json.dumps(dict(schema='radar-symmetry-catalog/v1',parent_setting_sha256='a'*64,k=['0','0','0'],candidates=[candidate])))
    p=validate_protocol(dict(kind='magnetic',source=dict(path='source.gpx',sha256=file_sha(inputs/'source.gpx')),
        catalog=dict(path='catalog.json',sha256=file_sha(inputs/'catalog.json')),parent_phase='Parent',histogram='Bank',
        parent_setting_sha256='a'*64,magnetic_function_sha256='b'*64,implementation_sha256=package_signature(),
        k=['0','0','0'],candidates=[candidate],seeds=[dict(id='start'+str(i),axis='x',magnitude=3.) for i in range(3)],sites=[dict(label='Mn',form_factor='Mn',lande_g=2.)],
        policy=dict(cycles=12,timeout_seconds=60,maximum_trials=3,maximum_cell_multiplier=8,maximum_moment=6.,fit_band_percent=.25,maximum_mode=.5,adequate_rwp_percent=12.)))
    source.update(experiment='IPTS-1',sample_key='specimen:A',bank='1')
    context=dict(sha256=source['source_sha256'])
    template=make_template(p,context,source,[(source,'radar')],origin='radar')
    c=Campaign(tmp_path/'original/campaign-aaaaaaaaaaaa',shared_inputs=True);c.enroll(p,inputs,approved_hash=digest(p))
    c.set_setting('template',template)
    t=GalaxyChunkTransport(runtime,c);t.bind_source(digest(p),source)
    original_submit=service.submit_utility
    def submit(**kwargs):
        result=original_submit(**kwargs);result.status=RunStatus.RUNNING
        raise TimeoutError('Viewer lost after the Galaxy job accepted the chunk')
    monkeypatch.setattr(service,'submit_utility',submit)
    c.tick(t,slots=2);assert len(service.submissions)==1
    assert any(label.startswith('RADAR-PD RADAR-PD Model campaign ') for label in service.labels.values())
    new=GalaxyModelRuntime(tmp_path/'new-runtime',service)
    monkeypatch.setattr(new,'find_dispatch',lambda key:service.jobs.get(key))
    service.data['original-data']=Path(source['source_path']).read_bytes()
    restore_history(tmp_path/'new-viewer',new)
    restored=Campaign(tmp_path/'new-viewer/campaign-aaaaaaaaaaaa',shared_inputs=True)
    transport=GalaxyChunkTransport(new,restored);restored.recover(transport)
    assert {r['state'] for r in restored.trials()}=={'running'}
    assert len(service.submissions)==1
    chunk=next(iter(transport.state['chunks'].values()));job=service.jobs[chunk['key']]
    archive=tmp_path/'finished-chunk.zip'
    response=dict(schema='radar-model-chunk-result/v1',dispatch_key=chunk['key'],protocol_sha256=digest(p),
        trials=[dict(id=uid,state='finished',status='unsupported') for uid in chunk['members']],pending=[],allocation=dict(workers=1))
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('response.json',json.dumps(response))
        for uid in chunk['members']:z.writestr('trials/'+uid+'/result.json',json.dumps(dict(schema='radar-symmetry-trial/v1',id=uid,protocol_sha256=digest(p),status='unsupported',reason='Retained control disposition')))
    job.outputs={'result_bundle':service._put(archive.read_bytes(),'result')};job.status=RunStatus.OK
    transport.cache.clear();restored.tick(transport)
    assert {r['state'] for r in restored.trials()}=={'terminal'}
    assert len(service.submissions)==1 and len(restored.trials())==3
