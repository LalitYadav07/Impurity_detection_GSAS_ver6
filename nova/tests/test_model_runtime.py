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
from radar_pd_nova.model_runtime import GalaxyModelRuntime,GalaxyChunkTransport,MODEL_TOOL_ID
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


def test_lost_ack_is_recovered_without_resubmission(remote):
    runtime,service,source=remote;service.lose_ack=True
    with pytest.raises(RuntimeUnavailable,match='did not acknowledge'):runtime._submit('chunk',source,{},'assets','a'*24)
    assert runtime._submit('chunk',source,{},'assets','a'*24).galaxy_job_id=='job0'
    assert len(service.submissions)==1


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
