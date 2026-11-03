"""An explicit preparation retry must not repeat an uncertain native job."""
import asyncio
import base64
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from gpx_recipe.symmetry.contracts import digest
from radar_pd_nova.checkpoint_uploads import CheckpointUploadPending
from radar_pd_nova.model_preparation_retry import (
    base_key, failure_proof, inspection_request, prepare_attempt, validate_attempt,
)
from radar_pd_nova.model_runtime import GalaxyModelRuntime
from radar_pd_nova.models import RunStatus
from radar_pd_nova.refinement_runtime import RuntimeUnavailable
from radar_pd_nova.symmetry_controller import SymmetryMixin
from radar_pd_nova.model_runtime import GalaxyChunkTransport
from gpx_recipe.symmetry.campaign import Campaign
from test_model_runtime import remote


@pytest.fixture
def failed_inspection(remote, tmp_path, monkeypatch):
    from radar_pd_nova.native_job_admission import HistoryJobAdmission
    monkeypatch.setattr(HistoryJobAdmission,'active_jobs',lambda self:[])
    runtime, service, source=remote
    service.tool_has_input=lambda *args:True
    root=tmp_path/'input';root.mkdir()
    (root/'source.gpx').write_bytes(Path(source['source_path']).read_bytes())
    body=inspection_request(source);key=base_key(runtime,source,body)
    old_submit=service.submit_utility
    def submit(**kwargs):
        job=old_submit(**kwargs)
        request=json.loads(base64.b64decode(kwargs['inputs']['model_request_b64']))
        if kwargs['inputs']['dispatch_key']==key:
            job.status=RunStatus.ERROR
            job.message='Pulsar metadata processing failed: duplicate rule'
            job.outputs={'result_bundle':'failed-bundle','console_output':'failed-console'}
        else:
            result=dict(schema='radar-model-native-result/v1',dispatch_key=request['dispatch_key'],
                request_sha256=digest(request),source_sha256=source['source_sha256'],
                worker_root='/worker/native',input_root='/worker/inputs',
                result={'checked':True},files={},native_seconds=0.)
            content=io.BytesIO()
            with zipfile.ZipFile(content,'w') as archive:archive.writestr('response.json',json.dumps(result))
            job.outputs={'result_bundle':service._put(content.getvalue(),'native result')}
        return job
    monkeypatch.setattr(service,'submit_utility',submit)
    metadata={name:dict(id=name,state='error',file_size=0) for name in ['failed-bundle','failed-console']}
    monkeypatch.setattr(service,'_dataset_metadata',lambda key:copy.deepcopy(metadata.get(key,dict(id=key,state='ok',file_size=10))),raising=False)
    details_override={}
    def details(identifier):
        key2,job=next((k,j) for k,j in service.jobs.items() if j.galaxy_job_id==identifier)
        inputs=next(p['inputs'] for p in service.submissions if p['inputs']['dispatch_key']==key2)
        value=dict(id=identifier,history_id=service.history_id,tool_id=runtime.tool_id,
            state=job.status.value,exit_code=0,stderr='Pulsar metadata processing failed: duplicate rule',
            params=dict(inputs),inputs={'source_gpx':{'id':inputs['source_gpx']['dataset_id']}},
            outputs={k:{'id':v} for k,v in job.outputs.items()})
        value.update(copy.deepcopy(details_override))
        return value
    monkeypatch.setattr(service,'_job_details',details)
    monkeypatch.setattr(service,'_job_output_ids',lambda d:{k:v['id'] for k,v in d.get('outputs',{}).items()},raising=False)
    request=dict(body,dispatch_key=key)
    old=runtime._submit('native',source,request,'assets',key)
    assert old.status==RunStatus.ERROR
    return SimpleNamespace(runtime=runtime,service=service,source=source,root=root,body=body,key=key,
        metadata=metadata,details=details_override,old_job=old)


def test_legacy_native_retry_reuses_the_failed_job(failed_inspection):
    f=failed_inspection
    with pytest.raises(RuntimeError,match='Pulsar'):
        f.runtime.native('inspect',{'source':str(f.root/'source.gpx')},f.root,source=f.source)
    assert len(f.service.submissions)==1


def test_explicit_retry_changes_only_dispatch_identity_after_ready_checkpoint(failed_inspection):
    f=failed_inspection
    before=copy.deepcopy(f.runtime.state['dispatches'][f.key])
    attempt,fresh=prepare_attempt(f.runtime,f.source,'campaign-aaaaaaaaaaaa','a'*64)
    assert fresh and attempt['dispatch_key']!=f.key and len(f.service.submissions)==1
    assert f.runtime.state['dispatches'][f.key]==before
    calls=[]
    result,_=f.runtime.native('inspect',{'source':str(f.root/'source.gpx')},f.root,source=f.source,
        preparation_attempt=attempt,preparation_guard=lambda:True,before_submit=lambda:calls.append('checkpoint-ready'))
    assert result=={'checked':True} and calls==['checkpoint-ready']
    assert len(f.service.submissions)==2 and f.key in f.runtime.state['dispatches']
    actual=json.loads(base64.b64decode(f.service.submissions[-1]['inputs']['model_request_b64']))
    assert {k:v for k,v in actual.items() if k!='dispatch_key'}==f.body
    assert actual['dispatch_key']==attempt['dispatch_key']
    # A second call collects the same accepted attempt; it does not POST again.
    result,_=f.runtime.native('inspect',{'source':str(f.root/'source.gpx')},f.root,source=f.source,
        preparation_attempt=attempt,preparation_guard=lambda:True,before_submit=lambda:pytest.fail('No new checkpoint for existing job'))
    assert result=={'checked':True} and len(f.service.submissions)==2


@pytest.mark.parametrize('change', ['running','cancelled','native_exit','boolean_exit','worker_error',
    'history','tool','dispatch','action','request','source_dataset','missing_inline','nonempty','active_output','deleted_output','same_outputs'])
def test_unverified_or_nonspecific_failure_never_authorizes_retry(failed_inspection,change):
    f=failed_inspection
    if change=='running':f.details['state']='running'
    elif change=='cancelled':f.details['state']='deleted'
    elif change=='native_exit':f.details['exit_code']=1
    elif change=='boolean_exit':f.details['exit_code']=False
    elif change=='worker_error':f.details['stderr']='Native calibration incompatible'
    elif change=='history':f.details['history_id']='other'
    elif change=='tool':f.details['tool_id']='different'
    elif change=='source_dataset':f.details['inputs']={'source_gpx':{'id':'different-source'}}
    elif change in {'dispatch','action','request','missing_inline'}:
        params=copy.deepcopy(f.service.submissions[0]['inputs'])
        if change=='dispatch':params['dispatch_key']='0'*24
        elif change=='action':params['action']='chunk'
        elif change=='missing_inline':params.pop('model_request_b64')
        else:
            request=json.loads(base64.b64decode(params['model_request_b64']))
            request['expected_source_sha256']='0'*64
            params['model_request_b64']=base64.b64encode(json.dumps(request).encode()).decode()
        f.details['params']=params
    elif change=='nonempty':f.metadata['failed-bundle']['file_size']=1
    elif change=='active_output':f.metadata['failed-console']['state']='running'
    elif change=='deleted_output':f.metadata['failed-bundle']['deleted']=True
    elif change=='same_outputs':f.details['outputs']={k:{'id':'failed-bundle'} for k in ['result_bundle','console_output']}
    with pytest.raises(RuntimeUnavailable):prepare_attempt(f.runtime,f.source,'campaign-aaaaaaaaaaaa','a'*64)
    assert len(f.service.submissions)==1 and not f.runtime._preparation_retry_permits


def test_checkpoint_timeout_never_posts_and_same_owner_can_resume_exact_attempt(failed_inspection):
    f=failed_inspection;attempt,_=prepare_attempt(f.runtime,f.source,'campaign-aaaaaaaaaaaa','a'*64)
    def pending():raise CheckpointUploadPending('owned-hda','queued')
    with pytest.raises(CheckpointUploadPending):
        f.runtime.native('inspect',{'source':str(f.root/'source.gpx')},f.root,source=f.source,
            preparation_attempt=attempt,preparation_guard=lambda:True,before_submit=pending)
    assert len(f.service.submissions)==1
    resumed,fresh=prepare_attempt(f.runtime,f.source,'campaign-aaaaaaaaaaaa','a'*64,attempt)
    assert resumed==attempt and not fresh
    f.runtime.native('inspect',{'source':str(f.root/'source.gpx')},f.root,source=f.source,
        preparation_attempt=attempt,preparation_guard=lambda:True,before_submit=lambda:None)
    assert len(f.service.submissions)==2


def test_prior_job_becoming_active_during_checkpoint_blocks_retry(failed_inspection):
    f=failed_inspection;attempt,_=prepare_attempt(f.runtime,f.source,'campaign-aaaaaaaaaaaa','a'*64)
    with pytest.raises(RuntimeUnavailable,match='terminal'):
        f.runtime.native('inspect',{'source':str(f.root/'source.gpx')},f.root,source=f.source,
            preparation_attempt=attempt,preparation_guard=lambda:True,before_submit=lambda:f.details.update(state='running'))
    assert len(f.service.submissions)==1


def test_lost_ack_is_held_then_reconciled_without_second_post(failed_inspection,monkeypatch,tmp_path):
    f=failed_inspection;attempt,_=prepare_attempt(f.runtime,f.source,'campaign-aaaaaaaaaaaa','a'*64)
    submit=f.service.submit_utility
    def lost(**kwargs):submit(**kwargs);raise TimeoutError('POST accepted, ACK lost')
    monkeypatch.setattr(f.service,'submit_utility',lost)
    with pytest.raises(RuntimeUnavailable,match='acknowledge'):
        f.runtime.native('inspect',{'source':str(f.root/'source.gpx')},f.root,source=f.source,
            preparation_attempt=attempt,preparation_guard=lambda:True,before_submit=lambda:None)
    assert len(f.service.submissions)==2
    old_find=f.runtime.find_dispatch
    monkeypatch.setattr(f.runtime,'find_dispatch',lambda key:None)
    with pytest.raises(RuntimeUnavailable,match='unresolved'):
        f.runtime.native('inspect',{'source':str(f.root/'source.gpx')},f.root,source=f.source,
            preparation_attempt=attempt,preparation_guard=lambda:True,before_submit=lambda:pytest.fail('No blind checkpoint/POST'))
    fresh=GalaxyModelRuntime(tmp_path/'new-viewer',f.service,poll_seconds=.001)
    monkeypatch.setattr(fresh,'find_dispatch',lambda key:None)
    with pytest.raises(RuntimeUnavailable,match='unresolved'):
        fresh.native('inspect',{'source':str(f.root/'source.gpx')},f.root,source=f.source,
            preparation_attempt=attempt,preparation_guard=lambda:True,before_submit=lambda:pytest.fail('No restored blind POST'))
    monkeypatch.setattr(fresh,'find_dispatch',old_find)
    fresh.native('inspect',{'source':str(f.root/'source.gpx')},f.root,source=f.source,
        preparation_attempt=attempt,preparation_guard=lambda:True,before_submit=lambda:pytest.fail('Existing job needs no POST'))
    assert len(f.service.submissions)==2


@pytest.mark.parametrize('field,value',[('template_sha256','b'*64),('source_sha256','b'*64),
    ('source_id','other'),('dispatch_key','b'*24),('generation',True),('previous_dispatch_key','b'*24)])
def test_tampered_attempt_binding_is_rejected(failed_inspection,field,value):
    f=failed_inspection;attempt,_=prepare_attempt(f.runtime,f.source,'campaign-aaaaaaaaaaaa','a'*64)
    changed=copy.deepcopy(attempt);changed[field]=value
    with pytest.raises(RuntimeUnavailable):validate_attempt(f.runtime,f.source,f.body,changed)
    assert len(f.service.submissions)==1


def test_another_active_inspection_blocks_new_generation(failed_inspection,monkeypatch):
    from radar_pd_nova.native_job_admission import HistoryJobAdmission
    f=failed_inspection
    monkeypatch.setattr(HistoryJobAdmission,'active_jobs',lambda self:[dict(id=f.old_job.galaxy_job_id,tool_id=f.runtime.tool_id)])
    with pytest.raises(RuntimeUnavailable,match='already active'):
        prepare_attempt(f.runtime,f.source,'campaign-aaaaaaaaaaaa','a'*64)
    assert len(f.service.submissions)==1 and not f.runtime._preparation_retry_permits


def test_new_active_inspection_during_checkpoint_blocks_post(failed_inspection,monkeypatch):
    from radar_pd_nova.native_job_admission import HistoryJobAdmission
    f=failed_inspection;attempt,_=prepare_attempt(f.runtime,f.source,'campaign-aaaaaaaaaaaa','a'*64)
    def checkpoint():
        monkeypatch.setattr(HistoryJobAdmission,'active_jobs',lambda self:[dict(id=f.old_job.galaxy_job_id,tool_id=f.runtime.tool_id)])
    with pytest.raises(RuntimeUnavailable,match='already active'):
        f.runtime.native('inspect',{'source':str(f.root/'source.gpx')},f.root,source=f.source,
            preparation_attempt=attempt,preparation_guard=lambda:True,before_submit=checkpoint)
    assert len(f.service.submissions)==1


def test_two_runtime_instances_cannot_overlap_same_source_retries(failed_inspection,monkeypatch,tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from radar_pd_nova.native_job_admission import HistoryJobAdmission
    from radar_pd_nova.model_preparation_retry import submit_attempt
    f=failed_inspection
    first,_=prepare_attempt(f.runtime,f.source,'campaign-aaaaaaaaaaaa','a'*64)
    other=GalaxyModelRuntime(tmp_path/'second-runtime',f.service,poll_seconds=.001)
    monkeypatch.setattr(other,'find_dispatch',lambda key:f.service.jobs.get(key))
    monkeypatch.setattr(other,'wait_capacity',lambda cancel=None:None)
    second,_=prepare_attempt(other,f.source,'campaign-bbbbbbbbbbbb','b'*64)
    monkeypatch.setattr(HistoryJobAdmission,'active_jobs',lambda self:[dict(id=j.galaxy_job_id,tool_id=f.runtime.tool_id)
        for key,j in f.service.jobs.items() if key!=f.key and j.status==RunStatus.RUNNING])
    submit=f.service.submit_utility
    def running(**kwargs):
        job=submit(**kwargs);job.status=RunStatus.RUNNING;return job
    monkeypatch.setattr(f.service,'submit_utility',running)
    entered=threading.Event();release=threading.Event();second_started=threading.Event();second_checkpoint=threading.Event()
    def hold():entered.set();assert release.wait(3)
    def start_second():
        second_started.set()
        return submit_attempt(other,f.source,dict(f.body,dispatch_key=second['dispatch_key']),
            'assets',second,second_checkpoint.set,threading.Event(),lambda:True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        one=pool.submit(submit_attempt,f.runtime,f.source,dict(f.body,dispatch_key=first['dispatch_key']),
            'assets',first,hold,threading.Event(),lambda:True)
        assert entered.wait(3)
        two=pool.submit(start_second)
        try:
            assert second_started.wait(3)
            assert not second_checkpoint.wait(.05)
        finally:release.set()
        assert one.result(3).status==RunStatus.RUNNING
        with pytest.raises(RuntimeUnavailable,match='already active'):two.result(3)
    assert len(f.service.submissions)==2 and not second_checkpoint.is_set()


@pytest.fixture
def retry_app(failed_inspection,tmp_path,monkeypatch):
    import radar_pd_nova.symmetry_controller as controller
    import radar_pd_nova.model_campaigns as campaigns
    f=failed_inspection
    c=Campaign(tmp_path/'campaign-aaaaaaaaaaaa')
    template=dict(scope={'experiment':'experiment'},existing_ids=[f.source['id']])
    c.set_setting('template',template);c.set_setting('paused',True)
    source=dict(f.source,experiment='experiment',sample_key='sample',bank='2',
        collection_id='source-collection',collection_name='final.gpx',scan_label='220611',sample_label='sample')
    entry=dict(id=source['id'],status='needs_review',origin='radar',scan='220611',
        source=campaigns.portable_source(source),reason='Pulsar metadata processing failed: duplicate rule')
    c.set_setting('enrollments',{source['id']:entry})
    monkeypatch.setattr(controller,'validate_template',lambda t:t)
    monkeypatch.setattr(controller,'matches',lambda *args:True)
    monkeypatch.setattr(campaigns,'matches',lambda *args:True)
    workspace=SimpleNamespace(runtime=SimpleNamespace(resolve=lambda value:copy.deepcopy(source)))
    t=GalaxyChunkTransport(f.runtime,c)
    snapshots=[]
    monkeypatch.setattr(t,'checkpoint',lambda **kwargs:snapshots.append(c.snapshot()))
    app=SymmetryMixin();app._symmetry_task=None;app._model_active_campaigns={};app._model_owned_campaigns={}
    app._symmetry_root=lambda:tmp_path;app._symmetry_campaign=lambda root:c
    app._refinement_store=lambda:workspace;app._model_remote=lambda:f.runtime
    app._symmetry_transport=lambda campaign:t;app.refresh_symmetry=lambda:None
    app._refinement_schedule=lambda work,name:asyncio.run(work)
    state=SimpleNamespace(sx_campaign=c.root.name,rf_experiment='experiment',sx_retry_acquisition=source['id'],flush=lambda:None)
    app.server=SimpleNamespace(state=state)
    return SimpleNamespace(app=app,state=state,c=c,t=t,f=f,original=copy.deepcopy(entry),snapshots=snapshots,workspace=workspace)


def test_retry_ui_preserves_failed_attempt_and_queues_only_same_capture(retry_app):
    f=retry_app;template=f.c.setting('template')
    f.app.retry_symmetry_preparation()
    assert not f.state.sx_error and 'Continue' in f.state.sx_retry_notice
    entries=f.c.setting('enrollments');assert set(entries)=={f.original['id']}
    entry=entries[f.original['id']]
    assert entry['status']=='queued' and f.c.setting('paused') is True
    assert entry['source']==f.original['source'] and f.c.setting('template')==template
    history=entry['previous_preparation_attempts']
    assert len(history)==1 and history[0]['enrollment']==f.original
    assert history[0]['failure_proof']['job_id']==f.f.old_job.galaxy_job_id
    assert len(f.f.service.submissions)==1 and len(f.snapshots)==1
    assert f.snapshots[0]['settings']['enrollments']==entries
    f.app.retry_symmetry_preparation()
    assert f.c.setting('enrollments')==entries and len(f.f.service.submissions)==1


@pytest.mark.parametrize('change',['active','incompatible','source','selection','declared','native_failure'])
def test_retry_ui_rejects_unverified_scope_or_failure_without_mutating_ledger(retry_app,monkeypatch,change):
    import radar_pd_nova.model_preparation_retry as retries
    f=retry_app
    if change=='active':f.app._model_active_campaigns[f.c.root.name]=(f.c,f.t)
    elif change=='incompatible':
        entry=copy.deepcopy(f.original);entry['failure_stage']='compatibility';f.c.set_setting('enrollments',{entry['id']:entry})
    elif change=='source':
        resolve=f.workspace.runtime.resolve
        f.workspace.runtime.resolve=lambda value:dict(resolve(value),source_sha256='0'*64)
    elif change=='selection':
        prepare=retries.prepare_attempt
        def changed(*args,**kwargs):
            result=prepare(*args,**kwargs);f.state.sx_campaign='another-campaign';return result
        monkeypatch.setattr(retries,'prepare_attempt',changed)
    elif change=='declared':
        with f.c.connect() as db:db.execute('INSERT INTO protocols VALUES(?,?,?,?)',
            ('p',json.dumps({'acquisition_id':f.original['id']}),'s','{}'))
    elif change=='native_failure':f.f.details['exit_code']=1
    original=f.c.setting('enrollments');template=f.c.setting('template')
    f.app.retry_symmetry_preparation()
    assert f.state.sx_error and f.state.sx_retry_notice=='No preparation retry was queued.'
    assert f.c.setting('enrollments')==original and f.c.setting('template')==template
    assert len(f.f.service.submissions)==1 and not f.snapshots and not f.f.runtime._preparation_retry_permits


def test_retry_ui_checkpoint_pending_is_not_reported_as_ready(retry_app,monkeypatch):
    f=retry_app
    def pending(**kwargs):raise CheckpointUploadPending('pending-checkpoint','queued')
    monkeypatch.setattr(f.t,'checkpoint',pending)
    f.app.retry_symmetry_preparation()
    assert 'pending-checkpoint' in f.state.sx_error
    assert 'queued locally' in f.state.sx_retry_notice
    entry=f.c.setting('enrollments')[f.original['id']]
    assert entry['status']=='queued' and len(entry['previous_preparation_attempts'])==1
    assert len(f.f.service.submissions)==1


@pytest.mark.parametrize('interruption',[None,'pause','owner','template','pending',
    'capacity_owner','capacity_pause','capacity_template','capacity_attempt','capacity_prior'])
def test_enrollment_retry_has_durable_identity_before_post_and_rechecks_scope(retry_app,monkeypatch,tmp_path,interruption):
    import radar_pd_nova.model_campaigns as campaigns
    f=retry_app
    f.app.retry_symmetry_preparation()
    entry=f.c.setting('enrollments')[f.original['id']]
    attempt=entry['preparation_attempt']
    template=f.c.setting('template')
    template['protocol']=dict(acquisition_id='another-reference',source={'path':'source.gpx','sha256':f.f.source['source_sha256']})
    # Rebind the witnessed attempt to this synthetic template before dispatch;
    # the production builder never edits an approved template this way.
    f.c.set_setting('template',template)
    f.f.runtime._preparation_retry_permits.discard(attempt['dispatch_key'])
    attempt,_=prepare_attempt(f.f.runtime,f.f.source,f.c.root.name,digest(template))
    entry['preparation_attempt']=attempt;f.c.set_setting('enrollments',{f.original['id']:entry})
    f.c.set_setting('reference_input_root',str(tmp_path));f.c.set_setting('paused',False)
    monkeypatch.setattr(campaigns,'validate_template',lambda value:value)
    monkeypatch.setattr(campaigns,'resolve_start',lambda *args:f.workspace.runtime.resolve(None))
    monkeypatch.setattr(campaigns,'assets',lambda protocol:[protocol['source']])
    monkeypatch.setattr(campaigns,'rebind',lambda *args:{'same':'reviewed-protocol'})
    monkeypatch.setattr(campaigns,'origin',lambda *args:'radar')
    monkeypatch.setattr(f.t,'prepare_source',lambda *args:None)
    enrolled=[];monkeypatch.setattr(f.c,'enroll',lambda *args,**kwargs:enrolled.append((args,kwargs)))
    snapshots=[]
    def checkpoint(**kwargs):
        snapshot=f.c.snapshot();snapshots.append(snapshot)
        assert snapshot['settings']['enrollments'][f.original['id']]['preparation_attempt']==attempt
        assert len(f.f.service.submissions)==1
        if interruption=='pause':f.c.set_setting('paused',True)
        elif interruption=='owner':f.t.dispatch_guard=lambda:False
        elif interruption=='template':f.c.set_setting('template',dict(template,changed=True))
        elif interruption=='pending':raise CheckpointUploadPending('same-hda','queued')
    monkeypatch.setattr(f.t,'checkpoint',checkpoint)
    def wait_capacity(cancel=None):
        if interruption=='capacity_owner':f.t.dispatch_guard=lambda:False
        elif interruption=='capacity_pause':f.c.set_setting('paused',True)
        elif interruption=='capacity_template':f.c.set_setting('template',dict(template,changed=True))
        elif interruption=='capacity_attempt':
            rows=f.c.setting('enrollments');rows[f.original['id']]['preparation_attempt']['generation']+=1
            f.c.set_setting('enrollments',rows)
        elif interruption=='capacity_prior':f.f.details['state']='running'
    monkeypatch.setattr(f.f.runtime,'wait_capacity',wait_capacity)
    def inspect(source,root,**kwargs):
        value,_=f.f.runtime.native('inspect',{'source':str(root/'source.gpx')},root,source=source,**kwargs)
        return value
    result=campaigns.enroll_source(f.c,f.workspace,f.original['id'],inspect,transport=f.t)
    assert len(snapshots)==1
    if interruption is None:
        assert result['status']=='enrolled' and len(enrolled)==1 and len(f.f.service.submissions)==2
        assert result['previous_preparation_attempts']==entry['previous_preparation_attempts']
    else:
        assert result['status'] in {'queued','needs_review'} and result['failure_stage']=='inspection'
        assert not enrolled and len(f.f.service.submissions)==1


@pytest.mark.parametrize('post_fails',[False,True])
def test_successful_prepost_checkpoint_clears_stale_notice_but_pending_keeps_it(tmp_path,monkeypatch,post_fails):
    from test_model_runtime import staged_wave
    from radar_pd_nova.checkpoint_uploads import CheckpointUploadPending
    # The real transport increments readiness only after save_workspace returns.
    from test_model_runtime import Service
    service=Service(tmp_path);runtime=GalaxyModelRuntime(tmp_path/'runtime',service)
    source_file=tmp_path/'source.gpx';source_file.write_bytes(b'opaque')
    source={'id':'s','dataset_id':'source','source_path':str(source_file),'source_sha256':__import__('hashlib').sha256(b'opaque').hexdigest()}
    monkeypatch.setattr(runtime,'find_dispatch',lambda key:service.jobs.get(key))
    monkeypatch.setattr(runtime,'wait_capacity',lambda cancel=None:None)
    t,service,entries,handles,_=staged_wave((runtime,service,source),tmp_path,monkeypatch,count=1)
    monkeypatch.setattr(t,'checkpoint',GalaxyChunkTransport.checkpoint.__get__(t))
    app=SymmetryMixin();app._model_checkpoint_errors={t.campaign.root.name:RuntimeError('old checkpoint')}
    app._model_stopping=False;app.refresh_symmetry=lambda:None
    app.server=SimpleNamespace(state=SimpleNamespace(sx_slots=2))
    t.dispatch_guard=lambda:True
    t.campaign.set_setting('checkpoint_notice','Old upload interrupted; new batches wait')
    before=t.ready_checkpoint_revision
    saved=[]
    def pending(document):
        saved.append(copy.deepcopy(document))
        raise CheckpointUploadPending('owned-checkpoint','queued')
    monkeypatch.setattr(t.checkpointer,'save_workspace',pending)
    async def retry_delay(seconds):
        assert t.ready_checkpoint_revision==before
        assert t.campaign.setting('checkpoint_notice')
        monkeypatch.setattr(t.checkpointer,'save_workspace',lambda document:saved.append(copy.deepcopy(document)))
    monkeypatch.setattr('radar_pd_nova.symmetry_controller.asyncio.sleep',retry_delay)
    if post_fails:
        def fail_post(*args,**kwargs):raise RuntimeError('POST failed after ready checkpoint')
        monkeypatch.setattr(runtime,'_submit',fail_post)
        with pytest.raises(RuntimeError,match='POST failed'):
            asyncio.run(app._flush_model_dispatch_wave(t.campaign,t))
    else:
        asyncio.run(app._flush_model_dispatch_wave(t.campaign,t))
    assert t.ready_checkpoint_revision==before+1
    assert t.campaign.setting('checkpoint_notice')==''
    assert t.campaign.root.name not in app._model_checkpoint_errors
    assert len(service.submissions)==(0 if post_fails else 1) and saved[0]==saved[1]


@pytest.mark.parametrize('change',['owner','prior','output'])
def test_final_global_admission_rechecks_live_scope_and_failure_proof(failed_inspection,monkeypatch,change):
    from contextlib import contextmanager
    from types import MethodType
    from radar_pd_nova.galaxy_service import GalaxyService
    from radar_pd_nova.native_job_admission import HistoryJobAdmission
    f=failed_inspection;attempt,_=prepare_attempt(f.runtime,f.source,'campaign-aaaaaaaaaaaa','a'*64)
    owner={'valid':True};checkpoint=[]
    @contextmanager
    def slot(self,*args,**kwargs):
        assert checkpoint==['ready']
        if change=='owner':owner['valid']=False
        elif change=='prior':f.details['state']='running'
        else:f.metadata['failed-bundle']['file_size']=1
        yield
    monkeypatch.setattr(HistoryJobAdmission,'slot',slot)
    monkeypatch.setattr(f.service,'submit_utility',MethodType(GalaxyService.submit_utility,f.service))
    monkeypatch.setattr('requests.post',lambda *args,**kwargs:pytest.fail('No POST after failed final admission check'))
    with pytest.raises(RuntimeUnavailable):
        f.runtime.native('inspect',{'source':str(f.root/'source.gpx')},f.root,source=f.source,
            preparation_attempt=attempt,preparation_guard=lambda:owner['valid'],
            before_submit=lambda:checkpoint.append('ready'))
    assert len(f.service.submissions)==1
    assert f.runtime.state['dispatches'][attempt['dispatch_key']]['pending'] is False


def test_owner_change_after_accepted_post_does_not_cancel_that_job(failed_inspection,monkeypatch):
    f=failed_inspection;attempt,_=prepare_attempt(f.runtime,f.source,'campaign-aaaaaaaaaaaa','a'*64)
    owner={'valid':True};submit=f.service.submit_utility
    def accepted(**kwargs):
        job=submit(**kwargs);job.status=RunStatus.RUNNING;owner['valid']=False;return job
    def complete(job):
        job.status=RunStatus.OK;f.service.jobs[attempt['dispatch_key']].status=RunStatus.OK
        return job
    monkeypatch.setattr(f.service,'submit_utility',accepted)
    monkeypatch.setattr(f.service,'refresh_utility',complete)
    monkeypatch.setattr(f.runtime,'cancel_job',lambda *args:pytest.fail('Owner loss cannot cancel an accepted job'))
    result,_=f.runtime.native('inspect',{'source':str(f.root/'source.gpx')},f.root,source=f.source,
        preparation_attempt=attempt,preparation_guard=lambda:owner['valid'],before_submit=lambda:None)
    assert result=={'checked':True} and len(f.service.submissions)==2
