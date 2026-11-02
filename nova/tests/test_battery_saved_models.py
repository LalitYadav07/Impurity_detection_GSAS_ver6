"""Synthetic publication and stopped-recovery controls; no native dispatch."""
import asyncio
from copy import deepcopy
import io
import json
from types import SimpleNamespace
import threading
import time
import zipfile

import pytest

from radar_pd_nova import battery_saved_models as saved
from radar_pd_nova import battery_preparation as preparation
from radar_pd_nova.battery_controller import BatteryMixin
from radar_pd_nova.battery_transport import write_preparation_packet
from radar_pd_nova.refinement_sources import file_sha
from test_battery_preparation_controller import State


class Service:
    galaxy_url='https://galaxy.invalid'
    history_id='history'
    _headers={'x-api-key':'not-a-credential'}

    @staticmethod
    def _job_output_ids(row):
        return {key:value['id'] for field in ('outputs','output_collections')
                for key,value in row.get(field,{}).items()}


class Response:
    status_code=200
    def __init__(self,value): self.data=value if isinstance(value,bytes) else json.dumps(value).encode()
    def iter_content(self,chunk_size):
        for offset in range(0,len(self.data),chunk_size):yield self.data[offset:offset+chunk_size]
    def close(self):pass


def job(ident='job',action='prepare'):
    return dict(id=ident,history_id='history',tool_id=saved.REFINEMENT_TOOL_ID,state='ok',
        create_time='2026-01-01T00:00:00',params={'action':json.dumps(action),'dispatch_key':json.dumps('d'*24)},
        inputs={'source_gpx':{'id':'packet','src':'hda'},'recipe_request':{'id':'request','src':'hda'}},
        outputs={'result_bundle':{'id':'bundle','src':'hda'}},
        output_collections={'stage_projects':{'id':'collection','src':'hdca'}})


class Router:
    def __init__(self):
        self.rows={};self.blobs={};self.jobs={'job':job()};self.calls=[]
        self.after=None;self.active=0;self.peak=0;self.lock=threading.Lock();self.delay=0
    def get(self,url,**kwargs):
        path=url.split('galaxy.invalid',1)[-1]
        self.calls.append((path,deepcopy(kwargs)))
        assert kwargs['allow_redirects'] is False
        assert kwargs['headers']=={'x-api-key':'not-a-credential'}
        with self.lock:self.active+=1;self.peak=max(self.peak,self.active)
        try:
            if self.delay:time.sleep(self.delay)
            if path=='/api/jobs':
                start=kwargs['params']['offset'];limit=kwargs['params']['limit']
                value=list(self.jobs.values())[start:start+limit]
            elif path.startswith('/api/jobs/'):value=self.jobs[path.rsplit('/',1)[-1]]
            elif path=='/api/histories/history/contents':value=[]
            elif path=='/api/dataset_collections/collection':
                value={'id':'collection','history_id':'history','elements':[{'element_identifier':'model','object':{'id':'model'}}]}
            elif path.endswith('/display'):value=self.blobs[path.split('/')[-2]]
            else:value=self.rows[path.rsplit('/',1)[-1]]
            if self.after:self.after(path)
            return Response(deepcopy(value))
        finally:
            with self.lock:self.active-=1
    def blob(self,ident,data):
        self.blobs[ident]=data
        self.rows[ident]=dict(id=ident,history_id='history',state='ok',file_size=len(data),deleted=False)


@pytest.fixture
def transport(monkeypatch):
    router=Router();monkeypatch.setattr(saved.requests,'get',router.get)
    return router


def bundle(tmp_path,router):
    root=tmp_path/'source';inputs=root/'inputs';model=root/'model'
    inputs.mkdir(parents=True);model.mkdir()
    for name,data in [('observation.gsa',b'example measured data'),('profile.instprm',b'profile'),('phase.cif',b'synthetic phase')]:
        (inputs/name).write_bytes(data)
    def ref(name):return dict(path=name,sha256=file_sha(inputs/name))
    spec=dict(schema='radar-battery-preparation/v1',
        acquisition=dict(instrument='NOM',ipts='IPTS-1',run=2,sample_id='sample',cell_id='cell',
            start_utc='2026-01-01T00:00:00Z',end_utc='2026-01-01T00:01:00Z',metadata_sha256='a'*64),
        reduction_family='original',review_note='Unqualified example profile',
        banks=[dict(key='b2',physical_bank=2,data=ref('observation.gsa'),profile=ref('profile.instprm'),
                    profile_bank=1,fit_d=[.7,4.5],profile_review='Explicit unqualified mapping')],
        phases=[dict(key='phase',name='Example',role='other',cif=ref('phase.cif'),
                     links=[dict(bank='b2',hap_scale=.1)],hypothesis_note='Synthetic test model')])
    (inputs/'preparation.json').write_text(json.dumps(spec))
    (model/'prepared.gpx').write_bytes(b'synthetic binary model')
    result=dict(schema='radar-battery-preparation-result/v1',status='prepared_needs_review',
                specification=spec,project_sha256=file_sha(model/'prepared.gpx'),banks=[],phases=[])
    (model/'preparation.json').write_text(json.dumps(result))
    (root/'verified_model.gpx').write_bytes((model/'prepared.gpx').read_bytes())
    packet=tmp_path/'packet.zip';write_preparation_packet(inputs,packet)
    response=dict(schema='radar-refinement-worker/v1',action='prepare',status='model_prepared',
        source_sha256=file_sha(packet),output_sha256=result['project_sha256'],
        final_gpx='verified_model.gpx',preparation=result,implementation_sha256='a'*64)
    response['files']={p.relative_to(root).as_posix():file_sha(p) for p in root.rglob('*') if p.is_file()}
    (root/'response.json').write_text(json.dumps(response))
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w') as z:
        for p in root.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(root))
    router.blob('bundle',output.getvalue());router.blob('packet',packet.read_bytes())
    router.blob('model',(model/'prepared.gpx').read_bytes())
    router.blob('request',json.dumps(dict(expected_source_sha256=response['source_sha256'],
        dispatch_key='d'*24,plan={'implementation_sha256':'a'*64})).encode())
    return root,result


def test_listing_only_metadata_bounded_parallel_and_one_failure_isolated(transport):
    transport.jobs={f'job{i}':job(f'job{i}',action='prepare' if i%2 else 'inspect') for i in range(7)}
    transport.jobs['job1']['history_id']='other';transport.delay=.01
    rows,failures,truncated=saved.SavedModelReader(Service()).list_models()
    assert [r['job_id'] for r in rows]==['job3','job5']
    assert failures==1 and not truncated and transport.peak==2
    assert all(p=='/api/jobs' or p.startswith('/api/jobs/') for p,_ in transport.calls)


def test_truthful_catalog_limit(transport,monkeypatch):
    monkeypatch.setattr(saved,'MAX_JOBS',2);monkeypatch.setattr(saved,'PAGE_SIZE',2)
    transport.jobs={f'job{i}':job(f'job{i}') for i in range(4)}
    rows,_,truncated=saved.SavedModelReader(Service()).list_models()
    assert len(rows)==2 and truncated
    assert len([p for p,_ in transport.calls if p=='/api/jobs'])==1


@pytest.mark.parametrize('change',['history','origin','actor'])
def test_context_revocation_stops_before_next_request(transport,change):
    service=Service();reader=saved.SavedModelReader(service)
    def mutate(path):
        if path=='/api/jobs':
            if change=='history':service.history_id='other'
            elif change=='origin':service.galaxy_url='https://other.invalid'
            else:service._headers={'x-api-key':'different'}
    transport.after=mutate
    with pytest.raises(saved.StaleSavedModel):reader.list_models()
    assert [p for p,_ in transport.calls]==['/api/jobs']


def test_explicit_open_verifies_packet_request_model_and_artifacts(transport,tmp_path):
    _,result=bundle(tmp_path,transport);reader=saved.SavedModelReader(Service())
    publication=reader.list_models()[0][0]
    root,recovered=saved.recover_preparation(reader,publication,tmp_path/'recovered')
    assert recovered==result and preparation.verify_prepared(root)==result
    record=json.loads((root/'preparation-publication.json').read_text())
    assert record['dataset_id']=='model' and record['galaxy_job_id']=='job' and record['history_id']=='history'
    assert record['source_sha256']==result['project_sha256']
    assert {p.name for p in (root/'inputs').iterdir()}=={p.name for p in (root/'original-inputs').iterdir()}


@pytest.mark.parametrize('change',['request_hash','model_hash','packet_bytes','artifact_bytes','wrong_history','deleted_model','job_output','request_dispatch'])
def test_incorrect_publication_fails_closed(transport,tmp_path,change):
    bundle(tmp_path,transport);reader=saved.SavedModelReader(Service());publication=reader.list_models()[0][0]
    if change in {'request_hash','request_dispatch'}:
        value=json.loads(transport.blobs['request'])
        value['expected_source_sha256' if change=='request_hash' else 'dispatch_key']='f'*64
        transport.blob('request',json.dumps(value).encode())
    elif change=='model_hash':transport.blob('model',b'other model')
    elif change=='packet_bytes':transport.blob('packet',b'not the captured packet')
    elif change=='artifact_bytes':
        old=zipfile.ZipFile(io.BytesIO(transport.blobs['bundle']));output=io.BytesIO()
        with zipfile.ZipFile(output,'w') as z:
            for name in old.namelist():z.writestr(name,b'changed' if name=='model/prepared.gpx' else old.read(name))
        transport.blob('bundle',output.getvalue())
    elif change=='wrong_history':transport.rows['model']['history_id']='other'
    elif change=='deleted_model':transport.rows['model']['deleted']=True
    else:transport.jobs['job']['outputs']['result_bundle']['id']='other'
    with pytest.raises(ValueError):saved.recover_preparation(reader,publication,tmp_path/'recovered')
    assert not list((tmp_path/'recovered').glob('*'))


@pytest.mark.parametrize('activation',['armed','queued','running','recovering','pump'])
def test_queue_activation_during_download_blocks_install(monkeypatch,tmp_path,activation):
    app=app_instance(Service());s=app.server.state
    app._battery_saved_models={'choice':dict(source='galaxy',job_id='job')};s.battery_saved_model='choice'
    w=SimpleNamespace(lock=threading.RLock(),state={'recipes':{},'jobs':{}},parallel_jobs=1)
    app._refinement_workspace=w
    prior=(tmp_path/'prior',{'previous':True});app._battery_prepared=prior
    def recover(*args):
        with w.lock:
            if activation=='armed':w.state['recipes']['r']={'armed':True}
            elif activation=='pump':s.rf_queue_running=True
            else:w.state['jobs']['j']={'status':activation}
        return tmp_path,{'new':True}
    monkeypatch.setattr(saved,'recover_preparation',recover)
    app.open_battery_refinement=lambda **_:pytest.fail('Active workspace was overwritten')
    async def run():
        app.open_saved_battery_model();await app._battery_task
    asyncio.run(run())
    assert app._refinement_workspace is w and app._battery_prepared==prior
    assert 'Pause the current' in s.battery_error and not s.battery_saved_busy
    assert s.battery_saved_model=='choice'


@pytest.mark.parametrize('alias',['./response.json','dir//file.json','dir/./file.json'])
def test_noncanonical_archive_paths_rejected(tmp_path,alias):
    path=tmp_path/'alias.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('response.json',b'{}')
        z.writestr(alias,b'{}')
    with pytest.raises(ValueError,match='unsafe or duplicate'):saved._checked_archive(path)



@pytest.mark.parametrize('name',['../escape','/escape','model\\escape','response.json'])
def test_unsafe_or_duplicate_archive(tmp_path,name):
    path=tmp_path/'bad.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('response.json',b'{}');z.writestr(name,b'{}')
    with pytest.raises(ValueError):saved._checked_archive(path)


def test_checkpoint_preserves_records_and_holds_work(transport,tmp_path,monkeypatch):
    document=dict(schema='radar-refinement-workspace/v1',history_id='history',sources={'source':{'id':'source'}},
        drafts={'draft':{'status':'ready'}},recipes={'r':{'armed':True}},
        jobs={key:{'status':key,'remote_action':{'id':'accepted'} if key=='running' else None}
              for key in ['queued','running','completed','failed','cancelled']},parallel_jobs=1)
    raw=json.dumps(document).encode();transport.blob('checkpoint',raw)
    original=transport.get
    def get(url,**kwargs):
        if '/contents' in url:
            transport.calls.append(('/api/histories/history/contents',kwargs))
            return Response([dict(id='checkpoint',history_content_type='dataset',state='ok')])
        return original(url,**kwargs)
    monkeypatch.setattr(saved.requests,'get',get)
    w=saved.recover_review_workspace(saved.SavedModelReader(Service()),tmp_path/'workspace')
    assert not w.state['recipes']['r']['armed']
    assert w.state['jobs']['queued']['status']==w.state['jobs']['running']['status']=='paused'
    assert [w.state['jobs'][k]['status'] for k in ['completed','failed','cancelled']]==['completed','failed','cancelled']
    assert w.state['jobs']['running']['remote_action']=={'id':'accepted'}
    assert w.state['sources']==document['sources'] and w.state['drafts']==document['drafts']
    assert transport.blobs['checkpoint']==raw and w.run_next() is None


def test_register_without_future_enrollment(tmp_path):
    from radar_pd_nova.refinement_workspace import RefinementWorkspace
    w=RefinementWorkspace(tmp_path,history_id='history',runtime=SimpleNamespace(kind='galaxy'))
    w.state['recipes']['r']={'armed':True}
    w._enroll_future=lambda recipe:pytest.fail('Historical model enrolled future work')
    w.observe([dict(id='source',source_sha256='a'*64)],enroll_future=False)
    assert w.state['sources']['source']['source_sha256']=='a'*64 and w.state['recipes']['r']['armed']


def app_instance(service):
    app=BatteryMixin();app.server=SimpleNamespace(state=State());app._init_battery()
    app.service=service;app._refinement_workspace=None;app._result_context=object()
    return app


@pytest.mark.parametrize('change',['clear','history','selection','workspace','revision'])
def test_inflight_open_does_not_replace_new_context(monkeypatch,tmp_path,change):
    app=app_instance(Service());s=app.server.state
    app._battery_saved_models={'choice':dict(source='galaxy',job_id='job')};s.battery_saved_model='choice'
    previous=(tmp_path/'previous',{'old':True});app._battery_prepared=previous
    def recover(reader,*args):
        if change=='clear':app._battery_context_changed()
        elif change=='history':app.service.history_id='other'
        elif change=='selection':s.battery_saved_model='other'
        elif change=='workspace':app._refinement_workspace=object()
        else:app._battery_preparation_edited()
        return tmp_path,{'recovered':True}
    monkeypatch.setattr(saved,'recover_preparation',recover)
    monkeypatch.setattr(saved,'recover_review_workspace',lambda *a:pytest.fail('Stale read opened a workspace'))
    app.open_battery_refinement=lambda **_:pytest.fail('Stale read changed the visible model')
    async def run():
        app.open_saved_battery_model();await app._battery_task
    asyncio.run(run())
    assert app._battery_prepared==(None if change in {'clear','revision'} else previous)
    assert not s.battery_saved_busy


def test_successful_open_installs_review_workspace_only(monkeypatch,tmp_path):
    app=app_instance(Service());s=app.server.state
    app._battery_saved_models={'choice':dict(source='galaxy',job_id='job')};s.battery_saved_model='choice'
    w=SimpleNamespace(runtime=SimpleNamespace(service=None),parallel_jobs=1,
        lock=threading.RLock(),state={'recipes':{},'jobs':{}})
    result={'specification':{'acquisition':dict(instrument='NOM',ipts='IPTS-1',run=2,sample_id='sample')}}
    monkeypatch.setattr(saved,'recover_preparation',lambda *a:(tmp_path,result))
    monkeypatch.setattr(saved,'recover_review_workspace',lambda *a:w)
    calls=[];app.open_battery_refinement=lambda **kwargs:calls.append(kwargs)
    async def run():
        app.open_saved_battery_model();await app._battery_task
    asyncio.run(run())
    assert app._refinement_workspace is w and w.runtime.service is app.service
    assert app._battery_prepared==(tmp_path,result)
    assert calls==[{'recovered':True,'raise_errors':True}]
    assert not s.battery_saved_busy and 'remain stopped' in s.battery_saved_notice
    assert s.battery_model_ready and s.rf_parallel_jobs==1


def test_original_source_identity_preserved_without_enrollment(transport,tmp_path):
    root,result=bundle(tmp_path,transport)
    (root/'preparation-publication.json').write_text(json.dumps(dict(
        dataset_id='model',source_sha256=result['project_sha256'],history_id='history')))
    acquisition=result['specification']['acquisition']
    old=dict(id='legacy-id',experiment='legacy-experiment',experiment_label='Saved experiment',
        scan_key='legacy-scan',record_uid='legacy-record',run_id='NOM_2',run_number=2,bank='2',
        sample_key='id:sample',sample_label='Saved sample',scan_label='Saved scan',metadata=acquisition,
        dataset_id='model',source_sha256=result['project_sha256'])
    calls=[]
    runtime=SimpleNamespace(kind='galaxy',resolve=lambda source:source)
    workspace=SimpleNamespace(history_id='history',runtime=runtime,state={'sources':{'legacy-id':old},'jobs':{}},
        observe=lambda sources,**kwargs:calls.append((sources,kwargs)))
    source=preparation.prepared_source(workspace,root,result,enroll_future=False)
    assert source['id']=='legacy-id' and source['scan_key']=='legacy-scan'
    assert source['experiment']=='legacy-experiment' and calls[0][1]=={'enroll_future':False}
    assert old['metadata']==acquisition
    old['bank']='5'
    with pytest.raises(ValueError,match='conflicts'):
        preparation.prepared_source(workspace,root,result,enroll_future=False)
    assert len(calls)==1


@pytest.mark.parametrize('change',['clear','history','selection'])
def test_second_await_stale_does_not_install_workspace(monkeypatch,tmp_path,change):
    app=app_instance(Service());s=app.server.state
    app._battery_saved_models={'choice':dict(source='galaxy',job_id='job')};s.battery_saved_model='choice'
    monkeypatch.setattr(saved,'recover_preparation',lambda *a:(tmp_path,{'ok':True}))
    def restore(*args):
        if change=='clear':app._battery_context_changed()
        elif change=='history':app.service.history_id='other'
        else:s.battery_saved_model='new'
        return SimpleNamespace(runtime=SimpleNamespace(service=None))
    monkeypatch.setattr(saved,'recover_review_workspace',restore)
    app.open_battery_refinement=lambda **kwargs:pytest.fail('Stale restore reached UI')
    async def run():
        app.open_saved_battery_model();await app._battery_task
    asyncio.run(run())
    assert app._refinement_workspace is None and not s.battery_saved_busy


def test_active_work_must_be_paused_before_historical_open(monkeypatch,tmp_path):
    app=app_instance(Service());s=app.server.state
    app._battery_saved_models={'choice':dict(source='galaxy',job_id='job')};s.battery_saved_model='choice'
    app._refinement_workspace=SimpleNamespace(lock=threading.RLock(),state={'recipes':{'r':{'armed':True}},'jobs':{}})
    monkeypatch.setattr(saved,'recover_preparation',lambda *a:pytest.fail('Active recipe was ignored'))
    async def run():
        app.open_saved_battery_model();await app._battery_task
    asyncio.run(run())
    assert 'Pause the current' in s.battery_error and not s.battery_saved_busy


def test_recovered_handoff_uses_exact_publication_and_configured_worker(monkeypatch,tmp_path):
    from radar_pd_nova import refinement_controller
    app=app_instance(Service());app._battery_prepared=(tmp_path,{'ok':True})
    w=SimpleNamespace(runtime=SimpleNamespace(kind='galaxy'))
    app._refinement_store=lambda:w
    calls=[]
    def source(*args,**kwargs):
        calls.append(kwargs)
        return {'id':'original-id','experiment':'original-experiment'}
    monkeypatch.setattr(preparation,'prepared_source',source)
    monkeypatch.setattr(refinement_controller,'refinement_experiment_options',lambda w:{'original-experiment':'Original'})
    app._refinement_sync=lambda:None
    app.open_battery_refinement(recovered=True,raise_errors=True)
    s=app.server.state
    assert calls==[{'enroll_future':False}]
    assert s.rf_scans==['source:original-id'] and s.rf_source=='original-id'
    assert s.rf_transport=='Galaxy scientific jobs' and not s.rf_future


@pytest.mark.parametrize('fault',['foreign_collection','duplicate_model','redirect','bad_size','stale_download'])
def test_additional_transport_and_collection_guards(transport,tmp_path,monkeypatch,fault):
    bundle(tmp_path,transport);alive=[True]
    reader=saved.SavedModelReader(Service(),current=lambda:alive[0])
    publication=reader.list_models()[0][0]
    original=transport.get
    def altered(url,**kw):
        if fault=='foreign_collection' and '/dataset_collections/' in url:
            return Response({'id':'collection','history_id':'other','elements':[]})
        if fault=='duplicate_model' and '/dataset_collections/' in url:
            return Response({'id':'collection','history_id':'history','elements':[
                {'element_identifier':'model','object':{'id':'model'}},
                {'element_identifier':'model','object':{'id':'model2'}}]})
        response=original(url,**kw)
        if fault=='redirect':
            response.status_code=302
        if fault=='stale_download' and '/datasets/packet/display' in url:alive[0]=False
        return response
    if fault=='bad_size':transport.rows['model']['file_size']=saved.MAX_PROJECT+1
    monkeypatch.setattr(saved.requests,'get',altered)
    with pytest.raises((ValueError,saved.StaleSavedModel)):
        saved.recover_preparation(reader,publication,tmp_path/'recovered')
    assert not list((tmp_path/'recovered').glob('*'))



