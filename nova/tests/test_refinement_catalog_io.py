"""Bounded catalog reads preserve source identity and main-thread ownership."""
import asyncio
import copy
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from radar_pd_nova.models import RunRecord
from radar_pd_nova.refinement_controller import (
    RefinementMixin, _prefetch_refinement_metadata,
    _refinement_catalog_directory, published_refinement_source,
)


def record(number):
    return RunRecord(uid=f'record-{number}', galaxy_job_id=f'job-{number}',
        name=f'scan-{number}', mode='full', history_id='history', status='ok',
        output_dataset_ids={'summary':f'summary-{number}',
            'gpx_index':f'index-{number}', 'gpx_projects':f'collection-{number}'})


def documents(records):
    values={}
    for number,rec in enumerate(records,1):
        project=dict(stage='final_refinement', status='accepted',
            sha256=f'{number:064x}', published_sha256=f'{number+100:064x}',
            collection_name='03_Final_accepted_model.gpx',
            collection_path='gpx/03_Final_accepted_model.gpx')
        values[rec.output_dataset_ids['summary']]=dict(gpx_projects=[project],
            scientific_status='needs_review', warnings=[dict(message=f'review-{number}')],
            acquisition=dict(instrument='NOM',run_number=number,sample_id=f'sample-{number}'))
        values[rec.output_dataset_ids['gpx_index']]=dict(projects=[project])
    return values


class Service:
    def __init__(self,records):
        self.documents=documents(records);self.calls=[];self.lock=threading.Lock()
        self.active=0;self.peak=0;self.entered=threading.Event();self.release=None
        self.fail=set();self.delays={}
    def _download_dataset(self,identifier,path):
        with self.lock:
            self.calls.append(identifier);self.active+=1;self.peak=max(self.peak,self.active)
            if self.active==2:self.entered.set()
        try:
            if self.release is not None:assert self.release.wait(5)
            time.sleep(self.delays.get(identifier,0))
            if identifier in self.fail:raise OSError('fixture unavailable')
            Path(path).write_text(json.dumps(self.documents[identifier]))
        finally:
            with self.lock:self.active-=1
    def collect_results(self,record):
        pytest.fail('Modern catalog metadata must not collect a full archive')


class Workspace:
    def __init__(self,path):
        self.root=path;self.history_id='history';self.runtime=SimpleNamespace(kind='galaxy')
        self.state={'sources':{},'recipes':{},'jobs':{}};self.observations=[]
    def observe(self,sources,**kwargs):
        self.observations.append((copy.deepcopy(sources),kwargs,threading.get_ident()))
        self.state['sources'].update({s['id']:s for s in sources})


class State(SimpleNamespace):
    def flush(self):pass


class App(RefinementMixin):
    def __init__(self,path,records,service):
        self.server=SimpleNamespace(state=State(**self._refinement_state_defaults(),
            powgen_all_scientific_rows=[],selected_run_uid=records[0].uid))
        self.records={r.uid:r for r in records};self.service=service
        self._refinement_workspace=Workspace(path);self._refinement_sha_cache={}
        self._refinement_refresh_again=False;self.syncs=0;self.pumps=0
    def _refinement_store(self):return self._refinement_workspace
    def _refinement_sync(self):self.syncs+=1
    def _refinement_experiment_changed(self):pass
    def _refinement_start_pump(self):self.pumps+=1


def test_metadata_prefetch_has_exactly_two_workers_and_reuses_complete_cache(tmp_path):
    records=[record(i) for i in range(4)];original=[r.model_dump() for r in records]
    service=Service(records);service.release=threading.Event();workspace=Workspace(tmp_path)
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as driver:
        pending=driver.submit(_prefetch_refinement_metadata,service,workspace,records)
        assert service.entered.wait(5)
        assert service.active==2 and len(service.calls)==2
        service.release.set()
        assert pending.result(timeout=5)=={}
    assert service.peak==2 and len(service.calls)==8
    assert time.monotonic()-started<5
    assert [r.model_dump() for r in records]==original
    sources=[published_refinement_source(service,workspace,r,None,{}) for r in records]
    assert [s['record_uid'] for s in sources]==[r.uid for r in records]
    assert [s['collection_id'] for s in sources]==[f'collection-{i}' for i in range(4)]
    assert [s['scientific_warnings'] for s in sources]==[[f'review-{i}'] for i in range(1,5)]
    assert _prefetch_refinement_metadata(service,workspace,records)=={}
    assert len(service.calls)==8
    assert not list(tmp_path.rglob('*.partial'))


def test_failed_metadata_does_not_retry_or_contaminate_sibling_source(tmp_path):
    records=[record(1),record(2),record(3)];service=Service(records)
    service.fail={'summary-2'};service.delays={'summary-1':.025,'index-3':.005}
    app=App(tmp_path,records,service)
    asyncio.run(app._refresh_refinement_sources())
    sources=app._refinement_workspace.observations[0][0]
    assert [s['record_uid'] for s in sources]==['record-1','record-3']
    assert [s['collection_id'] for s in sources]==['collection-1','collection-3']
    assert [s['scientific_warnings'] for s in sources]==[['review-1'],['review-3']]
    assert service.calls.count('summary-2')==1
    assert len(app.server.state.rf_source_warnings)==1
    assert app.server.state.rf_source_warnings[0]['scan']=='scan-2'
    assert 'fixture unavailable' in app.server.state.rf_source_warnings[0]['reason']
    assert app._refinement_workspace.observations[0][1]['active_live_ids'] is None
    assert app.server.state.rf_loading is False and app.server.state.rf_catalog_message==''
    assert app.pumps==1 and not app._refinement_workspace.state['jobs']


def test_current_selected_choices_remain_visible_and_unselected_options_wait(tmp_path):
    records=[record(1),record(2)];service=Service(records);service.release=threading.Event()
    app=App(tmp_path,records,service);s=app.server.state
    s.rf_experiment='selected-experiment';s.rf_source='selected-source';s.rf_scans=['selected-scan']
    s.rf_experiments=[dict(value='selected-experiment'),dict(value='old-experiment')]
    s.rf_sources=[dict(value='selected-source'),dict(value='old-source')]
    s.rf_scan_options=[dict(value='selected-scan'),dict(value='old-scan')]
    async def run():
        pending=asyncio.create_task(app._refresh_refinement_sources())
        assert await asyncio.to_thread(service.entered.wait,5)
        assert s.rf_loading and 'Loading published project details' in s.rf_catalog_message
        assert s.rf_sources==[dict(value='selected-source')]
        assert s.rf_experiments==[dict(value='selected-experiment')]
        assert s.rf_scan_options==[dict(value='selected-scan')]
        # A user selection made while reads are pending is not overwritten.
        s.rf_experiment='user-choice';s.rf_source='user-source';s.rf_selected_job='user-job'
        service.release.set();await pending
    asyncio.run(run())
    assert (s.rf_experiment,s.rf_source,s.rf_selected_job)==('user-choice','user-source','user-job')
    assert len(app._refinement_workspace.state['sources'])==2


@pytest.mark.parametrize('mutation',['record','row'])
def test_changed_source_identity_during_read_is_not_applied(tmp_path,mutation):
    records=[record(1),record(2)];service=Service(records);service.release=threading.Event()
    app=App(tmp_path,records,service)
    async def run():
        pending=asyncio.create_task(app._refresh_refinement_sources())
        assert await asyncio.to_thread(service.entered.wait,5)
        if mutation=='record':app.records['record-1'].output_dataset_ids['summary']='replacement-summary'
        else:app.server.state.powgen_all_scientific_rows=[dict(record_uid='record-1',sample_key='new-sample')]
        service.release.set();await pending
    asyncio.run(run())
    assert [s['record_uid'] for s in app._refinement_workspace.observations[0][0]]==['record-2']
    assert any('changed while' in w['reason'] for w in app.server.state.rf_source_warnings)
    assert 'replacement-summary' not in service.calls


def test_replaced_workspace_ignores_old_reads_and_keeps_new_ui_state(tmp_path):
    records=[record(1)];service=Service(records);service.release=threading.Event()
    app=App(tmp_path,records,service);old=app._refinement_workspace
    async def run():
        pending=asyncio.create_task(app._refresh_refinement_sources())
        assert await asyncio.to_thread(service.entered.wait,5)
        app._refinement_workspace=Workspace(tmp_path/'replacement')
        app.server.state.rf_catalog_message='new owner';app.server.state.rf_loading=True
        service.release.set();await pending
    asyncio.run(run())
    assert old.observations==[] and app._refinement_workspace.observations==[]
    assert app.server.state.rf_catalog_message=='new owner' and app.server.state.rf_loading is True
    assert app.syncs==0 and app.pumps==0


def test_invalid_document_is_not_published_or_left_as_a_partial(tmp_path):
    records=[record(1)];service=Service(records);service.documents['summary-1']=['not a summary']
    failures=_prefetch_refinement_metadata(service,Workspace(tmp_path),records)
    assert 'not a JSON object' in failures['record-1'][0]
    directory=_refinement_catalog_directory(Workspace(tmp_path),records[0])
    assert not (directory/'summary.json').exists()
    assert not list(directory.glob('*.partial'))


def test_catalog_refresh_preserves_other_experiment_sources(tmp_path):
    records=[record(1)];service=Service(records);app=App(tmp_path,records,service)
    other=dict(id='other-source',experiment='other-experiment',experiment_label='Other',
        record_uid='other-record',live_watch=False)
    app._refinement_workspace.state['sources']['other-source']=other
    asyncio.run(app._refresh_refinement_sources())
    assert app._refinement_workspace.state['sources']['other-source']==other
    assert len(app._refinement_workspace.state['sources'])==2


def test_recovery_uses_private_records_and_cannot_replace_a_changed_job(tmp_path):
    original=record(1);original.status='queued';original.output_dataset_ids={}
    service=Service([]);entered=threading.Event();release=threading.Event()
    app=App(tmp_path,[original],service)
    run=SimpleNamespace(galaxy_job_id='job-1',fingerprint='first')
    app._powgen_controller=SimpleNamespace(state=SimpleNamespace(completed={'scan':run}),
        records={'scan':original},_minimal_record=lambda run:None)
    def refresh(snapshot):
        assert snapshot is not original
        entered.set();assert release.wait(5)
        snapshot.status='ok';snapshot.output_dataset_ids={'summary':'recovered-old'}
        return snapshot
    service.refresh=refresh
    async def work():
        pending=asyncio.create_task(app._refresh_refinement_sources())
        assert await asyncio.to_thread(entered.wait,5)
        app._powgen_controller.state.completed['scan']=SimpleNamespace(galaxy_job_id='replacement',fingerprint='second')
        original.galaxy_job_id='replacement'
        release.set();await pending
    asyncio.run(work())
    assert original.output_dataset_ids=={} and original.status=='queued'
    assert app._powgen_controller.records['scan'] is original
    assert app._refinement_workspace.observations[0][0]==[]


def test_loading_message_is_rendered_in_refinement_workspace():
    from radar_pd_nova.app import RadarPdNovaApp
    app=RadarPdNovaApp()
    assert 'rf_catalog_message' in app.layout.html
    assert app.server.state.rf_catalog_message==''


def test_legacy_collection_stays_serial_and_retains_verified_cache(tmp_path):
    modern=record(1);legacy=record(2);legacy.output_dataset_ids={'results_archive':'old-archive'}
    service=Service([modern]);calls=[]
    def collect(snapshot):
        assert snapshot is not legacy and service.active==0
        calls.append(snapshot.uid)
        root=tmp_path/'legacy';(root/'gpx').mkdir(parents=True)
        raw=b'fixture published project';(root/'gpx/legacy.gpx').write_bytes(raw)
        checksum=hashlib.sha256(raw).hexdigest()
        project=dict(stage='final_refinement',status='accepted',sha256=checksum,
            published_sha256=checksum,collection_name='legacy.gpx',collection_path='gpx/legacy.gpx')
        (root/'summary.json').write_text(json.dumps(dict(gpx_projects=[project])))
        (root/'gpx_index.json').write_text(json.dumps(dict(projects=[project])))
        snapshot.output_dir=str(root);snapshot.result_status='ready'
        return snapshot
    service.collect_results=collect
    app=App(tmp_path,[modern,legacy],service)
    asyncio.run(app._refresh_refinement_sources())
    assert calls==['record-2'] and legacy.output_dir==str(tmp_path/'legacy')
    assert legacy.result_status=='ready'
    assert [s['record_uid'] for s in app._refinement_workspace.observations[0][0]]==['record-1','record-2']
