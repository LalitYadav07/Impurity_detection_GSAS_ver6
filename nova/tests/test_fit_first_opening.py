"""Partial fit display cannot publish caches, native authority or stale UI."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace

import pytest

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.galaxy_service import GalaxyService
from radar_pd_nova.models import AnalysisMode, InputSelection, ResultStatus, RunRecord, RunStatus
from radar_pd_nova.results import read_plot_payload
from test_results import _bound_plot_fixture


def setup_preview(tmp_path, *, status=ResultStatus.READY, failure=False):
    app=RadarPdNovaApp()
    record=RunRecord(uid='A',galaxy_job_id='A',history_id='history',name='Published A',
        mode=AnalysisMode.FULL,status=RunStatus.OK,output_dataset_ids={'summary':'summary-A'})
    app.records={record.uid:record};app._sync_runs=lambda:None
    app._publish_results_if_requested=lambda row,**_:row
    app._refinement_maybe_refresh=lambda:None
    preview_root=tmp_path/'preview';preview_root.mkdir()
    path,_=_bound_plot_fixture(preview_root)
    (preview_root/'.radar-pd-cache.json').unlink()
    receipt=dict(schema='radar-pd-result-preview/v1',job_id='A',history_id='history')
    (preview_root/'.radar-pd-preview.json').write_text(json.dumps(receipt))
    binding=json.loads((preview_root/'plot_observation_binding.json').read_text())
    binding['job_id']='A';(preview_root/'plot_observation_binding.json').write_text(json.dumps(binding))
    document=json.loads((preview_root/'summary.json').read_text())
    document.update({'$schema':'radar-pd-result/v1'},analysis_mode='full',status='complete',final_rwp=3.5)
    (preview_root/'summary.json').write_text(json.dumps(document))
    full_root=tmp_path/'canonical';shutil.copytree(preview_root,full_root)
    (full_root/'.radar-pd-preview.json').unlink()
    (full_root/'.radar-pd-cache.json').write_text(json.dumps({'job_id':'A'}))
    (full_root/'final.gpx').write_bytes(b'opaque native data')
    preview=SimpleNamespace(root=preview_root,output_dataset_ids=tuple(record.output_dataset_ids.items()))
    entered,release=threading.Event(),threading.Event();calls=[]
    def collect(row,**_):
        calls.append(row.uid);entered.set()
        assert release.wait(5),'The test must release canonical collection'
        if failure:raise OSError('canonical archive unavailable')
        row.output_dir=str(full_root);row.result_status=status
        if status!=ResultStatus.READY:row.message='GPX download incomplete'
        return row
    app.service=SimpleNamespace(collect_results=collect)
    return app,record,preview,path,entered,release,calls


async def until(predicate):
    async def wait():
        while not predicate():await asyncio.sleep(.005)
    await asyncio.wait_for(wait(),2)


def test_fit_is_visible_before_blocked_archive_and_native_actions_remain_gated(tmp_path,monkeypatch):
    app,row,preview,path,entered,release,calls=setup_preview(tmp_path)
    previews=[];native=[]
    def read(*_):previews.append(row.uid);return preview
    monkeypatch.setattr('radar_pd_nova.result_preview.collect_result_preview',read)
    app._schedule_utility=lambda *a,**k:native.append(a)
    app.download_file=lambda *a,**k:native.append(a)
    async def run():
        app._select_record(row);app._open_record_results(row)
        try:
            assert await asyncio.to_thread(entered.wait,1)
            await until(lambda:app.server.state.result_preview)
            s=app.server.state
            assert s.result_owner_uid==row.uid and s.plot_options and s.result_metrics
            assert 'Fit ready' in s.result_readiness_message and s.selected_run_loading
            assert row.output_dir is None and app.records[row.uid].output_dir is None
            assert row.result_status==ResultStatus.NOT_REQUESTED
            assert not s.file_groups and not s.checkpoint_rows and not s.artifact_options
            assert read_plot_payload(path)['_observation_validity']['status']=='verified'
            app.launch_result_explorer();app.open_selected_checkpoint_in_gsasii()
            app.handoff_selected_checkpoint();app.download_checkpoint();app.download_artifact(str(path))
            assert not native and not app._result_actions_ready(row.uid)
            # Opening through another entry point shares both phases.
            app._open_record_results(row)
            assert len(previews)==1 and calls==['A']
        finally:
            release.set();await asyncio.gather(*list(app._history_result_requests.values()))
        assert not s.result_preview and not s.result_readiness_message and not s.selected_run_loading
        assert app.records[row.uid].output_dir.endswith('canonical')
        assert app.records[row.uid].result_status==ResultStatus.READY
        assert app._result_actions_ready(row.uid) and s.file_groups
    asyncio.run(run())


@pytest.mark.parametrize('failure',[False,True])
def test_failed_canonical_enrichment_keeps_preview_partial(tmp_path,monkeypatch,failure):
    app,row,preview,_,entered,release,_=setup_preview(tmp_path,status=ResultStatus.STALE,failure=failure)
    published=[]
    app._publish_results_if_requested=lambda *a,**k:published.append(a) or a[0]
    monkeypatch.setattr('radar_pd_nova.result_preview.collect_result_preview',lambda *_:preview)
    async def run():
        app._select_record(row);app._open_record_results(row)
        try:await until(lambda:app.server.state.result_preview)
        finally:
            release.set();await asyncio.gather(*list(app._history_result_requests.values()))
        s=app.server.state
        assert s.workspace_view=='results' and s.result_owner_uid==row.uid and s.result_preview
        assert 'incomplete' in s.result_readiness_message and not s.selected_run_loading
        assert not app._result_actions_ready(row.uid)
        assert row.status==RunStatus.OK and not s.checkpoint_rows and not s.file_groups
        assert not published
    asyncio.run(run())


@pytest.mark.parametrize('change',['selection','context','service','hda'])
def test_late_preview_cannot_replace_current_source(tmp_path,monkeypatch,change):
    app,row,preview,_,entered,release,_=setup_preview(tmp_path)
    preview_entered,preview_release=threading.Event(),threading.Event()
    def read(*_):
        preview_entered.set();assert preview_release.wait(5);return preview
    monkeypatch.setattr('radar_pd_nova.result_preview.collect_result_preview',read)
    async def run():
        app._select_record(row);app._open_record_results(row)
        try:
            assert await asyncio.to_thread(preview_entered.wait,1)
            if change=='selection':app.server.state.selected_run_uid='B'
            elif change=='context':app._result_context=object()
            elif change=='service':app.service=SimpleNamespace()
            else:app.records[row.uid].output_dataset_ids={'summary':'replacement'}
            preview_release.set();await asyncio.sleep(.05)
            assert not app.server.state.result_preview
            assert app.server.state.result_owner_uid!=row.uid
        finally:
            preview_release.set();release.set()
            await asyncio.gather(*list(app._history_result_requests.values()))
    asyncio.run(run())


def test_wrong_preview_hda_never_becomes_a_displayed_result(tmp_path,monkeypatch):
    app,row,preview,_,entered,release,_=setup_preview(tmp_path)
    preview.output_dataset_ids=(('summary','another-job-summary'),)
    monkeypatch.setattr('radar_pd_nova.result_preview.collect_result_preview',lambda *_:preview)
    async def run():
        app._select_record(row);app._open_record_results(row)
        try:
            assert await asyncio.to_thread(entered.wait,1)
            await asyncio.sleep(.05)
            assert not app.server.state.result_preview and app.server.state.result_owner_uid!=row.uid
        finally:
            release.set();await asyncio.gather(*list(app._history_result_requests.values()))
    asyncio.run(run())


def test_monitor_first_collection_leaves_a_shared_preview_path(tmp_path,monkeypatch):
    app,row,preview,_,entered,release,calls=setup_preview(tmp_path)
    readers=[];previews=[]
    original=app.service.collect_results
    def display_reader(record,**kwargs):
        readers.append(record.uid);return original(record,**kwargs)
    app.service.collect_results_for_display=display_reader
    def preview_reader(*_):previews.append(row.uid);return preview
    monkeypatch.setattr('radar_pd_nova.result_preview.collect_result_preview',preview_reader)
    async def run():
        app._select_record(row)
        async def monitor():
            async with app._shared_result_read(row,row.model_copy(deep=True),app.service,
                                               context=app._result_context):pass
        monitor_task=asyncio.create_task(monitor())
        try:
            assert await asyncio.to_thread(entered.wait,1)
            app._open_record_results(row)
            await until(lambda:app.server.state.result_preview)
            assert readers==calls==previews==['A']
        finally:
            release.set()
            await asyncio.gather(monitor_task,*list(app._history_result_requests.values()))
    asyncio.run(run())


def test_new_selection_can_preview_while_another_canonical_read_holds_the_lock(tmp_path,monkeypatch):
    app,row,preview,_,entered,release,_=setup_preview(tmp_path)
    other=row.model_copy(deep=True,update={'uid':'B','galaxy_job_id':'B',
        'output_dataset_ids':{'summary':'summary-B'}})
    app.records['B']=other
    second=tmp_path/'preview-B';shutil.copytree(preview.root,second)
    for filename in ('.radar-pd-preview.json','plot_observation_binding.json'):
        document=json.loads((second/filename).read_text());document['job_id']='B'
        (second/filename).write_text(json.dumps(document))
    previews={'A':preview,'B':SimpleNamespace(root=second,output_dataset_ids=tuple(other.output_dataset_ids.items()))}
    monkeypatch.setattr('radar_pd_nova.result_preview.collect_result_preview',lambda service,r:previews[r.uid])
    async def run():
        app._select_record(row);app._open_record_results(row)
        try:
            await until(lambda:app.server.state.result_preview)
            assert entered.is_set()
            app._select_record(other);app._open_record_results(other)
            await until(lambda:app.server.state.result_owner_uid=='B')
            assert app.server.state.result_preview and not release.is_set()
            assert app.records['A'].output_dir is None and app.records['B'].output_dir is None
        finally:
            release.set();await asyncio.gather(*list(app._history_result_requests.values()))
        assert app.server.state.result_owner_uid=='B'
    asyncio.run(run())


def test_preview_mask_receipt_requires_its_own_job_identity(tmp_path):
    app,row,preview,path,_,_,_=setup_preview(tmp_path)
    assert read_plot_payload(path)['_observation_validity']['status']=='verified'
    receipt=json.loads((preview.root/'.radar-pd-preview.json').read_text())
    receipt['job_id']='other';(preview.root/'.radar-pd-preview.json').write_text(json.dumps(receipt))
    assert read_plot_payload(path)['_observation_validity']['status']=='unavailable'


def test_partial_renderer_failure_still_installs_the_complete_result(tmp_path,monkeypatch):
    app,row,preview,_,entered,release,_=setup_preview(tmp_path)
    rendered=threading.Event()
    monkeypatch.setattr('radar_pd_nova.result_preview.collect_result_preview',lambda *_:preview)
    def fail(*_):rendered.set();raise ValueError('partial renderer failed')
    app._display_result_preview=fail
    async def run():
        app._select_record(row);app._open_record_results(row)
        try:assert await asyncio.to_thread(rendered.wait,1)
        finally:
            release.set();await asyncio.gather(*list(app._history_result_requests.values()))
        assert not app.server.state.result_preview
        assert app.records[row.uid].result_status==ResultStatus.READY
        assert app.server.state.result_owner_uid==row.uid
    asyncio.run(run())


@pytest.mark.parametrize('preview_available',[False,True])
def test_stale_full_result_finishing_first_cannot_enable_native_actions(tmp_path,monkeypatch,preview_available):
    app,row,preview,_,entered,release,_=setup_preview(tmp_path,status=ResultStatus.STALE)
    preview_entered,preview_release=threading.Event(),threading.Event()
    def read(*_):
        preview_entered.set();assert preview_release.wait(5)
        return preview if preview_available else None
    monkeypatch.setattr('radar_pd_nova.result_preview.collect_result_preview',read)
    published=[];app._publish_results_if_requested=lambda *a,**k:published.append(a) or a[0]
    async def run():
        release.set();app._select_record(row);app._open_record_results(row)
        try:
            assert await asyncio.to_thread(preview_entered.wait,1)
            assert not app._result_actions_ready(row.uid)
        finally:
            preview_release.set();await asyncio.gather(*list(app._history_result_requests.values()))
        assert not app._result_actions_ready(row.uid)
        assert app.server.state.result_actions_pending and not app.server.state.checkpoint_rows
        assert not published
        assert app.server.state.result_preview is preview_available
    asyncio.run(run())


def test_preview_never_changes_preexisting_native_source_state(tmp_path,monkeypatch):
    app,row,preview,_,_,_,_=setup_preview(tmp_path)
    source={'record_uid':row.uid,'source_path':'independently-verified.gpx','source_sha256':'a'*64}
    workspace=SimpleNamespace(state={'sources':{'source':dict(source)},
        'jobs':{'approved':{'status':'queued','source':dict(source)}},'recipes':{'approved':{'approved':True}}})
    app._refinement_workspace=workspace
    import copy
    before=copy.deepcopy(workspace.state)
    app._select_record(row)
    assert app._display_result_preview(row,preview,app.service,app._result_source_identity(row),app._result_context)
    app.server.state.selected_run_uid='another-run';app._reset_result_state()
    assert workspace.state==before and row.output_dir is None
    assert not app._result_actions_ready(row.uid)


def test_replaced_hda_cannot_turn_an_old_preview_into_native_authority(tmp_path):
    app,row,preview,_,_,_,_=setup_preview(tmp_path)
    app._select_record(row)
    old_identity=app._result_source_identity(row)
    assert app._display_result_preview(row,preview,app.service,old_identity,app._result_context)
    row.output_dataset_ids={'summary':'replacement-summary'}
    assert app._result_preview_pending(row.uid) and not app._result_actions_ready(row.uid)
    stale=row.model_copy(deep=True,update={'result_status':ResultStatus.READY,
        'output_dataset_ids':dict(old_identity[2]),'output_dir':'stale-cache'})
    assert app._cache_result_record(row,stale,app.service,old_identity,app._result_context) is None
    assert app._result_preview_pending(row.uid)


def test_background_collection_leaves_two_shared_transfer_slots_for_preview(tmp_path,monkeypatch):
    service=GalaxyService('https://example.invalid','synthetic','history',output_root=tmp_path)
    lock=threading.Condition();release=threading.Event();active={'background':0,'preview':0};peaks=dict(active)
    def hold(kind):
        with lock:
            active[kind]+=1;peaks[kind]=max(peaks[kind],active[kind]);lock.notify_all()
        try:assert release.wait(3)
        finally:
            with lock:active[kind]-=1;lock.notify_all()
    monkeypatch.setattr(service,'_download_dataset',lambda *_:hold('background'))
    jobs={str(i):(str(i),tmp_path/str(i)) for i in range(4)}
    monkeypatch.setattr(service,'collect_results',lambda *a,**k:service._prefetch_results(jobs))
    def foreground():
        with service._result_download_slots:hold('preview')
    with ThreadPoolExecutor(max_workers=3) as pool:
        background=pool.submit(service.collect_results_for_display,None)
        try:
            with lock:assert lock.wait_for(lambda:active['background']==2,2)
            one,two=pool.submit(foreground),pool.submit(foreground)
            with lock:assert lock.wait_for(lambda:active['preview']==2,2)
            assert sum(active.values())==4
        finally:release.set()
        background.result();one.result();two.result()
    assert peaks=={'background':2,'preview':2}
    assert getattr(service._result_prefetch_context,'limit',4)==4


def test_incomplete_refresh_can_monitor_an_existing_export_but_cannot_submit_one(tmp_path):
    app,row,*_=setup_preview(tmp_path)
    app._publish_results_if_requested=RadarPdNovaApp._publish_results_if_requested.__get__(app)
    row.result_status=ResultStatus.STALE
    row.inputs=InputSelection(data_path='pattern',use_builtin_cuka=True,instrument='PG3',ipts='IPTS-1',
        publish_directory='shared',publish_results_to_ipts=True)
    row.output_dataset_ids['results_archive']='archive-A'
    calls=[]
    def refresh(action):
        calls.append(action.galaxy_job_id);action.status=RunStatus.OK;return action
    app.service=SimpleNamespace(refresh_utility=refresh,
        submit_results_export=lambda *a,**k:pytest.fail('Incomplete results cannot start an export'))
    assert app._publish_results_if_requested(row).publication_job_id is None
    assert not calls
    row.publication_job_id='existing-export';row.publication_status=RunStatus.RUNNING
    row.publication_target='shared/existing.zip'
    checked=app._publish_results_if_requested(row)
    assert checked.publication_status==RunStatus.OK and calls==['existing-export']
    assert checked.published_output_dir=='shared/existing.zip'


def test_rapid_missing_data_messages_are_pending_until_full_results_arrive():
    from ast import literal_eval
    from html.parser import HTMLParser
    class Controls(HTMLParser):
        def __init__(self, html):
            super().__init__();self.messages=[];self.feed(html)
        def handle_starttag(self, tag, attributes):
            attrs=dict(attributes)
            self.messages.extend(value for key,value in attrs.items()
                if key.replace('-','').replace('_','').lower()==':nodatatext')
    app=RadarPdNovaApp()
    expected={
        'The coarse-search table is not loaded yet.':'No coarse-search table was published.',
        'The lattice-nudge table is not loaded yet.':'No lattice-nudge table was published.',
        'The pattern-scoring table is not loaded yet.':'No pattern-scoring table was published.',
        'The final-refinement ranking is not loaded yet.':'No final-refinement ranking was published.',
        'Refined hypotheses are not loaded yet.':'No refined hypotheses are available',
        'Alternate hypotheses are not loaded yet.':'No alternate hypothesis is available',
        'Saved GPX checkpoints are not loaded yet.':'No published GPX checkpoint is available',
    }
    observed={}
    for expression in Controls(app.layout.html).messages:
        condition, choices=expression.split('?',1)
        assert condition.strip()=='result_actions_pending'
        pending, ready=choices.split(':',1)
        observed[literal_eval(pending.strip())]=literal_eval(ready.strip())
    assert observed==expected
