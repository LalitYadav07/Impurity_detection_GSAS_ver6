"""A completed result remains reachable after the user visits another workspace."""
import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.galaxy_service import GalaxyService
from radar_pd_nova.models import AnalysisMode, ResultStatus, RunRecord, RunStatus


@pytest.mark.parametrize('view', ['results', 'plots', 'files'])
def test_return_from_refinement_loads_selected_completed_cache_without_download(tmp_path, view):
    app = RadarPdNovaApp()
    app._sync_runs = lambda: None
    original = RunRecord(uid='selected', galaxy_job_id='job-selected', history_id='history',
                         name='Selected run', mode=AnalysisMode.FULL,
                         status=RunStatus.RUNNING, analysis_status=RunStatus.RUNNING)
    app.records = {original.uid: original}
    app._select_record(original)
    app.server.state.workspace_view = 'refinement'
    loaded = []
    def load(record, *, local_only=False):
        assert local_only
        loaded.append(record.uid)
        app.server.state.result_owner_uid = record.uid
    app._load_results = load
    app.service = SimpleNamespace(collect_results=lambda *_a, **_k: pytest.fail('Use the verified local cache'))

    completed = original.model_copy(deep=True, update={
        'status': RunStatus.OK, 'analysis_status': RunStatus.OK,
        'result_status': ResultStatus.READY, 'output_dir': str(tmp_path),
        'stage': 'Results ready', 'progress': 100,
    })
    app._monitor_update(completed)
    assert app.server.state.workspace_view == 'refinement'
    assert loaded == [], 'Background completion must not take over another workspace'

    app.select_workspace_view(view)

    assert loaded == ['selected']
    assert app.server.state.workspace_view == view
    assert app.server.state.selected_result_status == 'Ready'
    assert app.server.state.selected_run_stage == 'Results ready'
    assert app.server.state.result_owner_uid == 'selected'


def test_return_does_not_load_another_completed_run(tmp_path):
    app = RadarPdNovaApp()
    app._sync_runs = lambda: None
    selected = RunRecord(uid='selected', galaxy_job_id='job-selected', history_id='history',
                         name='Selected run', mode=AnalysisMode.FULL, status=RunStatus.RUNNING)
    other = RunRecord(uid='other', galaxy_job_id='job-other', history_id='history',
                      name='Other run', mode=AnalysisMode.FULL, status=RunStatus.OK,
                      result_status=ResultStatus.READY, output_dir=str(tmp_path))
    app.records = {selected.uid: selected, other.uid: other}
    app._select_record(selected)
    app.server.state.workspace_view = 'refinement'
    app._load_results = lambda *_a: pytest.fail('Unselected cached results must remain hidden')
    app.select_workspace_view('results')
    assert app.server.state.selected_run_uid == 'selected'
    assert app.server.state.result_owner_uid != 'other'


def test_return_while_selected_result_collects_does_not_begin_another_download():
    app = RadarPdNovaApp()
    selected = RunRecord(uid='selected', galaxy_job_id='job-selected', history_id='history',
                         name='Selected run', mode=AnalysisMode.FULL, status=RunStatus.OK,
                         result_status=ResultStatus.COLLECTING)
    app.records = {selected.uid: selected}
    app._select_record(selected)
    app.server.state.workspace_view = 'refinement'
    app._load_results = lambda *_a: pytest.fail('The selected result is not yet cached')
    app.service = SimpleNamespace(collect_results=lambda *_a, **_k: pytest.fail('No duplicate result transaction'))
    app.select_workspace_view('results')
    assert app.server.state.workspace_view == 'results'
    assert app.server.state.selected_result_status != 'Ready'


@pytest.mark.parametrize('reader', ['automatic', 'manual'])
@pytest.mark.parametrize('view', ['results', 'plots', 'files'])
def test_return_before_collection_finishes_publishes_only_the_selected_cached_result(tmp_path, reader, view):
    app = RadarPdNovaApp()
    app._sync_runs = lambda: None
    row = RunRecord(uid='selected', galaxy_job_id='job-selected', history_id='history',
                    name='Selected run', mode=AnalysisMode.FULL, status=RunStatus.OK)
    app.records = {row.uid: row}
    app._select_record(row)
    entered, release = threading.Event(), threading.Event()
    calls, loaded = [], []
    def collect(private):
        calls.append(private.uid)
        entered.set()
        assert release.wait(5)
        private.output_dir = str(tmp_path)
        private.result_status = ResultStatus.READY
        private.stage = 'Results ready'
        return private
    app.service = SimpleNamespace(collect_results=collect)
    app._load_results = lambda current, **_: loaded.append(current.uid)

    async def scenario():
        if reader == 'automatic':
            task = asyncio.create_task(app._monitor_record(row))
        else:
            app._open_record_results(row)
            task = app._history_result_requests[row.uid]
        assert await asyncio.to_thread(entered.wait, 2)
        try:
            app.select_workspace_view('refinement')
            app.select_workspace_view(view)
            assert loaded == []
            assert calls == [row.uid]
        finally:
            release.set()
            await task
        assert loaded == [row.uid]
        assert calls == [row.uid]
        assert app.server.state.workspace_view == view
        assert app.server.state.selected_result_status == 'Ready'

    asyncio.run(scenario())


def test_other_run_selected_during_collection_never_receives_finished_result(tmp_path):
    app = RadarPdNovaApp()
    app._sync_runs = lambda: None
    first = RunRecord(uid='first', galaxy_job_id='job-first', history_id='history',
                      name='First run', mode=AnalysisMode.FULL, status=RunStatus.OK)
    other = RunRecord(uid='other', galaxy_job_id='job-other', history_id='history',
                      name='Other run', mode=AnalysisMode.FULL, status=RunStatus.RUNNING)
    app.records = {r.uid:r for r in (first,other)}
    app._select_record(first)
    entered, release = threading.Event(), threading.Event()
    loaded, calls = [], []
    def collect(private):
        calls.append(private.uid)
        entered.set()
        assert release.wait(5)
        private.output_dir = str(tmp_path)
        private.result_status = ResultStatus.READY
        return private
    app.service = SimpleNamespace(collect_results=collect)
    app._load_results = lambda current, **_: loaded.append(current.uid)

    async def scenario():
        task = asyncio.create_task(app._monitor_record(first))
        assert await asyncio.to_thread(entered.wait,2)
        try:
            app._run_selection_changed([other.uid])
            app.select_workspace_view('results')
        finally:
            release.set()
            await task
        assert app.server.state.selected_run_uid == other.uid
        assert loaded == []
        assert calls == [first.uid]
        assert app.records[first.uid].result_status == ResultStatus.READY
        assert app.server.state.result_owner_uid != first.uid

    asyncio.run(scenario())


def test_cached_navigation_does_not_fetch_missing_observation_or_refresh_catalog(tmp_path, monkeypatch):
    root = tmp_path/'result'
    root.mkdir()
    summary = {'analysis_mode':'full', 'status':'complete', 'best_rwp':1.25, 'phases':[],
               'pattern_observation':{'name':'Pattern_observation.json','sha256':'a'*64,
                                      'source_gpx_sha256':'b'*64},
               'provenance':{'input':{'source_sha256':'c'*64}}}
    (root/'summary.json').write_text(json.dumps(summary))
    app = RadarPdNovaApp()
    app.service = GalaxyService('https://example.invalid','unused','history',output_root=tmp_path/'local')
    app.service.cache_pattern_observation = lambda *_a,**_k: pytest.fail('Navigation must not fetch optional masks')
    monkeypatch.setattr('radar_pd_nova.galaxy_service.requests.get',
                        lambda *_a,**_k: pytest.fail('Cached navigation must not make a GET'))
    app._refinement_maybe_refresh = lambda: pytest.fail('Navigation must not refresh the catalog')
    row = RunRecord(uid='selected',galaxy_job_id='job-selected',history_id='history',
                    name='Selected run',mode=AnalysisMode.FULL,status=RunStatus.OK,
                    result_status=ResultStatus.READY,output_dir=str(root),
                    output_dataset_ids={'pattern_observation':'packet-id'})
    app.records = {row.uid:row}
    app._select_record(row)
    app.server.state.workspace_view = 'refinement'

    app.select_workspace_view('results')

    assert app.server.state.result_owner_uid == row.uid
    assert app.server.state.workspace_view == 'results'
    assert app.server.state.selected_result_status == 'Ready'
    binding = json.loads((root/'plot_observation_binding.json').read_text())
    assert binding['status'] == 'unavailable'
    assert not binding.get('download_attempted')


def test_same_uid_new_publication_does_not_reuse_old_result_owner(tmp_path):
    app = RadarPdNovaApp()
    row = RunRecord(uid='same',galaxy_job_id='old-job',history_id='history',name='Old source',
                    mode=AnalysisMode.FULL,status=RunStatus.OK,result_status=ResultStatus.READY,
                    output_dir=str(tmp_path/'old'),output_dataset_ids={'summary':'old-summary'})
    app.records = {row.uid:row}
    app._select_record(row)
    app.server.state.result_owner_uid = row.uid
    app._result_display_binding = (app.service,app._result_context,*app._result_display_state(row))
    replacement = row.model_copy(deep=True,update={'galaxy_job_id':'new-job','name':'New source',
        'output_dir':str(tmp_path/'new'),'output_dataset_ids':{'summary':'new-summary'}})
    app.records[row.uid] = replacement
    loaded = []
    app._load_results = lambda current, **kwargs: loaded.append((current.galaxy_job_id,current.output_dir,kwargs))

    app.select_workspace_view('plots')

    assert loaded == [('new-job',str(tmp_path/'new'),{'local_only':True})]
    assert app.server.state.selected_run_name == 'New source'


@pytest.mark.parametrize('status', [RunStatus.RUNNING,RunStatus.ERROR,RunStatus.CANCELLED])
def test_unfinished_or_failed_source_cannot_reuse_a_ready_cache(tmp_path,status):
    app = RadarPdNovaApp()
    row = RunRecord(uid='selected',galaxy_job_id='job',history_id='history',name='Selected run',
                    mode=AnalysisMode.FULL,status=status,analysis_status=status,
                    result_status=ResultStatus.READY,output_dir=str(tmp_path))
    app.records = {row.uid:row}
    app._select_record(row)
    app._load_results = lambda *_a,**_k: pytest.fail('Not an accepted completed source')
    app.select_workspace_view('results')


def test_warm_navigation_preserves_controls_and_legacy_one_argument_provider(tmp_path,monkeypatch):
    (tmp_path/'summary.json').write_text(json.dumps({'analysis_mode':'full','status':'complete','phases':[]}))
    app = RadarPdNovaApp()
    # A legacy provider need not accept a new keyword or make a remote request.
    app.service = SimpleNamespace(result_payload=lambda _: pytest.fail('Use the local-file fallback'))
    monkeypatch.setattr('radar_pd_nova.galaxy_service.requests.get',
                        lambda *_a,**_k: pytest.fail('Cached navigation must not make a GET'))
    app._refinement_maybe_refresh = lambda: pytest.fail('Cached navigation must not refresh the catalog')
    row = RunRecord(uid='selected',galaxy_job_id='job',history_id='history',name='Selected run',
                    mode=AnalysisMode.FULL,status=RunStatus.OK,result_status=ResultStatus.READY,
                    output_dir=str(tmp_path),output_dataset_ids={'summary':'summary-id'})
    app.records = {row.uid:row}
    app._select_record(row)
    app.select_workspace_view('results')
    assert app.server.state.result_owner_uid == row.uid
    assert app._rendered_result_current(row)
    app.server.state.file_search = 'keep this filter'
    app.server.state.show_technical_files = True
    app._load_results = lambda *_a,**_k: pytest.fail('Warm navigation must not rebuild the result controls')
    for view in ('plots','files','results'):
        app.select_workspace_view(view)
        assert app.server.state.workspace_view == view
        assert app.server.state.file_search == 'keep this filter'
        assert app.server.state.show_technical_files is True


@pytest.mark.parametrize('changed', ['service','context','source','directory','manifest'])
def test_rendered_binding_rejects_changed_context_or_publication(tmp_path,changed):
    from radar_pd_nova.models import CacheManifest
    app = RadarPdNovaApp()
    row = RunRecord(uid='selected',galaxy_job_id='job',history_id='history',name='Selected run',
                    mode=AnalysisMode.FULL,status=RunStatus.OK,result_status=ResultStatus.READY,
                    output_dir=str(tmp_path),output_dataset_ids={'results_archive':'archive-id'},
                    cache_manifest=CacheManifest(job_id='job',archive_dataset_id='archive-id'))
    app.records = {row.uid:row}
    app._select_record(row)
    app.server.state.result_owner_uid = row.uid
    app._result_display_binding = (app.service,app._result_context,*app._result_display_state(row))
    assert app._rendered_result_current(row)
    if changed == 'service':app.service=SimpleNamespace()
    elif changed == 'context':app._result_context=object()
    elif changed == 'source':row.output_dataset_ids['results_archive']='replacement'
    elif changed == 'directory':row.output_dir=str(tmp_path/'replacement')
    else:row.cache_manifest.archive_size=1
    assert not app._rendered_result_current(row)
    loaded=[]
    app._load_results=lambda current,**kwargs:loaded.append((current.uid,kwargs))
    app.select_workspace_view('results')
    assert loaded == [('selected',{'local_only':True})]
