"""A completed result remains reachable after the user visits another workspace."""
import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.galaxy_service import GalaxyService
from radar_pd_nova.models import AnalysisMode, ResultStatus, RunRecord, RunStatus


def test_result_review_badge_is_bound_to_the_displayed_run_and_cleared_on_selection():
    app = RadarPdNovaApp()
    original = RunRecord(uid='selected', galaxy_job_id='job-selected', history_id='history',
                         name='Selected run', mode=AnalysisMode.FULL,
                         status=RunStatus.OK, result_status=ResultStatus.READY)
    app._select_record(original)
    state = app.server.state
    state.result_owner_uid = original.uid
    state.result_review_status = 'needs_review'
    app._sync_selected_run_status(original)
    assert state.selected_run_status == 'Ok', 'Job state remains available to lifecycle controls'
    assert state.selected_run_badge == 'Needs scientific review'

    other = original.model_copy(update={'uid': 'other', 'name': 'Other run'})
    app._select_record(other)
    assert state.result_owner_uid == ''
    assert state.selected_run_badge == 'Complete', 'Do not borrow the previous fit qualification'
    state.result_owner_uid = other.uid
    state.result_review_status = 'unknown'
    app._sync_selected_run_status(other)
    assert state.selected_run_badge == 'Convergence unknown'
    app._reset_result_state()
    assert state.selected_run_badge == 'Complete'


@pytest.mark.parametrize('workspace', ['refinement', 'battery', 'experiment', 'preparation'])
def test_selected_status_updates_without_taking_over_another_workspace(workspace):
    app = RadarPdNovaApp()
    app._sync_runs = lambda: None
    original = RunRecord(uid='selected', galaxy_job_id='job-selected', history_id='history',
                         name='Selected run', mode=AnalysisMode.FULL, status=RunStatus.QUEUED)
    app.records = {original.uid: original}
    app._select_record(original)
    app.server.state.workspace_view = workspace
    app._load_results = lambda *_a, **_k: pytest.fail('Do not render another workspace')
    completed = original.model_copy(deep=True, update={
        'status': RunStatus.OK, 'analysis_status': RunStatus.OK,
        'result_status': ResultStatus.COLLECTING, 'stage': 'Collecting results', 'progress': 100,
    })

    app._monitor_update(completed)

    assert app.server.state.workspace_view == workspace
    assert app.server.state.selected_run_status == 'Ok'
    assert app.server.state.selected_analysis_status == 'Ok'
    assert app.server.state.selected_run_progress == 100
    assert app.server.state.selected_run_stage == 'Analysis complete; loading results'
    assert app.server.state.result_owner_uid != completed.uid


@pytest.mark.parametrize('workspace',['battery','preparation'])
@pytest.mark.parametrize('mode',[AnalysisMode.FULL,AnalysisMode.RAPID])
def test_analysis_tab_refresh_preserves_global_preparation_workspaces(workspace,mode):
    app=SimpleNamespace(server=SimpleNamespace(state=SimpleNamespace(workspace_view=workspace)),
        _powgen_controller=None)
    RadarPdNovaApp._sync_workspace_options(app,mode)
    assert app.server.state.workspace_view==workspace
    assert workspace not in {r['value'] for r in app.server.state.workspace_options}


def test_return_to_monitor_uses_current_record_without_download():
    app = RadarPdNovaApp()
    original = RunRecord(uid='selected', galaxy_job_id='job-selected', history_id='history',
                         name='Selected run', mode=AnalysisMode.FULL, status=RunStatus.QUEUED)
    app.records = {original.uid: original}
    app._select_record(original)
    app.server.state.workspace_view = 'refinement'
    # Model a completion already in the cache while the rendered header still
    # contains the queued state, as observed in the actual two-scan NDIP test.
    app.records[original.uid] = original.model_copy(deep=True, update={
        'status': RunStatus.OK, 'analysis_status': RunStatus.OK,
        'result_status': ResultStatus.READY, 'stage': 'Results ready', 'progress': 100,
    })
    app._load_results = lambda *_a, **_k: pytest.fail('Monitor navigation needs no result download')

    app.select_workspace_view('monitor')

    assert app.server.state.workspace_view == 'monitor'
    assert app.server.state.selected_run_status == 'Ok'
    assert app.server.state.selected_result_status == 'Ready'
    assert app.server.state.selected_run_stage == 'Results ready'


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
               'observables':{'source_gpx_sha256':'b'*64},
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


@pytest.mark.parametrize('changed', [None,'dataset','collection','checksum','project','source','job'])
def test_local_reader_preserves_failed_fetch_memory_only_for_same_identity(tmp_path,changed):
    summary = {'analysis_mode':'full','phases':[],
        'pattern_observation':{'sha256':'a'*64,'source_gpx_sha256':'b'*64},
        'observables':{'source_gpx_sha256':'b'*64},
        'provenance':{'input':{'source_sha256':'c'*64}}}
    path=tmp_path/'summary.json';path.write_text(json.dumps(summary))
    service=GalaxyService('https://example.invalid','unused','history',output_root=tmp_path/'local')
    row=RunRecord(uid='selected',galaxy_job_id='job',history_id='history',name='Selected run',
        mode=AnalysisMode.FULL,status=RunStatus.OK,result_status=ResultStatus.READY,
        output_dir=str(tmp_path),output_dataset_ids={'pattern_observation':'packet','tables':'tables'})
    calls=[]
    def fail_fetch(*args,**kwargs):
        calls.append((args,kwargs))
        raise OSError('offline')
    service.cache_pattern_observation=fail_fetch
    service.result_payload(row)
    before=json.loads((tmp_path/'plot_observation_binding.json').read_text())
    assert len(calls)==1 and before['download_attempted'] is True
    if changed=='dataset':row.output_dataset_ids['pattern_observation']='new-packet'
    elif changed=='collection':row.output_dataset_ids['tables']='new-tables'
    elif changed=='checksum':summary['pattern_observation']['sha256']='d'*64
    elif changed=='project':
        summary['pattern_observation']['source_gpx_sha256']='d'*64
        summary['observables']['source_gpx_sha256']='d'*64
    elif changed=='source':summary['provenance']['input']['source_sha256']='d'*64
    elif changed=='job':row.galaxy_job_id='new-job'
    path.write_text(json.dumps(summary))
    service.cached_result_payload(row)
    after=json.loads((tmp_path/'plot_observation_binding.json').read_text())
    assert len(calls)==1, 'The local reader must never fetch a packet'
    assert bool(after.get('download_attempted')) is (changed is None)
    if changed is None:assert after['fetch_identity']==before['fetch_identity']
    service.result_payload(row)
    assert len(calls)==(1 if changed is None else 2)


@pytest.mark.parametrize('submitted,reported', [(AnalysisMode.FULL,'rapid'),(AnalysisMode.RAPID,'full')])
def test_warm_navigation_preserves_result_summary_mode_and_panels(tmp_path,submitted,reported):
    (tmp_path/'summary.json').write_text(json.dumps({'analysis_mode':reported,'status':'complete','phases':[]}))
    app=RadarPdNovaApp()
    app.service=SimpleNamespace(result_payload=lambda _:pytest.fail('Use local files'))
    row=RunRecord(uid='selected',galaxy_job_id='job',history_id='history',name='Selected run',
        mode=submitted,status=RunStatus.OK,result_status=ResultStatus.READY,output_dir=str(tmp_path))
    app.records={row.uid:row}
    app._select_record(row)
    app.select_workspace_view('results')
    assert app.server.state.viewed_run_mode==reported
    expected_options=list(app.server.state.workspace_options)
    for view in ('files','results'):
        app.select_workspace_view(view)
        assert app.server.state.viewed_run_mode==reported
        assert app.server.state.workspace_options==expected_options
        assert app.server.state.workspace_view==view
    assert row.mode==submitted


@pytest.mark.parametrize('new_mode,expected_panel', [('full','plots'),('rapid','monitor')])
def test_changed_cache_honors_requested_panel_only_if_result_mode_supports_it(tmp_path,new_mode,expected_panel):
    from radar_pd_nova.models import CacheManifest
    path=tmp_path/'summary.json'
    path.write_text(json.dumps({'analysis_mode':'full','status':'complete','phases':[]}))
    app=RadarPdNovaApp()
    app.service=SimpleNamespace(result_payload=lambda _:pytest.fail('Use local files'))
    row=RunRecord(uid='selected',galaxy_job_id='job',history_id='history',name='Selected run',
        mode=AnalysisMode.RAPID if new_mode=='full' else AnalysisMode.FULL,
        status=RunStatus.OK,result_status=ResultStatus.READY,output_dir=str(tmp_path),
        output_dataset_ids={'results_archive':'archive'},
        cache_manifest=CacheManifest(job_id='job',archive_dataset_id='archive'))
    app.records={row.uid:row};app._select_record(row);app.select_workspace_view('results')
    assert app.server.state.viewed_run_mode=='full'
    app.select_workspace_view('refinement')
    path.write_text(json.dumps({'analysis_mode':new_mode,'status':'complete','phases':[]}))
    row.cache_manifest.archive_size=1

    app.select_workspace_view('plots')

    assert app.server.state.viewed_run_mode==new_mode
    assert app.server.state.workspace_view==expected_panel
    assert ('plots' in {item['value'] for item in app.server.state.workspace_options}) is (new_mode=='full')
