"""A slow or failed result download must not freeze or relaunch an experiment."""
import asyncio
import threading
from types import SimpleNamespace

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.models import RunRecord, RunStatus, AnalysisMode


def app_with_completed_record():
    app=RadarPdNovaApp()
    record=RunRecord(uid='published-job',galaxy_job_id='published-job',history_id='test',
        name='IPTS-2767_PG3_50713',mode=AnalysisMode.FULL,status=RunStatus.OK,
        output_dataset_ids={'summary':'summary'})
    app._powgen_controller=SimpleNamespace(records={'PG3_50713':record})
    app.server.state.powgen_selected_run_id='PG3_50713'
    app.records={record.uid:record}
    app._select_record=lambda r:setattr(app.server.state,'selected_run_uid',r.uid)
    app._sync_runs=lambda:None
    app._publish_results_if_requested=lambda r:r
    return app,record


def test_slow_result_collection_yields_to_ui_and_reentrant_selection_does_not_download_twice(tmp_path):
    app,record=app_with_completed_record()
    entered,release=threading.Event(),threading.Event()
    downloads=[];loaded=[]
    def collect(r):
        downloads.append(r.uid);entered.set()
        if not release.wait(3):raise TimeoutError('Test download was not released')
        r.output_dir=str(tmp_path);return r
    app.service=SimpleNamespace(collect_results=collect)
    app._load_results=lambda r:loaded.append(r.uid)
    async def run():
        task=asyncio.create_task(app.open_powgen_selected_run_async())
        try:
            assert await asyncio.to_thread(entered.wait,1)
            assert not task.done() and app.server.state.selected_run_loading
            assert app.server.state.workspace_view=='experiment'
            assert 'downloading fit plots' in app.server.state.selected_result_load_message
            app._run_selection_changed(['published-job'])
            app.server.state.run_search='UI remains responsive'
            await app.open_powgen_selected_run_async()
            assert downloads==['published-job']
        finally:
            release.set();await task
        assert loaded==['published-job'] and not app.server.state.selected_run_loading
        assert app.server.state.workspace_view=='results' and not app.server.state.selected_result_load_error
    asyncio.run(run())


def test_result_button_returns_to_client_before_archive_download_finishes(tmp_path):
    app,record=app_with_completed_record()
    entered,release=threading.Event(),threading.Event()
    def collect(r):
        entered.set();assert release.wait(3)
        r.output_dir=str(tmp_path);return r
    app.service=SimpleNamespace(collect_results=collect)
    app._load_results=lambda r:None
    async def run():
        assert app.request_open_powgen_selected_run() is None
        try:
            assert await asyncio.to_thread(entered.wait,1)
            assert app.server.state.selected_run_loading
            assert app.server.state.workspace_view=='experiment'
        finally:
            tasks=list(app._result_open_tasks);release.set();await asyncio.gather(*tasks)
        await asyncio.sleep(0)
        assert not app._result_open_tasks
        assert app.server.state.workspace_view=='results'
    asyncio.run(run())


def test_failed_result_fetch_returns_to_existing_experiment_without_changing_successful_analysis():
    app,record=app_with_completed_record()
    app.service=SimpleNamespace(collect_results=lambda _:(_ for _ in ()).throw(OSError('object store unavailable')))
    app._load_results=lambda _:(_ for _ in ()).throw(AssertionError('Failed result was displayed'))
    asyncio.run(app.open_powgen_selected_run_async())
    assert not app.server.state.selected_run_loading
    assert app.server.state.workspace_view=='experiment'
    assert 'object store unavailable' in app.server.state.selected_result_load_error
    assert record.status is RunStatus.OK
    assert app._powgen_controller.records['PG3_50713'] is record
