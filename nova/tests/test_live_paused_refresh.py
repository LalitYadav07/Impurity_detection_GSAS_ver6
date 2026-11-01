"""Stopped Live result refresh never enrolls or retries science."""
import asyncio
from copy import deepcopy
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.models import RunRecord


def harness(*, blocked=False):
    trace=[]; entered=threading.Event(); release=threading.Event()
    if not blocked:release.set()
    state=SimpleNamespace(powgen_monitoring=False,live_launch_busy=False,
        powgen_results_refresh_busy=False,powgen_results_refresh_available=True,
        powgen_next_check='Stopped',powgen_message='stopped',flush=Mock(),
        workspace_clearing=False,session_closing=False,powgen_retry_requested=False)
    class Controller:
        refresh_errors={}
        def __init__(self):
            self.state=SimpleNamespace(submitted={'scan':object()},completed={},failed={'failed':'retained'},discovered={})
            self.records={}
        def refresh(self,*,checkpoint):
            assert checkpoint is False;trace.append('refresh');entered.set();assert release.wait(3)
            self.state.completed['scan']=self.state.submitted.pop('scan')
            return {'scan':RunRecord(uid='job',name='existing scan',galaxy_job_id='job',history_id='h',mode='full',status='ok')}
        def capture_checkpoint(self):
            trace.append('snapshot');return {'completed':list(self.state.completed)}
        def persist_state(self,snapshot):
            trace.append(('persist',deepcopy(snapshot)));return 'checkpoint'
        def discover(self,*a,**k):pytest.fail('Stopped refresh must not discover')
        def due_submissions(self,*a,**k):pytest.fail('Stopped refresh must not admit')
        def launch_submission(self,*a,**k):pytest.fail('Stopped refresh must not submit')
        def retry_failed_runs(self,*a,**k):pytest.fail('Stopped refresh must not retry')
    app=RadarPdNovaApp.__new__(RadarPdNovaApp)
    app.server=SimpleNamespace(state=state);app.service=object();app._result_context=object()
    app._powgen_controller=Controller();app.records={};app._powgen_stopped_refresh_task=None
    app._powgen_wake_event=asyncio.Event();app._sync_runs=Mock()
    app._sync_powgen_rows=Mock();app._powgen_monitor_task=None
    return app,trace,entered,release


def test_stopped_refresh_updates_existing_results_and_checkpoint_only():
    app,trace,_,_=harness()
    async def run():
        app.refresh_powgen_monitor_now()
        await app._powgen_stopped_refresh_task
    asyncio.run(run())
    assert trace==['refresh','snapshot',('persist',{'completed':['scan']})]
    assert app.records['job'].status.value=='ok'
    assert not app.server.state.powgen_monitoring and not app.server.state.live_launch_busy
    assert not app.server.state.powgen_results_refresh_busy
    assert not app.server.state.powgen_retry_requested
    assert 'stopped' in app.server.state.powgen_message.lower()
    app._sync_powgen_rows.assert_called_once_with(refresh_dependents=False)


def test_duplicate_refresh_coalesces_and_start_cannot_overlap():
    app,trace,entered,release=harness(blocked=True)
    async def run():
        app.refresh_powgen_monitor_now();task=app._powgen_stopped_refresh_task
        app.refresh_powgen_monitor_now()
        assert app._powgen_stopped_refresh_task is task
        assert await asyncio.to_thread(entered.wait,2)
        assert app.server.state.live_launch_busy and app.server.state.powgen_results_refresh_busy
        app._prepare_live_settings=Mock(side_effect=AssertionError('Start must remain blocked'))
        await app.start_powgen_monitoring()
        app._prepare_live_settings.assert_not_called()
        release.set();await task
    asyncio.run(run())
    assert trace.count('refresh')==1


@pytest.mark.parametrize('change',['controller','service','context','clear'])
def test_stale_refresh_never_applies_to_replaced_workspace(change):
    app,trace,entered,release=harness(blocked=True)
    async def run():
        app.refresh_powgen_monitor_now();task=app._powgen_stopped_refresh_task
        assert await asyncio.to_thread(entered.wait,2)
        if change=='controller':app._powgen_controller=object()
        elif change=='service':app.service=object()
        else:app._result_context=object()
        app.server.state.powgen_message='new workspace'
        app.server.state.live_launch_busy=False
        app.server.state.powgen_results_refresh_busy=False
        release.set();await task
    asyncio.run(run())
    assert app.records=={} and app.server.state.powgen_message=='new workspace'
    assert not any(isinstance(v,tuple) and v[0]=='persist' for v in trace)
    app._sync_runs.assert_not_called();app._sync_powgen_rows.assert_not_called()


def test_active_refresh_still_only_wakes_the_existing_monitor():
    app,trace,_,_=harness();app.server.state.powgen_monitoring=True
    app._live_adapter=lambda:SimpleNamespace(name='POWGEN')
    app.refresh_powgen_monitor_now()
    assert app._powgen_wake_event.is_set() and trace==[]
    assert app._powgen_stopped_refresh_task is None


def test_click_binds_controller_before_async_task_starts():
    app,trace,_,_=harness()
    async def run():
        app.refresh_powgen_monitor_now();task=app._powgen_stopped_refresh_task
        app._powgen_controller=object()
        app.server.state.powgen_message='new controller'
        app.server.state.live_launch_busy=False
        app.server.state.powgen_results_refresh_busy=False
        await task
    asyncio.run(run())
    assert trace==[] and app.records=={}
    assert app.server.state.powgen_message=='new controller'


def test_click_captures_owner_before_flushing_pending_ui_watchers():
    app,trace,_,_=harness()
    def changed():
        app._powgen_controller=object()
        app.server.state.powgen_message='replacement controller'
        app.server.state.live_launch_busy=False
        app.server.state.powgen_results_refresh_busy=False
    app.server.state.flush=Mock(side_effect=changed)
    async def run():
        app.refresh_powgen_monitor_now();await app._powgen_stopped_refresh_task
    asyncio.run(run())
    assert trace==[] and app.records=={}
    assert app.server.state.powgen_message=='replacement controller'


def test_no_controller_or_other_busy_work_does_not_launch_a_refresh():
    for target in ('no-controller','busy'):
        app,trace,_,_=harness()
        if target=='no-controller':app._powgen_controller=None
        else:app.server.state.live_launch_busy=True
        app.refresh_powgen_monitor_now()
        assert trace==[] and app._powgen_stopped_refresh_task is None


def test_stopped_refresh_error_is_visible_without_starting_work():
    app,trace,_,_=harness()
    def failed(**kwargs):raise OSError('temporary result metadata unavailable')
    app._powgen_controller.refresh=failed
    async def run():
        app.refresh_powgen_monitor_now();await app._powgen_stopped_refresh_task
    asyncio.run(run())
    assert 'unavailable' in app.server.state.powgen_message
    assert not app.server.state.powgen_monitoring
    assert not app.server.state.live_launch_busy and not app.server.state.powgen_results_refresh_busy


def test_refresh_control_explains_stopped_read_only_action():
    app=RadarPdNovaApp()
    assert 'Refresh results' in app.layout.html
    assert 'powgen_results_refresh_available' in app.layout.html
    assert 'without discovering or submitting' in app.layout.html
    assert 'Start monitoring to enable Retry failed runs' in app.layout.html


def test_checkpoint_failure_is_visible_and_does_not_resume_work():
    app,trace,_,_=harness()
    def failed(snapshot):raise OSError('checkpoint unavailable')
    app._powgen_controller.persist_state=failed
    async def run():
        app.refresh_powgen_monitor_now();await app._powgen_stopped_refresh_task
    asyncio.run(run())
    assert 'checkpoint unavailable' in app.server.state.powgen_message
    assert not app.server.state.powgen_monitoring
    assert not app.server.state.live_launch_busy and not app.server.state.powgen_results_refresh_busy
    assert isinstance(app._powgen_checkpoint_error,OSError)


def test_result_projection_can_skip_downstream_refinement_enrollment():
    app=RadarPdNovaApp()
    app._powgen_controller=SimpleNamespace(
        state=SimpleNamespace(failed={},submitted={},completed={},discovered={}),
        settings=SimpleNamespace(max_active_jobs=1),records={})
    app._sync_powgen_dashboard_provenance=Mock()
    app._refinement_maybe_refresh=Mock()
    app._update_powgen_dashboard_figures=Mock()
    app._sync_powgen_rows(refresh_dependents=False)
    app._refinement_maybe_refresh.assert_not_called()
    app._sync_powgen_rows()
    app._refinement_maybe_refresh.assert_called_once()
