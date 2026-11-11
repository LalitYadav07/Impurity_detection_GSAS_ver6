"""Session cleanup must not erase Galaxy work or silently resume old IPTS data."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from radar_pd_nova.app import RadarPdNovaApp


class _Workspace:
    def __init__(self):
        self.state = {"recipes": {"old": {"experiment": "PG3:2767", "armed": True}}, "jobs": {}}
        self.paused = []
        self.checkpoints = 0

    def pause(self, experiment):
        self.paused.append(experiment)
        self.state["recipes"]["old"]["armed"] = False
        for job in self.state["jobs"].values():
            if job["source"]["experiment"] == experiment and job["status"] == "queued":
                job["status"] = "paused"

    def checkpoint(self):
        self.checkpoints += 1



def _navigation(app, events=None, label="leave"):
    def dispatched(nonce):
        if events is not None:events.append(label)
        assert app._acknowledge_session_navigation(nonce)
    app._leave_session_page=dispatched


def test_cancelled_utility_wait_cannot_submit_later():
    import threading
    import pytest
    from radar_pd_nova.native_job_admission import AdmissionUnavailable
    entered=threading.Event();stopped=threading.Event();posted=[]
    def submit(*,cancel_event,**kwargs):
        entered.set()
        if cancel_event.wait(2):
            stopped.set()
            raise AdmissionUnavailable('Cancelled before POST')
        posted.append(kwargs)
    app=SimpleNamespace(service=SimpleNamespace(submit_utility=submit))
    async def scenario():
        task=asyncio.create_task(RadarPdNovaApp._submit_utility_action(
            app,tool_id='neutrons_radar_pd_configure_prototype',name='prepare',inputs={}))
        assert await asyncio.to_thread(entered.wait,1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert await asyncio.to_thread(stopped.wait,1)
    asyncio.run(scenario())
    assert not posted


def test_clear_workspace_releases_run_memory_and_disarms_future_work():
    app = RadarPdNovaApp()
    state = app.server.state
    workspace = app._refinement_workspace = _Workspace()
    workspace.state["jobs"]["waiting"] = {"source": {"experiment": "PG3:2767"}, "status": "queued"}
    app.records["old"] = object()
    app.utility_actions["old"] = object()
    app._powgen_sample_indexes["PG3:2767"] = object()
    app._refinement_view_cache["old"] = {"large": [1, 2, 3]}
    app._refinement_trend_cache["old"] = ({"parameters": [1, 2, 3]}, 128)
    app._refinement_trend_cache_bytes = 128
    app._post_analysis_scans = [{"run_id": "PG3_1"}]
    state.run_rows = [{"uid": "old"}]
    state.selected_run_uid = "old"
    state.powgen_ipts = "2767"
    state.live_sample_elements = "Al Mg O"
    state.powgen_configuration_dataset_id = "old-config"
    state.powgen_database_dataset_id = "old-library"
    state.rf_experiment = "PG3:2767"
    state.rf_context_atoms = [{"phase": "Al2O3", "atom": "Al1"}]
    state.rf_jobs = [{"scan": "PG3_43124"}]

    asyncio.run(app.clear_workspace())

    assert not app.records and not app.utility_actions
    assert not app._powgen_sample_indexes and not app._refinement_view_cache
    assert not app._refinement_trend_cache and app._refinement_trend_cache_bytes == 0
    assert not app._post_analysis_scans
    assert workspace.paused == ["PG3:2767"] and workspace.checkpoints == 1
    assert workspace.state["jobs"]["waiting"]["status"] == "paused"
    assert app._run_visibility_cutoff is not None
    assert state.previous_runs_hidden and not state.run_rows
    assert not state.selected_run_uid and not state.powgen_ipts
    assert not state.live_sample_elements
    assert not state.powgen_configuration_dataset_id and not state.powgen_database_dataset_id
    assert app._refinement_workspace is None
    assert not state.rf_experiment and not state.rf_context_atoms and not state.rf_jobs
    assert "Galaxy jobs, saved results and refinement receipts remain" in state.workspace_notice


def test_clear_workspace_waits_for_submission_acknowledgement():
    app = RadarPdNovaApp()
    state = app.server.state
    state.busy = True
    app.records["pending"] = object()

    asyncio.run(app.clear_workspace())

    assert "pending" in app.records
    assert app._run_visibility_cutoff is None
    assert "acknowledged" in state.workspace_notice


def test_clear_workspace_shows_pending_save_and_prevents_duplicate_actions():
    app = RadarPdNovaApp()
    calls = []
    app.server.state.clear_workspace_dialog = True

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def save():
            calls.append("save")
            entered.set()
            await release.wait()

        app._stop_session_enrollment = save
        pending = asyncio.create_task(app.clear_workspace())
        await entered.wait()
        assert app.server.state.workspace_clearing
        assert "before clearing" in app._session_action_blocker()
        await app.clear_workspace()
        assert app.server.state.clear_workspace_dialog
        assert calls == ["save"]
        release.set()
        await pending
        assert not app.server.state.workspace_clearing
        assert not app.server.state.clear_workspace_dialog

    asyncio.run(scenario())


def test_failed_clear_save_keeps_data_and_allows_explicit_retry():
    app = RadarPdNovaApp()
    app.records["saved"] = object()

    async def failed_save():
        raise RuntimeError("checkpoint temporarily unavailable")

    app._stop_session_enrollment = failed_save
    asyncio.run(app.clear_workspace())
    assert "saved" in app.records
    assert not app.server.state.workspace_clearing
    assert "was not cleared" in app.server.state.workspace_notice


def test_end_session_pauses_queued_refinements_even_without_future_enrollment():
    app = RadarPdNovaApp()
    workspace = app._refinement_workspace = _Workspace()
    workspace.state["recipes"]["old"]["armed"] = False
    workspace.state["jobs"]["waiting"] = {"source": {"experiment": "PG3:2767"}, "status": "queued"}
    _navigation(app)

    async def exit_callback():
        return None

    app.exit_callback = exit_callback
    asyncio.run(app.end_session())

    assert workspace.state["jobs"]["waiting"]["status"] == "paused"
    assert workspace.checkpoints == 1


def test_shutdown_reports_which_recovery_component_is_being_saved():
    app = RadarPdNovaApp()
    messages = []

    async def preparation():
        messages.append(app.server.state.session_save_message)

    async def webmon():
        messages.append(app.server.state.session_save_message)

    async def models():
        messages.append(app.server.state.session_save_message)

    class Workspace(_Workspace):
        def checkpoint(self):
            messages.append(app.server.state.session_save_message)
            super().checkpoint()

    app._preparation_shutdown = preparation
    app.webmon_shutdown = webmon
    app._pause_model_campaigns = models
    app._refinement_workspace = Workspace()
    asyncio.run(app._stop_session_enrollment())
    assert messages == [
        'Saving experiment-preparation recovery state.',
        'Saving live-acquisition recovery state.',
        'Saving model-comparison results and pending work.',
        'Saving refinement recipes and their completed results.',
    ]
    assert app.server.state.session_save_message == 'Recovery checkpoints saved.'


def test_recovery_respects_clear_cutoff_until_user_restores_old_runs():
    app = RadarPdNovaApp()
    now = datetime.now(timezone.utc)
    old = SimpleNamespace(uid="old", galaxy_job_id="old-job", created_utc=now - timedelta(hours=1))
    new = SimpleNamespace(uid="new", galaxy_job_id="new-job", created_utc=now + timedelta(seconds=1))
    app.service = SimpleNamespace(validate_connection=lambda: None,
                                  recent_runs=lambda: [old, new], refresh=lambda record: record)
    app.refresh_history = lambda: None
    app._start_monitor = lambda record: None
    app._sync_runs = lambda: None
    app._run_visibility_cutoff = now

    app._recover_runs()
    assert set(app.records) == {"new"}

    app.restore_earlier_runs()
    assert set(app.records) == {"new", "old"}
    assert not app.server.state.previous_runs_hidden


def test_end_session_uses_existing_exit_path_after_checkpointing():
    app = RadarPdNovaApp()
    workspace = app._refinement_workspace = _Workspace()
    events = []
    _navigation(app,events,"return-to-galaxy")

    async def exit_callback():
        events.append("exit-container")

    app.exit_callback = exit_callback
    asyncio.run(app.end_session())

    assert workspace.paused == ["PG3:2767"] and workspace.checkpoints == 1
    assert events == ["return-to-galaxy", "exit-container"]
    assert app.server.state.nova_kill_jobs_on_exit is False


def test_end_session_waits_for_the_monitor_final_checkpoint():
    app=RadarPdNovaApp();events=[]
    _navigation(app,events)
    async def exit_callback(): events.append('exit')
    app.exit_callback=exit_callback
    async def run():
        release=asyncio.Event()
        async def monitor():
            await release.wait();events.append('saved')
        task=app._powgen_monitor_task=asyncio.create_task(monitor())
        app.server.state.powgen_monitoring=True
        closing=asyncio.create_task(app.end_session())
        await asyncio.sleep(0)
        assert not closing.done() and not task.cancelled() and events==[]
        release.set();await closing
        assert events==['saved','leave','exit']
    asyncio.run(run())


def test_failed_monitor_checkpoint_keeps_the_session_open():
    app=RadarPdNovaApp();events=[]
    app._powgen_checkpoint_error=RuntimeError('checkpoint upload unavailable')
    _navigation(app,events)
    async def exit_callback(): events.append('exit')
    app.exit_callback=exit_callback
    asyncio.run(app.end_session())
    assert events==[]
    assert 'recovery state could not be saved' in app.server.state.workspace_notice


def test_close_retries_a_previously_failed_checkpoint_before_releasing_container():
    app=RadarPdNovaApp();events=[]
    app._powgen_checkpoint_error=RuntimeError('transient failure')
    app._powgen_controller=SimpleNamespace(persist_state=lambda:events.append('saved'))
    _navigation(app,events)
    async def exit_callback(): events.append('exit')
    app.exit_callback=exit_callback
    asyncio.run(app.end_session())
    assert events==['saved','leave','exit']
    assert app._powgen_checkpoint_error is None


def test_concurrent_end_actions_share_one_checkpoint_and_exit():
    app=RadarPdNovaApp();events=[]
    _navigation(app,events)
    async def exit_callback():events.append('exit')
    app.exit_callback=exit_callback
    async def run():
        entered=asyncio.Event();release=asyncio.Event()
        async def save():entered.set();await release.wait();events.append('saved')
        app._stop_session_enrollment=save
        first=asyncio.create_task(app.end_session());await entered.wait()
        assert app.server.state.session_closing
        await app.end_session()
        assert events==[] and not first.done()
        release.set();await first
    asyncio.run(run())
    assert events==['saved','leave','exit']


def test_pending_checkpoint_keeps_session_open_and_reenables_retry():
    from radar_pd_nova.checkpoint_uploads import CheckpointUploadPending
    app=RadarPdNovaApp();events=[]
    async def pending():raise CheckpointUploadPending('hda1','queued')
    app._stop_session_enrollment=pending
    _navigation(app,events)
    asyncio.run(app.end_session())
    assert not events and not app.server.state.session_closing
    assert not app.server.state.end_session_dialog
    assert 'No replacement upload was submitted' in app.server.state.workspace_notice


def test_close_does_not_stack_a_second_pending_wait_after_monitor_drain():
    from radar_pd_nova.checkpoint_uploads import CheckpointUploadPending
    app=RadarPdNovaApp();events=[]
    app._powgen_controller=SimpleNamespace(persist_state=lambda:events.append('explicit retry'))
    _navigation(app,events)
    async def exit_callback():events.append('exit')
    app.exit_callback=exit_callback
    async def run():
        release=asyncio.Event()
        async def monitor():
            await release.wait();app._powgen_checkpoint_error=CheckpointUploadPending('hda1','queued')
            app.server.state.live_launch_busy=False
        app._powgen_monitor_task=asyncio.create_task(monitor())
        app.server.state.powgen_monitoring=True
        closing=asyncio.create_task(app.end_session());await asyncio.sleep(0)
        release.set();await closing
        assert events==[] and not app.server.state.session_closing
        # An explicit subsequent close may retry the retained checkpoint once.
        await app.end_session()
        assert events==['explicit retry','leave','exit']
    asyncio.run(run())
