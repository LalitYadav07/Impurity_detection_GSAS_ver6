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


def test_clear_workspace_releases_run_memory_and_disarms_future_work():
    app = RadarPdNovaApp()
    state = app.server.state
    workspace = app._refinement_workspace = _Workspace()
    workspace.state["jobs"]["waiting"] = {"source": {"experiment": "PG3:2767"}, "status": "queued"}
    app.records["old"] = object()
    app.utility_actions["old"] = object()
    app._powgen_sample_indexes["PG3:2767"] = object()
    app._refinement_view_cache["old"] = {"large": [1, 2, 3]}
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


def test_end_session_pauses_queued_refinements_even_without_future_enrollment():
    app = RadarPdNovaApp()
    workspace = app._refinement_workspace = _Workspace()
    workspace.state["recipes"]["old"]["armed"] = False
    workspace.state["jobs"]["waiting"] = {"source": {"experiment": "PG3:2767"}, "status": "queued"}
    app._leave_session_page = lambda: None

    async def exit_callback():
        return None

    app.exit_callback = exit_callback
    asyncio.run(app.end_session())

    assert workspace.state["jobs"]["waiting"]["status"] == "paused"
    assert workspace.checkpoints == 1


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
    app._leave_session_page = lambda: events.append("return-to-galaxy")

    async def exit_callback():
        events.append("exit-container")

    app.exit_callback = exit_callback
    asyncio.run(app.end_session())

    assert workspace.paused == ["PG3:2767"] and workspace.checkpoints == 1
    assert events == ["return-to-galaxy", "exit-container"]
    assert app.server.state.nova_kill_jobs_on_exit is False


def test_end_session_waits_for_the_monitor_final_checkpoint():
    app=RadarPdNovaApp();events=[]
    app._leave_session_page=lambda:events.append('leave')
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
    app._leave_session_page=lambda:events.append('leave')
    async def exit_callback(): events.append('exit')
    app.exit_callback=exit_callback
    asyncio.run(app.end_session())
    assert events==[]
    assert 'recovery state could not be saved' in app.server.state.workspace_notice


def test_close_retries_a_previously_failed_checkpoint_before_releasing_container():
    app=RadarPdNovaApp();events=[]
    app._powgen_checkpoint_error=RuntimeError('transient failure')
    app._powgen_controller=SimpleNamespace(persist_state=lambda:events.append('saved'))
    app._leave_session_page=lambda:events.append('leave')
    async def exit_callback(): events.append('exit')
    app.exit_callback=exit_callback
    asyncio.run(app.end_session())
    assert events==['saved','leave','exit']
    assert app._powgen_checkpoint_error is None
