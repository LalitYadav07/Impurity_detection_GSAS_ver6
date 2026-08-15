import asyncio
from types import SimpleNamespace

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.models import AnalysisMode, RunRecord, RunStatus, selected_run_uid


def test_selected_run_uid_accepts_vuetify_payload_shapes() -> None:
    assert selected_run_uid(["job-1"]) == "job-1"
    assert selected_run_uid({"value": ["job-2"]}) == "job-2"
    assert selected_run_uid([{"uid": "job-3"}]) == "job-3"
    assert selected_run_uid({"item": {"raw": {"uid": "job-4"}}}) == "job-4"
    assert selected_run_uid([]) == ""


class _State(SimpleNamespace):
    def flush(self) -> None:
        self.flush_count = getattr(self, "flush_count", 0) + 1


def _app_with_record(record: RunRecord) -> tuple[RadarPdNovaApp, _State]:
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    state = _State(
        run_selection=[],
        selected_run_uid="",
        selected_run_name="",
        selected_run_status="",
        selected_run_stage="",
        selected_run_progress=0,
        selected_run_message="",
        selected_run_console="",
        selected_run_loading=False,
        viewed_configuration="",
        active_page="runs",
        error_message="",
        notice="",
    )
    app.server = SimpleNamespace(state=state)
    app.records = {record.uid: record}
    app._opening_run_uid = None
    return app, state


def test_selecting_completed_run_opens_results() -> None:
    record = RunRecord(
        uid="completed-1",
        name="completed run",
        mode=AnalysisMode.RAPID,
        history_id="history-1",
        status=RunStatus.OK,
    )
    app, state = _app_with_record(record)
    opened: list[str] = []
    app._open_record_results = lambda selected: opened.append(selected.uid)

    app._run_selection_changed([record.uid])

    assert state.selected_run_uid == record.uid
    assert opened == [record.uid]


def test_selecting_active_run_keeps_run_monitor_visible() -> None:
    record = RunRecord(
        uid="running-1",
        name="running run",
        mode=AnalysisMode.FULL,
        history_id="history-1",
        status=RunStatus.RUNNING,
        stage="Lattice nudging",
        progress=42,
    )
    app, state = _app_with_record(record)
    opened: list[str] = []
    app._open_record_results = lambda selected: opened.append(selected.uid)

    app._run_selection_changed([record.uid])

    assert state.active_page == "runs"
    assert state.selected_run_stage == "Lattice nudging"
    assert state.selected_run_progress == 42
    assert state.workspace_view == "monitor"
    assert state.setup_collapsed is True
    assert [item["value"] for item in state.workspace_options] == ["monitor", "results", "plots", "files"]
    assert opened == []


def test_monitor_timeline_uses_mode_specific_stages() -> None:
    rapid = RunRecord(
        uid="rapid-stage",
        name="rapid",
        mode=AnalysisMode.RAPID,
        history_id="history-1",
        status=RunStatus.RUNNING,
        stage="Pattern scoring",
        progress=70,
    )
    full = rapid.model_copy(update={"uid": "full-stage", "mode": AnalysisMode.FULL, "stage": "Refinement pass 2"})

    rapid_rows = RadarPdNovaApp._monitor_stage_rows(rapid)
    full_rows = RadarPdNovaApp._monitor_stage_rows(full)

    assert rapid_rows[3] == {"name": "Pattern scoring", "state": "active"}
    assert full_rows[3] == {"name": "Refinement passes", "state": "active"}


def test_async_monitor_publishes_updates_on_the_event_loop(tmp_path) -> None:
    record = RunRecord(
        uid="monitor-1",
        name="monitored run",
        mode=AnalysisMode.RAPID,
        history_id="history-1",
        status=RunStatus.QUEUED,
        stage="Waiting for compute",
        progress=3,
    )

    class _Service:
        refresh_count = 0

        def refresh(self, current: RunRecord) -> RunRecord:
            self.refresh_count += 1
            if self.refresh_count == 1:
                current.status = RunStatus.RUNNING
                current.stage = "Pattern scoring"
                current.progress = 72
            else:
                current.status = RunStatus.OK
                current.stage = "Results ready"
                current.progress = 100
            return current

        def collect_results(self, current: RunRecord) -> RunRecord:
            current.output_dir = str(tmp_path / "run")
            current.message = "Results loaded"
            return current

    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    app.service = _Service()
    app._monitored_uids = {record.uid}
    updates: list[tuple[RunStatus, str, int, str]] = []
    app._monitor_update = lambda current: updates.append(  # type: ignore[method-assign]
        (current.status, current.stage, current.progress, current.message)
    )

    asyncio.run(app._monitor_record(record, poll_seconds=0))

    assert updates == [
        (RunStatus.RUNNING, "Pattern scoring", 72, ""),
        (RunStatus.OK, "Results ready", 100, ""),
        (RunStatus.OK, "Results ready", 100, "Results loaded"),
    ]
    assert record.uid not in app._monitored_uids
