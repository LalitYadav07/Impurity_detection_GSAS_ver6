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


def test_history_search_preserves_selected_sibling_labels() -> None:
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    selected_data = {
        "id": "data-id",
        "name": "RADAR-PD diffraction data | pattern.dat",
        "display_name": "pattern.dat · data-id",
        "role": "diffraction",
        "generated": False,
    }
    state = _State(
        history_datasets=[selected_data],
        history_show_all=False,
        history_data_id="data-id",
        history_instrument_id="",
        history_main_cif_id="",
        history_database_id="",
        history_configuration_id="",
        library_builder_cif_ids=[],
    )
    app.server = SimpleNamespace(state=state)
    instrument = {
        "id": "instrument-id",
        "name": "RADAR-PD instrument profile | profile.instprm",
        "display_name": "profile.instprm · instrument-id",
        "role": "instrument",
        "generated": False,
    }

    app._apply_history_page([instrument], append=False)

    assert [item["id"] for item in state.history_data_datasets] == ["data-id"]
    assert [item["id"] for item in state.history_instrument_datasets] == ["instrument-id"]


def test_facility_instrument_dropdown_payload_is_normalized() -> None:
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    state = _State(facility_instrument={"title": "HB2A", "value": "HB2A"}, facility_ipts="old")
    app.server = SimpleNamespace(state=state)
    seen: list[str] = []
    app.facility = SimpleNamespace(
        list_ipts=lambda instrument: seen.append(instrument) or [{"title": "IPTS-1", "value": "IPTS-1"}]
    )

    app._facility_instrument_changed(state.facility_instrument)

    assert state.facility_instrument == "HB2A"
    assert state.facility_ipts == ""
    assert state.facility_ipts_options == [{"title": "IPTS-1", "value": "IPTS-1"}]
    assert seen == ["HB2A"]


def test_facility_ipts_dropdown_payload_is_normalized_before_refresh() -> None:
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    state = _State(facility_ipts="{'title': 'IPTS-38548', 'value': 'IPTS-38548'}")
    app.server = SimpleNamespace(state=state)
    refreshed: list[str] = []
    app.refresh_facility_browser = lambda **_: refreshed.append(state.facility_ipts)

    app._facility_ipts_changed(state.facility_ipts)

    assert state.facility_ipts == "IPTS-38548"
    assert refreshed == ["IPTS-38548"]


def test_selected_value_normalizes_serialized_option_payloads() -> None:
    assert RadarPdNovaApp._selected_value("{'title': 'HB2A', 'value': 'HB2A'}") == "HB2A"
    assert RadarPdNovaApp._selected_value('[{"title": "IPTS-1", "value": "IPTS-1"}]') == "IPTS-1"
