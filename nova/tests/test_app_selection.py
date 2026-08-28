import asyncio
from pathlib import Path
from types import SimpleNamespace

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.galaxy_service import GSASII_INTERACTIVE_TOOL_ID
from radar_pd_nova.models import (
    AnalysisConfig,
    AnalysisMode,
    InputSelection,
    RunRecord,
    RunStatus,
    UtilityActionRecord,
    selected_run_uid,
)


def test_selected_run_uid_accepts_vuetify_payload_shapes() -> None:
    assert selected_run_uid(["job-1"]) == "job-1"
    assert selected_run_uid({"value": ["job-2"]}) == "job-2"
    assert selected_run_uid([{"uid": "job-3"}]) == "job-3"
    assert selected_run_uid({"item": {"raw": {"uid": "job-4"}}}) == "job-4"
    assert selected_run_uid([]) == ""


def test_workflow_mode_change_collapses_previous_run_history() -> None:
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    state = _State(history_panels=["history"], run_search="PG3_63802")
    app.server = SimpleNamespace(state=state)

    app._workflow_mode_changed()

    assert state.history_panels == []
    assert state.run_search == ""
    assert state.flush_count == 1


def test_loading_results_publishes_only_the_final_plot_figure(monkeypatch, tmp_path: Path) -> None:
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    state = _State(workspace_view="monitor")
    app.server = SimpleNamespace(state=state)
    app.service = SimpleNamespace(result_payload=lambda _record: {"summary": {}})
    app._powgen_controller = None

    class _Widget:
        def __init__(self) -> None:
            self.updates: list[object] = []

        def update(self, figure: object) -> None:
            self.updates.append(figure)

    primary = _Widget()
    gallery = _Widget()
    app._primary_plot_widget = primary
    app._plot_widget = gallery
    expected_figure = object()
    plot_path = str(tmp_path / "best.plotdata.json")
    view = {
        "mode": "full",
        "metrics": [],
        "warnings": [],
        "phases": [],
        "phase_total": "-",
        "tables": [],
        "plots": [{"name": "Best refinement", "path": plot_path, "category": "Best refinement"}],
        "primary_plot_path": plot_path,
        "file_groups": [],
        "checkpoints": [],
        "rapid_stages": {
            "coarse_search": [],
            "lattice_nudge": [],
            "pattern_scoring": [],
            "final_refinement": [],
        },
        "top_refinements": [],
        "full_progression": [],
        "full_models": [],
    }
    monkeypatch.setattr(
        "radar_pd_nova.app.build_result_view",
        lambda *_args, **_kwargs: SimpleNamespace(to_state=lambda: view),
    )
    monkeypatch.setattr(
        "radar_pd_nova.app.load_plot_with_fallback",
        lambda _options, path: (path, {}, expected_figure),
    )
    record = RunRecord(
        uid="completed-with-plot",
        name="completed with plot",
        mode=AnalysisMode.FULL,
        history_id="history-1",
        status=RunStatus.OK,
        output_dir=str(tmp_path),
    )

    app._load_results(record)

    assert primary.updates == [expected_figure]
    assert gallery.updates == [expected_figure]


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
        result_explorer_available=False,
        viewed_configuration="",
        active_page="runs",
        error_message="",
        notice="",
    )
    app.server = SimpleNamespace(state=state)
    app.records = {record.uid: record}
    app._opening_run_uid = None
    return app, state


def test_run_rows_disambiguate_repeated_scans_with_eastern_time_and_job_id() -> None:
    record = RunRecord(
        uid="job-2a64efc6f78b0ea3",
        galaxy_job_id="2a64efc6f78b0ea3",
        name="IPTS-37876_PG3_63798",
        mode=AnalysisMode.FULL,
        history_id="history-1",
        status=RunStatus.OK,
        created_utc="2026-08-25T21:14:18Z",
    )
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    state = _State(run_rows=[])
    app.server = SimpleNamespace(state=state)
    app.records = {record.uid: record}

    app._sync_runs()

    assert state.run_rows[0]["created_display"] == "2026-08-25 17:14 EDT"
    assert state.run_rows[0]["job_short"] == "2a64efc6"


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
    assert state.result_explorer_available is False
    assert opened == [record.uid]


def test_result_explorer_is_available_only_with_a_complete_archive() -> None:
    record = RunRecord(
        uid="completed-archive",
        name="completed run with archive",
        mode=AnalysisMode.FULL,
        history_id="history-1",
        status=RunStatus.OK,
        output_dataset_ids={"results_archive": "archive-hda"},
    )
    app, state = _app_with_record(record)
    app._open_record_results = lambda selected: None

    app._run_selection_changed([record.uid])

    assert state.result_explorer_available is True


def test_selecting_another_run_clears_stale_gsasii_status() -> None:
    record = RunRecord(
        uid="completed-next",
        name="completed next run",
        mode=AnalysisMode.FULL,
        history_id="history-1",
        status=RunStatus.OK,
    )
    app, state = _app_with_record(record)
    state.selected_run_uid = "completed-previous"
    state.gsasii_launch_url = "/interactivetool/old"
    state.gsasii_session_status = "error"
    state.gsasii_status_message = "Galaxy did not publish a GPX collection for this run."

    app._select_record(record)

    assert state.gsasii_launch_url == ""
    assert state.gsasii_session_status == ""
    assert state.gsasii_status_message == ""


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


def test_history_refresh_keeps_reusable_configurations_visible_after_large_upload() -> None:
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    newest = [
        {
            "id": f"cif-{index}",
            "name": f"candidate-{index}.cif",
            "display_name": f"candidate-{index}.cif",
            "role": "cif",
            "generated": False,
        }
        for index in range(100)
    ]
    configuration = {
        "id": "config-id",
        "name": "RADAR-PD configuration | run_radar_pd_config.yaml",
        "display_name": "run_radar_pd_config.yaml",
        "role": "configuration",
        "generated": True,
    }
    library = {
        "id": "library-id",
        "name": "RADAR-PD portable custom library | Fe_Al.zip",
        "display_name": "Fe_Al custom library",
        "role": "candidate_library",
        "generated": False,
    }

    class _Service:
        def search_history_datasets(self, *, query: str, **_kwargs):
            if query == "radar_pd_config":
                return [configuration]
            if query == "portable custom library":
                return [library]
            return newest if not query or query == ".cif" else []

    state = _State(
        history_search="",
        history_offset=0,
        history_datasets=[],
        history_show_all=False,
        history_data_id="",
        history_instrument_id="",
        history_main_cif_id="",
        history_database_id="",
        history_configuration_id="",
        powgen_configuration_dataset_id="",
        powgen_main_cif_dataset_id="",
        library_builder_cif_ids=[],
        error_message="",
        notice="",
    )
    app.server = SimpleNamespace(state=state)
    app.service = _Service()

    app.refresh_history()

    assert [item["id"] for item in state.history_configuration_datasets] == ["config-id"]
    assert [item["id"] for item in state.history_archive_datasets] == ["library-id"]
    assert state.history_has_more is True
    assert state.flush_count == 1


def test_use_configuration_restores_durable_galaxy_inputs_and_full_mode() -> None:
    inputs = InputSelection(
        source="upload",
        instrument_source="upload",
        data_path="/transient/pattern.gsa",
        data_dataset_id="data-id",
        instrument_path="/transient/profile.instprm",
        instrument_dataset_id="instrument-id",
        main_cif_path="/transient/main.cif",
        main_cif_dataset_id="main-id",
        database_archive_path="/transient/library.zip",
        database_dataset_id="library-id",
    )
    record = RunRecord(
        uid="reusable-full-run",
        name="FeVAl full run",
        mode=AnalysisMode.FULL,
        history_id="history-1",
        status=RunStatus.OK,
        config=AnalysisConfig(mode=AnalysisMode.FULL, sample_elements=["Fe", "V", "Al"]),
        inputs=inputs,
    )
    app, state = _app_with_record(record)
    state.selected_run_uid = record.uid
    state.input_source = "upload"
    state.instrument_source = "upload"
    state.main_cif_source = "upload"
    state.database_source = "archive"
    state.library_archive_source = "computer"
    state.data_path = "/stale/data.gsa"
    state.instrument_path = "/stale/profile.instprm"
    state.main_cif_path = "/stale/main.cif"
    state.database_archive_path = "/stale/library.zip"
    state.history_datasets = []
    state.history_show_all = False
    state.history_data_id = ""
    state.history_instrument_id = ""
    state.history_main_cif_id = ""
    state.history_database_id = ""
    state.history_configuration_id = ""
    state.powgen_configuration_dataset_id = ""
    state.powgen_database_dataset_id = ""
    state.powgen_main_cif_dataset_id = ""
    state.library_builder_cif_ids = []

    items = {
        "data-id": {
            "id": "data-id",
            "name": "RADAR-PD diffraction data | pattern.gsa",
            "display_name": "pattern.gsa · data-id",
            "role": "diffraction",
            "generated": True,
        },
        "instrument-id": {
            "id": "instrument-id",
            "name": "RADAR-PD instrument profile | profile.instprm",
            "display_name": "profile.instprm · instrument-id",
            "role": "instrument",
            "generated": True,
        },
        "main-id": {
            "id": "main-id",
            "name": "RADAR-PD main phase | main.cif",
            "display_name": "main.cif · main-id",
            "role": "cif",
            "generated": True,
        },
        "library-id": {
            "id": "library-id",
            "name": "RADAR-PD portable custom library | FeVAl.zip",
            "display_name": "FeVAl candidate library · library-id",
            "role": "candidate_library",
            "generated": False,
        },
    }
    app.service = SimpleNamespace(history_dataset_item=lambda dataset_id: items[dataset_id])

    app.use_selected_configuration()

    assert state.analysis_mode == "full"
    assert state.input_source == "galaxy"
    assert state.instrument_source == "galaxy"
    assert state.main_cif_source == "galaxy"
    assert state.database_source == "archive"
    assert state.library_archive_source == "galaxy"
    assert state.data_path == ""
    assert state.instrument_path == ""
    assert state.main_cif_path == ""
    assert state.database_archive_path == ""
    assert state.history_data_id == "data-id"
    assert state.history_instrument_id == "instrument-id"
    assert state.history_main_cif_id == "main-id"
    assert state.history_database_id == "library-id"
    assert state.history_data_datasets[0]["display_name"] == "pattern.gsa · data-id"
    assert state.history_instrument_datasets[0]["display_name"] == "profile.instprm · instrument-id"
    assert state.history_cif_datasets[0]["display_name"] == "main.cif · main-id"
    assert state.history_archive_datasets[0]["display_name"] == "FeVAl candidate library · library-id"
    assert "durable Galaxy inputs" in state.notice
    assert state.active_page == "setup"


def test_xray_mode_clears_neutron_only_setup_and_restores_options() -> None:
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    state = _State(
        radiation="xray",
        instrument_mode="tof",
        input_source="ipts_browser",
        instrument_source="ipts",
        main_cif_source="ipts",
        use_facility_workspace=True,
        magnetic_precheck=True,
        use_builtin_cuka=True,
        busy=False,
        notice="",
    )
    app.server = SimpleNamespace(state=state)

    app._radiation_changed("xray")

    assert [item["value"] for item in state.instrument_mode_options] == ["auto", "cw"]
    assert [item["value"] for item in state.source_options] == ["upload", "galaxy"]
    assert [item["value"] for item in state.instrument_source_options] == ["upload", "galaxy"]
    assert [item["value"] for item in state.main_cif_source_options] == ["none", "upload", "galaxy"]
    assert state.instrument_mode == "auto"
    assert state.input_source == "upload"
    assert state.instrument_source == "upload"
    assert state.main_cif_source == "none"
    assert state.use_facility_workspace is False
    assert state.magnetic_precheck is False

    app._radiation_changed("neutron")

    assert [item["value"] for item in state.instrument_mode_options] == ["auto", "cw", "tof"]
    assert "ipts_browser" in [item["value"] for item in state.source_options]
    assert state.use_builtin_cuka is False


def test_radiation_change_clears_measurement_specific_restored_inputs() -> None:
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    state = _State(
        radiation="xray",
        last_radiation="neutron",
        instrument_mode="auto",
        input_source="galaxy",
        instrument_source="galaxy",
        main_cif_source="galaxy",
        database_source="archive",
        library_archive_source="galaxy",
        data_path="pattern.gsa",
        history_data_id="neutron-pattern-id",
        remote_data_uri="gxfiles://pattern",
        facility_data_path="/SNS/PG3/pattern.gsa",
        facility_data_relative_path="shared/pattern.gsa",
        event_file_path="events.nxs",
        instrument_path="powgen.instprm",
        history_instrument_id="neutron-profile-id",
        remote_instrument_uri="gxfiles://profile",
        facility_instrument_path="/SNS/PG3/profile.instprm",
        facility_instrument_relative_path="shared/profile.instprm",
        database_archive_path="neutron-library.zip",
        history_database_id="neutron-library-id",
        history_main_cif_id="main-phase-id",
        use_facility_workspace=False,
        magnetic_precheck=False,
        use_builtin_cuka=False,
        busy=False,
        notice="",
    )
    app.server = SimpleNamespace(state=state)

    app._radiation_changed("xray")

    assert state.input_source == "galaxy"
    assert state.history_data_id == ""
    assert state.instrument_source == "upload"
    assert state.history_instrument_id == ""
    assert state.instrument_path == ""
    assert state.database_source == "builtin"
    assert state.history_database_id == ""
    assert state.database_archive_path == ""
    assert state.main_cif_source == "galaxy"
    assert state.history_main_cif_id == "main-phase-id"
    assert state.last_radiation == "xray"
    assert "Reselect the diffraction pattern" in state.notice


def test_applying_saved_configuration_preserves_compatible_inputs() -> None:
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    state = _State(
        radiation="neutron",
        analysis_mode="rapid",
        instrument_mode="tof",
        input_source="galaxy",
        instrument_source="galaxy",
        main_cif_source="galaxy",
        database_source="archive",
        library_archive_source="galaxy",
        history_data_id="data-id",
        history_instrument_id="instrument-id",
        history_main_cif_id="main-id",
        history_database_id="library-id",
        use_facility_workspace=False,
        magnetic_precheck=False,
        busy=False,
        notice="",
        run_name="old name",
    )
    app.server = SimpleNamespace(state=state)
    config = AnalysisConfig(
        mode=AnalysisMode.FULL,
        radiation="xray",
        instrument_mode="cw",
        sample_elements=["Fe", "O"],
    )

    app._apply_configuration(config)

    assert state.analysis_mode == "full"
    assert state.radiation == "xray"
    assert state.instrument_mode == "cw"
    assert [item["value"] for item in state.instrument_mode_options] == ["auto", "cw"]
    assert [item["value"] for item in state.source_options] == ["upload", "galaxy"]
    assert state.input_source == "galaxy"
    assert state.instrument_source == "galaxy"
    assert state.main_cif_source == "galaxy"
    assert state.database_source == "archive"
    assert state.library_archive_source == "galaxy"
    assert state.history_data_id == "data-id"
    assert state.history_instrument_id == "instrument-id"
    assert state.history_main_cif_id == "main-id"
    assert state.history_database_id == "library-id"
    assert state.run_name == ""


def test_powgen_configuration_preview_shows_summary_and_exact_yaml() -> None:
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    config = AnalysisConfig(
        mode=AnalysisMode.FULL,
        sample_elements=["Fe", "Al", "V"],
        environment_elements=["V"],
        instrument_mode="tof",
        limits=(10000, 80000),
    )
    state = _State(
        powgen_configuration_dataset_id="",
        powgen_configuration_summary=[],
        powgen_configuration_yaml="",
        powgen_configuration_error="",
        powgen_configuration_title="",
        powgen_configuration_label="",
        history_configuration_datasets=[
            {"id": "config-id", "display_name": "FeVAl full configuration"}
        ],
    )
    app.server = SimpleNamespace(state=state)

    app._set_powgen_configuration_preview("config-id", config)

    assert state.powgen_configuration_label == "FeVAl full configuration"
    assert state.powgen_configuration_summary[0] == {"label": "Mode", "value": "Full"}
    assert "sample_elements:" in state.powgen_configuration_yaml
    assert "- Fe" in state.powgen_configuration_yaml
    assert "instrument_mode: tof" in state.powgen_configuration_yaml
    assert "created_utc" not in state.powgen_configuration_yaml


def test_standard_configuration_preview_shows_summary_and_exact_yaml() -> None:
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    config = AnalysisConfig(
        mode=AnalysisMode.RAPID,
        radiation="xray",
        instrument_mode="cw",
        sample_elements=["Fe", "O"],
        full_profile="quick",
    )
    state = _State(
        history_configuration_id="",
        history_configuration_summary=[],
        history_configuration_yaml="",
        history_configuration_error="",
        history_configuration_title="",
    )
    app.server = SimpleNamespace(state=state)

    app._set_history_configuration_preview("config-id", config)

    assert state.history_configuration_id == "config-id"
    assert state.history_configuration_title == "Rapid configuration details"
    assert state.history_configuration_summary[0] == {"label": "Mode", "value": "Rapid"}
    assert {"label": "Measurement", "value": "X-ray / CW"} in state.history_configuration_summary
    assert "sample_elements:" in state.history_configuration_yaml
    assert "- Fe" in state.history_configuration_yaml
    assert "created_utc" not in state.history_configuration_yaml


def test_powgen_completed_scan_projects_scientific_summary_into_dashboard() -> None:
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    state = _State(
        powgen_rows=[],
        powgen_scientific_rows=[],
        powgen_all_scientific_rows=[],
        powgen_sample_options=[],
        powgen_selected_samples=[],
        powgen_selected_phase_labels=[],
        powgen_x_axis="run_number",
        powgen_selected_run_id="",
    )
    app.server = SimpleNamespace(state=state)
    completed_run = SimpleNamespace(
        run_number=63802,
        source_path="/SNS/PG3/IPTS-37876/shared/autoreduce/PG3_63802.gsa",
        galaxy_result_ids=["result-1"],
        scientific_summary={
            "rwp": 8.25,
            "elapsed_seconds": 31.0,
            "analysis_mode": "full",
            "phases": [
                {
                    "phase": "Al Fe2 V",
                    "space_group": "F m -3 m (225)",
                    "weight_percent": 100.0,
                }
            ],
        },
        scan_metadata={
            "sample_id": "118400",
            "sample_name": "Fe2VAl Base Alloy",
            "temperature": {"value": 193.657, "unit": "K", "source": "NeXus"},
        },
        error="",
        galaxy_job_id="job-1",
    )
    app._powgen_controller = SimpleNamespace(
        state=SimpleNamespace(
            submitted={},
            failed={},
            completed={"PG3_63802": completed_run},
            discovered={},
        ),
        settings=SimpleNamespace(max_active_jobs=5),
        records={},
        source_directory=Path("/SNS/PG3/IPTS-37876/shared/autoreduce"),
    )
    app._powgen_phase_widget = None
    app._powgen_quality_widget = None
    app._powgen_heatmap_widget = None
    app._update_powgen_dashboard_figures = lambda rows: None  # type: ignore[method-assign]

    app._sync_powgen_rows()

    assert state.powgen_rows[0]["status"] == "Completed"
    assert state.powgen_dashboard_metrics[0]["value"] == "1"
    phase = state.powgen_scientific_rows[0]["phases"][0]
    assert phase["phase"] == "AlFe2V"
    assert phase["label"] == "AlFe2V (SG Fm-3m (225))"


def test_open_powgen_completed_scan_rehydrates_restored_galaxy_record() -> None:
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    restored = RunRecord(
        uid="job-1",
        galaxy_job_id="job-1",
        name="IPTS-37876_PG3_63802",
        mode=AnalysisMode.FULL,
        history_id="history-1",
        status=RunStatus.QUEUED,
        stage="Recovering Galaxy job",
    )
    refreshed = restored.model_copy(deep=True)
    refreshed.status = RunStatus.OK
    refreshed.analysis_status = RunStatus.OK
    refreshed.output_dataset_ids = {
        "summary": "summary-hda",
        "results_archive": "archive-hda",
    }
    state = _State(powgen_selected_run_id="PG3_63802", error_message="")
    app.server = SimpleNamespace(state=state)
    app.service = SimpleNamespace(refresh=lambda record: refreshed)
    controller = SimpleNamespace(
        records={"PG3_63802": restored},
        state=SimpleNamespace(completed={"PG3_63802": SimpleNamespace()}),
    )
    app._powgen_controller = controller
    app.records = {}
    selected: list[RunRecord] = []
    opened: list[RunRecord] = []
    app._select_record = selected.append  # type: ignore[method-assign]
    app._open_record_results = opened.append  # type: ignore[method-assign]

    app.open_powgen_selected_run()

    assert controller.records["PG3_63802"] is refreshed
    assert app.records["job-1"] is refreshed
    assert selected == [refreshed]
    assert opened == [refreshed]
    assert state.error_message == ""


def test_run_configuration_prefers_published_resolved_yaml(tmp_path: Path) -> None:
    output = tmp_path / "result"
    output.mkdir()
    resolved = output / "resolved_config.yaml"
    resolved.write_text(
        "$schema: radar-pd-config/v1\ncreated_utc: 2026-08-24T04:21:26Z\nanalysis:\n  mode: full\n",
        encoding="utf-8",
    )
    record = RunRecord(
        uid="run-with-resolved-config",
        name="resolved run",
        mode=AnalysisMode.FULL,
        history_id="history-1",
        status=RunStatus.OK,
        output_dir=str(output),
        config=AnalysisConfig(sample_elements=["Fe"], mode=AnalysisMode.RAPID),
    )

    rendered = RadarPdNovaApp._record_configuration_yaml(record)

    assert "created_utc: 2026-08-24T04:21:26Z" in rendered
    assert "mode: full" in rendered
    assert "mode: rapid" not in rendered


def test_run_provenance_carries_configuration_and_library_names_forward() -> None:
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    state = _State(
        application_version="v0.3.79",
        history_datasets=[
            {"id": "config-id", "display_name": "FeVAl full configuration"},
            {"id": "library-id", "display_name": "FeVAl candidate family"},
        ],
    )
    app.server = SimpleNamespace(state=state)
    app._dataset_label_cache = {}
    record = RunRecord(
        uid="provenance-run",
        name="provenance run",
        mode=AnalysisMode.FULL,
        history_id="history-1",
        status=RunStatus.OK,
        input_dataset_ids={"configuration": "config-id"},
        config=AnalysisConfig(
            sample_elements=["Fe", "V", "Al"],
            mode=AnalysisMode.FULL,
            instrument_mode="tof",
        ),
        inputs=InputSelection(
            source="upload",
            data_path="pattern.gsa",
            instrument_path="profile.instprm",
            database_dataset_id="library-id",
        ),
    )

    provenance = {item["label"]: item["value"] for item in app._run_provenance(record)}

    assert provenance["Saved configuration"] == "FeVAl full configuration"
    assert provenance["Analysis profile"] == "Full / Neutron / TOF"
    assert provenance["Candidate library"] == "FeVAl candidate family"


def test_phantom_checkpoint_is_not_exposed_as_gsasii_action(tmp_path: Path) -> None:
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    record = RunRecord(
        uid="job-with-phantom-gpx",
        name="old result",
        mode=AnalysisMode.FULL,
        history_id="history-1",
        status=RunStatus.OK,
    )
    rows = [
        {
            "id": "checkpoint-0",
            "path": "",
            "local_available": False,
            "handoff_available": True,
            "galaxy_element_name": "02_Main_phase_anchor",
        }
    ]

    assert app._launchable_checkpoint_rows(record, rows) == []


def test_real_galaxy_checkpoint_collection_enables_gsasii_action() -> None:
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    record = RunRecord(
        uid="job-with-real-gpx",
        name="new result",
        mode=AnalysisMode.FULL,
        history_id="history-1",
        status=RunStatus.OK,
        output_dataset_ids={"gpx_projects": "gpx-collection"},
    )
    rows = [{"id": "checkpoint-0", "path": "", "local_available": False}]

    launchable = app._launchable_checkpoint_rows(record, rows)

    assert record.output_dataset_ids["gpx_projects"] == "gpx-collection"
    assert launchable[0]["handoff_available"] is True
    assert launchable[0]["local_available"] is False


def test_archived_local_gpx_enables_action_without_collection(tmp_path: Path) -> None:
    project = tmp_path / "accepted.gpx"
    project.write_bytes(b"GPX")
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    record = RunRecord(
        uid="job-with-local-gpx",
        name="archived result",
        mode=AnalysisMode.FULL,
        history_id="history-1",
        status=RunStatus.OK,
    )

    launchable = app._launchable_checkpoint_rows(
        record,
        [{"id": "checkpoint-0", "path": str(project), "local_available": True}],
    )

    assert launchable[0]["local_available"] is True


def test_powgen_main_cif_can_be_reused_from_galaxy_history() -> None:
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    state = _State(
        powgen_main_cif_source="galaxy",
        powgen_main_cif_dataset_id="cif-dataset-1",
        powgen_main_cif_path="",
    )
    app.server = SimpleNamespace(state=state)

    assert app._resolve_powgen_main_cif_id() == "cif-dataset-1"


def test_powgen_main_cif_upload_is_saved_and_selected(tmp_path) -> None:
    cif_path = tmp_path / "known_phase.cif"
    cif_path.write_text("data_known_phase\n", encoding="utf-8")
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    state = _State(
        powgen_main_cif_source="computer",
        powgen_main_cif_dataset_id="",
        powgen_main_cif_path=str(cif_path),
        notice="",
    )
    app.server = SimpleNamespace(state=state)
    uploads: list[tuple[Path, str]] = []

    class _Service:
        def upload_document(self, path: Path, *, label: str) -> str:
            uploads.append((path, label))
            return "uploaded-cif-1"

    app.service = _Service()
    app.refresh_history = lambda: None

    assert app._resolve_powgen_main_cif_id() == "uploaded-cif-1"
    assert uploads == [(cif_path, "POWGEN main phase CIF")]
    assert state.powgen_main_cif_source == "galaxy"
    assert state.powgen_main_cif_dataset_id == "uploaded-cif-1"
    assert "known_phase.cif" in state.notice


def test_integrated_library_builder_bundles_reusable_history_cifs(tmp_path) -> None:
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    state = _State(
        library_builder_cif_ids=["cif-1", "cif-2"],
        library_builder_local_paths=[],
        library_builder_name="Y Fe Si candidates",
        library_builder_mode="mini",
        library_builder_active=False,
        library_builder_status="idle",
        library_builder_progress=0,
        library_builder_built_count=0,
        library_builder_skipped_count=0,
        library_builder_message="",
        radiation="neutron",
        database_source="custom_mini",
        error_message="",
    )
    app.server = SimpleNamespace(state=state)
    downloaded: list[str] = []
    uploads: list[tuple[str, str, str]] = []
    submitted: list[dict[str, object]] = []

    class _Service:
        def _dataset_metadata(self, dataset_id: str):
            return {"name": f"{dataset_id}.cif"}

        def _download_dataset(self, dataset_id: str, destination: Path) -> None:
            downloaded.append(dataset_id)
            destination.write_text(
                "\n".join(
                    (
                        f"data_{dataset_id}",
                        "_cell_length_a 2.86",
                        "_cell_length_b 2.86",
                        "_cell_length_c 2.86",
                        "_cell_angle_alpha 90",
                        "_cell_angle_beta 90",
                        "_cell_angle_gamma 90",
                    )
                ),
                encoding="utf-8",
            )

        def _upload_one(self, key: str, path: str, label: str):
            uploads.append((key, path, label))
            return key, "bundle-1"

    app.service = _Service()

    async def _submit(**kwargs):
        submitted.append(kwargs)
        return UtilityActionRecord(
            uid="utility-1",
            tool_id=str(kwargs["tool_id"]),
            name=str(kwargs["name"]),
            status=RunStatus.OK,
        )

    app._submit_utility_action = _submit  # type: ignore[method-assign]
    app._schedule_utility = lambda coroutine, _name: asyncio.run(coroutine)  # type: ignore[method-assign]

    app.build_candidate_library()

    assert downloaded == ["cif-1", "cif-2"]
    assert len(uploads) == 1
    assert submitted[0]["inputs"] == {
        "cif_archive": {"dataset_id": "bundle-1"},
        "library_name": "Y Fe Si candidates",
        "library_mode": "mini",
        "radiation": "neutron",
        "overwrite": "",
    }
    assert state.library_builder_active is False


def test_integrated_library_builder_uploads_one_bundle_for_many_local_cifs(tmp_path) -> None:
    cif = b"""data_Fe
_cell_length_a 2.86
_cell_length_b 2.86
_cell_length_c 2.86
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
"""
    paths = []
    for index in range(3):
        path = tmp_path / f"Fe_{index}.cif"
        path.write_bytes(cif + f"# {index}\n".encode())
        paths.append(str(path))
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    state = _State(
        library_builder_cif_ids=[],
        library_builder_local_paths=paths,
        library_builder_name="Fe candidates",
        library_builder_mode="mini",
        library_builder_active=False,
        library_builder_status="idle",
        library_builder_progress=0,
        library_builder_built_count=0,
        library_builder_skipped_count=0,
        library_builder_failure_rows=[],
        library_builder_message="",
        radiation="neutron",
        database_source="custom_augmented",
        error_message="",
    )
    app.server = SimpleNamespace(state=state)
    uploads: list[tuple[str, str, str]] = []
    submitted: list[dict[str, object]] = []

    class _Service:
        def _upload_one(self, key: str, path: str, label: str):
            uploads.append((key, path, label))
            return key, "bundle-1"

    app.service = _Service()

    async def _submit(**kwargs):
        submitted.append(kwargs)
        return UtilityActionRecord(uid="utility-2", tool_id=str(kwargs["tool_id"]), name=str(kwargs["name"]), status=RunStatus.OK)

    app._submit_utility_action = _submit  # type: ignore[method-assign]
    app._schedule_utility = lambda coroutine, _name: asyncio.run(coroutine)  # type: ignore[method-assign]

    app.build_candidate_library()

    assert len(uploads) == 1
    assert uploads[0][0] == "library_cif_bundle"
    assert submitted[0]["inputs"] == {
        "cif_archive": {"dataset_id": "bundle-1"},
        "library_name": "Fe candidates",
        "library_mode": "augmented",
        "radiation": "neutron",
        "overwrite": "",
    }


def test_selected_checkpoint_starts_native_gsasii_session() -> None:
    record = RunRecord(
        uid="run-with-gpx",
        name="completed run",
        mode=AnalysisMode.FULL,
        history_id="history",
        status=RunStatus.OK,
        output_dataset_ids={"gpx_projects": "gpx-collection"},
    )
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    state = _State(
        selected_run_uid=record.uid,
        selected_checkpoint="checkpoint-0",
        checkpoint_rows=[
            {
                "id": "checkpoint-0",
                "path": "",
                "galaxy_element_name": "Accepted_model_after_pass_2.gpx",
            }
        ],
        gsasii_launch_url="",
        gsasii_session_status="",
        notice="",
        error_message="",
    )
    app.server = SimpleNamespace(state=state)
    app.records = {record.uid: record}
    app.service = SimpleNamespace(
        collection_elements=lambda collection_id: [
            {"id": "accepted-gpx", "name": "Accepted_model_after_pass_2.gpx"}
        ]
    )
    submitted: list[dict[str, object]] = []

    async def _submit(**kwargs):
        submitted.append(kwargs)
        return UtilityActionRecord(
            uid="gsasii-action",
            tool_id=str(kwargs["tool_id"]),
            name=str(kwargs["name"]),
            status=RunStatus.RUNNING,
        )

    app._submit_utility_action = _submit  # type: ignore[method-assign]
    app._schedule_utility = lambda coroutine, _name: asyncio.run(coroutine)  # type: ignore[method-assign]

    app.open_selected_checkpoint_in_gsasii()

    assert state.gsasii_session_status == "starting"
    assert submitted[0]["tool_id"] == GSASII_INTERACTIVE_TOOL_ID
    assert submitted[0]["inputs"] == {
        "project_source|source_kind": "single",
        "project_source|gpx_project": {"dataset_id": "accepted-gpx"},
    }


def test_selected_checkpoint_uses_collection_when_galaxy_omits_element_view() -> None:
    record = RunRecord(
        uid="run-with-gpx-shell",
        name="IPTS-37876_PG3_63802",
        mode=AnalysisMode.FULL,
        history_id="history",
        status=RunStatus.OK,
        output_dataset_ids={"gpx_projects": "gpx-collection"},
    )
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    state = _State(
        selected_run_uid=record.uid,
        selected_checkpoint="checkpoint-0",
        checkpoint_rows=[
            {
                "id": "checkpoint-0",
                "name": "Seq final main polished (GPX)",
                "path": "",
                "galaxy_element_name": "02_Main_phase_anchor",
            }
        ],
        gsasii_launch_url="",
        gsasii_session_status="",
        notice="",
        error_message="stale error",
    )
    app.server = SimpleNamespace(state=state)
    app.records = {record.uid: record}
    app.service = SimpleNamespace(collection_elements=lambda collection_id: [])
    submitted: list[dict[str, object]] = []

    async def _submit(**kwargs):
        submitted.append(kwargs)
        return UtilityActionRecord(
            uid="gsasii-action",
            tool_id=str(kwargs["tool_id"]),
            name=str(kwargs["name"]),
            status=RunStatus.RUNNING,
        )

    app._submit_utility_action = _submit  # type: ignore[method-assign]
    app._schedule_utility = lambda coroutine, _name: asyncio.run(coroutine)  # type: ignore[method-assign]

    app.open_selected_checkpoint_in_gsasii()

    assert state.gsasii_session_status == "starting"
    assert state.error_message == ""
    assert "preferred accepted checkpoint" in state.notice
    assert submitted[0]["inputs"] == {
        "project_source|source_kind": "collection",
        "project_source|gpx_projects": {"collection_id": "gpx-collection"},
    }


def test_selected_checkpoint_recovers_collection_id_from_job_outputs() -> None:
    record = RunRecord(
        uid="recovered-run-without-gpx-id",
        galaxy_job_id="analysis-job",
        name="IPTS-37876_PG3_63802",
        mode=AnalysisMode.FULL,
        history_id="history",
        status=RunStatus.OK,
        output_dataset_ids={},
    )
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    state = _State(
        selected_run_uid=record.uid,
        selected_checkpoint="checkpoint-0",
        checkpoint_rows=[
            {
                "id": "checkpoint-0",
                "name": "Seq final main polished (GPX)",
                "path": "",
                "galaxy_element_name": "02_Main_phase_anchor",
            }
        ],
        gsasii_launch_url="",
        gsasii_session_status="",
        notice="",
        error_message="",
    )
    app.server = SimpleNamespace(state=state)
    app.records = {record.uid: record}
    requested_jobs: list[str] = []
    app.service = SimpleNamespace(
        job_output_ids=lambda job_id: requested_jobs.append(job_id)
        or {"gpx_projects": "recovered-gpx-collection"},
        collection_elements=lambda collection_id: [
            {"id": "main-anchor-gpx", "name": "02_Main_phase_anchor"}
        ],
    )
    submitted: list[dict[str, object]] = []

    async def _submit(**kwargs):
        submitted.append(kwargs)
        return UtilityActionRecord(
            uid="gsasii-action",
            tool_id=str(kwargs["tool_id"]),
            name=str(kwargs["name"]),
            status=RunStatus.RUNNING,
        )

    app._submit_utility_action = _submit  # type: ignore[method-assign]
    app._schedule_utility = lambda coroutine, _name: asyncio.run(coroutine)  # type: ignore[method-assign]

    app.open_selected_checkpoint_in_gsasii()

    assert requested_jobs == ["analysis-job"]
    assert record.output_dataset_ids["gpx_projects"] == "recovered-gpx-collection"
    assert state.gsasii_session_status == "starting"
    assert submitted[0]["inputs"] == {
        "project_source|source_kind": "single",
        "project_source|gpx_project": {"dataset_id": "main-anchor-gpx"},
    }


def test_monitor_checkpoint_is_published_before_native_gsasii_launch(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "Seq_final_main_polished.gpx"
    checkpoint_path.write_bytes(b"GSAS-II project")
    record = RunRecord(
        uid="monitor-run-with-archive-gpx",
        galaxy_job_id="monitor-job",
        name="IPTS-37876_PG3_63802",
        mode=AnalysisMode.FULL,
        history_id="history",
        status=RunStatus.OK,
        output_dataset_ids={},
    )
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    state = _State(
        selected_run_uid=record.uid,
        selected_checkpoint="checkpoint-0",
        checkpoint_rows=[
            {
                "id": "checkpoint-0",
                "name": "Seq final main polished (GPX)",
                "path": str(checkpoint_path),
                "galaxy_element_name": "Seq_final_main_polished",
            }
        ],
        gsasii_launch_url="",
        gsasii_session_status="",
        notice="",
        error_message="stale error",
    )
    app.server = SimpleNamespace(state=state)
    app.records = {record.uid: record}
    uploads: list[tuple[Path, str]] = []
    app.service = SimpleNamespace(
        job_output_ids=lambda _job_id: {},
        upload_document=lambda path, *, label: uploads.append((Path(path), label)) or "uploaded-gpx",
    )
    submitted: list[dict[str, object]] = []

    async def _submit(**kwargs):
        submitted.append(kwargs)
        return UtilityActionRecord(
            uid="gsasii-action",
            tool_id=str(kwargs["tool_id"]),
            name=str(kwargs["name"]),
            status=RunStatus.RUNNING,
        )

    app._submit_utility_action = _submit  # type: ignore[method-assign]
    app._schedule_utility = lambda coroutine, _name: asyncio.run(coroutine)  # type: ignore[method-assign]

    app.open_selected_checkpoint_in_gsasii()

    assert uploads == [(checkpoint_path, "GSAS-II checkpoint IPTS-37876_PG3_63802")]
    assert state.error_message == ""
    assert submitted[0]["inputs"] == {
        "project_source|source_kind": "single",
        "project_source|gpx_project": {"dataset_id": "uploaded-gpx"},
    }

    state.gsasii_session_status = ""
    app.open_selected_checkpoint_in_gsasii()

    assert len(uploads) == 1
    assert len(submitted) == 2


def test_monitor_checkpoint_reports_missing_archive_gpx() -> None:
    record = RunRecord(
        uid="monitor-run-without-gpx",
        galaxy_job_id="monitor-job",
        name="IPTS-37876_PG3_63802",
        mode=AnalysisMode.FULL,
        history_id="history",
        status=RunStatus.OK,
        output_dataset_ids={},
    )
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    state = _State(
        selected_run_uid=record.uid,
        selected_checkpoint="checkpoint-0",
        checkpoint_rows=[{"id": "checkpoint-0", "path": ""}],
        gsasii_launch_url="",
        gsasii_session_status="",
        notice="",
        error_message="",
    )
    app.server = SimpleNamespace(state=state)
    app.records = {record.uid: record}
    app.service = SimpleNamespace(job_output_ids=lambda _job_id: {})
    app._schedule_utility = lambda coroutine, _name: asyncio.run(coroutine)  # type: ignore[method-assign]

    app.open_selected_checkpoint_in_gsasii()

    assert state.gsasii_session_status == "error"
    assert "neither a published Galaxy GPX nor an available GPX checkpoint" in state.error_message


def test_checkpoint_alias_maps_technical_name_to_published_anchor() -> None:
    record = RunRecord(
        uid="run-with-renamed-gpx",
        name="completed run",
        mode=AnalysisMode.FULL,
        history_id="history",
        status=RunStatus.OK,
        output_dataset_ids={"gpx_projects": "gpx-collection"},
    )
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    state = _State(
        selected_run_uid=record.uid,
        selected_checkpoint="checkpoint-0",
        checkpoint_rows=[
            {
                "id": "checkpoint-0",
                "name": "Seq final main polished (GPX)",
                "path": "",
                "galaxy_element_name": "",
            }
        ],
        gsasii_launch_url="",
        gsasii_session_status="",
        notice="",
        error_message="",
    )
    app.server = SimpleNamespace(state=state)
    app.records = {record.uid: record}
    app.service = SimpleNamespace(
        collection_elements=lambda collection_id: [
            {"id": "main-anchor-gpx", "name": "02_Main_phase_anchor"},
            {"id": "pass-1-gpx", "name": "Accepted_model_after_pass_1"},
        ]
    )
    submitted: list[dict[str, object]] = []

    async def _submit(**kwargs):
        submitted.append(kwargs)
        return UtilityActionRecord(
            uid="gsasii-action",
            tool_id=str(kwargs["tool_id"]),
            name=str(kwargs["name"]),
            status=RunStatus.RUNNING,
        )

    app._submit_utility_action = _submit  # type: ignore[method-assign]
    app._schedule_utility = lambda coroutine, _name: asyncio.run(coroutine)  # type: ignore[method-assign]

    app.open_selected_checkpoint_in_gsasii()

    assert submitted[0]["inputs"] == {
        "project_source|source_kind": "single",
        "project_source|gpx_project": {"dataset_id": "main-anchor-gpx"},
    }


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
