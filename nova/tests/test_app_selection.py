import asyncio
from pathlib import Path
from types import SimpleNamespace

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.galaxy_service import GSASII_INTERACTIVE_TOOL_ID
from radar_pd_nova.models import AnalysisMode, RunRecord, RunStatus, UtilityActionRecord, selected_run_uid


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

    class _Service:
        def search_history_datasets(self, *, query: str, **_kwargs):
            return [configuration] if query == "radar_pd_config" else newest

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
    assert state.history_has_more is True
    assert state.flush_count == 1


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
