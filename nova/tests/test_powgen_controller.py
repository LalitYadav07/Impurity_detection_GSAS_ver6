from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from radar_pd_nova.models import AnalysisConfig, AnalysisMode, InputSelection, RunRecord, RunStatus
from radar_pd_nova.powgen_controller import (
    PowgenExperimentSettings,
    PowgenWatchController,
    bounded_directory_listing,
    preflight_powgen_experiment,
)
from radar_pd_nova import powgen_controller


class FakeGalaxyService:
    history_id = "history-1"

    def __init__(self) -> None:
        self.snapshots = []
        self.uploaded = []
        self.history_documents = []
        self.json_documents = {}
        self.recovered_runs = []

    def load_configuration_dataset(self, dataset_id: str) -> AnalysisConfig:
        assert dataset_id == "config-hda"
        return AnalysisConfig(mode=AnalysisMode.RAPID, sample_elements=["Y", "Fe", "Si", "Ga"])

    def create_submission_snapshot(self, config, inputs, *, idempotency_token):
        snapshot = SimpleNamespace(config=config, inputs=inputs, idempotency_token=idempotency_token)
        self.snapshots.append(snapshot)
        return snapshot

    def submit_snapshot(self, snapshot):
        return RunRecord(
            uid="job-1",
            name=snapshot.config.run_name,
            mode=snapshot.config.mode,
            history_id=self.history_id,
            galaxy_job_id="galaxy-job-1",
            status=RunStatus.QUEUED,
            analysis_status=RunStatus.QUEUED,
            config=snapshot.config,
            inputs=snapshot.inputs,
        )

    def upload_json_document(self, payload, *, name, label):
        self.uploaded.append((payload, name, label))
        return f"state-{len(self.uploaded)}"

    def search_history_datasets(self, **_kwargs):
        return list(self.history_documents)

    def load_json_document(self, dataset_id):
        if dataset_id in self.json_documents:
            return self.json_documents[dataset_id]
        return next(row["payload"] for row in self.history_documents if row["id"] == dataset_id)

    def recent_runs(self, *, limit=50):
        return list(self.recovered_runs[:limit])

    def refresh(self, record):
        record.status = RunStatus.OK
        record.analysis_status = RunStatus.OK
        record.output_dataset_ids = {"results_archive": "archive-hda"}
        return record

    def collect_results(self, _record):
        raise AssertionError("POWGEN status refresh must not download the result archive")


def test_bounded_listing_is_nonrecursive_and_ignores_directories(tmp_path: Path) -> None:
    (tmp_path / "PG3_63764.gsa").write_text("data", encoding="ascii")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "PG3_63765.gsa").write_text("data", encoding="ascii")

    rows = bounded_directory_listing(tmp_path)

    assert [Path(row["path"]).name for row in rows] == ["PG3_63764.gsa"]


def test_controller_submits_existing_analyze_contract_and_persists_state() -> None:
    service = FakeGalaxyService()
    settings = PowgenExperimentSettings(
        ipts="IPTS-38000",
        history_id="history-1",
        configuration_dataset_id="config-hda",
        wavelength_angstrom="1.5",
        frequency_hz="60",
    )
    controller = PowgenWatchController(service, settings)
    listing = [
        {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764-2.dat", "size": 10},
        {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764.gsa", "size": 20},
    ]

    run = controller.discover(listing)[0]
    record = controller.submit(run)

    assert run.source_path.endswith("PG3_63764.gsa")
    assert record.galaxy_job_id == "galaxy-job-1"
    snapshot = service.snapshots[0]
    assert snapshot.config.run_name == "IPTS-38000_PG3_63764"
    assert snapshot.inputs.data_path == run.source_path
    assert snapshot.inputs.instrument_path.endswith("2026B_HighRes_60HzB2_CWL1p5.instprm")
    assert snapshot.inputs.publish_results_to_ipts is False
    assert service.uploaded[-1][2] == "POWGEN watch state IPTS-38000"
    assert "PG3_63764" in service.uploaded[-1][0]["submitted"]
    assert service.uploaded[-1][0]["watch"]["source_directory"].endswith("shared/autoreduce")


def test_controller_propagates_custom_library_to_every_monitored_scan() -> None:
    service = FakeGalaxyService()
    settings = PowgenExperimentSettings(
        ipts="IPTS-38000",
        history_id="history-1",
        configuration_dataset_id="config-hda",
        database_dataset_id="custom-library-hda",
        wavelength_angstrom="1.5",
    )
    controller = PowgenWatchController(service, settings)
    run = controller.discover(
        [{"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764.gsa"}]
    )[0]

    controller.submit(run)

    snapshot = service.snapshots[0]
    assert snapshot.inputs.database_dataset_id == "custom-library-hda"
    assert controller.definition.config_refs["candidate_library"] == "custom-library-hda"
    assert service.uploaded[-1][0]["watch"]["database_dataset_id"] == "custom-library-hda"


def test_controller_backfills_all_existing_gsa_then_submits_all_new_runs() -> None:
    service = FakeGalaxyService()
    controller = PowgenWatchController(
        service,
        PowgenExperimentSettings(
            ipts="IPTS-38000",
            history_id="history-1",
            configuration_dataset_id="config-hda",
            wavelength_angstrom="1.5",
        ),
    )

    initial = controller.discover(
        [
            {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63762.gsa"},
            {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63763.gsa"},
            {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764.dat"},
            {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764.gsa"},
        ]
    )
    later = controller.discover(
        [
            {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764.gsa"},
            {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63765.gsa"},
            {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63766.gsa"},
        ]
    )

    assert [run.run_number for run in initial] == [63762, 63763, 63764]
    assert [run.run_number for run in later] == [63765, 63766]


def test_controller_limits_initial_backfill_without_leaking_older_scans_later() -> None:
    service = FakeGalaxyService()
    controller = PowgenWatchController(
        service,
        PowgenExperimentSettings(
            ipts="IPTS-38000",
            history_id="history-1",
            configuration_dataset_id="config-hda",
            wavelength_angstrom="1.5",
            initial_backfill_limit=2,
        ),
    )
    initial_listing = [
        {"path": f"/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_{run_number}.gsa"}
        for run_number in range(63760, 63766)
    ]

    initial = controller.discover(initial_listing)
    later = controller.discover(
        initial_listing
        + [{"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63766.gsa"}]
    )

    assert [run.run_number for run in initial] == [63764, 63765]
    assert [run.run_number for run in later] == [63766]


def test_controller_new_only_ignores_startup_files_and_discovers_next_scan() -> None:
    service = FakeGalaxyService()
    controller = PowgenWatchController(
        service,
        PowgenExperimentSettings(
            ipts="IPTS-38000",
            history_id="history-1",
            configuration_dataset_id="config-hda",
            wavelength_angstrom="1.5",
            initial_backfill_limit=0,
        ),
    )
    initial_listing = [
        {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764.gsa"},
        {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63765.gsa"},
    ]

    assert controller.discover(initial_listing) == []
    discovered = controller.discover(
        initial_listing
        + [{"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63766.gsa"}]
    )

    assert [run.run_number for run in discovered] == [63766]


def test_preflight_detects_latest_scan_wavelength_and_packaged_profile(monkeypatch) -> None:
    monkeypatch.setattr(
        powgen_controller,
        "bounded_directory_listing",
        lambda _directory: [
            {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63763.gsa"},
            {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764.gsa"},
        ],
    )
    monkeypatch.setattr(
        powgen_controller,
        "read_powgen_scan_metadata",
        lambda _ipts, _run: {
            "available": True,
            "sample_id": "118950",
            "temperature": {"value": 300.0, "unit": "K"},
            "wavelength": {"value": 1.5, "unit": "A"},
        },
    )

    result = preflight_powgen_experiment("IPTS-38000", requested_wavelength="0.8")

    assert result["ready"] is True
    assert result["scan_count"] == 2
    assert result["latest_run"] == "PG3_63764"
    assert result["detected_wavelength"] == "1.5"
    assert result["selected_wavelength"] == "1.5"
    assert result["profile_filename"].endswith("60HzB2_CWL1p5.instprm")


def test_controller_caps_concurrent_backfill_submissions() -> None:
    service = FakeGalaxyService()
    controller = PowgenWatchController(
        service,
        PowgenExperimentSettings(
            ipts="IPTS-38000",
            history_id="history-1",
            configuration_dataset_id="config-hda",
            wavelength_angstrom="1.5",
            max_active_jobs=3,
        ),
    )
    runs = controller.discover(
        [
            {"path": f"/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_{run_number}.gsa"}
            for run_number in range(63760, 63767)
        ]
    )
    controller.state.mark_submitted(runs[0], "job-a")

    assert [run.run_number for run in controller.due_submissions()] == [63761, 63762]


def test_launch_and_acknowledge_keep_checkpoint_transition_explicit() -> None:
    service = FakeGalaxyService()
    controller = PowgenWatchController(
        service,
        PowgenExperimentSettings(
            ipts="IPTS-38000",
            history_id="history-1",
            configuration_dataset_id="config-hda",
            wavelength_angstrom="1.5",
        ),
    )
    run = controller.discover(
        [{"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764.gsa"}]
    )[0]

    record = controller.launch_submission(run)

    assert run.run_id in controller.state.discovered
    assert run.run_id not in controller.state.submitted

    controller.acknowledge_submission(run, record)

    assert run.run_id not in controller.state.discovered
    assert controller.state.submitted[run.run_id].galaxy_job_id == "galaxy-job-1"
    assert service.uploaded[-1][2] == "POWGEN watch state IPTS-38000"


def test_legacy_checkpoint_triggers_missing_initial_backfill() -> None:
    service = FakeGalaxyService()
    settings = PowgenExperimentSettings(
        ipts="IPTS-38000",
        history_id="history-1",
        configuration_dataset_id="config-hda",
        wavelength_angstrom="1.5",
    )
    original = PowgenWatchController(service, settings)
    original.discover(
        [{"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764.gsa"}]
    )
    legacy_checkpoint = service.uploaded[-1][0]
    legacy_checkpoint.pop("initial_backfill_complete", None)
    service.history_documents = [{"id": "legacy-checkpoint", "payload": legacy_checkpoint}]

    restored = PowgenWatchController(service, settings)
    assert restored.restore_latest_state() is True
    missing = restored.discover(
        [
            {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63762.gsa"},
            {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63763.gsa"},
            {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764.gsa"},
        ]
    )

    assert [run.run_number for run in missing] == [63762, 63763]
    assert restored.state.initial_backfill_complete is True


def test_controller_marks_finished_job_complete_without_downloading_archive() -> None:
    service = FakeGalaxyService()
    controller = PowgenWatchController(
        service,
        PowgenExperimentSettings(
            ipts="IPTS-38000",
            history_id="history-1",
            configuration_dataset_id="config-hda",
            wavelength_angstrom="1.5",
        ),
    )
    run = controller.discover(
        [{"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63765.gsa"}]
    )[0]
    controller.submit(run)

    records = controller.refresh()

    assert records[run.run_id].status == RunStatus.OK
    assert run.run_id in controller.state.completed
    assert run.run_id not in controller.state.submitted
    assert controller.state.completed[run.run_id].galaxy_result_ids == ("archive-hda",)


def test_controller_persists_small_scientific_summary_without_archive_download() -> None:
    class SummaryService(FakeGalaxyService):
        def refresh(self, record):
            record = super().refresh(record)
            record.output_dataset_ids["summary"] = "summary-hda"
            return record

        def load_json_document(self, dataset_id):
            if dataset_id == "summary-hda":
                return {
                    "status": "complete",
                    "analysis_mode": "rapid",
                    "summary": {"live_run": {"timings": {"total_seconds": 90}}},
                    "hypotheses": [
                        {
                            "gsas_rwp_rank": "1",
                            "rwp": "9.5",
                            "status": "ok",
                            "weights_json": '{"YFeSi (SG 62)": 75, "Ga (SG 225)": 25}',
                        }
                    ],
                }
            return super().load_json_document(dataset_id)

    service = SummaryService()
    controller = PowgenWatchController(
        service,
        PowgenExperimentSettings(
            ipts="IPTS-38000",
            history_id="history-1",
            configuration_dataset_id="config-hda",
            wavelength_angstrom="1.5",
        ),
    )
    run = controller.discover(
        [{"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63765.gsa"}]
    )[0]
    controller.submit(run)

    controller.refresh()

    summary = controller.state.completed[run.run_id].scientific_summary
    assert summary["rwp"] == 9.5
    assert summary["phases"][0]["phase"] == "YFeSi"
    assert summary["elapsed_seconds"] == 90.0


def test_controller_restores_galaxy_checkpoint_without_resubmitting_old_scan() -> None:
    service = FakeGalaxyService()
    settings = PowgenExperimentSettings(
        ipts="IPTS-38000",
        history_id="history-1",
        configuration_dataset_id="config-hda",
        wavelength_angstrom="1.5",
    )
    original = PowgenWatchController(service, settings)
    old_run = original.discover(
        [{"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764.gsa"}]
    )[0]
    old_record = original.submit(old_run)
    checkpoint = service.uploaded[-1][0]
    service.history_documents = [{"id": "checkpoint-1", "payload": checkpoint}]
    service.recovered_runs = [old_record]

    restarted = PowgenWatchController(service, settings)
    assert restarted.restore_latest_state() is True
    discovered = restarted.discover(
        [
            {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764.gsa"},
            {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63765.gsa"},
        ]
    )

    assert [run.run_id for run in discovered] == ["PG3_63765"]
    assert restarted.records["PG3_63764"].galaxy_job_id == "galaxy-job-1"


def test_controller_recovers_acknowledged_job_directly_without_recent_history_scan() -> None:
    service = FakeGalaxyService()
    settings = PowgenExperimentSettings(
        ipts="IPTS-38000",
        history_id="history-1",
        configuration_dataset_id="config-hda",
        wavelength_angstrom="1.5",
    )
    original = PowgenWatchController(service, settings)
    run = original.discover(
        [{"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764.gsa"}]
    )[0]
    original.submit(run)
    service.history_documents = [{"id": "checkpoint-1", "payload": service.uploaded[-1][0]}]
    service.recovered_runs = []

    restarted = PowgenWatchController(service, settings)
    assert restarted.restore_latest_state() is True
    assert restarted.records[run.run_id].galaxy_job_id == "galaxy-job-1"

    restarted.refresh()

    assert run.run_id in restarted.state.completed
    assert run.run_id not in restarted.state.submitted


def test_controller_preserves_pending_retry_across_restart() -> None:
    service = FakeGalaxyService()
    settings = PowgenExperimentSettings(
        ipts="IPTS-38000",
        history_id="history-1",
        configuration_dataset_id="config-hda",
        wavelength_angstrom="1.5",
        retry_base_seconds=0,
        retry_max_seconds=0,
    )
    original = PowgenWatchController(service, settings)
    run = original.discover(
        [{"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764.gsa"}]
    )[0]
    original.defer_submission(run, "temporary failure")
    service.history_documents = [{"id": "checkpoint-1", "payload": service.uploaded[-1][0]}]

    restarted = PowgenWatchController(service, settings)
    assert restarted.restore_latest_state() is True

    assert [item.run_id for item in restarted.due_submissions()] == [run.run_id]
    assert restarted.state.discovered[run.run_id].submission_attempts == 1


def test_controller_detects_late_lower_numbered_reduction_inside_window() -> None:
    service = FakeGalaxyService()
    controller = PowgenWatchController(
        service,
        PowgenExperimentSettings(
            ipts="IPTS-38000",
            history_id="history-1",
            configuration_dataset_id="config-hda",
            wavelength_angstrom="1.5",
            late_arrival_window=10,
        ),
    )
    first = controller.discover(
        [{"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63766.gsa"}]
    )
    late = controller.discover(
        [
            {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63765.gsa"},
            {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63766.gsa"},
        ]
    )

    assert [run.run_number for run in first] == [63766]
    assert [run.run_number for run in late] == [63765]


def test_controller_refresh_failure_for_one_job_does_not_block_another() -> None:
    class PartiallyFailingService(FakeGalaxyService):
        def refresh(self, record):
            if record.galaxy_job_id == "bad-job":
                raise RuntimeError("temporary status endpoint failure")
            return super().refresh(record)

    service = PartiallyFailingService()
    controller = PowgenWatchController(
        service,
        PowgenExperimentSettings(
            ipts="IPTS-38000",
            history_id="history-1",
            configuration_dataset_id="config-hda",
            wavelength_angstrom="1.5",
        ),
    )
    runs = controller.discover(
        [
            {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764.gsa"},
            {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63765.gsa"},
        ]
    )
    by_id = {run.run_id: run for run in runs}
    controller.state.mark_submitted(by_id["PG3_63764"], "bad-job")
    controller.state.mark_submitted(by_id["PG3_63765"], "good-job")

    controller.refresh()

    assert "PG3_63764" in controller.state.submitted
    assert controller.refresh_errors == {"PG3_63764": "temporary status endpoint failure"}
    assert "PG3_63765" in controller.state.completed


def _recovered_record(
    *,
    run_number: int = 63764,
    status: RunStatus = RunStatus.OK,
    database_dataset_id: str | None = "custom-library-hda",
) -> RunRecord:
    return RunRecord(
        uid=f"job-{run_number}",
        galaxy_job_id=f"job-{run_number}",
        name=f"IPTS-38000_PG3_{run_number}",
        mode=AnalysisMode.RAPID,
        history_id="history-1",
        status=status,
        analysis_status=status,
        inputs=InputSelection(
            data_dataset_id=f"data-{run_number}",
            instrument_dataset_id="instrument-hda",
            database_dataset_id=database_dataset_id,
        ),
        output_dataset_ids=(
            {"summary": f"summary-{run_number}", "results_archive": f"archive-{run_number}"}
            if status == RunStatus.OK
            else {}
        ),
    )


def test_controller_reconciles_completed_galaxy_job_missing_from_checkpoint() -> None:
    service = FakeGalaxyService()
    controller = PowgenWatchController(
        service,
        PowgenExperimentSettings(
            ipts="IPTS-38000",
            history_id="history-1",
            configuration_dataset_id="config-hda",
            database_dataset_id="custom-library-hda",
            wavelength_angstrom="1.5",
        ),
    )
    run = controller.discover(
        [{"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764.gsa"}]
    )[0]
    record = _recovered_record()
    service.recovered_runs = [record]
    service.json_documents["summary-63764"] = {
        "status": "complete",
        "analysis_mode": "rapid",
        "hypotheses": [{"gsas_rwp_rank": "1", "rwp": "8.2", "status": "ok"}],
    }

    recovered = controller.reconcile_recent_runs()

    assert recovered == 1
    assert run.run_id in controller.state.completed
    assert run.run_id not in controller.state.discovered
    assert controller.state.completed[run.run_id].scientific_summary["rwp"] == 8.2
    assert run.run_id in service.uploaded[-1][0]["completed"]
    assert service.snapshots == []


def test_controller_reconciles_previously_submitted_job_that_finished() -> None:
    service = FakeGalaxyService()
    controller = PowgenWatchController(
        service,
        PowgenExperimentSettings(
            ipts="IPTS-38000",
            history_id="history-1",
            configuration_dataset_id="config-hda",
            database_dataset_id="custom-library-hda",
            wavelength_angstrom="1.5",
        ),
    )
    run = controller.discover(
        [{"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764.gsa"}]
    )[0]
    controller.state.mark_submitted(run, "job-63764")
    service.recovered_runs = [_recovered_record()]
    service.json_documents["summary-63764"] = {
        "status": "complete",
        "analysis_mode": "rapid",
        "hypotheses": [{"gsas_rwp_rank": "1", "rwp": "8.2", "status": "ok"}],
    }

    assert controller.reconcile_recent_runs() == 1
    assert run.run_id in controller.state.completed
    assert run.run_id in service.uploaded[-1][0]["completed"]


def test_controller_reconciles_queued_job_without_resubmitting() -> None:
    service = FakeGalaxyService()
    controller = PowgenWatchController(
        service,
        PowgenExperimentSettings(
            ipts="IPTS-38000",
            history_id="history-1",
            configuration_dataset_id="config-hda",
            database_dataset_id="custom-library-hda",
            wavelength_angstrom="1.5",
        ),
    )
    run = controller.discover(
        [{"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764.gsa"}]
    )[0]
    service.recovered_runs = [_recovered_record(status=RunStatus.QUEUED)]

    assert controller.reconcile_recent_runs() == 1
    assert controller.state.submitted[run.run_id].galaxy_job_id == "job-63764"
    assert service.snapshots == []


def test_controller_does_not_reconcile_job_using_another_candidate_library() -> None:
    service = FakeGalaxyService()
    controller = PowgenWatchController(
        service,
        PowgenExperimentSettings(
            ipts="IPTS-38000",
            history_id="history-1",
            configuration_dataset_id="config-hda",
            database_dataset_id="expected-library-hda",
            wavelength_angstrom="1.5",
        ),
    )
    run = controller.discover(
        [{"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764.gsa"}]
    )[0]
    service.recovered_runs = [_recovered_record(database_dataset_id="other-library-hda")]

    assert controller.reconcile_recent_runs() == 0
    assert run.run_id in controller.state.discovered


def test_checkpoint_upload_retries_transient_galaxy_failure() -> None:
    class FlakyCheckpointService(FakeGalaxyService):
        def __init__(self) -> None:
            super().__init__()
            self.upload_attempts = 0

        def upload_json_document(self, payload, *, name, label):
            self.upload_attempts += 1
            if self.upload_attempts < 3:
                raise RuntimeError("temporary Galaxy upload failure")
            return super().upload_json_document(payload, name=name, label=label)

    service = FlakyCheckpointService()
    controller = PowgenWatchController(
        service,
        PowgenExperimentSettings(
            ipts="IPTS-38000",
            history_id="history-1",
            configuration_dataset_id="config-hda",
            wavelength_angstrom="1.5",
        ),
    )

    controller.discover(
        [{"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764.gsa"}]
    )

    assert service.upload_attempts == 3
    assert service.uploaded[-1][2] == "POWGEN watch state IPTS-38000"
