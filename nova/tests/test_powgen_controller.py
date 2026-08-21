from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from radar_pd_nova.models import AnalysisConfig, AnalysisMode, RunRecord, RunStatus
from radar_pd_nova.powgen_controller import (
    PowgenExperimentSettings,
    PowgenWatchController,
    bounded_directory_listing,
)


class FakeGalaxyService:
    history_id = "history-1"

    def __init__(self) -> None:
        self.snapshots = []
        self.uploaded = []
        self.history_documents = []
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
        return next(row["payload"] for row in self.history_documents if row["id"] == dataset_id)

    def recent_runs(self, *, limit=50):
        return list(self.recovered_runs[:limit])


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


def test_controller_starts_with_latest_gsa_then_submits_all_new_runs() -> None:
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

    assert [run.run_number for run in initial] == [63764]
    assert [run.run_number for run in later] == [63765, 63766]


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
