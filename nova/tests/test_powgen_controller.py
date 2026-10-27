from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import pytest

from radar_pd_nova.models import AnalysisConfig, AnalysisMode, InputSelection, RunRecord, RunStatus
from radar_pd_nova.powgen_controller import (
    PowgenExperimentSettings,
    PowgenWatchController,
    bounded_directory_listing,
    preflight_powgen_experiment,
)
from radar_pd_nova import powgen_controller


@pytest.fixture(autouse=True)
def isolated_reduced_metadata(monkeypatch):
    """Controller lifecycle tests use an in-memory source; real parsing/staging
    and multibank submission are exercised in test_powgen_setup.py."""
    from radar_pd_nova.powgen_setup import ReducedSetup, ReducedBank
    import re
    def inspect(path):
        run = int(re.search(r"PG3_(\d+)", str(path))[1])
        return ReducedSetup(run, 1.5, 60, -54,
            (ReducedBank(2, 2, 2, "SLOG", "FXYE", "BANK 2 2 2 SLOG 1 2 .01 0 FXYE"),),
            ("/SNS/PG3/shared/CALIBRATION/2026-2_11A_CAL/PG3_PAC.h5",), (), (),
            "2026-2_11A_CAL", "HR", (10, 1))
    monkeypatch.setattr(powgen_controller, "inspect_reduced_setup", inspect)
    monkeypatch.setattr(PowgenWatchController, "_setup", lambda self, run: inspect(run.source_path))
    monkeypatch.setattr(powgen_controller, "stage_bank", lambda source, setup, bank, profile, dest, **kwargs: (source, profile.path))


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

    def create_submission_snapshot(
        self,
        config,
        inputs,
        *,
        idempotency_token,
        output_profile="full",
        prepared_dataset_ids=None,
    ):
        snapshot = SimpleNamespace(
            config=config,
            inputs=inputs,
            idempotency_token=idempotency_token,
            output_profile=output_profile,
            prepared_dataset_ids=dict(prepared_dataset_ids or {}),
        )
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

    def upload_document(self, path, *, label):
        self.uploaded.append((str(path), Path(path).name, label))
        return f"document-{len(self.uploaded)}"

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


def test_checkpoint_search_skips_other_settings_before_downloading():
    from dataclasses import replace
    service = FakeGalaxyService()
    settings = PowgenExperimentSettings("IPTS-2767", service.history_id, "config-hda", "1.5")
    original = PowgenWatchController(service, settings)
    original.persist_state()
    payload, _, label = service.uploaded[-1]
    other = PowgenWatchController(service, replace(settings, database_dataset_id="other-library"))
    other.persist_state()
    other_payload, _, other_label = service.uploaded[-1]
    rows = [{"id": f"other-{n}", "name": other_label, "payload": other_payload} for n in range(30)]
    rows.append({"id": "matching", "name": label, "payload": payload})
    queries, loaded = [], []

    def search(*, query, limit, **_):
        queries.append(query)
        return [row for row in rows if query in row["name"]][:limit]

    def load(identifier):
        loaded.append(identifier)
        return next(row["payload"] for row in rows if row["id"] == identifier)

    service.search_history_datasets, service.load_json_document = search, load
    restored = PowgenWatchController(service, settings)
    assert restored.restore_latest_state()
    assert loaded == ["matching"] and len(queries) == 1
    assert restored._checkpoint_key() != other._checkpoint_key()


def test_checkpoint_index_never_replaces_payload_validation():
    from copy import deepcopy
    service = FakeGalaxyService()
    settings = PowgenExperimentSettings("IPTS-2767", service.history_id, "config-hda", "1.5")
    first = PowgenWatchController(service, settings)
    first.persist_state()
    payload = deepcopy(service.uploaded[-1][0])
    payload["watch"]["configuration_dataset_id"] = "different-config"
    service.history_documents = [{"id": "wrong", "name": service.uploaded[-1][2], "payload": payload}]
    assert not PowgenWatchController(service, settings).restore_latest_state()


def test_job_reconciliation_supplies_configuration_scope_when_transport_supports_it(monkeypatch):
    seen = []
    class ScopedService(FakeGalaxyService):
        def recent_runs(self, *, limit, configuration_dataset_id=None):
            seen.append((limit, configuration_dataset_id))
            return []
    service = ScopedService()
    controller = PowgenWatchController(service,
        PowgenExperimentSettings("IPTS-38000", service.history_id, "config-hda", "1.5"))
    monkeypatch.setattr(controller, "_read_scan_metadata", lambda _: {})
    controller.discover([{"path": "PG3_63765.gsa"}])
    assert controller.reconcile_recent_runs() == 0
    assert seen == [(50, "config-hda")]


def test_monitor_scientific_summary_resolves_direct_curve_output_without_tables():
    service = FakeGalaxyService()
    descriptor = {"name": "Pattern_observation.json", "sha256": "a" * 64, "source_gpx_sha256": "b" * 64}
    service.json_documents["summary"] = {"status": "complete", "pattern_observation": descriptor}
    seen = []

    def cache(receipt, *, collection_id=None):
        seen.append((receipt, collection_id))
        return {**receipt, "local_path": "/owned-cache/curve.json"}

    service.cache_pattern_observation = cache
    controller = PowgenWatchController(service,
        PowgenExperimentSettings("IPTS-2767", service.history_id, "config-hda", "1.5"))
    record = RunRecord(uid="job", name="run", mode=AnalysisMode.FULL, history_id=service.history_id,
                       output_dataset_ids={"summary": "summary", "pattern_observation": "curve-hda"})
    summary = controller._scientific_summary(record)
    assert seen == [({**descriptor, "dataset_id": "curve-hda"}, None)]
    assert summary["pattern_observation"]["dataset_id"] == "curve-hda"


def test_refresh_can_publish_completed_result_before_checkpoint_upload(monkeypatch):
    service = FakeGalaxyService()
    controller = PowgenWatchController(service,
        PowgenExperimentSettings("IPTS-38000", service.history_id, "config-hda", "1.5"))
    monkeypatch.setattr(controller, "_read_scan_metadata", lambda _: {})
    run = controller.discover([{"path": "PG3_63765.gsa"}])[0]
    controller.submit(run)
    before = len(service.uploaded)
    refreshed = controller.refresh(checkpoint=False)
    assert run.run_id in controller.state.completed
    assert refreshed[run.run_id].status == RunStatus.OK
    assert len(service.uploaded) == before
    controller.persist_state()
    assert len(service.uploaded) == before + 1


def test_bounded_listing_is_nonrecursive_and_ignores_directories(tmp_path: Path) -> None:
    (tmp_path / "PG3_63764.gsa").write_text("data", encoding="ascii")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "PG3_63765.gsa").write_text("data", encoding="ascii")

    rows = bounded_directory_listing(tmp_path)

    assert [Path(row["path"]).name for row in rows] == ["PG3_63764.gsa"]


def test_large_listing_filters_before_stat_and_does_not_truncate(tmp_path, monkeypatch):
    class Entry:
        def __init__(self, name): self.name = name
        def is_file(self, **kwargs):
            assert self.name.endswith(".gsa"), "Unrelated entry was stat'ed"
            return True
        def stat(self, **kwargs): return SimpleNamespace(st_size=10, st_mtime_ns=1)
    class Listing:
        def __enter__(self):
            return iter([*[Entry(f"auxiliary-{n}.png") for n in range(6000)],
                         *[Entry(f"PG3_{n}.gsa") for n in range(6001)]])
        def __exit__(self, *args): pass
    monkeypatch.setattr(powgen_controller, "_scandir", lambda _: Listing())
    rows = bounded_directory_listing(tmp_path)
    assert len(rows) == 6001
    assert Path(rows[-1]["path"]).name == "PG3_6000.gsa"


def test_trial_discovery_does_not_publish_a_watch_checkpoint(monkeypatch):
    service = FakeGalaxyService()
    controller = PowgenWatchController(service,
        PowgenExperimentSettings("IPTS-38000", "history-1", "config-hda", "1.5", initial_backfill_limit=1),
        sample_index=SimpleNamespace(identity=lambda _: {}))
    monkeypatch.setattr(controller, "_read_scan_metadata", lambda _: {})
    rows = controller.discover([{"path": "PG3_1.gsa"}, {"path": "PG3_2.gsa"}], checkpoint=False)
    assert [row.run_number for row in rows] == [2]
    assert not service.uploaded and not service.snapshots


def test_sample_selection_precedes_backfill_and_applies_to_new_scans(monkeypatch):
    service = FakeGalaxyService()
    settings = PowgenExperimentSettings("IPTS-38000", "history-1", "config-hda", "1.5",
        initial_backfill_limit=2, selected_sample_ids=("A",))
    index = SimpleNamespace(identity=lambda run: {"sample_id": "A" if run.run_number % 2 else "B"})
    controller = PowgenWatchController(service, settings, sample_index=index)
    monkeypatch.setattr(controller, "_read_scan_metadata", lambda run: {})
    listing = [{"path": f"PG3_{n}.gsa"} for n in range(1, 7)]
    assert [r.run_number for r in controller.discover(listing)] == [3, 5]
    assert set(controller.state.discovered) == {"PG3_3", "PG3_5"}
    assert [r.run_number for r in controller.discover(listing + [{"path":"PG3_7.gsa"}, {"path":"PG3_8.gsa"}])] == [7]
    assert "PG3_8" not in controller.state.discovered
    assert controller.discover(listing + [{"path":"PG3_7.gsa"}]) == []
    watch = service.uploaded[0][0]["watch"]
    assert watch["selected_sample_ids"] == ["A"]


def test_all_samples_includes_unknown_ids_without_reading_identity(monkeypatch):
    controller = PowgenWatchController(FakeGalaxyService(),
        PowgenExperimentSettings("IPTS-38000", "history-1", "config-hda", "1.5"),
        sample_index=SimpleNamespace(identity=lambda _: pytest.fail("All samples read an identity")))
    monkeypatch.setattr(controller, "_read_scan_metadata", lambda _: {})
    assert len(controller.discover([{"path":"PG3_1.gsa"}, {"path":"PG3_2.gsa"}])) == 2


def test_selected_samples_excludes_unknown_and_retries_later_metadata(monkeypatch):
    metadata = {}
    controller = PowgenWatchController(FakeGalaxyService(),
        PowgenExperimentSettings("IPTS-38000", "history-1", "config-hda", "1.5", selected_sample_ids=("A",)),
        sample_index=SimpleNamespace(identity=lambda run: metadata.get(run.run_number, {})))
    monkeypatch.setattr(controller, "_read_scan_metadata", lambda _: {})
    rows = [{"path":"PG3_1.gsa"}]
    assert controller.discover(rows) == []
    metadata[1] = {"sample_id":"A"}
    assert [r.run_number for r in controller.discover(rows)] == [1]


def test_watch_recovery_requires_the_same_sample_selection(monkeypatch):
    service = FakeGalaxyService()
    settings = PowgenExperimentSettings("IPTS-38000", "history-1", "config-hda", "1.5", selected_sample_ids=("A",))
    first = PowgenWatchController(service, settings, sample_index=SimpleNamespace(identity=lambda _: {"sample_id":"A"}))
    monkeypatch.setattr(first, "_read_scan_metadata", lambda _: {})
    first.discover([{"path":"PG3_1.gsa"}])
    payload = service.uploaded[-1][0]
    service.history_documents = [{"id":"state-hda", "payload":payload}]
    restored = PowgenWatchController(service, settings)
    assert restored.restore_latest_state() is True
    assert set(restored.state.discovered) == {"PG3_1"}
    other = PowgenWatchController(service, PowgenExperimentSettings("IPTS-38000", "history-1", "config-hda", "1.5"))
    assert other.restore_latest_state() is False


def test_preflight_uses_latest_matching_sample_and_reports_total(monkeypatch):
    monkeypatch.setattr(powgen_controller, "bounded_directory_listing", lambda _: [{"path":"PG3_63763.gsa"}, {"path":"PG3_63764.gsa"}])
    calls = []
    def metadata(ipts, run):
        calls.append(run)
        return {"available":True, "sample_id":"A", "wavelength":{"value":1.5,"unit":"A"}}
    monkeypatch.setattr(powgen_controller, "read_powgen_scan_metadata", metadata)
    index = SimpleNamespace(identity=lambda run: {"sample_id":"A" if run.run_number==63763 else "B"})
    result = preflight_powgen_experiment("IPTS-38000", selected_sample_ids=("A",), sample_index=index)
    assert result["scan_count"] == 1 and result["total_scan_count"] == 2
    assert result["latest_run"] == "PG3_63763"
    assert calls == [63763]
    result = preflight_powgen_experiment("IPTS-38000", selected_sample_ids=("missing",), sample_index=index)
    assert result["ready"] is False and "match" in result["message"]


def test_empty_selected_sample_list_cannot_start():
    with pytest.raises(ValueError, match="at least one sample"):
        PowgenExperimentSettings("IPTS-1", "history-1", "config-hda", "1.5", selected_sample_ids=())


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
    assert snapshot.inputs.instrument_path is None
    assert snapshot.inputs.instrument_dataset_id.startswith("document-")
    assert snapshot.output_profile == "monitor"
    assert snapshot.prepared_dataset_ids == {"configuration": "config-hda"}
    assert snapshot.inputs.publish_results_to_ipts is False
    assert service.uploaded[-1][2].startswith("POWGEN watch state IPTS-38000 [watch:")
    assert "PG3_63764" in service.uploaded[-1][0]["submitted"]
    assert service.uploaded[-1][0]["watch"]["source_directory"].endswith("shared/autoreduce")


def test_controller_reuses_one_packaged_profile_dataset_across_scans() -> None:
    service = FakeGalaxyService()
    controller = PowgenWatchController(
        service,
        PowgenExperimentSettings(
            ipts="IPTS-38000",
            history_id="history-1",
            configuration_dataset_id="config-hda",
            wavelength_angstrom="1.5",
            frequency_hz="60",
        ),
    )
    runs = controller.discover(
        [
            {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63763.gsa"},
            {"path": "/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63764.gsa"},
        ]
    )

    records = [controller.launch_submission(run) for run in runs]

    profile_uploads = [item for item in service.uploaded if item[2] == "POWGEN instrument profile"]
    assert len(profile_uploads) == 1
    profile_ids = {record.inputs.instrument_dataset_id for record in records if record.inputs}
    assert len(profile_ids) == 1
    assert next(iter(profile_ids)).startswith("document-")


def test_controller_reuses_selected_history_profile_without_packaged_resolution() -> None:
    service = FakeGalaxyService()
    controller = PowgenWatchController(
        service,
        PowgenExperimentSettings(
            ipts="IPTS-12000",
            history_id="history-1",
            configuration_dataset_id="config-hda",
            wavelength_angstrom="1.5",
            instrument_profile_dataset_id="legacy-profile-hda",
            instrument_profile_name="2011A_bank2.instprm",
        ),
    )
    run = controller.discover(
        [{"path": "/SNS/PG3/IPTS-12000/shared/autoreduce/PG3_10000.gsa"}]
    )[0]

    record = controller.launch_submission(run)

    assert record.inputs is not None
    assert record.inputs.instrument_dataset_id == "legacy-profile-hda"
    assert controller.definition.instrument_profile_ref == "legacy-profile-hda"
    assert not [item for item in service.uploaded if item[2] == "POWGEN instrument profile"]
    checkpoint = next(item[0] for item in service.uploaded if item[2].startswith("POWGEN watch state IPTS-12000 [watch:"))
    assert checkpoint["watch"]["instrument_profile_dataset_id"] == "legacy-profile-hda"


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

    result = preflight_powgen_experiment("IPTS-38000")

    assert result["ready"] is True
    assert result["scan_count"] == 2
    assert result["latest_run"] == "PG3_63764"
    assert result["detected_wavelength"] == "1.5"
    assert result["selected_wavelength"] == "1.5"
    assert result["profile_filename"].endswith("60HzB2_CWL1p5.instprm")


def test_preflight_accepts_user_profile_when_cycle_is_not_packaged(monkeypatch) -> None:
    monkeypatch.setattr(
        powgen_controller,
        "bounded_directory_listing",
        lambda _directory: [
            {"path": "/SNS/PG3/IPTS-12000/shared/autoreduce/PG3_10000.gsa"},
        ],
    )
    monkeypatch.setattr(
        powgen_controller,
        "read_powgen_scan_metadata",
        lambda _ipts, _run: {
            "available": True,
            "wavelength": {"value": 1.5, "unit": "A"},
        },
    )
    monkeypatch.setattr(
        powgen_controller,
        "resolve_bank_profile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("packaged lookup was used")),
    )

    result = preflight_powgen_experiment(
        "IPTS-12000",
        instrument_profile_name="2011A_bank2.instprm",
    )

    assert result["ready"] is True
    assert result["profile_filename"] == "2011A_bank2.instprm"
    assert result["profile_rule"] == "user-supplied"
    assert result["profile_source"] == "User-supplied GSAS-II profile"


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
    assert [run.run_number for run in controller.due_submissions(
        reserved_run_ids={runs[1].run_id}
    )] == [63762]
    assert controller.due_submissions(reserved_run_ids={runs[1].run_id, runs[2].run_id}) == []


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
    assert service.uploaded[-1][2].startswith("POWGEN watch state IPTS-38000 [watch:")


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
    assert service.uploaded[-1][2].startswith("POWGEN watch state IPTS-38000 [watch:")
