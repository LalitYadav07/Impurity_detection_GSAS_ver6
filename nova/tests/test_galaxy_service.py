import sys
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import yaml

from radar_pd_nova.galaxy_service import GalaxyService, normalize_status, stage_from_console
from radar_pd_nova.models import AnalysisConfig, AnalysisMode, InputSelection, InputSource, RunRecord, RunStatus


def test_progress_stage_inference() -> None:
    assert stage_from_console("starting lattice nudge", RunStatus.RUNNING) == ("Lattice nudging", 55)
    assert stage_from_console("rerank 512 patterns", RunStatus.RUNNING) == ("Pattern scoring", 72)
    assert stage_from_console("", RunStatus.OK) == ("Results ready", 100)


def test_galaxy_status_normalization() -> None:
    assert normalize_status("queued") is RunStatus.QUEUED
    assert normalize_status("running") is RunStatus.RUNNING
    assert normalize_status("ok") is RunStatus.OK
    assert normalize_status("error") is RunStatus.ERROR
    assert normalize_status("deleted") is RunStatus.CANCELLED


def test_result_collector_uses_nova_collection_api(tmp_path: Path) -> None:
    class FakeData:
        id = "output-id"

        def download(self, target: str) -> None:
            path = Path(target)
            if path.suffix:
                path.write_text("{}", encoding="utf-8")
            else:
                path.mkdir(parents=True, exist_ok=True)
                (path / "member.txt").write_text("ok", encoding="utf-8")

    class FakeOutputs:
        def __init__(self) -> None:
            self.collection_names: list[str] = []

        def get_dataset(self, name: str) -> FakeData:
            if name != "summary":
                raise RuntimeError("not published")
            return FakeData()

        def get_collection(self, name: str) -> FakeData:
            self.collection_names.append(name)
            if name != "plots":
                raise RuntimeError("not published")
            return FakeData()

    outputs = FakeOutputs()

    class FakeTool:
        def get_results(self) -> FakeOutputs:
            return outputs

    service = GalaxyService("https://example.invalid", "key", "history", output_root=tmp_path)
    service._job_details = lambda uid: {}  # type: ignore[method-assign]
    service._recover_tool = lambda uid: FakeTool()  # type: ignore[method-assign]
    record = RunRecord(uid="job-1", name="run-1", mode=AnalysisMode.RAPID, history_id="history")

    result = service.collect_results(record)

    assert outputs.collection_names == ["plots", "tables", "phases", "gpx_projects", "diagnostics"]
    assert result.status is RunStatus.OK, result.message
    assert (Path(result.output_dir) / "plots" / "member.txt").read_text(encoding="utf-8") == "ok"


def test_result_collector_recovers_archive_when_collections_are_unavailable(tmp_path: Path) -> None:
    class ArchiveDataset:
        id = "archive-id"

        def download(self, target: str) -> None:
            with zipfile.ZipFile(target, "w") as handle:
                handle.writestr("ndip/summary.json", '{"analysis_mode":"full","phases":[]}')
                handle.writestr("ndip/tables/Final_phase_fractions.csv", "phase,weight_percent\nFe,100\n")
                handle.writestr(
                    "ndip/plots/final_fit.plotdata.json",
                    '{"plot_kind":"rapid_component_fit_v1","q":[1],"target":[1],"total_fit":[1]}',
                )

    class FakeOutputs:
        def get_dataset(self, name: str) -> ArchiveDataset:
            if name == "results_archive":
                return ArchiveDataset()
            raise RuntimeError("not published")

        def get_collection(self, name: str) -> Any:
            class BrokenCollection:
                id = f"{name}-id"

                def download(self, target: str) -> None:
                    raise OSError("collection object store unavailable")

            return BrokenCollection()

    class FakeTool:
        def get_results(self) -> FakeOutputs:
            return FakeOutputs()

    service = GalaxyService("https://example.invalid", "key", "history", output_root=tmp_path)
    service._job_details = lambda uid: {}  # type: ignore[method-assign]
    service._recover_tool = lambda uid: FakeTool()  # type: ignore[method-assign]
    record = RunRecord(uid="archive-job", name="run", mode=AnalysisMode.FULL, history_id="history")

    result = service.collect_results(record)
    payload = service.result_payload(result)

    assert result.status is RunStatus.OK
    assert "5 duplicate or optional" in result.message
    assert payload["summary"]["analysis_mode"] == "full"
    assert (Path(result.output_dir) / "ndip" / "tables" / "Final_phase_fractions.csv").is_file()


def test_result_collector_uses_durable_dataset_ids_for_recovered_jobs(tmp_path: Path) -> None:
    service = GalaxyService("https://example.invalid", "key", "history", output_root=tmp_path)
    service._job_details = lambda uid: {"outputs": {"results_archive": {"id": "archive-id"}}}  # type: ignore[method-assign]
    service._recover_tool = lambda uid: (_ for _ in ()).throw(RuntimeError("no live tool"))  # type: ignore[method-assign]

    def download(dataset_id: str, target: Path) -> None:
        assert dataset_id == "archive-id"
        with zipfile.ZipFile(target, "w") as handle:
            handle.writestr("ndip/summary.json", '{"analysis_mode":"rapid","phases":[]}')
            handle.writestr("ndip/tables/ranking.csv", "rank,phase\n1,Cu\n")

    service._download_dataset = download  # type: ignore[method-assign]
    record = RunRecord(uid="recovered-job", name="run", mode=AnalysisMode.RAPID, history_id="history")

    result = service.collect_results(record)

    assert result.status is RunStatus.OK, result.message
    assert result.output_dataset_ids["results_archive"] == "archive-id"
    assert (Path(result.output_dir) / "ndip" / "summary.json").is_file()


def test_result_archive_rejects_paths_outside_staging(tmp_path: Path) -> None:
    class UnsafeArchive:
        id = "unsafe-archive"

        def download(self, target: str) -> None:
            with zipfile.ZipFile(target, "w") as handle:
                handle.writestr("../outside.txt", "unsafe")

    class FakeOutputs:
        def get_dataset(self, name: str) -> UnsafeArchive:
            if name == "results_archive":
                return UnsafeArchive()
            raise RuntimeError("not published")

        def get_collection(self, name: str) -> Any:
            raise RuntimeError("not published")

    class FakeTool:
        def get_results(self) -> FakeOutputs:
            return FakeOutputs()

    service = GalaxyService("https://example.invalid", "key", "history", output_root=tmp_path)
    service._job_details = lambda uid: {}  # type: ignore[method-assign]
    service._recover_tool = lambda uid: FakeTool()  # type: ignore[method-assign]
    record = RunRecord(uid="unsafe-job", name="run", mode=AnalysisMode.FULL, history_id="history")

    result = service.collect_results(record)

    assert result.status is RunStatus.ERROR
    assert "Unsafe path" in result.message
    assert not (tmp_path / "outside.txt").exists()


def test_submit_persists_configuration_and_resolved_input_selection(tmp_path: Path, monkeypatch: Any) -> None:
    submitted_parameters: list[tuple[str, Any]] = []

    class FakeParameters:
        def add_input(self, **kwargs: Any) -> None:
            submitted_parameters.append((kwargs["name"], kwargs["value"]))

    class FakeTool:
        def __init__(self, id: str) -> None:
            self.id = id

        def run(self, store: Any, params: Any, *, wait: bool) -> None:
            assert wait is False

        def get_uid(self) -> str:
            return "submitted-job"

    nova_module = ModuleType("nova")
    galaxy_module = ModuleType("nova.galaxy")
    galaxy_module.Parameters = FakeParameters  # type: ignore[attr-defined]
    galaxy_module.Tool = FakeTool  # type: ignore[attr-defined]
    nova_module.galaxy = galaxy_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nova", nova_module)
    monkeypatch.setitem(sys.modules, "nova.galaxy", galaxy_module)

    class FakeDataset:
        def __init__(self, identifier: str) -> None:
            self.id = identifier

    @contextmanager
    def fake_store() -> Iterator[object]:
        yield object()

    service = GalaxyService("https://example.invalid", "key", "history", output_root=tmp_path)
    service._store = fake_store  # type: ignore[method-assign]
    service._upload_dataset = lambda path, store, label: FakeDataset(f"uploaded-{label}")  # type: ignore[method-assign]
    service._dataset_for_input = (  # type: ignore[method-assign]
        lambda *, path, dataset_id, store, label: FakeDataset(dataset_id or f"uploaded-{label}")
    )
    config = AnalysisConfig(run_name="snapshot-run", mode=AnalysisMode.RAPID, sample_elements=["Cu", "S"])
    inputs = InputSelection(
        source=InputSource.UPLOAD,
        data_path="pattern.dat",
        instrument_path="profile.instprm",
        main_cif_path="main.cif",
    )

    record = service.submit(config, inputs)
    config.run_name = "mutated-after-submit"
    inputs.data_path = "different.dat"

    assert record.config is not None
    assert record.config.run_name == "snapshot-run"
    assert record.inputs is not None
    assert record.inputs.data_path == "pattern.dat"
    assert record.inputs.data_dataset_id == "uploaded-diffraction data"
    assert record.inputs.instrument_dataset_id == "uploaded-instrument profile"
    assert record.inputs.main_cif_dataset_id == "uploaded-main phase CIF"
    assert record.input_dataset_ids["configuration"] == "uploaded-configuration"
    submitted_names = {name for name, _ in submitted_parameters}
    assert {
        "measurement|radiation",
        "measurement|instrument_mode",
        "chemistry|sample_elements",
        "chemistry|environment_elements",
        "analysis|strategy|analysis_mode",
        "reproducibility|configuration_override|config_kind",
        "reproducibility|configuration_override|config_file",
        "data_inputs|input_source|source_kind",
        "data_inputs|input_source|diffraction_pattern",
        "data_inputs|input_source|instrument_source|kind",
        "data_inputs|input_source|instrument_source|instrument_file",
        "data_inputs|main_cif",
        "library|database|database_kind",
        "reproducibility|run_name",
    }.issubset(submitted_names)
    assert not any(name.startswith("input_source|") for name in submitted_names)
    assert ("chemistry|sample_elements", "Cu, S") in submitted_parameters
    assert ("data_inputs|input_source|source_kind", "history") in submitted_parameters
    restored = RunRecord.model_validate_json(record.model_dump_json())
    assert restored.config is not None and restored.config.run_name == "snapshot-run"
    assert restored.inputs is not None and restored.inputs.data_dataset_id == "uploaded-diffraction data"


def test_submit_waits_for_async_galaxy_job_identifier(tmp_path: Path, monkeypatch: Any) -> None:
    class FakeParameters:
        def add_input(self, **kwargs: Any) -> None:
            pass

    class FakeStatus:
        state = "queued"
        details: dict[str, str] = {}

    class FakeTool:
        def __init__(self, id: str) -> None:
            self.id = id
            self.uid = ""

        def run(self, store: Any, params: Any, *, wait: bool) -> None:
            assert wait is False

            def acknowledge() -> None:
                time.sleep(0.02)
                self.uid = "delayed-job"

            threading.Thread(target=acknowledge, daemon=True).start()

        def get_uid(self) -> str:
            return self.uid

        def get_full_status(self) -> FakeStatus:
            return FakeStatus()

    nova_module = ModuleType("nova")
    galaxy_module = ModuleType("nova.galaxy")
    galaxy_module.Parameters = FakeParameters  # type: ignore[attr-defined]
    galaxy_module.Tool = FakeTool  # type: ignore[attr-defined]
    nova_module.galaxy = galaxy_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nova", nova_module)
    monkeypatch.setitem(sys.modules, "nova.galaxy", galaxy_module)

    class FakeDataset:
        def __init__(self, identifier: str) -> None:
            self.id = identifier

    @contextmanager
    def fake_store() -> Iterator[object]:
        yield object()

    service = GalaxyService("https://example.invalid", "key", "history", output_root=tmp_path)
    service._store = fake_store  # type: ignore[method-assign]
    service._upload_dataset = lambda path, store, label: FakeDataset(f"uploaded-{label}")  # type: ignore[method-assign]
    service._dataset_for_input = (  # type: ignore[method-assign]
        lambda *, path, dataset_id, store, label: FakeDataset(dataset_id or f"uploaded-{label}")
    )

    record = service.submit(
        AnalysisConfig(run_name="async-run", mode=AnalysisMode.RAPID, sample_elements=["Cu", "S"]),
        InputSelection(source=InputSource.UPLOAD, data_path="pattern.dat", instrument_path="profile.instprm"),
    )

    assert record.uid == "delayed-job"


def test_upload_dataset_targets_store_history_id(tmp_path: Path, monkeypatch: Any) -> None:
    source = tmp_path / "pattern.dat"
    source.write_text("1 2\n", encoding="utf-8")
    upload_calls: list[dict[str, Any]] = []
    waited: list[str] = []

    class FakeDataset:
        def __init__(self, *, name: str, force_upload: bool) -> None:
            self.name = name
            self.force_upload = force_upload
            self.id = ""
            self.store: Any = None

    class FakeTools:
        def upload_file(self, **kwargs: Any) -> dict[str, Any]:
            upload_calls.append(kwargs)
            return {"outputs": [{"id": "uploaded-id"}]}

    class FakeDatasets:
        def wait_for_dataset(self, dataset_id: str) -> None:
            waited.append(dataset_id)

    galaxy_module = ModuleType("nova.galaxy")
    galaxy_module.Dataset = FakeDataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nova.galaxy", galaxy_module)

    galaxy_instance = type("GalaxyInstance", (), {"tools": FakeTools(), "datasets": FakeDatasets()})()
    connection = type("Connection", (), {"galaxy_instance": galaxy_instance})()
    store = type(
        "Store",
        (),
        {
            "name": "RADAR-PD NOVA",
            "history_id": "actual-history-id",
            "nova_connection": connection,
        },
    )()

    dataset = GalaxyService._upload_dataset(str(source), store, "diffraction data")

    assert upload_calls == [
        {
            "path": str(source),
            "history_id": "actual-history-id",
            "file_name": "RADAR-PD diffraction data | pattern.dat",
        }
    ]
    assert waited == ["uploaded-id"]
    assert dataset.id == "uploaded-id"
    assert dataset.store is store


def test_recent_runs_recovers_config_and_inputs_without_command_guessing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    contract = AnalysisConfig(
        run_name="ignored-contract-name",
        mode=AnalysisMode.RAPID,
        sample_elements=["Cu", "S"],
    ).portable_contract()
    details = {
        "id": "recovered-job",
        "state": "running",
        "create_time": "2026-08-10T10:00:00Z",
        "update_time": "2026-08-10T10:01:00Z",
        "command_line": "python runner.py --unrelated full",
        "params": {
            "reproducibility|run_name": "recovered-rapid-run",
            "analysis|strategy|analysis_mode": "full",
            "data_inputs|input_source|source_kind": "history",
            "data_inputs|input_source|diffraction_pattern": {"src": "hda", "id": "data-id"},
            "data_inputs|input_source|instrument_source|kind": "uploaded",
            "data_inputs|input_source|instrument_source|instrument_file": {
                "src": "hda",
                "id": "instrument-id",
            },
            "chemistry|sample_elements": "Cu, S",
        },
        "outputs": {"resolved_config": {"id": "resolved-config-id"}},
    }

    class FakeResponse:
        def __init__(self, payload: Any, *, text: str = "") -> None:
            self.payload = payload
            self.text = text

        def raise_for_status(self) -> None:
            pass

        def json(self) -> Any:
            return self.payload

    def fake_get(url: str, **kwargs: Any) -> FakeResponse:
        if url.endswith("/api/jobs"):
            return FakeResponse([{"id": "recovered-job", "state": "running"}])
        if url.endswith("/api/jobs/recovered-job"):
            return FakeResponse(details)
        if url.endswith("/api/datasets/resolved-config-id/display"):
            return FakeResponse({}, text=yaml.safe_dump(contract))
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("radar_pd_nova.galaxy_service.requests.get", fake_get)
    service = GalaxyService("https://galaxy.example", "key", "history", output_root=tmp_path)

    records = service.recent_runs()

    assert len(records) == 1
    record = records[0]
    assert record.name == "recovered-rapid-run"
    assert record.mode is AnalysisMode.RAPID
    assert record.config is not None and record.config.sample_elements == ["Cu", "S"]
    assert record.inputs is not None
    assert record.inputs.source is InputSource.GALAXY
    assert record.inputs.data_dataset_id == "data-id"
    assert record.inputs.instrument_dataset_id == "instrument-id"
    assert record.output_dataset_ids["resolved_config"] == "resolved-config-id"


def test_recovered_active_run_refreshes_through_galaxy_rest(tmp_path: Path, monkeypatch: Any) -> None:
    details = {
        "id": "active-job",
        "state": "running",
        "stdout": "starting lattice nudge",
        "params": {
            "analysis|strategy|analysis_mode": "rapid",
            "reproducibility|run_name": "active-recovered-run",
            "measurement|radiation": "neutron",
            "chemistry|sample_elements": "Cu S",
            "data_inputs|input_source|source_kind": "history",
            "data_inputs|input_source|diffraction_pattern": {"id": "data-id"},
            "data_inputs|input_source|instrument_source|instrument_file": {"id": "instrument-id"},
        },
    }

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return details

    monkeypatch.setattr("radar_pd_nova.galaxy_service.requests.get", lambda *args, **kwargs: FakeResponse())
    service = GalaxyService("https://galaxy.example", "key", "history", output_root=tmp_path)
    service._recover_tool = lambda uid: (_ for _ in ()).throw(AssertionError("Tool recovery was used"))  # type: ignore[method-assign]
    record = RunRecord(
        uid="active-job",
        name="unknown",
        mode=AnalysisMode.FULL,
        history_id="history",
        status=RunStatus.QUEUED,
    )

    refreshed = service.refresh(record)

    assert refreshed.status is RunStatus.RUNNING
    assert refreshed.stage == "Lattice nudging"
    assert refreshed.progress == 55
    assert refreshed.console_tail == "starting lattice nudge"
    assert refreshed.name == "active-recovered-run"
    assert refreshed.mode is AnalysisMode.RAPID
    assert refreshed.config is not None
    assert refreshed.inputs is not None


def test_result_download_failure_is_an_error_and_preserves_existing_results(tmp_path: Path) -> None:
    class BrokenDataset:
        id = "broken-id"

        def download(self, target: str) -> None:
            raise OSError("object store unavailable")

    class FakeOutputs:
        def get_dataset(self, name: str) -> BrokenDataset:
            if name == "summary":
                return BrokenDataset()
            raise RuntimeError("not published")

        def get_collection(self, name: str) -> Any:
            raise RuntimeError("not published")

    class FakeTool:
        def get_results(self) -> FakeOutputs:
            return FakeOutputs()

    service = GalaxyService("https://example.invalid", "key", "history", output_root=tmp_path)
    service._job_details = lambda uid: {}  # type: ignore[method-assign]
    service._recover_tool = lambda uid: FakeTool()  # type: ignore[method-assign]
    destination = tmp_path / "runs" / "job-failed-download"
    destination.mkdir(parents=True)
    (destination / "previous.json").write_text("valid", encoding="utf-8")
    record = RunRecord(
        uid="job-failed-download",
        name="run",
        mode=AnalysisMode.FULL,
        history_id="history",
        status=RunStatus.OK,
    )

    result = service.collect_results(record)

    assert result.status is RunStatus.ERROR
    assert result.stage == "Result download failed"
    assert "object store unavailable" in result.message
    assert (destination / "previous.json").read_text(encoding="utf-8") == "valid"


def test_missing_galaxy_results_are_an_error(tmp_path: Path) -> None:
    class FakeTool:
        def get_results(self) -> None:
            return None

    service = GalaxyService("https://example.invalid", "key", "history", output_root=tmp_path)
    service._job_details = lambda uid: {}  # type: ignore[method-assign]
    service._recover_tool = lambda uid: FakeTool()  # type: ignore[method-assign]
    record = RunRecord(uid="missing", name="run", mode=AnalysisMode.FULL, history_id="history")

    result = service.collect_results(record)

    assert result.status is RunStatus.ERROR
    assert "did not return" in result.message
