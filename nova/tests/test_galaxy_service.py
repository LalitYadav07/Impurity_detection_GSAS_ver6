import sys
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import yaml

from radar_pd_nova.galaxy_service import GalaxyService, _upload_filename, normalize_status, stage_from_console
from radar_pd_nova.models import (
    AnalysisConfig,
    AnalysisMode,
    InputSelection,
    InputSource,
    ResultStatus,
    RunRecord,
    RunStatus,
    SubmissionPhase,
)


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


def test_results_export_uses_ndip_authenticated_export_contract() -> None:
    service = GalaxyService("https://example.invalid", "key", "history", output_root=Path("."))
    seen: dict[str, Any] = {}

    def submit_utility(**kwargs: Any) -> Any:
        seen.update(kwargs)
        from radar_pd_nova.models import UtilityActionRecord

        return UtilityActionRecord(
            uid="utility-export",
            tool_id=kwargs["tool_id"],
            name=kwargs["name"],
            associated_run_uid=kwargs["associated_run_uid"],
            galaxy_job_id="export-job",
            status=RunStatus.QUEUED,
        )

    service.submit_utility = submit_utility  # type: ignore[method-assign]
    selection = InputSelection(
        source=InputSource.GALAXY,
        data_dataset_id="data-id",
        instrument_dataset_id="instrument-id",
        facility_root="/HFIR",
        instrument="HB2A",
        ipts="IPTS-28749",
        publish_results_to_ipts=True,
        publish_directory="shared/Lalit_radarpd",
        publish_subfolder="results",
    )
    record = RunRecord(
        uid="run-uid",
        galaxy_job_id="analysis123456",
        name="scan 0003",
        mode=AnalysisMode.RAPID,
        history_id="history",
        inputs=selection,
    )

    action, destination = service.submit_results_export(record, archive_dataset_id="archive-id")

    assert action.galaxy_job_id == "export-job"
    assert seen["tool_id"] == "neutrons_export"
    assert seen["inputs"] == {
        "series_0|input_mode|input_mode_collection": False,
        "series_0|input_mode|input": {"dataset_id": "archive-id"},
        "series_0|input_mode|export_path": destination,
    }
    assert destination == (
        "/HFIR/HB2A/IPTS-28749/shared/Lalit_radarpd/"
        "scan_0003-analysis-results.zip"
    )


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
    assert result.analysis_status is RunStatus.OK
    assert result.result_status is ResultStatus.READY
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
    assert result.result_status is ResultStatus.READY
    assert result.message == ""
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

    assert result.status is RunStatus.OK
    assert result.analysis_status is RunStatus.OK
    assert result.result_status is ResultStatus.ERROR
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

    assert record.uid.startswith("pending-")
    assert record.galaxy_job_id == "delayed-job"


def test_submit_retries_after_transient_galaxy_submission_failure(tmp_path: Path, monkeypatch: Any) -> None:
    """A failed tool-run attempt never creates a Galaxy job, so retrying is safe.

    NDIP interactive-tool pods have occasionally seen the Analyze tool-run POST
    come back as a generic Galaxy legacy-webapp 400 page instead of a normal
    API response, with an identical retry succeeding immediately after.
    """

    monkeypatch.setattr("radar_pd_nova.galaxy_service.time.sleep", lambda _seconds: None)

    class FakeParameters:
        def add_input(self, **kwargs: Any) -> None:
            pass

    class FailingStatus:
        state = "error"
        details = {
            "message": (
                "Unexpected HTTP status code: 400: "
                '<div class="message mt-2 alert alert-info">'
                "Required parameter(s) kwd not provided in request.</div>"
            )
        }

    tool_instances: list["FakeTool"] = []

    class FakeTool:
        def __init__(self, id: str) -> None:
            self.id = id
            self.attempt_index = len(tool_instances)
            tool_instances.append(self)

        def run(self, store: Any, params: Any, *, wait: bool) -> None:
            assert wait is False

        def get_uid(self) -> str:
            return "" if self.attempt_index == 0 else "recovered-after-retry"

        def get_full_status(self) -> Any:
            return FailingStatus() if self.attempt_index == 0 else None

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
        AnalysisConfig(run_name="retry-run", mode=AnalysisMode.RAPID, sample_elements=["Cu", "S"]),
        InputSelection(source=InputSource.UPLOAD, data_path="pattern.dat", instrument_path="profile.instprm"),
    )

    assert record.galaxy_job_id == "recovered-after-retry"
    assert len(tool_instances) == 2, "expected exactly one retry after the transient failure"


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


def test_laptop_temporary_uploads_receive_scientific_filename_suffixes(tmp_path: Path) -> None:
    """Browser uploads expose extensionless temp paths, but Galaxy HDAs must not."""

    temporary = tmp_path / "tmpkc3r80cb"
    expected = {
        "diffraction data": "RADAR-PD diffraction data | tmpkc3r80cb.dat",
        "instrument profile": "RADAR-PD instrument profile | tmpkc3r80cb.instprm",
        "main phase CIF": "RADAR-PD main phase CIF | tmpkc3r80cb.cif",
        "candidate library": "RADAR-PD candidate library | tmpkc3r80cb.zip",
        "NeXus event file": "RADAR-PD NeXus event file | tmpkc3r80cb.nxs",
    }

    for label, filename in expected.items():
        assert _upload_filename(temporary, label) == filename


def test_server_selected_upload_keeps_supported_original_filename(tmp_path: Path) -> None:
    source = tmp_path / "TbSSL.CIF"

    assert _upload_filename(source, "main phase CIF") == "RADAR-PD main phase CIF | TbSSL.CIF"


def test_gsas_diffraction_uploads_keep_original_filename_suffix(tmp_path: Path) -> None:
    for suffix in (".gsa", ".gsas", ".gss"):
        source = tmp_path / f"pattern{suffix}"

        assert _upload_filename(source, "diffraction data") == f"RADAR-PD diffraction data | pattern{suffix}"


def test_gsas_diffraction_datasets_are_classified_from_filename_or_datatype() -> None:
    for extension in ("gsa", "gsas", "gss"):
        assert GalaxyService._dataset_scientific_role(
            {"name": f"RADAR-PD diffraction data | pattern.{extension}", "extension": "txt"}
        ) == ("diffraction", False)
        assert GalaxyService._dataset_scientific_role(
            {"name": "pattern", "extension": extension}
        ) == ("diffraction", False)


def test_remote_sources_and_directory_entries_are_normalized(monkeypatch: Any) -> None:
    responses = [
        [
            {
                "id": "sns",
                "label": "SNS experiment files",
                "uri_root": "gxfiles://sns/",
                "browsable": True,
                "writable": False,
            }
        ],
        [
            {"class": "Directory", "name": "HB2A", "uri": "gxfiles://sns/HB2A"},
            {"class": "File", "name": "scan.dat", "uri": "gxfiles://sns/scan.dat"},
            {"class": "File", "name": "scan.gsa", "uri": "gxfiles://sns/scan.gsa"},
            {"class": "File", "name": "scan.gsas", "uri": "gxfiles://sns/scan.gsas"},
            {"class": "File", "name": "scan.gss", "uri": "gxfiles://sns/scan.gss"},
            {"class": "File", "name": "notes.pdf", "uri": "gxfiles://sns/notes.pdf"},
        ],
    ]

    class FakeResponse:
        def __init__(self, payload: Any) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            pass

        def json(self) -> Any:
            return self.payload

    monkeypatch.setattr(
        "radar_pd_nova.galaxy_service.requests.get",
        lambda *args, **kwargs: FakeResponse(responses.pop(0)),
    )
    service = GalaxyService("https://galaxy.example", "key", "history")

    sources = service.list_remote_file_sources()
    entries = service.list_remote_files("gxfiles://sns/", role="data")

    assert sources == [
        {
            "id": "sns",
            "title": "SNS experiment files",
            "value": "gxfiles://sns/",
            "description": "",
            "writable": False,
        }
    ]
    assert [entry["title"] for entry in entries] == [
        "HB2A",
        "scan.dat",
        "scan.gsa",
        "scan.gsas",
        "scan.gss",
    ]
    assert service.remote_parent_uri("gxfiles://sns/HB2A/IPTS-123/shared/", "gxfiles://sns/") == (
        "gxfiles://sns/HB2A/IPTS-123/"
    )
    assert service.remote_parent_uri("gxfiles://other/private/", "gxfiles://sns/") == "gxfiles://sns/"


def test_remote_file_import_uses_galaxy_fetch_and_waits_for_dataset(monkeypatch: Any) -> None:
    posted: list[dict[str, Any]] = []

    class FakeResponse:
        def __init__(self, payload: Any) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            pass

        def json(self) -> Any:
            return self.payload

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        posted.append({"url": url, **kwargs})
        return FakeResponse({"outputs": [{"id": "imported-data"}]})

    monkeypatch.setattr("radar_pd_nova.galaxy_service.requests.post", fake_post)
    monkeypatch.setattr(
        "radar_pd_nova.galaxy_service.requests.get",
        lambda *args, **kwargs: FakeResponse({"state": "ok"}),
    )
    service = GalaxyService("https://galaxy.example", "key", "history")

    dataset_id = service._import_remote_dataset("gxfiles://sns/HB2A/IPTS-123/shared/scan.dat", "diffraction data")

    assert dataset_id == "imported-data"
    assert posted[0]["url"].endswith("/api/tools/fetch")
    assert posted[0]["json"]["history_id"] == "history"
    item = posted[0]["json"]["targets"][0]["items"][0]
    assert item["url"] == "gxfiles://sns/HB2A/IPTS-123/shared/scan.dat"
    assert item["name"].endswith("scan.dat")


def test_history_dataset_listing_paginates_before_filtering(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []
    pages = {
        0: [
            {"id": "new-output", "name": "result", "extension": "json", "state": "ok"},
            {"id": "data", "name": "pattern.dat", "extension": "dat", "state": "ok"},
        ],
        2: [
            {"id": "cif", "name": "TbSSL.cif", "extension": "cif", "state": "ok"},
        ],
    }

    class FakeResponse:
        def __init__(self, payload: list[dict[str, Any]]) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            pass

        def json(self) -> list[dict[str, Any]]:
            return self.payload

    def fake_get(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse(pages[int(kwargs["params"]["offset"])])

    monkeypatch.setattr("radar_pd_nova.galaxy_service.requests.get", fake_get)
    service = GalaxyService("https://galaxy.example", "key", "history")

    datasets = service.list_history_datasets(limit=4, page_size=2)

    assert [item["id"] for item in datasets] == ["new-output", "data", "cif"]
    assert [call["params"]["offset"] for call in calls] == [0, 2]


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
            "data_inputs|input_source|diffraction_pattern": {"src": "hda", "id": 292227},
            "data_inputs|input_source|instrument_source|kind": "uploaded",
            "data_inputs|input_source|instrument_source|instrument_file": {
                "src": "hda",
                "id": 292229,
            },
            "data_inputs|main_cif": {"src": "hda", "id": 292232},
            "chemistry|sample_elements": "Cu, S",
        },
        "inputs": {
            "data_inputs|input_source|diffraction_pattern": {"src": "hda", "id": "encoded-data-id"},
            "data_inputs|input_source|instrument_source|instrument_file": {
                "src": "hda",
                "id": "encoded-instrument-id",
            },
            "data_inputs|main_cif": {"src": "hda", "id": "encoded-cif-id"},
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
    assert record.inputs.data_dataset_id == "encoded-data-id"
    assert record.inputs.instrument_dataset_id == "encoded-instrument-id"
    assert record.inputs.main_cif_dataset_id == "encoded-cif-id"
    assert record.input_dataset_ids["data"] == "encoded-data-id"
    assert record.input_dataset_ids["instrument"] == "encoded-instrument-id"
    assert record.input_dataset_ids["main_cif"] == "encoded-cif-id"
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


def test_recovered_inputs_restore_persisted_ipts_export_selection(tmp_path: Path) -> None:
    service = GalaxyService("https://galaxy.example", "key", "history", output_root=tmp_path)
    service._dataset_document = lambda dataset_id: {  # type: ignore[method-assign]
        "$schema": "radar-pd-config/v1",
        "ndip_delivery": {
            "$schema": "radar-pd-delivery/v1",
            "facility_root": "/HFIR",
            "instrument": "HB2A",
            "ipts": "IPTS-28749",
            "data_relative_path": "shared/Lalit_radarpd/Data/HB2A.dat",
            "publish_results_to_ipts": True,
            "publish_directory": "shared/Lalit_radarpd",
            "publish_subfolder": "radar-pd-results",
        },
    }
    parameters = {
        "reproducibility|configuration_override|config_file": {"id": "config-id"},
        "data_inputs|input_source|source_kind": "history",
        "data_inputs|input_source|diffraction_pattern": {"id": "data-id"},
        "data_inputs|input_source|instrument_source|kind": "uploaded",
        "data_inputs|input_source|instrument_source|instrument_file": {"id": "instrument-id"},
    }

    inputs = service._inputs_from_parameters(parameters)

    assert inputs is not None
    assert inputs.source is InputSource.GALAXY
    assert inputs.facility_root == "/HFIR"
    assert inputs.instrument == "HB2A"
    assert inputs.ipts == "IPTS-28749"
    assert inputs.publish_results_to_ipts is True
    assert inputs.publish_directory == "shared/Lalit_radarpd"
    assert inputs.publish_subfolder == "radar-pd-results"


def test_refresh_preserves_click_time_ipts_export_selection(tmp_path: Path) -> None:
    service = GalaxyService("https://galaxy.example", "key", "history", output_root=tmp_path)
    service._job_details = lambda uid: {  # type: ignore[method-assign]
        "id": uid,
        "state": "running",
        "params": {
            "data_inputs|input_source|source_kind": "history",
            "data_inputs|input_source|diffraction_pattern": {"id": "recovered-data"},
            "data_inputs|input_source|instrument_source|instrument_file": {"id": "recovered-instrument"},
        },
    }
    original = InputSelection(
        source=InputSource.IPTS_BROWSER,
        data_dataset_id="submitted-data",
        instrument_dataset_id="submitted-instrument",
        facility_root="/HFIR",
        instrument="HB2A",
        ipts="IPTS-28749",
        data_relative_path="shared/Lalit_radarpd/Data/HB2A.dat",
        publish_results_to_ipts=True,
        publish_directory="shared/Lalit_radarpd",
        publish_subfolder="radar-pd-results",
    )
    record = RunRecord(
        uid="active-job",
        galaxy_job_id="active-job",
        name="run",
        mode=AnalysisMode.RAPID,
        history_id="history",
        inputs=original,
    )

    refreshed = service.refresh(record)

    assert refreshed.inputs is not None
    assert refreshed.inputs.source is InputSource.IPTS_BROWSER
    assert refreshed.inputs.data_dataset_id == "recovered-data"
    assert refreshed.inputs.instrument_dataset_id == "recovered-instrument"
    assert refreshed.inputs.publish_results_to_ipts is True
    assert refreshed.inputs.publish_directory == "shared/Lalit_radarpd"
    assert refreshed.inputs.publish_subfolder == "radar-pd-results"


def test_newly_submitted_run_uses_authoritative_galaxy_rest_state(tmp_path: Path) -> None:
    class StaleTool:
        def get_status(self) -> str:
            raise AssertionError("cached nova-galaxy state was used")

        def get_stdout(self) -> str:
            raise AssertionError("cached nova-galaxy stdout was used")

        def get_stderr(self) -> str:
            raise AssertionError("cached nova-galaxy stderr was used")

    service = GalaxyService("https://galaxy.example", "key", "history", output_root=tmp_path)
    service._tools["submitted-job"] = StaleTool()  # type: ignore[assignment]
    service._job_details = lambda uid: {  # type: ignore[method-assign]
        "id": uid,
        "state": "ok",
        "job_stdout": "Rapid Hypothesis Mode finished successfully",
        "outputs": {"results_archive": {"id": "archive-id"}},
    }
    record = RunRecord(
        uid="submitted-job",
        name="submitted-run",
        mode=AnalysisMode.RAPID,
        history_id="history",
        status=RunStatus.QUEUED,
        stage="Waiting for compute",
        progress=3,
    )

    refreshed = service.refresh(record)

    assert refreshed.status is RunStatus.OK
    assert refreshed.stage == "Results ready"
    assert refreshed.progress == 100
    assert refreshed.console_tail == "Rapid Hypothesis Mode finished successfully"
    assert refreshed.output_dataset_ids["results_archive"] == "archive-id"


def test_refresh_never_regresses_to_stale_cached_state_on_rest_failure(tmp_path: Path) -> None:
    stale_calls: list[str] = []

    class StaleTool:
        def get_status(self) -> str:
            stale_calls.append("status")
            return "queued"

        def get_stdout(self) -> str:
            stale_calls.append("stdout")
            return ""

        def get_stderr(self) -> str:
            stale_calls.append("stderr")
            return ""

    service = GalaxyService("https://galaxy.example", "key", "history", output_root=tmp_path)
    service._tools["completed-job"] = StaleTool()  # type: ignore[assignment]
    service._job_details = (  # type: ignore[method-assign]
        lambda uid: (_ for _ in ()).throw(RuntimeError("temporary Galaxy status failure"))
    )
    record = RunRecord(
        uid="completed-job",
        name="completed-run",
        mode=AnalysisMode.RAPID,
        history_id="history",
        status=RunStatus.OK,
        stage="Results ready",
        progress=100,
    )

    try:
        service.refresh(record)
    except RuntimeError as exc:
        assert str(exc) == "temporary Galaxy status failure"
    else:
        raise AssertionError("Galaxy REST failure was unexpectedly ignored")

    assert stale_calls == []
    assert record.status is RunStatus.OK
    assert record.stage == "Results ready"
    assert record.progress == 100


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

    assert result.status is RunStatus.OK
    assert result.analysis_status is RunStatus.OK
    assert result.result_status is ResultStatus.ERROR
    assert result.stage == "Analysis complete; results unavailable"
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

    assert result.status is RunStatus.OK
    assert result.result_status is ResultStatus.ERROR
    assert "did not return" in result.message
