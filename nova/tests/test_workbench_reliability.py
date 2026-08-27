import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from radar_pd_nova.galaxy_service import (
    COMPARE_SERIES_TOOL_ID,
    GPX_HANDOFF_TOOL_ID,
    GSASII_INTERACTIVE_TOOL_ID,
    LIBRARY_BUILDER_TOOL_ID,
    RESULT_EXPLORER_TOOL_ID,
    SNS_RESOLVER_TOOL_ID,
    GalaxyService,
)
from radar_pd_nova.models import (
    AnalysisConfig,
    AnalysisMode,
    CacheManifest,
    InputSelection,
    InputSource,
    ResultStatus,
    RunRecord,
    RunStatus,
    UtilityActionRecord,
)


def _config(mode: AnalysisMode = AnalysisMode.RAPID) -> AnalysisConfig:
    return AnalysisConfig(run_name="atomic-run", mode=mode, sample_elements=["Tb", "O"])


def _inputs() -> InputSelection:
    return InputSelection(
        source=InputSource.UPLOAD,
        data_path="pattern.dat",
        instrument_path="profile.instprm",
        main_cif_path="main.cif",
    )


def test_submission_snapshot_detaches_button_time_mode_and_inputs(tmp_path: Path) -> None:
    service = GalaxyService("https://example.invalid", "key", "history", output_root=tmp_path)
    config = _config(AnalysisMode.FULL)
    inputs = _inputs()
    snapshot = service.create_submission_snapshot(
        config,
        inputs,
        client_revision=17,
        idempotency_token="same-click-token",
    )

    config.mode = AnalysisMode.RAPID
    inputs.data_path = "changed-after-click.dat"

    assert snapshot.config.mode is AnalysisMode.FULL
    assert snapshot.inputs.data_path == "pattern.dat"
    assert snapshot.client_revision == 17
    assert snapshot.display_summary["mode"] == "full"


def test_idempotency_token_returns_one_pending_record(tmp_path: Path) -> None:
    service = GalaxyService("https://example.invalid", "key", "history", output_root=tmp_path)
    snapshot = service.create_submission_snapshot(_config(), _inputs(), idempotency_token="repeat-click-token")

    first = service.pending_record(snapshot)
    second = service.pending_record(snapshot)

    assert first is second
    assert first.uid.startswith("pending-")
    assert first.galaxy_job_id is None


def test_partial_uploads_are_reused_on_retry(tmp_path: Path) -> None:
    service = GalaxyService("https://example.invalid", "key", "history", output_root=tmp_path)
    snapshot = service.create_submission_snapshot(_config(), _inputs(), idempotency_token="partial-upload-token")
    record = service.pending_record(snapshot)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("$schema: radar-pd-config/v1\n", encoding="utf-8")
    calls: dict[str, int] = {}

    def upload(key: str, path: str, label: str) -> tuple[str, str]:
        calls[key] = calls.get(key, 0) + 1
        if key == "main_cif" and calls[key] == 1:
            raise OSError("temporary object store failure")
        return key, f"dataset-{key}"

    service._upload_one = upload  # type: ignore[method-assign]

    with pytest.raises(OSError):
        service._prepare_datasets(snapshot, record, config_path, None)

    retained = dict(record.prepared_dataset_ids)
    assert retained
    prepared = service._prepare_datasets(snapshot, record, config_path, None)

    assert prepared == {
        "configuration": "dataset-configuration",
        "data": "dataset-data",
        "instrument": "dataset-instrument",
        "main_cif": "dataset-main_cif",
    }
    for key in retained:
        assert calls[key] == 1, f"{key} should not be uploaded twice"
    assert calls["main_cif"] == 2


def test_acknowledgement_recovery_prevents_second_tool_run(tmp_path: Path, monkeypatch: Any) -> None:
    instances: list[Any] = []

    class Tool:
        def __init__(self, id: str) -> None:
            self.id = id
            self.assigned = ""
            instances.append(self)

        def run(self, store: Any, params: Any, *, wait: bool) -> None:
            pass

        def get_uid(self) -> str:
            return ""

        def get_full_status(self) -> Any:
            return type("Status", (), {"state": "error", "details": {"message": "lost acknowledgement"}})()

        def assign_id(self, uid: str, store: Any) -> None:
            self.assigned = uid

    nova = ModuleType("nova")
    galaxy = ModuleType("nova.galaxy")
    galaxy.Tool = Tool  # type: ignore[attr-defined]
    nova.galaxy = galaxy  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nova", nova)
    monkeypatch.setitem(sys.modules, "nova.galaxy", galaxy)
    service = GalaxyService("https://example.invalid", "key", "history", output_root=tmp_path)
    service._find_acknowledged_job = lambda run_name, config_id: "existing-job"  # type: ignore[method-assign]

    uid, recovered = service._submit_with_retry(
        object(),
        object(),
        run_name="atomic-run",
        config_dataset_id="config-id",
    )

    assert uid == "existing-job"
    assert recovered.assigned == "existing-job"
    assert len(instances) == 2  # failed client Tool + recovered assigned Tool, no second run attempt


def test_analysis_and_result_states_are_independent() -> None:
    record = RunRecord(
        uid="job",
        galaxy_job_id="job",
        name="run",
        mode=AnalysisMode.FULL,
        history_id="history",
        status=RunStatus.OK,
        analysis_status=RunStatus.OK,
        result_status=ResultStatus.ERROR,
        message="archive temporarily unavailable",
    )

    restored = RunRecord.model_validate_json(record.model_dump_json())

    assert restored.analysis_status is RunStatus.OK
    assert restored.status is RunStatus.OK
    assert restored.result_status is ResultStatus.ERROR


def test_cache_manifest_invalidates_changed_archive(tmp_path: Path) -> None:
    destination = tmp_path / "run"
    destination.mkdir()
    (destination / "fit.plotdata.json").write_text("{}", encoding="utf-8")
    cached = CacheManifest(job_id="job", archive_dataset_id="archive", archive_size=10, archive_update_time="old")
    current = cached.model_copy(update={"archive_size": 11, "archive_update_time": "new"})

    assert GalaxyService._cache_is_current(cached, cached, destination)
    assert not GalaxyService._cache_is_current(cached, current, destination)


def test_remote_history_search_uses_dev_pagination_and_query(tmp_path: Path, monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> list[dict[str, Any]]:
            return [
                {
                    "id": "data-id",
                    "name": "HB2A_TbSSL.dat",
                    "extension": "dat",
                    "state": "ok",
                    "history_content_type": "dataset",
                },
                {
                    "id": "output-id",
                    "name": "RADAR-PD summary",
                    "extension": "json",
                    "state": "ok",
                    "history_content_type": "dataset",
                },
            ]

    def get(url: str, **kwargs: Any) -> Response:
        captured.update(kwargs["params"])
        return Response()

    monkeypatch.setattr("radar_pd_nova.galaxy_service.requests.get", get)
    service = GalaxyService("https://example.invalid", "key", "history", output_root=tmp_path)

    rows = service.search_history_datasets(query="TbSSL", limit=100, offset=200)

    assert captured == {"v": "dev", "limit": 100, "offset": 200, "order": "update_time-dsc", "q": "name-contains", "qv": "TbSSL"}
    assert [row["id"] for row in rows] == ["data-id"]
    assert rows[0]["role"] == "diffraction"


def test_history_role_uses_preserved_filename_when_galaxy_sniffs_text(tmp_path: Path, monkeypatch: Any) -> None:
    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> list[dict[str, Any]]:
            return [
                {
                    "id": "instrument-dataset",
                    "name": "RADAR-PD instrument profile | hb2a_si_ge113.instprm",
                    "extension": "txt",
                    "state": "ok",
                    "update_time": "2026-08-15T20:51:00",
                    "history_content_type": "dataset",
                }
            ]

    monkeypatch.setattr("radar_pd_nova.galaxy_service.requests.get", lambda *args, **kwargs: Response())
    service = GalaxyService("https://example.invalid", "key", "history", output_root=tmp_path)

    rows = service.search_history_datasets(query="instprm")

    assert rows[0]["role"] == "instrument"
    assert rows[0]["display_name"] == "hb2a_si_ge113.instprm · 2026-08-15 20:51 · instrume"


def test_every_companion_action_serializes_dataset_and_collection_references(
    tmp_path: Path, monkeypatch: Any
) -> None:
    submissions: list[tuple[str, dict[str, Any]]] = []

    class Parameters:
        def __init__(self) -> None:
            self.values: dict[str, Any] = {}

        def add_input(self, *, name: str, value: Any) -> None:
            self.values[name] = value

    class Dataset:
        def __init__(self, path: str = "", name: str | None = None, **_: Any) -> None:
            self.name = name
            self.id = ""
            self.store: Any = None

    class DatasetCollection(Dataset):
        pass

    class Tool:
        def __init__(self, id: str) -> None:
            self.id = id
            self.parameters: Parameters | None = None

        def run(self, store: Any, params: Parameters, *, wait: bool) -> None:
            assert wait is False
            self.parameters = params
            submissions.append((self.id, params.values))

        def get_uid(self) -> str:
            return f"job-{len(submissions)}"

    nova = ModuleType("nova")
    galaxy = ModuleType("nova.galaxy")
    galaxy.Parameters = Parameters  # type: ignore[attr-defined]
    galaxy.Tool = Tool  # type: ignore[attr-defined]
    galaxy.Dataset = Dataset  # type: ignore[attr-defined]
    galaxy.DatasetCollection = DatasetCollection  # type: ignore[attr-defined]
    nova.galaxy = galaxy  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nova", nova)
    monkeypatch.setitem(sys.modules, "nova.galaxy", galaxy)

    store = object()

    @contextmanager
    def fake_store():
        yield store

    service = GalaxyService("https://example.invalid", "key", "history", output_root=tmp_path)
    service._store = fake_store  # type: ignore[method-assign]
    cases = [
        (
            SNS_RESOLVER_TOOL_ID,
            {"source|source_kind": "event_file", "source|event_file": {"dataset_id": "event"}, "source|bank": 1},
        ),
        (
            LIBRARY_BUILDER_TOOL_ID,
            {
                "cif_source|source_kind": "archives",
                "cif_source|cif_archives": [{"dataset_id": "zip-1"}, {"dataset_id": "zip-2"}],
                "library_mode": "mini",
                "radiation": "neutron",
            },
        ),
        (
            GPX_HANDOFF_TOOL_ID,
            {"gpx_project": {"dataset_id": "gpx"}, "gpx_index": {"dataset_id": "index"}},
        ),
        (COMPARE_SERIES_TOOL_ID, {"summaries": {"collection_id": "summaries"}}),
        (
            RESULT_EXPLORER_TOOL_ID,
            {"result_source|source_kind": "archive", "result_source|results_archive": {"dataset_id": "archive"}},
        ),
    ]

    for tool_id, inputs in cases:
        action = service.submit_utility(tool_id=tool_id, name=tool_id, inputs=inputs, associated_run_uid="run")
        assert action.tool_id == tool_id
        assert action.inputs == inputs
        assert action.associated_run_uid == "run"
        assert action.galaxy_job_id

    assert [tool_id for tool_id, _ in submissions] == [case[0] for case in cases]
    assert submissions[0][1]["source|event_file"] == {"src": "hda", "id": "event"}
    assert submissions[1][1]["cif_source|cif_archives"] == [
        {"src": "hda", "id": "zip-1"},
        {"src": "hda", "id": "zip-2"},
    ]
    assert submissions[2][1]["gpx_index"] == {"src": "hda", "id": "index"}
    assert submissions[3][1]["summaries"] == {"src": "hdca", "id": "summaries"}
    assert submissions[4][1]["result_source|results_archive"] == {"src": "hda", "id": "archive"}


def test_utility_refresh_exposes_outputs_and_result_explorer_entrypoint(
    tmp_path: Path, monkeypatch: Any
) -> None:
    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> list[dict[str, Any]]:
            return [{"id": "entry", "active": True, "target": "/interactivetool/ep/entry/token/"}]

    monkeypatch.setattr("radar_pd_nova.galaxy_service.requests.get", lambda *args, **kwargs: Response())
    service = GalaxyService("https://example.invalid", "key", "history", output_root=tmp_path)
    service._job_details = lambda job_id: {  # type: ignore[method-assign]
        "state": "running",
        "outputs": {"result_report": {"id": "report-id"}},
    }
    action = UtilityActionRecord(
        uid="utility",
        tool_id=RESULT_EXPLORER_TOOL_ID,
        name="Explorer",
        galaxy_job_id="job",
    )

    refreshed = service.refresh_utility(action)

    assert refreshed.status is RunStatus.RUNNING
    assert refreshed.outputs["result_report"] == "report-id"
    assert refreshed.outputs["launch_url"] == "/interactivetool/ep/entry/token/"
    assert refreshed.entrypoint_id == "entry"


def test_result_explorer_inactive_entrypoint_is_reported_as_error(tmp_path: Path, monkeypatch: Any) -> None:
    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> list[dict[str, Any]]:
            return [{"id": "entry", "active": False}]

    monkeypatch.setattr("radar_pd_nova.galaxy_service.requests.get", lambda *args, **kwargs: Response())
    service = GalaxyService("https://example.invalid", "key", "history", output_root=tmp_path)
    service._job_details = lambda job_id: {  # type: ignore[method-assign]
        "state": "ok",
        "job_stderr": "container stopped",
    }
    action = UtilityActionRecord(
        uid="utility",
        tool_id=RESULT_EXPLORER_TOOL_ID,
        name="Explorer",
        galaxy_job_id="job",
    )

    refreshed = service.refresh_utility(action)

    assert refreshed.status is RunStatus.ERROR
    assert refreshed.message == "container stopped"


def test_result_explorer_missing_entrypoint_is_reported_as_error(tmp_path: Path, monkeypatch: Any) -> None:
    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr("radar_pd_nova.galaxy_service.requests.get", lambda *args, **kwargs: Response())
    service = GalaxyService("https://example.invalid", "key", "history", output_root=tmp_path)
    service._job_details = lambda job_id: {"state": "ok"}  # type: ignore[method-assign]
    action = UtilityActionRecord(
        uid="utility",
        tool_id=RESULT_EXPLORER_TOOL_ID,
        name="Explorer",
        galaxy_job_id="job",
    )

    refreshed = service.refresh_utility(action)

    assert refreshed.status is RunStatus.ERROR
    assert refreshed.message == "Result Explorer stopped before its NDIP entry point became active"


def test_gsasii_interactive_session_tracks_ready_and_normal_close(
    tmp_path: Path, monkeypatch: Any
) -> None:
    class Response:
        def __init__(self, active: bool) -> None:
            self.active = active

        def raise_for_status(self) -> None:
            pass

        def json(self) -> list[dict[str, Any]]:
            return [
                {
                    "id": "gsas-entry",
                    "active": self.active,
                    "target": "/interactivetool/ep/gsas/token/" if self.active else None,
                }
            ]

    active = True
    monkeypatch.setattr(
        "radar_pd_nova.galaxy_service.requests.get",
        lambda *args, **kwargs: Response(active),
    )
    service = GalaxyService("https://example.invalid", "key", "history", output_root=tmp_path)
    service._job_details = lambda job_id: {  # type: ignore[method-assign]
        "state": "running" if active else "ok",
        "outputs": {"edited_project": {"id": "saved-gpx"}},
    }
    action = UtilityActionRecord(
        uid="utility",
        tool_id=GSASII_INTERACTIVE_TOOL_ID,
        name="GSAS-II",
        galaxy_job_id="job",
    )

    ready = service.refresh_utility(action)
    assert ready.status is RunStatus.RUNNING
    assert ready.outputs["launch_url"] == "/interactivetool/ep/gsas/token/"
    assert ready.message == "GSAS-II session is ready"

    active = False
    closed = service.refresh_utility(ready)
    assert closed.status is RunStatus.OK
    assert "launch_url" not in closed.outputs
    assert closed.outputs["edited_project"] == "saved-gpx"
    assert closed.message == "GSAS-II session closed; the saved project is available in Galaxy History"


def test_gsasii_interactive_session_explains_pending_entrypoint(tmp_path: Path, monkeypatch: Any) -> None:
    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr("radar_pd_nova.galaxy_service.requests.get", lambda *args, **kwargs: Response())
    service = GalaxyService("https://example.invalid", "key", "history", output_root=tmp_path)
    service._job_details = lambda job_id: {"state": "running"}  # type: ignore[method-assign]
    action = UtilityActionRecord(
        uid="utility",
        tool_id=GSASII_INTERACTIVE_TOOL_ID,
        name="GSAS-II",
        galaxy_job_id="job",
    )

    pending = service.refresh_utility(action)

    assert pending.status is RunStatus.RUNNING
    assert "Waiting for NDIP to allocate the GSAS-II desktop" in pending.message
    assert "Interactive Tool" in pending.message


def test_collection_creation_and_element_mapping_use_public_galaxy_ids(
    tmp_path: Path, monkeypatch: Any
) -> None:
    captured: dict[str, Any] = {}
    get_captured: dict[str, Any] = {}

    class Response:
        def __init__(self, payload: Any) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            pass

        def json(self) -> Any:
            return self.payload

    def post(url: str, **kwargs: Any) -> Response:
        captured.update(kwargs["json"])
        return Response({"id": "collection-id"})

    def get(url: str, **kwargs: Any) -> Response:
        get_captured.update(kwargs)
        return Response(
            {
                "elements": [
                    {"element_identifier": "best.gpx", "object": {"id": "gpx-dataset"}},
                    {"element_identifier": "second.gpx", "object": {"id": "gpx-dataset-2"}},
                ]
            }
        )

    monkeypatch.setattr("radar_pd_nova.galaxy_service.requests.post", post)
    monkeypatch.setattr("radar_pd_nova.galaxy_service.requests.get", get)
    service = GalaxyService("https://example.invalid", "key", "history", output_root=tmp_path)

    collection_id = service.create_dataset_collection("scan series", ["summary-1", "summary-2"])
    elements = service.collection_elements(collection_id)

    assert collection_id == "collection-id"
    assert captured["history_id"] == "history"
    assert [item["id"] for item in captured["element_identifiers"]] == ["summary-1", "summary-2"]
    assert get_captured["params"] == {"view": "element"}
    assert elements == [
        {"id": "gpx-dataset", "name": "best.gpx"},
        {"id": "gpx-dataset-2", "name": "second.gpx"},
    ]


def test_save_configuration_round_trip_publishes_validated_yaml(tmp_path: Path) -> None:
    service = GalaxyService("https://example.invalid", "key", "history", output_root=tmp_path)

    @contextmanager
    def fake_store():
        yield object()

    uploaded: dict[str, Any] = {}

    class Dataset:
        id = "saved-config-id"

    def upload(path: str, store: Any, label: str) -> Dataset:
        uploaded["document"] = Path(path).read_text(encoding="utf-8")
        uploaded["label"] = label
        return Dataset()

    service._store = fake_store  # type: ignore[method-assign]
    service._upload_dataset = upload  # type: ignore[method-assign]

    action = service.save_configuration(_config(AnalysisMode.FULL), associated_run_uid="run")

    assert "$schema: radar-pd-config/v1" in uploaded["document"]
    assert "mode: full" in uploaded["document"]
    assert uploaded["label"] == "configuration"
    assert action.status is RunStatus.OK
    assert action.outputs == {"config_output": "saved-config-id"}
