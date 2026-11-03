"""Incomplete result transfers retain usable artifacts and retry missing outputs."""
import asyncio
import json
from pathlib import Path
import zipfile

import pytest

from radar_pd_nova.galaxy_service import GalaxyService
from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.models import AnalysisMode, ResultStatus, RunRecord, RunStatus


@pytest.fixture
def collection(tmp_path):
    service = GalaxyService("https://example.invalid", "synthetic", "history", output_root=tmp_path)
    service._job_details = lambda _uid: {}
    service.job_output_ids = lambda _uid: {}
    service._dataset_metadata = lambda _uid: {"file_size": 1234, "update_time": "unchanged"}
    record = RunRecord(uid="job", galaxy_job_id="job", name="run", history_id="history",
                       mode=AnalysisMode.FULL, status=RunStatus.OK,
                       output_dataset_ids={"summary": "summary", "results_archive": "archive",
                                           "gpx_projects": "gpx"})
    calls = []
    transport = {"failed": set(), "revision": "old"}

    def dataset(identifier, target):
        calls.append(identifier)
        if identifier in transport["failed"]:
            raise TimeoutError(identifier + " transport timeout")
        summary = json.dumps({"analysis_mode": "full", "revision": transport["revision"], "phases": []})
        if identifier == "summary":
            target.write_text(summary)
        elif identifier == "archive":
            with zipfile.ZipFile(target, "w") as archive:
                archive.writestr("ndip/summary.json", summary)
                archive.writestr("ndip/plots/final_fit.plotdata.json", '{"synthetic": true}')
        else:
            raise AssertionError(identifier)

    def projects(identifier, target):
        calls.append(identifier)
        if identifier in transport["failed"]:
            raise TimeoutError(identifier + " transport timeout")
        target.mkdir(parents=True, exist_ok=True)
        (target / "synthetic.gpx").write_text("fixture " + transport["revision"])

    service._download_dataset = dataset
    service._download_collection = projects
    return service, record, calls, transport


def cached_files(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*")
            if path.is_file() and path.name != ".radar-pd-refresh.json"}


def test_partial_force_refresh_preserves_previous_cache_and_its_identity(collection):
    service, record, calls, transport = collection
    service.collect_results(record)
    root = Path(record.output_dir)
    before = cached_files(root)
    manifest = record.cache_manifest.model_dump()
    transport.update(failed={"archive", "gpx"}, revision="replacement")
    service._dataset_metadata = lambda _uid: {"file_size": 5678, "update_time": "replacement"}

    service.collect_results(record, force=True)

    assert cached_files(root) == before
    assert record.cache_manifest.model_dump() == manifest
    assert record.status is RunStatus.OK and record.analysis_status is RunStatus.OK
    assert record.result_status is ResultStatus.STALE
    assert "previous local cache" in record.message and "incomplete" in record.message
    assert "results_archive" in record.message and "gpx_projects" in record.message
    pending = json.loads((root / ".radar-pd-refresh.json").read_text())
    assert pending["attempted_manifest"]["archive_update_time"] == "replacement"
    assert sorted(pending["failures"]) == ["gpx_projects", "results_archive"]


def test_missing_unique_collection_is_visible_and_retry_reuses_successful_downloads(collection):
    service, record, calls, transport = collection
    transport["failed"] = {"gpx"}
    service.collect_results(record)
    root = Path(record.output_dir)
    assert record.result_status is ResultStatus.STALE
    assert "gpx_projects" in record.message and "incomplete" in record.message
    assert list(root.rglob("*.plotdata.json")) and not list(root.rglob("*.gpx"))
    assert calls.count("summary") == calls.count("archive") == calls.count("gpx") == 1

    transport["failed"].clear()
    service.collect_results(record)

    assert record.result_status is ResultStatus.READY and record.message == ""
    assert (root / "gpx_projects/synthetic.gpx").is_file()
    assert calls.count("summary") == calls.count("archive") == 1
    assert calls.count("gpx") == 2
    service.collect_results(record)
    assert calls.count("summary") == calls.count("archive") == 1
    assert calls.count("gpx") == 2


def test_removed_collection_is_repaired_without_downloading_the_archive_again(collection):
    service, record, calls, transport = collection
    service.collect_results(record)
    root = Path(record.output_dir)
    (root / "gpx_projects/synthetic.gpx").unlink()

    service.collect_results(record)

    assert (root / "gpx_projects/synthetic.gpx").is_file()
    assert calls.count("archive") == calls.count("summary") == 1
    assert calls.count("gpx") == 2


def test_changed_collection_failure_cannot_relabel_old_projects_as_current(collection):
    service, record, calls, transport = collection
    service.collect_results(record)
    root = Path(record.output_dir)
    before = cached_files(root)
    record.output_dataset_ids["gpx_projects"] = "replacement-gpx"
    transport["failed"] = {"replacement-gpx"}
    transport["revision"] = "replacement"

    service.collect_results(record)

    assert cached_files(root) == before
    assert record.result_status is ResultStatus.STALE
    assert "previous local cache" in record.message
    receipt = json.loads((root / ".radar-pd-artifacts.json").read_text())
    assert receipt["complete"]["gpx_projects"] == "gpx"
    assert service._result_cache_needs_collection(record, root)


def test_ordinary_saved_run_open_retries_only_the_missing_collection(collection):
    service, record, calls, transport = collection
    transport["failed"] = {"gpx"}
    service.collect_results(record)
    app = RadarPdNovaApp()
    app.records = {record.uid: record}
    app.service = service
    app._sync_runs = lambda: None
    app._publish_results_if_requested = lambda row, **kwargs: row
    displayed = []
    app._load_results = lambda row: displayed.append(row)
    transport["failed"].clear()

    async def run():
        app._run_selection_changed([record.uid])
        await asyncio.gather(*list(app._history_result_requests.values()))

    asyncio.run(run())
    assert len(displayed) == 1 and displayed[0].result_status is ResultStatus.READY
    assert calls.count("archive") == calls.count("summary") == 1
    assert calls.count("gpx") == 2
    assert list(Path(displayed[0].output_dir).rglob("*.gpx"))


def test_publication_failure_rolls_back_to_previous_complete_cache(collection, monkeypatch):
    service, record, calls, transport = collection
    service.collect_results(record)
    root = Path(record.output_dir)
    before = cached_files(root)
    transport["revision"] = "replacement"

    def fail_publication(staging, destination):
        destination.mkdir()
        (destination / "unfinished").write_text("partial copy")
        raise OSError("synthetic publication failure")

    monkeypatch.setattr("radar_pd_nova.galaxy_service._publish_staging_directory", fail_publication)
    service.collect_results(record, force=True)

    assert cached_files(root) == before
    assert record.result_status is ResultStatus.STALE
    assert "previous local cache" in record.message and "publication failure" in record.message
    assert not list(root.parent.glob("*.backup"))
