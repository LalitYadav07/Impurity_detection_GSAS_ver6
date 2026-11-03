"""Synchronous result rendering stays local while mask repair runs in a worker."""
import asyncio
import hashlib
import json
import threading

import pytest

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.galaxy_service import GalaxyService
from radar_pd_nova.models import AnalysisMode, ResultStatus, RunRecord, RunStatus


def setup(tmp_path):
    packet = {"$schema": "radar-pd-pattern-observation/v1", "x": list(range(1, 17)),
              "y": [3.] * 16, "sigma": [1.] * 16, "valid": [True] * 16,
              "fit_valid": [True] * 16, "ycalc": [1.] * 16,
              "source_gpx_sha256": "b" * 64, "context": {"source_product_sha256": "a" * 64}}
    raw = json.dumps(packet).encode()
    summary = {"analysis_mode": "full", "phases": [],
               "pattern_observation": {"name": "Pattern_observation.json",
                   "sha256": hashlib.sha256(raw).hexdigest(), "source_gpx_sha256": "b" * 64},
               "observables": {"source_gpx_sha256": "b" * 64},
               "provenance": {"input": {"source_sha256": "a" * 64}}}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    record = RunRecord(uid="job", galaxy_job_id="job", history_id="history", name="run",
                       mode=AnalysisMode.FULL, status=RunStatus.OK, result_status=ResultStatus.READY,
                       output_dir=str(tmp_path), output_dataset_ids={"pattern_observation": "pattern"})
    app = RadarPdNovaApp()
    app.records = {record.uid: record}
    app.service = GalaxyService("https://example.invalid", "synthetic", "history", output_root=tmp_path / "service")
    app._refinement_maybe_refresh = lambda: None
    app._sync_runs = lambda: None
    app._publish_results_if_requested = lambda row, **kwargs: row
    app.service.collect_results = lambda *a, **k: pytest.fail("Existing mask repair needs no archive collection")
    return app, record, raw


def test_default_synchronous_renderer_never_fetches_missing_observation(tmp_path):
    app, record, raw = setup(tmp_path)
    app.server.state.selected_run_uid = record.uid
    app.service.load_text_document = lambda *a, **k: pytest.fail("Renderer must only read local artifacts")

    app._load_results(record)

    assert app.server.state.result_owner_uid == record.uid
    binding = json.loads((tmp_path / "plot_observation_binding.json").read_text())
    assert binding["status"] == "unavailable" and not binding.get("download_attempted")


def test_retained_incomplete_cache_remains_reachable_with_its_warning(tmp_path):
    app, record, raw = setup(tmp_path)
    record.result_status = ResultStatus.STALE
    record.stage = "Results retained; refresh incomplete"
    record.message = "Showing the previous local cache; gpx_projects download incomplete."
    app._select_record(record)
    app.server.state.workspace_view = "refinement"
    app.service.load_text_document = lambda *a, **k: pytest.fail("Navigation must only read local artifacts")

    app.select_workspace_view("files")

    assert app.server.state.result_owner_uid == record.uid
    assert app.server.state.workspace_view == "files"
    assert app.server.state.selected_run_stage == record.stage
    assert app.server.state.selected_run_message == record.message


def test_existing_observation_repair_waits_in_background_and_keeps_ui_responsive(tmp_path):
    app, record, raw = setup(tmp_path)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def fetch(identifier, **kwargs):
        calls.append((identifier, threading.get_ident()))
        entered.set()
        assert release.wait(5)
        return raw.decode()

    app.service.load_text_document = fetch

    async def run():
        ui_thread = threading.get_ident()
        app._run_selection_changed([record.uid])
        assert await asyncio.to_thread(entered.wait, 2)
        try:
            assert app.server.state.selected_run_loading
            app.server.state.run_search = "UI heartbeat is responsive"
            heartbeat = asyncio.Event()
            asyncio.get_running_loop().call_soon(heartbeat.set)
            await asyncio.wait_for(heartbeat.wait(), .5)
            assert not release.is_set() and calls == [("pattern", calls[0][1])]
            assert calls[0][1] != ui_thread
        finally:
            release.set()
            await asyncio.gather(*list(app._history_result_requests.values()))

    asyncio.run(run())
    binding = json.loads((tmp_path / "plot_observation_binding.json").read_text())
    assert binding["status"] == "verified" and (tmp_path / "pattern_observation.json").read_bytes() == raw
    assert app.server.state.result_owner_uid == record.uid and not app.server.state.selected_run_loading
