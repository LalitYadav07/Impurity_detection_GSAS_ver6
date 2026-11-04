"""Completed Live watches recover portable fit observations after a restart."""

import asyncio
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.galaxy_service import GalaxyService
from radar_pd_nova.powgen_controller import PowgenExperimentSettings, PowgenWatchController
from radar_pd_nova.powgen_watch import WatchedRun, WatchState
from radar_pd_nova.scan_interpretation import PATTERN_SCHEMA, Pattern
from radar_pd_nova.unet_transition_ui import UnetTransitionMixin, _fit_figure_for_scan


def pattern_packet():
    return {
        "$schema": PATTERN_SCHEMA,
        "source_gpx_sha256": "b" * 64,
        "coordinate": {"kind": "tof", "unit": "us"},
        "context": {"profile_sha256": "profile", "reduction_family": "fixture"},
        "fit_domain": {"limits": [1, 32], "excluded": []},
        "x": list(range(1, 33)),
        "y": [float(n + 2) for n in range(32)],
        "ycalc": [float(n + 1) for n in range(32)],
        "sigma": [1.0] * 32,
        "valid": [True] * 32,
        "fit_valid": [True] * 32,
    }


def app_harness(controller):
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    app.server = SimpleNamespace(state=SimpleNamespace(
        powgen_selected_samples=[], powgen_selected_phase_labels=[],
        powgen_selected_run_id="", powgen_x_axis="run_number", flush=Mock(),
    ))
    app.records = {}
    app._result_context = object()
    app.service = getattr(controller, "service", None)
    app._powgen_controller = controller
    app._post_analysis_scope = "watch:" + str(controller.source_directory)
    app._sync_runs = Mock()
    app._sync_powgen_dashboard_provenance = Mock()
    app._refinement_maybe_refresh = Mock()
    app._update_powgen_dashboard_figures = Mock()
    app._live_adapter = lambda: SimpleNamespace(instrument="PG3", name="POWGEN")
    return app


@pytest.mark.parametrize("direct_dataset", [True, False], ids=["monitor-output", "legacy-tables"])
def test_completed_only_restore_fetches_curve_and_projects_current_cache(tmp_path, direct_dataset):
    raw = json.dumps(pattern_packet()).encode()
    checksum = sha256(raw).hexdigest()
    old_path = tmp_path / "previous-session" / "observations" / f"{checksum}.json"
    descriptor = {
        "name": "Pattern_observation.json", "sha256": checksum,
        "source_gpx_sha256": "b" * 64, "local_path": str(old_path),
        **({"dataset_id": "curve-hda"} if direct_dataset else {"collection_id": "tables-hdca"}),
    }
    service = GalaxyService("https://galaxy.invalid", "test-key", "history", output_root=tmp_path / "current-session")
    service.load_text_document = Mock(return_value=raw.decode())
    service.collection_elements = Mock(return_value=[{"name": "Pattern_observation.json", "id": "curve-hda"}])
    service.refresh = Mock(side_effect=AssertionError("Completed summaries need no scientific job refresh"))
    service.collect_results = Mock(side_effect=AssertionError("Recovery must not download a result archive"))
    settings = PowgenExperimentSettings("IPTS-1", "history", "config-hda", "1.5")
    original = PowgenWatchController(service, settings)
    summary = {"status": "complete", "analysis_mode": "full", "rwp": 3.2,
               "final_rwp": 3.2, "pattern_observation": descriptor}
    run = WatchedRun(1, "/SNS/PG3/IPTS-1/shared/autoreduce/PG3_1.gsa", "source-revision",
                     bank=2, galaxy_job_id="job-hda", galaxy_result_ids=("summary-hda", "curve-hda"),
                     scan_metadata={"schema_version": 3, "sample_id": "sample-1"},
                     scientific_summary=summary)
    original.state.completed[run.run_id] = run
    snapshot = json.loads(original.capture_checkpoint().payload_json)
    service.search_history_datasets = Mock(return_value=[{"id": "watch-hda"}])
    service.load_json_document = Mock(return_value=deepcopy(snapshot))
    restored = PowgenWatchController(service, settings)
    assert restored.restore_latest_state()
    assert not restored.state.submitted
    assert restored.state.completed[run.run_id].scientific_summary["pattern_observation"]["local_path"] == str(old_path)

    app = app_harness(restored)
    refresh = restored.refresh
    restored.refresh = Mock(wraps=refresh)
    warnings = []
    asyncio.run(app._refresh_powgen_jobs(restored, warnings))

    restored.refresh.assert_called_once_with(checkpoint=False)
    service.load_text_document.assert_called_once_with("curve-hda", max_bytes=16_000_000)
    assert service.collection_elements.call_count == (0 if direct_dataset else 1)
    service.refresh.assert_not_called()
    service.collect_results.assert_not_called()
    app._sync_runs.assert_called_once_with()
    assert warnings == []
    current_path = service.output_root / "observations" / f"{checksum}.json"
    assert current_path.read_bytes() == raw
    assert not old_path.exists()
    row = app.server.state.powgen_scientific_rows[0]
    assert row["pattern_path"] == str(current_path)
    assert row["pattern_observation"]["dataset_id"] == "curve-hda"
    source = app._unet_transition_sources([row], [])[0]
    assert source["pattern_path"] == str(current_path)
    figure, _ = _fit_figure_for_scan(source)
    assert "Calculated" in {trace["name"] for trace in figure["data"]}
    recovered = restored.state.completed[run.run_id]
    assert recovered.source_path == run.source_path and recovered.fingerprint == run.fingerprint
    assert recovered.galaxy_job_id == run.galaxy_job_id and recovered.galaxy_result_ids == run.galaxy_result_ids
    assert recovered.scientific_summary["final_rwp"] == summary["final_rwp"]
    assert snapshot["completed"][run.run_id]["scientific_summary"] == summary


def test_empty_watch_does_not_refresh_or_publish_rows():
    controller = SimpleNamespace(state=WatchState("history"), source_directory="/read-only/source", refresh=Mock())
    app = app_harness(controller)
    app._sync_powgen_rows = Mock()
    asyncio.run(app._refresh_powgen_jobs(controller, []))
    controller.refresh.assert_not_called()
    app._sync_runs.assert_not_called()
    app._sync_powgen_rows.assert_not_called()
    app.server.state.flush.assert_not_called()


@pytest.mark.parametrize("same_cache_path", [False, True], ids=["new-cache-path", "reused-cache-path"])
def test_newly_cached_receipt_replaces_previously_empty_compact_fit(tmp_path, same_cache_path):
    packet = pattern_packet()
    raw = json.dumps(packet).encode()
    descriptor = {"name": "Pattern_observation.json", "sha256": sha256(raw).hexdigest(),
                  "source_gpx_sha256": packet["source_gpx_sha256"], "dataset_id": "curve-hda"}
    local = tmp_path / "curve.json"
    if same_cache_path:
        descriptor["local_path"] = str(local)
    scan = {"run_id": "PG3_1_BANK2", "source_path": str(tmp_path / "raw.gsa"),
            "pattern_observation": descriptor}
    if same_cache_path:
        scan["pattern_path"] = str(local)
    engine = SimpleNamespace(evaluate=Mock(return_value={}),
                             load=Mock(return_value=SimpleNamespace(pattern=Pattern.parse(packet))))
    state = SimpleNamespace(unet_transition_order="acquisition", unet_transition_direction="ascending",
                            unet_transition_reference_gap=1, flush=Mock())
    harness = SimpleNamespace(server=SimpleNamespace(state=state), _post_analysis_scope="watch:test",
                              _unet_transition_engine=engine, _unet_transition_fit_cache={},
                              _unet_transition_signature=None, _unet_transition_fit_signature=None,
                              _unet_transition_generation=1, _unet_transition_task=None)
    rendered = []

    def apply(_report, sources, signature):
        harness._unet_transition_signature = signature
        try:
            rendered.append(UnetTransitionMixin._cached_unet_transition_fit(harness, sources[0])[0])
        except (OSError, ValueError, TypeError):
            # The real renderer leaves the compact plot empty and reports the
            # unavailable receipt until a later refresh can read it.
            rendered.append(None)

    harness._apply_unet_transition_tracking = apply
    harness._update_unet_transition_loading = lambda: UnetTransitionMixin._update_unet_transition_loading(harness)
    harness._render_unet_transition_tracking = lambda: apply(
        {}, list(harness._unet_transition_scans.values()), harness._unet_transition_signature)
    UnetTransitionMixin._schedule_unet_transition_tracking(harness, [scan])
    assert not (rendered[-1] or {}).get("data")
    local.write_bytes(raw)
    recovered = {**scan, "pattern_path": str(local),
                 "pattern_observation": {**descriptor, "local_path": str(local)}}
    UnetTransitionMixin._schedule_unet_transition_tracking(harness, [recovered])
    assert rendered[-1] is not None
    assert "Calculated" in {trace["name"] for trace in rendered[-1]["data"]}
