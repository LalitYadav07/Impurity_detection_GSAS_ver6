"""Regressions found by replaying real beamline file-arrival sequences."""

from dataclasses import replace
from pathlib import Path

import pytest

from radar_pd_nova.powgen_controller import (
    PowgenExperimentSettings,
    PowgenWatchController,
)
from radar_pd_nova.nomad_controller import NomadExperimentSettings, NomadWatchController
from radar_pd_nova.powgen_watch import WatchDefinition
from test_powgen_setup import source as powgen_source
from test_powgen_controller import FakeGalaxyService
from test_nomad import source, supplied


def test_partial_nomad_completion_replaces_placeholder_below_backfill_floor(source, monkeypatch):
    monkeypatch.setattr(
        WatchDefinition, "source_directory", property(lambda self: str(source.parent))
    )
    monkeypatch.setattr(NomadWatchController, "_read_scan_metadata", lambda *a: {})
    service = FakeGalaxyService()
    settings = NomadExperimentSettings(
        "IPTS-1",
        service.history_id,
        "config-hda",
        selected_banks=(5,),
        initial_backfill_limit=1,
    )
    controller = NomadWatchController(service, settings)
    complete = source.read_bytes()
    source.write_bytes(complete[:150])
    partial = controller.discover()
    assert len(partial) == 1 and partial[0].scan_metadata["setup_error"]
    assert not controller.due_submissions()
    assert partial[0].submission_attempts == 0
    source.write_bytes(complete)
    controller.discover()
    pending = controller.due_submissions()
    assert [r.run_id for r in pending] == ["NOM_42_BANK5"]
    assert "setup_error" not in pending[0].scan_metadata
    assert not pending[0].scan_metadata.get("source_revision_warning")
    assert controller.discover() == []


@pytest.mark.parametrize("receipt_change", [False, True])
def test_powgen_source_or_receipt_replacement_preserves_result_and_flags_review(
    tmp_path, monkeypatch, receipt_change
):
    path = powgen_source(tmp_path)
    monkeypatch.setattr(WatchDefinition, "source_directory", property(lambda self: str(tmp_path)))
    monkeypatch.setattr(PowgenWatchController, "_read_scan_metadata", lambda *a: {})
    service = FakeGalaxyService()
    settings = PowgenExperimentSettings(
        "IPTS-1", service.history_id, "config-hda", "", selected_banks=(2,)
    )
    controller = PowgenWatchController(service, settings)
    run = controller.discover()[0]
    record = controller.submit(run)
    controller.state.mark_completed(
        run,
        ("result-id",),
        galaxy_job_id=record.galaxy_job_id,
        scientific_summary={
            "scientific_status": "checks_passed",
            "observables": {"source_gpx_sha256": "original"},
        },
    )
    if receipt_change:
        side = path.with_suffix(".py")
        side.write_text(side.read_text() + "\n# new receipt revision\n")
    else:
        path.write_text(
            path.read_text().replace("calibration test", "replacement calibration test")
        )
    assert controller.discover() == []
    completed = controller.state.completed[run.run_id]
    assert completed.scan_metadata["source_revision_warning"]
    assert completed.scientific_summary["scientific_status"] == "needs_review"
    assert completed.scientific_summary["observables"]["source_gpx_sha256"] == "original"
    assert not controller.due_submissions()
    assert len(service.snapshots) == 1


def test_powgen_identical_atomic_rewrite_does_not_flag_changed_input(tmp_path, monkeypatch):
    path = powgen_source(tmp_path)
    monkeypatch.setattr(WatchDefinition, "source_directory", property(lambda self: str(tmp_path)))
    monkeypatch.setattr(PowgenWatchController, "_read_scan_metadata", lambda *a: {})
    service = FakeGalaxyService()
    controller = PowgenWatchController(
        service,
        PowgenExperimentSettings("IPTS-1", service.history_id, "config-hda", ""),
    )
    run = controller.discover()[0]
    controller.submit(run)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(path.read_bytes())
    temporary.replace(path)
    controller.discover()
    assert not controller.state.submitted[run.run_id].scan_metadata.get("source_revision_warning")
