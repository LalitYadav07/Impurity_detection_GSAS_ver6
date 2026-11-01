"""Reviewed conditions are consistent before and after refinement completion."""
import copy

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.post_analysis import PostAnalysisSettings, reviewed_axis_overrides
from radar_pd_nova.transition_tracking_ui import _conditions, curve_label
from test_transition_tracking_ui import scans


def mapped_settings(row):
    return PostAnalysisSettings.model_validate({"axis_mappings": {"temperature": {
        row["run_id"]: {"value": 114, "unit": "C", "source": "Reviewed thermometer units",
                        "kind": "user_supplied", "verified": True}
    }}})


def test_pending_and_completed_use_same_reviewed_temperature_without_mutating_measurement():
    app = RadarPdNovaApp()
    row = scans()[0]
    original = copy.deepcopy(row)
    app._set_post_settings(mapped_settings(row))
    pending = app._transition_sources([], [row])[0]
    completed = app._transition_sources([row], [row])[0]
    assert _conditions(pending)["temperature"]["value"] == 387.15
    assert _conditions(pending)["temperature"]["status"] == "reviewed_mapping"
    assert pending["axis_overrides"] == completed["axis_overrides"]
    assert "387.1" in curve_label(pending) or "387.2" in curve_label(pending)
    assert row == original
    app._set_post_settings(PostAnalysisSettings())
    assert app._transition_sources([completed], [])[0]["axis_overrides"] == {}


def test_display_mapping_does_not_change_pattern_membership():
    app = RadarPdNovaApp()
    rows = scans()
    raw = copy.deepcopy(rows)
    app._refresh_transition_tracking([], pending=rows)
    before = [(r["run_id"], r.get("state_id"), r.get("event")) for r in app._transition_report["records"]]
    app._set_post_settings(mapped_settings(rows[0]))
    app._refresh_transition_tracking([], pending=rows)
    after = [(r["run_id"], r.get("state_id"), r.get("event")) for r in app._transition_report["records"]]
    assert after == before
    assert rows == raw


def test_unverified_mapping_does_not_override_conditions():
    row = scans()[0]
    payload = mapped_settings(row).model_dump(mode="json")
    payload["axis_mappings"]["temperature"][row["run_id"]]["verified"] = False
    assert reviewed_axis_overrides(row, PostAnalysisSettings.model_validate(payload)) == {}
