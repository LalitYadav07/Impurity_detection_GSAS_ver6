"""Distinguish local component evolution from newly resolved components."""
import copy

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.transition_signals import BoundaryPolicy
from radar_pd_nova.transition_tracking_ui import (
    _component_evidence_status,
    _splitting_evidence_detail,
)
from test_scan_interpretation import scan
from test_splitting_profiles import doublet


def component_app(before, after):
    rows = []
    for i, pattern in enumerate((before, after), 1):
        row = scan(i)
        row["pattern"].update(
            coordinate={"kind": "tof", "unit": "us"},
            x=pattern.x.tolist(), y=pattern.intensity.tolist(),
            sigma=pattern.sigma.tolist(), valid=pattern.valid.tolist(),
            fit_valid=pattern.valid.tolist(), ycalc=pattern.intensity.tolist(),
        )
        rows.append(row)
    app = RadarPdNovaApp()
    app._refresh_transition_tracking(rows)
    app.update_transition_display("overlay", "both")
    return app


def test_existing_doublet_is_described_without_claiming_new_components():
    app = component_app(doublet(24), doublet(32, seed=7))
    state = app.server.state
    report = copy.deepcopy(app._transition_report)
    rule = copy.deepcopy(state.transition_policy)
    indicator = next(v for v in state.transition_signal_rows if "Splitting" in v["indicator"])
    assert indicator["result"] == "Triggered"
    assert "Existing doublet separation changed" in indicator["detail"]
    assert "physical cause is unresolved" in indicator["detail"]
    assert "Resolved component count changed" not in indicator["detail"]
    local = next(v for v in state.transition_region_evidence_rows if "Existing doublet separation changed" in v["status"])
    assert "cause unresolved" in local["status"]
    assert float(local["before"]) < float(local["after"])
    app._render_transition_tracking()
    assert app._transition_report == report
    assert state.transition_policy == rule
    assert report["physical_transition_confirmed"] is False
    assert "item.detail" in app.layout.html


def test_newly_resolved_local_component_count_has_distinct_wording():
    app = component_app(doublet(0), doublet(24, seed=7))
    state = app.server.state
    indicator = next(v for v in state.transition_signal_rows if "Splitting" in v["indicator"])
    assert indicator["result"] == "Triggered"
    assert "Resolved component count changed" in indicator["detail"]
    assert "Existing doublet separation changed" not in indicator["detail"]
    local = next(v for v in state.transition_region_evidence_rows if "Resolved component count changed" in v["status"])
    assert "cause unresolved" in local["status"]
    assert "Single component" in (local["before"], local["after"])
    assert app._transition_report["physical_transition_confirmed"] is False


def test_mixed_and_legacy_subtypes_are_reported_without_mutating_evidence():
    indicator = {"regions": [
        {"kind": "local_component_separation_change", "edge_neighbor_overlap": True},
        {"kind": "local_component_separation_change"},
        {"kind": "local_resolved_component_count_change"},
        {"kinds": ["resolved_component_count_change"]},
        {"kind": "legacy_unclassified"},
    ]}
    original = copy.deepcopy(indicator)
    result = _splitting_evidence_detail(indicator)
    assert "Existing doublet separation changed in 2 regions" in result
    assert "Resolved component count changed in 2 regions" in result
    assert "Component subtype is unavailable in 1 region" in result
    assert indicator == original
    assert _splitting_evidence_detail({"regions": []}) == ""


def test_below_rule_doublet_is_not_presented_as_a_detected_change():
    event = {"kind": "local_component_separation_change", "passes_rule": False}
    assert _component_evidence_status(event) == "Existing doublet; below change rule"
    assert "unavailable" in _component_evidence_status({"passes_rule": True})
    assert BoundaryPolicy().minimum_signals == 1
