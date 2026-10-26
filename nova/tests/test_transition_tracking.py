"""State memory must recognize recurrence without chaining no-change decisions."""

from dataclasses import replace

import numpy as np
from radar_pd_nova.pattern_transition import prepare_pattern
from radar_pd_nova.transition_input import observation, packet_observation
from radar_pd_nova.transition_tracking import (
    PatternStateMemory,
    TransitionTrackingEngine,
)
from test_pattern_transition import observed
from test_scan_interpretation import scan


def sequence():
    rows = []
    for i, kind in enumerate([0, 0, 0, 1, 1, 1, 2, 1]):
        kwargs = {"new": [(2700, 18, 0.0025), (6000, 18, 0.0025)]} if kind else {}
        if kind == 2:
            kwargs["factors"] = [1, 1, 1, 1, 1, 1, 2.5, 0.25]
        rows.append(observed(seed=i + 10, ident=str(i + 1), **kwargs))
    return rows


def test_same_state_repeats_and_a_return_reuses_the_original_symbol_identity():
    memory = PatternStateMemory()
    records = [
        memory.observe(prepare_pattern(p), observation(p).fingerprint)
        for p in sequence()
    ]
    ids = [r["state_id"] for r in records]
    assert all(ids)
    assert ids[0] == ids[1] == ids[2]
    assert ids[3] == ids[4] == ids[5] == ids[7]
    assert len({ids[0], ids[3], ids[6]}) == 3
    assert records[7]["event"] == "returned"
    assert records[7]["representative"] == "4"


def test_bad_scan_does_not_become_a_state_or_replace_a_good_previous_pattern():
    memory = PatternStateMemory()
    good = observed(ident="first")
    memory.observe(prepare_pattern(good), observation(good).fingerprint)
    mask = np.ones(len(good.x), dtype=bool)
    mask[::4] = False
    bad = replace(good, valid=mask, observation_id="bad")
    row = memory.observe(prepare_pattern(bad), observation(bad).fingerprint)
    assert row["state_id"] is None
    next_scan = replace(good, observation_id="next")
    row = memory.observe(prepare_pattern(next_scan), observation(next_scan).fingerprint)
    assert row["previous"] == "first" and row["event"] == "continued"


def test_uncertain_or_nontransitive_bridge_is_not_assigned_by_connected_components():
    first, second, bridge = [
        replace(observed(), observation_id=s) for s in ["a", "b", "c"]
    ]

    def pair(a, b, _):
        names = {a.observation.observation_id, b.observation.observation_id}
        return {
            "status": "transition_evidence"
            if names == {"a", "c"}
            else "no_resolved_change"
        }

    memory = PatternStateMemory(compare=pair)
    rows = [
        memory.observe(prepare_pattern(p), observation(p).fingerprint)
        for p in [first, second, bridge]
    ]
    assert rows[0]["state_id"] == rows[1]["state_id"]
    assert rows[2]["state_id"] is None
    assert rows[2]["reason"] == "ambiguous_or_nontransitive_state_membership"


def graph_observations(names, relations):
    def pair(a, b, _):
        x, y = a.observation.observation_id, b.observation.observation_id
        return {"status": "no_resolved_change" if x == y else relations.get(frozenset([x, y]), "uncertain")}

    memory = PatternStateMemory(compare=pair)
    return [memory.observe(prepare_pattern(p), observation(p).fingerprint)
            for p in [observed(seed=ord(name), ident=name) for name in names]]


def test_a_noisy_initial_reference_does_not_block_later_compatible_scans_or_invent_a_transition():
    rows = graph_observations("abcd", {frozenset(p): "no_resolved_change" for p in ["bc", "bd", "cd"]})
    assert [r["event"] for r in rows] == ["initial", "unassigned", "reference", "continued"]
    assert rows[2]["state_id"] == rows[3]["state_id"] != rows[0]["state_id"]
    assert rows[2]["relation_to_other_references"] == "unresolved"
    assert rows[1]["state_id"] is None  # no retrospective relabeling


def test_an_interior_group_member_can_veto_a_contradictory_merge():
    relations = {frozenset(p): "no_resolved_change" for p in ["ab", "ac", "bc", "ad", "cd"]}
    relations[frozenset("bd")] = "transition_evidence"
    rows = graph_observations("abcd", relations)
    assert rows[0]["state_id"] == rows[1]["state_id"] == rows[2]["state_id"]
    # First/last-only comparison would wrongly merge d despite its conflict with b.
    assert rows[3]["state_id"] is None


def test_matching_two_established_references_remains_ambiguous():
    rows = graph_observations("abc", {frozenset("ab"): "transition_evidence",
        frozenset("ac"): "no_resolved_change", frozenset("bc"): "no_resolved_change"})
    assert rows[0]["state_id"] != rows[1]["state_id"]
    assert rows[2]["state_id"] is None
    assert rows[2]["reason"] == "ambiguous_between_existing_pattern_references"


def test_statistical_input_and_fingerprint_ignore_fitted_labels_values_and_rwp():
    a = scan(1)
    b = scan(1)
    b["observables"] = {"phases": [{"phase_id": "totally different", "a": 900}]}
    b["rwp"] = 99
    b["pattern"]["source_gpx_sha256"] = "new fit"
    b["pattern"]["ycalc"] = [999.0] * len(b["pattern"]["x"])
    assert packet_observation(a).fingerprint == packet_observation(b).fingerprint


def test_scope_isolation_and_replay_order_are_independent_of_completion_order():
    engine = TransitionTrackingEngine()
    report = engine.evaluate(
        [scan(3), scan(1), scan(2), scan(4, bank=3), scan(5, sample="other")]
    )
    assert [r["run_id"] for r in report["records"]] == [
        "NOM_1_BANK2",
        "NOM_2_BANK2",
        "NOM_3_BANK2",
        "NOM_4_BANK3",
        "NOM_5_BANK2",
    ]
    assert len(report["groups"]) == 3
    assert all(
        r["event"] == "initial" for r in [report["records"][0], *report["records"][3:]]
    )


def test_unknown_specimen_and_multiple_result_revisions_are_explicitly_unassigned():
    report = TransitionTrackingEngine().evaluate([scan(1, sample=""), scan(2), scan(2)])
    assert all(r["state_id"] is None for r in report["records"])
    assert "sample ID" in report["records"][0]["reason"]
    assert "revisions" in report["records"][1]["reason"]


def test_checkpoint_rejects_invalid_color_limits_and_unbounded_marker_indices():
    from radar_pd_nova.transition_tracking import checkpoint

    clean = checkpoint(
        {
            "references": {"group": "run"},
            "styles": {"okay": 2, "bad": 33, "boolean": True},
            "limits": {
                "bad": {"low": 0, "high": float("nan"), "soft": 1},
                "good": {"low": -2, "high": 5, "soft": 0.1},
            },
        }
    )
    assert clean["references"] == {"group": "run"}
    assert clean["styles"] == {"okay": 2}
    assert list(clean["limits"]) == ["good"]
