from dataclasses import replace

import numpy as np
import pytest

from radar_pd_nova.pattern_transition import compare_observed_patterns, prepare_pattern
from radar_pd_nova.transition_input import observation
from radar_pd_nova.transition_signals import BoundaryPolicy, apply_policy, measure_signals
from radar_pd_nova.transition_tracking import TransitionTrackingEngine
from test_pattern_transition import observed


def decision(a, b, policy):
    a, b = prepare_pattern(a), prepare_pattern(b)
    raw = compare_observed_patterns(a, b)
    return apply_policy(measure_signals(a, b, raw), policy)


def source(pattern, number):
    return {"run_id": pattern.observation_id, "run_number": number, "measured": pattern,
            "pattern": {"x": pattern.x.tolist(), "y": pattern.intensity.tolist()},
            "metadata": {"sample_id": "one"}, "instrument": "PG3", "ipts": "IPTS-1", "bank": 2}


def engine(policy):
    return TransitionTrackingEngine(policy=policy, loader=lambda s: observation(s["measured"]))


def test_new_width_measurement_detects_modest_broadening_below_the_legacy_gate():
    a, b = observed(), observed(width=.00275, seed=5)
    assert compare_observed_patterns(a, b)["status"] == "no_resolved_change"
    result = decision(a, b, BoundaryPolicy(selected=("width",)))
    assert result["status"] == "transition_evidence", result
    assert result["indicators"]["width"]["support"] >= 2
    assert not result["indicators"]["splitting"]["detected"]


def test_normal_peak_motion_gain_and_count_noise_do_not_become_broadening():
    a = observed()
    b = observed(shifts=np.array([-.004,.002,-.003,.003,-.002,.004,-.001,.001]), gain=4, offset=24, seed=5)
    result = decision(a, b, BoundaryPolicy(selected=("width",)))
    assert result["status"] == "no_resolved_change", result


def test_unresolved_daughters_can_trigger_width_without_two_resolved_maxima():
    a = observed(noise=.03)
    x = a.x
    # Two offset copies replace each original Gaussian at fixed integrated area.
    y = 15 + .0005*x
    for center in [1300,1700,2300,3100,4000,5300,6800,8500]:
        sigma = center*.0025
        for offset in [-.4*sigma,.4*sigma]:
            y += 50*np.exp(-.5*((x-center-offset)/sigma)**2)
    b = replace(a, intensity=y, observation_id="unresolved")
    result = decision(a, b, BoundaryPolicy(selected=("width",)))
    assert result["indicators"]["width"]["detected"], result
    assert not result["indicators"]["splitting"]["detected"]


def test_indicator_selection_and_required_count_have_actual_scientific_effect():
    a = observed()
    b = observed(new=[(2700,25,.0025),(6000,25,.0025)],seed=8)
    result = decision(a, b, BoundaryPolicy(selected=("width","appearance"),minimum_signals=2))
    assert result["indicators"]["appearance"]["detected"]
    assert not result["indicators"]["width"]["detected"]
    assert result["status"] == "no_resolved_change"
    result = decision(a, b, BoundaryPolicy(selected=("appearance",)))
    assert result["status"] == "transition_evidence"
    result = decision(a, b, BoundaryPolicy(selected=("width",)))
    assert result["status"] == "no_resolved_change"


def test_unavailable_width_is_not_a_negative_vote_or_a_claim_of_similarity():
    result = decision(observed(), observed(), BoundaryPolicy(selected=("width",), fixed_broadening=False))
    assert result["status"] == "uncertain"
    assert not result["indicators"]["width"]["available"]
    result = decision(observed(), observed(), BoundaryPolicy(selected=("width","appearance"),fixed_broadening=False))
    assert result["status"] == "uncertain", "An unavailable selected signal could still satisfy an any-one rule"


def test_append_is_causal_and_recurrence_reuses_a_symbol_when_width_returns():
    policy = BoundaryPolicy(selected=("width",))
    tracker = engine(policy)
    rows = [source(observed(ident="a",seed=2),1), source(observed(ident="b",seed=3),2),
            source(observed(ident="c",width=.0028,seed=4),3), source(observed(ident="d",width=.0028,seed=5),4),
            source(observed(ident="e",seed=6),5)]
    previous = []
    for n, row in enumerate(rows,1):
        report = tracker.append(row)
        assert report["records"][:-1] == previous
        reference = engine(policy).evaluate(rows[:n])
        assert report["records"] == reference["records"]
        previous = list(report["records"])
    r = report["records"]
    assert r[0]["state_id"] == r[1]["state_id"] == r[4]["state_id"]
    assert r[2]["state_id"] == r[3]["state_id"] != r[0]["state_id"]
    assert r[4]["event"] == "returned" and r[4]["boundary"]


def test_adjacent_rule_recognizes_active_reference_after_gradual_width_drift():
    policy = BoundaryPolicy(selected=("width",), comparison="previous")
    widths = [.0025, .00255, .0026, .00265, .0027, .00275, .0028, .0025]
    rows = [source(observed(ident=f"width-{i}", width=width, noise=.003, seed=4), i+1)
        for i,width in enumerate(widths)]
    tracker = engine(policy)
    previous = []
    for n,row in enumerate(rows,1):
        report = tracker.append(row)
        assert report["records"][:-1] == previous
        assert report["records"] == engine(policy).evaluate(rows[:n])["records"]
        previous = list(report["records"])
    assert not any(r["boundary"] for r in previous[:-1])
    assert previous[-1]["boundary"] and previous[-1]["event"] == "returned"
    assert previous[-1]["state_id"] == previous[0]["state_id"]
    assert previous[-1]["anchor_comparison"]["status"] == "no_resolved_change"


def test_switching_rule_replays_existing_measurements_without_new_data():
    tracker = engine(BoundaryPolicy(selected=("width",)))
    rows = [source(observed(ident="a"),1),source(observed(ident="b",new=[(2700,30,.0025),(6000,30,.0025)]),2)]
    a = tracker.evaluate(rows)
    assert not a["records"][1]["boundary"]
    b = tracker.reclassify(BoundaryPolicy(selected=("appearance",)))
    assert b["records"][1]["boundary"]
    assert a["policy_sha256"] != b["policy_sha256"]


def test_an_ordinary_arrival_does_not_require_a_complete_scan_list(monkeypatch):
    tracker = engine(BoundaryPolicy(selected=("width",)))
    tracker.append(source(observed(ident="a"),1))
    def no_batch(*args,**kwargs):
        raise AssertionError("Normal arrivals must append without a batch replay")
    monkeypatch.setattr(tracker,"evaluate",no_batch)
    assert len(tracker.append(source(observed(ident="b",seed=9),2))["records"]) == 2


def test_fixed_reference_detects_accumulated_change_while_previous_scan_is_selectable():
    rows = [source(observed(ident="a"),1),source(observed(ident="b",width=.00256),2),
            source(observed(ident="c",width=.00263),3)]
    fixed = engine(BoundaryPolicy(selected=("width",))).evaluate(rows)
    adjacent = engine(BoundaryPolicy(selected=("width",),comparison="previous")).evaluate(rows)
    assert fixed["records"][-1]["boundary"]
    assert not adjacent["records"][-1]["boundary"]


def test_late_arrival_replays_received_history_in_acquisition_order():
    policy = BoundaryPolicy(selected=("width",))
    tracker = engine(policy)
    rows = [source(observed(ident="a"),1),source(observed(ident="b",width=.0028),2),source(observed(ident="c"),3)]
    for i in [0,2,1]: report=tracker.append(rows[i])
    assert report["records"] == engine(policy).evaluate(rows)["records"]


def test_masked_or_incompatible_measurements_cannot_vote_for_a_boundary():
    a = observed()
    mask = a.valid.copy();mask[::4] = False
    b = replace(a,valid=mask)
    assert decision(a,b,BoundaryPolicy())["status"] == "uncertain"


@pytest.mark.parametrize("kwargs", [{"minimum_signals":0},{"selected":["unknown"]},
    {"selected":["width"],"minimum_signals":2},{"width_fraction":float("nan")},{"comparison":"future"}])
def test_invalid_boundary_rules_are_rejected(kwargs):
    with pytest.raises(ValueError): BoundaryPolicy(**kwargs)
