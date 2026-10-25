"""Scientific behavior tests for observed-only transition evidence.

Synthetic cases establish controlled mechanisms; historical IPTS calibration
is a separate audit and must not be confused with these constructed examples.
"""

import math
from dataclasses import replace

import numpy as np
import pytest
from radar_pd_nova.pattern_transition import (
    ObservedPattern,
    PatternScope,
    TransitionMonitor,
    TransitionSettings,
    compare_observed_patterns,
    partial_assignment,
    prepare_pattern,
)

SCOPE = PatternScope(
    "explicit-specimen-17",
    "PG3",
    "2",
    "tof_us",
    "geometry-1",
    "reduction-1",
    "charge",
    "density",
)
CENTERS = np.array([1300, 1700, 2300, 3100, 4000, 5300, 6800, 8500], dtype=float)


def observed(
    *,
    shifts=None,
    factors=None,
    new=(),
    width=0.0025,
    noise=0.08,
    offset=15,
    gain=1,
    seed=2,
    x=None,
    scope=SCOPE,
    mask=None,
    ident="scan",
):
    x = np.geomspace(1000, 10000, 6000) if x is None else x
    y = offset + 0.0005 * x
    shifts = np.zeros(len(CENTERS)) if shifts is None else shifts
    factors = np.ones(len(CENTERS)) if factors is None else factors
    for p, shift, factor in zip(CENTERS, shifts, factors):
        y += 100 * factor * np.exp(-0.5 * ((x - p * (1 + shift)) / (p * width)) ** 2)
    for p, height, relative_width in new:
        y += height * np.exp(-0.5 * ((x - p) / (p * relative_width)) ** 2)
    y = gain * y + np.random.default_rng(seed).normal(0, noise * gain, len(x))
    return ObservedPattern(scope, x, y, np.full(len(x), noise * gain), mask, ident)


def test_partial_assignment_can_leave_both_sides_unmatched_with_equal_counts():
    m, a, b = partial_assignment([[0.1, 9], [9, 9]], 0.8)
    assert m == [(0, 0)] and a == [1] and b == [1]
    assert partial_assignment(np.full((3, 3), np.inf), 0.8) == (
        [],
        [0, 1, 2],
        [0, 1, 2],
    )


def test_assignment_does_not_impose_peak_order():
    assert partial_assignment([[9, 0.1], [0.1, 9]], 0.8)[0] == [(0, 1), (1, 0)]
    assert partial_assignment(np.empty((0, 2)), 0.8) == ([], [], [0, 1])


def test_different_directions_and_magnitudes_of_motion_are_not_a_transition():
    a = observed()
    b = observed(
        shifts=[0.01, -0.007, 0.004, -0.011, 0.008, -0.002, 0.006, -0.005], seed=3
    )
    r = compare_observed_patterns(a, b)
    assert r["status"] == "no_resolved_change", r["reasons"]
    assert r["metrics"]["min_relative_motion"] < -0.005
    assert r["metrics"]["max_relative_motion"] > 0.005
    assert r["metrics"]["matched_features"] == 8


def test_scale_background_and_negative_values_do_not_create_new_phases():
    a = observed(offset=-10)
    b = observed(gain=7, offset=1000, seed=4)
    r = compare_observed_patterns(a, b)
    assert r["status"] == "no_resolved_change", r["regions"]
    np.testing.assert_array_equal(a.intensity, a.intensity.copy())
    assert np.any(a.intensity < 0)


def test_smooth_debye_like_intensity_change_remains_a_nuisance():
    factors = np.exp(-0.2 * (CENTERS[0] / CENTERS) ** 2)
    r = compare_observed_patterns(observed(), observed(factors=factors, seed=4))
    assert r["status"] == "no_resolved_change", r["regions"]


def test_two_weak_new_peaks_are_detected_despite_very_high_global_correlation():
    a = observed()
    b = observed(new=[(2700, 8, 0.0025), (6000, 8, 0.0025)], seed=4)
    assert np.corrcoef(a.intensity, b.intensity)[0, 1] > 0.995
    r = compare_observed_patterns(a, b)
    assert r["status"] == "transition_evidence", r["metrics"]
    assert len(r["regions"]) >= 2


def test_existing_peak_intensity_redistribution_without_extra_peaks():
    a = observed()
    b = observed(factors=[1, 1, 1, 1, 1, 1, 2.5, 0.25], seed=4)
    r = compare_observed_patterns(a, b)
    assert r["status"] == "transition_evidence", r["metrics"]
    assert "relative_intensity_redistribution" in r["reasons"]
    assert r["metrics"]["before_features"] == r["metrics"]["after_features"] == 8


def test_multiple_resolved_doublets_after_independent_local_motion():
    factors = np.ones(8)
    factors[[1, 4, 6]] = 0
    new = []
    for p in CENTERS[[1, 4, 6]]:
        new.extend([(p * 0.992, 50, 0.0025), (p * 1.008, 50, 0.0025)])
    r = compare_observed_patterns(
        observed(), observed(factors=factors, new=new, seed=4)
    )
    assert r["status"] == "transition_evidence", r["metrics"]
    assert r["regions"]
    assert "phase" not in r and "propagation_vector" not in r


def test_wholesale_replacement_and_reverse_comparison():
    a = observed()
    b = observed(
        factors=[0] * 8, new=[(p * 1.10, 100, 0.0025) for p in CENTERS], seed=4
    )
    forward = compare_observed_patterns(a, b)
    reverse = compare_observed_patterns(b, a)
    assert forward["status"] == reverse["status"] == "transition_evidence"


def test_common_masked_regions_are_never_filled_or_used_as_emergent_signal():
    a = observed()
    b = observed(new=[(2700, 80, 0.0025), (6000, 80, 0.0025)], seed=4)
    mask = ~(((a.x > 2500) & (a.x < 2900)) | ((a.x > 5700) & (a.x < 6300)))
    a = replace(a, valid=mask)
    r = compare_observed_patterns(a, b)
    assert r["status"] == "no_resolved_change", r["regions"]
    assert all(
        not e["supported_in_both"] for e in r["unmatched"] if e["side"] == "after"
    )


def test_fragmented_low_count_product_is_not_reported_as_phase_disappearance():
    a = observed()
    mask = np.ones(len(a.x), dtype=bool)
    mask[::4] = False
    bad = replace(a, valid=mask)
    r = compare_observed_patterns(a, bad)
    assert r["status"] == "uncertain"
    assert r["reasons"] == ["fragmented_or_insufficient_valid_data"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("specimen", "other-specimen"),
        ("bank", "3"),
        ("geometry", "profile-2"),
        ("reduction", "background-2"),
        ("normalization", "unknown"),
        ("intensity_semantics", "bin-integrated"),
        ("coordinate", "d_angstrom"),
    ],
)
def test_incompatible_acquisitions_are_rejected_even_for_identical_curves(field, value):
    a = observed()
    b = replace(a, scope=replace(SCOPE, **{field: value}))
    r = compare_observed_patterns(a, b)
    assert r["status"] == "uncertain" and field in r["reasons"][0]


def test_rebinning_does_not_change_integrated_density_features():
    a = observed(noise=0.001)
    b = observed(x=np.geomspace(1000, 10000, 9000), noise=0.001, seed=4)
    r = compare_observed_patterns(a, b)
    assert r["status"] == "no_resolved_change", r["regions"]
    assert r["metrics"]["matched_features"] == 8


def test_reference_does_not_follow_a_gradual_onset_or_use_future_scans():
    monitor = TransitionMonitor()
    assert monitor.observe(observed(ident="0"))["status"] == "reference_initialized"
    result = None
    for i in range(1, 21):
        factors = np.ones(8)
        factors[-1] = 1 + i * 0.08
        result = monitor.observe(observed(factors=factors, ident=str(i), seed=3 + i))
    assert result["anchor_comparison"]["status"] == "transition_evidence"
    assert result["previous_comparison"]["status"] == "no_resolved_change"
    assert result["anchor_comparison"]["before"] == "0"
    assert result["physical_transition_confirmed"] is False
    monitor.reset_anchor(observed(factors=factors, ident="reviewed"))
    assert (
        monitor.observe(observed(factors=factors, ident="next", seed=40))["status"]
        == "no_resolved_change"
    )


def test_prepared_features_may_be_reused_only_for_compatible_extraction_settings():
    a = observed()
    p = prepare_pattern(a)
    classify_change = replace(TransitionSettings(), min_changed_mass=0.02)
    assert (
        compare_observed_patterns(p, p, classify_change)["status"]
        == "no_resolved_change"
    )
    with pytest.raises(ValueError, match="different sensitivity"):
        compare_observed_patterns(p, p, replace(TransitionSettings(), peak_snr=10))


def test_no_fitted_or_phase_data_are_accepted_as_observations():
    a = observed()
    for forbidden in ["ycalc", "rwp", "phases", "hkl", "lattice", "gpx"]:
        with pytest.raises(TypeError):
            ObservedPattern(a.scope, a.x, a.intensity, **{forbidden: "forbidden"})
    assert not a.x.flags.writeable and not a.intensity.flags.writeable


def test_bad_arrays_and_settings_fail_explicitly():
    a = observed()
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(a, x=a.x[::-1])
    with pytest.raises(ValueError, match="finite"):
        replace(a, intensity=np.full(len(a.x), np.nan))
    with pytest.raises(ValueError, match="explicit booleans"):
        replace(a, valid=np.ones(len(a.x), dtype=int))
    with pytest.raises(ValueError):
        TransitionSettings(motion_limit=math.inf)
    with pytest.raises(ValueError):
        TransitionSettings(baseline_bins=4)
    assert TransitionSettings(smooth_intensity_nuisance=False)


def test_flat_curves_do_not_certify_absence_of_a_transition():
    a = observed(factors=[0] * 8, noise=0.1)
    r = compare_observed_patterns(a, replace(a, observation_id="next"))
    assert r["status"] == "uncertain"


def test_changed_detection_limits_do_not_create_an_intensity_jump():
    # Same measured profiles, different counting errors. Several weak neighbors
    # cross the detection threshold only in the second scan. Integrating over
    # independently truncated feature windows would give false intensity jumps.
    a = observed(noise=0.01, new=[(p * 1.018, 1.2, 0.0015) for p in CENTERS])
    before = replace(a, sigma=np.full(len(a.x), 0.8))
    after = replace(a, sigma=np.full(len(a.x), 0.01))
    r = compare_observed_patterns(before, after)
    assert r["status"] == "no_resolved_change", r["regions"]


def test_broad_background_change_does_not_become_a_bragg_transition():
    a = observed()
    b = observed(seed=4)
    a = replace(
        a, intensity=a.intensity + 500 * np.exp(-0.5 * ((a.x - 3800) / 950) ** 2)
    )
    b = replace(
        b, intensity=b.intensity + 750 * np.exp(-0.5 * ((b.x - 3900) / 1000) ** 2)
    )
    r = compare_observed_patterns(a, b)
    assert r["status"] == "no_resolved_change", r["regions"]
    assert any("Broad/diffuse" in q for q in r["qualifications"])


def test_material_and_beamline_names_do_not_change_numerical_rules():
    a = observed()
    b = observed(
        shifts=[0.005, -0.004, 0.006, -0.003, 0.003, -0.001, 0.002, 0.004], seed=4
    )
    first = compare_observed_patterns(a, b)
    other = replace(SCOPE, instrument="NOM", specimen="unseen-material", bank="6")
    second = compare_observed_patterns(replace(a, scope=other), replace(b, scope=other))
    assert first["status"] == second["status"]
    assert first["metrics"] == second["metrics"]


def test_monitor_never_promotes_an_invalid_scan_to_a_reference():
    monitor = TransitionMonitor(max_scopes=1)
    a = observed(ident="good")
    monitor.observe(a)
    bad = replace(a, valid=np.arange(len(a.x)) % 4 != 0, observation_id="invalid")
    assert monitor.observe(bad)["status"] == "uncertain"
    final = monitor.observe(observed(ident="next", seed=9))
    assert final["previous_comparison"]["before"] == "good"
    with pytest.raises(ValueError, match="scope limit"):
        monitor.observe(replace(a, scope=replace(SCOPE, specimen="other")))
    monitor.close_scope(SCOPE)
    assert (
        monitor.observe(replace(a, scope=replace(SCOPE, specimen="other")))["status"]
        == "reference_initialized"
    )


def test_monitor_distinguishes_new_changes_from_a_stable_changed_state_and_return():
    monitor = TransitionMonitor()
    original = observed(ident="reference")
    changed = observed(factors=[1, 1, 1, 1, 1, 1, 2.5, 0.25], ident="new-state", seed=6)
    monitor.observe(original)
    onset = monitor.observe(changed)
    assert onset["new_change_evidence"] and onset["change_since_anchor"]
    stable = monitor.observe(replace(changed, observation_id="same-state"))
    assert stable["status"] == "no_resolved_change"
    assert not stable["new_change_evidence"] and stable["change_since_anchor"]
    returned = monitor.observe(replace(original, observation_id="returned"))
    assert returned["new_change_evidence"] and not returned["change_since_anchor"]
