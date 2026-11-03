"""Deterministic audit regressions; injected scores never open model weights."""

from copy import deepcopy

import numpy as np
import pytest

from radar_pd_nova.unet_core import production
from radar_pd_nova.unet_core.lag_morphology import _corr, lag_signature
from radar_pd_nova.unet_core.morphology_type_fallback import tag_unresolved_event
from radar_pd_nova.unet_core.native_type_policy import tag_event


def pattern(x, y, name, *, valid=None, sigma=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    scope = {field: "declared" for field in (
        "specimen", "instrument", "bank", "geometry", "reduction",
        "normalization", "intensity_semantics")}
    scope["coordinate"] = "q_inv_angstrom"
    return {"id": name, "scope": scope, "x": x, "y": y, "intensity": y,
            "valid": np.ones(len(x), bool) if valid is None else valid,
            "sigma": np.ones(len(x)) if sigma is None else sigma}


def probabilities(batch, attention):
    scores = np.broadcast_to(np.asarray(attention, np.float32), (len(batch), 156)).copy()
    return {"attention": scores, "type_probabilities": {
        name: np.zeros_like(scores) for name in production.FEATURE_TYPES}}


def predict(a, b, scores):
    predictor = production.FocusTransitionPredictor()
    predictor.predict_batch = scores
    return predictor.predict_pair(a, b)


def test_invalid_shared_padding_does_not_change_alignment_or_prediction():
    results, matches, batches = [], [], []
    for padded in (False, True):
        curves = []
        for name, shift, intensity in (("a", 0., 0.), ("b", .5, 20.)):
            x = np.arange(1000., 1100.) + shift
            valid = np.ones(len(x), bool)
            if padded:
                x = np.r_[np.arange(1., 201.), x]
                valid = np.r_[np.zeros(200, bool), valid]
            curves.append(pattern(x, np.full(len(x), intensity), name, valid=valid))
        captured = []

        def scores(batch):
            captured.append(batch.copy())
            attention = np.zeros(156, np.float32)
            attention[30:125] = .99
            return probabilities(batch, attention)

        results.append(predict(*curves, scores))
        matches.append(production._paired_native(*curves, (1000.5, 1099.)))
        batches.append(np.concatenate(captured))
    np.testing.assert_array_equal(batches[0], batches[1])
    for field in ("x", "good", "direct_z"):
        np.testing.assert_array_equal(matches[0][field], matches[1][field])
    assert np.sum(matches[1]["good"]) > 90
    assert all(result["attention"] == "yes" for result in results)
    assert results[0]["events"] == results[1]["events"]


def test_exact_alignment_ignores_shared_valid_points_outside_target():
    x = np.arange(1., 401.)
    a = pattern(x, np.zeros(len(x)), "a")
    b = pattern(x, np.full(len(x), 20.), "b")
    matched = production._paired_native(a, b, (100., 150.))
    assert len(matched["x"]) == 51
    assert np.all((matched["x"] >= 100.) & (matched["x"] <= 150.))


@pytest.mark.parametrize("tile_score,second_score,expected", [
    (.95, .4, "uncertain"), (.55, 0., "no")])
def test_final_uncertainty_uses_fused_overlapping_scores(tile_score, second_score, expected):
    x = np.arange(1., 227.)
    a, b = pattern(x, np.zeros(len(x)), "a"), pattern(x, np.full(len(x), 20.), "b")

    def disagree(batch):
        attention = np.zeros((len(batch), 156), np.float32)
        attention[0, 100:125] = tile_score
        attention[1, :] = second_score
        return probabilities(batch, attention)

    result = predict(a, b, disagree)
    assert result["events"] == []
    assert result["attention"] == expected
    maximum = result["postprocessing"]["maximum_supported_fused_attention"]
    if expected == "uncertain":
        assert result["status"] == "uncertain"
        assert "borderline_attention_score" in result["reasons"]
        assert .35 <= maximum < .8
    else:
        assert result["status"] == "valid" and result["reasons"] == []
        assert maximum < .35


def test_high_model_proposals_rejected_by_native_evidence_have_a_reason():
    x = np.arange(1., 201.)
    a, b = pattern(x, np.zeros(len(x)), "a"), pattern(x, np.ones(len(x)), "b")
    result = predict(a, b, lambda batch: probabilities(batch, .95))
    assert result["status"] == result["attention"] == "uncertain"
    assert result["events"] == []
    assert "local_model_proposals_rejected_by_measured_evidence" in result["reasons"]
    assert result["postprocessing"]["rejected_local_boxes"]["insufficient_native_evidence"] > 0


def test_uniform_fused_borderline_and_low_scores_remain_distinct():
    x = np.arange(1., 227.)
    a, b = pattern(x, np.zeros(len(x)), "a"), pattern(x, np.full(len(x), 20.), "b")
    borderline = predict(a, b, lambda batch: probabilities(batch, .65))
    low = predict(a, b, lambda batch: probabilities(batch, .1))
    assert borderline["attention"] == "uncertain"
    assert borderline["reasons"] == ["borderline_attention_score"]
    assert low["attention"] == "no" and low["reasons"] == []


def test_morphology_label_and_lag_are_invariant_to_intensity_units():
    x = np.linspace(1., 3., 201)
    a = 20. + 100. * np.exp(-.5 * ((x - 2.) / .06) ** 2)
    b = 20. + 100. * np.exp(-.5 * ((x - 2.04) / .06) ** 2)
    event = {"x0": 1.7, "x1": 2.4, "attention_score": .95,
             "feature_type_scores": {name: .6 if name == "peak_shift" else 0.
                                     for name in production.FEATURE_TYPES},
             "feature_types": [], "type_status": "uncertain"}
    signatures = []
    for gain in (1., 2. ** -27, 1e-150, 1e150):
        matched = {"x": x, "good": np.ones(len(x), bool),
                   "before_y": a * gain, "after_y": b * gain,
                   "before_sigma": np.ones(len(x)) * gain,
                   "after_sigma": np.ones(len(x)) * gain,
                   "direct_z": (b - a) / np.sqrt(2.)}
        native, _ = tag_event(deepcopy(event), matched)
        result, _ = tag_unresolved_event(native, matched)
        assert result["feature_types"] == ["peak_shift"]
        signatures.append(lag_signature(event, matched))
    assert all(signature["best_lag_bins"] == 4 for signature in signatures)
    np.testing.assert_allclose([signature["best_corr"] for signature in signatures], 1., atol=1e-3)


def test_correlation_handles_flat_vectors_and_extreme_scales():
    a = np.array([1., 2., 4., 3.])
    for gain in (1e-300, 1., 1e300):
        assert _corr(a * gain, a * gain) == pytest.approx(1.)
        assert _corr(a * gain, -a * gain) == pytest.approx(-1.)
        assert _corr(np.ones(4) * gain, a * gain) == 0.
