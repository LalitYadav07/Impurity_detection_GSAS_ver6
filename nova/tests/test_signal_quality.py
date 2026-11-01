"""Fixed uncertainty-aware zero-model bound; synthetic inputs only."""

import json

import numpy as np
import pytest

from radar_pd_nova.signal_quality import supplied_error_signal_quality


def test_entire_signal_below_unit_norm_requires_review_without_phase_claim():
    result = supplied_error_signal_quality([0.1, -0.2, 0.3], [1.0, 1.0, 1.0])
    assert result["point_count"] == 3
    assert result["total_norm_squared"] == pytest.approx(0.14)
    assert result["max_abs_snr"] == pytest.approx(0.3)
    assert result["unresolved"] is True and result["needs_review"] is True
    assert "under the supplied errors" in result["message"]
    assert "small Rwp does not establish phase identification" in result["message"]
    assert "does not establish phase absence" in result["message"]


@pytest.mark.parametrize("scale", [1e-12, 1e-4, 1.0, 1e4, 1e12])
def test_common_units_or_count_scale_does_not_change_evidence(scale):
    y = np.array([0.1, -0.2, 0.3])
    sigma = np.array([1.0, 2.0, 1.5])
    baseline = supplied_error_signal_quality(y, sigma)
    actual = supplied_error_signal_quality(y * scale, sigma * scale)
    assert actual["status"] == baseline["status"]
    assert actual["total_norm_squared"] == pytest.approx(baseline["total_norm_squared"])
    assert actual["max_abs_snr"] == pytest.approx(baseline["max_abs_snr"])


def test_masked_signal_cannot_make_selected_support_resolved():
    result = supplied_error_signal_quality([0.1, 1000.0], [1.0, 1.0], [True, False])
    assert result["point_count"] == 1 and result["masked_point_count"] == 1
    assert result["total_norm_squared"] == pytest.approx(0.01)
    assert result["needs_review"] is True


def test_collectively_resolved_weak_points_are_not_flagged():
    result = supplied_error_signal_quality([0.25] * 20, [1.0] * 20)
    assert result["max_abs_snr"] == 0.25
    assert result["total_norm_squared"] == 1.25
    assert result["needs_review"] is False
    assert result["unresolved"] is False


def test_strong_background_does_not_prove_bragg_signal_or_phase_identity():
    result = supplied_error_signal_quality([100.0] * 10, [1.0] * 10)
    assert result["status"] == "not_flagged"
    assert result["needs_review"] is False
    assert "does not establish phase identity" in result["message"]
    assert "Bragg-signal significance" in result["message"]


def test_unit_norm_boundary_is_not_below_one():
    result = supplied_error_signal_quality([1.0], [1.0])
    assert result["total_norm_squared"] == 1.0
    assert result["unresolved"] is False


@pytest.mark.parametrize(
    "y,sigma,valid",
    [([], [], None), ([np.nan], [1.0], None), ([1.0], [0.0], None), ([1.0], [np.inf], None), ([100.0], [1.0], [False])],
)
def test_no_finite_usable_support_remains_unknown(y, sigma, valid):
    result = supplied_error_signal_quality(y, sigma, valid)
    assert result["point_count"] == 0
    assert result["status"] == "unavailable"
    assert result["unresolved"] is None
    assert result["needs_review"] is True
    assert result["total_norm_squared"] is None


def test_nonfinite_pairs_do_not_contaminate_valid_support():
    result = supplied_error_signal_quality([0.1, np.nan, 5.0, 2.0], [1.0, 1.0, -1.0, np.inf])
    assert result["point_count"] == 1 and result["invalid_point_count"] == 3
    assert result["total_norm_squared"] == pytest.approx(0.01)


def test_overflow_is_json_safe_and_cannot_trigger_unresolved_bound():
    result = supplied_error_signal_quality([1e300], [1e-300])
    assert result["norm_overflow"] is True
    assert result["unresolved"] is False and result["needs_review"] is False
    json.dumps(result, allow_nan=False)


def test_inputs_are_not_mutated():
    y = np.array([0.1, -0.2]); sigma = np.array([1.0, 2.0]); valid = np.array([True, False])
    copies = [array.copy() for array in (y, sigma, valid)]
    supplied_error_signal_quality(y, sigma, valid)
    for value, original in zip((y, sigma, valid), copies):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize("y,sigma,valid", [([1.0], [1.0, 2.0], None), ([[1.0]], [[1.0]], None), ([1.0], [1.0], [True, False])])
def test_mismatched_support_raises_instead_of_broadcasting(y, sigma, valid):
    with pytest.raises(ValueError):
        supplied_error_signal_quality(y, sigma, valid)
