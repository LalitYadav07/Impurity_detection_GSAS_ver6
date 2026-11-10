import numpy as np
import pytest

from gpx_recipe.preparation.structural_features import pattern_features, with_parent_reference


def test_narrow_peak_between_bin_centers_is_retained_and_integrated():
    q = np.array([0., .99, 1., 1.01, 2.]); y = np.array([0., 0., 10., 0., 0.])
    before = y.copy()
    features = pattern_features(q, y, np.array([0., 2.]))
    # Triangle area=.1, mean over width two=.05. A center-only or downsampled
    # line can lose this peak; the separate maximum channel must retain it.
    np.testing.assert_allclose(features[0], np.arcsinh(30*.05/10)/np.arcsinh(30), rtol=1e-7)
    np.testing.assert_array_equal(features[1], [1.])
    np.testing.assert_array_equal(features[2], [1.])
    np.testing.assert_array_equal(y, before)


def test_masked_gaps_are_never_bridged_or_learned_as_real_zero_signal():
    q = np.arange(4.); y = np.array([1., 1., np.nan, 100.])
    result = pattern_features(q, y, np.arange(4.), valid=np.array([True, True, False, True]))
    np.testing.assert_array_equal(result[2], [1., 0., 0.])
    np.testing.assert_array_equal(result[:2, 1:], 0.)
    masked = np.ma.array(y, mask=[False, False, True, False])
    np.testing.assert_array_equal(pattern_features(q, masked, np.arange(4.)), result)


def test_partial_grid_coverage_is_visible_without_endpoint_extrapolation():
    features = pattern_features(np.array([1., 2.]), np.ones(2), np.array([0., 1.5, 3.]))
    np.testing.assert_allclose(features[2], [1/3, 1/3], rtol=1e-7)
    np.testing.assert_array_equal(features[0], [1., 1.])


def test_scale_invariance_and_signed_background_do_not_change_missingness():
    q = np.arange(5.); y = np.array([-2., -1., 5., 1., -2.]); edges = np.linspace(0, 4, 9)
    actual = pattern_features(q, y, edges)
    np.testing.assert_allclose(pattern_features(q, y*700., edges), actual, atol=2e-7)
    assert np.any(actual[0] < 0.)
    assert np.all(actual[2] == 1.)
    assert actual.dtype == np.float32


def test_zeros_remain_finite_features_and_do_not_imply_a_phase_label():
    features = pattern_features(np.arange(5.), np.zeros(5), np.arange(5.))
    np.testing.assert_array_equal(features[:2], 0.)
    assert np.isfinite(features).all()


def test_parent_pair_retains_reference_and_zero_change_without_editing_inputs():
    q = np.arange(1., 6.); y = np.array([0., 1., 8., 1., 0.])
    parent = pattern_features(q, y, q); before = parent.copy()
    paired = with_parent_reference(parent, parent)
    assert paired.shape == (8, 4)
    np.testing.assert_array_equal(paired[:4], parent)
    np.testing.assert_array_equal(paired[4:6], parent[:2])
    np.testing.assert_array_equal(paired[6:], 0.)
    np.testing.assert_array_equal(parent, before)


def test_tiny_parent_difference_cannot_normalize_itself_to_a_strong_signal():
    q = np.arange(1., 6.); y = np.array([0., 1., 8., 1., 0.])
    parent = pattern_features(q, y, q)
    tiny = parent.copy(); tiny[0, 1] += 1e-6
    larger = parent.copy(); larger[0, 1] += .01
    weak = with_parent_reference(tiny, parent)[6:]
    strong = with_parent_reference(larger, parent)[6:]
    assert 0 < np.max(abs(weak)) < 1e-4
    assert np.max(abs(strong)) > 100*np.max(abs(weak))


def test_parent_pair_requires_shared_coordinates_and_retains_joint_missingness():
    q = np.arange(1., 6.); parent = pattern_features(q, np.ones(5), q)
    reference = parent.copy(); reference[2, 1] = 0.
    paired = with_parent_reference(parent, reference)
    assert paired[2, 1] == 0.
    np.testing.assert_array_equal(paired[[0, 1, 4, 5, 6, 7], 1], 0.)
    reference[3, 1] += .001
    with pytest.raises(ValueError, match='same Q'): with_parent_reference(parent, reference)
    reference = parent.copy(); reference[2, 1] = 2.
    with pytest.raises(ValueError): with_parent_reference(parent, reference)


def test_outside_range_peak_cannot_change_the_features_inside_the_model_range():
    q = np.arange(5.); edges = np.array([0., 1., 2.])
    first = pattern_features(q, np.array([1., 2., 3., 4., 1.]), edges)
    second = pattern_features(q, np.array([1., 2., 3., 100000., 1.]), edges)
    np.testing.assert_array_equal(first, second)


def test_supplied_background_is_subtracted_without_editing_original_measurements():
    q = np.arange(5.); signal = np.array([0., 1., 8., 1., 0.]); background = np.linspace(100, 110, 5)
    observed = signal+background; before = observed.copy()
    expected = pattern_features(q, signal, q)
    actual = pattern_features(q, observed, q, background=background)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(observed, before)
    with pytest.raises(ValueError, match='background'):
        pattern_features(q, observed, q, background=np.ones(4))


@pytest.mark.parametrize('case', ['unsorted', 'bad_edges', 'bad_y', 'float_mask', 'one_valid', 'contrast'])
def test_invalid_patterns_do_not_silently_produce_predictions(case):
    q = np.arange(4.); y = np.ones(4); edges = np.arange(4.); kwargs = {}
    if case == 'unsorted': q[2] = q[1]
    elif case == 'bad_edges': edges[2] = np.nan
    elif case == 'bad_y': y[2] = np.nan
    elif case == 'float_mask': kwargs['valid'] = np.ones(4)
    elif case == 'one_valid': kwargs['valid'] = np.array([True, False, False, False])
    elif case == 'contrast': kwargs['contrast'] = True
    with pytest.raises(ValueError): pattern_features(q, y, edges, **kwargs)
