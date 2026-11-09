"""Measured references must preserve real changes and independently masked grids."""
import numpy as np
import pytest

from gpx_recipe.preparation.structural_features import (
    IndependentCurveProjector, independent_curve_features, three_curve_features,
)


def test_same_grid_matches_original_feature_bytes_with_masks_and_signed_values():
    rng = np.random.default_rng(765041)
    q = np.linspace(.5, 8, 401); edges = np.linspace(.6, 7.9, 129)
    before = rng.normal(2, 3, len(q)); after = rng.normal(4, 3, len(q))
    mask = rng.random(len(q)) > .1
    for contrast in (1., 30., 1000.):
        actual = independent_curve_features(q, after, q, before, edges,
            valid=mask, reference_valid=mask, contrast=contrast)
        expected = three_curve_features(q, after, before, edges, valid=mask, contrast=contrast)
        for key in ('features', 'coverage', 'q_edges'):
            assert actual[key].tobytes() == expected[key].tobytes()
        assert actual['intensity_scale'] == expected['intensity_scale']


def test_different_grids_integrate_the_same_curve_without_peak_registration():
    q = np.array([1., 1.2, 1.8, 2.7, 4., 5.])
    reference_q = np.array([1., 1.4, 2.2, 3.6, 4.8, 5.])
    edges = np.linspace(1, 5, 65)
    before = 2*reference_q-3; after = 2*q-3
    original = (q.copy(), reference_q.copy(), before.copy(), after.copy())
    result = independent_curve_features(q, after, reference_q, before, edges)
    np.testing.assert_array_equal(result['coverage'], 1.)
    np.testing.assert_allclose(result['features'][2], 0., atol=2e-14)
    assert result['features'].shape == (3, 64)
    for actual, expected in zip((q, reference_q, before, after), original):
        np.testing.assert_array_equal(actual, expected)


def test_common_valid_intervals_are_intersections_not_minimum_coverage():
    # In the first bin the before curve covers [1,2], while after covers
    # [2,3]. Each alone has 50% coverage, but their intersection has none.
    q = np.arange(1., 7.); edges = np.array([1., 3., 4., 6.])
    reference_mask = np.array([True, True, False, True, True, True])
    current_mask = np.array([False, True, True, True, True, True])
    result = independent_curve_features(q, 2*q, q, 2*q, edges,
        valid=current_mask, reference_valid=reference_mask)
    np.testing.assert_array_equal(result['coverage'], [0., 0., 1.])
    np.testing.assert_array_equal(result['features'][:, :2], 0.)
    np.testing.assert_array_equal(result['features'][2], 0.)


def test_partial_overlap_uses_identical_intervals_in_both_integrals():
    q = np.arange(1., 7.); reference_q = np.array([1., 1.5, 2., 3., 4., 5., 6.])
    mask = np.array([True, False, True, True, True, True, True])
    result = independent_curve_features(q, q, reference_q, reference_q,
        np.array([1., 3., 5.]), reference_valid=mask)
    np.testing.assert_array_equal(result['coverage'], [.5, 1.])
    np.testing.assert_array_equal(result['features'][2], 0.)


def test_no_common_valid_support_is_not_an_unchanged_pattern():
    q = np.arange(1., 5.)
    with pytest.raises(ValueError, match='no shared valid'):
        independent_curve_features(q, q, q+10, q, np.linspace(1, 14, 65))
    with pytest.raises(ValueError, match='no shared valid'):
        independent_curve_features(q, q, q, q, [1., 4.],
            valid=np.array([True, True, False, False]),
            reference_valid=np.array([False, False, True, True]))


def test_subtraction_preserves_intensity_changes_split_peaks_and_direction():
    q = np.linspace(1, 5, 4001); edges = np.linspace(1, 5, 1001)
    gaussian = lambda center: np.exp(-((q-center)/.02)**2)
    before = gaussian(3); after = .5*gaussian(2.97)+.5*gaussian(3.03)
    first = independent_curve_features(q, after, q, before, edges)
    reverse = independent_curve_features(q, before, q, after, edges)
    np.testing.assert_array_equal(first['features'][2], -reverse['features'][2])
    difference = first['features'][2]
    centers = (edges[:-1]+edges[1:])*.5
    assert difference[np.argmin(abs(centers-3))] < -.5
    assert difference[np.argmin(abs(centers-2.97))] > .5
    assert difference[np.argmin(abs(centers-3.03))] > .5
    doubled = independent_curve_features(q, before*2, q, before, edges)
    np.testing.assert_array_equal(doubled['features'][2], doubled['features'][0])


def test_reused_reference_detects_mutation_and_grid_ownership():
    q = np.linspace(1, 5, 401); reference_q = np.linspace(1, 5, 501)
    edges = np.linspace(1, 5, 129); before = 2*reference_q; after = 2*q
    projector = IndependentCurveProjector(q, reference_q, edges)
    q[:] = 0; reference_q[:] = 0; edges[:] = 0
    first = projector.features(after, before)
    first['features'][:] = 100; first['q_edges'][:] = 0
    before += .25
    updated = projector.features(after, before)
    assert (updated['features'][2] < 0).all()
    assert np.all(np.diff(updated['q_edges']) > 0)
    mask = np.ones(len(before), bool); mask[20:40] = False; before[30] = np.nan
    masked = projector.features(after, before, reference_valid=mask)
    assert (masked['coverage'] < 1).any()
    with pytest.raises(ValueError, match='nonfinite'):
        projector.features(after, before)


@pytest.mark.parametrize('case', ['unsorted', 'wrong_shape', 'float_mask', 'background', 'contrast'])
def test_invalid_measured_metadata_cannot_enter_features(case):
    q = np.arange(1., 7.); before = q.copy(); after = q.copy(); options = {}
    if case == 'unsorted': q = q[::-1]
    elif case == 'wrong_shape': before = before[:-1]
    elif case == 'float_mask': options['reference_valid'] = np.ones(6)
    elif case == 'background': options['reference_background'] = np.ones((6, 1))
    else: options['contrast'] = float('nan')
    with pytest.raises(ValueError):
        independent_curve_features(q, after, np.arange(1., 7.), before, [1., 6.], **options)
