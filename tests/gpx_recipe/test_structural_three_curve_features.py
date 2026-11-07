import numpy as np
import pytest

from gpx_recipe.preparation.structural_features import three_curve_features


def test_exactly_three_curves_share_one_scale_and_retain_a_real_intensity_change():
    q = np.arange(1., 6.); reference = np.array([1., 3., 2., 4., 1.])
    result = three_curve_features(q, 2*reference, reference, q)
    assert result['features'].shape == (3, 4)
    assert result['channels'] == ['reference', 'current', 'current_minus_reference']
    # Doubling intensity must not disappear through two independent scale
    # normalizations. Here the raw residual equals the original reference.
    np.testing.assert_array_equal(result['features'][2], result['features'][0])
    assert (result['features'][1] > result['features'][0]).all()
    assert result['intensity_scale'] == 8.
    np.testing.assert_array_equal(result['q_edges'], q)


def test_weak_changes_do_not_acquire_unit_strength_from_residual_normalization():
    q = np.arange(1., 6.); before = np.array([1., 3., 2., 4., 1.])
    weak = three_curve_features(q, before+1e-7, before, q)['features'][2]
    stronger = three_curve_features(q, before+.01, before, q)['features'][2]
    assert np.max(abs(weak)) < 1e-5
    assert np.max(abs(weak)) < np.min(abs(stronger))
    np.testing.assert_array_equal(three_curve_features(q, before, before, q)['features'][2], 0.)


def test_mask_metadata_prevents_subtracting_different_parts_of_the_two_curves():
    q = np.arange(1., 7.); curve = np.array([1., 4., 2., 6., 3., 1.])
    reference = np.ma.array(curve, mask=[False, True, False, False, False, False])
    current = np.ma.array(curve, mask=[False, False, False, False, True, False])
    result = three_curve_features(q, current, reference, np.array([1., 3., 4., 6.]))
    np.testing.assert_array_equal(result['features'][2], 0.)
    np.testing.assert_array_equal(result['coverage'], [0., 1., 0.])
    np.testing.assert_array_equal(result['features'][:, [0, 2]], 0.)
    assert result['features'].shape[0] == 3  # No mask or coordinate ML channel.


def test_background_treatment_and_compression_are_shared_without_modifying_data():
    q = np.arange(1., 5.); signal = np.array([2., 5., 1., 3.])
    result = three_curve_features(q, signal+7, signal+2, q,
        background=np.full(4, 7.), reference_background=np.full(4, 2.))
    np.testing.assert_array_equal(result['features'][2], 0.)
    with pytest.raises(ValueError): three_curve_features(q, signal, signal, q, valid=np.ones(4))
    with pytest.raises(ValueError): three_curve_features(q, signal, signal[:2], q)


def test_signed_difference_keeps_peak_loss_distinct_from_peak_emergence():
    q = np.arange(1., 5.); before = np.array([1., 2., 3., 1.]); after = np.array([2., 1., 4., 2.])
    first = three_curve_features(q, after, before, q)
    reverse = three_curve_features(q, before, after, q)
    np.testing.assert_array_equal(first['features'][2], -reverse['features'][2])
