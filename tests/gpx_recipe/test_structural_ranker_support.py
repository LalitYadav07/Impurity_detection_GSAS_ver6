import numpy as np
import pytest

from gpx_recipe.preparation.structural_support import distances, fit, fit_batches
from gpx_recipe.preparation.structural_training_data import feature_rows


def test_support_uses_training_rows_and_reports_validation_extrapolation():
    x = np.asarray([0., 1., 2., 10., 1000.], np.float32)[:, None, None]
    result = fit(x, np.array([True, True, True, False, False]),
                    np.array([False, False, False, True, False]), maximum_anchors=2)
    # The huge test row cannot widen the validation envelope or become a
    # support anchor; even validation data are excluded from anchor selection.
    assert all(v in [0., 1., 2.] for v in result['prototypes'][:, 0])
    expected = distances(result['prototypes'], x[[3]].reshape(1, 1))[0]
    assert result['limit'] == expected
    assert distances(result['prototypes'], x[[4]].reshape(1, 1))[0] > result['limit']
    assert result['statistical_coverage_qualified'] is False


def test_exact_prototypes_have_zero_distance_without_float32_cancellation():
    p = np.asarray([[1000., .0001], [1000., .0003]], np.float32)
    np.testing.assert_allclose(distances(p, p), 0., atol=1e-8)


def test_batched_support_excludes_test_rows_and_checks_every_validation_row():
    x = np.broadcast_to(np.asarray([0., 1., 2., 10., 20., 1000.], np.float32)[:, None, None],
                        (6, 8, 64)).copy()
    targets = np.asarray([[True, False], [False, True], [True, False],
                          [True, False], [False, True], [True, False]])
    result = fit_batches(feature_rows(x), np.asarray([0, 1, 2]), np.asarray([3, 4]),
                         targets, np.zeros((8, 1), np.float32), np.ones((8, 1), np.float32))
    assert set(result['prototypes'][:, 0]) == {0., 1., 2.}
    assert result['training_pool_rows'] == 3
    assert result['validation_rows'] == 2
    assert result['limit'] == pytest.approx(18.)
    assert distances(result['prototypes'], x[[5]].reshape(1, -1))[0] > result['limit']
    assert result['statistical_coverage_qualified'] is False


@pytest.mark.parametrize('case', ['overlap', 'no_validation', 'nonfinite', 'unbounded_anchors'])
def test_invalid_support_cannot_silently_admit_a_model(case):
    x = np.arange(12, dtype=np.float32).reshape(4, 1, 3)
    training = np.array([True, True, False, False]); validation = ~training
    kwargs = {}
    if case == 'overlap': validation[0] = True
    elif case == 'no_validation': validation[:] = False
    elif case == 'nonfinite': x[0, 0, 0] = np.nan
    else: kwargs['maximum_anchors'] = 10000
    with pytest.raises(ValueError): fit(x, training, validation, **kwargs)
