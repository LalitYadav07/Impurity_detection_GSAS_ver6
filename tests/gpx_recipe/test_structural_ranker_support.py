import numpy as np
import pytest

from gpx_recipe.preparation.structural_support import distances, fit


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
