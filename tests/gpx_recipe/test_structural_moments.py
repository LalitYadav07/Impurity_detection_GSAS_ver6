"""Certificates may accelerate statistics, never change returned float32 bytes."""
import numpy as np
import pytest

from gpx_recipe.preparation import structural_moments as moments
from gpx_recipe.preparation.structural_training_data import channel_statistics, feature_rows


def captured(values):
    result = moments.Moments(len(values), values.shape[2])
    for value in values:
        result.append(value)
    result.seal()
    return result


@pytest.mark.parametrize('bins', [64, 127, 4096, 8192])
@pytest.mark.parametrize('kind', ['normal', 'constant', 'cancellation', 'large', 'small', 'near_constant'])
def test_any_certificate_matches_original_two_pass_bytes(bins, kind):
    rng = np.random.default_rng(15261)
    values = rng.normal(size=(129, 3, bins)).astype(np.float32)
    if kind == 'constant': values[:] = np.array([0., 1., -.1], np.float32)[:, None]
    elif kind == 'cancellation':
        values[:, :, ::2] *= 1e10; values[:, :, 1::2] = -values[:, :, ::2][:, :, :values[:, :, 1::2].shape[2]]
    elif kind == 'large': values *= np.float32(1e35)
    elif kind == 'small': values *= np.float32(1e-40)
    elif kind == 'near_constant': values = 1.+values*np.float32(2.**-23)
    summary = captured(values)
    order = rng.permutation(len(values))
    for minimum in (.01, .02, 1.):
        for center in (False, True):
            result = summary.statistics(order, minimum_scale=minimum, center=center)
            expected = channel_statistics(feature_rows(values), order, minimum_scale=minimum, center=center)
            if result is not None:
                assert all(a.tobytes() == b.tobytes() for a, b in zip(result, expected))
    # No statistical approximation can influence prototype ties or ordering.
    with np.errstate(over='ignore'):
        expected_strength = np.linalg.norm(values[order, 2], axis=1)
    np.testing.assert_array_equal(summary.strengths(order), expected_strength)
    assert summary.nbytes == len(values)*moments.BYTES_PER_ROW


def test_common_data_certify_without_reconstruction_and_no_holdout_statistics_enter():
    rng = np.random.default_rng(82367)
    values = rng.normal(size=(257, 3, 2048)).astype(np.float32)
    summary = captured(values)
    order = np.array([9, 2, 0, 4, 3, 7, 5], np.int64)
    result = summary.statistics(order, minimum_scale=.01, center=False)
    assert result is not None
    expected = channel_statistics(feature_rows(values), order, minimum_scale=.01, center=False)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(result, expected))
    changed = values.copy(); changed[10:] *= 1e10
    other = captured(changed).statistics(order, minimum_scale=.01, center=False)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(result, other))


def test_rounding_tie_and_cancellation_must_not_receive_a_false_certificate():
    rng = np.random.default_rng(301)
    certified = fallback = 0
    # Sweep exact float32 scale boundaries. Reaching a boundary is a reason
    # to replay the original algorithm, never to choose either neighboring bit.
    for step in range(100):
        target = np.float32(.01+step*.0023)
        right = np.nextafter(target, np.float32(np.inf))
        scale = (np.float64(target)+np.float64(right))/2
        values = rng.choice([-1., 1.], size=(3, 3, 64)).astype(np.float32)
        values *= np.float32(scale)
        result = captured(values).statistics(np.arange(3), minimum_scale=.01, center=False)
        expected = channel_statistics(feature_rows(values), np.arange(3), minimum_scale=.01, center=False)
        if result is None: fallback += 1
        else:
            certified += 1
            assert all(a.tobytes() == b.tobytes() for a, b in zip(result, expected))
    assert certified+fallback == 100


def test_a_scale_at_a_float32_rounding_boundary_requires_original_replay():
    left=np.float32(.1);right=np.nextafter(left,np.float32(np.inf))
    values=np.tile(np.array([left,-left,right,-right],np.float32),16)
    values=np.broadcast_to(values,(3,3,64)).copy()
    # Equal squared contributions put the standard deviation within a few
    # binary64 ulps of the midpoint of two neighboring float32 values.
    # A wide enough enclosure must decline the certificate.
    assert captured(values).statistics(np.arange(3),minimum_scale=.01,center=False) is None


def test_unknown_arithmetic_uses_original_path(monkeypatch):
    values = np.random.default_rng(32).normal(size=(3, 3, 64)).astype(np.float32)
    summary = captured(values)
    monkeypatch.setattr(moments, '_rounding_supported', lambda: False)
    assert summary.statistics(np.arange(3), minimum_scale=.01, center=False) is None
    assert summary.strengths(np.arange(3)) is None


def test_repeated_rows_retain_the_original_count_and_normalization():
    values=np.random.default_rng(193).normal(size=(3,3,128)).astype(np.float32)
    order=np.array([1,1,0,1,2,1,1,0,0],np.int64)
    result=captured(values).statistics(order,minimum_scale=.01,center=True)
    assert result is not None
    expected=channel_statistics(feature_rows(values),order,minimum_scale=.01,center=True)
    assert all(a.tobytes()==b.tobytes() for a,b in zip(result,expected))


def test_summary_inventory_is_bounded_and_sealed():
    with pytest.raises(ValueError, match='bounded inventory'):
        moments.Moments(moments.MAX_BYTES//moments.BYTES_PER_ROW+1, 64)
    summary = captured(np.zeros((1, 3, 64), np.float32))
    with pytest.raises(ValueError, match='Capture complete'):
        summary.append(np.zeros((3, 64), np.float32))
    with pytest.raises(ValueError): summary._values[0, 0] = 1.
    for indices in (np.array([-1]), np.array([1]), np.array([0.]), np.empty(0, np.int64)):
        with pytest.raises(ValueError, match='bounded owned'):
            summary.statistics(indices, minimum_scale=.01, center=False)
