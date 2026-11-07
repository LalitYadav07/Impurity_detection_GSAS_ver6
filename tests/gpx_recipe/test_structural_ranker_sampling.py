"""Structural training examples must not learn dimension or split shortcuts."""
import numpy as np
import pytest

from gpx_recipe.preparation.structural_sampling import draw_trajectories, partition_trajectories


@pytest.mark.parametrize('modes', [1, 3, 16, 57])
def test_equal_physical_sizes_do_not_grow_with_mode_count(modes):
    # Deliberately different coordinate units and correlations; equal raw
    # Gaussian coefficients would give very different RMS displacements.
    rng = np.random.default_rng(modes)
    a = rng.normal(size=(modes, modes)); gram = a.T@a+np.eye(modes)
    data = draw_trajectories(gram, np.array([[9., 1.], [1., 4.]]),
        np.ones(modes+2), seed=23, trajectories=18, steps=5,
        displacement_range=(.02, .02), relative_metric_range=(.005, .005))
    assert data.points.shape == (90, modes+2)
    assert np.all(abs(data.points) <= 1.)
    for group in range(18):
        mask = data.trajectory == group
        expected = np.geomspace(.1, 1., 5)
        if group % 3 != 1:
            np.testing.assert_allclose(data.displacement_rms[mask], expected*.02, rtol=1e-10, atol=1e-14)
        else:
            np.testing.assert_array_equal(data.displacement_rms[mask], 0.)
        if group % 3 != 0:
            np.testing.assert_allclose(data.relative_metric_norm[mask], expected*.005, rtol=1e-10, atol=1e-14)
        else:
            np.testing.assert_array_equal(data.relative_metric_norm[mask], 0.)
    assert not data.points.flags.writeable


def test_feasibility_rejects_a_whole_trajectory_without_clipping_or_shortening():
    calls = []
    def feasible(rows):
        calls.append(rows.copy())
        return rows[:, 0] > 0
    data = draw_trajectories(np.eye(2), np.eye(1), np.ones(3),
        seed=4, trajectories=1, steps=4, feasible=feasible)
    assert data.points.shape == (4, 3)
    assert np.all(data.points[:, 0] > 0)
    assert data.rejected_directions == len(calls)-1
    assert np.all(data.trajectory == 0)


def test_infeasible_scope_is_not_a_partial_or_relabelled_dataset():
    with pytest.raises(ValueError, match='Could not complete'):
        draw_trajectories(np.eye(2), np.eye(1), np.ones(3), seed=4,
            trajectories=2, feasible=lambda rows: np.zeros(len(rows), bool), maximum_attempts=3)


@pytest.mark.parametrize('bad', [lambda rows: True, lambda rows: np.ones(len(rows)),
                                 lambda rows: np.ones(len(rows)+1, bool)])
def test_untyped_feasibility_is_not_training_truth(bad):
    with pytest.raises(ValueError, match='one boolean'):
        draw_trajectories(np.eye(1), np.eye(1), np.ones(2), seed=2, feasible=bad)


@pytest.mark.parametrize('gram', [np.zeros((2, 2)), np.array([[1., 2.], [2., 1.]]),
                                  np.array([[1., .1], [0., 1.]])])
def test_unresolved_or_nonphysical_coordinates_are_not_silently_removed(gram):
    with pytest.raises(ValueError):
        draw_trajectories(gram, np.eye(1), np.ones(3), seed=4)


def test_zero_displacement_dimension_still_samples_the_complete_cell_space():
    data = draw_trajectories(np.empty((0, 0)), np.eye(3), np.ones(3), seed=4, trajectories=8)
    assert data.points.shape == (32, 3)
    assert np.all(data.displacement_rms == 0.)
    assert np.all(data.relative_metric_norm > 0.)


def test_sampling_is_reproducible():
    args = (np.eye(3), np.eye(2), np.ones(5))
    first = draw_trajectories(*args, seed=91)
    second = draw_trajectories(*args, seed=91)
    np.testing.assert_array_equal(first.points, second.points)


@pytest.mark.parametrize('count', [3, 4, 16, 17, 100])
def test_complete_cohort_has_nonempty_disjoint_group_holdouts(count):
    groups = [f'trajectory-{i}' for i in range(count)]
    assigned = partition_trajectories(groups)
    assert set(assigned) == set(groups)
    assert set(assigned.values()) == {'train', 'validation', 'test'}
    assert assigned == partition_trajectories(list(reversed(groups))+groups)
    if count == 100:
        assert {k: list(assigned.values()).count(k) for k in set(assigned.values())} == dict(train=70, validation=15, test=15)


def test_nonindependent_cohort_cannot_create_a_test_split_by_dividing_amplitude_steps():
    with pytest.raises(ValueError, match='three independent'):
        partition_trajectories(['same-direction']*50)


@pytest.mark.parametrize('change', [{'seed': True}, {'trajectories': 0}, {'steps': 1.5},
    {'displacement_range': (0., .1)}, {'relative_metric_range': (.1, .01)},
    {'maximum_attempts': 0}])
def test_invalid_or_unbounded_sampling_requests_fail_before_generation(change):
    with pytest.raises(ValueError):
        draw_trajectories(np.eye(1), np.eye(1), np.ones(2), **dict({'seed': 4}, **change))
