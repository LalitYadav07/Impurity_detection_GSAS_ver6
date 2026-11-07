"""Shared parent variation must not erase a child's symmetry-breaking strain."""
from types import SimpleNamespace

import numpy as np
import pytest

from gpx_recipe.preparation.structural_conditioning import condition
from gpx_recipe.preparation.structural_parent_reference import (
    ParentMetricSpace, child_metric, coefficients, parent_metric,
)
from gpx_recipe.preparation.structural_sampling import TrajectorySamples
from gpx_recipe.structural_live.optimization.physical_prior_geometry import native_reciprocal_matrix


def setup(basis=None):
    baseline = np.diag([.04, .04, .01])
    quarter = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    space = ParentMetricSpace(baseline, np.array([np.linalg.matrix_power(quarter, i) for i in range(4)]))
    basis = np.eye(3) if basis is None else basis
    mapping = np.array([[[1.], [0.], [0.]], [[-1.], [0.], [0.]]])
    geometry = SimpleNamespace(a0=coefficients(child_metric(baseline, basis)),
        cell_mapping=np.eye(6), mapping=mapping, occupancy=np.ones(2))
    source = SimpleNamespace(mode_dimensions=1, cell_dimensions=6, lift=np.eye(7),
        radii=np.full(7, .1), fast=SimpleNamespace(empirical_constraint=SimpleNamespace(geometry=geometry)))

    def check(point):
        point = np.asarray(point, float)
        if point.shape != (7,) or not np.isfinite(point).all() or np.any(abs(point) > source.radii):
            raise ValueError('Outside physical domain')
        return point

    source.check = check
    original_metrics = [baseline + np.diag([.0003, .0001, -.0001]),
                        baseline + np.diag([-.0002, .0002, .0001])]
    points = np.array([np.r_[amplitude, coefficients(child_metric(metric, basis)) - geometry.a0]
                       for amplitude, metric in zip((.001, .002), original_metrics)])
    samples = TrajectorySamples(points, np.array([0, 1]), np.zeros(2), np.zeros(2), 0)
    parents = space.draw(seed=4361, count=2, relative_range=(.001, .001))
    return source, samples, space, basis, parents, original_metrics


@pytest.mark.parametrize('basis', [np.eye(3), np.array([[1., 2, 0], [0, 1, 0], [0, 0, 1]])])
def test_conditioning_preserves_every_mode_and_broken_metric_with_shared_parent_controls(basis):
    source, samples, space, basis, parents, originals = setup(basis)
    result = condition(source, samples, space, basis, parents)
    assert result.trajectory.tolist() == [-1, 0, 0, 1, 1]
    assert result.role.tolist() == [0, 1, 2, 1, 2]
    assert result.maximum_reconstruction_error < 1e-15
    for i, row in enumerate((2, 4)):
        point = result.points[row]
        assert point[0] == samples.points[i, 0]
        expected_parent = native_reciprocal_matrix(parents[i])
        actual_child = native_reciprocal_matrix(source.fast.empirical_constraint.geometry.a0 + point[1:])
        actual_parent = parent_metric(actual_child, basis)
        np.testing.assert_allclose(space.project(actual_parent), expected_parent, rtol=1e-12, atol=1e-16)
        np.testing.assert_allclose(actual_parent - space.project(actual_parent),
            originals[i] - space.project(originals[i]), rtol=1e-12, atol=1e-16)
        null = result.points[row-1]
        assert null[0] == 0.
        np.testing.assert_allclose(parent_metric(native_reciprocal_matrix(
            source.fast.empirical_constraint.geometry.a0 + null[1:]), basis), expected_parent, atol=1e-16)
        # Occupied sites move by +/- amplitude along child a. The norm must
        # use the changed direct metric rather than the baseline cell.
        expected_rms = abs(point[0]) * np.sqrt(np.linalg.inv(actual_child)[0, 0])
        assert result.displacement_rms[row] == pytest.approx(expected_rms, rel=1e-12)
        np.testing.assert_array_equal(result.parent_a[row], result.parent_a[row-1])
    assert all(not a.flags.writeable for a in (result.points, result.parent_a, result.role, result.trajectory))


def test_conditioning_rejects_whole_block_instead_of_clipping_an_infeasible_mode():
    source, samples, space, basis, parents, _ = setup()
    source.radii[0] = .0015
    before = samples.points.copy()
    with pytest.raises(ValueError, match='physical domain'):
        condition(source, samples, space, basis, parents)
    np.testing.assert_array_equal(samples.points, before)


def test_parent_contexts_do_not_depend_on_candidate_displacement_amplitudes():
    source, samples, space, basis, parents, _ = setup()
    first = condition(source, samples, space, basis, parents)
    changed = samples.points.copy(); changed[:, 0] *= 2
    second = condition(source, TrajectorySamples(changed, samples.trajectory,
        samples.displacement_rms, samples.relative_metric_norm, 0), space, basis, parents)
    np.testing.assert_array_equal(first.parent_a, second.parent_a)
    np.testing.assert_array_equal(first.points[[1, 3]], second.points[[1, 3]])


def test_wrong_parent_setting_cannot_produce_training_pairs():
    source, samples, space, basis, parents, _ = setup()
    source.fast.empirical_constraint.geometry.a0[0] *= 1.01
    with pytest.raises(ValueError, match='baseline'):
        condition(source, samples, space, basis, parents)


def test_empty_or_partial_trajectories_cannot_become_conditioned_examples():
    source, samples, space, basis, parents, _ = setup()
    empty = TrajectorySamples(np.empty((0, 7)), np.empty(0, np.int64), np.empty(0), np.empty(0), 0)
    with pytest.raises(ValueError, match='nonempty'):
        condition(source, empty, space, basis, parents)
    partial = TrajectorySamples(samples.points, np.array([0, 2]), np.zeros(2), np.zeros(2), 0)
    with pytest.raises(ValueError, match='complete'):
        condition(source, partial, space, basis, parents)
