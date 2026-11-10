"""Shared parent variation must not erase a child's symmetry-breaking strain."""
from types import SimpleNamespace

import numpy as np
import pytest

from gpx_recipe.preparation.structural_conditioning import ParentConditioner,condition
from gpx_recipe.preparation.structural_parent_reference import (
    ParentMetricSpace, child_metric, coefficients, parent_metric,
)
from gpx_recipe.preparation.structural_sampling import TrajectorySamples,draw_trajectories
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


def test_conditioned_admission_redraws_child_rays_without_changing_shared_parents_or_bounds():
    source,_,space,basis,parents,_=setup()
    # An anisotropic child strain can fit its original box, but after adding
    # this fixed parent-only context it violates one of the same cell bounds.
    source.radii[1:]=.00025
    source.chemistry=SimpleNamespace(log_kernel=lambda rows:np.zeros(len(rows)))
    parents=np.tile(coefficients(space.baseline+np.diag([.0002,.0002,0.])),(2,1))
    context=ParentConditioner(source,space,basis,parents)
    def complete(index,rows):
        # Same metric geometry as this test's native cell mapping.
        return context.feasible(index,rows)
    settings=dict(seed=3,trajectories=2,steps=4,displacement_range=(.001,.001),
        relative_metric_range=(.0002,.0002))
    original=draw_trajectories(np.eye(1),np.eye(6),source.radii,**settings)
    with pytest.raises(ValueError,match='physical domain'):
        condition(source,original,space,basis,parents)
    samples=draw_trajectories(np.eye(1),np.eye(6),source.radii,**settings,
        trajectory_feasible=complete)
    result=condition(source,samples,space,basis,parents)
    assert samples.rejected_directions>0
    assert result.points.shape==(11,7)
    assert result.role.tolist()==[0,1,2,2,2,2,1,2,2,2,2]
    assert set(result.trajectory)=={-1,0,1}
    assert np.all(abs(result.points)<=source.radii)
    for index in range(2):
        rows=result.trajectory==index
        np.testing.assert_array_equal(result.parent_a[rows],np.tile(parents[index],(5,1)))
        np.testing.assert_array_equal(result.points[rows][1:,0],samples.points[samples.trajectory==index,0])
    np.testing.assert_array_equal(context.parents,parents)
    assert all(not value.flags.writeable for value in
        (context.parents,context.basis,context.mapping,context.controls,context.metrics))


def test_impossible_shared_parent_is_not_resampled_for_a_candidate():
    from gpx_recipe.preparation.failures import PreparationScopeError
    source,_,space,basis,parents,_=setup()
    source.radii[1:]=1e-12
    before=parents.copy()
    with pytest.raises(PreparationScopeError,match='reviewed cell bounds') as caught:
        ParentConditioner(source,space,basis,parents)
    assert caught.value.code=='parent_cell_domain_conflict'
    np.testing.assert_array_equal(parents,before)


def test_conditioned_admission_keeps_chemistry_and_propagates_technical_errors():
    source,samples,space,basis,parents,_=setup()
    source.chemistry=SimpleNamespace(log_kernel=lambda rows:np.where(rows[:,0]>.001,-np.inf,0.))
    context=ParentConditioner(source,space,basis,parents)
    assert context.feasible(0,np.repeat(samples.points,2,axis=0)).tolist()==[True,True,False,False]
    def broken(rows):raise RuntimeError('Unavailable chemistry implementation')
    source.chemistry.log_kernel=broken
    with pytest.raises(RuntimeError,match='Unavailable chemistry'):
        context.feasible(0,samples.points)


@pytest.mark.parametrize('values',[lambda rows:np.nan,lambda rows:np.ones(len(rows)+1),
                                  lambda rows:np.full(len(rows),np.inf)])
def test_unusable_conditioned_chemistry_is_not_a_physical_rejection(values):
    source,samples,space,basis,parents,_=setup()
    source.chemistry=SimpleNamespace(log_kernel=values)
    context=ParentConditioner(source,space,basis,parents)
    with pytest.raises(ValueError,match='Conditioned chemistry'):
        context.feasible(0,samples.points)
