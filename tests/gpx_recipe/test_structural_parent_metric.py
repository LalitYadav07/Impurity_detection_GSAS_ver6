"""Parent reference projection must retain transition-producing strain."""
import numpy as np
import pytest

from gpx_recipe.preparation.structural_parent_reference import (
    ParentMetricSpace,coefficients,parent_metric,child_metric,
)
from gpx_recipe.structural_live.optimization.physical_prior_geometry import native_reciprocal_matrix


def tetragonal():
    quarter=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    rotations=np.asarray([np.linalg.matrix_power(quarter,i) for i in range(4)])
    return np.diag([.04,.04,.01]),rotations


def test_normal_anisotropic_parent_change_is_kept_but_split_producing_strain_is_not_removed():
    g,r=tetragonal();space=ParentMetricSpace(g,r)
    thermal=np.diag([.0001,.0001,-.00003])
    split=np.diag([.00004,-.00004,0.])
    result=space.project(g+thermal+split)
    np.testing.assert_allclose(result,g+thermal,rtol=0,atol=1e-16)
    assert space.norm(g+thermal+split-result)>0
    assert space.basis.shape==(6,2)
    with pytest.raises(ValueError,match='complete parent metric symmetry'):space.require(g+thermal+split)


def test_monoclinic_reference_preserves_allowed_shear_and_all_three_lengths():
    g=np.array([[.04,0,.002],[0,.03,0],[.002,0,.01]])
    r=np.asarray([np.eye(3,dtype=int),np.diag([-1,1,-1])]);space=ParentMetricSpace(g,r)
    allowed=np.array([[.0001,0,.00003],[0,-.00002,0],[.00003,0,.00001]])
    broken=np.array([[0,.00003,0],[.00003,0,.00004],[0,.00004,0]])
    np.testing.assert_allclose(space.project(g+allowed+broken),g+allowed,rtol=0,atol=1e-16)
    assert space.basis.shape==(6,4)


def test_projection_commutes_with_a_skew_change_of_parent_basis():
    g,r=tetragonal();space=ParentMetricSpace(g,r)
    t=np.array([[1,2,0],[0,1,0],[0,0,1.]])
    inverse=np.linalg.inv(t);transformed_r=inverse.T@r@t.T
    transformed=ParentMetricSpace(child_metric(g,t),transformed_r)
    changed=g+np.array([[.0002,.00001,.00002],[.00001,-.0001,.00003],[.00002,.00003,.0001]])
    np.testing.assert_allclose(transformed.project(child_metric(changed,t)),child_metric(space.project(changed),t),rtol=1e-12,atol=1e-16)
    np.testing.assert_allclose(parent_metric(child_metric(changed,t),t),changed,rtol=1e-12,atol=1e-16)


def test_parent_draws_use_equal_physical_norm_and_reproducible_full_symmetry():
    g,r=tetragonal();space=ParentMetricSpace(g,r)
    a=space.draw(seed=83621,count=16,relative_range=(.001,.001))
    b=space.draw(seed=83621,count=16,relative_range=(.001,.001))
    np.testing.assert_array_equal(a,b)
    for point in a:
        metric=native_reciprocal_matrix(point);space.require(metric)
        assert space.norm(metric-g)==pytest.approx(.001,rel=1e-10)


def test_new_parent_sampling_covers_thermal_offsets_and_keeps_saved_explicit_ranges():
    from gpx_recipe.preparation.structural_domain import DEFAULT_PARENT_METRIC_RANGE
    g,r=tetragonal();space=ParentMetricSpace(g,r)
    old_interval=(.0001,.001)
    old=space.draw(seed=83621,count=64,relative_range=old_interval)
    new=space.draw(seed=83621,count=64)
    explicit_new=space.draw(seed=83621,count=64,relative_range=DEFAULT_PARENT_METRIC_RANGE)
    np.testing.assert_array_equal(new,explicit_new)
    old_norms=np.array([space.norm(native_reciprocal_matrix(a)-g) for a in old])
    new_norms=np.array([space.norm(native_reciprocal_matrix(a)-g) for a in new])
    assert old_norms.max()<=old_interval[1]+1e-12
    assert .003<new_norms.max()<=DEFAULT_PARENT_METRIC_RANGE[1]+1e-12
    # New preparation does not mutate an explicit old request or draw a
    # candidate-specific parent distribution. Every point retains symmetry.
    np.testing.assert_array_equal(old,space.draw(seed=83621,count=64,relative_range=old_interval))
    for a in new:space.require(native_reciprocal_matrix(a))


def test_parent_metric_context_owns_immutable_scientific_inputs():
    g,r=tetragonal();space=ParentMetricSpace(g,r)
    baseline=space.baseline.copy();operations=space.operations.copy()
    g[:]=0.;r[:]=0
    np.testing.assert_array_equal(space.baseline,baseline)
    np.testing.assert_array_equal(space.operations,operations)
    assert not space.baseline.flags.writeable and not space.operations.flags.writeable and not space.basis.flags.writeable


@pytest.mark.parametrize('change',['not_closed','duplicate','singular','fractional','wrong_metric'])
def test_invalid_group_cannot_define_a_parent_reference(change):
    g,r=tetragonal()
    if change=='not_closed':r=r[:2]
    elif change=='duplicate':r=np.concatenate([r,r[:1]])
    elif change=='singular':r=np.zeros((1,3,3))
    elif change=='fractional':r=r.astype(float);r[0,0,0]=.5
    else:g[0,0]*=1.1
    with pytest.raises(ValueError):ParentMetricSpace(g,r)
