"""Complete algebraic strain span and both signed upper-range probes."""
from types import SimpleNamespace as NS
import copy
import json
import numpy as np
import pytest

from gpx_recipe.preparation.structural_sampling import draw_trajectories,physical_grams
from gpx_recipe.preparation.structural_conditioning import ParentConditioner
from gpx_recipe.preparation.structural_parent_reference import ParentMetricSpace,child_metric,coefficients
from gpx_recipe.structural_live.optimization.physical_prior_geometry import native_reciprocal_matrix


def context(*,basis=None,mapping=None,kind='tetragonal'):
    parent=np.diag([.04,.04,.01]) if kind=='tetragonal' else np.eye(3)*.04
    r=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    operations=[np.linalg.matrix_power(r,k) for k in range(4)]
    if kind=='cubic':
        from itertools import permutations,product
        operations=[]
        for order in permutations(range(3)):
            for signs in product([-1,1],repeat=3):operations.append(np.eye(3,dtype=int)[list(order)]*np.array(signs)[:,None])
    space=ParentMetricSpace(parent,np.array(operations));basis=np.eye(3) if basis is None else np.array(basis,float)
    mapping=np.eye(6) if mapping is None else np.array(mapping,float)
    nc=mapping.shape[1];a0=coefficients(child_metric(parent,basis))
    geometry=NS(a0=a0,cell_mapping=mapping,mapping=np.empty((1,3,0)),occupancy=np.ones(1))
    source=NS(mode_dimensions=0,cell_dimensions=nc,lift=np.eye(nc),
        fast=NS(empirical_constraint=NS(geometry=geometry)))
    ctx=ParentConditioner.__new__(ParentConditioner)
    ctx.mapping=mapping;ctx.basis=basis;ctx.nc=nc;ctx.nm=0;ctx.space=space;ctx.source=source
    return ctx


@pytest.mark.parametrize('kind,dimensions',[('tetragonal',4),('cubic',5)])
@pytest.mark.parametrize('basis',[np.eye(3),[[1,2,0],[0,1,0],[0,0,1]]])
def test_all_broken_cell_directions_are_kept_in_arbitrary_candidate_basis(kind,dimensions,basis):
    ctx=context(kind=kind,basis=basis);directions=ctx.strain_directions();_,gram=physical_grams(ctx.source)
    assert directions.shape==(dimensions,6)
    np.testing.assert_allclose(directions@gram@directions.T,np.eye(dimensions),atol=1e-12)
    for direction in directions:
        delta=ctx.basis.T@native_reciprocal_matrix(ctx.mapping@direction)@ctx.basis
        delta=(delta+delta.T)/2
        np.testing.assert_allclose(ctx.space.project(delta,positive=False),0,atol=1e-14)


def test_parent_invariant_lattice_space_has_no_transition_producing_direction():
    mapping=np.zeros((6,2));mapping[:2,0]=1;mapping[2,1]=1
    ctx=context(mapping=mapping)
    assert ctx.strain_directions().shape==(0,2)


def test_orthorhombic_split_has_one_complete_opposite_axis_direction():
    ctx=context(mapping=np.eye(6)[:,:3]);directions=ctx.strain_directions()
    assert directions.shape==(1,3)
    np.testing.assert_allclose(directions[0,0],-directions[0,1],rtol=1e-12)
    assert abs(directions[0,2])<1e-14


def test_sampler_guarantees_both_signs_and_upper_range_without_extra_examples():
    kwargs=dict(seed=15127,trajectories=12,steps=4)
    old=draw_trajectories(np.eye(2),np.eye(2),np.ones(4),**kwargs)
    new=draw_trajectories(np.eye(2),np.eye(2),np.ones(4),cell_directions=np.eye(2),**kwargs)
    assert len(new.points)==len(old.points)==48
    assert new.cell_coverage['dimensions']==2 and len(new.cell_coverage['rays'])==4
    for index,ray in enumerate(new.cell_coverage['rays']):
        rows=new.points[new.trajectory==index]
        np.testing.assert_array_equal(rows[:,:2],0)
        assert ray['maximum_relative_metric_norm']==pytest.approx(.01)
        np.testing.assert_allclose(rows[-1,2:],np.eye(2)[index//2]*ray['sign']*.01)
    # Unchanged explicit legacy call is still deterministic bit for bit.
    np.testing.assert_array_equal(old.points,draw_trajectories(np.eye(2),np.eye(2),np.ones(4),**kwargs).points)


def test_rejection_changes_amplitude_only_on_that_same_signed_axis_and_parent():
    seen=[]
    def feasible(index,points):
        seen.append(index)
        return points[:,0]<=.006
    result=draw_trajectories(np.empty((0,0)),np.eye(1),np.ones(1),seed=723,trajectories=4,steps=4,
        cell_directions=np.ones((1,1)),trajectory_feasible=feasible)
    first,second=result.cell_coverage['rays']
    assert first['sign']==-1 and first['maximum_relative_metric_norm']==pytest.approx(.01)
    assert second['sign']==1 and second['maximum_relative_metric_norm']==pytest.approx(.005)
    assert second['rejected_amplitudes']==1 and seen.count(1)==2
    assert np.all(result.points[result.trajectory==1]>0)


def test_required_physical_axis_cannot_be_removed_or_replaced_when_infeasible():
    with pytest.raises(ValueError,match='required signed strain ray'):
        draw_trajectories(np.empty((0,0)),np.eye(1),np.ones(1),seed=723,trajectories=4,steps=4,
            cell_directions=np.ones((1,1)),trajectory_feasible=lambda i,p:np.zeros(len(p),bool))
    with pytest.raises(ValueError,match='both signs'):
        draw_trajectories(np.empty((0,0)),np.eye(2),np.ones(2),seed=723,trajectories=3,steps=4,
            cell_directions=np.eye(2))


def test_ray_endpoint_uses_the_current_parent_box_before_any_chemistry_rejection():
    ctx=context(mapping=np.eye(6)[:,:3]);ctx.source.radii=np.full(3,.01)
    ctx.controls=np.array([[.002,-.001,.003]])
    assert ctx.strain_ray_limit(0,np.array([1.,-1.,0.]))==pytest.approx(.008)
    assert ctx.strain_ray_limit(0,np.array([-1.,1.,0.]))==pytest.approx(.011)
    result=draw_trajectories(np.empty((0,0)),np.eye(1),np.ones(1),seed=723,trajectories=4,steps=4,
        cell_directions=np.ones((1,1)),cell_ray_limit=lambda i,d:.008)
    assert all(r['maximum_relative_metric_norm']==pytest.approx(.008) for r in result.cell_coverage['rays'])
    assert all(r['rejected_amplitudes']==0 for r in result.cell_coverage['rays'])


def packet(root):
    from test_structural_paired_blocks import block
    sampling=draw_trajectories(np.eye(1),np.eye(2),np.ones(3),seed=417,trajectories=8,steps=1,cell_directions=np.eye(2))
    def arrays(a):
        rows=np.flatnonzero(a['role']==2);a['parameters'][rows]=sampling.points
        a['displacement_rms'][rows]=sampling.displacement_rms
        a['relative_metric_norm'][rows]=sampling.relative_metric_norm
    return block(root,change=arrays,metadata=lambda d:d.update(cell_coverage=copy.deepcopy(sampling.cell_coverage)))


def test_portable_pair_reader_checks_claimed_signed_coverage_and_every_step(tmp_path):
    from gpx_recipe.preparation.structural_paired_blocks import read
    block=read(packet(tmp_path/'source'))
    assert block.manifest['cell_coverage']['dimensions']==2


@pytest.mark.parametrize('case',['axis','missing_sign','endpoint','normalization','unexpected','point'])
def test_invalid_coverage_cannot_travel_as_verified_preparation_metadata(tmp_path,case):
    from gpx_recipe.preparation.structural_paired_blocks import read
    from gpx_recipe.symmetry.compiler import file_sha
    path=packet(tmp_path/'source');d=json.loads((path/'pairs.json').read_text())
    if case=='axis':d['cell_coverage']['rays'][0]['axis']=1
    elif case=='missing_sign':d['cell_coverage']['rays'].pop()
    elif case=='endpoint':d['cell_coverage']['rays'][0]['maximum_relative_metric_norm']/=2
    elif case=='normalization':d['cell_coverage']['cell_gram'][0][0]*=2
    elif case=='unexpected':d['cell_coverage']['scientific_result']=True
    else:
        with np.load(path/'pairs.npz',allow_pickle=False) as archive:a={k:archive[k] for k in archive.files}
        a['parameters'][2,1]/=2
        np.savez_compressed(path/'pairs.npz',**a);d['pairs_sha256']=file_sha(path/'pairs.npz')
    (path/'pairs.json').write_text(json.dumps(d))
    with pytest.raises(ValueError):read(path)
