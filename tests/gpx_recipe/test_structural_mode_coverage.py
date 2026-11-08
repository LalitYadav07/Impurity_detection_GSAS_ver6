"""Additive physical coverage must retain every original point and group."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from gpx_recipe.preparation.structural_mode_coverage import supplement
from gpx_recipe.preparation.structural_sampling import from_source,physical_grams


def source():
    mapping=np.zeros((2,3,2))
    mapping[0,:,0]=[1.,0.,0.];mapping[1,:,0]=[-.5,0.,0.]
    mapping[0,:,1]=[.4,1.,0.];mapping[1,:,1]=[0.,.3,0.]
    cell=np.zeros((6,1));cell[0]=.1
    geometry=NS(a0=np.r_[np.ones(3),np.zeros(3)],mapping=mapping,
        occupancy=np.ones(2),cell_mapping=cell)
    fast=NS(nm=2,empirical_constraint=NS(geometry=geometry),basis={'gauge_basis':np.empty((2,0))},
        meta={'inventory':{'native_mode_labels':['native-a','native-b']}})
    return NS(fast=fast,mode_dimensions=2,cell_dimensions=1,dimension=3,
        lift=np.eye(3),radii=np.ones(3),native_mode_map=np.eye(2),frozen_modes=[],
        chemistry=NS(log_kernel=lambda points:np.zeros(len(points))))


def portfolio(s):
    return from_source(s,seed=17,trajectories=6,steps=4,
        displacement_range=(.002,.002),relative_metric_range=(.0001,.0001))


def test_every_original_point_and_group_is_retained_before_complete_signed_axis_rays():
    s=source();base=portfolio(s)
    result=supplement(s,base,seed=19,steps=4,contexts=3,displacement_range=(.005,.005))
    n=len(base.points);assert result.base_examples==n
    for name in ['points','trajectory','displacement_rms','relative_metric_norm']:
        original=getattr(base,name);actual=getattr(result.samples,name)
        np.testing.assert_array_equal(actual[:n],original)
        assert not actual.flags.writeable
    assert len(result.rays)==12 and len(result.samples.points)==n+48
    assert result.gauge_axes==() and result.labels==('native-a','native-b')
    assert result.samples.rejected_directions==base.rejected_directions
    for mode in range(2):
        records=[row for row in result.rays if row['native_mode']==mode]
        assert len({r['parent'] for r in records})==3
        for parent in {r['parent'] for r in records}:
            assert {r['sign'] for r in records if r['parent']==parent}=={-1,1}
    for ray in result.rays:
        low,high=ray['rows'];points=result.samples.points[low:high]
        assert high-low==4 and np.all(result.samples.trajectory[low:high]==ray['parent'])
        np.testing.assert_array_equal(points[:,2],0.)
        np.testing.assert_array_equal(points[:,1-ray['native_mode']],0.)
        assert np.all(np.sign(points[:,ray['native_mode']])==ray['sign'])
        np.testing.assert_allclose(result.samples.displacement_rms[low:high],np.geomspace(.1,1.,4)*.005)
    replay=supplement(s,base,seed=19,steps=4,contexts=3,displacement_range=(.005,.005))
    np.testing.assert_array_equal(replay.samples.points,result.samples.points)
    assert replay.rays==result.rays


def test_rejection_changes_amplitude_without_changing_axis_sign_or_shared_parent():
    s=source();base=portfolio(s);gram,_=physical_grams(s);calls=[]
    def feasible(parent,points):
        calls.append((parent,points.copy()))
        return np.einsum('ni,ij,nj->n',points[:,:2],gram,points[:,:2])<=.003**2
    result=supplement(s,base,seed=19,contexts=3,displacement_range=(.002,.008),trajectory_feasible=feasible)
    assert result.rejected_amplitudes>0
    assert result.rejected_amplitudes==sum(r['rejected_amplitudes'] for r in result.rays)
    assert len(calls)==len(result.rays)+result.rejected_amplitudes
    cursor=0
    for ray in result.rays:
        for parent,points in calls[cursor:cursor+ray['rejected_amplitudes']+1]:
            assert parent==ray['parent']
            assert np.all(np.sign(points[:,ray['native_mode']])==ray['sign'])
        cursor+=ray['rejected_amplitudes']+1
        low,high=ray['rows']
        assert np.max(result.samples.displacement_rms[low:high])<=.003


def test_an_impossible_signed_axis_is_an_explicit_scope_failure_not_a_partial_dataset():
    s=source();base=portfolio(s);before=base.points.copy()
    s.chemistry.log_kernel=lambda points:np.where(points[:,0]>=0,0.,-np.inf)
    with pytest.raises(ValueError,match='native mode 0, sign -1'):
        supplement(s,base,seed=4,contexts=1,maximum_attempts=2)
    np.testing.assert_array_equal(base.points,before)


@pytest.mark.parametrize('callback',[lambda parent,rows:True,
    lambda parent,rows:np.ones(len(rows)),lambda parent,rows:np.ones(len(rows)+1,bool)])
def test_parent_admission_cannot_accept_untyped_or_incomplete_rays(callback):
    s=source()
    with pytest.raises(ValueError,match='one boolean'):
        supplement(s,portfolio(s),seed=4,contexts=1,trajectory_feasible=callback)


def test_technical_native_geometry_error_is_not_hidden_as_a_resampled_amplitude():
    s=source();base=portfolio(s)
    def broken(points):raise RuntimeError('native geometry implementation unavailable')
    s.chemistry.log_kernel=broken
    with pytest.raises(RuntimeError,match='implementation unavailable'):
        supplement(s,base,seed=4,contexts=1)


def test_explicit_origin_gauges_do_not_become_new_distorted_training_pairs():
    s=source();s.mode_dimensions=s.fast.nm=0;s.radii=np.ones(1);s.lift=np.eye(1)
    s.fast.empirical_constraint.geometry.mapping=np.empty((2,3,0))
    s.native_mode_map=np.empty((3,0));s.fast.basis['gauge_basis']=np.eye(3)
    s.fast.meta['inventory']['native_mode_labels']=['origin-x','origin-y','origin-z']
    base=portfolio(s);result=supplement(s,base,seed=4,contexts=3)
    assert result.gauge_axes==(0,1,2) and not result.rays
    np.testing.assert_array_equal(result.samples.points,base.points)


@pytest.mark.parametrize('kwargs',[{'seed':True},{'contexts':7},{'steps':0},
    {'displacement_range':(0.,.1)},{'maximum_attempts':0}])
def test_invalid_controls_fail_before_producing_partial_coverage(kwargs):
    s=source()
    with pytest.raises(ValueError):supplement(s,portfolio(s),**dict({'seed':4},**kwargs))
