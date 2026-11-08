"""Repeated native unit peaks never reuse fitted intensities or changed inputs."""
import numpy as np
import pytest

powder=pytest.importorskip('GSASII.GSASIIpwd')
from gpx_recipe.live import native_peak_batch as batch
from gpx_recipe.live.native_peak_groups import group
from test_native_peak_batch_native import installed_helper
from test_native_profile_capture import native_source


def inputs():
    axis=np.linspace(99985.,100030.,601)
    records={}
    for i in range(512):
        p=np.array([100000.+i%3,.5,.25,1.,.125])
        records[(i,0,0)]=(p,[1e16,1.,-1e16,0.][i%4],0,len(axis))
    return records,axis


def original(records,axis):
    result=np.zeros(len(axis))
    for p,height,low,high in records.values():
        result[low:high]+=np.float64(height)*powder.getEpsVoigt(*p,axis[low:high])[0]
    return result


def test_reused_plane_keeps_new_intensities_original_order_and_zero_height_reflections(installed_helper):
    records,axis=inputs();owner=installed_helper
    first=owner.render(records,axis);np.testing.assert_array_equal(first,original(records,axis))
    cold=owner.diagnostics();assert cold['cached_plane_bytes']>0 and cold['native_points']>0
    first[:]=-1234.
    changed={key:(p.copy(),float(i%7-3),low,high) for i,(key,(p,_,low,high)) in enumerate(records.items())}
    actual=owner.render(changed,axis);np.testing.assert_array_equal(actual,original(changed,axis))
    warm=owner.diagnostics()
    assert warm['native_points']==cold['native_points']
    assert warm['native_reflections']==cold['native_reflections']
    assert warm['previous_plane_hits']==1 and warm['reused_native_points']>0
    assert warm['reflections']==1024 and warm['maximum_accounted_workspace_bytes']<=batch.MAXIMUM_BYTES
    # Original reflection order is applied anew, even when all unit shapes
    # remain present and the cancellation-sensitive intensity order changes.
    reverse=dict(reversed(list(changed.items())))
    np.testing.assert_array_equal(owner.render(reverse,axis),original(reverse,axis))


@pytest.mark.parametrize('change',['grid_low_bit','position_low_bit','shape_low_bit','signed_zero','window','new_zero_peak'])
def test_any_changed_original_shape_input_forces_a_fresh_plane(installed_helper,change):
    records,axis=inputs();owner=installed_helper
    if change=='signed_zero':
        for p,_,_,_ in records.values():p[4]=0.
    owner.render(records,axis);cold=owner.diagnostics()
    if change=='grid_low_bit':axis[7]=np.nextafter(axis[7],np.inf)
    elif change=='new_zero_peak':records[(9999,0,0)]=(np.array([100010.,.5,.25,1.,.125]),0.,0,len(axis))
    else:
        for key,(p,h,low,high) in list(records.items()):
            if change=='position_low_bit':p[0]=np.nextafter(p[0],np.inf)
            elif change=='shape_low_bit':p[1]=np.nextafter(p[1],np.inf)
            elif change=='signed_zero':p[4]=-0.
            else:low+=1
            records[key]=(p,h,low,high)
    np.testing.assert_array_equal(owner.render(records,axis),original(records,axis))
    current=owner.diagnostics()
    assert current['previous_plane_hits']==0 and current['native_points']>cold['native_points']
    assert current['maximum_accounted_workspace_bytes']<=batch.MAXIMUM_BYTES


def test_a_previous_plane_cannot_reduce_the_next_calls_memory_scope(installed_helper):
    records,axis=inputs();owner=installed_helper;owner.render(records,axis)
    assert owner.cached_plane_bytes>0
    *_,required=batch.pack_records(records,axis,batch.MAXIMUM_BYTES)
    limit=required+axis.nbytes
    np.testing.assert_array_equal(owner.render(records,axis,maximum_bytes=limit),original(records,axis))
    assert owner.cached_plane_bytes==0 and owner.diagnostics()['previous_plane_evictions']>0
    # Per-call accounting remains bounded even when the previous call used a
    # larger budget. The lifetime maximum is descriptive across both calls.
    assert owner.diagnostics()['maximum_accounted_workspace_bytes']<=batch.MAXIMUM_BYTES


def test_retention_overhead_does_not_reduce_a_viable_single_call_grouping_budget(installed_helper):
    records,axis=inputs();owner=installed_helper
    parameters,_,bounds,_,required=batch.pack_records(records,axis,batch.MAXIMUM_BYTES)
    groups,reason=group(parameters,bounds,axis,maximum_bytes=batch.MAXIMUM_BYTES,
        base_bytes=required+axis.nbytes)
    assert reason is None
    limit=groups.required_bytes
    np.testing.assert_array_equal(owner.render(records,axis,maximum_bytes=limit),original(records,axis))
    proof=owner.diagnostics()
    assert proof['grouped_calls']==1 and proof['sequentially_split_calls']==0
    assert proof['cached_plane_bytes']==0 and proof['maximum_accounted_workspace_bytes']<=limit


@pytest.mark.parametrize('change',['compiled','wrapper','fork','floating'])
def test_native_controls_are_rechecked_before_a_primed_plane_hit(installed_helper,monkeypatch,change):
    owner=installed_helper;records,axis=inputs();owner.render(records,axis)
    assert owner.cached_plane_bytes>0
    if change=='compiled':monkeypatch.setattr(powder.pyd,'pyepsvoigt',lambda:None)
    elif change=='wrapper':monkeypatch.setattr(powder,'getEpsVoigt',lambda *a:None)
    elif change=='fork':monkeypatch.setattr(batch.os,'getpid',lambda:owner.pid+1)
    else:monkeypatch.setattr(batch,'floating_state',lambda:(1919,16256))
    with pytest.raises(batch.BatchUnavailable,match='changed'):owner.render(records,axis)
    assert owner.diagnostics()['previous_plane_hits']==0


def test_changed_grid_during_a_reused_sum_is_rejected_before_publication(installed_helper,monkeypatch):
    owner=installed_helper;records,axis=inputs();owner.render(records,axis);sum_native=owner.group_sum
    def mutate(*args):
        result=sum_native(*args);axis[0]-=1.
        return result
    monkeypatch.setattr(owner,'group_sum',mutate)
    with pytest.raises(batch.BatchUnavailable,match='grid changed'):owner.render(records,axis)
    assert owner.cached_plane_bytes==0 and owner.diagnostics()['calls']==1


def test_scalar_collection_evicts_before_reducing_the_available_native_workspace(tmp_path,installed_helper,monkeypatch):
    from gpx_recipe.structural_live.optimization.native_forward import for_source
    from gpx_recipe.structural_live.optimization.native_tof_scalars import ScalarBatchUnavailable
    _,source=native_source(tmp_path/'source','PNT');engine=for_source(source)
    records,axis=inputs();engine.native_batch.render(records,axis)
    cached=engine.native_batch.cached_plane_bytes;assert cached>0
    collect=engine.collector.collect_packed_physical;seen=[]
    def constrained(point,*,maximum_bytes):
        seen.append(maximum_bytes)
        if len(seen)==1:raise ScalarBatchUnavailable('Injected tight scalar workspace')
        return collect(point,maximum_bytes=maximum_bytes)
    monkeypatch.setattr(engine.collector,'collect_packed_physical',constrained)
    point=np.zeros(len(source.radii));expected=source.native(point).copy()
    np.testing.assert_array_equal(engine.evaluate(point),expected)
    assert seen==[engine.maximum_bytes-engine.fixed_workspace-cached,engine.maximum_bytes-engine.fixed_workspace]
    assert engine.fallback_calls==0 and engine.native_batch.diagnostics()['previous_plane_evictions']>0
