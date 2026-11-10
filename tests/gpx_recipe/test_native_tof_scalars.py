"""Native scalar batching retains reflection identity and original physics."""
import sys
import numpy as np
import pytest

pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from gpx_recipe.structural_live.optimization import native_tof_scalars as module
from gpx_recipe.structural_live.optimization import native_cell_proposals as proposals
from gpx_recipe.structural_live.optimization.native_profile_capture import _capture_lock


def native_records(source,point,*,physical=False):
    n=source.fast.native;driver=n.math.getPowderProfile;powder=n.math.G2pwd;original=powder.getEpsVoigt
    records={}
    def watched(*args,**kwargs):
        frame=sys._getframe(1)
        try:
            assert frame.f_code is driver.__code__
            local=frame.f_locals;r=local['refl'];key=tuple(map(int,r[:3]))
            records[key]=(np.asarray(args[:-1],float),float(r[9]*r[11]),int(local['iBeg']),int(local['iFin']))
        finally:del frame
        return original(*args,**kwargs)
    try:
        powder.getEpsVoigt=watched
        curve=source.native(point if physical else point*source.radii)
    finally:powder.getEpsVoigt=original
    return records,curve


@pytest.mark.parametrize('change',['parent','mixed','profile','absorption','lookup','cell_edge'])
def test_batched_scalars_match_complete_native_trace_at_current_geometry(tmp_path,change):
    _,s=native_source(tmp_path/'inputs','PNT');n=s.fast.native
    hfx=f':{s.fast.slow.hist.id}:'
    point=np.zeros(len(s.radii))
    if change=='mixed':point=np.linspace(-.06,.08,len(point))
    if change=='profile':
        n.parms[hfx+'X']=.015;n.parms[hfx+'Y']=.025;n.parms[hfx+'Z']=.01
    if change=='absorption':n.parms[hfx+'Absorption']=.1
    if change=='lookup':
        n.histdict[hfx+'pdabc']={'d':[.1,3.,20.],'sig':[.01,.07,.03]}
    if change=='cell_edge':point[s.mode_dimensions:]=np.linspace(-.8,.8,s.cell_dimensions)
    collector=module.for_source(s,n.math.getPowderProfile)
    assert collector is not None
    batched=collector.collect(point,maximum_bytes=64*1024*1024)
    packed=collector.collect_packed_physical(point*s.radii,maximum_bytes=64*1024*1024)
    assert [tuple(h) for h in packed.hkls]==list(batched)
    assert module.record_bytes(packed)<module.record_bytes(batched)
    for i,(parameters,height,low,high) in enumerate(batched.values()):
        np.testing.assert_array_equal(packed.parameters[i],parameters)
        assert packed.heights[i]==height and tuple(packed.bounds[i])==(low,high)
    native_rows=s.fast.native.histo_phases[0][s.fast.slow.hist.name]['Reflection Lists'][s.fast.slow.phase.name]['RefList']
    ds={tuple(map(int,r[:3])):r[4] for r in native_rows}
    np.testing.assert_array_equal(packed.spacings,[ds[tuple(h)] for h in packed.hkls])
    geometry=collector.collect_kernel_geometry(point*s.radii,maximum_bytes=64*1024*1024)
    original,curve=native_records(s,point)
    assert set(batched)==set(original)
    assert set(geometry)==set(original)
    for key in original:
        got,height,low,high=batched[key];expected,expected_height,nlow,nhigh=original[key]
        assert (low,high)==(nlow,nhigh)
        np.testing.assert_allclose(got,expected,rtol=1e-12,atol=1e-12)
        assert height==expected_height,(key,height,expected_height)
        shape,glo,ghi=geometry[key]
        np.testing.assert_array_equal(shape,got)
        assert (glo,ghi)==(nlow,nhigh)
    np.testing.assert_array_equal(curve,s.native(point*s.radii))
    assert collector.maximum_workspace_bytes<=64*1024*1024


def test_batch_does_not_read_or_change_observations(tmp_path):
    _,s=native_source(tmp_path/'inputs','PNT');n=s.fast.native
    data=n.histo_phases[0][s.fast.slow.hist.name]['Data'];data[1][:]=0.;data[2][:]=7.
    before=[np.ma.array(a,copy=True) for a in data[:3]]
    collector=module.for_source(s,n.math.getPowderProfile)
    records=collector.collect(np.zeros(len(s.radii)),maximum_bytes=64*1024*1024)
    assert records
    for actual,expected in zip(data[:3],before):np.testing.assert_array_equal(actual,expected)


def test_physical_collection_does_not_round_trip_coordinates_through_radii(tmp_path,monkeypatch):
    _,s=native_source(tmp_path/'inputs','PNT');collector=module.for_source(s,s.fast.native.math.getPowderProfile)
    physical=np.linspace(-.053,.071,len(s.radii))*s.radii
    checked=[];original=s.check
    def watch(point):checked.append(np.asarray(point).copy());return original(point)
    monkeypatch.setattr(s,'check',watch)
    collector.collect_physical(physical,maximum_bytes=64*1024*1024)
    assert checked
    np.testing.assert_array_equal(checked[0],physical)


def test_unreviewed_scalar_implementation_keeps_original_path(tmp_path,monkeypatch):
    _,s=native_source(tmp_path/'inputs','PNT');n=s.fast.native
    monkeypatch.setattr(n.math,'GetReflPos',lambda *a:0.)
    assert module.for_source(s,n.math.getPowderProfile) is None


def test_batched_failure_restores_full_native_state_and_lock(tmp_path,monkeypatch):
    _,s=native_source(tmp_path/'inputs','PNT');point=np.zeros(len(s.radii));expected=s.native(point)
    collector=module.for_source(s,s.fast.native.math.getPowderProfile);original=collector.collect;calls=[]
    def broken(p,**kwargs):
        calls.append(True)
        if len(calls)==2:raise RuntimeError('Induced batched scalar failure')
        return original(p,**kwargs)
    monkeypatch.setattr(collector,'collect',broken)
    monkeypatch.setattr(proposals,'scalar_collector_for_source',lambda *a:collector)
    engine=proposals.NativeCellProposals(s)
    with pytest.raises(RuntimeError,match='batched scalar failure'):
        engine.columns(point,np.full(len(point),.004),allowed=lambda p:np.all(abs(p)<=1))
    np.testing.assert_array_equal(s.native(point),expected)
    assert _capture_lock.acquire(blocking=False);_capture_lock.release()


def test_batched_anchor_mismatch_cannot_supply_directions(tmp_path,monkeypatch):
    _,s=native_source(tmp_path/'inputs','PNT');point=np.zeros(len(s.radii));expected=s.native(point)
    collector=module.for_source(s,s.fast.native.math.getPowderProfile);original=collector.collect
    def changed(p,**kwargs):
        records=original(p,**kwargs);key=next(iter(records));params,height,lo,hi=records[key]
        records[key]=(params+1.,height,lo,hi)
        return records
    monkeypatch.setattr(collector,'collect',changed)
    monkeypatch.setattr(proposals,'scalar_collector_for_source',lambda *a:collector)
    engine=proposals.NativeCellProposals(s)
    with pytest.raises(proposals.ProposalUnavailable,match='full native anchor'):
        engine.columns(point,np.full(len(point),.004),allowed=lambda p:np.all(abs(p)<=1))
    assert engine.diagnostics()['batched_scalar_anchor_mismatches']==1
    np.testing.assert_array_equal(s.native(point),expected)
    assert _capture_lock.acquire(blocking=False);_capture_lock.release()


def test_batched_workspace_is_bounded_without_dropping_reflections(tmp_path):
    _,s=native_source(tmp_path/'inputs','PNT')
    collector=module.for_source(s,s.fast.native.math.getPowderProfile)
    with pytest.raises(module.ScalarBatchUnavailable,match='workspace budget'):
        collector.collect(np.zeros(len(s.radii)),maximum_bytes=1)


def test_packed_representation_fits_a_real_bound_that_refuses_per_reflection_python_records(tmp_path):
    _,source=native_source(tmp_path/'inputs','PNT')
    collector=module.for_source(source,source.fast.native.math.getPowderProfile)
    point=np.zeros(len(source.radii))
    original=collector.collect_physical(point,maximum_bytes=64*1024*1024)
    rows=source.fast.native.histo_phases[0][source.fast.slow.hist.name]['Reflection Lists'][source.fast.slow.phase.name]['RefList']
    budget=len(rows)*(512+96)+4096
    assert budget<len(rows)*(512+1024)
    with pytest.raises(module.ScalarBatchUnavailable,match='workspace budget'):
        collector.collect_physical(point,maximum_bytes=budget)
    packed=collector.collect_packed_physical(point,maximum_bytes=budget)
    assert [tuple(h) for h in packed.hkls]==list(original)
    assert collector.last_workspace_bytes<=budget


def test_collector_cannot_bypass_replaced_native_owner(tmp_path,monkeypatch):
    _,s=native_source(tmp_path/'inputs','PNT');n=s.fast.native
    collector=module.for_source(s,n.math.getPowderProfile)
    def replaced():raise RuntimeError('Native owner replaced')
    monkeypatch.setattr(n,'_check',replaced)
    with pytest.raises(RuntimeError,match='owner replaced'):
        collector.collect(np.zeros(len(s.radii)),maximum_bytes=64*1024*1024)
