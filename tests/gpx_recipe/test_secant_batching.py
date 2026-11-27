"""Independent stencil work may share workers without changing its samples."""
from types import SimpleNamespace
import time

import numpy as np
import pytest

from gpx_recipe.structural_live.optimization import reference_secants as secants
from gpx_recipe.structural_live.optimization import parallel_bank


@pytest.mark.parametrize('point',[np.array([0.,0.]),np.array([.999,-.999])])
def test_batch_preserves_every_stencil_sample_order_and_result(point):
    visited=[];scheduled=[]
    allowed=lambda p: bool(np.all(abs(p)<=1.))
    def profile(p):
        visited.append(p.copy())
        return np.array([np.exp(p[0]),np.sin(p[1]),p@p])
    expected=secants.select_secant_steps(profile,point,[1,0],allowed=allowed)
    original=np.asarray(visited);visited.clear()
    def batch(points):
        scheduled.extend(points.copy());return np.asarray([profile(p) for p in points])
    actual=secants.select_secant_steps(profile,point,[1,0],allowed=allowed,sample_many=batch)
    np.testing.assert_array_equal(scheduled,original)
    np.testing.assert_array_equal(visited,original)
    np.testing.assert_array_equal(actual[0],expected[0])
    assert actual[1]==expected[1]
    assert any(row['scheme']!='central' for c in actual[1] for row in c['scales']) == bool(point[0])


@pytest.mark.parametrize('bad',[None,[],[float('nan')],[[1.]]])
def test_incomplete_or_nonfinite_batch_is_not_a_secant(bad):
    with pytest.raises(ValueError,match='every finite profile'):
        secants.select_secant_steps(lambda p:p,np.zeros(2),[0],allowed=lambda p:True,
                                   sample_many=lambda p:bad)


def test_parallel_native_bank_and_secants_match_the_serial_arrays(tmp_path,monkeypatch):
    from test_structural_live_offline_start_bank import source
    from gpx_recipe.structural_live.optimization import offline_native_start_bank as banks
    s,*_=source(monkeypatch);serial=tmp_path/'serial.npz';parallel=tmp_path/'parallel.npz'
    expected=banks.build(s,serial,policy='joint_cell_v1')
    class Batch:
        active=False
        calls=[]
        def __enter__(self):self.active=True;return self
        def __exit__(self,*unused):self.active=False
        def __call__(self,points,*,native):
            assert self.active and native
            self.calls.append(points.copy())
            return np.asarray([s.native(p) for p in points]),dict(pool_reused=len(self.calls)>1)
    batch=Batch();actual=banks.build(s,parallel,policy='joint_cell_v1',batch_renderer=batch)
    assert not batch.active and len(batch.calls)==2
    with np.load(serial,allow_pickle=False) as a,np.load(parallel,allow_pickle=False) as b:
        assert a.files==b.files
        for name in a.files:np.testing.assert_array_equal(a[name],b[name])
    assert actual['proposal_step_selection']['details']==expected['proposal_step_selection']['details']
    assert actual['proposal_step_selection']['parallel_execution']['pool_reused']
    assert actual['proposal_step_selection']['parallel_execution']['serial_anchor_points']==2
    assert actual['native_curves']==expected['native_curves']==185


def test_failed_parent_proposal_anchor_requires_native_secant_batch(tmp_path,monkeypatch):
    from test_structural_live_offline_start_bank import source
    from gpx_recipe.structural_live.optimization import offline_native_start_bank as banks
    s,*_=source(monkeypatch);disabled=[False];roles=[]
    def proposal(point):
        if np.any(point[s.mode_dimensions:]):disabled[0]=True
        return s.native(point)
    def batch(points,*,native):
        roles.append(native)
        return np.asarray([s.native(p) for p in points]),dict(pool_reused=False,native_curve_role=native)
    receipt=banks._construct(s,tmp_path/'proposals.npz',policy='joint_cell_v1',
        proposal_renderer=proposal,batch_renderer=batch,proposal_batch_native=lambda:disabled[0])
    assert roles==[False,True]
    assert receipt['native_curves']==0 and receipt['curve_role']=='starting_proposals'
    assert receipt['proposal_step_selection']['parallel_execution']['native_curve_role']


def fake_map(monkeypatch,tmp_path,*,fail=False):
    created=[]
    monkeypatch.setattr(parallel_bank,'allocation',lambda _:dict(workers=4,memory_mb=8192,reserve_mb=512))
    from gpx_recipe.structural_live.optimization import offline_native_start_bank as banks
    monkeypatch.setattr(banks,'context',lambda s:dict(current=True))
    from gpx_recipe.symmetry.contracts import digest
    identity=digest(dict(current=True))
    def evaluate(p):
        now=time.monotonic()
        return np.r_[p,0.],dict(pid=123,started=now,finished=now+.001,cpu_seconds=.001,
            context_sha256=identity,peak_rss_bytes=1024)
    monkeypatch.setattr(parallel_bank,'_evaluate',evaluate)
    class Pool:
        def __init__(self,**kwargs):
            assert all(p.closed for p in created),'Replacement pool overlaps its predecessor'
            self.kwargs=kwargs;self.closed=False;self.maps=0;created.append(self)
        def map(self,function,points,**kwargs):
            self.maps+=1
            if fail:raise RuntimeError('worker failed')
            return map(function,points)
        def shutdown(self,**kwargs):self.closed=True
    monkeypatch.setattr(parallel_bank,'ProcessPoolExecutor',Pool)
    source=SimpleNamespace(radii=np.array([.05,.01]),mode_dimensions=1,
        fast=SimpleNamespace(x=np.arange(3),slow=SimpleNamespace(root=tmp_path/'prepared')))
    return parallel_bank.NativeBankProcessMap(source,tmp_path/'work'),created


def test_allocated_pool_is_reused_and_closed_after_both_batches(tmp_path,monkeypatch):
    renderer,pools=fake_map(monkeypatch,tmp_path)
    with renderer:
        first,a=renderer(np.zeros((8,2)),native=True)
        second,b=renderer(np.ones((8,2)),native=True)
        assert len(pools)==1 and not pools[0].closed
        assert pools[0].kwargs['max_workers']==4
        assert not a['pool_reused'] and b['pool_reused']
        np.testing.assert_array_equal(second,np.column_stack([np.ones((8,2)),np.zeros(8)]))
    assert pools[0].closed
    with pytest.raises(RuntimeError,match='closed'):renderer(np.zeros((8,2)),native=True)


def test_changed_role_closes_the_previous_pool_before_replacement(tmp_path,monkeypatch):
    renderer,pools=fake_map(monkeypatch,tmp_path)
    with renderer:
        renderer(np.zeros((8,2)),native=False)
        _,receipt=renderer(np.ones((8,2)),native=True)
        assert len(pools)==2 and pools[0].closed and not receipt['pool_reused']
    assert all(pool.closed for pool in pools)


def test_worker_failure_closes_its_pool_without_silent_candidate_loss(tmp_path,monkeypatch):
    renderer,pools=fake_map(monkeypatch,tmp_path,fail=True)
    with pytest.raises(RuntimeError,match='worker failed'):
        with renderer:renderer(np.zeros((8,2)),native=True)
    assert len(pools)==1 and pools[0].closed
