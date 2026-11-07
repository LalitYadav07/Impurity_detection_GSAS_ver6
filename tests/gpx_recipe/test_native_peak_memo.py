"""Exact keys must not turn nearby profiles or edited grids into cache hits."""
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from gpx_recipe.live import native_peak_memo as module


def runtime(monkeypatch,native_work=None,seen_axes=None):
    calls=[]
    monkeypatch.setattr(module,'floating_state',lambda:(895,8064))
    def tof(pos,alpha,beta,sig,gam,x):
        if native_work is not None:native_work()
        calls.append(tuple(map(float,(pos,alpha,beta,sig,gam))))
        values=np.asarray(x)+sum((j+1)*v for j,v in enumerate((pos,alpha,beta,sig,gam)))
        return values.astype('float32'),float(np.sum(values))
    def cw(pos,sig,gam,shl,x):return tof(pos,sig,gam,shl,1.,x)
    powder=SimpleNamespace(getEpsVoigt=tof,getFCJVoigt3=cw)
    def driver(_a,_b,x,*,shift=0.,foreign=False,unique=False,prefix=False):
        if seen_axes is not None:seen_axes.append(x)
        view=x[1:-1].copy() if foreign else x[1:-1]
        if prefix:
            return [powder.getEpsVoigt(0.,.1,.2,3.,4.,view) for _ in range(2)]+[
                powder.getEpsVoigt(float(i),.1,.2,3.,4.,view) for i in range(1,101)]
        if unique:
            return [powder.getEpsVoigt(shift+i,.1,.2,3.,4.,view) for i in range(10)]
        first=powder.getEpsVoigt(shift,.1,.2,3.,4.,view)
        second=powder.getEpsVoigt(shift,.1,.2,3.,4.,view)
        return first,second
    math=SimpleNamespace(getPowderProfile=driver,G2pwd=powder,G2mp=SimpleNamespace(InitMP=lambda:(False,0)))
    identities={driver:module._DRIVER,tof:module._KERNELS['getEpsVoigt'][0],cw:module._KERNELS['getFCJVoigt3'][0]}
    monkeypatch.setattr(module,'_source_hash',lambda f:identities.get(f))
    return math,calls


@pytest.mark.parametrize('same_scope',[False,True])
def test_changed_arithmetic_controls_invalidate_exact_shape_values(monkeypatch,same_scope):
    math,calls=runtime(monkeypatch);memo=module.NativePeakMemo();x=np.arange(7.)
    state=[(895,8064)]
    monkeypatch.setattr(module,'floating_state',lambda:state[0])
    def twice():
        math.getPowderProfile(None,None,x)
        state[0]=(1919,16256)
        math.getPowderProfile(None,None,x)
    if same_scope:
        with memo.evaluation(math):twice()
    else:
        with memo.evaluation(math):math.getPowderProfile(None,None,x)
        state[0]=(1919,16256)
        with memo.evaluation(math):math.getPowderProfile(None,None,x)
    assert len(calls)==2 and memo.misses==2 and memo.hits==2


@pytest.mark.parametrize('becomes_unknown',[False,True])
def test_unknown_arithmetic_state_uses_original_driver_and_discards_cache(monkeypatch,becomes_unknown):
    math,calls=runtime(monkeypatch);memo=module.NativePeakMemo();x=np.arange(7.)
    driver=math.getPowderProfile;peak=math.G2pwd.getEpsVoigt
    state=[(895,8064) if becomes_unknown else None]
    monkeypatch.setattr(module,'floating_state',lambda:state[0])
    with memo.evaluation(math):
        if becomes_unknown:math.getPowderProfile(None,None,x)
        state[0]=None
        math.getPowderProfile(None,None,x)
    assert not memo._store.cache and memo._store.floating is None
    assert math.getPowderProfile is driver and math.G2pwd.getEpsVoigt is peak
    assert memo.scope_bypasses['unavailable native floating-point control state']==1
    assert len(calls)==(3 if becomes_unknown else 2)


@pytest.mark.parametrize('kind',['plain','nomask','all_false','excluded','strided','float32'])
def test_only_standard_unmasked_native_grids_use_a_zero_copy_plain_view(monkeypatch,kind):
    seen=[];math,_=runtime(monkeypatch,seen_axes=seen);memo=module.NativePeakMemo()
    x=np.arange(7.,dtype=float)
    if kind!='plain':x=np.ma.array(x,mask=False)
    if kind=='all_false':x.mask=np.zeros(7,dtype=bool)
    elif kind=='excluded':x.mask=[False,False,True,False,False,False,False]
    elif kind=='strided':x=np.ma.array(np.arange(14.,dtype=float)[::2],mask=False,copy=False)
    elif kind=='float32':x=x.astype('float32')
    values=np.ma.getdata(x).copy();mask=np.ma.getmaskarray(x).copy()
    with memo.evaluation(math):math.getPowderProfile(None,None,x)
    expected=kind in {'nomask','all_false'}
    assert memo.unmasked_grid_driver_calls==int(expected)
    if expected:
        assert type(seen[0]) is np.ndarray and np.shares_memory(seen[0],np.ma.getdata(x))
    else:assert seen[0] is x
    np.testing.assert_array_equal(np.ma.getdata(x),values)
    np.testing.assert_array_equal(np.ma.getmaskarray(x),mask)


def test_identical_native_arguments_reuse_independent_result_arrays(monkeypatch):
    math,calls=runtime(monkeypatch);memo=module.NativePeakMemo()
    driver=math.getPowderProfile;peak=math.G2pwd.getEpsVoigt;x=np.arange(7.)
    with memo.evaluation(math):
        first,second=math.getPowderProfile(None,None,x)
        np.testing.assert_array_equal(first[0],second[0]);assert first[1]==second[1]
        first[0][:]=-99;second[0][:]=-100
        third=math.getPowderProfile(None,None,x)[0]
    assert len(calls)==1 and memo.hits==3
    np.testing.assert_array_equal(third[0],driver(None,None,x)[0][0])
    assert math.getPowderProfile is driver and math.G2pwd.getEpsVoigt is peak


def test_same_storage_grid_edits_and_tiny_parameter_changes_are_not_hits(monkeypatch):
    math,calls=runtime(monkeypatch);memo=module.NativePeakMemo();x=np.arange(7.)
    with memo.evaluation(math):
        a=math.getPowderProfile(None,None,x)[0][0]
        x[2]+=.125
        b=math.getPowderProfile(None,None,x)[0][0]
        math.getPowderProfile(None,None,x,shift=np.nextafter(0.,1.))
    assert not np.array_equal(a,b)
    assert len(calls)==3 and memo.misses==3


def test_other_arrays_and_unknown_driver_use_unchanged_native(monkeypatch):
    math,calls=runtime(monkeypatch);memo=module.NativePeakMemo();x=np.arange(7.)
    with memo.evaluation(math):math.getPowderProfile(None,None,x,foreign=True)
    assert memo.bypasses==2 and len(calls)==2
    original=math.getPowderProfile
    math.getPowderProfile=lambda *a,**k:original(*a,**k)
    with memo.evaluation(math):math.getPowderProfile(None,None,x)
    assert len(calls)==4 and not memo.hits
    assert memo.scope_bypasses['unreviewed native driver or active peak hook']==1


def test_memory_bound_and_failed_driver_restore_hooks_and_ownership(monkeypatch):
    math,calls=runtime(monkeypatch);memo=module.NativePeakMemo(600);x=np.arange(7.)
    driver=math.getPowderProfile;peak=math.G2pwd.getEpsVoigt
    for shift in range(5):
        with memo.evaluation(math):math.getPowderProfile(None,None,x,shift=shift)
    assert memo.evictions==4 and memo.diagnostics()['accounted_cache_bytes']<=600
    with pytest.raises(RuntimeError,match='injected'):
        with memo.evaluation(math):
            math.getPowderProfile(None,None,x)
            raise RuntimeError('injected')
    assert math.getPowderProfile is driver and math.G2pwd.getEpsVoigt is peak
    with memo.evaluation(math):math.getPowderProfile(None,None,x)
    assert not module._LOCK.locked()


def test_native_multiprocessing_and_non_main_thread_never_patch_functions(monkeypatch):
    math,calls=runtime(monkeypatch);memo=module.NativePeakMemo();driver=math.getPowderProfile
    math.G2mp.InitMP=lambda:(True,2)
    with memo.evaluation(math):assert math.getPowderProfile is driver
    math.G2mp.InitMP=lambda:(False,0)
    def run():
        with memo.evaluation(math):assert math.getPowderProfile is driver
    thread=threading.Thread(target=run);thread.start();thread.join()
    assert memo.scope_bypasses=={'unsupported driver or native multiprocessing':1,'requires owned main thread':1}
    assert not calls


def test_unprofitable_unique_peaks_bypass_memo_without_omitting_work(monkeypatch):
    math,calls=runtime(monkeypatch);memo=module.NativePeakMemo();x=np.arange(7.)
    ticks=iter(range(1000))
    monkeypatch.setattr(module.time,'perf_counter',lambda:next(ticks))
    monkeypatch.setattr(module,'PROBE_CALLS',2)
    with memo.evaluation(math):actual=math.getPowderProfile(None,None,x,unique=True)
    assert len(calls)==10  # All ten original calculations still happened.
    assert memo.calls==2 and memo.adaptive_driver_bypasses==1
    with memo.evaluation(math):again=math.getPowderProfile(None,None,x,unique=True)
    for left,right in zip(actual,again):np.testing.assert_array_equal(left[0],right[0])


def test_compiled_dependency_change_cannot_reuse_an_earlier_result(monkeypatch):
    math,calls=runtime(monkeypatch);memo=module.NativePeakMemo();x=np.arange(7.)
    with memo.evaluation(math):math.getPowderProfile(None,None,x)
    assert len(calls)==1
    math.G2pwd.pyd=SimpleNamespace(pyepsvoigt=lambda:None)
    with memo.evaluation(math):math.getPowderProfile(None,None,x)
    assert len(calls)==2


def test_profitable_prefix_does_not_admit_an_unprofitable_complete_driver(monkeypatch):
    clock=[0.]
    def native_work():clock[0]+=100.
    def timer():clock[0]+=1.;return clock[0]
    math,calls=runtime(monkeypatch,native_work=native_work)
    memo=module.NativePeakMemo();x=np.arange(7.)
    monkeypatch.setattr(module.time,'perf_counter',timer)
    monkeypatch.setattr(module,'PROBE_CALLS',2)
    with memo.evaluation(math):result=math.getPowderProfile(None,None,x,prefix=True)
    assert len(result)==102 and len(calls)==101
    assert 2<memo.calls<102 and memo.adaptive_driver_bypasses==1


def test_fresh_contexts_share_only_numeric_storage_and_keep_independent_counters(monkeypatch):
    monkeypatch.setattr(module,'_PROCESS_STORE',None)
    math,calls=runtime(monkeypatch);x=np.arange(7.)
    first=module.shared_native_peak_memo()
    with first.evaluation(math):
        original=math.getPowderProfile(None,None,x)[0][0]
    expected=original.copy();original[:]=-100.
    second=module.shared_native_peak_memo()
    assert second is not first and second.calls==0
    with second.evaluation(math):
        actual=math.getPowderProfile(None,None,x)[0][0]
    np.testing.assert_array_equal(actual,expected)
    assert len(calls)==1
    assert first.misses==1 and first.hits==1
    assert second.misses==0 and second.hits==2
    for owner in (first,second):
        proof=owner.diagnostics()
        assert proof['shared_across_contexts'] and proof['cache_scope']=='owned process'
        assert proof['counters_scope']=='this adapter'
        assert proof['entries']==1 and proof['accounted_cache_bytes']<=module.MAXIMUM_BYTES
    private=module.NativePeakMemo()
    with private.evaluation(math):math.getPowderProfile(None,None,x)
    assert len(calls)==2 and not private.shared_across_contexts


def test_shared_store_invalidation_reaches_every_existing_adapter(monkeypatch):
    monkeypatch.setattr(module,'_PROCESS_STORE',None)
    math,calls=runtime(monkeypatch);x=np.arange(7.)
    first=module.shared_native_peak_memo();second=module.shared_native_peak_memo()
    with first.evaluation(math):math.getPowderProfile(None,None,x)
    math.G2pwd.pyd=SimpleNamespace(pyepsvoigt=lambda:None)
    with second.evaluation(math):math.getPowderProfile(None,None,x)
    assert len(calls)==2 and second.misses==1
    del math.G2pwd.pyd
    with first.evaluation(math):math.getPowderProfile(None,None,x)
    assert len(calls)==3 and first.misses==2
    driver=math.getPowderProfile
    math.getPowderProfile=lambda *a,**k:driver(*a,**k)
    with second.evaluation(math):math.getPowderProfile(None,None,x)
    assert not second.diagnostics()['entries']
    math.getPowderProfile=driver
    with first.evaluation(math):math.getPowderProfile(None,None,x)
    assert len(calls)==6 and first.misses==3


def test_shared_allocation_is_bounded_once_across_many_contexts(monkeypatch):
    math,calls=runtime(monkeypatch);x=np.arange(7.)
    store=module._PeakStore(600)
    owners=[module.NativePeakMemo(600,_store=store) for _ in range(5)]
    for shift,owner in enumerate(owners):
        with owner.evaluation(math):math.getPowderProfile(None,None,x,shift=shift)
    assert len(calls)==5 and sum(owner.evictions for owner in owners)==4
    assert all(owner.diagnostics()['entries']==1 for owner in owners)
    assert store.bytes<=600
    owners[0].clear()
    assert all(owner.diagnostics()['entries']==0 for owner in owners)
    with pytest.raises(ValueError,match='same bounded allocation'):
        module.NativePeakMemo(_store=store)


def test_nested_owner_cannot_invalidate_or_replace_shared_outer_hooks(monkeypatch):
    monkeypatch.setattr(module,'_PROCESS_STORE',None)
    math,calls=runtime(monkeypatch);x=np.arange(7.)
    first=module.shared_native_peak_memo();second=module.shared_native_peak_memo()
    driver=math.getPowderProfile
    with first.evaluation(math):
        math.getPowderProfile(None,None,x)
        owned=math.getPowderProfile
        with second.evaluation(math):
            assert math.getPowderProfile is owned
            math.getPowderProfile(None,None,x)
        assert math.getPowderProfile is owned
        math.getPowderProfile(None,None,x)
    assert math.getPowderProfile is driver and not module._LOCK.locked()
    assert len(calls)==1 and first.hits==5 and second.calls==0
    assert second.scope_bypasses=={'another native peak scope owns this process':1}


@pytest.mark.parametrize('inherited_first',[False,True])
def test_changed_process_identity_loses_inherited_qualification(monkeypatch,inherited_first):
    monkeypatch.setattr(module,'_PROCESS_STORE',None)
    math,calls=runtime(monkeypatch);x=np.arange(7.)
    first=module.shared_native_peak_memo()
    old_store=first._store
    with first.evaluation(math):math.getPowderProfile(None,None,x)
    actual_pid=module.os.getpid()
    monkeypatch.setattr(module.os,'getpid',lambda:actual_pid+1)
    if inherited_first:
        with first.evaluation(math):math.getPowderProfile(None,None,x)
        second=module.shared_native_peak_memo()
        with second.evaluation(math):math.getPowderProfile(None,None,x)
        assert first.misses==2 and second.misses==0
    else:
        second=module.shared_native_peak_memo()
        with second.evaluation(math):math.getPowderProfile(None,None,x)
        with first.evaluation(math):math.getPowderProfile(None,None,x)
        assert first.misses==1 and second.misses==1
    assert len(calls)==2
    assert first._store is second._store and first._store is not old_store
    assert old_store.bytes==0
