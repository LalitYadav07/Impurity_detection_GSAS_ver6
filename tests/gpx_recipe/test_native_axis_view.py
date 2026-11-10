"""Native axis storage is preserved across the performance-only view scope."""
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from gpx_recipe.live import native_axis_view as module


def engine(monkeypatch, *, failure=False):
    seen=[]
    def driver(*args, **kwargs):
        seen.append(args[2])
        if failure:raise RuntimeError('Induced native failure')
        return np.asarray(args[2]).copy()
    monkeypatch.setattr(module, '_source_hash', lambda fn:module._DRIVER if fn is driver else None)
    native=SimpleNamespace(getPowderProfile=driver,G2mp=SimpleNamespace(InitMP=lambda:(False,1)))
    return native,driver,seen


def test_unmasked_view_preserves_values_and_owned_storage_on_failure(monkeypatch):
    native,driver,seen=engine(monkeypatch,failure=True)
    axis=np.ma.array(np.arange(8,dtype=float),mask=False);before=axis.copy()
    view=module.NativeUnmaskedAxisView()
    with pytest.raises(RuntimeError,match='Induced native failure'):
        with view.evaluation(native):native.getPowderProfile(None,None,axis)
    assert type(seen[0]) is np.ndarray and np.shares_memory(seen[0],axis)
    np.testing.assert_array_equal(axis.data,before.data)
    np.testing.assert_array_equal(np.ma.getmaskarray(axis),np.ma.getmaskarray(before))
    assert native.getPowderProfile is driver
    assert module._LOCK.acquire(blocking=False);module._LOCK.release()


@pytest.mark.parametrize('kind',['masked','float32','strided','plain'])
def test_masks_and_unusual_storage_keep_the_original_axis(monkeypatch,kind):
    native,driver,seen=engine(monkeypatch)
    if kind=='masked':axis=np.ma.array(np.arange(8,dtype=float),mask=[False,True]*4)
    elif kind=='float32':axis=np.ma.array(np.arange(8,dtype=np.float32))
    elif kind=='strided':axis=np.ma.array(np.arange(16,dtype=float))[::2]
    else:axis=np.arange(8,dtype=float)
    with module.NativeUnmaskedAxisView().evaluation(native):native.getPowderProfile(None,None,axis)
    assert seen[0] is axis and native.getPowderProfile is driver


@pytest.mark.parametrize('reason',['disabled','driver','mp','lock','thread'])
def test_unsupported_ownership_keeps_the_original_driver(monkeypatch,reason):
    native,driver,seen=engine(monkeypatch)
    view=module.NativeUnmaskedAxisView(reason!='disabled')
    if reason=='driver':monkeypatch.setattr(module,'_source_hash',lambda fn:None)
    if reason=='mp':native.G2mp.InitMP=lambda:(True,2)
    axis=np.ma.array(np.arange(8,dtype=float))
    def run():
        with view.evaluation(native):native.getPowderProfile(None,None,axis)
    if reason=='lock':
        module._LOCK.acquire()
        try:run()
        finally:module._LOCK.release()
    elif reason=='thread':
        with ThreadPoolExecutor(max_workers=1) as pool:pool.submit(run).result()
    else:run()
    assert seen[0] is axis and native.getPowderProfile is driver
    assert sum(view.bypasses.values())==1


def test_nested_scope_does_not_replace_an_existing_owner(monkeypatch):
    native,driver,seen=engine(monkeypatch)
    axis=np.ma.array(np.arange(8,dtype=float))
    first=module.NativeUnmaskedAxisView();second=module.NativeUnmaskedAxisView()
    with first.evaluation(native):
        active=native.getPowderProfile
        with second.evaluation(native):
            assert native.getPowderProfile is active
            native.getPowderProfile(None,None,axis)
    assert native.getPowderProfile is driver and first.converted_calls==1
    assert sum(second.bypasses.values())==1
