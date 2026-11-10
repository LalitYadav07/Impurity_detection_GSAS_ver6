"""Unsupported reflection scopes preserve native behavior and ownership."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from gpx_recipe.live import native_reflection_memo as module


def fixture(monkeypatch):
    functions={}
    for name in module._SOURCES:
        def fn(*args, **kwargs):return 'native result'
        functions[name]=fn
    native=SimpleNamespace(**functions)
    lookup={id(fn):module._SOURCES[name] for name,fn in functions.items()}
    monkeypatch.setattr(module,'_hash',lambda fn:lookup.get(id(fn)))
    return native,functions


@pytest.mark.parametrize('reason',['disabled','bytes','driver','magnetic','modulated','pawley','thread','lock'])
def test_unsupported_scope_does_not_replace_native_enumeration(monkeypatch,reason):
    native,functions=fixture(monkeypatch)
    phases={'p':{'General':dict(Type='nuclear')}}
    memo=module.NativeReflectionMemo(0 if reason=='bytes' else module.MAXIMUM_BYTES,enabled=reason!='disabled')
    if reason=='driver':monkeypatch.setattr(module,'_hash',lambda fn:None)
    if reason=='magnetic':phases['p']['General']['Type']='magnetic'
    if reason=='modulated':phases['p']['General']['Modulated']=True
    if reason=='pawley':phases['p']['General']['doPawley']=True
    def run():
        with memo.evaluation(native,phases):
            assert native.GenHLaue is functions['GenHLaue']
            assert native.GenHLaue(None,None,None)=='native result'
    if reason=='thread':
        with ThreadPoolExecutor(max_workers=1) as pool:pool.submit(run).result()
    elif reason=='lock':
        module._LOCK.acquire()
        try:run()
        finally:module._LOCK.release()
    else:run()
    assert sum(memo.bypasses.values())==1
    assert native.GenHLaue is functions['GenHLaue']


def test_exception_and_nested_scope_restore_the_original_native_function(monkeypatch):
    native,functions=fixture(monkeypatch);phases={'p':{'General':dict(Type='nuclear')}}
    first=module.NativeReflectionMemo();second=module.NativeReflectionMemo()
    with pytest.raises(RuntimeError,match='induced failure'):
        with first.evaluation(native,phases):
            active=native.GenHLaue
            with second.evaluation(native,phases):assert native.GenHLaue is active
            raise RuntimeError('induced failure')
    assert native.GenHLaue is functions['GenHLaue']
    assert module._LOCK.acquire(blocking=False);module._LOCK.release()


@pytest.mark.parametrize('budget',[-1,True,64*1024*1024+1])
def test_cache_budget_is_bounded(budget):
    with pytest.raises(ValueError,match='allocation'):module.NativeReflectionMemo(budget)
