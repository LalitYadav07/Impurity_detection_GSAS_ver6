from types import SimpleNamespace
import numpy as np
import pytest
from gpx_recipe.structural_live.optimization import native_bank_reuse as module


@pytest.fixture(autouse=True)
def no_native_stack_for_host_contracts(monkeypatch):
    monkeypatch.setattr(module,'native_forward_for_source',lambda source:None)


def test_only_displacements_at_exact_fixed_cell_reuse_native_profiles(monkeypatch):
    calls=[]
    class Cache:
        def __init__(self,source,max_cells):assert max_cells==1;self.ready=False
        def has_response(self,p):return self.ready
        def native_evaluate(self,p):calls.append(('capture',tuple(p)));self.ready=True;return np.array([3.])
        def verify(self,p):calls.append(('check',tuple(p)))
        def evaluate(self,p):calls.append(('cache',tuple(p)));return np.array([3.])
    monkeypatch.setattr(module,'CapturedCellDerivative',Cache)
    def native(p):calls.append(('native',tuple(p)));return np.array([3.])
    s=SimpleNamespace(mode_dimensions=2,native=native,identity='same-candidate')
    proxy=module.NativeBankSource(s)
    for p in [[0,0,0],[.1,0,0],[0,.2,0],[0,0,.001],[.1,0,.00001]]:proxy.native(p)
    assert [r[0] for r in calls]==['capture','check','cache','cache','native','native']
    assert proxy.identity==s.identity and proxy.cached_calls==2 and proxy.direct_calls==3


def test_failed_native_identity_check_never_returns_a_cached_profile(monkeypatch):
    class Cache:
        def __init__(self,*a,**kw):self.ready=False
        def has_response(self,p):return self.ready
        def native_evaluate(self,p):self.ready=True;return np.array([3.])
        def verify(self,p):raise ValueError('native mismatch')
        def evaluate(self,p):raise AssertionError('unchecked cache used')
    monkeypatch.setattr(module,'CapturedCellDerivative',Cache)
    p=module.NativeBankSource(SimpleNamespace(mode_dimensions=1))
    p.native([0,0])
    with pytest.raises(ValueError,match='mismatch'):p.native([.1,0])
    assert not p.checked and p.cached_calls==0


def test_internal_build_dispatch_retains_full_native_reference(monkeypatch,tmp_path):
    from gpx_recipe.structural_live.optimization import offline_native_start_bank as bank
    calls=[]
    def reuse(source,path,policy):
        calls.append((source,path,policy));return {'accelerated':True}
    monkeypatch.setattr(module,'build_reusing_profiles',reuse)
    source=object();path=tmp_path/'bank.npz'
    assert bank.build(source,path,reuse_fixed_cell_profiles=True)=={'accelerated':True}
    assert calls==[(source,path,'multiscale_v1')]
