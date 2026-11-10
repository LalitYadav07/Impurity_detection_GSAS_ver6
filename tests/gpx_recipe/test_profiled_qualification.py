import pytest
from gpx_recipe.symmetry import profiled_qualification as module


@pytest.fixture(autouse=True)
def empty_ledger(monkeypatch):
    monkeypatch.setattr(module,'_entries',module.OrderedDict())


def test_only_complete_successful_owned_checks_can_be_reused():
    module.remember('checked',[1e-12]*4)
    assert module.prior('checked',4)==[1e-12]*4
    assert module.prior('checked',5) is None
    module.remember('checked',[1e-13])
    assert len(module.prior('checked',4))==4
    module.forget('checked')
    assert module.prior('checked',1) is None


@pytest.mark.parametrize('errors',[[],[float('nan')],[float('inf')],[1e-7]])
def test_bad_or_empty_native_evidence_is_never_admitted(errors):
    module.remember('bad',errors)
    assert module.prior('bad',1) is None


def test_qualification_cannot_be_inherited_by_another_process(monkeypatch):
    module.remember('parent',[1e-12]*4)
    monkeypatch.setattr(module.os,'getpid',lambda:module._pid+1)
    assert module.prior('parent',1) is None


def test_ledger_is_bounded_and_lru_eviction_preserves_recent_evidence():
    for i in range(module.MAX_ENTRIES):module.remember(str(i),[1e-12]*4)
    module.prior('0',4)
    module.remember('new',[1e-12]*4)
    assert module.prior('1',1) is None
    assert module.prior('0',4)==[1e-12]*4
    assert len(module._entries)==module.MAX_ENTRIES


def test_unknown_context_declines_reuse_and_array_masks_are_bound():
    import numpy as np
    assert module.identity(object(),object()) is None
    a=np.arange(5,dtype=float)
    assert module._plain(a)!=module._plain(np.ma.array(a,mask=[False,True,False,False,False]))
    with pytest.raises(ValueError,match='Object arrays'):module._plain(np.array([object()]))
