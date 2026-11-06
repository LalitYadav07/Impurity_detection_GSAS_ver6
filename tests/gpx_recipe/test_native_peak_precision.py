"""Native argument equivalence is not rounding a scientific calculation."""
from types import SimpleNamespace

import numpy as np
import pytest

from gpx_recipe.live import native_peak_memo as module


def runtime(monkeypatch,reviewed=True):
    calls=[]
    def tof(position,alpha,beta,sigma,gamma,x):
        calls.append((position,alpha))
        offset=np.asarray(x-position,dtype=np.float32)
        a,b,s,g=np.asarray([alpha,beta,sigma,gamma],dtype=np.float32)
        values=(np.sin(offset)*a+b+s+g).astype(np.float32)
        widths=np.diff(x);widths=np.append(widths,widths[-1])
        return values,np.sum(values*widths)
    def cw(position,sigma,gamma,shl,x):return tof(position,sigma,gamma,shl,1.,x)
    powder=SimpleNamespace(getEpsVoigt=tof,getFCJVoigt3=cw)
    def driver(_a,_b,x,*,position=1.,alpha=.7):
        return powder.getEpsVoigt(position,alpha,.3,1.,.02,x[1:-1])
    math=SimpleNamespace(getPowderProfile=driver,G2pwd=powder,G2mp=SimpleNamespace(InitMP=lambda:(False,0)))
    identities={driver:module._DRIVER,tof:module._KERNELS['getEpsVoigt'][0],cw:module._KERNELS['getFCJVoigt3'][0]}
    monkeypatch.setattr(module,'_source_hash',lambda f:identities.get(f))
    monkeypatch.setattr(module,'_compiled_tof_float32',lambda _:reviewed)
    return math,calls


def test_shape_scalars_with_identical_compiled_bits_reuse_without_changing_results(monkeypatch):
    math,calls=runtime(monkeypatch);memo=module.NativePeakMemo();x=np.arange(7.)
    original=math.getPowderProfile
    expected=[original(None,None,x,alpha=alpha) for alpha in [.7,.7+1e-12]]
    calls.clear()
    with memo.evaluation(math):
        actual=[math.getPowderProfile(None,None,x,alpha=alpha) for alpha in [.7,.7+1e-12]]
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a[0],b[0]);assert a[1]==b[1]
    assert len(calls)==1 and memo.native_float32_hits==1


def test_positions_need_equal_translated_compiled_arrays(monkeypatch):
    math,calls=runtime(monkeypatch);memo=module.NativePeakMemo();x=np.arange(1000.,1007.)
    with memo.evaluation(math):
        a=math.getPowderProfile(None,None,x,position=1.)
        b=math.getPowderProfile(None,None,x,position=1.+1e-12)
    np.testing.assert_array_equal(a[0],b[0]);assert a[1]==b[1]
    assert len(calls)==1 and memo.native_offset_checks==1 and memo.native_float32_hits==1


def test_equal_rounded_positions_with_different_native_offsets_are_not_hits(monkeypatch):
    math,calls=runtime(monkeypatch);memo=module.NativePeakMemo()
    x=np.array([0.,1.00000005,1.1,2.])
    with memo.evaluation(math):
        math.getPowderProfile(None,None,x,position=1.+1e-8)
        math.getPowderProfile(None,None,x,position=1.+2e-8)
    assert len(calls)==2 and memo.hits==0
    assert memo.native_offset_checks==memo.native_offset_collisions==1
    assert memo.diagnostics()['accounted_cache_bytes']<=memo.maximum_bytes


def test_grid_normalization_is_not_replaced_by_compiled_density_identity(monkeypatch):
    math,calls=runtime(monkeypatch);memo=module.NativePeakMemo();x=np.arange(1000.,1007.)
    with memo.evaluation(math):
        a=math.getPowderProfile(None,None,x)
        x[2]+=1e-8
        b=math.getPowderProfile(None,None,x)
    np.testing.assert_array_equal(a[0],b[0])
    assert a[1]!=b[1] and len(calls)==2 and memo.hits==0


@pytest.mark.parametrize('reviewed',[True,False])
def test_unreviewed_binary_and_distinct_native_scalars_keep_original_work(monkeypatch,reviewed):
    math,calls=runtime(monkeypatch,reviewed=reviewed);memo=module.NativePeakMemo();x=np.arange(7.)
    with memo.evaluation(math):
        for alpha in ([.7,.7+1e-5] if reviewed else [.7,.7+1e-12]):
            math.getPowderProfile(None,None,x,alpha=alpha)
    assert len(calls)==2 and not memo.hits


def test_signed_zero_offsets_are_compared_as_bits():
    assert not module._same_tof_offsets(np.array([-0.,1.]),0.,-0.)


def test_python_replacement_is_not_authorized_by_a_matching_binary_file(tmp_path):
    path=tmp_path/'fake.so';path.write_bytes(b'not a native module')
    assert not module._compiled_tof_float32(SimpleNamespace(__file__=str(path),pyepsvoigt=lambda:None))
