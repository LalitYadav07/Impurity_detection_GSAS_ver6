"""The actual compiled TOF routine, not a surrogate precision emulator."""
from types import SimpleNamespace

import numpy as np
import pytest

P=pytest.importorskip('GSASII.GSASIIpwd')
from gpx_recipe.live import native_peak_memo as module


def runtime(monkeypatch):
    def driver(_a,_b,x,*,position=0.,alpha=.7):
        return P.getEpsVoigt(position,alpha,.3,1.,.02,x[1:-1])
    original=module._source_hash
    monkeypatch.setattr(module,'_source_hash',lambda f:module._DRIVER if f is driver else original(f))
    return SimpleNamespace(getPowderProfile=driver,G2pwd=P,G2mp=SimpleNamespace(InitMP=lambda:(False,0)))


def test_reviewed_native_binary_has_the_declared_precision():
    assert type(P.pyd.pyepsvoigt).__name__=='fortran'
    assert module._compiled_tof_float32(P.pyd)
    assert P.getEpsVoigt(0.,.7,.3,1.,.02,np.linspace(-3.,3.,31))[0].dtype==np.float32


@pytest.mark.parametrize('changed',['shape','position','position_collision'])
def test_actual_kernel_equivalence_and_nearby_non_equivalence(tmp_path,monkeypatch,changed):
    math=runtime(monkeypatch);original=math.getPowderProfile;memo=module.NativePeakMemo()
    if changed=='shape':
        x=np.linspace(-3.,3.,31);settings=[dict(alpha=.7),dict(alpha=.7+1e-12)]
    elif changed=='position':
        x=np.arange(1000.,1007.);settings=[dict(position=1.),dict(position=1.+1e-12)]
    else:
        x=np.array([0.,1.00000005,1.1,2.]);settings=[dict(position=1.+1e-8),dict(position=1.+2e-8)]
    reference=[original(None,None,x,**kwargs) for kwargs in settings]
    with memo.evaluation(math):actual=[math.getPowderProfile(None,None,x,**kwargs) for kwargs in settings]
    for a,b in zip(actual,reference):
        np.testing.assert_array_equal(a[0],b[0]);assert a[1]==b[1]
    if changed=='position_collision':
        assert memo.native_offset_collisions==1 and memo.misses==2 and memo.hits==0
    else:assert memo.native_float32_hits==memo.hits==1 and memo.misses==1
    assert not module._LOCK.locked()
