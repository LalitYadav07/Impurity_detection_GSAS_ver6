"""Installed-kernel ABI and complete native profile regression controls."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

powder=pytest.importorskip('GSASII.GSASIIpwd')
from gpx_recipe.live import native_peak_batch as batch
from test_native_profile_capture import native_source


@pytest.fixture
def installed_helper(tmp_path,monkeypatch):
    try:batch.NativePeakBatch(powder)
    except batch.BatchUnavailable:
        source=Path(batch.__file__).with_name('_native_peak_batch.c')
        batch.build(tmp_path)
        monkeypatch.setattr(batch,'_files',lambda:(source,tmp_path/'_native_peak_batch.so',tmp_path/'_native_peak_batch.receipt'))
    return batch.NativePeakBatch(powder)


@pytest.mark.parametrize('case',['overlap','translated','mixed_widths','zero_height','empty'])
def test_complete_original_peak_sum_is_bitwise_identical(installed_helper,case):
    rng=np.random.default_rng(92713);axis=np.linspace(-15.,30.,601)
    if case=='translated':axis+=100000.
    records={}
    for i in range(17 if case!='empty' else 0):
        position=float(rng.uniform(-1,2))+(100000. if case=='translated' else 0.)
        shape=np.exp(rng.uniform(-3,2,size=4)) if case=='mixed_widths' else np.array([.5,.25,1.,.125])
        low=int(rng.integers(0,40));high=int(rng.integers(300,602))
        records[(i,0,0)]=(np.r_[position,shape],0. if case=='zero_height' else float(rng.uniform(-3,4)),low,high)
    expected=np.zeros(len(axis))
    for p,height,low,high in records.values():expected[low:high]+=np.float64(height)*powder.getEpsVoigt(*p,axis[low:high])[0]
    actual=installed_helper.render(records,axis)
    np.testing.assert_array_equal(actual.view(np.uint64),expected.view(np.uint64))
    actual[:]=-999.
    np.testing.assert_array_equal(installed_helper.render(records,axis),expected)
    proof=installed_helper.diagnostics()
    assert proof['reflections']==2*len(records) and proof['native_points']==2*sum(r[3]-r[2] for r in records.values())
    assert proof['maximum_accounted_workspace_bytes']<=batch.MAXIMUM_BYTES


@pytest.mark.parametrize('case',['identical','rounded_shape','offset_collision','rounded_offsets','mixed_windows','zero_height'])
def test_grouped_peaks_preserve_every_reflection_and_exact_original_sum(installed_helper,case):
    axis=np.linspace(99985.,100030.,601)
    if case=='rounded_offsets':axis+=100.
    records={};rng=np.random.default_rng(62914)
    for i in range(512):
        p=np.array([100000.,.5,.25,1.,.125]);low=0;high=len(axis)
        if case in ('offset_collision','rounded_offsets') and i%2:p[0]=np.nextafter(p[0],np.inf)
        if case=='rounded_shape' and i%2:p[1]=np.nextafter(p[1],np.inf)
        if case=='mixed_windows' and i%2:low=1;high-=1
        height=0. if case=='zero_height' else float(rng.uniform(-3.,4.))
        records[(i,0,0)]=(p,height,low,high)
    expected=np.zeros(len(axis))
    for p,height,low,high in records.values():
        expected[low:high]+=np.float64(height)*powder.getEpsVoigt(*p,axis[low:high])[0]
    actual=installed_helper.render(records,axis)
    np.testing.assert_array_equal(actual.view(np.uint64),expected.view(np.uint64))
    proof=installed_helper.diagnostics()
    assert proof['grouped_calls']==1 and proof['reflections']==512
    assert proof['native_reflections']==(2 if case in ('offset_collision','mixed_windows') else 1)
    assert proof['duplicate_reflections']==512-proof['native_reflections']
    assert proof['requested_peak_points']==sum(r[3]-r[2] for r in records.values())
    assert proof['maximum_accounted_workspace_bytes']<=batch.MAXIMUM_BYTES
    if case=='offset_collision':assert proof['compiled_offset_collisions']>0


def test_grouping_workspace_exhaustion_uses_complete_original_batch(installed_helper):
    axis=np.linspace(-10.,10.,101);p=np.array([0.,.5,.25,1.,.125])
    records={(i,0,0):(p,float(i%3-1),0,len(axis)) for i in range(512)}
    *_,required=batch.pack_records(records,axis,batch.MAXIMUM_BYTES)
    expected=np.zeros(len(axis))
    for args,height,low,high in records.values():
        expected+=np.float64(height)*powder.getEpsVoigt(*args,axis)[0]
    actual=installed_helper.render(records,axis,maximum_bytes=required+axis.nbytes)
    np.testing.assert_array_equal(actual.view(np.uint64),expected.view(np.uint64))
    proof=installed_helper.diagnostics()
    assert proof['native_reflections']==512 and proof['grouped_calls']==0


@pytest.mark.parametrize('bins,parts',[(101,2),(5001,4)])
def test_sequential_parts_keep_one_original_accumulator_with_cancellation(installed_helper,bins,parts):
    axis=np.linspace(-10.,10.,bins);records={}
    for i in range(1024):
        p=np.array([float(i%16)/10,.5,.25,1.,.125])
        records[(i,0,0)]=(p,[1e16,1.,-1e16,0.][i%4],0,len(axis))
    *_,required=batch.pack_records(records,axis,batch.MAXIMUM_BYTES)
    maximum=required+axis.nbytes+28*len(records)+8+15*(584+4*bins)
    expected=np.zeros(len(axis))
    for p,height,low,high in records.values():
        expected+=np.float64(height)*powder.getEpsVoigt(*p,axis)[0]
    actual=installed_helper.render(records,axis,maximum_bytes=maximum)
    np.testing.assert_array_equal(actual.view(np.uint64),expected.view(np.uint64))
    proof=installed_helper.diagnostics()
    assert proof['sequentially_split_calls']==1 and proof['grouped_parts']==parts
    assert proof['native_reflections']==16*parts and proof['reflections']==1024
    assert proof['maximum_accounted_workspace_bytes']<=maximum


def test_changed_grid_during_grouping_is_not_published(installed_helper,monkeypatch):
    axis=np.linspace(-10.,10.,101);p=np.array([0.,.5,.25,1.,.125])
    records={(i,0,0):(p,1.,0,len(axis)) for i in range(512)}
    original=installed_helper.group_values
    def changed_grid(*args):
        result=original(*args);axis[0]-=1.;return result
    monkeypatch.setattr(installed_helper,'group_values',changed_grid)
    with pytest.raises(batch.BatchUnavailable,match='grid changed'):installed_helper.render(records,axis)
    assert installed_helper.diagnostics()['calls']==0


def test_nonfinite_original_native_peak_is_rejected_even_with_zero_intensity(installed_helper):
    axis=np.linspace(99985.,100030.,601)+1e8;p=np.array([1e5,.5,.25,1.,.125])
    assert not np.isfinite(powder.getEpsVoigt(*p,axis)[0]).all()
    records={(i,0,0):(p,0.,0,len(axis)) for i in range(512)}
    with pytest.raises(ValueError,match='nonfinite profile'):installed_helper.render(records,axis)
    assert installed_helper.diagnostics()['calls']==0


@pytest.mark.parametrize('change',['compiled','wrapper','fork','thread','floating'])
def test_changed_native_ownership_or_controls_do_not_reuse_a_pointer(installed_helper,monkeypatch,change):
    owner=installed_helper
    if change=='compiled':monkeypatch.setattr(powder.pyd,'pyepsvoigt',lambda:None)
    elif change=='wrapper':monkeypatch.setattr(powder,'getEpsVoigt',lambda *a:None)
    elif change=='fork':monkeypatch.setattr(batch.os,'getpid',lambda:owner.pid+1)
    elif change=='thread':monkeypatch.setattr(batch.threading,'current_thread',lambda:object())
    elif change=='floating':monkeypatch.setattr(batch,'floating_state',lambda:(1919,16256))
    with pytest.raises(batch.BatchUnavailable,match='changed'):
        owner.render({},np.arange(8.))


def test_nondefault_arithmetic_modes_use_original_native_calculation(installed_helper):
    library=ctypes.CDLL('libm.so.6');environment=ctypes.create_string_buffer(256)
    library.fegetenv.argtypes=[ctypes.c_void_p];library.fegetenv.restype=ctypes.c_int
    library.fesetenv.argtypes=[ctypes.c_void_p];library.fesetenv.restype=ctypes.c_int
    library.fesetround.argtypes=[ctypes.c_int];library.fesetround.restype=ctypes.c_int
    assert library.fegetenv(environment)==0
    try:
        for flag in [0x400,0x800,0xc00]:
            assert library.fesetround(flag)==0
            with pytest.raises(batch.BatchUnavailable,match='controls'):installed_helper.render({},np.arange(8.))
            assert batch.for_powder(powder) is None
    finally:assert library.fesetenv(environment)==0


def test_frozen_job_uses_installed_helper_and_ignores_adjacent_executable(tmp_path,installed_helper):
    from gpx_recipe.symmetry.compiler import freeze_runtime
    root=tmp_path/'frozen';freeze_runtime(root)
    folder=root/'gpx_recipe/live'
    (folder/'_native_peak_batch.so').write_bytes(b'Untrusted job bytes must never be loaded')
    (folder/'_native_peak_batch.receipt').write_text('{}')
    command='''import json,numpy as np
from pathlib import Path
from GSASII import GSASIIpwd as p
from gpx_recipe.live.native_peak_batch import NativePeakBatch
b=NativePeakBatch(p)
assert b.helper_path.parent in (Path('/opt/radar-native/gpx_recipe/live'),Path('/opt/radar-symmetry/gpx_recipe/live'))
x=np.arange(8.);args=np.array([1.,.5,.25,1.,.125])
expected=p.getEpsVoigt(*args,x)[0].astype(float)
np.testing.assert_array_equal(b.render({(1,0,0):(args,1.,0,8)},x),expected)
print(json.dumps(b.diagnostics()))
'''
    result=subprocess.run([sys.executable,'-c',command],env={**os.environ,'PYTHONPATH':str(root)},
        capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr


def test_unavailable_batch_does_not_repeat_failed_work_and_reanchors_python_path(tmp_path,installed_helper,monkeypatch):
    from gpx_recipe.structural_live.optimization.native_forward import for_source
    _,source=native_source(tmp_path/'source','PNT');engine=for_source(source)
    point=np.zeros(len(source.radii));expected=source.native(point).copy();seen=[]
    def unavailable(*args,**kwargs):seen.append(True);raise batch.BatchUnavailable('Injected unavailable helper')
    monkeypatch.setattr(engine.native_batch,'render',unavailable)
    np.testing.assert_array_equal(engine.evaluate(point),expected)
    assert engine.native_batch_disabled and not engine.qualified
    np.testing.assert_array_equal(engine.evaluate(point),expected)
    assert engine.qualified and len(seen)==1 and engine.fallback_calls==1


def test_replaced_native_owner_during_batch_is_an_error_before_publication(tmp_path,installed_helper,monkeypatch):
    from gpx_recipe.structural_live.optimization.native_forward import for_source
    _,source=native_source(tmp_path/'source','PNT');engine=for_source(source)
    point=np.zeros(len(source.radii));engine.evaluate(point)
    original=engine.native_batch.render
    def replaced(*args,**kwargs):
        result=original(*args,**kwargs)
        def fail():raise RuntimeError('Native owner replaced during batch')
        monkeypatch.setattr(source.fast.native,'_check',fail)
        return result
    monkeypatch.setattr(engine.native_batch,'render',replaced)
    with pytest.raises(RuntimeError,match='replaced during batch'):engine.evaluate(point)


@pytest.mark.parametrize('change',['cell','profile','helper_absent','float_controls'])
def test_whole_native_forward_uses_batch_or_unchanged_fallback(tmp_path,installed_helper,monkeypatch,change):
    from gpx_recipe.structural_live.optimization import native_forward as module
    _,source=native_source(tmp_path/'source','PNT');native=source.fast.native
    if change=='helper_absent':monkeypatch.setattr(batch,'for_powder',lambda _:None)
    engine=module.for_source(source);assert engine is not None
    assert bool(engine.native_batch)==(change!='helper_absent')
    for i in range(3):
        point=np.linspace(-.1,.2,len(source.radii))*source.radii*(i+1)/3
        if change=='profile':native.parms[f':{source.fast.slow.hist.id}:X']+=.01
        expected=source.native(point).copy()
        np.testing.assert_array_equal(engine.evaluate(point),expected)
    if change!='helper_absent':assert engine.native_batch.calls>0
    if change=='float_controls':
        monkeypatch.setattr(module,'floating_state',lambda:(1919,16256))
        expected=source.native(point).copy()
        np.testing.assert_array_equal(engine.evaluate(point),expected)
        assert engine.fallback_calls==1
