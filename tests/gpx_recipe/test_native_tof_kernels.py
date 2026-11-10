"""Original native primitive kernels survive portable chunks and real edits."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from gpx_recipe.preparation import native_tof_kernels as kernels
from gpx_recipe.preparation import tof_kernel_arrays as arrays
from gpx_recipe.preparation.kernel_store import KernelStore
from gpx_recipe.structural_live.optimization.native_forward import NativeTOFForward


def captured(tmp_path):
    _,s=native_source(tmp_path/'inputs','PNT')
    return s,kernels.capture(s,np.zeros(len(s.radii)))


def test_owned_native_kernels_preserve_full_curves_and_current_observations(tmp_path):
    s,kernel=captured(tmp_path);engine=NativeTOFForward(s);h=s.fast.slow.hist.data['data'][1]
    before=[np.ma.array(a,copy=True) for a in h[:2]]
    for scale in [0.,.07,-.11]:
        p=np.zeros(len(s.radii));p[:s.mode_dimensions]=np.linspace(-scale,scale,s.mode_dimensions)*s.radii[:s.mode_dimensions]
        expected=s.native(p).copy()
        np.testing.assert_array_equal(engine.evaluate(p,kernel=kernel),expected)
    assert engine.kernel_hits>0 and engine.peak_calls==0
    assert kernel.diagnostics()['verified_all_peaks'] and kernel.diagnostics()['qualification_persisted'] is False
    for got,old in zip(h[:2],before):np.testing.assert_array_equal(got,old)


@pytest.mark.parametrize('change',['cell','width','calibration'])
def test_changed_native_inputs_retain_original_peaks_instead_of_shifting_cached_values(tmp_path,change):
    s,kernel=captured(tmp_path);engine=NativeTOFForward(s);p=np.zeros(len(s.radii));engine.evaluate(p,kernel=kernel)
    if change=='cell':p[s.mode_dimensions:]=np.linspace(-.3,.3,s.cell_dimensions)*s.radii[s.mode_dimensions:]
    if change=='width':s.fast.native.parms[f':{s.fast.slow.hist.id}:X']+=.04
    if change=='calibration':s.fast.native.parms[f':{s.fast.slow.hist.id}:Zero']+=.1
    expected=s.native(p).copy()
    np.testing.assert_array_equal(engine.evaluate(p,kernel=kernel),expected)
    assert engine.kernel_misses>0 and engine.peak_calls>0


def test_imported_data_need_complete_native_checks_then_reuse_only_in_this_process(tmp_path):
    s,owned=captured(tmp_path);data=arrays.decode(arrays.encode(owned.data),owned.data.metadata)
    kernels._VERIFIED.clear()
    verified=kernels.verify(data,s)
    assert verified.diagnostics()['native_peak_calls']==data.metadata['reflection_count']
    again=kernels.verify(arrays.decode(arrays.encode(data),data.metadata),s)
    assert again.diagnostics()['native_peak_calls']==0
    kernels._VERIFIED_PID=-1
    fresh=kernels.verify(data,s)
    assert fresh.diagnostics()['native_peak_calls']==data.metadata['reflection_count']


def test_self_consistent_checksum_does_not_authorize_a_false_native_peak(tmp_path):
    s,owned=captured(tmp_path);values={k:v.copy() for k,v in owned.data._arrays.items()}
    values['values'][0]+=np.float32(.125)
    meta=owned.data.metadata;meta['arrays_sha256']=arrays.array_identity(values)
    false=arrays.KernelData(values,meta)
    with pytest.raises(ValueError,match='complete native verification'):kernels.verify(false,s)
    with pytest.raises(TypeError,match='natively qualified'):
        NativeTOFForward(s).evaluate(np.zeros(len(s.radii)),kernel=false)


def test_qualification_cannot_outlive_its_native_function_or_process(tmp_path,monkeypatch):
    s,owned=captured(tmp_path);a=owned.data._arrays;lo,hi=a['bounds'][0]
    assert owned.lookup(a['hkls'][0],a['parameters'][0],a['axis'],lo,hi,function=lambda:None) is None
    owned._pid-=1
    assert owned.lookup(a['hkls'][0],a['parameters'][0],a['axis'],lo,hi,function=owned._function) is None
    original=s.fast.native.math.G2pwd.getEpsVoigt
    monkeypatch.setattr(s.fast.native.math.G2pwd,'getEpsVoigt',lambda *a:original(*a))
    with pytest.raises(ValueError):kernels.verify(owned.data,s)


def test_native_verification_can_be_cancelled_without_creating_a_permission(tmp_path):
    s,owned=captured(tmp_path);kernels._VERIFIED.clear()
    def cancel():raise InterruptedError('cancel')
    with pytest.raises(InterruptedError):kernels.verify(owned.data,s,check_cancel=cancel)
    assert not kernels._VERIFIED


def test_cancellation_is_honored_even_for_a_previously_verified_chunk(tmp_path):
    s,owned=captured(tmp_path)
    def cancel():raise InterruptedError('cancel')
    with pytest.raises(InterruptedError):kernels.verify(owned.data,s,check_cancel=cancel)


def test_cancelled_native_capture_has_no_partial_qualification(tmp_path):
    _,s=native_source(tmp_path/'inputs','PNT');kernels._VERIFIED.clear();calls=[]
    def cancel():
        calls.append(1)
        if len(calls)==2:raise InterruptedError('cancel')
    with pytest.raises(InterruptedError):kernels.capture(s,np.zeros(len(s.radii)),check_cancel=cancel)
    assert not kernels._VERIFIED


def test_cw_context_keeps_its_complete_native_path(tmp_path):
    _,s=native_source(tmp_path/'inputs','PNC')
    with pytest.raises(kernels.KernelUnavailable):kernels.capture(s,np.zeros(len(s.radii)))


CONSUMER='''import sys,json,numpy as np
from pathlib import Path
from test_native_profile_capture import native_source
from gpx_recipe.preparation.kernel_store import KernelStore
from gpx_recipe.preparation.native_tof_kernels import verify
from gpx_recipe.structural_live.optimization.native_forward import NativeTOFForward
work=Path(sys.argv[2]);store=KernelStore.import_part(sys.argv[1],work/'restored')
assert store.coverage()['scientific_qualification'] is False
_,source=native_source(work/'inputs','PNT')
data=store.get(0);kernel=verify(data,source)
assert kernel.diagnostics()['native_peak_calls']==data.metadata['reflection_count']
engine=NativeTOFForward(source);point=np.zeros(len(source.radii))
expected=source.native(point).copy();actual=engine.evaluate(point,kernel=kernel)
np.testing.assert_array_equal(actual,expected)
assert engine.kernel_hits>0 and engine.peak_calls==0
(work/'result.json').write_text(json.dumps(dict(verification=kernel.diagnostics(),forward=engine.diagnostics())))
'''


def test_new_process_imports_data_only_parts_and_independently_qualifies_every_peak(tmp_path):
    s,owned=captured(tmp_path)
    store=KernelStore.create(tmp_path/'store',dict(model_sha256='a'*64,bank='PNT bank 1',cell_count=2))
    store.put(0,owned.data);export=store.export_parts(tmp_path/'parts');assert len(export['parts'])==1
    script=tmp_path/'consumer.py';script.write_text(CONSUMER)
    work=tmp_path/'cold';work.mkdir()
    with (tmp_path/'cold.log').open('w') as log:
        subprocess.run([sys.executable,str(script),str(tmp_path/'parts'/export['parts'][0]['path']),str(work)],
            check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT)
    result=json.loads((work/'result.json').read_text())
    assert result['verification']['method']=='complete original native peak verification'
    assert result['forward']['kernel_hits']>0
