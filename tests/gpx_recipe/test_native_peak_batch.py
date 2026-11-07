"""Malformed data cannot reach a native pointer or silently omit reflections."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpx_recipe.live import native_peak_batch as batch


def record(low=1,high=6):
    return {(-1,2,3):(np.array([2.,.5,.25,1.,.125]),3.,low,high)}


def test_bounded_packing_retains_order_zero_negative_intensities_and_inputs():
    axis=np.arange(8.);records=record();records[(0,0,1)]=(np.array([3.,.7,.2,2.,.1]),0.,0,8)
    records[(1,0,0)]=(np.array([4.,.2,.5,1.,.2]),-2.,2,7)
    before=axis.copy()
    p,h,b,maximum,required=batch.pack_records(records,axis,batch.MAXIMUM_BYTES)
    np.testing.assert_array_equal(h,[3.,0.,-2.]);np.testing.assert_array_equal(b,[[1,6],[0,8],[2,7]])
    np.testing.assert_array_equal(p,np.array([r[0] for r in records.values()]))
    np.testing.assert_array_equal(axis,before)
    assert maximum==8 and required>=p.nbytes+h.nbytes+b.nbytes+16*len(axis)+8*maximum


@pytest.mark.parametrize('low,high',[(-1,4),(1,9),(1,2),(2,2),(True,4),(1.,5)])
def test_invalid_windows_are_refused_before_a_native_call(low,high):
    with pytest.raises(batch.BatchUnavailable,match='window'):
        batch.pack_records(record(low,high),np.arange(8.),batch.MAXIMUM_BYTES)


@pytest.mark.parametrize('change',['nan_grid','nonordered','strided','float32','nonfinite_shape','nonfinite_height','overflow_shape','wrong_parameters','wrong_record'])
def test_unsupported_inputs_keep_the_original_calculation(change):
    axis=np.arange(8.);records=record()
    p,h,lo,hi=next(iter(records.values()))
    if change=='nan_grid':axis[2]=np.nan
    elif change=='nonordered':axis[3]=axis[2]
    elif change=='strided':axis=np.arange(16.)[::2]
    elif change=='float32':axis=axis.astype(np.float32)
    elif change=='nonfinite_shape':p[1]=np.nan
    elif change=='overflow_shape':p[1]=1e100
    elif change=='nonfinite_height':h=np.inf
    elif change=='wrong_parameters':p=np.ones(6)
    elif change=='wrong_record':records={(-1,2,3):(p,h,lo)}
    if change!='wrong_record':records={(-1,2,3):(p,h,lo,hi)}
    with pytest.raises(batch.BatchUnavailable):batch.pack_records(records,axis,batch.MAXIMUM_BYTES)


def test_workspace_exhaustion_refuses_complete_batch_instead_of_truncation():
    with pytest.raises(batch.BatchUnavailable,match='workspace'):
        batch.pack_records(record(0,8),np.arange(8.),250)
    p,h,b,maximum,required=batch.pack_records({},np.arange(8.),1024)
    assert p.shape==(0,5) and b.shape==(0,2) and not len(h) and maximum==2 and required<=1024


@pytest.mark.parametrize('budget',[0,-1,True,1.5,batch.MAXIMUM_BYTES+1])
def test_invalid_workspace_limit_is_not_a_different_execution_mode(budget):
    with pytest.raises(ValueError,match='workspace'):batch.pack_records({},np.arange(8.),budget)


def test_receipt_cannot_authorize_changed_helper_bytes(tmp_path,monkeypatch):
    source=tmp_path/'helper.c';source.write_text('reviewed code')
    binary=tmp_path/'helper.so';binary.write_bytes(b'changed')
    receipt=tmp_path/'helper.receipt'
    receipt.write_text(json.dumps(dict(schema=batch.SCHEMA,source_sha256=batch._sha(source),
        binary_sha256='0'*64,flags=list(batch.FLAGS))))
    monkeypatch.setattr(batch,'_files',lambda:(source,binary,receipt))
    monkeypatch.setattr(batch.ctypes,'CDLL',lambda *_:pytest.fail('Changed bytes must never be loaded'))
    with pytest.raises(batch.BatchUnavailable,match='receipt'):batch._load()


def test_missing_helper_retains_original_kernel_without_runtime_compilation(monkeypatch):
    monkeypatch.setattr(batch,'floating_state',lambda:batch.DEFAULT_FLOATING)
    monkeypatch.setattr(batch,'_compiled_tof_float32',lambda _:False)
    monkeypatch.setattr(batch,'build',lambda *_:pytest.fail('No runtime compilation'))
    assert batch.for_powder(SimpleNamespace(pyd=object())) is None


def test_scientific_package_identity_includes_native_source_and_freezes_it(tmp_path,monkeypatch):
    from gpx_recipe.symmetry import compiler
    # Exercise the real package inventory; executable artifacts are excluded
    # while the installed reviewed C source is included in portable source.
    source=Path(batch.__file__).with_name('_native_peak_batch.c')
    assert source.is_file()
    compiler.freeze_runtime(tmp_path)
    assert (tmp_path/'gpx_recipe/live/_native_peak_batch.c').read_bytes()==source.read_bytes()
    assert not list(tmp_path.rglob('*.so')) and not list(tmp_path.rglob('*.receipt'))
