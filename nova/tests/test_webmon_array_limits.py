import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from gpx_recipe.live.array_limits import validate_npz
from gpx_recipe.live import packet
from test_webmon_hardening import scientific_request


def header_only(shape, dtype='<f8'):
    stream=io.BytesIO()
    np.lib.format.write_array_header_1_0(stream,dict(descr=dtype,fortran_order=False,shape=shape))
    return stream.getvalue()


def test_oversized_declared_array_is_rejected_without_allocation(tmp_path,monkeypatch):
    path=tmp_path/'bomb.npz'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ['x','y','sigma','mask']:
            z.writestr(name+'.npy',header_only((10**12,), '|b1' if name=='mask' else '<f8'))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('NumPy must not load oversized array'))
    with pytest.raises(ValueError,match='budget'):
        validate_npz(path,observation=True)


def test_observation_header_guard_is_used_before_request_load(tmp_path,monkeypatch):
    from gpx_recipe.live.contracts import validate_request,file_hash
    request,_=scientific_request(tmp_path)
    path=tmp_path/'bomb.npz'
    with zipfile.ZipFile(path,'w') as z:
        for name in ['x','y','sigma','mask']:
            z.writestr(name+'.npy',header_only((10**12,), '|b1' if name=='mask' else '<f8'))
    raw=tmp_path/'raw.dat';raw.write_bytes(b'raw')
    request.update(observation=str(path),observation_sha256=file_hash(path),
        observation_source=str(raw),observation_source_sha256=file_hash(raw),
        source_revision='PG3_1:'+file_hash(raw),cumulative_policy='replace')
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('NumPy must not load oversized array'))
    with pytest.raises(ValueError,match='budget'):validate_request(request)


@pytest.mark.parametrize('change',[dict(x=np.ones((2,2))),dict(mask=np.ones(2)),dict(x=np.array([object(),object()]))])
def test_invalid_observation_shape_or_dtype_is_rejected(tmp_path,change):
    path=tmp_path/'array.npz';values=dict(x=np.ones(2),y=np.ones(2),sigma=np.ones(2),mask=np.zeros(2,dtype=bool))
    values.update(change);np.savez_compressed(path,**values)
    with pytest.raises(ValueError):validate_npz(path,observation=True)


def test_valid_compressed_observation_and_scalar_metadata_are_supported(tmp_path):
    path=tmp_path/'array.npz'
    np.savez_compressed(path,x=np.arange(5.),y=np.ones(5),sigma=np.ones(5),mask=np.zeros(5,dtype=bool))
    assert validate_npz(path,observation=True)
    np.savez_compressed(path,coordinate=np.array('tof'),reviewed=np.array(True),x=np.arange(5.))
    assert validate_npz(path)


def test_truncated_payload_and_repeated_array_names_are_rejected(tmp_path):
    path=tmp_path/'array.npz'
    with zipfile.ZipFile(path,'w') as z:z.writestr('x.npy',header_only((2,)))
    with pytest.raises(ValueError,match='payload'):validate_npz(path)
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('x.npy',header_only((2,)))
        with pytest.warns(UserWarning):z.writestr('x.npy',header_only((2,)))
    with pytest.raises(ValueError,match='inventory'):validate_npz(path)


def test_oversized_cache_is_omitted_without_reading_or_invalid_archive(tmp_path,monkeypatch):
    model=tmp_path/'source/models/test';model.mkdir(parents=True)
    for name in packet.CACHE_FILES:(model/name).write_bytes(b'x'*50)
    monkeypatch.setattr(packet,'MAX_CACHE_BYTES',100)
    # A too-large cache group is omitted before any member or NPZ is read.
    monkeypatch.setattr(Path,'read_bytes',lambda *a:pytest.fail('Oversized cache should not be read'))
    archive=tmp_path/'cache.zip';packet.pack_cache(model.parents[1],archive)
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('manifest.json'))
        assert manifest['files']=={} and manifest['omitted_models']==['test']
    packet.unpack_cache(archive,tmp_path/'restored')


def test_result_extraction_checks_total_size_before_writing(tmp_path,monkeypatch):
    archive=tmp_path/'results.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('big.txt',b'x'*32)
    original=packet._entries
    monkeypatch.setattr(packet,'_entries',lambda z:original(z,maximum=16))
    output=tmp_path/'out'
    with pytest.raises(ValueError,match='limits'):packet.extract_result_archive(archive,output)
    assert list(output.iterdir())==[]


def test_result_nested_npz_cannot_bypass_expansion_budget(tmp_path):
    npz=io.BytesIO()
    with zipfile.ZipFile(npz,'w') as z:z.writestr('x.npy',header_only((10**12,)))
    archive=tmp_path/'results.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('models/test/kernel.npz',npz.getvalue())
    with pytest.raises(ValueError,match='budget'):
        packet.extract_result_archive(archive,tmp_path/'out')
