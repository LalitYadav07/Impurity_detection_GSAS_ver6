import hashlib,json,stat,zipfile
from pathlib import Path
import pytest
from gpx_recipe.live.packet import create_packet,unpack_packet,pack_cache,unpack_cache,canonical
from gpx_recipe.live.contracts import file_hash
from test_webmon_hardening import scientific_request


def test_packet_is_portable_deterministic_and_verifies_original_inputs(tmp_path):
    source=tmp_path/'source';source.mkdir();request,_=scientific_request(source)
    p1=tmp_path/'one.zip';p2=tmp_path/'two.zip'
    identity=create_packet(request,p1);assert create_packet(request,p2)==identity
    assert p1.read_bytes()==p2.read_bytes()
    restored,found=unpack_packet(p1,tmp_path/'out')
    assert found==identity
    assert Path(restored['source_gpx']).read_bytes()==Path(request['source_gpx']).read_bytes()
    assert str(tmp_path/'source') not in (tmp_path/'out'/'manifest.json').read_text() if (tmp_path/'out'/'manifest.json').exists() else True


def test_packet_rejects_tampering_traversal_and_symlink(tmp_path):
    source=tmp_path/'source';source.mkdir();request,_=scientific_request(source);packet=tmp_path/'packet.zip';create_packet(request,packet)
    with zipfile.ZipFile(packet) as z:payload={v:z.read(v) for v in z.namelist()}
    payload['inputs/source.gpx']=b'tampered'
    bad=tmp_path/'bad.zip'
    with zipfile.ZipFile(bad,'w') as z:
        for name,data in payload.items():z.writestr(name,data)
    with pytest.raises(ValueError,match='checksum'):unpack_packet(bad,tmp_path/'bad-work')
    with zipfile.ZipFile(bad,'w') as z:z.writestr('../escape',b'x')
    with pytest.raises(ValueError,match='Unsafe'):unpack_packet(bad,tmp_path/'unsafe-work')
    with zipfile.ZipFile(bad,'w') as z:
        info=zipfile.ZipInfo('link');info.external_attr=(stat.S_IFLNK|0o777)<<16;z.writestr(info,'elsewhere')
    with pytest.raises(ValueError,match='nonregular'):unpack_packet(bad,tmp_path/'link-work')


def test_response_cache_is_exact_and_does_not_overwrite_work(tmp_path):
    first=tmp_path/'first';model=first/'models'/'test';model.mkdir(parents=True)
    for name in ['kernel.json','magnetic-refinement.gpx']:(model/name).write_bytes(name.encode())
    import numpy as np
    np.savez_compressed(model/'kernel.npz',x=np.array([1.,2.]))
    (model/'posterior.npz').write_bytes(b'not cached')
    cache=tmp_path/'cache.zip';pack_cache(first,cache)
    second=tmp_path/'second';unpack_cache(cache,second)
    assert not (second/'models/test/posterior.npz').exists()
    assert (second/'models/test/kernel.npz').read_bytes()==(model/'kernel.npz').read_bytes()
    with pytest.raises(ValueError,match='replace'):unpack_cache(cache,second)


def test_worker_envelope_runs_without_nova_and_records_packet_identity(tmp_path,monkeypatch):
    from gpx_recipe.live.galaxy_worker import run_packet
    from gpx_recipe.live import worker
    source=tmp_path/'source';source.mkdir();request,_=scientific_request(source);packet=tmp_path/'packet.zip';identity=create_packet(request,packet)
    def native(path):
        root=path.parent;model=root/'models/test';model.mkdir(parents=True)
        for name in ['kernel.json','kernel.npz','magnetic-refinement.gpx','posterior.npz','posterior.json','posterior-maximum.gpx']:(model/name).write_text(name)
        (root/'result.json').write_text(json.dumps(dict(status='complete',source_revision=request['source_revision'],reference_revision=request['reference_revision'],request_sha256=file_hash(path),rows=[dict(id='test',posterior=str(model/'posterior.json'),fitted_gpx=str(model/'posterior-maximum.gpx'),fitted_gpx_sha256=file_hash(model/'posterior-maximum.gpx'))])))
    monkeypatch.setattr(worker,'run',native)
    result=run_packet(packet,tmp_path/'job',tmp_path/'result.zip',tmp_path/'cache.zip')
    assert result['packet_id']==identity
    assert result['rows'][0]['posterior']=='models/test/posterior.json'
    assert result['native_result_sha256']==file_hash(tmp_path/'job/native-result.json')
    with zipfile.ZipFile(tmp_path/'result.zip') as z:assert 'transport.json' in z.namelist() and 'verified_projects/test.gpx' in z.namelist()
