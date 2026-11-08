"""The actual Live launcher owns the same provenance boundary as Model jobs."""
import json
from pathlib import Path
import sys

import pytest

from gpx_recipe.live import galaxy_worker
from gpx_recipe.preparation import authority
from gpx_recipe.symmetry.compiler import file_sha
from test_preparation_authority import certified, envelope


@pytest.mark.parametrize('origin',['valid','absent','changed'])
def test_live_wrapper_accepts_only_separately_supplied_creating_job_evidence(tmp_path,monkeypatch,origin):
    *_,certificate,document=certified(tmp_path/'producer',kind='magnetic')
    evidence,_=envelope(tmp_path,certificate)
    if origin=='changed':certificate.write_text(certificate.read_text()+' ')
    entry=document['entries'][0];destination=tmp_path/'job';seen=[]
    def native(packet,directory,result,cache_archive,**kwargs):
        current=authority.magnetic_response_origin(entry['artifacts']['response.npz'],entry['physics_sha256'])
        assert bool(current)==(origin=='valid')
        environment=authority.child_environment()
        if environment:
            owned=Path(environment[authority.ENV_PATH])
            assert owned.is_file() and not owned.is_relative_to(destination)
            seen.append(owned)
        return {'finished':True}
    monkeypatch.setattr(galaxy_worker,'_run_packet',native)
    result=galaxy_worker.run_packet('packet.zip',destination,'result.zip','cache.zip',
        preparation_evidence=None if origin=='absent' else evidence)
    assert result=={'finished':True}
    assert all(not path.exists() for path in seen)
    assert authority.magnetic_response_origin(entry['artifacts']['response.npz'],entry['physics_sha256']) is None


def test_live_wrapper_without_server_evidence_suppresses_inherited_authority(tmp_path,monkeypatch):
    *_,certificate,document=certified(tmp_path/'producer',kind='magnetic')
    evidence,_=envelope(tmp_path,certificate)
    records,_=authority.verify_envelope(evidence)
    inherited=tmp_path/'.radar-preparation-authority-inherited';inherited.mkdir()
    path=inherited/'authority.json'
    path.write_text(json.dumps(dict(schema=authority.CAPSULE,
        implementation_sha256=document['implementation_sha256'],records=records)))
    monkeypatch.setenv(authority.ENV_PATH,str(path));monkeypatch.setenv(authority.ENV_HASH,file_sha(path))
    assert authority.child_environment()
    def native(*args,**kwargs):
        assert authority.child_environment()=={}
        return {'finished':True}
    monkeypatch.setattr(galaxy_worker,'_run_packet',native)
    assert galaxy_worker.run_packet('packet.zip',tmp_path/'job','result.zip','cache.zip')=={'finished':True}
    assert authority.child_environment()


def test_live_cli_keeps_server_provenance_outside_the_scientific_packet(tmp_path,monkeypatch):
    captured=[]
    monkeypatch.setattr(galaxy_worker,'run_packet',lambda *args,**kwargs:captured.append((args,kwargs)))
    monkeypatch.setattr(sys,'argv',['live-worker','packet.zip','job','result.zip','cache.zip',
        '--cache','previous.zip','--preparation-evidence','galaxy-rendered.json'])
    galaxy_worker.main()
    assert captured==[(('packet.zip','job','result.zip','cache.zip'),
        {'cache':'previous.zip','preparation_evidence':'galaxy-rendered.json'})]
