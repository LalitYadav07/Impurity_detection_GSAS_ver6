import json
from pathlib import Path
import stat
import zipfile

import pytest

from gpx_recipe.battery.transport import extract_preparation_packet


def packet(path, members):
    with zipfile.ZipFile(path, 'w') as archive:
        for name, value in members:
            archive.writestr(name, value)
    return path


def test_preparation_packet_round_trip_retains_exact_inputs(tmp_path):
    source = packet(tmp_path/'inputs.zip', [('preparation.json', '{"schema":"test"}'), ('bank.gsa', b'exact bytes')])
    assert extract_preparation_packet(source, tmp_path/'out') == {'schema':'test'}
    assert (tmp_path/'out/bank.gsa').read_bytes() == b'exact bytes'


@pytest.mark.parametrize('name', ['../secret', '/secret', 'nested/file', 'x\\file', '..', ''])
def test_preparation_packet_rejects_paths(tmp_path, name):
    source = packet(tmp_path/'inputs.zip', [('preparation.json', '{}'), (name, 'unsafe')])
    with pytest.raises(ValueError, match='unsafe'):
        extract_preparation_packet(source, tmp_path/'out')


def test_preparation_packet_rejects_symlinks(tmp_path):
    source = tmp_path/'inputs.zip'
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('preparation.json', '{}')
        link=zipfile.ZipInfo('link');link.external_attr=(stat.S_IFLNK|0o777)<<16
        archive.writestr(link, '/private/secret')
    with pytest.raises(ValueError, match='unsafe'):
        extract_preparation_packet(source, tmp_path/'out')


def test_prepare_worker_uses_shared_native_preparer_and_publishes_provenance(tmp_path, monkeypatch):
    from gpx_recipe import worker
    from gpx_recipe.battery import preparation
    import hashlib
    source=packet(tmp_path/'inputs.zip', [('preparation.json', '{}'), ('observed.gsa', 'private input')])
    def prepare(spec, inputs, destination):
        assert (inputs/'observed.gsa').read_text()=='private input'
        destination.mkdir();(destination/'prepared.gpx').write_bytes(b'native model')
        context={'source':{'sha256':hashlib.sha256(b'native model').hexdigest()}}
        (destination/'context.json').write_text(json.dumps(context))
        result={'status':'prepared_needs_review','project_sha256':context['source']['sha256']}
        (destination/'preparation.json').write_text(json.dumps(result))
        return result
    monkeypatch.setattr(preparation,'prepare_model',prepare)
    monkeypatch.setattr(worker,'write_project_view',lambda source,target: target.write_text('{}'))
    from gpx_recipe.battery.transport import implementation_signature
    response=worker.work('prepare',source,tmp_path/'worker',plan={'implementation_sha256':implementation_signature()})
    assert response['status']=='model_prepared'
    assert response['final_gpx']=='verified_model.gpx'
    assert 'inputs/observed.gsa' in response['files']
    assert 'model/preparation.json' in response['files']
    from radar_pd_nova.refinement_runtime import verify_result_files
    verify_result_files(tmp_path/'worker',response,action='prepare')


def test_prepare_rejects_other_implementation_before_unpacking(tmp_path,monkeypatch):
    from gpx_recipe import worker
    inspections=[]
    monkeypatch.setattr(worker,'write_project_view',lambda *args:inspections.append(args))
    source=packet(tmp_path/'inputs.zip', [('preparation.json', '{}')])
    response=worker.work('prepare',source,tmp_path/'worker',plan={'implementation_sha256':'0'*64})
    assert response['status']=='failed' and 'implementation differs' in response['error']
    assert not (tmp_path/'worker/inputs').exists()
    assert inspections == []  # A rejected ZIP is never opened as a GPX.
