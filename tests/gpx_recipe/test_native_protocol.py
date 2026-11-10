"""The Galaxy entry point accepts the same magnetic inventory as NOVA."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from gpx_recipe.symmetry.compiler import file_sha, package_signature
from gpx_recipe.symmetry.native_protocol import validate_native_payload,validate_native_timeout,native_timeout_limit


def test_only_offline_training_and_complete_simulation_units_have_long_deadlines():
    assert native_timeout_limit('preparation-structural-training')==7200
    assert validate_native_timeout('preparation-structural-training',7200)==7200
    assert native_timeout_limit('preparation-structural-examples')==3600
    assert validate_native_timeout('preparation-structural-examples',3600)==3600
    for operation in ('inspect','k-portfolio','preparation-batch','structural-snapshot'):
        assert native_timeout_limit(operation)==600
        with pytest.raises(ValueError,match='deadline'):validate_native_timeout(operation,601)
    with pytest.raises(ValueError,match='deadline'):
        validate_native_timeout('preparation-structural-examples',3601)


@pytest.mark.parametrize('value',[True,False,0,-1,7201,float('nan'),float('inf'),'7200'])
def test_invalid_offline_training_deadlines_are_rejected(value):
    with pytest.raises(ValueError,match='deadline'):
        validate_native_timeout('preparation-structural-training',value)


@pytest.mark.parametrize('moment_model', [None, 'collinear', 'symmetry_allowed'])
def test_inventory_reaches_native_worker_with_its_requested_geometry(tmp_path, monkeypatch, moment_model):
    script=Path(__file__).resolve().parents[2]/'scripts/ndip_model_campaign.py'
    spec=importlib.util.spec_from_file_location('inventory_worker_test',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=tmp_path/'source.gpx';source.write_bytes(b'exact captured project')
    assets=tmp_path/'assets.zip'
    with zipfile.ZipFile(assets,'w'):pass
    payload=dict(source='source.gpx',source_sha256=file_sha(source),parent='nuclear',
        parent_setting_sha256='a'*64,sites=[],directions=['x','y','z'],catalogs=[])
    if moment_model is not None:payload['moment_model']=moment_model
    request=dict(schema='radar-model-native/v1',operation='k-inventory',payload=payload,
        implementation_sha256=package_signature(),expected_source_sha256=file_sha(source),
        input_hashes={'source.gpx':file_sha(source)},dispatch_key='b'*24,timeout=120)
    path=tmp_path/'request.json';path.write_text(json.dumps(request))
    monkeypatch.setattr(module,'require_adapters',lambda:None)
    monkeypatch.setattr('gpx_recipe.symmetry.compiler.native_environment',lambda:{'PYTHONPATH':''})
    observed=[]
    def run(command,**kwargs):
        captured=json.loads(Path(command[-2]).read_text());observed.append(captured)
        assert file_sha(captured['source'])==file_sha(source)
        Path(command[-1]).write_text(json.dumps({'schema':'radar-k-native-inventory/v1','families':[]}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module,'run_scientific',run)
    result=module.work('native',source,assets,path,tmp_path/'worker',tmp_path/'result.zip')
    assert result['result']['schema']=='radar-k-native-inventory/v1'
    assert len(observed)==1 and observed[0].get('moment_model')==moment_model
    with zipfile.ZipFile(tmp_path/'result.zip') as archive:
        assert 'response.json' in archive.namelist()


@pytest.mark.parametrize('change', [{'moment_model':'invented'}, {'credentials':{}}, {'sites':None,'extra':True}])
def test_inventory_keeps_a_closed_payload_contract(change):
    payload=dict(source='source.gpx',source_sha256='a'*64,parent='nuclear',
        parent_setting_sha256='b'*64,sites=[],directions=['x'],catalogs=[])
    payload.update(change)
    with pytest.raises(ValueError):validate_native_payload('k-inventory',payload)
