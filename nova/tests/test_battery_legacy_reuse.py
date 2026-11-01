"""Old battery artifacts remain readable; lossy profiles cannot seed new fits."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from gpx_recipe.battery.profiles import legacy_profile_metadata
from radar_pd_nova import battery_preparation as preparation
from radar_pd_nova.refinement_runtime import LocalRefinementRuntime


@pytest.fixture
def bundle(tmp_path):
    inputs=tmp_path/'inputs';inputs.mkdir()
    profile=inputs/'synthetic.prm'
    rows={'INS   HTYPE ':'PNTR','INS   FPATH1':'19.5','INS  2 ICONS':'2900 0 0',
          'INS  2BNKPAR':'2 31','INS  2PRCF1 ':'3 21 .002',
          'INS  2PRCF11':'1 2 3 4','INS  2PRCF12':'5 6 7 8',
          'INS  2PRCF13':'9 0 0 0','INS  2PRCF14':'0 0 0 0',
          'INS  2PRCF15':'0 0 0 0','INS  2PRCF16':'0'}
    profile.write_text(''.join(key+value+'\n' for key,value in rows.items()))
    result=dict(specification=dict(banks=[dict(physical_bank=2,profile_bank=2,profile=dict(path=profile.name))]),
        banks=[dict(physical_bank=2,profile_conversion=dict(adapter='native-gsas-legacy-tof/v1',
            converted_parameters=legacy_profile_metadata(profile,2)['coefficient_mapping']))])
    return tmp_path,result


def test_matching_coefficients_are_allowed_regardless_of_converter_label(bundle):
    root,result=bundle
    preparation.check_prepared_profiles(root,result)


def test_omitted_width_cannot_be_hidden_by_a_current_converter_label(bundle):
    root,result=bundle
    result['banks'][0]['profile_conversion'].update(adapter='gsas-isotropic-tof-coefficients/v2')
    result['banks'][0]['profile_conversion']['converted_parameters']['Y']=0.
    before=deepcopy(result)
    with pytest.raises(ValueError,match='Physical bank 2 needs a new preparation.*Y'):
        preparation.check_prepared_profiles(root,result)
    assert result==before


def test_lossy_preparation_cannot_register_a_new_source(bundle,monkeypatch):
    root,result=bundle
    result['banks'][0]['profile_conversion']['converted_parameters']['sig-0']=0.
    monkeypatch.setattr(preparation,'verify_prepared',lambda path:result)
    workspace=SimpleNamespace(runtime=SimpleNamespace(kind='local'))
    with pytest.raises(ValueError,match='sig-0'):
        preparation.prepared_source(workspace,root,result)


@pytest.mark.parametrize('action',['preview','run'])
def test_cached_source_is_rechecked_before_starting_a_worker(bundle,monkeypatch,action):
    root,result=bundle
    result['banks'][0]['profile_conversion']['converted_parameters']['Y']=0.
    monkeypatch.setattr(preparation,'verify_prepared',lambda path:result)
    runtime=LocalRefinementRuntime(root/'runtime',python='unused')
    with pytest.raises(ValueError,match='needs a new preparation'):
        runtime.call(action,dict(metadata=dict(preparation=str(root/'model/preparation.json'))),root/'execution')
    assert not (root/'execution').exists()


def test_historical_inspection_does_not_require_repreparation(bundle,monkeypatch):
    root,result=bundle
    runtime=LocalRefinementRuntime(root/'runtime',python='unused')
    def reached(source):raise RuntimeError('inspection may resolve the retained GPX')
    def forbidden(path):raise AssertionError('Historical inspection must not be blocked')
    monkeypatch.setattr(runtime,'resolve',reached)
    monkeypatch.setattr(preparation,'verify_prepared',forbidden)
    with pytest.raises(RuntimeError,match='inspection may resolve'):
        runtime.call('inspect',dict(metadata=dict(preparation=str(root/'model/preparation.json'))),root/'inspection')
