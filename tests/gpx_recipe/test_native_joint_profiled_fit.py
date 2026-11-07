"""Coupled native updates retain full saved-project and observation checks."""
from functools import partial

import pytest

pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from gpx_recipe.structural_live.service import fit_snapshot,SearchSettings
from gpx_recipe.structural_live.optimization import profiled_native_fit
from gpx_recipe.symmetry.compiler import file_sha


@pytest.mark.parametrize('kind',['PNT','PNC'])
def test_joint_native_fit_publishes_the_same_observation_and_refinable_scope(tmp_path,monkeypatch,kind):
    original,_=native_source(tmp_path/'inputs',kind);before=file_sha(original)
    monkeypatch.setattr(profiled_native_fit,'fit',partial(profiled_native_fit.fit,joint_steps=True))
    result=fit_snapshot(tmp_path/'inputs/prepared',tmp_path/'fit',
        settings=SearchSettings(seconds=8.,starts=2))
    assert result['result']['method']=='profiled_native_joint'
    assert result['all_physical_modes_retained'] and result['physical_mode_count']>0
    assert result['cell_parameter_count']>0
    assert len(result['result']['attempts'])==result['result']['starts']==2
    assert result['verification']['relative_error']<1e-8
    assert result['verification']['max_sigma_error']<.01
    assert all(row['weighted_objective_verified'] for row in result['verification']['saved_observations'].values())
    assert result['continuation']['native_active_set_verified']
    assert not result['global_optimum_certified'] and result['model_probability'] is None
    assert file_sha(original)==before
