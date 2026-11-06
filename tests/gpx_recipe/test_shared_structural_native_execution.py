"""Preparation/recovery handoffs use the same owned native response engine."""
from pathlib import Path

import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from gpx_recipe.structural_live.optimization.offline_native_start_bank import build,OfflineNativeStartBank
from gpx_recipe.structural_live.optimization.native_derivative_source import NativeDerivativeSource
from gpx_recipe.structural_live.service import prepare_model,make_source,fit_snapshot,SearchSettings
from gpx_recipe.structural_live.native import sha
from gpx_recipe.structural_live.reduced import from_arrays


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_complete_joint_start_bank_matches_full_native_without_a_second_profile_build(tmp_path,kind):
    _,source=native_source(tmp_path/'inputs',kind)
    reference_path=tmp_path/'reference.npz';captured_path=tmp_path/'captured.npz'
    reference=build(source,reference_path,policy='joint_cell_v1')
    accelerated=build(source,captured_path,policy='joint_cell_v1',reuse_fixed_cell_profiles=True)
    old=OfflineNativeStartBank(source,reference_path);new=OfflineNativeStartBank(source,captured_path)
    assert reference['proposed_points']==accelerated['proposed_points']==185
    np.testing.assert_array_equal(new.points,old.points)
    np.testing.assert_array_equal(new.proposal_cell_steps,old.proposal_cell_steps)
    np.testing.assert_allclose(new.curves,old.curves,rtol=1e-10,atol=1e-8)
    execution=accelerated['construction_execution'];response=execution['diagnostics']
    assert execution['cached_calls']>=64 and execution['direct_calls']>0
    assert response['captured_native_calls']==1
    assert response['maximum_capture_relative_error']<1e-9
    assert response['maximum_check_error']<1e-9
    assert response['numeric_cache_bytes']<=response['maximum_numeric_cache_bytes']


@pytest.mark.parametrize('kind',['PNC','PNT'])
@pytest.mark.parametrize('excluded',[False,True])
def test_recovery_mode_derivatives_keep_native_curves_and_all_coordinates(tmp_path,kind,excluded):
    _,source=native_source(tmp_path/'inputs',kind)
    if excluded:
        name=source.fast.slow.hist.name;prefix=f":{source.fast.slow.hist.id}:"
        source.fast.native.controls[prefix+'Exclude']=[[30.,35.]]
    proxy=NativeDerivativeSource(source)
    for offset in [0.,.07]:
        point=np.linspace(-.09,.11,len(source.radii))*source.radii
        point[-1]+=offset*source.radii[-1]
        actual,jac=proxy.native_mode_jacobian(point)
        expected,reference=source.native_mode_jacobian(point)
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_allclose(jac,reference,rtol=2e-6,atol=2e-7)
        np.testing.assert_array_equal(proxy.native(point),source.native(point))
    proof=proxy.diagnostics()
    assert proof['scores_use_original_native'] and proof['all_mode_derivatives_retained']
    if excluded:
        assert proof['native_derivative_fallbacks']==2
    else:
        assert proof['response']['derivative_calls']==2
        assert proof['maximum_relative_curve_error']<1e-9


def test_shared_service_recovery_publishes_its_actual_best_objective_and_verified_gpx(tmp_path):
    source_gpx,_=native_source(tmp_path/'inputs','PNC')
    child=Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    settings=SearchSettings(seconds=3.,starts=2,recovery_seconds=6.)
    prepared=tmp_path/'prepared'
    prepare_model(source_gpx,child,[[-1.,0,0],[0,-1.,0],[0,0,1.]],
        [0,0,0],prepared,settings=settings)
    source=make_source(prepared,tmp_path/'truth',settings)
    point=np.linspace(-.07,.05,len(source.radii))*source.radii
    truth=source.native(point);arrays=source.fast.slow.hist.data['data'][1]
    observation=from_arrays([arrays[0],truth,np.full(len(truth),4.)],
        source.fast.slow.hist.data['Limits'],acquisition='public recovery control',
        protocol_sha256=sha(prepared/'prepared.json'))
    result=fit_snapshot(prepared,tmp_path/'fit',settings=settings,observation=observation)
    fitted=result['result'];recovery=fitted['recovery']
    assert fitted['rwp_percent']==recovery['rwp_percent']
    assert fitted['exact_native_deviance']==recovery['best']['deviance']==fitted['best']['deviance']
    execution=recovery['native_derivative_execution']
    assert execution['derivative_requests']>0 and execution['response']['derivative_calls']>0
    assert execution['scores_use_original_native']
    assert result['verification']['relative_error']<1e-8
    assert result['verification']['max_sigma_error']<.01
    assert set(result['verification']['saved_observations'])=={'frozen','continuation'}
    assert not result['optimization_qualified'] and not result['model_exclusion_authorized']
