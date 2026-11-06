"""Full structural point-fit/export controls using public artificial inputs."""
import json

import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from gpx_recipe.structural_live.service import SearchSettings,prepare_model,make_source,fit_snapshot
from gpx_recipe.structural_live.native import sha
from gpx_recipe.structural_live.reduced import from_arrays


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_nonlinear_service_keeps_all_modes_current_observations_and_native_gpx(tmp_path,kind):
    source_gpx,source=native_source(tmp_path/'inputs',kind)
    # Use the original reviewed public mode file, not the import scratch.
    from pathlib import Path
    child=Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    settings=SearchSettings(seconds=8.,starts=2)
    prepared=tmp_path/'prepared'
    prepare_model(source_gpx,child,[[-1.,0,0],[0,-1.,0],[0,0,1.]],
        [0,0,0],prepared,settings=settings)
    source=make_source(prepared,tmp_path/'truth-context',settings)
    point=np.linspace(-.04,.06,len(source.radii))*source.radii
    truth=source.native(point)
    arrays=source.fast.slow.hist.data['data'][1]
    x=np.asarray(np.ma.getdata(arrays[0]),float)
    sigma=max(float(np.max(abs(truth)))*1e-4,1e-3)
    observed=truth+np.random.default_rng(92831).normal(0.,sigma,len(truth))
    observation=from_arrays([x,observed,np.full(len(x),1/sigma**2)],
        source.fast.slow.hist.data['Limits'],acquisition='independent public artificial observation',
        protocol_sha256=sha(prepared/'prepared.json'))
    original_source_sha=sha(source_gpx)
    result=fit_snapshot(prepared,tmp_path/'fit',settings=settings,observation=observation)
    assert result['state']=='verified_point_fit'
    assert result['all_physical_modes_retained'] and result['physical_mode_count']==source.mode_dimensions
    assert result['cell_parameter_count']==source.cell_dimensions
    assert result['starting_curves_reused']
    assert result['observation_sha256']==observation.observation_sha256
    assert result['verification']['relative_error']<1e-8
    assert result['verification']['max_sigma_error']<.01
    assert result['verification']['saved_observations'].keys()=={'frozen','continuation'}
    assert result['result']['exact_profile_memo']['hits']>0
    assert result['result']['cell_derivative_proposals']['builds']>0
    response=result['result']['native_response_cache']
    assert response['captured_native_calls']>0 and response['derivative_calls']>0
    assert response['maximum_capture_relative_error']<1e-9
    assert response['numeric_cache_bytes']<=response['maximum_numeric_cache_bytes']
    assert not result['optimization_qualified'] and not result['model_exclusion_authorized']
    assert sha(source_gpx)==original_source_sha
    for role in ('frozen','continuation'):
        artifact=result[role]
        assert sha(artifact['path'])==artifact['sha256']
        saved=G.G2Project(gpxfile=artifact['path'])
        data=saved.histograms()[0].data['data'][1]
        np.testing.assert_array_equal(data[1],observation.y)
    (tmp_path/'acceptance.json').write_text(json.dumps(dict(kind=kind,
        rwp_percent=result['metrics']['rwp_native'],timings=result['timings'],
        optimizer=result['result'],scope='Artificial native point-fit/export control; no material identification'),indent=2)+'\n')
