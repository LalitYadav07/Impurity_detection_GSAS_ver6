"""An ordinary recipe holds imported modes in real CW and TOF projects."""
from pathlib import Path
import os

import numpy as np
import pytest

if os.environ.get('RADAR_REQUIRE_STRUCTURAL_NATIVE') == '1':
    import GSASII.GSASIIscriptable
else:
    pytest.importorskip('GSASII.GSASIIscriptable')

from test_native_profile_capture import native_source
from gpx_recipe.compiler import approve
from gpx_recipe.snapshot import gsas, snapshot
from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.structural_live.native import NativeCandidate
from gpx_recipe.structural_live.optimization.refinable_gpx import export_refinable
from gpx_recipe.worker import work


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_background_recipe_preserves_imported_structure_and_constraint_equations(tmp_path,kind):
    native_source(tmp_path/'source',kind)
    prepared=tmp_path/'source/prepared/prepared.gpx'
    # The preparation fixture has not materialized a fitted curve. Use the
    # native evaluator to publish a checked point before activating its modes.
    native=NativeCandidate(prepared.parent,tmp_path/'point')
    prediction=native.evaluate(np.zeros(native.basis['physical_basis'].shape[1]))
    point=native.save_verified_prediction(prediction)
    exported=export_refinable(point['path'],tmp_path/'export',expected_sha256=point['sha256'])
    source=Path(exported['path']);original_hash=file_sha(source)
    context=snapshot(source)
    plan=dict(schema='radar-gpx-recipe/v1',model_signature=context['model_signature'],
        request='Background and histogram scale only. Hold all imported modes and other parameters.',
        assumptions=[],questions=[],unsupported=[],stages=[
            dict(id='background',title='Background and scale',cycles=4,constraints=[],
                 free=[dict(kind='background',histogram='h0'),dict(kind='histogram_scale',histogram='h0')])])
    preview=work('preview',source,tmp_path/'preview',plan=plan)
    assert preview['status']=='preview_passed',preview
    approval=approve(plan,context,preview['preview'],approver='Native regression')
    result=work('run',source,tmp_path/'run',plan=plan,approval=approval)
    assert result['status']=='completed',result
    assert all('nv-' not in name for name in result['stage_outcomes'][0]['actual_vary'])
    G=gsas();before=G.G2Project(gpxfile=str(source));after=G.G2Project(gpxfile=str(tmp_path/'run'/result['final_gpx']))
    a,b=before.phases()[0],after.phases()[0]
    cx,_,_,cia=a.data['General']['AtomPtrs']
    for x,y in zip(a.data['Atoms'],b.data['Atoms']):
        np.testing.assert_allclose(x[cx:cx+4],y[cx:cx+4],rtol=0,atol=1e-12)
        np.testing.assert_allclose(x[cia+1:cia+8],y[cia+1:cia+8],rtol=0,atol=1e-12)
    np.testing.assert_allclose(a.data['General']['Cell'][1:],b.data['General']['Cell'][1:],rtol=0,atol=1e-12)
    assert len(a.data['ISODISTORT']['G2ModeList'])>0
    np.testing.assert_allclose(a.data['ISODISTORT']['modeDispl'],b.data['ISODISTORT']['modeDispl'],rtol=0,atol=1e-12)
    h0,h1=before.histograms()[0],after.histograms()[0]
    for x,y in zip(h0.data['data'][1][:3],h1.data['data'][1][:3]):
        np.testing.assert_array_equal(np.ma.getdata(x),np.ma.getdata(y))
        np.testing.assert_array_equal(np.ma.getmaskarray(x),np.ma.getmaskarray(y))
    assert h0.data['Limits']==h1.data['Limits']
    assert h0.data['Instrument Parameters']==h1.data['Instrument Parameters']
    old=before.data['Constraints']['data']['Phase'];new=after.data['Constraints']['data']['Phase']
    assert len(old)==len(new)
    for x,y in zip(old,new):
        assert repr(x[:-2])==repr(y[:-2]) and x[-1]==y[-1]
        assert y[-2] is False
    assert file_sha(source)==original_hash
