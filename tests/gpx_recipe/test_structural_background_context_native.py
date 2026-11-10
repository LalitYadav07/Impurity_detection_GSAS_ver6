"""Native structural background setup across all reviewed CW/TOF functions."""
import json

import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from gpx_recipe.live import native_background
from gpx_recipe.structural_live.native import sha
from gpx_recipe.structural_live.optimization.native_background_context import fit_table

KINDS=['chebyschev','cosine','chebyschev-1','Q^2 power series','Q^-2 power series',
       'lin interpolate','inv interpolate','log interpolate']


@pytest.mark.parametrize('kind',['PNC','PNT'])
@pytest.mark.parametrize('background',KINDS)
def test_complete_native_structural_response_retains_fixed_terms_and_changed_geometry(tmp_path,monkeypatch,kind,background):
    path,s=native_source(tmp_path/'inputs',kind);original=sha(path);n=s.fast.native
    hist=n.histo_phases[0][s.fast.slow.hist.name];hfx=f":{hist['hId']}:"
    n.controls[hfx+'bakType']=background
    hist['Background'][0][0]=background
    n.parms.update({hfx+'BF mult':1.3,hfx+'DebyeA;0':.1,hfx+'DebyeR;0':1.2,hfx+'DebyeU;0':.01})
    hist['Background'][1]['fixback']=np.linspace(.02,.08,len(s.fast.x[s.fast.slice]))
    observations=[np.ma.getdata(a).copy() for a in hist['Data'][:3]]
    current=fit_table(s,5.)
    proof=current.background_execution
    assert proof['method']=='unchanged native background function'
    assert proof['full_native_calls']==1 and proof['full_powder_calls_avoided']==3
    def unavailable(**_):raise native_background.BackgroundCompilerUnavailable('independent original full-powder reference')
    with monkeypatch.context() as patch:
        patch.setattr(native_background,'compile_columns',unavailable)
        reference=fit_table(s,5.)
    assert reference.background_execution['full_native_calls']==4
    for key in ['fixed_background','background_response']:
        np.testing.assert_allclose(getattr(current,key),getattr(reference,key),rtol=1e-10,atol=1e-10)
    np.testing.assert_array_equal(current.renderer.background,reference.renderer.background)
    # Atomic modes and all three independent cell coordinates stay free.
    # Background construction cannot depend on a particular structure curve.
    for offset in [.06,-.09]:
        point=np.linspace(-offset,offset,len(s.radii))*s.radii
        baseline=s.native(point)
        theta=s.theta(point);theta[-1]=.7
        actual=s.fast.evaluate(theta,chemistry_domain=(s.chemistry,point))
        np.testing.assert_allclose(actual,baseline+.7*current.background_response,rtol=1e-10,atol=1e-8)
    s.native(np.zeros(len(s.radii)))
    for before,after in zip(observations,hist['Data'][:3]):np.testing.assert_array_equal(before,np.ma.getdata(after))
    assert sha(path)==original
    (tmp_path/'background-check.json').write_text(json.dumps(proof,indent=2))


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_wrong_compilation_recovers_original_structural_background(tmp_path,monkeypatch,kind):
    _,s=native_source(tmp_path/'inputs',kind);reference=fit_table(s,5.)
    real=native_background.compile_columns
    def wrong(**kwargs):
        columns,proof=real(**kwargs);return columns*1.001,proof
    monkeypatch.setattr(native_background,'compile_columns',wrong)
    actual=fit_table(s,5.)
    assert actual.background_execution['method']=='native full powder'
    assert 'anchor/slope checks' in actual.background_execution['fallback_reason']
    np.testing.assert_allclose(actual.background_response,reference.background_response,rtol=1e-10,atol=1e-10)
