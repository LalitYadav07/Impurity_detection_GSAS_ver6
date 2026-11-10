"""Direct native background columns versus unchanged full powder evaluations.

Public artificial neutron cases only. Every background coefficient, full
moment basis, native holdout/derivative gate and fitted-state check is kept.
"""
import json
from pathlib import Path
import time

import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')

from test_preparation_theory_native import inputs
from gpx_recipe.live import native_background as background
from gpx_recipe.live.native_profile import NativeReducedProfile
from gpx_recipe.native_forward import forward_profile
from gpx_recipe.preparation.magnetic import ARRAYS
from gpx_recipe.preparation.native_magnetic import compile_response
from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.symmetry.native_magnetic import construct_theory
from gpx_recipe.symmetry.native_profiled import NativeMagneticStartingPoint
from gpx_recipe.symmetry.worker import Artifacts

KINDS=['chebyschev','cosine','chebyschev-1','Q^2 power series','Q^-2 power series',
       'lin interpolate','inv interpolate','log interpolate']


def measured(root,kind,bak_type='chebyschev-1',*,masked=False):
    root.mkdir();reference,protocol,candidate,seed=inputs(root,kind)
    out=root/'model';out.mkdir()
    result=construct_theory(reference['path'],Artifacts(out),protocol,candidate,seed)
    path=out/result['gpx'];project=G.G2Project(gpxfile=str(path));hist=project.histograms()[0]
    hist.data['Background'][0]=[bak_type,True,6,3.,.2,.1,.03,.01,.002]
    project.save();observed,_=forward_profile(path,hist.name)
    project=G.G2Project(gpxfile=str(path));hist=project.histograms()[0]
    hist.data['data'][1][1]=observed.copy()+np.random.default_rng(718).normal(0.,.001,len(observed))
    hist.data['data'][1][2]=np.linspace(4.,7.,len(observed))
    hist.data['data'][0]['Dummy']=False;hist.data['data'][0]['wtFactor']=2.5
    if masked:
        x=hist.data['data'][1][0]
        hist.data['Limits'].append([float(x[180]),float(x[195])])
        hist.data['data'][1][2]=np.ma.array(hist.data['data'][1][2],mask=np.arange(len(x))%43==0)
    project.save();return path


@pytest.mark.parametrize('kind',['PNC','PNT'])
@pytest.mark.parametrize('bak_type',KINDS)
def test_all_native_background_functions_match_full_powder_with_masks_weights_and_fixed_terms(tmp_path,kind,bak_type):
    path=measured(tmp_path/'inputs',kind,bak_type,masked=True);before=file_sha(path)
    native=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    hfx=f':{native.histogram_id}:';x=native.histogram['Data'][0]
    lower,upper=native.args[4][hfx+'Limits'];sl=slice(np.searchsorted(x,lower),np.searchsorted(x,upper)+1)
    # Fixed nuisance terms are added in this owned numerical control. Native
    # full-powder calls see the exact same terms; the source GPX stays intact.
    native.histogram['Background'][1]['fixback']=np.linspace(.02,.08,len(x[sl]))
    native.parms.update({hfx+'BF mult':1.3,hfx+'DebyeA;0':.1,hfx+'DebyeR;0':1.2,hfx+'DebyeU;0':.01})
    coefficients=native.initial[native.background_indices];moments=native.initial[native.moment_indices]
    zero=np.zeros(len(coefficients));base=native.residual(moments,1.,zero)
    columns,proof=background.compile_columns(math=native.math,histogram=native.histogram,
        parameters=native.parms,controls=native.args[4],axis=x[sl],
        names=[native.vary[i] for i in native.background_indices])
    weights=np.asarray(np.ma.getdata(native.histogram['Data'][2]))[sl][~native.residual_mask]
    columns=columns[~native.residual_mask]*np.sqrt(weights*native.args[4][hfx+'wtFactor'])[:,None]
    for bg in (coefficients,np.random.default_rng(719).normal(size=6)):
        actual=native.residual(moments,1.,bg)
        np.testing.assert_allclose(base+columns@bg,actual,rtol=1e-8,atol=1e-6)
    assert proof['native_background_calls']==7 and proof['full_powder_calls_avoided']==6
    assert file_sha(path)==before


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_shared_background_compiler_preserves_complete_model_and_preparation_native_checks(tmp_path,kind,monkeypatch):
    path=measured(tmp_path/'inputs',kind,masked=True);before=file_sha(path)
    started=time.monotonic()
    fast=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    model,proof=fast.build();fast_seconds=time.monotonic()-started
    assert proof['background_compiler']['method']=='unchanged native background function'
    assert len(proof['heldout_relative_errors'])==4 and proof['native_derivative_calls']==1
    native_calls=fast.calls;results=[]
    starts=[np.eye(model.dimension)[i]*m for m in (.4,1.8) for i in range(model.dimension)]
    for seed in starts:
        fit=model.fit(seed,max_iterations=250)
        residual=fast.residual(fit.coefficients,fit.scale,fit.background)
        assert model.heldout_error(fit.coefficients,fit.scale,fit.background,residual)<1e-8
        results.append(dict(cost=float(.5*(residual@residual)),residual=residual.copy()))
    def unavailable(**_):raise background.BackgroundCompilerUnavailable('Independent original-powder reference')
    with monkeypatch.context() as patch:
        patch.setattr(background,'compile_columns',unavailable)
        started=time.monotonic()
        original=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
        reference,receipt=original.build();original_seconds=time.monotonic()-started
        assert receipt['background_compiler']['method']=='native full powder'
        assert original.calls==native_calls+6
        for name in ('offset','nuclear','quadratic','background'):
            np.testing.assert_allclose(getattr(model,name),getattr(reference,name),rtol=1e-8,atol=1e-6)
        for seed,expected in zip(starts,results):
            fit=reference.fit(seed,max_iterations=250)
            residual=original.residual(fit.coefficients,fit.scale,fit.background)
            np.testing.assert_allclose(residual,expected['residual'],rtol=1e-6,atol=2e-5)
            assert .5*(residual@residual)==pytest.approx(expected['cost'],rel=1e-6,abs=1e-6)
    # The same helper is consumed by observation-independent preparation and
    # Live compilation. Native qualification remains complete in both paths.
    unmasked=measured(tmp_path/'unmasked',kind)
    response=compile_response(NativeReducedProfile(unmasked))
    assert response.proof['background_compiler']['method']=='unchanged native background function'
    with monkeypatch.context() as patch:
        patch.setattr(background,'compile_columns',unavailable)
        native_response=compile_response(NativeReducedProfile(unmasked))
    for name in ARRAYS:
        np.testing.assert_allclose(getattr(response,name),getattr(native_response,name),rtol=1e-8,atol=1e-6)
    assert file_sha(path)==before
    (tmp_path/'measurement.json').write_text(json.dumps(dict(kind=kind,starts=len(starts),
        fast_seconds=fast_seconds,original_seconds=original_seconds,native_powder_calls_avoided=6,
        scope='Public artificial paired same-process numerical control; timings are not independent cold-worker or NDIP latency')))


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_wrong_background_optimization_recovers_the_original_native_columns(tmp_path,kind,monkeypatch):
    path=measured(tmp_path/'inputs',kind)
    real=background.compile_columns
    def wrong(**kwargs):
        matrix,proof=real(**kwargs);return matrix*1.001,proof
    monkeypatch.setattr(background,'compile_columns',wrong)
    native=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    model,proof=native.build()
    assert proof['background_compiler']['method']=='native full powder'
    assert 'background-only compilation failed' in proof['compiler_note']
    assert len(proof['heldout_relative_errors'])==4
    response=compile_response(NativeReducedProfile(path))
    assert response.proof['background_compiler']['method']=='native full powder'
    assert 'failed native qualification' in response.proof['background_compiler']['fallback_reason']


@pytest.mark.parametrize('target',['driver','background'])
def test_active_native_function_replacement_uses_original_powder_path(tmp_path,target,monkeypatch):
    path=measured(tmp_path/'inputs','PNC')
    native=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    hfx=f':{native.histogram_id}:'
    owner,name=(native.math,'getPowderProfile') if target=='driver' else (native.math.G2pwd,'getBackground')
    original=getattr(owner,name)
    monkeypatch.setattr(owner,name,lambda *args,**kwargs:original(*args,**kwargs))
    with pytest.raises(background.BackgroundCompilerUnavailable,match='needs review'):
        background.compile_columns(math=native.math,histogram=native.histogram,parameters=native.parms,
            controls=native.args[4],axis=native.histogram['Data'][0],
            names=[native.vary[i] for i in native.background_indices])


def test_fixed_background_array_changes_rebuild_prepared_physics(tmp_path):
    from gpx_recipe.preparation.native_magnetic import physics_identity,adapt_response,_physics_record
    path=measured(tmp_path/'inputs','PNT');before=file_sha(path)
    native=NativeReducedProfile(path);initial_identity=physics_identity(native)
    assert 'fixed_background_profile' not in _physics_record(native)
    histogram=native.histo_phases[0][native.parts[0][0]]
    fixed=np.full(len(histogram['Data'][0]),.12)
    histogram['Background'][1]['fixback']=fixed
    changed_identity=physics_identity(native)
    assert changed_identity!=initial_identity
    fixed[10]+=.01
    assert physics_identity(native)!=changed_identity
    del histogram['Background'][1]['fixback']
    assert physics_identity(native)==initial_identity
    response=compile_response(native)
    histogram['Background'][1]['fixback']=np.full(len(histogram['Data'][0]),.12)
    adapted,report=adapt_response(response,native)
    assert report['action']=='current_context_rendered'
    assert adapted.physics_sha256!=response.physics_sha256
    np.testing.assert_allclose(adapted.fixed-response.fixed,.12,rtol=1e-10,atol=1e-11)
    assert file_sha(path)==before
