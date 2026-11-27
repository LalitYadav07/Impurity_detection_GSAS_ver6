"""Native regression for source GPXs carrying polynomial-background holds."""
import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from gpx_recipe.structural_live.native import prepare,sha
from gpx_recipe.structural_live.service import make_source,SearchSettings
from gpx_recipe.symmetry.native_magnetic import _analytic_background_preflight


def held_source(tmp_path,kind,held):
    source,_=native_source(tmp_path/'input',kind)
    output=tmp_path/'held.gpx'
    p=G.G2Project(gpxfile=str(source),newgpx=str(output));h=p.histograms()[0]
    if held:
        p.index_ids()
        for index in ((1,) if held=='middle' else (1,2)):p.add_HoldConstr([f':{h.id}:Back;{index}'])
    p.save()
    return output,p,h


@pytest.mark.parametrize('kind',['PNC','PNT'])
@pytest.mark.parametrize('held',[False,True,'middle'])
def test_prepared_derivatives_keep_source_holds_and_match_finite_differences(tmp_path,kind,held):
    source,p,h=held_source(tmp_path,kind,held)
    original=sha(source)
    holds=[str(c[0][1]) for c in p.data['Constraints']['data']['Hist']]
    coefficients=list(h.data['Background'][0][3:])
    from pathlib import Path
    child=Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    prepare(source,child,[[-1.,0,0],[0,-1.,0],[0,0,1.]],[0,0,0],tmp_path/'prepared')
    s=make_source(tmp_path/'prepared',tmp_path/'context',SearchSettings());fast=s.fast
    point=np.zeros(fast.dimension);point[:fast.nm]=np.linspace(-.002,.003,fast.nm)
    original_vary=list(fast.native.vary);original_values=fast.native.x0.copy()
    native_function=fast.native.math.dervRefine
    baseline=fast.evaluate(point)
    mean,jacobian=fast.evaluate(point,jacobian=True)
    np.testing.assert_array_equal(mean,baseline)
    for index in list(range(fast.nm))+[fast.dimension-2,fast.dimension-1]:
        step=1e-6;plus=point.copy();minus=point.copy();plus[index]+=step;minus[index]-=step
        finite=(fast.evaluate(plus)-fast.evaluate(minus))/(2*step)
        error=np.linalg.norm(jacobian[index]-finite)/max(np.linalg.norm(finite),1e-30)
        assert error<1e-6,(kind,held,index,error)
    assert fast.native.vary==original_vary
    np.testing.assert_array_equal(fast.native.x0,original_values)
    assert fast.native.math.dervRefine is native_function
    assert [str(c[0][1]) for c in fast.slow.project.data['Constraints']['data']['Hist']]==holds
    assert fast.slow.hist.data['Background'][0][3:]==coefficients
    assert sha(source)==original
    if not held:
        fast.native.complete_background_derivatives=False
        direct_mean,direct_jacobian=fast.evaluate(point,jacobian=True)
        np.testing.assert_array_equal(direct_mean,mean)
        np.testing.assert_array_equal(direct_jacobian,jacobian)


@pytest.mark.parametrize('held',[False,True])
def test_magnetic_analytic_preflight_uses_actual_native_background_freedom(tmp_path,held):
    source,p,h=held_source(tmp_path,'PNC',held)
    h.data['Background'][0][1]=True
    p.data['Controls']['data']['deriv type']='analytic Hessian';p.save();original=sha(source)
    if held:
        with pytest.raises(ValueError,match='partially constrained polynomial background'):
            _analytic_background_preflight(p,[h.name])
    else:
        result=_analytic_background_preflight(p,[h.name])
        assert result['checked'] and result['source_holds_preserved']
        assert result['background_variables']==[f':{h.id}:Back;{i}' for i in range(3)]
    assert sha(source)==original


def test_numeric_preflight_keeps_partial_source_holds(tmp_path):
    source,p,h=held_source(tmp_path,'PNC',True)
    h.data['Background'][0][1]=True;p.data['Controls']['data']['deriv type']='numeric';p.save()
    result=_analytic_background_preflight(p,[h.name])
    assert not result['checked']


@pytest.mark.parametrize('higher_held',[False,True])
def test_structural_required_constant_background_hold_fails_closed(tmp_path,higher_held):
    source,p,h=held_source(tmp_path,'PNC',higher_held)
    p.add_HoldConstr([f':{h.id}:Back;0']);p.save();original=sha(source)
    from pathlib import Path
    child=Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    prepare(source,child,[[-1.,0,0],[0,-1.,0],[0,0,1.]],[0,0,0],tmp_path/'prepared')
    with pytest.raises(ValueError,match='hold the constant background'):
        make_source(tmp_path/'prepared',tmp_path/'context',SearchSettings())
    assert sha(source)==original


def test_magnetic_all_held_background_fails_before_fit(tmp_path):
    source,p,h=held_source(tmp_path,'PNC',True)
    p.add_HoldConstr([f':{h.id}:Back;0']);h.data['Background'][0][1]=True
    p.data['Controls']['data']['deriv type']='analytic Hessian';p.save()
    with pytest.raises(ValueError,match='no independent polynomial coefficients'):
        _analytic_background_preflight(p,[h.name])


def test_legacy_structural_replay_returns_typed_unsupported_before_refinement(tmp_path,monkeypatch):
    import json,shutil
    from pathlib import Path
    from gpx_recipe.symmetry.native_structural import run
    from gpx_recipe.symmetry.worker import Artifacts,TrialDisposition
    from gsas_magnetic_preflight import setting_fingerprint
    source,p,h=held_source(tmp_path,'PNC',True)
    p.data['Controls']['data']['deriv type']='analytic Hessian';p.save();original=sha(source)
    child=tmp_path/'child.cif';shutil.copyfile(Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif',child)
    names=json.loads((tmp_path/'input/prepared/prepared.json').read_text())['inventory']['native_mode_names']
    candidate=dict(id='public',modes=names,child={'path':child.name},
        transformation_matrix=[[-1.,0,0],[0,-1.,0],[0,0,1.]],origin_shift=[0,0,0])
    policy=dict(maximum_cell_multiplier=1,maximum_mode=.5,maximum_cell_relative_change=.02,
        maximum_angle_change_degrees=3.,refine_cell=False,refine_background=True,
        refine_histogram_scale=True,cycles=1)
    protocol=dict(policy=policy,parent_phase=p.phases()[0].name,histogram=h.name,
        parent_setting_sha256=setting_fingerprint(p.phases()[0].data['General']['SGData']))
    output=tmp_path/'legacy';output.mkdir()
    monkeypatch.setattr(G.G2Project,'do_refinements',lambda *a,**k:pytest.fail('Unsupported analytic fit must not run'))
    with pytest.raises(TrialDisposition) as caught:
        run(source,Artifacts(output,input_root=tmp_path),protocol,candidate,{'mode_values':{n:.001 for n in names}})
    assert caught.value.status=='unsupported'
    assert 'source holds are preserved' in str(caught.value)
    assert sha(source)==original


def test_magnetic_returns_typed_unsupported_before_refinement(tmp_path,monkeypatch):
    from gpx_recipe.symmetry.native_magnetic import _finish
    from gpx_recipe.symmetry.worker import Artifacts,TrialDisposition
    source,p,h=held_source(tmp_path,'PNC',True)
    p.data['Controls']['data']['deriv type']='analytic Hessian';h.data['Background'][0][1]=True;p.save()
    original=sha(source);name=p.phases()[0].name
    keys=['candidate','baseline_residuals','baseline_target_profiles','magnetic_elements',
        'maximum_moment_magnitude','fixed_moment_direction','initial_moment_direction',
        'candidate_fixed_moment_direction','candidate_seed_moment_direction','minimum_relative_chi2_change',
        'moment_audit','moment_site_coupling','omitted_magnetic_sites','atom_constraint_warning','transferred_hstrain']
    state=dict.fromkeys(keys);state.update(selected_histograms=[h.name],magnetic_name=name,
        nuclear_name=name,mode_constraint_count=0)
    output=tmp_path/'magnetic';output.mkdir()
    monkeypatch.setattr(G.G2Project,'do_refinements',lambda *a,**k:pytest.fail('Unsupported analytic fit must not run'))
    with pytest.raises(TrialDisposition) as caught:
        _finish(Artifacts(output,input_root=tmp_path),{'policy':{'magnetic_optimizer':'gsas_hessian'}},{},p,state,construction_seconds=0.)
    assert caught.value.status=='unsupported' and 'source holds are preserved' in str(caught.value)
    assert sha(source)==original
