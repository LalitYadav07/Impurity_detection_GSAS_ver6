"""Real GSAS-II public-data controls for the opt-in execution boundary."""
import copy
import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profiled import artificial_project
from gpx_recipe.symmetry.profiled_solver import refine,ProfiledScopeError
from gpx_recipe.symmetry.native_audit import profile_residual_metrics,write_view
from GSASII import GSASIIstrMain as MAIN


def protocol(*,free=True,iterations=100):
    return dict(kind='magnetic',k=['0','0','0'],policy=dict(moment_model='symmetry_allowed',
        refine_cell=False,refine_histogram_scale=free,refine_background=free,
        maximum_moment=6.,timeout_seconds=120,optimizer_iterations=iterations))


@pytest.mark.parametrize('free,masked',[(True,True),(False,False)])
def test_serialized_native_oracle_and_arrays_preserve_scientific_inputs(tmp_path,free,masked,monkeypatch):
    project,path=artificial_project(tmp_path,masked=masked,scale_free=free,background_free=free)
    histogram=project.histograms()[0]
    fixed=copy.deepcopy([*histogram.data['data'][1][:3],histogram.data['Limits'],histogram.data['Instrument Parameters']])
    result=refine(project,path,protocol(free=free))
    assert result['success'] and result['status']==0
    assert result['qualification']['passed'],result['qualification']
    assert not result['native_convergence_declared'] and not result['uncertainty_available']
    covariance=project.data['Covariance']['data']
    assert covariance['Rvals']['converged'] is None
    assert not covariance['covMatrix'] and all(x is None for x in covariance['sig'])
    for before,after in zip(fixed[:3],histogram.data['data'][1][:3]):
        assert np.array_equal(np.ma.getdata(before),np.ma.getdata(after))
        assert np.array_equal(np.ma.getmaskarray(before),np.ma.getmaskarray(after))
    assert fixed[3:]==[histogram.data['Limits'],histogram.data['Instrument Parameters']]
    captured={};original=MAIN.G2stIO.GetUsedHistogramsAndPhases
    def capture(*args,**kwargs):
        value=original(*args,**kwargs);captured['histograms']=value[0];return value
    monkeypatch.setattr(MAIN.G2stIO,'GetUsedHistogramsAndPhases',capture)
    original_residual=MAIN.G2stMth.errRefine
    def capture_forward(*args,**kwargs):
        captured['forward_args']=args
        return original_residual(*args,**kwargs)
    monkeypatch.setattr(MAIN.G2stMth,'errRefine',capture_forward)
    native=MAIN.DoNoFit(str(path),histogram.name)
    # DoNoFit computes Fc but leaves the observed-F² extraction column zero.
    # Use GSAS-II's own extraction on that independent fresh forward state,
    # matching the final derivative evaluation that supplied the saved table.
    args=captured['forward_args']
    MAIN.G2stMth.GetFobsSq(args[1][0],args[1][1],args[2],args[5])
    metrics=profile_residual_metrics(histogram.data['data'][1][:3],native,histogram.data['Limits'])
    assert metrics['rwp_native']==pytest.approx(result['final_profile']['rwp_native'],rel=1e-10)
    assert result['diagnostics']['physical_gradient_norm']<=1e-7
    assert result['serialized_native_identity_error']<=1e-8
    reloaded=G.G2Project(gpxfile=str(path));stored=reloaded.histogram(histogram.name).data['Reflection Lists']
    expected=captured['histograms'][histogram.name]['Reflection Lists']
    active={phase.name for phase in reloaded.phases() if phase.data['Histograms'][histogram.name]['Use']}
    assert active.issubset(stored)
    for phase in active:
        assert len(stored[phase]['RefList'])>0
        np.testing.assert_allclose(stored[phase]['RefList'],expected[phase]['RefList'],rtol=1e-10,atol=1e-10)
    view=write_view(reloaded,histogram.name,tmp_path/'view.json')
    assert {tick['phase'] for tick in view['ticks']}==active
    for tick in view['ticks']:
        assert tick['native']==pytest.approx(np.asarray(stored[tick['phase']]['RefList'])[:,5])
        assert tick['d']==pytest.approx(np.asarray(stored[tick['phase']]['RefList'])[:,4])


def test_budget_failure_stays_unqualified_and_does_not_borrow_parent_covariance(tmp_path):
    project,path=artificial_project(tmp_path)
    project.data['Covariance']['data']={'Rvals':{'converged':True},'sig':[999.],'covMatrix':[[1.]]}
    project.save()
    result=refine(project,path,protocol(iterations=1))
    assert not result['success'] and result['status']==9
    assert not result['qualification']['passed'] and not result['candidate_applied']
    checks=result['qualification']['checks']
    assert set(checks)=={'raw_optimizer_success','independent_stationarity','no_local_descent',
        'curvature_nonnegative','interior_solution','improves_nuclear_baseline','nuclear_null_identity',
        'serialized_basis_preserved','serialized_native_stationarity','serialized_native_identity'}
    assert all(type(value) is bool for value in checks.values())
    assert checks['serialized_basis_preserved'] is False
    assert project.data['Covariance']['data']['Rvals']['converged'] is None
    assert not project.data['Covariance']['data']['covMatrix']


def test_hidden_nuisance_freedom_is_not_accepted(tmp_path):
    project,path=artificial_project(tmp_path,scale_free=True,background_free=True)
    with pytest.raises(ProfiledScopeError,match='scale freedom'):
        refine(project,path,protocol(free=False))


def test_unlisted_nuclear_parameters_are_not_accepted(tmp_path):
    project,path=artificial_project(tmp_path)
    project.phases()[0].data['Atoms'][0][2]='X'
    project.save()
    with pytest.raises(ProfiledScopeError,match='Only independent moments'):
        refine(project,path,protocol())


@pytest.mark.parametrize('constrained,masked',[(False,True),(True,False)])
def test_group_screen_reuses_solution_but_rechecks_native_and_saved_gpx(tmp_path,monkeypatch,constrained,masked):
    from gpx_recipe.symmetry.native_profiled import NativeMagneticStartingPoint
    from gpx_recipe.symmetry.profiled_magnetic import ProfiledMagneticObjective
    project,path=artificial_project(tmp_path,sites=2 if constrained else 1,constrained=constrained,masked=masked)
    native=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    model,_=native.build()
    fit=model.fit(native.initial[native.moment_indices],max_iterations=100)
    assert fit.success
    names=list(native.moment_names)
    def forbidden_second_optimization(*args,**kwargs):
        raise AssertionError('A screened start must not optimize again during native verification')
    monkeypatch.setattr(ProfiledMagneticObjective,'fit',forbidden_second_optimization)
    result=refine(project,path,protocol(),screened=dict(magnetic_variables=names,fit=fit))
    assert result['reused_group_screen'] and result['candidate_applied']
    assert result['qualification']['passed'],result['qualification']
    assert result['serialized_native_identity_error']<1e-8
    assert not result['native_convergence_declared'] and not result['native_covariance_supplied']
    reloaded=G.G2Project(gpxfile=str(path))
    assert not reloaded.data['Covariance']['data']['covMatrix']
    actual=np.asarray(MAIN.DoNoFit(str(path),reloaded.histograms()[0].name))
    expected=np.ma.getdata(reloaded.histograms()[0].data['data'][1][3])
    np.testing.assert_allclose(actual,expected,rtol=1e-8,atol=1e-8)
    histogram=reloaded.histograms()[0]
    metrics=profile_residual_metrics(histogram.data['data'][1][:3],actual,histogram.data['Limits'])
    assert result['final_profile']['rwp_native']==pytest.approx(metrics['rwp_native'],rel=1e-10)


def test_group_screen_cannot_use_a_different_native_variable_basis(tmp_path):
    from gpx_recipe.symmetry.native_profiled import NativeMagneticStartingPoint
    project,path=artificial_project(tmp_path)
    native=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    model,_=native.build();fit=model.fit(native.initial[native.moment_indices])
    with pytest.raises(ProfiledScopeError,match='independent variables differ'):
        refine(project,path,protocol(),screened=dict(magnetic_variables=['wrong'],fit=fit))


def test_boundary_solution_does_not_acquire_unreviewed_qualification(tmp_path):
    project,path=artificial_project(tmp_path)
    p=protocol();p['policy']['maximum_moment']=.5
    result=refine(project,path,p)
    assert not result['qualification']['passed']
    assert not result['qualification']['checks']['interior_solution']


def test_commensurate_antitranslation_full_basis_matches_native(tmp_path):
    """Half-order reflections in a doubled cell exercise nonzero-k magnetism."""
    project,path=artificial_project(tmp_path,sites=1)
    mag=project.phases()[1]
    sg=mag.data['General']['SGData']
    sg.update(SGCen=np.array([[0.,0.,0.],[.5,0.,0.]]),SGFixed=True,
              SpnFlp=np.array([1.,-1.]),MagMom=np.array([1.,-1.]))
    project.save()
    hist=project.histograms()[0]
    truth=np.asarray(MAIN.DoNoFit(str(path),hist.name),float)
    hist.data['data'][1][1]=truth+np.random.default_rng(912).normal(0.,.01,len(truth))
    hist.data['data'][1][2][:]=10000.
    for atom in mag.data['Atoms']:
        atom[7:10]=[v*.6 for v in atom[7:10]]
    project.save()
    request=protocol(iterations=200);request['k']=['1/2','0','0']
    result=refine(project,path,request)
    assert result['objective_identity']['linear_identity_error']<1e-8
    assert max(result['objective_identity']['heldout_relative_errors'])<1e-8
    assert result['candidate_applied'],result['qualification']
    assert result['serialized_native_identity_error']<1e-8
    assert result['final_profile']['rwp_native']<.5
