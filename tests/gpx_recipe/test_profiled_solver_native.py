"""Real GSAS-II public-data controls for the opt-in execution boundary."""
import copy
import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profiled import artificial_project
from gpx_recipe.symmetry.profiled_solver import refine,ProfiledScopeError
from gpx_recipe.symmetry.native_audit import profile_residual_metrics
from GSASII import GSASIIstrMain as MAIN


def protocol(*,free=True,iterations=100):
    return dict(kind='magnetic',k=['0','0','0'],policy=dict(moment_model='symmetry_allowed',
        refine_cell=False,refine_histogram_scale=free,refine_background=free,
        maximum_moment=6.,timeout_seconds=120,optimizer_iterations=iterations))


@pytest.mark.parametrize('free,masked',[(True,True),(False,False)])
def test_serialized_native_oracle_and_arrays_preserve_scientific_inputs(tmp_path,free,masked):
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
    native=MAIN.DoNoFit(str(path),histogram.name)
    metrics=profile_residual_metrics(histogram.data['data'][1][:3],native,histogram.data['Limits'])
    assert metrics['rwp_native']==pytest.approx(result['final_profile']['rwp_native'],rel=1e-10)
    assert result['diagnostics']['physical_gradient_norm']<=1e-7
    assert result['serialized_native_identity_error']<=1e-8


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


def test_boundary_solution_does_not_acquire_unreviewed_qualification(tmp_path):
    project,path=artificial_project(tmp_path)
    p=protocol();p['policy']['maximum_moment']=.5
    result=refine(project,path,p)
    assert not result['qualification']['passed']
    assert not result['qualification']['checks']['interior_solution']
