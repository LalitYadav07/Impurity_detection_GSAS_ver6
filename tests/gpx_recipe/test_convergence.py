import pytest

from gpx_recipe.convergence import convergence_evidence, assess_convergence
from gpx_recipe.studio_state import summarize_shift_over_sigma


def state():
    # A large total displacement is compatible with a small final-cycle step.
    covariance={'varyList':['p','q'],'sig':[2.,1.], 'Lastshft':[20.,-30.],
                'Rvals':{'Max shft/sig':30.,'lastShifts':{'p':.04,'q':-.03}}}
    return dict(fit_statistics={'converged':True,'DelChi2':1e-6},
                controls={'deriv type':'analytic Hessian','max cyc':6},
                parameter_space={'last_vary':['p','q']},
                covariance={'shift_over_sigma':summarize_shift_over_sigma(covariance)})


def test_final_cycle_screen_is_not_confused_with_total_run_displacement():
    evidence=convergence_evidence(state())
    assert evidence['last_cycle_complete']
    assert evidence['last_cycle_max_shift_over_sigma']==.03
    assert evidence['total_run_max_shift_over_sigma']==30.
    assert assess_convergence(evidence,.1)['status']=='passed'


def test_partial_shifts_and_legacy_total_only_cannot_pass_a_convergence_screen():
    source=state();source['covariance']['shift_over_sigma']['last_cycle']['ratios'].pop('q')
    evidence=convergence_evidence(source)
    assert evidence['missing_last_cycle_parameters']==['q']
    assert assess_convergence(evidence,.1)['status']=='unavailable'
    source['covariance']['shift_over_sigma']['last_cycle']={}
    assert assess_convergence(convergence_evidence(source),.1)['status']=='unavailable'
    assert assess_convergence(None,.1)['status']=='unavailable'
    assert assess_convergence(None,None)['status']=='not_reviewed'


@pytest.mark.parametrize('change,reason',[
    (lambda s:s['fit_statistics'].update(converged=False),'native_not_converged'),
    (lambda s:s['fit_statistics'].update(Aborted=True),'native_aborted'),
    (lambda s:s['fit_statistics'].update(DelChi2=-.002),'objective_increased'),
    (lambda s:s['covariance']['shift_over_sigma']['last_cycle']['ratios'].update(q=-3.9),'last_cycle_shift_exceeded'),
])
def test_native_success_or_small_residual_cannot_hide_numerical_problem(change,reason):
    source=state();change(source)
    screen=assess_convergence(convergence_evidence(source),.1)
    assert screen['status']=='needs_review' and reason in screen['reasons']


def test_unknown_native_method_does_not_mislabel_missing_delta_sentinel():
    source=state();source['controls']['deriv type']='numeric';source['fit_statistics']['DelChi2']=-1.
    assert convergence_evidence(source)['relative_objective_change'] is None
    source['fit_statistics']['converged']=None
    assert assess_convergence(convergence_evidence(source),.1)['status']=='unavailable'


def test_cycle_limit_is_only_reported_for_a_hessian_solver():
    source=state()
    assert convergence_evidence(source)['maximum_cycles']==6
    for method in ('numeric','analytic Jacobian'):
        source['controls']['deriv type']=method
        assert convergence_evidence(source)['maximum_cycles'] is None


def test_objective_screen_uses_recorded_native_tolerance_without_hiding_increase():
    source=state();source['controls']['min dM/M']=.001
    source['fit_statistics']['DelChi2']=-9.481126812924467e-8
    evidence=convergence_evidence(source)
    assert evidence['relative_objective_tolerance']==.001
    screen=assess_convergence(evidence,.1)
    assert screen['status']=='passed' and 'increase' in screen['message']
    source['fit_statistics']['DelChi2']=-.002
    assert 'objective_increased' in assess_convergence(convergence_evidence(source),.1)['reasons']
    source['fit_statistics']['DelChi2']=-1e-8
    source['controls']['min dM/M']=1e-9
    assert 'objective_increased' in assess_convergence(convergence_evidence(source),.1)['reasons']


@pytest.mark.parametrize('tolerance',[None,True,0,-1,float('nan'),float('inf'),'0.001'])
def test_missing_or_invalid_native_objective_tolerance_does_not_relax_legacy_screen(tolerance):
    source=state();source['controls']['min dM/M']=tolerance
    source['fit_statistics']['DelChi2']=-1e-8
    evidence=convergence_evidence(source)
    assert evidence['relative_objective_tolerance'] is None
    assert 'objective_increased' in assess_convergence(evidence,.1)['reasons']


@pytest.mark.parametrize('maximum',[True,-1,float('nan'),float('inf'),'0.1'])
def test_invalid_convergence_policy_rejected(maximum):
    with pytest.raises(ValueError):assess_convergence(convergence_evidence(state()),maximum)
