from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from scipy.optimize import leastsq

from gpx_recipe.native_budget import residual_budget, EvaluationBudgetExceeded, iteration_caption


def adapter(target, solver):
    return SimpleNamespace(so=SimpleNamespace(leastsq=solver), G2stMth=SimpleNamespace(errRefine=target))


def test_successful_budget_delegates_identical_results_and_restores_native_module():
    target=lambda x:x
    answer=object()
    def solver(f,x,**kwargs):
        assert kwargs=={'full_output':True}
        assert f(x)==x
        return answer
    main=adapter(target,solver);original=main.so
    with residual_budget(main,1) as receipt:
        assert main.so.leastsq(target,3,full_output=True) is answer
    assert receipt['evaluations']==1 and not receipt['exhausted']
    assert main.so is original


def test_limit_interrupts_before_an_extra_native_evaluation():
    calls=[]
    target=lambda x:calls.append(x) or x
    def solver(f,x):
        for _ in range(10):f(x)
    main=adapter(target,solver);original=main.so
    with pytest.raises(EvaluationBudgetExceeded):
        with residual_budget(main,3) as receipt:main.so.leastsq(target,7)
    assert len(calls)==receipt['evaluations']==3 and receipt['exhausted']
    assert main.so is original


def test_unrelated_functions_and_other_threads_are_not_rewritten():
    target=lambda x:x
    seen=[]
    def solver(f,*args,**kwargs):seen.append(f);return f(*args)
    main=adapter(target,solver);other=lambda x:x+1
    with residual_budget(main,2) as receipt:
        assert main.so.leastsq(other,3)==4
        with ThreadPoolExecutor(1) as pool:assert pool.submit(main.so.leastsq,target,3).result()==3
    assert seen==[other,target] and receipt['evaluations']==0


@pytest.mark.parametrize('maximum',[0,-1,True,2.5,'10',100001])
def test_invalid_budget_is_rejected(maximum):
    with pytest.raises(ValueError):
        with residual_budget(SimpleNamespace(),maximum):pass


def test_missing_adapter_and_nested_budget_cannot_claim_enforcement():
    with pytest.raises(ValueError):
        with residual_budget(SimpleNamespace(),3):pass
    main=adapter(lambda x:x,lambda f,x:f(x))
    with residual_budget(main,3):
        with pytest.raises(ValueError,match='already active'):
            with residual_budget(main,3):pass


def test_real_minpack_solution_is_unchanged_when_within_budget():
    target=lambda x:np.array([x[0]-3.,2*(x[0]-3.)])
    original=leastsq(target,[0.],full_output=True)
    main=adapter(target,leastsq)
    with residual_budget(main,100) as receipt:
        actual=main.so.leastsq(target,[0.],full_output=True)
    np.testing.assert_array_equal(actual[0],original[0])
    np.testing.assert_array_equal(actual[1],original[1])
    assert actual[3:]==original[3:] and receipt['evaluations']>=actual[2]['nfev']


def test_recipe_caption_never_calls_numeric_budget_a_cycle_count():
    assert iteration_caption({'cycles':12,'max_function_evaluations':500},{'controls':{'deriv type':'numeric'}})=='At most 500 residual evaluations · numeric minimizer'
    assert 'no evaluation limit recorded' in iteration_caption({'cycles':12},{'controls':{'deriv type':'numeric'}})
    assert iteration_caption({'cycles':12},{'controls':{'deriv type':'analytic Hessian'}})=='At most 12 cycles'


def test_unrelated_native_error_is_not_reclassified_as_budget_exhaustion():
    def target(value):raise ArithmeticError('native scientific calculation failed')
    main=adapter(target,lambda fn,x:fn(x));original=main.so
    with pytest.raises(ArithmeticError):
        with residual_budget(main,5) as receipt:main.so.leastsq(target,1)
    assert receipt['evaluations']==1 and not receipt['exhausted'] and main.so is original
