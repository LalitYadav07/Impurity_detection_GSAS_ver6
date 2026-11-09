from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import optimize

from gpx_recipe.native_termination import observe_leastsq


def adapter(function, leastsq):
    return SimpleNamespace(so=SimpleNamespace(leastsq=leastsq, sentinel=object()),
                           G2stMth=SimpleNamespace(errRefine=function))


@pytest.mark.parametrize('code,success', [(1,True),(2,True),(3,True),(4,True),
    (0,False),(5,False),(6,False),(7,False),(8,False),(9,None),(True,None)])
def test_termination_is_observed_without_mutating_native_result_or_call(code,success):
    target=lambda x:x
    result=(np.array([1.]),np.eye(1),{'nfev':np.int64(12)},'native message',code)
    args=[];token=object()
    def solver(*a,**k):args.append((a,k));return result
    main=adapter(target,solver);original=main.so
    with observe_leastsq(main) as evidence:
        assert main.so.sentinel is original.sentinel
        returned=main.so.leastsq(target,token,full_output=True,ftol=1e-8)
    assert returned is result and main.so is original
    assert args==[((target,token),{'full_output':True,'ftol':1e-8})]
    assert evidence['final']['solver_success'] is success
    assert evidence['final']['code']==(None if code is True else code)
    assert evidence['final']['function_evaluations']==12
    assert not evidence['qualification_changed'] and not evidence['final_cycle_shifts_supplied']
    assert 'converged' not in evidence


def test_only_actual_native_target_and_own_thread_are_recorded():
    target=lambda x:x;other=lambda x:x
    main=adapter(target,lambda *a,**k:([],[],{'nfev':1},'ok',1));original=main.so
    with observe_leastsq(main) as evidence:
        main.so.leastsq(other,[])
        with ThreadPoolExecutor(1) as pool:pool.submit(main.so.leastsq,target,[]).result()
        assert evidence['calls']==[]
        main.so.leastsq(target,[])
    assert len(evidence['calls'])==1 and main.so is original


def test_retries_keep_final_failure_and_exception_restores_module():
    target=lambda x:x
    codes=iter([1,5])
    main=adapter(target,lambda *a,**k:([],[],{},'native',next(codes)));original=main.so
    with pytest.raises(RuntimeError,match='after native'):
        with observe_leastsq(main) as evidence:
            main.so.leastsq(target,[]);main.so.leastsq(target,[])
            raise RuntimeError('after native')
    assert main.so is original and [r['code'] for r in evidence['calls']]==[1,5]
    assert evidence['final']['solver_success'] is False
    with observe_leastsq(main) as again:assert again['state']=='unavailable'


@pytest.mark.parametrize('main', [SimpleNamespace(),SimpleNamespace(so=optimize)])
def test_missing_adapter_remains_unavailable(main):
    with observe_leastsq(main) as evidence:pass
    assert evidence['state']=='unavailable' and evidence['final'] is None


def test_nested_observer_does_not_interfere_or_claim_success():
    main=adapter(lambda x:x,lambda *a,**k:([],[],{},'native',1));original=main.so
    with observe_leastsq(main) as first:
        proxy=main.so
        with observe_leastsq(main) as second:
            assert second['reason']=='native_observer_already_active' and main.so is proxy
        main.so.leastsq(main.G2stMth.errRefine,[])
    assert main.so is original and first['final']['solver_success'] is True


def test_native_scipy_solution_is_identical_with_observer():
    target=lambda p:np.array([p[0]-3.,2.*(p[0]-3.)])
    expected=optimize.leastsq(target,[0.],full_output=True,ftol=1e-8)
    original=optimize.leastsq;main=SimpleNamespace(so=optimize,G2stMth=SimpleNamespace(errRefine=target))
    with observe_leastsq(main) as evidence:
        result=main.so.leastsq(target,[0.],full_output=True,ftol=1e-8)
        assert optimize.leastsq is original
    np.testing.assert_array_equal(result[0],expected[0])
    np.testing.assert_array_equal(result[1],expected[1])
    assert result[3:]==expected[3:]
    assert evidence['final']['code']==result[4]


def test_native_scipy_evaluation_limit_is_not_reported_as_success():
    target=lambda p:np.array([10.*(p[1]-p[0]**2),1.-p[0]])
    main=SimpleNamespace(so=optimize,G2stMth=SimpleNamespace(errRefine=target))
    with observe_leastsq(main) as evidence:
        result=main.so.leastsq(target,[-1.2,1.],full_output=True,maxfev=1)
    assert result[4]==5 and evidence['final']['termination']=='function_evaluation_limit'
    assert evidence['final']['solver_success'] is False


def test_solver_exception_remains_exception_and_restores_adapter():
    def solver(*a,**k):raise ValueError('native failure')
    main=adapter(lambda x:x,solver);original=main.so
    with pytest.raises(ValueError,match='native failure'):
        with observe_leastsq(main) as evidence:main.so.leastsq(main.G2stMth.errRefine,[])
    assert main.so is original and evidence['calls']==[] and evidence['state']=='unavailable'


def test_missing_full_output_does_not_invent_status():
    result=(np.array([0.]),2)
    main=adapter(lambda x:x,lambda *a,**k:result)
    with observe_leastsq(main) as evidence:
        assert main.so.leastsq(main.G2stMth.errRefine,[]) is result
    assert evidence['state']=='unavailable' and evidence['final']['reason']=='native_full_output_not_returned'
