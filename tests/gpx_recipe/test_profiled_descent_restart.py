"""Escape verified saddles without changing physics, guards or budgets."""
from types import SimpleNamespace

import numpy as np
import pytest

from gpx_recipe.symmetry import profiled_magnetic as module


def saddle_model():
    # Unpolarized two-site interference. The antiparallel subspace is a
    # saddle for these observations, and time reversal does not escape it.
    factors=np.array([[1.,-1.],[1.,1.]])
    q=np.einsum('ni,nj->nij',factors,factors)
    nuclear=np.array([10.,10.])
    observed=nuclear+np.array([1.,9.])
    maps=np.zeros((2,3,2));maps[0,0,0]=maps[1,0,1]=1.
    return module.ProfiledMagneticObjective(offset=-observed,nuclear=nuclear,quadratic=q,
        background=np.empty((2,0)),moment_maps=maps,maximum_moment=4.,
        normalization=float(observed@observed),fixed_scale=1.)


@pytest.mark.parametrize('sign', [-1.,1.])
def test_time_reversed_saddle_starts_find_lower_cost_physical_solutions(sign):
    model=saddle_model()
    at_saddle=model.to_physical@np.array([.5,-.5])*sign
    initial_cost=model._evaluate(at_saddle)[0]
    assert initial_cost==pytest.approx(40.5)
    fit=model.fit(np.array([1.,-1.])*sign,max_iterations=80)
    assert fit.success and not fit.local_descent_found
    assert fit.descent_restarts>=1 and fit.descent_probe_evaluations>0
    assert fit.iterations<=80 and fit.weighted_cost<1e-10
    assert fit.weighted_cost<initial_cost
    assert fit.maximum_moment<=4.
    assert sorted(abs(fit.coefficients))==pytest.approx([1.,2.],abs=2e-5)
    checked=model.reassess(fit)
    assert checked.success and checked.descent_restarts==fit.descent_restarts
    assert checked.descent_probe_evaluations==fit.descent_probe_evaluations


def test_restarts_share_the_original_iteration_budget(monkeypatch):
    model=saddle_model();limits=[]
    def stuck(fun,x,**kwargs):
        limit=kwargs['options']['maxiter'];limits.append(limit)
        return SimpleNamespace(x=np.asarray(x).copy(),success=True,status=0,
            message='Injected stationary termination',nit=min(2,limit),nfev=1)
    monkeypatch.setattr(module,'minimize',stuck)
    fit=model.fit(np.array([.5,-.5]),max_iterations=5)
    assert limits==[5,3,1]
    assert fit.iterations==5 and fit.evaluations==3
    assert fit.descent_restarts==2
    assert not fit.success and fit.local_descent_found


def test_no_downhill_proposal_remains_unresolved(monkeypatch):
    model=saddle_model()
    monkeypatch.setattr(model,'_descent_restart',lambda point:(None,0))
    fit=model.fit(np.array([1.,-1.]))
    assert not fit.success and fit.local_descent_found
    assert fit.descent_restarts==0


def test_restart_checkpoint_preserves_cancellation():
    model=saddle_model()
    def cancel(point):
        raise InterruptedError('cancelled by owner')
    with pytest.raises(InterruptedError,match='cancelled by owner'):
        model.fit(np.array([1.,-1.]),callback=cancel)
