import numpy as np
import pytest

from gpx_recipe.structural_live.fitting import fit, FitScope
from gpx_recipe.structural_live.observations import Acquisition, CumulativeCounts


class ScaleBackgroundOracle:
    """Analytic two-parameter oracle, independent of native diffraction."""
    meta={'inventory':{'physical_rank':0,'cell_rank':0}}

    def __init__(self):
        self.signal=np.linspace(.05,2.,80)

    def bind_observation(self, observation):
        self.observation=observation

    def evaluate(self, modes, cell, *, log_scale, background0_delta):
        return self.signal*np.exp(log_scale)+background0_delta

    def save_verified_prediction(self, predicted):
        return {'test_oracle':True}


def test_poisson_fit_keeps_rwp_distinct_from_its_likelihood_objective():
    candidate=ScaleBackgroundOracle();rng=np.random.default_rng(812)
    counts=rng.poisson(4*(candidate.signal*np.exp(.08)+.2))
    source=Acquisition('synthetic','1','bank','raw','1','profile','tof_us','rate','s')
    obs=CumulativeCounts.from_counts(source,'protocol',1,4.,np.arange(80.),counts,1.,np.ones(80,bool))
    result,predicted=fit(candidate,obs,FitScope(background0_lower=0.,starts=1))
    assert result['state']=='fitted'
    numerator=np.sum((obs.y-predicted)**2/obs.variance)
    denominator=np.sum(obs.y**2/obs.variance)
    assert result['chi2']==pytest.approx(numerator)
    assert result['rwp_percent']==pytest.approx(100*np.sqrt(numerator/denominator))
    assert result['objective']=='poisson_deviance'
    assert abs(result['objective_value']-result['chi2'])>1.
    assert result['jacobian_rank']==2
    # Poisson score equations supply an independent stationarity check.
    mu=4*predicted
    score=np.stack([4*candidate.signal*np.exp(result['log_scale']),np.full(80,4.)])@(counts/mu-1)
    np.testing.assert_allclose(score,0.,atol=2e-4)
    assert not result['calibrated_intervals']


def test_invalid_scope_is_rejected_before_native_work():
    with pytest.raises(ValueError):FitScope(background0_bound=.1,background0_lower=.2)


def test_small_step_termination_with_a_large_gradient_is_not_verified_convergence(monkeypatch):
    from scipy.optimize import OptimizeResult
    import gpx_recipe.structural_live.fitting as module
    candidate=ScaleBackgroundOracle()
    source=Acquisition('synthetic','1','bank','raw','1','profile','tof_us','rate','s')
    obs=CumulativeCounts.from_counts(source,'protocol',1,4.,np.arange(80.),
                                     np.full(80,5),1.,np.ones(80,bool))
    def stopped_early(fun,initial,**kwargs):
        x=np.array([.01,.1]);residual=fun(x)
        return OptimizeResult(x=x,cost=.5*(residual@residual),jac=np.stack([candidate.signal,np.ones(80)],axis=1),
                              success=True,optimality=100.,nfev=4,message='xtol reached')
    monkeypatch.setattr(module,'least_squares',stopped_early)
    result,_=fit(candidate,obs,FitScope(background0_lower=0.,starts=1))
    assert result['optimizer_reported_success']
    assert result['state']=='optimization_unresolved'
    assert not result['stationarity_verified']
    assert not result['local_curvature_regular']
