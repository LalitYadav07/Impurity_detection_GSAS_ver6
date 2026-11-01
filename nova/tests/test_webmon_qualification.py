import pytest
from radar_pd_nova.webmon.qualification import wilson,summarize

def test_zero_observed_errors_is_not_zero_risk():
    assert wilson(0,20)[1]>.1
    assert wilson(0,400)[1]<.01

def test_unresolved_runs_and_repeated_looks_are_not_hidden():
    rows=[dict(experiment_id=str(i),resolved=i<2,all_target_intervals_cover_truth=True,truth_excluded_at_any_step=i==0,premature_stop_at_any_step=False) for i in range(4)]
    result=summarize(rows)
    assert result['simultaneous_coverage_including_unresolved']==.5
    assert result['unresolved']==2 and not result['policy_evaluated']
    with pytest.raises(ValueError):summarize(rows+rows[:1])


def test_marginal_normalization_matches_dense_gaussian_including_determinant():
    import numpy as np
    from radar_pd_nova.webmon.nuisance import MarginalNormalizationObservation
    model=MarginalNormalizationObservation(np.array([1.,2.,3.]),np.array([[[2.,1.,3.]]]),np.array([5.,4.,8.]),np.array([.3,.5,.7]),np.ones((1,1,1)),normalization_sd=.04)
    for p in (.2,.8,1.2):
        f=model.predict([p])[0];r=model.y-f;C=np.diag(model.sigma**2)+.04**2*np.outer(f,f)
        truth=-.5*r@np.linalg.solve(C,r)-.5*(np.linalg.slogdet(C)[1]-np.log(model.sigma**2).sum())
        np.testing.assert_allclose(model.loglike([p])[0],truth,rtol=1e-10,atol=1e-10)
