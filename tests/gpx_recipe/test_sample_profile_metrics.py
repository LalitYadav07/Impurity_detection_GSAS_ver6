"""Width diagnostics must respect native masks and remain finite at zero error."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from gpx_recipe.live.sample_profile_trials import residual_metrics


def histogram():
    x=np.arange(1.,7.);y=np.ones(6);w=np.ones(6);calculated=y.copy()
    return SimpleNamespace(data={'data':[None,[x,y,w,calculated,np.zeros(6),np.zeros(6)]],
        'Limits':[[1.,6.],[1.,6.],[3.,4.]]})


def test_masks_and_exclusions_cannot_inflate_profile_rwp():
    h=histogram();a=h.data['data'][1]
    a[1]=np.ma.array(a[1],mask=[False,True,False,False,False,False])
    a[1][1]=1e9;a[1].mask[1]=True;a[3][2:4]=1e8
    result=residual_metrics(h)
    assert result['usable_observations']==3 and result['rwp']==0
    assert result['residual_lag1'] is None
    json.dumps(result,allow_nan=False)


def test_bad_weights_and_masked_calculated_values_are_excluded():
    h=histogram();a=h.data['data'][1];a[2][0]=0.;a[2][1]=-1.
    a[3]=np.ma.array(a[3],mask=[False,False,False,False,True,False])
    a[3][-1]=0.
    result=residual_metrics(h)
    assert result['usable_observations']==1 and result['rwp']==100
    assert result['residual_lag1'] is None


def test_weighted_rwp_matches_independent_manual_calculation():
    h=histogram();a=h.data['data'][1]
    a[2]=np.arange(1.,7.);a[3]=np.zeros(6)
    result=residual_metrics(h)
    assert result['rwp']==100. and result['reduced_residual_square']==pytest.approx((1+2+5+6)/4)
    assert result['usable_observations']==4


def test_lag_correlation_cannot_join_points_across_excluded_ranges():
    h=histogram();a=h.data['data'][1]
    a[1][:]=[1.,2.,0.,0.,100.,4.];a[3][:]=0.
    result=residual_metrics(h)
    assert result['residual_adjacent_pairs']==2
    assert result['residual_lag1']==pytest.approx(1.)


@pytest.mark.parametrize('case',['masked','zero_intensity','bad_weights'])
def test_undefined_profile_metrics_fail_explicitly(case):
    h=histogram();a=h.data['data'][1]
    if case=='masked':a[1]=np.ma.masked_all(6)
    elif case=='zero_intensity':a[1][:]=0.
    else:a[2][:]=np.nan
    with pytest.raises(ValueError):residual_metrics(h)
