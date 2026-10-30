import numpy as np
import pytest
from gpx_recipe.symmetry.native_audit import profile_residual_metrics


def test_excluded_can_signal_never_enters_residual_and_native_normalization_is_retained():
    x=np.array([0.,1.,2.]);y=np.array([10.,1000.,20.]);w=np.ones(3)
    result=profile_residual_metrics([x,y,w],np.zeros(3),[[0,2],[0,2],[.9,1.1]])
    assert result['rwp_native']==pytest.approx(100*np.sqrt(500/1000500))
    assert result['rwp_fit_points_only']==100
    assert result['fitted_points']==2 and result['normalization_points']==3
    second=profile_residual_metrics([x,y,w],np.array([0,1e9,0]),[[0,2],[0,2],[.9,1.1]])
    assert second==result  # An arbitrarily bad excluded can peak changes neither metric.


def test_perfect_profile_and_native_upper_index_convention():
    x=np.array([0.,1.,2.,3.]);y=np.ones(4);w=np.ones(4)
    result=profile_residual_metrics([x,y,w],y,[[0,3],[0,1.5]])
    assert result['rwp_native']==0 and result['fitted_points']==3
    with pytest.raises(ValueError):profile_residual_metrics([x,y,w*0],y,[[0,3],[0,3]])
