import numpy as np
import pytest
from gpx_recipe.live.native_peak_derivative_batch import NativePeakDerivativeBatch,BatchUnavailable


@pytest.mark.parametrize('change',['shape','integer','nan','noncontiguous','too_many','height','budget'])
def test_invalid_coefficients_never_enter_native_code(change):
    engine=object.__new__(NativePeakDerivativeBatch)
    engine._check_derivative=lambda:None
    def forbidden(*args):raise AssertionError('Invalid coefficients reached the native entry point')
    engine.derivative_function=forbidden
    axis=np.arange(8.);records={(1,0,0):(np.array([2.,.5,.25,1.,.1]),1.,0,8)}
    dp=np.ones((1,2,5));dh=np.ones((1,2));budget=64*1024*1024
    if change=='shape':dp=dp[:,:,0]
    elif change=='integer':dp=dp.astype(np.int64)
    elif change=='nan':dp[0,0,0]=np.nan
    elif change=='noncontiguous':dp=np.ones((1,2,10))[:,:,::2]
    elif change=='too_many':dp=np.ones((1,25,5));dh=np.ones((1,25))
    elif change=='height':dh=dh[0]
    elif change=='budget':budget=64
    with pytest.raises(BatchUnavailable):engine.render_columns(records,axis,dp,dh,maximum_bytes=budget)
