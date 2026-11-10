from types import SimpleNamespace

import numpy as np
import pytest

from gpx_recipe.structural_live.optimization.native_derivative_source import NativeDerivativeSource,derivative_source


def fixture():
    curve=np.array([2.,3.,4.]);jac=np.array([[1.,-2.,3.],[4.,-5.,6.]])
    calls=[]
    def native(p):calls.append('raw');return curve.copy()
    def fallback(p):calls.append('fallback');return curve.copy(),jac.copy()
    source=SimpleNamespace(mode_dimensions=2,check=np.asarray,native=native,native_mode_jacobian=fallback)
    class Cache:
        def __init__(self):self.source=source;self.ready=False;self.supported=True;self.delta=0.
        def has_response(self,p):return self.ready
        def needs_capture(self,p):return self.supported and not self.ready
        def native_evaluate(self,p):calls.append('anchor');self.ready=self.supported;return source.native(p)
        def mode_jacobian(self,p):calls.append('cached_jacobian');return curve+self.delta,jac.copy()
        def diagnostics(self):return dict(ready=self.ready)
    cache=Cache()
    return source,cache,calls,curve,jac


def test_scores_stay_native_and_each_derivative_has_a_current_native_anchor():
    source,cache,calls,curve,jac=fixture();proxy=NativeDerivativeSource(source,cache)
    point=np.array([.1,.2,.3])
    np.testing.assert_array_equal(proxy.native(point),curve)
    assert calls==['raw'] and not cache.ready
    for _ in range(2):
        actual,derivative=proxy.native_mode_jacobian(point)
        np.testing.assert_array_equal(actual,curve);np.testing.assert_array_equal(derivative,jac)
    assert calls==['raw','anchor','raw','cached_jacobian','anchor','raw','cached_jacobian']
    assert proxy.diagnostics()['derivative_requests']==2
    assert proxy.diagnostics()['scores_use_original_native']


def test_unavailable_response_keeps_the_full_native_jacobian_without_repeated_capture():
    source,cache,calls,curve,jac=fixture();cache.supported=False
    proxy=NativeDerivativeSource(source,cache)
    for _ in range(2):
        y,j=proxy.native_mode_jacobian([0.,0.,0.])
        np.testing.assert_array_equal(y,curve);np.testing.assert_array_equal(j,jac)
    assert calls==['fallback','fallback']
    assert proxy.diagnostics()['native_derivative_fallbacks']==2


def test_a_cached_curve_disagreement_stops_derivative_use():
    source,cache,_,_,_=fixture();cache.delta=.01
    proxy=NativeDerivativeSource(source,cache)
    with pytest.raises(ValueError,match='full native anchor'):
        proxy.native_mode_jacobian([0.,0.,0.])


def test_derivative_ownership_and_explicit_native_reference_are_preserved():
    source,cache,_,_,_=fixture()
    with pytest.raises(ValueError,match='another source'):
        NativeDerivativeSource(SimpleNamespace(),cache)
    assert derivative_source(source,force_native=True,evaluator=cache) is source
    assert derivative_source(source) is source
    proxy=derivative_source(source,evaluator=cache)
    assert derivative_source(proxy) is proxy
    with pytest.raises(ValueError,match='replace'):
        derivative_source(proxy,evaluator=SimpleNamespace())
