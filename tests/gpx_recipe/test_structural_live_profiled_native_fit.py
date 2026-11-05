import numpy as np
import pytest
from types import SimpleNamespace
from test_structural_live_alternating_native_fit import setup
from gpx_recipe.structural_live.optimization.fit_fast_native import residual_and_factor
from gpx_recipe.structural_live.optimization import profiled_native_fit as module

def test_conditional_native_fit_preserves_objective_and_all_modes():
    s,t,o,z,c,b,_=setup()
    result,rate=module.fit(s,t,o,b,seconds=2,starts=1,evaluator=c)
    residual,_=residual_and_factor(o,rate)
    assert result['best']['deviance']==pytest.approx(residual@residual)
    assert result['all_allowed_modes_retained'] and len(result['best']['parameters'])==5
    assert result['independent_cache_checks']>0
    assert not result['model_exclusion_authorized']
    assert np.all(np.diff([p['deviance'] for p in result['trace']])<0)
    assert result['best']['deviance']<150

def test_profiled_cache_error_is_not_hidden_by_scale_or_background():
    s,t,o,z,c,b,_=setup()
    c.evaluate=lambda p:s.native(p)+.001*np.sin(o.x*23)
    with pytest.raises(ValueError,match='disagrees'):
        module.fit(s,t,o,b,seconds=2,starts=1,evaluator=c)

def test_screening_timeout_keeps_exact_conditional_incumbent(monkeypatch):
    s,t,o,z,c,b,_=setup();clock=[0.];original=s.native
    monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    def slow(point):
        value=original(point);clock[0]=3.;return value
    s.native=slow
    result,rate=module.fit(s,t,o,b,seconds=1,starts=2,evaluator=c)
    assert result['starts']==0 and not result['attempts']
    residual,_=residual_and_factor(o,rate)
    assert result['best']['deviance']==pytest.approx(residual@residual)


def test_short_start_updates_lattice_before_expensive_displacement_derivatives(monkeypatch):
    s,t,o,z,c,b,_=setup();clock=[0.]
    monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    original=c.mode_jacobian
    def expensive_modes(point):
        value=original(point);clock[0]+=2.;return value
    c.mode_jacobian=expensive_modes
    result,rate=module.fit(s,t,o,b,seconds=1.,starts=1,evaluator=c)
    assert result['blocks'][0]['kind']=='cell'
    assert result['blocks'][0]['accepted_costs']
    assert result['attempts'][0]['reason']=='per_start_time_budget'
    assert result['best']['deviance']==pytest.approx(residual_and_factor(o,rate)[0]@residual_and_factor(o,rate)[0])


def test_forced_native_derivatives_do_not_build_a_single_step_response_cache():
    s,t,o,z,c,b,_=setup()
    def forbidden(point):raise AssertionError('Exact-native fitting must not build a fixed-cell response matrix')
    c.evaluate=forbidden
    result,rate=module.fit(s,t,o,b,seconds=2.,starts=1,evaluator=c,force_native_mode_derivatives=True)
    residual,_=residual_and_factor(o,rate)
    assert result['independent_cache_checks']==0
    assert result['best']['deviance']==pytest.approx(residual@residual)
    assert result['best']['deviance']<150
    assert {'cell','modes'}<={block['kind'] for block in result['blocks']}
