from types import SimpleNamespace
import numpy as np
import pytest
from test_structural_live_alternating_native_fit import setup
from gpx_recipe.structural_live.optimization.fit_fast_native import residual_and_factor
from gpx_recipe.structural_live.optimization import profiled_native_fit as module

def test_cell_only_model_does_not_require_a_mode_jacobian():
    s,t,o,z,c,b,_=setup();s.mode_dimensions=0;s.cell_dimensions=3
    def forbidden(p):raise AssertionError('A zero-mode model must not request a mode derivative')
    c.mode_jacobian=forbidden
    result,rate=module.fit(s,t,o,b,seconds=2,starts=1,evaluator=c)
    residual,_=residual_and_factor(o,rate)
    assert result['best']['deviance']==pytest.approx(residual@residual)
    assert result['mode_dimensions']==0 and result['cell_dimensions']==3
    assert result['best']['deviance']<150

def test_fixed_structure_can_fit_only_nuisance_parameters():
    old,t,o,z,c,b,_=setup();curve=old.native(z[:-2]*old.radii)
    s=SimpleNamespace(radii=np.empty(0),mode_dimensions=0,cell_dimensions=0,
        native=lambda p:curve.copy(),check=lambda p:p,
        chemistry=SimpleNamespace(log_kernel=lambda p:np.zeros(len(p))))
    b=SimpleNamespace(points=np.empty((1,0)),curves=curve[None,:],proposal_cell_steps=np.empty(0))
    cache=SimpleNamespace(diagnostics=lambda:{})
    result,rate=module.fit(s,t,o,b,seconds=1,starts=1,evaluator=cache)
    residual,_=residual_and_factor(o,rate)
    assert len(result['best']['parameters'])==2
    assert result['best']['deviance']==pytest.approx(residual@residual)
    assert result['best']['deviance']<150
