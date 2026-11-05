import numpy as np
import pytest
from types import SimpleNamespace
from gpx_recipe.structural_live.optimization import automatic_fit_recovery as recovery
from gpx_recipe.structural_live.optimization import chemistry_sqp_native_fit as sqp

@pytest.mark.parametrize('value',[-1,float('inf'),float('nan')])
def test_bad_chemistry_budget(value):
    with pytest.raises(ValueError):recovery.RecoveryPolicy(chemistry_seconds=value)

def test_chemistry_bank_required_before_calculation():
    with pytest.raises(ValueError,match='bank'):
        recovery.improve(None,None,None,None,policy=recovery.RecoveryPolicy(chemistry_seconds=1))

@pytest.mark.parametrize('proposal,claimed,accepted',[(.5,.25,True),(-.5,2.25,False),(.5,0.,None)])
def test_chemistry_recovery_monotone_and_fresh_native_checked(monkeypatch,proposal,claimed,accepted):
    source=SimpleNamespace(radii=np.ones(1),mode_dimensions=0,check=lambda p:p,native=lambda p:np.array([1.+p[0]]))
    table=SimpleNamespace(renderer=SimpleNamespace(background=np.zeros(1)))
    monkeypatch.setattr(recovery,'background_components',lambda t:(np.zeros(1),np.ones(1)))
    monkeypatch.setattr(recovery,'background_reference',lambda t:1.)
    monkeypatch.setattr(recovery,'residual_and_factor',lambda o,r:(r-3.,None))
    monkeypatch.setattr(recovery,'observation_rwp_percent',lambda o,r:0.)
    monkeypatch.setattr(recovery,'fit_feasible_gn',lambda *a,**kw:(dict(best=dict(parameters=[0,0,0],deviance=1.)),np.array([2.])))
    monkeypatch.setattr(sqp,'fit',lambda *a,**kw:(dict(best=dict(parameters=[proposal,0,0],deviance=claimed)),None))
    if accepted is None:
        with pytest.raises(ValueError,match='native objective'):
            recovery.improve(source,table,None,[0,0,0],bank=object(),policy=recovery.RecoveryPolicy(chemistry_seconds=1))
    else:
        r,y=recovery.improve(source,table,None,[0,0,0],bank=object(),policy=recovery.RecoveryPolicy(chemistry_seconds=1))
        assert r['stages'][-1]['accepted']==accepted
        assert r['best']['deviance']==(.25 if accepted else 1.)
        assert not r['model_exclusion_authorized']
