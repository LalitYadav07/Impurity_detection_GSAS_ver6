"""Structural setup retains native backgrounds and recovers failed helpers."""
from types import SimpleNamespace

import numpy as np
import pytest

from gpx_recipe.live import native_background
from gpx_recipe.structural_live.optimization import native_background_context as module


def source(monkeypatch):
    monkeypatch.setattr(native_background,'_reviewed',lambda _:True)
    x=np.arange(9.,dtype=float);sl=slice(1,8);p={':0:Back;0':3.,':0:Back;1':.4}
    fixed=np.linspace(.1,.7,7);counts={'powder':0,'checks':0}
    hist={'hId':0,'Background':[[],{'fixback':fixed}],'Data':[x,np.ones(9),np.ones(9),np.zeros(9),np.zeros(9)]}
    def background(prefix,parameters,kind,data_type,axis,fixback):
        return parameters[prefix+'Back;0']*np.ones(len(axis))+parameters[prefix+'Back;1']*axis+fixback,{}
    def check():counts['checks']+=1
    native=SimpleNamespace(_check=check,parms=p,controls={':0:bakType':'chebyschev',':0:histType':'PNT'},
        math=SimpleNamespace(G2pwd=SimpleNamespace(getBackground=background)))
    fast=SimpleNamespace(native=native,x=x,slice=sl,slow=SimpleNamespace(background0=3.))
    def evaluate(theta,**kwargs):
        counts['powder']+=1;p[':0:Back;0']=3.+theta[-1]
        hist['Data'][4][:]=0.
        hist['Data'][4][sl]=background(':0:',p,'chebyschev','PNT',x[sl],fixed)[0]
        return hist['Data'][4]+12.
    fast.evaluate=evaluate
    def snapshot(point):
        full=evaluate(np.r_[point,0.]);return full,None,hist['Data'][4].copy(),None,hist
    s=SimpleNamespace(fast=fast,radii=np.ones(2),chemistry=object(),snapshot=snapshot,
        theta=lambda point:np.r_[point,0.],native=lambda point:evaluate(np.r_[point,0.]))
    return s,hist,counts


def test_native_only_setup_avoids_three_phase_evaluations_and_retains_fixed_terms(monkeypatch):
    s,hist,counts=source(monkeypatch);fixed=hist['Background'][1]['fixback'].copy()
    table=module.fit_table(s,5.)
    assert counts['powder']==1
    assert table.background_execution['full_powder_calls_avoided']==3
    assert table.background_execution['independent_background_checks']==3
    assert table.background_execution['native_background_calls']==5
    np.testing.assert_allclose(table.background_response,np.r_[0.,np.ones(7),0.],rtol=0,atol=1e-14)
    np.testing.assert_allclose(table.fixed_background[1:8],.4*np.arange(1.,8.)+fixed)
    np.testing.assert_array_equal(hist['Background'][1]['fixback'],fixed)
    assert s.fast.native.parms[':0:Back;0']==3.
    assert table.background_reference==5.


@pytest.mark.parametrize('failure',['unavailable','wrong_slope','wrong_shape','nonfinite'])
def test_unqualified_or_inconsistent_columns_use_original_full_native_setup(monkeypatch,failure):
    s,hist,counts=source(monkeypatch);real=native_background.compile_columns
    def broken(**kwargs):
        if failure=='unavailable':raise native_background.BackgroundCompilerUnavailable('unreviewed runtime')
        columns,proof=real(**kwargs)
        if failure=='wrong_slope':columns*=1.01
        elif failure=='wrong_shape':columns=columns[:-1]
        else:columns[1]=np.nan
        return columns,proof
    monkeypatch.setattr(native_background,'compile_columns',broken)
    table=module.fit_table(s,5.)
    assert counts['powder']==4
    assert table.background_execution['method']=='native full powder'
    assert table.background_execution['fallback_reason']
    np.testing.assert_allclose(table.background_response,np.r_[0.,np.ones(7),0.],rtol=0,atol=1e-14)
    assert s.fast.native.parms[':0:Back;0']==3.


def test_changed_native_baseline_cannot_pass_the_fast_anchor_check(monkeypatch):
    s,hist,counts=source(monkeypatch);real=s.snapshot
    def incompatible(point):
        full,refs,bg,phase,h=real(point);bg[3]+=.2
        return full,refs,bg,phase,h
    s.snapshot=incompatible
    with pytest.raises(ValueError,match='not a verified linear'):
        module.fit_table(s,5.)
    assert counts['powder']==4
    assert s.fast.native.parms[':0:Back;0']==3.


def test_native_failure_is_not_disguised_as_an_accelerator_limitation(monkeypatch):
    s,_,_=source(monkeypatch)
    def fail():raise RuntimeError('native owner changed')
    s.fast.native._check=fail
    with pytest.raises(RuntimeError,match='native owner changed'):module.fit_table(s,5.)
