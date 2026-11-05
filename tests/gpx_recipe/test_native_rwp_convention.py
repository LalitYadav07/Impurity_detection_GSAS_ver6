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


def test_browser_view_uses_the_same_native_bins_as_residual_metrics(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import gpx_recipe.symmetry.native_audit as module
    x=np.arange(5.);y=np.ones(5);w=np.ones(5)
    limits=[[0,4],[.2,3.5],[2,2]]
    histogram=SimpleNamespace(name='selected',InstrumentParameters={'Type':['PNT']},
        data={'data':[{},[x,y,w,y,y]],'Limits':limits})
    project=SimpleNamespace(histogram=lambda name:histogram,phases=lambda:[])
    monkeypatch.setattr(module,'gsas',lambda:SimpleNamespace(__package__='mocknative'))
    original=module.importlib.import_module
    def load(name):
        if name=='mocknative.GSASIIlattice':return SimpleNamespace(Pos2dsp=lambda inst,xx:xx)
        if name=='mocknative.GSASIImath':return SimpleNamespace()
        return original(name)
    monkeypatch.setattr(module.importlib,'import_module',load)
    view=module.write_view(project,'selected',tmp_path/'view.json')
    assert view['fitted']==[False,True,False,True,True]
    metric=profile_residual_metrics([x,y,w],y,limits)
    assert sum(view['fitted'])==metric['fitted_points']
