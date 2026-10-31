"""A fresh, unrefined GPX has no Rwp; exporting it must remain read-only."""
from types import SimpleNamespace

import numpy as np
import pytest

from gpx_recipe import artifacts


@pytest.mark.parametrize('reported,expected',[(None,None),(float('nan'),None),
    (float('inf'),None),('?',None),(True,None),(-1.,None),(np.float64(4.25),4.25),(0.,0.)])
def test_project_export_preserves_missing_rwp_without_inventing_a_fit(tmp_path,monkeypatch,reported,expected):
    path=tmp_path/'unrefined.gpx';path.write_bytes(b'unchanged unrefined project')
    calls=[]
    histogram=SimpleNamespace(name='pattern',id=0,
        InstrumentParameters={'Type':['PNT']},
        data={'data':[{},[np.array([1.,2.,3.]),np.array([10.,20.,30.]),np.ones(3),np.zeros(3),np.zeros(3)]],
              'Limits':[[],[1.,3.]]},
        Excluded=lambda:[],reflections=lambda:{})
    def rwp():calls.append(True);return reported
    histogram.get_wR=rwp
    project=SimpleNamespace(index_ids=lambda:None,histograms=lambda:[histogram])
    monkeypatch.setattr(artifacts,'snapshot',lambda _:dict(histograms=[dict(name='pattern',key='h0')],phases=[]))
    monkeypatch.setattr(artifacts,'gsas',lambda:SimpleNamespace(G2Project=lambda **_:project))
    monkeypatch.setattr(artifacts,'extract_gsasii_refinement_state',lambda *a,**k:dict(parameter_space=dict(parameters=[])))
    monkeypatch.setattr(artifacts,'lattice',lambda _:SimpleNamespace(Pos2dsp=lambda _inst,x:x))
    view=artifacts.write_project_view(path,tmp_path/'view.json')
    assert view['histograms'][0]['rwp']==expected
    assert view['histograms'][0]['observed']==[10.,20.,30.]
    assert view['histograms'][0]['calculated']==[0.,0.,0.]
    assert path.read_bytes()==b'unchanged unrefined project' and len(calls)==1
