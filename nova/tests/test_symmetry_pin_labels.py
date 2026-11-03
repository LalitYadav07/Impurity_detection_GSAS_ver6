import copy
import json
from types import SimpleNamespace

import pytest

from radar_pd_nova.symmetry_workspace import display_status,pin_label
from radar_pd_nova.symmetry_plots import fit


def checkpoint(tmp_path, name, status, rwp):
    path=tmp_path/(name+'.json')
    path.write_text(json.dumps(dict(x=[10.,20.,30.],d=[3.,2.,1.],coordinate='two_theta',
        observed=[20.,100.,50.],calculated=[20.,98.,50.],background=[1.,1.,1.],weights=[1.,1.,1.],
        fitted=[True]*3,limits=[10.,30.],ticks=[],phases=[])))
    return dict(id=name,name=name,candidate='candidate-'+name,seed='start-1',source='scan',
                has_view=True,view_path=str(path),status=status,rwp=rwp)


@pytest.mark.parametrize('status,rwp',[('valid',10.835),('optimization_unresolved',10.84),('inadequate_fit',22.4)])
def test_pin_labels_keep_residual_disposition_and_exact_start(status,rwp):
    row=dict(name='C2/c',candidate='mag:123456abcdef',seed='x-signs-1',status=status,rwp=rwp)
    before=copy.deepcopy(row)
    label=pin_label(row,include_start=True)
    assert label==f'C2/c · abcdef · x-signs-1 · Rwp {rwp:.3f}% · {display_status(status)}'
    assert pin_label(row)==f'C2/c · Rwp {rwp:.3f}% · {display_status(status)}'
    assert row==before


@pytest.mark.parametrize('rwp',[None,float('nan'),float('inf'),True,-1.])
def test_missing_or_invalid_residual_is_not_presented_as_a_measurement(rwp):
    assert pin_label(dict(name='Model',status='optimization_unresolved',rwp=rwp))==(
        'Model · Rwp unavailable · optimization unresolved')


def test_pinned_legend_hover_preserve_unresolved_attempts_and_two_pin_limit(tmp_path):
    selected=checkpoint(tmp_path,'selected','valid',10.835)
    valid=checkpoint(tmp_path,'valid','valid',10.836)
    unresolved=checkpoint(tmp_path,'C2c','optimization_unresolved',10.84)
    inadequate=checkpoint(tmp_path,'inadequate','inadequate_fit',22.4)
    figure=fit(selected,[valid,unresolved,inadequate])
    pins=[trace for trace in figure['data'] if trace['name'].startswith('Pinned')]
    assert len(pins)==2
    for trace,row in zip(pins,[valid,unresolved]):
        assert trace['name']=='Pinned · '+pin_label(row)
        assert trace['text']==pin_label(row,include_start=True)
        assert '%{text}' in trace['hovertemplate']
        assert trace['connectgaps'] is False
    figure=fit(selected,[inadequate])
    assert any('Rwp 22.400% · inadequate fit' in trace['name'] for trace in figure['data'])
    assert selected['status']=='valid' and unresolved['status']=='optimization_unresolved'


def test_pin_selector_keeps_attempts_inspectable_and_clamps_only_pin_count(tmp_path,monkeypatch):
    import radar_pd_nova.symmetry_controller as controller
    rows=[checkpoint(tmp_path,name,status,rwp) for name,status,rwp in [
        ('valid','valid',10.835),('unresolved','optimization_unresolved',10.84),('inadequate','inadequate_fit',22.4)]]
    state=SimpleNamespace(sx_selected='',sx_pins=[row['id'] for row in rows])
    app=controller.SymmetryMixin();app.server=SimpleNamespace(state=state)
    app._symmetry_rows=rows;app._symmetry_groups=[];app._symmetry_explicit_selection=False
    app._symmetry_group_changed=lambda:None
    monkeypatch.setattr(controller.plots,'overview',lambda *args:{})
    app._symmetry_publish_source()
    assert [option['value'] for option in state.sx_pin_options]==[row['id'] for row in rows]
    assert [option['title'] for option in state.sx_pin_options]==[pin_label(row,include_start=True) for row in rows]
    assert state.sx_pins==['unresolved','inadequate']
