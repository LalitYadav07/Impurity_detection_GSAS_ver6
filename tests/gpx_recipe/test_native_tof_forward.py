"""Full native start curves, including unsupported-scope and failure controls."""
import numpy as np
import pytest

pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from gpx_recipe.structural_live.optimization.native_forward import for_source,NativeTOFForward


@pytest.mark.parametrize('change',['parent','mixed','profile','absorption','lookup','cell_edge','zero_counts'])
def test_batched_native_curves_retain_original_values_and_observations(tmp_path,change):
    _,s=native_source(tmp_path/'inputs','PNT');n=s.fast.native
    h=n.histo_phases[0][s.fast.slow.hist.name];hfx=f":{h['hId']}:"
    if change=='profile':
        for name,value in [('X',.015),('Y',.025),('Z',.01)]:n.parms[hfx+name]=value
    if change=='absorption':n.parms[hfx+'Absorption']=.1
    if change=='lookup':n.histdict[hfx+'pdabc']={'d':[.1,3.,20.],'sig':[.01,.07,.03]}
    if change=='zero_counts':h['Data'][1][:]=0.
    original=[np.ma.array(a,copy=True) for a in h['Data'][:3]]
    engine=for_source(s);assert engine is not None
    for i in range(3):
        point=np.zeros(len(s.radii))
        if change=='mixed':point=np.linspace(-.06,.08,len(point))*(i+1)/3
        if change=='cell_edge':point[s.mode_dimensions:]=np.linspace(-.8,.8,s.cell_dimensions)*(i+1)/3
        physical=point*s.radii
        expected=s.native(physical).copy()
        np.testing.assert_array_equal(engine.evaluate(physical),expected)
    assert engine.diagnostics()['qualified_anchor'] and engine.anchor_checks==1,engine.diagnostics()
    assert engine.anchor_comparison['array_identical']
    assert engine.anchor_comparison['differing_bins']==0
    for got,expected in zip(h['Data'][:3],original):np.testing.assert_array_equal(got,expected)


@pytest.mark.parametrize('change',['cw','mask','anisotropic','extinction','background','factor'])
def test_unsupported_context_keeps_original_native_forward(tmp_path,monkeypatch,change):
    _,s=native_source(tmp_path/'inputs','PNC' if change=='cw' else 'PNT');n=s.fast.native
    h=n.histo_phases[0][s.fast.slow.hist.name];hfx=f":{h['hId']}:"
    phfx=f"{n.histo_phases[1][s.fast.slow.phase.name]['pId']}:{h['hId']}:"
    if change=='mask':
        h['Data'][0]=np.ma.array(h['Data'][0],mask=np.zeros(len(h['Data'][0]),bool))
        h['Data'][0].mask[10]=True
    if change=='anisotropic':n.controls[phfx+'MustrainType']='uniaxial'
    if change=='extinction':n.parms[phfx+'Extinction']=.01
    if change=='background':monkeypatch.setattr(n.math.G2pwd,'getBackground',lambda *a:None)
    if change=='factor':n.factor[0]=2.
    assert for_source(s) is None


def test_anchor_disagreement_disables_acceleration_and_keeps_original_curve(tmp_path,monkeypatch):
    _,s=native_source(tmp_path/'inputs','PNT');engine=for_source(s);assert engine is not None
    point=np.zeros(len(s.radii));expected=s.native(point).copy();original=engine.collector.collect_physical
    def changed(*args,**kwargs):
        records=original(*args,**kwargs);key=next(iter(records));par,height,lo,hi=records[key]
        records[key]=(par,height+1.,lo,hi);return records
    monkeypatch.setattr(engine.collector,'collect_physical',changed)
    np.testing.assert_array_equal(engine.evaluate(point),expected)
    assert engine.disabled and not engine.qualified
    assert not engine.anchor_comparison['array_identical']
    assert engine.anchor_comparison['differing_bins']>0
    assert engine.anchor_comparison['maximum_absolute_error']>0
    monkeypatch.setattr(engine.collector,'collect_physical',lambda *a,**k:pytest.fail('A disabled renderer cannot run again'))
    np.testing.assert_array_equal(engine.evaluate(point),expected)
    assert engine.diagnostics()['bypasses']['native anchor is not array-identical']==1


def test_actual_native_owner_cannot_be_bypassed(tmp_path,monkeypatch):
    _,s=native_source(tmp_path/'inputs','PNT');engine=for_source(s);assert engine is not None
    def replaced():raise RuntimeError('Native owner changed')
    monkeypatch.setattr(s.fast.native,'_check',replaced)
    with pytest.raises(RuntimeError,match='owner changed'):engine.evaluate(np.zeros(len(s.radii)))


def test_small_workspace_falls_back_without_removing_reflections(tmp_path):
    _,s=native_source(tmp_path/'inputs','PNT')
    assert for_source(s,maximum_bytes=1) is None


def test_later_profile_edit_recalculates_native_scalars(tmp_path):
    _,s=native_source(tmp_path/'inputs','PNT');engine=for_source(s)
    point=np.zeros(len(s.radii));before=engine.evaluate(point).copy()
    prefix=f':{s.fast.slow.hist.id}:'
    s.fast.native.parms[prefix+'X']+=.04
    expected=s.native(point).copy()
    np.testing.assert_array_equal(engine.evaluate(point),expected)
    assert not np.array_equal(before,expected)
    assert engine.anchor_checks==1 and engine.fallback_calls==0


def test_later_unreviewed_background_uses_original_native_path(tmp_path,monkeypatch):
    _,s=native_source(tmp_path/'inputs','PNT');engine=for_source(s)
    point=np.zeros(len(s.radii));engine.evaluate(point)
    powder=s.fast.native.math.G2pwd;original=powder.getBackground
    monkeypatch.setattr(powder,'getBackground',lambda *a,**kw:original(*a,**kw))
    expected=s.native(point).copy()
    np.testing.assert_array_equal(engine.evaluate(point),expected)
    assert engine.anchor_checks==1 and engine.fallback_calls==1
    assert engine.bypasses['Native TOF forward context changed']==1


@pytest.mark.parametrize('budget',[0,-1,True,1.5,64*1024*1024+1])
def test_invalid_workspace_cannot_start_native_construction(budget):
    with pytest.raises(ValueError,match='workspace'):NativeTOFForward(object(),maximum_bytes=budget)
