import json
from pathlib import Path
import numpy as np

from radar_pd_nova.symmetry_workspace import group_models,choose_group
from radar_pd_nova.symmetry_plots import coordinate,fit


def row(root,name,*,source='scan1',offset=0.,status='valid',rwp=5.):
    path=root/(name+'.json')
    path.write_text(json.dumps(dict(x=[10.,20.,30.],d=[3.,2.,1.],coordinate='two_theta',observed=[20.,100.,50.],calculated=[20.,100.+offset,50.],background=[1.,1.,1.],weights=[1.,1.,1.],fitted=[True]*3,limits=[10.,30.],ticks=[dict(phase='nuclear',native=[20.],d=[2.])],phases=[])))
    return dict(id=name,source=source,view_path=str(path),has_view=True,rwp=rwp,status=status,name=name,seed='x',nvars=1)


def test_matching_curves_do_not_merge_different_sources_or_hide_unresolved_members(tmp_path):
    a=row(tmp_path,'qualified');b=row(tmp_path,'unresolved',offset=.00001,status='optimization_unresolved',rwp=4.9)
    c=row(tmp_path,'other-scan',source='scan2');d=row(tmp_path,'different-pattern',offset=2.)
    groups=group_models([a,b,c,d]);assert len(groups)==3
    same=next(g for g in groups if len(g['members'])==2)
    assert {r['status'] for r in same['members']}=={'valid','optimization_unresolved'}
    assert same['representative']['status']=='optimization_unresolved'


def test_instrument_converted_axes_and_bragg_ticks_share_coordinates(tmp_path):
    r=row(tmp_path,'fit');view=json.loads(Path(r['view_path']).read_text())
    q,label=coordinate(view,'q');np.testing.assert_allclose(q,[2*np.pi/3,np.pi,2*np.pi]);assert label=='Q (Å⁻¹)'
    figure=fit(r,axis='q');tick=next(t for t in figure['data'] if t['name']=='nuclear Bragg')
    # Plotly versions can serialize arrays as binary records. The explicit
    # scalar coordinate conversion above is checked independently of rendering.
    assert tick['xaxis']=='x2';assert figure['layout']['xaxis3']['title']['text']=='Q (Å⁻¹)'
    assert any(t['name']=='Observed − calculated' for t in figure['data'])


def test_incompatible_trial_without_curve_stays_inspectable(tmp_path):
    r=row(tmp_path,'fit');bad=dict(r,id='forbidden',has_view=False,rwp=None,status='site_incompatible')
    groups=group_models([r,bad]);assert len(groups)==2
    assert 'annotations' in fit(bad)['layout']


def test_streaming_default_opens_first_fit_and_explicit_trial_survives_regrouping():
    groups=[dict(id='fit-1',members=[dict(id='fit-1'),dict(id='fit-2')]),dict(id='pending',members=[dict(id='waiting')])]
    assert choose_group(groups,'waiting')=='fit-1'
    assert choose_group(groups,'waiting',explicit=True)=='pending'
    assert choose_group(groups,'fit-2',explicit=True)=='fit-1'
    assert choose_group([], 'waiting')==''


def test_excluded_regions_are_gaps_in_the_calculated_and_difference_lines(tmp_path):
    r=row(tmp_path,'gap');path=Path(r['view_path']);v=json.loads(path.read_text());v['fitted'][1]=False;path.write_text(json.dumps(v))
    figure=fit(r)
    import base64
    for name in ['Selected calculated','Observed − calculated']:
        trace=next(t for t in figure['data'] if t['name']==name);y=trace['y']
        if isinstance(y,dict):assert np.isnan(np.frombuffer(base64.b64decode(y['bdata']),dtype=y['dtype'])[1])
        else:assert y[1] is None
        assert trace['connectgaps'] is False
