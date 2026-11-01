import numpy as np
import pytest
from radar_pd_nova.webmon.plots import comparison, coordinates, moments
from radar_pd_nova.webmon.viewdata import moment_rows, full_posterior

def post():
    return dict(x=np.array([100.,200.,300.]),y=np.array([-1.,4.,2.]),sigma=np.array([2.,1.,2.]),fit=np.array([0.,3.,1.]),mean_band=np.array([[0.,2.,0.],[0.,3.,1.],[0.,4.,2.]]),intervals=[[5.00001,5.00002,5.00003]],labels=['Er1'],relative_half_width=[.000002],prior_moment_draws=[[5.1],[6.],[6.9]])

def test_shared_axes_and_difference_are_in_same_error_units():
    a=post();b=post();b['fit']=a['fit']+a['sigma']*2
    f=comparison([dict(id='a',label='A',color='#123456',post=a),dict(id='b',label='B',color='#654321',post=b)],axis='d',difc=100)
    np.testing.assert_array_equal(f.data[0].x,[1,2,3])
    np.testing.assert_allclose(next(t.y for t in f.data if t.name=='B minus reference'),2)
    assert f.layout.xaxis.matches=='x4' and f.layout.xaxis2.matches=='x4'
    assert f.layout.xaxis4.title.text=='d (Å)'

def test_cannot_compare_different_data_or_guess_conversion():
    a=post();b=post();b['y']=a['y']+1
    with pytest.raises(ValueError,match='same observations'):comparison([dict(id='a',label='A',color='blue',post=a),dict(id='b',label='B',color='red',post=b)])
    with pytest.raises(ValueError,match='Verified'):coordinates(a,'d',None)

def test_bound_pressure_stays_visible_with_narrow_interval():
    p=post();r=moment_rows(p,5,7)
    assert r[0]['warning']=='Touches imposed bound'
    f=moments(p,5,7)
    assert len(f.layout.shapes)==2
    assert f.data[-1].marker.symbol=='diamond'

def test_saved_plot_reconstructs_peak_omitted_from_thinned_payload(tmp_path):
    import json
    (tmp_path/'posterior.json').write_text(json.dumps({'x':[1,3],'fit':[0,0]}))
    np.savez(tmp_path/'kernel.npz',x=[1,2,3],y=[0,100,0],sigma=[1,1,1],base=[0,0,0],quadratic=np.array([[[0,25,0]]]))
    np.savez(tmp_path/'posterior.npz',samples=[[1],[2]],loglike=[-20,0])
    p=full_posterior(tmp_path/'posterior.json')
    np.testing.assert_array_equal(p['fit'],[0,100,0]);assert len(p['x'])==3

def test_full_range_explicitly_clears_previous_focus():
    f=comparison([dict(id='a',label='A',color='blue',post=post())],axis='d',difc=100,region='full')
    assert list(f.layout.xaxis.range)==[1,3]
    assert list(f.layout.xaxis3.range)==[1,3]

def test_selection_limit_and_table_do_not_rewrite_valid_client_selection():
    from types import SimpleNamespace
    from radar_pd_nova.webmon.ui import WebmonMixin
    obj=WebmonMixin();s=SimpleNamespace(webmon_message='',webmon_compared=[],webmon_table_selected=['a','b'])
    obj.server=SimpleNamespace(state=s);obj._webmon_model_paths=set('abcdef');obj.webmon_render=lambda:None
    obj.webmon_compare(['a','b'],from_table=True)
    assert s.webmon_compared==['a','b'] and s.webmon_table_selected==['a','b']
    obj.webmon_compare(list('abcdef'),from_table=True)
    assert s.webmon_compared==list('abcde') and s.webmon_table_selected==list('abcde')
    assert 'five' in s.webmon_message

def test_control_event_applies_value_before_rendering():
    from types import SimpleNamespace
    from radar_pd_nova.webmon.ui import WebmonMixin
    obj=WebmonMixin();state={'webmon_axis':'tof'};obj.server=SimpleNamespace(state=state);seen=[]
    obj.webmon_render=lambda:seen.append(state['webmon_axis'])
    obj.webmon_control('webmon_axis','d')
    assert seen==['d']
    obj.webmon_control('unrelated','bad');assert 'unrelated' not in state


def test_q_sorting_per_model_envelopes_rwp_and_native_ticks():
    a=post();a['rwp']=35.2
    a['reflections']=[dict(phase='Er nuclear',kind='nuclear',tof=[200],hkl=[[1,2,3]],f2=[8]),dict(phase='Er magnetic',kind='magnetic',tof=[300],hkl=[[2,3,4]],f2=[9])]
    b=post();b['mean_band']=b['mean_band']+10;b['rwp']=33.3
    f=comparison([dict(id='a',label='A',color='#123456',post=a),dict(id='b',label='B',color='#654321',post=b)],axis='q',difc=100,ensemble=True)
    bands=[t for t in f.data if t.fill=='tonexty']
    assert len(bands)==2 and {t.legendgroup for t in bands}=={'a','b'}
    np.testing.assert_allclose(bands[0].x,2*np.pi/np.array([3,2,1]))
    np.testing.assert_allclose(bands[0].y,[2e6,4e6,0])
    assert any('Rwp 35.20%' in (t.name or '') for t in f.data)
    ticks=[t for t in f.data if t.yaxis=='y2']
    np.testing.assert_allclose(ticks[0].x,[np.pi]);assert '(1 2 3)' in ticks[0].text[0]
    assert f.layout.xaxis4.title.text=='Q (Å⁻¹)'


def test_single_orbit_and_invalid_pair_all_plots_serialize():
    from radar_pd_nova.webmon.plots import correlation,sampler
    from radar_pd_nova.webmon.viewdata import sampler_summary
    p=post();p.update(moment_draws=[[5.1],[5.2]],particles=256,sampler_history=[dict(beta=.5,ess=204.8,acceptance=.05)])
    figs=[comparison([dict(id='a',label='A',color='#123456',post=p)],ensemble=True),moments(p),correlation(p),correlation(p,(-1,10)),sampler(p)]
    assert 'One independent moment' in figs[2].layout.annotations[0].text
    for f in figs:assert f.to_json()
    assert figs[-1].data[0].y[0]==80 and figs[-1].data[1].y[0]==5
    assert 'Local acceptance was not saved' in sampler_summary(p)


def test_reflection_provenance_must_match_saved_fit(tmp_path):
    import json,hashlib
    (tmp_path/'posterior.json').write_text('{}')
    np.savez(tmp_path/'kernel.npz',x=[1],y=[1],sigma=[1],base=[0],quadratic=[[[1]]])
    np.savez(tmp_path/'posterior.npz',samples=[[1]],loglike=[0])
    metadata=dict(kernel_sha256=hashlib.sha256((tmp_path/'kernel.npz').read_bytes()).hexdigest(),posterior_sha256=hashlib.sha256((tmp_path/'posterior.npz').read_bytes()).hexdigest(),scope='Native ticks',tables=[{'kind':'nuclear'}])
    (tmp_path/'reflections.json').write_text(json.dumps(metadata))
    assert full_posterior(tmp_path/'posterior.json')['reflections']==metadata['tables']
    np.savez(tmp_path/'posterior.npz',samples=[[2]],loglike=[0])
    p=full_posterior(tmp_path/'posterior.json');assert not p['reflections'] and 'withheld' in p['reflection_status']
