import json
from pathlib import Path
import numpy as np

from radar_pd_nova.symmetry_workspace import group_models,choose_group,disposition_reason,coverage_rows,scope_rows,parameter_rows,moment_precision_note
from radar_pd_nova.symmetry_plots import coordinate,fit


def test_cached_evidence_rehashes_changed_bytes_even_if_mtime_is_restored(tmp_path,monkeypatch):
    import os
    import radar_pd_nova.symmetry_workspace as module
    module.clear_evidence_cache()
    p=dict(source=dict(path='source.gpx'),catalog=dict(path='catalog.json'),candidates=[])
    for name in ['source.gpx','catalog.json','job.json']:(tmp_path/name).write_text('data')
    receipt=dict(id='trial',protocol_sha256='p',files={'fit.gpx':'hash'})
    (tmp_path/'result.json').write_text(json.dumps(receipt));(tmp_path/'fit.gpx').write_text('AAAA')
    job=dict(id='trial',protocol_sha256='p',schema='radar-symmetry-job/v1',protocol=p)
    calls=[]
    def verify(*args,**kwargs):calls.append(1);return receipt
    monkeypatch.setattr(module,'verify_receipt',verify)
    module.checked_receipt(tmp_path,job);module.checked_receipt(tmp_path,job)
    assert len(calls)==1
    target=tmp_path/'fit.gpx';st=target.stat();target.write_text('BBBB');os.utime(target,ns=(st.st_atime_ns,st.st_mtime_ns))
    module.checked_receipt(tmp_path,job);assert len(calls)==2
    module.clear_evidence_cache();module.checked_receipt(tmp_path,job);assert len(calls)==3


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


def test_same_acquisition_with_a_different_starting_gpx_revision_does_not_merge(tmp_path):
    original=row(tmp_path,'original');revised=row(tmp_path,'revised')
    original['source_sha256']='a'*64;revised['source_sha256']='b'*64
    assert original['source']==revised['source']
    assert len(group_models([original,revised]))==2


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


def test_an_audited_checkpoint_is_not_described_as_a_qualified_optimization():
    reason=disposition_reason(dict(status='optimization_unresolved',has_view=True))
    assert 'did not qualify' in reason
    assert 'Awaiting' in disposition_reason(dict(status='running',has_view=False))
    assert disposition_reason(dict(status='unsupported',reason='Different BNS setting',has_view=False))=='Different BNS setting'


def test_excluded_regions_are_gaps_in_the_calculated_and_difference_lines(tmp_path):
    r=row(tmp_path,'gap');path=Path(r['view_path']);v=json.loads(path.read_text());v['fitted'][1]=False;path.write_text(json.dumps(v))
    figure=fit(r)
    import base64
    for name in ['Selected calculated','Observed − calculated']:
        trace=next(t for t in figure['data'] if t['name']==name);y=trace['y']
        if isinstance(y,dict):assert np.isnan(np.frombuffer(base64.b64decode(y['bdata']),dtype=y['dtype'])[1])
        else:assert y[1] is None
        assert trace['connectgaps'] is False


def test_coverage_distinguishes_hypotheses_without_claiming_a_structure(tmp_path):
    a=row(tmp_path,'qualified');b=row(tmp_path,'unresolved',status='optimization_unresolved',rwp=4.)
    for r in [a,b]:r.update(protocol='cr-only',metadata=dict(scan='PG3_1',condition=7.,condition_unit='K',bank=3),execution_state='terminal',sites=[dict(label='Cr1',form_factor='Cr+3')])
    c=dict(b,protocol='cr-pr',status='prepared',rwp=None,has_view=False,execution_state='prepared')
    result=coverage_rows([a,b,c])
    assert result[0]['qualified']==1 and result[0]['best_audited']=='4.000%' and result[0]['best_qualified']=='5.000%'
    assert result[0]['condition']=='7 K' and result[0]['coverage']=='2 / 2'
    assert result[1]['coverage']=='0 / 1' and result[1]['status']=='In progress'


def test_long_real_project_names_do_not_fill_the_plot_legend(tmp_path):
    r=row(tmp_path,'real');p=Path(r['view_path']);v=json.loads(p.read_text())
    name='A very long RADAR-PD publication phase label '*5
    v['phases']=[dict(name=name,type='nuclear')];v['ticks'][0]['phase']=name;p.write_text(json.dumps(v))
    figure=fit(r);tick=next(t for t in figure['data'] if t['name']=='Nuclear Bragg')
    assert name in tick['hovertemplate']
    assert figure['layout']['xaxis3']['title']['text']=='d (Å)'
    assert fit(r,axis='q')['layout']['xaxis3']['title']['text']=='Q (Å⁻¹)'


def test_log_intensity_keeps_the_residual_linear(tmp_path):
    r=row(tmp_path,'log');figure=fit(r,intensity_scale='log')
    assert figure['layout']['yaxis']['type']=='log'
    assert figure['layout']['yaxis3'].get('type','linear')=='linear'


def test_selected_experiment_does_not_silently_show_another_ipts():
    rows=[dict(metadata=dict(series='IPTS-1')),dict(metadata=dict(series='IPTS-2'))]
    assert scope_rows(rows,'IPTS-1')==[rows[0]]
    assert scope_rows(rows,'IPTS-3')==[]
    assert scope_rows(rows,'IPTS-1','campaign')==rows


def test_small_nonzero_moment_keeps_its_large_native_conditional_uncertainty():
    view=dict(phases=[dict(name='Fe magnetic',moments=[dict(label='Fe1_m2',magnitude=.2)])])
    r=dict(refined_moments=[dict(label='Fe1_m2',magnitude_esd=1.1)])
    value=parameter_rows(r,view)[0]
    assert value['value']==.2 and value['uncertainty']==1.1 and value['unit']=='µB'
    assert parameter_rows({},view)[0]['uncertainty']=='—'


def test_warm_scan_precision_warning_does_not_relabel_a_native_fit_or_invent_norm_uncertainty():
    row=dict(kind='magnetic',status='valid',refined_moments=[dict(label='Tb1',moment=[.238,.238,.264],
        component_esd={'mx':42.85,'my':42.67,'mz':.1445},magnitude=.428,magnitude_esd=None)])
    note=moment_precision_note(row)
    assert 'Weak conditional moment' in note and '1.83' in note
    assert 'not a joint significance test' in note
    assert row['status']=='valid' and row['refined_moments'][0]['magnitude_esd'] is None


def test_precision_screen_keeps_strong_and_unavailable_estimates_distinct():
    assert moment_precision_note(dict(kind='magnetic',refined_moments=[dict(label='Er1',moment=[1.7,1.3,6.1],
        component_esd={'mx':.1,'my':.2,'mz':.1})]))==''
    assert 'unavailable' in moment_precision_note(dict(kind='magnetic',refined_moments=[dict(label='Tb1',moment=[.2,0,0],
        component_esd={'mx':None,'my':None,'mz':None},magnitude=.2,magnitude_esd=None)]))
    assert moment_precision_note(dict(kind='structural'))==''


def test_experiment_scope_filters_before_reading_foreign_result_files(tmp_path,monkeypatch):
    from contextlib import nullcontext
    import radar_pd_nova.symmetry_workspace as module
    class DB:
        def execute(self,*args):return [dict(hash='a',metadata=json.dumps(dict(series='IPTS-1'))),dict(hash='b',metadata=json.dumps(dict(series='IPTS-2')))]
    class C:
        def __init__(self,root):pass
        def connect(self):return nullcontext(DB())
        def trials(self):return [dict(id='first',protocol='a',state='prepared',receipt=None,error=None),dict(id='foreign',protocol='b',state='prepared',receipt=None,error=None)]
        def trial_directory(self,row):return tmp_path/row['id']
        def summaries(self):return [dict(protocol='a'),dict(protocol='b')]
    accessed=[]
    def read(path,metadata,**kwargs):
        accessed.append(path.name);return dict(status='pending',metadata=metadata)
    monkeypatch.setattr(module,'Campaign',C);monkeypatch.setattr(module,'read_trial',read)
    rows,summaries=module.campaign_rows(tmp_path,experiment='IPTS-1')
    assert accessed==['first'] and len(rows)==1
    assert len(summaries)==2 and all(s['pending_count']==1 for s in summaries)


def test_changed_collected_receipt_never_exposes_a_verified_plot_or_download(tmp_path,monkeypatch):
    import radar_pd_nova.symmetry_workspace as module
    candidate=dict(id='c',magnetic_space_group='P1',irrep='',bns_number='1.1',transformation_matrix=np.eye(3).tolist(),origin_shift=[0,0,0])
    protocol=dict(candidates=[candidate],kind='magnetic',source={'sha256':'a'*64},parent_phase='Parent',histogram='h',k=['0','0','0'])
    (tmp_path/'job.json').write_text(json.dumps(dict(id='trial',protocol=protocol,protocol_sha256='p',candidate_id='c',seed_id='s')))
    (tmp_path/'result.json').write_text('{}')
    result=dict(status='valid',rwp=1.,nvars=1,files={'results/model.gpx':'a','results/model.view.json':'b'})
    monkeypatch.setattr(module,'verify_receipt',lambda *args:result)
    value=module.read_trial(tmp_path,expected_receipt=json.dumps(dict(result,rwp=2.)))
    assert value['status']=='integrity_failure' and 'changed after collection' in value['reason']
    assert not value['has_view'] and not value['has_gpx'] and value['rwp'] is None
    assert 'receipt_sha256' not in value
    (tmp_path/'result.json').unlink()
    assert module.read_trial(tmp_path,expected_receipt=json.dumps(result))['status']=='integrity_failure'


def test_a_late_native_evidence_read_failure_clears_all_verified_artifact_flags(tmp_path,monkeypatch):
    import radar_pd_nova.symmetry_workspace as module
    candidate=dict(id='c',magnetic_space_group='P1',irrep='',bns_number='1.1',transformation_matrix=np.eye(3).tolist(),origin_shift=[0,0,0])
    protocol=dict(candidates=[candidate],kind='magnetic',source={'sha256':'a'*64},parent_phase='Parent',histogram='h',k=['0','0','0'])
    (tmp_path/'job.json').write_text(json.dumps(dict(id='trial',protocol=protocol,protocol_sha256='p',candidate_id='c',seed_id='s')))
    (tmp_path/'result.json').write_text('{}');(tmp_path/'results').mkdir()
    (tmp_path/'results/native.json').write_text('changed during read')
    result=dict(status='valid',rwp=1.,nvars=1,gpx='model.gpx',native_receipt='native.json',
                files={'results/model.gpx':'a','results/model.view.json':'b','results/native.json':'c'})
    monkeypatch.setattr(module,'verify_receipt',lambda *args:result)
    value=module.read_trial(tmp_path)
    assert value['status']=='integrity_failure' and value['rwp'] is None
    assert not value['has_view'] and not value['has_gpx']
    assert not {'receipt_sha256','view_path','gpx_relative'} & set(value)


def test_canted_components_show_native_uncertainties_without_inventing_a_norm_error():
    view=dict(phases=[dict(name='Er magnetic',moments=[dict(label='Er1_m1',magnitude=6.5,components=[1.7,1.3,6.1])])])
    row=dict(refined_moments=[dict(label='Er1_m1',magnitude_esd=None,component_esd={'mx':.3,'my':.4,'mz':.1})])
    values=parameter_rows(row,view)
    assert values[0]['value']==6.5 and values[0]['uncertainty']=='—'
    assert [v['parameter'] for v in values[1:]]==['Er1_m1 · mx','Er1_m1 · my','Er1_m1 · mz']
    assert [v['value'] for v in values[1:]]==[1.7,1.3,6.1]
    assert [v['uncertainty'] for v in values[1:]]==[.3,.4,.1]
