from copy import deepcopy
from types import SimpleNamespace

import pytest

from radar_pd_nova.refinement_parameters import execution_parameters, parameter_table


SOURCE, PLAN, FIRST, SECOND, FINAL, RECEIPT = (c*64 for c in 'ab123e')


def native(variable, family, group, value, sigma=.01):
    return dict(variable=variable,family=family,group=group,after=value,sigma=sigma,
                native_status='refined_independent',meaning='Native '+family)


def fixture():
    view=dict(source_sha256=SECOND,phases=[dict(key='p0',native_id=7,name='Oxide')],
              histograms=[dict(key='h0',name='bank5')],parameters=[
                  dict(key='cell|Oxide|a',kind='cell',label='Oxide a',value=4.,sigma=.1,status='refined')])
    scale=native('7:3:Scale','Scale','phase_fraction',1.03,.02)
    background=native(':3:Back;0','Back;0','background',5.,.4)
    orientation=native('7:3:MD','MD','preferred_orientation',.973,.0048)
    def stage(key,records):
        return dict(id=key,status='completed',parameter_changes=dict(released_parameters=deepcopy(records)),
                    final_state=dict(phases=[dict(key='p0',hap={'bank5':dict(scale=[1.03,True],
                        texture=['MD',.973,True,[0,0,1]])})]))
    execution=dict(schema='radar-gpx-execution/v1',preview=False,source_sha256=SOURCE,plan_sha256=PLAN,
        status='completed_needs_scientific_review',output_gpx_sha256=FINAL,
        stages=[stage('stage1',[scale,background]),stage('stage2',[scale,background,orientation])])
    result=dict(source_sha256=SOURCE,status='completed',execution='polished.execution.json',
        final_gpx='verified_final.gpx',output_sha256=FINAL,final_view='final.view.json',
        stages=[dict(id='stage1',gpx='verified_stage1.gpx',view='stage1.view.json'),
                dict(id='stage2',gpx='verified_stage2.gpx',view='stage2.view.json')],
        files={'verified_stage1.gpx':FIRST,'verified_stage2.gpx':SECOND,'verified_final.gpx':FINAL,
               'polished.execution.json':RECEIPT})
    return view,result,execution


def rows(view,result,execution,selected='stage2',**kwargs):
    return execution_parameters(view,result,execution,selected,source_sha=SOURCE,plan_sha=PLAN,**kwargs)


def test_selected_texture_value_sigma_and_native_scale_are_visible_without_mutating_view():
    v,r,e=fixture();before=deepcopy((v,r,e))
    extra,notice=rows(v,r,e)
    md=next(p for p in extra if p['parameter']=='MD')
    assert md['value']==.973 and md['sigma']==.0048 and md['unit']=='dimensionless'
    assert 'Preferred orientation' in md['label'] and md['histogram']=='bank5'
    assert md['native_variable']=='7:3:MD' and md['evidence_stage']=='stage2'
    assert next(p for p in extra if p['parameter']=='Scale')['unit']=='native scale'
    assert 'not weight %' in next(p for p in extra if p['parameter']=='Scale')['label']
    assert 'conditional' in notice and 'not a calibrated' in notice
    merged=parameter_table(v,extra);merged[0]['value']=99
    assert (v,r,e)==before


def test_stage_one_cannot_see_future_texture_or_failed_stage_values():
    v,r,e=fixture();v['source_sha256']=FIRST
    e['status']='failed';e['stages'][1]['status']='refined_needs_review'
    e['stages'][1]['parameter_changes']['released_parameters'][0]['after']=999
    r.update(status='needs_review',stages=r['stages'][:1]);r.pop('final_gpx');r.pop('final_view')
    extra,_=rows(v,r,e,'stage1')
    assert all(p['parameter']!='MD' and p['value']!=999 for p in extra)
    assert next(p for p in extra if p['parameter']=='Scale')['value']==1.03
    with pytest.raises(ValueError):rows(v,r,e,'stage2')
    with pytest.raises(ValueError):rows(v,r,e,'final')


@pytest.mark.parametrize('fault',['source','plan','view','stage_hash','final_hash','preview','missing_receipt',
                                  'duplicate_stage','unpublished_stage','failed_before'])
def test_unbound_receipts_cannot_supply_values(fault):
    v,r,e=fixture();selected='stage2'
    if fault=='source':e['source_sha256']='f'*64
    elif fault=='plan':e['plan_sha256']='f'*64
    elif fault=='view':v['source_sha256']=FIRST
    elif fault=='stage_hash':r['files']['verified_stage2.gpx']='f'*64
    elif fault=='final_hash':selected='final';v['source_sha256']=FINAL;e['output_gpx_sha256']='f'*64
    elif fault=='preview':e['preview']=True
    elif fault=='missing_receipt':del r['files']['polished.execution.json']
    elif fault=='duplicate_stage':e['stages'].append(deepcopy(e['stages'][-1]))
    elif fault=='unpublished_stage':r['stages']=r['stages'][:1]
    else:e['stages'][0]['status']='failed'
    with pytest.raises(ValueError):rows(v,r,e,selected)


def test_completed_final_uses_exact_last_stage_and_final_gpx():
    v,r,e=fixture();v['source_sha256']=FINAL
    extra,_=rows(v,r,e,'final')
    assert next(p for p in extra if p['parameter']=='MD')['refined_stage']=='stage2'


@pytest.mark.parametrize('fault',['receipt_type','files_type','ledger_type'])
def test_malformed_receipt_is_rejected_without_replacing_native_view_rows(fault):
    v,r,e=fixture();before=deepcopy(v)
    if fault=='receipt_type':e=[]
    elif fault=='files_type':r['files']=['not a manifest']
    else:e['stages'][1]['parameter_changes']=['not a ledger']
    with pytest.raises(ValueError):rows(v,r,e)
    assert v==before


@pytest.mark.parametrize('fault',['nan_value','infinite_sigma','negative_sigma','duplicate_variable','wrong_phase'])
def test_invalid_native_parameter_rows_are_withheld(fault):
    v,r,e=fixture();record=e['stages'][1]['parameter_changes']['released_parameters'][-1]
    if fault=='nan_value':record['after']=float('nan')
    elif fault=='infinite_sigma':record['sigma']=float('inf')
    elif fault=='negative_sigma':record['sigma']=-1
    elif fault=='duplicate_variable':e['stages'][1]['parameter_changes']['released_parameters'].append(deepcopy(record))
    else:record['variable']='8:3:MD'
    with pytest.raises(ValueError):rows(v,r,e)


def test_existing_cell_atom_size_strain_and_weight_rows_are_not_duplicated():
    v,r,e=fixture()
    e['stages'][1]['parameter_changes']['released_parameters'].extend([
        native('7::A0','A0','lattice',.1),native('7::AUiso:0','AUiso','atomic_displacement',.02),
        native('7:3:Size;i','Size;i','crystallite_size',1.),native('7:3:Mustrain;i','Mustrain;i','microstrain',10),
        native('7:3:WgtFrac','WgtFrac','weight_fraction',1.)])
    extra,_=rows(v,r,e)
    assert len(extra)==3 and len(parameter_table(v,extra))==4


def test_existing_native_variable_export_takes_precedence_over_a_supplement():
    v,r,e=fixture();extra,_=rows(v,r,e)
    v['parameters'].append(dict(key='native-texture',native_variable='7:3:MD',value=.973))
    merged=parameter_table(v,extra)
    assert len([p for p in merged if p.get('native_variable')=='7:3:MD'])==1
    assert next(p for p in merged if p.get('native_variable')=='7:3:MD')['key']=='native-texture'


@pytest.mark.parametrize('fault',[None,'changed_value','changed_direction','unrecorded_held'])
def test_prior_sigma_is_used_only_for_unchanged_explicitly_held_texture(fault):
    v,r,e=fixture();v['source_sha256']=FINAL
    third=deepcopy(e['stages'][-1]);third['id']='stage3'
    third['parameter_changes']['released_parameters']=[]
    texture=third['final_state']['phases'][0]['hap']['bank5']['texture'];texture[2]=False
    third['final_state']['phases'][0]['hap']['bank5']['scale'][1]=False
    if fault=='changed_value':texture[1]=.5
    elif fault=='changed_direction':texture[3]=[1,0,0]
    elif fault=='unrecorded_held':texture[2]=True
    e['stages'].append(third);r['stages'].append(dict(id='stage3',gpx='stage3.gpx',view='stage3.view.json'))
    r['files']['stage3.gpx']='4'*64
    extra,notice=rows(v,r,e,'final')
    md=[p for p in extra if p['parameter']=='MD']
    if fault is None:
        assert len(md)==1 and md[0]['status']=='refined_in_prior_stage'
        assert md[0]['sigma']==.0048 and md[0]['refined_stage']=='stage2' and md[0]['evidence_stage']=='stage3'
    else:assert not md
    assert 'omitted' in notice  # Held background coefficients lack a final-state value.


def test_multibank_labels_need_exact_source_context_not_representative_ids():
    v,r,e=fixture();v['histograms'].append(dict(key='h1',name='bank2'))
    context=dict(source=dict(sha256='f'*64),histograms=[dict(key='h1',name='bank2',native_id=3)])
    extra,_=rows(v,r,e,context=context)
    assert all(p['histogram'] is None and 'Native histogram 3' in p['label'] for p in extra)
    context['source']['sha256']=SOURCE
    extra,_=rows(v,r,e,context=context)
    assert all(p['histogram']=='bank2' for p in extra)


def test_unknown_selected_stage_clears_previously_displayed_parameters():
    from radar_pd_nova.refinement_controller import RefinementMixin
    s=SimpleNamespace(**RefinementMixin._refinement_state_defaults())
    s.rf_has_output=True;s.rf_selected_job='job';s.rf_experiment='exp';s.rf_stage='missing'
    s.rf_parameters=[{'label':'stale texture'}];s.rf_parameter_notice='old notice'
    job={'source':{'experiment':'exp'},'result':{}}
    app=SimpleNamespace(server=SimpleNamespace(state=s),_refinement_workspace=SimpleNamespace(state={'jobs':{'job':job}}))
    RefinementMixin._refinement_plot_changed(app)
    assert s.rf_parameters==[] and s.rf_parameter_notice=='' and 'completed stage' in s.rf_error


def test_selection_change_during_read_cannot_publish_old_job_rows():
    from radar_pd_nova.refinement_controller import RefinementMixin
    s=SimpleNamespace(**RefinementMixin._refinement_state_defaults())
    s.rf_has_output=True;s.rf_selected_job='job';s.rf_experiment='exp';s.rf_stage='stage1'
    job={'source':{'experiment':'exp'},'result':{'baseline_view':'base','stages':[{'id':'stage1','view':'one'}]}}
    def read(*_):
        s.rf_selected_job='new';s.rf_parameters=[{'label':'new job owns these rows'}]
        return {'parameters':[]}
    app=SimpleNamespace(server=SimpleNamespace(state=s),_refinement_workspace=SimpleNamespace(state={'jobs':{'job':job}}),
                        _read_refinement_view=read)
    RefinementMixin._refinement_plot_changed(app)
    assert s.rf_parameters==[{'label':'new job owns these rows'}]


def test_changed_publication_during_read_cannot_publish_old_rows():
    from radar_pd_nova.refinement_controller import RefinementMixin
    s=SimpleNamespace(**RefinementMixin._refinement_state_defaults())
    s.rf_has_output=True;s.rf_selected_job='job';s.rf_experiment='exp';s.rf_stage='stage1'
    result={'baseline_view':'base','stages':[{'id':'stage1','view':'one'}],'files':{'one':FIRST}}
    job={'source':{'experiment':'exp'},'result':result}
    def read(*_):
        result['files']['one']=SECOND
        s.rf_parameters=[{'label':'new publication owns these rows'}]
        return {'parameters':[]}
    app=SimpleNamespace(server=SimpleNamespace(state=s),_refinement_workspace=SimpleNamespace(state={'jobs':{'job':job}}),
                        _read_refinement_view=read)
    RefinementMixin._refinement_plot_changed(app)
    assert s.rf_parameters==[{'label':'new publication owns these rows'}]


@pytest.mark.parametrize('approval',[{}, {'approved_plan_sha256':'f'*64}])
def test_unapproved_recipe_cannot_supply_execution_parameters(approval):
    from radar_pd_nova.refinement_controller import RefinementMixin
    from test_refinement_plots import view
    s=SimpleNamespace(**RefinementMixin._refinement_state_defaults())
    s.rf_has_output=True;s.rf_selected_job='job';s.rf_experiment='exp';s.rf_stage='stage1';s.rf_histogram='bank2'
    v,r,e=fixture();data=view();data['source_sha256']=FIRST
    for p in data['parameters']:p['status']='refined'
    r.update(baseline_view='base',execution='polished.execution.json')
    def read(job,name):
        assert name!='polished.execution.json'
        return data
    job=dict(source={'experiment':'exp','source_sha256':SOURCE},result=r,recipe_id='recipe')
    workspace=SimpleNamespace(state={'jobs':{'job':job},'recipes':{'recipe':{'plan':{'stages':[]},'approval':approval}}})
    app=SimpleNamespace(server=SimpleNamespace(state=s),_refinement_workspace=workspace,_read_refinement_view=read)
    RefinementMixin._refinement_plot_changed(app)
    assert not s.rf_error
    assert s.rf_parameters and all(not p.get('native_variable') for p in s.rf_parameters)
    assert 'no matching approval' in s.rf_parameter_notice
