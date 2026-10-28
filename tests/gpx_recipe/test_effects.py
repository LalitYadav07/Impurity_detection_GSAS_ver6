"""Actual native effects and fail-closed input/target checks."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from gpx_recipe.effects import parameter_changes,require_fixed_values_unchanged
from gpx_recipe.execution import check_preflight
from gpx_recipe import advanced


def row(name,value,group='occupancy',free=False,family='Afrac'):
    return dict(name=name,current_value=value,group=group,configured_now=free,effective_now=free,family=family,status='configured' if free else 'fixed',sigma=.01)


def state(*rows):return {'parameter_space':{'parameters':list(rows)}}


def test_real_parameter_changes_distinguish_fixed_and_released():
    before=state(row('0::Afrac:0',.5,free=True),row('0::Afrac:1',.5))
    after=state(row('0::Afrac:0',.6,free=True),row('0::Afrac:1',.5))
    d=parameter_changes(before,after);require_fixed_values_unchanged(d)
    assert d['released_parameters_changed']==1 and d['fixed_parameters_checked']==1
    assert d['released_parameters'][0]['after']==.6 and d['released_parameters'][0]['sigma']==.01


def test_zero_change_can_be_an_already_converged_refinement():
    s=state(row('0::Afrac:0',.5,free=True));d=parameter_changes(s,s)
    require_fixed_values_unchanged(d);assert d['released_parameters_changed']==0 and len(d['released_parameters'])==1


def test_any_unapproved_fixed_parameter_drift_is_detected():
    d=parameter_changes(state(row(':0:Zero',0,'instrument_zero',family='Zero')),state(row(':0:Zero',.002,'instrument_zero',family='Zero')))
    with pytest.raises(ValueError,match='held fixed'):require_fixed_values_unchanged(d)


def test_derived_weight_fraction_is_not_a_free_hap_scale():
    d=parameter_changes(state(row('0:0:WgtFrac',.3,'phase_fraction',family='WgtFrac')),state(row('0:0:WgtFrac',.4,'phase_fraction',family='WgtFrac')))
    require_fixed_values_unchanged(d);assert len(d['derived_changes'])==1 and not d['released_parameters']


def test_background_cannot_hide_an_inactive_requested_cell():
    context={'histograms':[{'key':'h0','native_id':0}], 'phases':[{'key':'p0','native_id':3}]}
    p=dict(name=':0:Back;0',group='background',phase_id=None,histogram_id=0,atom_id=None,configured_now=True,effective_now=True)
    s={'extraction':{'status':'complete'},'constraints':{},'parameter_space':{'effective_vary':[':0:Back;0'],'parameters':[p]}}
    stage={'free':[{'kind':'background','histogram':'h0'},{'kind':'cell','phase':'p0'}]}
    with pytest.raises(ValueError,match='no configured native'):check_preflight(s,stage,context)


def test_runtime_bound_is_rechecked_against_updated_held_occupancy():
    context={'phases':[{'key':'p0','atoms':[], 'occupancy_groups':[['A','B']]}]}
    phase=SimpleNamespace(atom=lambda name:SimpleNamespace(occupancy=.5 if name=='A' else .9))
    stage={'free':[{'kind':'atom','phase':'p0','atom':'A','flags':'F'}], 'bounds':[dict(phase='p0',atom='A',parameter='F',lower=0,upper=.5)]}
    with pytest.raises(ValueError,match='after preceding stages'):advanced.configure(None,stage,context,{'p0':phase},{},set())


def test_equivalent_scaled_normalization_equation_is_recognized():
    project=SimpleNamespace(data={'Constraints':{'data':{'HAP':[[[2.,'0:0:Scale'],[2.,'1:0:Scale'],2.,None,'c']]}}})
    assert advanced._scale_sum_exists(project,['0:0:Scale','1:0:Scale'])


def test_sequential_source_requires_explicit_supported_workflow(monkeypatch,tmp_path):
    import gpx_recipe.snapshot as module
    project=SimpleNamespace(data={'Controls':{'data':{'Seq Data':['PWDR T1','PWDR T2']}}})
    monkeypatch.setattr(module,'gsas',lambda:SimpleNamespace(G2Project=lambda **kw:project))
    with pytest.raises(ValueError,match='Sequential-refinement controls'):module.snapshot(tmp_path/'sequence.gpx')


def test_long_multiphase_joint_recipe_receives_room_for_complete_active_sets():
    from gpx_recipe.planner import output_allowance
    project={'phases':[{'histograms':['h0'],'atoms':[{'label':'O1'}]} for _ in range(8)],'histograms':[{'key':'h0'}]}
    payload={'project':project,'user_request':'Refine O1 in all phases over six stages.','explicit_stage_count':6,'previous_plan':None}
    assert output_allowance(payload)==8192
    payload['explicit_stage_count']=1;payload['project']['phases']=project['phases'][:1]
    assert output_allowance(payload)==2048


def test_equal_uiso_accepts_isotropic_atoms_and_rejects_anisotropic_or_single_free():
    from gpx_recipe.contracts import validate_plan
    context={'model_signature':'a','histograms':[{'key':'h0','name':'h'}], 'phases':[{'key':'p0','histograms':['h'],'atoms':[{'label':'O1','adp_type':'I'},{'label':'O2','adp_type':'I'}]}]}
    eq={'kind':'uiso_equal','atoms':[{'phase':'p0','atom':'O1'},{'phase':'p0','atom':'O2'}],'reason':'Explicit shared oxygen Uiso'}
    plan={'schema':'radar-gpx-recipe/v1','model_signature':'a','request':'share oxygen Uiso','assumptions':[],'questions':[],'unsupported':[],'stages':[{'id':'s1','title':'shared Uiso','cycles':4,'free':[{'kind':'atom','phase':'p0','atom':a,'flags':'U'} for a in ['O1','O2']],'constraints':[eq]}]}
    validate_plan(plan,context)
    plan['stages'][0]['free']=plan['stages'][0]['free'][:1]
    with pytest.raises(ValueError,match='only one released'):validate_plan(plan,context)
    plan['stages'][0]['free']=[];context['phases'][0]['atoms'][1]['adp_type']='A'
    plan['stages'][0]['free']=[{'kind':'background','histogram':'h0'}]
    with pytest.raises(ValueError,match='exact isotropic'):validate_plan(plan,context)


def test_equivalence_chain_is_idempotent_but_different_multipliers_are_not_equal():
    p=SimpleNamespace(data={'Constraints':{'data':{'Phase':[[[1.,'0::AUiso:0'],[1.,'0::AUiso:1'],None,None,'e'],[[1.,'0::AUiso:1'],[1.,'0::AUiso:2'],None,None,'e']]}}})
    assert advanced.has_equality(p,['0::AUiso:0','0::AUiso:2'])
    p.data['Constraints']['data']['Phase'][1][1][0]=2
    assert not advanced.has_equality(p,['0::AUiso:0','0::AUiso:2'])


def test_constraint_identity_is_stable_when_native_phase_histogram_atom_ids_change():
    from gpx_recipe.identity import constraint_binding
    def project(p,h,a):return ([{'key':'p0','native_id':p,'atoms':[{'label':'O1','native_id':a}]}],[{'key':'h0','native_id':h}])
    rows=lambda name:[{'kind':'h','section':'Phase','terms':[{'variable':name,'coefficient':1.}],'value':None,'definition':None}]
    assert constraint_binding(rows('3:9:Afrac:7'),*project(3,9,7))==constraint_binding(rows('0:2:Afrac:0'),*project(0,2,0))


def test_preserved_uiso_group_cannot_be_partially_released():
    from gpx_recipe.identity import validate_joint_uiso_release
    groups=[[{'phase':'p0','atom':'C1'},{'phase':'p0','atom':'O1'}]]
    s={'free':[{'kind':'atom','phase':'p0','atom':'O1','flags':'U'}]}
    with pytest.raises(ValueError,match='tied by a preserved equality'):validate_joint_uiso_release(s,groups)
    s['free'].append({'kind':'atom','phase':'p0','atom':'C1','flags':'U'});validate_joint_uiso_release(s,groups)
    validate_joint_uiso_release({'free':[]},groups)


def test_user_named_constraint_variable_needs_actual_approved_native_provenance():
    from gpx_recipe.execution import resolve_constraint_provenance
    row={'name':'0::AUiso:1','refinement_variable':'0::AUiso:1','group':'atomic_displacement','configured_now':True}
    s={'parameters':[row],'effective_vary':['::oxygen_U']}
    resolved=resolve_constraint_provenance(s,['::oxygen_U'],['0::AUiso:1'],lambda var,vary:(['::oxygen_U'],[1.]),{'atomic_displacement'},lambda row:True)
    assert resolved['::oxygen_U']==[{'variable':'0::AUiso:1','group':'atomic_displacement'}]
    assert not resolve_constraint_provenance(s,['::oxygen_U'],['0::AUiso:1'],lambda var,vary:(['::oxygen_U'],[1.]),{'atomic_displacement'},lambda row:False)


@pytest.mark.skipif(__import__('os').name!='posix',reason='POSIX process-group cleanup')
def test_timeout_stops_own_scientific_descendant(tmp_path):
    import subprocess,time,json
    from gpx_recipe.process import run_scientific
    marker=tmp_path/'orphan.txt';ready=tmp_path/'ready.txt'
    child="import time;from pathlib import Path;time.sleep(.8);Path("+repr(str(marker))+").write_text('should never complete')"
    parent="import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c',"+repr(child)+"]);Path("+repr(str(ready))+").write_text(str(p.pid));time.sleep(10)"
    with pytest.raises(subprocess.TimeoutExpired):run_scientific([sys.executable,'-c',parent],timeout=.3,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    assert ready.exists();time.sleep(.85);assert not marker.exists()


def test_fresh_covariance_requires_dimensions_finite_sigmas_and_identifiability():
    import numpy as np
    from gpx_recipe.effects import require_native_covariance
    good={'covMatrix':np.eye(2),'sig':[1.,1.]};require_native_covariance(good,2,{'rank':2})
    for bad in [{'covMatrix':[],'sig':[]},{'covMatrix':[[1.,float('nan')],[0.,1.]],'sig':[1.,1.]},{'covMatrix':np.eye(2),'sig':[1.]}]:
        with pytest.raises(ValueError,match='covariance'):require_native_covariance(bad,2,{'rank':2})
    with pytest.raises(ValueError,match='rank deficient'):require_native_covariance(good,2,{'rank':1})


def test_new_direct_cell_release_overrides_a_previous_strain_shift_alias():
    from gpx_recipe.studio_state import current_refinement_variable
    aliases={'0::A0':'0::D11','0::Ax:2':'0::dAx:2'}
    assert current_refinement_variable('0::A0',aliases,{'0::A0'})=='0::A0'
    assert current_refinement_variable('0::Ax:2',aliases,{'0::dAx:2'})=='0::dAx:2'


def test_reset_freezes_background_peak_debye_and_imported_flags_without_values_or_support_loss():
    from gpx_recipe.execution import freeze
    background=[['chebyschev-1',True,2,10.,20.],{'peaksList':[[5.,True,100.,True,2.,True,.1,False]],'debyeTerms':[[3.,True,4.,True,5.,True]],'background PWDR':['PWDR blank',1.2,True],'FixedPoints':[[1.,8.],[4.,3.]]}]
    class Histogram:
        SampleParameters={};InstrumentParameters={}
        data={'Background':background}
        def clear_refinements(self,value):
            assert value=={'Background':{'peaks':True}}
            background[0][1]=False
            for peak in background[1]['peaksList']:
                for i in range(1,len(peak),2):peak[i]=False
    freeze(SimpleNamespace(histograms=lambda:[Histogram()],phases=lambda:[]))
    assert background[0]==['chebyschev-1',False,2,10.,20.]
    assert background[1]['debyeTerms']==[[3.,False,4.,False,5.,False]]
    assert background[1]['background PWDR']==['PWDR blank',1.2,False]
    assert background[1]['FixedPoints']==[[1.,8.],[4.,3.]]


def test_covariance_rejects_indefinite_asymmetric_and_inconsistent_uncertainties():
    import numpy as np
    from gpx_recipe.effects import require_native_covariance
    for matrix,sigmas in [([[1.,2.],[2.,1.]],[1.,1.]),([[1.,.5],[.1,1.]],[1.,1.]),(np.eye(2),[2.,1.]),([[0.,0.],[0.,1.]],[0.,1.])]:
        with pytest.raises(ValueError,match='covariance'):require_native_covariance({'covMatrix':matrix,'sig':sigmas},2,{'rank':2})


def test_source_bound_identity_survives_id_remapping_but_preserves_changed_limits():
    from gpx_recipe.identity import bound_binding
    def bind(phase,hist,atom,value):
        return bound_binding({'parmMinDict':{f'{phase}:{hist}:Afrac:{atom}':value}},[{'key':'p0','native_id':phase,'atoms':[{'label':'O1','native_id':atom}]}],[{'key':'h0','native_id':hist}],lambda v:v)
    assert bind(2,7,4,.1)==bind(0,0,0,.1)
    assert bind(2,7,4,.1)!=bind(2,7,4,.2)


def test_ten_histogram_long_recipe_is_not_trapped_at_truncated_8192_budget():
    from gpx_recipe.planner import output_allowance
    project={'phases':[{'histograms':['h'+str(i) for i in range(10)],'atoms':[]} for _ in range(2)],'histograms':[{'key':'h'+str(i)} for i in range(10)]}
    assert output_allowance({'project':project,'user_request':'six stages','explicit_stage_count':6,'previous_plan':None})==16384


def test_inactive_constraint_definition_cannot_mask_unsatisfied_held_values():
    values={'A':.3,'B':.4};phase=SimpleNamespace(atom=lambda name:SimpleNamespace(occupancy=values[name],ADP=values[name]))
    project=SimpleNamespace(phase=lambda name:phase);context={'phases':[{'key':'p0','name':'phase'}]}
    eq={'kind':'occupancy_sum','terms':[{'phase':'p0','atom':a,'coefficient':1.} for a in ['A','B']],'total':1.,'reason':'explicit total'}
    with pytest.raises(ValueError,match='held values'):advanced.check_requested_relations(project,context,[eq],free=[{'kind':'background','histogram':'h0'}])
    # A preview may not enforce a not-yet-refined pair's numerical values.
    advanced.check_requested_relations(project,context,[eq],free=[{'kind':'atom','phase':'p0','atom':'A','flags':'F'}])
    with pytest.raises(ValueError,match='actual refined values'):advanced.check_requested_relations(project,context,[eq])
    values['B']=.7;advanced.check_requested_relations(project,context,[eq])
    u={'kind':'uiso_equal','atoms':[{'phase':'p0','atom':a} for a in ['A','B']],'reason':'shared'}
    with pytest.raises(ValueError,match='held values'):advanced.check_requested_relations(project,context,[u],free=[])
    values['B']=.3;advanced.check_requested_relations(project,context,[u])


def test_solver_dependent_negative_scale_cannot_hide_behind_native_positive_clipping():
    from gpx_recipe.effects import constraint_bound_audit,require_constraint_bounds
    p=row('0:0:Scale',.5,'phase_fraction',True,'Scale');p.update(lower_bound={'value':0.},upper_bound={'value':1.})
    cov={'varyList':['::constr0'],'variables':[-.0005]}
    audit=constraint_bound_audit(state(p),cov,lambda values:values.update({'0:0:Scale':values['::constr0']}))
    assert audit['violations'][0]['solver_value_before_native_clipping']==-.0005
    with pytest.raises(ValueError,match='before save-time clipping'):require_constraint_bounds(audit)
    audit=constraint_bound_audit(state(p),{'varyList':['::constr0'],'variables':[.4]},lambda v:v.update({'0:0:Scale':v['::constr0']}));require_constraint_bounds(audit)


def test_saved_normalization_must_hold_numerically_not_only_exist_as_an_equation():
    context={'histograms':[{'key':'h0','name':'PWDR sample'}]}
    def project(values):return SimpleNamespace(phases=lambda:[SimpleNamespace(data={'Histograms':{'PWDR sample':{'Use':True,'Scale':[v,True]}}}) for v in values])
    advanced.check_normalization(project([.4,.6]),context,{'h0'})
    with pytest.raises(ValueError,match='normalization drifted'):advanced.check_normalization(project([.4,.6005]),context,{'h0'})


def test_unrelated_frozen_history_is_retained_without_blocking_new_parameters():
    from gpx_recipe.effects import blocking_frozen_variables
    before={'parameter_space':{'frozen_variables':[':0:Zero'],'configured_vary':[':0:Back;0'],'effective_vary':[':0:Back;0']}}
    assert blocking_frozen_variables(before,{'parameter_space':{'frozen_variables':[':0:Zero']}})==[]
    assert blocking_frozen_variables(before,{'parameter_space':{'frozen_variables':[':0:Zero',':0:Back;0']}})==[':0:Back;0']
    before['parameter_space']['configured_vary'].append(':0:Zero')
    assert blocking_frozen_variables(before,{'parameter_space':{'frozen_variables':[':0:Zero']}})==[':0:Zero']


def test_requested_already_frozen_parameter_requires_explicit_source_review():
    s={'extraction':{'status':'complete'},'constraints':{},'parameter_space':{'frozen_variables':['0::AUiso:0'],'configured_vary':['0::AUiso:0'],'effective_vary':['0::AUiso:0']}}
    with pytest.raises(ValueError,match='already frozen by preserved source'):check_preflight(s,{'free':[]},{})


def test_polynomial_background_does_not_authorize_peak_or_debye_parameters():
    from gpx_recipe.execution import parameter_matches_target
    context={'phases':[],'histograms':[{'key':'h0','native_id':0}]};target={'kind':'background','histogram':'h0'}
    for family,expected in [('Back;0',True),('BkPkpos;0',False),('DebyeA;0',False)]:
        p={'name':':0:'+family,'family':family,'phase_id':None,'histogram_id':0,'group':'background'}
        assert parameter_matches_target(p,target,context)==expected


def test_phase_global_texture_sample_angles_keep_values_while_flags_are_frozen():
    from gpx_recipe.execution import freeze
    texture={'SH Coeff':[True,{'C(4,0,1)':-.4}],'Sample omega':[True,17.5],'Sample chi':[False,-3.],'Sample phi':[True,26.]}
    phase=SimpleNamespace(data={'General':{'SH Texture':texture},'Histograms':{}},clear_refinements=lambda x:None,set_refinements=lambda x:None,clear_HAP_refinements=lambda x:None)
    freeze(SimpleNamespace(histograms=lambda:[],phases=lambda:[phase]))
    assert texture['SH Coeff']==[False,{'C(4,0,1)':-.4}]
    assert texture['Sample omega']==[False,17.5] and texture['Sample chi']==[False,-3.] and texture['Sample phi']==[False,26.]


def test_model_topology_changes_require_review_but_refined_texture_values_can_change():
    from gpx_recipe.identity import model_topology
    import copy
    p={'global_texture':{'Order':4,'Model':'cylindrical','SH Coeff':[True,{'C(4,0,1)':.2}]},'histogram_phase_models':{'h0':{'Pref.Ori.':['MD',1.2,True,[0,0,1],0,{}],'Size':['isotropic',[1.,1.,1.],[True,False,False],[0,0,1]]}}}
    q=copy.deepcopy(p);q['global_texture']['SH Coeff'][1]['C(4,0,1)']=.8;q['histogram_phase_models']['h0']['Pref.Ori.'][1]=1.5
    assert model_topology(p)==model_topology(q)
    q['histogram_phase_models']['h0']['Pref.Ori.'][3]=[1,0,0]
    assert model_topology(p)!=model_topology(q)


def test_native_numpy_scalars_are_never_lost_from_change_or_fixed_audits():
    import numpy as np
    from gpx_recipe.effects import numeric
    assert numeric(np.float64(.3)) and numeric(np.float32(.4)) and numeric(np.int64(2))
    assert not numeric(True) and not numeric(np.bool_(True)) and not numeric(np.array([1.])) and not numeric(np.float64('nan'))
    before=state(row('0:0:Scale',.3,'phase_fraction',True,'Scale'),row(':0:Zero',np.float64(0),'instrument_zero',False,'Zero'))
    after=state(row('0:0:Scale',np.float64(.4),'phase_fraction',True,'Scale'),row(':0:Zero',np.float64(.01),'instrument_zero',False,'Zero'))
    ledger=parameter_changes(before,after)
    assert len(ledger['released_parameters'])==1 and ledger['released_parameters_changed']==1
    assert ledger['unexpected_fixed_changes'][0]['variable']==':0:Zero'
    with pytest.raises(ValueError,match='held fixed'):require_fixed_values_unchanged(ledger)


@pytest.mark.parametrize('free',[True,False])
def test_missing_or_nonfinite_requested_values_cannot_claim_a_complete_audit(free):
    before=state(row('0:0:Scale',.3,'phase_fraction',free,'Scale'))
    for after in [state(),state(row('0:0:Scale',None,'phase_fraction',free,'Scale'))]:
        ledger=parameter_changes(before,after)
        with pytest.raises(ValueError,match='Could not audit every'):require_fixed_values_unchanged(ledger)


def test_runtime_version_is_not_the_project_historical_save_label(monkeypatch):
    from gpx_recipe import snapshot as module
    g2=SimpleNamespace(__name__='GSASII.GSASIIscriptable',__package__='GSASII')
    project=SimpleNamespace(data={'Controls':{'data':{'LastSavedUsing':'old-source-version'}}})
    monkeypatch.setattr(module.importlib,'import_module',lambda name:SimpleNamespace(GetVersionNumber=lambda:5843))
    info=module.runtime_info(g2,project)
    assert info['version']=='5843' and info['source_saved_using']=='old-source-version'


def test_generated_independent_names_missing_from_parameter_table_require_provenance(monkeypatch):
    import gpx_recipe.execution as module
    context={'histograms':[{'key':'h0','native_id':0}], 'phases':[{'key':'p0','native_id':0}]}
    native=dict(name='0:0:Scale',refinement_variable='0:0:Scale',group='phase_fraction',phase_id=0,histogram_id=0,atom_id=None,configured_now=True,effective_now=False)
    source={'extraction':{'status':'complete'},'constraints':{},'parameter_space':{'effective_vary':['::constr0'],'parameters':[native]}}
    stage={'free':[dict(kind='phase_scale',phase='p0',histogram='h0')]}
    maps=SimpleNamespace(GetDependentVars=lambda:['0:0:Scale'],getInvConstraintEq=lambda name,vary:(['::constr0'],[1.]))
    monkeypatch.setattr(module,'gsas',lambda:SimpleNamespace(__package__='fake'))
    monkeypatch.setattr(module.importlib,'import_module',lambda name:maps)
    proven=module.check_preflight(source,stage,context)
    assert proven['::constr0']==[{'variable':'0:0:Scale','group':'phase_fraction'}]
    maps.GetDependentVars=lambda:[]
    with pytest.raises(ValueError,match='outside the approved vocabulary'):module.check_preflight(source,stage,context)


def test_replaced_model_covariance_history_is_not_a_current_fixed_parameter():
    historical=row('0:0:MD',.8,'preferred_orientation',False,'MD');historical.update(source_group='derived',status='refined_independent')
    ledger=parameter_changes(state(historical),state())
    require_fixed_values_unchanged(ledger)
    assert ledger['historical_only_parameters']==['0:0:MD'] and ledger['fixed_parameters_checked']==0
    historical['source_group']='hap'
    with pytest.raises(ValueError,match='numeric fixed parameter'):require_fixed_values_unchanged(parameter_changes(state(historical),state()))


def test_fresh_atomic_coordinates_bind_current_shifts_without_prior_covariance():
    from gpx_recipe.studio_state import current_refinement_variable
    assert current_refinement_variable('0::Ax:0',{}, {'0::dAx:0'})=='0::dAx:0'
    assert current_refinement_variable('0::Ay:0',{}, {'0::dAy:0'})=='0::dAy:0'
    assert current_refinement_variable('0::Ax:0',{}, {'0::dAx:1'})=='0::Ax:0'
    assert current_refinement_variable('0::A0',{}, {'0::D11'})=='0::A0'
    assert current_refinement_variable('0::Ax:0',{'0::Ax:0':'old'}, {'0::dAx:0'})=='0::dAx:0'
