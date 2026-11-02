import copy
import json

import pytest

from gpx_recipe.symmetry.chemistry import sites_from_inputs
from radar_pd_nova.model_planner import make_intent, payload, validate_intent, planning_failure_receipt


@pytest.fixture
def source():
    return dict(sha256='a'*64, phases=[dict(name='Parent', type='nuclear', space_group='C m m a',
        cell=[5., 5., 8., 90., 90., 90.], operations=[], histograms=['PWDR scan'],
        atoms=[dict(label='Fe1', type='Fe', occupancy=1.), dict(label='Pr1', type='Pr', occupancy=.5)])],
        histograms=[dict(name='PWDR scan', limits=[[0, 100], [1, 90]],
                         instrument={'Type': ['PNT', 'PNT', False]})])


@pytest.fixture
def intent():
    return dict(schema='radar-symmetry-intent/v1', source_sha256='a'*64, request='request',
        assumptions=[], questions=[], unsupported=[], settings=dict(kind='magnetic', parent='Parent',
        histogram='PWDR scan', k=['1', '0', '1'], sites=[dict(label='Fe1', form_factor='Fe+2', lande_g=2.)],
        directions=['x','y','z'], maximum_moment=4., magnetic_d_min=0., cycles=20, adequate_rwp=15.,relative_sign_starts=True))


def test_integer_nonzero_k_and_per_site_ionic_g_are_not_replaced(source, intent):
    intent['settings']['sites'].append(dict(label='Pr1', form_factor='Pr+3', lande_g=.8))
    p=validate_intent(intent,source,request='Original user text',declared={'k':['1','0','1']})
    assert p['settings']['k']==['1','0','1'] and p['request']=='Original user text'
    assert p['settings']['sites'][1]['lande_g']==.8


def test_unknown_k_prepares_reviewable_sites_but_never_invents_a_propagation_vector(source,intent):
    intent['settings'].update(k=None,moment_model='symmetry_allowed')
    result=validate_intent(intent,source,request='Find k from the data; allow canted moments')
    assert result['settings']['k'] is None
    assert result['settings']['moment_model']=='symmetry_allowed'
    assert result['settings']['sites']==intent['settings']['sites']
    with pytest.raises(ValueError,match='declared k'):
        validate_intent(intent,source,request='Keep my reviewed k',declared={'k':['0','0','0']})


def test_all_scan_replay_has_validated_enrollment_separate_from_native_refinement(source,intent):
    intent['enrollment']=dict(existing='sample',maximum_scans=57,follow_new=False)
    c=Client([intent])
    result=make_intent('Refine all 57 published matching scans; historical replay only',source,client=c)
    assert result['intent']['enrollment']==dict(existing='sample',maximum_scans=57,follow_new=False)
    assert 'enrollment' in c.calls[0][1]['required']
    assert 'enrollment' not in result['intent']['settings']
    assert any('Encode a requested scan scope' in rule for rule in payload('all scans',source)['rules'])


def test_explicit_multi_scan_cap_is_shown_to_the_model_as_structured_requirements(source,intent):
    request='Use all four existing scans 1, 2, 3 and 4. Use all four matching nuclear GPXs, maximum 4 acquisitions, and no future scans.'
    requirements=payload(request,source)['enrollment_requirements']
    assert requirements==dict(existing='sample',minimum_scans=4,maximum_scans=4,follow_new=False)
    intent['enrollment']=dict(existing='sample',maximum_scans=4,follow_new=False)
    assert validate_intent(intent,source,request=request)['enrollment']==intent['enrollment']
    intent['enrollment']['maximum_scans']=5
    with pytest.raises(ValueError,match='explicitly requested acquisition cap'):
        validate_intent(intent,source,request=request)


def test_multi_scan_repair_gives_scope_values_but_does_not_correct_a_bad_reply(source,intent):
    bad=copy.deepcopy(intent);bad['enrollment']=dict(existing='selected',maximum_scans=1,follow_new=False)
    good=copy.deepcopy(intent);good['enrollment']=dict(existing='sample',maximum_scans=4,follow_new=False)
    client=Client([bad,good])
    request='Use all four existing scans. Set a maximum of 4 acquisitions; no future scans.'
    result=make_intent(request,source,client=client)
    assert len(result['calls'])==2 and result['intent']['enrollment']==good['enrollment']
    repair=client.calls[1][0][-1]['content']
    assert '"existing": "sample"' in repair and '"maximum_scans": 4' in repair
    assert 'representative' in repair
    with pytest.raises(ValueError,match='multiple existing scans'):
        make_intent(request,source,client=Client([bad,bad]))


@pytest.mark.parametrize('user_request',[
    'Use all four existing scans. Refine only this scan.',
    'Use all four existing scans, maximum 3 acquisitions.',
    'Use all scans, maximum 4 acquisitions. Limit to 5 acquisitions.',
    'Use all scans, maximum 2001 acquisitions.',
    'Use all scans, maximum -1 acquisitions.',
    'Use all scans, maximum 1.5 acquisitions.',
    'Refine only this scan, maximum 4 acquisitions.',
])
def test_contradictory_or_invalid_caps_fail_before_a_model_call(source,user_request):
    with pytest.raises(ValueError):payload(user_request,source)


def test_negated_scope_cap_is_not_promoted_to_a_positive_requirement(source):
    request='Do not use all scans with a maximum of 4 acquisitions. Refine only this scan.'
    assert payload(request,source)['enrollment_requirements']=={'existing':'selected'}
    assert 'enrollment_requirements' not in payload('Refine the magnetic moments.',source)


@pytest.mark.parametrize('user_request',[
    'Use all four existing scans, maximum 4 acquisitions, and do not follow future scans.',
    'Do not follow future scans, but use all four existing scans with a maximum of 4 acquisitions.',
])
def test_no_future_negation_does_not_remove_a_positive_cap(source,intent,user_request):
    assert payload(user_request,source)['enrollment_requirements']['maximum_scans']==4
    intent['enrollment']=dict(existing='sample',maximum_scans=2000,follow_new=False)
    with pytest.raises(ValueError,match='acquisition cap'):
        validate_intent(intent,source,request=user_request)


@pytest.mark.parametrize('user_request',[
    'Use all existing scans, not limited to 4 acquisitions.',
    'Use all existing scans; no maximum of 4 acquisitions.',
])
def test_negated_cap_is_not_enforced(source,user_request):
    assert 'maximum_scans' not in payload(user_request,source)['enrollment_requirements']


@pytest.mark.parametrize('user_request',[
    'Use all existing scans, but do not exceed a maximum of 4 acquisitions.',
    'Use all existing scans; do not exceed 4 acquisitions.',
    'Use all existing scans; no more than 4 acquisitions.',
])
def test_positive_upper_bound_phrased_with_negation(source,user_request):
    assert payload(user_request,source)['enrollment_requirements']['maximum_scans']==4


@pytest.mark.parametrize('user_request',[
    'Use only these three existing scans and do not follow future scans.',
    'Refine all 57 published matching scans; historical replay only',
    'Process all existing scans',
])
def test_clear_multiple_scan_request_cannot_accept_selected_only_enrollment(source,intent,user_request):
    intent['enrollment']=dict(existing='selected',maximum_scans=1,follow_new=False)
    with pytest.raises(ValueError,match='multiple existing scans'):
        validate_intent(intent,source,request=user_request)


def test_scope_contradiction_is_repaired_before_any_settings_are_accepted(source,intent):
    bad=copy.deepcopy(intent);bad['enrollment']=dict(existing='selected',maximum_scans=1,follow_new=False)
    good=copy.deepcopy(intent);good['enrollment']=dict(existing='sample',maximum_scans=3,follow_new=False)
    client=Client([bad,good])
    result=make_intent('Use only these three existing scans and do not follow future scans.',source,client=client)
    assert len(result['calls'])==2 and result['intent']['enrollment']==good['enrollment']
    assert 'multiple existing scans' in client.calls[1][0][-1]['content']


@pytest.mark.parametrize('enrollment,message',[
    (None,'explicitly requested scan scope'),
    (dict(existing='sample',maximum_scans=2,follow_new=False),'maximum_scans is smaller'),
    (dict(existing='sample',maximum_scans=3,follow_new=True),'excludes future scans'),
])
def test_explicit_count_and_no_future_request_remain_binding(source,intent,enrollment,message):
    intent['enrollment']=enrollment
    with pytest.raises(ValueError,match=message):
        validate_intent(intent,source,request='Use only these three existing scans and do not follow future scans.')


def test_selected_only_request_cannot_expand_to_whole_sample(source,intent):
    intent['enrollment']=dict(existing='sample',maximum_scans=3,follow_new=False)
    with pytest.raises(ValueError,match='only the selected scan'):
        validate_intent(intent,source,request='Refine only this scan')


def test_negated_multi_scan_instruction_does_not_become_a_positive_scope(source,intent):
    intent['enrollment']=dict(existing='selected',maximum_scans=1,follow_new=False)
    result=validate_intent(intent,source,request='Do not refine all existing scans. Refine only this scan.')
    assert result['enrollment']==intent['enrollment']


def test_k_survey_cannot_silently_reduce_later_refinement_to_one_cycle(source,intent):
    request='Find k from the data, then screen and refine all magnetic subgroups across the series'
    bad=copy.deepcopy(intent);bad['settings'].update(k=None,cycles=1)
    good=copy.deepcopy(bad);good['settings']['cycles']=20
    client=Client([bad,good]);result=make_intent(request,source,client=client)
    assert result['intent']['settings']['cycles']==20 and len(result['calls'])==2
    data=json.loads(client.calls[0][0][1]['content'])
    assert data['declared']['cycles']==20
    assert 'declared cycles' in client.calls[1][0][-1]['content']


@pytest.mark.parametrize('cycles',[1,30,100])
def test_an_explicit_cycle_budget_is_preserved_and_displayed_as_declared(source,intent,cycles):
    request=f'Use {cycles} refinement cycles per start'
    intent['settings']['cycles']=cycles
    assert payload(request,source)['declared']['cycles']==cycles
    assert validate_intent(intent,source,request=request)['settings']['cycles']==cycles


def test_declared_cycle_budget_is_binding_and_conflicting_literals_need_review(source,intent):
    with pytest.raises(ValueError,match='declared cycles'):
        validate_intent(intent,source,request='Use the reviewed budget',declared={'cycles':30})
    with pytest.raises(ValueError,match='several cycle budgets'):
        payload('Use 1 cycle, then 30 cycles',source)


@pytest.mark.parametrize('literal,cycles',[('one',1),('single',1),('twenty-five',25),('one hundred',100)])
def test_written_cycle_limits_do_not_get_replaced_by_the_default(source,intent,literal,cycles):
    request=f'Use {literal} refinement cycles per start'
    intent['settings']['cycles']=cycles
    assert validate_intent(intent,source,request=request)['settings']['cycles']==cycles


@pytest.mark.parametrize('literal',['-1','0','1.5','101'])
def test_invalid_requested_cycle_limit_is_not_silently_replaced(source,literal):
    with pytest.raises(ValueError):payload(f'Use {literal} refinement cycles per start',source)


@pytest.mark.parametrize('update',[
    dict(existing='all_histories'),dict(maximum_scans=0),dict(maximum_scans=2001),
    dict(maximum_scans=True),dict(maximum_scans=2.5),dict(follow_new='yes'),
    dict(existing='selected',maximum_scans=57),dict(python='submit_jobs()')])
def test_enrollment_cannot_expand_to_unbounded_or_unrelated_inputs(source,intent,update):
    intent['enrollment']=dict(existing='sample',maximum_scans=57,follow_new=False,**{})
    intent['enrollment'].update(update)
    with pytest.raises((ValueError,TypeError)):
        validate_intent(intent,source,request='all scans')


def test_legacy_scope_stays_unchanged_and_blocked_intent_cannot_apply_enrollment(source,intent):
    assert validate_intent(intent,source,request='request')['enrollment'] is None
    intent.update(settings=None,questions=['Which bank?'],enrollment=dict(existing='sample',maximum_scans=2000,follow_new=True))
    with pytest.raises(ValueError,match='Unresolved'):
        validate_intent(intent,source,request='request')


def test_an_invalid_geometry_or_changed_declared_geometry_is_rejected(source,intent):
    intent['settings']['moment_model']='arbitrary_python'
    with pytest.raises(ValueError):validate_intent(intent,source,request='request')
    intent['settings']['moment_model']='symmetry_allowed'
    with pytest.raises(ValueError,match='declared moment_model'):
        validate_intent(intent,source,request='request',declared={'moment_model':'collinear'})


@pytest.mark.parametrize('field,value',[
    ('parent','Another parent'),('histogram','Another bank'),('k',[1.,0.,1.]),
    ('k',['0.33333','0','0']),('k',[['0','0','0'],['1','0','1']]),
    ('directions',['x','x']),('directions',['continuous']),('cycles',True),
    ('cycles',101),('maximum_moment',float('nan')),('adequate_rwp',0),('magnetic_d_min',-1)])
def test_incompatible_or_unbounded_choices_are_rejected(source,intent,field,value):
    intent['settings'][field]=value
    with pytest.raises((ValueError,TypeError)):validate_intent(intent,source,request='test')


@pytest.mark.parametrize('field,value,requirement',[
    ('cycles',1.5,'an integer, at least 1 and at most 100'),
    ('cycles',101,'an integer, at least 1 and at most 100'),
    ('adequate_rwp',0,'a finite number, greater than 0 and at most 100'),
    ('adequate_rwp',None,'a finite number, greater than 0 and at most 100'),
    ('maximum_moment',float('inf'),'a finite number, greater than 0 and at most 20'),
    ('magnetic_d_min',-1,'a finite number, at least 0 and at most 10'),
])
def test_numeric_failure_names_exact_setting_and_keeps_native_bounds(source,intent,field,value,requirement):
    intent['settings'][field]=value
    with pytest.raises(ValueError) as exc:
        validate_intent(intent,source,request='test')
    assert f'settings.{field}' in str(exc.value) and requirement in str(exc.value)


def test_site_and_enrollment_numeric_failures_identify_their_paths(source,intent):
    intent['settings']['sites'][0]['lande_g']=0
    with pytest.raises(ValueError,match=r'settings.sites\[0\].lande_g'):
        validate_intent(intent,source,request='test')
    intent['settings']['sites'][0]['lande_g']=2
    intent['enrollment']=dict(existing='sample',maximum_scans=True,follow_new=False)
    with pytest.raises(ValueError,match=r'enrollment.maximum_scans'):
        validate_intent(intent,source,request='test')


def test_bounded_repair_receives_numeric_field_and_can_ask_instead_of_defaulting(source,intent):
    bad=copy.deepcopy(intent);bad['settings']['adequate_rwp']=0
    clarified=copy.deepcopy(intent)
    clarified.update(settings=None,questions=['Which adequate Rwp criterion should be used?'])
    client=Client([bad,clarified]);result=make_intent('Survey k; I have not chosen a fit criterion',source,client=client)
    assert result['status']=='needs_input' and result['intent']['settings'] is None
    assert 'settings.adequate_rwp' in client.calls[1][0][-1]['content']
    assert len(result['calls'])==2


def test_failed_planning_receipt_is_bounded_and_omits_transport_or_credentials():
    error=RuntimeError('HTTPS error containing private transport content')
    error.planning_calls=[dict(model='fake',response='x'*70000,usage={'input_tokens':4},
        api_key='never-save',headers={'Authorization':'never-save'},request_dump='never-save')]*3
    receipt=planning_failure_receipt(error,request='a'*13000,source_sha256='a'*64)
    assert len(receipt['request'])==12000 and len(receipt['calls'])==2
    assert receipt['validation_error'] is None and receipt['settings_accepted'] is False
    assert len(receipt['calls'][0]['response'])==65536 and receipt['calls'][0]['response_truncated']
    assert 'never-save' not in json.dumps(receipt) and 'private transport' not in json.dumps(receipt)


def test_wrong_element_ambiguous_site_and_unsupported_code_never_compile(source,intent):
    intent['settings']['sites'][0]['form_factor']='Cr+3'
    with pytest.raises(ValueError,match='chemical element'):validate_intent(intent,source,request='test')
    intent['settings']['sites'][0]['form_factor']='Fe+2'
    source['phases'][0]['atoms'].append(copy.deepcopy(source['phases'][0]['atoms'][0]))
    with pytest.raises(ValueError,match='ambiguous'):validate_intent(intent,source,request='test')
    intent['python']='print(1)'
    with pytest.raises(ValueError,match='Unknown fields'):validate_intent(intent,source,request='test')


def test_ion_and_vector_cannot_silently_change_declared_choices(source,intent):
    with pytest.raises(ValueError,match='declared k'):
        validate_intent(intent,source,request='test',declared={'k':['0','0','0']})
    with pytest.raises(ValueError,match='declared sites'):
        validate_intent(intent,source,request='test',declared={'sites':[dict(label='Fe1',form_factor='Fe+3',lande_g=2.)]})


def test_blocked_request_cannot_execute_partial_settings(source,intent):
    intent['unsupported']=['Noncollinear refinement needs a separate executor']
    with pytest.raises(ValueError,match='Unresolved'):validate_intent(intent,source,request='test')
    intent['settings']=None
    assert validate_intent(intent,source,request='test')['settings'] is None


def test_changed_source_or_xray_project_cannot_be_used(source,intent):
    intent['source_sha256']='b'*64
    with pytest.raises(ValueError,match='different source'):validate_intent(intent,source,request='test')
    intent['source_sha256']='a'*64;source['histograms'][0]['instrument']['Type'][0]='PXC'
    with pytest.raises(ValueError,match='neutron'):validate_intent(intent,source,request='test')


class Client:
    def __init__(self,plans):self.plans=iter(plans);self.calls=[]
    def complete_structured(self,messages,schema,name):
        self.calls.append((copy.deepcopy(messages),schema,name))
        return json.dumps(next(self.plans)),dict(model='fake',finish_reason='stop')


def test_single_setup_call_produces_settings_without_subgroups_or_python(source,intent):
    c=Client([intent]);r=make_intent('Use k=(1,0,1), Fe1=Fe+2, g=2.',source,client=c)
    assert r['status']=='ready_for_discovery' and len(r['calls'])==1
    assert c.calls[0][2]=='radar_symmetry_intent'
    assert set(r['intent']['settings'])==set(intent['settings'])|{'moment_model','refine_histogram_scale','refine_background'}
    assert r['intent']['settings']['moment_model']=='collinear'
    assert r['intent']['settings']['refine_histogram_scale'] is False
    assert r['intent']['settings']['refine_background'] is False
    assert 'runtime' not in payload('test',source)['project']


def test_one_bounded_repair_and_no_third_call(source,intent):
    bad=copy.deepcopy(intent);bad['settings']['parent']='hallucinated'
    c=Client([bad,intent]);r=make_intent('test',source,client=c)
    assert len(r['calls'])==2
    c=Client([bad,bad,intent])
    with pytest.raises(ValueError,match='bounded validation') as exc:make_intent('test',source,client=c)
    assert len(c.calls)==2 and len(exc.value.planning_calls)==2


def test_refusal_stays_blocked_without_unnecessary_model_repair(source,intent):
    intent.update(questions=['Which exact propagation vector and magnetic ion?'],settings=None)
    c=Client([intent]);r=make_intent('Find magnetic order',source,client=c)
    assert r['status']=='needs_input' and len(c.calls)==1


def test_paragraphs_in_clarification_are_prose_and_do_not_discard_a_safe_block(source,intent):
    intent.update(questions=['Choose an adequate Rwp criterion.\n\tThe current reduction is unqualified.'],settings=None)
    c=Client([intent]);r=make_intent('Survey this series; review the fit criterion later',source,client=c)
    assert r['status']=='needs_input' and len(c.calls)==1
    assert r['intent']['questions']==['Choose an adequate Rwp criterion. The current reduction is unqualified.']
    assert r['intent']['settings'] is None


@pytest.mark.parametrize('note',['', '   ', 'bad\x00text', 'x'*2001, None])
def test_invalid_explanation_names_the_field_without_accepting_settings(source,intent,note):
    intent.update(questions=[note],settings=None)
    with pytest.raises(ValueError,match=r'questions\[0\]'):
        validate_intent(intent,source,request='test')


def test_ionic_notation_error_is_specific_enough_for_one_bounded_repair(source,intent):
    intent['settings']['sites'][0]['form_factor']='Fe+3'
    bad=copy.deepcopy(intent);bad['settings']['sites'][0]['form_factor']='Fe3+'
    c=Client([bad,intent]);r=make_intent('Use Fe3+, g=2',source,client=c)
    assert len(c.calls)==2 and r['intent']['settings']['sites'][0]['form_factor']=='Fe+3'
    repair=c.calls[1][0][-1]['content']
    assert 'ionic form factor for Fe1' in repair and 'Fe+3' in repair


def test_source_payload_does_not_export_arrays_or_paths(source):
    source['runtime']={'root':'/private/runtime'}
    source['histograms'][0].update(observed=[999],source='/private/pattern')
    data=json.dumps(payload('test',source))
    assert '999' not in data and '/private' not in data and 'observed' not in data


def test_multispecies_site_fields_keep_individual_lande_values():
    assert sites_from_inputs(['Fe1','Pr1'],'Fe1=Fe+2; Pr1=Pr+3','Fe1=2;Pr1=0.8')==[
        dict(label='Fe1',form_factor='Fe+2',lande_g=2.),dict(label='Pr1',form_factor='Pr+3',lande_g=.8)]
    assert sites_from_inputs(['Fe1'],'Fe1=Fe+3',2)[0]['lande_g']==2
    for forms,g in [('Fe1=Fe+2;Fe1=Fe+3',2),('Fe1=Fe+2;Pr1=Pr+3',2),('Fe1=Fe+2','nan'),('Fe1=Fe+2',True)]:
        with pytest.raises(ValueError):sites_from_inputs(['Fe1'],forms,g)


def test_common_scale_requires_a_boolean_and_preserves_reviewed_choices(source,intent):
    intent['settings']['refine_histogram_scale']=True
    result=validate_intent(intent,source,request='Refine a shared nuclear and magnetic scale',declared={'refine_histogram_scale':True})
    assert result['settings']['refine_histogram_scale'] is True
    with pytest.raises(ValueError,match='declared refine_histogram_scale'):
        validate_intent(intent,source,request='Keep all scales fixed',declared={'refine_histogram_scale':False})
    intent['settings']['refine_histogram_scale']='yes'
    with pytest.raises(ValueError):validate_intent(intent,source,request='request')


def test_existing_background_choice_cannot_override_a_reviewed_fixed_background(source,intent):
    intent['settings']['refine_background']=True
    result=validate_intent(intent,source,request='Refine the existing background coefficients',declared={'refine_background':True})
    assert result['settings']['refine_background'] is True
    with pytest.raises(ValueError,match='declared refine_background'):
        validate_intent(intent,source,request='Keep background fixed',declared={'refine_background':False})
    intent['settings']['refine_background']='true'
    with pytest.raises(ValueError):validate_intent(intent,source,request='request')
