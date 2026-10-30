import copy
import json

import pytest

from gpx_recipe.symmetry.chemistry import sites_from_inputs
from gpx_recipe.symmetry.planner import make_intent, payload, validate_intent


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
