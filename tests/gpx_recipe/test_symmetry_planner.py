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
    assert set(r['intent']['settings'])==set(intent['settings'])
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
