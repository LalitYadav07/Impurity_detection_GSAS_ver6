import copy
import json
from types import SimpleNamespace

import pytest

from gpx_recipe.contracts import validate_plan
from gpx_recipe.native_objective import apply, bind_request, check_request, validate_tolerance
from gpx_recipe.native_budget import iteration_caption
from gpx_recipe.planner import make_plan
from gpx_recipe.schema import normalize_optional_nulls


def context():
    return dict(model_signature='model', controls={'deriv type':'analytic Hessian','min dM/M':.001},
        histograms=[dict(key='h0',name='PWDR control')],
        phases=[dict(key='p0',name='control',histograms=['PWDR control'],atoms=[])])


def plan(tolerance=None):
    stage=dict(id='background',title='Background',cycles=12,
               free=[dict(kind='background',histogram='h0')],constraints=[],
               convergence={'max_shift_over_sigma':.5})
    if tolerance is not None: stage['relative_objective_tolerance']=tolerance
    return dict(schema='radar-gpx-recipe/v1',model_signature='model',request='Refine background.',
                assumptions=[],questions=[],unsupported=[],stages=[stage])


@pytest.mark.parametrize('value',[True,False,None,'1e-6',0,-1,1e-15,.1,float('nan'),float('inf')])
def test_invalid_tolerances_are_not_native_settings(value):
    with pytest.raises(ValueError,match='relative-objective'):
        validate_tolerance(value)


@pytest.mark.parametrize('text',[
    'Use native relative-objective tolerance 1e-6.',
    'Set relative_objective_tolerance=0.000001.',
    'Set solver ftol to 1e-6.',
    'Use min dM/M of 1e-6.',
    'Use native relative-objective\n tolerance 1e-6 in every stage.',
    'Use relative\nobjective stopping\n tolerance 1e-6.',
    'Use objective\nchange tolerance 1e-6.',
])
def test_explicit_native_tolerance_keeps_separate_shift_criterion(text):
    binding=bind_request(text,context())
    assert binding['global_tolerance']==1e-6 and not binding['questions']
    candidate=plan(1e-6)
    validate_plan(candidate,context());check_request(candidate,binding)
    assert candidate['stages'][0]['convergence']=={'max_shift_over_sigma':.5}
    assert '1e-06' in iteration_caption(candidate['stages'][0],context())
    with pytest.raises(ValueError,match='requested'):
        check_request(plan(.001),binding)


def test_omission_restores_source_control_between_explicit_stage_overrides():
    native=SimpleNamespace(data={'Controls':{'data':{'min dM/M':.001,'other':7}}})
    apply(native,plan(1e-6)['stages'][0],context())
    assert native.data['Controls']['data']['min dM/M']==1e-6
    apply(native,plan()['stages'][0],context())
    assert native.data['Controls']['data']=={'min dM/M':.001,'other':7}


def test_shift_sigma_request_does_not_authorize_a_native_tolerance_change():
    binding=bind_request('Require shift/sigma <= 0.5.',context())
    check_request(plan(),binding)
    with pytest.raises(ValueError,match='do not invent'):
        check_request(plan(.5e-5),binding)


def test_stage_specific_native_tolerances_are_not_flattened():
    binding=bind_request('Stage 1: background, ftol=1e-5.\nStage 2: background, ftol=1e-7.',context())
    candidate=plan(1e-5)
    candidate['stages'].append({**copy.deepcopy(candidate['stages'][0]),'id':'later','relative_objective_tolerance':1e-7})
    check_request(candidate,binding)
    candidate['stages'][1]['relative_objective_tolerance']=1e-5
    with pytest.raises(ValueError,match='Stage 2'):
        check_request(candidate,binding)


def test_invalid_or_conflicting_requests_ask_before_calling_a_model():
    for request in ('Set ftol=-1.','Set ftol=1e-6 and ftol=1e-4.',
                    'Set ftol=NaN.', 'Set ftol=inf.', 'Set ftol=1e-6junk.',
                    'Use ftol NaN.', 'Use ftol infinity.',
                    'Do not set ftol=1e-6.'):
        binding=bind_request(request,context())
        assert binding['questions']


def test_native_answer_after_numbered_recipe_applies_to_whole_recipe():
    request=('Stage 1: background.\nStage 2: cell.\n'
             'Clarification for this draft:\nUse native ftol=1e-6.')
    binding=bind_request(request,context())
    assert not binding['questions']
    assert binding['global_tolerance']==1e-6
    candidate=plan(1e-6)
    candidate['stages'].append({**copy.deepcopy(candidate['stages'][0]),'id':'later'})
    check_request(candidate,binding)


def test_global_multiline_instruction_after_numbered_stages_keeps_control_name():
    request=('Use exactly two stages.\nStage 1: background.\nStage 2: cell.\n'
             'Use native relative-objective\ntolerance 1e-5 in every stage.')
    binding=bind_request(request,context())
    assert binding['global_tolerance']==1e-5 and not binding['questions']
    candidate=plan(1e-5)
    candidate['stages'].append({**copy.deepcopy(candidate['stages'][0]),'id':'later'})
    check_request(candidate,binding)


def test_quoted_stage_in_answer_does_not_define_a_numeric_scope():
    request=('Stage 1: background.\nStage 2: cell.\n'
             'Clarification for this draft:\n"Stage 1: ftol=1e-2" is a quoted example. '
             'Use ftol=1e-6 for every stage.')
    binding=bind_request(request,context())
    assert binding['global_tolerance']==1e-6
    assert not binding['questions']


def test_retaining_missing_source_control_requires_clarification():
    incomplete=context()
    del incomplete['controls']['min dM/M']
    assert bind_request('Keep the source relative-objective tolerance.',incomplete)['questions']


def test_explicit_other_setting_retention_preserves_stage_native_controls():
    previous=plan(1e-5)
    previous['stages'].append({**copy.deepcopy(previous['stages'][0]),'id':'later','relative_objective_tolerance':1e-6})
    binding=bind_request('Change the oxygen choices. Keep everything else unchanged.',context(),previous)
    check_request(previous,binding)
    changed=copy.deepcopy(previous)
    del changed['stages'][1]['relative_objective_tolerance']
    with pytest.raises(ValueError,match='Stage 2'):
        check_request(changed,binding)
    changed['stages'].pop()
    with pytest.raises(ValueError,match='stage count'):
        check_request(changed,binding)


def test_source_and_previous_retention_are_distinct():
    assert bind_request('Keep the source relative-objective tolerance.',context())['global_tolerance']==.001
    assert bind_request('Keep the previous relative-objective tolerance.',context(),plan(1e-7))['global_tolerance']==1e-7
    assert bind_request('Keep the previous relative-objective tolerance.',context(),plan())['questions']


def test_optional_provider_null_does_not_change_legacy_plan():
    candidate=plan()
    candidate['stages'][0]['relative_objective_tolerance']=None
    assert 'relative_objective_tolerance' not in normalize_optional_nulls(candidate)['stages'][0]


def test_planner_repairs_wrong_native_control_without_relaxing_acceptance():
    class Client:
        def __init__(self): self.calls=[]
        def complete(self,messages):
            self.calls.append(copy.deepcopy(messages))
            value=.001 if len(self.calls)==1 else 1e-6
            return json.dumps(plan(value)), {'finish_reason':'stop','model':'test'}
    client=Client()
    result=make_plan('Refine background. Use native ftol=1e-6.',context(),client=client)
    assert len(client.calls)==2
    stage=result['plan']['stages'][0]
    assert stage['relative_objective_tolerance']==1e-6
    assert stage['convergence']['max_shift_over_sigma']==.5
    assert 'relative-objective' in client.calls[-1][-1]['content']
