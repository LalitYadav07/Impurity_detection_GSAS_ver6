import copy
import json
import pytest

from gpx_recipe.planner import make_plan, build_payload
from gpx_recipe.contracts import validate_plan
from gpx_recipe.schema import normalize_optional_nulls
from gpx_recipe.solver_plan_limits import requested_limits, check_limits


def context(method='numeric'):
    return dict(model_signature='model',controls={'deriv type':method},
                histograms=[dict(key='h0',name='PWDR test')],phases=[])


def plan(limit=1000):
    stage=dict(id='background',title='Background',cycles=1,free=[dict(kind='background',histogram='h0')],constraints=[])
    if limit is not None:stage['max_function_evaluations']=limit
    return dict(schema='radar-gpx-recipe/v1',model_signature='model',request='Refine background',
                questions=[],unsupported=[],assumptions=[],stages=[stage])


class Client:
    def __init__(self,*replies):self.replies=list(replies);self.calls=[]
    def complete(self,messages):
        self.calls.append(copy.deepcopy(messages));return json.dumps(self.replies.pop(0)),{}


def test_new_numeric_plans_require_reviewable_evaluation_budgets():
    client=Client(plan(None),plan(300))
    result=make_plan('Refine background',context(),client=client)
    assert result['status']=='ready_for_preflight' and len(client.calls)==2
    assert result['plan']['stages'][0]['max_function_evaluations']==300


@pytest.mark.parametrize('prompt',['Use at most 6 cycles.', 'Run three refinement cycles.'])
def test_explicit_numeric_cycle_request_needs_clarification_without_a_model_call(prompt):
    client=Client()
    result=make_plan(prompt+' Refine background.',context(),client=client)
    assert result['status']=='needs_input' and result['plan']['stages']==[] and not client.calls
    assert 'does not enforce a cycle count' in result['plan']['questions'][0]


def test_user_evaluation_ceiling_is_preserved_across_model_repair():
    client=Client(plan(1000),plan(400))
    result=make_plan('Refine background using at most 500 residual evaluations.',context(),client=client)
    assert len(client.calls)==2 and result['plan']['stages'][0]['max_function_evaluations']==400


def test_per_stage_ceilings_are_not_confused_with_each_other_or_quoted_context():
    ctx=context()
    request='Ignore the quoted label "at most 1 function evaluations".\nStage 1: Refine background, at most 200 residual evaluations.\nStage 2: Refine scale, at most 500 function evaluations.'
    limits=requested_limits(request,ctx)
    assert limits['global_maximum'] is None and limits['stage_maxima']=={1:200,2:500}
    p=plan(200);p['stages'].append(dict(p['stages'][0],id='scale',max_function_evaluations=500))
    check_limits(p,limits)
    p['stages'][1]['max_function_evaluations']=501
    with pytest.raises(ValueError):check_limits(p,limits)


def test_hessian_budget_and_invalid_counts_are_rejected():
    with pytest.raises(ValueError,match='numerical/Jacobian'):validate_plan(plan(500),context('analytic Hessian'))
    for bad in [0,-1,True,1.5,100001]:
        with pytest.raises(ValueError):validate_plan(plan(bad),context())
    validate_plan(plan(None),context('analytic Hessian'))
    with pytest.raises(ValueError,match='stored cycle count cannot limit'):
        validate_plan(plan(None),context())


def test_provider_null_remains_absent_for_hessian_and_budget_is_in_payload():
    p=plan(None);p['stages'][0]['max_function_evaluations']=None
    assert 'max_function_evaluations' not in normalize_optional_nulls(p)['stages'][0]
    payload=build_payload('Refine background',context())
    assert payload['native_solver_limits']['requires_evaluation_budget']
    assert 'max_function_evaluations' in payload['response_schema']['properties']['stages']['items']['properties']


def test_global_budget_is_not_silently_multiplied_across_stages():
    p=plan(200);p['stages'].append(dict(p['stages'][0],id='second'))
    with pytest.raises(ValueError,match='whole recipe'):
        check_limits(p,requested_limits('Use at most 400 residual evaluations.',context()))
    check_limits(p,requested_limits('Use at most 400 residual evaluations per stage.',context()))


@pytest.mark.parametrize('text', [
    'Use at most 200 residual/function evaluations per stage.',
    'Maximum function evaluations = 200 per stage.',
    'Maximum residual evaluations: 200 per stage.',
    'max_function_evaluations=200 per stage.',
])
def test_common_evaluation_budget_wording_is_bound(text):
    limits = requested_limits(text, context())
    assert limits['global_maximum'] == 200
    assert limits['global_per_stage'] and not limits['questions']
    check_limits(plan(200), limits)
    with pytest.raises(ValueError, match='ceiling of 200'):
        check_limits(plan(201), limits)


def test_answering_the_cycle_question_resolves_the_old_request_only():
    original = ('Use at most 30 cycles per stage.\n'
                'Stage 1: Refine background.\nStage 2: Refine histogram scale.')
    request = original + '\n\nClarification for this draft:\nUse at most 200 residual/function evaluations per stage instead of cycles.'
    limits = requested_limits(request, context())
    assert not limits['questions']
    assert limits['global_maximum'] == 200 and limits['global_per_stage']
    assert limits['stage_maxima'] == {}
    p = plan(200)
    p['stages'].append(dict(p['stages'][0], id='scale'))
    check_limits(p, limits)


def test_a_clarification_with_a_new_cycle_request_still_needs_an_answer():
    request = ('Use at most 30 cycles.\n\nClarification for this draft:\n'
               'Use at most 200 function evaluations per stage, but run three cycles.')
    assert requested_limits(request, context())['questions']


def test_a_partial_stage_answer_does_not_clear_other_stage_cycle_limits():
    request = ('Stage 1: Run three cycles for background.\n'
               'Stage 2: Run three cycles for scale.\n\n'
               'Clarification for this draft:\nStage 1: Use at most 200 function evaluations.')
    limits = requested_limits(request, context())
    assert limits['questions'] and limits['stage_maxima'] == {1: 200}


def test_clarification_does_not_silently_raise_an_existing_evaluation_limit():
    request = ('Use at most 100 residual evaluations per stage.\n\n'
               'Clarification for this draft:\nUse at most 200 function evaluations per stage.')
    assert requested_limits(request, context())['questions']


def test_quoted_clarification_and_budget_do_not_override_an_instruction():
    request = ('Use at most 30 cycles. The label is "Clarification for this draft:\n'
               'Use at most 200 function evaluations per stage."')
    limits = requested_limits(request, context())
    assert limits['questions'] and limits['global_maximum'] is None
