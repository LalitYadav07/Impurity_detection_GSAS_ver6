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
