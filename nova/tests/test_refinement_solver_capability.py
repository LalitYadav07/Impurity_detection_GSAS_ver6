"""The planner must distinguish Hessian diagnostics from Jacobian diagnostics."""
import pytest
import json

from gpx_recipe.convergence import require_screen_support
from gpx_recipe import planner as native
from radar_pd_nova.refinement_planner import solver_capability, build_payload
from radar_pd_nova.refinement_planner import make_plan


@pytest.mark.parametrize('method,supported', [
    ('analytic Hessian', True), ('Hessian SVD', True),
    ('numeric', False), ('analytic Jacobian', False),
    ('', None), ('future solver', None),
])
def test_planner_reports_inspected_solver_capability(method, supported):
    context = dict(model_signature='test', controls={'deriv type': method},
                   phases=[], histograms=[])
    previous = dict(questions=['The analytic Hessian cannot report shift/sigma.'])
    payload = build_payload('Retain the solver and use shift/sigma <= 0.1.', context, previous)
    capability = payload['native_convergence_capability']
    assert capability['method'] == method
    assert capability['final_cycle_shift_over_sigma_supported'] is supported
    assert payload['previous_plan'] == previous
    assert 'prior draft question is not a solver-capability fact' in payload['convergence_instruction']
    assert payload['convergence_request']['global_maximum'] == .1
    if supported is False:
        with pytest.raises(ValueError, match='does not save'):
            require_screen_support(context)
    elif supported is True:
        require_screen_support(context)


def test_solver_capability_does_not_infer_a_successful_or_complete_stage():
    facts = solver_capability({'controls': {'deriv type': 'analytic Hessian'}})
    assert 'converged' not in facts and 'last_cycle_complete' not in facts
    assert 'capability alone does not pass' in facts['evidence_required']


def test_capability_enrichment_keeps_the_shared_planner_unchanged():
    context = dict(model_signature='test', controls={'deriv type': 'analytic Hessian'},
                   phases=[], histograms=[])
    request = 'Refine the background and require shift/sigma <= 0.1.'
    before = native.build_payload(request, context)
    enriched = build_payload(request, context)
    assert 'native_convergence_capability' not in before
    assert 'native_convergence_capability' in enriched
    assert native.build_payload(request, context) == before
    assert enriched['user_request'] == before['user_request']
    assert enriched['convergence_request'] == before['convergence_request']


def test_numeric_clarification_reaches_the_planner_with_its_actual_ceiling():
    context = dict(model_signature='public-control', controls={'deriv type': 'numeric'},
                   phases=[], histograms=[dict(key='h0', name='PWDR control')])
    original = ('Use at most 30 cycles per stage.\nStage 1: Refine background.\n'
                'Stage 2: Refine histogram scale.')
    request = original + ('\n\nClarification for this draft:\n'
        'Use maximum function evaluations = 200 per stage instead of cycles. Retain native convergence.')
    calls = []

    class Client:
        def complete(self, messages):
            calls.append(messages)
            return json.dumps(dict(schema='radar-gpx-recipe/v1', model_signature='public-control',
                request=request, questions=[], unsupported=[], assumptions=[], stages=[
                    dict(id='background', title='Background', cycles=1, max_function_evaluations=200,
                         free=[dict(kind='background', histogram='h0')], constraints=[]),
                    dict(id='scale', title='Scale', cycles=1, max_function_evaluations=200,
                         free=[dict(kind='histogram_scale', histogram='h0')], constraints=[]),
                ])), {}

    result = make_plan(request, context, client=Client())
    assert result['status'] == 'ready_for_preflight' and len(calls) == 1
    payload = json.loads(calls[0][1]['content'])
    assert payload['user_request'] == request
    assert payload['native_solver_limits']['global_maximum'] == 200
    assert payload['native_solver_limits']['global_per_stage']
    assert payload['convergence_request']['global_maximum'] is None
    assert not payload['convergence_request']['questions']
