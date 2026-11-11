"""Request fidelity across initial plans, revisions and native-tolerance context."""
import copy
import json

import pytest

from gpx_recipe.convergence_intent import bind_convergence, check_convergence
from gpx_recipe.planner import PlanningValidationError, make_plan


def context():
    return dict(model_signature='model', controls={'min dM/M': .001},
                histograms=[dict(key='h0', name='PWDR control')],
                phases=[dict(key='p0', name='control', histograms=['PWDR control'], atoms=[])])


def plan(ceiling=.5):
    return dict(schema='radar-gpx-recipe/v1', model_signature='model', request='control',
                questions=[], unsupported=[], assumptions=[], stages=[
                    dict(id='background', title='Background', cycles=30,
                         free=[dict(kind='background', histogram='h0')], constraints=[],
                         convergence={'max_shift_over_sigma': ceiling})])


class Client:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.messages = []

    def complete(self, messages):
        self.messages.append(copy.deepcopy(messages))
        return json.dumps(next(self.replies)), {'model': 'synthetic-test', 'finish_reason': 'stop'}


@pytest.mark.parametrize('text', [
    'Require final-cycle maximum absolute shift/sigma <= 0.5 in both stages.',
    'Use max_shift_over_sigma=0.5.',
    'The shift/sigma threshold must be at most 5e-1.',
    'Require |shift/sigma| ≤ .5.',
    'Set shift/s.u. limit to 0.5.',
])
def test_explicit_ceiling_is_bound(text):
    bound = bind_convergence(text, None, 'model')
    assert bound['global_maximum'] == .5 and bound['questions'] == []
    check_convergence(plan(), bound)
    with pytest.raises(ValueError, match='relative-objective tolerance'):
        check_convergence(plan(.001), bound)


def test_retained_ceiling_comes_from_compatible_recipe_not_controls():
    bound = bind_convergence('Keep the original convergence threshold.', plan(), 'model')
    assert bound['global_maximum'] == .5 and bound['source'] == 'previous compatible recipe'
    check_convergence(plan(), bound)


@pytest.mark.parametrize('previous', [None, dict(plan(), model_signature='other'),
    dict(plan(), questions=['unresolved']), dict(plan(), stages=[]),
    dict(plan(), stages=[{**plan()['stages'][0], 'convergence': None}]),
    dict(plan(), stages=[plan()['stages'][0], {**plan()['stages'][0], 'convergence': {'max_shift_over_sigma': .1}}])])
def test_missing_or_ambiguous_retained_criterion_asks_before_model_call(previous):
    client = Client([])
    result = make_plan('Retain that convergence requirement.', context(), previous=previous, client=client)
    assert result['status'] == 'needs_input' and result['plan']['stages'] == []
    assert 'relative-objective tolerance' in result['plan']['questions'][0]
    assert client.messages == []


def test_explicit_new_ceiling_overrides_a_previous_draft():
    bound = bind_convergence('Keep the remaining settings. Require shift/sigma <= 0.5.', plan(.001), 'model')
    assert bound['global_maximum'] == .5


def test_stage_specific_ceiling_and_retention_are_not_flattened():
    text = 'Retain the original convergence threshold.\nStage 1: background.\nStage 2: cell, shift/sigma <= 0.1.'
    bound = bind_convergence(text, plan(), 'model')
    candidate = plan()
    candidate['stages'].append({**candidate['stages'][0], 'id': 'cell', 'convergence': {'max_shift_over_sigma': .1}})
    check_convergence(candidate, bound)
    candidate['stages'][0]['convergence']['max_shift_over_sigma'] = .001
    with pytest.raises(ValueError, match='Stage 1'):
        check_convergence(candidate, bound)


def test_concluding_all_stage_limit_applies_to_every_stage():
    text = 'Stage 1: background.\nStage 2: cell.\nFor both stages require shift/sigma <= 0.5.'
    bound = bind_convergence(text, None, 'model')
    assert bound['global_maximum'] == .5
    candidate = plan(.001)
    candidate['stages'].append({**plan()['stages'][0], 'id': 'cell'})
    with pytest.raises(ValueError, match='Stage 1'):
        check_convergence(candidate, bound)


def test_wrong_model_ceiling_is_repaired_within_existing_call_budget():
    client = Client([plan(.001), plan(.5)])
    result = make_plan('Keep the original convergence threshold.', context(), previous=plan(), client=client)
    assert len(client.messages) == 2 and result['plan']['stages'][0]['convergence']['max_shift_over_sigma'] == .5
    assert 'relative-objective tolerance' in client.messages[1][-1]['content']
    payload = json.loads(client.messages[0][1]['content'])
    assert payload['convergence_request']['global_maximum'] == .5


def test_repeated_wrong_ceiling_does_not_reach_preflight():
    client = Client([plan(.001), plan(.001)])
    with pytest.raises(PlanningValidationError, match='proposed value is 0.001'):
        make_plan('Require shift/sigma <= 0.5.', context(), client=client)
    assert len(client.messages) == 2


def test_unspecified_or_quoted_numbers_are_not_guessed_as_a_ceiling():
    for request in ['The previous observed shift/sigma was 0.77371. Refine the background.',
                    'The objective tolerance is 0.001. Refine the background.',
                    'The label is "shift/sigma <= 0.1". Refine the background.']:
        bound = bind_convergence(request, None, 'model')
        assert bound['global_maximum'] is None and bound['questions'] == []


@pytest.mark.parametrize('text', [
    'Do not retain the previous convergence threshold.',
    "Don't keep the original convergence screen.",
    'Never use the previous convergence threshold.',
    'The original convergence threshold was not specified.',
])
def test_negated_or_descriptive_retention_does_not_force_a_previous_limit(text):
    bound = bind_convergence(text, plan(.5), 'model')
    assert bound['global_maximum'] is None and bound['questions'] == []


@pytest.mark.parametrize('text', ['shift/sigma <= 0.5; shift/sigma <= 0.1.', 'shift/sigma <= 0.', 'shift/sigma <= 2.'])
def test_conflicting_or_invalid_limits_require_clarification(text):
    assert bind_convergence(text, None, 'model')['questions']


@pytest.mark.parametrize('text', [
    'Retain native convergence.',
    'Keep the GSAS-II convergence criterion.',
    'Preserve the native solver convergence settings.',
    'Keep the minimizer convergence threshold unchanged.',
])
@pytest.mark.parametrize('previous', [None, plan(.5)])
def test_native_solver_convergence_does_not_request_a_shift_sigma_screen(text, previous):
    bound = bind_convergence(text, previous, 'model')
    assert bound['global_maximum'] is None and bound['questions'] == []


def test_native_convergence_does_not_hide_an_explicit_shift_sigma_requirement():
    bound = bind_convergence('Keep native convergence. Retain the original shift/sigma threshold.', plan(.5), 'model')
    assert bound['global_maximum'] == .5 and bound['source'] == 'previous compatible recipe'
