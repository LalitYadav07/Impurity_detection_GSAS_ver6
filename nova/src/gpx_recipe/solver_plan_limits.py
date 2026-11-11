"""Keep native numerical evaluation limits distinct from Hessian cycles."""
import re

from .intent import mask_quoted, stage_sections
from .native_budget import NUMERICAL_METHODS

_EVALUATION_KIND = r'(?:residual(?:\s*/\s*function)?|function(?:\s*/\s*residual)?)'
_EVALUATIONS = re.compile(
    r'\b(?:at most|up to|no more than|maximum(?: of)?)\s+(\d+)\s+'
    + _EVALUATION_KIND + r'\s+evaluations?\b'
    r'|\bmax(?:imum)?(?:_|\s+(?:number\s+of\s+)?)' + _EVALUATION_KIND
    + r'(?:_|\s+)evaluations?\s*[:=]?\s*(\d+)\b', re.I)
_CYCLES = re.compile(r'\b(?:at most|up to|no more than|maximum(?: of)?|exactly|use|run|perform)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:refinement\s+)?cycles?\b', re.I)
_CLARIFICATION = re.compile(r'^\s*Clarification for this draft:\s*$', re.I | re.M)
_PER_STAGE = re.compile(r'\b(?:per|each|every)\s+stage\b', re.I)


def _scopes(request):
    # The UI appends answers rather than replacing the scientific request. A
    # new answer's global limit must not become part of the last numbered stage.
    # Mask quotes before looking for the UI marker, so copied labels cannot
    # acquire the semantics of an answer.
    scopes = []
    for revision, text in enumerate(_CLARIFICATION.split(mask_quoted(request))):
        sections = stage_sections(text)
        marker = re.search(r'^\s*(?:stage\s+)?1\s*[:.)-]', text, re.I | re.M) if sections else None
        parts = [(None, text[:marker.start()] if marker else text)]
        parts.extend((row['stage'], row['text']) for row in sections)
        for stage, content in parts:
            values = {int(match.group(1) or match.group(2)) for match in _EVALUATIONS.finditer(content)}
            scopes.append(dict(revision=revision, stage=stage, text=content,
                               values=values, per_stage=bool(_PER_STAGE.search(content))))
    return scopes


def _answered_cycle_request(scope, scopes):
    # Only an actual later answer with a definite evaluation count can resolve
    # the older cycle question. An unrelated answer, a conflicting count, or a
    # limit for a different stage cannot discard the user's earlier constraint.
    for answer in scopes:
        if answer['revision'] <= scope['revision'] or len(answer['values']) != 1:
            continue
        if answer['stage'] == scope['stage']:
            return True
        if scope['stage'] is not None and answer['stage'] is None and answer['per_stage']:
            return True
    return False


def requested_limits(request, context):
    method = str((context.get('controls') or {}).get('deriv type') or '')
    scopes = _scopes(request)
    questions = []

    def cap(stage):
        values = set().union(*(scope['values'] for scope in scopes if scope['stage'] == stage))
        if len(values)>1:
            questions.append('Specify one maximum residual-evaluation count for each affected stage.')
            return None
        return next(iter(values), None)

    global_cap = cap(None)
    stages = {stage: value for stage in sorted({scope['stage'] for scope in scopes if scope['stage'] is not None})
              if (value := cap(stage)) is not None}
    if method in NUMERICAL_METHODS and any(_CYCLES.search(scope['text']) and not _answered_cycle_request(scope, scopes)
                                          for scope in scopes):
        questions.append('This GPX uses the GSAS-II '+method+' minimizer, which does not enforce a cycle count. '
                         'Specify a maximum number of residual/function evaluations instead; the worker will not silently convert cycles.')
    if method and method not in NUMERICAL_METHODS and (global_cap is not None or stages):
        questions.append('This GPX uses the '+method+' minimizer. Specify its maximum cycles; a MINPACK residual-evaluation budget does not apply to it.')
    return dict(method=method, requires_evaluation_budget=method in NUMERICAL_METHODS,
                global_maximum=global_cap, stage_maxima=stages,
                global_per_stage=global_cap is not None and all(scope['per_stage'] for scope in scopes
                    if scope['stage'] is None and scope['values']), questions=questions)


def check_limits(plan, limits):
    if not plan['stages']:
        return
    if len(plan['stages'])>1 and limits['global_maximum'] is not None and not limits.get('global_per_stage'):
        raise ValueError('Clarify whether the requested global evaluation budget is for the whole recipe or for each stage; do not multiply it across stages silently.')
    for index, stage in enumerate(plan['stages'], 1):
        value = stage.get('max_function_evaluations')
        if limits['requires_evaluation_budget'] and value is None:
            raise ValueError('A new numerical recipe needs an explicit max_function_evaluations in every stage; cycles do not cap MINPACK.')
        ceiling = limits['stage_maxima'].get(index, limits['global_maximum'])
        if ceiling is not None and (value is None or value > ceiling):
            raise ValueError(f'Stage {index} exceeds the user residual-evaluation ceiling of {ceiling}')
