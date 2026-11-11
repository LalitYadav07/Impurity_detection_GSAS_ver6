"""Keep native numerical evaluation limits distinct from Hessian cycles."""
import re

from .intent import mask_quoted, stage_sections
from .native_budget import NUMERICAL_METHODS

_EVALUATIONS = re.compile(r'\b(?:at most|up to|no more than|maximum(?: of)?)\s+(\d+)\s+(?:residual|function)\s+evaluations?\b', re.I)
_CYCLES = re.compile(r'\b(?:at most|up to|no more than|maximum(?: of)?|exactly|use|run|perform)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:refinement\s+)?cycles?\b', re.I)


def requested_limits(request, context):
    method = str((context.get('controls') or {}).get('deriv type') or '')
    sections = stage_sections(request)
    raw = mask_quoted(request)
    marker = re.search(r'^\s*(?:stage\s+)?1\s*[:.)-]', raw, re.I | re.M) if sections else None
    prefix = raw[:marker.start()] if marker else raw
    questions = []

    def cap(text):
        values = set(int(m.group(1)) for m in _EVALUATIONS.finditer(mask_quoted(text)))
        if len(values)>1:
            questions.append('Specify one maximum residual-evaluation count for each affected stage.')
            return None
        return next(iter(values), None)

    global_cap = cap(prefix)
    stages = {row['stage']: value for row in sections if (value:=cap(row['text'])) is not None}
    if method in NUMERICAL_METHODS and _CYCLES.search(raw):
        questions.append('This GPX uses the GSAS-II '+method+' minimizer, which does not enforce a cycle count. '
                         'Specify a maximum number of residual/function evaluations instead; the worker will not silently convert cycles.')
    if method and method not in NUMERICAL_METHODS and (global_cap is not None or stages):
        questions.append('This GPX uses the '+method+' minimizer. Specify its maximum cycles; a MINPACK residual-evaluation budget does not apply to it.')
    return dict(method=method, requires_evaluation_budget=method in NUMERICAL_METHODS,
                global_maximum=global_cap, stage_maxima=stages,
                global_per_stage=bool(re.search(r'\b(?:per|each|every)\s+stage\b',prefix,re.I)), questions=questions)


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
