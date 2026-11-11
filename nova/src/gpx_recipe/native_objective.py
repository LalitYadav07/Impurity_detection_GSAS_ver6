"""Explicit native stopping tolerances, separate from shift/sigma acceptance."""
import math
import re

from .intent import mask_quoted

MINIMUM = 1e-12
MAXIMUM = 1e-2
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
NAME = (r"(?:relative[-_\s]+objective(?:[-\s]+(?:stopping|convergence))?[-_\s]+tolerance"
        r"|objective[-\s]+change\s+tolerance|(?:native\s+|solver\s+)?ftol|min\s+dM\s*/\s*M)")
VALUE = re.compile(r'(?<![\w])' + NAME + r"\s*(?:=|:|to|of)?\s*(" + NUMBER
                   + r")(?=$|[\s,;:)`]|[.!?](?=\s|$))", re.I)
ASSIGNMENT = re.compile(r'(?<![\w])' + NAME
    + r'\s*(?:(?:=|:|to|of)\s*|(?=[+\-\d]|\.\d|nan\b|inf(?:inity)?\b))', re.I)
RETAIN = re.compile(r"\b(?:keep|retain|preserve)\b[^.;\n]{0,70}" + NAME, re.I)
CLARIFICATION = re.compile(r'^\s*Clarification for this draft:\s*$', re.I | re.M)
STAGE = re.compile(r'^\s*(?:stage\s+)?(\d+)\s*[:.)-]\s*', re.I | re.M)
KEEP_OTHER = re.compile(r'\b(?:keep|leave|retain)\b[^.;\n]{0,50}\b(?:everything|all|other|rest)\b[^.;\n]{0,45}\b(?:unchanged|same)\b',re.I)


def validate_tolerance(value):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not MINIMUM <= value <= MAXIMUM):
        raise ValueError('Native relative-objective tolerance must be a finite number from 1e-12 to 0.01.')
    return float(value)


def _requested(text):
    text = mask_quoted(text)
    matches = list(VALUE.finditer(text))
    if any(not any(value.start() == assignment.start() for value in matches)
           for assignment in ASSIGNMENT.finditer(text)):
        return None, 'Specify a finite numeric native relative-objective tolerance, for example ftol=1e-6.'
    for match in matches:
        prefix = re.split(r'[.!?;\n]', text[:match.start()])[-1]
        if re.search(r"\b(?:do\s+not|don't|never|avoid|not\s+to)\b", prefix, re.I):
            return None, 'A native tolerance occurs in a negated instruction. State the value to use, or request preserving the source tolerance.'
    values = [float(m.group(1)) for m in matches]
    if not values:
        return None, None
    try:
        values = [validate_tolerance(value) for value in values]
    except ValueError as error:
        return None, str(error)
    if len(set(values)) != 1:
        return None, 'Specify one native relative-objective tolerance for this stage; conflicting values were supplied.'
    return values[0], None


def bind_request(request, context, previous=None):
    # The UI appends clarification answers. A new global answer must not be
    # mistaken for part of the final numbered stage in the original request.
    global_parts, stage_parts = [], {}
    for text in CLARIFICATION.split(mask_quoted(request)):
        headers = list(STAGE.finditer(text))
        global_parts.append(text[:headers[0].start()] if headers else text)
        if headers:
            # Global instructions often follow a numbered recipe and wrap
            # across lines. Retain the whole sentence, not only the line
            # containing "every stage", or the control name may be lost.
            global_parts.extend(sentence for sentence in re.split(r'(?<=[.!?;])\s+', text)
                if re.search(r'\b(?:all|both|each|every)\s+(?:the\s+)?stages?\b', sentence, re.I))
        for index, header in enumerate(headers):
            number = int(header.group(1))
            stop = headers[index+1].start() if index+1 < len(headers) else len(text)
            stage_parts.setdefault(number, []).append(text[header.end():stop])
    global_text = '\n'.join(global_parts)
    value, error = _requested(global_text)
    questions = [error] if error else []
    stages = []
    for number, parts in sorted(stage_parts.items()):
        selected, error = _requested('\n'.join(parts))
        if error:
            questions.append(f"Stage {number}: {error}")
        if selected is not None:
            stages.append(dict(stage=number, tolerance=selected))
    source = 'explicit request' if value is not None or stages else None
    expected_stage_count = None
    eligible = (isinstance(previous, dict) and previous.get('model_signature') == context['model_signature']
                and not previous.get('questions') and not previous.get('unsupported'))
    if value is None and not stages and not questions and eligible and KEEP_OTHER.search(mask_quoted(request)):
        old = previous.get('stages', [])
        if any(s.get('relative_objective_tolerance') is not None for s in old):
            stages = [dict(stage=i,tolerance=s['relative_objective_tolerance'])
                      for i,s in enumerate(old,1) if s.get('relative_objective_tolerance') is not None]
            expected_stage_count = len(old)
            source = 'previous recipe; other settings explicitly retained'
    if value is None and not questions and RETAIN.search(mask_quoted(global_text)):
        if re.search(r'\b(?:previous|same)\b', global_text, re.I):
            old = [s.get('relative_objective_tolerance') for s in previous.get('stages', [])] if eligible else []
            if old and None not in old and len(set(old)) == 1:
                value = old[0]; source = 'previous compatible recipe'
            else:
                questions.append('Specify the native relative-objective tolerance to retain; the previous recipe has no single explicit value.')
        else:
            value = (context.get('controls') or {}).get('min dM/M')
            source = 'source GPX native stopping control'
            if value is None:
                questions.append('The source GPX has no recorded native relative-objective tolerance to retain. Reinspect the native project or specify the value.')
        if value is not None:
            try:
                value = validate_tolerance(value)
            except ValueError as error:
                questions.append(str(error)); value = None
    return dict(global_tolerance=value, stages=stages, source=source, questions=questions,
                expected_stage_count=expected_stage_count)


def check_request(plan, binding):
    if not isinstance(plan, dict) or plan.get('questions') or plan.get('unsupported'):
        return
    if binding.get('expected_stage_count') is not None and len(plan.get('stages', [])) != binding['expected_stage_count']:
        raise ValueError('The stage count changed while retaining previous native controls. Specify their mapping to the new stages.')
    per_stage = {row['stage']: row['tolerance'] for row in binding['stages']}
    if set(per_stage) - set(range(1, len(plan.get('stages', []))+1)):
        raise ValueError('A requested native-tolerance stage is absent from the proposed recipe.')
    for index, stage in enumerate(plan.get('stages', []), 1):
        expected = per_stage.get(index, binding['global_tolerance'])
        actual = stage.get('relative_objective_tolerance')
        if expected != actual:
            if expected is None:
                raise ValueError(f'Stage {index}: do not invent a native relative-objective tolerance; preserve the source stopping control unless the user specifies it.')
            raise ValueError(f'Stage {index} must use the requested native relative-objective tolerance {expected:g}; the proposed value is {actual!r}. This is separate from shift/sigma acceptance.')


def apply(project, stage, context):
    """An omitted stage override retains the original GPX control, not a prior override."""
    value = stage.get('relative_objective_tolerance')
    if value is not None:
        value = validate_tolerance(value)
    else:
        value = (context.get('controls') or {}).get('min dM/M')
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            return
    controls = project.data.get('Controls', {}).get('data', {})
    if 'min dM/M' not in controls:
        raise ValueError('The native project does not expose its relative-objective stopping control.')
    controls['min dM/M'] = value
