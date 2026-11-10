"""Bind stated shift/sigma criteria without confusing native solver tolerances.

Only explicit ceilings and unambiguous retention of a previous recipe bind.
This does not infer a scientific convergence criterion from observed shifts or
the GSAS-II relative-objective stopping tolerance.
"""
import math
import re

from .intent import mask_quoted, stage_sections

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
SHIFT = r"(?:\|?\s*shift\s*/\s*(?:sigma|s\.?u\.?)\s*\|?|max_shift_over_sigma)"
CEILING = re.compile(
    SHIFT + r"\s*(?:(?:convergence\s+)?(?:threshold|limit|ceiling|criterion)\s*)?"
    r"(?:<=|≤|=|(?:must\s+be\s+)?(?:at\s+most|no\s+more\s+than|not\s+above)|of|to)\s*(" + NUMBER + r")",
    re.I,
)
RETAIN = re.compile(
    r"\b(?:keep|retain|preserve)\b[^.;\n]{0,90}\b(?:convergence|shift\s*/\s*sigma)\b"
    r"|\b(?:same|original|previous|existing|unchanged)\s+(?:final[- ]cycle\s+)?"
    r"(?:convergence\s+(?:threshold|criterion|criteria|screen|requirement)|shift\s*/\s*sigma\s+(?:limit|threshold|ceiling))",
    re.I,
)


def _ceiling(text):
    values = [float(match.group(1)) for match in CEILING.finditer(mask_quoted(text))]
    if not values:
        return None, None
    if any(not math.isfinite(value) or not 0 < value <= 1 for value in values):
        return None, "Specify a final-cycle maximum absolute shift/sigma greater than 0 and at most 1."
    if len(set(values)) != 1:
        return None, "Specify one final-cycle maximum absolute shift/sigma for this scope; conflicting ceilings were supplied."
    return values[0], None


def bind_convergence(request, previous, model_signature):
    sections = stage_sections(request)
    first = re.search(r"^\s*(?:stage\s+)?1\s*[:.)-]\s*", request, re.I | re.M)
    global_text = request[:first.start()] if sections and first else request
    if sections:
        # A concluding instruction for every stage is global even when the
        # numbered-stage parser places that line in the final stage's body.
        global_text += '\n' + '\n'.join(line for line in request.splitlines()
            if re.search(r'\b(?:all|both|each|every)\s+(?:the\s+)?stages?\b', mask_quoted(line), re.I))
    global_limit, problem = _ceiling(global_text)
    questions = [problem] if problem else []
    stages = []
    for section in sections:
        limit, error = _ceiling(section['text'])
        if error:
            questions.append(f"Stage {section['stage']}: {error}")
        if limit is not None:
            stages.append(dict(stage=section['stage'], maximum=limit))
    source = 'explicit request' if global_limit is not None or stages else None
    retaining = bool(RETAIN.search(mask_quoted(request)))
    if retaining and global_limit is None and not questions:
        prior_stages = previous.get('stages', []) if isinstance(previous, dict) else []
        eligible = (isinstance(previous, dict) and previous.get('model_signature') == model_signature
                    and not previous.get('questions') and not previous.get('unsupported') and prior_stages)
        limits = [(stage.get('convergence') or {}).get('max_shift_over_sigma') for stage in prior_stages] if eligible else []
        if (limits and all(type(value) in (int, float) and math.isfinite(value) and 0 < value <= 1
                           for value in limits) and len(set(limits)) == 1):
            global_limit = float(limits[0])
            source = 'previous compatible recipe'
        elif not sections or len(stages) != len(sections):
            questions.append("Which final-cycle maximum absolute shift/sigma should be retained? "
                             "No single criterion is available from a previous compatible recipe. "
                             "Specify the number (for example, shift/sigma <= 0.5); the native relative-objective tolerance is a different quantity.")
    return dict(global_maximum=global_limit, stages=stages, source=source, questions=questions)


def check_convergence(plan, binding):
    if not isinstance(plan, dict) or plan.get('questions') or plan.get('unsupported'):
        return
    stage_limits = {row['stage']: row['maximum'] for row in binding['stages']}
    for index, stage in enumerate(plan.get('stages', []), 1):
        expected = stage_limits.get(index, binding['global_maximum'])
        if expected is None:
            continue
        actual = (stage.get('convergence') or {}).get('max_shift_over_sigma')
        if actual != expected:
            raise ValueError(f"Stage {index} must retain the requested final-cycle maximum absolute shift/sigma "
                             f"{expected:g}; the proposed value is {actual!r}. "
                             "Do not substitute the GSAS-II relative-objective tolerance or silently omit the criterion.")
