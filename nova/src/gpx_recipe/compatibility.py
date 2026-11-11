"""Cheap, source-bound recipe compatibility previews; native validation still runs."""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re


SCHEMA = 'radar-recipe-model/v1'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=1)
def inspection_revision():
    # Application/worker upgrades start a new process. Do not hash the complete
    # scientific package on every UI refresh or once for every selected scan.
    from .symmetry.compiler import package_signature
    return package_signature()


def descriptor(context, *, revision=None):
    binding = context['binding']
    if _digest(binding) != context['model_signature']:
        raise ValueError('The model binding differs from its native signature')
    return dict(schema=SCHEMA, source_sha256=context['source']['sha256'],
                implementation_sha256=revision or inspection_revision(),
                model_signature=context['model_signature'], binding=copy.deepcopy(binding))


def capture(path, *, expected_sha256):
    """Inspect a final published GPX in its existing numerical worker, once."""
    from .snapshot import snapshot
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('Published GPX changed before its model inspection')
    context = snapshot(path)
    if (context['source']['sha256'] != expected_sha256
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256):
        raise ValueError('Published GPX changed during its model inspection')
    return descriptor(context)


def _validated(value, sha, revision):
    if not isinstance(value, dict) or value.get('schema') != SCHEMA:
        return None
    if not re.fullmatch(r'[a-f0-9]{64}', str(sha)) or value.get('source_sha256') != sha:
        return None
    if value.get('implementation_sha256') != revision:
        return None
    binding = value.get('binding')
    if not isinstance(binding, dict):
        return None
    try:
        return value if value.get('model_signature') == _digest(binding) else None
    except (TypeError, ValueError, OverflowError):
        return None


def differences(expected, actual):
    """Explain canonical binding differences without changing recipe matching."""
    messages = []
    ep, ap = expected.get('phases', []), actual.get('phases', [])
    names = lambda phases: ', '.join(str(p.get('name', p.get('key', '?'))) for p in phases) or 'none'
    if [(p.get('key'), p.get('name')) for p in ep] != [(p.get('key'), p.get('name')) for p in ap]:
        messages.append(f"Phase inventory differs: recipe has {names(ep)}; this scan has {names(ap)}.")
    else:
        for left, right in zip(ep, ap):
            if left.get('space_group') != right.get('space_group'):
                messages.append(f"Space group differs for {left.get('name', left.get('key'))}.")
            if left.get('atoms') != right.get('atoms'):
                la={a.get('label'):a for a in left.get('atoms',[])}
                ra={a.get('label'):a for a in right.get('atoms',[])}
                if la.keys()!=ra.keys():
                    expected_sites=', '.join(str(k) for k in sorted(la.keys()-ra.keys(),key=str)) or 'none'
                    found_sites=', '.join(str(k) for k in sorted(ra.keys()-la.keys(),key=str)) or 'none'
                    messages.append(f"Atom sites differ for {left.get('name', left.get('key'))}: recipe-only {expected_sites}; scan-only {found_sites}.")
                else:
                    changed=', '.join(str(k) for k in la if la[k]!=ra[k])
                    messages.append(f"Atom elements, displacement models or site symmetry differ for {left.get('name', left.get('key'))}: {changed}.")
    eh, ah = expected.get('histograms', []), actual.get('histograms', [])
    if [h.get('key') for h in eh] != [h.get('key') for h in ah]:
        messages.append('The histogram/bank inventory differs.')
    else:
        for index, (left, right) in enumerate(zip(eh, ah), 1):
            if left.get('instrument') != right.get('instrument'):
                messages.append(f"Instrument type or calibration differs for histogram {index}.")
            if left.get('background') != right.get('background'):
                messages.append(f"Background function or coefficient count differs for histogram {index}.")
    labels = dict(source_constraints='Existing constraints', source_bounds='Parameter bounds',
                  occupancy_groups='Occupancy groups', model_topology='Profile or texture model',
                  source_restraints='Existing restraints', phase_histogram_links='Phase-to-histogram links',
                  occupancy_reviews='Occupancy review')
    for key, label in labels.items():
        if expected.get(key) != actual.get(key):
            messages.append(label + ' differ.')
    return messages or ['The native model binding differs.']


def assess(source, reference, *, revision=None):
    """Use only an exact current inspection; old/missing metadata is unverified."""
    revision = revision or inspection_revision()
    if not reference.get('model_signature'):
        return dict(status='unverified', phases='Not recorded',
                    reason='No reviewed model identity is available. Native recipe validation remains required.')
    sha = source.get('source_sha256')
    context = source.get('context') or {}
    candidate = None
    if (source.get('inspection_revision') == revision
            and context.get('source', {}).get('sha256') == sha
            and context.get('model_signature')):
        candidate = context
    if candidate is None:
        candidate = _validated(source.get('recipe_model') or source.get('project', {}).get('recipe_model'), sha, revision)
    if candidate is None:
        return dict(status='unverified', phases='Not inspected',
                    reason='No current model inspection for this exact GPX. The native worker will check it before refinement.')
    signature = candidate['model_signature']
    binding = candidate.get('binding') or {}
    phases = ', '.join(str(p.get('name', p.get('key', '?'))) for p in binding.get('phases', []))
    if not phases:
        phases = ', '.join(str(p.get('name', p.get('key', '?'))) for p in candidate.get('phases', [])) or 'Not recorded'
    if signature == reference.get('model_signature'):
        return dict(status='compatible', phases=phases, reason='Matches the reviewed recipe model. Native validation still runs before refinement.')
    return dict(status='incompatible', phases=phases,
                reason=' '.join(differences(reference.get('binding') or {}, binding)))


def preview(sources, reference, *, revision=None):
    revision = revision or inspection_revision()
    rows = [dict(id=s['id'], scan=s.get('scan_label', s['id']), **assess(s, reference, revision=revision)) for s in sources]
    for row in rows:
        row['status_label'] = {'compatible':'Matching model', 'incompatible':'Different model',
                               'unverified':'Not yet checked'}[row['status']]
    counts = {status: sum(row['status'] == status for row in rows)
              for status in ('compatible', 'incompatible', 'unverified')}
    return dict(rows=rows, **counts)
