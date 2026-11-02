"""Presentation of nuisance parameters from one verified execution stage.

This module never reads a GPX or computes parameter uncertainties. The caller
hash-verifies both the selected view and execution receipt before using it.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re


_GROUPS = {
    'background', 'histogram_scale', 'phase_fraction', 'preferred_orientation',
    'instrument_zero', 'instrument_profile', 'sample_displacement',
    'hydrostatic_strain', 'correction', 'histogram_or_instrument', 'rigid_body',
}
_SHA = re.compile(r'[a-f0-9]{64}')


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _sha(value):
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _selected_stages(view, result, execution, selected, source_sha, plan_sha):
    if (not isinstance(execution, dict) or not isinstance(result, dict)
            or execution.get('schema') != 'radar-gpx-execution/v1' or execution.get('preview') is not False
            or not _sha(source_sha) or result.get('source_sha256') != source_sha
            or execution.get('source_sha256') != source_sha or not _sha(plan_sha)
            or execution.get('plan_sha256') != plan_sha):
        raise ValueError('Execution parameters do not match the selected source and approved recipe')
    published = result.get('stages') or []
    stages = execution.get('stages') or []
    if (not isinstance(published, list) or not isinstance(stages, list)
            or not stages or len(stages) > 100 or len(published) > 100
            or any(not isinstance(s, dict) or not isinstance(s.get('id'), str) for s in stages + published)
            or len({s['id'] for s in stages}) != len(stages)
            or len({s['id'] for s in published}) != len(published)):
        raise ValueError('Execution has no unambiguous completed stage history')
    files = result.get('files') or {}
    if not isinstance(files, dict) or not _sha(files.get(result.get('execution'))):
        raise ValueError('The execution receipt is not a verified result artifact')
    if selected == 'final':
        if (result.get('status') != 'completed' or execution.get('status') != 'completed_needs_scientific_review'
                or [s['id'] for s in published] != [s['id'] for s in stages]
                or any(s.get('status') != 'completed' for s in stages)):
            raise ValueError('Final parameters require every stage to have completed')
        expected = files.get(result.get('final_gpx'))
        if not _sha(expected) or expected != result.get('output_sha256') or expected != execution.get('output_gpx_sha256'):
            raise ValueError('Final parameters do not match the published final GPX')
        prefix = stages
    else:
        matches = [i for i, s in enumerate(stages) if s['id'] == selected]
        output = next((s for s in published if s['id'] == selected), None)
        if len(matches) != 1 or output is None:
            raise ValueError('Choose a published completed stage for execution parameters')
        prefix = stages[:matches[0]+1]
        if any(s.get('status') != 'completed' for s in prefix):
            raise ValueError('Execution parameters cannot come from an incomplete stage')
        expected = files.get(output.get('gpx'))
    if not _sha(expected) or view.get('source_sha256') != expected:
        raise ValueError('Execution parameters do not match the selected GPX view')
    return prefix


def _location(variable, view, context, source_sha):
    parts = variable.split(':')
    if len(parts) < 3 or any(p and not p.isdecimal() for p in parts[:2]):
        raise ValueError('Execution parameter has an unsupported native identity')
    phase = None
    if parts[0]:
        matches = [p for p in view.get('phases', []) if p.get('native_id') == int(parts[0])]
        if len(matches) != 1:
            raise ValueError('Execution parameter phase is absent from the selected view')
        phase = matches[0]
    histogram = None
    if parts[1]:
        native_id = int(parts[1])
        # Representative-scan IDs can differ. Only use context from this exact
        # source; otherwise a single view histogram is unambiguous. Multiple
        # unmatched histograms retain the honest native numerical identity.
        if context and (context.get('source') or {}).get('sha256') == source_sha:
            matches = [h for h in context.get('histograms', []) if h.get('native_id') == native_id]
            if len(matches) == 1:
                histogram = next((h for h in view.get('histograms', [])
                                  if h.get('name') == matches[0].get('name') and h.get('key') == matches[0].get('key')), None)
        if histogram is None and len(view.get('histograms', [])) == 1:
            histogram = view['histograms'][0]
    return phase, histogram, ('Native histogram '+parts[1] if parts[1] else '')


def _held_value(record, stage, phase, histogram, previous):
    """Only carry a prior estimate when this stage explicitly records it held."""
    state = stage.get('final_state') or {}
    group, family = record.get('group'), record.get('family')
    value = None
    if group == 'histogram_scale' and histogram:
        matches = [h for h in state.get('histograms', []) if h.get('key') == histogram.get('key')]
        stored = matches[0].get('histogram_scale') if len(matches) == 1 else None
        if isinstance(stored, list) and len(stored) >= 2 and stored[1] is False:
            value = stored[0]
    elif phase and histogram:
        matches = [p for p in state.get('phases', []) if p.get('key') == phase.get('key')]
        hap = (matches[0].get('hap') or {}).get(histogram.get('name'), {}) if len(matches) == 1 else {}
        if group == 'phase_fraction':
            stored = hap.get('scale')
            if isinstance(stored, list) and len(stored) >= 2 and stored[1] is False:
                value = stored[0]
        elif group == 'preferred_orientation' and family == 'MD':
            stored = hap.get('texture')
            prior_phases = (previous.get('final_state') or {}).get('phases', [])
            prior_matches = [p for p in prior_phases if p.get('key') == phase.get('key')]
            old = ((prior_matches[0].get('hap') or {}).get(histogram.get('name'), {}).get('texture')
                   if len(prior_matches) == 1 else None)
            if (isinstance(stored, list) and len(stored) >= 4 and stored[0] == 'MD' and stored[2] is False
                    and isinstance(old, list) and len(old) >= 4 and old[0] == stored[0] and old[3] == stored[3]):
                value = stored[1]
    return value if _finite(value) else None


def execution_parameters(view, result, execution, selected, *, source_sha, plan_sha, context=None):
    """Supplement existing physical rows without borrowing another stage's values."""
    stages = _selected_stages(view, result, execution, selected, source_sha, plan_sha)
    current = stages[-1]
    records = {}
    for stage in stages:
        changes = stage.get('parameter_changes') or {}
        if not isinstance(changes, dict):
            raise ValueError('Execution parameter ledger is invalid')
        ledger = changes.get('released_parameters') or []
        if not isinstance(ledger, list) or len(ledger) > 10000:
            raise ValueError('Execution parameter ledger is invalid or too large')
        seen = set()
        for record in ledger:
            if not isinstance(record, dict) or record.get('group') not in _GROUPS:
                continue  # Cell/atom/size/strain/weight rows already have native view exports.
            variable = record.get('variable')
            if not isinstance(variable, str) or variable in seen:
                raise ValueError('Execution has ambiguous native parameter rows')
            seen.add(variable)
            records[variable] = (record, stage['id'])
    rows, omitted = [], False
    for variable, (record, refined_stage) in records.items():
        phase, hist, fallback = _location(variable, view, context, source_sha)
        value = record.get('after')
        prior = refined_stage != current['id']
        if prior:
            previous = next(s for s in stages if s['id'] == refined_stage)
            value = _held_value(record, current, phase, hist, previous)
            if value is None or not _finite(record.get('after')) or not math.isclose(value, record['after'], rel_tol=1e-10, abs_tol=1e-12):
                omitted = True
                continue
        if not _finite(value):
            raise ValueError('Execution parameter value is not finite')
        sigma = record.get('sigma')
        if sigma is not None and (not _finite(sigma) or sigma < 0):
            raise ValueError('Execution parameter uncertainty is invalid')
        group, family = record['group'], str(record.get('family') or variable.split(':')[2])
        if group == 'preferred_orientation' and family == 'MD':
            label, unit = 'Preferred orientation · March–Dollase ratio', 'dimensionless'
        elif group == 'preferred_orientation':
            label, unit = 'Preferred orientation · '+str(record.get('meaning') or family), 'native'
        elif group == 'phase_fraction':
            label, unit = 'Phase scale (HAP; not weight %)', 'native scale'
        elif group == 'histogram_scale':
            label, unit = 'Histogram scale', 'native scale'
        elif group == 'background':
            number = family.removeprefix('Back;')
            label, unit = 'Background coefficient '+number, 'native'
        else:
            label, unit = str(record.get('meaning') or group.replace('_', ' ')), 'native'
        place = ' · '.join(p for p in [(phase or {}).get('name'), (hist or {}).get('name') or fallback] if p)
        native_status = record.get('native_status')
        estimated = native_status in {'refined_independent', 'refined_dependent'}
        rows.append(dict(key='execution|'+variable,kind=group,parameter=family,
            phase=(phase or {}).get('name'),histogram=(hist or {}).get('name'),
            histogram_key=(hist or {}).get('key'),label=' · '.join(p for p in [place,label] if p),
            unit=unit,value=value,sigma=sigma if estimated else None,
            status='refined_in_prior_stage' if prior else 'refined' if estimated else 'configured',
            native_status=native_status,native_variable=variable,refined_stage=refined_stage,
            evidence='Selected-stage execution receipt',evidence_stage=current['id']))
    notice = ('Additional scale, background and texture values come from the verified selected-stage execution receipt. '
              'Their s.u. values are conditional native uncertainties. “Native” units are not a calibrated physical quantity.') if rows else ''
    if omitted:
        notice += ' Prior-stage nuisance estimates without an unchanged held value recorded in this stage are omitted.'
    return rows, notice.strip()


def parameter_table(view, supplements=()):
    """Copy cached rows so presentation cannot mutate the verified artifact."""
    rows = deepcopy(view.get('parameters', []))
    keys = {p.get('key') for p in rows}
    native = {p.get('native_variable') for p in rows if p.get('native_variable')}
    rows.extend(deepcopy(p) for p in supplements
                if p.get('key') not in keys and p.get('native_variable') not in native)
    return rows
