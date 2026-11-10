"""Closed outer-worker admission; the shared numerical package is unchanged.

The web process validates paths and the receipt only. ``execute`` runs in the
Model worker and delegates physical comparisons to the existing native guard.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

OPERATION = 'structural-library-admission'
SCHEMA = 'radar-structural-library-admission/v1'
MAX_CANDIDATES = 500
PATH_FIELDS = ('source', 'parent', 'instrument', 'model', 'weights')
MAX_BYTES = {'source': 128*1024**2, 'parent': 10*1024**2, 'instrument': 1024**2,
             'model': 2*1024**2, 'weights': 32*1024**2, 'reference': 128*1024**2,
             'child': 10*1024**2}


def vector(value):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError('Select one rational parent-basis k vector')
    result = []
    for item in value:
        if not isinstance(item, str) or len(item) > 40:
            raise ValueError('A bounded rational k vector is required')
        number = Fraction(item)
        if number.denominator > 96 or abs(number) > 8:
            raise ValueError('The selected k vector exceeds the supported rational bound')
        result.append(str(number))
    return result


def validate_payload(payload):
    required = set(PATH_FIELDS) | {'children', 'phase', 'histogram', 'k', 'grid', 'destination', 'library_id'}
    if (not isinstance(payload, dict) or required-set(payload)
            or set(payload)-required-{'reference'}):
        raise ValueError('Unexpected structural-library admission fields')
    for key in (*PATH_FIELDS, 'phase', 'histogram', 'destination', *(['reference'] if 'reference' in payload else [])):
        if not isinstance(payload[key], str) or not 0 < len(payload[key]) <= 4096:
            raise ValueError('A bounded admission path or native selector is required')
    if (not isinstance(payload['library_id'], str)
            or not re.fullmatch('[a-f0-9]{24}', payload['library_id'])):
        raise ValueError('Admission must bind one exact preparation library')
    children = payload['children']
    if (not isinstance(children, list) or not 1 <= len(children) <= MAX_CANDIDATES
            or any(not isinstance(p, str) or not 0 < len(p) <= 4096 for p in children)
            or len(set(children)) != len(children)):
        raise ValueError('Admission needs the complete bounded child-CIF inventory')
    vector(payload['k'])
    from gpx_recipe.preparation.reference import validate_grid
    if not isinstance(payload['grid'], dict):
        raise ValueError('Admission needs the captured training bank and grid')
    validate_grid(**payload['grid'])
    return payload


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def request_identity(payload):
    for key in (*PATH_FIELDS, *(['reference'] if 'reference' in payload else [])):
        path = Path(payload[key])
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_BYTES[key]:
            raise ValueError('A bounded regular admission input is required: '+key)
    for name in payload['children']:
        path = Path(name)
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_BYTES['child']:
            raise ValueError('A bounded regular child CIF is required')
    return dict(library_id=payload['library_id'], k=vector(payload['k']),
        phase=payload['phase'], histogram=payload['histogram'], grid=payload['grid'],
        files={key: file_hash(payload[key]) for key in PATH_FIELDS},
        children=[file_hash(path) for path in payload['children']],
        **({'reference_sha256': file_hash(payload['reference'])} if 'reference' in payload else {}))


def validate_receipt(result, payload):
    if (not isinstance(result, dict) or result.get('schema') != SCHEMA
            or result.get('adapter_sha256') != file_hash(__file__)
            or result.get('identity') != request_identity(payload)
            or type(result.get('compatible')) is not bool
            or result.get('no_refinement') is not True or result.get('inputs_unchanged') is not True
            or not isinstance(result.get('reason'), str) or len(result['reason']) > 1000):
        raise ValueError('Native admission did not bind the selected library and current source')
    if not result['compatible']:
        raise ValueError('Selected prepared library is incompatible: '+result['reason'])
    from gpx_recipe.preparation.structural_ranking_hints import _metadata
    _, candidates = _metadata(Path(payload['model']))
    expected = [dict(id=c['id'], child_sha256=c['child']['sha256'],
        modes_sha256=hashlib.sha256(json.dumps(c['modes'], separators=(',', ':')).encode()).hexdigest()) for c in candidates]
    checks = result.get('candidate_checks')
    if (result.get('k_verified') is not True or result.get('mode_inventory_verified') is not True
            or not isinstance(checks, list) or len(checks) != len(expected)
            or any(not isinstance(row, dict) or {key:row.get(key) for key in item} != item
                   or not isinstance(row.get('mode_k_vectors'), list) or not row['mode_k_vectors']
                   or row.get('eligibility') not in {'eligible','excluded'}
                   or not isinstance(row.get('eligibility_reason'),str) or not 1<=len(row['eligibility_reason'])<=500
                   for row, item in zip(checks, expected))):
        raise ValueError('Native admission omitted the complete candidate/k verification')
    eligible = [row['id'] for row in checks if row['eligibility']=='eligible']
    excluded = [row['id'] for row in checks if row['eligibility']=='excluded']
    if (not eligible or result.get('eligible_candidate_ids')!=eligible
            or result.get('excluded_candidate_ids')!=excluded
            or result.get('candidate_ids')!=[c['id'] for c in candidates]):
        raise ValueError('Native admission changed the complete selected-k coverage')
    return result


def _native_k(labels, requested, operations):
    """Read numerical wavevectors from actual ISODISTORT mode records."""
    from gpx_recipe.commensurate_k import reciprocal_equivalent
    vectors = []
    for label in labels:
        matches = re.findall(r'\[([+\-0-9./ ]+,[+\-0-9./ ]+,[+\-0-9./ ]+)\]', str(label))
        if len(matches) != 1:
            raise ValueError('The exact child lacks an unambiguous numerical mode wavevector')
        vectors.append(vector([part.strip() for part in matches[0].split(',')]))
    if not vectors:
        raise ValueError('The retained child has no explicit native mode wavevectors')
    gamma = ['0', '0', '0']; nonzero=[]
    for wave in vectors:
        if not reciprocal_equivalent(wave,gamma,operations) and not any(
                reciprocal_equivalent(wave,seen,operations) for seen in nonzero):nonzero.append(wave)
    if len(nonzero)>1:
        status,reason='excluded','Unsupported mixed-k construction: more than one non-Gamma k star'
    elif reciprocal_equivalent(requested,gamma,operations):
        status,reason=('eligible','All native modes belong to Gamma') if not nonzero else (
            'excluded','Native modes belong to a non-Gamma k star outside the requested Gamma scope')
    elif not nonzero:
        status,reason='excluded','Gamma-only candidate is outside the requested nonzero k scope; no null-control role is declared'
    elif reciprocal_equivalent(nonzero[0],requested,operations):
        status,reason='eligible','Native modes match the requested k star; Gamma secondary modes are retained'
    else:
        status,reason='excluded','Native modes belong to a different k star'
    return dict(mode_k_vectors=vectors,eligibility=status,eligibility_reason=reason)


def execute(payload):
    """Native-only, no fit and no exported observations or GPX objects."""
    validate_payload(payload)
    from gpx_recipe.execution import write_json
    from gpx_recipe.preparation.structural_ranking_hints import _metadata
    from gpx_recipe.preparation.structural_domain import maximum_parent_metric
    from gpx_recipe.preparation.reference import create_reference
    from gpx_recipe.preparation.structural_ranking_native import require_compatible, require_unconstrained_project, observed_features
    from gpx_recipe.preparation.structural_ranker import Ranker
    from gpx_recipe.preparation.structural_features import pattern_features
    from gpx_recipe.snapshot import gsas
    from gpx_recipe.symmetry.native_audit import inspect_source
    from gsas_isodistort_import import add_isodistort_phase
    before = request_identity(payload)
    result = dict(schema=SCHEMA, adapter_sha256=file_hash(__file__), identity=before, compatible=False, no_refinement=True,
        inputs_unchanged=False, reason='', k_verified=False, mode_inventory_verified=False)
    output = Path(payload['destination']); output.mkdir(parents=True, exist_ok=False)
    try:
        metadata, candidates = _metadata(Path(payload['model']))
        if len(candidates) != len(payload['children']):
            raise ValueError('The native child inventory differs from the complete trained scope')
        if (metadata['parent_sha256'] != before['files']['parent']
                or metadata['instrument_sha256'] != before['files']['instrument']
                or metadata['weights_sha256'] != before['files']['weights']):
            raise ValueError('The model differs from its captured parent, profile or weights')
        maximum = maximum_parent_metric(metadata)
        from gpx_recipe.preparation.structural_attention_weights import FIXED_REFERENCE_KIND
        if metadata.get('reference_kind') == FIXED_REFERENCE_KIND and 'reference' not in payload:
            raise ValueError('This trained model requires a before-transition reference GPX before discovery')
        # Loading the bounded numerical model validates its exact grid/weights;
        # no inference or optimizer is run during admission.
        ranker = Ranker(Path(payload['model']).parent, metadata_name=Path(payload['model']).name,
            weights_name=Path(payload['weights']).name, threads=1)
        G = gsas()
        prepared = create_reference(payload['parent'], payload['instrument'], output/'reference', **payload['grid'])
        reference = G.G2Project(gpxfile=prepared['path'])
        expected_parent = reference.phase(prepared['parent_phase']); expected_hist = reference.histogram(prepared['histogram'])
        context = inspect_source(prepared['path'], include_runtime=False)
        distances = []

        def check(source, phase, histogram=None):
            project = G.G2Project(gpxfile=source); require_unconstrained_project(project)
            current = project.phase(phase)
            histograms = [project.histogram(histogram)] if histogram else project.histograms()
            accepted = []
            reasons = []
            for hist in histograms:
                if not current.data.get('Histograms', {}).get(hist.name, {}).get('Use'): continue
                try:
                    distance = require_compatible(current, hist, expected_parent, expected_hist, maximum_metric=maximum)
                    active = [p for p in project.phases() if p.data.get('Histograms', {}).get(hist.name, {}).get('Use')]
                    if len(active) != 1 or active[0].name != current.name:
                        raise ValueError('Prepared ranking needs one active nuclear phase')
                    features = observed_features(source, hist)
                    projected = pattern_features(features['q'], features['intensity'], ranker.edges,
                        background=features['background'], valid=features['valid'])
                    if (projected[2] < .999).any():
                        raise ValueError('The current native grid does not cover the complete trained Q range')
                    accepted.append((hist, distance))
                except ValueError as error: reasons.append(str(error))
            if len(accepted) != 1:
                raise ValueError('No unique compatible native bank: '+('; '.join(reasons)[:700] or 'parent is inactive'))
            distances.append(accepted[0][1])
            return accepted[0][0].name

        current_hist = check(payload['source'], payload['phase'], payload['histogram'])
        before_hist = check(payload['reference'], payload['phase']) if 'reference' in payload else None
        candidate_checks = []
        for index, (candidate, child) in enumerate(zip(candidates, payload['children'])):
            if candidate['child']['sha256'] != before['children'][index]:
                raise ValueError('The retained child CIF differs from its exact trained candidate')
            project = G.G2Project(newgpx=str(output/f'child-{index}.gpx'))
            child_phase, receipt = add_isodistort_phase(project, child, allow_strain_only=True)
            modes = [mode.name for mode in child_phase.data['ISODISTORT']['G2ModeList']]
            if modes != candidate['modes']:
                raise ValueError('The retained child native mode inventory differs from the trained candidate')
            eligibility = _native_k(receipt['source_mode_labels'], payload['k'], context['phases'][0]['operations'])
            candidate_checks.append(dict(id=candidate['id'], child_sha256=before['children'][index],
                modes_sha256=hashlib.sha256(json.dumps(modes, separators=(',', ':')).encode()).hexdigest(),
                **eligibility))
        eligible=[row['id'] for row in candidate_checks if row['eligibility']=='eligible']
        excluded=[row['id'] for row in candidate_checks if row['eligibility']=='excluded']
        result.update(compatible=bool(eligible), candidate_ids=[c['id'] for c in candidates],
            eligible_candidate_ids=eligible,excluded_candidate_ids=excluded,
            k_verified=True, mode_inventory_verified=True, candidate_checks=candidate_checks,
            parent_setting_sha256=context['phases'][0]['setting_sha256'],
            maximum_parent_metric=maximum, actual_parent_metrics=distances,
            histogram=current_hist, reference_histogram=before_hist,
            reason=(f'Native context matched; {len(eligible)} of {len(candidates)} stored candidates are eligible for the selected k; '
                    f'{len(excluded)} excluded with native reasons' if eligible else
                    f'No stored candidates are eligible for the selected k; all {len(candidates)} checked and excluded with native reasons'))
    except (ValueError, KeyError, TypeError, IndexError, OSError) as error:
        result['reason'] = str(error)[:1000]
    finally:
        result['inputs_unchanged'] = request_identity(payload) == before
        if not result['inputs_unchanged']: result.update(compatible=False, reason='An admission input changed during native verification')
        write_json(output/'admission.json', result)
    return result


def main():
    import sys
    from gpx_recipe.execution import write_json
    write_json(Path(sys.argv[2]), execute(json.loads(Path(sys.argv[1]).read_text())))


if __name__ == '__main__': main()
