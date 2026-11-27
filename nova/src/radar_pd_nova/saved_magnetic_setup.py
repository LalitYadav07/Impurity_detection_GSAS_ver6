"""Load declarative magnetic setup from one selected, recovered campaign.

No solver context, result or approval crosses this boundary. The current
builder and native BNS/operator audit remain the execution authorities.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import uuid

from gpx_recipe.symmetry.chemistry import validate_sites
from gpx_recipe.symmetry.compiler import assets, confined, file_sha, verify_inputs
from gpx_recipe.symmetry.contracts import digest, validate_candidate
from gpx_recipe.symmetry.discovery import magnetic_request, operation_text
from gpx_recipe.symmetry.templates import matches, validate_template

MAX_METADATA = 8 * 1024 * 1024


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_METADATA:
        raise ValueError('The saved magnetic setup is missing, linked or oversized; load its campaign again.')
    return path.read_bytes()


def _asset(root, relative):
    root = Path(root).resolve()
    path = root / relative
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError('The saved magnetic setup contains a linked input.')
    return confined(root, relative)


def _parent(context, name):
    rows = [p for p in context.get('phases', []) if p.get('name') == name]
    if len(rows) != 1 or rows[0].get('type') != 'nuclear':
        raise ValueError('The saved magnetic parent is absent or ambiguous in the inspected GPX.')
    return rows[0]


def _operations(parent):
    values = [operation_text(op) for op in parent.get('operations', [])]
    if not values or len(set(values)) != len(values):
        raise ValueError('The magnetic parent needs a complete unique operation inventory.')
    return sorted(values)


def load(campaign, *, context, source, origin, draft_root):
    """Return a verified setup from immutable inputs of this exact source.

    ``campaign`` has already been selected in the current History. Derive its
    input directory from the sealed protocol; never follow a checkpoint's
    arbitrary reference_input_root pathname.
    """
    template = deepcopy(validate_template(campaign.setting('template')))
    protocol = template['protocol']
    if protocol['kind'] != 'magnetic':
        raise ValueError('Select a saved magnetic campaign to reuse its setup.')
    if (context.get('sha256') != protocol['source']['sha256']
            or source.get('source_sha256') != context.get('sha256')
            or source.get('id') not in template['seen_ids']
            or not matches(template, source, origin)):
        raise ValueError('The saved magnetic setup belongs to another source, sample, bank or recipe checkpoint.')
    current_source = _asset(draft_root, 'source.gpx')
    if current_source.is_symlink() or file_sha(current_source) != context['sha256']:
        raise ValueError('The freshly inspected source GPX changed.')
    root = Path(campaign.root).resolve()
    inputs = root / 'inputs' / digest(protocol)
    if inputs.is_symlink() or not inputs.is_dir() or not inputs.resolve().is_relative_to(root):
        raise ValueError('The selected campaign inputs are unavailable; load the saved campaign first.')
    saved_protocol = json.loads(_read(_asset(root, str(Path('inputs') / digest(protocol) / 'protocol.json'))))
    if saved_protocol != protocol:
        raise ValueError('The selected campaign protocol changed after recovery.')
    for asset in assets(protocol):
        _asset(root, str(Path('inputs') / digest(protocol) / asset['path']))
    verify_inputs(protocol, inputs)
    path = _asset(inputs, protocol['catalog']['path'])
    raw = _read(path)
    if hashlib.sha256(raw).hexdigest() != protocol['catalog']['sha256']:
        raise ValueError('The saved magnetic catalog changed.')
    catalog = json.loads(raw)
    old_parent = _parent(template['context'], protocol['parent_phase'])
    parent = _parent(context, protocol['parent_phase'])
    if (parent['setting_sha256'] != protocol['parent_setting_sha256']
            or old_parent['setting_sha256'] != parent['setting_sha256']
            or _operations(old_parent) != _operations(parent)):
        raise ValueError('The saved magnetic catalog has a different parent setting or operation inventory.')
    histogram = protocol['histogram']
    if (histogram not in parent['histograms']
            or len([h for h in context['histograms'] if h['name'] == histogram]) != 1):
        raise ValueError('The saved magnetic histogram is absent or ambiguous in the inspected source.')
    validate_sites(old_parent['atoms'], protocol['sites'])
    validate_sites(parent['atoms'], protocol['sites'])
    for site in protocol['sites']:
        old = next(a for a in old_parent['atoms'] if a['label'] == site['label'])
        new = next(a for a in parent['atoms'] if a['label'] == site['label'])
        if any(old.get(key) != new.get(key) for key in ('type', 'occupancy', 'xyz')):
            raise ValueError('A saved magnetic site changed in the inspected source.')
    coverage = catalog.get('coverage', {})
    candidates = catalog.get('candidates', [])
    if (coverage.get('discovery_complete') is not True
            or type(coverage.get('constructed_count')) is not int
            or coverage['constructed_count'] != len(candidates) or not 0 < len(candidates) <= 2000):
        raise ValueError('The saved magnetic discovery scope is incomplete.')
    if len({c['id'] for c in candidates}) != len(candidates):
        raise ValueError('The saved magnetic catalog contains duplicate candidates.')
    for candidate in candidates:
        checked = validate_candidate(candidate)
        if (not checked['magnetic_space_group'] or checked['child'] or checked['modes']
                or not re.fullmatch(r'\d+\.\d+', checked.get('bns_number') or '')):
            raise ValueError('A saved magnetic candidate lacks its declared BNS identity.')
    provenance = catalog.get('provenance', {})
    request = provenance.get('request', {})
    star = request.get('inputs', {}).get('starmagnetica')
    if (star not in {'yes', 'no'} or not re.fullmatch(r'[a-f0-9]{64}', provenance.get('output_sha256', ''))
            or request != magnetic_request(parent, protocol['k'], whole_star=star == 'yes')):
        raise ValueError('The saved catalog is not bound to the current complete parent operations and k vector.')
    settings = dict(parent=protocol['parent_phase'], histogram=histogram, k=protocol['k'],
        sites=deepcopy(protocol['sites']), candidates=[c['id'] for c in protocol['candidates']],
        directions=list(dict.fromkeys(s['axis'] for s in protocol['seeds'])),
        relative_sign_starts=any('site_signs' in s for s in protocol['seeds']),
        **{key:protocol['policy'].get(key, default) for key, default in {
            'moment_model':'collinear', 'magnetic_optimizer':'gsas_hessian',
            'optimizer_iterations':120, 'refine_histogram_scale':False, 'refine_background':False,
            'magnetic_d_min':0., 'maximum_moment':6., 'adequate_rwp_percent':12., 'cycles':20}.items()})
    binding = dict(source_sha256=context['sha256'], parent=settings['parent'], histogram=histogram,
        parent_setting_sha256=parent['setting_sha256'], parent_operations_sha256=digest(_operations(parent)),
        k=deepcopy(settings['k']), sites=deepcopy(settings['sites']), catalog_sha256=protocol['catalog']['sha256'],
        campaign=root.name, old_protocol_sha256=digest(protocol),
        reused_results=False, reused_approval=False, native_operator_audit_required=True)
    return dict(catalog=catalog, catalog_bytes=raw, settings=settings, binding=binding)


def stage(root, setup):
    """Copy verified catalog bytes into an owned fresh draft, preserving hash."""
    root = Path(root)
    raw = setup['catalog_bytes']
    if hashlib.sha256(raw).hexdigest() != setup['binding']['catalog_sha256']:
        raise ValueError('The saved magnetic catalog changed before staging.')
    temporary = root / ('catalog-' + uuid.uuid4().hex + '.pending')
    try:
        with temporary.open('xb') as stream:
            stream.write(raw)
        temporary.replace(root / 'catalog.json')
    finally:
        temporary.unlink(missing_ok=True)


def verify_binding(draft, settings):
    """A later form edit cannot silently extend the saved site's scope."""
    binding = draft.get('saved_magnetic_setup')
    if not binding:
        return
    if (draft['context'].get('sha256') != binding['source_sha256']
            or file_sha(Path(draft['root']) / 'source.gpx') != binding['source_sha256']
            or settings['kind'] != 'magnetic' or settings['parent'] != binding['parent']
            or settings['histogram'] != binding['histogram'] or settings['k'] != binding['k']
            or settings['sites'] != binding['sites']
            or file_sha(Path(draft['root']) / 'catalog.json') != binding['catalog_sha256']):
        raise ValueError('The saved magnetic setup source, sites or catalog changed; reuse it again or explicitly discover a new catalog.')
    parent = _parent(draft['context'], binding['parent'])
    if (parent['setting_sha256'] != binding['parent_setting_sha256']
            or digest(_operations(parent)) != binding['parent_operations_sha256']):
        raise ValueError('The inspected magnetic parent changed; inspect and reuse the saved setup again.')
