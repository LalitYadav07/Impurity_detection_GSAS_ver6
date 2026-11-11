"""Use exact stored structural candidates after a native compatibility check."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

from gpx_recipe.execution import write_json
from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.symmetry.contracts import digest
from .model_admission import OPERATION, validate_receipt, vector


class PreparedLibraryUnavailable(RuntimeError):
    """An explicitly selected ranker must not silently become unprepared work."""
    def __init__(self,message,*,candidate_checks=()):
        super().__init__(message)
        self.candidate_checks=list(candidate_checks)


def reference_notice(kind, *, has_reference=False):
    """Describe the stored model's inputs without changing its inference."""
    from gpx_recipe.preparation.structural_attention_weights import REFERENCE_KIND, FIXED_REFERENCE_KIND
    if kind == FIXED_REFERENCE_KIND:
        return ('ML reference: the selected before-transition measured scan.' if has_reference else
                'ML reference: this model requires a before-transition measured scan.')
    if kind == REFERENCE_KIND:
        detail = (' The selected before-transition GPX supplies structure parameters for fitting; '
                  'its measured curve is not used by this stored model.' if has_reference else
                  ' No measured before-transition curve is used by this stored model.')
        return 'ML reference: a calculated parent pattern.' + detail
    return ('This older model does not record its ML reference type. '
            'Do not interpret its ranking as a measured before/after comparison.')


def selected_library(app):
    workspace = getattr(app, '_preparation_workspace', None)
    if workspace is None or not workspace.registry.selected_key(): return None
    library = workspace.registry.get(); library.refresh()
    if not any(row['model']['kind'] in {'structural-training', 'structural-ranking-model'}
               for row in library.document['entries']): return None
    return library


def require_capability(runtime):
    """A registered action proves the outer adapter exists; never assume latest."""
    import requests
    from urllib.parse import quote
    service = runtime.service
    response = requests.get(service.galaxy_url+'/api/tools/'+quote(runtime.tool_id, safe=''),
        headers=service._headers, params={'io_details': 'true'}, timeout=30, allow_redirects=False)
    response.raise_for_status(); document = response.json()
    if not isinstance(document, dict) or document.get('id') != runtime.tool_id:
        raise PreparedLibraryUnavailable('Galaxy did not return the exact registered Model tool contract.')
    def actions(node):
        if isinstance(node, dict):
            if node.get('name') == 'action':
                for option in node.get('options', []):
                    if isinstance(option, dict): yield option.get('value')
                    elif isinstance(option, (list, tuple)) and len(option) >= 2: yield option[1]
            for value in node.values(): yield from actions(value)
        elif isinstance(node, list):
            for value in node: yield from actions(value)
    if 'admission' not in set(actions(document.get('inputs', []))):
        raise PreparedLibraryUnavailable('The registered Model worker does not support prepared-library admission. '
            'Use a qualified admission-capable worker or explicitly deselect the library for ordinary discovery.')


def stage_models(library, root):
    """Verify with the shared portable reader; preserve every candidate byte."""
    from gpx_recipe.preparation.structural_ranking_hints import _metadata, capture, checked_files
    library.refresh(); definitions = {}
    rows = [r for r in library.document['entries']
            if r['model']['kind'] in {'structural-training', 'structural-ranking-model'}]
    if not rows or len(rows) > 8 or any(row['state'] != 'prepared' for row in rows):
        raise PreparedLibraryUnavailable('Select a completed structural model library with at most eight bank contexts.')
    for row in rows:
        assets = [a['path'] for a in row['files'] if Path(a['path']).name == 'model.json']
        if len(assets) != 1: raise PreparedLibraryUnavailable('The selected library has no unique captured model metadata.')
        paths = library.verify_assets(assets)
        _, candidates = _metadata(paths[assets[0]])
        for candidate in candidates:
            old = definitions.setdefault(candidate['id'], candidate)
            if old != candidate: raise PreparedLibraryUnavailable('The selected library has conflicting candidate definitions.')
    protocol = dict(kind='structural', candidates=list(definitions.values()))
    inventory = capture(library, library.document, protocol, root)
    if inventory['omitted_entries'] or len(inventory['entries']) != len(rows):
        raise PreparedLibraryUnavailable('The complete selected model scope exceeds the portable capture budget.')
    original_catalogs = []
    scope_name = library.document['specification'].get('scope')
    if scope_name:
        scope_path = library.verify_assets([scope_name])[scope_name]
        if scope_path.stat().st_size > 4*1024*1024:
            raise PreparedLibraryUnavailable('The retained discovery scope is oversized.')
        scope = json.loads(scope_path.read_text())
        for search in scope.get('searches', []):
            catalog = search.get('catalog')
            if catalog is not None:
                if search.get('catalog_sha256') != digest(catalog):
                    raise PreparedLibraryUnavailable('A retained authoritative catalog changed.')
                original_catalogs.append(catalog)
    result = []
    for row in inventory['entries']:
        files = checked_files(root, row, protocol)
        metadata, candidates = _metadata(files['model'])
        children = []
        for index, candidate in enumerate(candidates):
            names = [a['path'] for a in library.document['inputs']
                     if a['sha256'] == candidate['child']['sha256'] and Path(a['path']).suffix.lower() == '.cif']
            if not names: raise PreparedLibraryUnavailable('An exact trained child CIF is absent from the selected library.')
            source = library.verify_assets([sorted(names)[0]])[sorted(names)[0]]
            # Different mode constructions can legitimately share child bytes.
            # Keep one positional input per candidate for native inventory checks.
            target = Path(root)/'children'/digest({'entry': row['entry']})/(f'{index:04d}-'+candidate['child']['sha256']+'.cif')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists(): shutil.copyfile(source, target)
            if file_sha(source) != candidate['child']['sha256'] or file_sha(target) != candidate['child']['sha256']:
                raise PreparedLibraryUnavailable('A captured trained child CIF changed.')
            children.append(target)
        from gpx_recipe.preparation.structural_hints import candidate_identity
        identities = {candidate_identity(c) for c in candidates}
        original = [c for c in original_catalogs if c.get('candidates')
                    and {candidate_identity(v) for v in c['candidates']} <= identities]
        result.append(dict(row=row, files=files, candidates=candidates, children=children,
            original_catalogs=original,
            reference_kind=metadata.get('reference_kind'),
            library_id=library.document['id'], library_name=library.document['name']))
    return result


def discover(app, draft, parent, k, progress):
    library = selected_library(app)
    if library is None: return None
    if app.server.state.sx_opd_policy != 'all':
        raise PreparedLibraryUnavailable('Choose All discovered OPDs to reuse the complete stored training candidate scope.')
    remote = app._model_remote()
    if remote is not None: require_capability(remote)
    root = Path(draft['root'])
    binding = dict(source=draft['context']['sha256'], reference=draft.get('reference_asset'),
        library=file_sha(library.path), library_id=library.document['id'], k=vector(k),
        phase=app.server.state.sx_parent, histogram=app.server.state.sx_histogram)
    from . import model_admission
    binding['adapter_sha256'] = file_sha(model_admission.__file__)
    directory = root/'prepared-library'/digest(binding)
    models = stage_models(library, directory)
    accepted = []; reasons = []; checked_candidates=[]
    for model in models:
        row = model['row']; files = model['files']
        payload = dict(source=str(root/'source.gpx'), parent=str(files['parent']),
            instrument=str(files['instrument']), model=str(files['model']), weights=str(files['weights']),
            children=[str(p) for p in model['children']], phase=app.server.state.sx_parent,
            histogram=app.server.state.sx_histogram, k=vector(k), grid=row['grid'],
            library_id=model['library_id'], destination=str(directory/('admission-'+row['entry'])))
        if draft.get('reference_asset'):
            payload['reference'] = str(root/draft['reference_asset']['path'])
        progress('Checking the selected library against the current native parent, bank, cell and stored modes. No fit has started.')
        cache = directory/('admission-'+row['entry']+'.json')
        if cache.is_file():
            if cache.is_symlink() or cache.stat().st_size > 4*1024*1024:
                raise PreparedLibraryUnavailable('The cached native admission receipt is invalid.')
            result = json.loads(cache.read_text())
        else:
            result = app._symmetry_native(OPERATION, payload, root)
            write_json(cache, result)
        checked_candidates.extend(result.get('candidate_checks',[]))
        try: validate_receipt(result, payload)
        except ValueError as error: reasons.append(str(error)); continue
        if result['parent_setting_sha256'] != parent['setting_sha256']:
            raise PreparedLibraryUnavailable('The selected library admission returned another parent setting.')
        accepted.append((model, result))
    if len(accepted) != 1:
        raise PreparedLibraryUnavailable('The selected library has no unique compatible bank/model: '+
            ('; '.join(reasons)[:1000] if not accepted else 'multiple contexts match; select a single-context library.'),
            candidate_checks=checked_candidates)
    model, receipt = accepted[0]
    eligible=set(receipt['eligible_candidate_ids'])
    candidates = []
    for old, child in zip(model['candidates'], model['children']):
        if old['id'] not in eligible:continue
        candidate=copy.deepcopy(old)
        candidate['child']['path'] = str(child.relative_to(root))
        candidates.append(candidate)
    if len(candidates)>int(app.server.state.sx_construct_limit):
        raise PreparedLibraryUnavailable('The eligible selected-k scope exceeds the construction limit; increase it explicitly.',
            candidate_checks=receipt['candidate_checks'])
    provenance = dict(library_id=model['library_id'], library_name=model['library_name'],
        entry_id=model['row']['entry'], model_sha256=model['row']['files']['model']['sha256'],
        reference_kind=model.get('reference_kind'),
        candidate_scope_sha256=digest(model['candidates']), admission=receipt,
        scope='Reconstructed from native-eligible exact stored training candidates for the selected k; full model retained')
    catalog = dict(schema='radar-symmetry-catalog/v1', parent_setting_sha256=parent['setting_sha256'],
        k=vector(k), candidates=candidates,
        coverage=dict(discovery_complete=True, discovered_count=len(candidates), constructed_count=len(candidates),
            opd_policy='all', scope=provenance['scope'],
            stored_count=len(model['candidates']),eligible_count=len(candidates),
            excluded_count=len(receipt['excluded_candidate_ids']),stored_candidates=receipt['candidate_checks'],
            candidates=[dict(id=c['id'], label=c.get('irrep') or c['id'], state='constructed',
                reason='Exact stored training construction; native admission passed') for c in candidates]),
        provenance=dict(prepared_library=provenance))
    from gpx_recipe.commensurate_k import reciprocal_equivalent
    original=[]
    for item in model['original_catalogs']:
        same_k=vector(item['k'])==vector(k)
        if not same_k and parent.get('operations'):
            same_k=reciprocal_equivalent(item['k'],k,parent['operations'])
        if same_k and {c['id'] for c in item['candidates']}==eligible:original.append(item)
    if len({digest(c) for c in original})>1:
        raise PreparedLibraryUnavailable('Several retained original catalogs match the selected k; select an unambiguous library.')
    if original:
        original=original[0]
        from .preparation_discovery import require_structural_coverage
        require_structural_coverage(original)
        if original['parent_setting_sha256'] != parent['setting_sha256']:
            raise PreparedLibraryUnavailable('The original catalog belongs to another parent setting.')
        provenance.update(original_catalog=copy.deepcopy(original), original_catalog_sha256=digest(original),
            scope='Exact captured authoritative catalog for the selected k and child constructions; full model retained')
    write_json(directory/'catalog.json', catalog)
    return catalog


def required_attachment(protocol, root, document):
    """A declared prepared catalog is a requirement, not an optional speedup."""
    if protocol.get('kind') != 'structural' or not protocol.get('catalog'): return
    from gpx_recipe.symmetry.compiler import confined
    try:
        catalog_path = confined(root, protocol['catalog']['path'])
        if (catalog_path.is_symlink() or catalog_path.stat().st_size > 8*1024*1024
                or protocol['catalog'].get('sha256') not in (None, file_sha(catalog_path))):
            raise ValueError('changed catalog')
        catalog = json.loads(catalog_path.read_text())
        if not isinstance(catalog, dict):raise ValueError('invalid catalog')
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise PreparedLibraryUnavailable('The captured structural catalog is missing, corrupt or changed; '
            'recover its original inputs before resuming.') from error
    provenance = catalog.get('provenance', {})
    if not isinstance(provenance, dict):
        raise PreparedLibraryUnavailable('The captured structural catalog has invalid provenance metadata.')
    marker = provenance.get('prepared_library')
    if not marker: return
    if (not isinstance(marker, dict) or any(not isinstance(marker.get(key), str)
            for key in ('library_id', 'entry_id', 'model_sha256'))):
        raise PreparedLibraryUnavailable('The captured prepared-library requirement is invalid.')
    entries = (document or {}).get('ranking_models', {}).get('entries', [])
    match = [r for r in entries if r['entry'] == marker['entry_id']
             and r['files']['model']['sha256'] == marker['model_sha256']]
    if (not document or document.get('library_id') != marker['library_id'] or len(match) != 1
            or set(match[0]['routes'].values()) != {c['id'] for c in protocol['candidates']}):
        raise PreparedLibraryUnavailable('The explicitly admitted trained model could not be attached unchanged; '
            'select and recheck the original library before starting this plan.')
    from gpx_recipe.preparation.structural_ranking_hints import checked_files
    try:checked_files(root, match[0], protocol)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise PreparedLibraryUnavailable('The explicitly admitted model files are missing, corrupt or changed; '
            'recover the original prepared inputs before resuming.') from error
