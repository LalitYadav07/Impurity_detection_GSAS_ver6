"""Shared admission and publication boundary for preparation clients.

Scheduling and Galaxy ownership stay with the caller. This module stages the
same immutable scientific request for every client and publishes only the
declared data artifacts after checking the exact input and output receipts.
"""
from __future__ import annotations

import json
import heapq
from pathlib import Path
import shutil
import tempfile

from ..symmetry.compiler import file_sha
from ..symmetry.contracts import digest, fields
from .library import PreparationLibrary, confined, relative
from .worker import validate_request, validate_settings, verify_structural_child

PLAN_SCHEMA = 'radar-experiment-preparation-plan/v1'
PORTABLE = {'reference': set(), 'magnetic': {'response.npz'},
            'structural': {'starts.npz', 'starts.json'},'kernel':{'kernel.npz','kernel.json'},
            'structural-examples':{'examples.json','examples.npz'},
            'structural-training':{'model/model.json','model/weights.npz','training.json'}}
CHILD_KINDS={'structural','structural-examples'}
OPTIONAL_PORTABLE = {'magnetic': {'scattering.npz'},'reference':{'profile.npz','profile.json'}}


def portable_artifacts(kind, names):
    """Closed inventories, preserving response-only legacy magnetic entries."""
    if kind not in PORTABLE or not isinstance(names,(list,set,dict,tuple)):
        raise ValueError('Unknown portable preparation inventory')
    if any(not isinstance(name,str) for name in names):
        raise ValueError('Portable artifact names must be strings')
    inventory=set(names)
    if (len(inventory)!=len(names) or not PORTABLE[kind] <= inventory
            or not inventory <= PORTABLE[kind] | OPTIONAL_PORTABLE.get(kind,set())):
        raise ValueError('Unexpected portable preparation artifact inventory')
    if kind=='reference' and inventory not in (set(),{'profile.npz','profile.json'}):
        raise ValueError('TOF reference preparation requires both profile arrays and metadata')
    return inventory


def portable_entry_artifacts(kind, manifest):
    """One artifact/identity contract for workers, publication and import."""
    if not isinstance(manifest,dict) or not isinstance(manifest.get('artifacts'),dict):
        raise ValueError('A preparation needs its declared portable artifacts')
    names=portable_artifacts(kind,manifest['artifacts'])
    if 'profile.npz' in names:
        from .tof_kernel_arrays import checksum
        detail=manifest.get('detail');profile=detail.get('profile_table') if isinstance(detail,dict) else None
        if (not isinstance(profile,dict) or profile.get('status')!='prepared'
                or profile.get('role')!='starting_proposals' or profile.get('observations_included') is not False
                or profile.get('native_likelihood_authority') is not False
                or not checksum(detail.get('physics_sha256'))
                or detail['physics_sha256']!=profile.get('physics_sha256')):
            raise ValueError('Portable TOF tables need their proposal-only physical identity')
    if 'scattering.npz' in names:
        import re
        detail=manifest.get('detail')
        scattering=detail.get('scattering') if isinstance(detail,dict) else None
        if (not isinstance(scattering,dict) or scattering.get('status')!='prepared'
                or scattering.get('artifact')!='scattering.npz'
                or not isinstance(scattering.get('identity'),str)
                or not re.fullmatch('[a-f0-9]{64}',scattering['identity'])):
            raise ValueError('Portable scattering factors need their native physical identity')
    return names


def inherit_reference(library,inspection,identifier):
    """Carry an immutable completed reference into its dependent model ZIP.

    This copies the original manifest/arrays and origin routing references;
    it performs no new native calculation and grants no new native authority.
    Ordinary publication rechecks the new library's parent/profile inputs.
    """
    source=inspection.row(identifier)
    if source['model']['kind']!='reference' or source['state']!='prepared':
        raise ValueError('Only a completed shared reference can be inherited')
    assets={Path(name).name:path for name,path in inspection.prepared_assets(identifier).items()}
    if set(assets)!={'entry.json','profile.npz','profile.json'}:
        raise ValueError('The reference has no complete shared TOF table')
    if assets['entry.json'].stat().st_size>8*1024*1024:
        raise ValueError('The inherited reference metadata exceeds its bound')
    matches=[r for r in library.document['entries'] if r['model']['kind']=='reference'
             and r['model']['bank']==source['model']['bank']]
    if len(matches)!=1:raise ValueError('The model library must declare the exact shared reference dependency')
    target=matches[0]
    if target['model']['grid']!=source['model']['grid'] or target['model']['settings']!=source['model']['settings']:
        raise ValueError('The inherited reference has different profile or grid choices')
    if target['state']=='prepared':
        if {Path(a['path']).name:a['sha256'] for a in target['files']}!={name:file_sha(path) for name,path in assets.items()}:
            raise ValueError('The completed dependency contains different shared-reference bytes')
        return library.coverage()
    if target['state']!='pending':raise ValueError('The reference dependency already has an active or failed attempt')
    attempt=library.start(target['id'])
    try:
        with tempfile.TemporaryDirectory(prefix='radar-reference-transfer-',dir=library.root.parent) as temporary:
            request=stage_entry(library,target['id'],attempt,Path(temporary)/'inputs')
            output=Path(request['destination']);output.mkdir()
            for name,path in assets.items():shutil.copyfile(path,output/name)
            manifest=json.loads((output/'entry.json').read_text())
            portable_entry_artifacts('reference',manifest)
            entry_sha=file_sha(output/'entry.json')
            result=dict(schema='radar-experiment-preparation-worker/v1',kind='reference',status='prepared',
                output=str(output),artifacts=sorted(assets),entry_sha256=entry_sha,
                observations_included=False,scientific_result=False)
            publish_entry(library,target['id'],attempt,request,result)
            from .evidence import select
            for reference in select(inspection.document.get('evidence',[]),[entry_sha]):library.add_evidence(reference)
    except Exception:
        if library.row(target['id'])['state']=='preparing':
            library.finish(target['id'],attempt,state='failed',reason='Shared reference transfer did not complete')
        raise
    return library.coverage()


def pending_entries(library, *, deferred=()):
    """Balance finite k/irrep/bank coverage across repeated admission passes.

    Counting prior admissions (including active/failed ones) prevents a new
    pass from restarting at the first family whenever one job finishes.
    """
    library.coverage()  # Refresh the durable, atomically written manifest.
    if library.document['specification'].get('schema') != PLAN_SCHEMA:
        raise ValueError('This library needs a supported scientific preparation plan before admission')
    deferred=set(deferred)
    if any(library.row(identifier)['state']!='preparing' for identifier in deferred):
        raise ValueError('Only verified unfinished attempts may be deferred')
    groups = {}
    for row in library.document['entries']:
        model = row['model'];settings = model['settings']
        from .structural_dependencies import ready
        if not ready(library,model):continue
        family = json.dumps([model['kind'], model['bank'], settings.get('k'),
            settings.get('candidate',{}).get('irrep')], sort_keys=True)
        group = groups.setdefault(family, dict(admitted=0, pending=[]))
        group['admitted'] += int(row['attempt'] > 0)
        if row['state'] == 'pending' or row['id'] in deferred:
            group['pending'].append(row['id'])
    queue = [(group['admitted'], family, 0) for family, group in groups.items() if group['pending']]
    heapq.heapify(queue)
    ordered = []
    while queue:
        admitted, family, offset = heapq.heappop(queue)
        pending = groups[family]['pending']
        ordered.append(pending[offset])
        if offset+1 < len(pending):
            heapq.heappush(queue, (admitted+1, family, offset+1))
    return ordered


def create_library(root, *, name, source, banks, models, scope=None):
    """Capture one parent, selected bank profiles and the finite model scope."""
    if not isinstance(banks, dict) or not banks or len(banks) > 100:
        raise ValueError('Select a bounded nonempty set of bank profiles')
    inputs = {'parent.cif': source}
    bank_records = {}
    for label, bank in banks.items():
        if not isinstance(label, str) or not label or len(label) > 100:
            raise ValueError('Each bank needs a stable descriptive identity')
        bank = fields(bank, ['instrument', 'grid'])
        suffix = Path(bank['instrument']).suffix.lower()
        if suffix not in {'.instprm', '.prm', '.ins', '.inst'}:
            raise ValueError('Select a native instrument profile for each bank')
        path = 'profile-'+digest(label)[:12]+suffix
        inputs[path] = bank['instrument']
        bank_records[label] = dict(instrument='inputs/'+path, grid=bank['grid'])
    if not isinstance(models, list) or not models:
        raise ValueError('Declare which magnetic or structural models to prepare')
    declared = []
    for model in models:
        model = fields(model, ['kind', 'bank', 'settings'], {'child': None})
        if model['kind'] not in {'reference','magnetic','structural','structural-examples'} or model['bank'] not in bank_records:
            raise ValueError('The model must refer to a selected physical bank')
        if model['kind']=='structural-examples':
            from .structural_jobs import validate_examples
            settings=validate_examples(dict(source='parent.cif',instrument='profile.instprm',child='child.cif',
                grid=bank_records[model['bank']]['grid'],destination='compiled',settings=model['settings']))['settings']
        else:settings = validate_settings(model['kind'], model['settings'])
        captured = dict(kind=model['kind'], bank=model['bank'], settings=settings,
            **bank_records[model['bank']])
        if model['kind'] in CHILD_KINDS:
            if not model['child']:
                raise ValueError('A structural model requires its authoritative child CIF')
            child_sha = file_sha(model['child'])
            verify_structural_child(settings, child_sha)
            child = 'child-'+child_sha+'.cif'
            inputs[child] = model['child']
            captured['child'] = 'inputs/'+child
        elif model['child'] is not None:
            raise ValueError('A magnetic model uses its MSG embedding, not a supplied GPX or child')
        # Validate the same operation contract before persisting a scope.
        validate_request(model['kind'], dict(source='parent.cif', instrument=captured['instrument'],
            grid=captured['grid'], settings=settings, destination='compiled',
            **({'child': captured['child']} if 'child' in captured else {})))
        declared.append(captured)
    from .structural_dependencies import append_training
    append_training(declared,bank_records)
    specification=dict(schema=PLAN_SCHEMA,source='inputs/parent.cif')
    if scope is None:
        return PreparationLibrary.create(root,name=name,specification=specification,models=declared,inputs=inputs)
    # Keep the selected discovery denominator portable. It is provenance data,
    # not a source of native executable instructions or an inferred result.
    if (not isinstance(scope,dict) or scope.get('schema')!='radar-preparation-declared-scope/v1'
            or scope.get('parent_cif_sha256')!=file_sha(source)
            or scope.get('selected_banks')!=sorted(bank_records)
            or not isinstance(scope.get('searches'),list) or not scope['searches']
            or len(json.dumps(scope,allow_nan=False).encode())>4*1024*1024):
        raise ValueError('Preparation needs its exact declared parent, banks and discovery scope')
    with tempfile.TemporaryDirectory(prefix='radar-preparation-scope-') as temporary:
        path=Path(temporary)/'scope.json';path.write_text(json.dumps(scope,sort_keys=True,allow_nan=False)+'\n')
        inputs['scope.json']=path;specification['scope']='inputs/scope.json'
        return PreparationLibrary.create(root,name=name,specification=specification,models=declared,inputs=inputs)


def stage_entry(library, identifier, attempt, directory, *, recover_completed=False):
    """Capture or recover this exact attempt without changing its inputs."""
    library.refresh()
    row = library.row(identifier)
    states={'preparing','prepared'} if recover_completed else {'preparing'}
    if row['state'] not in states or row['attempt'] != attempt:
        raise ValueError('Stage only the currently owned preparation attempt')
    specification = fields(library.document['specification'], ['schema', 'source'], {'scope':None})
    if specification['schema'] != PLAN_SCHEMA:
        raise ValueError('This preparation plan cannot be reconstructed by the current client')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    model = row['model']
    captured = {asset['path']: asset['sha256'] for asset in library.document['inputs']}
    roles = dict(source=specification['source'], instrument=model['instrument'])
    if model['kind'] in CHILD_KINDS:
        roles['child'] = model['child']
    library.verify_assets(roles.values())
    payload = dict(grid=model['grid'], settings=model['settings'], destination=str(directory/'compiled'))
    for role, name in roles.items():
        if name not in captured:
            raise ValueError('A preparation role is outside the captured inputs')
        source = confined(library.root, name)
        target = directory/(role+source.suffix)
        if target.is_symlink() or (target.exists() and file_sha(target) != captured[name]):
            raise ValueError('An immutable preparation attempt input changed')
        if not target.exists():
            shutil.copyfile(source, target)
        if file_sha(target) != captured[name]:
            raise ValueError('Preparation input changed while staging')
        payload[role] = str(target)
    if model['kind']=='structural-training':
        from .structural_dependencies import stage
        payload=stage(library,model,payload,directory)
    return validate_request(model['kind'], payload)


def publish_entry(library, identifier, attempt, request, result):
    """Accept a checked worker receipt; an upload alone never marks preparation done."""
    library.refresh()
    row = library.row(identifier)
    if row['state'] != 'preparing' or row['attempt'] != attempt:
        raise ValueError('Late preparation receipt belongs to another attempt')
    kind = row['model']['kind']
    request = validate_request(kind, request)
    model = row['model']
    roles = [library.document['specification']['source'], model['instrument']]
    if kind in CHILD_KINDS:
        roles.append(model['child'])
    library.verify_assets(roles)
    captured = {asset['path']: asset['sha256'] for asset in library.document['inputs']}
    if request['grid'] != model['grid'] or request['settings'] != model['settings']:
        raise ValueError('The returned preparation changed its declared model or bank')
    checksums = [captured[name] for name in roles]
    if checksums != [file_sha(request[role]) for role in ('source', 'instrument')+
                     (('child',) if kind in CHILD_KINDS else ())]:
        raise ValueError('The staged preparation no longer matches its captured library inputs')
    if kind=='structural-training':
        from .structural_dependencies import verify
        verify(library,model,request)
    if (not isinstance(result, dict) or result.get('schema') != 'radar-experiment-preparation-worker/v1'
            or result.get('status') != 'prepared' or result.get('kind') != kind
            or result.get('observations_included') is not False or result.get('scientific_result') is not False):
        raise ValueError('The worker did not produce a theory-only preparation receipt')
    output = Path(request['destination']).resolve()
    if output != Path(result.get('output', '')).resolve():
        raise ValueError('The returned preparation belongs to another output context')
    if (not isinstance(result.get('artifacts'),list)
            or result['artifacts'].count('entry.json')!=1):
        raise ValueError('The preparation returned an unexpected portable artifact inventory')
    numeric = portable_artifacts(kind,[name for name in result['artifacts'] if name!='entry.json'])
    names = numeric | {'entry.json'}
    entry = confined(output, 'entry.json')
    if entry.stat().st_size > 8*1024*1024 or file_sha(entry) != result.get('entry_sha256'):
        raise ValueError('The preparation manifest differs from its collected receipt')
    manifest = json.loads(entry.read_text())
    if (manifest.get('schema') != 'radar-experiment-preparation-entry/v1'
            or manifest.get('kind') != kind or manifest.get('grid') != request['grid']
            or manifest.get('settings') != request['settings'] or manifest.get('input_sha256') != checksums
            or manifest.get('observations_included') is not False or manifest.get('scientific_result') is not False
            or set(manifest.get('artifacts', {})) != numeric):
        raise ValueError('The worker preparation differs from its exact reviewed inputs')
    portable_entry_artifacts(kind,manifest)
    for name, checksum in manifest['artifacts'].items():
        if file_sha(confined(output, name)) != checksum:
            raise ValueError('A prepared numeric artifact changed before publication')
    target = library.root/'entries'/identifier/('attempt-'+str(attempt))
    target.mkdir(parents=True, exist_ok=True)
    # A recovered attempt can encounter already copied immutable files after
    # an interrupted manifest write. Accept byte identity, never overwrite.
    paths = []
    expected = dict(manifest['artifacts'], **{'entry.json': result['entry_sha256']})
    for name in sorted(names):
        destination = confined(library.root,
            relative(f'entries/{identifier}/attempt-{attempt}/{name}'), exists=False)
        source = confined(output, name)
        if destination.exists():
            if file_sha(destination) != file_sha(source):
                raise ValueError('A different preparation already occupies this entry')
        else:
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source, destination)
        if file_sha(destination) != expected[name]:
            raise ValueError('A preparation artifact changed while copying its verified output')
        paths.append(str(destination.relative_to(library.root)))
    library.finish(identifier, attempt, state='prepared', files=paths)
    return library.coverage()
