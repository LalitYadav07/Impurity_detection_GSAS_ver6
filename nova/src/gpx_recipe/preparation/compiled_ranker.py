"""Read-only, portable trained-model libraries without invented training jobs.

An externally completed trainer can publish a compact model library. The
library contains its exact parent, profile, child CIFs and typed weights. It
does not claim a Galaxy creating job, native-bank authority, experimental
accuracy, or permission to exclude candidates. Raw training data remain with
the originating preparation rather than being replaced by dummy dependencies.
"""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import uuid

from ..execution import write_json
from ..symmetry.compiler import file_sha,package_signature
from ..symmetry.contracts import digest,fields
from .reference import validate_grid
from .tof_kernel_arrays import checksum,read_bounded

KIND='structural-ranking-model'
ORIGIN_SCHEMA='radar-compiled-structural-ranking-origin/v1'
FILES={'model/model.json','model/weights.npz','model-origin.json','entry.json'}


def validate_entry(row):
    model=fields(row['model'],['kind','bank','instrument','grid','settings'])
    settings=fields(model['settings'],['model_sha256','weights_sha256','candidate_scope_sha256','training_context_sha256'])
    if (model['kind']!=KIND or row['state']!='prepared' or row['attempt']!=0
            or any(not checksum(v) for v in settings.values())):
        raise ValueError('Imported rankers are complete read-only model data, not queued native jobs')
    from .library import relative
    relative(model['instrument'])
    if not isinstance(model['grid'],dict):raise ValueError('A compiled ranker needs its declared native grid')
    validate_grid(**fields(model['grid'],['kind','lower','upper','bins','bank']))
    prefix='entries/'+row['id']+'/compiled/'
    if {a['path'] for a in row['files']}!={prefix+n for n in FILES}:
        raise ValueError('A compiled ranker needs its exact closed data inventory')


def validate_origin(value,settings,*,parent_sha256,instrument_sha256,grid,input_sha256):
    if (not isinstance(input_sha256,list) or not input_sha256 or len(input_sha256)>5000
            or any(not checksum(v) for v in input_sha256)):
        raise ValueError('Compiled models need their complete native training input identities')
    origin=fields(value,['schema','model_sha256','weights_sha256','candidate_scope_sha256',
        'parent_sha256','instrument_sha256','training_context','training_context_sha256','observations_included','native_origin_certificate',
        'candidate_pruning_authorized'])
    if (origin['schema']!=ORIGIN_SCHEMA or any(origin[k]!=settings[k] for k in settings)
            or origin['parent_sha256']!=parent_sha256 or origin['instrument_sha256']!=instrument_sha256
            or any(origin[k] is not False for k in ['observations_included',
                'native_origin_certificate','candidate_pruning_authorized'])):
        raise ValueError('Compiled model origin differs from its exact advisory data')
    context=fields(origin['training_context'],['grid','native_manifests','input_sha256'])
    records=context['native_manifests']
    if (context['grid']!=grid or context['input_sha256']!=input_sha256
            or not isinstance(records,list) or len(records)!=len(input_sha256)
            or not records or len(records)>5000 or any(not checksum(v) for v in records)
            or digest(context)!=origin['training_context_sha256']):
        raise ValueError('Compiled model context differs from its complete native training inputs')
    return origin


def create(model_directory,parent,instrument,destination,*,children,name,bank,grid,training_manifests):
    """Capture a completed model through the ordinary library/archive reader.

    ``children`` maps every original candidate ID to its exact child CIF. No
    candidate may disappear merely because it is not in the leading scores.
    This adapter changes placement only; all model bytes remain unchanged.
    """
    from .library import PreparationLibrary,relative,confined,validate_manifest,MAX_ARCHIVE_BYTES,MAX_METADATA
    from .structural_ranker import Ranker
    from .structural_ranking_hints import LIMITS,_metadata
    source=Path(model_directory).resolve();parent=Path(parent);instrument=Path(instrument)
    destination=Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError('Create a compiled model library at a fresh destination')
    destination=destination.resolve()
    for path,limit in [(source/'model.json',LIMITS['model']),
                       (source/'weights.npz',LIMITS['weights']),
                       (parent,LIMITS['parent']),(instrument,LIMITS['instrument'])]:
        if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=limit:
            raise ValueError('A compiled ranker needs bounded regular captured files')
    model_hashes={n:file_sha(source/n) for n in ['model.json','weights.npz']}
    metadata,candidates=_metadata(source/'model.json')
    # Read only typed numeric tensors; this also checks the installed model
    # layout, metadata/weights binding and all original class memberships.
    Ranker(source,threads=1)
    if (metadata['parent_sha256']!=file_sha(parent)
            or metadata['instrument_sha256']!=file_sha(instrument)):
        raise ValueError('Compiled model parent or profile differs from the original training context')
    identifiers={c['id'] for c in candidates}
    if not isinstance(children,dict) or set(children)!=identifiers:
        raise ValueError('Capture the complete original candidate-to-CIF inventory')
    if not isinstance(bank,str) or not 1<=len(bank)<=100:
        raise ValueError('A compiled model needs its physical bank label')
    grid=validate_grid(**fields(grid,['kind','lower','upper','bins','bank']))
    # A multi-bank profile's file hash alone cannot distinguish its blocks.
    # Derive/check the declared bank and grid against every original native
    # input, rather than allowing a caller to retag the trained weights.
    from .structural_mode_projection import PAIR_SCHEMAS
    from .structural_hints import candidate_identity
    expected=metadata.get('input_sha256')
    if (not isinstance(expected,list) or not expected or len(expected)>5000
            or any(not checksum(v) for v in expected)
            or not isinstance(training_manifests,(list,tuple))
            or len(training_manifests)!=len(expected)):
        raise ValueError('Capture every original native training manifest for this compiled model')
    definitions={c['id']:candidate_identity(c) for c in candidates};headers=[]
    from ..symmetry.contracts import validate_candidate
    for path,expected_hash in zip(training_manifests,expected):
        path=Path(path)
        if path.is_symlink():raise ValueError('Native training manifests must be regular captured files')
        raw=read_bounded(path,8*1024**2);header=json.loads(raw)
        if (not isinstance(header,dict) or header.get('schema') not in PAIR_SCHEMAS
                or header.get('source_sha256')!=metadata['parent_sha256']
                or header.get('instrument_sha256')!=metadata['instrument_sha256']
                or header.get('grid')!=grid or header.get('pairs_sha256')!=expected_hash
                or header.get('observations_included') is not False):
            raise ValueError('Compiled bank/grid differs from its original native training context')
        candidate=validate_candidate(header.get('candidate'))
        if definitions.get(candidate['id'])!=candidate_identity(candidate):
            raise ValueError('A native training input belongs to another structural construction')
        header_hash=hashlib.sha256(raw).hexdigest()
        if file_sha(path)!=header_hash:raise ValueError('A native training manifest changed during capture')
        headers.append(header_hash)
    context=dict(grid=grid,native_manifests=headers,input_sha256=list(expected))
    snapshots={}
    def capture(relative_name,path,*,expected=None):
        relative(relative_name);path=Path(path)
        if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=10*1024**2:
            raise ValueError('Compiled inputs must be bounded regular scientific files')
        checksum_value=file_sha(path)
        if expected is not None and checksum_value!=expected:
            raise ValueError('Compiled child differs from its original model identity')
        if relative_name in snapshots:
            if snapshots[relative_name][1]!=checksum_value:
                raise ValueError('Conflicting compiled child paths')
        else:snapshots[relative_name]=(path,checksum_value)
    capture('inputs/parent.cif',parent,expected=metadata['parent_sha256'])
    capture('inputs/profile.instprm',instrument,expected=metadata['instrument_sha256'])
    for candidate in candidates:
        capture('inputs/'+candidate['child']['path'],children[candidate['id']],
            expected=candidate['child']['sha256'])
    if (sum(p.stat().st_size for p,_ in snapshots.values())
            +sum((source/n).stat().st_size for n in model_hashes)+MAX_METADATA>MAX_ARCHIVE_BYTES):
        raise ValueError('The complete compiled model exceeds the portable byte budget')
    settings=dict(model_sha256=model_hashes['model.json'],
        weights_sha256=model_hashes['weights.npz'],candidate_scope_sha256=digest(candidates),
        training_context_sha256=digest(context))
    model=dict(kind=KIND,bank=bank,instrument='inputs/profile.instprm',grid=grid,settings=settings)
    identifier=digest(model)[:24];prefix='entries/'+identifier+'/compiled/'
    specification=dict(source='inputs/parent.cif',
        banks=[dict(label=bank,instrument=model['instrument'],grid=grid)],
        imported_compiled_ranker=True)
    inputs=[dict(path=n,sha256=h) for n,(_,h) in sorted(snapshots.items())]
    identity=digest(dict(specification=specification,inputs=inputs,models=[model]))
    staging=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.partial')
    staging.mkdir(parents=True)
    try:
        for n,(path,h) in snapshots.items():
            target=confined(staging,n,exists=False);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
            if file_sha(path)!=h or file_sha(target)!=h:raise ValueError('Compiled scientific input changed during capture')
        target=staging/prefix;target.mkdir(parents=True)
        (target/'model').mkdir()
        for n,key in [('model.json','model_sha256'),('weights.npz','weights_sha256')]:
            shutil.copyfile(source/n,target/'model'/n)
            if file_sha(source/n)!=settings[key] or file_sha(target/'model'/n)!=settings[key]:
                raise ValueError('Compiled model changed during capture')
        origin=dict(schema=ORIGIN_SCHEMA,**settings,training_context=context,parent_sha256=metadata['parent_sha256'],
            instrument_sha256=metadata['instrument_sha256'],observations_included=False,
            native_origin_certificate=False,candidate_pruning_authorized=False)
        validate_origin(origin,settings,parent_sha256=metadata['parent_sha256'],
            instrument_sha256=metadata['instrument_sha256'],grid=grid,input_sha256=expected)
        write_json(target/'model-origin.json',origin)
        detail={k:copy.deepcopy(metadata[k]) for k in ['dataset','test'] if k in metadata}
        detail.update(imported_model=True,candidate_count=len(candidates),classes=metadata['classes'],
            mode_families=len(metadata.get('mode_readout',{}).get('families',[])),
            native_origin_certificate=False,candidate_pruning_authorized=False)
        artifact_names=FILES-{'entry.json'}
        entry=dict(schema='radar-experiment-preparation-entry/v1',kind=KIND,
            grid=grid,settings=settings,input_sha256=[metadata['parent_sha256'],metadata['instrument_sha256']],
            implementation_sha256=package_signature(),detail=detail,
            observations_included=False,scientific_result=False,
            artifacts={n:file_sha(target/n) for n in sorted(artifact_names)})
        write_json(target/'entry.json',entry)
        row=dict(id=identifier,model=model,state='prepared',attempt=0,
            reason='Imported trained simulations; no native fit or training job is claimed.',
            files=[dict(path=prefix+n,sha256=file_sha(target/n)) for n in sorted(FILES)])
        document=dict(schema='radar-experiment-preparation-library/v1',id=identity[:24],name=name,
            specification=specification,specification_sha256=digest(specification),
            inputs=inputs,entries=[row],observations_included=False,scientific_result=False)
        validate_manifest(document);write_json(staging/'library.json',document,compact=True)
        checked=PreparationLibrary(staging)
        staging.replace(destination)
        checked.root=destination;checked.path=destination/'library.json'
        return checked
    finally:
        if staging.exists():shutil.rmtree(staging)
