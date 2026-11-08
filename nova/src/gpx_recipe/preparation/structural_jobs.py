"""Closed scientific-worker operations for structural ranking preparation.

Native example blocks and trained weights are portable data. Feature caches
and native construction projects stay in the owned worker. A bounded optimizer
checkpoint can continue the same captured training run on another worker.
Neither operation qualifies a classifier to remove refinement candidates.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
import time

from ..execution import write_json
from ..symmetry.compiler import file_sha, package_signature
from ..symmetry.contracts import digest, fields, number
from .batch import relative
from .structural_augmentation import Controls, ObservationAugmenter
from .tof_kernel_arrays import checksum, read_bounded
from .reference import validate_grid
from .structural_training_blocks import (
    REPRESENTATION, PAIRED_FIELDS, parent_context, validate_representation,
    read as read_training_block, metadata as block_metadata,
    representation as block_representation, context as block_context,
)

PLAN_SCHEMA = 'radar-structural-training-plan/v1'
PAIRED_PLAN_SCHEMA = 'radar-structural-training-plan/v2'
CONTINUED_PLAN_SCHEMA = 'radar-structural-training-plan/v3'
STRATIFIED_PLAN_SCHEMA = 'radar-structural-training-plan/v4'
STRATIFIED_CONTINUED_PLAN_SCHEMA = 'radar-structural-training-plan/v5'
PROJECTED_PLAN_SCHEMA = 'radar-structural-training-plan/v6'
PROJECTED_CONTINUED_PLAN_SCHEMA = 'radar-structural-training-plan/v7'
ADDITIVE_PLAN_SCHEMA='radar-structural-training-plan/v8'
ADDITIVE_CONTINUED_PLAN_SCHEMA='radar-structural-training-plan/v9'
ADDITIVE_PLAN_SCHEMAS={ADDITIVE_PLAN_SCHEMA,ADDITIVE_CONTINUED_PLAN_SCHEMA}
PROJECTED_PLAN_SCHEMAS = {PROJECTED_PLAN_SCHEMA,PROJECTED_CONTINUED_PLAN_SCHEMA,*ADDITIVE_PLAN_SCHEMAS}
CONTINUED_PLAN_SCHEMAS = {CONTINUED_PLAN_SCHEMA, STRATIFIED_CONTINUED_PLAN_SCHEMA,PROJECTED_CONTINUED_PLAN_SCHEMA,ADDITIVE_CONTINUED_PLAN_SCHEMA}
STRATIFIED_PLAN_SCHEMAS = {STRATIFIED_PLAN_SCHEMA, STRATIFIED_CONTINUED_PLAN_SCHEMA,*PROJECTED_PLAN_SCHEMAS}
PAIRED_PLAN_SCHEMAS = {PAIRED_PLAN_SCHEMA, *CONTINUED_PLAN_SCHEMAS, *STRATIFIED_PLAN_SCHEMAS}
MAX_PLAN_BYTES = 8*1024**2
MAX_INPUT_BYTES = 480*1024**2
# Only typed three-curve training may span several separately bounded packets.
# Ordinary model archives and legacy feature plans keep their original limits.
MAX_PAIRED_INPUT_BYTES = 8*1024**3
MAX_INPUT_FILES=10000
# The shared packet admits parent, profile, plan, and two files per block.
MAX_BLOCKS = (10000-3)//2


def validate_examples(request):
    from .worker import validate_request
    request = fields(request, ['source', 'instrument', 'child', 'grid', 'settings', 'destination'])
    settings = fields(request['settings'], ['candidate', 'mode_radius', 'cell_radius', 'seed'],
                      dict(trajectories=64, steps=4, trajectory_slice=None,mode_coverage=None,
                           **dict.fromkeys(PAIRED_FIELDS)))
    for key, lower, upper in [('seed', 0, 2**32-1), ('trajectories', 1, 4096), ('steps', 1, 32)]:
        settings[key] = number(settings[key], lower, upper, integer=True)
    interval = settings['trajectory_slice']
    if interval is None:
        interval = [0, settings['trajectories']]
    if (not isinstance(interval, (list, tuple)) or len(interval) != 2
            or any(type(i) is not int for i in interval)
            or not 0 <= interval[0] < interval[1] <= settings['trajectories']):
        raise ValueError('Select a nonempty slice of the complete structural trajectory portfolio')
    settings['trajectory_slice'] = list(interval)
    from .structural_mode_coverage import SCHEMAS as MODE_COVERAGE
    coverage=settings['mode_coverage']
    if coverage is None:settings.pop('mode_coverage')
    elif not isinstance(coverage,str) or coverage not in MODE_COVERAGE or settings['representation']!=REPRESENTATION:
        raise ValueError('Named-mode coverage requires its declared paired sampling version')
    if settings['representation'] is None:
        if any(settings[k] is not None for k in PAIRED_FIELDS[1:]):
            raise ValueError('Parent-paired controls require their three-curve representation')
        for key in PAIRED_FIELDS:settings.pop(key)
    else:
        context=parent_context(settings)
        settings['parent_relative_metric_range']=list(context[2:4])
        if settings['trajectories']*settings['steps']>32768:
            raise ValueError('Declare a bounded complete paired trajectory portfolio')
    base = dict(request, settings={k: settings[k] for k in ('candidate', 'mode_radius', 'cell_radius')})
    validated = validate_request('structural', base)
    settings.update(validated['settings'])
    return dict(validated, settings=settings)


def validate_training(request):
    request = fields(request, ['source', 'instrument', 'grid', 'plan', 'destination', 'settings'])
    if not isinstance(request['settings'],dict):raise ValueError('Training settings must be a closed contract object')
    paired='representation' in request['settings']
    if paired:validate_representation(request['settings'])
    defaults=dict(bins=4096 if paired else 2048, split_seed=69271 if paired else 3917,
        epochs=60, seed=89113, workers=4, augmentation=asdict(Controls()), dependencies=[])
    if paired:defaults['representation']=REPRESENTATION
    settings = fields(request['settings'], [], defaults)
    for key, lower, upper in [('bins', 64, 8192), ('split_seed', 0, 2**32-1),
                              ('epochs', 1, 1000), ('seed', 0, 2**32-1), ('workers', 1, 16)]:
        settings[key] = number(settings[key], lower, upper, integer=True)
    if settings['augmentation'] is not None:
        controls = fields(settings['augmentation'], [], asdict(Controls()))
        controls = Controls(**controls)
        if controls.impurity_fraction != (0., 0.):
            raise ValueError('This training job has no declared native impurity templates')
        settings['augmentation'] = {key:list(value) if isinstance(value,tuple) else value
                                    for key,value in asdict(controls).items()}
    dependencies=settings['dependencies']
    if (not isinstance(dependencies,list) or len(dependencies)>MAX_BLOCKS
            or any(not isinstance(key,str) or len(key)!=24
                   or any(c not in '0123456789abcdef' for c in key) for key in dependencies)
            or dependencies!=sorted(set(dependencies))):
        raise ValueError('Training dependencies must be unique captured example-entry identities')
    request['grid']=validate_grid(**fields(request['grid'], ['kind','lower','upper','bins','bank']))
    for key in ('source', 'instrument', 'plan', 'destination'):
        if not isinstance(request[key], str) or not request[key]:
            raise ValueError('Training requires owned source, plan and destination paths')
    return dict(request, settings=settings)


def publish_result(kind, request, detail, names):
    """Use the same entry receipt and library publication as other preparation."""
    out=Path(request['destination']).resolve()
    roles=('source','instrument')+(('child',) if kind=='structural-examples' else ())
    manifest=dict(schema='radar-experiment-preparation-entry/v1',kind=kind,
        grid=request['grid'],settings=request['settings'],input_sha256=[file_sha(request[k]) for k in roles],
        implementation_sha256=package_signature(),detail=detail,
        observations_included=False,scientific_result=False,
        artifacts={name:file_sha(out/name) for name in names})
    write_json(out/'entry.json',manifest)
    return dict(schema='radar-experiment-preparation-worker/v1',kind=kind,status='prepared',
        output=str(out),artifacts=names+['entry.json'],entry_sha256=file_sha(out/'entry.json'),
        observations_included=False,scientific_result=False,screening_qualified=False,
        candidate_pruning_authorized=False,seconds=detail['seconds'])


def read_plan(path):
    value = json.loads(read_bounded(path, MAX_PLAN_BYTES))
    continued=isinstance(value,dict) and value.get('schema') in CONTINUED_PLAN_SCHEMAS
    paired=isinstance(value,dict) and value.get('schema') in PAIRED_PLAN_SCHEMAS
    stratified=isinstance(value,dict) and value.get('schema') in STRATIFIED_PLAN_SCHEMAS
    projected=isinstance(value,dict) and value.get('schema') in PROJECTED_PLAN_SCHEMAS
    additive=isinstance(value,dict) and value.get('schema') in ADDITIVE_PLAN_SCHEMAS
    value = fields(value, ['schema', 'parent_sha256', 'instrument_sha256', 'blocks']+
                   (['representation'] if paired else [])+(['resume_state'] if continued else [])+
                   (['parent_strata'] if stratified else [])+(['additive_sources'] if additive else []))
    if paired:validate_representation(value)
    if (value['schema'] not in {PLAN_SCHEMA, *PAIRED_PLAN_SCHEMAS} or not checksum(value['parent_sha256'])
            or not checksum(value['instrument_sha256']) or not isinstance(value['blocks'], list)
            or not 1 <= len(value['blocks']) <= MAX_BLOCKS):
        raise ValueError('Training requires a bounded parent/profile example inventory')
    seen = set()
    for row in value['blocks']:
        fields(row, ['directory', 'manifest_sha256', 'examples_sha256']+(['mode_projection'] if projected else []))
        name = relative(row['directory'])
        if name in seen or any(not checksum(row[k]) for k in ('manifest_sha256', 'examples_sha256')):
            raise ValueError('Training blocks need unique directories and exact content identities')
        seen.add(name)
        if projected and row['mode_projection'] is not None:
            from .structural_mode_projection import marker
            marker(row['mode_projection'])
    if projected and not any(row['mode_projection'] is not None for row in value['blocks']):
        raise ValueError('A projected training plan needs its captured native mode coordinates')
    if additive:
        from .structural_supplement_pairs import validate_inventory
        validate_inventory(value['additive_sources'],{row['directory']:row for row in value['blocks']})
    if stratified and (not projected or value['parent_strata'] is not None):
        from .structural_training_strata import validate
        validate(value['parent_strata'], seen-set(value.get('additive_sources',{})))
    if continued:
        resume=fields(value['resume_state'],['directory','state_sha256','receipt_sha256'])
        if (resume['directory']!='resume' or not checksum(resume['state_sha256'])
                or not checksum(resume['receipt_sha256']) or len(value['blocks'])>MAX_BLOCKS-1):
            raise ValueError('A training continuation needs its bounded numerical resume inventory')
    if 3+sum(2+2*bool(row.get('mode_projection')) for row in value['blocks'])+2*continued>MAX_INPUT_FILES:
        raise ValueError('Training inputs exceed the complete packet file inventory')
    return value


def plan_inputs(path):
    """One packet inventory for NOVA capture and worker-side verification."""
    plan = read_plan(path)
    from .structural_mode_projection import FILES
    result=[str(Path(row['directory'])/name) for row in plan['blocks']
            for name in ('examples.json', 'examples.npz')+(FILES if row.get('mode_projection') is not None else ())]
    if plan['schema'] in CONTINUED_PLAN_SCHEMAS:
        from .structural_training_transfer import STATE,RECEIPT
        result.extend(str(Path('resume')/name) for name in (STATE,RECEIPT))
    return result


def resume_directory(plan_path,plan):
    if plan['schema'] not in CONTINUED_PLAN_SCHEMAS:return None
    from ..symmetry.compiler import confined
    from .structural_training_transfer import STATE,RECEIPT,read
    directory=Path(plan_path).parent/'resume';record=plan['resume_state']
    for name,key in ((STATE,'state_sha256'),(RECEIPT,'receipt_sha256')):
        if file_sha(confined(Path(plan_path).parent,str(Path('resume')/name)))!=record[key]:
            raise ValueError('Captured training continuation changed')
    read(directory)
    return directory


def create_plan(source, directories, destination, *, recover_identical=False,resume=None):
    """Bind existing observation-free blocks without copying large arrays."""
    destination = Path(destination).resolve()
    source = Path(source)
    if source.is_symlink() or not source.is_file() or not 0 < source.stat().st_size <= 10*1024**2:
        raise ValueError('Training requires a bounded regular parent CIF')
    from .structural_training_strata import Inventory
    from . import structural_mode_projection as projection
    from . import structural_supplement_pairs as additive
    parent = file_sha(source); rows = []; profile = None; total = 0; formats=set(); strata=Inventory();mode_stamps=[]
    additions=additive.Catalog(arrays=False);documents=[]
    directories = list(directories)
    if not 1 <= len(directories) <= MAX_BLOCKS:
        raise ValueError('Declare a bounded structural training inventory')
    for directory in directories:
        directory = Path(directory).resolve()
        name = str(directory.relative_to(destination.parent))
        relative(name)
        metadata=block_metadata(directory)
        names=('examples.json','examples.npz')+(projection.FILES if metadata['schema'] in projection.PAIR_SCHEMAS else ())
        total += sum((directory/name).stat().st_size for name in names)
        formats.add(block_representation(metadata))
        if len(formats)!=1:
            raise ValueError('Training blocks must share one representation and parent trajectory context')
        limit=MAX_PAIRED_INPUT_BYTES if formats=={REPRESENTATION} else MAX_INPUT_BYTES
        if total > limit:
            raise ValueError('Training input arrays exceed their typed byte budget')
        block = read_training_block(directory)
        documents.append(block.manifest)
        mode_stamps.append(None if getattr(block,'mode_projection',None) is None else block.mode_projection.stamp)
        if block_representation(metadata)==REPRESENTATION:
            additions.add(block.manifest,name=name,arrays=block.arrays,projection=getattr(block,'mode_projection',None))
            if not additive.present(block.manifest):strata.add(name,block.manifest)
        if block.manifest['source_sha256'] != parent:
            raise ValueError('A training block belongs to another parent CIF')
        current = block.manifest['instrument_sha256']
        if profile is not None and current != profile:
            raise ValueError('Train each exact instrument-profile context separately')
        profile = current
        rows.append(dict(directory=name, manifest_sha256=file_sha(directory/'examples.json'),
                         examples_sha256=file_sha(directory/'examples.npz')))
    value = dict(schema=PLAN_SCHEMA, parent_sha256=parent, instrument_sha256=profile, blocks=rows)
    if formats=={REPRESENTATION}:
        added_inventory=additions.finish()
        if added_inventory!=additive.verify_context(documents,lambda:(),
                subset_factory=lambda indices:(read_training_block(directories[i]) for i in indices),
                names=[r['directory'] for r in rows]):
            raise ValueError('Sparse original contexts differ from their captured plan inventory')
        value.update(schema=PAIRED_PLAN_SCHEMA,representation=REPRESENTATION)
        parent_strata=strata.describe()
        if parent_strata is not None:
            value.update(schema=STRATIFIED_PLAN_SCHEMA,parent_strata=parent_strata)
        if any(stamp is not None for stamp in mode_stamps):
            value.update(schema=PROJECTED_PLAN_SCHEMA,parent_strata=parent_strata)
            for row,stamp in zip(rows,mode_stamps):row['mode_projection']=stamp
        if added_inventory:
            value.update(schema=ADDITIVE_PLAN_SCHEMA,additive_sources=added_inventory)
    if resume is not None:
        from .structural_training_transfer import STATE,RECEIPT,read
        if formats!={REPRESENTATION} or len(directories)>MAX_BLOCKS-1:
            raise ValueError('Only a bounded three-curve training plan can continue numerical state')
        resume=Path(resume);receipt,raw=read(resume);target=destination.parent/'resume'
        total+=len(raw)+len(read_bounded(resume/RECEIPT,8192))
        if total>MAX_PAIRED_INPUT_BYTES:
            raise ValueError('Training continuation exceeds its typed byte budget')
        if target.is_symlink():raise ValueError('Training resume directory must be owned')
        target.mkdir(parents=True,exist_ok=True)
        for name,data in ((STATE,raw),(RECEIPT,read_bounded(resume/RECEIPT,8192))):
            path=target/name
            if path.is_symlink() or (path.exists() and path.read_bytes()!=data):
                raise ValueError('An immutable training resume input changed')
            if not path.exists():path.write_bytes(data)
        value.update(schema=ADDITIVE_CONTINUED_PLAN_SCHEMA if value['schema']==ADDITIVE_PLAN_SCHEMA else
            PROJECTED_CONTINUED_PLAN_SCHEMA if value['schema']==PROJECTED_PLAN_SCHEMA else
            STRATIFIED_CONTINUED_PLAN_SCHEMA if 'parent_strata' in value else CONTINUED_PLAN_SCHEMA,
            resume_state=dict(directory='resume',
            state_sha256=file_sha(target/STATE),receipt_sha256=file_sha(target/RECEIPT)))
    # Validate before publishing, including duplicate logical block paths.
    if 3+sum(2+2*bool(row.get('mode_projection')) for row in rows)+2*(resume is not None)>MAX_INPUT_FILES:
        raise ValueError('Training inputs exceed the complete packet file inventory')
    if len((json.dumps(value, indent=2)+'\n').encode()) > MAX_PLAN_BYTES:
        raise ValueError('Training plan exceeds its metadata budget')
    if len({row['directory'] for row in rows}) != len(rows):
        raise ValueError('Training blocks need unique directories and exact content identities')
    if destination.exists():
        if recover_identical and read_plan(destination)==value:return value
        raise ValueError('Training plans are immutable; use a new path')
    write_json(destination, value)
    read_plan(destination)
    return value


def checked_blocks(plan_path, plan, *, validate_arrays=True,grid=None):
    from ..symmetry.compiler import confined
    from .structural_training_strata import Inventory
    from . import structural_mode_projection as projection
    from . import structural_supplement_pairs as additive
    uses_strata=plan.get('parent_strata') is not None
    projected=plan['schema'] in PROJECTED_PLAN_SCHEMAS
    total = 0; directories = [];contexts=set();strata=Inventory()
    additions=additive.Catalog(arrays=False);documents=[]
    for row in plan['blocks']:
        names=('examples.json','examples.npz')+(projection.FILES if row.get('mode_projection') is not None else ())
        paths = [confined(Path(plan_path).parent, str(Path(row['directory'])/name)) for name in names]
        total += sum(path.stat().st_size for path in paths)
        limit=MAX_PAIRED_INPUT_BYTES if plan.get('representation')==REPRESENTATION else MAX_INPUT_BYTES
        if total > limit:
            raise ValueError('Training input arrays exceed their typed byte budget')
        hashes=[row['manifest_sha256'],row['examples_sha256']]+([] if row.get('mode_projection') is None else
            [row['mode_projection']['metadata_sha256'],row['mode_projection']['arrays_sha256']])
        if [file_sha(path) for path in paths] != hashes:
            raise ValueError('A captured structural training block changed')
        block=read_training_block(paths[0].parent) if validate_arrays else None
        metadata=block.manifest if block is not None else block_metadata(paths[0].parent)
        documents.append(metadata)
        if block_representation(metadata)==REPRESENTATION:
            additions.add(metadata,name=row['directory'],arrays=None if block is None else block.arrays,
                projection=None if block is None else getattr(block,'mode_projection',None))
        if (metadata['source_sha256'] != plan['parent_sha256']
                or metadata['instrument_sha256'] != plan['instrument_sha256']):
            raise ValueError('Training block has a different parent or profile context')
        contexts.add(block_context(metadata))
        if (block_representation(metadata)!=plan.get('representation')
                or (not uses_strata and len(contexts)!=1)):
            raise ValueError('Training block representation or parent trajectory context differs from its plan')
        has_map=metadata['schema'] in projection.PAIR_SCHEMAS
        if has_map and (not projected or projection.marker(metadata.get('mode_projection'))!=row['mode_projection']):
            raise ValueError('Paired native mode coordinates differ from their captured training plan')
        if not has_map and row.get('mode_projection') is not None:
            raise ValueError('A legacy block cannot acquire an unbound native mode projection')
        if uses_strata and not additive.present(metadata):strata.add(row['directory'],metadata)
        if grid is not None and metadata.get('grid')!=grid:
            raise ValueError('Training examples differ from the captured physical bank grid')
        directories.append(paths[0].parent)
    if uses_strata and strata.describe()!=plan['parent_strata']:
        raise ValueError('Training blocks differ from their complete captured parent-range strata')
    if additions.finish()!=plan.get('additive_sources',{}):
        raise ValueError('Sparse additions differ from their explicit captured training plan')
    if validate_arrays and additions.finish()!=additive.verify_context(documents,lambda:(),
            subset_factory=lambda indices:(read_training_block(directories[i]) for i in indices),
            names=[r['directory'] for r in plan['blocks']]):
        raise ValueError('Sparse original contexts differ from their captured plan arrays')
    resumed=resume_directory(plan_path,plan)
    if resumed is not None:
        from .structural_training_transfer import STATE,RECEIPT
        total+=sum((resumed/name).stat().st_size for name in (STATE,RECEIPT))
        if total>MAX_PAIRED_INPUT_BYTES:
            raise ValueError('Training continuation exceeds its typed byte budget')
    return directories


def execute_examples(request):
    request = validate_examples(request); settings = request['settings']
    base = dict(request, settings={k: settings[k] for k in ('candidate', 'mode_radius', 'cell_radius')})
    options={k:settings[k] for k in ('seed','trajectories','steps','trajectory_slice')}
    if 'mode_coverage' in settings:options['mode_coverage']=settings['mode_coverage']
    began=time.monotonic()
    if settings.get('representation')==REPRESENTATION:
        from .structural_pairs import generate
        from .structural_paired_blocks import pack_block
        from .native_size import publish as publish_size
        out=Path(request['destination']).resolve();out.mkdir(parents=True,exist_ok=False)
        detail=generate(dict(base,destination=str(out/'working-native')),
            parent_seed=settings['parent_seed'],parent_relative_range=settings['parent_relative_metric_range'],
            report_size=lambda **dimensions:publish_size(request,**dimensions),**options)
        packed=pack_block(out/'working-native',out/'portable',names='examples')
        from .structural_mode_projection import FILES as PROJECTION_FILES,descriptor
        names=('examples.json','examples.npz')+(PROJECTION_FILES if packed.mode_projection is not None else ())
        for name in names:(out/'portable'/name).replace(out/name)
        examples_sha256=packed.manifest['pairs_sha256']
    else:
        from .structural_examples import generate
        detail=generate(base,**options)
        examples_sha256=detail['examples_sha256']
        names=('examples.json','examples.npz')
    receipt=dict(examples=detail['examples'],
        candidate_id=detail['candidate']['id'],trajectory_slice=detail['trajectory_slice'],
        mode_dimensions=detail['mode_dimensions'],cell_dimensions=detail['cell_dimensions'],
        examples_sha256=examples_sha256,seconds=time.monotonic()-began)
    if settings.get('representation')==REPRESENTATION:
        receipt.update(representation=REPRESENTATION,parent_seed=settings['parent_seed'],
            parent_relative_metric_range=settings['parent_relative_metric_range'],timing=detail['timing'])
        if 'mode_coverage' in settings:receipt['mode_coverage']=settings['mode_coverage']
        if packed.mode_projection is not None:receipt['mode_projection']=descriptor(packed.mode_projection)
    return publish_result('structural-examples',request,receipt,list(names))


def _completed_training(request,out,plan_sha):
    """Recover immutable published files without rebuilding or retraining."""
    path=out/'entry.json'
    if not path.exists() and not path.is_symlink():return None
    entry_raw=read_bounded(path,MAX_PLAN_BYTES);entry=json.loads(entry_raw)
    names=['model/model.json','model/weights.npz','training.json']
    if (not isinstance(entry,dict) or entry.get('schema')!='radar-experiment-preparation-entry/v1'
            or entry.get('kind')!='structural-training' or entry.get('settings')!=request['settings']
            or entry.get('grid')!=request['grid'] or entry.get('implementation_sha256')!=package_signature()
            or entry.get('input_sha256')!=[file_sha(request[k]) for k in ('source','instrument')]
            or not isinstance(entry.get('artifacts'),dict) or set(entry['artifacts'])!=set(names)
            or entry.get('observations_included') is not False or entry.get('scientific_result') is not False):
        raise ValueError('Published training recovery receipt changed')
    for name in names:
        raw=read_bounded(out/name,32*1024**2)
        if hashlib.sha256(raw).hexdigest()!=entry['artifacts'][name]:
            raise ValueError('Published training artifact changed; refusing to overwrite it')
    detail=json.loads(read_bounded(out/'training.json',MAX_PLAN_BYTES))
    if (detail!=entry.get('detail') or detail.get('plan_sha256')!=plan_sha
            or detail.get('status')!='trained' or detail.get('screening_qualified') is not False
            or detail.get('candidate_pruning_authorized') is not False):
        raise ValueError('Published training result differs from its captured plan')
    from .structural_ranker import Ranker
    model=Ranker(out/'model').metadata
    if (model.get('weights_sha256')!=detail.get('weights_sha256')
            or model.get('parent_sha256')!=detail.get('parent_sha256')
            or model.get('instrument_sha256')!=detail.get('instrument_sha256')):
        raise ValueError('Published training model differs from its worker receipt')
    return dict(schema='radar-experiment-preparation-worker/v1',kind='structural-training',status='prepared',
        output=str(out),artifacts=names+['entry.json'],entry_sha256=hashlib.sha256(entry_raw).hexdigest(),
        observations_included=False,scientific_result=False,screening_qualified=False,
        candidate_pruning_authorized=False,seconds=detail['seconds'])


def execute_training(request):
    from .structural_ranker import train
    request = validate_training(request); settings = request['settings']
    plan_path = Path(request['plan']).resolve(); plan_sha = file_sha(plan_path); plan = read_plan(plan_path)
    if settings.get('representation')!=plan.get('representation'):
        raise ValueError('Training settings differ from the captured example representation')
    paired=plan.get('representation')==REPRESENTATION
    if paired:
        from .structural_paired_store import load
        from .structural_replay import build as build_replay
    else:
        from .structural_store import build
    source = Path(request['source'])
    if source.is_symlink() or not 0 < source.stat().st_size <= 10*1024**2 or file_sha(source) != plan['parent_sha256']:
        raise ValueError('Training source differs from the captured parent CIF')
    instrument=Path(request['instrument'])
    if instrument.is_symlink() or not 0<instrument.stat().st_size<=1024**2 or file_sha(instrument)!=plan['instrument_sha256']:
        raise ValueError('Training profile differs from the captured instrument file')
    if request['settings']['dependencies'] and [row['directory'] for row in plan['blocks']]!=[
            'blocks/'+key for key in request['settings']['dependencies']]:
        raise ValueError('Training plan does not cover the exact declared library dependencies')
    directories = checked_blocks(plan_path, plan,grid=request['grid'])
    out = Path(request['destination'])
    if out.is_symlink():raise ValueError('Training recovery needs an owned regular directory')
    out=out.resolve()
    if paired:
        from .structural_training_checkpoint import owned_run
        run_identity=digest(dict(plan_sha256=plan_sha,parent_sha256=plan['parent_sha256'],
            instrument_sha256=plan['instrument_sha256'],grid=request['grid'],settings=settings,
            implementation_sha256=package_signature()))
        owner=owned_run(out,run_identity)
    else:
        out.mkdir(parents=True,exist_ok=False);owner=nullcontext(out)
    with owner:
        if paired:
            completed=_completed_training(request,out,plan_sha)
            if completed is not None:return completed
            from . import structural_training_segments as segments
            completed=segments.completed(request,out,plan_sha)
            if completed is not None:return completed
        began = time.monotonic()
        write_json(out/'progress.json', dict(state='assembling_examples', blocks=len(directories)))
        controls = settings['augmentation']
        augmenter = (None if controls is None else ObservationAugmenter(Controls(**controls),
                     instrument_sha256=plan['instrument_sha256']))
        cache=out/'working-features'
        if paired and cache.is_symlink():raise ValueError('Training working data must remain in an owned directory')
        if paired and ((cache/'store.json').exists() or (cache/'store.json').is_symlink()):
            cohort=load(cache)
            if (cohort['input_sha256']!=[row['examples_sha256'] for row in plan['blocks']]
                    or cohort['parent_sha256']!=plan['parent_sha256']
                    or cohort['instrument_sha256']!=plan['instrument_sha256']):
                cohort['features'].close()
                raise ValueError('Recovered feature cache differs from the captured native inputs')
        elif paired:
            cache.mkdir(exist_ok=True)
            if any(cache.iterdir()):
                raise ValueError('Unverified partial feature-cache files remain; refusing to replace or use them')
            # One automatic path for new paired training. The reader retains
            # native arrays and verifies exact reconstructed feature rows;
            # it never materializes the expanded multi-gigabyte feature plane.
            cohort=build_replay(directories,bins=settings['bins'],split_seed=settings['split_seed'],
                augmenter=augmenter,names='examples',destination=cache)
        else:
            cohort = build(directories, cache, bins=settings['bins'],
                           split_seed=settings['split_seed'], augmenter=augmenter)
        try:
            if paired and cohort.get('mode_projection_inventory'):
                from .structural_family_targets import build as family_targets
                from .structural_paired_blocks import read as read_pairs
                write_json(out/'progress.json',dict(state='assembling_mode_family_targets',blocks=len(directories)))
                families=family_targets(lambda:(read_pairs(path,names='examples') for path in directories),cohort)
                cohort.update(mode_family_targets=families,mode_family_target_identity=families.metadata['identity'],
                    mode_family_supervision=families.metadata)
            write_json(out/'progress.json', dict(state='training', examples=len(cohort['features']),
                independent_groups=len(set(cohort['groups'])),feature_storage=cohort['features'].storage))
            remaining=(None if not paired or segments.SEGMENT_SECONDS is None else
                max(1e-9,float(segments.SEGMENT_SECONDS)-(time.monotonic()-began)))
            model = train(cohort, out/'model', epochs=settings['epochs'], seed=settings['seed'],
                          threads=settings['workers'],**(dict(_segment_seconds=remaining,
                              _resume=resume_directory(plan_path,plan)) if paired else {}))
        finally:
            cohort['features'].close()
        if (file_sha(plan_path) != plan_sha or file_sha(source) != plan['parent_sha256']
                or file_sha(instrument)!=plan['instrument_sha256']):
            raise ValueError('Training source or declared plan changed during the job')
        checked_blocks(plan_path, plan)
        if paired and model.get('schema')=='radar-attention-training-segment/v1':
            return segments.publish(request,out,plan_sha,model,seconds=time.monotonic()-began)
        receipt = dict(schema='radar-structural-training-worker/v1', status='trained',
            plan_sha256=plan_sha, settings=copy.deepcopy(settings),
            parent_sha256=plan['parent_sha256'], instrument_sha256=plan['instrument_sha256'],
            implementation_sha256=package_signature(), examples=model['examples'],
            independent_groups=model['independent_groups'], classes=model['classes'],
            allocation=model['allocation'], test=model['test'],
            output=str(out), artifacts=['model/model.json', 'model/weights.npz', 'training.json'],
            weights_sha256=model['weights_sha256'], observations_included=False,
            screening_qualified=False, candidate_pruning_authorized=False, scientific_result=False,
            seconds=time.monotonic()-began)
        if paired:receipt['dataset']=model['dataset']
        write_json(out/'training.json', receipt)
        write_json(out/'progress.json', dict(state='complete', examples=model['examples'],
                                           screening_qualified=False))
        return publish_result('structural-training',request,receipt,
            ['model/model.json','model/weights.npz','training.json'])
