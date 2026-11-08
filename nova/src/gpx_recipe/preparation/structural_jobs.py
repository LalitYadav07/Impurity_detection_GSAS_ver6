"""Closed scientific-worker operations for structural ranking preparation.

Native example blocks and trained weights are portable data. Feature caches,
native construction projects and optimizer state stay in the owned worker.
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
MAX_PLAN_BYTES = 8*1024**2
MAX_INPUT_BYTES = 480*1024**2
# The shared packet admits parent, profile, plan, and two files per block.
MAX_BLOCKS = (10000-3)//2


def validate_examples(request):
    from .worker import validate_request
    request = fields(request, ['source', 'instrument', 'child', 'grid', 'settings', 'destination'])
    settings = fields(request['settings'], ['candidate', 'mode_radius', 'cell_radius', 'seed'],
                      dict(trajectories=64, steps=4, trajectory_slice=None,
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
    paired=isinstance(value,dict) and value.get('schema')==PAIRED_PLAN_SCHEMA
    value = fields(value, ['schema', 'parent_sha256', 'instrument_sha256', 'blocks']+
                   (['representation'] if paired else []))
    if paired:validate_representation(value)
    if (value['schema'] not in {PLAN_SCHEMA,PAIRED_PLAN_SCHEMA} or not checksum(value['parent_sha256'])
            or not checksum(value['instrument_sha256']) or not isinstance(value['blocks'], list)
            or not 1 <= len(value['blocks']) <= MAX_BLOCKS):
        raise ValueError('Training requires a bounded parent/profile example inventory')
    seen = set()
    for row in value['blocks']:
        fields(row, ['directory', 'manifest_sha256', 'examples_sha256'])
        name = relative(row['directory'])
        if name in seen or any(not checksum(row[k]) for k in ('manifest_sha256', 'examples_sha256')):
            raise ValueError('Training blocks need unique directories and exact content identities')
        seen.add(name)
    return value


def plan_inputs(path):
    """One packet inventory for NOVA capture and worker-side verification."""
    plan = read_plan(path)
    return [str(Path(row['directory'])/name) for row in plan['blocks']
            for name in ('examples.json', 'examples.npz')]


def create_plan(source, directories, destination, *, recover_identical=False):
    """Bind existing observation-free blocks without copying large arrays."""
    destination = Path(destination).resolve()
    source = Path(source)
    if source.is_symlink() or not source.is_file() or not 0 < source.stat().st_size <= 10*1024**2:
        raise ValueError('Training requires a bounded regular parent CIF')
    parent = file_sha(source); rows = []; profile = None; total = 0; formats=set();contexts=set()
    directories = list(directories)
    if not 1 <= len(directories) <= MAX_BLOCKS:
        raise ValueError('Declare a bounded structural training inventory')
    for directory in directories:
        directory = Path(directory).resolve()
        name = str(directory.relative_to(destination.parent))
        relative(name)
        total += sum((directory/name).stat().st_size for name in ('examples.json','examples.npz'))
        if total > MAX_INPUT_BYTES:
            raise ValueError('Training input arrays exceed one bounded worker packet')
        block = read_training_block(directory)
        formats.add(block_representation(block.manifest));contexts.add(block_context(block.manifest))
        if len(formats)!=1 or len(contexts)!=1:
            raise ValueError('Training blocks must share one representation and parent trajectory context')
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
        value.update(schema=PAIRED_PLAN_SCHEMA,representation=REPRESENTATION)
    # Validate before publishing, including duplicate logical block paths.
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


def checked_blocks(plan_path, plan, *, validate_arrays=True):
    from ..symmetry.compiler import confined
    total = 0; directories = [];contexts=set()
    for row in plan['blocks']:
        paths = [confined(Path(plan_path).parent, str(Path(row['directory'])/name))
                 for name in ('examples.json', 'examples.npz')]
        total += sum(path.stat().st_size for path in paths)
        if total > MAX_INPUT_BYTES:
            raise ValueError('Training input arrays exceed one bounded worker packet')
        if [file_sha(path) for path in paths] != [row['manifest_sha256'], row['examples_sha256']]:
            raise ValueError('A captured structural training block changed')
        metadata = (read_training_block(paths[0].parent).manifest if validate_arrays else
                    block_metadata(paths[0].parent))
        if (metadata['source_sha256'] != plan['parent_sha256']
                or metadata['instrument_sha256'] != plan['instrument_sha256']):
            raise ValueError('Training block has a different parent or profile context')
        contexts.add(block_context(metadata))
        if block_representation(metadata)!=plan.get('representation') or len(contexts)!=1:
            raise ValueError('Training block representation or parent trajectory context differs from its plan')
        directories.append(paths[0].parent)
    return directories


def execute_examples(request):
    request = validate_examples(request); settings = request['settings']
    base = dict(request, settings={k: settings[k] for k in ('candidate', 'mode_radius', 'cell_radius')})
    options={k:settings[k] for k in ('seed','trajectories','steps','trajectory_slice')}
    began=time.monotonic()
    if settings.get('representation')==REPRESENTATION:
        from .structural_pairs import generate
        from .structural_paired_blocks import pack_block
        out=Path(request['destination']).resolve();out.mkdir(parents=True,exist_ok=False)
        detail=generate(dict(base,destination=str(out/'working-native')),
            parent_seed=settings['parent_seed'],parent_relative_range=settings['parent_relative_metric_range'],**options)
        packed=pack_block(out/'working-native',out/'portable',names='examples')
        for name in ('examples.json','examples.npz'):(out/'portable'/name).replace(out/name)
        examples_sha256=packed.manifest['pairs_sha256']
    else:
        from .structural_examples import generate
        detail=generate(base,**options)
        examples_sha256=detail['examples_sha256']
    receipt=dict(examples=detail['examples'],
        candidate_id=detail['candidate']['id'],trajectory_slice=detail['trajectory_slice'],
        mode_dimensions=detail['mode_dimensions'],cell_dimensions=detail['cell_dimensions'],
        examples_sha256=examples_sha256,seconds=time.monotonic()-began)
    if settings.get('representation')==REPRESENTATION:
        receipt.update(representation=REPRESENTATION,parent_seed=settings['parent_seed'],
            parent_relative_metric_range=settings['parent_relative_metric_range'],timing=detail['timing'])
    return publish_result('structural-examples',request,receipt,['examples.json','examples.npz'])


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
        from .structural_paired_store import build,load
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
    directories = checked_blocks(plan_path, plan)
    if any(json.loads(read_bounded(p/'examples.json',MAX_PLAN_BYTES)).get('grid')!=request['grid']
           for p in directories):
        raise ValueError('Training examples differ from the captured physical bank grid')
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
        began = time.monotonic()
        write_json(out/'progress.json', dict(state='assembling_examples', blocks=len(directories)))
        controls = settings['augmentation']
        augmenter = (None if controls is None else ObservationAugmenter(Controls(**controls),
                     instrument_sha256=plan['instrument_sha256']))
        cache=out/'working-features'
        if paired and (cache/'store.json').exists():
            cohort=load(cache)
            if (cohort['input_sha256']!=[row['examples_sha256'] for row in plan['blocks']]
                    or cohort['parent_sha256']!=plan['parent_sha256']
                    or cohort['instrument_sha256']!=plan['instrument_sha256']):
                cohort['features'].close()
                raise ValueError('Recovered feature cache differs from the captured native inputs')
        else:
            cohort = build(directories, cache, bins=settings['bins'],
                           split_seed=settings['split_seed'], augmenter=augmenter,
                           **({'names':'examples'} if paired else {}))
        try:
            write_json(out/'progress.json', dict(state='training', examples=len(cohort['features']),
                                                independent_groups=len(set(cohort['groups']))))
            model = train(cohort, out/'model', epochs=settings['epochs'], seed=settings['seed'],
                          threads=settings['workers'])
        finally:
            cohort['features'].close()
        if (file_sha(plan_path) != plan_sha or file_sha(source) != plan['parent_sha256']
                or file_sha(instrument)!=plan['instrument_sha256']):
            raise ValueError('Training source or declared plan changed during the job')
        checked_blocks(plan_path, plan)
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
