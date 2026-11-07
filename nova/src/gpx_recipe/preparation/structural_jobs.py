"""Closed scientific-worker operations for structural ranking preparation.

Native example blocks and trained weights are portable data. Feature caches,
native construction projects and optimizer state stay in the owned worker.
Neither operation qualifies a classifier to remove refinement candidates.
"""
from __future__ import annotations

from dataclasses import asdict
import copy
import json
from pathlib import Path
import time

from ..execution import write_json
from ..symmetry.compiler import file_sha, package_signature
from ..symmetry.contracts import fields, number
from .batch import relative
from .structural_augmentation import Controls, ObservationAugmenter
from .tof_kernel_arrays import checksum, read_bounded
from .reference import validate_grid

PLAN_SCHEMA = 'radar-structural-training-plan/v1'
MAX_PLAN_BYTES = 8*1024**2
MAX_INPUT_BYTES = 480*1024**2
# The shared packet admits parent, profile, plan, and two files per block.
MAX_BLOCKS = (10000-3)//2


def validate_examples(request):
    from .worker import validate_request
    request = fields(request, ['source', 'instrument', 'child', 'grid', 'settings', 'destination'])
    settings = fields(request['settings'], ['candidate', 'mode_radius', 'cell_radius', 'seed'],
                      dict(trajectories=64, steps=4, trajectory_slice=None))
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
    base = dict(request, settings={k: settings[k] for k in ('candidate', 'mode_radius', 'cell_radius')})
    validated = validate_request('structural', base)
    settings.update(validated['settings'])
    return dict(validated, settings=settings)


def validate_training(request):
    request = fields(request, ['source', 'instrument', 'grid', 'plan', 'destination', 'settings'])
    settings = fields(request['settings'], [], dict(bins=2048, split_seed=3917,
        epochs=60, seed=89113, workers=4, augmentation=asdict(Controls()), dependencies=[]))
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
    value = fields(value, ['schema', 'parent_sha256', 'instrument_sha256', 'blocks'])
    if (value['schema'] != PLAN_SCHEMA or not checksum(value['parent_sha256'])
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
    from .structural_cohort import read_block
    destination = Path(destination).resolve()
    source = Path(source)
    if source.is_symlink() or not source.is_file() or not 0 < source.stat().st_size <= 10*1024**2:
        raise ValueError('Training requires a bounded regular parent CIF')
    parent = file_sha(source); rows = []; profile = None; total = 0
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
        block = read_block(directory)
        if block.manifest['source_sha256'] != parent:
            raise ValueError('A training block belongs to another parent CIF')
        current = block.manifest['instrument_sha256']
        if profile is not None and current != profile:
            raise ValueError('Train each exact instrument-profile context separately')
        profile = current
        rows.append(dict(directory=name, manifest_sha256=file_sha(directory/'examples.json'),
                         examples_sha256=file_sha(directory/'examples.npz')))
    value = dict(schema=PLAN_SCHEMA, parent_sha256=parent, instrument_sha256=profile, blocks=rows)
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
    from .structural_cohort import read_block
    total = 0; directories = []
    for row in plan['blocks']:
        paths = [confined(Path(plan_path).parent, str(Path(row['directory'])/name))
                 for name in ('examples.json', 'examples.npz')]
        total += sum(path.stat().st_size for path in paths)
        if total > MAX_INPUT_BYTES:
            raise ValueError('Training input arrays exceed one bounded worker packet')
        if [file_sha(path) for path in paths] != [row['manifest_sha256'], row['examples_sha256']]:
            raise ValueError('A captured structural training block changed')
        metadata = (read_block(paths[0].parent).manifest if validate_arrays else
                    json.loads(read_bounded(paths[0], MAX_PLAN_BYTES)))
        if (metadata['source_sha256'] != plan['parent_sha256']
                or metadata['instrument_sha256'] != plan['instrument_sha256']):
            raise ValueError('Training block has a different parent or profile context')
        directories.append(paths[0].parent)
    return directories


def execute_examples(request):
    from .structural_examples import generate
    request = validate_examples(request); settings = request['settings']
    base = dict(request, settings={k: settings[k] for k in ('candidate', 'mode_radius', 'cell_radius')})
    detail = generate(base, **{k: settings[k] for k in ('seed', 'trajectories', 'steps', 'trajectory_slice')})
    return publish_result('structural-examples',request,dict(examples=detail['examples'],
        candidate_id=detail['candidate']['id'],trajectory_slice=detail['trajectory_slice'],
        mode_dimensions=detail['mode_dimensions'],cell_dimensions=detail['cell_dimensions'],
        examples_sha256=detail['examples_sha256'],seconds=detail['seconds']),
        ['examples.json','examples.npz'])


def execute_training(request):
    from .structural_store import build
    from .structural_ranker import train
    request = validate_training(request); settings = request['settings']
    plan_path = Path(request['plan']).resolve(); plan_sha = file_sha(plan_path); plan = read_plan(plan_path)
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
    out = Path(request['destination']).resolve(); out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    write_json(out/'progress.json', dict(state='assembling_examples', blocks=len(directories)))
    controls = settings['augmentation']
    augmenter = (None if controls is None else ObservationAugmenter(Controls(**controls),
                 instrument_sha256=plan['instrument_sha256']))
    cohort = build(directories, out/'working-features', bins=settings['bins'],
                   split_seed=settings['split_seed'], augmenter=augmenter)
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
    write_json(out/'training.json', receipt)
    write_json(out/'progress.json', dict(state='complete', examples=model['examples'],
                                       screening_qualified=False))
    return publish_result('structural-training',request,receipt,
        ['model/model.json','model/weights.npz','training.json'])
