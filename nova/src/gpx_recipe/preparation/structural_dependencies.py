"""Finite example/training dependencies inside the existing preparation queue."""
import copy
import json
from pathlib import Path
import shutil

from ..symmetry.compiler import file_sha
from ..symmetry.contracts import digest,fields
from .structural_training_blocks import REPRESENTATION,PARENT_RANGE,parent_context

EXAMPLES='structural-examples'
TRAINING='structural-training'
DEFAULT_TRAJECTORY_UNIT=8
COORDINATE_ROWS_PER_UNIT=2560


def trajectory_unit(model,steps,*,grid=None):
    """Choose scheduling granularity, never reduce the physical portfolio.

    A complete measured 228-coordinate two-trajectory unit took 233 seconds;
    its eight-trajectory counterpart exceeds a batch budget. Discovery mode
    counts are a conservative cost estimate, not native scientific authority.
    Unknown counts use small units. Actual native dimensions, byte limits,
    allocation and deadline checks remain authoritative in the worker.
    """
    modes=model.get('settings',{}).get('candidate',{}).get('modes')
    if not isinstance(modes,list) or not modes:return 2
    grid=model.get('grid',{}) if grid is None else grid
    if not isinstance(grid,dict):return 2
    bins=grid.get('bins',4096)
    if type(bins)is not int or bins<3:return 2
    resolution=max(1,(bins+4095)//4096)
    # Retain the parent-only control beside every whole trajectory and allow
    # all six lattice coordinates when estimating this computational unit.
    work=(len(modes)+6)*(steps+1)*resolution
    return max(1,min(DEFAULT_TRAJECTORY_UNIT,COORDINATE_ROWS_PER_UNIT//work))


def validate_entries(entries):
    """A library cannot declare training on a missing or partial sample scope."""
    from .structural_jobs import validate_examples,validate_training
    examples={};training={};portfolios={};contexts={};training_contexts={}
    for row in entries:
        model=row['model'];kind=model['kind']
        if kind in {EXAMPLES,TRAINING}:
            fields(model,['kind','bank','instrument','grid','settings']+(['child'] if kind==EXAMPLES else []))
        if kind==EXAMPLES:
            request=dict(source='parent.cif',instrument=model['instrument'],child=model['child'],
                grid=model['grid'],settings=model['settings'],destination='compiled')
            settings=validate_examples(request)['settings']
            if settings!=model['settings']:raise ValueError('Example settings must retain their complete declared defaults')
            examples.setdefault(model['bank'],[]).append(row['id'])
            representation=settings.get('representation')
            contexts.setdefault(model['bank'],set()).add(digest(
                [model['instrument'],model['grid'],representation]))
            if representation is not None:parent_context(settings)
            portfolio=copy.deepcopy(model);portfolio['settings'].pop('trajectory_slice')
            group=portfolios.setdefault(digest(portfolio),dict(total=settings['trajectories'],intervals=[]))
            group['intervals'].append(settings['trajectory_slice'])
        elif kind==TRAINING:
            request=dict(source='parent.cif',instrument=model['instrument'],grid=model['grid'],
                plan='plan.json',settings=model['settings'],destination='model')
            settings=validate_training(request)['settings']
            if settings!=model['settings'] or model['bank'] in training:
                raise ValueError('A physical bank needs one complete training declaration')
            training[model['bank']]=settings['dependencies']
            training_contexts[model['bank']]=digest([model['instrument'],model['grid'],settings.get('representation')])
    if set(training)!=set(examples):
        raise ValueError('Every structural example bank needs its dependent training entry')
    for bank,keys in examples.items():
        if training[bank]!=sorted(keys) or contexts[bank]!={training_contexts[bank]}:
            raise ValueError('Training must include every declared example block for its bank')
        paired_contexts={parent_context(row['model']['settings']) for row in entries
            if row['model']['kind']==EXAMPLES and row['model']['bank']==bank
            and row['model']['settings'].get('representation')==REPRESENTATION}
        if len(paired_contexts)>1:
            from .structural_training_strata import Inventory
            strata=Inventory()
            for row in entries:
                if row['model']['kind']==EXAMPLES and row['model']['bank']==bank:
                    strata.add('blocks/'+row['id'],dict(row['model']['settings'],grid=row['model']['grid']))
            strata.describe()
    for group in portfolios.values():
        start=0
        for lower,upper in sorted(group['intervals']):
            if lower!=start:raise ValueError('Structural example trajectories must have no missing or overlapping slices')
            start=upper
        if start!=group['total']:
            raise ValueError('Structural example slices must cover the complete declared trajectory portfolio')


def example_models(models, *, trajectories=64, shard=None, steps=4, seed=89113,bank_grids=None):
    """Expand each structural context into complete, disjoint trajectory slices."""
    from .structural_mode_coverage import TARGETED_SCHEMA
    if (type(trajectories) is not int or not 1<=trajectories<=4096
            or type(steps)is not int or not 1<=steps<=32
            or (shard is not None and (type(shard) is not int or not 1<=shard<=trajectories))):
        raise ValueError('Declare bounded complete structural trajectories and shard size')
    result=[]
    for model in models:
        if model['kind']!='structural':continue
        grid=None if bank_grids is None else bank_grids.get(model['bank'])
        size=min(trajectories,trajectory_unit(model,steps,grid=grid)) if shard is None else shard
        for first in range(0,trajectories,size):
            row=copy.deepcopy(model);row['kind']=EXAMPLES
            row['settings'].update(seed=seed,trajectories=trajectories,steps=steps,
                trajectory_slice=[first,min(first+size,trajectories)],
                representation=REPRESENTATION,parent_seed=seed^0x63BC32D9,
                parent_relative_metric_range=list(PARENT_RANGE),mode_coverage=TARGETED_SCHEMA)
            result.append(row)
    return result


def append_training(declared, banks):
    """Seal a single training entry per bank after its example identities exist."""
    from .structural_jobs import validate_training
    by_bank={};representations={}
    for row in declared:
        if row['kind']==EXAMPLES:
            by_bank.setdefault(row['bank'],[]).append(digest(row)[:24])
            representations.setdefault(row['bank'],set()).add(row['settings'].get('representation'))
    for bank,dependencies in sorted(by_bank.items()):
        if len(representations[bank])!=1:
            raise ValueError('A bank cannot mix structural training representations')
        profile=banks[bank]
        settings=dict(dependencies=sorted(dependencies))
        if representations[bank]=={REPRESENTATION}:
            from .structural_attention_weights import FIXED_REFERENCE_KIND
            settings.update(representation=REPRESENTATION,reference_kind=FIXED_REFERENCE_KIND)
        settings=validate_training(dict(source='parent.cif',instrument=profile['instrument'],
            grid=profile['grid'],plan='plan.json',destination='model',
            settings=settings))['settings']
        declared.append(dict(kind=TRAINING,bank=bank,settings=settings,**profile))


def ready(library, model):
    if model['kind']!=TRAINING:return True
    return all(library.row(key)['state']=='prepared' for key in model['settings']['dependencies'])


def blocked_training(library):
    """Report failed prerequisites without changing or narrowing the plan."""
    library.refresh();rows={r['id']:r for r in library.document['entries']};blocked=[]
    for row in rows.values():
        if row['model']['kind']!=TRAINING or row['state']!='pending':continue
        failures=[key for key in row['model']['settings']['dependencies']
                  if rows[key]['state'] in {'failed','cancelled','unsupported'}]
        if failures:blocked.append(dict(entry=row['id'],dependencies=failures))
    return blocked


def dependencies(library, model):
    """Read only verified artifacts belonging to this exact bank and scope."""
    from .structural_training_blocks import metadata,representation,context
    result=[]
    keys=model['settings']['dependencies']
    if not keys:raise ValueError('A library training entry needs its complete example dependencies')
    for key in keys:
        row=library.row(key);other=row['model']
        if (row['state']!='prepared' or other['kind']!=EXAMPLES
                or other['bank']!=model['bank'] or other['instrument']!=model['instrument']
                or other['grid']!=model['grid']):
            raise ValueError('Complete the exact bank’s structural example dependencies before training')
        assets={Path(name).name:path for name,path in library.prepared_assets(key).items()}
        from .structural_mode_projection import FILES
        base={'entry.json','examples.json','examples.npz'}
        if set(assets) not in (base,base|set(FILES)):
            raise ValueError('A training dependency has an incomplete example inventory')
        document=metadata(assets['examples.json'].parent)
        if (representation(document)!=model['settings'].get('representation')
                or representation(document)!=other['settings'].get('representation')):
            raise ValueError('A training dependency changed its structural representation')
        if representation(document)==REPRESENTATION and (
                context(document)!=parent_context(other['settings'])
                or document.get('grid')!=other['grid']
                or document.get('trajectory_slice')!=other['settings']['trajectory_slice']
                or document.get('candidate')!=other['settings']['candidate']
                or document.get('mode_coverage',{}).get('schema')!=other['settings'].get('mode_coverage')):
            raise ValueError('A paired dependency changed its declared candidate or parent trajectories')
        result.append((key,assets))
    return result


def stage(library, model, payload, directory, *, resume=None):
    """Copy immutable dependency bytes into the job's recoverable input packet."""
    from .structural_jobs import create_plan
    directory=Path(directory);paths=[]
    for key,assets in dependencies(library,model):
        target=directory/'blocks'/key;target.mkdir(parents=True,exist_ok=True)
        from .structural_mode_projection import FILES
        for name in ('examples.json','examples.npz')+tuple(n for n in FILES if n in assets):
            path=target/name;expected=file_sha(assets[name])
            if path.is_symlink() or (path.exists() and file_sha(path)!=expected):
                raise ValueError('A captured training dependency changed')
            if not path.exists():shutil.copyfile(assets[name],path)
            if file_sha(path)!=expected:raise ValueError('Training input changed while staging')
        paths.append(target)
    plan=directory/'training-plan.json'
    create_plan(payload['source'],paths,plan,recover_identical=True,resume=resume)
    payload['plan']=str(plan)
    return payload


def verify(library, model, request,*,expected_implementation=None):
    """Publication must still refer to every declared dependency, after recovery."""
    from .structural_jobs import read_plan,resume_directory,PROJECTED_PLAN_SCHEMAS
    plan=read_plan(request['plan'])
    resume_directory(Path(request['plan']),plan,expected_implementation=expected_implementation)
    expected=[]
    for key,assets in dependencies(library,model):
        row=dict(directory='blocks/'+key,manifest_sha256=file_sha(assets['examples.json']),
                 examples_sha256=file_sha(assets['examples.npz']))
        d=json.loads(assets['examples.json'].read_text())
        stamp=d.get('mode_projection')
        if stamp is not None and plan['schema'] not in PROJECTED_PLAN_SCHEMAS:
            raise ValueError('The training plan omitted its native mode coordinates')
        if plan['schema'] in PROJECTED_PLAN_SCHEMAS:row['mode_projection']=stamp
        expected.append(row)
    if (plan['blocks']!=expected or plan['parent_sha256']!=file_sha(request['source'])
            or plan['instrument_sha256']!=file_sha(request['instrument'])
            or plan.get('representation')!=model['settings'].get('representation')):
        raise ValueError('Training result no longer covers its exact library dependencies')
