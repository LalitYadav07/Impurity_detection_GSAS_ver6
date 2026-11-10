"""Descriptive cohort counts; no numerical imports or scientific qualification.

Generated curves, deduplicated clean pairs and noisy feature rows are different
denominators. Keep them explicit in portable receipts and lightweight UI reads.
"""

SCHEMA='radar-structural-training-counts/v1'
COVERAGE_SCHEMA='radar-structural-training-counts/v2'
REPLAY_SCHEMA='radar-structural-training-counts/v3'
CALIBRATION_SCHEMA='radar-structural-training-counts/v4'
CALIBRATION_REPLAY_SCHEMA='radar-structural-training-counts/v5'
PARTITIONS=('train','validation','test')
CALIBRATED_PARTITIONS=(*PARTITIONS,'calibration')


def partition_names(value):
    schema=value.get('schema')
    return CALIBRATED_PARTITIONS if isinstance(schema,str) and schema in {CALIBRATION_SCHEMA,CALIBRATION_REPLAY_SCHEMA} else PARTITIONS


def describe(groups,split,clean,*,generated_pairs,simulated_observations,
             compatible_labels=None,classes=None,row_layout=None,calibration=False):
    from . import structural_replay_limits as limits
    if type(calibration)is not bool or not isinstance(groups,list) or not groups:
        raise ValueError('Complete unique feature rows and physical trajectory assignments required')
    if row_layout is not None:limits.validate(row_layout,rows=len(groups))
    large=len(groups)>131072
    partitions=CALIBRATED_PARTITIONS if calibration else PARTITIONS
    if calibration and classes is None:raise ValueError('Calibration counts need complete native family coverage')
    if (large and (row_layout is None or classes is None)):
        raise ValueError('Large training counts need the bounded native-row layout and complete family inventory')
    if (not isinstance(groups,list) or not groups or len(groups)>(limits.MAX_ROWS if large else 131072)
            or len(split)!=len(groups) or len(clean)!=len(groups)
            or any(not isinstance(g,str) or not g or len(g)>500 for g in groups)
            or any(type(code) is not int or code not in range(len(partitions)) for code in split)
            or any(type(value) is not bool for value in clean)):
        raise ValueError('Complete unique feature rows and physical trajectory assignments required')
    assigned={};parts={}
    for group,code in zip(groups,split):
        if group in assigned and assigned[group]!=code:
            raise ValueError('Related clean/noisy trajectories cannot cross holdouts')
        assigned[group]=code
    for code,name in enumerate(partitions):
        selected=[i for i,value in enumerate(split) if value==code]
        pairs=sum(clean[i] for i in selected)
        parts[name]=dict(rows=len(selected),clean_pairs=pairs,
            simulated_only_rows=len(selected)-pairs,
            trajectory_groups=len({groups[i] for i in selected}))
    result=dict(schema=SCHEMA,generated_pairs=generated_pairs,
        simulated_observations_generated=simulated_observations,
        unique_rows=len(groups),unique_clean_pairs=sum(clean),
        unique_simulated_only_rows=len(groups)-sum(clean),
        trajectory_groups=len(assigned),partitions=parts)
    if (compatible_labels is None)!=(classes is None):
        raise ValueError('Family coverage needs both native classes and compatible labels')
    if classes is not None:
        result.update(schema=CALIBRATION_SCHEMA if calibration else COVERAGE_SCHEMA,
            family_coverage=_families(groups,split,clean,compatible_labels,classes,partitions=partitions))
    if large:result.update(schema=CALIBRATION_REPLAY_SCHEMA if calibration else REPLAY_SCHEMA,row_layout=row_layout.copy())
    return validate(result)


def _families(groups,split,clean,labels,classes,*,partitions=PARTITIONS):
    if (not isinstance(classes,list) or not 2<=len(classes)<=230
            or any(type(c)is not int or not 1<=c<=230 for c in classes)
            or classes!=sorted(set(classes))):
        raise ValueError('Family coverage needs the exact ordered native class inventory')
    cells=[[dict(rows=0,clean_pairs=0,simulated_only_rows=0,groups=set(),clean_groups=set())
            for _ in classes] for _ in partitions]
    seen=0
    for i,row in enumerate(labels):
        if (i>=len(groups) or not isinstance(row,list) or len(row)!=len(classes)
                or any(type(v)is not bool for v in row) or not any(row)):
            raise ValueError('Family coverage needs one complete compatible-label row per feature')
        seen+=1
        for c,allowed in enumerate(row):
            if not allowed:continue
            cell=cells[split[i]][c];cell['rows']+=1;cell['groups'].add(groups[i])
            cell['clean_pairs' if clean[i] else 'simulated_only_rows']+=1
            if clean[i]:cell['clean_groups'].add(groups[i])
    if seen!=len(groups):raise ValueError('Family coverage is missing feature rows')
    parts={}
    for name,part in zip(partitions,cells):
        parts[name]={str(c):dict(rows=cell['rows'],clean_pairs=cell['clean_pairs'],
            simulated_only_rows=cell['simulated_only_rows'],trajectory_groups=len(cell['groups']),
            clean_trajectory_groups=len(cell['clean_groups'])) for c,cell in zip(classes,part)}
    return dict(classes=classes,partitions=parts,membership='compatible_native_labels_nonadditive')


def validate(value):
    keys={'schema','generated_pairs','simulated_observations_generated','unique_rows',
          'unique_clean_pairs','unique_simulated_only_rows','trajectory_groups','partitions'}
    large=isinstance(value,dict) and isinstance(value.get('schema'),str) and value['schema'] in {REPLAY_SCHEMA,CALIBRATION_REPLAY_SCHEMA}
    covered=isinstance(value,dict) and isinstance(value.get('schema'),str) and value['schema'] in {COVERAGE_SCHEMA,REPLAY_SCHEMA,CALIBRATION_SCHEMA,CALIBRATION_REPLAY_SCHEMA}
    partitions=partition_names(value) if isinstance(value,dict) else PARTITIONS
    from . import structural_replay_limits as limits
    maximum=limits.MAX_ROWS if large else 131072
    if (not isinstance(value,dict) or set(value)!=keys|({'family_coverage'} if covered else set())|({'row_layout'} if large else set())
            or not isinstance(value.get('schema'),str) or value['schema'] not in {SCHEMA,COVERAGE_SCHEMA,REPLAY_SCHEMA,CALIBRATION_SCHEMA,CALIBRATION_REPLAY_SCHEMA}
            or any(type(value[k]) is not int or not 0<=value[k]<=10**9
                   for k in keys-{'schema','partitions'})
            or not isinstance(value['partitions'],dict) or set(value['partitions'])!=set(partitions)):
        raise ValueError('Invalid structural training-count receipt')
    if large:
        limits.validate(value['row_layout'],rows=value['unique_rows'])
        if value['trajectory_groups']>4096:
            raise ValueError('Replay counts exceed the independent-group partition inventory')
    rows=clean=simulated=groups=0
    for name in partitions:
        part=value['partitions'][name]
        if (not isinstance(part,dict) or set(part)!={'rows','clean_pairs','simulated_only_rows','trajectory_groups'}
                or any(type(n) is not int or not 0<=n<=maximum for n in part.values())
                or part['rows']!=part['clean_pairs']+part['simulated_only_rows']
                or not 0<part['trajectory_groups']<=part['clean_pairs']<=part['rows']):
            raise ValueError('Invalid independent training/validation/test counts')
        rows+=part['rows'];clean+=part['clean_pairs'];simulated+=part['simulated_only_rows'];groups+=part['trajectory_groups']
    if (not 0<rows<=maximum or value['unique_rows']!=rows or value['unique_clean_pairs']!=clean
            or value['unique_simulated_only_rows']!=simulated or value['trajectory_groups']!=groups
            or value['generated_pairs']<clean or value['simulated_observations_generated']<simulated):
        raise ValueError('Structural training-count totals are inconsistent')
    if covered:_validate_families(value['family_coverage'],value['partitions'],maximum=maximum,partitions=partitions)
    return value


def _validate_families(value,parts,*,maximum=131072,partitions=PARTITIONS):
    if (not isinstance(value,dict) or set(value)!={'classes','partitions','membership'}
            or value['membership']!='compatible_native_labels_nonadditive'
            or not isinstance(value['classes'],list) or not 2<=len(value['classes'])<=230
            or any(type(c)is not int or not 1<=c<=230 for c in value['classes'])
            or value['classes']!=sorted(set(value['classes']))
            or not isinstance(value['partitions'],dict) or set(value['partitions'])!=set(partitions)):
        raise ValueError('Invalid native family-coverage inventory')
    for name in partitions:
        part=value['partitions'][name];total=parts[name]
        if not isinstance(part,dict) or set(part)!={str(c) for c in value['classes']}:
            raise ValueError('Family coverage does not cover every declared class and holdout')
        for cell in part.values():
            if (not isinstance(cell,dict) or set(cell)!={'rows','clean_pairs','simulated_only_rows',
                    'trajectory_groups','clean_trajectory_groups'}
                    or any(type(v)is not int or not 0<=v<=maximum for v in cell.values())
                    or cell['rows']!=cell['clean_pairs']+cell['simulated_only_rows']
                    or any(cell[k]>total[k] for k in ('rows','clean_pairs','simulated_only_rows','trajectory_groups'))
                    or not 0<=cell['clean_trajectory_groups']<=min(cell['trajectory_groups'],cell['clean_pairs'])
                    or cell['trajectory_groups']>cell['rows']
                    or bool(cell['rows'])!=bool(cell['trajectory_groups'])
                    or bool(cell['clean_pairs'])!=bool(cell['clean_trajectory_groups'])):
                raise ValueError('Native family-coverage counts are inconsistent with their partition')
        if any(sum(cell[k] for cell in part.values())<total[k]
               for k in ('rows','clean_pairs','simulated_only_rows','trajectory_groups')):
            raise ValueError('Native family coverage omitted partition members')
