"""Descriptive cohort counts; no numerical imports or scientific qualification.

Generated curves, deduplicated clean pairs and noisy feature rows are different
denominators. Keep them explicit in portable receipts and lightweight UI reads.
"""

SCHEMA='radar-structural-training-counts/v1'
PARTITIONS=('train','validation','test')


def describe(groups,split,clean,*,generated_pairs,simulated_observations):
    if (not isinstance(groups,list) or not groups or len(groups)>131072
            or len(split)!=len(groups) or len(clean)!=len(groups)
            or any(not isinstance(g,str) or not g or len(g)>500 for g in groups)
            or any(type(code) is not int or code not in (0,1,2) for code in split)
            or any(type(value) is not bool for value in clean)):
        raise ValueError('Complete unique feature rows and physical trajectory assignments required')
    assigned={};parts={}
    for group,code in zip(groups,split):
        if group in assigned and assigned[group]!=code:
            raise ValueError('Related clean/noisy trajectories cannot cross holdouts')
        assigned[group]=code
    for code,name in enumerate(PARTITIONS):
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
    return validate(result)


def validate(value):
    keys={'schema','generated_pairs','simulated_observations_generated','unique_rows',
          'unique_clean_pairs','unique_simulated_only_rows','trajectory_groups','partitions'}
    if (not isinstance(value,dict) or set(value)!=keys or value.get('schema')!=SCHEMA
            or any(type(value[k]) is not int or not 0<=value[k]<=10**9
                   for k in keys-{'schema','partitions'})
            or not isinstance(value['partitions'],dict) or set(value['partitions'])!=set(PARTITIONS)):
        raise ValueError('Invalid structural training-count receipt')
    rows=clean=simulated=groups=0
    for name in PARTITIONS:
        part=value['partitions'][name]
        if (not isinstance(part,dict) or set(part)!={'rows','clean_pairs','simulated_only_rows','trajectory_groups'}
                or any(type(n) is not int or not 0<=n<=131072 for n in part.values())
                or part['rows']!=part['clean_pairs']+part['simulated_only_rows']
                or not 0<part['trajectory_groups']<=part['clean_pairs']<=part['rows']):
            raise ValueError('Invalid independent training/validation/test counts')
        rows+=part['rows'];clean+=part['clean_pairs'];simulated+=part['simulated_only_rows'];groups+=part['trajectory_groups']
    if (not 0<rows<=131072 or value['unique_rows']!=rows or value['unique_clean_pairs']!=clean
            or value['unique_simulated_only_rows']!=simulated or value['trajectory_groups']!=groups
            or value['generated_pairs']<clean or value['simulated_observations_generated']<simulated):
        raise ValueError('Structural training-count totals are inconsistent')
    return value
