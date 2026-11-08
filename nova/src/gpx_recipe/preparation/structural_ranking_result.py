"""Observation-bound advisory ranking receipts and complete-scope ordering."""
import math
import re

from ..symmetry.contracts import digest

SCHEMA='radar-structural-ranking-evidence/v1'
SNAPSHOT_SCHEMA='radar-structural-snapshot-ranking-evidence/v1'
ADAPTIVE_SCHEMA='radar-structural-ranking-evidence/v2'
ADAPTIVE_SNAPSHOT_SCHEMA='radar-structural-snapshot-ranking-evidence/v2'


def _hash(value):return isinstance(value,str) and re.fullmatch('[a-f0-9]{64}',value) is not None


def observation_binding(value):
    if (not isinstance(value,dict) or set(value)!={'snapshot_sha256','calculation_source_sha256','acquisition_sha256','revision'}
            or any(not _hash(value[k]) for k in ('snapshot_sha256','calculation_source_sha256','acquisition_sha256'))
            or type(value['revision']) is not int or value['revision']<0):
        raise ValueError('Structural ranking needs its exact native snapshot and acquisition revision')
    return dict(value)


def unavailable(protocol,reason,*,snapshot=None,seconds=0.,checks=(),support=None,observation=None):
    result=dict(schema=SCHEMA,protocol_sha256=digest(protocol),source_sha256=protocol['source']['sha256'],
        implementation_sha256=protocol['implementation_sha256'],preparation_snapshot_sha256=snapshot,
        candidate_order=[c['id'] for c in protocol['candidates']],ranking_available=False,
        families=[],entry_id=None,bank=None,support=support,checks=list(checks),seconds=float(seconds),reason=reason[:1000],
        candidate_pruning_authorized=False,screening_qualified=False,
        native_refinement_required=True,complete_native_search_required=True)
    if observation is not None:result.update(schema=SNAPSHOT_SCHEMA,observation=observation_binding(observation))
    return result


def attach_adaptive(value,work_set):
    from .structural_adaptive_sets import validate_work_set
    validate_work_set(work_set,[r['space_group'] for r in value['families']])
    if not value['ranking_available']:raise ValueError('Unavailable ranking cannot attach an adaptive priority')
    return dict(value,schema=ADAPTIVE_SNAPSHOT_SCHEMA if 'observation' in value else ADAPTIVE_SCHEMA,
        adaptive_work_set=work_set)


def validate(value,protocol,*,snapshot=None,observation=None):
    required={'schema','protocol_sha256','source_sha256','implementation_sha256','preparation_snapshot_sha256',
        'candidate_order','ranking_available','families','entry_id','bank','support','checks','seconds','reason',
        'candidate_pruning_authorized','screening_qualified','native_refinement_required','complete_native_search_required'}
    schema=SCHEMA
    if observation is not None:
        observation=observation_binding(observation);required.add('observation');schema=SNAPSHOT_SCHEMA
        if not isinstance(value,dict) or value.get('observation')!=observation:
            raise ValueError('Structural ranking belongs to another snapshot revision')
    adaptive=isinstance(value,dict) and 'adaptive_work_set' in value
    if adaptive:
        required.add('adaptive_work_set');schema=ADAPTIVE_SNAPSHOT_SCHEMA if observation is not None else ADAPTIVE_SCHEMA
    candidates=[c['id'] for c in protocol['candidates']]
    if (protocol['kind']!='structural' or not isinstance(value,dict) or set(value)!=required
            or value['schema']!=schema or value['protocol_sha256']!=digest(protocol)
            or value['source_sha256']!=protocol['source']['sha256']
            or value['implementation_sha256']!=protocol['implementation_sha256']
            or (value['preparation_snapshot_sha256'] is not None and not _hash(value['preparation_snapshot_sha256']))
            or (snapshot is not None and value['preparation_snapshot_sha256']!=snapshot)
            or value['candidate_pruning_authorized'] is not False or value['screening_qualified'] is not False
            or value['native_refinement_required'] is not True or value['complete_native_search_required'] is not True
            or type(value['ranking_available']) is not bool
            or not isinstance(value['candidate_order'],list)
            or any(not isinstance(key,str) for key in value['candidate_order'])
            or len(value['candidate_order'])!=len(candidates) or set(value['candidate_order'])!=set(candidates)
            or not isinstance(value['reason'],str) or len(value['reason'])>1000
            or type(value['seconds']) not in (int,float) or not math.isfinite(value['seconds']) or not 0<=value['seconds']<=3600):
        raise ValueError('Structural ranking receipt differs from the complete captured source and scope')
    checks=value['checks']
    if not isinstance(checks,list) or len(checks)>128:raise ValueError('Structural ranking checks exceed their bounded inventory')
    seen=set()
    for row in checks:
        if (not isinstance(row,dict) or set(row)!={'entry','status','reason'}
                or not isinstance(row['entry'],str) or not re.fullmatch('[a-f0-9]{24}',row['entry'])
                or row['entry'] in seen or not isinstance(row['status'],str)
                or row['status'] not in {'compatible','incompatible','invalid_data','inference_unavailable'}
                or not isinstance(row['reason'],str) or len(row['reason'])>600):
            raise ValueError('Invalid structural ranking compatibility check')
        seen.add(row['entry'])
    support=value['support']
    if support is not None:
        if (not isinstance(support,dict) or set(support)!={'status','nearest_training_rms','validation_maximum_rms','statistical_coverage_qualified'}
                or not isinstance(support['status'],str)
                or support['status'] not in {'within_empirical_synthetic_support','outside_prepared_support'}
                or support['statistical_coverage_qualified'] is not False
                or any(type(support[k]) not in (int,float) or not math.isfinite(support[k]) or support[k]<0
                       for k in ('nearest_training_rms','validation_maximum_rms'))):
            raise ValueError('Invalid empirical structural ranking support')
    families=value['families']
    if not isinstance(families,list) or len(families)>230:raise ValueError('Invalid structural ranking family inventory')
    groups=set()
    for row in families:
        if (not isinstance(row,dict) or set(row)!={'space_group','model_score','candidate_ids'}
                or type(row['space_group']) is not int or not 1<=row['space_group']<=230 or row['space_group'] in groups
                or type(row['model_score']) not in (int,float) or not math.isfinite(row['model_score']) or not 0<=row['model_score']<=1
                or not isinstance(row['candidate_ids'],list) or not row['candidate_ids']
                or any(not isinstance(key,str) for key in row['candidate_ids'])
                or len(set(row['candidate_ids']))!=len(row['candidate_ids']) or not set(row['candidate_ids'])<=set(candidates)):
            raise ValueError('Structural ranking families must lead to declared native constructions')
        groups.add(row['space_group'])
    if value['ranking_available']:
        head=list(dict.fromkeys(key for row in families for key in row['candidate_ids']))
        if (not families or not isinstance(value['entry_id'],str) or not re.fullmatch('[a-f0-9]{24}',value['entry_id'])
                or not isinstance(value['bank'],str) or not 1<=len(value['bank'])<=100
                or not any(r['entry']==value['entry_id'] and r['status']=='compatible' for r in checks)
                or support is None or support['status']!='within_empirical_synthetic_support'
                or value['preparation_snapshot_sha256'] is None
                or value['candidate_order']!=head+[key for key in candidates if key not in head]):
            raise ValueError('Available ranking needs its checked bank, support and complete native routing')
        if adaptive:
            from .structural_adaptive_sets import validate_work_set
            validate_work_set(value['adaptive_work_set'],[r['space_group'] for r in families])
    elif (families or value['entry_id'] is not None or value['bank'] is not None or value['candidate_order']!=candidates):
        raise ValueError('Unavailable structural ranking must preserve the complete declared order')
    elif adaptive:raise ValueError('Unavailable structural ranking cannot acquire an adaptive work set')
    return value


def prioritize(rows,value,protocol,*,observation=None):
    validate(value,protocol,observation=observation)
    if not value['ranking_available']:return list(rows)
    positions={key:i for i,key in enumerate(value['candidate_order'])}
    if any(row['candidate'] not in positions for row in rows):raise ValueError('Ranking cannot introduce a new native candidate')
    return sorted(rows,key=lambda row:positions[row['candidate']])
