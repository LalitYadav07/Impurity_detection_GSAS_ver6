"""Closed queue records: one durable job handle for each finite batch."""
import copy
import json
import re

from gpx_recipe.symmetry.contracts import digest
from .native_operations import validate_handle

SCHEMA='radar-preparation-queue/v2'
MAX_METADATA=64*1024*1024
TERMINAL={'prepared','failed','cancelled','unsupported'}
JOB_PHASES={'awaiting_submission','submitted','needs_recovery','collected'}


def validate(value,library,runtime,identity):
    if (not isinstance(value,dict) or set(value)!={'schema','history_id','library_id','source',
            'base_attempts','paused','maximum_jobs','batch_size','jobs'} or value['schema']!=SCHEMA
            or value['history_id']!=runtime.service.history_id or value['library_id']!=library.document['id']
            or value['source']!=identity or type(value['paused']) is not bool
            or type(value['maximum_jobs']) is not int or value['maximum_jobs'] not in {1,2,4,8}
            or type(value['batch_size']) is not int or not 1<=value['batch_size']<=64
            or not isinstance(value['jobs'],list) or len(value['jobs'])>20000):
        raise ValueError('Preparation queue belongs to another library, History or schema')
    base=value['base_attempts']
    if (not isinstance(base,dict) or set(base)!={r['id'] for r in library.document['entries']}
            or any(type(v) is not int or not 0<=v<=1000000 for v in base.values())):
        raise ValueError('Preparation baseline attempt inventory changed')
    seen=set();latest={};members_total=0
    for job in value['jobs']:
        if (not isinstance(job,dict) or set(job)!={'id','mode','phase','handle','members','outcomes','reason','result_sha256'}
                or not isinstance(job['id'],str) or not re.fullmatch('[a-f0-9]{32}',job['id']) or job['id'] in seen
                or not isinstance(job['mode'],str) or job['mode'] not in {'batch','single'}
                or not isinstance(job['phase'],str) or job['phase'] not in JOB_PHASES
                or not isinstance(job['reason'],str) or len(job['reason'])>2000
                or not isinstance(job['members'],list) or not 1<=len(job['members'])<=64
                or not isinstance(job['outcomes'],dict)
                or (job['mode']=='single' and len(job['members'])!=1)
                or (job['result_sha256'] is not None and (not isinstance(job['result_sha256'],str)
                    or not re.fullmatch('[a-f0-9]{64}',job['result_sha256'])))):
            raise ValueError('Invalid preparation job record')
        seen.add(job['id']);members_total+=len(job['members']);owned=set()
        if job['handle'] is not None:
            validate_handle(runtime,job['handle'])
            operation=job['handle']['request']['operation']
            if (job['handle']['source']!=identity or (job['mode']=='batch' and operation!='preparation-batch')
                    or (job['mode']=='single' and operation not in {'preparation-magnetic','preparation-structural',
                        'preparation-structural-examples','preparation-structural-training'})):
                raise ValueError('Preparation handle names another source or operation')
        for member in job['members']:
            if (not isinstance(member,dict) or set(member)!={'entry','attempt'}
                    or not isinstance(member['entry'],str) or member['entry'] not in base
                    or member['entry'] in owned or type(member['attempt']) is not int
                    or not 1<=member['attempt']<=1000000):
                raise ValueError('Invalid or duplicate preparation member')
            entry=member['entry'];attempt=member['attempt'];owned.add(entry)
            previous=latest.get(entry)
            if previous is None:
                allowed=attempt==base[entry]+1
            else:
                prior,number=previous;outcome=prior['outcomes'].get(entry,{})
                allowed=prior['phase']=='collected' and (
                    (attempt==number and outcome.get('state')=='pending' and prior['result_sha256'] is not None)
                    or (attempt==number+1 and outcome.get('state') in TERMINAL-{'prepared'}))
            if not allowed:raise ValueError('Invalid or reordered preparation attempt lineage')
            latest[entry]=(job,attempt)
        if set(job['outcomes'])-owned or (job['phase']=='collected' and set(job['outcomes'])!=owned):
            raise ValueError('Preparation outcomes do not cover their exact members')
        for outcome in job['outcomes'].values():
            if (not isinstance(outcome,dict) or set(outcome)!={'state','reason','source'}
                    or not isinstance(outcome['state'],str) or outcome['state'] not in TERMINAL|{'pending'}
                    or not isinstance(outcome['reason'],str) or len(outcome['reason'])>2000
                    or not isinstance(outcome['source'],str) or outcome['source'] not in {'worker','verification','job','legacy'}):
                raise ValueError('Invalid preparation member outcome')
            if outcome['state']=='pending' and (job['mode']!='batch' or job['result_sha256'] is None
                    or outcome['source']!='worker'):
                raise ValueError('Only a verified batch may defer an unfinished member')
            if outcome['state']=='prepared' and (not job['handle'] or not job['handle']['job']):
                raise ValueError('Prepared work needs its acknowledged native job')
        if job['phase']=='collected' and (not job['handle'] or not job['handle']['job']):
            raise ValueError('Collected preparation needs its acknowledged job')
    if members_total>20000 or len(json.dumps(value,allow_nan=False).encode())>MAX_METADATA:
        raise ValueError('Preparation queue exceeds its bounded metadata scope')
    return value


def upgrade(value):
    """Preserve the earlier per-entry queue as single-member durable jobs."""
    if not isinstance(value,dict) or value.get('schema')!='radar-preparation-queue/v1':return value
    if set(value)!={'schema','history_id','library_id','source','base_attempts','paused','maximum_jobs','tasks'}:
        raise ValueError('Invalid legacy preparation queue')
    result={k:copy.deepcopy(v) for k,v in value.items() if k!='tasks'}
    result.update(schema=SCHEMA,batch_size=16,jobs=[],paused=True)
    if not isinstance(value['tasks'],list) or len(value['tasks'])>20000:
        raise ValueError('Legacy preparation task scope is oversized')
    for task in value['tasks']:
        if not isinstance(task,dict) or set(task)!={'entry','attempt','phase','handle','reason'}:
            raise ValueError('Invalid legacy preparation task')
        phase=task['phase'];terminal=isinstance(phase,str) and phase in TERMINAL
        if not isinstance(phase,str) or phase not in TERMINAL|{'awaiting_submission','submitted','needs_recovery'}:
            raise ValueError('Invalid legacy preparation phase')
        result['jobs'].append(dict(id=digest(['legacy',task['entry'],task['attempt']])[:32],mode='single',
            phase='collected' if terminal else phase,handle=copy.deepcopy(task['handle']),
            members=[dict(entry=task['entry'],attempt=task['attempt'])],
            outcomes={task['entry']:dict(state=phase,reason=task['reason'],source='legacy')} if terminal else {},
            reason=task['reason'],result_sha256=None))
    return result
