"""Bind collected preparation receipts to their exact immutable job members."""
import copy
import math
from pathlib import Path

from gpx_recipe.preparation import batch
from gpx_recipe.preparation.worker import validate_request,input_roles
from gpx_recipe.symmetry.compiler import confined, file_sha
from gpx_recipe.symmetry.contracts import digest


def requests_for(job, root):
    """Use captured inputs, never rebuild a recovered job with today's planner."""
    root=Path(root).resolve();handle=job['handle'];payload=handle['request']['payload']
    for name,checksum in handle['request']['input_hashes'].items():
        if file_sha(confined(root,name))!=checksum:
            raise ValueError('Captured preparation input changed')
    if job['mode']=='single':
        member=job['members'][0];request=copy.deepcopy(payload)
        for role in ('source','instrument','child','plan','destination'):
            if role in request:request[role]=str(confined(root,request[role],exists=role!='destination'))
        kind=handle['request']['operation'].removeprefix('preparation-')
        return {member['entry']:validate_request(kind,request)},None
    plan_path=confined(root,payload['plan']);plan=batch.read_plan(plan_path)
    members=[{k:t[k] for k in ('entry','attempt')} for t in plan['tasks']]
    if members!=job['members']:raise ValueError('Batch plan names another entry or attempt')
    output=confined(root,payload['destination'],exists=False);requests={}
    for task in plan['tasks']:
        request=copy.deepcopy(task['request'])
        for role in input_roles(task['kind']):
            request[role]=str(confined(plan_path.parent,request[role]))
        request['destination']=str(confined(output,request['destination'],exists=False))
        requests[task['entry']]=validate_request(task['kind'],request)
    return requests,plan


def stable_digest(result, root):
    """Exclude only viewer-local relocation from an otherwise exact receipt."""
    prefix=str(Path(root).resolve())
    def normalize(value):
        if isinstance(value,dict):return {k:normalize(v) for k,v in value.items()}
        if isinstance(value,list):return [normalize(v) for v in value]
        if isinstance(value,str) and (value==prefix or value.startswith(prefix+'/')):
            return 'workspace:'+value[len(prefix):]
        return value
    return digest(normalize(result))


def outcomes(job, result, root, plan, *, requests=None):
    """Check whole-batch completeness before publishing independent siblings."""
    if job['mode']=='single':
        from gpx_recipe.preparation import structural_training_segments as segments
        if isinstance(result,dict) and result.get('schema')==segments.SCHEMA:
            if job['handle']['request']['operation']!='preparation-structural-training':
                raise ValueError('An optimizer continuation belongs only to its training operation')
            if requests is None:requests,_=requests_for(job,root)
            member=job['members'][0];request=requests[member['entry']]
            segments.validate(result,request,file_sha(request['plan']),
                expected_implementation=job['handle']['request']['implementation_sha256'])
            return [dict(member,state='pending',reason=segments.REASON,result=result)]
        return [dict(job['members'][0],state='prepared',reason='',result=result)]
    expected={'schema','plan_sha256','status','counts','all_prepared','entries','allocation',
              'output','observations_included','scientific_result','artifacts','seconds'}
    output=confined(root,job['handle']['request']['payload']['destination'],exists=False)
    if (not isinstance(result,dict) or set(result)!=expected or result['schema']!=batch.RESULT_SCHEMA
            or result['plan_sha256']!=digest(plan) or result['output']!=str(output)
            or result['observations_included'] is not False or result['scientific_result'] is not False
            or not isinstance(result['entries'],list) or not isinstance(result['allocation'],dict)
            or not isinstance(result['artifacts'],list) or not all(isinstance(v,str) for v in result['artifacts'])
            or len(set(result['artifacts']))!=len(result['artifacts'])
            or type(result['seconds']) not in (int,float) or not math.isfinite(result['seconds']) or result['seconds']<0):
        raise ValueError('Preparation batch returned an unrelated or incomplete receipt')
    for name in result['artifacts']:
        batch.relative(name);confined(output,name)
    counts=dict.fromkeys(('prepared','failed','cancelled','pending'),0)
    actual=[]
    for row in result['entries']:
        if (not isinstance(row,dict) or set(row)-{'entry','attempt','state','reason','result','seconds'}
                or not {'entry','attempt','state','reason'}<=set(row)
                or not isinstance(row['state'],str) or row['state'] not in counts
                or type(row['attempt']) is not int or not isinstance(row['entry'],str)
                or not isinstance(row['reason'],str) or len(row['reason'])>2000
                or (row['state']=='prepared')!=('result' in row)
                or ('seconds' in row and (type(row['seconds']) not in (int,float)
                    or not math.isfinite(row['seconds']) or row['seconds']<0))):
            raise ValueError('Invalid preparation batch member receipt')
        if row['state']=='pending' and row['reason'] not in {
                'batch_time_budget','batch_artifact_budget','cancelled_before_start','not_started',
                batch.DEADLINE_INTERRUPTED}:
            raise ValueError('An unfinished preparation needs its explicit worker disposition')
        actual.append({k:row[k] for k in ('entry','attempt')});counts[row['state']]+=1
    if (actual!=job['members'] or not isinstance(result['counts'],dict) or result['counts']!=counts
            or any(type(n) is not int for n in result['counts'].values())
            or result['status']!=('partial' if counts['pending'] else 'complete')
            or type(result['all_prepared']) is not bool
            or result['all_prepared']!=(counts['prepared']==len(actual))):
        raise ValueError('Preparation batch receipt does not cover its complete planned scope')
    return result['entries']
