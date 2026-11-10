"""Worker-bound continuation receipts, distinct from completed ranking models."""
import json
from pathlib import Path

from ..execution import write_json
from ..symmetry.compiler import file_sha,package_signature
from . import structural_training_transfer as transfer
from .tof_kernel_arrays import read_bounded

SCHEMA='radar-structural-training-continuation-worker/v1'
REASON='training_checkpoint_available'
SEGMENT_SECONDS=5400
RESULT='training-continuation-result.json'
ARTIFACTS=['continuation/'+transfer.STATE,'continuation/'+transfer.RECEIPT]


def validate(result,request,plan_sha):
    expected={'schema','status','kind','output','artifacts','plan_sha256','input_sha256',
        'grid','settings','implementation_sha256','identity','checkpoint_sha256','epochs',
        'epochs_completed','warmup_completed','dataset','seconds','training_seconds_cumulative',
        'observations_included','scientific_result','trained_model','candidate_pruning_authorized'}
    if (not isinstance(result,dict) or set(result)!=expected or result['schema']!=SCHEMA
            or result['status']!='checkpointed' or result['kind']!='structural-training'
            or result['output']!=str(Path(request['destination']).resolve())
            or result['artifacts']!=ARTIFACTS or result['plan_sha256']!=plan_sha
            or result['input_sha256']!=[file_sha(request[k]) for k in ('source','instrument')]
            or result['grid']!=request['grid'] or result['settings']!=request['settings']
            or result['implementation_sha256']!=package_signature()
            or any(type(result[key])is not int for key in ('epochs','epochs_completed','warmup_completed'))
            or result['epochs']!=request['settings']['epochs']
            or any(result[key] is not False for key in ('observations_included','scientific_result',
                'trained_model','candidate_pruning_authorized'))):
        raise ValueError('Training continuation differs from its captured scientific request')
    receipt,_=transfer.read(Path(result['output'])/'continuation',identity=result['identity'],epochs=result['epochs'])
    if any(result[key]!=receipt[key] for key in ('checkpoint_sha256','epochs_completed','warmup_completed')):
        raise ValueError('Training continuation progress differs from its complete numerical state')
    from .structural_training_counts import validate as validate_counts
    validate_counts(result['dataset'])
    import math
    if any(type(result[key]) not in (int,float) or not math.isfinite(result[key]) or result[key]<0
           for key in ('seconds','training_seconds_cumulative')):
        raise ValueError('Training continuation timing is invalid')
    from .structural_jobs import read_plan,resume_directory
    plan=read_plan(request['plan']);previous=resume_directory(Path(request['plan']),plan)
    prior=(0,0)
    if previous is not None:
        before,_=transfer.read(previous,identity=result['identity'],epochs=result['epochs'])
        prior=(before['epochs_completed'],before['warmup_completed'])
    if (result['epochs_completed'],result['warmup_completed'])<=prior:
        raise ValueError('A training continuation must advance a complete checkpoint boundary')
    return receipt


def completed(request,out,plan_sha):
    """An acknowledged partial output is immutable on worker-local replay."""
    path=Path(out)/RESULT
    if not path.exists() and not path.is_symlink():return None
    result=json.loads(read_bounded(path,8*1024**2))
    validate(result,request,plan_sha)
    return result


def publish(request,out,plan_sha,segment,*,seconds):
    if segment.get('schema')!='radar-attention-training-segment/v1' or segment.get('status')!='checkpointed':
        raise ValueError('A complete internal optimizer segment is required')
    out=Path(out).resolve()
    receipt=transfer.snapshot(out/'model',out/'continuation',identity=segment['identity'],epochs=segment['epochs'])
    result=dict(schema=SCHEMA,status='checkpointed',kind='structural-training',output=str(out),
        artifacts=ARTIFACTS.copy(),plan_sha256=plan_sha,
        input_sha256=[file_sha(request[k]) for k in ('source','instrument')],
        grid=request['grid'],settings=request['settings'],implementation_sha256=package_signature(),
        identity=segment['identity'],checkpoint_sha256=receipt['checkpoint_sha256'],epochs=segment['epochs'],
        epochs_completed=segment['epochs_completed'],warmup_completed=segment['warmup_completed'],
        dataset=segment['dataset'],seconds=seconds,training_seconds_cumulative=segment['seconds'],
        observations_included=False,scientific_result=False,trained_model=False,candidate_pruning_authorized=False)
    validate(result,request,plan_sha)
    write_json(out/RESULT,result)
    return result
