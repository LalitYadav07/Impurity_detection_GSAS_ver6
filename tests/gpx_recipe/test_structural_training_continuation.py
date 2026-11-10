"""Bounded continuation plans and receipts, without training or physics claims."""
import copy
import json
from pathlib import Path

import pytest

from gpx_recipe.preparation import structural_jobs as jobs
from gpx_recipe.preparation import structural_training_segments as segments
from gpx_recipe.preparation import structural_training_transfer as transfer
from gpx_recipe.preparation.structural_training_counts import describe
from gpx_recipe.preparation.worker import validate_request
from gpx_recipe.symmetry.compiler import file_sha
from test_structural_paired_preparation import inputs
from test_structural_training_transfer import owned


def request(root):
    examples,directory=inputs(root)
    plan=root/'training-plan.json'
    jobs.create_plan(examples['source'],[directory],plan)
    return validate_request('structural-training',dict(source=examples['source'],
        instrument=examples['instrument'],grid=examples['grid'],plan=str(plan),
        destination=str(root/'worker'),settings=dict(representation=jobs.REPRESENTATION))),directory


def partial(payload,*,warmup=20):
    output=Path(payload['destination']);output.mkdir(parents=True)
    owned(output/'model',warmup=warmup)
    counts=describe(['train','validation','test'],[0,1,2],[True]*3,
        generated_pairs=3,simulated_observations=0)
    segment=dict(schema='radar-attention-training-segment/v1',status='checkpointed',
        identity='a'*64,epochs=payload['settings']['epochs'],epochs_completed=0,
        warmup_completed=warmup,dataset=counts,seconds=10.)
    return segments.publish(payload,output,file_sha(payload['plan']),segment,seconds=1.)


def test_partial_output_is_not_a_model_and_acknowledged_replay_is_immutable(tmp_path):
    payload,_=request(tmp_path);result=partial(payload)
    before={p.name:p.read_bytes() for p in (tmp_path/'worker/continuation').iterdir()}
    assert result['artifacts']==segments.ARTIFACTS
    assert result['trained_model'] is False and result['candidate_pruning_authorized'] is False
    assert not (tmp_path/'worker/entry.json').exists()
    assert not (tmp_path/'worker/model/model.json').exists()
    assert segments.completed(payload,tmp_path/'worker',file_sha(payload['plan']))==result
    assert partial_snapshot(payload)==before


def partial_snapshot(payload):
    output=Path(payload['destination'])
    receipt=transfer.snapshot(output/'model',output/'continuation',identity='a'*64,
        epochs=payload['settings']['epochs'])
    assert receipt['trained_model'] is False
    return {p.name:p.read_bytes() for p in (output/'continuation').iterdir()}


def test_new_plan_captures_exactly_two_resume_files_and_preserves_every_block(tmp_path):
    payload,directory=request(tmp_path);result=partial(payload)
    original=jobs.read_plan(payload['plan']);path=tmp_path/'continued.json'
    plan=jobs.create_plan(payload['source'],[directory],path,resume=tmp_path/'worker/continuation')
    assert plan['schema']==jobs.CONTINUED_PLAN_SCHEMA and plan['blocks']==original['blocks']
    assert jobs.plan_inputs(path)==['examples/examples.json','examples/examples.npz',
        'resume/training-state.npz','resume/training-state.json']
    assert jobs.checked_blocks(path,plan)==[directory]
    assert jobs.resume_directory(path,plan)==tmp_path/'resume'
    assert jobs.create_plan(payload['source'],[directory],path,recover_identical=True,
        resume=tmp_path/'worker/continuation')==plan
    continued=dict(payload,plan=str(path),destination=str(tmp_path/'second'))
    with pytest.raises(ValueError,match='advance'):
        partial(continued,warmup=20)
    assert result['warmup_completed']==20


@pytest.mark.parametrize('change',['state','receipt','directory','extra'])
def test_changed_resume_packet_cannot_be_used_or_published(tmp_path,change):
    payload,directory=request(tmp_path);partial(payload);path=tmp_path/'continued.json'
    jobs.create_plan(payload['source'],[directory],path,resume=tmp_path/'worker/continuation')
    if change in {'state','receipt'}:
        name=transfer.STATE if change=='state' else transfer.RECEIPT
        with (tmp_path/'resume'/name).open('ab') as stream:stream.write(b'changed')
    else:
        plan=json.loads(path.read_text())
        if change=='directory':plan['resume_state']['directory']='../outside'
        else:plan['resume_state']['python']='not allowed'
        path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):jobs.checked_blocks(path,jobs.read_plan(path))


@pytest.mark.parametrize('change',['epochs','profile','plan','progress','qualified','dataset','code'])
def test_partial_receipt_cannot_change_request_progress_or_qualification(tmp_path,change):
    payload,_=request(tmp_path);result=partial(payload);changed=copy.deepcopy(result)
    if change=='epochs':changed['epochs']=30
    elif change=='profile':changed['input_sha256'][1]='b'*64
    elif change=='plan':changed['plan_sha256']='b'*64
    elif change=='progress':changed['warmup_completed']=40
    elif change=='qualified':changed['trained_model']=True
    elif change=='dataset':changed['dataset']['unique_clean_pairs']+=1
    else:changed['implementation_sha256']='b'*64
    with pytest.raises(ValueError):segments.validate(changed,payload,file_sha(payload['plan']))
