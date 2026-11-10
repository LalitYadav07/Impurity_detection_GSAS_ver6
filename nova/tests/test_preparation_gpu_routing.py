"""GPU producer identity survives lost acknowledgments and new viewers.

These are application transport tests with opaque data, not GPU computation
or scientific qualification. Real-device acceptance is a separate NDIP job.
"""
import copy
from types import SimpleNamespace

import pytest

from gpx_recipe.symmetry.compiler import file_sha
from radar_pd_nova.models import RunStatus
from radar_pd_nova.native_operations import capture,handle_path,validate_handle,SCHEMA,BOUND_SCHEMA
from radar_pd_nova.preparation_training_runtime import (
    PreparationRuntime,TRAINING_OPERATION,TrainingCapacityPending,TrainingContractUnsupported,
)
from radar_pd_nova.preparation_limits import CUDA_TRAINING_TOOL_ID as TRAINING_TOOL_ID
from radar_pd_nova.refinement_runtime import RuntimeUnavailable
from test_model_runtime import Service
from test_native_operations import inputs as inspect_inputs,reply
from test_structural_training_jobs import inputs as legacy_training_inputs


def setup(tmp_path,monkeypatch,*,device='cuda'):
    monkeypatch.setenv('RADAR_PD_PREPARATION_TRAINING_DEVICE',device)
    service=Service(tmp_path)
    original=service.submit_utility
    def submit(**kwargs):
        try:return original(**kwargs)
        finally:
            job=service.jobs.get(kwargs['inputs']['dispatch_key'])
            if job is not None:job.tool_id=kwargs['tool_id'];job.status=RunStatus.QUEUED
    monkeypatch.setattr(service,'submit_utility',submit)
    router=PreparationRuntime(tmp_path/'runtime',service,poll_seconds=.001)
    patch_runtime(router,service,monkeypatch)
    from test_structural_paired_preparation import inputs as paired_inputs
    from gpx_recipe.preparation.structural_jobs import create_plan,validate_training
    example,directory=paired_inputs(tmp_path/'capture')
    plan=tmp_path/'capture/plan.json';create_plan(example['source'],[directory],plan)
    request=validate_training(dict(source=example['source'],instrument=example['instrument'],grid=example['grid'],
        plan=str(plan),destination=str(tmp_path/'capture/trained'),
        settings=dict(representation=example['settings']['representation'])))
    descriptor=dict(id='training-parent',dataset_id='original-parent',source_path=request['source'],
        source_sha256=file_sha(request['source']))
    return router,service,request,descriptor


def patch_runtime(router,service,monkeypatch):
    for target in (router.model,router.cpu_training,router.cuda_training):
        monkeypatch.setattr(target,'find_dispatch',lambda key:service.jobs.get(key))
        monkeypatch.setattr(target,'available_capacity',lambda:1)


def submit_training(router,request,source,*,root=None):
    from pathlib import Path
    return router.submit_native(TRAINING_OPERATION,request,root or Path(request['source']).parent,
        source=source,timeout=7200)


def test_new_training_has_a_fixed_gpu_producer_and_different_dispatch_from_the_cpu_contract(tmp_path,monkeypatch):
    router,service,request,source=setup(tmp_path,monkeypatch)
    cpu,_=capture(router.model,TRAINING_OPERATION,request,tmp_path/'capture',source,7200)
    gpu=submit_training(router,request,source)
    assert gpu['schema']==BOUND_SCHEMA and gpu['tool_id']==TRAINING_TOOL_ID
    assert cpu['schema']==SCHEMA and gpu['key']!=cpu['key']
    assert service.submissions[0]['tool_id']==TRAINING_TOOL_ID
    assert router.model.tool_id!=router.training.tool_id
    assert router.max_active_jobs==8 and router.training.max_active_jobs==1
    validate_handle(router,gpu)
    with pytest.raises(ValueError,match='original CPU producer'):validate_handle(router.training,cpu)


def test_inspection_keeps_the_existing_cpu_producer(tmp_path,monkeypatch):
    router,service,request,source=setup(tmp_path,monkeypatch)
    root,payload=inspect_inputs(tmp_path,source)
    result=router.submit_native('inspect',payload,root,source=source)
    assert result['schema']==SCHEMA and 'tool_id' not in result
    assert service.submissions[0]['tool_id']==router.model.tool_id
    assert not router.training.state['dispatches']


def test_an_old_eight_channel_plan_cannot_silently_run_on_cpu_under_a_gpu_label(tmp_path,monkeypatch):
    router,service,_,_=setup(tmp_path,monkeypatch)
    request,_=legacy_training_inputs(tmp_path/'legacy')
    source=dict(id='legacy-parent',dataset_id='legacy-source',source_path=request['source'],
        source_sha256=file_sha(request['source']))
    before=len(service.data)
    with pytest.raises(TrainingContractUnsupported,match='create a new preparation'):
        submit_training(router,request,source)
    assert len(service.data)==before and not service.submissions


def test_legacy_training_queue_pauses_with_a_durable_reason_before_any_gpu_upload(tmp_path,monkeypatch):
    from test_structural_preparation_dependencies import library_inputs,complete_examples
    from radar_pd_nova.preparation_queue import PreparationQueue
    router,service,_,_=setup(tmp_path,monkeypatch)
    library,originals=library_inputs(tmp_path/'old-library')
    complete_examples(library,originals,tmp_path/'old-examples')
    parent=library.root/library.document['specification']['source']
    source=dict(id='legacy-parent',source_sha256=file_sha(parent),source_path=str(parent),dataset_id='legacy-source')
    snapshots=[]
    queue=PreparationQueue(library.root/'queue',library,router,source=source,checkpoint=snapshots.append)
    queue.resume();before=len(service.data);status=queue.tick()
    assert status['queue']['paused'] and not service.submissions and len(service.data)==before
    assert status['queue']['jobs'][0]['phase']=='awaiting_submission'
    assert 'create a new preparation' in status['queue']['jobs'][0]['reason']
    assert snapshots[-1]['paused'] and snapshots[-1]['jobs'][0]['handle'] is None
    again=queue.tick()
    assert again['queue']==status['queue'] and not service.submissions


def test_unacknowledged_gpu_post_is_recovered_after_viewer_loss_without_another_submission(tmp_path,monkeypatch):
    import json
    router,service,request,source=setup(tmp_path,monkeypatch);service.lose_ack=True
    with pytest.raises(RuntimeUnavailable):submit_training(router,request,source)
    captured,_=capture(router,TRAINING_OPERATION,request,tmp_path/'capture',source,7200)
    saved=json.loads(handle_path(tmp_path/'capture',captured['key']).read_text())
    assert saved['submission_started'] and saved['job'] is None
    fresh=PreparationRuntime(tmp_path/'other-viewer',service);patch_runtime(fresh,service,monkeypatch)
    result=fresh.poll_native(saved)
    assert result['handle']['job']['tool_id']==TRAINING_TOOL_ID and result['state']=='queued'
    assert len(service.submissions)==1
    validate_handle(fresh,result['handle'])


def test_gpu_results_collect_in_a_new_viewer_and_legacy_cpu_handles_keep_their_owner(tmp_path,monkeypatch):
    router,service,request,source=setup(tmp_path,monkeypatch)
    handle=submit_training(router,request,source)
    reply(service,tmp_path,handle)
    fresh=PreparationRuntime(tmp_path/'fresh',service);patch_runtime(fresh,service,monkeypatch)
    result,_=fresh.collect_native(handle,tmp_path/'received')
    assert result['path'].endswith('nested/value.json') and len(service.submissions)==1
    cpu=router.model.submit_native(TRAINING_OPERATION,request,tmp_path/'capture',source=source,timeout=7200)
    assert fresh.native_runtime_for_handle(cpu) is fresh.model
    assert fresh.poll_native(cpu)['handle']['job']['tool_id']==router.model.tool_id
    assert len(service.submissions)==2


def test_gpu_origin_receipt_uses_the_captured_producer_after_viewer_loss(tmp_path,monkeypatch):
    from radar_pd_nova import preparation_evidence
    from test_preparation_evidence import setup as certificate_fixture
    router,service,request,source=setup(tmp_path,monkeypatch)
    handle=submit_training(router,request,source)
    certificate,_,_,_,_=certificate_fixture(tmp_path)
    identifier=service._put(certificate.read_bytes(),'origin receipt')
    service.jobs[handle['key']].outputs['preparation_certificate']=identifier
    fresh=PreparationRuntime(tmp_path/'fresh-evidence-viewer',service)
    patch_runtime(fresh,service,monkeypatch)
    references=[]
    library=SimpleNamespace(document=dict(entries=[dict(files=[
        dict(path='entries/opaque/entry.json',sha256='a'*64)])]),add_evidence=references.append)
    assert preparation_evidence.remember(library,fresh,handle,tmp_path)
    assert references[0]['dataset_id']==identifier and len(service.submissions)==1


@pytest.mark.parametrize('damage',['tool','operation','identity','request_type'])
def test_changed_or_foreign_training_routes_are_rejected_before_api_access(tmp_path,monkeypatch,damage):
    router,service,request,source=setup(tmp_path,monkeypatch);handle=submit_training(router,request,source)
    changed=copy.deepcopy(handle)
    if damage=='tool':changed['tool_id']=router.model.tool_id
    elif damage=='operation':changed['request']['operation']='inspect'
    elif damage=='identity':changed['identity']='f'*64
    else:changed['request']=[]
    with pytest.raises(ValueError):router.poll_native(changed)
    assert len(service.submissions)==1


def test_busy_gpu_capacity_returns_to_the_queue_without_a_post_or_false_ambiguous_intent(tmp_path,monkeypatch):
    import json
    router,service,request,source=setup(tmp_path,monkeypatch)
    monkeypatch.setattr(router.training,'available_capacity',lambda:0)
    with pytest.raises(TrainingCapacityPending):submit_training(router,request,source)
    captured,_=capture(router,TRAINING_OPERATION,request,tmp_path/'capture',source,7200)
    saved=json.loads(handle_path(tmp_path/'capture',captured['key']).read_text())
    assert not saved['submission_started'] and not service.submissions
    assert router.poll_native(saved)['state']=='not_submitted'


def test_an_active_gpu_bank_defers_other_trainers_but_leaves_cpu_generation_ready(tmp_path,monkeypatch):
    router,_,request,source=setup(tmp_path,monkeypatch);handle=submit_training(router,request,source)
    jobs=[dict(phase='submitted',handle=handle,members=[])]
    library=SimpleNamespace(row=lambda _:dict(model=dict(kind='structural-training')))
    assert not router.preparation_operation_ready(TRAINING_OPERATION,jobs,library)
    assert router.preparation_operation_ready('preparation-structural-examples',jobs,library)
    jobs[0]['phase']='collected'
    assert router.preparation_operation_ready(TRAINING_OPERATION,jobs,library)


@pytest.mark.parametrize('gpu_capable',[False,True])
def test_multipart_capability_is_checked_on_the_gpu_producer_and_inputs_recover_there(tmp_path,monkeypatch,gpu_capable):
    import shutil
    from gpx_recipe.preparation import structural_jobs
    from gpx_recipe.preparation.structural_input_parts import references
    from test_structural_input_parts import captured
    router,service,_,_=setup(tmp_path,monkeypatch)
    root=tmp_path/'large-training';request,names=captured(root,monkeypatch)
    monkeypatch.setattr(structural_jobs,'MAX_INPUT_BYTES',12000)
    calls=[]
    def capable(tool,key):
        calls.append((tool,key));return gpu_capable and tool==TRAINING_TOOL_ID and key=='model_asset_parts'
    monkeypatch.setattr(service,'tool_has_input',capable,raising=False)
    source=dict(id='multipart-parent',source_sha256=request['expected_source_sha256'],
        source_path=str(root/'parent.cif'),dataset_id='parent-data')
    payload=copy.deepcopy(request['payload'])
    for field in ('source','instrument','plan','destination'):payload[field]=str(root/payload[field])
    if not gpu_capable:
        before=len(service.data)
        with pytest.raises(RuntimeUnavailable,match='No examples were dropped'):
            submit_training(router,payload,source,root=root)
        assert not service.submissions and len(service.data)==before
    else:
        handle=submit_training(router,payload,source,root=root)
        assert len(references(handle['asset_id'])[1])==2 and len(service.submissions)==1
        fresh=PreparationRuntime(tmp_path/'fresh-multipart-viewer',service)
        patch_runtime(fresh,service,monkeypatch)
        shutil.rmtree(root);shutil.rmtree(router.training.root)
        restored=fresh.restore_native_inputs(handle,tmp_path/'restored-multipart')
        assert {name:file_sha(path) for name,path in restored.items()}==handle['request']['input_hashes']
        assert len(service.submissions)==1
    assert (TRAINING_TOOL_ID,'model_asset_parts') in calls
    assert all(tool==TRAINING_TOOL_ID for tool,_ in calls)


@pytest.mark.parametrize('tool,limit',[(TRAINING_TOOL_ID,1),('neutrons_radar_pd_model_campaign_prototype',8)])
def test_actual_utility_submission_uses_shared_admission_with_the_correct_family_limit(tmp_path,monkeypatch,tool,limit):
    from contextlib import contextmanager
    from radar_pd_nova.galaxy_service import GalaxyService
    from radar_pd_nova.native_job_admission import HistoryJobAdmission
    seen=[]
    @contextmanager
    def slot(self,tool_id,*,family_limit,cancel=None,operation=None):
        seen.append((tool_id,family_limit,self.total_limit));yield
    monkeypatch.setattr(HistoryJobAdmission,'slot',slot)
    monkeypatch.setattr('radar_pd_nova.galaxy_service.requests.post',lambda *a,**k:
        SimpleNamespace(raise_for_status=lambda:None,json=lambda:dict(jobs=[dict(id='accepted-job')])))
    monkeypatch.setenv('RADAR_PD_MAX_ACTIVE_JOBS','8')
    service=GalaxyService('https://galaxy.invalid','private-control','history',output_root=tmp_path)
    import base64,json
    request=dict(schema='radar-model-native/v1',operation=TRAINING_OPERATION,payload={})
    inputs=(dict(action='native',model_request_b64=base64.b64encode(json.dumps(request).encode()).decode())
            if tool==TRAINING_TOOL_ID else {})
    result=service.submit_utility(tool_id=tool,name='control',inputs=inputs)
    assert result.galaxy_job_id=='accepted-job' and seen==[(tool,limit,8)]


def test_two_bank_preparation_ticks_keep_one_gpu_job_and_continue_after_a_failed_bank(tmp_path,monkeypatch):
    import json,shutil
    from pathlib import Path
    from gpx_recipe.preparation.operations import create_library,stage_entry,publish_entry
    from gpx_recipe.preparation.structural_jobs import publish_result
    from gpx_recipe.preparation.structural_dependencies import EXAMPLES
    from radar_pd_nova.preparation_queue import PreparationQueue
    from test_structural_paired_preparation import inputs as paired_inputs
    router,service,_,_=setup(tmp_path,monkeypatch)
    request,directory=paired_inputs(tmp_path/'paired')
    banks={label:dict(instrument=request['instrument'],grid=dict(request['grid'],bank=bank))
        for bank,label in enumerate(('bank 1','bank 2'),1)}
    library=create_library(tmp_path/'library',name='Opaque two-bank transport control',
        source=request['source'],banks=banks,models=[dict(kind=EXAMPLES,bank=label,
            child=request['child'],settings=request['settings']) for label in banks])
    for row in list(library.document['entries']):
        if row['model']['kind']!=EXAMPLES:continue
        attempt=library.start(row['id']);staged=stage_entry(library,row['id'],attempt,tmp_path/('example-'+row['id']))
        output=Path(staged['destination']);shutil.copytree(directory,output)
        meta=json.loads((output/'examples.json').read_text());meta['grid']=staged['grid']
        (output/'examples.json').write_text(json.dumps(meta))
        publish_entry(library,row['id'],attempt,staged,
            publish_result(EXAMPLES,staged,dict(seconds=.1),['examples.json','examples.npz']))
    parent=Path(request['source']);source=dict(id='paired-source',source_path=str(parent),source_sha256=file_sha(parent),
        dataset_id=service._put(parent.read_bytes(),'paired parent'))
    queue=PreparationQueue(library.root/'queue',library,router,source=source,checkpoint=lambda _:None)
    queue.resume();first=queue.tick()
    assert first['active']==1 and len(service.submissions)==1
    assert service.submissions[0]['tool_id']==TRAINING_TOOL_ID
    queue.tick();assert len(service.submissions)==1
    service.jobs[first['queue']['jobs'][0]['handle']['key']].status=RunStatus.ERROR
    second=queue.tick()
    assert second['active']==1 and len(service.submissions)==2
    assert all(row['tool_id']==TRAINING_TOOL_ID for row in service.submissions)
    assert second['queue']['jobs'][0]['phase']=='collected'
