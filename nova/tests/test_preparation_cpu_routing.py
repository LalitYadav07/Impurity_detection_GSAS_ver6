"""CPU is the new-run default; owned CPU and CUDA state retain their producer."""
import copy
import json
from types import SimpleNamespace

import pytest

from radar_pd_nova.native_operations import BOUND_SCHEMA,SCHEMA,capture,handle_path,validate_handle
from radar_pd_nova.preparation_limits import CPU_TRAINING_TOOL_ID,CUDA_TRAINING_TOOL_ID,training_device
from radar_pd_nova.preparation_training_runtime import PreparationRuntime,TRAINING_OPERATION,TrainingCapacityPending
from radar_pd_nova.refinement_runtime import RuntimeUnavailable
from test_preparation_gpu_routing import setup,patch_runtime,submit_training
from test_native_operations import reply


def test_cpu_default_is_deployment_policy_only(monkeypatch):
    monkeypatch.delenv('RADAR_PD_PREPARATION_TRAINING_DEVICE',raising=False)
    assert training_device()=='cpu'
    assert training_device({'RADAR_PD_PREPARATION_TRAINING_DEVICE':'cuda'})=='cuda'
    for invalid in ('auto','mps','CUDA','',None):
        with pytest.raises(ValueError):training_device({'RADAR_PD_PREPARATION_TRAINING_DEVICE':invalid})


def test_new_cpu_handle_names_the_existing_tool_and_has_its_own_identity(tmp_path,monkeypatch):
    router,service,request,source=setup(tmp_path,monkeypatch,device='cpu')
    old,_=capture(router.model,TRAINING_OPERATION,request,tmp_path/'capture',source,7200)
    handle=submit_training(router,request,source)
    assert handle['schema']==BOUND_SCHEMA and handle['tool_id']==CPU_TRAINING_TOOL_ID
    assert old['schema']==SCHEMA and old['key']!=handle['key']
    assert service.submissions[0]['tool_id']==CPU_TRAINING_TOOL_ID
    assert router.training is router.cpu_training and router.training.max_active_jobs==1
    validate_handle(router,handle)


@pytest.mark.parametrize('origin',['legacy','cpu','cuda'])
def test_original_producer_recovers_collects_and_cancels_after_policy_change(tmp_path,monkeypatch,origin):
    router,service,request,source=setup(tmp_path,monkeypatch,device='cuda' if origin=='cuda' else 'cpu')
    producer=router.model if origin=='legacy' else router.training
    handle=producer.submit_native(TRAINING_OPERATION,request,tmp_path/'capture',source=source,timeout=7200)
    monkeypatch.setenv('RADAR_PD_PREPARATION_TRAINING_DEVICE','cpu' if origin=='cuda' else 'cuda')
    fresh=PreparationRuntime(tmp_path/'fresh',service);patch_runtime(fresh,service,monkeypatch)
    selected=fresh.native_runtime_for_handle(handle)
    assert selected.tool_id==producer.tool_id
    assert (selected is fresh.model)==(origin=='legacy')
    assert fresh.poll_native(handle)['state']=='queued'
    deleted=[]
    monkeypatch.setattr(selected,'cancel_job',lambda job,key:deleted.append((job.galaxy_job_id,key)))
    fresh.cancel_native(handle)
    assert deleted==[(handle['job']['galaxy_job_id'],handle['key'])]
    reply(service,tmp_path,handle)
    result,_=fresh.collect_native(handle,tmp_path/'collected')
    assert result['path'].endswith('nested/value.json') and len(service.submissions)==1


@pytest.mark.parametrize('origin',['legacy','cpu','cuda'])
def test_uncertain_submission_never_reposts_after_policy_change(tmp_path,monkeypatch,origin):
    router,service,request,source=setup(tmp_path,monkeypatch,device='cuda' if origin=='cuda' else 'cpu')
    producer=router.model if origin=='legacy' else router.training
    service.lose_ack=True
    with pytest.raises(RuntimeUnavailable):
        producer.submit_native(TRAINING_OPERATION,request,tmp_path/'capture',source=source,timeout=7200)
    captured,_=capture(producer,TRAINING_OPERATION,request,tmp_path/'capture',source,7200)
    handle=json.loads(handle_path(tmp_path/'capture',captured['key']).read_text())
    monkeypatch.setenv('RADAR_PD_PREPARATION_TRAINING_DEVICE','cpu' if origin=='cuda' else 'cuda')
    fresh=PreparationRuntime(tmp_path/'fresh',service);patch_runtime(fresh,service,monkeypatch)
    assert fresh.poll_native(handle)['state']=='queued'
    assert len(service.submissions)==1 and handle['submission_started']


def test_training_admission_defers_any_owned_trainer_but_keeps_generation_ready(tmp_path,monkeypatch):
    router,_,request,source=setup(tmp_path,monkeypatch,device='cpu')
    library=SimpleNamespace(row=lambda _:dict(model=dict(kind='structural-training')))
    for producer in (router.model,router.cpu_training,router.cuda_training):
        handle,_=capture(producer,TRAINING_OPERATION,request,tmp_path/'capture',source,7200)
        jobs=[dict(phase='needs_recovery',handle=handle,members=[])]
        assert not router.preparation_operation_ready(TRAINING_OPERATION,jobs,library)
        assert router.preparation_operation_ready('preparation-structural-examples',jobs,library)
    monkeypatch.setattr(router.cpu_training,'available_capacity',lambda:0)
    with pytest.raises(TrainingCapacityPending):submit_training(router,request,source)
    captured,_=capture(router,TRAINING_OPERATION,request,tmp_path/'capture',source,7200)
    saved=json.loads(handle_path(tmp_path/'capture',captured['key']).read_text())
    assert not saved['submission_started']


def test_cpu_continuation_keeps_cpu_even_when_deployment_selects_cuda(tmp_path,monkeypatch):
    from test_structural_training_continuation import request,partial
    from gpx_recipe.preparation import structural_jobs
    payload,directory=request(tmp_path/'continuation');partial(payload)
    plan=tmp_path/'continuation'/'continued.json'
    structural_jobs.create_plan(payload['source'],[directory],plan,resume=tmp_path/'continuation'/'worker'/'continuation')
    payload=dict(payload,plan=str(plan),destination=str(tmp_path/'next'))
    router,service,_,source=setup(tmp_path/'router',monkeypatch,device='cuda')
    assert router.native_runtime_for_operation(TRAINING_OPERATION,payload=payload) is router.cpu_training
    broken=copy.deepcopy(payload);broken['plan']=str(tmp_path/'missing.json')
    with pytest.raises(ValueError):router.native_runtime_for_operation(TRAINING_OPERATION,payload=broken)


@pytest.mark.parametrize('origin',['legacy','cpu','cuda'])
def test_same_request_keeps_its_uncertain_producer_when_submitted_again(tmp_path,monkeypatch,origin):
    router,service,request,source=setup(tmp_path,monkeypatch,device='cuda' if origin=='cuda' else 'cpu')
    producer=router.model if origin=='legacy' else router.training
    service.lose_ack=True
    with pytest.raises(RuntimeUnavailable):
        producer.submit_native(TRAINING_OPERATION,request,tmp_path/'capture',source=source,timeout=7200)
    monkeypatch.setenv('RADAR_PD_PREPARATION_TRAINING_DEVICE','cpu' if origin=='cuda' else 'cuda')
    fresh=PreparationRuntime(tmp_path/'fresh',service);patch_runtime(fresh,service,monkeypatch)
    recovered=submit_training(fresh,request,source)
    assert recovered['job']['tool_id']==producer.tool_id and len(service.submissions)==1
    assert (recovered['schema']==SCHEMA)==(origin=='legacy')


def test_legacy_cpu_contract_recovers_under_future_cuda_policy_before_device_guard(tmp_path,monkeypatch):
    from gpx_recipe.symmetry.compiler import file_sha
    from test_structural_training_jobs import inputs
    router,service,_,_=setup(tmp_path,monkeypatch,device='cpu')
    payload,_=inputs(tmp_path/'legacy')
    source=dict(id='old-cpu-parent',dataset_id='old-source',source_path=payload['source'],
        source_sha256=file_sha(payload['source']))
    service.lose_ack=True
    with pytest.raises(RuntimeUnavailable):
        router.model.submit_native(TRAINING_OPERATION,payload,tmp_path/'legacy',source=source,timeout=7200)
    monkeypatch.setenv('RADAR_PD_PREPARATION_TRAINING_DEVICE','cuda')
    fresh=PreparationRuntime(tmp_path/'fresh',service);patch_runtime(fresh,service,monkeypatch)
    result=fresh.submit_native(TRAINING_OPERATION,payload,tmp_path/'legacy',source=source,timeout=7200)
    assert result['schema']==SCHEMA and result['job']['tool_id']==CPU_TRAINING_TOOL_ID
    assert len(service.submissions)==1
