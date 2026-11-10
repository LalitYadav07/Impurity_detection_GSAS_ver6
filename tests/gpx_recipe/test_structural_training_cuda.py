"""CUDA placement is explicit; real-device recovery is a separate required gate."""
import copy
import json
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from gpx_recipe.preparation import structural_training_backend as backend


@pytest.mark.parametrize('kind,label',[('cpu','CPU'),('mps','MPS'),('cuda','CUDA')])
def test_recovery_description_names_its_actual_device(kind,label):
    text=backend.recovery_scope(kind)
    assert label in text and 'checkpoints are not model-library artifacts' in text
    if kind=='cuda':assert 'MPS' not in text


def test_managed_cuda_rejects_a_legacy_cpu_trainer_before_device_initialization(tmp_path,monkeypatch):
    from test_structural_training_jobs import inputs
    from gpx_recipe.preparation.structural_jobs import execute_training
    request,_=inputs(tmp_path/'legacy')
    monkeypatch.setenv('RADAR_PREPARATION_TRAINING_DEVICE','cuda')
    monkeypatch.setattr(backend,'cuda_backend',lambda:pytest.fail('Reject the old trainer before initializing CUDA'))
    with pytest.raises(ValueError,match='current three-curve preparation'):
        execute_training(request)
    from pathlib import Path
    assert not Path(request['destination']).exists()


@pytest.mark.parametrize('operation',['preparation-structural-training','preparation-reference'])
def test_galaxy_native_launch_keeps_gpu_binding_only_for_training_without_credentials(tmp_path,monkeypatch,operation):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    script=Path(__file__).resolve().parents[2]/'scripts/ndip_model_campaign.py'
    spec=importlib.util.spec_from_file_location('cuda_launch_control',script)
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    monkeypatch.setenv('RADAR_PREPARATION_TRAINING_DEVICE','cuda')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-reviewed-binding')
    monkeypatch.setenv('CUDA_DEVICE_ORDER','PCI_BUS_ID')
    monkeypatch.setenv('AZURE_ILLUMINE_API_KEY','must-not-transfer')
    monkeypatch.setattr(worker,'require_adapters',lambda:None)
    monkeypatch.setattr(worker,'validate_native_payload',lambda *args:None)
    monkeypatch.setattr(worker,'validate_native_timeout',lambda *args:None)
    inputs=tmp_path/'inputs';inputs.mkdir();root=tmp_path/'native';root.mkdir()
    captured={}
    def run(command,**kwargs):
        captured.update(kwargs['env']);(root/'native-response.json').write_text('{}')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(worker,'run_scientific',run)
    worker.native(dict(schema='radar-model-native/v1',operation=operation,payload={},
        implementation_sha256=worker.package_signature(),expected_source_sha256='a'*64,
        dispatch_key='b'*24,timeout=60,input_hashes={}),inputs,root,None)
    assert not any('KEY' in key or 'TOKEN' in key for key in captured)
    if operation=='preparation-structural-training':
        assert captured['RADAR_PREPARATION_TRAINING_DEVICE']=='cuda'
        assert captured['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
        assert captured['CUDA_VISIBLE_DEVICES']=='GPU-reviewed-binding'
        assert captured['CUDA_DEVICE_ORDER']=='PCI_BUS_ID'
    else:assert not any(key in captured for key in ('RADAR_PREPARATION_TRAINING_DEVICE','CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG'))


@pytest.mark.parametrize('device',['wrong','gpu',''])
def test_unknown_managed_device_is_not_reinterpreted_as_a_cpu_run(tmp_path,monkeypatch,device):
    monkeypatch.setenv('RADAR_PREPARATION_TRAINING_DEVICE',device)
    with pytest.raises(ValueError,match='Unknown managed'):backend.select(tmp_path)


@pytest.mark.parametrize('checkpoint',['training-checkpoint.npz','training-mps-checkpoint.npz'])
def test_cuda_request_cannot_replace_an_owned_optimizer(tmp_path,monkeypatch,checkpoint):
    (tmp_path/checkpoint).touch();monkeypatch.setenv('RADAR_PREPARATION_TRAINING_DEVICE','cuda')
    with pytest.raises(ValueError,match='cannot move silently'):backend.select(tmp_path)


@pytest.mark.parametrize('available,count',[(False,0),(True,0),(True,2),(True,4)])
def test_visible_host_devices_are_not_treated_as_an_allocation(tmp_path,monkeypatch,available,count):
    monkeypatch.setenv('RADAR_PREPARATION_TRAINING_DEVICE','cuda')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:available)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:count)
    with pytest.raises(ValueError,match='exactly one bound usable device'):backend.select(tmp_path)


def test_managed_cuda_never_falls_back_when_deterministic_launch_configuration_is_missing(tmp_path,monkeypatch):
    monkeypatch.setenv('RADAR_PREPARATION_TRAINING_DEVICE','cuda')
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    with pytest.raises(ValueError,match='cuBLAS workspace'):backend.select(tmp_path)


@pytest.mark.parametrize('other',['training-checkpoint.npz','training-mps-checkpoint.npz'])
def test_conflicting_cpu_gpu_owners_are_rejected(tmp_path,other):
    (tmp_path/'training-cuda-checkpoint.npz').touch();(tmp_path/other).touch()
    with pytest.raises(ValueError,match='Conflicting'):backend.select(tmp_path)


CUDA=pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count()!=1,
    reason='A real single bound CUDA device is required; skipped tests do not qualify GPU training')


@pytest.fixture(autouse=True)
def measured_real_cuda_case(request,record_property):
    measured=request.node.name.startswith('test_real_cuda') and torch.cuda.is_available() and torch.cuda.device_count()==1
    if measured:torch.cuda.reset_peak_memory_stats(0)
    yield
    if measured:
        torch.cuda.synchronize(0)
        record_property('cuda_peak_allocated_bytes',int(torch.cuda.max_memory_allocated(0)))
        record_property('cuda_peak_reserved_bytes',int(torch.cuda.max_memory_reserved(0)))


@pytest.mark.parametrize('length',[1,16,25,128,129,257,1024])
@pytest.mark.parametrize('maximum',[False,True])
def test_gather_only_pool_has_native_values_and_first_tie_derivatives(length,maximum):
    from gpx_recipe.preparation.structural_cuda_pool import adaptive
    torch.manual_seed(14)
    source=torch.round(torch.randn(2,3,length)*2)/2
    actual=source.clone().requires_grad_();reference=source.clone().requires_grad_()
    native=torch.nn.functional.adaptive_max_pool1d if maximum else torch.nn.functional.adaptive_avg_pool1d
    a=adaptive(actual,128,maximum=maximum);b=native(reference,128)
    assert torch.allclose(a,b,rtol=1e-6,atol=1e-6)
    weighted=torch.randn_like(a)
    ga=torch.autograd.grad((a*weighted).sum(),actual)[0]
    gb=torch.autograd.grad((b*weighted).sum(),reference)[0]
    assert torch.allclose(ga,gb,rtol=1e-5,atol=2e-6)


@CUDA
@pytest.mark.parametrize('maximum',[False,True])
def test_real_cuda_pool_runs_deterministically_with_native_cpu_values_and_gradients(maximum):
    from gpx_recipe.preparation.structural_attention_pool import adaptive
    torch.manual_seed(14);source=torch.round(torch.randn(2,3,129)*2)/2
    cpu=source.clone().requires_grad_();gpu=source.to('cuda').requires_grad_()
    with backend.Backend('cuda').random_state(14):
        actual=adaptive(gpu,128,maximum=maximum)
        native=torch.nn.functional.adaptive_max_pool1d if maximum else torch.nn.functional.adaptive_avg_pool1d
        expected=native(cpu,128);weight=torch.randn_like(expected)
        ga=torch.autograd.grad((actual*weight.to('cuda')).sum(),gpu)[0].cpu()
        gb=torch.autograd.grad((expected*weight).sum(),cpu)[0]
    assert torch.allclose(actual.cpu(),expected,rtol=1e-6,atol=1e-6)
    assert torch.allclose(ga,gb,rtol=1e-5,atol=2e-6)


@CUDA
@pytest.mark.parametrize('stage',['warmup','epoch','publication'])
def test_real_cuda_shared_training_recovers_exactly_and_exports_cpu_weights(tmp_path,monkeypatch,stage):
    from gpx_recipe.preparation import structural_paired_training as training
    from gpx_recipe.preparation.structural_ranker import Ranker
    from test_structural_paired_training_native import data
    monkeypatch.setenv('RADAR_PREPARATION_TRAINING_DEVICE','cuda')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    _,cohort=data(tmp_path/'pairs')
    before=torch.cuda.get_rng_state(0).clone()
    reference=training.train(cohort,tmp_path/'reference',epochs=4,threads=1)
    assert torch.equal(before,torch.cuda.get_rng_state(0))
    write=training.write_json
    def interrupt(path,value,**kwargs):
        write(path,value,**kwargs)
        if (stage=='warmup' and value.get('state')=='clean_prototype_warmup' and value.get('step')==40
                or stage=='epoch' and value.get('state')=='training' and value.get('epoch')==2
                or stage=='publication' and value.get('state')=='calibrating_training_support'):
            raise InterruptedError('deliberate CUDA interruption')
    monkeypatch.setattr(training,'write_json',interrupt)
    with pytest.raises(InterruptedError,match='deliberate CUDA'):
        training.train(cohort,tmp_path/'recovered',epochs=4,threads=1)
    assert torch.equal(before,torch.cuda.get_rng_state(0))
    assert (tmp_path/'recovered/training-cuda-checkpoint.npz').is_file()
    assert not (tmp_path/'recovered/model.json').exists()
    monkeypatch.setattr(training,'write_json',write)
    recovered=training.train(cohort,tmp_path/'recovered',epochs=4,threads=1)
    assert recovered['weights_sha256']==reference['weights_sha256']
    for name in ['history','test','prior_only_test','selected_epoch','dataset','training_device_transport']:
        assert recovered[name]==reference[name]
    assert torch.equal(before,torch.cuda.get_rng_state(0))
    assert recovered['training_device_transport']['device']=='cuda'
    assert 'CUDA' in recovered['training_recovery']['scope'] and 'MPS' not in recovered['training_recovery']['scope']
    ranker=Ranker(tmp_path/'recovered')
    assert next(ranker.model.parameters()).device.type=='cpu'
    assert json.loads((tmp_path/'recovered/model.json').read_text())['channels']==3


@CUDA
def test_real_cuda_checkpoint_can_continue_on_a_fresh_matching_worker_and_publish_cpu_weights(tmp_path,monkeypatch):
    from gpx_recipe.preparation import structural_paired_training as training
    from gpx_recipe.preparation import structural_training_transfer as transfer
    from test_structural_paired_training_native import data
    monkeypatch.setenv('RADAR_PREPARATION_TRAINING_DEVICE','cuda')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    _,cohort=data(tmp_path/'pairs');write=training.write_json
    reference=training.train(cohort,tmp_path/'reference',epochs=4,threads=1)
    def interrupt(path,value,**kwargs):
        write(path,value,**kwargs)
        if value.get('state')=='training' and value.get('epoch')==2:raise InterruptedError('CUDA transfer')
    monkeypatch.setattr(training,'write_json',interrupt)
    with pytest.raises(InterruptedError):training.train(cohort,tmp_path/'first-worker',epochs=4,threads=1)
    identity=json.loads((tmp_path/'first-worker/training-identity.json').read_text())['identity']
    transfer.snapshot(tmp_path/'first-worker',tmp_path/'transfer',identity=identity,epochs=4)
    monkeypatch.setattr(training,'write_json',write)
    actual=training.train(cohort,tmp_path/'second-worker',epochs=4,threads=1,_resume=tmp_path/'transfer')
    assert actual['weights_sha256']==reference['weights_sha256']
    for name in ('history','test','selected_epoch','dataset','training_device_transport'):
        assert actual[name]==reference[name]


@CUDA
def test_real_cuda_joint_class_and_mode_training_exports_all_cpu_readouts(tmp_path,monkeypatch):
    from test_structural_joint_modes_native import cohort
    from gpx_recipe.preparation.structural_paired_training import train
    from gpx_recipe.preparation.structural_ranker import Ranker
    monkeypatch.setenv('RADAR_PREPARATION_TRAINING_DEVICE','cuda')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    _,value=cohort(tmp_path/'source')
    result=train(value,tmp_path/'model',epochs=2,threads=1)
    transport=result['training_device_transport']
    assert transport['device']=='cuda' and transport['mode_queries']>0
    assert np.isfinite(transport['maximum_mode_absolute_error'])
    assert result['mode_head_trained'] and result['test']['mode_validation']['available']
    model=Ranker(tmp_path/'model')
    assert {p.device.type for p in model.model.parameters()}=={'cpu'}
    assert not model.metadata['mode_readout']['experimental_mode_absence_qualified']
