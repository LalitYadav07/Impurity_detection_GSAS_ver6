"""One trainer across devices; exact owned recovery and portable CPU weights."""
import copy
import io
import json

import numpy as np
import pytest

torch=pytest.importorskip('torch')
from gpx_recipe.preparation import structural_training_backend as backend
from gpx_recipe.preparation import structural_paired_training as training
from gpx_recipe.preparation.structural_attention import network
from gpx_recipe.preparation.structural_attention_pool import adaptive
from gpx_recipe.preparation import structural_attention_weights as weights
from test_structural_paired_training_native import data

MPS=pytest.mark.skipif(not torch.backends.mps.is_available(),reason='A real MPS device is required')


@pytest.mark.parametrize('kind',['cpu','mps'])
def test_worker_continuation_retains_cpu_and_device_checkpoint_never_silently_moves(tmp_path,monkeypatch,kind):
    monkeypatch.setattr(backend.sys,'platform','darwin')
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:True)
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK','0')
    assert backend.select(tmp_path).kind=='mps'
    assert backend.select(tmp_path,segmented=True).kind=='cpu'
    assert backend.select(tmp_path,transferred=True).kind=='cpu'
    (tmp_path/f'training{ "-mps" if kind=="mps" else ""}-checkpoint.npz').touch()
    assert backend.select(tmp_path).kind==kind
    if kind=='mps':
        with pytest.raises(ValueError,match='original local device'):
            backend.select(tmp_path,segmented=True)
        monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
        with pytest.raises(ValueError,match='original local device'):backend.select(tmp_path)
    else:
        monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
        assert backend.select(tmp_path).kind=='cpu'


def test_conflicting_checkpoint_devices_cannot_choose_an_alternate_optimizer(tmp_path):
    for name in ['training-checkpoint.npz','training-mps-checkpoint.npz']:(tmp_path/name).touch()
    with pytest.raises(ValueError,match='Conflicting'):backend.select(tmp_path)


@pytest.mark.parametrize('length',[16,25,32,75,128,129,257,1025])
@pytest.mark.parametrize('maximum',[False,True])
def test_cpu_pooling_retains_exact_native_values_and_gradients(length,maximum):
    torch.manual_seed(14)
    values=torch.randn(2,3,length,requires_grad=True)
    native=torch.nn.functional.adaptive_max_pool1d if maximum else torch.nn.functional.adaptive_avg_pool1d
    actual=adaptive(values,128,maximum=maximum);expected=native(values,128)
    assert torch.equal(actual,expected)
    assert torch.equal(torch.autograd.grad(actual.sum(),values)[0],torch.autograd.grad(expected.sum(),values)[0])


@MPS
@pytest.mark.parametrize('length',[16,25,32,75,128,129,257,1025])
@pytest.mark.parametrize('maximum',[False,True])
def test_real_mps_pooling_handles_nondivisible_lengths_and_tie_derivatives(length,maximum):
    torch.manual_seed(14)
    source=torch.round(torch.randn(2,3,length)*2)/2
    cpu=source.clone().requires_grad_();gpu=source.to('mps').requires_grad_()
    native=torch.nn.functional.adaptive_max_pool1d if maximum else torch.nn.functional.adaptive_avg_pool1d
    expected=native(cpu,128);actual=adaptive(gpu,128,maximum=maximum)
    assert torch.allclose(actual.cpu(),expected,rtol=1e-6,atol=1e-6)
    a=torch.autograd.grad(actual.sum(),gpu)[0].cpu();b=torch.autograd.grad(expected.sum(),cpu)[0]
    assert torch.allclose(a,b,rtol=1e-6,atol=1e-6)


def test_execution_placement_does_not_change_portable_parameter_layout():
    torch.manual_seed(8);model=network(4)
    original=copy.deepcopy(model)
    original.pool=torch.nn.AdaptiveAvgPool1d(128)
    plane,layout=weights.pack(model);before,previous=weights.pack(original)
    assert layout==previous and np.array_equal(plane,before)
    values=torch.randn(2,3,129);model.eval();original.eval()
    with torch.no_grad():assert torch.equal(backend.forward(model,values),original(values))


@MPS
@pytest.mark.parametrize('stage',['warmup','epoch','publication'])
def test_shared_mps_trainer_recovers_exactly_and_exports_a_cpu_readable_model(tmp_path,monkeypatch,stage):
    # Force the real local backend explicitly in this device test; ordinary
    # worker segmentation/transfer continues to exercise the CPU format.
    monkeypatch.setattr(backend,'select',lambda *args,**kwargs:backend.Backend('mps'))
    _,cohort=data(tmp_path/'pairs')
    before=torch.mps.get_rng_state().clone()
    reference=training.train(cohort,tmp_path/'reference',epochs=4,threads=1)
    assert torch.equal(before,torch.mps.get_rng_state())
    write=training.write_json
    def interrupt(path,value,**kwargs):
        write(path,value,**kwargs)
        if (stage=='warmup' and value.get('state')=='clean_prototype_warmup' and value.get('step')==40
                or stage=='epoch' and value.get('state')=='training' and value.get('epoch')==2
                or stage=='publication' and value.get('state')=='calibrating_training_support'):
            raise InterruptedError('deliberate device interruption')
    monkeypatch.setattr(training,'write_json',interrupt)
    with pytest.raises(InterruptedError,match='deliberate device'):
        training.train(cohort,tmp_path/'recovered',epochs=4,threads=1)
    assert torch.equal(before,torch.mps.get_rng_state())
    assert (tmp_path/'recovered/training-mps-checkpoint.npz').is_file()
    assert not (tmp_path/'recovered/model.json').exists()
    monkeypatch.setattr(training,'write_json',write)
    recovered=training.train(cohort,tmp_path/'recovered',epochs=4,threads=1)
    assert torch.equal(before,torch.mps.get_rng_state())
    assert recovered['weights_sha256']==reference['weights_sha256']
    for name in ['history','test','prior_only_test','selected_epoch','dataset','training_device_transport']:
        assert recovered[name]==reference[name]
    report=recovered['training_device_transport']
    assert report['device']=='mps' and report['portable_inference_device']=='cpu'
    from gpx_recipe.preparation.structural_ranker import Ranker
    ranker=Ranker(tmp_path/'recovered')
    assert next(ranker.model.parameters()).device.type=='cpu'
    assert json.loads((tmp_path/'recovered/model.json').read_text())['channels']==3


@MPS
@pytest.mark.parametrize('change',['inventory','checksum','runtime','optimizer'])
def test_damaged_mps_checkpoint_rejected_before_model_optimizer_or_rng_changes(tmp_path,change):
    from gpx_recipe.preparation import structural_mps_checkpoint as checkpoint
    from gpx_recipe.preparation.tof_kernel_arrays import array_identity
    model=network(4).to('mps')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=4,eta_min=.00001)
    rng=np.random.default_rng(15);visited=np.zeros(20,bool);identity='a'*64
    with checkpoint.owned_run(tmp_path,identity):
        checkpoint.save(tmp_path,identity,model,optimizer,schedule,rng,visited,
            warmup=0,epochs=0,history=[],best=None,selected_epoch=None,best_key=None,elapsed_seconds=0.)
        path=tmp_path/checkpoint.STATE
        with np.load(io.BytesIO(path.read_bytes()),allow_pickle=False) as archive:
            arrays={key:archive[key] for key in archive.files}
        metadata=json.loads(arrays.pop('metadata').tobytes())
        if change=='inventory':arrays['unexpected']=np.zeros(1,np.uint8)
        elif change=='checksum':arrays['cpu_checkpoint'][0]^=1
        elif change=='runtime':metadata['runtime']['torch']='other-runtime'
        else:
            with np.load(io.BytesIO(arrays['cpu_checkpoint'].tobytes()),allow_pickle=False) as archive:
                inner={key:archive[key] for key in archive.files}
            settings=json.loads(inner['metadata'].tobytes())
            settings['optimizer_groups'][0]['lr']=.003
            inner['metadata']=np.frombuffer(json.dumps(settings).encode(),np.uint8)
            stream=io.BytesIO();np.savez(stream,**inner)
            arrays['cpu_checkpoint']=np.frombuffer(stream.getvalue(),np.uint8)
            metadata['arrays_sha256']=array_identity(arrays)
        arrays['metadata']=np.frombuffer(json.dumps(metadata).encode(),np.uint8)
        np.savez(path,**arrays)
        parameters={key:value.clone() for key,value in model.state_dict().items()}
        before_cpu=torch.get_rng_state().clone();before_gpu=torch.mps.get_rng_state().clone()
        before_numpy=copy.deepcopy(rng.bit_generator.state)
        with pytest.raises(ValueError):
            checkpoint.restore(tmp_path,identity,model,optimizer,schedule,rng,visited,
                epochs=4,warmup_steps=200)
        assert all(torch.equal(value,model.state_dict()[key]) for key,value in parameters.items())
        assert torch.equal(before_cpu,torch.get_rng_state())
        assert torch.equal(before_gpu,torch.mps.get_rng_state())
        assert rng.bit_generator.state==before_numpy and not visited.any()
        assert not optimizer.state and schedule.last_epoch==0


@MPS
def test_failed_mps_publication_preserves_the_previous_atomic_checkpoint(tmp_path,monkeypatch):
    from pathlib import Path
    from gpx_recipe.preparation import structural_mps_checkpoint as checkpoint
    model=network(4).to('mps')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=4,eta_min=.00001)
    rng=np.random.default_rng(15);visited=np.zeros(20,bool);identity='a'*64
    settings=dict(warmup=0,epochs=0,history=[],best=None,selected_epoch=None,best_key=None,elapsed_seconds=0.)
    with checkpoint.owned_run(tmp_path,identity):
        checkpoint.save(tmp_path,identity,model,optimizer,schedule,rng,visited,**settings)
        path=tmp_path/checkpoint.STATE;before=path.read_bytes();replace=checkpoint.os.replace
        def fail(source,destination):
            if Path(destination).name==checkpoint.STATE:raise OSError('deliberate atomic device failure')
            return replace(source,destination)
        monkeypatch.setattr(checkpoint.os,'replace',fail)
        with pytest.raises(OSError,match='deliberate atomic device'):
            checkpoint.save(tmp_path,identity,model,optimizer,schedule,rng,visited,**settings)
        assert path.read_bytes()==before
        assert not list(tmp_path.glob('mps-state-*.partial'))
