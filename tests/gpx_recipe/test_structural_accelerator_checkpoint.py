"""Exercise the shared envelope independently of GPU availability."""
import copy
import io
import json
from contextlib import nullcontext
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from gpx_recipe.preparation.structural_accelerator_checkpoint import Checkpoint
from gpx_recipe.preparation import structural_paired_training as training
from gpx_recipe.preparation import structural_training_backend as backend
from test_structural_paired_training_native import data


class CpuDeviceControl:
    SCHEMA='test-owned-device-control/v1'
    STATE='training-control-checkpoint.npz'
    @staticmethod
    def runtime():return dict(device='CPU numerical control',torch=str(torch.__version__))
    @staticmethod
    def require_model(model):assert {p.device.type for p in model.parameters()}=={'cpu'}
    @staticmethod
    def synchronize():pass
    @staticmethod
    def get_rng():return torch.get_rng_state()
    @staticmethod
    def set_rng(state):torch.set_rng_state(state)


class Engine:
    kind='cpu'
    checkpoint=Checkpoint(CpuDeviceControl)
    def place(self,model):return model
    def random_state(self,seed):return nullcontext()
    def portable_model(self,model,*args,**kwargs):return model,None


@pytest.mark.parametrize('stage',['warmup','epoch','publication'])
def test_generic_device_envelope_recovers_the_same_complete_shared_training(tmp_path,monkeypatch,stage):
    monkeypatch.setattr(backend,'select',lambda *args,**kwargs:Engine())
    _,cohort=data(tmp_path/'pairs')
    reference=training.train(cohort,tmp_path/'reference',epochs=4,threads=1)
    write=training.write_json
    def interrupt(path,value,**kwargs):
        write(path,value,**kwargs)
        if (stage=='warmup' and value.get('state')=='clean_prototype_warmup' and value.get('step')==40
                or stage=='epoch' and value.get('state')=='training' and value.get('epoch')==2
                or stage=='publication' and value.get('state')=='calibrating_training_support'):
            raise InterruptedError('deliberate generic-device interruption')
    monkeypatch.setattr(training,'write_json',interrupt)
    with pytest.raises(InterruptedError):training.train(cohort,tmp_path/'recovered',epochs=4,threads=1)
    assert (tmp_path/'recovered'/CpuDeviceControl.STATE).exists()
    assert not (tmp_path/'recovered/model.json').exists()
    monkeypatch.setattr(training,'write_json',write)
    actual=training.train(cohort,tmp_path/'recovered',epochs=4,threads=1)
    assert actual['weights_sha256']==reference['weights_sha256']
    for name in ('history','test','prior_only_test','selected_epoch','dataset'):
        assert actual[name]==reference[name]


def test_damaged_device_envelope_does_not_change_random_or_optimization_state(tmp_path):
    from gpx_recipe.preparation.structural_attention import network
    model=network(4);optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=4,eta_min=.00001)
    rng=np.random.default_rng(15);visited=np.zeros(20,bool);identity='a'*64
    checkpoint=Checkpoint(CpuDeviceControl)
    with checkpoint.owned_run(tmp_path,identity):
        checkpoint.save(tmp_path,identity,model,optimizer,schedule,rng,visited,
            warmup=0,epochs=0,history=[],best=None,selected_epoch=None,best_key=None,elapsed_seconds=0.)
        path=tmp_path/CpuDeviceControl.STATE
        with np.load(io.BytesIO(path.read_bytes()),allow_pickle=False) as archive:
            arrays={key:archive[key] for key in archive.files}
        arrays['cpu_checkpoint'][0]^=1
        np.savez(path,**arrays)
        parameters={name:p.clone() for name,p in model.state_dict().items()}
        before=torch.get_rng_state().clone();numpy_before=copy.deepcopy(rng.bit_generator.state)
        with pytest.raises(ValueError):checkpoint.restore(tmp_path,identity,model,optimizer,schedule,rng,visited,epochs=4,warmup_steps=200)
        assert all(torch.equal(value,model.state_dict()[name]) for name,value in parameters.items())
        assert torch.equal(before,torch.get_rng_state()) and numpy_before==rng.bit_generator.state
        assert not optimizer.state and not visited.any()


def test_typed_cuda_envelope_transfers_through_the_shared_canonical_audit_on_a_fresh_worker(tmp_path,monkeypatch):
    from gpx_recipe.preparation import structural_training_transfer as transfer
    from gpx_recipe.preparation import structural_cuda_checkpoint
    class TypedControl(CpuDeviceControl):
        SCHEMA=structural_cuda_checkpoint.SCHEMA
        STATE=structural_cuda_checkpoint.STATE
        @staticmethod
        def runtime():
            return dict(device='cuda',torch=str(torch.__version__),cuda='12.6',cudnn=99999,
                machine='synthetic-control',capability=[8,9],name='CPU numerical envelope control',deterministic=True,tf32=False)
    class TypedEngine(Engine):checkpoint=Checkpoint(TypedControl)
    monkeypatch.setattr(backend,'select',lambda *args,**kwargs:TypedEngine())
    _,cohort=data(tmp_path/'pairs');write=training.write_json
    reference=training.train(cohort,tmp_path/'reference',epochs=4,threads=1)
    def interrupt(path,value,**kwargs):
        write(path,value,**kwargs)
        if value.get('state')=='clean_prototype_warmup' and value.get('step')==40:
            raise InterruptedError('transfer control')
    monkeypatch.setattr(training,'write_json',interrupt)
    with pytest.raises(InterruptedError):training.train(cohort,tmp_path/'first-worker',epochs=4,threads=1)
    identity=json.loads((tmp_path/'first-worker/training-identity.json').read_text())['identity']
    receipt=transfer.snapshot(tmp_path/'first-worker',tmp_path/'transfer',identity=identity,epochs=4)
    assert receipt['schema']==transfer.CUDA_SCHEMA and receipt['device']=='cuda'
    checked,_=transfer.read(tmp_path/'transfer',identity=identity,epochs=4)
    assert checked==receipt
    # Standalone inspection must bind the outer device owner to the audited
    # inner optimizer even when the caller has not yet supplied an identity.
    with np.load(tmp_path/'transfer'/transfer.STATE,allow_pickle=False) as archive:
        altered={key:archive[key] for key in archive.files}
    metadata=json.loads(altered['metadata'].tobytes());metadata['identity']='b'*64
    altered['metadata']=np.frombuffer(json.dumps(metadata).encode(),np.uint8)
    np.savez(tmp_path/'mismatched-device-owner.npz',**altered)
    with pytest.raises(ValueError,match='identity or numerical bytes'):
        transfer.inspect(tmp_path/'mismatched-device-owner.npz',epochs=4)
    monkeypatch.setattr(training,'write_json',write)
    actual=training.train(cohort,tmp_path/'second-worker',epochs=4,threads=1,_resume=tmp_path/'transfer')
    assert actual['weights_sha256']==reference['weights_sha256']
    for name in ('history','test','prior_only_test','selected_epoch','dataset'):assert actual[name]==reference[name]
