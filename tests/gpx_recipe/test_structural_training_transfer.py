"""Closed numeric transport; these controls do not validate a trained model."""
import io
import json

import numpy as np
import pytest

from gpx_recipe.execution import write_json
from gpx_recipe.preparation import structural_training_checkpoint as checkpoint
from gpx_recipe.preparation import structural_training_transfer as transfer
from gpx_recipe.preparation.tof_kernel_arrays import array_identity


def owned(root,*,identity='a'*64,warmup=20,value=1.):
    root.mkdir()
    arrays=dict(parameters=np.full(4,value,np.float32),best_parameters=np.empty(0,np.float32),
        first_moment=np.zeros(4,np.float32),second_moment=np.zeros(4,np.float32),
        steps=np.ones(2,np.float32),initialized=np.ones(2,bool),visited=np.zeros(12,bool),
        torch_rng=np.arange(16,dtype=np.uint8))
    metadata=dict(schema=checkpoint.SCHEMA,identity=identity,layout='b'*64,arrays_sha256=array_identity(arrays),
        epochs_completed=0,warmup_completed=warmup,elapsed_seconds=10.,history=[])
    arrays['metadata']=np.frombuffer(json.dumps(metadata).encode(),np.uint8)
    np.savez(root/'training-checkpoint.npz',**arrays)
    write_json(root/'training-identity.json',dict(schema='radar-owned-attention-training-run/v1',identity=identity))
    return root


def test_data_only_transfer_to_a_fresh_owner_is_lossless_and_idempotent(tmp_path):
    source=owned(tmp_path/'source');snapshot=tmp_path/'snapshot'
    receipt=transfer.snapshot(source,snapshot,identity='a'*64,epochs=60)
    assert receipt['trained_model'] is False and receipt['candidate_pruning_authorized'] is False
    assert {p.name for p in snapshot.iterdir()}=={transfer.STATE,transfer.RECEIPT}
    target=tmp_path/'fresh'
    result=transfer.restore(snapshot,target,identity='a'*64,epochs=60)
    assert result['disposition']=='copied_for_complete_native_checkpoint_audit'
    assert (target/'training-checkpoint.npz').read_bytes()==(source/'training-checkpoint.npz').read_bytes()
    assert transfer.restore(snapshot,target,identity='a'*64,epochs=60)['disposition']=='identical_owned_checkpoint_retained'


def test_archival_read_checks_original_implementation_but_does_not_authorize_resume(tmp_path,monkeypatch):
    source=owned(tmp_path/'source');snapshot=tmp_path/'snapshot'
    saved=transfer.snapshot(source,snapshot,identity='a'*64,epochs=60)
    monkeypatch.setattr(transfer,'package_signature',lambda:'b'*64)
    with pytest.raises(ValueError,match='continuation receipt'):transfer.read(snapshot)
    receipt,raw=transfer.read(snapshot,expected_implementation=saved['implementation_sha256'])
    assert receipt==saved and raw==(source/'training-checkpoint.npz').read_bytes()
    with pytest.raises(ValueError,match='continuation receipt'):
        transfer.read(snapshot,expected_implementation='c'*64)
    with pytest.raises(ValueError,match='implementation digest'):
        transfer.read(snapshot,expected_implementation='not-a-digest')
    with pytest.raises(ValueError,match='continuation receipt'):
        transfer.restore(snapshot,tmp_path/'must-not-resume',identity='a'*64,epochs=60)
    assert not (tmp_path/'must-not-resume').exists()


@pytest.mark.parametrize('mutation',['identity','target_epochs','code','hash','trained_flag','bytes','inventory'])
def test_changed_transfer_cannot_create_or_replace_an_optimizer(tmp_path,mutation):
    source=owned(tmp_path/'source');snapshot=tmp_path/'snapshot'
    transfer.snapshot(source,snapshot,identity='a'*64,epochs=60)
    receipt=json.loads((snapshot/transfer.RECEIPT).read_text())
    if mutation=='identity':receipt['identity']='c'*64
    elif mutation=='target_epochs':receipt['epochs']=30
    elif mutation=='code':receipt['implementation_sha256']='d'*64
    elif mutation=='hash':receipt['checkpoint_sha256']='e'*64
    elif mutation=='trained_flag':receipt['trained_model']=True
    elif mutation=='bytes':
        with (snapshot/transfer.STATE).open('ab') as stream:stream.write(b'changed')
    else:
        with np.load(snapshot/transfer.STATE,allow_pickle=False) as archive:arrays={key:archive[key] for key in archive.files}
        arrays['unexpected']=np.ones(1);np.savez(snapshot/transfer.STATE,**arrays)
    write_json(snapshot/transfer.RECEIPT,receipt)
    with pytest.raises(ValueError):transfer.restore(snapshot,tmp_path/'fresh',identity='a'*64,epochs=60)
    assert not (tmp_path/'fresh').exists()


def test_a_newer_owned_checkpoint_is_retained_and_an_alternate_branch_is_rejected(tmp_path):
    source=owned(tmp_path/'source');snapshot=tmp_path/'snapshot'
    transfer.snapshot(source,snapshot,identity='a'*64,epochs=60)
    target=owned(tmp_path/'newer',warmup=40,value=2.);before=(target/'training-checkpoint.npz').read_bytes()
    assert transfer.restore(snapshot,target,identity='a'*64,epochs=60)['disposition']=='newer_owned_checkpoint_retained'
    assert (target/'training-checkpoint.npz').read_bytes()==before
    alternate=owned(tmp_path/'alternate',warmup=20,value=2.);before=(alternate/'training-checkpoint.npz').read_bytes()
    with pytest.raises(ValueError,match='different owned checkpoint'):
        transfer.restore(snapshot,alternate,identity='a'*64,epochs=60)
    assert (alternate/'training-checkpoint.npz').read_bytes()==before


def test_transfer_respects_an_active_training_owner(tmp_path):
    source=owned(tmp_path/'source');snapshot=tmp_path/'snapshot'
    transfer.snapshot(source,snapshot,identity='a'*64,epochs=60)
    with checkpoint.owned_run(tmp_path/'target','a'*64):
        with pytest.raises(ValueError,match='active owner'):
            transfer.restore(snapshot,tmp_path/'target',identity='a'*64,epochs=60)
    assert not (tmp_path/'target/training-checkpoint.npz').exists()
