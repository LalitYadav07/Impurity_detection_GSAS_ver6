"""Interrupt real attention optimization and compare complete resumed outputs."""
import io
import json

import numpy as np
import pytest

pytest.importorskip('torch')
from gpx_recipe.preparation import structural_paired_training as training
from test_structural_paired_training_native import data


def interrupt(monkeypatch,stage):
    original=training.write_json
    if stage=='initial':
        save=training.checkpoint.save
        def initial(*args,**kwargs):
            save(*args,**kwargs)
            if kwargs['warmup']==kwargs['epochs']==0:raise InterruptedError('deliberate training interruption')
        monkeypatch.setattr(training.checkpoint,'save',initial)
    def write(path,value,**kwargs):
        original(path,value,**kwargs)
        stopped=(stage=='warmup' and value.get('state')=='clean_prototype_warmup' and value.get('step')==40
            or stage=='epoch' and value.get('state')=='training' and value.get('epoch')==2
            or stage=='publication' and value.get('state')=='calibrating_training_support')
        if stopped:raise InterruptedError('deliberate training interruption')
    monkeypatch.setattr(training,'write_json',write)
    return original


@pytest.mark.parametrize('stage',['initial','warmup','epoch','publication'])
def test_resumed_training_reproduces_uninterrupted_weights_history_and_holdouts(tmp_path,monkeypatch,stage):
    _,cohort=data(tmp_path/'pairs')
    reference=training.train(cohort,tmp_path/'reference',epochs=4,threads=1)
    original=interrupt(monkeypatch,stage)
    with pytest.raises(InterruptedError,match='deliberate'):
        training.train(cohort,tmp_path/'recovered',epochs=4,threads=1)
    assert (tmp_path/'recovered/training-checkpoint.npz').is_file()
    assert not (tmp_path/'recovered/model.json').exists()
    monkeypatch.setattr(training,'write_json',original)
    recovered=training.train(cohort,tmp_path/'recovered',epochs=4,threads=1)
    assert recovered['weights_sha256']==reference['weights_sha256']
    for key in ('history','test','prior_only_test','selected_epoch','curriculum','dataset'):
        assert recovered[key]==reference[key]
    receipt=recovered['training_recovery']
    assert receipt['warmup_completed_on_resume']==dict(initial=0,warmup=40,epoch=200,publication=200)[stage]
    assert receipt['epochs_completed_on_resume']==dict(initial=0,warmup=0,epoch=2,publication=4)[stage]
    assert recovered['candidate_pruning_authorized'] is False


@pytest.mark.parametrize('change',['feature','label','group','seed','epochs'])
def test_changed_training_context_cannot_resume_a_checkpoint(tmp_path,monkeypatch,change):
    _,cohort=data(tmp_path/'pairs');original=interrupt(monkeypatch,'warmup')
    out=tmp_path/'interrupted'
    with pytest.raises(InterruptedError):training.train(cohort,out,epochs=4,threads=1)
    monkeypatch.setattr(training,'write_json',original)
    before=(out/'training-checkpoint.npz').read_bytes()
    settings=dict(epochs=4,threads=1)
    if change=='feature':cohort['features'][0,0,0]+=.01
    elif change=='label':cohort['compatible_labels'][0]=True
    elif change=='group':cohort['groups'][0]='a-new-whole-trajectory'
    elif change=='seed':settings['seed']=55
    else:settings['epochs']=5
    with pytest.raises(ValueError,match='changed'):training.train(cohort,out,**settings)
    assert (out/'training-checkpoint.npz').read_bytes()==before
    assert not (out/'model.json').exists()


def test_damaged_numeric_checkpoint_cannot_publish_a_model(tmp_path,monkeypatch):
    _,cohort=data(tmp_path/'pairs');original=interrupt(monkeypatch,'warmup');out=tmp_path/'interrupted'
    with pytest.raises(InterruptedError):training.train(cohort,out,epochs=4,threads=1)
    monkeypatch.setattr(training,'write_json',original)
    path=out/'training-checkpoint.npz'
    with np.load(io.BytesIO(path.read_bytes()),allow_pickle=False) as archive:
        arrays={key:archive[key] for key in archive.files}
    arrays['parameters'][0]+=.01
    np.savez(path,**arrays)
    with pytest.raises(ValueError,match='changed'):training.train(cohort,out,epochs=4,threads=1)
    assert not (out/'model.json').exists()


def test_interrupted_atomic_write_preserves_the_previous_recoverable_optimizer(tmp_path,monkeypatch):
    from pathlib import Path
    _,cohort=data(tmp_path/'pairs');out=tmp_path/'interrupted';original=interrupt(monkeypatch,'warmup')
    with pytest.raises(InterruptedError):training.train(cohort,out,epochs=4,threads=1)
    monkeypatch.setattr(training,'write_json',original)
    before=(out/'training-checkpoint.npz').read_bytes();replace=training.checkpoint.os.replace
    def unavailable(source,destination):
        if Path(destination).name=='training-checkpoint.npz':raise OSError('simulated atomic publication failure')
        return replace(source,destination)
    monkeypatch.setattr(training.checkpoint.os,'replace',unavailable)
    with pytest.raises(OSError,match='simulated atomic'):training.train(cohort,out,epochs=4,threads=1)
    assert (out/'training-checkpoint.npz').read_bytes()==before
    assert not list(out.glob('training-checkpoint-*.partial'))
    monkeypatch.setattr(training.checkpoint.os,'replace',replace)
    recovered=training.train(cohort,out,epochs=4,threads=1)
    assert recovered['training_recovery']['warmup_completed_on_resume']==40


def test_shared_worker_recovers_verified_features_and_keeps_checkpoints_out_of_library_artifacts(tmp_path,monkeypatch):
    from gpx_recipe.preparation.worker import execute
    from gpx_recipe.preparation.structural_jobs import create_plan
    from gpx_recipe.preparation import structural_paired_store
    from gpx_recipe.preparation.structural_training_blocks import REPRESENTATION
    from test_structural_paired_preparation import inputs
    example,directory=inputs(tmp_path/'inputs')
    plan=tmp_path/'inputs/plan.json';create_plan(example['source'],[directory],plan)
    request=dict(source=example['source'],instrument=example['instrument'],plan=str(plan),grid=example['grid'],
        destination=str(tmp_path/'reference'),settings=dict(representation=REPRESENTATION,bins=128,epochs=4,workers=1,augmentation=None))
    reference=execute('structural-training',request)
    request['destination']=str(tmp_path/'interrupted');original=interrupt(monkeypatch,'epoch')
    with pytest.raises(InterruptedError):execute('structural-training',request)
    monkeypatch.setattr(training,'write_json',original)
    def no_reassembly(*args,**kwargs):raise AssertionError('Completed features were rebuilt')
    monkeypatch.setattr(structural_paired_store,'build',no_reassembly)
    recovered=execute('structural-training',request)
    assert recovered['status']=='prepared'
    first=json.loads((tmp_path/'reference/model/model.json').read_text())
    second=json.loads((tmp_path/'interrupted/model/model.json').read_text())
    assert first['weights_sha256']==second['weights_sha256']
    assert second['training_recovery']['epochs_completed_on_resume']==2
    assert set(recovered['artifacts'])=={'entry.json','model/model.json','model/weights.npz','training.json'}
    assert not any('checkpoint' in name or 'working-features' in name for name in recovered['artifacts'])
    before={name:(tmp_path/'interrupted'/name).read_bytes() for name in recovered['artifacts']}
    monkeypatch.setattr(training,'train',no_reassembly)
    assert execute('structural-training',request)==recovered
    assert all((tmp_path/'interrupted'/name).read_bytes()==raw for name,raw in before.items())
    with (tmp_path/'interrupted/model/weights.npz').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):execute('structural-training',request)
