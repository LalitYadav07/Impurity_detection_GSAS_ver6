"""Fresh workers must reproduce the complete uninterrupted optimizer run."""
import json
from pathlib import Path

import pytest

pytest.importorskip('torch')
from gpx_recipe.preparation import structural_paired_training as training
from gpx_recipe.preparation import structural_training_transfer as transfer
from test_structural_paired_training_native import data


def test_complete_training_across_fresh_directories_keeps_weights_history_and_holdouts(tmp_path):
    _,cohort=data(tmp_path/'inputs')
    reference=training.train(cohort,tmp_path/'reference',epochs=4,threads=1)
    previous=None;progress=[];result=None
    for index in range(20):
        directory=tmp_path/('worker-'+str(index))
        result=training.train(cohort,directory,epochs=4,threads=1,
            _segment_seconds=1e-9,_resume=previous)
        if result.get('status')!='checkpointed':break
        assert result['trained_model'] is False and result['candidate_pruning_authorized'] is False
        assert not (directory/'model.json').exists()
        point=(result['epochs_completed'],result['warmup_completed'])
        assert not progress or point>progress[-1]
        progress.append(point)
        previous=tmp_path/('transfer-'+str(index))
        transfer.snapshot(directory,previous,identity=result['identity'],epochs=4)
        assert {p.name for p in previous.iterdir()}=={transfer.STATE,transfer.RECEIPT}
    assert result is not None and result.get('status')!='checkpointed'
    assert len(progress)>=12 and progress[-1]==(4,200)
    assert result['weights_sha256']==reference['weights_sha256']
    for key in ('history','test','prior_only_test','selected_epoch','curriculum','dataset'):
        assert result[key]==reference[key]
    assert result['training_recovery']['epochs_completed_on_resume']==4
    assert result['candidate_pruning_authorized'] is False


def test_fresh_worker_rejects_a_changed_curriculum_without_replacing_the_transfer(tmp_path):
    _,cohort=data(tmp_path/'inputs')
    result=training.train(cohort,tmp_path/'first',epochs=4,threads=1,_segment_seconds=1e-9)
    snapshot=tmp_path/'transfer'
    transfer.snapshot(tmp_path/'first',snapshot,identity=result['identity'],epochs=4)
    before={p.name:p.read_bytes() for p in snapshot.iterdir()}
    with pytest.raises(ValueError,match='continuation receipt'):
        training.train(cohort,tmp_path/'changed',epochs=5,threads=1,_resume=snapshot)
    assert {p.name:p.read_bytes() for p in snapshot.iterdir()}==before
    assert not (tmp_path/'changed/model.json').exists()


@pytest.mark.parametrize('seconds',[0,-1,float('inf'),True])
def test_invalid_internal_budget_is_refused_before_work(tmp_path,seconds):
    _,cohort=data(tmp_path/'inputs')
    with pytest.raises(ValueError,match='work budget'):
        training.train(cohort,tmp_path/'output',epochs=4,threads=1,_segment_seconds=seconds)
    assert not (tmp_path/'output').exists()


def test_worker_plans_continue_on_fresh_workers_and_publish_only_the_complete_model(tmp_path,monkeypatch):
    from gpx_recipe.preparation import structural_jobs as jobs
    from gpx_recipe.preparation import structural_training_segments as segments
    from test_structural_training_continuation import request
    payload,directory=request(tmp_path/'inputs')
    payload['settings'].update(epochs=1,workers=1,bins=128,augmentation=None)
    monkeypatch.setattr(segments,'SEGMENT_SECONDS',None)
    reference=jobs.execute_training(dict(payload,destination=str(tmp_path/'reference')))
    reference_model=json.loads((tmp_path/'reference/model/model.json').read_text())
    assert reference_model['feature_storage']=='native_pair_replay'
    assert not list((tmp_path/'reference').rglob('features.npy'))
    monkeypatch.setattr(segments,'SEGMENT_SECONDS',1e-9)
    resume=None;previous=None;partial_results=[]
    for index in range(20):
        job=tmp_path/('job-'+str(index));job.mkdir()
        # Native input packets restore the original exact example arrays.
        import shutil
        shutil.copytree(directory,job/'examples')
        plan=job/'training-plan.json'
        jobs.create_plan(payload['source'],[job/'examples'],plan,resume=resume)
        current=dict(payload,plan=str(plan),destination=str(job/'compiled'))
        result=jobs.execute_training(current)
        assert not list((job/'compiled').rglob('features.npy'))
        assert not list((job/'compiled/working-features').iterdir())
        if result.get('schema')!=segments.SCHEMA:break
        assert result['status']=='checkpointed' and result['trained_model'] is False
        assert result['artifacts']==segments.ARTIFACTS
        assert not (job/'compiled/entry.json').exists()
        assert not (job/'compiled/model/model.json').exists()
        # Retrying a completed worker segment is an immutable replay, not
        # permission to advance its already acknowledged output in place.
        assert jobs.execute_training(current)==result
        point=(result['epochs_completed'],result['warmup_completed'])
        assert previous is None or point>previous
        previous=point;partial_results.append(result)
        resume=job/'compiled/continuation'
    assert len(partial_results)>=11 and previous==(1,200)
    assert result['status']=='prepared' and result['kind']=='structural-training'
    final=json.loads((Path(result['output'])/'model/model.json').read_text())
    for key in ('weights_sha256','history','test','prior_only_test','selected_epoch','curriculum','dataset'):
        assert final[key]==reference_model[key]
    assert result['candidate_pruning_authorized'] is False
    assert jobs.execute_training(current)==result


def test_worker_charges_both_assembly_and_normalization_before_checkpoint_admission(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from gpx_recipe.preparation import structural_jobs as jobs,structural_replay
    from gpx_recipe.preparation import structural_training_segments as segments
    from test_structural_training_continuation import request
    payload,_=request(tmp_path/'inputs')
    payload['settings'].update(epochs=1,workers=1,bins=128,augmentation=None)
    job_clock=[0.];training_clock=[0.]
    monkeypatch.setattr(jobs,'time',SimpleNamespace(monotonic=lambda:job_clock[0]))
    monkeypatch.setattr(training,'time',SimpleNamespace(monotonic=lambda:training_clock[0]))
    monkeypatch.setattr(segments,'SEGMENT_SECONDS',100.)
    original_build=structural_replay.build;statistics=training.channel_statistics
    def build(*args,**kwargs):
        result=original_build(*args,**kwargs);job_clock[0]+=60.;return result
    def setup(*args,**kwargs):
        result=statistics(*args,**kwargs);training_clock[0]+=45.;return result
    monkeypatch.setattr(structural_replay,'build',build)
    monkeypatch.setattr(training,'channel_statistics',setup)
    monkeypatch.setattr(training,'evaluate',lambda *a,**k:pytest.fail('The complete checkpoint boundary should precede validation'))
    result=jobs.execute_training(payload)
    assert result['schema']==segments.SCHEMA and result['warmup_completed']==20
    assert result['epochs_completed']==0 and result['trained_model'] is False
    progress=json.loads((Path(result['output'])/'model/progress.json').read_text())
    assert progress['data_preparation_seconds']==45.
    assert not list(Path(result['output']).rglob('features.npy'))
