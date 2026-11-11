"""Multi-model orchestration; mocked arrays do not establish scattering physics."""
import copy

import pytest

from gpx_recipe.preparation.library import import_library
from radar_pd_nova.model_runtime import GalaxyModelRuntime
from radar_pd_nova.preparation_queue import PreparationQueue
from test_model_runtime import remote
from test_preparation_queue import setup, complete, entry
from gpx_recipe.preparation.batch import DEADLINE_INTERRUPTED


def restore(queue,base,runtime,service,source,tmp_path,monkeypatch):
    library=import_library(base,tmp_path/'fresh-library')
    runtime=GalaxyModelRuntime(tmp_path/'fresh-runtime',service,max_active_jobs=2)
    monkeypatch.setattr(runtime,'find_dispatch',lambda key:service.jobs.get(key))
    monkeypatch.setattr(runtime,'wait_capacity',lambda cancel=None:None)
    source=dict(source,source_path=str(library.root/'inputs/parent.cif'))
    restored=PreparationQueue(library.root/'queue',library,runtime,source=source,checkpoint=queue.checkpoint)
    restored.restore(queue.snapshot())
    return restored


def test_default_queue_batches_sixteen_models_and_respects_actual_job_slots(remote,tmp_path):
    queue,_,_=setup(remote,tmp_path,count=35,batch_size=None);runtime,service,_=remote
    runtime.max_active_jobs=2;queue.resume(maximum_jobs=4)
    result=queue.tick();jobs=queue.snapshot()['jobs']
    assert result['active']==2 and result['effective_job_limit']==2
    assert [len(j['members']) for j in jobs]==[16,16]
    assert result['coverage']['counts']['preparing']==32
    assert result['coverage']['counts']['pending']==3
    assert all(j['handle']['request']['operation']=='preparation-batch' for j in jobs)
    complete(queue,service,jobs[0],tmp_path);queue.tick()
    assert [len(j['members']) for j in queue.snapshot()['jobs']]==[16,16,3]
    for job in queue.snapshot()['jobs'][1:]:complete(queue,service,job,tmp_path)
    assert queue.tick()['coverage']['all_prepared']
    assert len(service.submissions)==3  # 35 independent models, three launches.


def test_one_collection_per_batch_not_per_member(remote,tmp_path,monkeypatch):
    queue,_,_=setup(remote,tmp_path,count=9,batch_size=None);runtime,service,_=remote
    queue.resume();queue.tick();job=queue.snapshot()['jobs'][0]
    complete(queue,service,job,tmp_path);calls=[];writes=[];original=runtime.collect_native
    def collect(*args,**kwargs):
        calls.append(args[0]['key']);return original(*args,**kwargs)
    monkeypatch.setattr(runtime,'collect_native',collect)
    save=queue._save
    def saved(*args,**kwargs):
        writes.append(kwargs.get('remote',False));return save(*args,**kwargs)
    monkeypatch.setattr(queue,'_save',saved)
    assert queue.tick()['coverage']['all_prepared']
    assert len(writes)<=3 and sum(writes)==1
    assert queue.tick()['coverage']['all_prepared']
    assert calls==[job['handle']['key']]


def test_partial_batch_defers_unstarted_members_but_failures_need_review(remote,tmp_path):
    queue,_,_=setup(remote,tmp_path,count=5,batch_size=None);_,service,_=remote
    queue.resume();queue.tick();first=queue.snapshot()['jobs'][0]
    states={entry(first,1):'failed',entry(first,2):'pending',entry(first,3):'pending'}
    complete(queue,service,first,tmp_path,states=states)
    result=queue.tick();first,second=queue.snapshot()['jobs']
    assert result['coverage']['counts']['prepared']==2
    assert result['coverage']['counts']['failed']==1 and result['active']==1
    assert {m['entry'] for m in second['members']}=={entry(first,2),entry(first,3)}
    assert all(m['attempt']==1 for m in second['members'])
    assert first['handle']['key']!=second['handle']['key']
    complete(queue,service,second,tmp_path);result=queue.tick()
    assert result['coverage']['complete'] and not result['coverage']['all_prepared']
    queue.library.retry(entry(first,1));queue.tick();third=queue.snapshot()['jobs'][2]
    assert third['members']==[dict(entry=entry(first,1),attempt=2)]
    complete(queue,service,third,tmp_path)
    assert queue.tick()['coverage']['all_prepared'] and len(service.submissions)==3


def test_late_interrupted_member_gets_one_isolated_full_budget_then_failure_needs_review(remote,tmp_path):
    queue,_,_=setup(remote,tmp_path,count=3,batch_size=None);_,service,_=remote
    queue.resume();queue.tick();first=queue.snapshot()['jobs'][0]
    interrupted=entry(first,1);unstarted=entry(first,2)
    def disposition(result):
        result['entries'][1]['reason']=DEADLINE_INTERRUPTED
        result['entries'][1]['seconds']=300.
    complete(queue,service,first,tmp_path,
        states={interrupted:'pending',unstarted:'pending'},mutate=disposition)
    queue.tick();second=queue.snapshot()['jobs'][1]
    assert second['mode']=='batch' and second['members']==[dict(entry=interrupted,attempt=1)]
    complete(queue,service,second,tmp_path,states={interrupted:'failed'})
    queue.tick();third=queue.snapshot()['jobs'][2]
    assert third['members']==[dict(entry=unstarted,attempt=1)]
    complete(queue,service,third,tmp_path)
    status=queue.tick();assert status['coverage']['counts']['failed']==1
    assert len(service.submissions)==3 and status['active']==0
    assert queue.library.row(interrupted)['state']=='failed'
    # Full-budget failure does not become another automatic attempt.
    queue.tick();assert len(service.submissions)==3


@pytest.mark.parametrize('recovery_stage',['deferred','isolated_job_submitted'])
def test_interrupted_unit_recovery_retains_attempt_and_existing_dispatch(remote,tmp_path,monkeypatch,recovery_stage):
    queue,_,base=setup(remote,tmp_path,count=2,batch_size=None);runtime,service,source=remote
    queue.resume();queue.tick();first=queue.snapshot()['jobs'][0];interrupted=entry(first,1)
    def disposition(result):result['entries'][1].update(reason=DEADLINE_INTERRUPTED,seconds=100.)
    complete(queue,service,first,tmp_path,states={interrupted:'pending'},mutate=disposition)
    queue.pause();queue.tick()
    if recovery_stage=='isolated_job_submitted':queue.resume();queue.tick()
    restored=restore(queue,base,runtime,service,source,tmp_path,monkeypatch)
    restored.tick()
    assert len(service.submissions)==(1 if recovery_stage=='deferred' else 2)
    restored.resume();restored.tick();isolated=restored.snapshot()['jobs'][1]
    assert isolated['members']==[dict(entry=interrupted,attempt=1)]
    assert isolated['handle']['key']!=first['handle']['key']
    assert len(service.submissions)==2
    # A job already submitted before viewer loss finishes from its original
    # captured inputs. The restored viewer downloads that packet on collection.
    owner=queue if recovery_stage=='isolated_job_submitted' else restored
    complete(owner,service,isolated,tmp_path)
    assert restored.tick()['coverage']['all_prepared']
    assert restored.tick()['active']==0 and len(service.submissions)==2


def test_partial_batch_restore_checks_original_packet_and_keeps_deferral_attempt(remote,tmp_path,monkeypatch):
    queue,_,base=setup(remote,tmp_path,count=4,batch_size=None);runtime,service,source=remote
    queue.resume();queue.tick();job=queue.snapshot()['jobs'][0]
    complete(queue,service,job,tmp_path,states={entry(job,2):'pending',entry(job,3):'pending'})
    queue.pause();queue.tick()
    restored=restore(queue,base,runtime,service,source,tmp_path,monkeypatch)
    result=restored.tick()
    assert result['coverage']['counts']['prepared']==2 and result['deferred']==2
    assert result['active']==0 and len(service.submissions)==1
    errors=[];submit=restored.runtime.submit_native
    def checked_submit(*args,**kwargs):
        try:return submit(*args,**kwargs)
        except Exception as exc:
            errors.append(str(exc));raise
    monkeypatch.setattr(restored.runtime,'submit_native',checked_submit)
    restored.resume();restored.tick();continued=restored.snapshot()['jobs'][1]
    assert not errors,errors
    assert all(m['attempt']==1 for m in continued['members'])
    complete(restored,service,continued,tmp_path)
    assert restored.tick()['coverage']['all_prepared'] and len(service.submissions)==2


def test_bad_member_does_not_discard_valid_siblings_and_rejection_survives_restore(remote,tmp_path,monkeypatch):
    queue,_,base=setup(remote,tmp_path,count=4,batch_size=None);runtime,service,source=remote
    queue.resume();queue.tick();job=queue.snapshot()['jobs'][0]
    def corrupt(result):result['entries'][1]['result']['entry_sha256']='f'*64
    complete(queue,service,job,tmp_path,mutate=corrupt)
    result=queue.tick()
    assert result['coverage']['counts']['prepared']==3
    assert result['coverage']['counts']['failed']==1
    rejected=queue.snapshot()['jobs'][0]['outcomes'][entry(job,1)]
    assert rejected['source']=='verification'
    restored=restore(queue,base,runtime,service,source,tmp_path,monkeypatch)
    assert restored.tick()['coverage']['counts']==result['coverage']['counts']
    assert len(service.submissions)==1


def test_recovery_rechecks_a_prior_local_rejection_without_rerunning_the_worker(remote,tmp_path,monkeypatch):
    import radar_pd_nova.preparation_queue as module
    queue,_,base=setup(remote,tmp_path,count=4,batch_size=None);runtime,service,source=remote
    queue.resume();queue.tick();job=queue.snapshot()['jobs'][0]
    complete(queue,service,job,tmp_path)
    original=module.publish_entry
    def unavailable(library,identifier,*args,**kwargs):
        if identifier==entry(job,1):raise ValueError('Local publication temporarily unavailable')
        return original(library,identifier,*args,**kwargs)
    monkeypatch.setattr(module,'publish_entry',unavailable)
    assert queue.tick()['coverage']['counts']['failed']==1
    monkeypatch.setattr(module,'publish_entry',original)
    restored=restore(queue,base,runtime,service,source,tmp_path,monkeypatch)
    assert restored.tick()['coverage']['all_prepared']
    assert len(service.submissions)==1


@pytest.mark.parametrize('mutation',['missing','duplicate','wrong_attempt','wrong_plan','wrong_counts','wrong_output'])
def test_invalid_global_batch_receipt_never_becomes_partial_success(remote,tmp_path,mutation):
    queue,_,_=setup(remote,tmp_path,count=3,batch_size=None);_,service,_=remote
    queue.resume();queue.tick();job=queue.snapshot()['jobs'][0]
    def corrupt(result):
        if mutation=='missing':result['entries'].pop()
        if mutation=='duplicate':result['entries'][1]=copy.deepcopy(result['entries'][0])
        if mutation=='wrong_attempt':result['entries'][0]['attempt']+=1
        if mutation=='wrong_plan':result['plan_sha256']='0'*64
        if mutation=='wrong_counts':result['counts']['prepared']=0
        if mutation=='wrong_output':result['output']='/worker/native/unrelated'
    complete(queue,service,job,tmp_path,mutate=corrupt)
    result=queue.tick()
    assert result['coverage']['counts']['prepared']==0
    assert result['coverage']['counts']['failed']==3
    assert queue.tick()['active']==0 and len(service.submissions)==1


def test_duplicate_active_members_and_unproved_deferral_are_rejected(remote,tmp_path):
    queue,_,_=setup(remote,tmp_path,count=2,batch_size=None)
    queue.resume();queue.tick();state=queue.snapshot()
    duplicate=copy.deepcopy(state['jobs'][0]);duplicate['id']='f'*32
    state['jobs'].append(duplicate)
    with pytest.raises(ValueError,match='lineage'):queue._validate(state)
    state=queue.snapshot();job=state['jobs'][0];job['phase']='collected'
    job['outcomes']={m['entry']:dict(state='pending',reason='not_started',source='worker') for m in job['members']}
    with pytest.raises(ValueError,match='verified batch'):queue._validate(state)


def test_transient_result_download_failure_does_not_restart_or_fail_members(remote,tmp_path,monkeypatch):
    queue,_,_=setup(remote,tmp_path,count=4,batch_size=None);runtime,service,_=remote
    queue.resume();queue.tick();job=queue.snapshot()['jobs'][0];complete(queue,service,job,tmp_path)
    original=runtime.collect_native
    def unavailable(*args,**kwargs):raise TimeoutError('temporary network failure')
    monkeypatch.setattr(runtime,'collect_native',unavailable)
    with pytest.raises(TimeoutError):queue.tick()
    assert queue.library.coverage()['counts']['preparing']==4
    assert queue.snapshot()['jobs'][0]['phase']=='submitted'
    monkeypatch.setattr(runtime,'collect_native',original)
    assert queue.tick()['coverage']['all_prepared'] and len(service.submissions)==1


@pytest.mark.parametrize('finished_members',[0,1,3])
def test_partial_library_publication_ack_recovers_whole_batch(remote,tmp_path,monkeypatch,finished_members):
    queue,_,_=setup(remote,tmp_path,count=4,batch_size=None);_,service,_=remote
    queue.resume();queue.tick();job=queue.snapshot()['jobs'][0];complete(queue,service,job,tmp_path)
    original=queue._publish;published=[]
    def interrupted(*args,**kwargs):
        original(*args,**kwargs);published.append(args[0]['entry'])
        if len(published)>finished_members:raise OSError('lost local publication acknowledgment')
    monkeypatch.setattr(queue,'_publish',interrupted)
    with pytest.raises(OSError):queue.tick()
    assert queue.library.coverage()['counts']['prepared']==finished_members+1
    monkeypatch.setattr(queue,'_publish',original)
    assert queue.tick()['coverage']['all_prepared']
    assert len(service.submissions)==1


def test_reviewed_retry_revalidates_old_job_after_restart_before_any_new_post(remote,tmp_path):
    from gpx_recipe.execution import write_json
    queue,_,_=setup(remote,tmp_path,count=1,batch_size=None);runtime,service,_=remote
    queue.resume();queue.tick();state=queue.snapshot();job=state['jobs'][0];identifier=entry(job)
    queue.library.finish(identifier,1,state='failed',reason='saved failure')
    queue.library.retry(identifier)
    job.update(phase='collected',outcomes={identifier:dict(state='failed',reason='saved failure',source='job')})
    write_json(queue.path,state)
    # The real remote job is still queued. A claimed checkpoint completion
    # cannot justify a duplicate refinement after viewer restoration.
    with pytest.raises(ValueError,match='not terminal'):queue.tick()
    assert len(service.submissions)==1
    assert queue.library.row(identifier)['state']=='pending'


def test_legacy_queue_migrates_paused_without_losing_its_acknowledged_job(remote,tmp_path):
    from gpx_recipe.execution import write_json
    from gpx_recipe.preparation.operations import stage_entry
    queue,snapshots,base=setup(remote,tmp_path,count=1);runtime,service,source=remote
    identifier=queue.library.document['entries'][0]['id'];attempt=queue.library.start(identifier)
    payload=stage_entry(queue.library,identifier,attempt,
        queue.root/'attempts'/identifier/'attempt-1')
    legacy=dict(schema='radar-preparation-queue/v1',history_id=service.history_id,
        library_id=queue.library.document['id'],source=queue.identity,base_attempts={identifier:0},
        paused=False,maximum_jobs=1,tasks=[dict(entry=identifier,attempt=attempt,
            phase='awaiting_submission',handle=None,reason='')])
    def intent(handle):
        legacy['tasks'][0]['handle']=copy.deepcopy(handle)
        snapshots.append(dict(jobs=[dict(handle=handle)]))
    handle=runtime.submit_native('preparation-magnetic',payload,queue.root,
        source=queue.source,timeout=600,on_intent=intent)
    legacy['tasks'][0].update(handle=handle,phase='submitted');write_json(queue.path,legacy)
    recovered=PreparationQueue(queue.root,queue.library,runtime,source=source,checkpoint=queue.checkpoint)
    state=recovered.snapshot();assert state['schema']=='radar-preparation-queue/v2' and state['paused']
    assert state['jobs'][0]['mode']=='single'
    assert state['jobs'][0]['handle']==handle
    assert recovered.tick()['active']==1 and len(service.submissions)==1
    complete(recovered,service,state['jobs'][0],tmp_path)
    assert recovered.tick()['coverage']['all_prepared']
    legacy['tasks'][0].update(phase='prepared',handle=recovered.snapshot()['jobs'][0]['handle'])
    fresh_library=import_library(base,tmp_path/'legacy-restored-library')
    fresh=PreparationQueue(fresh_library.root/'queue',fresh_library,runtime,
        source=dict(source,source_path=str(fresh_library.root/'inputs/parent.cif')),checkpoint=queue.checkpoint)
    fresh.restore(legacy)
    assert fresh.tick()['coverage']['all_prepared'] and len(service.submissions)==1


def test_incorrect_claimed_deferral_is_replaced_by_verified_original_outcome(remote,tmp_path,monkeypatch):
    queue,_,base=setup(remote,tmp_path,count=2,batch_size=None);runtime,service,source=remote
    queue.resume();queue.tick();job=queue.snapshot()['jobs'][0]
    complete(queue,service,job,tmp_path);queue.tick()
    state=queue.snapshot();job=state['jobs'][0]
    job['outcomes']={m['entry']:dict(state='pending',reason='not_started',source='worker') for m in job['members']}
    original=queue.snapshot
    monkeypatch.setattr(queue,'snapshot',lambda:copy.deepcopy(state))
    recovered=restore(queue,base,runtime,service,source,tmp_path,monkeypatch)
    monkeypatch.setattr(queue,'snapshot',original)
    assert recovered.tick()['coverage']['all_prepared']
    recovered.resume();assert recovered.tick()['active']==0
    assert len(service.submissions)==1


def test_interrupted_batch_staging_resumes_the_exact_plan_before_post(remote,tmp_path,monkeypatch):
    from gpx_recipe.preparation import batch
    queue,_,_=setup(remote,tmp_path,count=4,batch_size=None);_,service,_=remote
    queue.resume();original=batch.stage
    def staged_then_interrupted(*args,**kwargs):
        original(*args,**kwargs);raise OSError('lost staging acknowledgment')
    monkeypatch.setattr(batch,'stage',staged_then_interrupted)
    with pytest.raises(OSError):queue.tick()
    assert not service.submissions and len(queue.snapshot()['jobs'])==1
    monkeypatch.setattr(batch,'stage',original)
    assert queue.tick()['active']==1 and len(service.submissions)==1


def test_worker_cancellation_does_not_create_an_automatic_resubmission_loop(remote,tmp_path):
    queue,_,_=setup(remote,tmp_path,count=3,batch_size=None);_,service,_=remote
    queue.resume();queue.tick();job=queue.snapshot()['jobs'][0]
    def cancelled(result):
        for row in result['entries']:row['reason']='cancelled_before_start'
    complete(queue,service,job,tmp_path,
        states={m['entry']:'pending' for m in job['members']},mutate=cancelled)
    result=queue.tick()
    assert result['queue']['paused'] and result['active']==0 and result['deferred']==3
    assert queue.tick()['active']==0 and len(service.submissions)==1
    queue.resume();queue.tick()
    assert len(service.submissions)==2
    assert all(m['attempt']==1 for m in queue.snapshot()['jobs'][1]['members'])
