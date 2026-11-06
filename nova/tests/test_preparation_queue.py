"""Scheduling and recovery contracts; fake worker arrays are not physics tests."""
import copy
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from gpx_recipe.preparation.operations import create_library
from gpx_recipe.preparation.library import import_library
from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.symmetry.contracts import digest
from radar_pd_nova.model_runtime import GalaxyModelRuntime
from radar_pd_nova.models import RunStatus
from radar_pd_nova.preparation_queue import PreparationQueue
from test_model_runtime import remote


def setup(remote, tmp_path, count=3, *, batch_size=1):
    runtime, service, source = remote
    instrument=tmp_path/'bank.instprm';instrument.write_text('# queue contract placeholder\n')
    models=[]
    for i in range(count):
        settings=dict(candidate=dict(id='queue:'+str(i),magnetic_space_group='P1',bns_number='1.1',
            transformation_matrix=np.eye(3).tolist(),origin_shift=[0,0,0]),k=['0','0','0'],
            sites=[dict(label='Fe1',form_factor='Fe+3',lande_g=2.)],maximum_moment=6.,
            refine_histogram_scale=True,refine_background=True)
        models.append(dict(kind='magnetic',bank='bank 1',settings=settings))
    library=create_library(tmp_path/'library',name='Queue transport control',source=source['source_path'],
        banks={'bank 1':dict(instrument=instrument,
            grid=dict(kind='PNT',lower=3000.,upper=40000.,bins=701,bank=1))},models=models)
    snapshots=[]
    options={} if batch_size is None else {'_batch_size':batch_size}
    queue=PreparationQueue(library.root/'queue',library,runtime,source=source,
        checkpoint=lambda state:snapshots.append(copy.deepcopy(state)),**options)
    # An owned base archive is captured once, before jobs are admitted.
    base=tmp_path/'base.zip';library.export(base)
    original=service.submit_utility
    def queued(**kwargs):
        key=kwargs['inputs']['dispatch_key']
        assert any(t['handle'] and t['handle']['key']==key and t['handle']['submission_started']
                   for t in snapshots[-1]['jobs']), 'History checkpoint must precede POST'
        job=original(**kwargs);job.status=RunStatus.QUEUED;return job
    service.submit_utility=queued
    return queue, snapshots, base


def complete(queue, service, task, tmp_path, *, states=None, mutate=None):
    """Protocol-only fake worker; the native worker has separate integration tests."""
    from gpx_recipe.preparation.batch import read_plan, RESULT_SCHEMA
    handle=task['handle'];request=handle['request'];payload=request['payload']
    single=task['mode']=='single';states=states or {}
    plan=(dict(tasks=[dict(task['members'][0],kind='magnetic',request=payload)]) if single
          else read_plan(queue.root/payload['plan']))
    binary=io.BytesIO();np.savez_compressed(binary,x=np.arange(4.))
    numeric=binary.getvalue();files={};rows=[];artifacts=[]
    for member in plan['tasks']:
        status=states.get(member['entry'],'prepared')
        row=dict(entry=member['entry'],attempt=member['attempt'],state=status,
                 reason='batch_time_budget' if status=='pending' else '')
        if status=='prepared':
            child=member['request']
            entry=dict(schema='radar-experiment-preparation-entry/v1',kind=member['kind'],
                grid=child['grid'],settings=child['settings'],
                input_sha256=[file_sha(queue.root/(Path('.') if single else Path(payload['plan']).parent)/child[k])
                              for k in ('source','instrument')],
                observations_included=False,scientific_result=False,
                artifacts={'response.npz':hashlib.sha256(numeric).hexdigest()})
            manifest=json.dumps(entry).encode();prefix=(child['destination'] if single
                else payload['destination']+'/'+child['destination'])
            files['native/'+prefix+'/entry.json']=manifest
            files['native/'+prefix+'/response.npz']=numeric
            artifacts.extend(child['destination']+'/'+name for name in ('entry.json','response.npz'))
            row['result']=dict(schema='radar-experiment-preparation-worker/v1',status='prepared',kind=member['kind'],
                output='/worker/native/'+prefix,observations_included=False,scientific_result=False,
                artifacts=['entry.json','response.npz'],entry_sha256=hashlib.sha256(manifest).hexdigest())
        rows.append(row)
    counts={name:sum(r['state']==name for r in rows) for name in ('prepared','failed','cancelled','pending')}
    result=dict(schema=RESULT_SCHEMA,plan_sha256=digest(plan),status='partial' if counts['pending'] else 'complete',
        counts=counts,all_prepared=counts['prepared']==len(rows),entries=rows,allocation={'workers':1},
        output='/worker/native/'+payload['destination'],observations_included=False,scientific_result=False,
        artifacts=artifacts+['batch.json','progress.json'],seconds=.25)
    if single:result=rows[0]['result']
    if mutate:mutate(result)
    if not single:
        files['native/'+payload['destination']+'/batch.json']=json.dumps(result).encode()
        files['native/'+payload['destination']+'/progress.json']=b'{}'
    response=dict(schema='radar-model-native-result/v1',dispatch_key=handle['key'],
        request_sha256=digest(request),source_sha256=handle['source']['source_sha256'],
        worker_root='/worker/native',input_root='/worker/inputs-packet',result=result,
        files={name:hashlib.sha256(data).hexdigest() for name,data in files.items()},native_seconds=.25)
    archive=tmp_path/('reply-'+handle['key']+'.zip')
    with zipfile.ZipFile(archive,'w') as output:
        output.writestr('response.json',json.dumps(response))
        for name,data in files.items():output.writestr(name,data)
    job=service.jobs[handle['key']];job.status=RunStatus.OK
    job.outputs={'result_bundle':service._put(archive.read_bytes(),'preparation result')}


def entry(job, index=0):
    return job['members'][index]['entry']


def test_paused_queue_admits_nothing_then_uses_only_allowed_job_slots(remote,tmp_path):
    queue,snapshots,_=setup(remote,tmp_path);runtime,service,_=remote
    assert queue.tick()['active']==0 and not service.submissions
    runtime.max_active_jobs=2;queue.resume(maximum_jobs=4)
    result=queue.tick()
    assert result['active']==2 and result['effective_job_limit']==2
    assert result['coverage']['counts']['preparing']==2
    assert len(service.submissions)==2
    assert all('PRIVATE-' not in json.dumps(s) for s in snapshots)


def test_out_of_order_completions_publish_exact_entries_and_fill_freed_slots(remote,tmp_path):
    queue,_,_=setup(remote,tmp_path);runtime,service,_=remote
    runtime.max_active_jobs=2;queue.resume(maximum_jobs=2);queue.tick()
    first,second=queue.snapshot()['jobs']
    complete(queue,service,second,tmp_path)
    result=queue.tick()
    assert result['coverage']['counts']['prepared']==1 and result['active']==2
    assert queue.library.row(entry(second))['state']=='prepared'
    assert queue.library.row(entry(first))['state']=='preparing'
    assert len(service.submissions)==3
    for task in queue.snapshot()['jobs']:
        if task['phase']!='collected':complete(queue,service,task,tmp_path)
    assert queue.tick()['coverage']['all_prepared']


def test_pause_keeps_running_work_collectible_without_starting_more(remote,tmp_path):
    queue,_,_=setup(remote,tmp_path);_,service,_=remote
    queue.resume();queue.tick();task=queue.snapshot()['jobs'][0]
    queue.pause();complete(queue,service,task,tmp_path)
    result=queue.tick()
    assert result['coverage']['counts']['prepared']==1
    assert result['coverage']['counts']['pending']==2
    assert result['active']==0 and len(service.submissions)==1


def test_restore_reconstructs_published_arrays_from_jobs_not_status_flags(remote,tmp_path,monkeypatch):
    queue,_,base=setup(remote,tmp_path);runtime,service,source=remote
    runtime.max_active_jobs=2;queue.resume(maximum_jobs=2);queue.tick()
    task=queue.snapshot()['jobs'][0];complete(queue,service,task,tmp_path);queue.pause();queue.tick()
    snapshot=queue.snapshot();assert queue.library.coverage()['counts']['prepared']==1
    fresh_library=import_library(base,tmp_path/'restored-library')
    fresh_runtime=GalaxyModelRuntime(tmp_path/'new-runtime',service,max_active_jobs=2)
    monkeypatch.setattr(fresh_runtime,'find_dispatch',lambda key:service.jobs.get(key))
    source=dict(source,source_path=str(fresh_library.root/'inputs/parent.cif'))
    restored=PreparationQueue(fresh_library.root/'queue',fresh_library,fresh_runtime,source=source,checkpoint=lambda _:None)
    restored.restore(snapshot)
    assert fresh_library.coverage()['counts']['prepared']==0
    result=restored.tick()
    assert result['coverage']['counts']['prepared']==1
    assert result['coverage']['counts']['preparing']==1
    assert len(service.submissions)==2
    assert fresh_library.prepared_assets(entry(task))


def test_known_failed_job_needs_explicit_retry_and_gets_a_new_attempt_key(remote,tmp_path):
    queue,_,_=setup(remote,tmp_path,count=1);_,service,_=remote
    queue.resume();queue.tick();first=queue.snapshot()['jobs'][0]
    service.jobs[first['handle']['key']].status=RunStatus.ERROR
    assert queue.tick()['coverage']['counts']['failed']==1
    assert queue.tick()['active']==0 and len(service.submissions)==1
    queue.library.retry(entry(first));queue.tick()
    second=queue.snapshot()['jobs'][1]
    assert second['members'][0]['attempt']==2 and first['handle']['key']!=second['handle']['key']
    assert len(service.submissions)==2


@pytest.mark.parametrize('after_write',[False,True])
def test_interrupted_library_start_recovers_the_same_locally_journaled_attempt(remote,tmp_path,monkeypatch,after_write):
    queue,_,_=setup(remote,tmp_path,count=1);_,service,_=remote
    queue.resume();start=queue.library.start
    def interrupted(identifier):
        if after_write:start(identifier)
        raise OSError('interrupted local acknowledgment')
    monkeypatch.setattr(queue.library,'start',interrupted)
    with pytest.raises(OSError):queue.tick()
    assert queue.snapshot()['jobs'][0]['members'][0]['attempt']==1 and not service.submissions
    monkeypatch.setattr(queue.library,'start',start)
    result=queue.tick()
    assert result['active']==1 and len(service.submissions)==1
    assert queue.library.document['entries'][0]['attempt']==1


def test_unready_checkpoint_never_submits_and_can_resume_without_new_attempt(remote,tmp_path):
    queue,_,_=setup(remote,tmp_path,count=1);_,service,_=remote
    queue.resume();checkpoint=queue.checkpoint
    def unavailable(_):raise TimeoutError('History upload still queued')
    queue.checkpoint=unavailable
    result=queue.tick()
    assert result['active']==1 and not service.submissions
    assert result['queue']['jobs'][0]['phase']=='awaiting_submission'
    queue.checkpoint=checkpoint
    queue.tick()
    assert len(service.submissions)==1 and len(queue.snapshot()['jobs'])==1


def test_lost_job_ack_holds_its_slot_until_existing_job_can_be_recovered(remote,tmp_path,monkeypatch):
    queue,_,_=setup(remote,tmp_path,count=2);runtime,service,_=remote
    queue.resume();service.lose_ack=True;queue.tick()
    monkeypatch.setattr(runtime,'find_dispatch',lambda key:None)
    assert queue.tick()['active']==1 and len(service.submissions)==1
    task=queue.snapshot()['jobs'][0];assert task['phase']=='needs_recovery'
    monkeypatch.setattr(runtime,'find_dispatch',lambda key:service.jobs.get(key))
    queue.tick()
    assert queue.snapshot()['jobs'][0]['handle']['job']
    assert len(service.submissions)==1


def test_foreign_history_restore_and_arbitrary_queue_locations_are_rejected(remote,tmp_path):
    queue,_,_=setup(remote,tmp_path);runtime,_,source=remote
    state=queue.snapshot();state['history_id']='other-history'
    with pytest.raises(ValueError,match='another library'):queue.restore(state)
    with pytest.raises(ValueError,match='single preparation queue'):
        PreparationQueue(tmp_path/'elsewhere',queue.library,runtime,source=source,checkpoint=lambda _:None)


def test_finished_job_waits_for_published_result_without_restarting(remote,tmp_path):
    queue,_,_=setup(remote,tmp_path,count=1);_,service,_=remote
    queue.resume();queue.tick();task=queue.snapshot()['jobs'][0]
    service.jobs[task['handle']['key']].status=RunStatus.OK
    result=queue.tick()
    assert result['active']==1 and 'publish' in result['queue']['jobs'][0]['reason']
    assert len(service.submissions)==1
    complete(queue,service,task,tmp_path)
    assert queue.tick()['coverage']['all_prepared']


def test_lost_queue_publication_ack_reuses_the_already_verified_library_entry(remote,tmp_path,monkeypatch):
    queue,_,_=setup(remote,tmp_path,count=1);_,service,_=remote
    queue.resume();queue.tick();task=queue.snapshot()['jobs'][0]
    complete(queue,service,task,tmp_path)
    save=queue._save
    def lost(state,**kwargs):
        if state['jobs'][0]['phase']=='collected':raise OSError('lost queue write acknowledgment')
        return save(state,**kwargs)
    monkeypatch.setattr(queue,'_save',lost)
    with pytest.raises(OSError):queue.tick()
    assert queue.library.coverage()['all_prepared']
    assert queue.snapshot()['jobs'][0]['phase']=='submitted'
    monkeypatch.setattr(queue,'_save',save)
    assert queue.tick()['coverage']['all_prepared']
    assert queue.snapshot()['jobs'][0]['phase']=='collected' and len(service.submissions)==1


def test_invalid_result_remains_failed_without_blocking_other_models(remote,tmp_path,monkeypatch):
    queue,_,_=setup(remote,tmp_path,count=3);runtime,service,_=remote
    runtime.max_active_jobs=2;queue.resume(maximum_jobs=2);queue.tick()
    first,second=queue.snapshot()['jobs']
    complete(queue,service,first,tmp_path);complete(queue,service,second,tmp_path)
    collect=runtime.collect_native
    def checked(handle,*args,**kwargs):
        if handle['key']==first['handle']['key']:raise ValueError('wrong source revision')
        return collect(handle,*args,**kwargs)
    monkeypatch.setattr(runtime,'collect_native',checked)
    result=queue.tick()
    assert result['coverage']['counts']['failed']==1
    assert result['coverage']['counts']['prepared']==1
    assert result['active']==1 and len(service.submissions)==3
    assert 'verification' in queue.library.row(entry(first))['reason']
