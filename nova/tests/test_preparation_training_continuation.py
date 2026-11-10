"""Real packet/receipt recovery with artificial optimizer arrays, not physics."""
import copy
import io
import json
from pathlib import Path
import zipfile

import pytest

from gpx_recipe.preparation.library import import_library
from gpx_recipe.preparation.structural_jobs import read_plan,CONTINUED_PLAN_SCHEMA
from gpx_recipe.preparation.structural_training_segments import REASON
from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.symmetry.contracts import digest
from radar_pd_nova.model_runtime import GalaxyModelRuntime
from radar_pd_nova.models import RunStatus
from radar_pd_nova.preparation_queue import PreparationQueue
from radar_pd_nova.preparation_queue_results import requests_for
from test_model_runtime import remote
from test_structural_paired_preparation import completed_library
from test_structural_training_continuation import partial


def setup(remote,root):
    runtime,service,_=remote;library=completed_library(root/'paired')
    source=root/'paired/parent.cif'
    descriptor=dict(id='paired-parent',source_path=str(source),source_sha256=file_sha(source),
        dataset_id=service._put(source.read_bytes(),'paired parent CIF'))
    queue=PreparationQueue(library.root/'queue',library,runtime,source=descriptor,checkpoint=lambda _:None)
    base=root/'base.zip';library.export(base)
    queue.resume();queue.tick()
    return queue,descriptor,base


def finish_segment(queue,service,job,*,warmup=20,mutate=None):
    requests,_=requests_for(job,queue.root)
    request=requests[job['members'][0]['entry']]
    result=partial(request,warmup=warmup);output=Path(result['output'])
    result=copy.deepcopy(result);result['output']='/worker/native/'+str(output.relative_to(queue.root))
    if mutate:mutate(result)
    files={'native/'+str((output/name).relative_to(queue.root)):(output/name).read_bytes()
        for name in result['artifacts']}
    handle=job['handle']
    response=dict(schema='radar-model-native-result/v1',dispatch_key=handle['key'],
        request_sha256=digest(handle['request']),source_sha256=handle['source']['source_sha256'],
        worker_root='/worker/native',input_root='/worker/inputs-packet',result=result,
        files={name:__import__('hashlib').sha256(data).hexdigest() for name,data in files.items()},native_seconds=1.)
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w') as archive:
        archive.writestr('response.json',json.dumps(response))
        for name,raw in files.items():archive.writestr(name,raw)
    remote_job=service.jobs[handle['key']];remote_job.status=RunStatus.OK
    remote_job.outputs={'result_bundle':service._put(data.getvalue(),'training segment')}


def test_verified_checkpoint_automatically_dispatches_same_attempt_with_fresh_immutable_packet(remote,tmp_path):
    queue,_,_=setup(remote,tmp_path);_,service,_=remote
    first=queue.snapshot()['jobs'][0];finish_segment(queue,service,first)
    result=queue.tick();old,new=result['queue']['jobs']
    identifier=first['members'][0]['entry']
    assert old['outcomes'][identifier]['state']=='pending'
    assert old['outcomes'][identifier]['reason']==REASON
    assert '0 / 60 epochs' in old['reason'] and '20 warmup' in old['reason']
    assert new['members']==first['members'] and new['handle']['key']!=first['handle']['key']
    assert new['handle']['request']['input_hashes']!=first['handle']['request']['input_hashes']
    assert new['handle']['request']['payload']['destination']!=first['handle']['request']['payload']['destination']
    requests,_=requests_for(new,queue.root);plan=read_plan(requests[identifier]['plan'])
    assert plan['schema']==CONTINUED_PLAN_SCHEMA
    assert len(new['handle']['request']['input_hashes'])==7
    assert queue.library.row(identifier)['state']=='preparing' and result['active']==1
    assert len(service.submissions)==2
    queue.tick()
    assert len(service.submissions)==2


@pytest.mark.parametrize('after_dispatch',[False,True])
def test_fresh_viewer_recovers_checkpoint_from_its_original_job_and_continues(remote,tmp_path,monkeypatch,after_dispatch):
    queue,descriptor,base=setup(remote,tmp_path);runtime,service,_=remote
    first=queue.snapshot()['jobs'][0];finish_segment(queue,service,first)
    if not after_dispatch:queue.pause()
    queue.tick();snapshot=queue.snapshot()
    fresh=import_library(base,tmp_path/'restored')
    viewer=GalaxyModelRuntime(tmp_path/'fresh-runtime',service,max_active_jobs=2)
    monkeypatch.setattr(viewer,'find_dispatch',lambda key:service.jobs.get(key))
    monkeypatch.setattr(viewer,'wait_capacity',lambda cancel=None:None)
    descriptor=dict(descriptor,source_path=str(fresh.root/'inputs/parent.cif'))
    restored=PreparationQueue(fresh.root/'queue',fresh,viewer,source=descriptor,checkpoint=lambda _:None)
    restored.restore(snapshot);restored.tick()
    assert len(service.submissions)==(2 if after_dispatch else 1)
    restored.resume();restored.tick();new=restored.snapshot()['jobs'][1]
    assert new['members']==first['members'] and len(service.submissions)==2
    assert new['handle']['request']['timeout']==7200
    # Reopen after dispatch: recover that same job rather than enrolling a third.
    again=PreparationQueue(fresh.root/'queue',fresh,viewer,source=descriptor,checkpoint=lambda _:None)
    again.tick();assert len(service.submissions)==2


def test_multiple_segments_keep_one_attempt_and_advance_the_captured_state(remote,tmp_path):
    queue,_,_=setup(remote,tmp_path);_,service,_=remote
    first=queue.snapshot()['jobs'][0];finish_segment(queue,service,first,warmup=20)
    queue.tick();second=queue.snapshot()['jobs'][1]
    finish_segment(queue,service,second,warmup=40)
    queue.tick();third=queue.snapshot()['jobs'][2]
    assert first['members']==second['members']==third['members']
    requests,_=requests_for(third,queue.root)
    from gpx_recipe.preparation.structural_training_transfer import read
    path=Path(requests[first['members'][0]['entry']]['plan']).parent/'resume'
    receipt,_=read(path)
    assert receipt['warmup_completed']==40
    assert len({j['handle']['key'] for j in queue.snapshot()['jobs']})==3
    queue.tick();assert len(service.submissions)==3


def test_explicit_retry_after_a_failed_continuation_retains_acknowledged_optimizer_progress(remote,tmp_path):
    queue,_,_=setup(remote,tmp_path);_,service,_=remote
    first=queue.snapshot()['jobs'][0];finish_segment(queue,service,first)
    queue.tick();second=queue.snapshot()['jobs'][1]
    service.jobs[second['handle']['key']].status=RunStatus.ERROR
    queue.tick();identifier=first['members'][0]['entry']
    assert queue.library.row(identifier)['state']=='failed'
    queue.library.retry(identifier);queue.tick();third=queue.snapshot()['jobs'][2]
    assert third['members']==[dict(entry=identifier,attempt=2)]
    requests,_=requests_for(third,queue.root)
    assert read_plan(requests[identifier]['plan'])['schema']==CONTINUED_PLAN_SCHEMA
    assert len(service.submissions)==3


def test_transient_checkpoint_download_keeps_the_existing_dispatch_intent(remote,tmp_path,monkeypatch):
    queue,_,_=setup(remote,tmp_path);runtime,service,_=remote
    first=queue.snapshot()['jobs'][0];finish_segment(queue,service,first)
    queue.pause();queue.tick()
    collect=runtime.collect_native;failures=[True]
    def unavailable_once(*args,**kwargs):
        if failures.pop() if failures else False:raise ConnectionError('temporary checkpoint download failure')
        return collect(*args,**kwargs)
    monkeypatch.setattr(runtime,'collect_native',unavailable_once)
    queue.resume()
    with pytest.raises(ConnectionError):queue.tick()
    saved=queue.snapshot()['jobs'][1]
    assert saved['phase']=='awaiting_submission' and saved['handle'] is None
    assert len(service.submissions)==1
    queue.tick();recovered=queue.snapshot()['jobs'][1]
    assert recovered['id']==saved['id'] and recovered['members']==first['members']
    assert len(service.submissions)==2


@pytest.mark.parametrize('change',['progress','settings','model'])
def test_invalid_partial_result_stays_visible_and_cannot_start_a_replacement(remote,tmp_path,change):
    queue,_,_=setup(remote,tmp_path);_,service,_=remote
    first=queue.snapshot()['jobs'][0]
    def mutate(result):
        if change=='progress':result['warmup_completed']=40
        elif change=='settings':result['settings']['epochs']=30
        else:result['trained_model']=True
    finish_segment(queue,service,first,mutate=mutate)
    result=queue.tick()
    assert result['coverage']['counts']['failed']==1 and result['active']==0
    assert len(service.submissions)==1
    queue.tick();assert len(service.submissions)==1
