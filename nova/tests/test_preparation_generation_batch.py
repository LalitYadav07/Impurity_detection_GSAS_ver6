"""Capability-gated full-unit generation keeps the existing recovery contract."""
from pathlib import Path

import pytest

from gpx_recipe.preparation import batch
from gpx_recipe.preparation.library import import_library
from gpx_recipe.preparation.operations import stage_entry
from gpx_recipe.symmetry.compiler import file_sha
from radar_pd_nova.model_runtime import GalaxyModelRuntime
from radar_pd_nova.preparation_queue import PreparationQueue
from radar_pd_nova.refinement_runtime import RuntimeUnavailable
from test_model_runtime import remote
from test_preparation_queue import complete
from test_structural_preparation_dependencies import library_inputs,complete_examples


def queue_for(remote,tmp_path,*,enabled=True,complete_first=False):
    runtime,service,source=remote
    library,originals=library_inputs(tmp_path/'fixture')
    if complete_first:complete_examples(library,originals,tmp_path/'completed-examples')
    parent=library.root/'inputs/parent.cif'
    source=dict(source,source_path=str(parent),source_sha256=file_sha(parent))
    service.tool_has_input=lambda tool,name:enabled and name==batch.EXAMPLES_CAPABILITY
    queue=PreparationQueue(library.root/'queue',library,runtime,source=source,checkpoint=lambda _:None)
    base=tmp_path/'base.zip';library.export(base)
    return queue,base,source


def test_supported_backend_groups_complete_units_and_preserves_scientific_requests(remote,tmp_path):
    queue,_,_=queue_for(remote,tmp_path);runtime,service,_=remote
    runtime.max_active_jobs=4;queue.resume(maximum_jobs=4);result=queue.tick()
    jobs=queue.snapshot()['jobs'];assert result['active']==len(jobs)==1
    job=jobs[0];assert job['mode']=='examples-batch' and len(job['members'])==2
    handle=job['handle'];assert handle['request']['operation']==batch.EXAMPLES_OPERATION
    assert handle['request']['timeout']==3690
    assert handle['request']['payload']['seconds']==3660
    requests,_=queue._inputs(job)
    for member in job['members']:
        captured=requests[member['entry']]
        single=stage_entry(queue.library,member['entry'],member['attempt'],
            queue.root/'comparison'/member['entry'])
        assert captured['settings']==single['settings'] and captured['grid']==single['grid']
        assert all(file_sha(captured[k])==file_sha(single[k]) for k in ('source','instrument','child'))
        assert queue.library.row(member['entry'])['model']['kind']=='structural-examples'
    assert len(service.submissions)==1


def test_older_backend_keeps_single_full_unit_jobs_without_claiming_a_worker_pool(remote,tmp_path):
    queue,_,_=queue_for(remote,tmp_path,enabled=False);_,service,_=remote
    queue.resume();queue.tick();job=queue.snapshot()['jobs'][0]
    assert job['mode']=='single' and len(job['members'])==1
    assert job['handle']['request']['operation']=='preparation-structural-examples'
    assert job['handle']['request']['timeout']==3600
    assert 'workers' not in job['handle']['request']['payload']
    assert len(service.submissions)==1


def test_training_remains_a_separate_single_job(remote,tmp_path):
    queue,_,_=queue_for(remote,tmp_path,complete_first=True)
    queue.resume();queue.tick();job=queue.snapshot()['jobs'][0]
    assert job['mode']=='single' and len(job['members'])==1
    assert job['handle']['request']['operation']=='preparation-structural-training'


def test_unstarted_units_recover_same_attempt_even_if_backend_capability_changes(remote,tmp_path,monkeypatch):
    queue,base,source=queue_for(remote,tmp_path);_,service,_=remote
    queue.resume();queue.tick();first=queue.snapshot()['jobs'][0]
    queue.pause()
    complete(queue,service,first,tmp_path,states={m['entry']:'pending' for m in first['members']})
    queue.tick();snapshot=queue.snapshot()
    library=import_library(base,tmp_path/'restored-library')
    runtime=GalaxyModelRuntime(tmp_path/'restored-runtime',service,max_active_jobs=1)
    monkeypatch.setattr(runtime,'find_dispatch',lambda key:service.jobs.get(key))
    monkeypatch.setattr(runtime,'wait_capacity',lambda cancel=None:None)
    service.tool_has_input=lambda *args:False
    source=dict(source,source_path=str(library.root/'inputs/parent.cif'))
    restored=PreparationQueue(library.root/'queue',library,runtime,source=source,checkpoint=lambda _:None)
    restored.restore(snapshot);restored.tick()
    assert len(service.submissions)==1
    restored.resume();restored.tick();jobs=restored.snapshot()['jobs']
    assert len(jobs)==2 and jobs[0]['mode']=='examples-batch' and jobs[1]['mode']=='single'
    assert jobs[1]['members'][0]['attempt']==1
    assert jobs[1]['members'][0]['entry'] in {m['entry'] for m in first['members']}
    assert len(service.submissions)==2


def test_full_budget_failure_is_not_automatically_resubmitted(remote,tmp_path):
    queue,_,_=queue_for(remote,tmp_path);_,service,_=remote
    queue.resume();queue.tick();job=queue.snapshot()['jobs'][0]
    complete(queue,service,job,tmp_path,states={m['entry']:'failed' for m in job['members']})
    queue.tick();queue.tick()
    assert len(service.submissions)==1
    assert queue.library.coverage()['counts']['failed']==2


def test_direct_new_submission_cannot_bypass_the_backend_marker(remote,tmp_path):
    runtime,service,source=remote;service.tool_has_input=lambda *args:False
    with pytest.raises(RuntimeUnavailable,match='does not support full-budget'):
        runtime._submit('native',source,dict(operation=batch.EXAMPLES_OPERATION),'asset','a'*24)
    assert not service.submissions
