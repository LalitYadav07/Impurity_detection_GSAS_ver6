"""Recover preparation after total viewer filesystem loss using owned HDAs."""
import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from radar_pd_nova.galaxy_service import GalaxyService
from radar_pd_nova.model_runtime import GalaxyModelRuntime
from radar_pd_nova.preparation_history import PreparationHistory, saved_preparations, INPUT_LABEL, STATE_LABEL
from radar_pd_nova.preparation_registry import PreparationRegistry
from test_model_runtime import remote
from test_preparation_queue import setup, complete


def history_setup(remote,tmp_path,count=2):
    queue,snapshots,_=setup(remote,tmp_path,count=count)
    runtime,service,_=remote
    search=service.search_history_datasets
    def rows(query,**kwargs):
        return [dict(row,state='ok',file_size=len(service.data[row['id']]))
                for row in search(query,**kwargs)]
    service.search_history_datasets=rows
    def download(key,path,*,maximum_bytes=None):
        data=service.data[key]
        if maximum_bytes is not None and len(data)>maximum_bytes:raise ValueError('download byte budget')
        Path(path).write_bytes(data)
    service._download_dataset=download
    history=PreparationHistory(queue.library.root/'history',queue.library,runtime)
    def checkpoint(state):
        snapshots.append(copy.deepcopy(state))
        return history.save(state)
    queue.checkpoint=checkpoint
    return queue,history


def test_base_uploaded_once_and_progress_snapshots_contain_only_metadata(remote,tmp_path):
    queue,history=history_setup(remote,tmp_path);_,service,_=remote
    queue.resume();queue.tick()
    first=saved_preparations(service)[0]
    task=queue.snapshot()['tasks'][0];complete(queue,service,task,tmp_path);queue.pause();queue.tick()
    second=saved_preparations(service)[0]
    assert first['queue_id']==second['queue_id'] and first['dataset_id']!=second['dataset_id']
    bases=[key for key,label in service.labels.items() if INPUT_LABEL in label]
    assert len(bases)==1
    import io
    with zipfile.ZipFile(io.BytesIO(service.data[bases[0]])) as archive:
        assert not any(name.endswith('.npz') for name in archive.namelist())
    envelope=json.loads(gzip.decompress(service.data[second['dataset_id']]))
    assert set(envelope)=={'schema','history_id','queue_id','base','queue'}
    assert envelope['base']['dataset_id']==bases[0]
    assert len(service.data[second['dataset_id']])<100000
    count=len(service.data)
    assert history.save(queue.snapshot())==second['dataset_id']
    assert len(service.data)==count


def test_total_viewer_loss_restores_inputs_and_verified_arrays_without_new_jobs(remote,tmp_path,monkeypatch):
    queue,history=history_setup(remote,tmp_path);_,service,_=remote
    queue.resume();queue.tick();task=queue.snapshot()['tasks'][0]
    complete(queue,service,task,tmp_path);queue.pause();queue.tick()
    dataset=saved_preparations(service)[0]['dataset_id']
    registry=PreparationRegistry(tmp_path/'fresh-workspace')
    runtime=GalaxyModelRuntime(tmp_path/'fresh-runtime',service)
    monkeypatch.setattr(runtime,'find_dispatch',lambda key:service.jobs.get(key))
    key,restored,reference=PreparationHistory.recover(tmp_path/'recovery',runtime,registry,dataset)
    assert restored.snapshot()['paused']
    assert restored.library.coverage()['counts']['prepared']==0
    result=restored.tick()
    assert result['coverage']['counts']['prepared']==1
    assert len(service.submissions)==1
    assert registry.get(key).prepared_assets(task['entry'])
    assert reference.reference['queue_id']==history.reference['queue_id']


def test_recovery_preserves_failed_attempt_and_later_successful_retry(remote,tmp_path,monkeypatch):
    from radar_pd_nova.models import RunStatus
    queue,_=history_setup(remote,tmp_path,count=1);_,service,_=remote
    queue.resume();queue.tick();first=queue.snapshot()['tasks'][0]
    service.jobs[first['handle']['key']].status=RunStatus.ERROR;queue.tick()
    queue.library.retry(first['entry']);queue.tick();second=queue.snapshot()['tasks'][1]
    complete(queue,service,second,tmp_path);queue.tick()
    dataset=saved_preparations(service)[0]['dataset_id']
    registry=PreparationRegistry(tmp_path/'recovered-registry')
    runtime=GalaxyModelRuntime(tmp_path/'fresh-runtime',service)
    monkeypatch.setattr(runtime,'find_dispatch',lambda key:service.jobs.get(key))
    _,restored,_=PreparationHistory.recover(tmp_path/'restore',runtime,registry,dataset)
    assert restored.tick()['coverage']['all_prepared']
    assert [t['phase'] for t in restored.snapshot()['tasks']]==['failed','prepared']
    assert restored.library.row(first['entry'])['attempt']==2
    assert len(service.submissions)==2


def test_old_checkpoint_is_rejected_when_newer_ready_state_exists(remote,tmp_path):
    queue,_=history_setup(remote,tmp_path);runtime,service,_=remote
    queue.resume();old=saved_preparations(service)[0]['dataset_id'];queue.tick()
    assert saved_preparations(service)[0]['dataset_id']!=old
    registry=PreparationRegistry(tmp_path/'registry')
    with pytest.raises(ValueError,match='latest ready'):
        PreparationHistory.recover(tmp_path/'restore',runtime,registry,old)
    assert registry.summaries()==[]


def test_changed_base_is_rejected_before_registering_a_library(remote,tmp_path):
    queue,history=history_setup(remote,tmp_path);runtime,service,_=remote
    queue.resume();dataset=saved_preparations(service)[0]['dataset_id']
    service.data[history.reference['base']['dataset_id']]=b'changed private scientific input'
    registry=PreparationRegistry(tmp_path/'registry')
    with pytest.raises(ValueError,match='base changed'):
        PreparationHistory.recover(tmp_path/'restore',runtime,registry,dataset)
    assert registry.summaries()==[]


def test_foreign_history_checkpoint_is_rejected(remote,tmp_path):
    queue,_=history_setup(remote,tmp_path);runtime,service,_=remote
    queue.resume();dataset=saved_preparations(service)[0]['dataset_id']
    envelope=json.loads(gzip.decompress(service.data[dataset]));envelope['history_id']='other-history'
    service.data[dataset]=gzip.compress(json.dumps(envelope).encode())
    registry=PreparationRegistry(tmp_path/'registry')
    with pytest.raises(ValueError,match='another History'):
        PreparationHistory.recover(tmp_path/'restore',runtime,registry,dataset)
    assert registry.summaries()==[]


def test_pending_newest_upload_preserves_last_ready_intent(remote,tmp_path):
    queue,history=history_setup(remote,tmp_path);_,service,_=remote
    queue.resume();queue.tick();ready=saved_preparations(service)[0]
    original=service.search_history_datasets
    service.search_history_datasets=lambda query,**kwargs:[
        dict(id='pending',name='RADAR-PD '+STATE_LABEL+history.reference['queue_id']+' | newer.json.gz',state='queued'),
        *original(query,**kwargs)]
    assert saved_preparations(service)==[ready]


def test_editing_old_dataset_does_not_restore_stale_intent(remote,tmp_path):
    queue,history=history_setup(remote,tmp_path);_,service,_=remote
    queue.resume();old=saved_preparations(service)[0]['dataset_id'];queue.tick()
    ready=saved_preparations(service)[0]
    assert ready['dataset_id']!=old and ready['name']=='Queue transport control'
    old_row=dict(id=old,name=service.labels[old],state='ok',
        create_time='2026-11-06T01:00:00',update_time='2026-11-06T03:00:00')
    new_row=dict(id=ready['dataset_id'],name=service.labels[ready['dataset_id']],state='ok',
        create_time='2026-11-06T02:00:00',update_time='2026-11-06T02:00:00')
    service.search_history_datasets=lambda **kwargs:[old_row,new_row]
    assert saved_preparations(service)==[ready]


@pytest.mark.parametrize('failure',['oversized','interrupted'])
def test_bounded_download_preserves_old_artifact_and_removes_only_partial(remote,tmp_path,monkeypatch,failure):
    from radar_pd_nova import galaxy_service as module
    service=GalaxyService('https://galaxy.invalid','private','history',output_root=tmp_path/'service')
    target=tmp_path/'retained';target.write_bytes(b'original')
    closed=[]
    def chunks(**kwargs):
        yield b'abc'
        if failure=='interrupted':raise OSError('network interrupted')
        yield b'def'
    response=SimpleNamespace(raise_for_status=lambda:None,iter_content=chunks,close=lambda:closed.append(True))
    monkeypatch.setattr(module.requests,'get',lambda *a,**k:response)
    with pytest.raises(ValueError if failure=='oversized' else OSError):
        service._download_dataset('dataset',target,maximum_bytes=4)
    assert target.read_bytes()==b'original' and closed==[True]
    assert not list(tmp_path.glob('*.partial'))


def test_bounded_download_accepts_exact_limit_and_publishes_atomically(tmp_path,monkeypatch):
    from radar_pd_nova import galaxy_service as module
    service=GalaxyService('https://galaxy.invalid','private','history',output_root=tmp_path/'service')
    target=tmp_path/'result'
    response=SimpleNamespace(raise_for_status=lambda:None,iter_content=lambda **k:iter([b'ab',b'cd']),close=lambda:None)
    monkeypatch.setattr(module.requests,'get',lambda *a,**k:response)
    service._download_dataset('dataset',target,maximum_bytes=4)
    assert target.read_bytes()==b'abcd' and not list(tmp_path.glob('*.partial'))
