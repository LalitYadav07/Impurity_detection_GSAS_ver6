"""Queued metadata uploads must not stall sessions or multiply on retry."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from radar_pd_nova.checkpoint_uploads import (
    CheckpointUploads, CheckpointUploadPending, CheckpointUploadFailed,
)


class Clock:
    def __init__(self): self.now = 0.
    def __call__(self): return self.now
    def sleep(self, seconds): self.now += seconds


def manager(tmp_path, *, initial='queued'):
    path = tmp_path/'checkpoint.json';path.write_text('{"revision":1}')
    clock = Clock();posts = [];states = {}
    def upload(path, *, label):
        identifier = 'hda'+str(len(posts)+1)
        posts.append((identifier, label, Path(path).read_bytes()))
        states[identifier] = dict(state=initial)
        return identifier
    queue = CheckpointUploads(upload, lambda key, seconds: states[key], clock=clock, sleep=clock.sleep)
    return queue, path, clock, posts, states


def test_timeout_retains_acknowledged_upload_and_retry_polls_same_hda(tmp_path):
    queue,path,clock,posts,states=manager(tmp_path)
    with pytest.raises(CheckpointUploadPending,match='still queued') as caught:
        queue.save(path,label='watch',timeout=6)
    assert clock.now==6 and caught.value.dataset_id=='hda1' and len(posts)==1
    with pytest.raises(CheckpointUploadPending):queue.save(path,label='watch',timeout=2)
    assert len(posts)==1
    states['hda1']['state']='ok'
    assert queue.save(path,label='watch',timeout=2)=='hda1'
    assert queue.save(path,label='watch',timeout=2)=='hda1'
    assert len(posts)==1


def test_new_snapshot_waits_for_prior_ack_and_never_reuses_older_state(tmp_path):
    queue,path,clock,posts,states=manager(tmp_path)
    with pytest.raises(CheckpointUploadPending):queue.save(path,label='watch',timeout=2)
    path.write_text('{"revision":2}')
    with pytest.raises(CheckpointUploadPending):queue.save(path,label='watch',timeout=2)
    assert len(posts)==1
    states['hda1']['state']='ok'
    with pytest.raises(CheckpointUploadPending) as caught:queue.save(path,label='watch',timeout=2)
    assert caught.value.dataset_id=='hda2' and len(posts)==2
    states['hda2']['state']='ok'
    assert queue.save(path,label='watch',timeout=2)=='hda2'
    path.write_text('{"revision":1}')
    with pytest.raises(CheckpointUploadPending) as caught:queue.save(path,label='watch',timeout=2)
    assert caught.value.dataset_id=='hda3' and len(posts)==3


@pytest.mark.parametrize('terminal',['discarded','error','empty','failed_metadata','deleted'])
def test_only_confirmed_terminal_failure_allows_a_new_upload(tmp_path,terminal):
    queue,path,clock,posts,states=manager(tmp_path,initial=terminal)
    with pytest.raises(CheckpointUploadFailed):queue.save(path,label='watch',timeout=2)
    with pytest.raises(CheckpointUploadFailed):queue.save(path,label='watch',timeout=2)
    assert len(posts)==2


def test_deleted_ok_hda_is_not_reused(tmp_path):
    queue,path,clock,posts,states=manager(tmp_path,initial='ok')
    assert queue.save(path,label='watch')=='hda1'
    states['hda1']['deleted']=True
    with pytest.raises(CheckpointUploadFailed):queue.save(path,label='watch')
    assert queue.save(path,label='watch')=='hda2'


def test_status_read_failure_preserves_ack_and_does_not_expose_response(tmp_path):
    queue,path,clock,posts,states=manager(tmp_path)
    def broken(*args):raise TimeoutError('PRIVATE-RESPONSE-SENTINEL')
    queue.status=broken
    with pytest.raises(CheckpointUploadPending) as caught:queue.save(path,label='watch')
    assert 'SENTINEL' not in str(caught.value)
    queue.status=lambda key,seconds:dict(state='ok')
    assert queue.save(path,label='watch')=='hda1' and len(posts)==1


def test_other_dataset_status_cannot_acknowledge_this_checkpoint(tmp_path):
    queue,path,clock,posts,states=manager(tmp_path,initial='ok')
    queue.status=lambda *args:dict(id='different-hda',state='ok')
    with pytest.raises(CheckpointUploadPending,match='matching status'):
        queue.save(path,label='watch',timeout=2)
    queue.status=lambda key,seconds:dict(id=key,state='ok')
    assert queue.save(path,label='watch',timeout=2)=='hda1' and len(posts)==1


def test_identical_concurrent_callers_share_one_ack(tmp_path):
    path=tmp_path/'checkpoint.json';path.write_text('{}')
    entered=threading.Event();release=threading.Event();posts=[]
    def upload(*args,**kwargs):posts.append(1);return 'hda1'
    def status(*args):entered.set();assert release.wait(3);return dict(state='ok')
    queue=CheckpointUploads(upload,status)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(queue.save,path,label='watch');assert entered.wait(3)
        second=pool.submit(queue.save,path,label='watch');release.set()
        assert first.result(3)==second.result(3)=='hda1'
    assert posts==[1]


def test_independent_checkpoint_streams_can_publish_in_parallel(tmp_path):
    path=tmp_path/'checkpoint.json';path.write_text('{}')
    arrived=threading.Barrier(2)
    def upload(path,*,label):return label
    def status(*args):arrived.wait(timeout=3);return dict(state='ok')
    queue=CheckpointUploads(upload,status)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(queue.save,path,label=label) for label in ['watch','models']]
        assert {f.result(4) for f in futures}=={'watch','models'}


@pytest.mark.parametrize('timeout',[0,-1,601,True,float('nan'),float('inf')])
def test_invalid_wait_is_rejected(tmp_path,timeout):
    queue,path,*_=manager(tmp_path)
    with pytest.raises(ValueError):queue.save(path,label='watch',timeout=timeout)


def test_galaxy_service_uses_ack_only_then_bounded_metadata_status(tmp_path,monkeypatch):
    from radar_pd_nova.galaxy_service import GalaxyService
    import radar_pd_nova.galaxy_service as module
    service=GalaxyService('https://galaxy.invalid','PRIVATE-TEST-SENTINEL','history',output_root=tmp_path)
    posts=[];gets=[]
    def upload(path,*,label,wait_ready=True):
        posts.append(wait_ready);return 'hda1'
    service.upload_document=upload
    def get(url,**kwargs):
        gets.append((url,kwargs['timeout']))
        return SimpleNamespace(raise_for_status=lambda:None,json=lambda:dict(state='ok'))
    monkeypatch.setattr(module.requests,'get',get)
    path=tmp_path/'checkpoint.json';path.write_text('{}')
    assert service.upload_checkpoint_document(path,label='watch')=='hda1'
    assert service.upload_checkpoint_document(path,label='watch')=='hda1'
    assert posts==[False] and all(url=='https://galaxy.invalid/api/datasets/hda1' and seconds<=15 for url,seconds in gets)


def test_refinement_checkpoint_fingerprint_ignores_only_top_level_clock(tmp_path):
    from radar_pd_nova.refinement_runtime import GalaxyRefinementRuntime
    saved=[]
    def checkpoint(path,**kwargs):
        saved.append((json.loads(gzip.decompress(Path(path).read_bytes())),kwargs['semantic_sha256']))
        return 'hda1'
    service=SimpleNamespace(upload_checkpoint_document=checkpoint)
    runtime=GalaxyRefinementRuntime(tmp_path,service)
    runtime.save_workspace(dict(updated_utc='first',jobs={'scan':dict(status='running')}))
    runtime.save_workspace(dict(updated_utc='later',jobs={'scan':dict(status='running')}))
    runtime.save_workspace(dict(updated_utc='later',jobs={'scan':dict(status='completed')}))
    assert saved[0][0]['updated_utc']=='first' and saved[1][0]['updated_utc']=='later'
    assert saved[0][1]==saved[1][1]!=saved[2][1]


def test_watch_does_not_retry_an_acknowledged_pending_upload(tmp_path):
    from radar_pd_nova.powgen_controller import PowgenWatchController,WatchCheckpoint
    calls=[]
    def pending(*args,**kwargs):
        calls.append(1);raise CheckpointUploadPending('hda1','queued')
    controller=PowgenWatchController.__new__(PowgenWatchController)
    controller.settings=SimpleNamespace(ipts='IPTS-1',instrument='PG3')
    controller.service=SimpleNamespace(upload_checkpoint_json_document=pending,upload_json_document=lambda *a,**k:None)
    controller._persist_lock=threading.Lock();controller.state_dataset_ids=[]
    controller._persisted_checkpoint_generation=0;controller._last_persisted_state=None
    controller._checkpoint_key=lambda:'stable'
    with pytest.raises(CheckpointUploadPending):controller.persist_state(WatchCheckpoint(1,'{}','{}'))
    assert calls==[1] and controller.state_dataset_ids==[]
