"""One collection pass shares metadata, never per-member result disposition."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpx_recipe.symmetry.campaign import Campaign
from radar_pd_nova.model_runtime import GalaxyChunkTransport, CollectOnlyTransport, tick_model_campaign
from radar_pd_nova.models import UtilityActionRecord, RunStatus


def transport(tmp_path, *, chunks=1, members=2):
    campaign=Campaign(tmp_path/'campaign')
    service=SimpleNamespace(history_id='history', states={}, calls=[], failures=set(), clock=0., delay=.25, cancelled=[])
    def refresh(job):
        service.calls.append(job.uid)
        service.clock+=service.delay
        if job.uid in service.failures:
            service.failures.remove(job.uid)
            raise OSError('Transient metadata failure')
        job.status=service.states[job.uid]
        return job
    service.refresh_utility=refresh
    runtime=SimpleNamespace(service=service, max_active_jobs=8, failed_request_proof=lambda *a,**k:None)
    result=GalaxyChunkTransport(runtime,campaign)
    result.checkpoint=lambda **kwargs:None
    handles=[]
    for index in range(chunks):
        key='chunk-'+str(index);ids=[f'trial-{index}-{member}' for member in range(members)]
        service.states[key]=RunStatus.QUEUED
        result.state['chunks'][key]=dict(key=key,members=ids,protocol='protocol',job=None)
        for uid in ids:
            (campaign.root/'trials'/uid).mkdir(parents=True)
            result.state['members'][uid]=key
            handles.append(dict(dispatch_key=key,key=uid))
    def job(chunk):
        key=chunk['key']
        return UtilityActionRecord(uid=key,tool_id='model-tool',name=key,galaxy_job_id='job-'+key,status=RunStatus.QUEUED)
    result._job=job
    def cancel(job,key):
        service.cancelled.append(key);service.states[key]=RunStatus.CANCELLED
    runtime.cancel_job=cancel
    return result,service,handles


def reply(tmp_path, t, key, *, pending=()):
    directory=tmp_path/('reply-'+key);directory.mkdir(exist_ok=True)
    members=t.state['chunks'][key]['members']
    terminal=[]
    for uid in members:
        if uid in pending:continue
        target=directory/'trials'/uid;target.mkdir(parents=True,exist_ok=True)
        (target/'result.json').write_text(json.dumps(dict(schema='radar-symmetry-trial/v1',id=uid,
            protocol_sha256='protocol',status='unsupported')))
        terminal.append(dict(id=uid,state='finished',status='unsupported'))
    (directory/'response.json').write_text(json.dumps(dict(schema='radar-model-chunk-result/v1',
        dispatch_key=key,protocol_sha256='protocol',trials=terminal,pending=list(pending),allocation=dict(workers=1))))
    return directory


class PollLedger:
    tick=Campaign.tick
    def __init__(self,root,handles):
        self.root=root
        self.rows=[dict(id=h['key'],state='running',handle=json.dumps(h)) for h in handles]
    def exclusive(self):return nullcontext()
    def trials(self):return copy.deepcopy(self.rows)
    def trial_directory(self,row):return self.root/row['id']
    def setting(self,key,default=None):return default
    def set_state(self,key,state,**kwargs):
        row=next(r for r in self.rows if r['id']==key);row['state']=state
        if kwargs.get('handle'):row['handle']=json.dumps(kwargs['handle'])


def test_eight_chunks_64_interleaved_members_use_eight_metadata_reads(tmp_path,monkeypatch):
    t,service,handles=transport(tmp_path,chunks=8,members=8)
    handles.sort(key=lambda h:(int(h['key'].rsplit('-',1)[-1]),h['dispatch_key']))
    ledger=PollLedger(tmp_path/'absent',handles)
    monkeypatch.setattr('radar_pd_nova.model_runtime.time.monotonic',lambda:service.clock)
    tick_model_campaign(ledger,CollectOnlyTransport(t),slots=4)
    assert Counter(service.calls)=={f'chunk-{index}':1 for index in range(8)}
    assert service.clock==2.
    assert all(row['state']=='running' for row in ledger.rows)
    # A later collection pass is fresh even with no added wall-clock delay.
    tick_model_campaign(ledger,CollectOnlyTransport(t),slots=4)
    assert Counter(service.calls)=={f'chunk-{index}':2 for index in range(8)}


def test_next_pass_observes_queued_running_then_finished_without_ttl_delay(tmp_path,monkeypatch):
    t,service,handles=transport(tmp_path)
    monkeypatch.setattr('radar_pd_nova.model_runtime.time.monotonic',lambda:0.)
    for state in [RunStatus.QUEUED,RunStatus.RUNNING]:
        service.states['chunk-0']=state
        with t.collection_pass():
            assert all(t.poll(h,tmp_path)['state']=='running' for h in handles)
    assert service.calls==['chunk-0','chunk-0']
    service.states['chunk-0']=RunStatus.OK
    directory=reply(tmp_path,t,'chunk-0')
    t.runtime.result=lambda *args:directory
    with t.collection_pass():
        assert all(t.poll(h,tmp_path)['state']=='finished' for h in handles)
    assert service.calls==['chunk-0']*3
    assert all((t.campaign.root/'trials'/h['key']/'result.json').is_file() for h in handles)


def test_metadata_failure_is_retryable_and_does_not_poison_other_chunks(tmp_path):
    t,service,handles=transport(tmp_path,chunks=2)
    bad,again,good=handles[0],handles[1],handles[2]
    service.failures.add('chunk-0')
    with t.collection_pass():
        with pytest.raises(OSError,match='Transient'):t.poll(bad,tmp_path)
        assert t.poll(good,tmp_path)['state']=='running'
        assert t.poll(again,tmp_path)['state']=='running'
        assert t.poll(bad,tmp_path)['state']=='running'
    assert Counter(service.calls)=={'chunk-0':2,'chunk-1':1}
    with t.collection_pass():t.poll(bad,tmp_path)
    assert service.calls.count('chunk-0')==3


def test_partial_result_download_can_retry_without_an_extra_metadata_read(tmp_path):
    t,service,handles=transport(tmp_path)
    with t.collection_pass():assert t.poll(handles[0],tmp_path)['state']=='running'
    service.states['chunk-0']=RunStatus.OK
    directory=reply(tmp_path,t,'chunk-0');downloads=[]
    def result(*args):
        downloads.append(True)
        if len(downloads)==1:raise OSError('Interrupted archive transfer')
        return directory
    t.runtime.result=result
    with t.collection_pass():
        with pytest.raises(OSError,match='Interrupted'):t.poll(handles[0],tmp_path)
        assert not (t.campaign.root/'trials'/handles[0]['key']/'result.json').exists()
        assert t.poll(handles[1],tmp_path)['state']=='finished'
    assert len(downloads)==2 and service.calls==['chunk-0','chunk-0']


def test_cached_metadata_does_not_bypass_archive_identity_or_member_dispositions(tmp_path):
    t,service,handles=transport(tmp_path);service.states['chunk-0']=RunStatus.OK
    directory=reply(tmp_path,t,'chunk-0',pending=[handles[1]['key']]);t.runtime.result=lambda *a:directory
    response=json.loads((directory/'response.json').read_text());response['dispatch_key']='wrong'
    (directory/'response.json').write_text(json.dumps(response))
    with t.collection_pass():
        with pytest.raises(ValueError,match='differs'):t.poll(handles[0],tmp_path)
    assert not t.state['chunks']['chunk-0'].get('complete')
    response['dispatch_key']='chunk-0';(directory/'response.json').write_text(json.dumps(response))
    with t.collection_pass():
        assert t.poll(handles[0],tmp_path)['state']=='finished'
        assert t.poll(handles[1],tmp_path)['state']=='not_started'
    assert service.calls==['chunk-0','chunk-0']


@pytest.mark.parametrize('other_thread',[False,True])
def test_cancellation_invalidates_same_pass_snapshot_and_never_downloads_a_result(tmp_path,other_thread):
    t,service,handles=transport(tmp_path)
    t.runtime.result=lambda *a:pytest.fail('Cancelled work cannot produce an accepted result')
    with t.collection_pass():
        assert t.poll(handles[0],tmp_path)['state']=='running'
        if other_thread:
            with ThreadPoolExecutor(max_workers=1) as pool:
                assert pool.submit(t.cancel,handles[0],tmp_path).result(timeout=2) is False
        else:assert t.cancel(handles[0],tmp_path) is False
        assert t.poll(handles[1],tmp_path)['state']=='cancelled'
    assert service.cancelled==['chunk-0'] and service.calls==['chunk-0','chunk-0']


def test_scope_cleanup_and_thread_isolation(tmp_path):
    t,service,handles=transport(tmp_path)
    with pytest.raises(ValueError):
        with t.collection_pass():
            t.poll(handles[0],tmp_path)
            raise ValueError('Interrupted pass')
    assert t._collection_metadata.jobs is None
    with t.collection_pass():
        t.poll(handles[0],tmp_path)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(tick_model_campaign,PollLedger(tmp_path/'absent',[handles[1]]),
                        CollectOnlyTransport(t),slots=1).result(timeout=2)
        t.poll(handles[1],tmp_path)
    assert service.calls==['chunk-0']*3


def test_non_galaxy_transport_retains_original_identity_and_admission(tmp_path):
    original=SimpleNamespace(max_active_jobs=2)
    calls=[]
    campaign=SimpleNamespace(tick=lambda transport,**kwargs:calls.append((transport,kwargs)) or 'done')
    assert tick_model_campaign(campaign,original,slots=2)=='done'
    assert calls==[(original,{'slots':2})]
