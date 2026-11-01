"""A revoked fresh retry must still acknowledge work Galaxy already accepted."""
from types import SimpleNamespace
import pytest
from radar_pd_nova.galaxy_service import GalaxyService
from radar_pd_nova.live_recovery import RetrySourceChanged


class Tool:
    def __init__(self, id): self.tool_id=id
    def assign_id(self,uid,store): self.uid=uid;self.store=store


def setup(tmp_path,monkeypatch):
    service=GalaxyService('https://galaxy.invalid','test-key','history',output_root=tmp_path)
    monkeypatch.setattr('nova.galaxy.Tool',Tool)
    posts=[]
    def post(*args):posts.append(1);raise OSError('ACK lost')
    monkeypatch.setattr(service,'_submit_analysis',post)
    return service,posts


def test_receipt_visible_during_backoff_is_returned_even_after_stop(tmp_path,monkeypatch):
    service,posts=setup(tmp_path,monkeypatch)
    allowed=[True];visible=[False];queries=[];guards=[]
    def lookup(name,configuration):
        queries.append((name,configuration));return 'accepted-retry' if visible[0] else None
    def sleep(_):allowed[0]=False;visible[0]=True
    def guard():
        guards.append(1)
        if not allowed[0]:raise RetrySourceChanged('Stopped')
    monkeypatch.setattr(service,'_find_acknowledged_job',lookup)
    monkeypatch.setattr('radar_pd_nova.galaxy_service.time.sleep',sleep)
    uid,tool=service._submit_with_retry(object(),object(),run_name='IPTS-1_PG3_123_retry1',
        config_dataset_id='same-config',before_submit=guard)
    assert uid==tool.uid=='accepted-retry' and len(posts)==1 and len(guards)==1
    assert queries==[('IPTS-1_PG3_123_retry1','same-config')]*2


def test_receipt_visible_while_guard_rejects_is_still_reconciled(tmp_path,monkeypatch):
    service,posts=setup(tmp_path,monkeypatch)
    results=iter([None,None,'accepted-retry']);guard_calls=[]
    monkeypatch.setattr(service,'_find_acknowledged_job',lambda *a:next(results))
    monkeypatch.setattr('radar_pd_nova.galaxy_service.time.sleep',lambda *a:None)
    def guard():
        guard_calls.append(1)
        if len(guard_calls)>1:raise RetrySourceChanged('Original error changed to OK')
    uid,tool=service._submit_with_retry(object(),object(),run_name='retry-generation1',
        config_dataset_id='same-config',before_submit=guard)
    assert uid==tool.uid=='accepted-retry' and len(posts)==1 and len(guard_calls)==2


def test_visible_receipt_prevents_second_post_even_if_guard_still_allows(tmp_path,monkeypatch):
    service,posts=setup(tmp_path,monkeypatch)
    results=iter([None,'accepted-retry'])
    monkeypatch.setattr(service,'_find_acknowledged_job',lambda *a:next(results))
    monkeypatch.setattr('radar_pd_nova.galaxy_service.time.sleep',lambda *a:None)
    uid,_=service._submit_with_retry(object(),object(),run_name='retry-generation1',
        config_dataset_id='same-config',before_submit=lambda:None)
    assert uid=='accepted-retry' and len(posts)==1


def test_revoked_guard_without_receipt_never_reposts(tmp_path,monkeypatch):
    service,posts=setup(tmp_path,monkeypatch)
    count=[0]
    monkeypatch.setattr(service,'_find_acknowledged_job',lambda *a:None)
    monkeypatch.setattr('radar_pd_nova.galaxy_service.time.sleep',lambda *a:None)
    def guard():
        count[0]+=1
        if count[0]>1:raise RetrySourceChanged('Stopped')
    with pytest.raises(RetrySourceChanged):
        service._submit_with_retry(object(),object(),run_name='retry-generation1',
            config_dataset_id='same-config',before_submit=guard)
    assert len(posts)==1
