"""Bounded, read-only reconciliation must never confuse failure with absence."""
from collections import Counter
import threading
from types import SimpleNamespace

import pytest

from radar_pd_nova.model_runtime import GalaxyModelRuntime, MODEL_RECOVERY_READERS, MODEL_TOOL_ID
from radar_pd_nova.refinement_runtime import RuntimeUnavailable


def runtime_with_jobs(tmp_path,monkeypatch,count=9):
    jobs={f'job{i}':dict(id=f'job{i}',history_id='history',tool_id=MODEL_TOOL_ID,
                       state='running',params={'dispatch_key':f'key{i}'}) for i in range(count)}
    reads=[];pages=[]
    def details(uid):
        reads.append(uid)
        return jobs[uid]
    service=SimpleNamespace(galaxy_url='https://galaxy.invalid',history_id='history',_headers={},
                            _job_details=details,_job_output_ids=lambda job:{})
    class Response:
        def __init__(self,rows):self.rows=rows
        def raise_for_status(self):pass
        def json(self):return self.rows
    def get(url,**kw):
        assert url=='https://galaxy.invalid/api/jobs'
        assert kw['allow_redirects'] is False
        params=kw['params']
        assert params['history_id']=='history' and params['tool_id']==MODEL_TOOL_ID
        pages.append(params['offset'])
        return Response([{'id':uid} for uid in list(jobs)[params['offset']:params['offset']+params['limit']]])
    monkeypatch.setattr('requests.get',get)
    return GalaxyModelRuntime(tmp_path,service),jobs,reads,pages


def test_complete_lookup_uses_at_most_four_concurrent_readers(tmp_path,monkeypatch):
    runtime,jobs,reads,pages=runtime_with_jobs(tmp_path,monkeypatch,8)
    lock=threading.Lock();barrier=threading.Barrier(MODEL_RECOVERY_READERS)
    active=peak=0
    def details(uid):
        nonlocal active,peak
        with lock:
            reads.append(uid);active+=1;peak=max(peak,active)
        try:
            barrier.wait(timeout=5)
            return jobs[uid]
        finally:
            with lock:active-=1
    monkeypatch.setattr(runtime.service,'_job_details',details)
    progress=[]
    found=runtime.find_dispatches_complete({'key0','key7','absent'},progress=progress.append)
    assert peak==4 and active==0
    assert found['key0'].galaxy_job_id=='job0' and found['key7'].galaxy_job_id=='job7'
    assert found['absent'] is None and pages==[0] and progress==[0,4,8]
    assert Counter(reads)==Counter(jobs.keys())
    runtime.verify_job(found['key0'],'key0')
    assert Counter(reads)==Counter(jobs.keys()), 'A matched, verified record must not be fetched twice'


def test_complete_lookup_checks_later_pages_even_after_a_match(tmp_path,monkeypatch):
    runtime,jobs,reads,pages=runtime_with_jobs(tmp_path,monkeypatch,203)
    found=runtime.find_dispatches_complete({'key0','key202','missing'})
    assert found['key202'].galaxy_job_id=='job202' and found['missing'] is None
    assert pages==[0,200] and len(reads)==203
    jobs['job202']['params']['dispatch_key']='key0'
    with pytest.raises(RuntimeUnavailable,match='Several'):
        runtime.find_dispatches_complete({'key0'})


@pytest.mark.parametrize('damage',['timeout','malformed','id','history_id','tool_id'])
def test_failed_batch_stops_reading_without_establishing_absence(tmp_path,monkeypatch,damage):
    runtime,jobs,reads,pages=runtime_with_jobs(tmp_path,monkeypatch,12)
    def details(uid):
        reads.append(uid)
        if uid=='job2':
            if damage=='timeout':raise TimeoutError('Interrupted metadata read')
            if damage=='malformed':return []
            return dict(jobs[uid],**{damage:'wrong'})
        return jobs[uid]
    monkeypatch.setattr(runtime.service,'_job_details',details)
    with pytest.raises((TimeoutError,RuntimeUnavailable)):
        runtime.find_dispatches_complete({'missing'})
    assert set(reads)<=set(list(jobs)[:4])
    assert not runtime._verified_jobs and pages==[0]


def test_page_identity_errors_are_rejected_before_detail_requests(tmp_path,monkeypatch):
    runtime,_,reads,_=runtime_with_jobs(tmp_path,monkeypatch)
    class Response:
        def raise_for_status(self):pass
        def json(self):return [{'id':'job0'},{'id':'job0'}]
    monkeypatch.setattr('requests.get',lambda *a,**kw:Response())
    with pytest.raises(RuntimeUnavailable,match='repeated'):
        runtime.find_dispatches_complete({'missing'})
    assert reads==[]


def test_no_missing_dispatches_requires_no_history_reads(tmp_path,monkeypatch):
    runtime,_,reads,pages=runtime_with_jobs(tmp_path,monkeypatch)
    assert runtime.find_dispatches_complete(set())=={}
    assert not reads and not pages


def test_changing_history_during_lookup_cannot_authorize_missing_retry(tmp_path,monkeypatch):
    runtime,_,reads,_=runtime_with_jobs(tmp_path,monkeypatch)
    def progress(checked):
        if checked:runtime.service.history_id='another-history'
    with pytest.raises(RuntimeUnavailable,match='History changed'):
        runtime.find_dispatches_complete({'missing'},progress=progress)
    assert len(reads)==4
