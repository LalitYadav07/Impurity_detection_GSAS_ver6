"""Result loading telemetry must diagnose waits without changing authority."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

import pytest
import requests

from radar_pd_nova.galaxy_service import GalaxyService
from radar_pd_nova.models import AnalysisMode, RunRecord, ResultStatus
from radar_pd_nova.result_loading import ResultLoadTrace, measure, submit_context, subject, transfer_slot


class Clock:
    def __init__(self): self.value = 0.
    def __call__(self): return self.value
    def advance(self, value): self.value += value


def test_file_response_wait_and_body_time_are_separate(tmp_path, monkeypatch):
    clock = Clock()
    trace = ResultLoadTrace('job', clock=clock)
    service = GalaxyService('https://example.invalid', 'secret', 'history', output_root=tmp_path)
    class Response:
        def raise_for_status(self): pass
        def iter_content(self, chunk_size):
            clock.advance(7)
            yield b'synthetic data'
        def close(self): pass
    def get(url, **kwargs):
        clock.advance(3)
        return Response()
    monkeypatch.setattr('radar_pd_nova.galaxy_service.requests.get', get)
    with trace.activate('preview'), subject('selected_plot_arrays'):
        service._download_dataset('dataset', tmp_path/'file', metadata={'id':'dataset','history_id':'history'})
    events = trace.snapshot()['events']
    assert [(e['stage'],e['duration_seconds'],e['bytes']) for e in events] == [
        ('response_headers',3.,0),('response_body',7.,14)]
    assert all(e['subject']=='selected_plot_arrays' for e in events)
    assert 'secret' not in json.dumps(trace.snapshot())
    assert 'https:' not in json.dumps(trace.snapshot())


def test_failure_type_is_recorded_without_error_contents_or_partial_file(tmp_path, monkeypatch):
    trace = ResultLoadTrace('job')
    service = GalaxyService('https://example.invalid', 'secret', 'history', output_root=tmp_path)
    class Response:
        def raise_for_status(self): pass
        def iter_content(self, chunk_size):
            yield b'partial'
            raise requests.Timeout('private URL and credentials must not enter the receipt')
        def close(self): pass
    monkeypatch.setattr('radar_pd_nova.galaxy_service.requests.get', lambda *a,**kw:Response())
    with pytest.raises(requests.Timeout), trace.activate('complete'):
        service._download_dataset('dataset', tmp_path/'file', metadata={'id':'dataset','history_id':'history'})
    event = trace.snapshot()['events'][-1]
    assert event['status']=='failed' and event['error_type']=='Timeout' and event['bytes']==7
    assert not (tmp_path/'file').exists() and not list(tmp_path.glob('*.partial'))
    assert 'private URL' not in json.dumps(trace.snapshot())


def test_concurrent_transfers_keep_transaction_context_and_wait_separate():
    trace = ResultLoadTrace('job')
    gate = threading.Semaphore(1)
    entered, release = threading.Event(), threading.Event()
    def work(first):
        with transfer_slot(gate), measure('response_body'):
            if first:
                entered.set(); assert release.wait(2)
    with trace.activate('complete'), subject('plots'), ThreadPoolExecutor(max_workers=2) as pool:
        first = submit_context(pool, work, True)
        assert entered.wait(2)
        second = submit_context(pool, work, False)
        release.set(); first.result(); second.result()
    events = trace.snapshot()['events']
    assert [e['stage'] for e in events].count('transfer_wait')==2
    assert [e['stage'] for e in events].count('response_body')==2
    assert all(e['branch']=='complete' and e['subject']=='plots' for e in events)
    assert all(e['status']=='complete' and e['duration_seconds']>=0 for e in events)


def test_measurement_is_bounded_and_optional_persistence_cannot_fail_load(tmp_path, monkeypatch):
    trace = ResultLoadTrace('job')
    for _ in range(trace.MAX_EVENTS+5):
        with trace.activate('preview'), measure('preview_parse'): pass
    assert len(trace.snapshot()['events'])==trace.MAX_EVENTS
    assert trace.snapshot()['dropped_events']==5
    assert trace.snapshot()['aggregates'][0]['count']==trace.MAX_EVENTS+5
    trace.persist(tmp_path)
    saved=json.loads(next((tmp_path/'result-loading').glob('*.json')).read_text())
    assert saved['trace_id']==trace.identifier and saved['dropped_events']==5
    monkeypatch.setattr(Path,'mkdir',lambda *a,**kw:(_ for _ in ()).throw(OSError('readonly')))
    trace.persist(tmp_path/'unwritable')


def test_untraced_code_keeps_original_exception():
    with pytest.raises(ValueError,match='original'), measure('body'):
        raise ValueError('original')


@pytest.mark.parametrize('complete',[True,False])
def test_monitor_uses_complete_job_associations_and_retains_missing_collection_fallback(tmp_path,complete):
    service=GalaxyService('https://example.invalid','secret','history',output_root=tmp_path)
    collections={'plots':{'id':'plots'}}
    if complete:collections['gpx_projects']={'id':'projects'}
    service._job_details=lambda _:dict(outputs={'summary':{'id':'summary'}},output_collections=collections)
    calls=[]
    def lookup(_):calls.append(1);return {'gpx_projects':'projects'}
    service.job_output_ids=lookup
    service._download_dataset=lambda identifier,path: path.write_text('{}')
    service._download_collection=lambda identifier,path:path.mkdir()
    service._ensure_plot_observation=lambda *a,**kw:None
    record=RunRecord(uid='job',history_id='history',name='monitor',mode=AnalysisMode.FULL,output_profile='monitor')
    result=service.collect_results(record)
    assert result.result_status==ResultStatus.READY
    assert result.output_dataset_ids=={'summary':'summary','plots':'plots','gpx_projects':'projects'}
    assert calls==([] if complete else [1])
