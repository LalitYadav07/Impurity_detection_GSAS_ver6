"""A private transfer shared by preview/enrichment cannot mix jobs or revisions."""
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path

import pytest
import requests

from radar_pd_nova.galaxy_service import GalaxyService
from radar_pd_nova.result_loading import ResultLoadTrace, subject
from radar_pd_nova.result_transfers import ResultTransfers


def metadata(**changes):
    return dict(id='dataset',history_id='history',state='ok',file_size=9,**changes)


def test_simultaneous_preview_and_canonical_read_one_identical_hda_once(tmp_path, monkeypatch):
    trace=ResultLoadTrace('job');trace.transfers=ResultTransfers(tmp_path,'history')
    service=GalaxyService('https://example.invalid','secret','history',output_root=tmp_path)
    entered,release=threading.Event(),threading.Event();calls=[]
    class Response:
        def raise_for_status(self): pass
        def close(self): pass
        def iter_content(self,chunk_size):
            entered.set();assert release.wait(2);yield b'synthetic'
    monkeypatch.setattr('radar_pd_nova.galaxy_service.requests.get',lambda *a,**k:(calls.append(a[0]) or Response()))
    def load(branch):
        with trace.activate(branch),subject('pattern_observation'):
            service._download_dataset('dataset',tmp_path/branch,metadata=metadata(),maximum_bytes=100)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(load,'preview');assert entered.wait(2)
        second=pool.submit(load,'complete');release.set();first.result();second.result()
    assert len(calls)==1
    assert (tmp_path/'preview').read_bytes()==(tmp_path/'complete').read_bytes()==b'synthetic'
    assert len([e for e in trace.snapshot()['events'] if e['stage']=='response_body'])==1
    assert len([e for e in trace.snapshot()['events'] if e['stage']=='shared_transfer_copy'])==2
    root=trace.transfers.root;trace.transfers.close();assert not root.exists()


@pytest.mark.parametrize('change',[{'file_size':8},{'update_time':'new'},{'uuid':'other'}])
def test_changed_hda_revision_cannot_reuse_completed_bytes(tmp_path,change):
    cache=ResultTransfers(tmp_path,'history');calls=[]
    first=metadata();first.update(update_time='old',uuid='original')
    def fetch(path): calls.append(1);path.write_bytes(b'synthetic')
    assert cache.download('dataset',tmp_path/'a',first,None,fetch,subject='summary')
    second={**first,**change}
    with pytest.raises(ValueError,match='revision changed'):
        cache.download('dataset',tmp_path/'b',second,None,fetch,subject='summary')
    assert calls==[1] and not (tmp_path/'b').exists()
    cache.close()


def test_failed_owner_is_shared_without_retry_or_partial_files(tmp_path):
    cache=ResultTransfers(tmp_path,'history');calls=[]
    def fetch(path):
        calls.append(1);path.write_bytes(b'partial');raise requests.Timeout('failed')
    for name in ('a','b'):
        with pytest.raises(requests.Timeout):
            cache.download('dataset',tmp_path/name,metadata(),None,fetch,subject='summary')
    assert calls==[1] and not list(tmp_path.glob('*.partial'))
    assert not (tmp_path/'a').exists() and not (tmp_path/'b').exists()
    cache.close()


def test_concurrent_waiter_receives_owner_failure_without_deadlock(tmp_path):
    cache=ResultTransfers(tmp_path,'history')
    entered,release=threading.Event(),threading.Event();calls=[]
    def fetch(path):
        calls.append(1);entered.set();assert release.wait(2)
        raise requests.Timeout('owner failed')
    def read(name):
        with pytest.raises(requests.Timeout):
            cache.download('dataset',tmp_path/name,metadata(),None,fetch,subject='summary')
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner=pool.submit(read,'a');assert entered.wait(2)
        waiter=pool.submit(read,'b');release.set()
        owner.result(timeout=2);waiter.result(timeout=2)
    assert calls==[1] and not (tmp_path/'a').exists() and not (tmp_path/'b').exists()
    cache.close()


def test_cleanup_failure_cannot_leave_a_waiter_blocked(tmp_path,monkeypatch):
    cache=ResultTransfers(tmp_path,'history')
    monkeypatch.setattr(Path,'unlink',lambda *a,**kw:(_ for _ in ()).throw(OSError('cleanup denied')))
    def fetch(path):raise requests.Timeout('original transport failure')
    for name in ('a','b'):
        with pytest.raises(requests.Timeout,match='original transport failure'):
            cache.download('dataset',tmp_path/name,metadata(),None,fetch,subject='summary')
    assert next(iter(cache.files.values()))['future'].done()
    cache.close()


def test_corrupt_shared_file_never_replaces_existing_destination(tmp_path):
    cache=ResultTransfers(tmp_path,'history')
    def fetch(path):path.write_bytes(b'synthetic')
    cache.download('dataset',tmp_path/'a',metadata(),None,fetch,subject='summary')
    next(iter(cache.files.values()))['path'].write_bytes(b'corrupted')
    destination=tmp_path/'b';destination.write_bytes(b'previous good result')
    with pytest.raises(ValueError,match='changed before copying'):
        cache.download('dataset',destination,metadata(),None,fetch,subject='summary')
    assert destination.read_bytes()==b'previous good result'
    assert not list(tmp_path.glob('*.partial'))
    cache.close()


def test_native_and_archive_files_are_never_substituted(tmp_path):
    cache=ResultTransfers(tmp_path,'history')
    def forbidden(path):pytest.fail('Native/archive bypass should leave the original downloader in control')
    for role in ('gpx_projects','results_archive','phases','diagnostics'):
        assert not cache.download('dataset',tmp_path/role,metadata(),None,forbidden,subject=role)
    assert not cache.files
    cache.close()


def test_separate_transactions_never_share_a_cache(tmp_path):
    calls=[]
    def fetch(path): calls.append(1);path.write_bytes(b'synthetic')
    for name in ('first','second'):
        cache=ResultTransfers(tmp_path,'history')
        cache.download('dataset',tmp_path/name,metadata(),None,fetch,subject='summary')
        cache.close()
    assert calls==[1,1]


def test_abandoned_transaction_files_do_not_seed_recovery(tmp_path):
    abandoned=ResultTransfers(tmp_path,'history')
    abandoned.download('dataset',tmp_path/'old',metadata(),None,
        lambda path:path.write_bytes(b'synthetic'),subject='summary')
    # Simulate a process interruption by retaining its private files. A new
    # transaction must read the HDA anew, even with exactly the same identity.
    recovered=ResultTransfers(tmp_path,'history')
    calls=[]
    def fetch(path):calls.append(1);path.write_bytes(b'recovered')
    recovered.download('dataset',tmp_path/'new',metadata(),None,fetch,subject='summary')
    assert calls==[1] and (tmp_path/'new').read_bytes()==b'recovered'
    assert recovered.root != abandoned.root
    recovered.close();abandoned.close()


def test_reused_bytes_still_respect_preview_budget(tmp_path):
    cache=ResultTransfers(tmp_path,'history')
    cache.download('dataset',tmp_path/'a',metadata(),None,lambda p:p.write_bytes(b'synthetic'),subject='summary')
    with pytest.raises(ValueError,match='byte budget'):
        cache.download('dataset',tmp_path/'b',metadata(),8,lambda p:None,subject='summary')
    assert not (tmp_path/'b').exists()
    cache.close()


def test_oversized_response_stops_at_declared_size_without_reading_rest(tmp_path,monkeypatch):
    service=GalaxyService('https://example.invalid','secret','history',output_root=tmp_path)
    trace=ResultLoadTrace('job');trace.transfers=ResultTransfers(tmp_path,'history',service=service)
    class Response:
        def raise_for_status(self):pass
        def close(self):pass
        def iter_content(self,chunk_size):
            yield b'too many bytes'
            pytest.fail('Oversized response must stop before consuming additional body chunks')
    monkeypatch.setattr('radar_pd_nova.galaxy_service.requests.get',lambda *a,**k:Response())
    with trace.activate('complete'),subject('summary'),pytest.raises(ValueError,match='byte budget'):
        service._download_dataset('dataset',tmp_path/'result',metadata=metadata())
    assert not (tmp_path/'result').exists()
    trace.transfers.close()
