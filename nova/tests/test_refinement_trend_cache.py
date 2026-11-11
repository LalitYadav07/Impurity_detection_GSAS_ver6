"""Compact plot records retain native evidence and exact artifact ownership."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import threading

import pytest

from radar_pd_nova import refinement_controller as module
from radar_pd_nova.refinement_controller import RefinementMixin
from radar_pd_nova.refinement_plots import same_fit_domain
from test_refinement_plots import view


def saved(root,value=4.):
    root.mkdir()
    full=view();full['parameters'][0]['value']=value
    path=root/'view.json';path.write_text(json.dumps(full))
    return {'result':{'directory':str(root),'files':{'view.json':hashlib.sha256(path.read_bytes()).hexdigest()}}},full


class App(RefinementMixin):
    def __init__(self):
        self._refinement_view_cache={};self.full_reads=0
    def _read_refinement_view(self,*args):
        self.full_reads+=1
        return super()._read_refinement_view(*args)


def test_repeated_trend_requests_reuse_only_compact_verified_metadata(tmp_path):
    job,full=saved(tmp_path/'first');app=App()
    before=deepcopy(full)
    compact=app._read_refinement_trend_view(job,'view.json')
    assert app._read_refinement_trend_view(job,'view.json') is compact
    assert app.full_reads==1 and compact['parameters']==full['parameters']
    assert same_fit_domain(compact['histograms'][0],full['histograms'][0])
    assert not {'x','d','observed','calculated','weights','fitted','valid'} & compact['histograms'][0].keys()
    assert json.loads((tmp_path/'first/view.json').read_text())==before


def test_changed_artifact_is_rejected_instead_of_serving_cached_trend(tmp_path):
    job,_=saved(tmp_path/'first');app=App()
    app._read_refinement_trend_view(job,'view.json')
    path=tmp_path/'first/view.json';original=path.stat()
    changed=view();changed['parameters'][0]['value']=9.
    path.write_text(json.dumps(changed))
    os.utime(path,ns=(original.st_atime_ns,original.st_mtime_ns+1_000_000))
    with pytest.raises(ValueError,match='changed'):
        app._read_refinement_trend_view(job,'view.json')


def test_recovered_revision_and_other_directory_never_reuse_old_parameters(tmp_path):
    job,_=saved(tmp_path/'first');app=App()
    first=app._read_refinement_trend_view(job,'view.json')
    second_job,_=saved(tmp_path/'recovered',9.)
    second=app._read_refinement_trend_view(second_job,'view.json')
    assert first['parameters'][0]['value']==4. and second['parameters'][0]['value']==9.
    job['result'].update(second_job['result'])
    assert app._read_refinement_trend_view(job,'view.json') is second


def test_cache_is_lru_and_bounded_by_entry_count(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'REFINEMENT_TREND_CACHE_ENTRIES',2)
    jobs=[saved(tmp_path/str(i),4.+i)[0] for i in range(3)];app=App()
    first=app._read_refinement_trend_view(jobs[0],'view.json')
    second=app._read_refinement_trend_view(jobs[1],'view.json')
    assert app._read_refinement_trend_view(jobs[0],'view.json') is first
    app._read_refinement_trend_view(jobs[2],'view.json')
    assert len(app._refinement_trend_cache)==2
    assert app._read_refinement_trend_view(jobs[1],'view.json') is not second
    assert app._refinement_trend_cache_bytes==sum(size for _,size in app._refinement_trend_cache.values())


def test_oversized_compact_record_is_returned_without_breaching_cache_bound(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'REFINEMENT_TREND_CACHE_BYTES',1)
    job,_=saved(tmp_path/'first');app=App()
    assert app._read_refinement_trend_view(job,'view.json')['parameters'][0]['value']==4.
    assert app._refinement_trend_cache=={} and app._refinement_trend_cache_bytes==0


def test_cache_evicts_by_metadata_size_even_below_entry_limit(tmp_path,monkeypatch):
    jobs=[saved(tmp_path/str(i),4.+i)[0] for i in range(2)];app=App()
    app._read_refinement_trend_view(jobs[0],'view.json')
    one_size=app._refinement_trend_cache_bytes
    monkeypatch.setattr(module,'REFINEMENT_TREND_CACHE_BYTES',one_size+10)
    app._read_refinement_trend_view(jobs[1],'view.json')
    assert len(app._refinement_trend_cache)==1
    assert app._refinement_trend_cache_bytes<=one_size+10


def test_artifact_change_during_projection_is_not_cached(tmp_path):
    job,_=saved(tmp_path/'first');app=App();read=app._read_refinement_view
    def changed(*args):
        full=read(*args)
        path=tmp_path/'first/view.json';stamp=path.stat()
        os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns+1_000_000))
        return full
    app._read_refinement_view=changed
    with pytest.raises(ValueError,match='changed during trend preparation'):
        app._read_refinement_trend_view(job,'view.json')
    assert app._refinement_trend_cache=={}


def test_compact_cache_cannot_read_outside_the_published_result_directory(tmp_path):
    job,_=saved(tmp_path/'first');outside=tmp_path/'outside.json';outside.write_text('{}')
    job['result']['files']['../outside.json']=hashlib.sha256(outside.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='verified refinement receipt'):
        App()._read_refinement_trend_view(job,'../outside.json')


def test_inflight_old_read_cannot_fill_a_reset_workspace_cache(tmp_path,monkeypatch):
    from radar_pd_nova import refinement_sources
    job,_=saved(tmp_path/'first');app=App()
    entered=threading.Event();release=threading.Event()
    sha=refinement_sources.file_sha
    def delayed(path):
        entered.set()
        assert release.wait(5)
        return sha(path)
    monkeypatch.setattr(refinement_sources,'file_sha',delayed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending=pool.submit(app._read_refinement_view,job,'view.json')
        assert entered.wait(5)
        app._refinement_view_cache={}
        release.set()
        assert pending.result(timeout=5)['parameters'][0]['value']==4.
    assert app._refinement_view_cache=={}
