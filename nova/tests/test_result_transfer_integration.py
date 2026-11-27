"""Synthetic monitor-profile publication exercises both real collectors."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import io
import json
import threading
import time

import numpy as np

from radar_pd_nova.galaxy_service import GalaxyService
from radar_pd_nova.models import ResultStatus
from radar_pd_nova.result_loading import ResultLoadTrace
from radar_pd_nova.result_preview import collect_result_preview
from radar_pd_nova.result_transfers import ResultTransfers
from test_result_preview import published


def run_collectors(published, monkeypatch, shared):
    p=published
    p.record.output_dataset_ids={k:v for k,v in p.record.output_dataset_ids.items()
                                 if k in {'summary','pattern_observation','plots','gpx_projects'}}
    p.job['outputs']={k:{'id':v} for k,v in p.record.output_dataset_ids.items()
                      if k in {'summary','pattern_observation'}}
    p.job['output_collections']={k:{'id':v} for k,v in p.record.output_dataset_ids.items()
                                 if k in {'plots','gpx_projects'}}
    # Multi-hundred-KB companions contain only generated random numbers. The
    # extra array is ignored by the renderer, but is still checked by the NPZ
    # size/type guard and exercises the actual file-transfer cost.
    for index in (0,1):
        identifier=f'plot{index}-npz'
        with np.load(io.BytesIO(p.data[identifier])) as original:
            arrays={name:original[name] for name in original.files}
        arrays['synthetic_padding']=np.random.default_rng(index).normal(size=40000)
        buffer=io.BytesIO();np.savez_compressed(buffer,**arrays)
        p.data[identifier]=buffer.getvalue();p.metadata[identifier]['file_size']=len(p.data[identifier])
    p.data['native']=b'opaque synthetic native project'
    p.metadata['native']=dict(id='native',history_id='history',state='ok',file_size=len(p.data['native']),file_ext='binary')
    projects=dict(id='projects',collection_type='list',populated_state='ok',element_count=1,
        elements=[dict(element_type='hda',element_identifier='Final_model',object=p.metadata['native'])])
    p.service._download_dataset=GalaxyService._download_dataset.__get__(p.service)
    calls=Counter();lock=threading.Lock();active=0;maximum=0
    class Response:
        def __init__(self,raw,document=None):self.raw=raw;self.document=document
        def raise_for_status(self):pass
        def close(self):pass
        def json(self):return deepcopy(self.document)
        def iter_content(self,chunk_size):
            nonlocal active,maximum
            with lock:
                active+=1;maximum=max(maximum,active)
            try:
                time.sleep(.003+len(self.raw)/2_000_000)
                yield self.raw
            finally:
                with lock:active-=1
    def get(url,**kwargs):
        if '/api/dataset_collections/' in url:
            document=p.collection if url.endswith('/plots') else projects
            return Response(json.dumps(document).encode(),document)
        identifier=url.split('/contents/')[1].split('/')[0]
        with lock:calls[identifier]+=1
        time.sleep(.003)
        return Response(p.data[identifier])
    monkeypatch.setattr('radar_pd_nova.galaxy_service.requests.get',get)
    trace=ResultLoadTrace('job')
    if shared:trace.transfers=ResultTransfers(p.service.output_root,'history')
    def complete():
        with trace.activate('complete'):
            return p.service.collect_results_for_display(p.record.model_copy(deep=True))
    def preview():
        with trace.activate('preview'):
            return collect_result_preview(p.service,p.record.model_copy(deep=True))
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        full=pool.submit(complete);partial=pool.submit(preview)
        result,small=full.result(),partial.result()
    elapsed=time.monotonic()-start
    assert result.result_status==ResultStatus.READY
    assert small is not None and not p.record.output_dir
    assert (result.local_output_dir/'gpx_projects/Final_model.gpx').read_bytes()==p.data['native']
    assert not list(small.root.rglob('*.gpx'))
    report=dict(shared=shared,elapsed_seconds=elapsed,requests=dict(calls),
                body_bytes=sum(len(p.data[key])*count for key,count in calls.items()),
                max_active_body_transfers=maximum)
    assert 1 < maximum <= 4
    if trace.transfers:trace.transfers.close()
    return report


def test_real_preview_and_canonical_collectors_reuse_only_identical_non_native_files(published,monkeypatch):
    report=run_collectors(published,monkeypatch,True)
    assert report['requests']['native']==1
    assert report['requests']['summary']==report['requests']['observation']==1
    assert report['requests']['plot0-json']==report['requests']['plot1-json']==1
    assert report['requests']['plot1-npz']==1
    print(json.dumps(report,sort_keys=True))


def test_unshared_control_demonstrates_duplicate_body_reads(published,monkeypatch):
    report=run_collectors(published,monkeypatch,False)
    assert report['requests']['native']==1
    assert report['requests']['summary']==report['requests']['observation']==2
    assert report['requests']['plot0-json']==report['requests']['plot1-json']==2
    assert report['requests']['plot1-npz']==2
    print(json.dumps(report,sort_keys=True))
