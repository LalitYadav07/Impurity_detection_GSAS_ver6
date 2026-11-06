"""Actual source-owner capture, latest revisions and bounded native dispatch."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from gpx_recipe.execution import write_json
from gpx_recipe.live.snapshot import parse_live_ascii
from gpx_recipe.symmetry.compiler import compile_native_bundle,file_sha
from gpx_recipe.symmetry.contracts import digest,validate_protocol
from radar_pd_nova.webmon.structural_stream import StructuralObservationStream,analyse_snapshot
from radar_pd_nova.webmon.stream import ReviewRequired
from test_structural_snapshot_transport import captured,RAW


def setup(tmp_path,count=1):
    path=tmp_path/'source.gpx';path.write_bytes(b'opaque public source')
    source=dict(id='public',source_path=str(path),source_sha256=file_sha(path),dataset_id='hda1')
    root,payload,protocol=captured(tmp_path,source);templates=[payload]
    if count==2:
        candidate=copy.deepcopy(protocol['candidates'][0]);candidate['id']='struct:second'
        protocol['candidates'].append(candidate);protocol['policy']['maximum_trials']=2
        inputs=root/'model-inputs'
        write_json(inputs/'catalog.json',dict(schema='radar-symmetry-catalog/v1',parent_setting_sha256='a'*64,
            k=protocol['k'],candidates=protocol['candidates']))
        protocol['catalog']['sha256']=file_sha(inputs/'catalog.json');protocol=validate_protocol(protocol)
        templates=[]
        for index,candidate in enumerate(protocol['candidates']):
            directory=root/('child-'+str(index))
            compile_native_bundle(protocol,inputs,directory,candidate_id=candidate['id'],seed_id='prepared',approved_hash=digest(protocol))
            templates.append(dict(copy.deepcopy(payload),directory=str(directory),source=str(directory/'source.gpx')))
    state=dict(raw=Path(payload['snapshot']).read_bytes(),metadata={},row=copy.deepcopy(payload['source_scope']),
        identity=copy.deepcopy(payload['mapping']['identity']))
    archive={}
    def capture(*args):
        raw=state['raw'];sha=hashlib.sha256(raw).hexdigest();archive[sha]=raw
        receipt=dict(content_sha256=sha,identity=copy.deepcopy(state['identity']),metadata=copy.deepcopy(state['metadata']))
        return copy.deepcopy(state['row']),receipt,parse_live_ascii(raw)
    session=SimpleNamespace(capture=capture,archive=SimpleNamespace(read=lambda receipt:archive[receipt['content_sha256']]))
    stream=StructuralObservationStream(session,'1','2',templates,root)
    return stream,state,source


def test_structural_capture_preserves_signed_data_and_admits_newly_valid_bins(tmp_path):
    stream,state,_=setup(tmp_path)
    state['raw']=state['raw'].replace(b'3,30,4',b'3,0,0')
    first=stream.capture();assert stream.capture() is None
    state['raw']=RAW
    second=stream.capture()
    assert first['revision']==1 and second['revision']==2
    assert first['id']!=second['id'] and second['payloads'][0]['snapshot']!=first['payloads'][0]['snapshot']
    assert Path(first['binding']['source_dat']).read_bytes().endswith(b'3,0,0\n')
    assert Path(second['binding']['source_dat']).read_bytes()==RAW
    assert second['payloads'][0]['mapping']==first['payloads'][0]['mapping']
    assert second['payloads'][0]['receipt']['content_sha256']==hashlib.sha256(RAW).hexdigest()
    assert second['payloads'][0]['revision']==2
    assert json.loads((stream.root/'structural-stream.json').read_text())['cumulative_policy']=='replace'


@pytest.mark.parametrize('change',['sample','bank','producer','epoch','grid','spectrum','exposure','exposure_semantics'])
def test_structural_capture_requires_new_review_for_changed_physical_scope(tmp_path,change):
    stream,state,_=setup(tmp_path);state['metadata']['cumulative_exposure']=10.;stream.capture()
    state['raw']=state['raw'].replace(b'-12',b'-13')
    if change=='sample':state['row']['sample_id']='another'
    if change=='bank':state['metadata']['physical_bank']=2
    if change=='producer':state['metadata']['producer_config_sha256']='d'*64
    if change=='epoch':state['identity']['epoch']='new producer'
    if change=='grid':state['raw']=state['raw'].replace(b'3,30,4',b'4,30,4')
    if change=='spectrum':state['raw']=state['raw'].replace(b'\n0\n',b'\n1\n')
    if change=='exposure':state['metadata']['cumulative_exposure']=9.
    if change=='exposure_semantics':state['metadata']['cumulative_exposure']=None
    with pytest.raises(ReviewRequired):stream.capture()
    assert stream.revision==1


def test_structural_capture_keeps_every_reviewed_model_and_repeated_curve_return(tmp_path):
    stream,state,_=setup(tmp_path,2);first=stream.capture()
    assert len(first['payloads'])==2
    state['raw']=RAW;second=stream.capture()
    state['raw']=Path(first['binding']['source_dat']).read_bytes();third=stream.capture()
    assert third['revision']==3 and third['id']!=first['id']
    assert third['receipt']['content_sha256']==first['receipt']['content_sha256']
    assert third['payloads'][0]['destination']!=first['payloads'][0]['destination']
    assert second['protocol_sha256']==first['protocol_sha256']


def adapter_for(stream,*,failure=None):
    events=[];lock=threading.Lock();active=[0];peak=[0]
    def submit(payload,root,**kwargs):
        candidate=Path(payload['directory']).name
        handle=dict(key=digest(candidate)[:24],submission_started=True,candidate=candidate)
        if kwargs.get('on_intent'):kwargs['on_intent'](handle)
        with lock:active[0]+=1;peak[0]=max(peak[0],active[0]);events.append(candidate)
        if failure=='uncertain':raise TimeoutError('acknowledgment lost')
        return handle
    def poll(handle):
        with lock:active[0]-=1
        return dict(state='failed' if failure=='known' and handle['candidate'].endswith('0') else 'finished',handle=handle)
    def collect(handle,root):
        return dict(result=dict(metrics=dict(rwp=2.,rwp_native=2.)),view='public-view.json'),1.
    adapter=SimpleNamespace(runtime=SimpleNamespace(max_active_jobs=2,timeout=3),submit=submit,poll=poll,collect=collect)
    return adapter,events,peak


def test_structural_portfolio_uses_bounded_existing_jobs_and_retains_each_outcome(tmp_path):
    stream,_,source=setup(tmp_path,2);item=stream.capture();adapter,events,peak=adapter_for(stream,failure='known')
    progress=[]
    rows=asyncio.run(analyse_snapshot(adapter,item,stream.root,source=source,on_progress=progress.append,poll_seconds=.001))
    assert len(rows)==2 and len(events)==2 and peak[0]<=2
    assert {r['state'] for r in rows}=={'verified_point_fit','failed'}
    assert len(progress[-1])==2
    saved=json.loads((Path(item['folder'])/'progress.json').read_text())
    assert saved['terminal']==saved['total']==2
    assert len(list(Path(item['folder']).glob('handle-*.json')))==2


def test_uncertain_dispatch_stops_the_snapshot_queue_without_automatic_retry(tmp_path):
    stream,_,source=setup(tmp_path);item=stream.capture();adapter,events,_=adapter_for(stream,failure='uncertain')
    with pytest.raises(ReviewRequired,match='unresolved'):
        asyncio.run(analyse_snapshot(adapter,item,stream.root,source=source,poll_seconds=.001))
    assert len(events)==1
    saved=json.loads((Path(item['folder'])/'progress.json').read_text())
    assert saved['rows'][0]['state']=='needs_recovery' and saved['terminal']==0


def test_pause_before_admission_never_submits_any_structural_job(tmp_path):
    stream,_,source=setup(tmp_path,2);item=stream.capture();adapter,events,_=adapter_for(stream)
    cancel=threading.Event();cancel.set()
    rows=asyncio.run(analyse_snapshot(adapter,item,stream.root,source=source,cancel=cancel,poll_seconds=.001))
    assert not events and all(r['state']=='not_started' for r in rows)
