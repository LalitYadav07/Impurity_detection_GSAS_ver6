"""Actual-snapshot ranking orders the full Live portfolio and reuses dispatch."""
import asyncio
import copy
import json
from pathlib import Path
import threading

import pytest

from gpx_recipe.execution import write_json
from gpx_recipe.preparation.structural_ranking_result import unavailable
from gpx_recipe.structural_live.acquisition import _mapping
from gpx_recipe.structural_live.job import reviewed_job
from gpx_recipe.symmetry.compiler import file_sha,package_signature
from gpx_recipe.symmetry.contracts import digest
from radar_pd_nova.webmon.structural_runtime import StructuralSnapshotRuntime
from radar_pd_nova.webmon.structural_stream import analyse_snapshot
from radar_pd_nova.webmon.stream import ReviewRequired
from test_structural_stream import setup,adapter_for,RAW
from test_structural_snapshot_transport import captured
from test_model_runtime import remote


def ranking_result(payload,*,available=True):
    _,protocol,_=reviewed_job(payload['directory'])
    observation=dict(snapshot_sha256=payload['receipt']['content_sha256'],calculation_source_sha256='e'*64,
        acquisition_sha256=digest([_mapping(payload['mapping']),payload['source_scope']]),revision=payload['revision'])
    evidence=unavailable(protocol,'Public transport fixture',snapshot='d'*64,observation=observation)
    if available:
        order=[c['id'] for c in reversed(protocol['candidates'])]
        evidence.update(ranking_available=True,candidate_order=order,entry_id='f'*24,bank='bank 1',
            checks=[dict(entry='f'*24,status='compatible',reason='Transport fixture; no model accuracy claim')],
            families=[dict(space_group=63,model_score=1.,candidate_ids=order)],
            support=dict(status='within_empirical_synthetic_support',nearest_training_rms=.1,
                validation_maximum_rms=.2,statistical_coverage_qualified=False))
    return dict(schema='radar-structural-snapshot-ranking-result/v1',kind='structural_ranking',
        protocol_sha256=digest(protocol),implementation_sha256=package_signature(),source_sha256=protocol['source']['sha256'],
        observation=observation,ranking=evidence,ranking_sha256='b'*64,output=payload['destination'],
        artifacts=['ranking/ranking.json','transfer.json','result.json'],seconds=1.,
        candidate_pruning_authorized=False,scientific_result=False)


def ranking_adapter(stream,*,problem=None,available=True):
    adapter,events,peak=adapter_for(stream);payloads={};rank_events=[];poll=adapter.poll
    adapter.has_prepared_ranking=lambda payload:True
    def submit(payload,root,**kwargs):
        rank_events.append('ranking');handle=dict(key='a'*24,submission_started=True,kind='ranking')
        kwargs['on_intent'](handle);payloads[handle['key']]=payload
        if problem=='ack':raise TimeoutError('Ranking acknowledgement lost')
        return handle
    def ranked_poll(handle):
        if handle.get('kind')=='ranking':
            return dict(state='failed' if problem=='known' else 'finished',handle=handle)
        return poll(handle)
    def collect(handle,root):
        if problem=='collect':raise OSError('Ranking download interrupted')
        result=ranking_result(payloads[handle['key']],available=available)
        if problem=='old_scan':result['observation']['revision']+=1
        return result,1.
    adapter.submit_ranking=submit;adapter.collect_ranking=collect;adapter.poll=ranked_poll
    return adapter,events,rank_events,peak


@pytest.mark.parametrize('available',[True,False])
def test_one_current_ranking_orders_without_dropping_any_native_model(tmp_path,available):
    stream,state,source=setup(tmp_path,2);item=stream.capture()
    adapter,events,ranks,peak=ranking_adapter(stream,available=available);progress=[]
    rows=asyncio.run(analyse_snapshot(adapter,item,stream.root,source=source,
        on_ranking_progress=progress.append,poll_seconds=.001))
    assert ranks==['ranking'] and len(events)==2 and peak[0]<=2
    assert events==(['child-1','child-0'] if available else ['child-0','child-1'])
    assert [r['priority'] for r in rows]==[1,2]
    assert all(r['state']=='verified_point_fit' for r in rows)
    assert progress[-1]['state']=='ranked'
    previous=json.loads((Path(item['folder'])/'ranking-evidence.json').read_text())
    state['raw']=RAW;later=stream.capture()
    asyncio.run(analyse_snapshot(adapter,later,stream.root,source=source,poll_seconds=.001))
    current=json.loads((Path(later['folder'])/'ranking-evidence.json').read_text())
    assert current['observation']['revision']!=previous['observation']['revision']
    assert current['observation']['snapshot_sha256']!=previous['observation']['snapshot_sha256']


@pytest.mark.parametrize('problem',['ack','collect','old_scan'])
def test_unresolved_or_wrong_ranking_retains_handle_and_never_dispatches_fits(tmp_path,problem):
    stream,_,source=setup(tmp_path,2);item=stream.capture()
    adapter,events,ranks,_=ranking_adapter(stream,problem=problem)
    with pytest.raises(ReviewRequired,match='ranking dispatch is unresolved'):
        asyncio.run(analyse_snapshot(adapter,item,stream.root,source=source,poll_seconds=.001))
    assert ranks==['ranking'] and not events
    progress=json.loads((Path(item['folder'])/'progress.json').read_text())
    assert progress['ranking']['state']=='needs_recovery' and progress['terminal']==0
    assert (Path(item['folder'])/'ranking-handle.json').is_file()


def test_known_ranking_failure_uses_complete_native_portfolio(tmp_path):
    stream,_,source=setup(tmp_path,2);item=stream.capture()
    adapter,events,ranks,_=ranking_adapter(stream,problem='known')
    rows=asyncio.run(analyse_snapshot(adapter,item,stream.root,source=source,poll_seconds=.001))
    assert ranks==['ranking'] and events==['child-0','child-1']
    assert len(rows)==2 and all(r['state']=='verified_point_fit' for r in rows)


@pytest.mark.parametrize('pause',['before','after_ranking'])
def test_pausing_ranked_stream_never_admits_new_native_fits(tmp_path,pause):
    stream,_,source=setup(tmp_path,2);item=stream.capture()
    adapter,events,ranks,_=ranking_adapter(stream);cancel=threading.Event()
    if pause=='before':cancel.set()
    def progress(row):
        if pause=='after_ranking' and row['state']=='ranked':cancel.set()
    rows=asyncio.run(analyse_snapshot(adapter,item,stream.root,source=source,cancel=cancel,
        on_ranking_progress=progress,poll_seconds=.001))
    assert not events and len(ranks)==int(pause=='after_ranking')
    assert all(r['state']=='not_started' for r in rows)


def test_zero_allocation_and_incomplete_portfolio_fail_before_ranking_post(tmp_path):
    stream,_,source=setup(tmp_path,2);item=stream.capture()
    adapter,events,ranks,_=ranking_adapter(stream);adapter.runtime.max_active_jobs=0
    with pytest.raises(ValueError,match='positive structural job allocation'):
        asyncio.run(analyse_snapshot(adapter,item,stream.root,source=source))
    adapter.runtime.max_active_jobs=2;item['payloads'].pop()
    with pytest.raises(ValueError,match='retain every reviewed model'):
        asyncio.run(analyse_snapshot(adapter,item,stream.root,source=source))
    assert not events and not ranks


def test_rank_only_uses_same_durable_transport_and_recovers_finished_job(remote,tmp_path,monkeypatch):
    runtime,service,source=remote;root,payload,_=captured(tmp_path,source)
    folder=root/'revision';folder.mkdir()
    item=dict(payloads=[payload],revision=payload['revision'],receipt=payload['receipt'],
        captured_monotonic=0.,folder=str(folder))
    adapter=StructuralSnapshotRuntime(runtime);monkeypatch.setattr(adapter,'has_prepared_ranking',lambda payload:True)
    collections=[]
    def collect(handle,root):
        collections.append(handle['key'])
        if len(collections)==1:raise OSError('Result download interrupted')
        actual=copy.deepcopy(payload);actual['destination']=str(folder/'ranking-output')
        return ranking_result(actual),1.
    monkeypatch.setattr(adapter,'collect_ranking',collect)
    monkeypatch.setattr(adapter,'collect',lambda *a:(dict(result=dict(metrics=dict(rwp_native=2.))),1.))
    with pytest.raises(ReviewRequired,match='ranking dispatch is unresolved'):
        asyncio.run(analyse_snapshot(adapter,item,root,source=source,poll_seconds=.001))
    assert len(service.submissions)==1
    rows=asyncio.run(analyse_snapshot(adapter,item,root,source=source,poll_seconds=.001))
    assert len(service.submissions)==2 and collections[0]==collections[1]
    assert rows[0]['state']=='verified_point_fit'


@pytest.mark.parametrize('change',[None,'revision','acquisition','source','implementation','drop','artifact','output','observation','seconds'])
def test_rank_collection_binds_every_receipt_to_captured_bytes(remote,tmp_path,monkeypatch,change):
    runtime,service,source=remote;root,payload,_=captured(tmp_path,source)
    service.data[source['dataset_id']]=Path(source['source_path']).read_bytes()
    adapter=StructuralSnapshotRuntime(runtime);handle=adapter.submit_ranking(payload,root,source=source,timeout=15)
    assert handle['request']['operation']=='structural-snapshot-ranking'
    assert set(handle['request']['input_hashes'])=={'model/job.json','model/source.gpx','model/reference.gpx',
        'model/catalog.json','model/child.cif','anchor.dat','current.dat'}
    assert adapter.submit_ranking(payload,root,source=source,timeout=15)==handle and len(service.submissions)==1
    result=ranking_result(payload);path=root/'output/ranking/ranking.json';path.parent.mkdir(parents=True)
    write_json(path,result['ranking']);result['ranking_sha256']=file_sha(path)
    if change=='revision':result['observation']['revision']+=1
    if change=='acquisition':result['observation']['acquisition_sha256']='0'*64
    if change=='source':result['source_sha256']='0'*64
    if change=='implementation':result['implementation_sha256']='0'*64
    if change=='drop':result['ranking']['candidate_order']=[]
    if change=='artifact':path.write_text('{}')
    if change=='output':
        (root/'elsewhere').mkdir();result['output']=str(root/'elsewhere')
    if change=='observation':result['ranking']['observation']['snapshot_sha256']='0'*64
    if change=='seconds':result['seconds']=float('nan')
    monkeypatch.setattr(runtime,'collect_native',lambda *a:(result,1.))
    if change is None:assert adapter.collect_ranking(handle,root)==(result,1.)
    else:
        with pytest.raises(ValueError):adapter.collect_ranking(handle,root)
