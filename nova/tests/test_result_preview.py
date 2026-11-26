"""A bounded same-job fit is available without a complete/native result cache."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import io
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from radar_pd_nova import result_preview as preview
from radar_pd_nova.galaxy_service import GalaxyService
from radar_pd_nova.models import AnalysisMode, RunRecord, RunStatus
from radar_pd_nova.tool_ids import tool_id


def json_bytes(value):
    return json.dumps(value).encode()


@pytest.fixture
def published(tmp_path, monkeypatch):
    service = GalaxyService('https://example.invalid', 'synthetic', 'history', output_root=tmp_path)
    record = RunRecord(uid='job', galaxy_job_id='job', history_id='history', name='Synthetic fit',
        mode=AnalysisMode.FULL, status=RunStatus.OK, output_dataset_ids={
            'summary':'summary', 'plots':'plots', 'pattern_observation':'observation',
            'gpx_projects':'projects', 'results_archive':'archive', 'phases':'cifs'})
    x=list(map(float,range(1,17)));y=[2.+(i%3) for i in range(16)]
    calculated=[v-.1 for v in y]
    packet = {'$schema':'radar-pd-pattern-observation/v1', 'x':x, 'y':y,
        'sigma':[1.]*16, 'valid':[True]*16, 'fit_valid':[True]*16,
        'ycalc':calculated, 'coordinate':{'kind':'tof','unit':'us'},
        'source_gpx_sha256':'b'*64, 'context':{'source_product_sha256':'a'*64}}
    data = {'observation':json_bytes(packet)}
    summary = {'$schema':'radar-pd-result/v1', 'analysis_mode':'full', 'phases':[],
        'pattern_observation':{'name':'Pattern_observation.json',
            'sha256':hashlib.sha256(data['observation']).hexdigest(), 'source_gpx_sha256':'b'*64},
        'observables':{'source_gpx_sha256':'b'*64}, 'provenance':{'input':{'source_sha256':'a'*64}}}
    data['summary'] = json_bytes(summary)
    metadata = {}
    elements = []
    for index, name in enumerate(('Accepted_fit_after_pass_4.png', 'Results__Plots__seq_final_joint_accepted_model.png')):
        payload = {'plot_kind':'gsas_fit_with_ticks_v1', 'rwp':2. if index==0 else 5.,
            'arrays_npz':name+'.plotdata_arrays.npz', 'source_plot':name,
            'fit_limits':[1.,16.], 'excluded_ranges':[]}
        arrays = io.BytesIO()
        np.savez_compressed(arrays, x=x, yobs=y, ycalc=calculated, resid=[.1]*16)
        for suffix, raw in (('.plotdata.json',json_bytes(payload)), ('.plotdata_arrays.npz',arrays.getvalue()), ('',b'synthetic-png')):
            filename = name+suffix
            identifier = f'plot{index}-'+('json' if suffix.endswith('json') else 'npz' if suffix.endswith('npz') else 'png')
            data[identifier] = raw
            obj = dict(id=identifier, history_id='history', state='ok', file_size=len(raw), file_ext=Path(filename).suffix[1:])
            metadata[identifier] = obj
            elements.append(dict(element_type='hda', element_identifier=filename.rsplit('.',1)[0], object=obj))
    for identifier in ('summary','observation'):
        metadata[identifier] = dict(id=identifier, history_id='history', state='ok', file_size=len(data[identifier]), file_ext='json')
    collection = dict(id='plots', collection_type='list', populated_state='ok', element_count=len(elements), elements=elements)
    job = dict(id='job', history_id='history', state='ok', tool_id=tool_id('analyze'),
        outputs={key:dict(id=value) for key,value in record.output_dataset_ids.items() if key not in {'plots','projects','cifs'}},
        output_collections={key:dict(id=value) for key,value in record.output_dataset_ids.items() if key in {'plots','projects','cifs'}})
    calls = []
    service._job_details = lambda _:deepcopy(job)
    service.job_output_ids = lambda _:dict(record.output_dataset_ids)
    service._dataset_metadata = lambda identifier:deepcopy(metadata[identifier])
    def download(identifier, target, *, maximum_bytes=None, metadata=None):
        calls.append(identifier)
        assert maximum_bytes is not None and metadata['id']==identifier
        raw = data[identifier]
        if len(raw)>maximum_bytes:raise ValueError('synthetic byte limit')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
    service._download_dataset = download
    class Response:
        def raise_for_status(self):pass
        def iter_content(self,chunk_size):yield json_bytes(collection)
        def close(self):pass
    def get(url, **kwargs):
        assert url=='https://example.invalid/api/dataset_collections/plots'
        assert kwargs['params']=={'view':'element'} and kwargs['allow_redirects'] is False
        return Response()
    monkeypatch.setattr(preview.requests,'get',get)
    return SimpleNamespace(service=service,record=record,data=data,metadata=metadata,
        collection=collection,job=job,calls=calls,summary=summary,download=download)


def test_preview_reads_final_joint_fit_without_archive_or_native_content(published):
    p=published; before=p.record.model_dump(mode='json')
    result=preview.collect_result_preview(p.service,p.record)
    assert isinstance(result,preview.ResultPreview)
    assert result.root.is_relative_to(p.service.output_root/'result-previews/job')
    assert p.record.model_dump(mode='json')==before and not p.record.output_dir
    assert not (p.service.output_root/'runs').exists()
    assert sorted(p.calls)==['observation','plot0-json','plot1-json','plot1-npz','summary']
    receipt=json.loads((result.root/preview.PREVIEW_RECEIPT).read_text())
    assert receipt['schema']=='radar-pd-result-preview/v1' and receipt['job_id']=='job'
    assert receipt['history_id']=='history' and 'seq_final_joint' in receipt['selected_plot']
    assert receipt['output_dataset_ids']==dict(result.output_dataset_ids)
    assert len(list(result.root.rglob('*.plotdata.json')))==1
    assert json.loads((result.root/'plot_observation_binding.json').read_text())['status']=='verified'
    assert not (result.root/'.radar-pd-cache.json').exists()
    assert not list(result.root.rglob('*.gpx')) and not list(result.root.rglob('*.cif'))
    with pytest.raises(FrozenInstanceError):result.root=Path('elsewhere')


def test_preview_uses_existing_sidecar_rwp_tiebreaker(published):
    p=published
    for index,element in enumerate(p.collection['elements']):
        old=element['element_identifier']
        element['element_identifier']=old.replace('Accepted_fit_after_pass_4','seq_final_model_a').replace('Results__Plots__seq_final_joint_accepted_model','seq_final_model_b')
        identifier=element['object']['id']
        if identifier.endswith('json'):
            payload=json.loads(p.data[identifier])
            payload['source_plot']=payload['source_plot'].replace('Accepted_fit_after_pass_4','seq_final_model_a').replace('Results__Plots__seq_final_joint_accepted_model','seq_final_model_b')
            payload['arrays_npz']=payload['source_plot']+'.plotdata_arrays.npz'
            p.data[identifier]=json_bytes(payload);element['object']['file_size']=len(p.data[identifier])
    result=preview.collect_result_preview(p.service,p.record)
    assert 'seq_final_model_a' in json.loads((result.root/preview.PREVIEW_RECEIPT).read_text())['selected_plot']
    assert 'plot0-npz' in p.calls and 'plot1-npz' not in p.calls


@pytest.mark.parametrize('mutation', ['job','history','tool','output','dataset','dataset_history','member_history'])
def test_preview_rejects_changed_source_identity_before_publication(published,mutation):
    p=published
    if mutation=='job':p.job['id']='other'
    elif mutation=='history':p.job['history_id']='other'
    elif mutation=='tool':p.job['tool_id']='other'
    elif mutation=='output':p.job['outputs']['summary']['id']='changed'
    elif mutation=='dataset':p.metadata['summary']['id']='other'
    elif mutation=='dataset_history':p.metadata['summary']['history_id']='other'
    else:p.collection['elements'][0]['object']['history_id']='other'
    with pytest.raises(ValueError,match='identity|History|Analyze'):
        preview.collect_result_preview(p.service,p.record)
    assert not list(p.service.output_root.glob('result-previews/job/*'))


@pytest.mark.parametrize('mutation', ['duplicate','filename','extension','companion','wrong_npz_history',
                                    'collection_id','missing_collection_id'])
def test_preview_rejects_ambiguous_or_unsafe_collection_members(published,mutation):
    p=published
    if mutation=='duplicate':
        p.collection['elements'].append(deepcopy(p.collection['elements'][0]));p.collection['element_count']+=1
    elif mutation=='filename':p.collection['elements'][0]['element_identifier']='../escape'
    elif mutation=='extension':p.collection['elements'][0]['object']['file_ext']='../json'
    elif mutation=='wrong_npz_history':p.collection['elements'][4]['object']['history_id']='other'
    elif mutation=='collection_id':p.collection['id']='another'
    elif mutation=='missing_collection_id':p.collection.pop('id')
    else:
        payload=json.loads(p.data['plot1-json']);payload['arrays_npz']='../escape.npz'
        p.data['plot1-json']=json_bytes(payload);p.metadata['plot1-json']['file_size']=len(p.data['plot1-json'])
    with pytest.raises(ValueError):preview.collect_result_preview(p.service,p.record)
    assert not list(p.service.output_root.glob('result-previews/job/*'))


@pytest.mark.parametrize('limit', ['MAX_SIDECARS','MAX_SIDECAR_TOTAL','MAX_SUMMARY_BYTES','MAX_PREVIEW_BYTES'])
def test_preview_budgets_fall_back_without_an_incomplete_cache(published,monkeypatch,limit):
    p=published;monkeypatch.setattr(preview,limit,1)
    assert preview.collect_result_preview(p.service,p.record) is None
    assert not list(p.service.output_root.glob('result-previews/job/*'))
    assert not p.record.output_dir


@pytest.mark.parametrize('kind', ['incomplete','nested','old_summary','missing_plots','running'])
def test_unsupported_publication_uses_complete_loader(published,kind):
    p=published
    if kind=='incomplete':p.collection['element_count']+=1
    elif kind=='nested':p.collection['collection_type']='list:list'
    elif kind=='old_summary':
        p.data['summary']=b'{}';p.metadata['summary']['file_size']=2
    elif kind=='missing_plots':
        p.record.output_dataset_ids.pop('plots');p.job['output_collections'].pop('plots')
    else:p.job['state']='running'
    assert preview.collect_result_preview(p.service,p.record) is None
    assert not list(p.service.output_root.glob('result-previews/job/*'))


def test_missing_observation_never_invents_verified_validity(published):
    p=published
    p.record.output_dataset_ids.pop('pattern_observation');p.job['outputs'].pop('pattern_observation')
    result=preview.collect_result_preview(p.service,p.record)
    binding=json.loads((result.root/'plot_observation_binding.json').read_text())
    assert binding['status']=='unavailable' and 'observation' not in p.calls


def test_changed_observation_checksum_remains_unverified(published):
    p=published
    packet=json.loads(p.data['observation']);packet['y'][1]=42.
    p.data['observation']=json_bytes(packet);p.metadata['observation']['file_size']=len(p.data['observation'])
    result=preview.collect_result_preview(p.service,p.record)
    assert json.loads((result.root/'plot_observation_binding.json').read_text())['status']=='unavailable'


def test_corrupt_selected_npz_uses_same_plot_static_fallback(published):
    p=published;p.data['plot1-npz']=b'not an npz';p.metadata['plot1-npz']['file_size']=len(p.data['plot1-npz'])
    result=preview.collect_result_preview(p.service,p.record)
    assert result is not None and 'plot1-png' in p.calls and 'plot0-npz' not in p.calls
    assert not list(result.root.rglob('*seq_final_joint*.npz'))


def test_transfer_failure_removes_only_its_private_preview(published):
    p=published
    existing=p.service.output_root/'runs/job';existing.mkdir(parents=True);(existing/'summary.json').write_text('old complete result')
    def broken(identifier,*args,**kwargs):
        if identifier=='plot1-npz':raise OSError('synthetic content failure')
        return p.download(identifier,*args,**kwargs)
    p.service._download_dataset=broken
    with pytest.raises(OSError,match='synthetic content failure'):preview.collect_result_preview(p.service,p.record)
    assert (existing/'summary.json').read_text()=='old complete result'
    assert not list(p.service.output_root.glob('result-previews/job/*'))


def test_retained_record_ids_need_authoritative_job_associations(published):
    p=published;p.job['outputs']={};p.job['output_collections']={}
    p.service.job_output_ids=lambda _:{}
    assert preview.collect_result_preview(p.service,p.record) is None
    assert p.calls==[]


def test_fallback_reuses_full_primary_policy_after_a_corrupt_final_fit(published):
    p=published
    additions=[]
    for element in deepcopy(p.collection['elements'][:3]):
        original=element['object']['id'];new=original.replace('plot0','extra')
        element['object']['id']=new
        element['element_identifier']=element['element_identifier'].replace('after_pass_4','after_pass_1')
        raw=p.data[original]
        if original.endswith('json'):
            payload=json.loads(raw)
            payload['source_plot']=payload['source_plot'].replace('after_pass_4','after_pass_1')
            payload['arrays_npz']=payload['arrays_npz'].replace('after_pass_4','after_pass_1')
            raw=json_bytes(payload)
        p.data[new]=raw;element['object']['file_size']=len(raw);p.metadata[new]=element['object']
        additions.append(element)
    p.collection['elements']=additions+[e for e in p.collection['elements'] if e['object']['id']!='plot1-png']
    p.collection['element_count']=len(p.collection['elements'])
    p.data['plot1-npz']=b'bad npz';p.metadata['plot1-npz']['file_size']=len(p.data['plot1-npz'])
    result=preview.collect_result_preview(p.service,p.record)
    receipt=json.loads((result.root/preview.PREVIEW_RECEIPT).read_text())
    assert 'after_pass_4' in receipt['selected_plot']
    assert 'plot0-npz' in p.calls and 'extra-npz' not in p.calls


def test_parallel_previews_share_the_existing_four_transfer_bound(published):
    p=published;lock=threading.Lock();active=0;maximum=0
    def tracked(*args,**kwargs):
        nonlocal active,maximum
        with lock:active+=1;maximum=max(maximum,active)
        try:
            time.sleep(.01)
            return p.download(*args,**kwargs)
        finally:
            with lock:active-=1
    p.service._download_dataset=tracked
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(lambda _:preview.collect_result_preview(p.service,p.record),range(3)))
    assert all(results) and len({r.root for r in results})==3
    assert 1 < maximum <= 4 and active==0


def test_simple_service_without_preview_capability_falls_back(published):
    assert preview.collect_result_preview(SimpleNamespace(),published.record) is None
