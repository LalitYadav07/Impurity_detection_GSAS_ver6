"""Recorded device/epoch evidence is inspectable without loading neural weights."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from radar_pd_nova.preparation_training_record import describe


def fixture(tmp_path,metadata):
    model=dict(kind='structural-training',settings=dict(epochs=3),grid=dict(bank=1))
    root=tmp_path/'entry';root.mkdir()
    manifest=root/'entry.json';manifest.write_text(json.dumps(dict(
        schema='radar-experiment-preparation-entry/v1',kind=model['kind'],
        settings=model['settings'],grid=model['grid'],detail={})))
    path=root/'model.json';path.write_text(json.dumps(metadata))
    paths={p.name:str(p.relative_to(tmp_path)) for p in (manifest,path)}
    assets=[dict(path=name,sha256=hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()) for name in paths.values()]
    seen=[]
    def verify(names):
        seen.extend(names)
        for name in names:
            checksum=next(a['sha256'] for a in assets if a['path']==name)
            if hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()!=checksum:raise ValueError('Changed artifact')
        return {name:tmp_path/name for name in names}
    return SimpleNamespace(root=tmp_path,verify_assets=verify),dict(
        state='prepared',model=model,files=assets),seen,path


def metadata(**values):
    result=dict(schema='radar-three-curve-structural-ranking-model/v1',
        epochs=3,history=[dict(epoch=i) for i in range(1,4)],selected_epoch=2,seconds=77.5)
    result.update(values);return result


@pytest.mark.parametrize('device,label',[('cuda','CUDA GPU'),('mps','Apple GPU (MPS)'),('cpu','CPU')])
def test_recorded_device_and_completed_history_are_distinct_from_selected_checkpoint(tmp_path,device,label):
    library,entry,seen,_=fixture(tmp_path,metadata(training_device_transport=dict(
        device=device,portable_inference_device='cpu',representatives=4,maximum_logit_absolute_error=1e-7)))
    record=describe(library,entry);rows={r['label']:r['value'] for r in record['rows']}
    assert record['status']=='complete' and rows['Recorded training device']==label
    assert rows['Completed epochs']=='3 of 3' and rows['Selected checkpoint']=='Epoch 2'
    assert rows['Trainer time']=='1 min 18 s'
    assert rows['Portable inference'].startswith('CPU') and 'excludes Galaxy queue' in record['note']
    assert seen==['entry/entry.json','entry/model.json']  # No weights or large training arrays are read.


def test_old_cpu_metadata_does_not_invent_a_recorded_gpu_or_cpu_device(tmp_path):
    library,entry,_,_=fixture(tmp_path,metadata())
    rows={r['label']:r['value'] for r in describe(library,entry)['rows']}
    assert rows['Recorded training device']==rows['Portable inference']=='Not recorded'


def test_partial_history_cannot_be_presented_as_completed_training(tmp_path):
    value=metadata();value.update(history=[dict(epoch=1)],selected_epoch=1)
    library,entry,_,_=fixture(tmp_path,value);record=describe(library,entry)
    assert record['status']=='partial' and 'does not establish completed training' in record['note']
    assert next(r['value'] for r in record['rows'] if r['label']=='Completed epochs')=='1 of 3'


def test_lossless_packed_history_keeps_all_completed_epochs(tmp_path):
    from gpx_recipe.preparation.structural_history import pack
    value=metadata();value['history']=pack([dict(epoch=i,padding='x'*100000) for i in range(1,4)])
    assert isinstance(value['history'],dict)
    library,entry,_,_=fixture(tmp_path,value)
    assert describe(library,entry)['status']=='complete'


@pytest.mark.parametrize('change',[
    dict(schema='unknown-model/v1'),dict(schema=None),dict(schema=[]),dict(schema={}),
    dict(epochs=True),dict(epochs=2),dict(selected_epoch=4),dict(selected_epoch=True),
    dict(history=[dict(epoch=1),dict(epoch=3)]),dict(history=[dict(epoch=True)]),
    dict(seconds=-1),dict(seconds=float('nan')),
    dict(training_device_transport=dict(device=['cuda'])),
    dict(training_device_transport=dict(device='gpu',portable_inference_device='cpu')),
    dict(training_device_transport=dict(device='cuda',portable_inference_device='cuda',representatives=1,maximum_logit_absolute_error=0)),
])
def test_malformed_or_conflicting_training_records_are_not_displayed_as_evidence(tmp_path,change):
    value=metadata();value.update(change)
    library,entry,_,_=fixture(tmp_path,value)
    with pytest.raises(ValueError):describe(library,entry)


def test_changed_artifact_is_rejected_and_incomplete_entries_are_not_read(tmp_path):
    library,entry,seen,path=fixture(tmp_path,metadata())
    assert describe(library,dict(entry,state='preparing')) is None and not seen
    path.write_text(json.dumps(metadata(seconds=100)))
    with pytest.raises(ValueError,match='Changed artifact'):describe(library,entry)


def test_legacy_library_without_metadata_explains_missing_training_details(tmp_path):
    library,entry,_,_=fixture(tmp_path,metadata());entry['files']=entry['files'][:1]
    record=describe(library,entry)
    assert record['status']=='unrecorded' and record['rows']==[] and 'No device' in record['note']


def test_oversized_training_metadata_is_rejected_before_its_checksum_is_read(tmp_path):
    library,entry,seen,path=fixture(tmp_path,metadata());path.write_bytes(b' '*(2*1024**2+1))
    with pytest.raises(ValueError,match='metadata bound'):describe(library,entry)
    assert seen==['entry/entry.json']
