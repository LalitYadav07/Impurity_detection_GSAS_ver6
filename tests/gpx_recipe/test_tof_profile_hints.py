"""Reference profile fields travel independently of selected child models."""
import copy
import json
from pathlib import Path

import pytest

from gpx_recipe.execution import write_json
from gpx_recipe.preparation import hints,tof_profile_hints
from gpx_recipe.preparation.operations import create_library,stage_entry,publish_entry,portable_artifacts,inherit_reference
from gpx_recipe.preparation.tof_profile_arrays import encode,origin_identity
from gpx_recipe.symmetry.compiler import file_sha
from test_tof_profile_arrays import data


def library_for(tmp_path,data):
    cif=tmp_path/'parent.cif';cif.write_text('data_public\n')
    inst=tmp_path/'profile.instprm';inst.write_text('# public fixture\n')
    library=create_library(tmp_path/'library',name='Shared profile fixture',source=cif,
        banks={'nominal bank':dict(instrument=inst,grid=dict(kind='PNT',lower=3000.,upper=40000.,bins=701,bank=1))},
        models=[dict(kind='reference',bank='nominal bank',settings=dict(maximum_denominator=8))])
    row=library.document['entries'][0];attempt=library.start(row['id'])
    request=stage_entry(library,row['id'],attempt,tmp_path/'attempt');output=Path(request['destination']);output.mkdir()
    (output/'profile.npz').write_bytes(encode(data));write_json(output/'profile.json',data.metadata)
    names=['profile.npz','profile.json'];identity=origin_identity(data.metadata)
    entry=dict(schema='radar-experiment-preparation-entry/v1',kind='reference',grid=request['grid'],settings=request['settings'],
        input_sha256=[file_sha(request[k]) for k in ('source','instrument')],
        detail=dict(physics_sha256=identity,profile_table=dict(status='prepared',role='starting_proposals',
            physics_sha256=identity,observations_included=False,native_likelihood_authority=False)),
        artifacts={k:file_sha(output/k) for k in names},observations_included=False,scientific_result=False)
    write_json(output/'entry.json',entry)
    publish_entry(library,row['id'],attempt,request,dict(schema='radar-experiment-preparation-worker/v1',kind='reference',
        status='prepared',output=str(output),artifacts=names+['entry.json'],entry_sha256=file_sha(output/'entry.json'),
        observations_included=False,scientific_result=False))
    library.add_evidence(dict(site='https://galaxy.invalid',dataset_id='reference-receipt',sha256='a'*64,
        entries=[file_sha(output/'entry.json')]))
    return library


def test_profile_only_library_survives_subset_capture_and_restore(tmp_path,data):
    library=library_for(tmp_path,data);protocol=dict(kind='structural',candidates=[dict(id='a-future-child')])
    before=copy.deepcopy(protocol);packet=tmp_path/'packet';doc=hints.capture(library,protocol,packet)
    assert doc['entries']==[] and len(doc['profile_tables']['entries'])==1
    assert len(hints.packet_paths(packet,protocol))==3
    subset=hints.subset(doc,protocol,[])
    assert subset['profile_tables']==doc['profile_tables'] and subset['evidence']==doc['evidence']
    restored=tmp_path/'received';proof=hints.restore(subset,protocol,packet,restored)
    assert proof['profile_tables']==dict(attached=1,discarded_entries=[])
    choices,attachment=tof_profile_hints.candidates(restored,protocol)
    assert len(choices)==1 and choices[0]['identity']==origin_identity(data.metadata)
    assert attachment['attached']==1 and not attachment['discarded_entries']
    assert protocol==before


@pytest.mark.parametrize('damage',['missing','bytes','metadata'])
def test_optional_corruption_keeps_scientific_protocol_and_reports_rebuild(tmp_path,data,damage):
    library=library_for(tmp_path,data);protocol=dict(kind='structural',candidates=[dict(id='a-future-child')])
    packet=tmp_path/'packet';doc=hints.capture(library,protocol,packet);row=doc['profile_tables']['entries'][0]
    if damage=='missing':(packet/row['path']).unlink()
    elif damage=='bytes':(packet/row['path']).write_bytes(b'invalid data')
    else:(packet/row['receipt_path']).write_text('{}')
    report=hints.restore(doc,protocol,packet,tmp_path/'received')
    assert report['profile_tables']==dict(attached=0,discarded_entries=[row['entry']])
    choices,proof=tof_profile_hints.candidates(tmp_path/'received',protocol)
    assert choices==[] and proof['discarded_entries']
    assert protocol['candidates']==[dict(id='a-future-child')]


def test_transport_budget_omits_optional_fields_without_removing_models(tmp_path,data,monkeypatch):
    library=library_for(tmp_path,data);protocol=dict(kind='structural',candidates=[dict(id='a-future-child')])
    monkeypatch.setattr(tof_profile_hints,'MAX_TOTAL',1)
    doc=hints.capture(library,protocol,tmp_path/'packet')
    assert doc['profile_tables']['eligible_entries']==doc['profile_tables']['omitted_entries']==1
    assert doc['profile_tables']['entries']==[] and doc['evidence']==[]
    assert protocol['candidates']==[dict(id='a-future-child')]


def test_model_library_inherits_the_same_profile_bytes_and_origin(tmp_path,data):
    inspection=library_for(tmp_path,data)
    model=inspection.document['entries'][0]['model']
    target=create_library(tmp_path/'model-library',name='Model preparation with shared dependencies',
        source=tmp_path/'parent.cif',banks={'nominal bank':dict(instrument=tmp_path/'profile.instprm',grid=model['grid'])},
        models=[dict(kind='reference',bank='nominal bank',settings=model['settings'])])
    source=inspection.document['entries'][0]
    assert inherit_reference(target,inspection,source['id'])['all_prepared']
    copied=target.document['entries'][0]
    assert {Path(a['path']).name:a['sha256'] for a in copied['files']}=={
        Path(a['path']).name:a['sha256'] for a in source['files']}
    assert target.document['evidence']==inspection.document['evidence']
    assert not list(tmp_path.glob('radar-reference-transfer-*'))
    assert inherit_reference(target,inspection,source['id'])['all_prepared']


def test_changed_parent_cannot_inherit_an_unrelated_reference(tmp_path,data):
    inspection=library_for(tmp_path,data);model=inspection.document['entries'][0]['model']
    other=tmp_path/'other.cif';other.write_text('data_different_parent\n')
    target=create_library(tmp_path/'different-model-library',name='Different parent',source=other,
        banks={'nominal bank':dict(instrument=tmp_path/'profile.instprm',grid=model['grid'])},
        models=[dict(kind='reference',bank='nominal bank',settings=model['settings'])])
    with pytest.raises(ValueError,match='exact reviewed inputs'):
        inherit_reference(target,inspection,inspection.document['entries'][0]['id'])
    assert target.document['entries'][0]['state']=='failed'
    assert not list(tmp_path.glob('radar-reference-transfer-*'))


@pytest.mark.parametrize('names',[['profile.npz'],['profile.json'],['profile.npz','profile.json','native.py']])
def test_reference_field_inventory_is_a_closed_atomic_pair(names):
    with pytest.raises(ValueError):portable_artifacts('reference',names)
