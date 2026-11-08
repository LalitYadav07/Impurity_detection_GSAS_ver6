"""Data-only ranking transport. Opaque test weights are never used for physics."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gpx_recipe.execution import write_json
from gpx_recipe.preparation import hints,structural_hints,structural_ranking_hints as ranking
from gpx_recipe.preparation.operations import stage_entry,publish_entry
from gpx_recipe.preparation.structural_jobs import publish_result
from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.symmetry.contracts import digest
from test_structural_preparation_dependencies import library_inputs,complete_examples,TRAINING


def prepared(root,*,three_curve=False,architecture=None):
    library,originals=library_inputs(root)
    complete_examples(library,originals,root/'examples-attempts')
    trainer=next(r for r in library.document['entries'] if r['model']['kind']==TRAINING)
    attempt=library.start(trainer['id']);request=stage_entry(library,trainer['id'],attempt,root/'training-attempt')
    out=Path(request['destination']);(out/'model').mkdir(parents=True)
    np.savez_compressed(out/'model/weights.npz',opaque_transport_fixture=np.ones(3))
    candidates=[r['model']['settings']['candidate'] for r in library.document['entries']
                if r['model']['kind']=='structural-examples']
    metadata=dict(schema='radar-structural-ranking-model/v4',candidates=candidates,
        parent_sha256=file_sha(request['source']),instrument_sha256=file_sha(request['instrument']),
        weights_sha256=file_sha(out/'model/weights.npz'))
    if three_curve:
        from gpx_recipe.preparation.structural_attention_weights import SCHEMA,ARCHITECTURE
        metadata.update(schema=SCHEMA,architecture=ARCHITECTURE if architecture is None else architecture)
    write_json(out/'model/model.json',metadata);write_json(out/'training.json',dict(status='trained'))
    result=publish_result(TRAINING,request,dict(seconds=.1),['model/model.json','model/weights.npz','training.json'])
    publish_entry(library,trainer['id'],attempt,request,result)
    protocol=dict(kind='structural',candidates=copy.deepcopy(candidates))
    return library,protocol


@pytest.mark.parametrize('architecture',[
    'parent-q-convolution-attention-3-24-32-t128-h4/v1',
    'parent-q-convolution-attention-3-24-32-t128-h4-normalized-readout/v2'])
def test_three_curve_attention_data_use_the_same_portable_routes_and_integrity_checks(tmp_path,architecture):
    library,protocol=prepared(tmp_path/'control',three_curve=True,architecture=architecture);packet=tmp_path/'packet'
    document=hints.capture(library,protocol,packet);row=document['ranking_models']['entries'][0]
    assert row['routes']=={'first':'first','second':'second'}
    restored=hints.restore(document,protocol,packet,tmp_path/'restored')
    assert restored['ranking_models']['attached']==1 and not restored['ranking_models']['discarded_entries']
    files=ranking.checked_files(packet,row,protocol)
    assert json.loads(files['model'].read_text())['schema']=='radar-three-curve-structural-ranking-model/v1'
    assert json.loads(files['model'].read_text())['architecture']==architecture


def test_model_capture_restore_and_subset_keep_exact_routes_and_all_candidates(tmp_path):
    library,protocol=prepared(tmp_path/'control');original=copy.deepcopy(protocol)
    packet=tmp_path/'packet';document=hints.capture(library,protocol,packet)
    assert document['entries']==[]
    model=document['ranking_models'];assert model['eligible_entries']==len(model['entries'])==1
    assert model['entries'][0]['routes']=={'first':'first','second':'second'}
    assert len(hints.packet_paths(packet,protocol))==5
    assert {p.suffix for p in packet.rglob('*') if p.is_file()}=={'.json','.npz','.cif','.instprm'}
    received=tmp_path/'received';report=hints.restore(document,protocol,packet,received)
    assert report['ranking_models']==dict(attached=1,discarded_entries=[],candidate_pruning_authorized=False)
    ranking.checked_files(received,model['entries'][0],protocol)
    assert hints.subset(document,protocol,[])['ranking_models']==model
    assert hints.read(received,protocol)==document and protocol==original


@pytest.mark.parametrize('role',list(ranking.LIMITS))
def test_corrupt_or_missing_model_assets_fall_back_without_discarding_native_models(tmp_path,role):
    library,protocol=prepared(tmp_path/'control');packet=tmp_path/'packet'
    document=hints.capture(library,protocol,packet);row=document['ranking_models']['entries'][0]
    (packet/row['files'][role]['path']).write_bytes(b'changed')
    report=hints.restore(document,protocol,packet,tmp_path/'received')
    assert report['ranking_models']['attached']==0
    assert report['ranking_models']['discarded_entries']==[row['entry']]
    result=ranking.rank(packet,protocol,[1,2],[3,4],background=[0,0],
        parent_sha256=row['files']['parent']['sha256'],instrument_sha256=row['files']['instrument']['sha256'],entry_id=row['entry'])
    assert result['candidate_order']==[c['id'] for c in protocol['candidates']]
    assert result['ranking_available'] is result['candidate_pruning_authorized'] is False
    assert result['complete_native_search_required']
    assert result['discarded_entries']==[dict(entry=row['entry'],error_type='ValueError')]


@pytest.mark.parametrize('change',['route','duplicate','path','parent','extra','counts','grid'])
def test_closed_ranking_inventory_cannot_redirect_a_native_construction(tmp_path,change):
    library,protocol=prepared(tmp_path/'control');packet=tmp_path/'packet'
    document=hints.capture(library,protocol,packet);model=document['ranking_models'];row=model['entries'][0]
    if change=='route':row['routes']['first']='second'
    elif change=='duplicate':model['entries']*=2;model['eligible_entries']=2
    elif change=='path':row['files']['weights']['path']='../outside.npz'
    elif change=='parent':row['files']['parent']['sha256']='f'*64
    elif change=='extra':row['python']='print(1)'
    elif change=='counts':model['eligible_entries']=2
    else:row['grid']['bins']=False
    document['snapshot_sha256']=digest({k:v for k,v in document.items() if k!='snapshot_sha256'})
    with pytest.raises(ValueError):hints.restore(document,protocol,packet,tmp_path/'received')
    assert not (tmp_path/'received').exists()


def test_hash_valid_but_wrong_candidate_geometry_is_not_reused(tmp_path):
    library,protocol=prepared(tmp_path/'control');packet=tmp_path/'packet'
    document=hints.capture(library,protocol,packet);row=document['ranking_models']['entries'][0]
    row['routes']={'first':'second','second':'first'}
    document['snapshot_sha256']=digest({k:v for k,v in document.items() if k!='snapshot_sha256'})
    report=hints.restore(document,protocol,packet,tmp_path/'received')
    assert report['ranking_models']['attached']==0
    assert report['ranking_models']['discarded_entries']==[row['entry']]


def test_ranking_transfer_budget_only_changes_acceleration_coverage(tmp_path,monkeypatch):
    library,protocol=prepared(tmp_path/'control');monkeypatch.setattr(ranking,'MAX_BYTES',1)
    document=hints.capture(library,protocol,tmp_path/'packet')
    assert document['ranking_models']['entries']==[]
    assert document['ranking_models']['eligible_entries']==document['ranking_models']['omitted_entries']==1
    assert len(protocol['candidates'])==2


def test_a_selected_k_scope_can_attach_a_broader_trained_library_without_adding_models(tmp_path):
    library,protocol=prepared(tmp_path/'control');protocol['candidates']=protocol['candidates'][1:]
    document=hints.capture(library,protocol,tmp_path/'packet')
    assert document['ranking_models']['entries'][0]['routes']=={'second':'second'}
    assert len(protocol['candidates'])==1


def test_a_foreign_profile_or_parent_does_not_load_the_model(tmp_path,monkeypatch):
    library,protocol=prepared(tmp_path/'control');packet=tmp_path/'packet'
    hints.capture(library,protocol,packet)
    result=ranking.rank(packet,protocol,[1,2],[3,4],background=[0,0],
                        parent_sha256='f'*64,instrument_sha256='e'*64,entry_id='a'*24)
    assert not result['ranking_available'] and not result['discarded_entries']
    assert result['candidate_order']==['first','second']


def test_another_bank_entry_cannot_be_selected_by_a_shared_instrument_file_hash(tmp_path):
    library,protocol=prepared(tmp_path/'control');packet=tmp_path/'packet'
    document=hints.capture(library,protocol,packet);row=document['ranking_models']['entries'][0]
    result=ranking.rank(packet,protocol,[1,2],[3,4],background=[0,0],
        parent_sha256=row['files']['parent']['sha256'],instrument_sha256=row['files']['instrument']['sha256'],
        entry_id='a'*24 if row['entry']!='a'*24 else 'b'*24)
    assert not result['ranking_available'] and not result['discarded_entries']
    assert result['candidate_order']==['first','second']


def test_exact_entry_selection_disambiguates_models_sharing_a_profile_file(tmp_path,monkeypatch):
    import sys
    from types import ModuleType
    library,protocol=prepared(tmp_path/'control');packet=tmp_path/'packet'
    document=hints.capture(library,protocol,packet);first=document['ranking_models']['entries'][0]
    second=copy.deepcopy(first);second.update(entry='f'*24,bank='bank 2')
    second['grid']['bank']=2
    document['ranking_models']['entries'].append(second);document['ranking_models']['eligible_entries']=2
    document['snapshot_sha256']=digest({k:v for k,v in document.items() if k!='snapshot_sha256'})
    write_json(packet/structural_hints.MANIFEST,document)
    calls=[]
    class FakeRanker:
        def __init__(self,*args,**kwargs):calls.append(kwargs)
        def rank(self,*args,**kwargs):
            return dict(ranking_available=True,families=[dict(space_group=65,model_score=.6,candidate_ids=['second']),
                dict(space_group=63,model_score=.4,candidate_ids=['first'])],
                candidate_pruning_authorized=False,native_refinement_required=True)
    module=ModuleType('gpx_recipe.preparation.structural_ranker');module.Ranker=FakeRanker
    monkeypatch.setitem(sys.modules,module.__name__,module)
    result=ranking.rank(packet,protocol,[1,2],[3,4],background=[0,0],entry_id=second['entry'],
        parent_sha256=second['files']['parent']['sha256'],instrument_sha256=second['files']['instrument']['sha256'])
    assert result['entry']==second['entry'] and result['bank']=='bank 2' and len(calls)==1
    assert result['candidate_order']==['second','first'] and result['complete_native_search_required']


def test_budget_counts_identical_bytes_stored_under_distinct_role_filenames(tmp_path,monkeypatch):
    library,protocol=prepared(tmp_path/'control');packet=tmp_path/'packet'
    document=hints.capture(library,protocol,packet);row=document['ranking_models']['entries'][0]
    # Opaque contract-only parent/profile bytes intentionally coincide. They
    # are still two physical files with different data-role extensions.
    profile=row['files']['instrument'];parent=dict(path=ranking.PREFIX+profile['sha256']+'.cif',sha256=profile['sha256'])
    (packet/parent['path']).write_bytes((packet/profile['path']).read_bytes());row['files']['parent']=parent
    metadata=json.loads((packet/row['files']['model']['path']).read_text())
    metadata['parent_sha256']=parent['sha256']
    changed=packet/'changed.json';write_json(changed,metadata);sha=file_sha(changed)
    row['files']['model']=dict(path=ranking.PREFIX+sha+'.json',sha256=sha)
    changed.rename(packet/row['files']['model']['path'])
    document['snapshot_sha256']=digest({k:v for k,v in document.items() if k!='snapshot_sha256'})
    actual=sum((packet/a['path']).stat().st_size for a in row['files'].values())
    monkeypatch.setattr(ranking,'MAX_BYTES',actual-1)
    result=hints.restore(document,protocol,packet,tmp_path/'received')
    assert result['ranking_models']['attached']==0
    assert result['ranking_models']['discarded_entries']==[row['entry']]
    assert len(protocol['candidates'])==2
