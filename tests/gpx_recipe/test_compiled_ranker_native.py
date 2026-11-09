"""Real typed-weight/library handoff on authored arrays; not model accuracy."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('torch')

from gpx_recipe.preparation.compiled_ranker import create,KIND
from gpx_recipe.preparation.library import import_library,validate_manifest
from gpx_recipe.preparation import structural_hints,structural_ranking_hints
from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.symmetry.contracts import validate_candidate
from test_structural_attention_portable_native import artifact


def prepared(root):
    root.mkdir()
    _,q,current,request,arrays,metadata=artifact(root/'model')
    parent=root/'parent.cif';parent.write_text('data_authored_parent\n')
    profile=root/'bank.instprm';profile.write_text('Authored profile; transport test only\n')
    children={}
    metadata['parent_sha256']=file_sha(parent);metadata['instrument_sha256']=file_sha(profile)
    for c in metadata['candidates']:
        child=root/(c['id']+'.cif');child.write_text('data_authored_'+c['id']+'\n')
        c['child']['sha256']=file_sha(child);children[c['id']]=child
    kwargs=dict(children=children,name='Authored portable model',bank='bank 2',
        grid=dict(kind='PNT',lower=3000.,upper=40000.,bins=701,bank=1))
    metadata['input_sha256']=['e'*64]
    header=root/'examples.json';header.write_text(json.dumps(dict(
        schema='radar-structural-ranking-pairs/v3',source_sha256=metadata['parent_sha256'],
        instrument_sha256=metadata['instrument_sha256'],grid=kwargs['grid'],pairs_sha256='e'*64,
        observations_included=False,candidate=metadata['candidates'][0])))
    kwargs['training_manifests']=[header]
    (root/'model/model.json').write_text(json.dumps(metadata))
    request.update(parent_sha256=metadata['parent_sha256'],instrument_sha256=metadata['instrument_sha256'])
    return root/'model',parent,profile,kwargs,(q,current,request)


def test_complete_model_library_cold_import_uses_the_existing_shared_inference_path(tmp_path):
    model,parent,profile,kwargs,observation=prepared(tmp_path/'source')
    before={n:file_sha(model/n) for n in ['model.json','weights.npz']}
    library=create(model,parent,profile,tmp_path/'library',**kwargs)
    assert library.coverage()['all_prepared']
    assert len(library.document['entries'])==1
    entry=library.document['entries'][0]
    assert entry['model']['kind']==KIND and entry['attempt']==0
    assert 'dependencies' not in entry['model']['settings']
    # The parent-class candidate legitimately reuses the identical parent CIF.
    assert len(library.document['inputs'])==3
    assert not list(library.root.rglob('*.gpx'))
    assert not list(library.root.rglob('examples.npz'))
    with pytest.raises(ValueError):library.start(entry['id'])
    library.export(tmp_path/'portable.zip')
    restored=import_library(tmp_path/'portable.zip',tmp_path/'restored')
    assert restored.document==library.document
    source_metadata=json.loads((model/'model.json').read_text())
    protocol=dict(kind='structural',candidates=[validate_candidate(c) for c in source_metadata['candidates']])
    packet=tmp_path/'packet';manifest=structural_hints.capture(restored,protocol,packet)
    assert manifest['entries']==[] and manifest['ranking_models']['eligible_entries']==1
    attached=structural_hints.restore(manifest,protocol,packet,tmp_path/'worker')
    assert attached['ranking_models']['attached']==1
    q,current,request=observation
    ranked=structural_ranking_hints.rank(tmp_path/'worker',protocol,q,current,
        entry_id=entry['id'],**request)
    assert ranked['ranking_available']
    assert set(ranked['candidate_order'])=={'child','parent'}
    assert ranked['complete_native_search_required'] and not ranked['candidate_pruning_authorized']
    assert before=={n:file_sha(model/n) for n in before}


@pytest.mark.parametrize('change',['parent','profile','missing_child','extra_child','changed_child','weights','bank','missing_manifest','changed_native_input'])
def test_compiled_import_cannot_replace_the_original_physical_or_candidate_scope(tmp_path,change):
    model,parent,profile,kwargs,_=prepared(tmp_path/'source')
    if change=='parent':parent.write_text('changed parent\n')
    elif change=='profile':profile.write_text('changed profile\n')
    elif change=='missing_child':kwargs['children'].pop('child')
    elif change=='extra_child':kwargs['children']['extra']=parent
    elif change=='changed_child':kwargs['children']['child'].write_text('changed child\n')
    elif change=='bank':kwargs['grid']['bank']=2
    elif change=='missing_manifest':kwargs['training_manifests']=[]
    elif change=='changed_native_input':
        path=kwargs['training_manifests'][0];header=json.loads(path.read_text())
        header['pairs_sha256']='f'*64;path.write_text(json.dumps(header))
    else:(model/'weights.npz').write_bytes(b'damaged')
    with pytest.raises(ValueError):create(model,parent,profile,tmp_path/'library',**kwargs)
    assert not (tmp_path/'library').exists()


@pytest.mark.parametrize('state',['pending','preparing','failed','unsupported','cancelled'])
def test_imported_rankers_cannot_be_reinterpreted_as_native_or_training_work(tmp_path,state):
    model,parent,profile,kwargs,_=prepared(tmp_path/'source')
    library=create(model,parent,profile,tmp_path/'library',**kwargs)
    document=copy.deepcopy(library.document);document['entries'][0]['state']=state
    if state!='prepared':document['entries'][0]['files']=[]
    with pytest.raises(ValueError):validate_manifest(document)


def test_hash_valid_origin_flags_cannot_grant_native_authority(tmp_path):
    model,parent,profile,kwargs,_=prepared(tmp_path/'source')
    library=create(model,parent,profile,tmp_path/'library',**kwargs)
    row=library.document['entries'][0]
    named={Path(a['path']).name:a for a in row['files']}
    origin=library.root/named['model-origin.json']['path'];d=json.loads(origin.read_text())
    d['native_origin_certificate']=True;origin.write_text(json.dumps(d))
    named['model-origin.json']['sha256']=file_sha(origin)
    entry=library.root/named['entry.json']['path'];d=json.loads(entry.read_text())
    d['artifacts']['model-origin.json']=file_sha(origin);entry.write_text(json.dumps(d))
    named['entry.json']['sha256']=file_sha(entry);library._save()
    protocol=dict(kind='structural',candidates=[validate_candidate(c) for c in json.loads((model/'model.json').read_text())['candidates']])
    with pytest.raises(ValueError,match='advisory data'):
        structural_hints.capture(library,protocol,tmp_path/'capture')


def test_completed_destination_and_model_bytes_are_never_overwritten(tmp_path):
    model,parent,profile,kwargs,_=prepared(tmp_path/'source')
    library=create(model,parent,profile,tmp_path/'library',**kwargs)
    before=file_sha(library.path)
    with pytest.raises(ValueError,match='fresh destination'):
        create(model,parent,profile,tmp_path/'library',**kwargs)
    assert file_sha(library.path)==before


def test_portable_budget_is_checked_before_copying_the_model(tmp_path,monkeypatch):
    from gpx_recipe.preparation import library as storage
    model,parent,profile,kwargs,_=prepared(tmp_path/'source')
    monkeypatch.setattr(storage,'MAX_ARCHIVE_BYTES',1024)
    with pytest.raises(ValueError,match='portable byte budget'):
        create(model,parent,profile,tmp_path/'library',**kwargs)
    assert not (tmp_path/'library').exists()
    assert not list(tmp_path.glob('library.*.partial'))
