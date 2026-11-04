"""Checked quadratic hints survive separate native jobs without altering scope."""
import json
import shutil

import numpy as np
import pytest

from gpx_recipe.symmetry import profiled_cache as cache
from gpx_recipe.symmetry.chunks import execute_chunk
from gpx_recipe.symmetry.compiler import file_sha, package_signature
from gpx_recipe.symmetry.contracts import digest, iter_trials, trial_id, validate_protocol
from test_symmetry_contracts import make_protocol


def make_cache(root, protocol, identity='d'*64, candidate='mag:fixture'):
    relative='_profiled_cache/'+identity+'.npz'
    target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,identity=identity,offset=np.zeros(5),maps=np.ones((1,3,1)),
        nuclear=np.ones(5),quadratic=np.ones((5,1,1)),background=np.ones((5,1)),linear_error=0.)
    return dict(schema=cache.SCHEMA,protocol_sha256=digest(protocol),entries=[dict(
        identity=identity,candidate=candidate,path=relative,sha256=file_sha(target))])


def test_two_fresh_chunks_reuse_only_sealed_source_and_candidate(tmp_path):
    source=tmp_path/'source';p=make_protocol(source)
    p['policy'].update(magnetic_optimizer='profiled_quadratic',optimizer_iterations=100,moment_model='symmetry_allowed')
    p['implementation_sha256']=package_signature();p=validate_protocol(p)
    uid=trial_id(p,*[(c['id'],s['id']) for c,s in iter_trials(p)][0])
    original=file_sha(source/'source.gpx')
    def runner(path,**_):
        root=path.parent.parent/'inputs'/digest(p)
        manifest=make_cache(root,p)
        proof=path/'results'/'optimizer-receipt.json';proof.parent.mkdir()
        proof.write_text(json.dumps({'objective_identity':{'cache_identity':'d'*64}}))
        return dict(state='finished',status='inadequate_fit',wall_seconds=12.)
    request=dict(schema='radar-model-chunk/v1',protocol=p,protocol_sha256=digest(p),
        trials=[uid],workers=1,dispatch_key='a'*24,seconds=240)
    first=execute_chunk(request,source,tmp_path/'first',runner=runner,environ={})
    manifest=first['profiled_cache'];assert len(manifest['entries'])==1
    packet=tmp_path/'packet';shutil.copytree(source,packet)
    good,bad=cache.restore(manifest,p,tmp_path/'first',packet)
    assert len(good)==1 and not bad
    def warm(path,**_):
        received=path.parent.parent/'inputs'/digest(p)/good[0]['path']
        assert file_sha(received)==good[0]['sha256']
        return runner(path)
    second=execute_chunk(dict(request,profiled_cache=manifest,dispatch_key='b'*24),packet,
                         tmp_path/'second',runner=warm,environ={})
    assert second['trials'][0]['id']==uid
    assert second['profiled_cache']==manifest and not second['discarded_cache_candidates']
    assert file_sha(source/'source.gpx')==original


@pytest.mark.parametrize('mutation',['protocol','candidate','traversal','duplicate','unknown'])
def test_cross_scope_and_unsafe_cache_manifest_rejected_before_copy(tmp_path,mutation):
    p=make_protocol(tmp_path/'source');manifest=make_cache(tmp_path/'hints',p)
    if mutation=='protocol':manifest['protocol_sha256']='f'*64
    elif mutation=='candidate':manifest['entries'][0]['candidate']='mag:other'
    elif mutation=='traversal':manifest['entries'][0]['path']='../source.gpx'
    elif mutation=='duplicate':manifest['entries']*=2
    else:manifest['entries'][0]['executable']='evil.py'
    with pytest.raises(ValueError):cache.restore(manifest,p,tmp_path/'hints',tmp_path/'target')
    assert not (tmp_path/'target').exists()


@pytest.mark.parametrize('damage',['checksum','identity','missing','pickle'])
def test_bad_optional_hint_is_discarded_and_cannot_replace_inputs(tmp_path,damage):
    p=make_protocol(tmp_path/'source');root=tmp_path/'hints';manifest=make_cache(root,p)
    row=manifest['entries'][0];path=root/row['path']
    if damage=='checksum':path.write_bytes(path.read_bytes()+b'changed')
    elif damage=='missing':path.unlink()
    else:
        with np.load(path,allow_pickle=False) as data:values={k:data[k] for k in data.files}
        values['identity']= 'e'*64 if damage=='identity' else np.array({'bad':'object'},dtype=object)
        np.savez_compressed(path,**values);row['sha256']=file_sha(path)
    good,bad=cache.restore(manifest,p,root,tmp_path/'target')
    assert not good and bad==['mag:fixture']
    assert not (tmp_path/'target'/row['path']).exists()


def test_archive_expansion_is_bounded_before_array_load(tmp_path,monkeypatch):
    p=make_protocol(tmp_path/'source');manifest=make_cache(tmp_path/'hints',p)
    monkeypatch.setattr(cache,'MAX_BYTES',1000)
    assert not cache.bounded_npz(tmp_path/'hints'/manifest['entries'][0]['path'],'d'*64)
