"""Whole-ray contracts survive packing; legacy layouts remain strict."""
import copy
import hashlib
import json

import numpy as np
import pytest

from gpx_recipe.preparation import structural_mode_layout as layout
from gpx_recipe.preparation.structural_mode_coverage import SCHEMA
from gpx_recipe.preparation.structural_paired_blocks import read,pack_block
from gpx_recipe.preparation.structural_training_blocks import representation,REPRESENTATION
from test_structural_paired_blocks import block


def record():
    rays=[]
    for parent in [0,3,7]:
        for sign in [-1,1]:
            start=8+len(rays)
            rays.append(dict(native_mode=0,parent=parent,sign=sign,rows=[start,start+1],
                maximum_amplitude=.01,rejected_amplitudes=0))
    return dict(schema=SCHEMA,contexts=3,labels=['native mode','origin gauge'],gauge_axes=[1],
        base_examples=8,rays=rays,rejected_amplitudes=0,maximum_reconstruction_error=0.)


def named_block(root,interval=(0,8)):
    source=read(block(root/'old'));d=copy.deepcopy(source.manifest)
    d.update(schema=layout.RAW_SCHEMA,mode_coverage=record(),trajectory_slice=list(interval))
    trajectory,role,indices,_=layout.rows(d,8,1,interval)
    old=source.arrays;rows=[0]
    for parent in range(*interval):
        # Copy that parent's exact original control and original hypothesis;
        # added test curves are hypotheses, never fabricated native evidence.
        rows.append(1+2*parent)
        rows.extend([2+2*parent]*np.count_nonzero((trajectory==parent)&(role==2)))
    arrays={k:(a[rows].copy() if a.ndim and a.shape[0]==len(old['role']) else a.copy())
            for k,a in old.items()}
    arrays.update(trajectory=trajectory,role=role)
    d.update(examples=len(role),groups=['parent:'+d['source_sha256']]+[
        f'parent:{d["source_sha256"]}:1:2:trajectory-{p}' for p in trajectory[1:]])
    directory=root/'named';directory.mkdir()
    save(directory,d,arrays)
    return directory,source,indices


def save(directory,d,a):
    np.savez_compressed(directory/'pairs.npz',**a)
    d['pairs_sha256']=hashlib.sha256((directory/'pairs.npz').read_bytes()).hexdigest()
    (directory/'pairs.json').write_text(json.dumps(d))


def test_complete_signed_axes_preserve_original_parent_groups_and_pack_losslessly(tmp_path):
    directory,old,_=named_block(tmp_path)
    new=read(directory);packed=pack_block(directory,tmp_path/'packed')
    assert new.manifest['schema']==layout.RAW_SCHEMA
    assert packed.manifest['schema']==layout.PACKED_SCHEMA
    assert packed.manifest['mode_coverage']==record()
    assert representation(packed.manifest)==REPRESENTATION
    for key in new.arrays:np.testing.assert_array_equal(packed.arrays[key],new.arrays[key])
    for p in range(8):
        before=np.flatnonzero(old.arrays['trajectory']==p)
        after=np.flatnonzero(new.arrays['trajectory']==p)[:len(before)]
        for key,a in old.arrays.items():
            if a.ndim and a.shape[0]==len(old.arrays['role']):
                np.testing.assert_array_equal(new.arrays[key][after],a[before])


@pytest.mark.parametrize('interval',[(0,1),(1,3),(3,8)])
def test_shards_keep_all_declared_amplitudes_for_each_selected_parent(tmp_path,interval):
    directory,_,indices=named_block(tmp_path,interval)
    d=read(directory)
    expected_extra=sum(interval[0]<=r['parent']<interval[1] for r in record()['rays'])
    assert len(d.arrays['role'])==1+2*(interval[1]-interval[0])+expected_extra
    assert len(set(indices[indices>=0]))==np.count_nonzero(d.arrays['role']==2)


@pytest.mark.parametrize('case',['missing','version','unknown_field','missing_ray','sign','axis','partial_steps',
    'repeat_parent','different_sign_parent','gauge','rejections','reconstruction','nonfinite','boolean_index'])
def test_recomputed_checksums_cannot_hide_incomplete_or_relabelled_ray_scope(tmp_path,case):
    directory,_,_=named_block(tmp_path);d=json.loads((directory/'pairs.json').read_text());c=d['mode_coverage']
    if case=='missing':del d['mode_coverage']
    elif case=='version':c['schema']='unknown'
    elif case=='unknown_field':c['scientific_result']=True
    elif case=='missing_ray':c['rays'].pop()
    elif case=='sign':c['rays'][0]['sign']=1
    elif case=='axis':c['rays'][0]['native_mode']=1
    elif case=='partial_steps':c['rays'][0]['rows'][1]-=1
    elif case=='repeat_parent':
        c['rays'][2]['parent']=c['rays'][3]['parent']=0
    elif case=='different_sign_parent':c['rays'][1]['parent']=1
    elif case=='gauge':c['gauge_axes']=[0,1]
    elif case=='rejections':c['rejected_amplitudes']=1
    elif case=='reconstruction':c['maximum_reconstruction_error']=1e-5
    elif case=='nonfinite':c['rays'][0]['maximum_amplitude']=float('nan')
    else:c['rays'][0]['native_mode']=False
    (directory/'pairs.json').write_text(json.dumps(d))
    with pytest.raises(ValueError):read(directory)


def test_old_schema_cannot_claim_named_coverage_or_accept_the_enlarged_layout(tmp_path):
    directory,_,_=named_block(tmp_path)
    d=json.loads((directory/'pairs.json').read_text());d['schema']='radar-structural-ranking-paired-pilot/v1'
    (directory/'pairs.json').write_text(json.dumps(d))
    with pytest.raises(ValueError,match='explicit paired schema'):read(directory)
    del d['mode_coverage'];(directory/'pairs.json').write_text(json.dumps(d))
    with pytest.raises(ValueError,match='complete child trajectories'):read(directory)


def test_worker_contract_binds_coverage_without_migrating_saved_requests(tmp_path):
    from test_structural_paired_preparation import inputs
    from gpx_recipe.preparation.structural_jobs import validate_examples
    request,_=inputs(tmp_path)
    assert 'mode_coverage' not in request['settings'] and validate_examples(request)==request
    named=copy.deepcopy(request);named['settings']['mode_coverage']=SCHEMA
    assert validate_examples(named)==named
    for wrong in ['unknown',True,{'schema':SCHEMA}]:
        bad=copy.deepcopy(named);bad['settings']['mode_coverage']=wrong
        with pytest.raises(ValueError,match='paired sampling version'):validate_examples(bad)
    bad=copy.deepcopy(named);bad['settings'].pop('representation')
    with pytest.raises(ValueError):validate_examples(bad)


@pytest.mark.parametrize('case',['valid','missing','odd','short','unknown','wrong_version','wrong_request','boolean'])
def test_owned_native_sizing_retains_all_additional_mode_rows(tmp_path,case):
    from pathlib import Path
    from test_structural_paired_preparation import inputs
    from gpx_recipe.preparation import native_size
    from gpx_recipe.symmetry.compiler import package_signature
    request,_=inputs(tmp_path);request['settings']['mode_coverage']=SCHEMA
    Path(request['destination']).mkdir()
    counts=dict(schema=SCHEMA,native_modes=2,gauge_modes=1,contexts=3,rays=6,selected_rays=6)
    n=23
    if case=='missing':counts=None
    elif case=='odd':counts['selected_rays']=5;n=22
    elif case=='short':n=22
    elif case=='unknown':counts['scientific_result']=True
    elif case=='wrong_version':counts['schema']='unknown'
    elif case=='wrong_request':request['settings'].pop('mode_coverage')
    elif case=='boolean':counts['native_modes']=True
    kwargs=dict(rows=n,bins=101,mode_dimensions=1,cell_dimensions=2,
        numeric_bytes=native_size.paired_array_bytes(n,101,1,2),mode_coverage_rows=counts)
    if case=='valid':
        native_size.publish(request,**kwargs);d=native_size.read(request,package_signature())
        assert d['schema']==native_size.MODE_SCHEMA and d['rows']==23
        assert d['mode_coverage_rows']==counts
        assert d['reservation_bytes']>d['numeric_bytes']
    else:
        with pytest.raises(ValueError):native_size.publish(request,**kwargs)
