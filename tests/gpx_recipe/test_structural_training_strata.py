"""Old examples survive widening a complete parent-range preparation."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from gpx_recipe.preparation import structural_jobs as jobs
from gpx_recipe.preparation.structural_cohort import assemble_pairs
from gpx_recipe.preparation.structural_paired_blocks import read, SCHEMA
from gpx_recipe.preparation.structural_training_blocks import REPRESENTATION
from gpx_recipe.symmetry.compiler import file_sha
from test_structural_paired_preparation import inputs as single_inputs


def variant(source, destination, identifier, high, interval):
    original=read(source,names='examples');d=copy.deepcopy(original.manifest)
    a={k:v.copy() for k,v in original.arrays.items()}
    selected=(a['trajectory']==-1)|((a['trajectory']>=interval[0])&(a['trajectory']<interval[1]))
    positions=np.flatnonzero(selected)
    for key in a.keys()-{'q','background'}:a[key]=a[key][positions]
    a['theory_intensity'][a['role']==2]+=np.exp(-((a['q']-3.)/.1)**2)*(high+.001*(identifier=='second'))
    d.update(schema=SCHEMA,examples=len(positions),trajectory_slice=list(interval))
    d['parent_relative_metric_range']=[.0001,high];d['candidate']['id']=identifier
    d['groups']=[d['groups'][i] for i in positions]
    destination.mkdir()
    np.savez_compressed(destination/'examples.npz',**a)
    d['pairs_sha256']=file_sha(destination/'examples.npz')
    (destination/'examples.json').write_text(json.dumps(d))
    read(destination,names='examples')
    return destination


def inputs(root):
    payload,source=single_inputs(root)
    paths=[]
    for identifier in ['first','second']:
        paths.append(variant(source,root/('old-'+identifier),identifier,.001,[0,8]))
    for identifier in ['first','second']:
        for lower,upper in [(0,4),(4,8)]:
            paths.append(variant(source,root/f'wide-{identifier}-{lower}',identifier,.005,[lower,upper]))
    return payload,paths


def test_balanced_complete_ranges_retain_every_original_block_and_group(tmp_path):
    payload,paths=inputs(tmp_path);plan_path=tmp_path/'plan.json'
    before={(p.name,n):file_sha(p/n) for p in paths for n in ['examples.json','examples.npz']}
    plan=jobs.create_plan(payload['source'],paths,plan_path)
    assert plan['schema']==jobs.STRATIFIED_PLAN_SCHEMA
    assert [row['parent_relative_metric_range'] for row in plan['parent_strata']]==[[.0001,.001],[.0001,.005]]
    assert [row['candidate_ids'] for row in plan['parent_strata']]==[['first','second']]*2
    assert jobs.read_plan(plan_path)==plan
    assert jobs.checked_blocks(plan_path,plan,grid=payload['grid'])==paths
    assert jobs.checked_blocks(plan_path,plan,validate_arrays=False,grid=payload['grid'])==paths
    assert jobs.plan_inputs(plan_path)==[p.name+'/'+n for p in paths for n in ['examples.json','examples.npz']]
    assert jobs.create_plan(payload['source'],paths,plan_path,recover_identical=True)==plan
    assert before=={(p.name,n):file_sha(p/n) for p in paths for n in ['examples.json','examples.npz']}
    old=assemble_pairs([read(p,names='examples') for p in paths[:2]],bins=128)
    combined=assemble_pairs([read(p,names='examples') for p in paths],bins=128)
    digest=lambda v:hashlib.sha256(v.tobytes()).hexdigest()
    assert {digest(v) for v in old['features']}<={digest(v) for v in combined['features']}
    assert len(combined['features'])>len(old['features'])
    assert combined['original_examples']==sum(read(p,names='examples').manifest['examples'] for p in paths)
    # A raw trajectory can never move to another fold merely because its
    # parent metric was sampled over another range or its shard name changed.
    assignments={}
    blocks=[read(p,names='examples') for p in paths]
    for code,references in zip(combined['split'],combined['references']):
        for block,index in references:
            group=blocks[block].manifest['groups'][index]
            assignments.setdefault(group,set()).add(int(code))
    assert all(len(codes)==1 for codes in assignments.values())


def test_equivalent_grid_numbers_and_additional_original_aliases_are_preserved(tmp_path):
    payload,paths=inputs(tmp_path)
    duplicate=tmp_path/'additional-original';shutil.copytree(paths[0],duplicate);paths.append(duplicate)
    for directory in paths[2:-1]:
        path=directory/'examples.json';d=json.loads(path.read_text())
        d['grid']['lower']=int(d['grid']['lower']);d['grid']['upper']=int(d['grid']['upper'])
        path.write_text(json.dumps(d))
    plan_path=tmp_path/'plan.json';plan=jobs.create_plan(payload['source'],paths,plan_path)
    assert len(plan['blocks'])==7 and jobs.checked_blocks(plan_path,plan,grid=payload['grid'])==paths


def test_added_mode_labels_cannot_reshuffle_original_validation_and_test_trajectories(tmp_path):
    _,paths=inputs(tmp_path)
    original_blocks=[read(p,names='examples') for p in paths[:2]]
    original=assemble_pairs(original_blocks,bins=128)
    assignment={}
    for code,references in zip(original['split'],original['references']):
        for b,i in references:assignment[original_blocks[b].manifest['groups'][i]]=int(code)
    # Add a label on a complete new range. It has original training,
    # validation and test support, but its uneven coverage changes the
    # greedy partitioner's preferred groups if it is run again from scratch.
    by_code={code:sorted(g for g,c in assignment.items() if c==code) for code in range(3)}
    support={by_code[0][-1],*by_code[1],*by_code[2]}
    for path in paths[2:]:
        data=read(path,names='examples');a={k:v.copy() for k,v in data.arrays.items()};d=copy.deepcopy(data.manifest)
        for i,g in enumerate(d['groups']):
            if g in support and a['role'][i]==2:a['geometry_groups'][i]=13
        np.savez_compressed(path/'examples.npz',**a);d['pairs_sha256']=file_sha(path/'examples.npz')
        (path/'examples.json').write_text(json.dumps(d))
    blocks=[read(p,names='examples') for p in paths];combined=assemble_pairs(blocks,bins=128)
    assert set(combined['classes'])=={12,13,166}
    for code,references in zip(combined['split'],combined['references']):
        for b,i in references:assert int(code)==assignment[blocks[b].manifest['groups'][i]]


@pytest.mark.parametrize('storage',['native_pair_replay','file_backed'])
def test_all_storage_paths_share_original_anchor_and_identical_feature_rows(tmp_path,storage):
    from gpx_recipe.preparation.structural_augmentation import Controls,ObservationAugmenter
    from gpx_recipe.preparation.structural_training_data import feature_digest
    from gpx_recipe.preparation import structural_replay,structural_paired_store
    _,paths=inputs(tmp_path/'inputs')
    profile=read(paths[0],names='examples').manifest['instrument_sha256']
    options=dict(bins=128,augmenter=ObservationAugmenter(Controls(variants=1),instrument_sha256=profile))
    dense=assemble_pairs([read(p,names='examples') for p in paths],**options)
    if storage=='native_pair_replay':other=structural_replay.build(paths,names='examples',**options)
    else:other=structural_paired_store.build(paths,tmp_path/'store',names='examples',**options)
    try:
        assert other['partition_anchor']==dense['partition_anchor']
        for key in dense.keys()-{'features'}:
            if isinstance(dense[key],np.ndarray):np.testing.assert_array_equal(other[key],dense[key])
            else:assert json.dumps(other[key],sort_keys=True)==json.dumps(dense[key],sort_keys=True),key
        assert feature_digest(other['features'])==feature_digest(dense['features'])
        if storage=='native_pair_replay':assert other['features'].diagnostics()['source_blocks_captured']==len(paths)
    finally:other['features'].close()


@pytest.mark.parametrize('case',['unsupported_new_label','cross_fold_alias','changed_q','disjoint_ranges'])
def test_expansion_refuses_to_relabel_move_or_hide_unresolved_original_holdouts(tmp_path,case):
    _,paths=inputs(tmp_path)
    original_blocks=[read(p,names='examples') for p in paths[:2]]
    original=assemble_pairs(original_blocks,bins=128);assignment={}
    for code,references in zip(original['split'],original['references']):
        for b,i in references:assignment[original_blocks[b].manifest['groups'][i]]=int(code)
    training=sorted(g for g,c in assignment.items() if c==0 and 'trajectory-' in g)
    heldout=next(g for g,c in assignment.items() if c==1 and 'trajectory-' in g)
    for path in paths[2:]:
        data=read(path,names='examples');a={k:v.copy() for k,v in data.arrays.items()};d=copy.deepcopy(data.manifest)
        if case=='unsupported_new_label':
            for i,g in enumerate(d['groups']):
                if g in training[:3] and a['role'][i]==2:a['geometry_groups'][i]=13
        elif case=='cross_fold_alias':
            if d['candidate']['id']=='first':
                source=np.array([i for i,g in enumerate(original_blocks[0].manifest['groups']) if g==heldout])
                target=np.array([i for i,g in enumerate(d['groups']) if g==training[0]])
                if len(target):
                    for key in a.keys()-{'q','background','role','trajectory'}:
                        a[key][target]=original_blocks[0].arrays[key][source]
        elif case=='changed_q':a['q']*=1.0001
        else:d['parent_relative_metric_range']=[.002,.005]
        np.savez_compressed(path/'examples.npz',**a);d['pairs_sha256']=file_sha(path/'examples.npz')
        (path/'examples.json').write_text(json.dumps(d))
    patterns=dict(unsupported_new_label='retained original holdouts',cross_fold_alias='links original training',
        changed_q='original shared feature grid',disjoint_ranges='retained reference interval')
    with pytest.raises(ValueError,match=patterns[case]):
        assemble_pairs([read(p,names='examples') for p in paths],bins=128)


def test_wider_interval_with_lower_minimum_still_uses_the_retained_narrow_reference(tmp_path):
    _,paths=inputs(tmp_path)
    for path in paths[2:]:
        target=path/'examples.json';d=json.loads(target.read_text())
        d['parent_relative_metric_range'][0]=.00001;target.write_text(json.dumps(d))
    combined=assemble_pairs([read(p,names='examples') for p in paths],bins=128)
    assert combined['partition_anchor']['original_relative_metric_range']==[.0001,.001]


@pytest.mark.parametrize('case',['missing_candidate','missing_trajectory','different_seed','candidate_changed','grid_changed'])
def test_unbalanced_or_unrelated_ranges_cannot_be_used_to_train(tmp_path,case):
    payload,paths=inputs(tmp_path)
    if case=='missing_candidate':paths=[p for p in paths if not p.name.startswith('wide-second')]
    elif case=='missing_trajectory':paths=[p for p in paths if p.name!='wide-first-4']
    else:
        path=paths[-1]/'examples.json';d=json.loads(path.read_text())
        if case=='different_seed':
            d['parent_seed']=3;d['groups']=[g.replace(':1:2:',':1:3:') for g in d['groups']]
        elif case=='candidate_changed':d['candidate']['child']['path']='another-child.cif'
        else:d['grid']['upper']+=1.
        path.write_text(json.dumps(d))
    with pytest.raises(ValueError):jobs.create_plan(payload['source'],paths,tmp_path/'plan.json')
    assert not (tmp_path/'plan.json').exists()


@pytest.mark.parametrize('case',['missing','duplicated','wrong_range','wrong_catalog','foreign_field','legacy_schema'])
def test_strata_are_closed_and_verified_against_actual_bound_metadata(tmp_path,case):
    payload,paths=inputs(tmp_path);path=tmp_path/'plan.json'
    d=jobs.create_plan(payload['source'],paths,path)
    if case=='missing':d['parent_strata'][0]['blocks'].pop()
    elif case=='duplicated':d['parent_strata'][1]['blocks'].append(d['parent_strata'][0]['blocks'][0])
    elif case=='wrong_range':d['parent_strata'][1]['parent_relative_metric_range'][1]=.006
    elif case=='wrong_catalog':
        for row in d['parent_strata']:row['candidate_ids']=['invented']
    elif case=='foreign_field':d['parent_strata'][0]['python']='not allowed'
    else:d['schema']=jobs.PAIRED_PLAN_SCHEMA;d.pop('parent_strata')
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError):jobs.checked_blocks(path,jobs.read_plan(path),validate_arrays=False)


def test_resume_preserves_every_range_and_keeps_numerical_state_separate(tmp_path):
    from test_structural_training_transfer import owned
    from gpx_recipe.preparation import structural_training_transfer as transfer
    payload,paths=inputs(tmp_path);model=tmp_path/'previous-model';owned(model,warmup=20)
    transfer.snapshot(model,tmp_path/'previous-transfer',identity='a'*64,epochs=4)
    path=tmp_path/'continued.json'
    plan=jobs.create_plan(payload['source'],paths,path,resume=tmp_path/'previous-transfer')
    assert plan['schema']==jobs.STRATIFIED_CONTINUED_PLAN_SCHEMA
    assert len(plan['parent_strata'])==2 and len(plan['blocks'])==len(paths)
    assert jobs.resume_directory(path,plan)==tmp_path/'resume'
    assert jobs.plan_inputs(path)[-2:]==['resume/training-state.npz','resume/training-state.json']
    assert jobs.checked_blocks(path,plan)==paths


def test_preparation_queue_accepts_only_balanced_range_dependencies(tmp_path):
    from gpx_recipe.preparation.operations import create_library
    payload,_=single_inputs(tmp_path)
    models=[]
    for high in [.001,.005]:
        for identifier in ['first','second']:
            settings=copy.deepcopy(payload['settings']);settings['candidate']['id']=identifier
            settings['parent_relative_metric_range']=[.0001,high]
            models.append(dict(kind='structural-examples',bank='bank 1',child=payload['child'],settings=settings))
    options=dict(name='Balanced existing and widened pairs',source=payload['source'],
        banks={'bank 1':dict(instrument=payload['instrument'],grid=payload['grid'])})
    library=create_library(tmp_path/'library',models=models,**options)
    examples=[r for r in library.document['entries'] if r['model']['kind']=='structural-examples']
    training=next(r for r in library.document['entries'] if r['model']['kind']=='structural-training')
    assert len(examples)==4 and training['model']['settings']['dependencies']==sorted(r['id'] for r in examples)
    with pytest.raises(ValueError,match='complete candidate catalog'):
        create_library(tmp_path/'unbalanced',models=models[:-1],**options)


@pytest.mark.parametrize('resume',[False,True])
def test_multipart_handoff_preserves_every_stratum_and_its_resume_state(tmp_path,monkeypatch,resume):
    from gpx_recipe.preparation import structural_input_parts as parts,structural_training_transfer as transfer
    from gpx_recipe.symmetry.compiler import package_signature
    from test_structural_input_parts import packet
    from test_structural_training_transfer import owned
    root=tmp_path/'capture';payload,paths=inputs(root)
    previous=None
    if resume:
        model=root/'previous';owned(model,warmup=20)
        previous=root/'transfer';transfer.snapshot(model,previous,identity='a'*64,epochs=60)
    plan=root/'plan.json';jobs.create_plan(payload['source'],paths,plan,resume=previous)
    names=['parent.cif','profile.instprm','plan.json']+jobs.plan_inputs(plan)
    request=dict(schema='radar-model-native/v1',operation=parts.OPERATION,
        implementation_sha256=package_signature(),expected_source_sha256=file_sha(root/'parent.cif'),
        dispatch_key='d'*24,timeout=120,payload=dict(source='parent.cif',instrument='profile.instprm',
            plan='plan.json',destination='trained',grid=payload['grid'],settings=dict(representation=REPRESENTATION)),
        input_hashes={name:file_sha(root/name) for name in names})
    monkeypatch.setattr(parts,'MAX_PART_BYTES',28000);monkeypatch.setattr(parts,'MAX_METADATA',20000)
    primary,more,_=packet(root,request,tmp_path/'packets')
    assert len(more)>1
    restored=parts.stage(primary,more,request,tmp_path/'fresh-worker')
    actual=jobs.read_plan(restored/'plan.json')
    assert actual['schema']==(jobs.STRATIFIED_CONTINUED_PLAN_SCHEMA if resume else jobs.STRATIFIED_PLAN_SCHEMA)
    assert {str(p.relative_to(restored)):file_sha(p) for p in restored.rglob('*') if p.is_file()}==request['input_hashes']
    assert len(jobs.checked_blocks(restored/'plan.json',actual,grid=payload['grid']))==len(paths)
    if resume:
        receipt,_=transfer.read(jobs.resume_directory(restored/'plan.json',actual),identity='a'*64,epochs=60)
        assert receipt['warmup_completed']==20 and receipt['trained_model'] is False


def test_actual_training_and_cold_continuation_keep_both_ranges_and_identical_optimizer_state(tmp_path,monkeypatch):
    pytest.importorskip('torch')
    from gpx_recipe.preparation import structural_training_segments as segments
    from gpx_recipe.preparation import structural_paired_training as training
    from gpx_recipe.preparation.structural_replay import build
    payload,paths=inputs(tmp_path/'inputs');root=Path(payload['source']).parent
    plan=root/'plan.json';jobs.create_plan(payload['source'],paths,plan)
    request=dict(source=payload['source'],instrument=payload['instrument'],grid=payload['grid'],
        plan=str(plan),destination=str(tmp_path/'first'),
        settings=dict(representation=REPRESENTATION,bins=128,epochs=1,workers=1,augmentation=None))
    cohort=build(paths,bins=128,names='examples')
    try:
        reference=training.train(cohort,tmp_path/'reference',epochs=1,threads=1,_segment_seconds=1e-9)
        reference=training.train(cohort,tmp_path/'reference',epochs=1,threads=1,_segment_seconds=1e-9)
        assert reference['warmup_completed']==40
        monkeypatch.setattr(segments,'SEGMENT_SECONDS',1e-9)
        first=jobs.execute_training(request)
        assert first['warmup_completed']==20 and first['identity']==reference['identity']
        fresh=tmp_path/'fresh';fresh.mkdir();fresh_paths=[]
        for path in paths:
            destination=fresh/path.name;shutil.copytree(path,destination);fresh_paths.append(destination)
        for source,name in [(payload['source'],'parent.cif'),(payload['instrument'],'profile.instprm')]:
            shutil.copyfile(source,fresh/name)
        fresh_plan=fresh/'plan.json'
        jobs.create_plan(fresh/'parent.cif',fresh_paths,fresh_plan,resume=tmp_path/'first/continuation')
        assert jobs.read_plan(fresh_plan)['schema']==jobs.STRATIFIED_CONTINUED_PLAN_SCHEMA
        second=jobs.execute_training(dict(request,source=str(fresh/'parent.cif'),instrument=str(fresh/'profile.instprm'),
            plan=str(fresh_plan),destination=str(tmp_path/'second')))
        assert second['identity']==reference['identity'] and second['warmup_completed']==40
        assert first['dataset']==second['dataset']==reference['dataset']
        with np.load(tmp_path/'reference/training-checkpoint.npz',allow_pickle=False) as left,np.load(
                tmp_path/'second/model/training-checkpoint.npz',allow_pickle=False) as right:
            assert set(left.files)==set(right.files)
            for key in set(left.files)-{'metadata'}:np.testing.assert_array_equal(left[key],right[key],err_msg=key)
            a=json.loads(left['metadata'].tobytes());b=json.loads(right['metadata'].tobytes())
            for key in set(a)-{'elapsed_seconds'}:assert a[key]==b[key],key
        monkeypatch.setattr(segments,'SEGMENT_SECONDS',None)
        full=jobs.execute_training(dict(request,destination=str(tmp_path/'full')))
        assert full['status']=='prepared' and full['candidate_pruning_authorized'] is False
        model=json.loads((tmp_path/'full/model/model.json').read_text())
        assert model['dataset']==reference['dataset'] and model['feature_storage']=='native_pair_replay'
        assert jobs.execute_training(dict(request,destination=str(tmp_path/'full')))==full
    finally:cohort['features'].close()
