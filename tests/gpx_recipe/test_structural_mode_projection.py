"""Complete mode coordinates survive every existing preparation handoff."""
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from gpx_recipe.preparation import structural_mode_projection as projection
from gpx_recipe.preparation import structural_jobs as jobs
from gpx_recipe.preparation.structural_cohort import assemble_pairs
from gpx_recipe.preparation.structural_paired_blocks import read,pack_block,source_paths
from gpx_recipe.symmetry.compiler import file_sha,package_signature
from test_structural_paired_blocks import block
from test_structural_paired_preparation import inputs


def attach(directory,*,names='pairs'):
    """Fixture with one physical direction and an independently kept gauge."""
    directory=Path(directory);paired=read(directory,names=names);d=copy.deepcopy(paired.manifest)
    np.savez_compressed(directory/projection.ARRAY_FILE,physical_to_native=np.array([[2.],[0.]]),
        gauge_basis=np.array([[0.],[1.]]),physical_gram=np.array([[4.]]))
    metadata=dict(schema=projection.SCHEMA,normalization=projection.NORMALIZATION,
        **{k:d[k] for k in ['source_sha256','instrument_sha256','child_sha256','candidate']},
        native_dimensions=2,physical_dimensions=1,native_mode_labels=['physical-irrep','origin'],
        native_normalization=[1.,1.],native_module_sha256={'GSASIIstrMath':'d'*64},
        producer_implementation_sha256='e'*64,geometry_sha256='f'*64,
        arrays_sha256=file_sha(directory/projection.ARRAY_FILE),observations_included=False,
        scientific_result=False,candidate_pruning_authorized=False)
    (directory/projection.METADATA).write_text(json.dumps(metadata))
    stamp=dict(schema=projection.SCHEMA,metadata_sha256=file_sha(directory/projection.METADATA),
        arrays_sha256=metadata['arrays_sha256'])
    d.update(schema=projection.PACKED_SCHEMA if 'array_encoding' in d else projection.RAW_SCHEMA,
        mode_projection=stamp)
    (directory/(names+'.json')).write_text(json.dumps(d))
    return read(directory,names=names)


def rewrite_projection(path,change=None,arrays=None):
    d=json.loads((path/projection.METADATA).read_text())
    if arrays:
        with np.load(path/projection.ARRAY_FILE,allow_pickle=False) as a:values={k:a[k] for k in a.files}
        arrays(values);np.savez_compressed(path/projection.ARRAY_FILE,**values)
        d['arrays_sha256']=file_sha(path/projection.ARRAY_FILE)
    if change:change(d)
    (path/projection.METADATA).write_text(json.dumps(d))
    return d


def test_complete_native_map_preserves_gauge_units_and_read_only_arrays(tmp_path):
    path=block(tmp_path/'source');paired=attach(path);value=paired.mode_projection
    assert value.metadata['native_mode_labels']==['physical-irrep','origin']
    assert value.metadata['physical_dimensions']==1 and value.metadata['native_dimensions']==2
    np.testing.assert_array_equal(value.arrays['physical_to_native'],[[2.],[0.]])
    assert projection.read(path,expected=paired.manifest,stamp=value.stamp).stamp==value.stamp
    assert source_paths(path)==tuple(path/n for n in ('pairs.json','pairs.npz',*projection.FILES))
    for a in value.arrays.values():
        with pytest.raises(ValueError):a.flags.writeable=True
    before={n:(path/n).read_bytes() for n in projection.FILES}
    with pytest.raises(ValueError,match='immutable'):projection.copy(path,path)
    assert before=={n:(path/n).read_bytes() for n in projection.FILES}


@pytest.mark.parametrize('case',['hash','flag','child','candidate','label','normalization','native_module',
    'extra','rank','gram','dtype','gauge','dimension','symlink','missing'])
def test_hashes_cannot_override_the_mode_coordinate_contract(tmp_path,case):
    path=block(tmp_path/'source');paired=attach(path)
    if case=='hash':
        with (path/projection.ARRAY_FILE).open('ab') as stream:stream.write(b'changed')
    elif case=='symlink':
        target=path/'actual.npz';(path/projection.ARRAY_FILE).rename(target)
        (path/projection.ARRAY_FILE).symlink_to(target)
    elif case=='missing':(path/projection.METADATA).unlink()
    elif case in {'rank','gram','dtype','gauge','dimension'}:
        def change(a):
            if case=='rank':a['physical_to_native'][:]=0
            elif case=='gram':a['physical_gram'][:]=-1
            elif case=='dtype':a['physical_to_native']=a['physical_to_native'].astype(np.float32)
            elif case=='gauge':a['gauge_basis']=a['physical_to_native'].copy()
            else:a['physical_gram']=np.eye(2)
        rewrite_projection(path,arrays=change)
    else:
        def change(d):
            if case=='flag':d['candidate_pruning_authorized']=True
            elif case=='child':d['child_sha256']='9'*64
            elif case=='candidate':d['candidate']['origin_shift']=[.5,0,0]
            elif case=='label':d['native_mode_labels'].pop()
            elif case=='normalization':d['native_normalization'][0]=0
            elif case=='native_module':d['native_module_sha256']={'../code':'d'*64}
            else:d['unexpected']='not an authority'
        rewrite_projection(path,change=change)
    with pytest.raises((ValueError,OSError)):
        projection.read(path,expected=paired.manifest)


def test_pair_repacking_keeps_every_old_array_bit_and_exact_native_maps(tmp_path):
    path=block(tmp_path/'source');old=read(path);mapped=attach(path)
    result=pack_block(path,tmp_path/'packed',names='examples')
    assert result.manifest['schema']==projection.PACKED_SCHEMA
    assert result.manifest['groups']==old.manifest['groups']
    for key,value in old.arrays.items():
        assert result.arrays[key].tobytes()==mapped.arrays[key].tobytes()==value.tobytes()
    for name in projection.FILES:assert (path/name).read_bytes()==(tmp_path/'packed'/name).read_bytes()
    d=json.loads((path/'pairs.json').read_text());d['schema']='radar-structural-ranking-paired-pilot/v1'
    (path/'pairs.json').write_text(json.dumps(d))
    with pytest.raises(ValueError,match='explicit paired schema'):read(path)


def test_projected_plan_binds_optional_maps_without_relabelling_old_blocks(tmp_path):
    request,path=inputs(tmp_path);attach(path,names='examples')
    legacy=tmp_path/'legacy';shutil.copytree(path,legacy)
    d=json.loads((legacy/'examples.json').read_text());d.pop('mode_projection')
    d['schema']='radar-structural-ranking-pairs/v1';(legacy/'examples.json').write_text(json.dumps(d))
    for name in projection.FILES:(legacy/name).unlink()
    target=tmp_path/'plan.json';plan=jobs.create_plan(request['source'],[path,legacy],target)
    assert plan['schema']==jobs.PROJECTED_PLAN_SCHEMA and plan['parent_strata'] is None
    assert plan['blocks'][0]['mode_projection']==read(path,names='examples').mode_projection.stamp
    assert plan['blocks'][1]['mode_projection'] is None
    assert jobs.read_plan(target)==plan
    assert jobs.checked_blocks(target,plan)==[path,legacy]
    assert len(jobs.plan_inputs(target))==6
    assert jobs.checked_blocks(target,plan,validate_arrays=False)==[path,legacy]
    (path/projection.METADATA).write_text((path/projection.METADATA).read_text()+' ')
    with pytest.raises(ValueError):jobs.checked_blocks(target,plan)


def test_projection_files_join_the_packet_limit_before_publishing_a_plan(tmp_path,monkeypatch):
    request,path=inputs(tmp_path);attach(path,names='examples');target=tmp_path/'plan.json'
    monkeypatch.setattr(jobs,'MAX_INPUT_FILES',6)
    with pytest.raises(ValueError,match='file inventory'):jobs.create_plan(request['source'],[path],target)
    assert not target.exists()


@pytest.mark.parametrize('storage',['dense','file_backed','native_pair_replay'])
def test_map_metadata_never_changes_curves_classes_or_holdouts(tmp_path,storage):
    from gpx_recipe.preparation import structural_paired_store,structural_replay
    paths=[block(tmp_path/'one'),block(tmp_path/'two')]
    before=assemble_pairs([read(p) for p in paths],bins=128)
    attach(paths[0])
    if storage=='dense':after=assemble_pairs([read(p) for p in paths],bins=128)
    elif storage=='file_backed':after=structural_paired_store.build(paths,tmp_path/'store',bins=128)
    else:after=structural_replay.build(paths,bins=128)
    try:
        from gpx_recipe.preparation.structural_training_data import feature_rows
        np.testing.assert_array_equal(feature_rows(after['features']).take(np.arange(len(before['features']))),before['features'])
        for key in ['compatible_labels','classes','split','reference_features']:
            np.testing.assert_array_equal(before[key],after[key])
        assert after['mode_projection_coverage']==dict(mapped_blocks=1,total_blocks=2,mapped_candidates=1,
            total_candidates=1,complete=False,mode_targets_inferred=False)
        assert after['mode_projection_references'][1] is None
        assert after['groups']==before['groups']
    finally:
        if storage!='dense':after['features'].close()


@pytest.mark.parametrize('name',projection.FILES)
def test_replay_checks_mode_files_even_when_all_curves_are_cached(tmp_path,name):
    from gpx_recipe.preparation.structural_replay import build
    path=block(tmp_path/'source');attach(path);rows=build([path],bins=128)['features']
    try:
        rows.take(np.array([0]))
        with (path/name).open('ab') as stream:stream.write(b' ')
        with pytest.raises(ValueError,match='sources changed'):rows.take(np.array([0]))
    finally:rows.close()


def test_checkpoint_identity_cannot_reuse_other_mode_coordinates_for_identical_curves(tmp_path):
    from gpx_recipe.preparation.structural_training_checkpoint import identity
    from gpx_recipe.preparation.structural_training_data import feature_rows
    path=block(tmp_path/'source');attach(path)
    before=assemble_pairs([read(path)],bins=128)
    def key(cohort):
        clean=np.ones(len(cohort['features']),bool)
        return identity(cohort,feature_rows(cohort['features']),cohort['compatible_labels'],cohort['split'],
            clean,np.zeros((3,1),np.float32),np.ones((3,1),np.float32),epochs=1,seed=1,workers=1)
    first=key(before)
    rewrite_projection(path,change=lambda d:d.update(geometry_sha256='8'*64))
    d=json.loads((path/'pairs.json').read_text())
    d['mode_projection']=projection.read(path).stamp;(path/'pairs.json').write_text(json.dumps(d))
    after=assemble_pairs([read(path)],bins=128)
    np.testing.assert_array_equal(before['features'],after['features'])
    assert before['input_sha256']==after['input_sha256'] and key(after)!=first


def test_library_export_import_and_training_handoff_keep_all_mode_files(tmp_path):
    from gpx_recipe.preparation.operations import create_library,stage_entry,publish_entry
    from gpx_recipe.preparation.library import import_library
    from gpx_recipe.preparation.structural_dependencies import verify
    request,path=inputs(tmp_path);paired=attach(path,names='examples')
    library=create_library(tmp_path/'library',name='Coordinates retained',source=request['source'],
        banks={'bank':dict(instrument=request['instrument'],grid=request['grid'])},
        models=[dict(kind='structural-examples',bank='bank',child=request['child'],settings=request['settings'])])
    row=next(r for r in library.document['entries'] if r['model']['kind']=='structural-examples')
    attempt=library.start(row['id']);staged=stage_entry(library,row['id'],attempt,tmp_path/'example')
    shutil.copytree(path,staged['destination'])
    detail=dict(seconds=.1,mode_projection=projection.descriptor(paired.mode_projection))
    result=jobs.publish_result('structural-examples',staged,detail,['examples.json','examples.npz',*projection.FILES])
    publish_entry(library,row['id'],attempt,staged,result)
    library.export(tmp_path/'library.zip');restored=import_library(tmp_path/'library.zip',tmp_path/'restored')
    assert restored.document==library.document
    train=next(r for r in restored.document['entries'] if r['model']['kind']=='structural-training')
    attempt=restored.start(train['id']);payload=stage_entry(restored,train['id'],attempt,tmp_path/'training')
    plan=jobs.read_plan(payload['plan']);assert plan['schema']==jobs.PROJECTED_PLAN_SCHEMA
    assert len(jobs.plan_inputs(payload['plan']))==4
    verify(restored,train['model'],payload)


def test_native_size_reserves_complete_projection_without_widening_artifact_cap(tmp_path):
    from gpx_recipe.preparation import native_size
    request,_=inputs(tmp_path)
    size=native_size.paired_array_bytes(17,101,1,2)
    native_size.publish(request,rows=17,bins=101,mode_dimensions=1,cell_dimensions=2,
        numeric_bytes=size,native_mode_dimensions=2)
    record=native_size.read(request,package_signature())
    assert record['schema']==native_size.PROJECTED_SCHEMA
    assert record['reservation_bytes']==size+8*(2*2+1)+native_size.OVERHEAD
    assert record['reservation_bytes']<=native_size.EXAMPLE_RESERVE


def test_projected_plan_continuation_preserves_map_identities(tmp_path):
    from test_structural_training_continuation import partial
    request,path=inputs(tmp_path);attach(path,names='examples')
    plan_path=tmp_path/'plan.json';first=jobs.create_plan(request['source'],[path],plan_path)
    payload=dict(source=request['source'],instrument=request['instrument'],grid=request['grid'],
        plan=str(plan_path),destination=str(tmp_path/'worker'),settings=dict(representation=jobs.REPRESENTATION))
    from gpx_recipe.preparation.worker import validate_request
    payload=validate_request('structural-training',payload);partial(payload)
    continued=tmp_path/'continued.json'
    result=jobs.create_plan(request['source'],[path],continued,resume=tmp_path/'worker/continuation')
    assert result['schema']==jobs.PROJECTED_CONTINUED_PLAN_SCHEMA
    assert result['blocks']==first['blocks'] and result['parent_strata'] is None
    assert jobs.read_plan(continued)==result and jobs.checked_blocks(continued,result)==[path]
    assert len(jobs.plan_inputs(continued))==6


def test_multipart_transport_captures_projection_files_in_the_same_typed_plan(tmp_path,monkeypatch):
    from gpx_recipe.preparation import structural_input_parts as parts
    from test_structural_input_parts import captured,packet
    root=tmp_path/'capture';request,_=captured(root,monkeypatch)
    for path in [root/'examples',root/'second']:attach(path,names='examples')
    plan=root/'projected.json';jobs.create_plan(root/'parent.cif',[root/'examples',root/'second'],plan)
    request['payload']['plan']=plan.name
    names=['parent.cif','profile.instprm',plan.name]+jobs.plan_inputs(plan)
    request['input_hashes']={name:file_sha(root/name) for name in names}
    primary,more,_=packet(root,request,tmp_path/'packets')
    target=parts.stage(primary,more,request,tmp_path/'worker-inputs')
    assert {str(p.relative_to(target)):file_sha(p) for p in target.rglob('*') if p.is_file()}==request['input_hashes']
    assert len(jobs.checked_blocks(target/plan.name,jobs.read_plan(target/plan.name)))==2
