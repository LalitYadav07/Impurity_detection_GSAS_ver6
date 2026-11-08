"""Default paired preparation, immutable replay and complete context binding."""
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from gpx_recipe.preparation.structural_dependencies import example_models,EXAMPLES,TRAINING
from gpx_recipe.preparation.structural_jobs import (
    validate_examples,validate_training,create_plan,read_plan,checked_blocks,publish_result,
    PAIRED_PLAN_SCHEMA,
)
from gpx_recipe.preparation.structural_paired_blocks import read,pack_block
from gpx_recipe.preparation.structural_training_blocks import REPRESENTATION,PARENT_RANGE
from gpx_recipe.preparation.worker import validate_request
from gpx_recipe.preparation.operations import create_library,stage_entry,publish_entry,pending_entries
from gpx_recipe.preparation.library import import_library
from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.symmetry.contracts import validate_candidate
from test_structural_paired_blocks import block
from test_structural_training_jobs import inputs as legacy_inputs


def inputs(root):
    root.mkdir(parents=True,exist_ok=True)
    parent=root/'parent.cif';parent.write_text('data_paired_contract\n')
    profile=root/'profile.instprm';profile.write_text('# synthetic profile contract\n')
    child=root/'child.cif';child.write_text('data_synthetic_child\n')
    grid=dict(kind='PNT',lower=3000.,upper=40000.,bins=101,bank=1)
    def metadata(d):
        old=d['source_sha256'];d.update(source_sha256=file_sha(parent),instrument_sha256=file_sha(profile),
            child_sha256=file_sha(child),grid=grid,parent_relative_metric_range=list(PARENT_RANGE))
        d['candidate']['child']['sha256']=file_sha(child)
        d['candidate']=validate_candidate(d['candidate'])
        d['groups']=[g.replace(old,d['source_sha256']) for g in d['groups']]
    raw=block(root/'raw',metadata=metadata)
    value=pack_block(raw,root/'examples',names='examples')
    settings=dict(candidate=value.manifest['candidate'],mode_radius=.05,cell_radius=.002,
        seed=1,parent_seed=2,parent_relative_metric_range=list(PARENT_RANGE),
        representation=REPRESENTATION,trajectories=8,steps=1,trajectory_slice=[0,8])
    request=dict(source=str(parent),instrument=str(profile),child=str(child),grid=grid,
        destination=str(root/'generated'),settings=settings)
    return validate_examples(request),root/'examples'


def test_app_default_expands_to_three_curves_with_shared_parent_sampling():
    candidate=dict(id='one',transformation_matrix=np.eye(3).tolist(),origin_shift=[0,0,0],
        child=dict(path='child.cif',sha256='c'*64))
    models=[dict(kind='structural',bank='bank 1',child='child.cif',
        settings=dict(candidate=candidate,mode_radius=.05,cell_radius=.002))]
    before=copy.deepcopy(models);rows=example_models(models,trajectories=8,shard=4,steps=2)
    assert models==before and len(rows)==2
    for row in rows:
        assert row['kind']==EXAMPLES and row['settings']['representation']==REPRESENTATION
        assert row['settings']['parent_relative_metric_range']==list(PARENT_RANGE)
        validate_examples(dict(source='parent.cif',instrument='profile.instprm',child='child.cif',
            grid=dict(kind='PNT',lower=3000.,upper=40000.,bins=101,bank=1),
            settings=row['settings'],destination='output'))
    assert rows[0]['settings']['parent_seed']==rows[1]['settings']['parent_seed']
    assert [r['settings']['trajectory_slice'] for r in rows]==[[0,4],[4,8]]


@pytest.mark.parametrize('modes,bins,unit',[(27,4096,8),(228,4096,2),(237,4096,2),
    (130,4096,3),(228,8192,1),(None,4096,2)])
def test_automatic_work_units_keep_every_original_trajectory_and_coordinate(modes,bins,unit):
    candidate=dict(id='one',modes=None if modes is None else [f'm{i}' for i in range(modes)])
    model=dict(kind='structural',bank='bank 1',child='original-child.cif',
        settings=dict(candidate=candidate,mode_radius=.05,cell_radius=.01))
    original=copy.deepcopy(model)
    rows=example_models([model],bank_grids={'bank 1':dict(bins=bins)})
    assert model==original
    intervals=[row['settings']['trajectory_slice'] for row in rows]
    assert intervals==[[i,min(i+unit,64)] for i in range(0,64,unit)]
    assert [i for a,b in intervals for i in range(a,b)]==list(range(64))
    for row in rows:
        assert row['settings']['candidate']==candidate
        assert row['settings']['trajectories']==64 and row['settings']['steps']==4
        assert row['settings']['mode_radius']==.05 and row['settings']['cell_radius']==.01
    assert len({row['settings']['seed'] for row in rows})==1
    assert len({row['settings']['parent_seed'] for row in rows})==1


@pytest.mark.parametrize('grid',[None,'unavailable',dict(bins=True),dict(bins=0),dict(bins=4096.5)])
def test_missing_or_malformed_resolution_keeps_all_trajectories_in_small_units(grid):
    model=dict(kind='structural',bank='one',settings=dict(candidate=dict(modes=['m']*228)))
    rows=example_models([model],bank_grids={'one':grid})
    assert [i for row in rows for i in range(*row['settings']['trajectory_slice'])]==list(range(64))
    assert all(row['settings']['trajectory_slice'][1]-row['settings']['trajectory_slice'][0]<=2 for row in rows)


def test_paired_names_keep_lossless_arrays_and_default_training_contract(tmp_path):
    request,directory=inputs(tmp_path)
    raw=read(tmp_path/'raw');paired=read(directory,names='examples')
    for key in raw.arrays:np.testing.assert_array_equal(raw.arrays[key],paired.arrays[key])
    settings=validate_training(dict(source=request['source'],instrument=request['instrument'],
        grid=request['grid'],plan='plan.json',destination='trained',
        settings=dict(representation=REPRESENTATION)))['settings']
    assert settings['bins']==4096 and settings['split_seed']==69271
    assert settings['representation']==REPRESENTATION


@pytest.mark.parametrize('case',['wrong_representation','missing_seed','range_zero','range_too_large','foreign_fields'])
def test_invalid_paired_sampling_is_rejected_before_dispatch(tmp_path,case):
    request,_=inputs(tmp_path);s=request['settings']
    if case=='wrong_representation':s['representation']='eight-optional-curves'
    elif case=='missing_seed':del s['parent_seed']
    elif case=='range_zero':s['parent_relative_metric_range']=[0,.001]
    elif case=='range_too_large':s['parent_relative_metric_range']=[.001,1.]
    else:del s['representation']
    with pytest.raises(ValueError):validate_request(EXAMPLES,request)


def test_paired_plan_replay_binds_format_parent_scope_and_bytes(tmp_path):
    request,directory=inputs(tmp_path)
    path=tmp_path/'plan.json';plan=create_plan(request['source'],[directory],path)
    assert plan['schema']==PAIRED_PLAN_SCHEMA and plan['representation']==REPRESENTATION
    assert create_plan(request['source'],[directory],path,recover_identical=True)==plan
    assert checked_blocks(path,read_plan(path))==[directory]
    assert checked_blocks(path,read_plan(path),validate_arrays=False)==[directory]
    changed=copy.deepcopy(plan);changed.pop('representation');changed['schema']='radar-structural-training-plan/v1'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='representation'):checked_blocks(path,read_plan(path))


def test_old_saved_training_settings_and_plan_are_not_silently_migrated(tmp_path):
    request,directories=legacy_inputs(tmp_path/'old')
    value=validate_training(request)
    assert 'representation' not in value['settings'] and value['settings']['bins']==128
    assert 'representation' not in read_plan(request['plan'])
    assert checked_blocks(request['plan'],read_plan(request['plan']))==directories


@pytest.mark.parametrize('case',['mixed','parent_seed'])
def test_paired_training_cannot_mix_old_features_or_unrelated_parent_trajectories(tmp_path,case):
    request,directory=inputs(tmp_path/'paired')
    if case=='mixed':
        _,others=legacy_inputs(tmp_path/'old')
        other=others[0]
    else:
        other=tmp_path/'paired/other';shutil.copytree(directory,other)
        path=other/'examples.json';d=json.loads(path.read_text());d['parent_seed']=3
        d['groups']=[g.replace(':1:2:',':1:3:') for g in d['groups']];path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='representation and parent trajectory'):
        create_plan(request['source'],[directory,other],tmp_path/'plan.json')


def completed_library(root):
    request,directory=inputs(root)
    library=create_library(root/'library',name='Paired default preparation',source=request['source'],
        banks={'bank 1':dict(instrument=request['instrument'],grid=request['grid'])},
        models=[dict(kind=EXAMPLES,bank='bank 1',child=request['child'],settings=request['settings'])])
    row=next(r for r in library.document['entries'] if r['model']['kind']==EXAMPLES)
    attempt=library.start(row['id']);staged=stage_entry(library,row['id'],attempt,root/'example-attempt')
    out=Path(staged['destination']);shutil.copytree(directory,out)
    result=publish_result(EXAMPLES,staged,dict(seconds=.1),['examples.json','examples.npz'])
    publish_entry(library,row['id'],attempt,staged,result)
    return library


def test_library_stages_and_recovers_three_curve_training_without_losing_any_dependency(tmp_path):
    library=completed_library(tmp_path)
    row=next(r for r in library.document['entries'] if r['model']['kind']==TRAINING)
    assert row['model']['settings']['representation']==REPRESENTATION
    assert pending_entries(library)==[row['id']]
    attempt=library.start(row['id'])
    first=stage_entry(library,row['id'],attempt,tmp_path/'training-attempt')
    assert first==stage_entry(library,row['id'],attempt,tmp_path/'training-attempt')
    assert read_plan(first['plan'])['representation']==REPRESENTATION
    library.export(tmp_path/'library.zip')
    restored=import_library(tmp_path/'library.zip',tmp_path/'restored')
    assert restored.document==library.document


def test_library_rejects_a_saved_candidate_with_different_parent_sampling(tmp_path):
    from gpx_recipe.preparation.structural_dependencies import validate_entries
    library=completed_library(tmp_path);entries=copy.deepcopy(library.document['entries'])
    example=next(r for r in entries if r['model']['kind']==EXAMPLES)
    duplicate=copy.deepcopy(example);duplicate['id']='f'*24
    duplicate['model']['settings']['parent_seed']+=1;entries.append(duplicate)
    training=next(r for r in entries if r['model']['kind']==TRAINING)
    training['model']['settings']['dependencies']=sorted([example['id'],duplicate['id']])
    with pytest.raises(ValueError,match='shared parent trajectories'):validate_entries(entries)
