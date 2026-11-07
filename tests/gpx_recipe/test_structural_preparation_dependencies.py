"""Finite training dependencies, replay and portability; fixture curves are synthetic."""
import copy
import json
from pathlib import Path
import shutil

import pytest

from gpx_recipe.preparation.operations import create_library,pending_entries,stage_entry,publish_entry
from gpx_recipe.preparation.structural_dependencies import EXAMPLES,TRAINING
from gpx_recipe.preparation.structural_jobs import publish_result
from gpx_recipe.preparation.library import import_library
from gpx_recipe.symmetry.compiler import file_sha
from test_structural_training_jobs import inputs


def library_inputs(root):
    request,directories=inputs(root)
    child=root/'child.cif';child.write_text('data_public_dependency_contract\n')
    models=[]
    for directory in directories:
        d=json.loads((directory/'examples.json').read_text());candidate=d['candidate']
        candidate['child']['sha256']=file_sha(child)
        models.append(dict(kind=EXAMPLES,bank='bank 1',child=child,
            settings=dict(candidate=candidate,mode_radius=.05,cell_radius=.002,
                seed=1947,trajectories=6,steps=2,trajectory_slice=[0,6])))
    library=create_library(root/'library',name='Synthetic dependency control',source=request['source'],
        banks={'bank 1':dict(instrument=request['instrument'],grid=request['grid'])},models=models)
    originals={json.loads((p/'examples.json').read_text())['candidate']['id']:p for p in directories}
    return library,originals


def complete_examples(library,originals,directory):
    for row in library.document['entries']:
        if row['model']['kind']!=EXAMPLES:continue
        identifier=row['id'];attempt=library.start(identifier)
        request=stage_entry(library,identifier,attempt,directory/identifier)
        out=Path(request['destination']);out.mkdir()
        original=originals[request['settings']['candidate']['id']]
        shutil.copyfile(original/'examples.npz',out/'examples.npz')
        d=json.loads((original/'examples.json').read_text())
        d.update(candidate=request['settings']['candidate'],child_sha256=file_sha(request['child']),
            grid=request['grid'],trajectory_slice=[0,6],generated_trajectories=6)
        (out/'examples.json').write_text(json.dumps(d))
        result=publish_result(EXAMPLES,request,dict(seconds=.1),['examples.json','examples.npz'])
        publish_entry(library,identifier,attempt,request,result)


def test_training_waits_for_every_example_and_stages_identical_inputs_on_replay(tmp_path):
    library,originals=library_inputs(tmp_path)
    examples=[row for row in library.document['entries'] if row['model']['kind']==EXAMPLES]
    training=next(row for row in library.document['entries'] if row['model']['kind']==TRAINING)
    assert training['model']['settings']['dependencies']==sorted(row['id'] for row in examples)
    assert set(pending_entries(library))=={row['id'] for row in examples}
    complete_examples(library,originals,tmp_path/'attempts')
    assert pending_entries(library)==[training['id']]
    attempt=library.start(training['id'])
    first=stage_entry(library,training['id'],attempt,tmp_path/'training-attempt')
    second=stage_entry(library,training['id'],attempt,tmp_path/'training-attempt')
    assert first==second
    plan=json.loads(Path(first['plan']).read_text())
    assert len(plan['blocks'])==2
    assert [row['directory'] for row in plan['blocks']]==[
        'blocks/'+key for key in training['model']['settings']['dependencies']]
    library.export(tmp_path/'prepared-inputs.zip')
    restored=import_library(tmp_path/'prepared-inputs.zip',tmp_path/'restored')
    assert restored.document==library.document


@pytest.mark.parametrize('change',['missing_slice','overlap','missing_dependency'])
def test_incomplete_or_ambiguous_training_scope_cannot_be_persisted(tmp_path,change):
    from gpx_recipe.preparation.library import PreparationLibrary
    from gpx_recipe.symmetry.contracts import digest
    library,_=library_inputs(tmp_path)
    entries=copy.deepcopy(library.document['entries'])
    example=next(row for row in entries if row['model']['kind']==EXAMPLES)
    training=next(row for row in entries if row['model']['kind']==TRAINING)
    if change=='missing_dependency':training['model']['settings']['dependencies']=[]
    else:
        old=example['id']
        example['model']['settings']['trajectory_slice']=[1,6] if change=='missing_slice' else [0,5]
        example['id']=digest(example['model'])[:24]
        training['model']['settings']['dependencies']=sorted(
            example['id'] if key==old else key for key in training['model']['settings']['dependencies'])
        if change=='overlap':
            extra=copy.deepcopy(example);extra['model']['settings']['trajectory_slice']=[4,6]
            extra['id']=digest(extra['model'])[:24];entries.append(extra)
            training['model']['settings']['dependencies']=sorted(training['model']['settings']['dependencies']+[extra['id']])
    training['id']=digest(training['model'])[:24]
    with pytest.raises(ValueError,match='every declared|missing or overlapping|complete declared'):
        PreparationLibrary.create(tmp_path/'invalid',name='Invalid sample scope',
            specification={'scope':'control'},models=[row['model'] for row in entries],
            inputs={'parent.cif':tmp_path/'parent.cif'})


def test_training_cannot_substitute_a_different_dependency_array(tmp_path):
    library,originals=library_inputs(tmp_path);complete_examples(library,originals,tmp_path/'attempts')
    training=next(row for row in library.document['entries'] if row['model']['kind']==TRAINING)
    attempt=library.start(training['id'])
    request=stage_entry(library,training['id'],attempt,tmp_path/'training-attempt')
    dependency=next(Path(request['plan']).parent.glob('blocks/*/examples.npz'))
    dependency.write_bytes(dependency.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='dependency changed'):
        stage_entry(library,training['id'],attempt,tmp_path/'training-attempt')
