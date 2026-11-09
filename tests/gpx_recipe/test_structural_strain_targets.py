"""Typed strain/displacement targets survive the existing shared pipeline."""
import copy
import json
import shutil

import numpy as np
import pytest

from gpx_recipe.preparation import structural_mode_projection as projection
from gpx_recipe.preparation.structural_strain_cif import from_metadata
from gpx_recipe.preparation.structural_mode_families import TYPED_NORMALIZATION
from gpx_recipe.preparation.structural_paired_blocks import read,pack_block
from gpx_recipe.preparation.structural_cohort import assemble_pairs
from gpx_recipe.preparation.structural_family_targets import build
from gpx_recipe.symmetry.compiler import file_sha
from test_structural_family_targets import mapped
from test_structural_strain_cif import model as strain_fixture


def attach_strain(path,tmp_path,*,names='pairs'):
    original=read(path,names=names);d=copy.deepcopy(original.manifest);a={k:v.copy() for k,v in original.arrays.items()}
    metadata=json.loads((path/projection.METADATA).read_text())
    value=strain_fixture(tmp_path);candidate=copy.deepcopy(d['candidate'])
    candidate.update(transformation_matrix=[[1.,1.,0.],[-1.,1.,0.],[0.,0.,1.]],origin_shift=[0.,.5,0.])
    d['candidate']=candidate;metadata['candidate']=candidate
    strain=dict(value.metadata,parent_sha256=d['source_sha256'],child_sha256=d['child_sha256'],candidate=candidate)
    compiled=from_metadata(strain)
    mapping=np.zeros((6,2));mapping[0,0]=1;mapping[1,0]=-1;mapping[2,1]=1
    metadata.update(schema=projection.STRAIN_SCHEMA,strain=dict(map=compiled.metadata,
        baseline_a=[1/32,1/32,1/81,0.,0.,0.],cell_mapping=mapping.tolist()))
    a['parent_a'][:]=[1/16,1/16,1/81,0.,0.,0.]
    cold=a['role']==2;a['parameters'][cold,1]=.0001*(1+a['trajectory'][cold]/10)
    # A packed fixture is rewritten to a raw mapped block for this authored
    # geometry test; production repacking is exercised separately below.
    d['schema']=projection.RAW_SCHEMA;d.pop('array_encoding',None)
    np.savez_compressed(path/(names+'.npz'),**a);d['pairs_sha256']=file_sha(path/(names+'.npz'))
    (path/projection.METADATA).write_text(json.dumps(metadata))
    d['mode_projection']=projection.read(path).stamp
    (path/(names+'.json')).write_text(json.dumps(d))
    return read(path,names=names)


def target_cohort(path):
    value=assemble_pairs([read(path)],bins=128)
    table=build(lambda:iter([read(path)]),value)
    value.update(mode_family_targets=table,mode_family_target_identity=table.metadata['identity'],mode_family_supervision=table.metadata)
    return value,table


def test_native_cell_coordinates_add_strain_without_changing_displacement_units(tmp_path):
    path=mapped(tmp_path/'source');block=attach_strain(path,tmp_path)
    cohort,table=target_cohort(path);table.verify(cohort)
    assert table.metadata['normalization']==TYPED_NORMALIZATION
    families=table.metadata['families'];assert len(families)==3
    dsp=next(i for i,r in enumerate(families) if 'quantity' not in r)
    strain=next(i for i,r in enumerate(families) if r['irrep']=='GM4+')
    row=next(i for i,t in enumerate(cohort['compatible_labels']) if t[0])
    batch=table.batch(np.array([row]));by={int(f):(lo,hi) for f,lo,hi in zip(batch['family'],batch['lower'],batch['upper'])}
    assert by[dsp][0]==pytest.approx(.2) and by[strain][0]>0
    # Independent diagonal-cell formula for the first authored cold row.
    point=block.arrays['parameters'][2,1];x=np.sqrt((1/32)/(1/32+point))-1;y=np.sqrt((1/32)/(1/32-point))-1
    assert by[strain][0]==pytest.approx(abs(x-y)/np.sqrt(2),rel=1e-9)
    assert families[strain]['units']=='dimensionless' and families[strain]['id']!=families[dsp]['id']
    assert not table.metadata['candidate_pruning_authorized']


def test_existing_packet_repacking_preserves_all_strain_coordinates_and_rows(tmp_path):
    path=mapped(tmp_path/'source');block=attach_strain(path,tmp_path)
    copied=pack_block(path,tmp_path/'packed')
    assert copied.mode_projection.stamp==block.mode_projection.stamp
    for k,v in block.arrays.items():assert copied.arrays[k].tobytes()==v.tobytes()
    assert copied.mode_projection.metadata['strain']==block.mode_projection.metadata['strain']


def test_checked_strain_addition_reuses_every_curve_without_replacing_old_displacements(tmp_path):
    path=mapped(tmp_path/'source');upgraded=attach_strain(path,tmp_path)
    legacy=tmp_path/'legacy';shutil.copytree(path,legacy)
    d=copy.deepcopy(upgraded.mode_projection.metadata);d.pop('strain');d['schema']=projection.SCHEMA
    (legacy/projection.METADATA).write_text(json.dumps(d))
    manifest=copy.deepcopy(upgraded.manifest);manifest['mode_projection']=projection.read(legacy).stamp
    (legacy/'pairs.json').write_text(json.dumps(manifest))
    original=read(legacy)
    result=projection.bind_strain(legacy,tmp_path/'added',path,names='pairs',stamp=upgraded.mode_projection.stamp)
    assert result.mode_projection.stamp==upgraded.mode_projection.stamp
    assert result.manifest['mode_projection_source_manifest_sha256']==file_sha(legacy/'pairs.json')
    for k,v in original.arrays.items():assert result.arrays[k].tobytes()==v.tobytes()
    with pytest.raises(ValueError,match='Append strain once'):
        projection.bind_strain(path,tmp_path/'again',path,names='pairs',stamp=upgraded.mode_projection.stamp)
    changed=copy.deepcopy(d);changed['geometry_sha256']='9'*64
    (legacy/projection.METADATA).write_text(json.dumps(changed))
    manifest['mode_projection']=projection.read(legacy).stamp
    (legacy/'pairs.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='original native displacement geometry'):
        projection.bind_strain(legacy,tmp_path/'wrong',path,names='pairs',stamp=upgraded.mode_projection.stamp)


def test_typed_library_download_upload_and_training_handoff_preserve_strain_maps(tmp_path):
    from gpx_recipe.preparation import structural_jobs as jobs
    from gpx_recipe.preparation.operations import create_library,stage_entry,publish_entry
    from gpx_recipe.preparation.library import import_library
    from gpx_recipe.preparation.structural_dependencies import verify
    from test_structural_paired_preparation import inputs
    from test_structural_mode_projection import attach
    request,path=inputs(tmp_path/'input');attach(path,names='examples')
    paired=attach_strain(path,tmp_path,names='examples')
    request['settings']['candidate']=paired.manifest['candidate']
    library=create_library(tmp_path/'library',name='Typed strain controls',source=request['source'],
        banks={'bank':dict(instrument=request['instrument'],grid=request['grid'])},
        models=[dict(kind='structural-examples',bank='bank',child=request['child'],settings=request['settings'])])
    row=next(r for r in library.document['entries'] if r['model']['kind']=='structural-examples')
    attempt=library.start(row['id']);staged=stage_entry(library,row['id'],attempt,tmp_path/'example')
    shutil.copytree(path,staged['destination'])
    detail=dict(seconds=.1,mode_projection=projection.descriptor(paired.mode_projection))
    result=jobs.publish_result('structural-examples',staged,detail,['examples.json','examples.npz',*projection.FILES])
    publish_entry(library,row['id'],attempt,staged,result)
    library.export(tmp_path/'library.zip');restored=import_library(tmp_path/'library.zip',tmp_path/'restored')
    train=next(r for r in restored.document['entries'] if r['model']['kind']=='structural-training')
    attempt=restored.start(train['id']);payload=stage_entry(restored,train['id'],attempt,tmp_path/'training')
    plan=jobs.read_plan(payload['plan']);assert plan['blocks'][0]['mode_projection']['schema']==projection.STRAIN_SCHEMA
    verify(restored,train['model'],payload)
    paths=jobs.checked_blocks(payload['plan'],plan)
    cohort=assemble_pairs([read(p,names='examples') for p in paths],bins=128)
    targets=build(lambda:(read(p,names='examples') for p in paths),cohort)
    assert targets.metadata['normalization']==TYPED_NORMALIZATION
    assert any(f.get('quantity')=='lattice_strain' for f in targets.metadata['families'])


@pytest.mark.parametrize('case',['parent','child','embedding','dimensions','cell_map','missing_axis','authority'])
def test_tampered_strain_context_is_rejected_before_target_training(tmp_path,case):
    path=mapped(tmp_path/'source');block=attach_strain(path,tmp_path)
    d=copy.deepcopy(block.mode_projection.metadata)
    if case=='parent':d['strain']['map']['parent_sha256']='b'*64
    elif case=='child':d['strain']['map']['child_sha256']='b'*64
    elif case=='embedding':d['strain']['map']['basis'][0][0]=2
    elif case=='dimensions':d['strain']['cell_mapping']=[r+[0] for r in d['strain']['cell_mapping']]
    elif case=='cell_map':d['strain']['cell_mapping'][0][0]=float('nan')
    elif case=='missing_axis':d['strain']['map']['normalizations'].pop()
    else:d['strain']['map']['scientific_result']=True
    (path/projection.METADATA).write_text(json.dumps(d))
    manifest=copy.deepcopy(block.manifest)
    manifest['mode_projection']=dict(schema=projection.STRAIN_SCHEMA,
        metadata_sha256=file_sha(path/projection.METADATA),arrays_sha256=d['arrays_sha256'])
    (path/'pairs.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):read(path)


def test_mixed_joint_model_trains_and_serializes_units_without_another_encoder(tmp_path):
    pytest.importorskip('torch')
    from gpx_recipe.preparation import structural_joint_modes as joint,structural_paired_training as training
    from gpx_recipe.preparation.structural_ranker import Ranker
    from test_structural_joint_modes_native import cohort as authored_cohort
    path,_=authored_cohort(tmp_path/'source');attach_strain(path,tmp_path)
    value,table=target_cohort(path)
    result=training.train(value,tmp_path/'model',epochs=2,threads=1)
    spec=result['mode_readout'];assert spec['normalization']==TYPED_NORMALIZATION
    assert len(spec['families'])==3 and result['test']['mode_validation']['known_queries']>0
    assert joint.validate(spec)==spec
    imported=Ranker(tmp_path/'model');assert imported.mode_spec==spec
    assert sum(f.get('quantity')=='lattice_strain' for f in spec['families'])==2
    changed=copy.deepcopy(spec);changed['normalization']=joint.NORMALIZATION
    with pytest.raises(ValueError,match='units or normalization'):joint.validate(changed)
    import torch
    from radar_pd_nova.structural_ranking_labels import mode_rows
    with torch.no_grad():
        inputs=torch.from_numpy((value['features'][:1]-imported.mean)/imported.scale)
        _,embedding=joint.shared(imported.model,inputs)
        evidence=joint.infer(imported.model,embedding,spec,space_groups=spec['classes'])
    shown=mode_rows(dict(mode_estimates=evidence))
    assert any(' · strain' in row['family'] and 'dimensionless' in row['amplitude'] for row in shown)
    assert any(' · strain' not in row['family'] and 'Å RMS' in row['amplitude'] for row in shown)
    assert all(row['status'].startswith('Unresolved') for row in shown)
