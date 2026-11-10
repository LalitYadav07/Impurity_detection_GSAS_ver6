"""Paired preparation uses the installed native worker and Galaxy transport."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
pytest.importorskip('torch')

from gpx_recipe.preparation.worker import execute
from gpx_recipe.preparation.structural_jobs import create_plan,plan_inputs
from gpx_recipe.preparation.structural_paired_blocks import read
from gpx_recipe.preparation.structural_training_blocks import REPRESENTATION,PARENT_RANGE
from gpx_recipe.preparation.structural_attention_weights import SCHEMA,ARCHITECTURE,FIXED_REFERENCE_KIND,REFERENCE_KIND
from gpx_recipe.preparation.structural_ranker import Ranker
from gpx_recipe.symmetry.bundle import pack,unpack
from gpx_recipe.symmetry.compiler import file_sha,package_signature
from gpx_recipe.execution import write_json
from test_native_profile_capture import native_source
from test_structural_paired_preparation import inputs,completed_library


@pytest.mark.parametrize('kind',['PNC','PNT'])
@pytest.mark.parametrize('parent_range',[PARENT_RANGE,(.0001,.001),(.0001,.003)])
def test_native_example_worker_returns_lossless_pairs_without_exporting_working_projects(tmp_path,kind,parent_range):
    source,_=native_source(tmp_path/'input',kind)
    parent=tmp_path/'parent.cif';G.G2Project(gpxfile=str(source)).phase(0).export_CIF(str(parent))
    child=Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    profile=tmp_path/'input/profile.instprm'
    request=dict(source=str(parent),instrument=str(profile),child=str(child),
        grid=dict(kind=kind,lower=12. if kind=='PNC' else 3000.,
            upper=125. if kind=='PNC' else 40000.,bins=701,bank=1),
        settings=dict(candidate=dict(id='public-ersi',transformation_matrix=[[-1.,0,0],[0,-1.,0],[0,0,1.]],
            origin_shift=[0,0,0],child=dict(path='child.cif',sha256=file_sha(child))),
            mode_radius=.05,cell_radius=.01,seed=1947,parent_seed=8341,
            parent_relative_metric_range=list(parent_range),representation=REPRESENTATION,
            trajectories=3,steps=2,trajectory_slice=[1,3]),destination=str(tmp_path/'output'))
    result=execute('structural-examples',request);root=Path(result['output'])
    original=read(root/'working-native');paired=read(root,names='examples')
    from gpx_recipe.preparation.native_size import read as read_size
    size=read_size(request,package_signature())
    assert size['rows']==7 and size['bins']==paired.manifest['bins']
    assert size['mode_dimensions']==paired.manifest['mode_dimensions']
    assert size['cell_dimensions']==paired.manifest['cell_dimensions']
    assert size['numeric_bytes']==sum(a.nbytes for a in paired.arrays.values())
    assert sum((root/name).stat().st_size for name in result['artifacts'])+65536<size['reservation_bytes']
    entry=json.loads((root/'entry.json').read_text())
    assert entry['detail']['timing']==original.manifest['timing']
    assert paired.manifest['examples']==7 and paired.manifest['trajectory_slice']==[1,3]
    assert paired.manifest['parent_controls_shape_qualified']
    assert paired.manifest['trajectory_admission']=='whole-child-rejection-at-fixed-shared-parent-context/v1'
    for name in original.arrays:assert original.arrays[name].tobytes()==paired.arrays[name].tobytes()
    assert set(result['artifacts'])=={'examples.json','examples.npz','entry.json','mode-projection.json','mode-projection.npz'}
    assert paired.mode_projection is not None
    assert size['native_mode_dimensions']==paired.mode_projection.metadata['native_dimensions']
    pack(root,tmp_path/'portable.zip',result['artifacts']);restored=unpack(tmp_path/'portable.zip',tmp_path/'restored')
    assert not list(restored.rglob('*.gpx'))
    assert read(restored,names='examples').manifest==paired.manifest


@pytest.mark.parametrize('multipart',[False,True])
@pytest.mark.parametrize('native_maps,additive',[(False,False),(True,False),(True,True),('families',True)])
@pytest.mark.parametrize('reference_kind',[None,FIXED_REFERENCE_KIND])
def test_three_curve_training_runs_through_the_actual_galaxy_wrapper(tmp_path,monkeypatch,multipart,native_maps,additive,reference_kind):
    root=tmp_path/'capture';example,directory=inputs(root)
    if native_maps or additive:
        from test_structural_mode_projection import attach
        attach(directory,names='examples')
    known_modes=native_maps=='families'
    if known_modes:
        from test_structural_mode_projection import rewrite_projection
        rewrite_projection(directory,change=lambda d:d.update(native_mode_labels=[
            'R-3m[0,0,0]GM1+(a)[A:c:dsp]A(a)','unresolved origin']))
        metadata=json.loads((directory/'examples.json').read_text())
        metadata['mode_projection']['metadata_sha256']=file_sha(directory/'mode-projection.json')
        (directory/'examples.json').write_text(json.dumps(metadata))
    directories=[directory]
    if additive:
        from test_structural_supplement_pairs import addition
        addition(directory,root/'added');directories.append(root/'added')
    create_plan(example['source'],directories,root/'plan.json')
    request=dict(source=example['source'],instrument=example['instrument'],plan=str(root/'plan.json'),
        grid=example['grid'],destination=str(root/'trained'),
        settings=dict(representation=REPRESENTATION,bins=128,epochs=1,workers=1,augmentation=None))
    if reference_kind is not None:request['settings']['reference_kind']=reference_kind
    names=['parent.cif','profile.instprm','plan.json']+plan_inputs(root/'plan.json')
    pack(root,tmp_path/'inputs.zip',names)
    payload=dict(request,source='parent.cif',instrument='profile.instprm',plan='plan.json',destination='trained')
    message=dict(schema='radar-model-native/v1',operation='preparation-structural-training',payload=payload,
        implementation_sha256=package_signature(),expected_source_sha256=file_sha(root/'parent.cif'),
        dispatch_key='a'*24,timeout=7200,input_hashes={name:file_sha(root/name) for name in names})
    write_json(tmp_path/'request.json',message)
    more=[]
    if multipart:
        from gpx_recipe.preparation import structural_input_parts as parts
        monkeypatch.setattr(parts,'MAX_PART_BYTES',6000)
        monkeypatch.setattr(parts,'MAX_METADATA',3000)
        captured={}
        def publish(signature,path):
            key='input'+str(len(captured));target=tmp_path/(key+'.zip')
            __import__('shutil').copyfile(path,target);captured[key]=target;return key
        refs=parts.prepare(root,message,tmp_path/'input-work',publish)
        input_archive=captured[refs['primary']];more=[captured[key] for key in refs['parts']]
        assert len(more)>=2 if native_maps or additive else len(more)==2
    else:input_archive=tmp_path/'inputs.zip'
    script=Path(__file__).resolve().parents[2]/'scripts/ndip_model_campaign.py'
    spec=importlib.util.spec_from_file_location('paired_preparation_galaxy_control',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module._work('native',root/'parent.cif',input_archive,tmp_path/'request.json',
        tmp_path/'job',tmp_path/'result.zip',asset_parts=more)
    assert result['result']['status']=='prepared'
    collected=unpack(tmp_path/'result.zip',tmp_path/'collected');model=Ranker(collected/'native/trained/model')
    assert model.metadata['reference_kind']==(REFERENCE_KIND if reference_kind is None else reference_kind)
    if native_maps or additive:
        supervision=model.metadata['mode_family_supervision']
        assert supervision['mapped_blocks']==supervision['total_blocks']==len(directories)
        assert supervision['rows']==model.metadata['examples']
        assert bool(supervision['intervals'])==known_modes
        assert model.metadata['mode_head_trained']==known_modes
    else:assert 'mode_family_supervision' not in model.metadata
    assert model.metadata['schema']==SCHEMA and model.metadata['channels']==3
    from gpx_recipe.preparation.structural_attention_weights import MODE_ARCHITECTURE
    assert model.metadata['architecture']==(MODE_ARCHITECTURE if known_modes else ARCHITECTURE)
    assert model.metadata['readout_health']['readout_active']
    assert model.metadata['classes']==[12,166]
    if native_maps or additive:
        assert model.metadata['mode_projection_coverage']['complete']
        assert model.metadata['mode_projection_coverage']['mode_targets_inferred'] is False
    if additive:
        assert set(model.metadata['additive_pair_inventory'])=={'1'}
        assert model.metadata['partition_anchor']['schema']=='radar-owned-original-additive-partition/v1'
        assert model.metadata['mode_head_trained']==known_modes
    if known_modes:
        from gpx_recipe.preparation.structural_joint_modes import OBJECTIVE,SELECTION
        assert model.metadata['training_objective']==OBJECTIVE and model.metadata['checkpoint_selection']==SELECTION
        assert model.metadata['test']['mode_validation']['available']
    assert model.metadata['feature_storage']=='file_backed'
    entry=json.loads((collected/'native/trained/entry.json').read_text())
    assert entry['detail']['dataset']==model.metadata['dataset']
    assert not model.metadata['screening_qualified'] and not model.metadata['candidate_pruning_authorized']
    assert set(result['files'])=={'native/trained/model/model.json','native/trained/model/weights.npz',
        'native/trained/training.json','native/trained/entry.json'}
    assert not list(collected.rglob('features.npy'))
    assert not list((tmp_path/'job').rglob('features.npy'))
    assert not list((tmp_path/'job/native/trained/working-features').iterdir())


@pytest.mark.parametrize('known_modes',[False,True])
def test_default_paired_library_training_remains_usable_after_export_and_model_handoff(tmp_path,known_modes):
    from gpx_recipe.preparation.operations import stage_entry,publish_entry
    from gpx_recipe.preparation.library import import_library
    from gpx_recipe.preparation import hints,structural_ranking_hints as ranking
    library=completed_library(tmp_path,known_modes=known_modes)
    trainer=next(r for r in library.document['entries'] if r['model']['kind']=='structural-training')
    attempt=library.start(trainer['id']);request=stage_entry(library,trainer['id'],attempt,tmp_path/'training-attempt')
    assert request['settings']['bins']==4096 and request['settings']['epochs']==60
    result=execute('structural-training',request)
    assert publish_entry(library,trainer['id'],attempt,request,result)['all_prepared']
    library.export(tmp_path/'library.zip');restored=import_library(tmp_path/'library.zip',tmp_path/'restored-library')
    assert restored.document.get('storage',{}).get('deferred_assets')
    assert restored.document['entries']==library.document['entries']
    assert not list((restored.root/'entries').rglob('examples.npz'))
    assert list((library.root/'entries').rglob('examples.npz'))
    assets=restored.prepared_assets(trainer['id'])
    metadata=json.loads(next(p for p in assets.values() if p.name=='model.json').read_text())
    assert bool(metadata.get('mode_head_trained'))==known_modes
    candidate=next(r['model']['settings']['candidate'] for r in restored.document['entries']
        if r['model']['kind']=='structural-examples')
    protocol=dict(kind='structural',candidates=[candidate]);capture=tmp_path/'model-capture'
    document=hints.capture(restored,protocol,capture)
    pack(capture,tmp_path/'model-packet.zip',hints.packet_paths(capture,protocol))
    received=unpack(tmp_path/'model-packet.zip',tmp_path/'model-received');target=tmp_path/'model-worker'
    attached=hints.restore(document,protocol,received,target)
    assert attached['ranking_models']['attached']==1
    d=read(tmp_path/'examples',names='examples');a=d.arrays
    from gpx_recipe.preparation.structural_attention_weights import FIXED_REFERENCE_KIND
    assert metadata['reference_kind']==FIXED_REFERENCE_KIND
    # New libraries train a fixed measured-before contract. A native caller
    # supplies the independently captured reference and its physical identity.
    reference=dict(q=a['q'],intensity=a['parent_intensity'][0],background=a['parent_background'][0],
        kind='measured_before_transition',parent_sha256=d.manifest['source_sha256'],
        instrument_sha256=d.manifest['instrument_sha256'],source_sha256='a'*64)
    ranked=ranking.rank(target,protocol,a['q'],a['theory_intensity'][2],background=a['background'],
        parent_sha256=d.manifest['source_sha256'],instrument_sha256=d.manifest['instrument_sha256'],
        entry_id=trainer['id'],
        reference=reference)
    assert ranked['ranking_available'] and not ranked['discarded_entries']
    assert ranked['candidate_order']==['child'] and ranked['complete_native_search_required']
    assert ranked['candidate_pruning_authorized'] is False
    if known_modes:
        estimates=ranked['mode_estimates']
        assert estimates['available'] and estimates['rows'] and estimates['readout_identity']
        assert estimates['intervals_calibrated'] is False
        assert estimates['experimental_mode_absence_qualified'] is False
    assert not list(restored.root.rglob('features.npy'))
