"""Actual Galaxy wrapper with independent calibration and >5000 fitting pairs.

Authored contract curves are not a diffraction-accuracy benchmark.
"""
import copy,importlib.util,json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('GSASII.GSASIIscriptable')


def test_calibrated_model_handoff_preserves_four_distinct_data_roles(tmp_path):
    from test_structural_calibration_training import packet_fixture
    from gpx_recipe.preparation.structural_jobs import plan_inputs
    from gpx_recipe.preparation.structural_ranker import Ranker
    from gpx_recipe.preparation.structural_paired_blocks import read
    from gpx_recipe.preparation.structural_training_blocks import REPRESENTATION
    from gpx_recipe.symmetry.bundle import pack,unpack
    from gpx_recipe.symmetry.compiler import file_sha,package_signature
    from gpx_recipe.execution import write_json
    root=tmp_path/'capture';example,plan=packet_fixture(root)
    names=['parent.cif','profile.instprm',plan.name]+plan_inputs(plan)
    payload=dict(source='parent.cif',instrument='profile.instprm',plan=plan.name,grid=example['grid'],
        destination='trained',settings=dict(representation=REPRESENTATION,bins=64,epochs=1,workers=4,augmentation=None))
    request=dict(schema='radar-model-native/v1',operation='preparation-structural-training',payload=payload,
        implementation_sha256=package_signature(),expected_source_sha256=file_sha(root/'parent.cif'),
        dispatch_key='a'*24,timeout=7200,input_hashes={name:file_sha(root/name) for name in names})
    pack(root,tmp_path/'input.zip',names);write_json(tmp_path/'request.json',request)
    script=Path(__file__).resolve().parents[2]/'scripts/ndip_model_campaign.py'
    spec=importlib.util.spec_from_file_location('calibrated_training_galaxy_control',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module._work('native',root/'parent.cif',tmp_path/'input.zip',tmp_path/'request.json',
        tmp_path/'job',tmp_path/'result.zip')
    assert result['result']['status']=='prepared'
    collected=unpack(tmp_path/'result.zip',tmp_path/'collected');model=Ranker(collected/'native/trained/model')
    d=model.metadata;reservation=d['calibration_partition'];assert reservation['available']
    assert len(reservation['calibration_groups'])==20 and reservation['fitting_clean_pairs']>=5001
    assert set(d['counts'])=={'train','validation','test','calibration'} and len(d['partition_rows'])==4
    assert d['adaptive_calibration']['identities']['weights_sha256']==d['weights_sha256']
    assert d['adaptive_test']['rows']==d['dataset']['partitions']['test']['rows']
    first=read(root/'cohort-0',names='examples');a=first.arrays
    ranked=model.rank(a['q'],a['theory_intensity'][-1],background=a['background'],
        parent_sha256=d['parent_sha256'],instrument_sha256=d['instrument_sha256'],
        reference=dict(q=a['q'],intensity=a['parent_intensity'][-1],background=a['parent_background'][-1]))
    selected=ranked['adaptive_work_set']
    assert sorted(selected['retained_space_groups']+selected['deferred_space_groups'])==d['classes']
    assert not selected['candidate_pruning_authorized']
    entry=json.loads((collected/'native/trained/entry.json').read_text())
    assert entry['detail']['dataset']==d['dataset']
    metadata=collected/'native/trained/model/model.json';original=metadata.read_bytes()
    for change in ['weights','groups','counts','test_metrics','authority']:
        modified=copy.deepcopy(d)
        if change=='weights':modified['adaptive_calibration']['identities']['weights_sha256']='f'*64
        elif change=='groups':modified['calibration_partition']['calibration_groups'].pop()
        elif change=='counts':modified['dataset']['partitions']['calibration']['clean_pairs']-=1
        elif change=='test_metrics':modified['adaptive_test']['excluded_compatible_labels']+=1
        else:modified['adaptive_calibration']['experimental_coverage_qualified']=True
        metadata.write_text(json.dumps(modified))
        with pytest.raises(ValueError):Ranker(metadata.parent)
        metadata.write_bytes(original)
