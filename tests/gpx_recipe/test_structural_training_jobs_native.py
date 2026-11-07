"""The real training worker and Galaxy packet return only portable model data."""
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip('torch')

from gpx_recipe.preparation.worker import execute
from gpx_recipe.preparation.structural_jobs import plan_inputs
from gpx_recipe.preparation.structural_ranker import Ranker
from gpx_recipe.symmetry.bundle import pack, unpack
from gpx_recipe.symmetry.compiler import file_sha, package_signature
from gpx_recipe.execution import write_json
from test_structural_training_jobs import inputs


def test_training_worker_exports_data_only_model_and_preserves_whole_groups(tmp_path):
    request, directories = inputs(tmp_path)
    result = execute('structural-training', request)
    assert result['status'] == 'trained'
    assert result['observations_included'] is False
    assert result['candidate_pruning_authorized'] is False
    assert result['screening_qualified'] is False
    # Six physical trajectories plus the common undistorted-parent control.
    assert result['independent_groups'] == 7
    assert result['allocation']['workers'] == 1
    root = Path(result['output'])
    assert set(result['artifacts']) == {'model/model.json','model/weights.npz','training.json'}
    assert all(not name.endswith(('.npy','.gpx','.pt','.pkl')) for name in result['artifacts'])
    pack(root, tmp_path/'portable.zip', result['artifacts'])
    restored = unpack(tmp_path/'portable.zip', tmp_path/'restored')
    assert Ranker(restored/'model').metadata['weights_sha256'] == result['weights_sha256']
    progress = json.loads((root/'progress.json').read_text())
    assert progress['state'] == 'complete'


def test_training_runs_through_actual_galaxy_native_wrapper(tmp_path):
    pytest.importorskip('GSASII.GSASIIscriptable')
    root = tmp_path/'capture'; request, _ = inputs(root)
    names = ['parent.cif','plan.json']+plan_inputs(root/'plan.json')
    pack(root, tmp_path/'inputs.zip', names)
    payload = dict(request, source='parent.cif', plan='plan.json', destination='trained')
    message = dict(schema='radar-model-native/v1', operation='preparation-structural-training',
        payload=payload, implementation_sha256=package_signature(),
        expected_source_sha256=file_sha(root/'parent.cif'), dispatch_key='a'*24,
        timeout=60, input_hashes={name:file_sha(root/name) for name in names})
    write_json(tmp_path/'request.json', message)
    script = Path(__file__).resolve().parents[2]/'scripts/ndip_model_campaign.py'
    spec = importlib.util.spec_from_file_location('structural_training_galaxy_control', script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result = module._work('native', root/'parent.cif', tmp_path/'inputs.zip',
        tmp_path/'request.json', tmp_path/'job', tmp_path/'result.zip')
    assert result['result']['status'] == 'trained'
    collected = unpack(tmp_path/'result.zip', tmp_path/'collected')
    assert Ranker(collected/'native/trained/model').metadata['classes'] == [63,65]
    assert set(result['files']) == {'native/trained/model/model.json',
        'native/trained/model/weights.npz','native/trained/training.json'}
    assert not list(collected.rglob('features.npy'))
