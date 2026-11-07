import hashlib
import io
import json

import numpy as np
import pytest

pytest.importorskip('torch')

from gpx_recipe.preparation.structural_cohort import assemble, read_block
from gpx_recipe.preparation.structural_ranker import Ranker, train
from test_structural_ranker_cohort import block


def cohort(tmp_path):
    first = read_block(block(tmp_path))
    second = read_block(block(tmp_path, 'second', label=65, offset=.013))
    return assemble([first, second], bins=128), first


def test_training_export_import_and_inference_preserve_all_hypotheses(tmp_path):
    data, first = cohort(tmp_path)
    result = train(data, tmp_path/'model', epochs=3, threads=1)
    assert result['screening_qualified'] is False
    assert result['scientific_result'] is False
    assert result['allocation']['workers'] == 1
    assert 0 <= result['test']['top1'] <= result['test']['top5'] <= 1
    a = first.arrays
    request = dict(background=a['background'], parent_sha256='a'*64, instrument_sha256='b'*64)
    model = Ranker(tmp_path/'model')
    first_result = model.rank(a['q'], a['theory_intensity'][1], **request)
    second_result = Ranker(tmp_path/'model').rank(a['q'], a['theory_intensity'][1], **request)
    assert first_result == second_result
    assert len(first_result['families']) == 2
    assert {r['space_group'] for r in first_result['families']} == {63, 65}
    assert first_result['screening_qualified'] is False
    assert first_result['native_refinement_required'] is True
    assert {c for row in first_result['families'] for c in row['candidate_ids']} == {'first', 'second'}
    assert abs(sum(r['model_score'] for r in first_result['families'])-1) < 1e-6
    with pytest.raises(ValueError, match='exact prepared'):
        model.rank(a['q'], a['theory_intensity'][1], **dict(request, instrument_sha256='d'*64))
    valid = np.ones(len(a['q']), bool); valid[30:40] = False
    with pytest.raises(ValueError, match='missing coordinate'):
        model.rank(a['q'], a['theory_intensity'][1], valid=valid, **request)
    # A user-edited saved flag never grants authority to eliminate a family.
    path = tmp_path/'model/model.json'; d = json.loads(path.read_text()); d['screening_qualified'] = True
    path.write_text(json.dumps(d))
    assert Ranker(tmp_path/'model').rank(a['q'], a['theory_intensity'][1], **request)['screening_qualified'] is False
    d['candidate_groups']['first'] = 999; path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='child-to-family'): Ranker(tmp_path/'model')


def test_changed_and_malformed_weights_are_rejected_before_inference(tmp_path):
    data, _ = cohort(tmp_path); train(data, tmp_path/'model', epochs=1, threads=1)
    weights = tmp_path/'model/weights.npz'; raw = weights.read_bytes()
    weights.write_bytes(raw+b'changed')
    with pytest.raises(ValueError, match='changed'): Ranker(tmp_path/'model')
    with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    arrays['0.weight'] = np.zeros((1,), np.float32)
    np.savez_compressed(weights, **arrays)
    path = tmp_path/'model/model.json'; d = json.loads(path.read_text())
    d['weights_sha256'] = hashlib.sha256(weights.read_bytes()).hexdigest(); path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='tensor dimensions'): Ranker(tmp_path/'model')


def test_reassigned_trajectory_steps_cannot_pass_as_a_valid_holdout(tmp_path):
    data, _ = cohort(tmp_path)
    data['split'] = data['split'].copy()
    repeated = next(g for g in data['groups'] if data['groups'].count(g) > 1)
    index = data['groups'].index(repeated)
    data['split'][index] = (data['split'][index]+1) % 3
    with pytest.raises(ValueError, match='trajectory groups'):
        train(data, tmp_path/'invalid', epochs=1, threads=1)
