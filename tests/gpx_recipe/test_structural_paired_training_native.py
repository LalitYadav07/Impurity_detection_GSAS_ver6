"""Real attention optimization and portable inference on artificial paired data."""
import json

import numpy as np
import pytest

pytest.importorskip('torch')
from gpx_recipe.preparation.structural_cohort import assemble_pairs
from gpx_recipe.preparation.structural_paired_blocks import read
from gpx_recipe.preparation.structural_paired_store import build
from gpx_recipe.preparation.structural_ranker import Ranker,train
from gpx_recipe.preparation.structural_attention_weights import SCHEMA,ARCHITECTURE
from gpx_recipe.preparation.structural_augmentation import ObservationAugmenter,Controls
from test_structural_paired_blocks import block


def data(root):
    path=block(root)
    return path,assemble_pairs([read(path)],bins=128,
        augmenter=ObservationAugmenter(Controls(variants=1),instrument_sha256='b'*64))


def test_training_and_portable_inference_use_three_curves_and_attention(tmp_path):
    path,cohort=data(tmp_path/'pairs')
    result=train(cohort,tmp_path/'model',epochs=1,threads=1)
    assert result['schema']==SCHEMA and result['architecture']==ARCHITECTURE and result['channels']==3
    assert result['curriculum']['training_rows_seen']==np.count_nonzero(cohort['split']==0)
    assert result['selected_epoch']==1 and result['history'][-1]['checkpoint_eligible']
    progress=json.loads((tmp_path/'model/progress.json').read_text())
    assert progress['state']=='complete' and progress['examples']==result['examples']
    assert progress['selected_epoch']==result['selected_epoch']
    assert all(cohort['split'][i]==0 for i in result['curriculum']['clean_training_representatives'])
    for key in ('test','prior_only_test'):
        metric=result[key]
        assert 0<=metric['top1']<=metric['top5']<=1
        assert set(metric['per_class'])==set(map(str,cohort['classes']))
        assert 0<=metric['worst_class_top5']<=metric['macro_top5']<=1
    assert result['candidate_pruning_authorized'] is result['screening_qualified'] is False
    model=Ranker(tmp_path/'model');assert hasattr(model.model,'attention')
    a=read(path).arrays
    ranked=model.rank(a['q'],a['theory_intensity'][2],background=a['background'],
        reference=dict(q=a['q'],intensity=a['parent_intensity'][2],background=a['parent_background'][2]),
        parent_sha256='a'*64,instrument_sha256='b'*64)
    assert ranked['candidate_order']==['child'] and ranked['complete_native_search_required']
    assert {row['space_group'] for row in ranked['families']}=={12,166}


def test_file_backed_and_dense_training_have_identical_parameters_and_holdout_scores(tmp_path):
    path,cohort=data(tmp_path/'pairs')
    disk=build([path],tmp_path/'cache',bins=128,
        augmenter=ObservationAugmenter(Controls(variants=1),instrument_sha256='b'*64))
    try:
        first=train(cohort,tmp_path/'dense-model',epochs=1,threads=1)
        second=train(disk,tmp_path/'disk-model',epochs=1,threads=1)
    finally:disk['features'].close()
    assert first['weights_sha256']==second['weights_sha256']
    assert first['test']==second['test'] and first['prior_only_test']==second['prior_only_test']
    assert first['curriculum']==second['curriculum']
    assert second['feature_storage']=='file_backed'


def test_validation_can_select_checkpoints_after_first_full_training_epoch(tmp_path):
    _,cohort=data(tmp_path/'pairs')
    result=train(cohort,tmp_path/'model',epochs=4,threads=1)
    assert result['curriculum']['clean_epochs']==1
    assert not result['history'][0]['checkpoint_eligible']
    assert all(row['checkpoint_eligible'] for row in result['history'][1:])
    assert all(row['training_rows_seen']==np.count_nonzero(cohort['split']==0) for row in result['history'][1:])


@pytest.mark.parametrize('case',['class_holdout','crossing_group','wrong_reference','wrong_channels'])
def test_invalid_training_data_cannot_publish_an_attention_model(tmp_path,case):
    _,cohort=data(tmp_path/'pairs')
    if case=='class_holdout':
        selected=np.flatnonzero(cohort['split']==1)
        cohort['compatible_labels'][selected]=False;cohort['compatible_labels'][selected,0]=True
    elif case=='crossing_group':cohort['groups'][np.flatnonzero(cohort['split']==1)[0]]=cohort['groups'][np.flatnonzero(cohort['split']==0)[0]]
    elif case=='wrong_reference':cohort['reference_features'][0,0]+=1.
    else:cohort['channels']=8
    with pytest.raises(ValueError):train(cohort,tmp_path/'refused',epochs=1,threads=1)
    assert not (tmp_path/'refused/model.json').exists()
