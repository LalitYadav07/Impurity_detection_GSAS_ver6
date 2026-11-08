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
    from gpx_recipe.preparation.structural_compatible_objective import OBJECTIVE,SELECTION,checkpoint_key
    assert result['training_objective']==OBJECTIVE and result['checkpoint_selection']==SELECTION
    eligible=[row for row in result['history'] if row['checkpoint_eligible']]
    assert result['selected_epoch']==min(eligible,key=lambda row:checkpoint_key(row['validation']))['epoch']
    progress=json.loads((tmp_path/'model/progress.json').read_text())
    assert progress['state']=='complete' and progress['examples']==result['examples']
    assert progress['selected_epoch']==result['selected_epoch']
    dataset=result['dataset']
    assert progress['dataset']==dataset
    coverage=dataset['family_coverage']
    assert coverage['classes']==cohort['classes'].tolist()
    assert dataset['generated_pairs']==cohort['original_examples']
    clean=np.array([any(p.get('kind')=='clean_native' for p in values) for values in cohort['example_simulations']])
    assert dataset['unique_clean_pairs']==int(clean.sum())
    for code,name in enumerate(('train','validation','test')):
        part=dataset['partitions'][name]
        assert part['rows']==int(np.count_nonzero(cohort['split']==code))
        assert part['clean_pairs']==int(np.count_nonzero((cohort['split']==code)&clean))
        for column,group in enumerate(cohort['classes']):
            selected=(cohort['split']==code)&cohort['compatible_labels'][:,column]
            cell=coverage['partitions'][name][str(group)]
            assert cell['rows']==int(selected.sum())
            assert cell['clean_pairs']==int(np.count_nonzero(selected&clean))
            assert cell['trajectory_groups']==len({g for g,keep in zip(cohort['groups'],selected) if keep})
            assert cell['clean_trajectory_groups']==len({g for g,keep in zip(cohort['groups'],selected&clean) if keep})
    assert all(cohort['split'][i]==0 for i in result['curriculum']['clean_training_representatives'])
    assert result['curriculum']['clean_replay']['maximum_rows_per_update']==16
    assert result['curriculum']['clean_replay']['additional_independent_pairs']==0
    retention=result['history'][-1]['strong_training_recall']
    assert retention['representatives']==len(cohort['classes']) and 0<=retention['top1']<=retention['top5']<=1
    assert 'per_class' not in retention
    for key in ('test','prior_only_test'):
        metric=result[key]
        assert 0<=metric['top1']<=metric['top5']<=1
        assert set(metric['per_class'])==set(map(str,cohort['classes']))
        assert 0<=metric['worst_class_top5']<=metric['macro_top5']<=1
        observations=metric['observation_metrics']['partitions']
        assert observations['clean_native']['rows']==dataset['partitions']['test']['clean_pairs']
        assert observations['simulated_observation']['rows']==dataset['partitions']['test']['simulated_only_rows']
    observed=result['history'][-1]['validation']['observation_metrics']['partitions']
    assert observed['clean_native']['rows']==dataset['partitions']['validation']['clean_pairs']
    assert observed['simulated_observation']['rows']==dataset['partitions']['validation']['simulated_only_rows']
    assert result['candidate_pruning_authorized'] is result['screening_qualified'] is False
    model=Ranker(tmp_path/'model');assert hasattr(model.model,'attention')
    a=read(path).arrays
    ranked=model.rank(a['q'],a['theory_intensity'][2],background=a['background'],
        reference=dict(q=a['q'],intensity=a['parent_intensity'][2],background=a['parent_background'][2]),
        parent_sha256='a'*64,instrument_sha256='b'*64)
    assert ranked['candidate_order']==['child'] and ranked['complete_native_search_required']
    assert {row['space_group'] for row in ranked['families']}=={12,166}


@pytest.mark.parametrize('storage', ['file_backed', 'native_pair_replay'])
def test_bounded_and_dense_training_have_identical_parameters_and_holdout_scores(tmp_path, storage):
    path,cohort=data(tmp_path/'pairs')
    kwargs=dict(bins=128,augmenter=ObservationAugmenter(Controls(variants=1),instrument_sha256='b'*64))
    if storage=='file_backed':disk=build([path],tmp_path/'cache',**kwargs)
    else:
        from gpx_recipe.preparation.structural_replay import build as replay
        disk=replay([path],**kwargs)
    try:
        first=train(cohort,tmp_path/'dense-model',epochs=1,threads=1)
        second=train(disk,tmp_path/'disk-model',epochs=1,threads=1)
    finally:disk['features'].close()
    assert first['weights_sha256']==second['weights_sha256']
    assert (tmp_path/'dense-model/training-identity.json').read_bytes()==(tmp_path/'disk-model/training-identity.json').read_bytes()
    assert first['test']==second['test'] and first['prior_only_test']==second['prior_only_test']
    assert first['curriculum']==second['curriculum']
    assert second['feature_storage']==storage
    assert first['dataset']==second['dataset']


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
