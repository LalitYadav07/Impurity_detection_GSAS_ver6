"""Automatic caching changes execution cost, never the scientific cohort."""
import json
import numpy as np
import pytest

from gpx_recipe.preparation import structural_training_cache as cache
from gpx_recipe.preparation.structural_augmentation import Controls,ObservationAugmenter
from gpx_recipe.preparation.structural_replay import build as replay
from test_structural_paired_blocks import block as paired_block


def block(root):
    return paired_block(root,metadata=lambda d:d.update(parent_relative_metric_range=[.001,.01]))


def test_placement_keeps_row_disk_and_free_space_limits():
    assert cache.placement(100,4096,20*1024**3)['storage']=='file_backed'
    assert cache.placement(cache.MAX_ROWS+1,128,100*1024**3)['storage']=='native_pair_replay'
    assert cache.placement(100,4096,cache.MINIMUM_HEADROOM)['storage']=='native_pair_replay'
    choice=cache.placement(100,4096,20*1024**3)
    assert choice['required_working_bytes']+choice['reserved_free_bytes']<=choice['observed_free_bytes']
    assert choice['scientific_scope_changed'] is False


@pytest.mark.parametrize('limited',[False,True])
def test_automatic_storage_retains_complete_features_aliases_noise_and_folds(tmp_path,monkeypatch,limited):
    source=block(tmp_path/'source')
    augmenter=ObservationAugmenter(Controls(variants=2),instrument_sha256='b'*64)
    if limited:monkeypatch.setattr(cache,'MAX_ROWS',1)
    actual=cache.build([source],tmp_path/'auto',bins=128,augmenter=augmenter,names='pairs')
    expected=replay([source],bins=128,augmenter=augmenter)
    try:
        assert actual['features'].storage==('native_pair_replay' if limited else 'file_backed')
        indices=np.arange(len(expected['features']))[::-1]
        np.testing.assert_array_equal(actual['features'].take(indices),expected['features'].take(indices))
        for key in ('classes','split','compatible_labels','q_edges','reference_features'):
            np.testing.assert_array_equal(actual[key],expected[key])
        for key in ('input_sha256','groups','references','example_simulations','class_candidate_ids','candidate_groups','reference_feature_sha256'):
            assert json.dumps(actual[key],sort_keys=True)==json.dumps(expected[key],sort_keys=True)
    finally:actual['features'].close();expected['features'].close()


def test_completed_storage_has_identical_cpu_weights_and_holdout_metrics(tmp_path,monkeypatch):
    from gpx_recipe.preparation import structural_training_backend as backend
    from gpx_recipe.preparation.structural_paired_training import train
    monkeypatch.setattr(backend,'select',lambda *a,**k:backend.Backend('cpu'))
    source=block(tmp_path/'source');options=dict(bins=128,augmenter=ObservationAugmenter(Controls(variants=1),instrument_sha256='b'*64))
    first=cache.build([source],tmp_path/'auto',names='pairs',**options)
    second=replay([source],**options)
    try:
        one=train(first,tmp_path/'cached-model',epochs=1,threads=1)
        two=train(second,tmp_path/'replayed-model',epochs=1,threads=1)
        assert one['weights_sha256']==two['weights_sha256']
        assert one['test']==two['test'] and one['prior_only_test']==two['prior_only_test']
        assert one['dataset']==two['dataset']
        assert (tmp_path/'cached-model/training-identity.json').read_bytes()==(tmp_path/'replayed-model/training-identity.json').read_bytes()
    finally:first['features'].close();second['features'].close()


def test_partial_or_symlinked_working_state_is_not_replaced(tmp_path):
    source=block(tmp_path/'source');partial=tmp_path/'partial';partial.mkdir();(partial/'owned-data').write_text('preserve')
    with pytest.raises(ValueError,match='fresh owned'):cache.build([source],partial,bins=128,names='pairs')
    assert (partial/'owned-data').read_text()=='preserve'
    link=tmp_path/'link';link.symlink_to(partial,target_is_directory=True)
    with pytest.raises(ValueError,match='fresh owned'):cache.build([source],link,bins=128,names='pairs')


def test_selected_cache_io_failure_does_not_silently_change_strategy(tmp_path,monkeypatch):
    from gpx_recipe.preparation import structural_paired_store,structural_replay
    source=block(tmp_path/'source')
    def fail(*a,**k):raise OSError('disk became unavailable')
    def forbidden(*a,**k):raise AssertionError('An unexpected alternate build was attempted')
    monkeypatch.setattr(structural_paired_store,'build',fail)
    monkeypatch.setattr(structural_replay,'build',forbidden)
    with pytest.raises(OSError,match='disk became'):cache.build([source],tmp_path/'auto',bins=128,names='pairs')


def test_finished_cache_is_discarded_after_close_without_deleting_the_directory(tmp_path):
    source=block(tmp_path/'source');destination=tmp_path/'cache'
    result=cache.build([source],destination,bins=128,names='pairs')
    result['features'].close();cache.discard_completed(destination)
    assert destination.is_dir() and not list(destination.iterdir())
    cache.discard_completed(destination)  # Empty replay state follows the same cleanup.


@pytest.mark.parametrize('case',['partial','extra','symlink'])
def test_cleanup_preserves_unexpected_or_partial_state(tmp_path,case):
    destination=tmp_path/'cache';destination.mkdir()
    for name in ('features.npy','targets.npz','store.json'):(destination/name).write_bytes(b'owned')
    if case=='partial':(destination/'store.json').unlink()
    elif case=='extra':(destination/'unexpected').write_bytes(b'preserve')
    else:
        (destination/'features.npy').unlink();(destination/'features.npy').symlink_to(destination/'targets.npz')
    before={p.name:p.lstat() for p in destination.iterdir()}
    with pytest.raises(ValueError,match='Preserve'):cache.discard_completed(destination)
    assert {p.name:p.lstat() for p in destination.iterdir()}==before
