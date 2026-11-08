import json

import numpy as np
import pytest

from gpx_recipe.preparation.structural_augmentation import Controls, ObservationAugmenter
from gpx_recipe.preparation.structural_cohort import assemble_pairs
from gpx_recipe.preparation.structural_paired_blocks import read
from gpx_recipe.preparation.structural_replay import build
from gpx_recipe.preparation.structural_training_data import feature_rows, channel_statistics, feature_digest
from test_structural_paired_blocks import block


@pytest.mark.parametrize('observations', [False, True, 'impurity'])
def test_replay_preserves_all_feature_bytes_labels_aliases_and_folds(tmp_path, observations):
    def alias(d): d['candidate']['id'] = 'alias'; d['declared_child_group']['number'] = 8
    def labels(a): a['geometry_groups'][a['role'] == 2] = 8
    paths = [block(tmp_path/'first'), block(tmp_path/'second', change=labels, metadata=alias)]
    augmenter = None
    if observations:
        kwargs = {}
        if observations == 'impurity':
            q = read(paths[0]).arrays['q']
            kwargs['impurities'] = [dict(q=q, signal=np.exp(-((q-2.5)/.05)**2),
                source_sha256='f'*64, instrument_sha256='b'*64)]
        augmenter = ObservationAugmenter(Controls(impurity_fraction=(.01, .1)
            if observations == 'impurity' else (0., 0.)), instrument_sha256='b'*64, **kwargs)
    dense = assemble_pairs([read(p) for p in paths], bins=128, augmenter=augmenter)
    replay = build(paths, bins=128, augmenter=augmenter); rows = replay['features']
    try:
        assert feature_rows(rows) is rows and rows.shape == dense['features'].shape
        assert feature_digest(rows)==feature_digest(dense['features'])
        assert rows.row_layout['index_bytes']==len(rows)*76
        order = np.random.default_rng(571).permutation(len(rows))
        for selected in (order, order[::-1], np.array([0, 0, len(rows)-1])):
            np.testing.assert_array_equal(rows.take(selected), dense['features'][selected])
        for key in dense.keys()-{'features'}:
            if isinstance(dense[key], np.ndarray):
                np.testing.assert_array_equal(dense[key], replay[key])
            else: assert dense[key] == replay[key], key
        for actual, expected in zip(channel_statistics(rows, order),
                                    channel_statistics(feature_rows(dense['features']), order)):
            np.testing.assert_array_equal(actual, expected)
        report = rows.diagnostics()
        assert report['expanded_feature_disk_bytes'] == 0 and report['screening_qualified'] is False
        assert report['retained_native_bytes'] <= report['native_cache_byte_limit']
        assert not list(tmp_path.rglob('features.npy'))
    finally: rows.close()
    with pytest.raises(ValueError, match='closed'): rows.take(np.array([0]))


def test_replay_eviction_reloads_validated_native_blocks_without_changing_rows(tmp_path, monkeypatch):
    from gpx_recipe.preparation import structural_replay
    paths = [block(tmp_path/'one'), block(tmp_path/'two')]
    # Make the second child distinct, while retaining a valid native control.
    data = read(paths[1]); a = dict(data.arrays)
    a['theory_intensity'] = a['theory_intensity'].copy()
    a['theory_intensity'][a['role'] == 2] += np.exp(-((a['q']-3)/.1)**2)*.02
    import hashlib
    np.savez_compressed(paths[1]/'pairs.npz', **a)
    d = data.manifest; d['pairs_sha256'] = hashlib.sha256((paths[1]/'pairs.npz').read_bytes()).hexdigest()
    (paths[1]/'pairs.json').write_text(json.dumps(d))
    dense = assemble_pairs([read(p) for p in paths], bins=128)
    replay = build(paths, bins=128); rows = replay['features']
    monkeypatch.setattr(structural_replay, 'MAX_NATIVE_CACHE_BYTES', 3*a['q'].nbytes)
    try:
        order = np.arange(len(rows))[::-1]
        for i in list(order)+list(order[::-1]):
            np.testing.assert_array_equal(rows.take(np.array([i]))[0], dense['features'][i])
        assert rows.diagnostics()['source_blocks_captured'] == 2
        assert rows.diagnostics()['native_array_reads'] > len(rows)
        assert rows.diagnostics()['retained_native_bytes'] <= structural_replay.MAX_NATIVE_CACHE_BYTES
        returned = rows.take(np.array([0])); returned[:] = 98.
        np.testing.assert_array_equal(rows.take(np.array([0])), dense['features'][[0]])
    finally: rows.close()


@pytest.mark.parametrize('file', ['pairs.json', 'pairs.npz'])
def test_changed_files_are_rejected_even_for_already_cached_rows(tmp_path, file):
    path = block(tmp_path/'source'); rows = build([path], bins=128)['features']
    try:
        rows.take(np.array([0]))
        with (path/file).open('ab') as stream: stream.write(b' ')
        with pytest.raises(ValueError, match='sources changed'): rows.take(np.array([0]))
    finally: rows.close()


def test_replay_reads_are_bounded_and_reject_symlinked_sources(tmp_path):
    path = block(tmp_path/'source'); rows = build([path], bins=128)['features']
    try:
        for indices in (np.array([-1]), np.array([len(rows)]), np.zeros(257, int), np.array([1.])):
            with pytest.raises(ValueError, match='bounded batch'): rows.take(indices)
    finally: rows.close()
    (path/'pairs.npz').rename(path/'actual.npz')
    (path/'pairs.npz').symlink_to('actual.npz')
    with pytest.raises(ValueError, match='regular owned'): build([path], bins=128)


def test_captured_array_mutation_and_disk_exhaustion_cannot_change_training_rows(tmp_path, monkeypatch):
    from gpx_recipe.preparation import structural_replay
    path = block(tmp_path/'source'); rows = build([path], bins=128)['features']
    try:
        rows._bank.stream.seek(0); rows._bank.stream.write(b'changed!'); rows._bank.stream.flush()
        with pytest.raises(ValueError, match='captured native replay arrays changed'):
            rows.take(np.array([0]))
    finally: rows.close()
    monkeypatch.setattr(structural_replay, 'MAX_STORE_BYTES', 100)
    with pytest.raises(ValueError, match='working-store budget'): build([path], bins=128)


def test_oversized_cohort_fails_before_any_native_bank_array_is_written(tmp_path, monkeypatch):
    from gpx_recipe.preparation import structural_replay
    path = block(tmp_path/'source')
    monkeypatch.setattr(structural_replay.limits, 'MAX_ROWS', 3)
    def forbidden(*args): raise AssertionError('Admission must precede native bank writes')
    monkeypatch.setattr(structural_replay._NativeArrays, '_put', forbidden)
    with pytest.raises(ValueError, match='bounded seekable index'): build([path], bins=128)


def test_captured_stream_digest_needs_no_rendering_but_still_rejects_source_mutation(tmp_path,monkeypatch):
    path=block(tmp_path/'source');rows=build([path],bins=128)['features']
    try:
        original=feature_digest(rows)
        def forbidden(*args):raise AssertionError('Identity may reuse its actual assembly digest')
        monkeypatch.setattr(rows,'take',forbidden)
        assert feature_digest(rows)==original
        with (path/'pairs.npz').open('ab') as stream:stream.write(b'changed')
        with pytest.raises(ValueError,match='sources changed'):feature_digest(rows)
    finally:rows.close()
