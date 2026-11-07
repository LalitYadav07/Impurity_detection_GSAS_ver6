import hashlib
import json

import numpy as np
import pytest

from gpx_recipe.preparation.structural_cohort import assemble, read_block
from gpx_recipe.preparation.structural_store import build, load
from gpx_recipe.preparation.structural_training_data import feature_rows, channel_statistics, open_rows
from test_structural_ranker_cohort import block


def test_streamed_assembly_preserves_aliases_labels_and_trajectory_holdouts(tmp_path):
    paths = [block(tmp_path), block(tmp_path, 'second', label=65, offset=.013),
             block(tmp_path, 'alias', label=65)]
    dense = assemble([read_block(p) for p in paths], bins=128)
    stored = build(paths, tmp_path/'cache', bins=128)
    try:
        assert stored['features'].storage == 'file_backed'
        assert stored['features']._array is None
        indices = np.arange(stored['unique_examples'])
        np.testing.assert_array_equal(stored['features'].take(indices), dense['features'])
        for key in ('classes', 'compatible_labels', 'split', 'q_edges', 'reference_features'):
            np.testing.assert_array_equal(stored[key], dense[key])
        for key in ('groups', 'candidate_groups', 'candidates', 'input_sha256', 'ambiguous_examples'):
            assert stored[key] == dense[key]
        assert stored['references'] == [[list(r) for r in refs] for refs in dense['references']]
        train = np.flatnonzero(dense['split'] == 0)
        for actual, expected in zip(channel_statistics(stored['features'], train),
                                    channel_statistics(feature_rows(dense['features']), train)):
            np.testing.assert_array_equal(actual, expected)
        with pytest.raises(ValueError, match='bounded batch'):
            stored['features'].take(np.zeros(257, np.int64))
    finally:
        stored['features'].close()
    with pytest.raises(ValueError, match='closed'):
        stored['features'].take(np.array([0]))


@pytest.mark.parametrize('case', ['features', 'targets', 'metadata', 'symlink', 'dimensions'])
def test_changed_cache_cannot_enter_training(tmp_path, case):
    paths = [block(tmp_path), block(tmp_path, 'second', label=65, offset=.013)]
    stored = build(paths, tmp_path/'cache', bins=128); stored['features'].close()
    root = tmp_path/'cache'
    if case == 'features':
        with (root/'features.npy').open('r+b') as f:
            f.seek(-4, 2); f.write(b'bad!')
    elif case == 'targets':
        with (root/'targets.npz').open('ab') as f: f.write(b'changed')
    elif case == 'symlink':
        path = root/'features.npy'; path.rename(root/'elsewhere.npy'); path.symlink_to(root/'elsewhere.npy')
    else:
        d = json.loads((root/'store.json').read_text())
        if case == 'metadata': d['rows'] += 1
        else: d['capacity'] += 1
        (root/'store.json').write_text(json.dumps(d))
    with pytest.raises((ValueError, OSError)):
        load(root)


def test_incomplete_cache_does_not_replace_a_complete_cache(tmp_path):
    path = block(tmp_path)
    root = tmp_path/'cache'; stored = build([path], root, bins=128); stored['features'].close()
    before = (root/'store.json').read_bytes()
    with pytest.raises(FileExistsError): build([path], root, bins=128)
    assert (root/'store.json').read_bytes() == before


def test_invalid_dimensions_are_rejected_before_opening_files(tmp_path):
    with pytest.raises(ValueError, match='dimensions'):
        open_rows(tmp_path/'missing', sha256='a'*64, capacity=10, rows=None, bins=128)


def test_file_reads_detect_changes_after_cache_admission(tmp_path):
    stored = build([block(tmp_path)], tmp_path/'cache', bins=128)
    try:
        with (tmp_path/'cache/features.npy').open('ab') as stream: stream.write(b'changed')
        with pytest.raises(ValueError, match='changed during training'):
            stored['features'].take(np.array([0]))
    finally:
        stored['features'].close()


def test_metadata_budget_covers_the_actual_published_representation(tmp_path, monkeypatch):
    from gpx_recipe.preparation import structural_store
    stored = build([block(tmp_path)], tmp_path/'first-cache', bins=128)
    stored['features'].close()
    raw = (tmp_path/'first-cache/store.json').read_bytes()
    compact_size = len(json.dumps(json.loads(raw)).encode())
    assert compact_size < len(raw)
    monkeypatch.setattr(structural_store, 'MAX_METADATA_BYTES', (compact_size+len(raw))//2)
    with pytest.raises(ValueError, match='metadata exceeds'):
        build([tmp_path/'first'], tmp_path/'oversized-cache', bins=128)
    assert not (tmp_path/'oversized-cache/store.json').exists()
