import hashlib
import json

import numpy as np
import pytest

from gpx_recipe.preparation.structural_cohort import assemble, read_block


def block(tmp_path, name='first', *, label=63, offset=0., change=None):
    directory = tmp_path/name; directory.mkdir()
    q = np.linspace(.5, 4., 101); background = np.full_like(q, 100.)
    trajectory = np.r_[-1, np.repeat(np.arange(6), 2)]
    curves = np.asarray([background+np.exp(-((q-(1.+offset+.04*i))/.04)**2)
                         for i in range(len(trajectory))])
    arrays = dict(q=q, theory_intensity=curves, background=background,
        parameters=np.arange(len(trajectory), dtype=float)[:, None], trajectory=trajectory,
        geometry_groups=np.full((len(trajectory), 2), label, np.int32),
        ambiguous_geometry=np.zeros(len(trajectory), bool),
        displacement_rms=np.arange(len(trajectory), dtype=float)*.001,
        relative_metric_norm=np.zeros(len(trajectory)))
    if change is not None:
        change(arrays)
    np.savez_compressed(directory/'examples.npz', **arrays)
    d = dict(schema='radar-structural-ranking-examples/v3', observations_included=False,
        scientific_result=False, examples_sha256=hashlib.sha256((directory/'examples.npz').read_bytes()).hexdigest(),
        source_sha256='a'*64, instrument_sha256='b'*64, child_sha256='c'*64,
        candidate=dict(id=name, transformation_matrix=np.eye(3).tolist(), origin_shift=[0, 0, 0],
            child=dict(path='child.cif', sha256='c'*64)),
        declared_child_group=dict(number=label),
        examples=len(trajectory), bins=len(q), seed=1947,
        groups=['parent:'+'a'*64]+[f'parent:{"a"*64}:1947:trajectory-{i}' for i in trajectory[1:]])
    (directory/'examples.json').write_text(json.dumps(d))
    return directory


def test_background_removed_cohort_has_whole_trajectory_holdouts(tmp_path):
    first = read_block(block(tmp_path))
    second = read_block(block(tmp_path, 'second', label=65, offset=.01))
    result = assemble([first, second], bins=128)
    assert result['features'].shape == (26, 4, 128)
    assert result['classes'].tolist() == [63, 65]
    assert set(result['split']) == {0, 1, 2}
    for group in result['groups']:
        indices = [i for i, g in enumerate(result['groups']) if g == group]
        assert len(set(result['split'][indices])) == 1
    assert result['screening_qualified'] is False
    assert np.max(result['features'][:, 0]) <= 1.
    with pytest.raises(ValueError):
        first.arrays['theory_intensity'].flags.writeable = True


def test_identical_patterns_retain_ambiguous_labels_instead_of_discarding_them(tmp_path):
    result = assemble([read_block(block(tmp_path)), read_block(block(tmp_path, 'alias', label=65))], bins=128)
    assert result['original_examples'] == 26
    assert result['unique_examples'] == 13
    assert result['ambiguous_examples'] == 13
    assert result['compatible_labels'].all()
    assert all(len(refs) == 2 for refs in result['references'])


def test_transitive_alias_chain_cannot_create_independent_holdout_evidence(tmp_path):
    # The second child's shifted examples coincide with later steps of the
    # first. Connecting complete trajectories collapses every apparent fold.
    with pytest.raises(ValueError, match='independent trajectory'):
        assemble([read_block(block(tmp_path)),
                  read_block(block(tmp_path, 'alias', label=65, offset=.2))], bins=128)


def test_ambiguity_from_symmetry_tolerances_remains_a_set_valued_target(tmp_path):
    def change(a):
        a['geometry_groups'][1, 1] = 65
        a['ambiguous_geometry'][1] = True
    result = assemble([read_block(block(tmp_path, change=change)),
                       read_block(block(tmp_path, 'other', label=65, offset=.01))], bins=128)
    assert result['ambiguous_examples'] == 1
    assert result['compatible_labels'][1].tolist() == [True, True]


def test_different_exact_profiles_cannot_silently_train_one_context(tmp_path):
    first = read_block(block(tmp_path))
    directory = block(tmp_path, 'other')
    d = json.loads((directory/'examples.json').read_text()); d['instrument_sha256'] = 'd'*64
    (directory/'examples.json').write_text(json.dumps(d))
    with pytest.raises(ValueError, match='profile'):
        assemble([first, read_block(directory)], bins=128)


@pytest.mark.parametrize('case', ['checksum', 'observations', 'seed', 'group', 'schema', 'nonfinite', 'labels', 'background'])
def test_invalid_example_block_cannot_enter_training(tmp_path, case):
    def change(a):
        if case == 'nonfinite': a['theory_intensity'][2, 10] = np.nan
        elif case == 'labels': a['geometry_groups'][1] = [0, 999]
        elif case == 'background': a['background'] = np.zeros(3)
    directory = block(tmp_path, change=change)
    path = directory/'examples.json'; d = json.loads(path.read_text())
    if case == 'checksum': d['examples_sha256'] = '0'*64
    elif case == 'observations': d['observations_included'] = True
    elif case == 'seed': del d['seed']
    elif case == 'group': d['groups'][1] = 'unrelated-child-private-fold'
    elif case == 'schema': d['schema'] = 'radar-structural-ranking-examples/v1'
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError): read_block(directory)


def test_a_symlink_is_not_a_training_asset(tmp_path):
    directory = block(tmp_path)
    path = directory/'examples.npz'; saved = directory/'other.npz'; path.rename(saved); path.symlink_to(saved)
    with pytest.raises(ValueError): read_block(directory)
