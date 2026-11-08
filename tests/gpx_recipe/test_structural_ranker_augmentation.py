import numpy as np
import pytest
import hashlib
import json

from gpx_recipe.preparation.structural_augmentation import Controls, ObservationAugmenter
from gpx_recipe.preparation.structural_cohort import assemble, read_block
from gpx_recipe.preparation.structural_store import build
from test_structural_ranker_cohort import block


def signal():
    q = np.linspace(.5, 5., 2001)
    return q, 10*np.exp(-((q-2)/.05)**2)


def test_augmentation_is_reproducible_preserves_inputs_and_has_explicit_counting_semantics():
    q, y = signal(); background = np.full(len(q), 100.)
    original = y.copy()
    augmenter = ObservationAugmenter(Controls(), instrument_sha256='b'*64)
    one = list(augmenter.observations(q, y+background, background, clean_identity='a'*64))
    two = list(augmenter.observations(q, y+background, background, clean_identity='a'*64))
    assert len(one) == 3
    for first, second in zip(one, two):
        for key in ('intensity', 'background', 'sigma'):
            np.testing.assert_array_equal(first[key], second[key])
        assert first['provenance'] == second['provenance']
        assert np.all(first['intensity'] >= 0)
        assert np.all(first['sigma'] > 0)
    assert not np.array_equal(one[0]['intensity'], one[1]['intensity'])
    np.testing.assert_array_equal(y, original)
    assert augmenter.metadata['experimental_counting_statistics_established'] is False
    assert augmenter.metadata['profile_variation_included'] is False
    assert augmenter.metadata['impurity_fraction_kind'] == 'integrated_Q_pattern_intensity'


def test_poisson_draws_recover_declared_mean_and_variance():
    q = np.linspace(.5, 5., 20001); y = np.ones(len(q))*10
    controls = Controls(variants=1, peak_counts=(100., 100.), background_fraction=(.2, .2),
                        background_error=(0., 0.))
    draw = next(ObservationAugmenter(controls, instrument_sha256='b'*64).observations(
        q, y, np.zeros_like(y), clean_identity='a'*64))
    z = (draw['intensity']-draw['background']-y)/draw['sigma']
    assert abs(z.mean()) < .03
    assert abs(z.var()-1.) < .04


def test_individual_siblings_retain_pre_replay_numeric_bytes_and_provenance():
    # Captured from eb86a04 before the single-sibling API was introduced.
    expected = [
        '84e53a5ea275d4883c0055bcc2a99c25e32c9862831b3d44f291e2531620421d',
        'da4b1c4fba24bb94bba2ea70d7626eb118dd7bbf7c4ed5eca42d0583f9d8eca4',
        'd6bee642f0af441d46bb3c6d234dd5b2b9468852e981c4cfd6b6bdceeb3cc35f',
    ]
    q, y = signal(); background = np.full(len(q), 100.)
    augmenter = ObservationAugmenter(Controls(), instrument_sha256='b'*64)
    sequential = list(augmenter.observations(q, y+background, background, clean_identity='a'*64))
    for index in (2, 0, 1, 2):
        row = augmenter.observation(q, y+background, background,
                                    clean_identity='a'*64, variant=index)
        digest = hashlib.sha256()
        for key in ('intensity', 'background', 'sigma'):
            np.testing.assert_array_equal(row[key], sequential[index][key])
            digest.update(row[key].astype('<f8').tobytes())
        digest.update(json.dumps(row['provenance'], sort_keys=True).encode())
        assert digest.hexdigest() == expected[index]


@pytest.mark.parametrize('variant', [-1, 3, True, 1., '1', None])
def test_single_observation_cannot_invent_an_undeclared_variant(variant):
    q, y = signal()
    with pytest.raises(ValueError, match='existing declared'):
        ObservationAugmenter(Controls(), instrument_sha256='b'*64).observation(
            q, y, np.zeros_like(y), clean_identity='a'*64, variant=variant)


def test_native_impurity_fraction_uses_integrated_intensity_and_exact_grid():
    q, y = signal(); minor = np.exp(-((q-3.)/.1)**2)
    template = dict(q=q, signal=minor, source_sha256='c'*64, instrument_sha256='b'*64)
    controls = Controls(variants=1, peak_counts=(1e8, 1e8), background_fraction=(0., 0.),
                        background_error=(0., 0.), impurity_fraction=(.1, .1))
    augmenter = ObservationAugmenter(controls, instrument_sha256='b'*64, impurities=[template])
    # Caller edits cannot mutate an admitted native template.
    template['signal'][:] = 0
    draw = next(augmenter.observations(q, y, np.zeros_like(y), clean_identity='a'*64))
    area = lambda x: np.dot(np.diff(q), (x[:-1]+x[1:])*.5)
    assert area(draw['intensity'][...])/area(y) == pytest.approx(1/.9, rel=1e-4)
    assert draw['provenance']['impurity_source_sha256'] == 'c'*64
    with pytest.raises(ValueError, match='exact same Q'):
        next(augmenter.observations(q+.001, y, np.zeros_like(y), clean_identity='a'*64))


def test_noisy_aliases_keep_all_labels_references_and_whole_trajectory_holdouts(tmp_path):
    paths = [block(tmp_path), block(tmp_path, 'alias', label=65)]
    augmenter = ObservationAugmenter(Controls(variants=2), instrument_sha256='b'*64)
    dense = assemble([read_block(p) for p in paths], bins=128, augmenter=augmenter)
    stored = build(paths, tmp_path/'cache', bins=128, augmenter=augmenter)
    try:
        assert dense['original_examples'] == 26
        assert dense['augmented_examples'] == 26
        assert dense['unique_examples'] == 39
        assert dense['ambiguous_examples'] == 36
        assert all(len(refs) == 2 for refs in dense['references'])
        for group in set(dense['groups']):
            assert len(set(dense['split'][[i for i, g in enumerate(dense['groups']) if g == group]])) == 1
        np.testing.assert_array_equal(stored['features'].take(np.arange(39)), dense['features'])
        np.testing.assert_array_equal(stored['compatible_labels'], dense['compatible_labels'])
        assert stored['augmentation_sha256'] == augmenter.sha256
        assert stored['example_simulations'] == dense['example_simulations']
    finally:
        stored['features'].close()


@pytest.mark.parametrize('case', ['bounds', 'variants', 'seed', 'template', 'profile', 'negative'])
def test_invalid_augmentation_cannot_silently_produce_training_data(case):
    q, y = signal()
    with pytest.raises(ValueError):
        if case == 'bounds': Controls(peak_counts=(-1., 10.))
        elif case == 'variants': Controls(variants=1000)
        elif case == 'seed': Controls(seed=-1)
        elif case == 'template': ObservationAugmenter(Controls(impurity_fraction=(0., .1)), instrument_sha256='b'*64)
        elif case == 'profile': ObservationAugmenter(Controls(), instrument_sha256='not-a-profile')
        else:
            y[0] = -1
            next(ObservationAugmenter(Controls(), instrument_sha256='b'*64).observations(
                q, y, np.zeros_like(y), clean_identity='a'*64))
