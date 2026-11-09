"""A fixed measured-reference model must be trained and consumed as such."""
import hashlib

import numpy as np
import pytest

pytest.importorskip('torch')
from gpx_recipe.preparation.structural_attention_weights import FIXED_REFERENCE_KIND
from gpx_recipe.preparation.structural_augmentation import Controls, ObservationAugmenter
from gpx_recipe.preparation.structural_cohort import assemble_pairs
from gpx_recipe.preparation.structural_features import independent_curve_features
from gpx_recipe.preparation.structural_paired_blocks import read
from gpx_recipe.preparation.structural_paired_store import build
from gpx_recipe.preparation.structural_ranker import Ranker, train
from test_structural_paired_blocks import block


def test_fixed_before_keeps_parent_thermal_shifts_in_difference_channel(tmp_path):
    native = read(block(tmp_path/'native'))
    old = assemble_pairs([native], bins=128)
    new = assemble_pairs([native], bins=128, reference_kind=FIXED_REFERENCE_KIND)
    a = native.arrays
    row = next(i for i, refs in enumerate(new['references']) if [0, 3] in [list(r) for r in refs])
    expected = independent_curve_features(a['q'], a['theory_intensity'][3], a['q'],
        a['parent_intensity'][0], new['q_edges'], background=a['background'],
        reference_background=a['parent_background'][0])
    np.testing.assert_array_equal(new['features'][row], expected['features'])
    assert np.linalg.norm(new['features'][row, 2]) > .01
    # In the old contract every parent-only control is paired at its own
    # cell, so its residual vanishes; these contracts cannot share weights.
    old_row = next(i for i, refs in enumerate(old['references']) if [0, 3] in [list(r) for r in refs])
    np.testing.assert_array_equal(old['features'][old_row, 2], 0.)


def test_both_reference_and_current_noise_are_declared_and_storage_preserves_rows(tmp_path):
    path = block(tmp_path/'native')
    options = dict(bins=128, reference_kind=FIXED_REFERENCE_KIND,
        augmenter=ObservationAugmenter(Controls(variants=2), instrument_sha256='b'*64))
    dense = assemble_pairs([read(path)], **options)
    disk = build([path], tmp_path/'cache', **options)
    try:
        assert disk['reference_kind'] == dense['reference_kind'] == FIXED_REFERENCE_KIND
        np.testing.assert_array_equal(disk['features'].take(np.arange(len(dense['features']))), dense['features'])
        assert disk['groups'] == dense['groups'] and disk['example_simulations'] == dense['example_simulations']
        assignments = {}
        for group, code in zip(dense['groups'], dense['split']):
            assignments.setdefault(group, set()).add(int(code))
        assert all(len(codes) == 1 for codes in assignments.values())
        for simulations in dense['example_simulations']:
            for simulated in simulations:
                if simulated['kind'] == 'simulated_observation':
                    before = simulated['reference_observation']
                    assert before['seed'] != simulated['seed']
                    assert before['variant'] == simulated['variant']
                    assert before['effective_peak_counts'] >= 1000
    finally:
        disk['features'].close()


def test_model_retains_fixed_reference_contract_and_rejects_a_calculated_reference(tmp_path):
    native = read(block(tmp_path/'native'))
    cohort = assemble_pairs([native], bins=128, reference_kind=FIXED_REFERENCE_KIND,
        augmenter=ObservationAugmenter(Controls(variants=1), instrument_sha256='b'*64))
    result = train(cohort, tmp_path/'model', epochs=1, threads=1)
    assert result['reference_kind'] == FIXED_REFERENCE_KIND
    model = Ranker(tmp_path/'model'); a = native.arrays
    reference = dict(q=a['q'], intensity=a['parent_intensity'][0], background=a['parent_background'][0])
    options = dict(background=a['background'], parent_sha256='a'*64, instrument_sha256='b'*64)
    with pytest.raises(ValueError, match='measured before-transition'):
        model.rank(a['q'], a['theory_intensity'][2], reference=reference, **options)
    reference.update(kind='measured_before_transition', parent_sha256='a'*64, instrument_sha256='b'*64,
        source_sha256=hashlib.sha256(a['parent_intensity'][0].tobytes()).hexdigest())
    ranked = model.rank(a['q'], a['theory_intensity'][2], reference=reference, **options)
    assert ranked['candidate_pruning_authorized'] is False
    assert ranked['complete_native_search_required'] is True
    assert {r['space_group'] for r in ranked['families']} == {12, 166}
    unchanged = model.rank(a['q'], a['parent_intensity'][0],
        background=a['parent_background'][0], reference=reference,
        parent_sha256='a'*64, instrument_sha256='b'*64)
    assert unchanged['ranking_available'] is False
    assert 'no difference' in unchanged['reason']
    assert all(row['model_score'] is None for row in unchanged['families'])
    assert unchanged['candidate_pruning_authorized'] is False
    assert unchanged['complete_native_search_required'] is True
    for field in ('kind', 'parent_sha256', 'instrument_sha256', 'source_sha256'):
        wrong = dict(reference, **{field:'wrong'})
        with pytest.raises(ValueError, match='measured before-transition'):
            model.rank(a['q'], a['theory_intensity'][2], reference=wrong, **options)
    # A separately sampled but identically calibrated reference need not
    # have the current acquisition's native point grid.
    reference_q = np.sort(np.r_[a['q'], (a['q'][:-1]+a['q'][1:])*.5])
    separate = dict(reference, q=reference_q,
        intensity=np.interp(reference_q, a['q'], a['parent_intensity'][0]),
        background=np.interp(reference_q, a['q'], a['parent_background'][0]))
    ranked_separate = model.rank(a['q'], a['theory_intensity'][2], reference=separate, **options)
    assert ranked_separate['candidate_order'] == ranked['candidate_order']
    assert ranked_separate['ranking_available'] == ranked['ranking_available']


@pytest.mark.parametrize('kind', ['measured', [], {}, 7])
def test_undeclared_reference_contract_is_rejected(tmp_path, kind):
    native = read(block(tmp_path/'native'))
    with pytest.raises(ValueError, match='reference contract'):
        assemble_pairs([native], bins=128, reference_kind=kind)
