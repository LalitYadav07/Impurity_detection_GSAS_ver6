from __future__ import annotations
import numpy as np
import pytest
from gpx_recipe.commensurate_k import point_candidates, reciprocal_equivalent, rank_profiles


def operations(translations=((0, 0, 0),)):
    return [dict(rotation=np.eye(3).tolist(), translation=list(t)) for t in translations]


FCC = operations(((0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)))


def test_fcc_integer_k_is_not_gamma():
    assert not reciprocal_equivalent([0, 1, 0], [0, 0, 0], FCC)
    assert reciprocal_equivalent([1, 1, 1], [0, 0, 0], FCC)
    assert reciprocal_equivalent([0, 2, 0], [0, 0, 0], FCC)


def test_screw_translation_is_not_a_centering_translation():
    ops = operations() + [dict(rotation=np.diag([-1, 1, -1]).tolist(), translation=[0, .5, 0])]
    assert reciprocal_equivalent([0, 1, 0], [0, 0, 0], ops)


def test_conjugate_and_parent_rotation_equivalence():
    ops = operations() + [dict(rotation=[[0, -1, 0], [1, 0, 0], [0, 0, 1]], translation=[0, 0, 0])]
    assert reciprocal_equivalent(['1/3', 0, 0], [0, '-1/3', 0], ops)
    assert not reciprocal_equivalent(['1/3', 0, 0], [0, 0, '1/3'], ops)


def test_points_preserve_fcc_x_and_report_nonrational_point():
    result = point_candidates(dict(point_coords={'X': [0, 1, 0], 'L': [.5] * 3, 'G': [0, 0, 0], 'metric': [.18374625, 0, 0]}), FCC)
    assert [r['k'] for r in result['candidates']] == [['0', '0', '0'], ['0', '1', '0'], ['1/2'] * 3]
    assert result['excluded'][0]['label'] == 'metric'
    assert not result['exhaustive_commensurate_search']


def test_conventional_fcc_cell_is_nonprimitive_but_valid():
    result = point_candidates(dict(is_supercell=True, point_coords={'X': [0, 1, 0]}), FCC)
    assert result['input_nonprimitive']
    assert len(result['candidates']) == 2


@pytest.mark.parametrize('bad', [[], [dict(rotation=np.zeros((3, 3)).tolist(), translation=[0, 0, 0])]])
def test_bad_parent_inventory_refused(bad):
    with pytest.raises(ValueError):
        point_candidates(dict(point_coords={}), bad)


def profiles():
    x = np.linspace(0, 10, 1000)
    a = np.exp(-((x - 3) / .08) ** 2) + .4 * np.exp(-((x - 7) / .1) ** 2)
    b = np.exp(-((x - 5) / .08) ** 2)
    templates = [dict(id='true', k=['0', '0', '1/2'], intensity=a), dict(id='wrong', k=['0', '0', '0'], intensity=b)]
    return a, b, templates


def test_signed_profile_likelihood_prefers_right_k():
    a, b, templates = profiles()
    result = rank_profiles(10 + 5 * a, np.full_like(a, 10), np.ones_like(a), templates)
    assert result['status'] == 'shortlist'
    assert result['families'][0]['k'] == ['0', '0', '1/2']
    assert result['templates'][0]['amplitude'] == pytest.approx(5)
    assert not result['unique_k_established']


def test_no_signal_does_not_select_a_k():
    a, b, templates = profiles()
    result = rank_profiles(np.ones_like(a), np.ones_like(a), np.ones_like(a), templates)
    assert result['status'] == 'no_support'


def test_identical_powder_patterns_remain_ambiguous():
    a, b, templates = profiles()
    templates[1]['intensity'] = a.copy()
    assert rank_profiles(a * 10, a * 0, np.ones_like(a), templates)['status'] == 'ambiguous'


def test_expected_but_absent_peaks_reduce_score():
    a, b, _ = profiles()
    result = rank_profiles(a * 5, a * 0, np.ones_like(a), [dict(id='right', k=[0, 0, 0], intensity=a), dict(id='extra', k=[0, 0, .5], intensity=a + b * 3)])
    assert result['families'][0]['best_template'] == 'right'


def test_negative_residual_is_not_relabelled_positive():
    a, b, templates = profiles()
    result = rank_profiles(a * -5, a * 0, np.ones_like(a), templates)
    assert result['status'] == 'no_support'
    assert all(r['amplitude'] == 0 for r in result['templates'])


def test_masks_weights_and_excluded_regions_respected():
    a, b, templates = profiles()
    observed = 5 * a + 200 * b
    mask = np.arange(len(a)) < 400
    result = rank_profiles(observed, a * 0, np.ones_like(a), templates, mask=mask)
    assert result['families'][0]['best_template'] == 'true'
    assert result['valid_points'] == mask.sum()


def test_zero_weight_bins_do_not_create_evidence():
    a, b, templates = profiles()
    weights = np.zeros_like(a); weights[:200] = 1
    assert rank_profiles(a * 1000, a * 0, weights, templates)['status'] == 'no_support'


def test_more_trials_receive_explicit_family_penalty():
    a, b, templates = profiles()
    templates = [dict(id=str(i), k=[0, 0, 0], intensity=a) for i in range(10)] + [dict(id='single', k=[0, 0, .5], intensity=a)]
    result = rank_profiles(a * 5, a * 0, np.ones_like(a), templates)
    assert result['families'][0]['best_template'] == 'single'
    assert result['status'] == 'ambiguous'


def test_incomplete_search_never_claims_a_completed_shortlist():
    a, b, templates = profiles()
    assert rank_profiles(a * 5, a * 0, np.ones_like(a), templates, complete=False)['status'] == 'incomplete'


def test_declared_moment_ceiling_bounds_the_intensity_fit():
    a,b,templates=profiles();templates[0]['maximum_amplitude']=2.
    result=rank_profiles(a*100,a*0,np.ones_like(a),templates)
    assert result['templates'][0]['amplitude']==2.


@pytest.mark.parametrize('problem', ['duplicate', 'negative', 'wrong_grid', 'missing_bin'])
def test_bad_templates_refused(problem):
    a, b, templates = profiles()
    if problem == 'duplicate': templates[1]['id'] = 'true'
    if problem == 'negative': templates[0]['intensity'] = -a
    if problem == 'wrong_grid': templates[0]['intensity'] = a[:-1]
    if problem == 'missing_bin': templates[0]['intensity'] = np.ma.array(a, mask=np.arange(len(a)) == 30)
    with pytest.raises(ValueError):
        rank_profiles(a * 5, a * 0, np.ones_like(a), templates)
