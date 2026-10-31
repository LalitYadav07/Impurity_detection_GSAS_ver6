"""Fit-domain regressions using synthetic profiles, not experiment-specific pins."""
import copy
from types import SimpleNamespace

import pytest

from scripts.gsas_safe_limits import apply_safe_limits, calculate_safe_d_min


def histogram(lower, upper, *, sigma=(1.0, 0.0, 0.0, 0.0), coordinates=None, name='Bank test'):
    # sigma order is constant, d, d², d⁴. All values are synthetic.
    inst = {'Type': ['PNT'], 'difC': [1000.0, 1000.0, False]}
    inst.update({key: [value, value, False] for key, value in zip(
        ('sig-0', 'sig-q', 'sig-1', 'sig-2'), sigma)})
    inst.update(coordinates or {})
    def value(key):
        row = inst.get(key, [0.0, 0.0])
        return row[1]
    def tof(d):
        return value('difC')*d + value('difA')*d*d + value('difB')/d + value('Zero')
    fitted = [tof(lower), tof(upper)]
    return SimpleNamespace(name=name, data={'Instrument Parameters': [inst, {}],
                          'Limits': [(0.0, fitted[1]*2), fitted, [fitted[0]+1, fitted[0]+2]]})


def project(*histograms):
    return SimpleNamespace(histograms=lambda: list(histograms))


def test_valid_range_and_exclusions_are_unchanged():
    h = histogram(0.4, 6.0)
    before = copy.deepcopy(h.data)
    assert apply_safe_limits(project(h)) is False
    assert h.data == before


def test_valid_small_d_range_is_not_cut_at_an_arbitrary_default():
    h = histogram(0.08, 0.5)
    before = copy.deepcopy(h.data)
    assert not apply_safe_limits(project(h))
    assert h.data == before


def test_only_invalid_low_tail_retains_existing_buffer_policy():
    h = histogram(0.2, 2.0, sigma=(-0.25, 0.0, 1.0, 0.0))
    original = copy.deepcopy(h.data['Limits'])
    assert apply_safe_limits(project(h)) is True
    assert h.data['Limits'][1] == pytest.approx([550.0, 2000.0])
    assert h.data['Limits'][0] == original[0]
    assert h.data['Limits'][2] == original[2]


def test_negative_quartic_cannot_be_fixed_by_largest_root():
    h = histogram(0.4, 3.0, sigma=(-1.0, 0.0, 4.0, -1.0))
    before = copy.deepcopy(h.data)
    with pytest.raises(ValueError, match='lower cutoff cannot safely repair'):
        apply_safe_limits(project(h))
    assert h.data == before
    with pytest.raises(ValueError, match='invalid high-d tail'):
        calculate_safe_d_min(h.data['Instrument Parameters'][0])


def test_negative_coefficients_are_allowed_on_a_valid_fitted_interval():
    h = histogram(0.8, 1.5, sigma=(-1.0, 0.0, 4.0, -1.0))
    before = copy.deepcopy(h.data)
    assert not apply_safe_limits(project(h))
    assert h.data == before


def test_bounded_profile_can_repair_only_its_low_tail_with_finite_upper_bound():
    h = histogram(0.4, 1.5, sigma=(-1.0, 0.0, 4.0, -1.0))
    assert apply_safe_limits(project(h))
    assert h.data['Limits'][1][0] == pytest.approx(((2-3**0.5)**0.5 + 0.05)*1000)


@pytest.mark.parametrize('sigma', [(4.0, 0.0, -5.0, 1.0), (0.999999, 0.0, -2.0, 1.0)])
def test_internal_invalid_region_is_not_hidden_by_a_grid_or_one_cutoff(sigma):
    h = histogram(0.4, 3.0, sigma=sigma)
    before = copy.deepcopy(h.data)
    with pytest.raises(ValueError, match='lower cutoff cannot safely repair'):
        apply_safe_limits(project(h))
    assert h.data == before


def test_tangent_zero_inside_the_range_is_not_treated_as_strictly_positive():
    h = histogram(0.4, 3.0, sigma=(1.0, 0.0, -2.0, 1.0))
    with pytest.raises(ValueError, match='lower cutoff cannot safely repair'):
        apply_safe_limits(project(h))


@pytest.mark.parametrize('sigma', [(-1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)])
def test_no_positive_variance_fails_instead_of_falling_back(sigma):
    h = histogram(0.4, 3.0, sigma=sigma)
    with pytest.raises(ValueError, match='positive intervals: none'):
        apply_safe_limits(project(h))


def test_current_not_original_native_parameters_define_the_safety_check():
    h = histogram(0.5, 3.0, sigma=(1.0, 0.0, 1.0, 0.0))
    h.data['Instrument Parameters'][0]['sig-0'] = [1.0, -1.0, False]
    assert apply_safe_limits(project(h))
    assert h.data['Limits'][1][0] == pytest.approx(1050.0)


def test_full_tof_polynomial_and_current_coordinate_values_are_used():
    h = histogram(0.2, 2.0, sigma=(-0.25, 0.0, 1.0, 0.0), coordinates={
        'difC': [999999.0, 1000.0, False], 'difA': [0.0, 2.0, False],
        'difB': [0.0, 3.0, False], 'Zero': [0.0, 7.0, False]})
    assert apply_safe_limits(project(h))
    assert h.data['Limits'][1][0] == pytest.approx(1000*.55 + 2*.55**2 + 3/.55 + 7)


def test_large_cutoff_fails_instead_of_skipping_the_guard():
    h = histogram(0.4, 3.0, sigma=(-4.0, 0.0, 1.0, 0.0))
    before = copy.deepcopy(h.data)
    with pytest.raises(ValueError, match='more than half'):
        apply_safe_limits(project(h))
    assert h.data == before


def test_multi_bank_failure_does_not_partially_change_other_limits():
    first = histogram(0.2, 2.0, sigma=(-.25, 0.0, 1.0, 0.0), name='lower-tail')
    second = histogram(0.4, 3.0, sigma=(-1.0, 0.0, 4.0, -1.0), name='invalid-upper-tail')
    before = [copy.deepcopy(h.data) for h in (first, second)]
    with pytest.raises(ValueError, match='invalid-upper-tail'):
        apply_safe_limits(project(first, second))
    assert [first.data, second.data] == before


def test_cw_profiles_are_not_changed_or_interpreted_as_tof():
    h = histogram(0.2, 2.0, sigma=(-999.0, 0.0, 0.0, 0.0))
    h.data['Instrument Parameters'][0]['Type'] = ['PXC']
    before = copy.deepcopy(h.data)
    assert not apply_safe_limits(project(h))
    assert h.data == before


@pytest.mark.parametrize('key,value', [('sig-1', float('nan')), ('difC', 0.0), ('difA', float('inf'))])
def test_malformed_profile_fails_with_histogram_context(key, value):
    h = histogram(0.4, 3.0)
    h.data['Instrument Parameters'][0][key] = [value, value, False]
    with pytest.raises(ValueError, match='Histogram Bank test'):
        apply_safe_limits(project(h))
