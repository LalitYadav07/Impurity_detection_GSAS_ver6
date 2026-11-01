"""Native measured and controlled checks of moving component support."""
from dataclasses import replace

import pytest

from radar_pd_nova.pattern_transition import compare_observed_patterns, prepare_pattern
from radar_pd_nova.transition_signals import BoundaryPolicy, apply_policy, measure_signals
from test_pattern_transition import observed
from test_splitting_profiles import doublet


def dilate(pattern, fraction):
    factor = 1 + fraction
    return replace(pattern, x=pattern.x * factor,
                   intensity=pattern.intensity / factor,
                   sigma=pattern.sigma / factor,
                   observation_id=f'dilated-{fraction}')


def decision(before, after):
    a, b = prepare_pattern(before), prepare_pattern(after)
    raw = compare_observed_patterns(a, b)
    return apply_policy(measure_signals(a, b, raw), BoundaryPolicy())


@pytest.fixture(scope='module')
def multi_peak_pattern():
    # Four synthetic, resolved doublets. A fixed absolute crop can contain
    # two components on only one side of an otherwise rigid translation.
    peaks = [(center * ratio, height, .0013)
             for center in (1300, 2300, 4000, 6800)
             for ratio, height in ((1, 60), (1.01, 45))]
    peaks += [(1700, 70, .0025), (8500, 70, .0025)]
    return observed(factors=[0] * 8, new=peaks, noise=.03)


@pytest.mark.parametrize('fraction', [
    -.019, -.015, -.01, -.008, -.005, -.003, -.001,
    .001, .003, .005, .008, .01, .015, .019,
])
def test_rigid_dilation_of_native_multi_peak_pattern_does_not_create_components(multi_peak_pattern, fraction):
    result = decision(multi_peak_pattern, dilate(multi_peak_pattern, fraction))
    assert result['status'] == 'no_resolved_change', result
    assert not result['indicators']['splitting']['detected']


@pytest.mark.parametrize('fraction', [-.015, -.005, .005, .015])
def test_rigid_motion_preserves_synthetic_existing_doublet(fraction):
    a = doublet(gap=24)
    result = decision(a, dilate(a, fraction))
    assert not result['indicators']['splitting']['detected'], result


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('fraction', [-.01, 0., .01])
def test_real_split_or_merge_survives_common_motion(reverse, fraction):
    a, b = doublet(gap=0), dilate(doublet(gap=24, seed=7), fraction)
    if reverse:
        a, b = b, a
    result = decision(a, b)
    assert result['indicators']['splitting']['detected'], result
    changes = [v for v in result['indicators']['splitting']['regions']
               if v.get('kind') == 'local_resolved_component_count_change']
    assert changes
    assert any(v['before_components'] == (2 if reverse else 1)
               and v['after_components'] == (1 if reverse else 2) for v in changes)


@pytest.mark.parametrize('fraction', [-.01, 0., .01])
def test_differential_doublet_separation_remains_evidence(fraction):
    result = decision(doublet(gap=24), dilate(doublet(gap=32, seed=7), fraction))
    events = result['indicators']['splitting']['regions']
    assert any(v.get('kind') == 'local_component_separation_change'
               and v['separation_change_fwhm'] > .4 for v in events), result


@pytest.mark.parametrize('fraction', [-.01, 0., .01])
def test_independent_new_peaks_remain_detectable(fraction):
    a = observed()
    b = dilate(observed(new=[(2700, 25, .0025), (6000, 25, .0025)], seed=8), fraction)
    result = decision(a, b)
    assert result['indicators']['appearance']['detected'], result
