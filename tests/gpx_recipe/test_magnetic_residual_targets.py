import copy

import numpy as np
import pytest

from gpx_recipe.symmetry.native_magnetic import (
    audit_target_peak_windows,
    find_positive_residual_targets,
)


class Histogram:
    name = 'synthetic neutron pattern'

    def __init__(self, x, observed, *, limits, exclusions=(), weights=None,
                 calculated=None, mask=None, storage='matrix', instrument='PNC'):
        x = np.asarray(x, dtype=float)
        background = np.full_like(x, 10.)
        matrix = np.ma.array([x, observed, np.ones_like(x) if weights is None else weights,
                              background if calculated is None else calculated,
                              background, np.zeros_like(x)], mask=False)
        if mask is not None:
            matrix.mask[mask] = True
        arrays = [row.copy() for row in matrix] if storage == 'list' else matrix
        self.data = {'data': [{'wtFactor': 1.}, arrays],
                     'Limits': [[float(x[0]), float(x[-1])], list(limits), *map(list, exclusions)]}
        self.InstrumentParameters = {'Type': [instrument]}

    def Excluded(self):
        return self.data['Limits'][2:]


def bump(x, center, amplitude=200., width=1.5):
    return amplitude*np.exp(-.5*((x-center)/width)**2)


def test_stronger_below_limit_er_like_peaks_do_not_become_targets():
    x = np.arange(5., 127., .05)
    y = 10.+bump(x, 7.875, 5000., .12)+bump(x, 9.325, 4000., .12)+bump(x, 18.775, 200., .12)
    h = Histogram(x, y, limits=(10., 126.))
    before = copy.deepcopy(h.data)
    baseline = find_positive_residual_targets(h)
    assert baseline['targets']
    assert all(10. <= row['x'] <= 126.05 for row in baseline['targets'])
    assert any(abs(row['x']-18.775) < .2 for row in baseline['targets'])
    final = Histogram(x, y, limits=(10., 126.), calculated=y)
    result = audit_target_peak_windows(final, baseline)
    assert result['underfit_peak_count'] == 0
    assert result['weighted_residual_improvement'] == pytest.approx(1.)
    np.testing.assert_array_equal(h.data['data'][1], before['data'][1])


@pytest.mark.parametrize('scale,offset,instrument', [(1.,0.,'PNC'), (100.,1000.,'PNT')])
def test_cw_and_tof_use_native_upper_searchsorted_bin(scale, offset, instrument):
    indices = np.arange(201.)
    x = offset+scale*indices
    y = 10.+bump(indices, 81., 200., .1)+bump(indices, 82., 1e6, .1)
    limits = (offset+40.2*scale, offset+80.2*scale)
    h = Histogram(x, y, limits=limits, instrument=instrument)
    baseline = find_positive_residual_targets(h)
    assert baseline['valid'][81] and not baseline['valid'][82]
    assert not baseline['valid'][40] and baseline['valid'][41]
    final = Histogram(x, y, limits=limits, calculated=y, instrument=instrument)
    result = audit_target_peak_windows(final, baseline)
    assert result['target_count'] == 1
    row = result['peaks'][0]
    assert row['right_x'] == x[81]
    assert row['left_x'] >= x[41]
    assert row['final_coverage'] == pytest.approx(1.)
    assert result['fitted_point_count'] == 41


@pytest.mark.parametrize('storage', ['matrix', 'list'])
def test_true_masks_and_nonpositive_weights_cannot_create_targets(storage):
    x = np.arange(201.)
    y = 10.+sum(bump(x, center, 500. if center != 140 else 100.) for center in (50.,80.,110.,140.))
    weights = np.ones_like(x)
    weights[76:85] = 0.
    weights[106:115] = -1.
    mask = (1, slice(46,55))
    h = Histogram(x, y, limits=(20.,180.), weights=weights, mask=mask, storage=storage)
    baseline = find_positive_residual_targets(h)
    assert not baseline['valid'][50] and not baseline['valid'][80] and not baseline['valid'][110]
    assert baseline['targets'] and all(135. < r['x'] < 145. for r in baseline['targets'])


@pytest.mark.parametrize('different_grid', [False, True])
def test_windows_stop_at_masked_gaps_and_exclusions(different_grid):
    x = np.arange(201.)
    y = 10.+bump(x, 96., width=1.)
    baseline = find_positive_residual_targets(
        Histogram(x, y, limits=(30.,170.), exclusions=[(100.,110.)], mask=(1,slice(90,93))))
    fx = np.arange(0.,200.5,.5) if different_grid else x
    fy = 10.+bump(fx, 96., width=1.)
    # A large calculated signal beyond the gap must not enter this target's integral.
    calc = fy.copy()
    calc[fx >= 100.] = 1e7
    mask = (1, (fx >= 90.) & (fx <= 92.))
    final = Histogram(fx, fy, limits=(30.,170.), exclusions=[(100.,110.)], calculated=calc, mask=mask)
    result = audit_target_peak_windows(final, baseline)
    assert result['target_count'] == 1
    row = result['peaks'][0]
    assert row['left_x'] == 93. and row['right_x'] == 99.
    assert row['final_coverage'] == pytest.approx(1.)
    assert row['weighted_residual_improvement'] == pytest.approx(1.)
    assert result['grid_alignment'] == ('final_interpolated_to_baseline' if different_grid else 'native')


def test_no_supported_weighted_bins_reports_no_targets():
    x = np.arange(201.)
    h = Histogram(x, 10.+bump(x, 100.), limits=(30.,170.), weights=np.zeros_like(x))
    baseline = find_positive_residual_targets(h)
    result = audit_target_peak_windows(h, baseline)
    assert baseline['targets'] == [] and result['peaks'] == []
    assert result['fitted_point_count'] == 0
    assert result['weighted_coverage_error'] is None
    assert result['weighted_residual_improvement'] is None


@pytest.mark.parametrize('count', [1, 2, 3])
def test_short_segments_preserve_shape_and_require_three_window_bins(count):
    x = np.arange(201.)
    weights = np.zeros_like(x)
    weights[99:99+count] = 1.
    h = Histogram(x, 10.+bump(x, 100.), limits=(20.,180.), weights=weights)
    baseline = find_positive_residual_targets(h)
    assert baseline['targets']
    result = audit_target_peak_windows(h, baseline)
    assert result['fitted_point_count'] == count
    assert baseline['valid'].shape == x.shape
    if count < 3:
        assert result['peaks'] == [] and result['target_count'] == 0
        assert result['weighted_coverage_error'] is None
        assert result['weighted_residual_improvement'] is None
    else:
        assert result['target_count'] == 1
        assert result['peaks'][0]['native_point_count'] == 3


def test_supported_window_retains_original_signal_and_residual_definitions():
    x = np.arange(201.)
    y = np.full_like(x, 10.)
    y[98:103] += [0., 10., 20., 10., 0.]
    baseline = find_positive_residual_targets(Histogram(x, y, limits=(20.,180.)))
    baseline['targets'] = [dict(index=100, x=100., baseline_significance=20.)]
    calc = 10.+.5*(y-10.)
    final = Histogram(x, y, limits=(20.,180.), calculated=calc)
    result = audit_target_peak_windows(final, baseline, half_width=2)
    row = result['peaks'][0]
    assert (row['left_x'], row['right_x']) == (98., 102.)
    assert row['observed_signal_above_background'] == 40.
    assert row['final_calculated_signal_above_background'] == 20.
    assert row['baseline_weighted_absolute_residual'] == 40.
    assert row['final_weighted_absolute_residual'] == 20.
    assert row['final_coverage'] == .5 and row['underfit'] is False
    assert result['weighted_residual_improvement'] == .5


def test_baseline_snapshot_survives_native_in_place_profile_update():
    x = np.arange(201.)
    y = 10.+bump(x, 100.)
    histogram = Histogram(x, y, limits=(20.,180.))
    baseline = find_positive_residual_targets(histogram)
    histogram.data['data'][1][3] = y
    result = audit_target_peak_windows(histogram, baseline)
    assert result['weighted_residual_improvement'] == pytest.approx(1.)
    assert result['peaks'][0]['baseline_calculated_signal_above_background'] == 0.


def test_smoothing_cannot_move_a_peak_across_one_invalid_bin():
    x = np.arange(101.)
    observed = np.full_like(x, 20.)
    observed[49] = 1000.
    calculated = np.full_like(x, 20.)
    weights = np.ones_like(x)
    weights[50] = 0.
    baseline = find_positive_residual_targets(
        Histogram(x, observed, calculated=calculated, weights=weights, limits=(20.,80.)))
    assert [row['index'] for row in baseline['targets']] == [49]
    assert all(observed[row['index']] > calculated[row['index']] for row in baseline['targets'])


@pytest.mark.parametrize('extra_native_support', [False, True])
def test_interpolation_cannot_invent_three_native_points_in_a_window(extra_native_support):
    x = np.arange(101.)
    observed = np.full_like(x, 20.)
    observed[48:53] = 100.
    baseline = find_positive_residual_targets(
        Histogram(x, observed, calculated=np.full_like(x, 20.), limits=(20.,80.)))
    baseline['targets'] = [dict(index=50, x=50., baseline_significance=80.)]
    final_x = np.array([0.,30.,45.,55.,70.,100.]) if extra_native_support else np.array([0.,45.,55.,100.])
    final_y = np.full_like(final_x, 100.)
    final_y[[0,-1]] = 20.
    weights = np.ones_like(final_x)
    weights[[0,-1]] = 0.
    final = Histogram(final_x, final_y, calculated=final_y, weights=weights, limits=(20.,80.))
    result = audit_target_peak_windows(final, baseline)
    # Even a longer valid native run has only45 and55 within the38..62 window.
    assert result['fitted_point_count'] == (41 if extra_native_support else 0)
    assert result['target_count'] == 0 and result['peaks'] == []
    assert result['weighted_coverage_error'] is None
    assert result['weighted_residual_improvement'] is None


def test_coarse_baseline_cannot_hide_a_gap_between_native_segments():
    x = np.arange(0.,101.,10.)
    observed = np.full_like(x, 20.)
    observed[5] = 100.
    baseline = find_positive_residual_targets(
        Histogram(x, observed, calculated=np.full_like(x, 20.), limits=(20.,80.)))
    baseline['targets'] = [dict(index=5, x=50., baseline_significance=80.)]
    final_x = np.arange(101.)
    final_y = np.full_like(final_x, 20.)
    final_y[50] = 100.
    weights = np.ones_like(final_x)
    weights[55] = 0.
    final = Histogram(final_x, final_y, weights=weights, limits=(20.,80.))
    result = audit_target_peak_windows(final, baseline, half_width=2)
    assert result['target_count'] == 1
    assert result['peaks'][0]['left_x'] == 30.
    assert result['peaks'][0]['right_x'] == 50.
    assert result['peaks'][0]['native_point_count'] == 21


def test_three_actual_native_points_can_support_an_interpolated_window():
    x = np.arange(101.)
    baseline = find_positive_residual_targets(
        Histogram(x, 20.+bump(x, 50.), calculated=np.full_like(x, 20.), limits=(20.,80.)))
    baseline['targets'] = [dict(index=50, x=50., baseline_significance=80.)]
    final = Histogram([0.,45.,50.,55.,100.], [20.,100.,100.,100.,20.],
                      calculated=[20.,100.,100.,100.,20.], weights=[0.,1.,1.,1.,0.], limits=(20.,80.))
    result = audit_target_peak_windows(final, baseline)
    assert result['target_count'] == 1
    assert result['peaks'][0]['native_point_count'] == 3
    assert result['peaks'][0]['final_coverage'] == pytest.approx(1.)


def test_subgrid_exclusion_still_splits_adjacent_baseline_points():
    x = np.arange(101.)
    observed = np.full_like(x, 20.)
    observed[48:53] = 100.
    baseline = find_positive_residual_targets(
        Histogram(x, observed, calculated=np.full_like(x, 20.), limits=(20.,80.)))
    baseline['targets'] = [dict(index=50, x=50., baseline_significance=80.)]
    final_x = np.arange(0.,100.25,.25)
    final_y = np.where(np.abs(final_x-50.) <= 2., 100., 20.)
    final = Histogram(final_x, final_y, calculated=final_y, limits=(20.,80.), exclusions=[(49.25,49.75)])
    result = audit_target_peak_windows(final, baseline)
    assert result['target_count'] == 1
    assert result['peaks'][0]['left_x'] >= 50.
    assert result['peaks'][0]['native_point_count'] >= 3
