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
