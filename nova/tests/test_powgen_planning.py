"""Scientific failure controls for planning and experimental qualification."""
import json
from pathlib import Path

import numpy as np
import pytest

from gpx_recipe.planning.contracts import validate_scenario
from gpx_recipe.planning.corpus import locked_split, read_inventory, summarize_run
from gpx_recipe.planning.instrument import profiles, selected_profile, coverage
from gpx_recipe.planning.validation import fit_repeat_reference, score_repeats, read_xye
from gpx_recipe.planning.counting import Accumulator
from gpx_recipe.planning.physics import Cylinder, charge_from_power, cylinder_attenuation


def test_profile_inventory_is_hash_checked_and_does_not_certify_rates():
    rows = profiles()
    assert len(rows) >= 100 and len({row['cycle'] for row in rows}) >= 20
    assert {'HR', 'HI'} <= {row['mode'] for row in rows}
    assert 4.797 in {row['wavelength'] for row in rows}
    assert coverage()['validated_count_response_mappings'] == 0
    with pytest.raises(ValueError): selected_profile('closest-wavelength')


def test_packaged_response_contains_only_traceable_derived_coefficients():
    import gpx_recipe.planning.response as response
    root=Path(response.__file__).parent/'data'
    for path in root.rglob('response.npz'):
        response.verify_calibration_asset(path,root)
        with np.load(path,allow_pickle=False) as arrays:
            assert set(arrays.files)=={'d','effective_response_per_uah','background_y','bin_width_us','standard_transmission'}
    assert len(list(root.rglob('response.npz')))==3


def test_same_acquisition_label_across_proposals_cannot_leak():
    rows = [{'ipts': 'IPTS-1', 'sample_id': 12}, {'ipts': 'IPTS-2', 'sample_id': 12},
            {'ipts': 'IPTS-3', 'sample_id': 15}]
    lock = locked_split(rows, development_proposals=['IPTS-1'])
    assert lock['proposal_roles']['IPTS-1'] == lock['proposal_roles']['IPTS-2'] == 'development'
    assert not lock['physical_specimen_independence_verified']
    assert lock == locked_split(rows[::-1], development_proposals=['IPTS-1'])


def test_partial_inventory_cannot_become_validation_corpus(tmp_path):
    p = tmp_path / 'inventory.jsonl'
    p.write_text(json.dumps({'type': 'start', 'schema': 'powgen-planning-inventory/v1'}) + '\n')
    with pytest.raises(ValueError, match='incomplete'): read_inventory(p)


def test_missing_units_and_zero_placeholders_do_not_establish_geometry():
    row = dict(ipts='IPTS-1', run=1, issues=[], products=[], logs={}, metadata={
        'entry/sample/mass': {'value': 1., 'units': ''},
        'entry/sample/height_in_container': {'value': 0., 'units': 'mm'},
        'entry/sample/interior_diameter': {'value': 6., 'units': 'mm'}})
    summary = summarize_run(row)
    assert summary['mass_g'] is None and summary['fill_height_cm'] is None
    assert not summary['eligible_for_absolute_metadata_review']
    assert not summary['absolute_validation_eligible']


def test_signed_and_zero_error_observations_survive_reader(tmp_path):
    p = tmp_path / 'observed.xye'
    p.write_text("'TOF reduced observation\n1 -2 1\n2 0 0\n3 4 2\n")
    arrays, receipt = read_xye(p)
    assert arrays[0, 1] == -2 and arrays[1, 2] == 0
    assert receipt['negative_y_bins'] == receipt['zero_error_bins'] == 1
    p.write_text('1 2 1\n2 4 -1\n3 5 1\n')
    with pytest.raises(ValueError): read_xye(p)


def test_common_run_drift_is_detected_despite_plausible_individual_errors():
    rng = np.random.default_rng(93821)
    n, bins = 80, 600
    mean = np.linspace(5, 20, bins)
    train = mean + rng.normal(size=(n, bins))
    model = fit_repeat_reference(train, np.ones_like(train), np.ones(n), [f't{i}' for i in range(n)])
    observed = mean + rng.normal(size=(n, bins)) + rng.normal(size=(n, 1)) * .3
    result = score_repeats(model, observed, np.ones_like(observed), np.ones(n), [f'v{i}' for i in range(n)])
    assert 0.90 < result['marginal_95_percent_fraction'] < 0.97
    assert 'all_bins' in result['rejected_independent_bin_variance_regions']
    assert result['regions']['all_bins']['variance_ratio'] > 10
    assert not result['experimental_accuracy_certified']


def test_poisson_exposure_model_has_calibrated_gaussian_limit_and_no_leakage():
    rng = np.random.default_rng(2124)
    mean = np.linspace(100, 300, 80)
    charge = np.linspace(10, 80, 100)
    counts = rng.poisson(charge[:, None] * mean)
    observed = counts / charge[:, None]
    errors = np.sqrt(counts + .5) / charge[:, None]
    train = fit_repeat_reference(observed[:40], errors[:40], charge[:40], [f't{i}' for i in range(40)])
    result = score_repeats(train, observed[40:], errors[40:], charge[40:], [f'v{i}' for i in range(60)])
    assert .91 < result['marginal_95_percent_fraction'] < .98
    assert .8 < result['regions']['all_bins']['variance_ratio'] < 1.3
    with pytest.raises(ValueError, match='leakage'):
        score_repeats(train, observed[40:], errors[40:], charge[40:], list(train.training_ids[:1])+[f'v{i}' for i in range(59)])


def test_cumulative_exposures_reuse_the_actual_earlier_counts():
    a = Accumulator([10., 20.], [1., 1.], 37, background_rate=5., blank_exposure=50., reference_counts=1000.)
    first, second = a.advance(10), a.advance(20)
    np.testing.assert_array_equal(first.counts + second.increment, second.counts)
    assert not np.shares_memory(first.counts, second.counts)


def test_shared_monitor_covariance_matches_independent_monte_carlo_ensemble():
    observed=[];reported=[]
    for seed in range(2500):
        sample=Accumulator([800.,1200.],[1.,1.],seed,monitor_rate=1000.).advance(1.)
        observed.append(sample.y)
        reported.append(np.diag(sample.diagonal_variance)+sample.covariance_factors.T@sample.covariance_factors)
    empirical=np.cov(np.asarray(observed).T)
    predicted=np.mean(reported,axis=0)
    np.testing.assert_allclose(empirical,predicted,rtol=.13)
    assert empirical[0,1]>700  # A diagonal-only Poisson model would miss this.


def test_finite_blank_leaves_a_variance_floor_and_correlated_cumulative_looks():
    pairs=[];reported=[]
    for seed in range(2500):
        accumulator=Accumulator([50.],[1.],seed,background_rate=20.,blank_exposure=10.)
        early=accumulator.advance(100.);late=accumulator.advance(10000.)
        pairs.append([early.y[0],late.y[0]]);reported.append(late.sigma[0]**2)
    covariance=np.cov(np.asarray(pairs).T)
    # Analytic Poisson result: finite blank contributes 20/10 at both looks;
    # the common earlier sample counts add 70/10000 to their covariance.
    np.testing.assert_allclose(covariance[1,1],2.007,rtol=.10)
    np.testing.assert_allclose(covariance[0,1],2.007,rtol=.10)
    np.testing.assert_allclose(covariance[1,1],np.mean(reported),rtol=.10)


def test_attenuation_and_exposure_obey_analytic_limits():
    geometry=Cylinder(mass_g=1.,radius_cm=.3,beam_width_cm=.6,beam_height_cm=4.)
    angles=np.array([10.,90.,170.])
    np.testing.assert_allclose(cylinder_attenuation(geometry,angles,0),1.,atol=1e-14)
    medium=cylinder_attenuation(geometry,angles,2.,quadrature=48)
    dense=cylinder_attenuation(geometry,angles,20.,quadrature=48)
    assert np.all((0<dense)&(dense<medium)&(medium<1))
    np.testing.assert_allclose(medium,cylinder_attenuation(geometry,angles,2.,quadrature=96),rtol=2e-4)
    assert charge_from_power(3600.,1.,1.)==1000.
    assert charge_from_power(3600.,1.,2.)==500.


def test_scenario_rejects_nonphysical_values_and_unbounded_grids(tmp_path):
    cif = tmp_path / 'source.cif'; cif.write_text('data_test\n')
    source = {'schema': 'powgen-planning-scenario/v1', 'cif': str(cif),
              'sample': {'mass_g': 1., 'radius_cm': .3},
              'beam': {'power_mw': 1.4, 'proton_energy_gev': 1.3}}
    assert validate_scenario(source)['response_mode'] == 'profile_only'
    for change in [{'seconds': [10, 5]}, {'seed': True}, {'d_min': float('nan')},
                   {'relative_step': 1e-10}, {'beam': {'power_mw': 1.4}},
                   {'sample': {'mass_g': float('nan'), 'radius_cm': .3}},
                   {'isotopes': {'B': 11}}, {'response_mode': 'pretend_validated'}]:
        with pytest.raises((ValueError, TypeError)): validate_scenario({**source, **change})
