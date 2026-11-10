"""Current-cycle effective V response for conditional POWGEN count planning.

Flux, solid angle and efficiency remain a measured combined response. This
first adapter uses an explicitly elastic, isotropic vanadium reference law,
simple cylinder attenuation, measured angular cells and empirical peak profiles.
It returns actual integer-count predictions and a focused pattern normalized
with the captured native processed V and smoothed empty-can references. The
terminal reduction has a separate numerical replay check; this is not physical
qualification of the raw-count response or a simulated raw-event reduction.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .count_response import on_off_posterior
from .empirical_counts import ReferenceCountCampaign
from .focused_reduction import overlap_map, project_reduction_reference
from .physics import AVOGADRO, Cylinder, cylinder_attenuation
from .response import cross_sections, verify_calibration_asset

DATA = Path(__file__).parent / 'data/current-2026b-hr-pac-b2'
BUNDLE_SHA256 = '181c18d7d5458793293709b52e1dc9b5219132854dd31f812b77736255cf54f2'
DEFAULT_PROFILE = '2026B_HighRes_60HzB2_CWL1p5.instprm'
BUNDLES = {
    DEFAULT_PROFILE: (DATA, BUNDLE_SHA256),
    '2026B_HighRes_60HzB1_CWL0p8.instprm': (
        DATA.parent/'current-2026b-hr-pac-b1',
        'a00401a3241b6376dcdf1a3ac27eac145767fc4d1cb993bd85122ba3b4297c04'),
    '2026B_HighRes_60HzB3_CWL2p665.instprm': (
        DATA.parent/'current-2026b-hr-pac-b3',
        '22b9ad29c8acd4964bb8d95e1a40b3b78ccc070be0cf0c946f833316ac0b1e78'),
}


def bundle_directory(profile_id=DEFAULT_PROFILE):
    try:
        return BUNDLES[profile_id][0]
    except KeyError:
        raise ValueError('Current count response requires a supported 2026B HR 60 Hz PAC configuration') from None


def load_bundle(profile_id=DEFAULT_PROFILE):
    directory = bundle_directory(profile_id)
    raw = (directory/'bundle.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != BUNDLES[profile_id][1]:
        raise ValueError('The current-cycle response manifest changed')
    meta = json.loads(raw)
    if meta['profile_id'] != profile_id:
        raise ValueError('The current-cycle response belongs to another configuration')
    for row in meta['assets']:
        path = (directory/row['name']).resolve()
        if not path.is_relative_to(directory.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('A current-cycle calibration asset is missing or changed')
    with np.load(directory/'counts.npz', allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    with np.load(directory/'reduction.npz', allow_pickle=False) as data:
        arrays.update({'reduction_'+key: data[key].copy() for key in data.files})
    if meta.get('standard_calibration'):
        with np.load(directory/'standard-calibration.npz', allow_pickle=False) as data:
            arrays.update({'standard_'+key: data[key].copy() for key in data.files})
    return meta, arrays


def profile_bin_edges(x, d):
    # Same center/edge convention as the existing synthetic SLOG exporter.
    def edges(v):
        out = np.empty(len(v)+1)
        out[0] = v[0] - (v[1]-v[0])/2
        for i in range(len(v)):
            out[i+1] = 2*v[i] - out[i]
        if np.any(np.diff(out) <= 0):
            raise ValueError('Requested centers do not admit increasing bin edges')
        return out
    return edges(np.asarray(x, float)), edges(np.asarray(d, float))


def _attenuation(geometry, quantities, metadata, d, angle, table):
    # Smooth wavelength-dependent attenuation is interpolated; sharp resonant
    # laws are evaluated on the requested grid directly.
    from .isotopes import resonant
    sharp = any(resonant(row[1]) for row in metadata.get('neutron_table', {}).values())
    points = d if sharp else np.geomspace(d[0], d[-1], 160)
    wave = 2*points*np.sin(np.deg2rad(angle/2))
    absorption, _, scattering, _ = cross_sections(metadata, wave, table)
    mu = quantities['cells_per_cm3']*(absorption+scattering)*1e-24
    result = cylinder_attenuation(geometry, np.full_like(points, angle), mu, quadrature=16)
    return result if sharp else np.interp(d, points, result)


def current_response(config, metadata, arrays, sample, seed=0, *, container='PAC06', intensity_calibration='vanadium', reference_law=None, sample_diffuse=None, sample_elastic=None):
    bundle, observed = load_bundle(config['id'])
    if config['id'] != bundle['profile_id'] or metadata['profile_sha256'] != bundle['profile_sha256']:
        raise ValueError('The instrument profile does not match this current-cycle count response')
    if intensity_calibration not in {'vanadium','silicon_63584'}:
        raise ValueError('Choose the V-based physical estimate or explicit Si integrated-count normalization')
    if intensity_calibration=='silicon_63584' and not bundle.get('standard_calibration'):
        raise ValueError('Si integrated-count normalization is available only for 2026B HR 60 Hz 1.5 Å')
    if sample.get('wall_thickness_cm', 0):
        raise ValueError('This response needs a separately specified can attenuation model for nonzero wall thickness')
    if container not in {'PAC06','none'} or (container=='PAC06' and sample['radius_cm']>.3):
        raise ValueError('Select a supported container with sufficient diameter for the sample')
    d, x = arrays['d'], arrays['x']
    xedges, dedges = profile_bin_edges(x, d)
    overlap = overlap_map(observed['d_edges_a'], dedges)
    on, off = observed['on_counts'], observed['off_counts']
    supported = (on+off) > 0
    if sample_diffuse is not None and (reference_law is None or sample_diffuse.reference_law is not reference_law):
        raise ValueError('Diffuse and Bragg predictions must share the same physical reference-law object')
    if sample_elastic is not None and (reference_law is None or sample_elastic.reference_law is not reference_law):
        raise ValueError('Elastic geometry and other predictions must share the same physical reference-law object')
    correction = None
    diffuse_factors = None
    elastic_factors = None
    if reference_law is not None:
        if intensity_calibration != 'vanadium':
            raise ValueError('A reference-law transfer cannot silently reuse the old Si normalization')
        required = supported & (np.asarray(overlap.getnnz(axis=0)) > 0)[None, :]
        if not np.any(required):
            raise ValueError('A reference-law transfer needs measured cells in the requested range')
        correction = reference_law.factors(config['id'], BUNDLES[config['id']][1],
            observed['d_edges_a'], required)
        if sample_diffuse is not None:
            diffuse_factors = sample_diffuse.factors(config['id'], BUNDLES[config['id']][1],
                observed['d_edges_a'], required, metadata, sample)
        if sample_elastic is not None:
            elastic_factors = sample_elastic.factors(config['id'], BUNDLES[config['id']][1],
                observed['d_edges_a'], required, metadata, sample)
    # Empty cells supply no measured coverage; do not populate them with prior
    # counts and pretend that an unmeasured angle/wavelength cell is calibrated.
    posterior = on_off_posterior(on[supported], off[supported],
        bundle['on_charge_uah'], bundle['off_charge_uah'])
    base = Path(__file__).parent/'data'
    table_path = base/'calibration/cross-sections.json'
    verify_calibration_asset(table_path, base)
    table = json.loads(table_path.read_text())
    geometry = Cylinder(**sample)
    q = geometry.quantities(metadata['cell_mass_da'], metadata['cell_volume_a3'])
    reference = bundle['vanadium_assumptions']
    ref = Cylinder(mass_g=reference['density_g_cm3']*np.pi*reference['radius_cm']**2*reference['height_cm'],
        radius_cm=reference['radius_cm'], packing_fraction=1.,
        beam_width_cm=reference['beam_width_cm'], beam_height_cm=reference['beam_height_cm'])
    ref_q = ref.quantities(reference['atom_mass_da'], reference['atom_mass_da']/AVOGADRO/reference['density_g_cm3']*1e24)
    cells_ratio = q['unit_cells']*q['vertical_illumination_fraction'] / (ref_q['unit_cells']*ref_q['vertical_illumination_fraction'])
    calibration = metadata['tof_calibration']
    sine = np.sin(np.deg2rad(calibration['2-theta']/2))
    jacobian = calibration['difC'] + 2*calibration['difA']*d - calibration['difB']/d**2
    if sine <= 0 or np.any(jacobian <= 0):
        raise ValueError('Invalid physical TOF Jacobian')
    # Native GSAS A_t = sin(theta_focus)*m*d_h**4*|F|**2/V.
    # Physical fixed-angle lambda area A_lambda/A_t=sin(theta)/(2*pi*sin(theta_focus)).
    # dSigma/dOmega(lambda)=y_t(t(d))*dt/dd/(4*pi*sin(theta_focus)).
    # Divide by the assumed V incoherent law sigma_V/(4*pi).
    normalizer_b = reference['incoherent_b'] if sample_elastic is None else sample_elastic.normalizer_b
    profile_factor = jacobian / (sine*normalizer_b)
    maps = {key: [] for key in ('nuclear', 'magnetic', 'incoherent', 'blank')}
    if sample_diffuse is not None:
        maps['diffuse'] = []
    vxs = table['elements']['V']
    for g, angle in enumerate(observed['angles_deg']):
        wave = 2*d*np.sin(np.deg2rad(angle/2))
        transfer = 1.
        if sample_elastic is None:
            ts = _attenuation(geometry, q, metadata, d, angle, table)
            points = np.geomspace(d[0], d[-1], 160)
            vwave = 2*points*np.sin(np.deg2rad(angle/2))
            vmu = ref_q['cells_per_cm3']*(vxs['scattering_b']+vxs['absorption_2200_b']*vwave/table['reference_wavelength_a'])*1e-24
            tv = np.interp(d, points, cylinder_attenuation(ref, np.full_like(points, angle), vmu, quadrature=16))
            transfer = cells_ratio*ts/tv
        _, incoherent, _, _ = cross_sections(metadata, wave, table)
        if sample_diffuse is not None:
            incoherent = np.zeros_like(incoherent)
        nuclear = arrays['nuclear']
        if 'angular_nuclear' in arrays:
            angles = arrays['kernel_angles_deg']
            hi = min(len(angles)-1, max(1, int(np.searchsorted(angles, angle))))
            weight = np.clip((angle-angles[hi-1])/(angles[hi]-angles[hi-1]), 0., 1.)
            nuclear = (1-weight)*arrays['angular_nuclear'][hi-1]+weight*arrays['angular_nuclear'][hi]
        block = overlap[:, supported[g]]
        source_block = block if correction is None else block.multiply(correction[g, supported[g]])
        if elastic_factors is not None:
            # This numerator already contains the whole specimen's cell count
            # and path weights. Its physical V denominator is shared directly;
            # do not also apply E_V/P_V or the old transverse-cylinder ratio.
            source_block = block.multiply(elastic_factors[g, supported[g]])
        for name, factor in [('nuclear', nuclear*profile_factor*transfer),
                             ('magnetic', arrays['magnetic']*profile_factor*transfer),
                             ('incoherent', incoherent/normalizer_b*transfer)]:
            maps[name].append(source_block.multiply(factor[:, None]))
        maps['blank'].append(block)
        if sample_diffuse is not None:
            # The whole-specimen diffuse/reference ratio already includes its
            # scattering and path weights. Do not apply the elastic correction
            # or the sample/reference cell-number ratio a second time.
            maps['diffuse'].append(block.multiply(diffuse_factors[g, supported[g]]))
    maps = {key: sparse.hstack(parts, format='csr') for key, parts in maps.items()}
    components = {key: np.asarray(maps[key] @ posterior.source_mean) for key in ('nuclear', 'magnetic', 'incoherent')}
    if sample_diffuse is not None:
        components['diffuse'] = np.asarray(maps['diffuse'] @ posterior.source_mean)
    normalization, subtraction = project_reduction_reference(
        observed['reduction_tof_edges_us'], observed['reduction_vanadium_rate'],
        observed['reduction_empty_smoothed_rate'], xedges,
        scale=bundle['terminal_reduction']['scale_data'])
    source_map = maps['nuclear'] + maps['magnetic'] + maps['incoherent']
    if sample_diffuse is not None:
        source_map = source_map + maps['diffuse']
    options={}
    background_map=maps['blank']
    if container=='PAC06':
        # Empty-can run already contains instrumental background. Replace the
        # V blank contribution; adding both would count that background twice.
        background_map=sparse.csr_matrix(source_map.shape)
        options=dict(independent_background_counts=observed['can_counts'],
            independent_background_charge=bundle['can_reference']['charge_uah'],
            independent_background_map=overlap)
    if container=='PAC06':
        options['reduction_background']=subtraction
    # No-can mode removes its conditional instrument-blank mean; it does not
    # pretend to have a native no-can subtraction reference for this cycle.
    standard=None
    if intensity_calibration=='silicon_63584':
        from .standard_counts import StandardCountCampaign
        if not np.array_equal(observed['standard_source_cell_indices'],np.flatnonzero(supported.ravel())):
            raise ValueError('Standard transfer vector does not match current reference cells')
        campaign=StandardCountCampaign(posterior,source_map,background_map,normalization,seed,
            calibration_weights=observed['standard_source_weights'],
            standard_on_counts=int(observed['standard_on_counts']),standard_charge=float(observed['standard_on_charge_uah']),
            can_counts=observed['can_counts'],can_charge=bundle['can_reference']['charge_uah'],
            calibration_can_mask=observed['standard_can_mask'],can_map=overlap if container=='PAC06' else None,
            reduction_background=subtraction if container=='PAC06' else None)
        components={key:value*campaign.gain for key,value in components.items()}
        standard=dict(bundle['standard_calibration'],mean_gain=campaign.gain,
            applied=True,denominator_cv=campaign.denominator_cv,
            shared_V_and_standard_can_covariance_included=True,
            moments='First-order in the normalized V response; exact standard ON/OFF and can Dirichlet moments.',
            sampling='Positive nonlinear standard/V ratio; shared standard/can posterior and all earlier counts retained.',
            physically_validated=False)
    else:
        campaign = ReferenceCountCampaign(posterior, source_map, background_map, normalization, seed, **options)
    details = dict(model='Current-cycle empirical V/blank effective response',
        calibration_status='current_cycle_conditional_elastic_reference',
        cycle=bundle['cycle'], environment='PAC', container=container, wavelength_a=bundle.get('wavelength_a',1.5),
        bundle_sha256=BUNDLES[config['id']][1], response_version=bundle['schema'],
        measured_reference_runs=bundle['reference_runs'], reference_proton_energy_gev=1.3,
        y_unit='POWGEN reduced intensity (ScaleData=100)',
        raw_count_unit='unweighted accepted neutron events per focused bin',
        geometry=q, sample_assumptions=sample, vanadium_assumptions=reference,
        reference_inference=posterior.metadata, measured_reference_cells=int(supported.sum()),
        container_reference=bundle['can_reference'] if container=='PAC06' else None,
        container_reference_likelihood='Independent Poisson empty-can counts; flat nonnegative rate prior; shared Gamma posterior across looks.' if container=='PAC06' else None,
        reduction=('Normalize sample counts by charge; subtract captured FFT-smoothed PAC06; divide by captured native processed V; multiply by 100.' if container=='PAC06' else
            'Normalize sample counts by charge; subtract conditional instrument-blank mean; divide by captured native processed V; multiply by 100.'),
        reduction_matches_facility_pipeline=False,
        terminal_reduction_matches_facility_convention=container=='PAC06',
        terminal_reference=bundle['terminal_reduction'],
        reduction_grid='Project processed reference bin rates onto requested TOF bin edges before subtraction/division; piecewise constant native-bin rate densities.',
        synthetic_raw_events_reduced_with_mantid=False,
        error_columns='Conditional predictive uncertainty includes shared finite references; these are not the facility conventional E columns with V errors reset to zero.',
        reference_uncertainty='Conditional predictive ON/OFF posterior, shared across bins and cumulative looks; exact sparse covariance operators exported.',
        intensity_calibration=intensity_calibration,standard_normalization=standard,
        intensity_normalization=('Si63584 integrated-count normalization under declared geometry/background assumptions; no NAC or held target fitted.' if standard else
            'GSAS integrated powder-line cross-section identity; no target scale fitted'),
        standard_intensity_transfer_calibrated=standard is not None, experimental_accuracy_certified=False,
        assumptions=['Sample mass, homogeneous cylinder, packing and illumination are supplied assumptions.',
            'Measured V includes the effective flux, acceptance and efficiency of this exact configuration.',
            ('Matched physical and elastic-equivalent V predictions modify the measured response; the planner rod/sample geometry remains explicit.' if reference_law is not None else
                'Vanadium is represented by an elastic isotropic incoherent law and conventional rod geometry.'),
            'Effective reference density is constant within the recorded 20-bin aggregation cells.',
            'Single scattering with transverse-cylinder attenuation; thermal incoherent sample law.',
            'Stationary Poisson sample increments conditional on one shared positive reference-rate draw.'],
        warnings=['Conditional current-cycle prediction; independent absolute and noise qualification are pending.',
            ('Reference-law correction has only the scattering/transport scope of its supplied physical model; remaining discrepancy is transferred multiplicatively.' if reference_law is not None else
                'Vanadium inelastic/coherent contributions and multiple scattering are absorbed approximately into the effective response.'),
            ('Measured PAC06 empty-can background replaces the instrument blank. Sample shielding of can/environment scattering and wall attenuation are not yet applied.' if container=='PAC06' else 'Background is the measured instrument blank; no sample-can scattering is included.'),
            'Native terminal reference processing is replay-verified; raw-count physics, requested-grid approximation and independent predictive coverage remain unqualified.'])
    if standard:
        details['reference_uncertainty']='Joint standard ON/OFF and can fractions, with shared V shape; positive nonlinear draws and first-order ratio moments. No independent fitted-scale error is added.'
        details['container_reference_likelihood']='Selected can cells share a joint ON/OFF standard/background rate and Dirichlet proportions; other cells retain independent Gamma posteriors.' if container=='PAC06' else None
        details['warnings'].append('Experimental Si integrated-count normalization assumes the measured empty-can background transfers. It does not remove the retained peak/continuum or NAC discrepancies; physical transfer is not validated.')
    if reference_law is not None:
        details['calibration_status'] = 'conditional_physical_reference_law_transfer'
        details['reference_law_transfer'] = dict(provenance=reference_law.provenance,
            method='Ratio of matched elastic-equivalent and physical V model bin rates; applied to source transfer columns before propagation of measured ON/OFF uncertainty.',
            factor_range_in_required_cells=[float(correction[required].min()), float(correction[required].max())],
            physical_model_uncertainty_included=False, experimental_accuracy_certified=False)
        details['assumptions'].append('Reference model discrepancy transfers multiplicatively within each measured angular/d cell; source, efficiency and geometry in the two reference predictions must match.')
        details['warnings'].append('Experimental reference-law transfer retains conditional measured-count covariance; source/efficiency/material/geometry model uncertainty and sample inelastic continuum remain unqualified.')
    if sample_diffuse is not None:
        details['sample_diffuse_transfer'] = dict(provenance=sample_diffuse.provenance,
            sample_fingerprint=sample_diffuse.sample_fingerprint,
            thermal_incoherent_source_replaced=True, coherent_bragg_source_duplicated=False,
            shared_reference_cross_covariance_retained=True, material_dynamics_uncertainty_included=False,
            experimental_accuracy_certified=False)
        details['assumptions'] = [a for a in details['assumptions'] if a != 'Single scattering with transverse-cylinder attenuation; thermal incoherent sample law.']
        details['assumptions'].append('Explicit diffuse/inelastic material response replaces thermal incoherence; its dynamics, transport and temperature assumptions are recorded separately.')
    if sample_elastic is not None:
        details['sample_elastic_transfer'] = dict(provenance=sample_elastic.provenance,
            sample_fingerprint=sample_elastic.sample_fingerprint,normalizer_b=normalizer_b,
            whole_specimen_cell_count_and_path_weights_applied_once=True,
            transverse_cylinder_ratio_replaced=True,shared_reference_cross_covariance_retained=True,
            angular_and_wavelength_variation_within_reference_cells_resolved=False,
            physical_model_uncertainty_included=False,experimental_accuracy_certified=False)
        details['assumptions'] = [a for a in details['assumptions'] if a != 'Single scattering with transverse-cylinder attenuation; thermal incoherent sample law.']
        details['assumptions'].append('Matched isotropic elastic unit-law response supplies sample geometry and amount; the crystallographic profile is applied within measured reference cells. Source/path/profile variation within cells remains approximate.')
    return campaign, {**components, 'blank': campaign.background, 'normalization': normalization,
                      'background_subtraction': campaign.subtraction,
                      'tof_edges_us': xedges, 'd_edges_a': dedges}, details
