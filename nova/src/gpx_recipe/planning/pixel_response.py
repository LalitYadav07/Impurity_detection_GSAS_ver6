"""Single-collision response retaining the calibration of each detector pixel.

Only the scattering law is averaged over polar rings. A joint energy/angle
draw selects a pixel within its ring in proportion to solid angle. Arrival
times, diffraction conversion and reduction cuts then use that pixel's actual
angle, flight path and DIFC, never their ring averages. Angular refinement is
still required: this is not integration over the finite surface of a pixel.

The current pixel contract is DIFC-only, matching the checked POWGEN archive.
It does not invent source delays, DIFA/TZERO, transport paths or count noise.
"""
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from .elastic_quadrature import ElasticCellQuadrature
from .energy_response import (EnergyResponse, ReductionWindow, TOF_US_PER_M_A,
                              _bin, _edges, _positive, apparent_d)


VERSION = 'powgen-individual-pixel-energy-response/1'


@dataclass(frozen=True)
class DetectorPixels:
    detector_ids: np.ndarray
    cos_two_theta: np.ndarray
    solid_angle_sr: np.ndarray
    l2_m: np.ndarray
    difc_us_per_a: np.ndarray
    observed_group_index: np.ndarray

    def __post_init__(self):
        ids = np.asarray(self.detector_ids)
        groups = np.asarray(self.observed_group_index)
        if (ids.ndim != 1 or not 1 <= len(ids) <= 200000 or ids.dtype.kind not in 'iu'
                or np.any(ids < 0) or len(np.unique(ids)) != len(ids)
                or groups.shape != ids.shape or groups.dtype.kind not in 'iu'
                or np.any(groups < 0) or np.any(groups >= 1000000)):
            raise ValueError('Unique integer detector IDs and bounded observed groups are required')
        for name in ('cos_two_theta', 'solid_angle_sr', 'l2_m', 'difc_us_per_a'):
            a = np.array(getattr(self, name), dtype=float, copy=True)
            if a.shape != ids.shape or not np.isfinite(a).all():
                raise ValueError('Pixel geometry/calibration must be finite matching vectors')
            if name == 'cos_two_theta':
                if np.any(a < -1) or np.any(a >= 1):
                    raise ValueError('Pixel angles must be physical and nonforward')
            elif np.any(a <= 0):
                raise ValueError('Active pixels need positive solid angle, path and DIFC')
            a.setflags(write=False)
            object.__setattr__(self, name, a)
        for name, a in (('detector_ids', ids), ('observed_group_index', groups)):
            a = np.array(a, dtype=np.int64, copy=True)
            a.setflags(write=False)
            object.__setattr__(self, name, a)

    def difc_ratio(self, l1_m):
        l1 = _positive(l1_m, 'Incident flight path')
        return self.difc_us_per_a / (TOF_US_PER_M_A * (l1 + self.l2_m)
                                    * np.sqrt(2 * (1 - self.cos_two_theta)))


def pixel_elastic_quadrature(cos_two_theta, difc_us_per_a, d_edges_a,
                             wavelength_nodes_a, *, l1_m, l2_m, window=None,
                             clock_model='calibrated_elastic_mean',
                             wavelength_breaks_a=(), energy_order=4):
    """Integrate d(lambda) at one pixel, splitting at every native event cut.

    Weights exclude solid angle, scattering cross section and specimen amount.
    No angular averaging or renormalization of accepted intervals is performed.
    """
    cosine = float(cos_two_theta)
    if not np.isfinite(cosine) or not -1 <= cosine < 1:
        raise ValueError('A finite nonforward pixel angle is required')
    difc = _positive(difc_us_per_a, 'Pixel DIFC')
    l1, l2 = _positive(l1_m, 'Incident path'), _positive(l2_m, 'Final path')
    edges, nodes = _edges(d_edges_a, 'Observed d spacing'), _edges(wavelength_nodes_a, 'Source nodes')
    if len(edges) > 4097 or len(nodes) > 4097:
        raise ValueError('Pixel quadrature exceeds the edge/node budget')
    breaks = np.asarray(wavelength_breaks_a, float)
    if (breaks.ndim != 1 or len(breaks) > 100001 or not np.isfinite(breaks).all()
            or np.any(breaks <= 0)):
        raise ValueError('Wavelength breaks must be positive finite values')
    if type(energy_order) is not int or not 1 <= energy_order <= 32:
        raise ValueError('Quadrature order must be an integer from 1 to 32')
    if clock_model not in {'physical', 'calibrated_elastic_mean'}:
        raise ValueError('An explicit supported elastic clock is required')
    window = ReductionWindow() if window is None else window
    if not isinstance(window, ReductionWindow):
        raise ValueError('An explicit reduction window is required')
    angle_factor = np.sqrt(2 * (1 - cosine))
    time_scale = (TOF_US_PER_M_A * (l1 + l2) if clock_model == 'physical'
                  else difc / angle_factor)
    wave_per_d = difc / time_scale
    cuts = [nodes, breaks, edges * wave_per_d]
    if window.tof_range_us is not None:
        cuts.append(np.asarray(window.tof_range_us) / time_scale)
    if window.apparent_wavelength_range_a is not None:
        cuts.append(np.asarray(window.apparent_wavelength_range_a) * wave_per_d / angle_factor)
    if window.prompt_period_us is not None and window.prompt_width_us > 0:
        last = int(np.ceil(nodes[-1] * time_scale / window.prompt_period_us))
        if last > 10000:
            raise ValueError('Prompt integration exceeds the interval budget')
        starts = np.arange(last + 1) * window.prompt_period_us
        cuts.extend((starts / time_scale, (starts + window.prompt_width_us) / time_scale))
    cuts = np.unique(np.clip(np.concatenate(cuts), nodes[0], nodes[-1]))
    low, high = cuts[:-1], cuts[1:]
    mid = (low + high) / 2
    _, valid_d = _bin(edges, mid / wave_per_d)
    good = valid_d & window.accept(mid * time_scale, mid / wave_per_d, cosine)
    low, high = low[good], high[good]
    x, w = leggauss(energy_order)
    wave = ((low + high)[:, None] / 2 + (high - low)[:, None] * x / 2).ravel()
    weight = ((high - low)[:, None] * w / 2).ravel()
    index, valid = _bin(edges, wave / wave_per_d)
    if not valid.all() or not window.accept(wave * time_scale, wave / wave_per_d, cosine).all():
        raise ValueError('Pixel quadrature node crossed a declared acceptance boundary')
    arrays = (wave, np.full(len(wave), cosine), index, weight)
    for a in arrays:
        a.setflags(write=False)
    return ElasticCellQuadrature(*arrays)


def build_pixel_energy_response(incident_edges_a, outgoing_edges_a, d_edges_a,
                                angular_edges_deg, *, pixels, scattering_batches,
                                scattering_law_id, samples_per_incident_bin,
                                path_weight, l1_m=60., reduction_window=None,
                                clock_model='physical', incident_nodes_a=None,
                                event_acceptance=None, pulse_delay=None, seed=0,
                                max_events=20000000, max_bytes=256 * 1024**2):
    """Positive sparse response using original pixels and joint scattering draws.

    The scattering provider has the same contract as build_energy_response.
    path_weight, event_acceptance and pulse_delay receive ORIGINAL pixel indices.
    Independent pixel sampling estimates the solid-angle-weighted discrete
    pixel mixture within each polar ring. It does not preserve an energy-angle
    variation *within* that ring: convergence under angular refinement is needed.
    Geometry and calibration remain correlated through the selected pixel.
    """
    if not isinstance(pixels, DetectorPixels):
        raise ValueError('Explicit original detector pixels are required')
    incident = _edges(incident_edges_a, 'Incident wavelength')
    outgoing = _edges(outgoing_edges_a, 'Outgoing wavelength')
    edges = _edges(d_edges_a, 'Observed d spacing')
    polar_edges = _edges(angular_edges_deg, 'Polar angle', angle=True)
    nodes = None if incident_nodes_a is None else _edges(incident_nodes_a, 'Source nodes')
    if nodes is not None and (len(nodes) > 4097 or nodes[0] != incident[0] or nodes[-1] != incident[-1]):
        raise ValueError('Density nodes must span the exact integration domain within budget')
    if not isinstance(scattering_law_id, str) or not scattering_law_id.strip():
        raise ValueError('A traceable scattering-law identity is required')
    if reduction_window is not None and not isinstance(reduction_window, ReductionWindow):
        raise ValueError('An explicit reduction window is required')
    if clock_model not in {'physical', 'calibrated_elastic_mean'}:
        raise ValueError('An explicit supported clock is required')
    ratio = pixels.difc_ratio(l1_m)
    ni, no, nd = len(incident)-1, len(outgoing)-1, len(edges)-1
    ns, ng = ni if nodes is None else len(nodes), int(pixels.observed_group_index.max()) + 1
    work_bytes = samples_per_incident_bin * 512 + len(pixels.detector_ids) * 128
    if (type(samples_per_incident_bin) is not int or samples_per_incident_bin < 1
            or type(seed) is not int or not 0 <= seed < 2**32 or type(max_events) is not int
            or ni * samples_per_incident_bin > max_events or ng * nd > 1000000
            or ns * no * ng * nd > np.iinfo(np.int64).max
            or type(max_bytes) is not int or work_bytes > max_bytes):
        raise ValueError('Pixel response exceeds its event, dimension or memory budget')
    pixel_polar, covered = _bin(polar_edges, np.rad2deg(np.arccos(pixels.cos_two_theta)))
    if not covered.all():
        raise ValueError('Polar integration domain must cover every active pixel')
    ring = 2 * np.pi * (np.cos(np.deg2rad(polar_edges[:-1])) - np.cos(np.deg2rad(polar_edges[1:])))
    order = np.argsort(pixel_polar, kind='stable')
    cumulative = np.cumsum(pixels.solid_angle_sr[order])
    starts = np.searchsorted(pixel_polar[order], np.arange(len(ring)), side='left')
    ends = np.searchsorted(pixel_polar[order], np.arange(len(ring)), side='right')
    prefix = np.r_[0., cumulative]
    offsets, omega = prefix[starts], prefix[ends] - prefix[starts]
    if np.any(omega > ring * (1 + 1e-12)):
        raise ValueError('Pixel solid angles exceed a complete polar ring')
    keys, coefficients, omitted_rows, omitted_cols, omitted_values, records = [], [], [], [], [], []

    def source_terms(column, wave):
        if nodes is None:
            return ((column, 1.),)
        left = np.minimum(np.searchsorted(nodes, wave, side='right')-1, len(nodes)-2)
        right = (wave - nodes[left]) / (nodes[left+1] - nodes[left])
        return ((left, 1-right), (left+1, right))

    for column in range(ni):
        child = int(np.random.SeedSequence([seed, column]).generate_state(1)[0])
        batch = scattering_batches(float(incident[column]), float(incident[column+1]), samples_per_incident_bin, child)
        if len(batch) != 4:
            raise ValueError('Provider must return joint incident/final wavelength, angle and weight')
        lin, lout, cosine, weight = [np.asarray(v, float) for v in batch]
        if any(v.shape != (samples_per_incident_bin,) or not np.isfinite(v).all() for v in (lin,lout,cosine,weight)):
            raise ValueError('Scattering draws must be finite matching vectors')
        if (np.any(lin < incident[column]) or np.any(lin > incident[column+1]) or np.any(lout <= 0)
                or np.any(cosine < -1) or np.any(cosine > 1) or np.any(weight < 0)):
            raise ValueError('Scattering draw is outside its physical domain')
        polar, good = _bin(polar_edges, np.rad2deg(np.arccos(cosine)))
        selected = np.flatnonzero(good & (weight > 0))
        selected = selected[omega[polar[selected]] > 0]
        rings = polar[selected]
        rng = np.random.default_rng(np.random.SeedSequence([seed, column, 901]))
        draw = offsets[rings] + rng.random(len(selected)) * omega[rings]
        positions = np.searchsorted(cumulative, draw, side='right')
        # Roundoff at a cumulative endpoint must stay in the selected ring.
        positions = np.maximum(starts[rings], np.minimum(positions, ends[rings]-1))
        pi = order[positions]
        if len(selected):
            delay = 0. if pulse_delay is None else pulse_delay(pi, lin[selected], lout[selected])
            d, time = apparent_d(lin[selected], lout[selected], pixels.cos_two_theta[pi],
                l1_m=l1_m, l2_m=pixels.l2_m[pi], difc_ratio=ratio[pi], delay_us=delay, clock_model=clock_model)
            di, good_d = _bin(edges, d)
            if reduction_window is not None:
                good_d &= reduction_window.accept(time, d, pixels.cos_two_theta[pi])
            selected, pi, rings, di, time = (a[good_d] for a in (selected, pi, rings, di, time))
            acceptance = (np.ones(len(selected)) if event_acceptance is None else
                np.broadcast_to(np.asarray(event_acceptance(pi, lin[selected], lout[selected], time), float), (len(selected),)))
            if not np.isfinite(acceptance).all() or np.any(acceptance < 0) or np.any(acceptance > 1):
                raise ValueError('Physical event acceptance must lie in [0,1]')
            values = weight[selected] * omega[rings] / ring[rings] * acceptance
            rows = pixels.observed_group_index[pi] * nd + di
            oi, outgoing_good = _bin(outgoing, lout[selected])
            if np.any(~outgoing_good):
                for col, basis in source_terms(column, lin[selected[~outgoing_good]]):
                    omitted_rows.append(rows[~outgoing_good])
                    omitted_cols.append(np.broadcast_to(col, ((~outgoing_good).sum(),)))
                    omitted_values.append(values[~outgoing_good] * basis)
            selected, pi, rows, values, oi = (a[outgoing_good] for a in (selected, pi, rows, values, oi))
            if len(selected):
                path = np.broadcast_to(np.asarray(path_weight(pi, lin[selected], lout[selected]), float), (len(selected),))
                if not np.isfinite(path).all() or np.any(path < 0) or np.any(path > 1):
                    raise ValueError('Whole-specimen path weights must lie in [0,1]')
                for col, basis in source_terms(column, lin[selected]):
                    key = (rows * ns + col) * no + oi
                    unique, inverse = np.unique(key, return_inverse=True)
                    keys.append(unique)
                    coefficients.append(np.bincount(inverse, weights=values * path * basis))
        records.append(dict(incident_bin=column, input_integrated_cross_section_b_a=float(weight.sum()),
                            sampling_points=samples_per_incident_bin))
        retained = sum(map(len, keys)) + sum(map(len, omitted_rows))
        if retained * 64 + work_bytes > max_bytes:
            raise ValueError('Sparse pixel response exceeds its working memory budget')
    key = np.concatenate(keys) if keys else np.empty(0, dtype=np.int64)
    values = np.concatenate(coefficients) if coefficients else np.empty(0)
    unique, inverse = np.unique(key, return_inverse=True)
    values = np.bincount(inverse, weights=values)
    positive = values > 0
    unique, values = unique[positive], values[positive]
    oi, ci, ri = unique % no, (unique // no) % ns, unique // (no * ns)
    omitted = sparse.csr_matrix((ng*nd, ns))
    if omitted_rows:
        omitted = sparse.coo_matrix((np.concatenate(omitted_values),
            (np.concatenate(omitted_rows), np.concatenate(omitted_cols))), shape=omitted.shape).tocsr()
        omitted.eliminate_zeros()
    for a in (incident, outgoing, edges, oi, ci, ri, values):
        a.setflags(write=False)
    metadata = dict(version=VERSION, scattering_law_id=scattering_law_id, seed=seed,
        integration_samples=ni*samples_per_incident_bin, incident_bin_records=records,
        coefficient_unit='barn*angstrom per scattering cell', detector_pixels=len(pixels.detector_ids),
        pixel_calibration_averaged=False, pixel_solid_angle_selection=True,
        scattering_angular_approximation='Joint law averaged within each polar ring; discrete original pixel mixture.',
        polar_edges_deg=polar_edges.tolist(), finite_pixel_face_integration=False,
        original_pixel_center_used_for_clock_and_cuts=True, clock_model=clock_model,
        outgoing_energy_not_renamed_incident_energy=True, outgoing_domain_loss_renormalized=False,
        whole_powder_path_includes_illumination_once=True, numerical_samples_are_physical_count_noise=False,
        observed_groups=ng, incident_density_basis='constant_bins' if nodes is None else 'linear_nodes',
        incident_parameter_count=ns, azimuthally_isotropic_scattering_assumed=True,
        reduction_window=None if reduction_window is None else dict(
            tof_range_us=reduction_window.tof_range_us,
            apparent_wavelength_range_a=reduction_window.apparent_wavelength_range_a,
            prompt_period_us=reduction_window.prompt_period_us, prompt_width_us=reduction_window.prompt_width_us),
        multiple_scattering_included=False, source_detector_identifiability_established=False,
        physical_Poisson_independence_or_stationarity_verified=False, experimental_accuracy_certified=False)
    return EnergyResponse(incident, outgoing, edges, ng, ri, ci, oi, values, omitted, metadata, nodes)
