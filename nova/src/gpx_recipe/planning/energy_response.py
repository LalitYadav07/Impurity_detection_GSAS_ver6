"""Sparse single-collision response with distinct incident/final wavelengths.

A scattering-law integrator supplies positive, per-cell cross-section weights
integrated over each incident wavelength bin (barn*angstrom). Its final energy
and angle must be a JOINT law, not independent marginal draws. Geometry, path
weights, source pulse/filter and detector efficiency remain explicit. Numerical
integration samples are not physical neutron-count noise.

The response preserves outgoing-wavelength indices so efficiency is evaluated
at the detected energy while incident fluence is evaluated at the incident
energy. A truncated outgoing domain is never renormalized: a separate unattenuated
upper response bounds the omitted contribution under efficiency <= 1.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy import sparse

VERSION = 'powgen-energy-changing-single-collision-response/3'
# CODATA 2022 neutron mass / exact SI Planck constant, with m,A -> microseconds.
TOF_US_PER_M_A = 1.67492750056e-27 / 6.62607015e-34 * 1e-4


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(name + ' must be a positive finite scalar')
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be a positive finite scalar')
    return value


def _edges(values, name, *, angle=False):
    result = np.asarray(values, float)
    if (result.ndim != 1 or not 2 <= len(result) <= 100001 or not np.isfinite(result).all()
            or np.any(np.diff(result) <= 0) or result[0] < (0 if angle else np.finfo(float).tiny)
            or (angle and result[-1] > 180)):
        raise ValueError(name + ' needs finite increasing physical bin edges')
    return result.copy()


def _bin(edges, values):
    index = np.searchsorted(edges, values, side='right') - 1
    index = np.where(values == edges[-1], len(edges) - 2, index)
    return index, (index >= 0) & (index < len(edges) - 1)


def apparent_d(wavelength_in_a, wavelength_out_a, cos_two_theta, *, l1_m, l2_m,
               difc_ratio=1., difa_us_per_a2=0., tzero_us=0., delay_us=0.,
               tof_us_per_m_a=TOF_US_PER_M_A, clock_model='physical'):
    """Convert a declared raw event clock through detector diffraction calibration.

    The physical clock is k*(L1*lambda_in+L2*lambda_out)+delay. Detector
    DIFA/DIFC/TZERO then map that raw time to d; they do not generate its arrival
    time. Source emission/path delays require an explicit input.

    The alternative calibrated_elastic_mean clock preserves the earlier
    phenomenological assumption: t_elastic=DIFC*d_in+DIFA*d_in^2+TZERO, then
    add k*L2*(lambda_out-lambda_in)+delay. Its equal-energy limit returns d_in,
    even when calibration differs from nominal physical geometry. That is an
    explicit mean-clock hypothesis, not an independently measured source delay.
    GSAS profile DIFB is not used by either event-coordinate calculation.
    """
    lin, lout, cosine, l2, ratio, a, zero, delay = np.broadcast_arrays(*map(lambda x: np.asarray(x, float),
        (wavelength_in_a, wavelength_out_a, cos_two_theta, l2_m, difc_ratio, difa_us_per_a2, tzero_us, delay_us)))
    l1, k = _positive(l1_m, 'Incident flight path'), _positive(tof_us_per_m_a, 'TOF conversion constant')
    if type(clock_model) is not str or clock_model not in ('physical', 'calibrated_elastic_mean'):
        raise ValueError('An explicit physical or calibrated_elastic_mean clock is required')
    if (not all(np.isfinite(x).all() for x in (lin,lout,cosine,l2,ratio,a,zero,delay))
            or np.any(lin <= 0) or np.any(lout <= 0) or np.any(l2 <= 0) or np.any(ratio <= 0)
            or np.any(cosine < -1) or np.any(cosine >= 1)):
        raise ValueError('Finite positive wavelengths/paths and physical nonforward angles are required')
    sine = np.sqrt((1 - cosine) / 2)
    d0 = lin / (2 * sine)
    b = 2 * k * (l1 + l2) * sine * ratio
    if clock_model == 'physical':
        time = k * (l1 * lin + l2 * lout) + delay
    else:
        if np.any(b + 2 * a * d0 <= 0):
            raise ValueError('Incident wavelength is outside the monotonic calibration branch')
        time = b * d0 + a * d0**2 + zero + k * l2 * (lout - lin) + delay
    offset = time - zero
    discriminant = b**2 + 4 * a * offset
    if np.any(time <= 0) or np.any(discriminant <= 0) or np.any(offset <= 0):
        raise ValueError('Energy-changing event is outside the positive monotonic calibration branch')
    d = 2 * offset / (b + np.sqrt(discriminant))
    if not np.isfinite(d).all() or np.any(d <= 0):
        raise ValueError('Invalid apparent diffraction coordinate')
    return d, time


@dataclass(frozen=True)
class ReductionWindow:
    """Event cuts for the pinned AlignAndFocusPowder coordinate sequence.

    TOF and prompt cuts use the raw detector clock. After detector calibration
    is converted to d and cleared, the wavelength cut uses 2*d*sin(theta) with
    the physical detector angle. It is neither incident nor final wavelength.
    Crop bounds include both endpoints; prompt intervals veto both endpoints.
    Output histogram d bounds belong to build_energy_response, separately.
    """
    tof_range_us: tuple = None
    apparent_wavelength_range_a: tuple = None
    prompt_period_us: float = None
    prompt_width_us: float = 0.

    def __post_init__(self):
        for name in ('tof_range_us', 'apparent_wavelength_range_a'):
            values = getattr(self, name)
            if values is None:
                continue
            values = np.asarray(values, float)
            if (values.shape != (2,) or not np.isfinite(values).all()
                    or values[0] < 0 or values[1] <= values[0]
                    or (name == 'apparent_wavelength_range_a' and values[0] == 0)):
                raise ValueError('Reduction ranges need finite increasing physical bounds')
            object.__setattr__(self, name, tuple(map(float, values)))
        width = self.prompt_width_us
        if (isinstance(width, (bool, np.bool_)) or not np.isscalar(width)
                or not math.isfinite(float(width)) or width < 0):
            raise ValueError('Prompt width must be finite and nonnegative')
        if self.prompt_period_us is None:
            if width != 0:
                raise ValueError('A nonzero prompt width needs an explicit period')
        else:
            period = _positive(self.prompt_period_us, 'Prompt period')
            if width >= period:
                raise ValueError('Prompt width must be smaller than its period')
            object.__setattr__(self, 'prompt_period_us', period)
        object.__setattr__(self, 'prompt_width_us', float(width))

    def accept(self, physical_time_us, apparent_d_a, cos_two_theta):
        time, d, cosine = np.broadcast_arrays(*map(lambda v: np.asarray(v, float),
            (physical_time_us, apparent_d_a, cos_two_theta)))
        if (not all(np.isfinite(v).all() for v in (time, d, cosine)) or np.any(d <= 0)
                or np.any(cosine < -1) or np.any(cosine >= 1)):
            raise ValueError('Reduction cuts need finite clocks, positive d and physical angles')
        accepted = np.ones(time.shape, dtype=bool)
        for values, bounds in ((time, self.tof_range_us),
                (2 * d * np.sqrt((1 - cosine) / 2), self.apparent_wavelength_range_a)):
            if bounds is not None:
                accepted &= (values >= bounds[0]) & (values <= bounds[1])
        if self.prompt_period_us is not None and self.prompt_width_us > 0:
            from .prompt_gate import periodic_prompt_acceptance
            accepted &= periodic_prompt_acceptance(time, self.prompt_period_us, self.prompt_width_us,
                                                   upper_inclusive=True)
        return accepted


@dataclass(frozen=True)
class EnergyResponse:
    incident_edges_a: np.ndarray
    outgoing_edges_a: np.ndarray
    d_edges_a: np.ndarray
    observed_groups: int
    observation_index: np.ndarray
    incident_index: np.ndarray
    outgoing_index: np.ndarray
    coefficient_b_a: np.ndarray
    outgoing_omitted_upper_b_a: sparse.csr_matrix
    metadata: dict
    incident_nodes_a: np.ndarray | None = None

    def __post_init__(self):
        if self.incident_nodes_a is not None:
            nodes = _edges(self.incident_nodes_a, 'Incident density nodes')
            if (len(nodes) > 4097 or nodes[0] != self.incident_edges_a[0]
                    or nodes[-1] != self.incident_edges_a[-1]):
                raise ValueError('Density nodes must exactly span the integration domain within the node budget')
            nodes.setflags(write=False)
            object.__setattr__(self, 'incident_nodes_a', nodes)

    @property
    def shape(self):
        columns = len(self.incident_edges_a) - 1 if self.incident_nodes_a is None else len(self.incident_nodes_a)
        return (self.observed_groups * (len(self.d_edges_a) - 1), columns)

    @property
    def incident_parameter_wavelength_a(self):
        """Node coordinates, or bin centers for the constant-density convention."""
        return ((self.incident_edges_a[:-1] + self.incident_edges_a[1:]) / 2
                if self.incident_nodes_a is None else self.incident_nodes_a)

    def incident_density_at(self, wavelength_a, incident_density):
        """Evaluate the declared nonnegative source basis without extrapolation."""
        wave, density = np.asarray(wavelength_a, float), np.asarray(incident_density, float)
        if (density.shape != (self.shape[1],) or not np.isfinite(density).all() or np.any(density < 0)
                or not np.isfinite(wave).all() or np.any(wave < self.incident_edges_a[0])
                or np.any(wave > self.incident_edges_a[-1])):
            raise ValueError('Source queries need matching nonnegative density and a covered finite wavelength')
        if self.incident_nodes_a is not None:
            return np.interp(wave, self.incident_nodes_a, density)
        index, _ = _bin(self.incident_edges_a, wave)
        return density[index]

    def _efficiency(self, values, angular_efficiency):
        efficiency = np.asarray(values, float)
        angular = np.ones(self.observed_groups) if angular_efficiency is None else np.asarray(angular_efficiency, float)
        if efficiency.shape not in ((len(self.outgoing_edges_a) - 1,),
                (self.observed_groups, len(self.outgoing_edges_a) - 1)) or angular.shape != (self.observed_groups,):
            raise ValueError('Outgoing and observed-angle efficiency arrays must match the response')
        if any(not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1) for x in (efficiency, angular)):
            raise ValueError('Absolute detection efficiencies must lie in [0,1]')
        return efficiency, angular

    def source_matrix(self, outgoing_efficiency, *, number_cells=1., angular_efficiency=None):
        """cm²*A per whole specimen, mapping incident fluence density to rates.

        Incident density is neutrons/(cm²*exposure*A). No source/reference scale
        is fitted here. Cells are for the WHOLE specimen; path weights already
        contain the illuminated fraction once.
        """
        efficiency, angular = self._efficiency(outgoing_efficiency, angular_efficiency)
        scale = _positive(number_cells, 'Whole specimen cell count') * 1e-24
        angles = self.observation_index // (len(self.d_edges_a) - 1)
        detected = efficiency[self.outgoing_index] if efficiency.ndim == 1 else efficiency[angles, self.outgoing_index]
        weights = scale * self.coefficient_b_a * detected * angular[angles]
        if not np.isfinite(weights).all():
            raise ValueError('Response scaling exceeds its finite numerical domain')
        result = sparse.coo_matrix((weights, (self.observation_index, self.incident_index)), shape=self.shape).tocsr()
        result.eliminate_zeros()
        return result

    def efficiency_matrix(self, incident_density, *, number_cells=1., angular_efficiency=None, by_angle=False):
        density = np.asarray(incident_density, float)
        if density.shape != (self.shape[1],) or not np.isfinite(density).all() or np.any(density < 0):
            raise ValueError('Incident fluence density must be finite, nonnegative and match incident bins')
        _, angular = self._efficiency(np.ones(len(self.outgoing_edges_a) - 1), angular_efficiency)
        angles = self.observation_index // (len(self.d_edges_a) - 1)
        weights = (_positive(number_cells, 'Whole specimen cell count') * 1e-24
                   * self.coefficient_b_a * density[self.incident_index] * angular[angles])
        if not np.isfinite(weights).all():
            raise ValueError('Response scaling exceeds its finite numerical domain')
        columns = angles * (len(self.outgoing_edges_a) - 1) + self.outgoing_index if by_angle else self.outgoing_index
        return sparse.coo_matrix((weights, (self.observation_index, columns)),
            shape=(self.shape[0], (self.observed_groups if by_angle else 1) * (len(self.outgoing_edges_a) - 1))).tocsr()

    def predict(self, incident_density, outgoing_efficiency, *, number_cells=1., angular_efficiency=None):
        """Covered rate and a separate upper rate for unmodeled outgoing energies.

        The upper term assumes unit path/detection weight and is never folded
        into the mean. It concerns outgoing-domain truncation inside the declared
        observation domain, not intensity cropped outside that observation domain.
        """
        density = np.asarray(incident_density, float)
        if density.shape != (self.shape[1],) or not np.isfinite(density).all() or np.any(density < 0):
            raise ValueError('Incident fluence density must be finite, nonnegative and match incident bins')
        matrix = self.source_matrix(outgoing_efficiency, number_cells=number_cells, angular_efficiency=angular_efficiency)
        _, angular = self._efficiency(outgoing_efficiency, angular_efficiency)
        upper = (_positive(number_cells, 'Whole specimen cell count') * 1e-24
                 * np.asarray(self.outgoing_omitted_upper_b_a @ density)
                 * np.repeat(angular, len(self.d_edges_a) - 1))
        return np.asarray(matrix @ density).reshape(self.observed_groups, -1), upper.reshape(self.observed_groups, -1)

    def log_jacobian(self, incident_density, outgoing_efficiency, *, number_cells=1., angular_efficiency=None):
        """Derivatives of rates w.r.t. log incident density and log efficiency.

        Their global opposite rescaling is a gauge: J*[1_source,-1_eff]=0.
        A gauge convention is not a measured absolute flux calibration.
        """
        density = np.asarray(incident_density, float)
        efficiency, _ = self._efficiency(outgoing_efficiency, angular_efficiency)
        a = self.source_matrix(efficiency, number_cells=number_cells, angular_efficiency=angular_efficiency)
        b = self.efficiency_matrix(density, number_cells=number_cells, angular_efficiency=angular_efficiency,
            by_angle=efficiency.ndim == 2)
        return sparse.hstack((a.multiply(density), b.multiply(efficiency.ravel())), format='csr')


def build_energy_response(incident_edges_a, outgoing_edges_a, d_edges_a, angular_edges_deg, *,
                          solid_angle_sr, l2_m, observed_group_index, scattering_batches,
                          scattering_law_id, samples_per_incident_bin, path_weight,
                          l1_m=60., difc_ratio=1., difa_us_per_a2=0., tzero_us=0.,
                          event_acceptance=None, pulse_delay=None, reduction_window=None,
                          clock_model='physical', incident_nodes_a=None, seed=0,
                          max_events=20000000, max_bytes=256 * 1024**2):
    """Accumulate a positive energy/angle/TOF response from joint scattering draws.

    scattering_batches(lo,hi,n,seed) returns four matching n-vectors:
    lambda_in, lambda_out, cos(2theta), weight_b_a. Weights integrate the
    per-cell material cross section over the incident bin, including 1/n.
    Angular cells have shape (polar bins, azimuth quadrature cells); their
    solid angles are fractions of the complete 2*pi*dcos(theta) polar rings.
    Acceptance within a cell is approximated uniform in cos(theta); angular
    refinement is a separate numerical requirement. Several azimuth cells may
    map to the same observed group. Their paths/calibration need not coincide.

    path_weight(cell_index,lambda_in,lambda_out) is a whole-powder [0,1] weight.
    event_acceptance(cell_index,lambda_in,lambda_out,physical_time) defaults to
    one. pulse_delay has the same first three arguments and returns microseconds.
    An explicit ReductionWindow applies matched cuts using the apparent event
    coordinate before any additional event_acceptance. It is not inferred from
    the incident-wavelength integration domain.
    clock_model declares physical arrival times or the calibrated_elastic_mean
    hypothesis; metadata retains that choice. Calibration coefficients alone
    do not establish a physical source-emission delay.
    With incident_nodes_a, source parameters are piecewise-linear density
    values at those nodes. The two nonnegative hat weights are evaluated at
    each sampled incident wavelength, not at its integration-bin midpoint.
    Nodes and scattering integration bins may differ, but have the same domain.
    Without nodes, source density is constant on each incident integration bin.
    Geometry/path-delay distributions and multiple scattering require explicit
    source/integrator support; this routine does not invent them.
    """
    incident = _edges(incident_edges_a, 'Incident wavelength')
    outgoing = _edges(outgoing_edges_a, 'Outgoing wavelength')
    d_edges = _edges(d_edges_a, 'Observed d spacing')
    angular_edges = _edges(angular_edges_deg, 'Polar angle', angle=True)
    nodes = None if incident_nodes_a is None else _edges(incident_nodes_a, 'Incident density nodes')
    if nodes is not None and (len(nodes) > 4097 or nodes[0] != incident[0] or nodes[-1] != incident[-1]):
        raise ValueError('Density nodes must exactly span the integration domain within the node budget')
    omega = np.asarray(solid_angle_sr, float)
    if omega.ndim != 2 or omega.shape[0] != len(angular_edges) - 1 or not 1 <= omega.shape[1] <= 32:
        raise ValueError('Angular acceptance must have explicit polar/azimuth-cell dimensions')
    if not np.isfinite(omega).all() or np.any(omega < 0):
        raise ValueError('Solid angles must be finite and nonnegative')
    ring = 2 * np.pi * (np.cos(np.deg2rad(angular_edges[:-1])) - np.cos(np.deg2rad(angular_edges[1:])))
    if np.any(omega.sum(axis=1) > ring * (1 + 1e-12)):
        raise ValueError('Angular cells exceed their full polar-ring solid angle')
    geometry = [np.broadcast_to(np.asarray(v, float), omega.shape).ravel() for v in
                (l2_m, difc_ratio, difa_us_per_a2, tzero_us)]
    if any(not np.isfinite(v).all() for v in geometry) or np.any(geometry[0] <= 0) or np.any(geometry[1] <= 0):
        raise ValueError('Finite physical flight paths and monotonic calibration coefficients are required')
    raw_groups = np.asarray(observed_group_index)
    if raw_groups.shape != omega.shape or raw_groups.dtype.kind not in 'iu' or np.any(raw_groups < 0):
        raise ValueError('Each angular cell needs an explicit nonnegative observed-group index')
    groups = raw_groups.ravel()
    ngroup = int(groups.max()) + 1
    ni, no, nd = len(incident) - 1, len(outgoing) - 1, len(d_edges) - 1
    ns = ni if nodes is None else len(nodes)
    event_work_bytes = 256 if nodes is None else 384
    if (type(samples_per_incident_bin) is not int or samples_per_incident_bin < 1
            or type(seed) is not int or not 0 <= seed < 2**32 or type(max_events) is not int
            or ni * samples_per_incident_bin > max_events or ngroup * nd > 1000000
            or ns * no * ngroup * nd > np.iinfo(np.int64).max
            or type(max_bytes) is not int or samples_per_incident_bin * event_work_bytes > max_bytes):
        raise ValueError('Energy response exceeds its explicit event, dimension or memory budget')
    if not isinstance(scattering_law_id, str) or not scattering_law_id.strip():
        raise ValueError('A traceable scattering-law identity is required')
    if reduction_window is not None and not isinstance(reduction_window, ReductionWindow):
        raise ValueError('Reduction cuts need an explicit ReductionWindow')
    if type(clock_model) is not str or clock_model not in ('physical', 'calibrated_elastic_mean'):
        raise ValueError('An explicit physical or calibrated_elastic_mean clock is required')
    _positive(l1_m, 'Incident flight path')
    keys, coefficients, omitted_rows, omitted_columns, omitted_values = [], [], [], [], []
    def source_terms(column, wavelengths):
        if nodes is None:
            return ((column, 1.),)
        left = np.minimum(np.searchsorted(nodes, wavelengths, side='right') - 1, len(nodes) - 2)
        right_weight = (wavelengths - nodes[left]) / (nodes[left + 1] - nodes[left])
        return ((left, 1 - right_weight), (left + 1, right_weight))
    raw_records = []
    for column in range(ni):
        child_seed = int(np.random.SeedSequence([seed, column]).generate_state(1)[0])
        batch = scattering_batches(float(incident[column]), float(incident[column + 1]), samples_per_incident_bin, child_seed)
        if len(batch) != 4:
            raise ValueError('Scattering provider must return joint incident/final wavelength, angle and weight')
        lin, lout, cosine, weight = [np.asarray(v, float) for v in batch]
        if any(v.shape != (samples_per_incident_bin,) or not np.isfinite(v).all() for v in (lin,lout,cosine,weight)):
            raise ValueError('Scattering batches must be finite matching vectors')
        if (np.any(lin < incident[column]) or np.any(lin > incident[column+1]) or np.any(lout <= 0)
                or np.any(cosine < -1) or np.any(cosine > 1) or np.any(weight < 0)):
            raise ValueError('Scattering batch is outside its physical incident/energy/angle domain')
        theta = np.rad2deg(np.arccos(cosine))
        polar, angular_good = _bin(angular_edges, theta)
        angular_good &= cosine < 1  # exactly forward scattering has infinite d
        oi, outgoing_good = _bin(outgoing, lout)
        for azimuth in range(omega.shape[1]):
            selected = np.flatnonzero(angular_good & (weight > 0))
            cells = polar[selected] * omega.shape[1] + azimuth
            active = omega.ravel()[cells] > 0
            selected, cells = selected[active], cells[active]
            if not len(selected):
                continue
            delay = 0. if pulse_delay is None else pulse_delay(cells, lin[selected], lout[selected])
            apparent, time = apparent_d(lin[selected], lout[selected], cosine[selected], l1_m=l1_m,
                l2_m=geometry[0][cells], difc_ratio=geometry[1][cells], difa_us_per_a2=geometry[2][cells],
                tzero_us=geometry[3][cells], delay_us=delay, clock_model=clock_model)
            di, good_d = _bin(d_edges, apparent)
            if reduction_window is not None:
                good_d &= reduction_window.accept(time, apparent, cosine[selected])
            selected, cells, di, time = selected[good_d], cells[good_d], di[good_d], time[good_d]
            if not len(selected):
                continue
            acceptance = (np.ones(len(selected)) if event_acceptance is None else
                np.broadcast_to(np.asarray(event_acceptance(cells, lin[selected], lout[selected], time), float), (len(selected),)))
            if not np.isfinite(acceptance).all() or np.any(acceptance < 0) or np.any(acceptance > 1):
                raise ValueError('Physical event acceptance must lie in [0,1]')
            values = weight[selected] * omega.ravel()[cells] / ring[polar[selected]] * acceptance
            rows = groups[cells] * nd + di
            covered = outgoing_good[selected]
            if np.any(~covered):
                # Unit path/detection efficiency is an upper bound, not a mean.
                for source_column, basis_weight in source_terms(column, lin[selected[~covered]]):
                    omitted_rows.append(rows[~covered])
                    omitted_columns.append(np.broadcast_to(source_column, ((~covered).sum(),)))
                    omitted_values.append(values[~covered] * basis_weight)
            if covered.any():
                selected, cells, rows, values = selected[covered], cells[covered], rows[covered], values[covered]
                path = np.broadcast_to(np.asarray(path_weight(cells, lin[selected], lout[selected]), float), (len(selected),))
                if not np.isfinite(path).all() or np.any(path < 0) or np.any(path > 1):
                    raise ValueError('Whole-powder path weights must lie in [0,1]')
                for source_column, basis_weight in source_terms(column, lin[selected]):
                    key = (rows * ns + source_column) * no + oi[selected]
                    unique, inverse = np.unique(key, return_inverse=True)
                    keys.append(unique); coefficients.append(np.bincount(inverse, weights=values * path * basis_weight))
        raw_records.append(dict(incident_bin=column, input_integrated_cross_section_b_a=float(weight.sum()),
            sampling_points=samples_per_incident_bin))
        retained_entries = sum(len(x) for x in keys) + sum(len(x) for x in omitted_rows)
        if retained_entries * 64 + samples_per_incident_bin * event_work_bytes > max_bytes:
            raise ValueError('Sparse response construction exceeds its declared working memory budget')
    key = np.concatenate(keys) if keys else np.array([], dtype=np.int64)
    coeff = np.concatenate(coefficients) if coefficients else np.array([], dtype=float)
    unique, inverse = np.unique(key, return_inverse=True)
    coeff = np.bincount(inverse, weights=coeff)
    positive = coeff > 0
    unique, coeff = unique[positive], coeff[positive]
    outgoing_index = unique % no
    incident_index = (unique // no) % ns
    observation_index = unique // (no * ns)
    omitted = sparse.csr_matrix((ngroup * nd, ns))
    if omitted_rows:
        omitted = sparse.coo_matrix((np.concatenate(omitted_values),
            (np.concatenate(omitted_rows), np.concatenate(omitted_columns))), shape=omitted.shape).tocsr()
        omitted.eliminate_zeros()
    for array in (incident, outgoing, d_edges, observation_index, incident_index, outgoing_index, coeff):
        array.setflags(write=False)
    metadata = dict(version=VERSION, scattering_law_id=scattering_law_id, seed=seed,
        integration_samples=ni * samples_per_incident_bin, incident_bin_records=raw_records,
        coefficient_unit='barn*angstrom per scattering cell', solid_angle_convention='fraction of full 2*pi*dcos(theta) ring',
        outgoing_energy_not_renamed_incident_energy=True, numerical_samples_are_physical_count_noise=False,
        outgoing_domain_loss_renormalized=False, whole_powder_path_includes_illumination_once=True,
        angular_cells=omega.size, observed_groups=ngroup, clock_model=clock_model,
        incident_density_basis='constant_bins' if nodes is None else 'linear_nodes',
        incident_parameter_count=ns,
        azimuthally_isotropic_scattering_assumed=True,
        reduction_window=None if reduction_window is None else dict(
            tof_range_us=reduction_window.tof_range_us,
            apparent_wavelength_range_a=reduction_window.apparent_wavelength_range_a,
            prompt_period_us=reduction_window.prompt_period_us,
            prompt_width_us=reduction_window.prompt_width_us),
        multiple_scattering_included=False, source_detector_identifiability_established=False,
        physical_Poisson_independence_or_stationarity_verified=False, experimental_accuracy_certified=False)
    return EnergyResponse(incident, outgoing, d_edges, ngroup, observation_index, incident_index,
        outgoing_index, coeff, omitted, metadata, nodes)
