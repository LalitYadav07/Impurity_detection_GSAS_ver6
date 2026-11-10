"""Elastic powder lines in physical count units, before experimental qualification.

For an equivalent reflection family, A_Q=m|F|²d²/(2V) is the Q-integrated
differential cross section per crystallographic cell. At a fixed physical
detector angle, lambda_h=2d sin(theta) and its wavelength-integrated area is
A_lambda=A_Q*lambda_h*d/(2*pi), in barn*angstrom/sr. Incident fluence density
is evaluated at lambda_h. A native TOF profile area is not this coefficient.

Factors are supplied in barns per declared cell and may depend on wavelength
(e.g. isotope amplitudes). Multiplicities must already include Friedel signs.
For unpolarized magnetic lines the provider must supply the correctly averaged
transverse magnetic amplitude squared, in the same units. This module neither
constructs magnetic symmetry nor establishes that factor convention itself.

Only supplied lines and the declared source domain are modeled. Missing source
coverage, finite timing mass, reduction losses and output-bin losses are never
renormalized. Ideal powder, one elastic collision and supplied geometry/path/
efficiency/timing laws are hypotheses, not a validated POWGEN instrument pack.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy import sparse
from .energy_response import apparent_d, ReductionWindow, TOF_US_PER_M_A

VERSION = 'powgen-elastic-powder-line-count-response/1'


def _positive(value, name):
    if isinstance(value, (bool, np.bool_, str, bytes)) or not np.isscalar(value):
        raise ValueError(name+' must be a finite positive scalar')
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name+' must be a finite positive scalar')
    return value


def _vector(value, name, *, minimum=0., strict=False, maximum=math.inf):
    result = np.array(value, dtype=float, copy=True)
    if (result.ndim != 1 or not 1 <= len(result) <= 100000 or not np.isfinite(result).all()
            or np.any(result <= minimum if strict else result < minimum) or np.any(result > maximum)):
        raise ValueError(name+' needs a bounded finite physical vector')
    return result


def _edges(value, name):
    result = _vector(value, name, strict=True)
    if len(result) < 2 or np.any(np.diff(result) <= 0):
        raise ValueError(name+' needs increasing positive edges/nodes')
    return result


def _lines(d_a, multiplicity, cell_volume_a3):
    d = _vector(d_a, 'Reflection d spacings', strict=True)
    m = _vector(multiplicity, 'Reflection multiplicities', strict=True)
    if m.shape != d.shape or np.any(m != np.rint(m)) or np.any(m > 1000000):
        raise ValueError('Matching positive integer multiplicities, including Friedel signs, are required')
    return d, m, _positive(cell_volume_a3, 'Crystallographic cell volume')


def _factors(provider, indices, wave):
    value = np.asarray(provider(indices, wave), float)
    if value.shape != wave.shape or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError('Factor provider must return matching finite nonnegative squared amplitudes in barns')
    return value


def powder_cross_section(d_a, multiplicity, cell_volume_a3, factor_squared_b, wavelength_a, *,
                         max_factor_values=2000000):
    """Supplied-family elastic powder removal in barn per declared cell.

    Sigma(lambda)=lambda²/(2V) sum[2d>=lambda] m*d*|F(lambda)|².
    The inclusive Bragg-edge convention is explicit; a single endpoint has zero
    weight in a continuous wavelength integral. Reflection completeness and the
    physical factor/ADP/isotope model remain the caller's responsibility.
    """
    d, m, volume = _lines(d_a, multiplicity, cell_volume_a3)
    wave = _vector(wavelength_a, 'Incident wavelengths', strict=True)
    if not callable(factor_squared_b) or type(max_factor_values) is not int or max_factor_values < 1:
        raise ValueError('An explicit factor provider and bounded evaluation budget are required')
    if len(d)*len(wave) > max_factor_values:
        raise ValueError('Powder cross-section evaluation exceeds its factor budget')
    answer = np.zeros_like(wave)
    for i, wavelength in enumerate(wave):
        indices = np.flatnonzero(2*d >= wavelength)
        if len(indices):
            factors = _factors(factor_squared_b, indices, np.full(len(indices), wavelength))
            answer[i] = wavelength**2/(2*volume)*np.sum(m[indices]*d[indices]*factors)
    if not np.isfinite(answer).all():
        raise ValueError('Powder cross sections exceed their finite numerical domain')
    return answer


@dataclass(frozen=True)
class PowderLineResponse:
    source_edges_a: np.ndarray
    source_nodes_a: np.ndarray | None
    d_edges_a: np.ndarray
    observed_groups: int
    coefficient_b_a: sparse.csr_matrix
    unobserved_b_a: sparse.csr_matrix
    detected_before_cuts_b_a: sparse.csr_matrix
    metadata: dict

    def source_matrix(self, *, number_cells):
        """cm²*A for a WHOLE specimen; multiply by incident fluence density.

        Density units are neutrons/(cm²*exposure*A). Supplied path weights
        already include whole-powder illumination exactly once. This does not
        fit a scale, establish absolute source calibration or add count noise.
        """
        scale = 1e-24*_positive(number_cells, 'Whole specimen crystallographic cell count')
        result = self.coefficient_b_a*scale
        if not np.isfinite(result.data).all():
            raise ValueError('Specimen scaling exceeds the finite count domain')
        return result


def build_powder_line_response(d_a, multiplicity, cell_volume_a3, factor_squared_b,
        source_edges_a, d_edges_a, *, cos_two_theta, solid_angle_sr, observed_group,
        l1_m, l2_m, path_weight, detector_efficiency, timing_offsets_us,
        timing_probability_mass, mean_delay_us, timing_law_label,
        source_nodes_a=None, difc_ratio=1., difa_us_per_a2=0., tzero_us=0.,
        clock_model='physical', window=None, tof_us_per_m_a=TOF_US_PER_M_A,
        max_bytes=512*1024**2, max_line_detector_pairs=2000000):
    """Build a positive elastic count operator at supplied angular quadrature points.

    path_weight(cell, lambda) is a whole-powder [0,1] single-collision weight.
    detector_efficiency(cell, lambda) is an absolute [0,1] probability.
    mean_delay_us(cell, lambda) supplies a physical-clock delay hypothesis.
    timing_probability_mass(cell, lambda) returns one row per wavelength and
    one column per supplied timing offset. Row sums <=1 are permitted: missing
    timing mass is retained as unobserved, rather than redistributed.

    Source density is constant by interval or linear between independent
    nodes spanning the declared domain. No source extrapolation is performed.
    Lines outside that domain are explicitly reported as unmodeled candidates;
    their contribution is not claimed zero or bounded without source evidence.
    Geometry and reflection/timing quadrature convergence require separate tests.
    """
    d, multiplicities, volume = _lines(d_a, multiplicity, cell_volume_a3)
    edges = _edges(source_edges_a, 'Incident source domain')
    output = _edges(d_edges_a, 'Observed d domain')
    nodes = None if source_nodes_a is None else _edges(source_nodes_a, 'Incident density nodes')
    if nodes is not None and (len(nodes) > 4097 or nodes[0] != edges[0] or nodes[-1] != edges[-1]):
        raise ValueError('Density nodes must exactly span the declared source domain within budget')
    if len(edges) > 4098 or len(output) > 100001:
        raise ValueError('Source/observed bin count exceeds its declared bound')
    cosine = _vector(cos_two_theta, 'Physical scattering cosines', minimum=-1., maximum=1.)
    if np.any(cosine >= 1):
        raise ValueError('Nonforward physical detector angles are required')
    omega = _vector(solid_angle_sr, 'Detector solid-angle weights', strict=True, maximum=4*np.pi)
    if omega.shape != cosine.shape or omega.sum() > 4*np.pi*(1+1e-14):
        raise ValueError('Matching solid-angle weights must not exceed one complete sphere')
    groups = np.asarray(observed_group)
    if (groups.shape != cosine.shape or groups.dtype.kind not in 'iu' or np.any(groups < 0)
            or groups.max() >= 2048 or not np.array_equal(np.unique(groups), np.arange(groups.max()+1))):
        raise ValueError('Contiguous bounded observed group indices are required')
    offsets = _vector(timing_offsets_us, 'Recorded timing offsets', minimum=-np.finfo(float).max)
    if len(offsets) > 4097 or np.any(np.diff(offsets) <= 0):
        raise ValueError('Timing offsets need a bounded increasing finite grid')
    if (type(timing_law_label) is not str or not timing_law_label.strip() or len(timing_law_label) > 512
            or not all(callable(f) for f in (factor_squared_b, path_weight, detector_efficiency,
                                            timing_probability_mass, mean_delay_us))):
        raise ValueError('Explicit bounded timing provenance and factor/path/efficiency/timing providers are required')
    if (type(max_bytes) is not int or not 1024 <= max_bytes <= 2*1024**3
            or type(max_line_detector_pairs) is not int or max_line_detector_pairs < 1):
        raise ValueError('Explicit bounded working memory and line/detector budgets are required')
    pairs = len(d)*len(cosine)
    source_width = 1 if nodes is None else 2
    estimated = pairs*(len(offsets)+2)*source_width*96 + (len(d)+len(cosine))*256
    ngroups, nbins = int(groups.max()+1), len(output)-1
    nsource = len(edges)-1 if nodes is None else len(nodes)
    if pairs > max_line_detector_pairs or estimated > max_bytes or ngroups*nbins > 1000000:
        raise ValueError('Powder line response exceeds its explicit working budget')
    l1 = _positive(l1_m, 'Incident flight path')
    calibration = []
    for value, name, positive in [(l2_m, 'Final flight paths', True), (difc_ratio, 'DIFC ratios', True),
            (difa_us_per_a2, 'DIFA coefficients', False), (tzero_us, 'TZERO coefficients', False)]:
        array = np.broadcast_to(np.asarray(value, float), cosine.shape)
        if not np.isfinite(array).all() or (positive and np.any(array <= 0)):
            raise ValueError('Matching finite physical '+name+' required')
        calibration.append(array)
    l2, ratios, difa, zero = calibration
    if window is None:
        window = ReductionWindow()
    if not isinstance(window, ReductionWindow):
        raise ValueError('An explicit ReductionWindow is required')
    # Validate clock/constants even if every candidate is outside the source domain.
    apparent_d(np.array([1.]), np.array([1.]), np.array([0.]), l1_m=l1, l2_m=1.,
        clock_model=clock_model, tof_us_per_m_a=tof_us_per_m_a)
    parts = [[], [], []]
    missed = [[], [], []]
    total = [[], [], []]
    uncovered = 0
    outside_min, outside_max = math.inf, -math.inf
    maximum_missing_timing_mass = 0.

    def append(storage, row, column, weight):
        row, column, weight = np.broadcast_arrays(row, column, weight)
        active = weight > 0
        if active.any():
            storage[0].append(np.asarray(row[active], dtype=np.int64))
            storage[1].append(np.asarray(column[active], dtype=np.int64))
            storage[2].append(np.asarray(weight[active], dtype=float))

    for cell in range(len(cosine)):
        wave = 2*d*np.sqrt((1-cosine[cell])/2)
        inside = (wave >= edges[0]) & (wave <= edges[-1])
        if (~inside).any():
            uncovered += int((~inside).sum())
            outside_min = min(outside_min, float(wave[~inside].min()))
            outside_max = max(outside_max, float(wave[~inside].max()))
        indices = np.flatnonzero(inside)
        wave = wave[inside]
        if not len(wave):
            continue
        factors = _factors(factor_squared_b, indices, wave)
        path = np.asarray(path_weight(cell, wave), float)
        efficiency = np.asarray(detector_efficiency(cell, wave), float)
        delay = np.asarray(mean_delay_us(cell, wave), float)
        probability = np.asarray(timing_probability_mass(cell, wave), float)
        if (path.shape != wave.shape or efficiency.shape != wave.shape or delay.shape != wave.shape
                or not all(np.isfinite(a).all() for a in (path, efficiency, delay, probability))
                or any(np.any(a < 0) or np.any(a > 1) for a in (path, efficiency))
                or probability.shape != (len(wave), len(offsets)) or np.any(probability < 0)
                or np.any(probability.sum(axis=1) > 1)):
            raise ValueError('Matching physical path/efficiency, finite delay and subprobability timing laws are required')
        missing = 1-probability.sum(axis=1)
        maximum_missing_timing_mass = max(maximum_missing_timing_mass, float(missing.max()))
        # dQ/dlambda = -2*pi/(d*lambda_h) at the ideal elastic line.
        weight = (multiplicities[indices]*factors*d[indices]**3*wave/(4*np.pi*volume)
                  * omega[cell]*path*efficiency)
        if not np.isfinite(weight).all():
            raise ValueError('Elastic line response exceeds its finite numerical domain')
        basis = edges if nodes is None else nodes
        left = np.minimum(np.searchsorted(basis, wave, side='right')-1, len(basis)-2)
        if nodes is None:
            source_columns = left[:, None]
            source_weights = np.ones((len(wave), 1))
        else:
            fraction = (wave-nodes[left])/(nodes[left+1]-nodes[left])
            source_columns = np.column_stack((left, left+1))
            source_weights = np.column_stack((1-fraction, fraction))
        weighted = weight[:, None]*source_weights
        append(total, groups[cell], source_columns, weighted)
        append(missed, groups[cell], source_columns, weighted*missing[:, None])
        for j, offset in enumerate(offsets):
            active = (probability[:, j] > 0) & (weight > 0)
            if not active.any():
                continue
            w = wave[active]
            coordinate, arrival = apparent_d(w, w, cosine[cell], l1_m=l1, l2_m=l2[cell],
                difc_ratio=ratios[cell], difa_us_per_a2=difa[cell], tzero_us=zero[cell],
                delay_us=delay[active]+offset, clock_model=clock_model, tof_us_per_m_a=tof_us_per_m_a)
            bin_index = np.searchsorted(output, coordinate, side='right')-1
            bin_index[coordinate == output[-1]] = nbins-1
            accepted = (bin_index >= 0) & (bin_index < nbins) & window.accept(arrival, coordinate, cosine[cell])
            piece = weighted[active]*probability[active, j, None]
            append(parts, (groups[cell]*nbins+bin_index[accepted])[:, None],
                source_columns[active][accepted], piece[accepted])
            append(missed, groups[cell], source_columns[active][~accepted], piece[~accepted])

    def matrix(storage, rows):
        arrays = [np.concatenate(p) if p else np.array([], dtype=float) for p in storage]
        result = sparse.coo_matrix((arrays[2], (arrays[0].astype(np.int64), arrays[1].astype(np.int64))),
            shape=(rows, nsource)).tocsr()
        result.eliminate_zeros()
        if not np.isfinite(result.data).all():
            raise ValueError('Accumulated elastic response exceeds its finite numerical domain')
        for array in (result.data, result.indices, result.indptr):
            array.setflags(write=False)
        return result

    for array in (edges, nodes, output):
        if array is not None:
            array.setflags(write=False)
    return PowderLineResponse(edges, nodes, output, ngroups, matrix(parts, ngroups*nbins),
        matrix(missed, ngroups), matrix(total, ngroups), dict(version=VERSION,
            coefficient_unit='barn*angstrom per declared crystallographic cell',
            factor_unit='barn per declared cell; multiplicity supplied separately including Friedel signs',
            line_area_convention='A_lambda=m*|F(lambda_h)|^2*d^3*lambda_h/(4*pi*V)',
            incident_density_unit='neutrons/(cm^2*declared exposure*angstrom)',
            source_basis='piecewise_constant' if nodes is None else 'piecewise_linear',
            number_reflection_families=len(d), number_detector_quadrature_points=len(cosine),
            solid_angle_sr=float(omega.sum()), clock_model=clock_model,
            timing_law_label=timing_law_label, timing_offsets_us=offsets.tolist(),
            maximum_missing_timing_probability_mass=maximum_missing_timing_mass,
            missing_timing_or_reduction_mass_renormalized=False,
            whole_powder_path_includes_illumination_once=True,
            outside_source_domain_candidate_line_detector_pairs=uncovered,
            outside_source_domain_candidate_wavelength_range_a=None if uncovered == 0 else [outside_min, outside_max],
            outside_source_domain_contribution_established_zero_or_bounded=False,
            source_support_or_reflection_completeness_verified=False,
            angular_or_timing_quadrature_convergence_established=False,
            multiple_scattering_or_texture_included=False,
            physical_instrument_response_or_absolute_source_calibration_verified=False,
            experimental_accuracy_certified=False))
