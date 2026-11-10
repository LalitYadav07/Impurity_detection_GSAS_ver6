"""Positive, bin-integrated incident-monitor response with explicit timing.

The incident fluence density and effective monitor area use piecewise-linear
nodes. The effective area (cm²) includes sampled beam area, efficiency and any
declared transport from the sample-referenced fluence to the upstream monitor.
It must be supplied; monitor counts alone do not identify it separately from
fluence. A wavelength-dependent discrete delay law is integrated jointly with
both curves. Its quadrature weights are not physical counting noise.

An unwindowed calculation is an explicit hypothesis. Neither missing coverage
nor a zero observed bin establishes zero incident flux outside an independently
established acquisition window. No response here is experimentally qualified.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy import sparse
from .energy_response import TOF_US_PER_M_A

VERSION = 'powgen-positive-monitor-response/1'


def _positive(value, name):
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
            or not math.isfinite(float(value)) or value <= 0):
        raise ValueError(name + ' must be a finite positive scalar')
    return float(value)


def _grid(values, name, maximum, *, positive=True):
    values = np.asarray(values, float)
    if (values.ndim != 1 or not 2 <= len(values) <= maximum
            or not np.isfinite(values).all() or np.any(np.diff(values) <= 0)
            or values[0] < 0 or (positive and values[0] == 0)):
        raise ValueError(name + ' needs bounded, increasing physical coordinates')
    return values.copy()


@dataclass(frozen=True)
class MonitorResponse:
    incident_nodes_a: np.ndarray
    tof_edges_us: np.ndarray
    matrix_cm2_a: sparse.csr_matrix
    unobserved_cm2_a: np.ndarray
    metadata: dict

    def predict(self, incident_density, *, exposure=1.):
        """Expected integer-count means and separate unobserved-domain mean.

        Density units are neutrons/(cm² * exposure-unit * angstrom). This
        function returns means; a caller generates Poisson increments only when
        a physical count-process model is justified. No shape is renormalized
        to compensate for finite TOF coverage, gates or vetoes.
        """
        density = np.asarray(incident_density, float)
        if (density.shape != (self.matrix_cm2_a.shape[1],) or not np.isfinite(density).all()
                or np.any(density < 0)):
            raise ValueError('Matching finite nonnegative incident density is required')
        q = _positive(exposure, 'Exposure')
        observed = q * np.asarray(self.matrix_cm2_a @ density)
        unobserved = q * float(np.sum(self.unobserved_cm2_a * density))
        if not np.isfinite(observed).all() or not math.isfinite(unobserved):
            raise ValueError('Expected monitor counts exceed their finite numerical domain')
        return observed, unobserved


def build_monitor_response(incident_nodes_a, effective_area_cm2, tof_edges_us, *,
        source_to_monitor_m, mean_delay_us=None, delay_offsets_us=(0.,),
        delay_probabilities=None, tof_windows_us=None, prompt_period_us=None,
        prompt_width_us=0., tof_us_per_m_a=TOF_US_PER_M_A,
        incident_density_basis='linear_nodes', max_bytes=256*1024**2):
    """Integrate a supplied monitor model exactly within its interpolation basis.

    t = k*L*lambda + mean_delay(lambda) + offset. Mean delay, effective area
    and delay probabilities are linear between the incident nodes. Probabilities
    sum to one at each node, allowing wavelength-dependent broadening. Each
    offset's arrival time must increase with wavelength. Source density is
    linear on these nodes, or constant on their intervals with constant_bins.
    Near-unit probability row sums are explicitly normalized within 1e-12 to
    remove floating arithmetic error, never to repair a truncated source law.
    The two-point Gaussian
    rule is exact for the resulting cubic integrands after all bin/window/prompt
    discontinuities are split. It does not establish convergence to a continuous
    physical source-time law or justify the supplied monitor response.

    tof_windows_us=None means an unwindowed hypothesis, never verified hardware
    coverage. Otherwise supply sorted, disjoint recorded windows. Unobserved
    response is retained explicitly; it is not added to any observed bin.
    """
    nodes = _grid(incident_nodes_a, 'Incident nodes', 4097)
    edges = _grid(tof_edges_us, 'Monitor TOF edges', 100001, positive=False)
    if type(incident_density_basis) is not str or incident_density_basis not in ('linear_nodes','constant_bins'):
        raise ValueError('An explicit linear_nodes or constant_bins source basis is required')
    area = np.asarray(effective_area_cm2, float)
    mean = np.zeros(len(nodes)) if mean_delay_us is None else np.asarray(mean_delay_us, float)
    offsets = np.asarray(delay_offsets_us, float)
    if (area.shape != nodes.shape or mean.shape != nodes.shape
            or not np.isfinite(area).all() or np.any(area < 0) or not np.isfinite(mean).all()
            or offsets.ndim != 1 or not 1 <= len(offsets) <= 64
            or not np.isfinite(offsets).all() or np.any(np.diff(offsets) <= 0)):
        raise ValueError('Matching physical area/delay nodes and bounded sorted delay offsets are required')
    if delay_probabilities is None:
        if len(offsets) != 1:
            raise ValueError('Several delay offsets need explicit probabilities')
        probabilities = np.ones((len(nodes), 1))
    else:
        probabilities = np.asarray(delay_probabilities, float)
        if probabilities.shape == offsets.shape:
            probabilities = np.broadcast_to(probabilities, (len(nodes), len(offsets)))
    if (probabilities.shape != (len(nodes), len(offsets)) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1., rtol=0, atol=1e-12)):
        raise ValueError('Nonnegative normalized delay probabilities are required at every incident node')
    probability_error = float(np.max(abs(probabilities.sum(axis=1)-1)))
    probabilities = probabilities/probabilities.sum(axis=1)[:,None]
    arrival = (_positive(tof_us_per_m_a, 'TOF constant')
               * _positive(source_to_monitor_m, 'Monitor flight path') * nodes + mean)
    if (not np.isfinite(arrival).all() or np.any(np.diff(arrival) <= 0)
            or arrival[0] + offsets[0] < 0):
        raise ValueError('Each delay branch needs a finite increasing nonnegative physical event clock')
    if type(max_bytes) is not int or max_bytes < 1024:
        raise ValueError('An explicit positive working memory budget is required')
    if tof_windows_us is None:
        windows = None
    else:
        windows = np.asarray(tof_windows_us, float)
        if (windows.ndim != 2 or windows.shape[1] != 2 or not 1 <= len(windows) <= 1024
                or not np.isfinite(windows).all() or np.any(windows[:, 0] < 0)
                or np.any(windows[:, 1] <= windows[:, 0])
                or np.any(windows[1:, 0] < windows[:-1, 1])):
            raise ValueError('Recorded TOF windows must be bounded, sorted and disjoint')
    if (isinstance(prompt_width_us, (bool, np.bool_)) or not np.isscalar(prompt_width_us)
            or not math.isfinite(float(prompt_width_us)) or prompt_width_us < 0):
        raise ValueError('Prompt width must be finite and nonnegative')
    if prompt_period_us is None:
        if prompt_width_us:
            raise ValueError('A prompt veto needs an explicit period')
        period = None
    else:
        period = _positive(prompt_period_us, 'Prompt period')
        if prompt_width_us >= period:
            raise ValueError('Prompt width must be smaller than its period')

    cuts = [edges]
    if windows is not None:
        cuts.append(np.clip(windows.ravel(), edges[0], edges[-1]))
    if period is not None and prompt_width_us:
        first = math.floor((edges[0] - prompt_width_us)/period)
        last = math.floor(edges[-1]/period)
        if last-first+1 > 100000:
            raise ValueError('Prompt-veto partition exceeds the time-piece budget')
        pulse = np.arange(first, last+1) * period
        cuts.extend([np.clip(pulse, edges[0], edges[-1]),
                     np.clip(pulse + prompt_width_us, edges[0], edges[-1])])
    boundaries = np.unique(np.concatenate(cuts))
    left, right = boundaries[:-1], boundaries[1:]
    middle = (left+right)/2
    accepted = np.ones(len(middle), bool)
    if windows is not None:
        index = np.searchsorted(windows[:, 0], middle, side='right') - 1
        accepted &= (index >= 0) & (middle <= windows[np.maximum(index, 0), 1])
    if period is not None and prompt_width_us:
        from .prompt_gate import periodic_prompt_acceptance
        accepted &= periodic_prompt_acceptance(middle, period, prompt_width_us, upper_inclusive=True)
    left, right, middle = left[accepted], right[accepted], middle[accepted]
    rows = np.searchsorted(edges, middle, side='right') - 1
    partition_bytes = (len(boundaries)*64 + len(nodes)*len(offsets)*64 + len(edges)*64)
    if partition_bytes > max_bytes:
        raise ValueError('Monitor time partition exceeds its working memory budget')
    row_parts, col_parts, value_parts = [], [], []
    terms = 0
    for j in range(len(nodes)-1):
        span = nodes[j+1]-nodes[j]
        slope = (arrival[j+1]-arrival[j])/span
        for k, offset in enumerate(offsets):
            if probabilities[j, k] == probabilities[j+1, k] == 0 or area[j] == area[j+1] == 0:
                continue
            lower = np.searchsorted(right, arrival[j]+offset, side='right')
            upper = np.searchsorted(left, arrival[j+1]+offset, side='left')
            if lower >= upper:
                continue
            lo = np.maximum(left[lower:upper], arrival[j]+offset)
            hi = np.minimum(right[lower:upper], arrival[j+1]+offset)
            # Fractions of the source-node interval at exact time intersections.
            a = (lo-arrival[j]-offset)/(slope*span)
            b = (hi-arrival[j]-offset)/(slope*span)
            a, b = np.clip(a, 0., 1.), np.clip(b, 0., 1.)
            width = (b-a)*span
            midpoint, radius = (a+b)/2, (b-a)/(2*math.sqrt(3))
            fraction = np.stack([midpoint-radius, midpoint+radius], axis=1)
            effective = area[j] + fraction*(area[j+1]-area[j])
            probability = probabilities[j,k] + fraction*(probabilities[j+1,k]-probabilities[j,k])
            weight = width[:, None]/2 * effective * probability
            terms += (2 if incident_density_basis=='linear_nodes' else 1)*len(lo)
            if partition_bytes + terms*128 > max_bytes:
                raise ValueError('Monitor sparse integration exceeds its working memory budget')
            if incident_density_basis=='linear_nodes':
                row_parts.extend([rows[lower:upper], rows[lower:upper]])
                col_parts.extend([np.full(len(lo), j, dtype=np.int32), np.full(len(lo), j+1, dtype=np.int32)])
                value_parts.extend([(weight*(1-fraction)).sum(axis=1), (weight*fraction).sum(axis=1)])
            else:
                row_parts.append(rows[lower:upper])
                col_parts.append(np.full(len(lo), j, dtype=np.int32))
                value_parts.append(weight.sum(axis=1))
    shape = (len(edges)-1, len(nodes) if incident_density_basis=='linear_nodes' else len(nodes)-1)
    matrix = (sparse.coo_matrix((np.concatenate(value_parts),
              (np.concatenate(row_parts), np.concatenate(col_parts))), shape=shape).tocsr()
              if terms else sparse.csr_matrix(shape))
    matrix.eliminate_zeros();matrix.sort_indices()
    if incident_density_basis=='linear_nodes':
        total = np.zeros(len(nodes))
        total[:-1] += np.diff(nodes)*(2*area[:-1]+area[1:])/6
        total[1:] += np.diff(nodes)*(area[:-1]+2*area[1:])/6
    else:
        total = np.diff(nodes)*(area[:-1]+area[1:])/2
    unobserved = total - np.asarray(matrix.sum(axis=0)).ravel()
    allowed = 256*np.finfo(float).eps * total
    if np.any(unobserved < -allowed) or not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError('Monitor integration failed positive count conservation')
    clipped = float(np.sum(np.maximum(-unobserved, 0.)))
    unobserved = np.maximum(unobserved, 0.)
    for array in (nodes, edges, unobserved): array.setflags(write=False)
    return MonitorResponse(nodes, edges, matrix, unobserved, dict(version=VERSION,
        effective_area_units='cm2; includes monitor efficiency, sampled beam area and declared upstream transfer',
        clock='k*source_to_monitor*lambda + piecewise_linear_mean_delay(lambda) + discrete_offset',
        tof_us_per_m_a=float(tof_us_per_m_a), source_to_monitor_m=float(source_to_monitor_m),
        delay_offsets=len(offsets), normalized_probability_tolerance=1e-12,
        maximum_input_probability_row_sum_error=probability_error,
        near_unit_probability_rows_explicitly_normalized=True,
        acquisition_model='unwindowed_hypothesis' if windows is None else 'specified_windows',
        tof_windows_us=None if windows is None else windows.tolist(),
        prompt_period_us=period, prompt_width_us=float(prompt_width_us),
        quadrature='two-point Gaussian, exact for cubic basis products after time discontinuity splitting',
        source_basis=incident_density_basis, sparse_coefficients=matrix.nnz,
        maximum_working_memory_budget_bytes=max_bytes, floating_subtraction_clipping_cm2_a=clipped,
        unobserved_response_renormalized=False, source_count_floor_added=False,
        acquisition_window_or_monitor_response_independently_verified=False,
        physical_source_time_law_convergence_established=False, experimental_accuracy_certified=False))
