"""Positive elastic quadrature split at diffraction and reduction boundaries.

Fixed angular quadrature can miss a thin allowed strip at a TOF cut even when
its whole-pattern convergence looks excellent. Split the angular integration
where accepted wavelength endpoints cross output d edges. Then split energy
at source nodes, detector-efficiency edges, d edges and reduction boundaries.
Weights integrate d(lambda) * d(cos(2 theta)) / full_cell_cosine_width. They
never renormalize the accepted region. Multiply by the cell solid angle and
the differential scattering law separately.

This helper covers elastic clocks with DIFC only. It does not approximate
energy-changing scattering or silently support nonzero DIFA/TZERO/delay.
"""
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss

from .energy_response import ReductionWindow, TOF_US_PER_M_A, apparent_d


@dataclass(frozen=True)
class ElasticCellQuadrature:
    wavelength_a: np.ndarray
    cos_two_theta: np.ndarray
    bin_index: np.ndarray
    weight_a: np.ndarray


def elastic_cell_quadrature(polar_bounds_deg, d_edges_a, wavelength_nodes_a, *,
                            l1_m, l2_m, difc_ratio=1., window=None,
                            clock_model='calibrated_elastic_mean',
                            wavelength_breaks_a=(), angular_order=4, energy_order=4):
    """Return accepted nodes for one angular cell without dropping narrow bins."""
    polar = np.asarray(polar_bounds_deg, float)
    edges = np.asarray(d_edges_a, float)
    nodes = np.asarray(wavelength_nodes_a, float)
    breaks = np.asarray(wavelength_breaks_a, float)
    if (polar.shape != (2,) or not np.isfinite(polar).all()
            or not 0 < polar[0] < polar[1] <= 180):
        raise ValueError('A positive finite angular cell is required')
    for a in (edges, nodes):
        if (a.ndim != 1 or not 2 <= len(a) <= 4097 or not np.isfinite(a).all()
                or a[0] <= 0 or np.any(np.diff(a) <= 0)):
            raise ValueError('Finite increasing d edges and source nodes are required')
    if (breaks.ndim != 1 or len(breaks) > 100001 or not np.isfinite(breaks).all()
            or np.any(breaks <= 0)):
        raise ValueError('Wavelength integration breaks must be finite and positive')
    for n in (angular_order, energy_order):
        if type(n) is not int or not 1 <= n <= 32:
            raise ValueError('Quadrature orders must be integers from 1 to 32')
    for v in (l1_m, l2_m, difc_ratio):
        if isinstance(v, (bool, np.bool_)) or not np.isscalar(v) or not np.isfinite(v) or v <= 0:
            raise ValueError('Positive finite flight paths and DIFC ratio are required')
    if clock_model not in {'physical', 'calibrated_elastic_mean'}:
        raise ValueError('An explicit supported elastic clock is required')
    window = ReductionWindow() if window is None else window
    if not isinstance(window, ReductionWindow):
        raise ValueError('An explicit reduction window is required')
    coordinate_ratio = difc_ratio if clock_model == 'physical' else 1.
    time_scale = TOF_US_PER_M_A * (l1_m + l2_m)
    if clock_model == 'calibrated_elastic_mean':
        time_scale *= difc_ratio

    # All energy acceptance boundaries are independent of angle for this
    # elastic, zero-DIFA/TZERO clock. Keep every accepted interval, including
    # pieces split by prompt vetoes. Endpoint inclusivity has zero measure.
    cuts = [nodes[0], nodes[-1]]
    if window.tof_range_us is not None:
        cuts.extend(np.asarray(window.tof_range_us) / time_scale)
    if window.apparent_wavelength_range_a is not None:
        cuts.extend(np.asarray(window.apparent_wavelength_range_a) * coordinate_ratio)
    if window.prompt_period_us is not None and window.prompt_width_us > 0:
        last = int(np.ceil(nodes[-1] * time_scale / window.prompt_period_us))
        if last > 10000:
            raise ValueError('Prompt integration exceeds the interval budget')
        starts = np.arange(last + 1) * window.prompt_period_us
        cuts.extend(starts / time_scale)
        cuts.extend((starts + window.prompt_width_us) / time_scale)
    cuts = np.unique(np.clip(cuts, nodes[0], nodes[-1]))
    mid = (cuts[:-1] + cuts[1:]) / 2
    # At cosine=0, apparent wavelength remains lambda/coordinate_ratio.
    ok = window.accept(mid * time_scale, mid / (np.sqrt(2.) * coordinate_ratio),
                       np.zeros_like(mid))
    intervals = np.c_[cuts[:-1][ok], cuts[1:][ok]]

    coshi, coslo = np.cos(np.deg2rad(polar))
    full_width = coshi - coslo
    angular_cuts = [coslo, coshi]
    if len(intervals):
        s = intervals.ravel()[:, None] / (2 * coordinate_ratio * edges)
        boundary_cos = 1 - 2 * s**2
        angular_cuts.extend(boundary_cos[(s > 0) & (s < 1)
                            & (boundary_cos > coslo) & (boundary_cos < coshi)])
    angular_cuts = np.unique(angular_cuts)
    u, uw = leggauss(angular_order)
    e, ew = leggauss(energy_order)
    output = [[], [], [], []]
    for ca, cb in zip(angular_cuts[:-1], angular_cuts[1:]):
        cosines = (ca + cb) / 2 + (cb - ca) * u / 2
        angular_weights = (cb - ca) * uw / (2 * full_width)
        for cosine, aw in zip(cosines, angular_weights):
            d_to_wave = 2 * np.sqrt((1 - cosine) / 2) * coordinate_ratio
            for low, high in intervals:
                energy_cuts = np.unique(np.clip(np.r_[low, high, nodes, breaks,
                    edges * d_to_wave], low, high))
                lo, hi = energy_cuts[:-1], energy_cuts[1:]
                wave = ((lo + hi)[:, None] / 2 + (hi - lo)[:, None] * e / 2).ravel()
                weights = ((hi - lo)[:, None] * ew / 2).ravel() * aw
                d, time = apparent_d(wave, wave, cosine, l1_m=l1_m, l2_m=l2_m,
                    difc_ratio=difc_ratio, clock_model=clock_model)
                index = np.searchsorted(edges, d, side='right') - 1
                accepted = (index >= 0) & (index < len(edges) - 1)
                accepted &= window.accept(time, d, cosine)
                output[0].append(wave[accepted])
                output[1].append(np.full(accepted.sum(), cosine))
                output[2].append(index[accepted])
                output[3].append(weights[accepted])
    arrays = [np.concatenate(v) if v else np.empty(0, dtype=int if i == 2 else float)
              for i, v in enumerate(output)]
    for a in arrays:
        a.setflags(write=False)
    return ElasticCellQuadrature(*arrays)
