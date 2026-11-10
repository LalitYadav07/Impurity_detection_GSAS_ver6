"""Map complete collision histories into calibrated, unweighted count cells.

Weighted numerical trajectories remain weighted here. Experimental integer
acquisition, source/area integration and detector efficiency are separate.
The raw-time clock, calibration, mask, cuts and half-open histogram are explicit.
"""
from dataclasses import dataclass
import numpy as np
from .collision_transport import EscapeBatch
from .pixel_rays import PixelRayHits
from .energy_response import ReductionWindow, TOF_US_PER_M_A


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CalibratedPixelObservation:
    detector_ids: np.ndarray
    difc_us_per_a: np.ndarray
    difa_us_per_a2: np.ndarray
    tzero_us: np.ndarray
    cos_two_theta: np.ndarray
    l2_m: np.ndarray
    group_index: np.ndarray
    observed_groups: int
    provenance: str

    def __post_init__(self):
        ids, groups = np.asarray(self.detector_ids), np.asarray(self.group_index)
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or not 1 <= len(ids) <= 2**20
                or np.any(ids < 0) or np.any(ids > np.iinfo(np.int64).max) or len(np.unique(ids)) != len(ids)
                or groups.shape != ids.shape or groups.dtype.kind not in 'iu'
                or type(self.observed_groups) is not int or not 1 <= self.observed_groups <= 4096
                or np.any(groups < -1) or np.any(groups >= self.observed_groups)
                or not isinstance(self.provenance, str) or not self.provenance.strip()):
            raise ValueError('Explicit detector identities, masks/groups and calibration provenance are required')
        arrays = [np.asarray(getattr(self, key), float) for key in
            ['difc_us_per_a', 'difa_us_per_a2', 'tzero_us', 'cos_two_theta', 'l2_m']]
        if any(a.shape != ids.shape or not np.isfinite(a).all() for a in arrays):
            raise ValueError('Finite matching per-pixel calibration arrays are required')
        active = groups >= 0
        b, _, _, cosine, l2 = arrays
        if np.any(b[active] <= 0) or np.any(l2[active] <= 0) or np.any((cosine[active] < -1) | (cosine[active] >= 1)):
            raise ValueError('Active pixels require positive calibration/path and physical nonforward angles')
        object.__setattr__(self, 'detector_ids', _readonly(ids.astype(np.int64)))
        object.__setattr__(self, 'group_index', _readonly(groups.astype(np.int64)))
        for key, value in zip(['difc_us_per_a', 'difa_us_per_a2', 'tzero_us', 'cos_two_theta', 'l2_m'], arrays):
            object.__setattr__(self, key, _readonly(value))


@dataclass(frozen=True)
class ObservedTrajectoryBatch:
    row_index: np.ndarray
    primary_index: np.ndarray
    pixel_index: np.ndarray
    incident_wavelength_a: np.ndarray
    outgoing_wavelength_a: np.ndarray
    raw_time_us: np.ndarray
    apparent_d_a: np.ndarray
    weight: np.ndarray
    ledger: dict


def observe_escape_batch(batch, hits, calibration, d_edges_a, *, l1_m,
                         source_plane_x_cm, window, clock_model='physical',
                         emission_delay_us=0., tof_us_per_m_a=TOF_US_PER_M_A,
                         source_plane_arrival_us=None, max_bytes=256*1024**2):
    """Project matching geometric hits and complete source-plane flight times.

    Both the history integrator and this function must use the explicitly
    supplied TOF conversion constant. The physical raw time is incident flight
    to the source plane plus the saved history time plus flight to the collecting
    face. The alternative calibrated_elastic_mean clock adds the same declared
    per-pixel elastic-reference offset as the point-response hypothesis, while
    retaining all internal trajectory times. It is a hypothesis, not a physical
    emission-time measurement. Source emission delay is an additional explicit
    scalar or per-history array.

    Alternatively, source_plane_arrival_us is the complete time from the pulse
    trigger to the source plane, including emission and all upstream flight.
    It is scalar or aligned with this escape batch. Map primary phase-space
    arrays through batch.primary_index before calling. An additional nonzero
    emission_delay_us is rejected with this complete upstream time.

    The fixed physical detector angle defines the apparent wavelength cut.
    Histogram cells are half open, including rejection at the final upper edge,
    matching the frozen raw-histogram observation contract. Invalid calibration
    branches fail explicitly rather than being silently removed.
    The window argument is required; supply None explicitly for no cuts.
    """
    if (not isinstance(batch, EscapeBatch) or not isinstance(hits, PixelRayHits)
            or not isinstance(calibration, CalibratedPixelObservation)
            or (window is not None and not isinstance(window, ReductionWindow))
            or clock_model not in ['physical', 'calibrated_elastic_mean']
            or not np.isfinite(l1_m) or l1_m <= 0 or not np.isfinite(source_plane_x_cm)
            or l1_m+source_plane_x_cm*.01 <= 0
            or not np.isfinite(tof_us_per_m_a) or tof_us_per_m_a <= 0):
        raise ValueError('Explicit trajectory, calibration, clock and physical flight paths are required')
    n = len(batch.weight)
    if type(max_bytes) is not int or not 1 <= n <= 2**20 or 768*n > max_bytes:
        raise ValueError('Observation batch exceeds its declared memory budget')
    for a in [batch.incident_wavelength_a, batch.wavelength_a, batch.flight_time_from_source_plane_us, batch.weight]:
        if a.shape != (n,) or not np.isfinite(a).all():
            raise ValueError('Finite matching trajectory arrays are required')
    if (np.any(batch.incident_wavelength_a <= 0) or np.any(batch.wavelength_a <= 0)
            or np.any(batch.weight < 0) or np.any(batch.flight_time_from_source_plane_us < 0)
            or batch.primary_index.shape != (n,) or batch.primary_index.dtype.kind not in 'iu'
            or np.any(batch.primary_index < 0)):
        raise ValueError('Physical wavelengths, times, weights and primary identities are required')
    index = np.asarray(hits.pixel_index)
    if (index.shape != (n,) or index.dtype.kind not in 'iu' or np.any(index < -1)
            or np.any(index >= len(calibration.detector_ids)) or hits.distance_cm.shape != (n,)
            or not np.isfinite(hits.distance_cm).all() or np.any(hits.distance_cm < 0)
            or hits.detector_id.shape != (n,)):
        raise ValueError('Geometric hit identities and distances must match the trajectory batch')
    geometric = index >= 0
    if (np.any(hits.distance_cm[geometric] <= 0)
            or not np.array_equal(hits.detector_id[geometric], calibration.detector_ids[index[geometric]])):
        raise ValueError('Geometric hit identities do not match the calibration ordering')
    edges = np.asarray(d_edges_a, float)
    if (edges.ndim != 1 or not 2 <= len(edges) <= 100001 or not np.isfinite(edges).all()
            or edges[0] <= 0 or np.any(np.diff(edges) <= 0)):
        raise ValueError('Finite increasing positive diffraction bin edges are required')
    delay = np.broadcast_to(np.asarray(emission_delay_us, float), (n,))
    if not np.isfinite(delay).all():
        raise ValueError('An explicit finite emission delay is required')
    arrival=None
    if source_plane_arrival_us is not None:
        arrival=np.broadcast_to(np.asarray(source_plane_arrival_us,float),(n,))
        if not np.isfinite(arrival).all() or np.any(arrival<0):
            raise ValueError('Complete source-plane arrival times must be finite and nonnegative')
        if np.any(delay!=0):
            raise ValueError('Complete source-plane arrival time already includes emission delay')
    active = geometric.copy()
    active[geometric] &= calibration.group_index[index[geometric]] >= 0
    positions = np.flatnonzero(active)
    px = index[positions]
    lin, lout = batch.incident_wavelength_a[positions], batch.wavelength_a[positions]
    k = float(tof_us_per_m_a)
    upstream=k*(l1_m+source_plane_x_cm*.01)*lin if arrival is None else arrival[positions]
    time = (upstream+batch.flight_time_from_source_plane_us[positions]
        +k*.01*hits.distance_cm[positions]*lout+delay[positions])
    b, a, zero, cosine = [getattr(calibration, key)[px] for key in
        ['difc_us_per_a', 'difa_us_per_a2', 'tzero_us', 'cos_two_theta']]
    if clock_model == 'calibrated_elastic_mean':
        d0 = lin/np.sqrt(2*(1-cosine))
        if np.any(b+2*a*d0 <= 0):
            raise ValueError('Incident wavelength is outside the monotonic calibration branch')
        time += b*d0+a*d0*d0+zero-k*(l1_m+calibration.l2_m[px])*lin
    offset = time-zero
    disc = b*b+4*a*offset
    if np.any(time <= 0) or np.any(offset <= 0) or np.any(disc <= 0):
        raise ValueError('Trajectory is outside the positive monotonic calibration branch')
    d = 2*offset/(b+np.sqrt(disc))
    if not np.isfinite(d).all() or np.any(d <= 0):
        raise ValueError('Invalid calibrated trajectory coordinate')
    accepted = np.ones(len(d), dtype=bool) if window is None else window.accept(time, d, cosine)
    cell = np.searchsorted(edges, d, side='right')-1
    inside = (cell >= 0) & (cell < len(edges)-1)
    retained = accepted & inside
    selected = positions[retained]
    weights = batch.weight
    ledger = dict(input_history_weight=float(weights.sum()),
        no_geometric_surface_weight=float(weights[~geometric].sum()),
        masked_first_surface_weight=float(weights[geometric & ~active].sum()),
        observation_window_rejected_weight=float(weights[positions[~accepted]].sum()),
        histogram_rejected_weight=float(weights[positions[accepted & ~inside]].sum()),
        retained_weight=float(weights[selected].sum()), input_histories=n, retained_histories=len(selected),
        scattering_order=int(batch.scattering_order), clock_model=clock_model,
        calibration_provenance=calibration.provenance, no_window_or_mask_renormalization=True)
    ledger['upstream_time_model']='nominal_parallel_path_plus_emission' if arrival is None else 'explicit_complete_source_plane_arrival'
    total = sum(ledger[key] for key in ['no_geometric_surface_weight', 'masked_first_surface_weight',
        'observation_window_rejected_weight', 'histogram_rejected_weight', 'retained_weight'])
    if abs(total-ledger['input_history_weight']) > 2e-12*max(1., ledger['input_history_weight']):
        raise ValueError('Observation weight partition failed')
    row = calibration.group_index[px[retained]]*(len(edges)-1)+cell[retained]
    return ObservedTrajectoryBatch(*map(_readonly, (row, batch.primary_index[selected], index[selected],
        lin[retained], lout[retained], time[retained], d[retained], weights[selected])), ledger)
