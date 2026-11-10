"""Weighted multiple-collision transport through one homogeneous finite cylinder.

Coordinates are cm: the nominal incident beam is +x and z is vertical.
Cross sections and scattering draws are supplied by a traceable, isotropic
material provider. Every free flight splits into escape, absorption and a
forced scattering contribution. A collision cap leaves an explicit positive
remaining-weight bound. No finite-order or detector-window renormalization is
performed. These are numerical histories, not experimental Poisson events.

The callback receives one bounded batch per scattering order, with primary
identifiers retained for clustered numerical uncertainty calculations. It must
consume each batch without accumulating unbounded storage. Engine memory limits
cover engine arrays and one callback batch, not storage retained by user code.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np
from .energy_response import TOF_US_PER_M_A
from .transport import SolidCylinder, CylindricalRegion, ray_length_cm

VERSION = 'finite-cylinder-multiple-collision/2'


@dataclass(frozen=True)
class IsotropicCollisionLaw:
    """Macroscopic removal [cm^-1] and conditional scattering law.

    cross_sections(wavelength_a) returns (scattering, absorption). scatter takes
    (wavelength_a, numpy_generator) and returns (final_wavelength_a, cos_angle).
    Azimuth is uniform and sampled by the transport engine. Neither function
    may change its input arrays or keep references to them.
    """
    cross_sections: Callable
    scatter: Callable
    provenance: str

    def __post_init__(self):
        if (not callable(self.cross_sections) or not callable(self.scatter)
                or not isinstance(self.provenance, str) or not self.provenance.strip()):
            raise ValueError('A traceable isotropic collision law is required')


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class EscapeBatch:
    primary_index: np.ndarray
    scattering_order: int
    position_cm: np.ndarray
    direction: np.ndarray
    incident_wavelength_a: np.ndarray
    wavelength_a: np.ndarray
    flight_time_from_source_plane_us: np.ndarray
    weight: np.ndarray


@dataclass(frozen=True)
class CollisionTransportResult:
    escaped_weight: np.ndarray
    absorbed_weight: np.ndarray
    missed_geometry_weight: np.ndarray
    remaining_scatter_weight_upper: np.ndarray
    metadata: dict


def _forward_entry_cm(solid, positions, directions):
    """Positive entry of forward rays into the intersection of disk and z slab."""
    p=positions-np.asarray(solid.center_cm)
    radial=np.sum(directions[:,:2]**2,axis=1)
    dot=np.sum(p[:,:2]*directions[:,:2],axis=1)
    c=np.sum(p[:,:2]**2,axis=1)-solid.radius_cm**2
    disc=dot*dot-radial*c
    root=np.sqrt(np.maximum(disc,0.))
    # Stable quadratic roots, including entry close to the front surface.
    q=-dot-np.copysign(root,dot)
    first=q/radial
    second=np.divide(c,q,out=np.zeros_like(q),where=q!=0)
    lower=np.minimum(first,second);upper=np.maximum(first,second)
    vertical=np.abs(directions[:,2])>1e-14
    dz=np.where(vertical,directions[:,2],1.)
    a=(-solid.height_cm/2-p[:,2])/dz
    b=(solid.height_cm/2-p[:,2])/dz
    lower=np.maximum(np.maximum(lower,0.),np.where(vertical,np.minimum(a,b),-np.inf))
    upper=np.minimum(upper,np.where(vertical,np.maximum(a,b),np.inf))
    hit=(disc>0)&(upper>lower)&(vertical|(abs(p[:,2])<solid.height_cm/2))
    return np.where(hit,lower,0.),hit


def trace_cylinder_neutrons(solid, source_positions_cm, incident_wavelength_a, law,
                            *, incident_directions=None, seed=0, maximum_collisions=64, on_escape=None,
                            tof_us_per_m_a=TOF_US_PER_M_A, max_bytes=512*1024**2):
    """Trace source rays starting on the cylinder front plane.

    Each primary starts with unit weight. Positions must have x=center_x-radius.
    Default directions are +x; explicit unit directions may be one vector or
    one per primary, with dx>=1e-8. Position/direction/wavelength correlations
    are retained. Source/beam area, wavelength density and experimental fluence are
    external factors; this function invents none of them. Missed rays are
    recorded separately and do not emit an escape batch. Flight times include
    vacuum from the source plane to entry and every internal segment at its
    current neutron wavelength. Detector flight and source emission come later.

    remaining_scatter_weight_upper bounds any further escape contribution from
    that primary, before external source/detection weighting. It is a
    truncation bound, not a Monte Carlo integration error or material-model
    uncertainty. Scattering order one retains removal on the exit path.
    """
    if (not isinstance(solid, SolidCylinder) or not isinstance(law, IsotropicCollisionLaw)
            or type(seed) is not int or not 0 <= seed < 2**32
            or type(maximum_collisions) is not int or not 0 <= maximum_collisions <= 256
            or (on_escape is not None and not callable(on_escape))
            or not np.isfinite(tof_us_per_m_a) or not 0 < tof_us_per_m_a <= 1e6
            or type(max_bytes) is not int or max_bytes < 1):
        raise ValueError('Finite cylinder, material, seed, collision and memory limits are required')
    initial = np.asarray(source_positions_cm, float)
    if (initial.ndim != 2 or initial.shape[1] != 3 or not 1 <= len(initial) <= 2**20
            or not np.isfinite(initial).all() or np.max(abs(initial)) > 1e8):
        raise ValueError('Bounded finite source points are required')
    # Includes live free-flight/scattering temporaries, geometry temporaries,
    # result arrays, and a copied immutable callback batch.
    required = (1536 if incident_directions is None else 1792)*len(initial)
    if required > max_bytes:
        raise ValueError('Collision transport exceeds its declared memory budget')
    plane = solid.center_cm[0]-solid.radius_cm
    if not np.all(abs(initial[:, 0]-plane) <= 1e-12*max(1., abs(plane), solid.radius_cm)):
        raise ValueError('Source points must lie on the cylinder front plane')
    wave0 = np.broadcast_to(np.asarray(incident_wavelength_a, float), (len(initial),)).copy()
    if not np.isfinite(wave0).all() or np.any((wave0 <= 0) | (wave0 > 1e6)):
        raise ValueError('Finite positive incident wavelengths are required')
    n = len(initial)
    incoming=None
    if incident_directions is not None:
        supplied=np.asarray(incident_directions,float)
        if (supplied.shape not in ((3,),initial.shape) or not np.isfinite(supplied).all() or
            not np.allclose(np.linalg.norm(supplied,axis=-1),1.,rtol=0,atol=1e-12)):
            raise ValueError('Explicit incident directions must be finite unit vectors')
        incoming=np.broadcast_to(supplied,initial.shape).copy()
        if np.any(incoming[:,0]<1e-8):
            raise ValueError('Incident rays must point forward with dx at least 1e-8')
    escaped, absorbed, missed, remaining = [np.zeros(n) for _ in range(4)]
    yy = initial[:, 1]-solid.center_cm[1]
    zz = initial[:, 2]-solid.center_cm[2]
    parallel=incoming is None or np.all(incoming==np.array([1.,0.,0.]))
    if parallel:
        hit = (abs(yy) < solid.radius_cm) & (abs(zz) < solid.height_cm/2)
    else:
        entry,hit=_forward_entry_cm(solid,initial,incoming)
    missed[~hit] = 1.
    ids = np.flatnonzero(hit)
    position = initial[ids].copy()
    wave = wave0[ids].copy()
    if parallel:
        # Preserve the exact established arithmetic/RNG path for parallel inputs.
        position[:, 0] = solid.center_cm[0]-np.sqrt(np.maximum(0., solid.radius_cm**2-yy[ids]**2))
        clock = (position[:, 0]-plane)*.01*tof_us_per_m_a*wave
        direction = np.zeros_like(position)
        direction[:, 0] = 1.
    else:
        direction=incoming[ids].copy()
        position+=entry[ids,None]*direction
        clock=entry[ids]*.01*tof_us_per_m_a*wave
    weight = np.ones(len(ids))
    rng = np.random.default_rng(seed)
    region = CylindricalRegion(solid)
    emitted, last_order = 0, 0
    for order in range(maximum_collisions+1):
        if not len(ids):
            break
        last_order = order
        ms, ma = [np.broadcast_to(np.asarray(v, float), wave.shape)
                  for v in law.cross_sections(wave)]
        total = ms+ma
        if (not np.isfinite(ms).all() or not np.isfinite(ma).all()
                or np.any(ms < 0) or np.any(ma < 0) or not np.isfinite(total).all()):
            raise ValueError('Material removal must be finite and nonnegative')
        distance = ray_length_cm(region, position, direction)
        tau = total*distance
        probability = -np.expm1(-tau)
        exit_weight = weight*np.exp(-tau)
        collision_weight = weight*probability
        scatter_fraction = np.divide(ms, total, out=np.zeros_like(ms), where=total > 0)
        absorption_fraction = np.divide(ma, total, out=np.zeros_like(ma), where=total > 0)
        absorbed[ids] += collision_weight*absorption_fraction
        escaped[ids] += exit_weight
        if on_escape is not None:
            on_escape(EscapeBatch(_readonly(ids), order,
                _readonly(position+distance[:, None]*direction), _readonly(direction),
                _readonly(wave0[ids]), _readonly(wave),
                _readonly(clock+distance*.01*tof_us_per_m_a*wave), _readonly(exit_weight)))
        emitted += len(ids)
        weight = collision_weight*scatter_fraction
        if order == maximum_collisions:
            remaining[ids] = weight
            break
        active = weight > 0
        ids, position, direction, wave, clock, weight, probability, total = [v[active] for v in
            (ids, position, direction, wave, clock, weight, probability, total)]
        if not len(ids):
            break
        flight = -np.log1p(-rng.random(len(ids))*probability)/total
        position += flight[:, None]*direction
        clock += flight*.01*tof_us_per_m_a*wave
        final, cosine = [np.asarray(v, float) for v in law.scatter(wave, rng)]
        if (final.shape != wave.shape or cosine.shape != wave.shape
                or not np.isfinite(final).all() or not np.isfinite(cosine).all()
                or np.any((final <= 0) | (final > 1e6)) or np.any(abs(cosine) > 1)):
            raise ValueError('Scattering draws must have finite matching energies and physical angles')
        # Choose the coordinate axis least parallel to each current direction.
        # This remains stable at forward/backward and vertical directions.
        axis = np.eye(3)[np.argmin(abs(direction), axis=1)]
        first = np.cross(direction, axis)
        first /= np.linalg.norm(first, axis=1)[:, None]
        second = np.cross(direction, first)
        phi = 2*np.pi*rng.random(len(ids))
        sine = np.sqrt(np.maximum(0., (1-cosine)*(1+cosine)))
        direction = (cosine[:, None]*direction+sine[:, None]*
            (np.cos(phi)[:, None]*first+np.sin(phi)[:, None]*second))
        norms = np.linalg.norm(direction, axis=1)
        if not np.isfinite(norms).all() or np.max(abs(norms-1)) > 2e-12:
            raise ValueError('Transport rotation lost its unit-direction invariant')
        direction /= norms[:, None]
        wave = final.copy()
    closure = escaped+absorbed+missed+remaining
    error = float(np.max(abs(closure-1)))
    if error > 5e-12:
        raise ValueError('Per-primary escape/absorption/truncation conservation failed')
    return CollisionTransportResult(*map(_readonly, (escaped, absorbed, missed, remaining)), dict(
        version=VERSION, material_law=law.provenance, source_primaries=n, intersecting_primaries=int(hit.sum()),
        seed=seed, maximum_collisions=maximum_collisions, last_order=last_order,
        emitted_history_rows=emitted, conservative_working_memory_estimate_bytes=required,
        maximum_per_primary_conservation_error=error, source_plane_x_cm=plane,
        tof_us_per_m_a=float(tof_us_per_m_a),
        incident_direction_model='parallel_x' if parallel else 'explicit_correlated_forward_rays',
        remaining_weight_is_positive_truncation_bound=True,
        floating_point_error_bound_not_formally_certified=True,
        source_area_fluence_and_detector_efficiency_not_supplied=True,
        numerical_histories_are_not_Poisson_count_events=True,
        material_geometry_and_experimental_accuracy_not_certified=True))
