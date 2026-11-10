"""Finite radial-blade removal for explicit specimen-to-detector rays.

Geometry and the rotation angle for every ray are caller-supplied. This does
not infer POWGEN blade dimensions, a drive waveform, an oscillation period or
the cause of a measured count modulation. Rotation is frozen during each ray
segment. Blade-generated scattered/background neutrons are not generated.

Coordinates are cm, with the collimator rotating about the vertical z axis.
Each blade is a finite rectangular prism spanning radial coordinate Rin..Rout,
transverse thickness w and height H. Origins must lie inside the inner bore.
Exact box intersections follow a conservative angular candidate search; only
the search is padded for roundoff. Rays and material thickness are not widened.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np

VERSION = 'finite-radial-blade-removal/1'


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _provenance(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError('Explicit bounded collimator/material provenance is required')


@dataclass(frozen=True)
class CollimatorTransmission:
    transmission: np.ndarray
    metadata: dict


@dataclass(frozen=True)
class RadialBladePaths:
    length_cm: np.ndarray
    crossed_blades: np.ndarray
    metadata: dict

    def transmission(self, removal_cm_inv, *, provenance):
        _provenance(provenance)
        raw = np.asarray(removal_cm_inv)
        if raw.dtype.kind not in 'iuf':
            raise ValueError('Finite nonnegative blade removal coefficients in cm^-1 are required')
        mu = np.broadcast_to(raw, self.length_cm.shape).astype(float)
        if not np.isfinite(mu).all() or np.any(mu < 0):
            raise ValueError('Finite nonnegative blade removal coefficients in cm^-1 are required')
        # An optical depth above float range has exactly zero representable
        # transmission. No positive transmission/count floor is inserted.
        with np.errstate(over='ignore', under='ignore'):
            weight = np.exp(-mu * self.length_cm)
        metadata = {**self.metadata, 'removal_provenance': provenance,
            'removal_cm_inv_sha256': hashlib.sha256(mu.astype('<f8').tobytes()).hexdigest(),
            'absorption_and_scattering_out_of_ray_only': True,
            'blade_generated_scattered_neutrons_not_simulated': True,
            'no_positive_transmission_floor_or_window_renormalization': True}
        return CollimatorTransmission(_readonly(weight), metadata)


@dataclass(frozen=True)
class RadialBladeArray:
    inner_radius_cm: float
    outer_radius_cm: float
    height_cm: float
    thickness_cm: float
    blades: int
    provenance: str
    center_cm: tuple = (0., 0., 0.)
    blade_angles_rad: tuple | None = None

    def __post_init__(self):
        values = [self.inner_radius_cm, self.outer_radius_cm, self.height_cm, self.thickness_cm]
        if (any(isinstance(x, (bool, np.bool_)) or not np.isscalar(x) or np.asarray(x).dtype.kind not in 'iuf'
                or not math.isfinite(float(x)) or not 0 < float(x) <= 1e6 for x in values)
                or self.outer_radius_cm <= self.inner_radius_cm
                or type(self.blades) is not int or not 1 <= self.blades <= 4096):
            raise ValueError('Finite positive radial blade dimensions and 1..4096 blades are required')
        angles = (np.arange(self.blades)*2*np.pi/self.blades if self.blade_angles_rad is None
                  else np.asarray(self.blade_angles_rad))
        if (angles.shape != (self.blades,) or angles.dtype.kind not in 'iuf'
                or not np.isfinite(angles).all() or np.any(angles<0) or np.any(angles>=2*np.pi)):
            raise ValueError('Matching distinct blade angles in [0,2pi) are required')
        angles = np.sort(angles.astype(float))
        gap = float(np.diff(np.r_[angles, angles[0]+2*np.pi]).min())
        # A conservative non-overlap condition and an explicit finite precision
        # domain; neither replaces or broadens a supplied physical thickness.
        if (gap<=0 or self.thickness_cm >= 2*self.inner_radius_cm
                or (self.blades>1 and self.thickness_cm >= 2*self.inner_radius_cm*math.sin(gap/2))
                or self.thickness_cm < 256*np.finfo(float).eps*self.outer_radius_cm):
            raise ValueError('Blade thickness must be resolved and radial prisms must not overlap')
        center = np.asarray(self.center_cm)
        if (center.shape != (3,) or center.dtype.kind not in 'iuf'
                or not np.isfinite(center).all() or np.max(abs(center)) > 1e8):
            raise ValueError('An explicit bounded finite collimator center is required')
        _provenance(self.provenance)
        for name in ['inner_radius_cm', 'outer_radius_cm', 'height_cm', 'thickness_cm']:
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, 'center_cm', tuple(float(v) for v in center))
        object.__setattr__(self, 'blade_angles_rad', _readonly(angles))

    def _metadata(self):
        parameters = dict(inner_radius_cm=self.inner_radius_cm, outer_radius_cm=self.outer_radius_cm,
            height_cm=self.height_cm, thickness_cm=self.thickness_cm, blades=self.blades, center_cm=self.center_cm,
            blade_angles_rad=self.blade_angles_rad.tolist())
        return dict(version=VERSION, parameters=parameters,
            geometry_sha256=hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest(),
            provenance=self.provenance, geometry_units='cm', angle_units='rad',
            finite_rectangular_radial_prisms_not_zero_width_planes=True,
            explicit_per_ray_rotation_frozen_during_segment=True,
            drive_waveform_period_and_physical_motion_not_inferred=True,
            actual_POWGEN_geometry_or_noise_cause_not_established=True,
            physical_metrology_and_material_removal_not_qualified=True,
            experimental_accuracy_certified=False)

    def path_lengths(self, origins_cm, directions, rotation_rad, *, maximum_distance_cm=np.inf,
                     max_bytes=256*1024**2):
        origin, rays = np.asarray(origins_cm), np.asarray(directions)
        if (origin.ndim != 2 or origin.shape[1] != 3 or rays.shape != origin.shape
                or not 1 <= len(origin) <= 2**20
                or any(a.dtype.kind not in 'iuf' or not np.isfinite(a).all() for a in [origin, rays])
                or np.max(abs(origin)) > 1e8
                or np.max(abs(np.linalg.norm(rays, axis=1)-1)) > 1e-12):
            raise ValueError('Bounded finite ray origins and matching unit directions are required')
        n = len(origin)
        if type(max_bytes) is not int or max_bytes < 1024 or 512*n+128*self.blades > max_bytes:
            raise ValueError('Radial blade rays exceed the explicit working-memory budget')
        p = origin.astype(float)-np.asarray(self.center_cm)
        d = rays.astype(float)
        radial2 = np.sum(p[:, :2]**2, axis=1)
        if np.any(radial2 >= self.inner_radius_cm**2):
            raise ValueError('Radial-blade ray origins must lie strictly inside the inner bore')
        angle_raw = np.asarray(rotation_rad)
        distance_raw = np.asarray(maximum_distance_cm)
        if angle_raw.dtype.kind not in 'iuf' or distance_raw.dtype.kind not in 'iuf':
            raise ValueError('Explicit finite rotation angles and nonnegative segment limits are required')
        angle = np.broadcast_to(angle_raw, (n,)).astype(float)
        distance = np.broadcast_to(distance_raw, (n,)).astype(float)
        if (not np.isfinite(angle).all() or np.max(abs(angle)) > 1e6
                or np.any(np.isnan(distance)) or np.any(distance < 0)):
            raise ValueError('Explicit finite rotation angles and nonnegative segment limits are required')
        angle = np.remainder(angle, 2*np.pi)
        # Solve in horizontal path length with a unit horizontal direction.
        # This avoids squaring a nearly vertical direction into underflow.
        horizontal = np.hypot(d[:, 0], d[:, 1])
        u = np.divide(d[:, :2], horizontal[:, None], out=np.zeros((n, 2)), where=horizontal[:, None] > 0)
        dot = np.sum(p[:, :2]*u, axis=1)

        def circle(radius):
            root = np.sqrt(dot**2+radius**2-radial2)
            outward = dot >= 0
            result = -dot+root
            result[outward] = (radius**2-radial2[outward])/(dot[outward]+root[outward])
            return result

        inner_s = circle(self.inner_radius_cm)
        outer_s = circle(np.hypot(self.outer_radius_cm, self.thickness_cm/2))
        inner_xy = p[:, :2]+inner_s[:, None]*u
        outer_xy = p[:, :2]+outer_s[:, None]*u
        entry_angle = np.arctan2(inner_xy[:, 1], inner_xy[:, 0])
        cross = inner_xy[:, 0]*outer_xy[:, 1]-inner_xy[:, 1]*outer_xy[:, 0]
        delta = np.arctan2(cross, np.sum(inner_xy*outer_xy, axis=1))
        margin = math.asin(self.thickness_cm/(2*self.inner_radius_cm))
        relative = np.remainder(entry_angle-angle, 2*np.pi)
        low = relative+np.minimum(delta, 0)-margin
        high = relative+np.maximum(delta, 0)+margin
        padding = 128*np.finfo(float).eps*(1+abs(low)+abs(high))
        extended = np.r_[self.blade_angles_rad-2*np.pi, self.blade_angles_rad, self.blade_angles_rad+2*np.pi]
        first = np.searchsorted(extended, low-padding, side='left')
        last = np.searchsorted(extended, high+padding, side='right')-1
        with np.errstate(over='ignore'):
            entry_t = np.divide(inner_s, horizontal, out=np.full(n, np.inf), where=horizontal>0)
        # A global height slab discards rays that cannot reach the radial body
        # before leaving its finite vertical extent. The exact box test follows.
        dz = d[:, 2]
        with np.errstate(over='ignore'):
            z0 = np.divide(-self.height_cm/2-p[:, 2], dz, out=np.zeros(n), where=dz!=0)
            z1 = np.divide(self.height_cm/2-p[:, 2], dz, out=np.zeros(n), where=dz!=0)
        height_inside = (p[:, 2] >= -self.height_cm/2) & (p[:, 2] < self.height_cm/2)
        height_exit = np.where(dz!=0, np.maximum(z0, z1), np.where(height_inside, np.inf, -np.inf))
        eligible = (horizontal>0) & (entry_t<np.minimum(distance, height_exit))
        count = np.where(eligible, np.maximum(last-first+1, 0), 0)
        if np.any(count > self.blades):
            raise ValueError('Angular candidate interval exceeds the non-overlapping geometry domain')
        candidates = int(count.sum())
        if 512*n+128*self.blades+768*candidates > max_bytes:
            raise ValueError('Radial blade candidate intersections exceed the working-memory budget')
        total = np.zeros(n)
        crossed = np.zeros(n, np.int64)
        if candidates:
            owner = np.repeat(np.arange(n), count)
            starts = np.repeat(np.cumsum(count)-count, count)
            blade = first[owner]+np.arange(candidates)-starts
            phi = np.remainder(angle[owner]+extended[blade], 2*np.pi)
            co, si = np.cos(phi), np.sin(phi)
            pp, dd = p[owner], d[owner]
            local_p = [pp[:, 0]*co+pp[:, 1]*si, -pp[:, 0]*si+pp[:, 1]*co, pp[:, 2]]
            local_d = [dd[:, 0]*co+dd[:, 1]*si, -dd[:, 0]*si+dd[:, 1]*co, dd[:, 2]]
            lower, upper = np.zeros(candidates), distance[owner].copy()
            bounds = [(self.inner_radius_cm, self.outer_radius_cm),
                      (-self.thickness_cm/2, self.thickness_cm/2), (-self.height_cm/2, self.height_cm/2)]
            for position, direction, (lo, hi) in zip(local_p, local_d, bounds):
                nonzero = direction != 0
                with np.errstate(over='ignore'):
                    a = np.divide(lo-position, direction, out=np.zeros(candidates), where=nonzero)
                    b = np.divide(hi-position, direction, out=np.zeros(candidates), where=nonzero)
                inside = (position>=lo) & (position<hi)
                near = np.where(nonzero, np.minimum(a, b), np.where(inside, -np.inf, np.inf))
                far = np.where(nonzero, np.maximum(a, b), np.where(inside, np.inf, -np.inf))
                lower = np.maximum(lower, near)
                upper = np.minimum(upper, far)
            length = np.maximum(upper-lower, 0)
            assert np.isfinite(length).all()
            total = np.bincount(owner, weights=length, minlength=n)
            crossed = np.bincount(owner[length>0], minlength=n)
        metadata = self._metadata()
        metadata.update(rays=n, angular_candidates=candidates, candidate_search_padding_only=True,
            explicit_nonuniform_angular_positions_and_missing_sectors_supported=True,
            exact_finite_box_intersections=True, half_open_parallel_faces=True,
            every_origin_strictly_inside_bore=True, no_ray_motion_or_width_adjustment=True)
        return RadialBladePaths(_readonly(total), _readonly(crossed), metadata)
