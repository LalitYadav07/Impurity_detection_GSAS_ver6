"""Finite-ray intersection with nominal rectangular collecting surfaces.

Index coplanar rectangular pixel grids by panel. A conservative enclosing-sphere
test removes impossible panels; exact plane and rectangle tests choose the
nearest collecting surface. No solid-angle factor, efficiency, mask or detector
depth is inferred. Apply those separately to the returned geometric identities.
"""
from dataclasses import dataclass
import numpy as np
from .pixel_geometry import PixelGeometry

VERSION = 'rectangular-pixel-ray-intersection/1'


def _dot_rows(array, vector):
    return np.einsum('ij,j->i', array, vector)


def _cluster(values, tolerance):
    order = np.argsort(values)
    groups = np.cumsum(np.r_[True, np.diff(values[order]) > tolerance])-1
    centers = np.bincount(groups, weights=values[order])/np.bincount(groups)
    inverse = np.empty(len(values), int)
    inverse[order] = groups
    if np.max(abs(values-centers[inverse])) > tolerance:
        raise ValueError('Pixel centers do not form a coplanar Cartesian grid')
    return centers, inverse


def _nearest(values, points):
    upper = np.clip(np.searchsorted(values, points), 0, len(values)-1)
    lower = np.maximum(0, upper-1)
    # At a touching edge choose the upper center: rectangles are half open.
    return np.where(abs(points-values[lower]) < abs(points-values[upper]), lower, upper)


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PixelRayHits:
    pixel_index: np.ndarray
    detector_id: np.ndarray
    distance_cm: np.ndarray
    local_face_coordinates: np.ndarray


class RectangularPixelRayIndex:
    """Compile geometric panels; source arrays are copied into the index.

    Pixel faces in each named panel must share orientation, dimensions and a
    plane, within declared floating roundoff tolerance. The Cartesian grid may
    have gaps/missing entries but may not overlap. Rectangle coordinates are
    half open [-0.5,0.5) along each supplied edge. Exact equal-distance ties
    between panels choose the lower detector ID. Faces collect from both sides;
    material transmission, shielding and front/back response are external.
    """
    def __init__(self, geometry, *, max_bytes=128*1024**2):
        if not isinstance(geometry, PixelGeometry) or type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('Explicit pixel geometry and memory budget are required')
        ids = np.asarray(geometry.detector_ids)
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or not 1 <= len(ids) <= 2**20
                or np.any(ids < 0) or np.any(ids > np.iinfo(np.int64).max) or len(np.unique(ids)) != len(ids)):
            raise ValueError('Bounded unique nonnegative detector identities are required')
        if 512*len(ids) > max_bytes:
            raise ValueError('Pixel index exceeds its declared memory budget')
        center, u, v = [np.asarray(x, float) for x in (geometry.face_centers_cm, geometry.u_cm, geometry.v_cm)]
        if any(x.shape != (len(ids), 3) or not np.isfinite(x).all() or np.max(abs(x)) > 1e8 for x in [center, u, v]):
            raise ValueError('Finite bounded pixel collecting faces are required')
        names = np.asarray(geometry.bank_names)
        if (names.shape != ids.shape or names.dtype.kind not in 'US' or names.dtype.itemsize > 256
                or len(np.unique(names)) > 1024):
            raise ValueError('Bounded explicit panel identities are required')
        self.detector_ids = _readonly(ids.astype(np.int64))
        self.panels = []
        tolerances = []
        for name in np.unique(names):
            selected = np.flatnonzero(names == name)
            c, uu, vv = center[selected], u[selected], v[selected]
            lu, lv = np.linalg.norm(uu[0]), np.linalg.norm(vv[0])
            if min(lu, lv) <= 0:
                raise ValueError('Pixel edges must be nonzero')
            a, b = uu[0]/lu, vv[0]/lv
            if abs(a@b) > 1e-12 or np.max(abs(uu-uu[0])) > 1e-12*lu or np.max(abs(vv-vv[0])) > 1e-12*lv:
                raise ValueError('A panel requires common orthogonal pixel edges')
            normal = np.cross(a, b)
            origin = c.mean(axis=0)
            tolerance = 128*np.finfo(float).eps*max(1., float(np.max(abs(c))), lu, lv)
            tolerances.append(tolerance)
            relative = c-origin
            if np.max(abs(_dot_rows(relative, normal))) > tolerance:
                raise ValueError('Pixel collecting faces must be coplanar within a panel')
            uc, ui = _cluster(_dot_rows(relative, a), tolerance)
            vc, vi = _cluster(_dot_rows(relative, b), tolerance)
            if np.any(np.diff(uc) < lu-tolerance) or np.any(np.diff(vc) < lv-tolerance):
                raise ValueError('Collecting rectangles may not overlap within a panel')
            if len(uc)*len(vc) > 4*len(ids):
                raise ValueError('Sparse panel grid exceeds its explicit allocation budget')
            grid = np.full((len(uc), len(vc)), -1, dtype=np.int64)
            if len(np.unique(ui*len(vc)+vi)) != len(selected):
                raise ValueError('A panel has duplicate collecting grid cells')
            grid[ui, vi] = selected
            radius = float(np.linalg.norm(relative, axis=1).max()+.5*np.hypot(lu, lv))
            self.panels.append(dict(origin=_readonly(origin), a=_readonly(a), b=_readonly(b), normal=_readonly(normal),
                u_centers=_readonly(uc), v_centers=_readonly(vc), grid=_readonly(grid),
                u_length=float(lu), v_length=float(lv), enclosing_radius_cm=radius))
        self.metadata = dict(version=VERSION, pixels=len(ids), panels=len(self.panels),
            construction_roundoff_tolerance_cm=max(tolerances),
            input_geometry_source=geometry.metadata.get('geometry_source', 'Explicit supplied collecting surfaces'),
            first_geometric_surface_only=True, masking_efficiency_depth_and_real_hardware_not_inferred=True)

    def intersect(self, origins_cm, directions, *, max_bytes=256*1024**2):
        """Nearest positive ray hit; misses have index/ID -1 and zero distance.

        Returned local coordinates are dimensionless fractions of the pixel
        edge vectors, in [-0.5,0.5). Ray directions must already be unit vectors.
        No input ray is moved or widened to make it hit a detector.
        """
        origins, rays = np.asarray(origins_cm, float), np.asarray(directions, float)
        if (origins.ndim != 2 or origins.shape[1] != 3 or rays.shape != origins.shape
                or not 1 <= len(origins) <= 2**20 or not np.isfinite(origins).all() or not np.isfinite(rays).all()
                or np.max(abs(origins)) > 1e8 or np.max(abs(np.linalg.norm(rays, axis=1)-1)) > 1e-12
                or type(max_bytes) is not int or 768*len(origins) > max_bytes):
            raise ValueError('Bounded finite rays, unit directions and adequate memory are required')
        n = len(origins)
        nearest = np.full(n, np.inf)
        pixel = np.full(n, -1, dtype=np.int64)
        coordinates = np.zeros((n, 2))
        origin_radius = float(np.linalg.norm(origins, axis=1).max())
        for panel in self.panels:
            center = panel['origin']
            length = float(np.linalg.norm(center))
            radius = panel['enclosing_radius_cm']+origin_radius
            if length > radius:
                padded = min(1., radius/length+128*np.finfo(float).eps)
                cutoff = np.sqrt(max(0., (1-padded)*(1+padded)))-128*np.finfo(float).eps
                candidate = np.flatnonzero(_dot_rows(rays, center/length) >= cutoff)
            else:
                candidate = np.arange(n)
            if not len(candidate):
                continue
            normal = panel['normal']
            denominator = _dot_rows(rays[candidate], normal)
            numerator = _dot_rows(center-origins[candidate], normal)
            distance = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
            valid = (distance > 0) & np.isfinite(distance) & (distance <= nearest[candidate])
            candidate, distance = candidate[valid], distance[valid]
            if not len(candidate):
                continue
            point = origins[candidate]+distance[:, None]*rays[candidate]-center
            cu, cv = _dot_rows(point, panel['a']), _dot_rows(point, panel['b'])
            ui, vi = _nearest(panel['u_centers'], cu), _nearest(panel['v_centers'], cv)
            du = (cu-panel['u_centers'][ui])/panel['u_length']
            dv = (cv-panel['v_centers'][vi])/panel['v_length']
            target = panel['grid'][ui, vi]
            valid = (target >= 0) & (du >= -.5) & (du < .5) & (dv >= -.5) & (dv < .5)
            candidate, distance, target, du, dv = [x[valid] for x in (candidate, distance, target, du, dv)]
            if not len(candidate):
                continue
            old = pixel[candidate]
            better = ((distance < nearest[candidate]) | (old < 0) |
                (self.detector_ids[target] < self.detector_ids[np.maximum(old, 0)]))
            candidate, distance, target, du, dv = [x[better] for x in (candidate, distance, target, du, dv)]
            nearest[candidate], pixel[candidate] = distance, target
            coordinates[candidate] = np.column_stack((du, dv))
        hit = pixel >= 0
        ids = np.full(n, -1, dtype=np.int64)
        ids[hit] = self.detector_ids[pixel[hit]]
        nearest[~hit] = 0.
        return PixelRayHits(*map(_readonly, (pixel, ids, nearest, coordinates)))
