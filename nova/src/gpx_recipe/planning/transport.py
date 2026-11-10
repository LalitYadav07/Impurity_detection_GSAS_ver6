"""Finite 3D cylinder/sample-environment single-scattering attenuation.

Coordinates are cm: +x is the parallel incident beam, z is vertical. The
rectangular beam footprint is uniform. Solid or annular powder is sampled
uniformly inside the illuminated volume; the exact illuminated volume fraction
is included once in the returned weight. Other cylindrical regions can include
voids/endcaps and offsets. These are uncollided incident/exit path weights, not
multiple-scattering, environment-scattering or physical counting-noise draws.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import qmc

VERSION = 'powgen-finite-cylinder-single-scattering/4'


def _positive(value, name):
    if not np.isfinite(value) or not 0 < value <= 1e6:
        raise ValueError(name + ' must be finite and positive')


@dataclass(frozen=True)
class SolidCylinder:
    radius_cm: float
    height_cm: float
    center_cm: tuple = (0., 0., 0.)

    def __post_init__(self):
        _positive(self.radius_cm, 'Cylinder radius')
        _positive(self.height_cm, 'Cylinder height')
        if (np.shape(self.center_cm) != (3,) or not np.isfinite(self.center_cm).all() or
                np.max(np.abs(self.center_cm)) > 1e8):
            raise ValueError('A finite 3D cylinder center is required')


@dataclass(frozen=True)
class CylindricalRegion:
    outer: SolidCylinder
    void: SolidCylinder | None = None

    def __post_init__(self):
        if not isinstance(self.outer, SolidCylinder) or (self.void is not None and not isinstance(self.void, SolidCylinder)):
            raise ValueError('Regions need explicit finite-cylinder solids and optional voids')
        if self.void is not None:
            displacement = np.asarray(self.void.center_cm) - self.outer.center_cm
            if (np.linalg.norm(displacement[:2]) + self.void.radius_cm > self.outer.radius_cm or
                    abs(displacement[2]) + self.void.height_cm / 2 > self.outer.height_cm / 2):
                raise ValueError('A region void must be contained in its outer cylinder')
            if (self.void.radius_cm == self.outer.radius_cm and self.void.height_cm == self.outer.height_cm):
                raise ValueError('A cylindrical region must have positive volume')


@dataclass(frozen=True)
class RectangularBeam:
    width_cm: float
    height_cm: float
    center_y_cm: float = 0.
    center_z_cm: float = 0.

    def __post_init__(self):
        _positive(self.width_cm, 'Beam width')
        _positive(self.height_cm, 'Beam height')
        if not np.isfinite([self.center_y_cm, self.center_z_cm]).all():
            raise ValueError('Finite beam offsets are required')


def _solid_length(points, direction, solid, maximum_cm=None):
    p = points - solid.center_cm
    v = np.broadcast_to(direction, p.shape)
    radial = v[:, 0]**2 + v[:, 1]**2
    dot = p[:, 0] * v[:, 0] + p[:, 1] * v[:, 1]
    c = np.sum(p[:, :2]**2, axis=1) - solid.radius_cm**2
    disc = dot**2 - radial * c
    moving = radial > 1e-28
    divisor = np.where(moving, radial, 1.)
    root = np.sqrt(np.maximum(disc, 0.))
    lower = np.where(moving, (-dot - root) / divisor, -np.inf)
    upper = np.where(moving, (-dot + root) / divisor, np.inf)
    valid = np.where(moving, disc >= 0, c <= 0)
    vertical = np.abs(v[:, 2]) > 1e-14
    dz = np.where(vertical, v[:, 2], 1.)
    first = (-solid.height_cm / 2 - p[:, 2]) / dz
    last = (solid.height_cm / 2 - p[:, 2]) / dz
    lower = np.maximum(lower, np.where(vertical, np.minimum(first, last), -np.inf))
    upper = np.minimum(upper, np.where(vertical, np.maximum(first, last), np.inf))
    if maximum_cm is not None:
        upper = np.minimum(upper, maximum_cm)
    valid &= vertical | (np.abs(p[:, 2]) <= solid.height_cm / 2)
    return np.where(valid, np.maximum(0., upper - np.maximum(lower, 0.)), 0.)


def ray_length_cm(region, points_cm, directions):
    """Length of a positive unit ray inside an outer finite solid minus void."""
    points = np.asarray(points_cm, float)
    direction = np.asarray(directions, float)
    if (not isinstance(region, CylindricalRegion) or
            points.ndim != 2 or points.shape[1] != 3 or not 1 <= len(points) <= 2**20 or
            direction.shape not in ((3,), points.shape) or
            not np.isfinite(points).all() or np.max(np.abs(points)) > 1e8 or not np.isfinite(direction).all() or
            not np.allclose(np.linalg.norm(direction, axis=-1), 1., rtol=0, atol=1e-12)):
        raise ValueError('Bounded finite 3D points and unit ray directions are required')
    outer = _solid_length(points, direction, region.outer)
    inner = _solid_length(points, direction, region.void) if region.void else 0.
    if np.any(outer - inner < -1e-10):
        raise ValueError('A ray void exceeds its enclosing solid')
    return np.maximum(0., outer - inner)


def segment_length_cm(region, points_cm, endpoints_cm):
    """Material length on finite point-to-endpoint segments, including voids.

    Unlike a positive infinite ray, a detector segment cannot traverse material
    behind the detector. Endpoint arrays broadcast from a single 3D point;
    zero-length segments return zero. Coordinates and lengths are cm.
    """
    points=np.asarray(points_cm,float);ends=np.asarray(endpoints_cm,float)
    if (not isinstance(region,CylindricalRegion) or points.ndim!=2 or points.shape[1]!=3
            or not 1<=len(points)<=2**20 or ends.shape not in ((3,),points.shape)
            or not np.isfinite(points).all() or not np.isfinite(ends).all()
            or np.max(abs(points))>1e8 or np.max(abs(ends))>1e8):
        raise ValueError('Bounded finite 3D segment endpoints and material region are required')
    delta=ends-points;distance=np.linalg.norm(delta,axis=1)
    direction=delta/np.where(distance>0,distance,1.)[:,None]
    outer=_solid_length(points,direction,region.outer,distance)
    inner=_solid_length(points,direction,region.void,distance) if region.void else 0.
    if np.any(outer-inner < -1e-10):raise ValueError('A segment void exceeds its enclosing material')
    return np.maximum(0.,outer-inner)


def _disk_area_below(y, radius):
    if radius == 0:
        return np.zeros_like(np.asarray(y, float))
    y = np.clip(y, -radius, radius)
    # Evaluate the smaller circular segment before reflecting about y=0.
    # Near a tangent, asin(y/r)+pi/2 cancels against y*sqrt(r²-y²),
    # which can perturb the volume CDF under a mere coordinate translation.
    absolute=np.abs(y)
    chord=np.sqrt(np.maximum((radius-absolute)*(radius+absolute),0.))
    angle=np.arctan2(chord,absolute)
    small=angle<.01;square=angle**2
    series=angle**3*(2/3+square*(-2/15+square*(4/315+square*(-2/2835+square*4/155925))))
    cap=radius**2*np.where(small,series,angle-np.sin(angle)*np.cos(angle))
    return np.where(y<=0,cap,np.pi*radius**2-cap)


def _contains(container, content):
    delta = np.asarray(container.center_cm) - content.center_cm
    tolerance = 1e-12 * max(1., container.radius_cm, container.height_cm)
    return (np.linalg.norm(delta[:2]) + content.radius_cm <= container.radius_cm + tolerance and
            abs(delta[2]) + content.height_cm / 2 <= container.height_cm / 2 + tolerance)


def _disjoint(left, right):
    a, b = left.outer, right.outer
    delta = np.asarray(a.center_cm) - b.center_cm
    return (np.linalg.norm(delta[:2]) >= a.radius_cm + b.radius_cm or
            abs(delta[2]) >= (a.height_cm + b.height_cm) / 2 or
            (left.void is not None and _contains(left.void, b)) or
            (right.void is not None and _contains(right.void, a)))


def _inside(solid, points):
    p = points - solid.center_cm
    return ((np.linalg.norm(p[:, :2], axis=1) <= solid.radius_cm) &
            (np.abs(p[:, 2]) <= solid.height_cm / 2))


def illuminated_points(sample, beam, *, power=14, seed=0):
    """Scrambled Sobol volume points and exact full-powder illuminated fraction.

    This conditions directly on the rectangularly illuminated volume. Its
    integration discrepancy is not an iid counting error or a Poisson draw.
    Reuse one kernel across exposure looks and assess numerical convergence
    separately using more points/independent scrambles.
    """
    if (not isinstance(sample, CylindricalRegion) or not isinstance(beam, RectangularBeam) or
            type(power) is not int or not 8 <= power <= 20 or
            type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError('Bounded cylinder, beam and scrambled integration settings are required')
    u = qmc.Sobol(3, scramble=True, seed=seed).random_base2(power)
    return illuminated_points_from_unit_cube(sample,beam,u)


def illuminated_points_from_unit_cube(sample,beam,unit_points):
    """Map explicit unit-cube quadrature to the illuminated powder volume.

    Use columns of ONE higher-dimensional cubature rule when coupling volume
    to a detector face or another latent coordinate. Independently constructing
    same-seed 3D/2D Sobol rules and pairing their rows introduces correlations.
    No statistical counting noise or measured beam profile is supplied here.
    """
    u=np.asarray(unit_points,float)
    if (not isinstance(sample,CylindricalRegion) or not isinstance(beam,RectangularBeam)
            or u.ndim!=2 or u.shape[1]!=3 or not 1<=len(u)<=2**20
            or not np.isfinite(u).all() or np.any(u<0) or np.any(u>1)):
        raise ValueError('Explicit bounded unit-cube volume coordinates are required')
    outer, void = sample.outer, sample.void
    if void and (tuple(void.center_cm) != tuple(outer.center_cm) or void.height_cm != outer.height_cm):
        raise ValueError('Powder sampling supports solid cylinders or coaxial full-height annuli')
    radius, inner = outer.radius_cm, (void.radius_cm if void else 0.)
    cy, cz = outer.center_cm[1:]
    relative_y,relative_z=beam.center_y_cm-cy,beam.center_z_cm-cz
    low_y = max(-radius, relative_y - beam.width_cm / 2)
    high_y = min(radius, relative_y + beam.width_cm / 2)
    low_z = max(-outer.height_cm / 2, relative_z - beam.height_cm / 2)
    high_z = min(outer.height_cm / 2, relative_z + beam.height_cm / 2)
    if high_y <= low_y or high_z <= low_z:
        return np.empty((0, 3)), 0.
    def area(y):
        return _disk_area_below(y, radius) - _disk_area_below(y, inner)
    area_low, area_high = area(low_y), area(high_y)
    area_fraction=1. if low_y==-radius and high_y==radius else (
        area_high-area_low)/(np.pi*(radius**2-inner**2))
    # Exact geometry lies in [0,1]; limit only possible endpoint roundoff.
    fraction = float(np.clip(area_fraction*(high_z-low_z)/outer.height_cm,0.,1.))
    target = area_low + (area_high - area_low) * u[:, 0]
    left, right = np.full(len(u), low_y), np.full(len(u), high_y)
    for _ in range(48):
        mid = (left + right) / 2
        greater = area(mid) > target
        right = np.where(greater, mid, right)
        left = np.where(greater, left, mid)
    y = (left + right) / 2
    outer_x = np.sqrt(np.maximum(radius**2 - y**2, 0.))
    inner_x = np.sqrt(np.maximum(inner**2 - y**2, 0.))
    chord = outer_x - inner_x
    offset = 2 * chord * u[:, 1]
    x = np.where(offset < chord, -outer_x + offset, inner_x + offset - chord)
    points = np.column_stack((x + outer.center_cm[0], y + cy, cz + low_z + (high_z - low_z) * u[:, 2]))
    return points, fraction


def attenuation(sample, beam, directions, sample_mu_cm_inv, *, layers=(),
                outgoing_sample_mu_cm_inv=None, detector_positions_cm=None,
                power=14, seed=0, max_bytes=256 * 1024**2):
    """Whole-powder source weight for one scatter and no other collision.

    Mu includes removal by every separately declared material process. A vector
    is shared by directions; (directions, energy-pairs) arrays are also accepted.
    Supply outgoing_sample_mu_cm_inv for energy-changing scattering; otherwise
    the same removal applies on both legs. A layer is (region, mu) for equal
    coefficients or (region, incident_mu, outgoing_mu). Regions cannot overlap.
    Finite detector positions optionally replace parallel outgoing directions.
    Returned weights contain the exact illuminated fraction once. This does
    not create an inelastic source law or model multiple scattering.
    """
    if not isinstance(sample, CylindricalRegion) or not isinstance(beam, RectangularBeam):
        raise ValueError('Explicit sample region and rectangular beam are required')
    vectors = np.asarray(directions, float)
    if (vectors.ndim != 2 or vectors.shape[1] != 3 or not 1 <= len(vectors) <= 2048 or
            not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1., rtol=0, atol=1e-12)):
        raise ValueError('A bounded list of unit outgoing directions is required')
    mu = np.asarray(sample_mu_cm_inv, float)
    if mu.ndim == 1:
        mu = np.broadcast_to(mu, (len(vectors), len(mu)))
    if mu.ndim != 2 or mu.shape[0] != len(vectors) or not 1 <= mu.shape[1] <= 100000:
        raise ValueError('Removal coefficients need explicit direction/wavelength dimensions')
    if len(layers) > 16 or type(max_bytes) is not int or not 1024 <= max_bytes <= 1024**3:
        raise ValueError('Explicit bounded region and memory budgets are required')
    outgoing_mu = mu if outgoing_sample_mu_cm_inv is None else np.broadcast_to(
        np.asarray(outgoing_sample_mu_cm_inv, float), mu.shape)
    regions, coefficients, outgoing_coefficients = [sample], [mu], [outgoing_mu]
    distinct_legs = outgoing_sample_mu_cm_inv is not None
    for layer in layers:
        if len(layer) not in (2, 3):
            raise ValueError('Layers need a region and explicit incident/optional outgoing removal')
        region, value = layer[:2]
        if not isinstance(region, CylindricalRegion):
            raise ValueError('Each material layer needs a finite cylindrical region')
        regions.append(region)
        coefficients.append(np.broadcast_to(np.asarray(value, float), mu.shape))
        outgoing_coefficients.append(coefficients[-1] if len(layer) == 2 else
            np.broadcast_to(np.asarray(layer[2], float), mu.shape))
        distinct_legs |= len(layer) == 3
    if any(not _disjoint(a, b) for i, a in enumerate(regions) for b in regions[i + 1:]):
        raise ValueError('Material regions overlap or their disjointness is unverified')
    if any(not np.isfinite(v).all() or np.any(v < 0) for v in coefficients + outgoing_coefficients):
        raise ValueError('Finite nonnegative material removal coefficients are required')
    detector = None if detector_positions_cm is None else np.asarray(detector_positions_cm, float)
    if detector is not None and (detector.shape != vectors.shape or not np.isfinite(detector).all() or
                                 np.max(np.abs(detector)) > 1e8):
        raise ValueError('Explicit finite detector positions must match outgoing directions')
    if detector is not None:
        radial = detector - sample.outer.center_cm
        distance = np.linalg.norm(radial, axis=1)
        if np.any(distance == 0) or not np.allclose(vectors, radial / distance[:, None], rtol=0, atol=1e-12):
            raise ValueError('Detector positions and outgoing directions describe different rays')
        for region in regions:
            inside = _inside(region.outer, detector)
            if region.void is not None:
                inside &= ~_inside(region.void, detector)
            if inside.any():
                raise ValueError('Detectors must lie outside all material regions')
    if type(power) is not int or not 8 <= power <= 20:
        raise ValueError('Bounded scrambled integration settings are required')
    if 2**power * (256 + 16 * len(regions)) + mu.nbytes + outgoing_mu.nbytes > max_bytes:
        raise ValueError('Ray integration exceeds its declared working memory budget')
    points, fraction = illuminated_points(sample, beam, power=power, seed=seed)
    result = np.zeros_like(mu)
    if not len(points):
        return dict(weight=result, illuminated_fraction=fraction, integration_points=0,
                    version=VERSION, experimental_accuracy_certified=False)
    fixed_bytes = points.nbytes * 5 + len(points) * (3 * len(regions) + 3) * 8 + result.nbytes
    block = min(64, (max_bytes - fixed_bytes) // (len(points) * (32 if distinct_legs else 16)))
    if block < 1:
        raise ValueError('Ray integration exceeds its declared working memory budget')
    incident = np.column_stack([ray_length_cm(region, points, [-1., 0., 0.]) for region in regions])
    for i, direction in enumerate(vectors):
        if detector is not None:
            outgoing = detector[i] - points
            distance = np.linalg.norm(outgoing, axis=1)
            if np.any(distance == 0):
                raise ValueError('A detector cannot coincide with a scattering point')
            outgoing /= distance[:, None]
        else:
            outgoing = direction
        exit_paths = np.column_stack([segment_length_cm(region,points,detector[i])
            if detector is not None else ray_length_cm(region,points,outgoing) for region in regions])
        if not distinct_legs:
            paths = incident + exit_paths
        for start in range(0, mu.shape[1], block):
            end = min(start + block, mu.shape[1])
            coefficient = np.array([v[i, start:end] for v in coefficients])
            if distinct_legs:
                outgoing_coefficient = np.array([v[i, start:end] for v in outgoing_coefficients])
                optical_depth = incident @ coefficient + exit_paths @ outgoing_coefficient
            else:
                optical_depth = paths @ coefficient
            result[i, start:end] = fraction * np.exp(-optical_depth).mean(axis=0)
    return dict(weight=result, illuminated_fraction=fraction, integration_points=len(points),
                integration_power=power, scramble_seed=seed, version=VERSION,
                separate_incident_exit_removal=distinct_legs,
                includes_illumination_once=True, multiple_scattering_included=False,
                beam_divergence_modelled=False, experimental_accuracy_certified=False)


@dataclass(frozen=True)
class SingleRegionGrid:
    incident_mu_cm_inv: np.ndarray
    outgoing_mu_cm_inv: np.ndarray
    whole_powder_weight: np.ndarray
    maximum_interpolation_error: np.ndarray
    metadata: dict

    def evaluate(self, direction_index, incident_mu_cm_inv, outgoing_mu_cm_inv):
        """Positive bilinear interpolation; no extrapolation beyond removal grids."""
        raw = np.asarray(direction_index)
        if raw.dtype.kind not in 'iu':
            raise ValueError('Directions need explicit integer indices')
        direction, first, last = np.broadcast_arrays(raw, np.asarray(incident_mu_cm_inv, float),
                                                     np.asarray(outgoing_mu_cm_inv, float))
        x, y = self.incident_mu_cm_inv, self.outgoing_mu_cm_inv
        if (not np.isfinite(first).all() or not np.isfinite(last).all() or np.any(direction < 0)
                or np.any(direction >= len(self.whole_powder_weight)) or np.any(first < x[0])
                or np.any(first > x[-1]) or np.any(last < y[0]) or np.any(last > y[-1])):
            raise ValueError('Removal query extends beyond its declared direction/grid domain')
        i = np.minimum(np.searchsorted(x, first, side='right') - 1, len(x) - 2)
        j = np.minimum(np.searchsorted(y, last, side='right') - 1, len(y) - 2)
        a, b = (first-x[i])/(x[i+1]-x[i]), (last-y[j])/(y[j+1]-y[j])
        w = self.whole_powder_weight
        return ((1-a)*(1-b)*w[direction,i,j] + a*(1-b)*w[direction,i+1,j]
                + (1-a)*b*w[direction,i,j+1] + a*b*w[direction,i+1,j+1])


def single_region_grid(sample, beam, directions, incident_mu_cm_inv, outgoing_mu_cm_inv, *,
                       detector_positions_cm=None, power=12, seed=0, max_bytes=256*1024**2):
    """Fast single-region two-energy Laplace grid, with an interpolation bound.

    T(mu_i,mu_f)=f*mean(exp(-mu_i*L_i)*exp(-mu_f*L_f)). Factoring the two
    exponentials avoids a separate ray integration for every energy pair.
    There are no container layers in this helper; use attenuation for layers.
    The absolute bilinear bound is f/8*(h_i^2*L_i,max^2+h_f^2*L_f,max^2).
    It bounds interpolation of this numerical integral, not its geometric
    integration discrepancy or physical material/geometry uncertainty.
    """
    grids = [np.array(v, dtype=float, copy=True) for v in (incident_mu_cm_inv, outgoing_mu_cm_inv)]
    if any(v.ndim != 1 or not 2 <= len(v) <= 512 or not np.isfinite(v).all()
           or v[0] < 0 or np.any(np.diff(v) <= 0) for v in grids):
        raise ValueError('Two bounded increasing nonnegative removal grids are required')
    vectors = np.asarray(directions, float)
    if (vectors.ndim != 2 or vectors.shape[1] != 3 or not 1 <= len(vectors) <= 2048
            or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1.,rtol=0,atol=1e-12)):
        raise ValueError('Explicit bounded unit directions are required')
    if (type(max_bytes) is not int or max_bytes < 1024 or type(power) is not int or not 8 <= power <= 20):
        raise ValueError('Bounded integration and memory settings are required')
    required = len(vectors)*len(grids[0])*len(grids[1])*8 + 2**power*(256+16*(len(grids[0])+len(grids[1])))
    if required > max_bytes:
        raise ValueError('Two-energy grid exceeds its declared working memory budget')
    points, fraction = illuminated_points(sample,beam,power=power,seed=seed)
    detector = None if detector_positions_cm is None else np.asarray(detector_positions_cm,float)
    center=np.asarray(sample.outer.center_cm)
    radius,height=sample.outer.radius_cm,sample.outer.height_cm
    if detector is not None:
        if detector.shape != vectors.shape or not np.isfinite(detector).all() or np.max(np.abs(detector))>1e8:
            raise ValueError('Finite external detectors must match the outgoing directions')
        displacement=detector-center
        distance=np.linalg.norm(displacement,axis=1)
        if (detector.shape != vectors.shape or not np.isfinite(detector).all() or np.any(distance==0)
                or not np.allclose(displacement/distance[:,None],vectors,rtol=0,atol=1e-12)
                or _inside(sample.outer,detector).any()):
            raise ValueError('Finite external detectors must match the outgoing directions')
        # Every scatter point is within the outer cylinder bounding sphere.
        max_distance=distance+np.hypot(radius,height/2)
        min_horizontal=np.maximum(0.,np.linalg.norm(displacement[:,:2],axis=1)-radius)/max_distance
        min_vertical=np.maximum(0.,np.abs(displacement[:,2])-height/2)/max_distance
    else:
        min_horizontal=np.linalg.norm(vectors[:,:2],axis=1)
        min_vertical=np.abs(vectors[:,2])
    with np.errstate(divide='ignore'):
        exit_max=np.minimum(np.hypot(2*radius,height),np.minimum(2*radius/min_horizontal,height/min_vertical))
    bound=fraction/8*(np.max(np.diff(grids[0]))**2*(2*radius)**2
                      + np.max(np.diff(grids[1]))**2*exit_max**2)
    table=np.zeros((len(vectors),len(grids[0]),len(grids[1])))
    if len(points):
        incoming=ray_length_cm(sample,points,[-1.,0.,0.])
        first=np.exp(-incoming[:,None]*grids[0])
        for index,direction in enumerate(vectors):
            if detector is not None:
                direction=detector[index]-points
                direction/=np.linalg.norm(direction,axis=1)[:,None]
            last=ray_length_cm(sample,points,direction)
            table[index]=fraction/len(points)*(first.T@np.exp(-last[:,None]*grids[1]))
    for array in (*grids,table,bound):array.setflags(write=False)
    return SingleRegionGrid(*grids,table,bound,dict(version=VERSION,
        integration_power=power,seed=seed,illuminated_fraction=fraction,
        whole_powder_illumination_included_once=True,maximum_interpolation_error_is_physical_uncertainty=False,
        multiple_scattering_included=False,experimental_accuracy_certified=False))


@dataclass(frozen=True)
class ElasticRegionGrid:
    mu_cm_inv: np.ndarray
    whole_powder_weight: np.ndarray
    maximum_interpolation_error: np.ndarray
    metadata: dict

    def evaluate(self,direction_index,mu_cm_inv):
        """Positive linear interpolation in physical removal, without extrapolation."""
        raw=np.asarray(direction_index)
        if raw.dtype.kind not in 'iu':
            raise ValueError('Directions need explicit integer indices')
        direction,mu=np.broadcast_arrays(raw,np.asarray(mu_cm_inv,float))
        grid=self.mu_cm_inv
        if (not np.isfinite(mu).all() or np.any(direction<0) or np.any(direction>=len(self.whole_powder_weight))
                or np.any(mu<grid[0]) or np.any(mu>grid[-1])):
            raise ValueError('Elastic removal query exceeds its direction/grid domain')
        i=np.minimum(np.searchsorted(grid,mu,side='right')-1,len(grid)-2)
        fraction=(mu-grid[i])/(grid[i+1]-grid[i])
        w=self.whole_powder_weight
        return (1-fraction)*w[direction,i]+fraction*w[direction,i+1]


def elastic_region_grid(sample,beam,directions,mu_cm_inv,*,detector_positions_cm=None,
                        power=12,seed=0,max_bytes=256*1024**2):
    """Finite-cylinder elastic Laplace grid, T(mu)=f*mean(exp[-mu*(Li+Lo)]).

    The same removal coefficient applies before and after this elastic scatter.
    The interpolation bound f*Lmax²*h²/8 is for this numerical path integral;
    its sampling error and actual specimen metrology remain separate. Evaluating
    removal at the actual wavelength avoids interpolation across Bragg edges.
    This single-region helper contains no layer or multiple-scattering source.
    """
    if not isinstance(sample,CylindricalRegion) or not isinstance(beam,RectangularBeam):
        raise ValueError('Explicit sample region and rectangular beam are required')
    mu=np.array(mu_cm_inv,float,copy=True)
    vectors=np.asarray(directions,float)
    if (mu.ndim!=1 or not 2<=len(mu)<=512 or not np.isfinite(mu).all() or mu[0]<0
            or np.any(np.diff(mu)<=0) or vectors.ndim!=2 or vectors.shape[1]!=3
            or not 1<=len(vectors)<=2048 or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors,axis=1),1.,rtol=0,atol=1e-12)):
        raise ValueError('Bounded increasing nonnegative removal and explicit unit directions are required')
    if (type(power) is not int or not 8<=power<=20 or type(max_bytes) is not int
            or not 1024<=max_bytes<=1024**3):
        raise ValueError('Bounded integration and memory settings are required')
    required=len(vectors)*len(mu)*8+2**power*(384+8*len(mu))
    if required>max_bytes:
        raise ValueError('Elastic path grid exceeds its declared working memory budget')
    detector=None if detector_positions_cm is None else np.asarray(detector_positions_cm,float)
    if detector is not None:
        center=np.asarray(sample.outer.center_cm)
        if detector.shape!=vectors.shape or not np.isfinite(detector).all() or np.max(abs(detector))>1e8:
            raise ValueError('Finite external detector positions must match the directions')
        displacement=detector-center;distance=np.linalg.norm(displacement,axis=1)
        if (np.any(distance==0) or not np.allclose(displacement/distance[:,None],vectors,rtol=0,atol=1e-12)
                or _inside(sample.outer,detector).any()):
            raise ValueError('Finite external detector positions must match the directions')
    points,fraction=illuminated_points(sample,beam,power=power,seed=seed)
    table=np.zeros((len(vectors),len(mu)));bound=np.zeros(len(vectors));maximum=np.zeros(len(vectors))
    if len(points):
        incoming=ray_length_cm(sample,points,[-1.,0.,0.])
        for index,direction in enumerate(vectors):
            if detector is not None:
                direction=detector[index]-points
                direction/=np.linalg.norm(direction,axis=1)[:,None]
            total=incoming+ray_length_cm(sample,points,direction)
            table[index]=fraction*np.exp(-total[:,None]*mu).mean(axis=0)
            maximum[index]=float(total.max())
            bound[index]=fraction*maximum[index]**2*np.diff(mu).max()**2/8
    for array in (mu,table,bound,maximum):array.setflags(write=False)
    return ElasticRegionGrid(mu,table,bound,dict(version='powgen-elastic-finite-cylinder-path-grid/1',
        path_primitive_version=VERSION,integration_power=power,seed=seed,illuminated_fraction=fraction,
        sampled_maximum_total_path_cm=maximum.tolist(),
        whole_powder_illumination_included_once=True,finite_detector_positions_supplied=detector is not None,
        maximum_interpolation_error_is_for_numeric_path_integral=True,
        geometric_sampling_or_physical_metrology_error_bounded=False,
        multiple_scattering_or_container_layers_included=False,experimental_accuracy_certified=False))
