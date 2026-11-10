"""Single-scattering geometry and explicit exposure units (cm, g, s, Angstrom).

The transverse cylinder integral includes incident and exit attenuation. The
quadrature is deterministic: it is not counting noise and is reused at all looks.
The finite-height approximation assumes scattering in the horizontal plane.
"""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss

AVOGADRO = 6.02214076e23


def positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def bin_widths(x):
    x = np.asarray(x, float)
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("x must be a finite increasing vector with >=3 points")
    # Midpoint edges, including half-width extrapolation at the ends.
    edges = np.r_[x[0]-(x[1]-x[0])/2, (x[1:]+x[:-1])/2, x[-1]+(x[-1]-x[-2])/2]
    return np.diff(edges)


@dataclass(frozen=True)
class Cylinder:
    mass_g: float
    radius_cm: float
    packing_fraction: float = 0.5
    beam_width_cm: float = 2.5
    beam_height_cm: float = 5.0
    inner_radius_cm: float = 0.0
    wall_thickness_cm: float = 0.0

    def __post_init__(self):
        for name in ("mass_g", "radius_cm", "packing_fraction", "beam_width_cm", "beam_height_cm"):
            positive(getattr(self, name), name)
        if self.packing_fraction > 1 or not 0 <= self.inner_radius_cm < self.radius_cm or self.wall_thickness_cm < 0:
            raise ValueError("Invalid packing fraction, annulus or can thickness")

    def quantities(self, cell_mass_da, cell_volume_a3):
        density = positive(cell_mass_da, "cell mass") / AVOGADRO / (positive(cell_volume_a3, "cell volume")*1e-24)
        bulk = density*self.packing_fraction
        volume = self.mass_g/bulk
        height = volume/(np.pi*(self.radius_cm**2-self.inner_radius_cm**2))
        return dict(crystal_density_g_cm3=density, bulk_density_g_cm3=bulk,
                    fill_height_cm=height, unit_cells=self.mass_g/cell_mass_da*AVOGADRO,
                    cells_per_cm3=bulk/cell_mass_da*AVOGADRO,
                    vertical_illumination_fraction=min(1., self.beam_height_cm/height))


def _disk_paths(points, direction, radius):
    dot = points @ direction
    disc = dot**2 + radius**2 - np.sum(points**2, axis=1)
    return -dot + np.sqrt(np.maximum(disc, 0))


def _hole_length(points, direction, radius):
    if radius == 0:
        return np.zeros(len(points))
    dot = points @ direction
    disc = dot**2 + radius**2 - np.sum(points**2, axis=1)
    root = np.sqrt(np.maximum(disc, 0))
    return np.where(disc > 0, np.maximum(0, -dot+root)-np.maximum(0, -dot-root), 0)


def cylinder_attenuation(geometry, angles_deg, mu_cm_inv, can_mu_cm_inv=0., quadrature=24):
    """Return <beam mask * exp[-mu(Lin+Lout)-mu_can*Lcan]> over sample.

    Includes horizontal underillumination in the average; vertical illumination
    is applied separately. Annuli and side-wall attenuation are supported.
    End caps, vertical divergence, multiple scattering and texture are not.
    """
    angles = np.asarray(angles_deg, float)
    mu = np.broadcast_to(mu_cm_inv, angles.shape)
    can_mu = np.broadcast_to(can_mu_cm_inv, angles.shape)
    if not np.isfinite(angles).all() or np.any(mu < 0) or np.any(can_mu < 0) or not np.isfinite(mu+can_mu).all():
        raise ValueError("Nonfinite or negative attenuation")
    z, w = leggauss(quadrature)
    r = np.sqrt((z+1)/2*(geometry.radius_cm**2-geometry.inner_radius_cm**2)+geometry.inner_radius_cm**2)
    phi = (np.arange(4*quadrature)+.5)*2*np.pi/(4*quadrature)
    points = np.stack(np.broadcast_arrays(r[:, None]*np.cos(phi), r[:, None]*np.sin(phi)), axis=-1).reshape(-1, 2)
    weight = np.repeat(w/2/(4*quadrature), 4*quadrature)
    weight *= np.abs(points[:, 1]) <= geometry.beam_width_cm/2
    incident_dir = np.array([-1., 0.])
    lin_outer = _disk_paths(points, incident_dir, geometry.radius_cm)
    lin = lin_outer-_hole_length(points, incident_dir, geometry.inner_radius_cm)
    rout = geometry.radius_cm+geometry.wall_thickness_cm
    can_in = _disk_paths(points, incident_dir, rout)-lin_outer
    result = np.empty(angles.size)
    for i, angle in enumerate(angles.ravel()):
        direction = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        lout_outer = _disk_paths(points, direction, geometry.radius_cm)
        lout = lout_outer-_hole_length(points, direction, geometry.inner_radius_cm)
        can_out = _disk_paths(points, direction, rout)-lout_outer
        result[i] = np.dot(weight, np.exp(-mu.flat[i]*(lin+lout)-can_mu.flat[i]*(can_in+can_out)))
    return result.reshape(angles.shape)


def charge_from_power(seconds, power_mw, proton_energy_gev):
    """Integrated charge in microampere-hours; MW/GeV = mA."""
    return positive(seconds, "seconds")*positive(power_mw, "power MW")/positive(proton_energy_gev, "proton energy GeV")*1000/3600


def hb2a_cross_section(gsas_profile, wavelength_a):
    """Unit-scale GSAS-II PNC profile -> b/sr/unit cell, before absorption.

    Paddison et al., arXiv:2506.03201 Eq. 8. GSAS-II neutron b are in
    sqrt(barn) and its CW profile uses centidegree normalization.
    Nuclear and magnetic terms must share this conversion and their cell scales.
    """
    return np.asarray(gsas_profile)*4500*positive(wavelength_a, "wavelength")**3/(2*np.pi**2)
