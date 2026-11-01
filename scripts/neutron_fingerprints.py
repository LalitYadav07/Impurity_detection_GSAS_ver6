"""Neutron fingerprints with explicit D/T identity and versioned provenance.

Ordinary elements use pymatgen's calculator without alteration. Its species
``symbol`` is H for D and T, so isotope-bearing structures need an explicit
per-species coherent scattering length. No global scattering table is changed.

D/T values below are from the frozen GSAS-II ``AtmBlens`` table. Its unit is
10^-12 cm; multiplying by ten gives fm, the unit used by pymatgen. Natural
elements deliberately retain pymatgen's existing table, including H=-3.739 fm.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from types import MappingProxyType

import numpy as np
from pymatgen.analysis.diffraction.core import DiffractionPattern, get_unique_families
from pymatgen.analysis.diffraction.neutron import ATOMIC_SCATTERING_LEN, NDCalculator
from pymatgen.core.periodic_table import DummySpecies

GENERATOR_VERSION = "radar-neutron-fingerprint/1"
ISOTOPE_LENGTHS_FM = MappingProxyType({"D": 6.681, "T": 4.792})
ISOTOPE_TABLE_PROVENANCE = MappingProxyType({
    "source": "GSAS-II/GSASII/atmdata.py:AtmBlens",
    "source_sha256": "f34011ec9e9f47088b42b840626443e0e4d7ff757de92b303d2be4a0e9f8ce66",
    "source_unit": "10^-12 cm",
    "fingerprint_unit": "fm",
    "conversion_factor": 10,
    "entries": {"D": "D_ / H_2: 0.6681", "T": "T_ / H_3: 0.4792"},
    "deuterium_reference": "Snow (2020), doi:10.1103/PhysRevD.101.062004 (GSAS-II table annotation)",
})


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def species_identity(species) -> str:
    """Keep named isotope identity; retain charge in cache/symmetry identity."""
    if isinstance(species, DummySpecies):
        raise ValueError(f"Unsupported neutron atom/isotope {species!s}; an explicit supported species is required")
    name = getattr(species, "name", None)
    if name in ISOTOPE_LENGTHS_FM:
        return name
    symbol = getattr(species, "symbol", None)
    # Hydrogen's supported mass-labelled forms are H, D and T. A future
    # pymatgen isotope representation must not silently fall back to H.
    if symbol == "H" and getattr(species, "A", 1) not in (None, 1):
        raise ValueError(f"Unsupported hydrogen isotope {species!s}; only H, D and T are supported")
    return str(species)


def named_isotopes(structure) -> tuple[str, ...]:
    identities = {species_identity(sp) for site in structure for sp in site.species}
    return tuple(sorted(identities.intersection(ISOTOPE_LENGTHS_FM)))


def generator_metadata(radiation: str = "neutron") -> dict:
    """A portable receipt for the actual calculator/table generation."""
    version = importlib.metadata.version("pymatgen")
    if radiation.lower() != "neutron":
        return {"generator": "pymatgen-xrd", "pymatgen_version": version}
    return {
        "generator": GENERATOR_VERSION,
        "pymatgen_version": version,
        "natural_table": "pymatgen/neutron_scattering_length.json (unchanged)",
        "natural_table_sha256": _digest(ATOMIC_SCATTERING_LEN),
        "isotope_lengths_fm": dict(ISOTOPE_LENGTHS_FM),
        "isotope_table": dict(ISOTOPE_TABLE_PROVENANCE),
        "symmetry_refinement": "disabled for isotope-bearing structures",
    }


def request_metadata(cif_content: str, radiation: str, wavelength: float,
                     two_theta_min: float, two_theta_max: float, top_m: int) -> dict:
    """Bind an on-disk fingerprint to its source, physics and generation window."""
    body = {"generator": generator_metadata(radiation),
            "source_sha256": hashlib.sha256(cif_content.encode("utf-8")).hexdigest(),
            "radiation": radiation.lower(), "wavelength_angstrom": float(wavelength),
            "two_theta_range_degrees": [float(two_theta_min), float(two_theta_max)],
            "top_m": int(top_m)}
    return {**body, "request_sha256": _digest(body)}


def structure_cache_key(structure, radiation: str = "neutron") -> str:
    """Include isotope identity, exact site values and calculator generation."""
    return _digest({
        "generator": generator_metadata(radiation),
        "lattice": [[float(v).hex() for v in row] for row in structure.lattice.matrix],
        "sites": [{"fractional": [float(v).hex() for v in site.frac_coords],
                   "species": sorted((species_identity(sp), float(n).hex()) for sp, n in site.species.items())}
                  for site in structure],
    })


def isotope_preserving_space_group_info(structure, symprec=1e-2, angle_tolerance=5.0):
    """Analyze identity without allowing Composition equality to merge H/D.

    spglib accepts arbitrary positive integer type labels. These labels encode
    the complete species/occupancy identity and never enter a physical model.
    The source Structure is not standardized, averaged or mutated.
    """
    if not named_isotopes(structure):
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        analyzer = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
        return analyzer.get_space_group_symbol(), int(analyzer.get_space_group_number())
    import spglib
    types = {}
    labels = []
    for site in structure:
        identity = tuple(sorted((species_identity(sp), float(n).hex()) for sp, n in site.species.items()))
        labels.append(types.setdefault(identity, len(types) + 1))
    dataset = spglib.get_symmetry_dataset((structure.lattice.matrix, structure.frac_coords, labels),
                                         symprec=symprec, angle_tolerance=angle_tolerance)
    if dataset is None:
        raise ValueError("No symmetry model preserves the supplied isotope identities")
    return str(dataset.international), int(dataset.number)


class NeutronFingerprintCalculator(NDCalculator):
    """Use upstream physics exactly for natural species and explicit b for D/T."""

    def get_pattern(self, structure, scaled=True, two_theta_range=(0, 90)):
        for site in structure:
            for species in site.species:
                if isinstance(species, DummySpecies) or species.symbol not in ATOMIC_SCATTERING_LEN:
                    raise ValueError(f"Unsupported neutron atom/isotope {species!s}; no coherent scattering length is available")
        isotopes = named_isotopes(structure)
        if not isotopes:
            return super().get_pattern(structure, scaled=scaled, two_theta_range=two_theta_range)
        if self.symprec:
            raise ValueError("Symmetry refinement of an isotope-bearing structure is not supported; preserve the source model")
        # Same reciprocal-point enumeration, Lorentz correction and peak
        # grouping as pymatgen NDCalculator (Materials Project, MIT license),
        # with the scattering-length lookup resolved per species below.
        wavelength = self.wavelength
        lattice = structure.lattice
        is_hex = lattice.is_hexagonal()
        min_r, max_r = ((0, 2 / wavelength) if two_theta_range is None else
                        [2 * math.sin(math.radians(t / 2)) / wavelength for t in two_theta_range])
        points = lattice.reciprocal_lattice_crystallographic.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], max_r)
        if min_r:
            points = [point for point in points if point[1] >= min_r]
        lengths, coordinates, occupancies, factors = [], [], [], []
        for site in structure:
            for species, occupancy in site.species.items():
                identity = species_identity(species)
                lengths.append(ISOTOPE_LENGTHS_FM[identity] if identity in ISOTOPE_LENGTHS_FM
                               else ATOMIC_SCATTERING_LEN[species.symbol])
                coordinates.append(site.frac_coords)
                occupancies.append(occupancy)
                factors.append(self.debye_waller_factors.get(identity, self.debye_waller_factors.get(species.symbol, 0)))
        lengths, coordinates, occupancies, factors = map(np.asarray, (lengths, coordinates, occupancies, factors))
        peaks, angles = {}, []
        for hkl, reciprocal_length, _, _ in sorted(points, key=lambda p: (p[1], -p[0][0], -p[0][1], -p[0][2])):
            hkl = [round(v) for v in hkl]
            if reciprocal_length == 0:
                continue
            d_hkl = 1 / reciprocal_length
            theta = math.asin(wavelength * reciprocal_length / 2)
            dw = np.exp(-factors * (reciprocal_length / 2) ** 2)
            phase = np.dot(coordinates, np.transpose([hkl])).T[0]
            amplitude = np.sum(lengths * occupancies * np.exp(2j * np.pi * phase) * dw)
            intensity = (amplitude * amplitude.conjugate()).real / (math.sin(theta) ** 2 * math.cos(theta))
            angle = math.degrees(2 * theta)
            if is_hex:
                hkl = (hkl[0], hkl[1], -hkl[0] - hkl[1], hkl[2])
            matches = np.where(np.abs(np.subtract(angles, angle)) < self.TWO_THETA_TOL)
            if len(matches[0]):
                peaks[angles[matches[0][0]]][0] += intensity
                peaks[angles[matches[0][0]]][1].append(tuple(hkl))
            else:
                peaks[angle] = [intensity, [tuple(hkl)], d_hkl]
                angles.append(angle)
        maximum = max((p[0] for p in peaks.values()), default=0)
        x, y, hkls, spacings = [], [], [], []
        for angle, (intensity, families, spacing) in sorted(peaks.items()):
            if maximum > 0 and intensity / maximum * 100 > self.SCALED_INTENSITY_TOL:
                x.append(angle); y.append(intensity); spacings.append(spacing)
                hkls.append([{"hkl": hkl, "multiplicity": count} for hkl, count in get_unique_families(families).items()])
        result = DiffractionPattern(x, y, hkls, spacings)
        if scaled and len(y):
            result.normalize(mode="max", value=100)
        return result
