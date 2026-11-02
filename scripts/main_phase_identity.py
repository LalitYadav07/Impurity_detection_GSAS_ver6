"""Comparison-only identities for excluding an already supplied main phase.

Oxidation labels can differ after a catalog CIF export. Ignore that annotation
for this identity check while preserving isotope, spin and site occupancies.
The returned structures are temporary matcher inputs, never scientific models.
"""
from __future__ import annotations

import math

from pymatgen.core import Structure
from pymatgen.core.periodic_table import DummySpecies

from cif_occupancy import species_key


def _identity(species):
    if isinstance(species, DummySpecies):
        raise ValueError("A main-phase comparison needs identified elements/isotopes")
    spin = getattr(species, "spin", None)
    if spin is not None and not math.isfinite(float(spin)):
        raise ValueError("A main-phase comparison needs finite spin labels")
    return species_key(species), None if spin is None else float(spin)


def _inventory(structure):
    amounts = {}
    for site in structure:
        for species, amount in site.species.items():
            value = float(amount)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("A main-phase comparison needs positive finite occupancies")
            key = _identity(species)
            amounts[key] = amounts.get(key, 0.0) + value
    total = sum(amounts.values())
    if total <= 0:
        raise ValueError("A main-phase comparison needs occupied sites")
    return {key: value/total for key, value in amounts.items()}


def main_phase_comparison_pair(reference, candidate):
    """Return copies suitable for exact-species geometric matching, or None.

    Labels are shared by this pair and distinguish H/D/T and explicit spin.
    Dummy labels prevent pymatgen's element-based composition/primitive-cell
    logic from merging hydrogen isotopes. Every original site amount and
    coordinate is copied. Only the oxidation annotation is omitted.
    """
    first, second = _inventory(reference), _inventory(candidate)
    if first.keys() != second.keys() or any(
        not math.isclose(first[key], second[key], rel_tol=0, abs_tol=1e-10)
        for key in first
    ):
        return None
    labels = {key: DummySpecies(f"Xq{index}")
              for index, key in enumerate(sorted(first, key=repr))}

    def copy_for_comparison(structure):
        sites = []
        for site in structure:
            amounts = {}
            for species, amount in site.species.items():
                label = labels[_identity(species)]
                amounts[label] = amounts.get(label, 0.0) + float(amount)
            sites.append(amounts)
        return Structure(structure.lattice.copy(), sites, structure.frac_coords.copy())

    return copy_for_comparison(reference), copy_for_comparison(candidate)
