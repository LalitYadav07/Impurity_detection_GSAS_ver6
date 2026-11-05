"""Numerical parent handoff, independent of publication-style CIF rounding."""
from __future__ import annotations

import math
from pathlib import Path


def export_exact_parent(phase, destination):
    """Preserve the native fractional structure while retaining CIF metadata.

    GSAS-II's ordinary exporter rounds refined values to their uncertainties.
    That is appropriate for tables, but can break exact site symmetry or lose
    the parent origin needed by a subsequent zero-distortion construction.
    The GPX is authoritative; neither its coordinates nor uncertainties move.
    """
    import CifFile

    destination = Path(destination)
    phase.export_CIF(str(destination))
    cif = CifFile.ReadCif(str(destination))
    blocks = [cif[key] for key in cif.keys() if '_atom_site_label' in cif[key]]
    if len(blocks) != 1:
        raise ValueError('Exact parent export needs one unambiguous atom-site block')
    block = blocks[0]
    cx, ct, _, _ = phase.data['General']['AtomPtrs']
    atoms = {atom[ct-1]: atom for atom in phase.data['Atoms']}
    labels = list(block['_atom_site_label'])
    if len(atoms) != len(phase.data['Atoms']) or len(labels) != len(set(labels)) or set(labels) != set(atoms):
        raise ValueError('Exact parent export lost native site identities')

    def number(value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('Exact parent export requires finite native parameters')
        return format(value, '.17g')

    for index, axis in enumerate('xyz'):
        block['_atom_site_fract_'+axis] = [number(atoms[label][cx+index]) for label in labels]
    block['_atom_site_occupancy'] = [number(atoms[label][cx+3]) for label in labels]
    for key, value in zip(('length_a', 'length_b', 'length_c', 'angle_alpha', 'angle_beta', 'angle_gamma'),
                          phase.data['General']['Cell'][1:7]):
        # Native reciprocal/direct conversions can leave symmetry-equivalent
        # lengths one ULP apart or an exact 120-degree angle just below 120.
        # ISODISTORT compares those values strictly. Twelve significant digits
        # remove arithmetic roundoff without uncertainty-driven table rounding.
        block['_cell_'+key] = format(float(number(value)), '.12g')
    destination.write_text(cif.WriteOut())
    return dict(path=str(destination), precision_policy='native_coordinates_precise_cell',
                cell_significant_digits=12,
                source_coordinates_unchanged=True, sites=len(labels))
