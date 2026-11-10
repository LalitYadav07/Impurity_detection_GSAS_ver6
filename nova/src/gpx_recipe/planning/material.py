"""Fixed-structure, anisotropic powder Bragg removal in barns per nuclear cell.

This is one scattering component, not the total material scattering law.
Inelastic, incoherent, diffuse and magnetic removal require separate inputs.
It must not be added as a second source to a powder pattern already containing
the same coherent elastic process. Shared neutron atomic parameters remain
explicit; the cell sum itself uses no GSAS symmetry/structure-factor routine.
"""
from dataclasses import dataclass
from pathlib import Path
import copy
import hashlib
import json
import math
import re
import numpy as np
from .isotopes import amplitude
from .nuclear import full_cell, number

VERSION = 'powgen-fixed-structure-bragg-removal/1'


def atom_type(value):
    """Translate explicit CIF charges to the native element+charge spelling."""
    match = re.fullmatch(r'([A-Z][a-z]?)(?:(\d+)([+-])|([+-])(\d+))?', value)
    if not match:
        raise ValueError('Unsupported exact CIF atom type: ' + str(value))
    symbol, magnitude, sign, other_sign, other_magnitude = match.groups()
    return symbol + ((sign or other_sign) + (magnitude or other_magnitude)
                     if sign or other_sign else '')


def _metric(path):
    import CifFile
    data = CifFile.ReadCif(str(path))
    block = data[data.keys()[0]]
    lengths = np.array([number(block['_cell_length_' + v]) for v in 'abc'])
    angles = np.deg2rad([number(block['_cell_angle_' + v]) for v in ('alpha', 'beta', 'gamma')])
    cosines = np.cos(angles)
    metric = np.diag(lengths**2)
    for i, j, k in ((1, 2, 0), (0, 2, 1), (0, 1, 2)):
        metric[i, j] = metric[j, i] = lengths[i] * lengths[j] * cosines[k]
    if np.any(lengths <= 0) or np.linalg.eigvalsh(metric).min() <= 1e-12:
        raise ValueError('A positive nondegenerate cell metric is required')
    return lengths, np.linalg.inv(metric), float(np.sqrt(np.linalg.det(metric)))


def _partial(hkl, xyz, tensors, occupation, species, types):
    """Site/ADP sums by exact atom type, before neutron amplitudes enter."""
    phase = 2 * np.pi * (hkl @ xyz.T)
    thermal = np.exp(-2 * np.pi**2 * np.einsum('hi,aij,hj->ha', hkl, tensors, hkl))
    values = np.exp(1j * phase) * thermal * occupation
    return np.column_stack([values[:, species == kind].sum(axis=1) for kind in types])


@dataclass(frozen=True)
class PowderBraggKernel:
    atom_types: tuple
    neutron_records: tuple
    d_a: np.ndarray
    cumulative_gram_bare: np.ndarray
    cell_volume_a3: float
    cell_mass_da: float
    atom_count: float
    minimum_wavelength_a: float
    provenance: dict

    def cross_section(self, wavelength_a):
        """Powder-averaged coherent elastic removal, barn per nuclear cell.

        All reciprocal points with 2d >= lambda enter, including both Friedel
        signs. Complex and resonant isotope amplitudes are evaluated at the
        actual wavelength. No interpolation across Bragg edges is used.
        """
        wave = np.asarray(wavelength_a, float)
        if (wave.ndim != 1 or not 1 <= len(wave) <= 100000 or
                not np.isfinite(wave).all() or np.any(wave < self.minimum_wavelength_a)):
            raise ValueError('Wavelengths must be finite vectors inside the declared Bragg domain')
        if hashlib.sha256(json.dumps(dict(zip(self.atom_types, self.neutron_records)), sort_keys=True).encode()).hexdigest() != self.provenance['neutron_records_sha256']:
            raise ValueError('Compiled neutron records changed; rebuild the Bragg kernel')
        indices = np.searchsorted(self.d_a, wave / 2, side='left')
        coeff = np.array([amplitude(record, wave) for record in self.neutron_records]).T
        gram = self.cumulative_gram_bare[indices]
        summed = np.einsum('ne,nef,nf->n', coeff.conj(), gram, coeff).real
        result = wave**2 / (2 * self.cell_volume_a3) * summed
        if not np.isfinite(result).all() or result.min() < -1e-12 * max(1., result.max()):
            raise ValueError('Nonphysical powder Bragg removal')
        return np.maximum(result, 0.)


def powder_bragg_kernel(path, neutron_records, minimum_wavelength_a, *,
                        max_reflections=250000, max_bytes=256 * 1024**2):
    """Compile a complete reciprocal sum for a fixed explicit CIF and domain.

    ADPs and special-position consistency are enforced by the independent CIF
    expansion. A cumulative real Gram matrix retains exact tensor/phase sums
    while allowing fast wavelength-dependent complex neutron amplitudes.
    Only one of each Friedel pair is stored, with an explicit factor of two.
    No displacement tensor is replaced by an isotropic equivalent.
    """
    path = Path(path).resolve()
    if (type(minimum_wavelength_a) not in (int, float) or
            not math.isfinite(minimum_wavelength_a) or minimum_wavelength_a <= 0):
        raise ValueError('A finite positive minimum wavelength is required')
    if (type(max_reflections) is not int or not 1 <= max_reflections <= 1000000 or
            type(max_bytes) is not int or not 1024 <= max_bytes <= 1024**3):
        raise ValueError('Explicit bounded reciprocal-space resources are required')
    xyz, tensors, raw_species, occupation, counts = full_cell(path, exact_atom_types=True)
    species = np.array([atom_type(value) for value in raw_species])
    types = tuple(sorted(set(species)))
    if len(types) > 16 or not isinstance(neutron_records, dict) or set(types) - set(neutron_records):
        raise ValueError('Explicit neutron records for at most sixteen exact CIF atom types are required')
    records = tuple(copy.deepcopy(neutron_records[kind]) for kind in types)
    for record in records:
        if not isinstance(record, dict) or 'Mass' not in record:
            raise ValueError('Neutron records must be explicit isotope data dictionaries')
        amplitude(record, np.array([minimum_wavelength_a]))
        if not math.isfinite(record['Mass']) or record['Mass'] <= 0:
            raise ValueError('Positive isotope-weighted nuclear masses are required')
    lengths, reciprocal, volume = _metric(path)
    cutoff = minimum_wavelength_a / 2
    # |h_i| <= |a_i|/d follows from Cauchy-Schwarz, also for oblique cells.
    ratios = lengths / cutoff
    if not np.isfinite(ratios).all() or np.any(ratios > 1000000):
        raise ValueError('Reciprocal enumeration exceeds its declared resource budget')
    bounds = np.floor(ratios).astype(np.int32)
    cube = math.prod(int(2 * value + 1) for value in bounds)
    if cube > 2000000 or cube * 40 > max_bytes:
        raise ValueError('Reciprocal enumeration exceeds its declared resource budget')
    candidates = np.indices(tuple(2 * bounds + 1), dtype=np.int32).reshape(3, -1).T - bounds
    h, k, l = candidates.T
    chosen = (h > 0) | ((h == 0) & (k > 0)) | ((h == 0) & (k == 0) & (l > 0))
    hkl = candidates[chosen]
    d = 1 / np.sqrt(np.einsum('hi,ij,hj->h', hkl, reciprocal, hkl))
    chosen = d >= cutoff
    hkl, d = hkl[chosen], d[chosen]
    if len(d) > max_reflections:
        raise ValueError('Reciprocal reflection count exceeds its declared domain budget')
    required = 2 * (len(d) + 1) * len(types)**2 * 8 + 4 * hkl.nbytes + 32 * len(d)
    if required > max_bytes:
        raise ValueError('Bragg-kernel arrays exceed their declared memory budget')
    order = np.argsort(d)
    hkl, d = hkl[order], d[order]
    del candidates, h, k, l, chosen, order
    chunk_size = min(1024, (max_bytes - required) // (len(xyz) * 64 + len(types)**2 * 24))
    if chunk_size < 1:
        raise ValueError('Site/ADP sums exceed their declared working memory budget')
    gram = np.zeros((len(d) + 1, len(types), len(types)))
    for first in range(0, len(d), chunk_size):
        last = min(first + chunk_size, len(d))
        partial = _partial(hkl[first:last], xyz, tensors, occupation, species, types)
        # Friedel averaging retains Re(A_e A_f*) for complex neutron b.
        gram[first:last] = 2 * d[first:last, None, None] * (
            partial[:, :, None] * partial[:, None, :].conj()).real
    gram[:-1] = np.cumsum(gram[-2::-1], axis=0)[::-1]
    cell_mass = float(sum(occupation[species == kind].sum() * record['Mass']
                          for kind, record in zip(types, records)))
    if cell_mass <= 0:
        raise ValueError('At least one occupied nuclear site is required')
    source = {'version': VERSION, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cif_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'nuclear_expansion_source_sha256': hashlib.sha256(Path(__file__).with_name('nuclear.py').read_bytes()).hexdigest(),
        'neutron_amplitude_source_sha256': hashlib.sha256(Path(__file__).with_name('isotopes.py').read_bytes()).hexdigest(),
        'neutron_records_sha256': hashlib.sha256(json.dumps(dict(zip(types, records)), sort_keys=True).encode()).hexdigest(),
        'minimum_wavelength_a': minimum_wavelength_a, 'd_cutoff_a': cutoff,
        'friedel_half_rows': len(d), 'full_reciprocal_rows': 2 * len(d),
        'independent_site_multiplicities': counts, 'exact_atom_types': types,
        'all_displacement_tensors_preserved': True,
        'partial_occupancy_diffuse_scattering_included': False,
        'includes_incoherent_inelastic_magnetic_or_multiple_scattering': False,
        'experimental_accuracy_certified': False}
    d.setflags(write=False)
    gram.setflags(write=False)
    return PowderBraggKernel(types, records, d, gram, volume, cell_mass,
        float(occupation.sum()), float(minimum_wavelength_a), source)
