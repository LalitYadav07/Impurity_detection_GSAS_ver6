"""Native independent moments mapped to physical Cartesian site moments."""
from dataclasses import dataclass
import copy
import numpy as np


@dataclass(frozen=True)
class MomentBasis:
    atoms: tuple
    components: np.ndarray
    cartesian: np.ndarray


def from_native(native, reference, indices, *, nuisance=()):
    """Extract the same checked map from a direct native forward adapter."""
    def map_values(values):
        native._check()
        native.parms.update(zip(native.vary, values))
        native.maps.Dict2Map(native.parms)
    return derive(native.histo_phases[1], native.parms, native.vary, reference,
        indices, map_values=map_values, lattice=native.lattice, nuisance=nuisance)


def derive(phases, parameters, variables, reference, indices, *, map_values,
           lattice, nuisance=()):
    """Extract and check the native homogeneous constraint map.

    This maps the native asymmetric atoms, not a common supercell or k star.
    Sharing between different embeddings additionally requires an explicit
    physical-site correspondence and native forward qualification.
    """
    reference = np.asarray(reference, float)
    indices = list(indices)
    if (reference.shape != (len(variables),) or not np.isfinite(reference).all()
            or not indices or len(set(indices)) != len(indices)
            or any(i < 0 or i >= len(variables) for i in indices)):
        raise ValueError('Complete finite native reference and independent moments required')
    atoms, units = [], []
    for name, phase in phases.items():
        general = phase['General']
        if general['Type'] != 'magnetic':
            continue
        if general.get('Modulated'):
            raise ValueError('Modulated magnetic profiles need a different adapter')
        cell = np.asarray(general['Cell'][1:7], float)
        unit = lattice.cell2AB(cell)[0]/cell[:3][None, :]
        for index, _ in enumerate(phase['Atoms']):
            atoms.append((name, index, tuple(f"{phase['pId']}::AM{a}:{index}" for a in 'xyz')))
            units.append(unit)
    if not atoms:
        raise ValueError('The source has no native magnetic atoms')

    def moments():
        return np.asarray([[parameters[k] for k in keys] for _, _, keys in atoms], float)

    zero = reference.copy()
    zero[indices] = 0.
    try:
        map_values(zero)
        if not np.allclose(moments(), 0., rtol=0, atol=1e-10):
            raise ValueError('Fixed nonzero magnetic moments are outside the homogeneous starting-point adapter')
        dynamic = {k for _, _, keys in atoms for k in keys} | set(nuisance)
        dynamic.update(variables[i] for i in indices)
        fixed = {k: copy.deepcopy(v) for k, v in parameters.items() if k not in dynamic}
        columns = []
        for i in indices:
            point = zero.copy()
            point[i] = 1.
            map_values(point)
            for key, value in fixed.items():
                try:
                    same = np.allclose(parameters[key], value, rtol=0, atol=1e-10, equal_nan=True)
                except (TypeError, ValueError):
                    same = parameters[key] == value
                if not same:
                    raise ValueError('An independent moment variable also changes a fixed parameter')
            columns.append(moments())
        components = np.stack(columns, axis=-1)
        cartesian = np.einsum('aij,ajk->aik', np.asarray(units), components)
        map_values(reference)
        if not np.allclose(components @ reference[indices], moments(), rtol=1e-9, atol=1e-10):
            raise ValueError('The native magnetic constraints are not the checked homogeneous linear map')
        if not np.isfinite(cartesian).all():
            raise ValueError('Nonfinite physical moment basis')
        components.flags.writeable = cartesian.flags.writeable = False
        return MomentBasis(tuple(atoms), components, cartesian)
    finally:
        map_values(reference)
