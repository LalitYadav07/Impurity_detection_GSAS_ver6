"""Native, site-specific preflight for a declared commensurate MSG candidate.

No subgroup is selected here. A caller supplies an already enumerated candidate,
the parent setting fingerprint and explicit magnetic sites. GSAS-II constructs
the site stabilizers and permitted moment bases before any refinement is run.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from fractions import Fraction
from collections import deque

import numpy as np

from gsas_cif_import import _gsas_operations, _operator_sets


class NativeRepresentationError(ValueError):
    """A native construction needs a verified basis/site transfer, not a fit."""


def integer_supercell_translations(basis_rows, *, maximum_index=128):
    """Exact coset representatives of Z^3 / T.T Z^3 for an integer supercell.

    Enumerating a box from maximum matrix entries misses rotated cells such as
    a+b,-a+b,c. The finite quotient has abs(det(T)) classes; breadth-first
    generation uses exact cofactor residues and visits at most 3*index edges.
    Fractional conventional-to-primitive transformations require a separate
    primitive-lattice conversion and are deliberately not approximated here.
    """
    if len(basis_rows) != 3 or any(len(row) != 3 for row in basis_rows):
        raise ValueError('A supercell basis requires three rows')
    t = [[Fraction(str(v)) for v in row] for row in basis_rows]
    if any(v.denominator != 1 for row in t for v in row):
        raise ValueError('Exact integer supercell translations require an integer parent-lattice basis')
    t = [[int(v) for v in row] for row in t]
    # Cyclic minors already carry the cofactor parity.
    cof = [[t[(i+1)%3][(j+1)%3]*t[(i+2)%3][(j+2)%3]
             -t[(i+1)%3][(j+2)%3]*t[(i+2)%3][(j+1)%3] for j in range(3)] for i in range(3)]
    determinant = sum(t[0][j]*cof[0][j] for j in range(3))
    index = abs(determinant)
    if not 1 <= index <= maximum_index:
        raise ValueError('Singular supercell or declared expansion budget exceeded')
    sign = 1 if determinant > 0 else -1
    generators = [tuple(sign*cof[i][j] % index for i in range(3)) for j in range(3)]
    seen = {(0,0,0):(0,0,0)};queue=deque([(0,0,0)])
    while queue:
        residue=queue.popleft();translation=seen[residue]
        for axis,generator in enumerate(generators):
            new=tuple((residue[j]+generator[j])%index for j in range(3))
            if new not in seen:
                seen[new]=tuple(translation[j]+int(j==axis) for j in range(3));queue.append(new)
    if len(seen) != index:
        raise ValueError('Integer supercell quotient did not match its determinant')
    return dict(index=index,parent_translations=[list(v) for v in seen.values()],
                child_offsets=[[str(Fraction(v,index)) for v in key] for key in seen])


def direct_unit_metric(cell):
    """Metric for GSAS-II magnetic components along unit crystallographic axes.

    Matches the g / outer(sqrt(diag(g)), sqrt(diag(g))) convention in native
    GSAS-II MagStructureFactor2. Components are not generally Cartesian.
    """
    values = np.asarray(cell, dtype=float)
    if values.shape != (6,) or not np.all(np.isfinite(values)) or np.any(values[:3] <= 0) or np.any(values[3:] <= 0) or np.any(values[3:] >= 180):
        raise ValueError('A finite physical unit cell is required')
    alpha, beta, gamma = np.deg2rad(values[3:])
    metric = np.array([[1.,np.cos(gamma),np.cos(beta)],
                       [np.cos(gamma),1.,np.cos(alpha)],
                       [np.cos(beta),np.cos(alpha),1.]])
    if np.linalg.eigvalsh(metric).min() <= 1e-10:
        raise ValueError('The unit-cell metric is singular or nonphysical')
    return metric


def magnetic_magnitude(components, cell):
    moment = np.asarray(components, dtype=float)
    if moment.shape != (3,) or not np.all(np.isfinite(moment)):
        raise ValueError('Three finite native moment components are required')
    return float(np.sqrt(max(0., moment @ direct_unit_metric(cell) @ moment)))


def transform_native_moment(components, parent_cell, basis_rows):
    """Passive axial-vector conversion for a'=T a in native unit-axis components.

    m' = sign(det(T)) D' inv(T).T inv(D) m. D contains direct-axis lengths.
    The sign records axial parity for a handedness change. It is not inferred
    from the space-group name. Physical magnitude is preserved, including in
    nonorthogonal cells and supercells; origin translations do not rotate it.
    """
    cell = np.asarray(parent_cell, dtype=float)
    metric = direct_unit_metric(cell)
    moment = np.asarray(components, dtype=float)
    transform = np.asarray(basis_rows, dtype=float)
    if moment.shape != (3,) or transform.shape != (3,3) or not np.all(np.isfinite(moment)) or not np.all(np.isfinite(transform)):
        raise ValueError('Invalid native moment or basis matrix')
    determinant = float(np.linalg.det(transform))
    if abs(determinant) < 1e-10:
        raise ValueError('A moment cannot be transformed through a singular basis')
    lengths = np.diag(cell[:3])
    child_metric = transform @ lengths @ metric @ lengths @ transform.T
    child_lengths = np.sqrt(np.diag(child_metric))
    child = np.sign(determinant) * child_lengths * (np.linalg.inv(transform).T @ (moment/cell[:3]))
    child_unit_metric = child_metric/np.outer(child_lengths,child_lengths)
    if not np.isclose(moment @ metric @ moment, child @ child_unit_metric @ child, rtol=1e-9,atol=1e-10):
        raise ValueError('Moment conversion failed physical magnitude invariance')
    return child


def transform_k_vector(vector, basis_rows, *, maximum_denominator=96):
    """k' = T k for a' = T a and x' = inv(T).T x (GSAS-II convention).

    Rational strings/integers are required. Never silently approximate a fitted
    floating-point propagation vector to a nearby commensurate value.
    """
    def rational(value):
        if isinstance(value, (float, bool, np.floating)):
            raise ValueError('Use reviewed rational strings or integers for the k-vector basis conversion')
        result = Fraction(value)
        if result.denominator > maximum_denominator:
            raise ValueError('Propagation-vector denominator exceeds the declared commensurate budget')
        return result
    if len(vector) != 3 or len(basis_rows) != 3 or any(len(row) != 3 for row in basis_rows):
        raise ValueError('A k-vector has three components and a basis matrix has three rows')
    k = [rational(v) for v in vector]
    t = [[rational(v) for v in row] for row in basis_rows]
    determinant = (t[0][0]*(t[1][1]*t[2][2]-t[1][2]*t[2][1])
                   - t[0][1]*(t[1][0]*t[2][2]-t[1][2]*t[2][0])
                   + t[0][2]*(t[1][0]*t[2][1]-t[1][1]*t[2][0]))
    if not determinant:
        raise ValueError('A k-vector cannot be converted through a singular basis')
    transformed = [sum(row[j]*k[j] for j in range(3)) for row in t]
    if any(v.denominator > maximum_denominator for v in transformed):
        raise ValueError('Transformed k-vector exceeds the declared commensurate budget')
    return [str(v) for v in transformed]


def setting_fingerprint(sgdata):
    """Fingerprint full operations in this basis/origin, including centering."""
    groups = _operator_sets(_gsas_operations(sgdata))
    payload = [[[int(v) for v in rotation], [[int(v) for v in t] for t in sorted(translations)]]
               for rotation, translations in sorted(groups.items())]
    return hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()


def moment_basis(component_ids, coefficients):
    """Columns are independent native moment amplitudes, not axis checkboxes."""
    if len(component_ids) != 3 or len(coefficients) != 3:
        raise ValueError('A native moment basis requires three components')
    ids = [int(v) for v in component_ids]
    values = np.asarray(coefficients, dtype=float)
    if any(i < 0 for i in ids) or not np.all(np.isfinite(values)):
        raise ValueError('Invalid native magnetic-site constraint')
    columns = []
    for mode in sorted(set(ids) - {0}):
        column = np.array([values[i] if ids[i] == mode else 0. for i in range(3)])
        if not np.any(np.abs(column) > 1e-12):
            raise ValueError('A permitted magnetic basis has no nonzero coefficient')
        columns.append(column)
    return np.stack(columns, axis=1) if columns else np.zeros((3, 0))


def direction_in_basis(basis, direction, *, tolerance=1e-7):
    """A requested direction must obey all coupled/forbidden components."""
    basis = np.asarray(basis, dtype=float)
    vector = np.asarray(direction, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)) or np.linalg.norm(vector) <= 1e-12:
        raise ValueError('A finite, nonzero direction in the declared child basis is required')
    if basis.ndim != 2 or basis.shape[0] != 3 or not np.all(np.isfinite(basis)):
        raise ValueError('Invalid magnetic moment basis')
    if basis.shape[1] == 0:
        return False
    solution = np.linalg.lstsq(basis, vector, rcond=None)[0]
    return bool(np.linalg.norm(basis @ solution - vector) <= tolerance * np.linalg.norm(vector))


def preflight_magnetic_sites(parent, candidate, *, magnetic_sites, site_policy,
                             expected_parent_setting, fixed_direction_child=None,
                             maximum_cell_multiplier=64):
    """Return the transformed phase and a site-coverage report, without refining.

`magnetic_sites` contains exact parent labels. `all` requires every generated
orbit of every selected site to permit a nonzero moment (and requested direction).
`any` explicitly permits partial ordering and preserves every forbidden orbit.
Moment directions use GSAS-II's native child-component basis, not a presumed
Cartesian basis. Frame conversion and ionic form-factor choices belong to the
approved protocol, not an inference performed by this function.
"""
    from GSASII import GSASIIlattice as G2lat, GSASIIspc as G2spc, GSASIIElem as G2elem

    general = parent['General']
    if general['Type'] != 'nuclear' or general.get('Modulated'):
        raise ValueError('A nuclear, nonmodulated parent is required')
    fingerprint = setting_fingerprint(general['SGData'])
    if expected_parent_setting != fingerprint:
        raise ValueError('Candidate catalog belongs to another parent basis/origin; rediscover or explicitly transform it')
    if site_policy not in {'all', 'any'}:
        raise ValueError('Select all or any magnetic-site compatibility explicitly')
    sites = list(magnetic_sites)
    if not sites or len(set(sites)) != len(sites):
        raise ValueError('Select unique, explicit magnetic parent sites')
    cx, ct, _, _ = general['AtomPtrs']
    labels = [a[ct-1] for a in parent['Atoms']]
    if len(set(labels)) != len(labels) or any(s not in labels for s in sites):
        raise ValueError('A magnetic site label is absent or ambiguous')
    for site in sites:
        atom = parent['Atoms'][labels.index(site)]
        if not G2elem.GetMFtable([atom[ct]], [2.]):
            raise ValueError(f'No native magnetic form factor for {site}: {atom[ct]}')
    transform = np.asarray(candidate['transformation_matrix'], dtype=float)
    origin = np.asarray(candidate['origin_shift'], dtype=float)
    if transform.shape != (3, 3) or origin.shape != (3,) or not np.all(np.isfinite(transform)) or not np.all(np.isfinite(origin)):
        raise ValueError('Invalid candidate basis/origin')
    determinant = float(np.linalg.det(transform))
    if not math.isfinite(determinant) or abs(determinant) < 1e-10:
        raise ValueError('Singular candidate basis')
    if abs(determinant) > maximum_cell_multiplier:
        raise ValueError('Candidate exceeds the declared cell-expansion budget; deferred, not physically rejected')
    child = copy.deepcopy(parent)
    child['General']['SGData'] = G2lat.makeBilbaoPhase(
        [candidate['magnetic_space_group'], candidate.get('bns_lattice', '')], origin, transform, True)['SGData']
    child['General']['Cell'][1:] = list(G2lat.TransformCell(general['Cell'][1:7], transform))
    child, codes = G2lat.TransformPhase(parent, child, transform, origin, np.zeros(3), True)
    ccx, cct, _, _ = child['General']['AtomPtrs']
    rows = []
    represented = {site:0 for site in sites}
    for atom, code in zip(child['Atoms'], codes):
        match = re.match(r'^(\d+):', str(code))
        if not match or not 0 <= int(match[1]) < len(labels):
            raise NativeRepresentationError('GSAS-II returned an unresolvable parent-to-child atom map')
        parent_label = labels[int(match[1])]
        if parent_label not in sites:
            continue
        syms = G2spc.SytSym(atom[ccx:ccx+3], child['General']['SGData'])
        represented[parent_label] += int(syms[1])
        ids, coefficients = G2spc.GetCSpqinel(child['General']['SGData']['SpnFlp'], syms[3])
        basis = moment_basis(ids, coefficients)
        direction_allowed = fixed_direction_child is None or direction_in_basis(basis, fixed_direction_child)
        rows.append(dict(parent_site=parent_label, child_site=atom[cct-1], atom_code=str(code),
                         position=[float(v) for v in atom[ccx:ccx+3]], site_symmetry=str(syms[0]),
                         component_ids=[int(v) for v in ids], coefficients=[float(v) for v in coefficients],
                         independent_moment_count=basis.shape[1], basis=basis.tolist(),
                         nonzero_moment_allowed=bool(basis.shape[1]), direction_allowed=bool(direction_allowed),
                         compatible=bool(basis.shape[1] and direction_allowed)))
    for site in sites:
        atom = parent['Atoms'][labels.index(site)]
        multiplicity = G2spc.SytSym(atom[cx:cx+3],general['SGData'])[1]
        if not np.isclose(represented[site],multiplicity*abs(determinant),rtol=0,atol=1e-6):
            raise NativeRepresentationError('Native transformation did not preserve selected-site atomic density; a coset-complete construction is required')
    missing = sorted(set(sites) - {row['parent_site'] for row in rows})
    compatible = bool(rows) and not missing and (all(row['compatible'] for row in rows) if site_policy == 'all' else any(row['compatible'] for row in rows))
    return child, dict(schema='radar-magnetic-site-preflight/v1', candidate_id=candidate['candidate_id'],
                       parent_setting_sha256=fingerprint, site_policy=site_policy, selected_parent_sites=sites,
                       missing_parent_sites=missing, compatible=compatible, sites=rows,
                       component_basis='GSAS-II native child magnetic components', determinant=determinant)
