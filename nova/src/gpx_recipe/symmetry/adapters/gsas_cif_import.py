"""Verify CIF origin settings and elemental inventories at GSAS-II import.

An origin-choice mismatch can change special-position multiplicities even
when the cell and space-group number appear correct. Only an exact origin
translation supported by the supplied symmetry operators is automated here.
Basis changes, missing symmetry evidence and changed compositions are never
repaired by changing atom occupancies or guessing a different structure.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

VERSION = "gsas-cif-origin/1"
DENOMINATOR = 24


def elemental_counts(atom_types):
    """Aggregate charged GSAS-II species without changing the stored atom types."""
    result = defaultdict(float)
    for species, count in atom_types.items():
        match = re.fullmatch(r"([A-Z][a-z]?)(?:(?:[+-]\d*)|(?:\d+[+-]))?", species)
        if not match:
            raise ValueError("Unrecognized atom type in imported CIF; structure review required")
        result[match[1]] += float(count)
    return dict(result)


def _operator_sets(operations):
    groups = defaultdict(set)
    for rotation, translation in operations:
        rotation = np.asarray(rotation, dtype=float)
        translation = np.asarray(translation, dtype=float)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("A crystallographic operation needs a 3x3 rotation and three translations")
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            raise ValueError("Non-finite crystallographic operation")
        rounded_rotation = np.rint(rotation).astype(int)
        rounded_translation = np.rint(translation * DENOMINATOR).astype(int)
        if not np.allclose(rotation, rounded_rotation, atol=1e-8, rtol=0) or not np.allclose(
            translation * DENOMINATOR, rounded_translation, atol=1e-6, rtol=0
        ):
            raise ValueError("Unsupported crystallographic operation precision; manual setting review required")
        groups[tuple(rounded_rotation.flat)].add(tuple(rounded_translation % DENOMINATOR))
    if not groups:
        raise ValueError("No explicit crystallographic symmetry operations")
    return dict(groups)


def matching_origin_shift(source_operations, target_operations):
    """Find a verified fractional origin translation, without changing the basis.

The finite grid covers conventional rational origin choices. Every supplied
operator must match after conjugation, not just one generator or atom count.
Unrepresentable settings fail for review instead of using a near match.
"""
    source, target = _operator_sets(source_operations), _operator_sets(target_operations)
    if set(source) != set(target) or any(len(source[k]) != len(target[k]) for k in source):
        raise ValueError("CIF and GSAS-II require a basis/setting transformation, not a verified origin translation")
    if source == target:
        return np.zeros(3)
    candidates = np.indices((DENOMINATOR,) * 3).reshape(3, -1).T
    keys = sorted(source, key=lambda k: -np.linalg.matrix_rank(np.eye(3) - np.array(k).reshape(3, 3)))
    for key in keys:
        rotation = np.array(key).reshape(3, 3)
        offsets = candidates - candidates @ rotation.T
        keep = np.ones(len(candidates), dtype=bool)
        targets = np.array(sorted(target[key]))
        for translation in source[key]:
            transformed = (offsets + np.array(translation)) % DENOMINATOR
            keep &= np.any(np.all(transformed[:, None, :] == targets[None, :, :], axis=2), axis=1)
        candidates = candidates[keep]
        if not len(candidates):
            raise ValueError("No exact supported origin translation maps the supplied CIF operators to GSAS-II")
    signed = (candidates + DENOMINATOR // 2) % DENOMINATOR - DENOMINATOR // 2
    order = sorted(range(len(candidates)), key=lambda i: (int(np.dot(signed[i], signed[i])), tuple(signed[i])))
    shift = signed[order[0]] / DENOMINATOR
    translated = [(r, np.asarray(t) + shift - np.asarray(r) @ shift) for r, t in source_operations]
    if _operator_sets(translated) != target:
        raise ValueError("Origin transformation failed its complete symmetry-operator check")
    return shift


def _gsas_operations(sgdata):
    return [(sign * np.asarray(rotation), sign * np.asarray(translation) + np.asarray(center))
            for center in sgdata["SGCen"]
            for sign in ([1, -1] if sgdata["SGInv"] else [1])
            for rotation, translation in sgdata["SGOps"]]


def _shift_coordinate(token, shift):
    # Preserve coordinate uncertainty units when converting an exponent or
    # translating across the unit-cell boundary.
    text = str(token).strip()
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:\((\d+)\))?([eE][+-]?\d+)?", text)
    if not match:
        raise ValueError("Unreadable fractional coordinate; manual CIF review required")
    mantissa, uncertainty, exponent = match.groups()
    power = int(exponent[1:]) if exponent else 0
    value = (float(mantissa) * 10 ** power + float(shift)) % 1.0
    if uncertainty is not None:
        decimals = len(mantissa.partition(".")[2]) - power
        if not 0 <= decimals <= 14:
            raise ValueError("Coordinate uncertainty precision requires manual review")
        return f"{value:.{decimals}f}({uncertainty})"
    return f"{value:.12g}"


def prepare_cif_for_gsas(cif_file, cache_dir):
    """Return the original file or a proven origin-equivalent derived copy."""
    import CifFile
    from GSASII import GSASIIspc as G2spc
    from pymatgen.core.operations import SymmOp

    source = Path(cif_file).resolve(strict=True)
    document = CifFile.ReadCif(str(source))
    structures = [document[key] for key in document.keys() if "_atom_site_fract_x" in document[key]]
    if len(structures) != 1:
        raise ValueError("Select a CIF containing exactly one structural data block")
    block = structures[0]
    operation_tag = next((tag for tag in ["_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz"] if tag in block), None)
    if operation_tag is None:
        # The composition check below still applies. Without explicit supplied
        # operators there is no evidence for an automatic setting conversion.
        return source
    symbols = [block[tag] for tag in ["_space_group_name_H-M_alt", "_symmetry_space_group_name_H-M"] if tag in block]
    error, sgdata = G2spc.SpcGroup(str(symbols[0])) if symbols else (1, None)
    if error:
        number = next((int(block[tag]) for tag in ["_space_group_IT_number", "_symmetry_Int_Tables_number"] if tag in block), None)
        if number is None or not 1 <= number <= 230:
            raise ValueError("CIF space-group setting cannot be resolved")
        error, sgdata = G2spc.SpcGroup(G2spc.spgbyNum[number])
    if error:
        raise ValueError("CIF space group is not supported by the installed GSAS-II")
    expressions = block[operation_tag]
    if isinstance(expressions, str):
        expressions = [expressions]
    source_operations = []
    for text in expressions:
        operation = SymmOp.from_xyz_str(text)
        source_operations.append((operation.rotation_matrix, operation.translation_vector))
    target_operations = _gsas_operations(sgdata)
    shift = matching_origin_shift(source_operations, target_operations)
    if np.allclose(shift, 0, atol=1e-12, rtol=0):
        return source
    input_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / (source.stem + "__gsas_" + input_sha[:12] + ".cif")
    for i, tag in enumerate(["_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]):
        block[tag] = [_shift_coordinate(value, shift[i]) for value in block[tag]]
    # Conjugation above proves that this target operation set acts on the
    # translated coordinates exactly as the original set acted before.
    block[operation_tag] = [SymmOp.from_rotation_and_translation(r, t).as_xyz_str() for r, t in target_operations]
    id_tag = next((tag for tag in ["_space_group_symop_id", "_symmetry_equiv_pos_site_id"] if tag in block), None)
    if id_tag:
        block[id_tag] = [str(i + 1) for i in range(len(target_operations))]
    for tag in ["_space_group_name_H-M_alt", "_symmetry_space_group_name_H-M"]:
        if tag in block:
            block[tag] = sgdata["SpGrp"]
    text = document.WriteOut()
    copyright_lines = [line for line in source.read_text().splitlines()
                       if (line.startswith('#') and 'copyright' in line.lower()) or line.startswith('#(C)')]
    text = '\n'.join(copyright_lines) + '\n# Verified fractional origin translation; original retained.\n' + text
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(text)
    temporary.replace(destination)
    receipt = {"schema": VERSION, "source": str(source), "source_sha256": input_sha,
               "derived": str(destination), "derived_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
               "fractional_origin_shift": shift.tolist(), "basis_changed": False,
               "complete_symmetry_operator_equivalence": True, "atomic_occupancies_changed": False}
    destination.with_suffix('.origin.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return destination


def add_verified_phase(project, cif_file, **kwargs):
    """Import an equivalent setting, then reject any changed elemental inventory."""
    import warnings
    from pymatgen.core import Structure

    source = Path(cif_file).resolve(strict=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        original = Structure.from_file(source)
    original.remove_oxidation_states()
    expected = original.composition.get_el_amt_dict()
    cache = Path(project.filename).resolve().parent / "verified_cif_settings"
    prepared = prepare_cif_for_gsas(source, cache)
    phase = project.add_phase(str(prepared), **kwargs)
    actual = elemental_counts(phase.data['General']['NoAtoms'])
    e_sum, a_sum = sum(expected.values()), sum(actual.values())
    if not e_sum > 0 or not a_sum > 0 or set(expected) != set(actual) or any(
        not math.isclose(expected[k] / e_sum, actual[k] / a_sum, abs_tol=1e-6, rel_tol=1e-6) for k in expected
    ):
        raise ValueError("CIF import changed the relative elemental atom counts; refinement withheld for structure-setting review")
    cache.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": VERSION, "source": str(source),
               "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "imported_cif": str(prepared), "imported_cif_sha256": hashlib.sha256(prepared.read_bytes()).hexdigest(),
               "expected_elemental_counts": expected, "imported_elemental_counts": actual,
               "relative_elemental_inventory_preserved": True, "phase": phase.name}
    target = cache / (source.stem + '__' + receipt['source_sha256'][:12] + '.import.json')
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    temporary.replace(target)
    return phase
