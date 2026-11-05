"""Verify CIF origin settings and elemental inventories at GSAS-II import.

An origin-choice mismatch can change special-position multiplicities even
when the cell and space-group number appear correct. Only an exact origin
translation supported by the supplied symmetry operators is automated here.
Basis changes, missing symmetry evidence and changed compositions are never
repaired by changing atom occupancies or guessing a different structure.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

import numpy as np

VERSION = "gsas-cif-origin/1"
DENOMINATOR = 24


def _occupancy_review_module():
    """Load the shared pure validator in batch images and source checkouts."""
    try:
        from gpx_recipe import occupancy_review
        return occupancy_review
    except ModuleNotFoundError as exc:
        if exc.name != 'gpx_recipe':
            raise
        # Local scientific scripts need no installed NOVA web dependencies.
        path = Path(__file__).resolve().parents[1] / 'nova/src/gpx_recipe/occupancy_review.py'
        spec = importlib.util.spec_from_file_location('radar_native_occupancy_review', path)
        if spec is None or spec.loader is None:
            raise ValueError('The shared native occupancy-review validator is unavailable') from exc
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def elemental_counts(atom_types):
    """Aggregate charged GSAS-II species without changing the stored atom types."""
    result = defaultdict(float)
    for species, count in atom_types.items():
        match = re.fullmatch(r"([A-Z][a-z]?)(?:(?:[+-]\d*)|(?:\d+[+-]))?", species)
        if not match:
            raise ValueError("Unrecognized atom type in imported CIF; structure review required")
        amount = float(count)
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("Invalid atom count in imported CIF; structure review required")
        if amount:
            result[match[1]] += amount
    return dict(result)


def refresh_native_inventory(phase, setup_general):
    """Rebuild stale native totals after the importer's ionic normalization.

    GSAS-II can normalize an unsupported charge label in the atom list while
    overwriting (rather than accumulating) NoAtoms for repeated instances.
    Re-running native setup on the normalized records fixes bookkeeping. It
    must not change an atom or replace the independent composition check.
    """
    atoms=copy.deepcopy(phase['Atoms'])
    before=copy.deepcopy(phase['General']['NoAtoms'])
    setup_general(phase, '')
    if atoms != phase['Atoms']:
        raise ValueError('Native inventory refresh changed atom records; import review required')
    cx,ct,cs,_=phase['General']['AtomPtrs']
    expected=defaultdict(float)
    for atom in atoms:
        contribution=float(atom[cx+3])*float(atom[cs+1])
        if not math.isfinite(contribution) or contribution < 0:
            raise ValueError('Invalid atom population during inventory refresh')
        expected[atom[ct]]+=contribution
    actual=phase['General']['NoAtoms']
    if set(expected)!=set(actual) or any(not math.isclose(expected[k],actual[k],rel_tol=1e-10,abs_tol=1e-10) for k in expected):
        raise ValueError('Native totals disagree with unchanged atom populations')
    return dict(before=before,after=copy.deepcopy(actual),atom_records_unchanged=True)

def _operator_sets(operations):
    return _occupancy_review_module().operator_sets(operations)


def matching_origin_shift(source_operations, target_operations):
    """Use the same complete origin proof at import and native receipt readback."""
    return _occupancy_review_module().matching_origin_shift(source_operations, target_operations)


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
    source_bytes = source.read_bytes()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from gpx_recipe.cif_occupancy import load_authorized_structure
        original = load_authorized_structure(source)
    # Elemental aggregation already ignores oxidation states. Mutating each
    # species through pymatgen's setter would reject an explicitly reviewed
    # small excess, even though its unchanged values are the import reference.
    from gpx_recipe.cif_occupancy import authorized_review, raw_sites, species_inventory, species_key, _species
    expected = species_inventory(original)
    review = authorized_review(source)
    cache = Path(project.filename).resolve().parent / "verified_cif_settings"
    prepared = prepare_cif_for_gsas(source, cache)
    prepared_bytes = prepared.read_bytes()
    phase = project.add_phase(str(prepared), **kwargs)
    if source.read_bytes() != source_bytes or prepared.read_bytes() != prepared_bytes:
        raise ValueError("CIF changed during verified import; retry from an immutable reviewed source")
    inventory_refresh=None
    # The actual native project has already imported its scriptable module.
    # Keep strict CIF validation usable before GSAS-II is loaded.
    if type(project).__module__.startswith('GSASII'):
        from GSASII import GSASIIscriptable as G2sc
        inventory_refresh=refresh_native_inventory(phase.data,G2sc.SetupGeneral)
    actual = elemental_counts(phase.data['General']['NoAtoms'])
    e_sum, a_sum = sum(expected.values()), sum(actual.values())
    if not e_sum > 0 or not a_sum > 0 or set(expected) != set(actual) or any(
        not math.isclose(expected[k] / e_sum, actual[k] / a_sum, abs_tol=1e-6, rel_tol=1e-6) for k in expected
    ):
        raise ValueError("CIF import changed the relative elemental atom counts; refinement withheld for structure-setting review")
    if review is not None:
        if any(not math.isclose(expected[key],actual[key],rel_tol=0,abs_tol=1e-8) for key in expected):
            raise ValueError("CIF import changed absolute reviewed atom counts; refinement withheld")
        # Equal totals alone can hide redistribution or uniform normalization.
        # Compare the imported asymmetric rows with the proven prepared CIF.
        rows,_ = raw_sites(prepared_bytes)
        general = phase.data['General']
        if 'AtomPtrs' not in general or 'Atoms' not in phase.data:
            raise ValueError("Reviewed CIF import has no auditable atom-site inventory")
        cx,ct,_,_ = general['AtomPtrs']
        atoms = phase.data['Atoms']
        by_label = {str(atom[ct-1]):atom for atom in atoms}
        if len(by_label) != len(atoms) or set(by_label) != {row['label'] for row in rows}:
            raise ValueError("Reviewed CIF import changed the atom-site labels")
        for row in rows:
            atom = by_label[row['label']]
            element = species_key(_species(str(atom[ct])))
            if (element != species_key(_species(row['symbol']))
                or not math.isclose(float(atom[cx+3]),row['occupancy'],rel_tol=0,abs_tol=1e-10)
                or not np.allclose((np.asarray(atom[cx:cx+3],float)-row['xyz']+.5)%1-.5,0,rtol=0,atol=1e-8)):
                raise ValueError("CIF import changed reviewed site occupancies or positions; refinement withheld")
    cache.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": VERSION, "source": str(source),
               "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
               "imported_cif": str(prepared), "imported_cif_sha256": hashlib.sha256(prepared_bytes).hexdigest(),
               "native_inventory_refresh": inventory_refresh,
               "expected_elemental_counts": expected, "imported_elemental_counts": actual,
               "relative_elemental_inventory_preserved": True, "phase": phase.name}
    if review is not None:
        from gpx_recipe.cif_occupancy import model_fields
        receipt['occupancy_review']=review
        receipt['absolute_counts_and_site_values_preserved']=True
        native_review = _occupancy_review_module()
        source_rows, _ = raw_sites(source_bytes)
        phase.data[native_review.DATA_KEY] = native_review.make_receipt(
            phase, review, source_sha256=receipt['source_sha256'],
            imported_cif_sha256=receipt['imported_cif_sha256'], source_atoms=source_rows,
            source_model=model_fields(source_bytes, neutral_elements=True))
        native_review.validate_receipt(phase)
    target = cache / (source.stem + '__' + receipt['source_sha256'][:12] + '.import.json')
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    temporary.replace(target)
    return phase
