"""Import a nuclear ISODISTORT displacement model without rewriting its basis.

This is a preparatory adapter, not a symmetry-search or refinement executor.
The ordinary CIF origin converter cannot safely translate a mode-bearing CIF:
its mode offsets, parent coordinates and transformation metadata would also
need a consistent transformation. This adapter permits a name-only repair
after complete symmetry-operator equivalence in the supplied setting.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from gsas_cif_import import _gsas_operations, _operator_sets, add_verified_phase

VERSION = "radar-isodistort-import/v1"
_HM_TAGS = {"_symmetry_space_group_name_h-m", "_space_group_name_h-m_alt"}


def _replace_inline_names(text, short_symbol):
    """Change only single-line H-M scalars; retain all other bytes as text."""
    if not short_symbol or any(c in short_symbol for c in "\r\n\"'"):
        raise ValueError("Invalid GSAS-II Hermann–Mauguin name")
    output, replaced = [], 0
    for line in text.splitlines(keepends=True):
        match = re.match(r"^(\s*)(_\S+)([ \t]+)(.*?)(\r?\n)?$", line)
        if match and match[2].lower() in _HM_TAGS:
            scalar = re.fullmatch(r'''(?:"[^"]*"|'[^']*'|[^\s#;]+)([ \t]*(?:#.*)?)''', match[4])
            if scalar is None:
                raise ValueError("Unreadable H-M scalar needs explicit setting review")
            output.append(f'{match[1]}{match[2]}{match[3]}"{short_symbol}"{scalar[1]}{match[5] or ""}')
            replaced += 1
        else:
            if line.strip().lower() in _HM_TAGS:
                raise ValueError("Multi-line H-M field needs explicit setting review")
            output.append(line)
    if not replaced:
        raise ValueError("ISODISTORT child has no supported H-M scalar")
    return "".join(output)


def _blocks(document):
    return [document[key] for key in document.keys() if "_atom_site_fract_x" in document[key]]


def _contents(block, *, omit_names=False):
    """CIF values independently compared after rewriting; no mode-table edits."""
    return {str(tag).lower(): list(value) if isinstance(value, (list, tuple)) else str(value)
            for tag in block.keys()
            if not (omit_names and str(tag).lower() in _HM_TAGS)
            for value in [block[tag]]}


def _values(value):
    return [value] if isinstance(value, str) else list(value)


def prepare_isodistort_cif(source, cache_dir):
    """Return (private import copy, receipt), or fail without guessing a setting.

Only displacive nuclear models are supported here. Strain modes are recorded
but not compiled; magnetic, occupancy and rotational mode executors are separate
work. All non-H-M CIF values and every supplied symmetry operator must survive.
"""
    import CifFile
    from GSASII import GSASIIspc as G2spc
    from pymatgen.core.operations import SymmOp

    source = Path(source).resolve(strict=True)
    before = source.read_bytes()
    text = before.decode("utf-8", errors="strict")
    document = CifFile.ReadCif(str(source))
    blocks = _blocks(document)
    if len(document.keys()) != 1 or len(blocks) != 1:
        raise ValueError("Select exactly one ISODISTORT structural data block")
    block = blocks[0]
    counts = {kind: int(block.get(f"_iso_{kind}mode_number", "0"))
              for kind in ["displacive", "strain", "magnetic", "occupancy", "rotational"]}
    if any(value < 0 for value in counts.values()) or not counts['displacive']:
        raise ValueError("A positive native displacement-mode inventory is required")
    if any(counts[kind] for kind in ["magnetic", "occupancy", "rotational"]):
        raise ValueError("This adapter supports nuclear displacement modes only")
    labels = [str(value) for value in _values(block.get('_iso_displacivemode_label', []))]
    norm_values = [float(value) for value in _values(block.get('_iso_displacivemodenorm_value', []))]
    if (len(labels) != counts['displacive'] or len(set(labels)) != len(labels)
            or len(norm_values) != counts['displacive']
            or any(not math.isfinite(n) or n <= 0 for n in norm_values)):
        raise ValueError("Displacement-mode labels and positive normalization must be complete")
    number_values = [int(block[tag]) for tag in ["_space_group_IT_number", "_symmetry_Int_Tables_number"] if tag in block]
    if not number_values or len(set(number_values)) != 1 or not 1 <= number_values[0] <= 230:
        raise ValueError("A consistent International Tables number is required")
    number = number_values[0]
    error, sgdata = G2spc.SpcGroup(G2spc.spgbyNum[number])
    if error:
        raise ValueError("The installed GSAS-II cannot resolve this space group")
    operation_tag = next((tag for tag in ["_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz"] if tag in block), None)
    if operation_tag is None:
        raise ValueError("Explicit child symmetry operators are required")
    expressions = block[operation_tag]
    if isinstance(expressions, str):
        expressions = [expressions]
    supplied = []
    for expression in expressions:
        op = SymmOp.from_xyz_str(expression)
        supplied.append((op.rotation_matrix, op.translation_vector))
    if _operator_sets(supplied) != _operator_sets(_gsas_operations(sgdata)):
        raise ValueError("Mode-bearing CIF requires an origin/basis transformation; name-only repair withheld")

    derived_text = _replace_inline_names(text, sgdata['SpGrp'])
    # Parse a temporary private copy before publishing the derived artifact.
    directory = Path(cache_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(before).hexdigest()
    destination = directory / (source.stem + '__native_' + digest[:12] + '.cif')
    if destination == source:
        raise ValueError("The authoritative child must remain separate from its import copy")
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(derived_text, encoding='utf-8')
    try:
        derived_blocks = _blocks(CifFile.ReadCif(str(temporary)))
        if len(derived_blocks) != 1 or _contents(block, omit_names=True) != _contents(derived_blocks[0], omit_names=True):
            raise ValueError("Name normalization changed structural or mode content")
        if source.read_bytes() != before:
            raise ValueError("Authoritative child changed during import preparation")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    mode_contents = {k:v for k,v in _contents(block).items() if k.startswith('_iso_')}
    receipt = dict(schema=VERSION, source=str(source), source_sha256=digest,
                   derived=str(destination), derived_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                   space_group_number=number, imported_symbol=sgdata['SpGrp'],
                   symmetry_operators_unchanged=True, non_name_values_unchanged=True,
                   origin_changed=False, basis_changed=False, source_mode_counts=counts,
                   source_mode_labels=labels, source_mode_normalization=norm_values,
                   mode_content_sha256=hashlib.sha256(json.dumps(mode_contents, sort_keys=True).encode()).hexdigest(),
                   scope='nuclear displacement-mode import; strain modes not compiled')
    destination.with_suffix('.import.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return destination, receipt


def add_isodistort_phase(project, source, **kwargs):
    """Import and verify GSAS-II's exact native displacement-mode inventory."""
    import numpy as np
    directory = Path(project.filename).resolve().parent / 'isodistort_import'
    derived, receipt = prepare_isodistort_cif(source, directory)
    phase = add_verified_phase(project, derived, **kwargs)
    iso = phase.data.get('ISODISTORT', {})
    expected = receipt['source_mode_counts']['displacive']
    labels, variables = iso.get('IsoModeList', []), iso.get('G2ModeList', [])
    norms = iso.get('NormList', [])
    if not (len(labels) == len(variables) == len(norms) == expected):
        raise ValueError("GSAS-II did not import the complete displacement-mode basis")
    if list(labels) != receipt['source_mode_labels'] or not np.allclose(
            norms, receipt['source_mode_normalization'], rtol=1e-10, atol=1e-12):
        raise ValueError("GSAS-II changed native mode identities or normalization")
    forward = np.asarray(iso.get('Mode2VarMatrix', []), dtype=float)
    inverse = np.asarray(iso.get('Var2ModeMatrix', []), dtype=float)
    if (forward.shape != (expected, expected) or inverse.shape != (expected, expected)
            or not np.all(np.isfinite(forward)) or not np.all(np.isfinite(inverse))
            or not np.allclose(inverse @ forward, np.eye(expected), rtol=1e-8, atol=1e-8)):
        raise ValueError("GSAS-II displacement-mode coordinate mapping is incomplete or singular")
    receipt.update(native_mode_labels=list(labels), native_mode_variables=[str(v) for v in variables],
                   native_normalization=[float(n) for n in norms], native_displacement_inventory_verified=True,
                   native_coordinate_mapping_invertible=True)
    derived.with_suffix('.import.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return phase, receipt
