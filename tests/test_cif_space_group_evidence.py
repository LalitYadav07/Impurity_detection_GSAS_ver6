"""Library classification uses complete synthetic symmetry evidence only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.groups import SpaceGroup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from db_pack_builder import _preserve_cif_space_group, build_mini_db_pack


def cif(symbol="P 4/n m m z", *, number=None, operations=None):
    # Arbitrary synthetic Na/Cl positions; no measured structure or pattern.
    if operations is None:
        operations = [op.as_xyz_str() for op in SpaceGroup("P4/nmm:2").symmetry_ops]
    text = "data_synthetic_setting\n"
    if symbol is not None:
        text += f"_symmetry_space_group_name_H-M '{symbol}'\n"
    if number is not None:
        text += f"_space_group_IT_number {number}\n"
    text += """_cell_length_a 4.1
_cell_length_b 4.1
_cell_length_c 7.3
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
"""
    if operations:
        text += "loop_\n_space_group_symop_operation_xyz\n"
        text += "".join(f"'{value}'\n" for value in operations)
    return text + """loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Na1 Na 0.173 0.229 0.137 1
Cl1 Cl 0.313 0.191 0.411 1
"""


def classify(text):
    row = {"space_group": 123, "SG_symbol": "P4/mmm"}
    _preserve_cif_space_group(row, text)
    return row


def test_gsas_label_resolves_from_all_operators_without_rewriting_model():
    text = cif()
    original = text.encode()
    row = classify(text)
    assert text.encode() == original
    assert row["space_group"] == 129
    assert row["SG_symbol"] == "P 4/n m m z"
    provenance = row["symmetry_provenance"]
    assert provenance["declared_number"] is None
    assert provenance["declared_symbol"] == "P 4/n m m z"
    assert provenance["resolved_number"] == 129
    assert provenance["declared_symbol_status"] == "unrecognized by symbol parser; explicit operators authoritative"
    assert provenance["analyzer_number"] == 123
    assert provenance["operator_verification"]["exact_tabulated_settings"] == [{
        "hall_number": 409, "number": 129, "symbol": "P4/nmm",
        "choice": "2", "hall_symbol": "-P 4a 2a",
    }]


@pytest.mark.parametrize("number", [None, 129])
def test_unresolved_label_without_explicit_operations_is_not_authorized(number):
    with pytest.raises(ValueError, match="symbol needs setting review"):
        classify(cif(number=number, operations=[]))


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_incomplete_extra_or_duplicate_operations_fail(mutation):
    operations = [op.as_xyz_str() for op in SpaceGroup("P4/nmm:2").symmetry_ops]
    if mutation == "missing":
        operations.pop()
    elif mutation == "extra":
        operations.append("x+1/3,y,z")
    else:
        operations.append(operations[0])
    with pytest.raises(ValueError, match="complete tabulated|duplicates"):
        classify(cif(operations=operations))


def test_operator_and_explicit_number_contradiction_fails():
    with pytest.raises(ValueError, match="symbol and number disagree"):
        classify(cif(number=47))


def test_falsely_claimed_known_group_fails_even_when_number_agrees():
    with pytest.raises(ValueError, match="operators disagree with the declared space group"):
        classify(cif("Pmmm", number=47))


def test_explicit_origin_choice_cannot_be_overridden_by_operators():
    with pytest.raises(ValueError, match="operators disagree with the declared setting"):
        classify(cif("P4/nmm:1", number=129))


@pytest.mark.parametrize("symbol", ["P4/nmm", "P4/nmm:2"])
def test_standard_documented_name_preserves_explicit_origin_evidence(symbol):
    row = classify(cif(symbol, number=129))
    assert row["space_group"] == 129
    assert row["symmetry_provenance"]["operator_verification"]["operator_count"] == 16


def shifted_operations():
    shift = np.array([0.125, 0, 0])
    return [SymmOp.from_rotation_and_translation(
        op.rotation_matrix,
        op.translation_vector + shift - op.rotation_matrix @ shift,
    ).as_xyz_str() for op in SpaceGroup("P4/nmm:2").symmetry_ops]


def test_nonstandard_origin_requires_complete_declared_group_proof():
    operations = shifted_operations()
    with pytest.raises(ValueError, match="complete tabulated"):
        classify(cif(operations=operations))
    row = classify(cif("P4/nmm", operations=operations))
    assert row["space_group"] == 129
    assert not row["symmetry_provenance"]["operator_verification"]["exact_tabulated_settings"]
    with pytest.raises(ValueError, match="operators disagree"):
        classify(cif("P4/nmm", operations=operations[:-1]))


def test_two_operator_declarations_must_agree():
    text = cif() + "loop_\n_symmetry_equiv_pos_as_xyz\n'x,y,z'\n"
    with pytest.raises(ValueError, match="operator declarations disagree"):
        classify(text)


def test_ordinary_symbol_only_declaration_remains_supported():
    row = classify(cif("Pmmm", operations=[]))
    assert row["space_group"] == 47
    assert "operator_verification" not in row["symmetry_provenance"]


def test_failure_does_not_change_existing_classification():
    row = {"space_group": 123, "SG_symbol": "P4/mmm"}
    before = deepcopy(row)
    with pytest.raises(ValueError):
        _preserve_cif_space_group(row, cif("Pmmm", number=47))
    assert row == before


@pytest.mark.parametrize("symbol", ["P 4/n m m z", "P4/nmm"])
def test_near_rational_translation_cannot_round_into_exact_proof(symbol):
    operations = [op.as_xyz_str() for op in SpaceGroup("P4/nmm:2").symmetry_ops]
    index = next(i for i, value in enumerate(operations) if "1/2" in value)
    operations[index] = operations[index].replace("1/2", "0.50000000001", 1)
    with pytest.raises(ValueError, match="complete tabulated|disagree"):
        classify(cif(symbol, operations=operations))


@pytest.mark.parametrize("operation", ["1/2x,y,z", "2x,y,z", "x,x,z", "x+1/0,y,z", "x+nan,y,z"])
def test_invalid_rotation_or_translation_fails(operation):
    with pytest.raises(ValueError, match="integral|determinant|translation"):
        classify(cif(operations=[operation]))


def test_integer_translations_are_the_same_exact_operation():
    # Replace the identity, regardless of set iteration order.
    others = [op.as_xyz_str() for op in SpaceGroup("P4/nmm:2").symmetry_ops
              if np.array_equal(op.rotation_matrix, np.eye(3))
              and np.array_equal(op.translation_vector, np.zeros(3))]
    assert len(others) == 1
    operations = [op.as_xyz_str() for op in SpaceGroup("P4/nmm:2").symmetry_ops
                  if op.as_xyz_str() != others[0]] + ["x+1,y-2,z+3"]
    assert classify(cif(operations=operations))["space_group"] == 129


def test_unknown_label_is_not_represented_as_verified_declaration():
    row = classify(cif("unrecognized descriptive label"))
    provenance = row["symmetry_provenance"]
    assert provenance["declared_number"] is None
    assert "unrecognized" in provenance["declared_symbol_status"]
    assert provenance["source"] == "explicit CIF operators; original model and label retained"


def test_mini_library_preserves_source_bytes_and_numerical_fingerprint(tmp_path):
    results = []
    for name, text in [("alias", cif()), ("documented", cif("P4/nmm:2", number=129))]:
        source = tmp_path / (name + ".cif")
        original = text.encode()
        source.write_bytes(original)
        digest = hashlib.sha256(original).hexdigest()
        result = build_mini_db_pack([source], tmp_path / name, source_type="neutron")
        assert not result.failures and len(result.phase_ids) == 1
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        stored = json.loads(result.layout.original_json.read_text())
        entry = stored[result.phase_ids[0]]
        assert entry["cif_content"].encode() == original
        assert next(result.layout.cifs_dir.glob("*.cif")).read_bytes() == original
        with np.load(result.layout.profiles_npz, allow_pickle=False) as values:
            results.append({key: values[key].copy() for key in values.files if values[key].dtype.kind != "U"})
    assert results[0].keys() == results[1].keys()
    for key in results[0]:
        np.testing.assert_array_equal(results[0][key], results[1][key])
