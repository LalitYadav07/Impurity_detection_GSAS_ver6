"""Origin-setting conversion must preserve the complete operator action."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gsas_cif_import import matching_origin_shift, _operator_sets, _shift_coordinate, elemental_counts


def p21c_operations():
    return [(np.eye(3), np.zeros(3)), (-np.eye(3), np.zeros(3)),
            (np.diag([-1, 1, -1]), np.array([0, .5, .5])),
            (np.diag([1, -1, 1]), np.array([0, .5, .5]))]


@pytest.mark.parametrize("shift", [[.25, -.25, 0], [.125, .25, .5], [1/3, -1/6, 0]])
def test_exact_origin_translation_maps_every_operator_without_a_basis_change(shift):
    target = p21c_operations()
    shift = np.array(shift)
    source = [(r, t - shift + r @ shift) for r, t in target]
    recovered = matching_origin_shift(source, target)
    transformed = [(r, t + recovered - r @ recovered) for r, t in source]
    assert _operator_sets(transformed) == _operator_sets(target)


def test_correct_setting_is_unchanged():
    np.testing.assert_array_equal(matching_origin_shift(p21c_operations(), p21c_operations()), [0, 0, 0])


def test_a_basis_change_or_nonmatching_symmetry_is_not_guessed():
    with pytest.raises(ValueError, match="basis/setting"):
        matching_origin_shift(p21c_operations()[:2], p21c_operations())


def test_inexact_unsupported_translations_are_rejected():
    with pytest.raises(ValueError, match="precision"):
        matching_origin_shift([(np.eye(3), [np.pi / 9, 0, 0])], [(np.eye(3), [0, 0, 0])])


def test_same_rotations_do_not_hide_an_incompatible_translation_group():
    with pytest.raises(ValueError, match="No exact"):
        matching_origin_shift([(np.eye(3), [.5, 0, 0])], [(np.eye(3), [0, 0, 0])])


def test_coordinate_esds_keep_their_absolute_precision_after_translation():
    assert _shift_coordinate("0.0946(2)", .25) == "0.3446(2)"
    assert _shift_coordinate("1.23(4)e-2", .25) == "0.2623(4)"
    assert _shift_coordinate("0.9999(12)", .25) == "0.2499(12)"
    with pytest.raises(ValueError, match="Unreadable"):
        _shift_coordinate("?", .25)


def test_species_charge_labels_do_not_create_false_composition_changes():
    assert elemental_counts({"Ba+2": 4, "Fe+2": 1, "Fe+3": 3, "O-2": 12}) == {"Ba": 4, "Fe": 4, "O": 12}
    with pytest.raises(ValueError, match="atom type"):
        elemental_counts({"unknown": 1})


def test_bootstrap_cif_passes_the_installed_gsas_import_guard(tmp_path):
    pytest.importorskip("pymatgen")
    G2sc = pytest.importorskip("GSASII.GSASIIscriptable")
    from gsas_cif_import import add_verified_phase
    from gsas_legacy_bridge import write_dummy_nacl_cif

    source = write_dummy_nacl_cif(str(tmp_path))
    project = G2sc.G2Project(newgpx=str(tmp_path / "bootstrap.gpx"))
    phase = add_verified_phase(project, source, phasename="bootstrap", histograms=[])
    assert elemental_counts(phase.data["General"]["NoAtoms"]) == {"Na": 4.0, "Cl": 4.0}
