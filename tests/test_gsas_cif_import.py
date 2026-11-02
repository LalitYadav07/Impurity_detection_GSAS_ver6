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


def test_reviewed_occupancy_receipt_survives_native_gpx_save_and_reload(tmp_path, monkeypatch):
    pytest.importorskip("pymatgen")
    G2 = pytest.importorskip("GSASII.GSASIIscriptable")
    import json
    import gsas_cif_import
    from cif_occupancy import AUTH_ENV, make_review, write_review
    from test_cif_occupancy_policy_characterization import HEADER

    raw = (HEADER + 'C1 C 0 0 0 0.6\nLi1 Li 0 0 0 0.40066\nO1 O .3 .4 .5 1\n').encode()
    source = tmp_path / 'reviewed.cif'
    source.write_bytes(raw)
    review = make_review(raw, .001, 'Reviewed synthetic source rounding, retain unchanged')
    write_review(source, review)
    ledger = tmp_path / 'authorization.json'
    ledger.write_text(json.dumps({'root': str(tmp_path), 'approved_policies': {review['policy_sha256']: review}}))
    monkeypatch.setenv(AUTH_ENV, str(ledger))
    project = G2.G2Project(newgpx=str(tmp_path / 'reviewed.gpx'))
    phase = gsas_cif_import.add_verified_phase(project, source, phasename='reviewed', histograms=[])
    native_review = gsas_cif_import._occupancy_review_module()
    receipt = phase.data[native_review.DATA_KEY]
    project.save()
    saved = (tmp_path / 'reviewed.gpx').read_bytes()
    reopened = G2.G2Project(gpxfile=str(tmp_path / 'reviewed.gpx')).phase('reviewed')
    assert native_review.validate_receipt(reopened) == receipt
    assert reopened.atom('C1').occupancy == .6
    assert reopened.atom('Li1').occupancy == .40066
    cx, ct, _, _ = reopened.data['General']['AtomPtrs']
    oxygen = next(row for row in reopened.data['Atoms'] if row[ct-1] == 'O1')
    oxygen[cx] += .01
    native_review.validate_receipt(reopened)  # Unrelated oxygen motion is allowed.
    lithium = next(row for row in reopened.data['Atoms'] if row[ct-1] == 'Li1')
    lithium[cx+3] += 1e-8
    with pytest.raises(ValueError, match='exactly held'):
        native_review.validate_receipt(reopened)
    assert source.read_bytes() == raw
    assert (tmp_path / 'reviewed.gpx').read_bytes() == saved


def test_final_gpx_snapshot_retains_recipe_identity_after_allowed_oxygen_and_cell_changes(tmp_path, monkeypatch):
    pytest.importorskip('pymatgen')
    G2 = pytest.importorskip('GSASII.GSASIIscriptable')
    import json
    from cif_occupancy import AUTH_ENV, make_review, write_review
    from gsas_cif_import import add_verified_phase
    from gpx_recipe.snapshot import snapshot
    from test_cif_occupancy_policy_characterization import HEADER

    raw = (HEADER + 'C1 C 0 0 0 0.6\nLi1 Li 0 0 0 0.40066\nO1 O .3 .4 .5 1\n').encode()
    cif = tmp_path / 'reviewed.cif'
    cif.write_bytes(raw)
    policy = make_review(raw, .001, 'Retain source rounding in the fixed shared site')
    write_review(cif, policy)
    ledger = tmp_path / 'activation.json'
    ledger.write_text(json.dumps({'root':str(tmp_path),'approved_policies':{policy['policy_sha256']:policy}}))
    monkeypatch.setenv(AUTH_ENV,str(ledger))
    profile = tmp_path / 'cw.instprm'
    profile.write_text('#Bank 1: GSAS-II instrument parameter file\nType:PXC\nBank:1\n'
                       'Lam:1.54\nU:.1\nV:0\nW:.1\nZero:0\nX:0\nY:0\n'
                       'SH/L:.002\nPolariz:.99\nAzimuth:0\n')
    first = tmp_path / 'first.gpx'
    project = G2.G2Project(newgpx=str(first))
    phase = add_verified_phase(project,cif,phasename='reviewed',histograms=[])
    histogram = project.add_simulated_powder_histogram('synthetic',str(profile),10.,80.,Npoints=32,phases=[phase])
    histogram.data['data'][1][1][:] = 100.
    project.save()
    # Production publication uses this exact cosmetic renamer after import.
    from ndip_outputs import _rewrite_gpx_phase_names
    pretty = 'C0.6 Li0.40066 O (SG P1, 1)'
    assert _rewrite_gpx_phase_names(first, {'reviewed': pretty})
    project = G2.G2Project(gpxfile=str(first))
    phase = project.phase(pretty)
    before = snapshot(first)
    # Stand in for a completed allowed refinement, then really save and read
    # the resulting GPX through the complete native snapshot path.
    cx, ct, _, cia = phase.data['General']['AtomPtrs']
    oxygen = next(row for row in phase.data['Atoms'] if row[ct-1] == 'O1')
    oxygen[cx] += .01
    oxygen[cia+1] += .005
    phase.data['General']['Cell'][1] = 4.1
    phase.data['General']['Cell'][7] = 65.6
    final = tmp_path / 'final.gpx'
    project.save(str(final))
    after = snapshot(final)
    assert before['model_signature'] == after['model_signature']
    assert before['source']['sha256'] != after['source']['sha256']
    assert after['phases'][0]['occupancy_review'] == before['phases'][0]['occupancy_review']
    assert after['phases'][0]['atoms'][2]['coordinates'] != before['phases'][0]['atoms'][2]['coordinates']
    assert cif.read_bytes() == raw


def test_reviewed_native_origin_conversion_is_reproved_from_the_source_model(tmp_path, monkeypatch):
    pytest.importorskip('pymatgen')
    G2 = pytest.importorskip('GSASII.GSASIIscriptable')
    from GSASII import GSASIIspc
    from pymatgen.core.operations import SymmOp
    from cif_occupancy import AUTH_ENV, make_review, write_review
    import gsas_cif_import
    import json

    target = gsas_cif_import._gsas_operations(GSASIIspc.SpcGroup('P 21/c')[1])
    shift = np.array([.25, 0., 0.])
    operations = [SymmOp.from_rotation_and_translation(r, t - shift + np.asarray(r) @ shift).as_xyz_str()
                  for r, t in target]
    text = """data_origin
_cell_length_a 5
_cell_length_b 6
_cell_length_c 7
_cell_angle_alpha 90
_cell_angle_beta 105
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P 21/c'
_space_group_IT_number 14
loop_
_space_group_symop_operation_xyz
""" + ''.join("'" + operation + "'\n" for operation in operations) + """loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C1 C .1 .2 .3 .6
Li1 Li .1 .2 .3 .40066
O1 O .2 .3 .4 1
"""
    source = tmp_path / 'origin.cif'
    source.write_text(text)
    policy = make_review(source.read_bytes(), .001, 'Retain reviewed source in its proved origin setting')
    write_review(source, policy)
    ledger = tmp_path / 'authorization.json'
    ledger.write_text(json.dumps({'root':str(tmp_path),'approved_policies':{policy['policy_sha256']:policy}}))
    monkeypatch.setenv(AUTH_ENV,str(ledger))
    project = G2.G2Project(newgpx=str(tmp_path / 'origin.gpx'))
    phase = gsas_cif_import.add_verified_phase(project, source, phasename='origin', histograms=[])
    module = gsas_cif_import._occupancy_review_module()
    assert module.validate_receipt(phase)
    assert list((tmp_path / 'verified_cif_settings').glob('*.origin.json'))
    project.save()
    reopened = G2.G2Project(gpxfile=str(tmp_path / 'origin.gpx')).phase('origin')
    assert module.validate_receipt(reopened)
    assert source.read_text() == text
