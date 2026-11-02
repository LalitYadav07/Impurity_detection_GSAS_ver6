"""Candidate screening must export a cell derivative, not rounded atom rows."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cif_occupancy import (AUTH_ENV, load_authorized_structure, make_review,
                          model_signature, raw_sites, write_cell_derivative, write_review)
import gsas_main_phase_refiner as refiner

SOURCE = """data_example
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P 1'
_space_group_IT_number 1
loop_
_space_group_symop_operation_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_U_iso_or_equiv
C1 C .123456789 .25 .5 .6 .0123456789
Li1 Li .123456789 .25 .5 .40066 .0234567891
"""


def reviewed_source(tmp_path, monkeypatch, text=SOURCE):
    path = tmp_path / "source.cif"
    path.write_text(text)
    review = make_review(path.read_bytes(), .001, "Synthetic explicitly reviewed source")
    write_review(path, review)
    ledger = tmp_path / "authorization.json"
    ledger.write_text(json.dumps({"root": str(tmp_path), "approved_policies": {review["policy_sha256"]: review}}))
    monkeypatch.setenv(AUTH_ENV, str(ledger))
    return path


def fake_manager(monkeypatch, source, *, mutation=None):
    """Keep the real screen/export path; isolate only the numerical engine."""
    import gsas_core_infrastructure

    class Histogram:
        def set_refinements(self, *_args, **_kwargs): pass
        def clear_refinements(self, *_args, **_kwargs): pass
        def getdata(self, name): return np.array([1., 3., 2., 5.])

    class Phase:
        name = source.stem
        def __init__(self):
            rows, _ = raw_sites(source.read_bytes())
            self.data = {"General": {"AtomPtrs": [3, 1, 7, 9],
                "Cell": [False, 4., 4., 4., 90., 90., 90., 64.],
                "SGData": {"SpGrp": "P 1", "SGOps": [(np.eye(3), np.zeros(3))]}},
                "Atoms": [[r["label"], r["symbol"], "", *r["xyz"], r["occupancy"], "1", 1, "I", .0123456789, *([0.] * 6)] for r in rows]}
            self.exports = 0
        def set_HAP_refinements(self, *_args, **_kwargs): pass
        def HAPvalue(self, *_args, **_kwargs): pass
        def set_refinements(self, values): self.data["General"]["Cell"][0] = values.get("Cell", False)
        def export_CIF(self, target, **_kwargs):
            self.exports += 1
            Path(target).write_text(source.read_text().replace(".40066", ".4006"))

    phase, histogram = Phase(), Histogram()
    class Project:
        def __init__(self): self.data = {"Controls": {"data": {}}}
        def phases(self): return [phase]
        def histograms(self): return [histogram]
        def save(self): pass
        def do_refinements(self, _steps):
            if mutation: mutation(phase)

    class Manager:
        def __init__(self, *_args):
            self.project = Project()
            self.main_histogram = histogram
            self.main_phase = phase
        def create_project(self, **_kwargs): return True
        def add_histogram(self, *_args, **_kwargs): return True
        def add_phase_from_cif(self, path, _name):
            load_authorized_structure(path)
            return True

    monkeypatch.setattr(gsas_core_infrastructure, "GSASProjectManager", Manager)
    monkeypatch.setattr(refiner, "_init_gsas_process", lambda: None)
    monkeypatch.setattr(refiner, "apply_safe_limits", lambda _project: None)
    return phase


def run_screen(source, tmp_path):
    return refiner.compute_gsas_pearson_for_cif(
        "observed.xye", "profile.instprm", "xye", str(source), str(tmp_path), None, None,
        "test", cell_refine_min_r=2., return_diagnostics=True)


def test_screen_export_retains_activated_raw_values_and_reimport(tmp_path, monkeypatch):
    source = reviewed_source(tmp_path, monkeypatch)
    original = source.read_bytes()
    phase = fake_manager(monkeypatch, source)
    result = run_screen(source, tmp_path)
    target = Path(result["refined_cif_path"])
    assert model_signature(target.read_bytes()) == model_signature(original)
    assert sum(amount for _, amount in load_authorized_structure(target)[0].species.items()) == pytest.approx(1.00066)
    assert phase.exports == 0  # GSAS quick export rounds the immutable atom rows.
    assert source.read_bytes() == original
    receipt = json.loads(Path(str(target) + ".cell-derivation.json").read_text())
    assert receipt["atomic_sites_preserved"] is True


@pytest.mark.parametrize("field", ["coordinate", "occupancy", "adp", "species", "symmetry"])
def test_screen_never_hides_an_unexpected_atom_or_symmetry_change(tmp_path, monkeypatch, field):
    source = reviewed_source(tmp_path, monkeypatch)
    def mutate(phase):
        if field == "coordinate": phase.data["Atoms"][0][3] += .01
        elif field == "occupancy": phase.data["Atoms"][0][6] -= .01
        elif field == "adp": phase.data["Atoms"][0][10] += .01
        elif field == "species": phase.data["Atoms"][0][1] = "N"
        else: phase.data["General"]["SGData"]["SpGrp"] = "P -1"
    fake_manager(monkeypatch, source, mutation=mutate)
    with pytest.raises(ValueError, match="held atom|symmetry"):
        run_screen(source, tmp_path)
    assert not source.with_name("source_refined.cif").exists()


def test_unreviewed_overfill_still_fails_before_screening(tmp_path, monkeypatch):
    monkeypatch.delenv(AUTH_ENV, raising=False)
    source = tmp_path / "source.cif"
    source.write_text(SOURCE)
    phase = fake_manager(monkeypatch, source)
    with pytest.raises(ValueError, match="occupancy total"):
        run_screen(source, tmp_path)
    assert phase.exports == 0


def test_ordinary_natural_source_does_not_need_an_allowance(tmp_path, monkeypatch):
    monkeypatch.delenv(AUTH_ENV, raising=False)
    source = tmp_path / "source.cif"
    source.write_text(SOURCE.replace(".40066", ".4"))
    original = source.read_bytes()
    fake_manager(monkeypatch, source)
    target = Path(run_screen(source, tmp_path)["refined_cif_path"])
    assert model_signature(target.read_bytes()) == model_signature(original)
    assert sum(load_authorized_structure(target)[0].species.values()) == pytest.approx(1.)
    assert not Path(str(target) + ".occupancy-review.json").exists()


@pytest.mark.parametrize('symbols,neutral', [(('C', 'Li'), False), (('D', 'T'), False),
                                           (('D', 'T'), True), (('H', 'D'), True)])
def test_cell_writer_preserves_raw_precision_species_and_symmetry(tmp_path, monkeypatch, symbols, neutral):
    text = SOURCE.replace('C1 C ', 'C1 ' + symbols[0] + ' ').replace('Li1 Li ', 'Li1 ' + symbols[1] + ' ')
    source = reviewed_source(tmp_path, monkeypatch, text)
    original = source.read_bytes()
    target = tmp_path / 'derived.cif'
    proof = write_cell_derivative(source, target, [4.1, 4.2, 4.3, 90., 91., 92.],
                                  data_name='reviewed cell', neutral_elements=neutral)
    import CifFile
    before = CifFile.ReadCif(str(source))['example']
    after = CifFile.ReadCif(str(target))['reviewed_cell']
    for key in before.keys():
        if not key.startswith(('_cell_length_', '_cell_angle_')) and key != '_cell_volume':
            assert before[key] == after[key], key
    assert float(after['_cell_length_a']) == 4.1
    assert proof['source_model_signature'] == proof['derived_model_signature']
    assert sum(amount for _, amount in load_authorized_structure(target)[0].species.items()) == pytest.approx(1.00066)
    assert source.read_bytes() == original


@pytest.mark.parametrize('cell', [[-1,4,4,90,90,90], [4,4,4,0,90,90], [4,4,float('nan'),90,90,90], [4,4,4,10,10,170]])
def test_invalid_cell_leaves_previous_output_and_source_unchanged(tmp_path, monkeypatch, cell):
    source = reviewed_source(tmp_path, monkeypatch)
    original = source.read_bytes()
    target = tmp_path / 'derived.cif'
    target.write_text('existing output')
    with pytest.raises(ValueError):
        write_cell_derivative(source, target, cell)
    assert target.read_text() == 'existing output'
    assert source.read_bytes() == original


def test_revoked_review_cannot_create_or_reuse_a_cell_derivative(tmp_path, monkeypatch):
    source = reviewed_source(tmp_path, monkeypatch)
    target = tmp_path / 'derived.cif'
    write_cell_derivative(source, target, [4,4,4,90,90,90])
    prior = target.read_bytes()
    monkeypatch.delenv(AUTH_ENV)
    with pytest.raises(ValueError, match='occupancy total'):
        write_cell_derivative(source, target, [4.1,4,4,90,90,90])
    assert target.read_bytes() == prior


def test_cell_writer_refuses_in_place_output(tmp_path, monkeypatch):
    source = reviewed_source(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='overwrite'):
        write_cell_derivative(source, source, [4,4,4,90,90,90])


def test_cell_writer_retains_anisotropic_displacement_tokens(tmp_path, monkeypatch):
    text = SOURCE + '''loop_
_atom_site_aniso_label
_atom_site_aniso_U_11
_atom_site_aniso_U_22
_atom_site_aniso_U_33
_atom_site_aniso_U_12
_atom_site_aniso_U_13
_atom_site_aniso_U_23
C1 .012345(2) .023456(3) .034567(4) .001234(1) .002345(1) .003456(1)
'''
    source = reviewed_source(tmp_path, monkeypatch, text)
    target = tmp_path / 'derived.cif'
    write_cell_derivative(source, target, [4.1,4.1,4.1,90,90,90])
    assert model_signature(source.read_bytes()) == model_signature(target.read_bytes())


def test_strict_new_source_does_not_inherit_previous_output_review(tmp_path, monkeypatch):
    source = reviewed_source(tmp_path, monkeypatch)
    target = tmp_path / 'derived.cif'
    write_cell_derivative(source, target, [4,4,4,90,90,90])
    strict = tmp_path / 'strict.cif'
    strict.write_text(SOURCE.replace('.40066','.4'))
    write_cell_derivative(strict, target, [4,4,4,90,90,90])
    assert not Path(str(target)+'.occupancy-review.json').exists()
    assert sum(amount for _,amount in load_authorized_structure(target)[0].species.items()) == pytest.approx(1.)
