from pathlib import Path

from scripts.ndip_lattice import lattice_from_records, lattice_quality, final_project_key


def phase(name, scale, mass, cell):
    return [name, {"General": {"Cell": [False, *cell], "Mass": mass, "SGData": {"SpGrp": "P 1"}},
        "Histograms": {"PWDR test": {"Scale": [scale, True], "Use": True}}}]


def test_cells_and_weight_fractions_use_gsas_mass_normalization():
    cell = [4, 4, 4, 90, 90, 90, 64]
    records = [[["Phases", {}], phase("a", 1, 600, cell), phase("b", 1.5, 160, [5, 5, 5, 90, 90, 90, 125])]]
    rows, rwp = lattice_from_records(records, source="final.gpx")
    assert rows[0]["weight_percent"] == 100 * 600 / 840
    assert rows[1]["a"] == 5
    assert rows[1]["a_esd"] is None
    assert rows[1]["uncertainty_status"] == "not_estimated"
    assert lattice_quality(rows) == []
    rows[0]["weight_percent"] = 0
    assert lattice_quality(rows)[0]["code"] == "vanishing_phase"
    rows[0]["a"] = -1
    assert lattice_quality(rows)[0]["code"] == "invalid_cell"


def test_final_project_prefers_final_polish_then_highest_accepted_pass():
    files = [Path(p) for p in ["pattern_project.gpx", "seq_pass1_kept_polished.gpx", "seq_pass10_kept_polished.gpx"]]
    assert max(files, key=final_project_key).name == "seq_pass10_kept_polished.gpx"
    assert max([*files, Path("seq_final_main_polished.gpx")], key=final_project_key).name == "seq_final_main_polished.gpx"
    assert max([*files,Path('seq_final_joint_polished.gpx'),Path('seq_final_accepted_model.gpx')],key=final_project_key).name=='seq_final_accepted_model.gpx'
