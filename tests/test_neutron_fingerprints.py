"""Analytic isotope controls and natural-element regression checks."""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import DummySpecies, get_el_sp
from pymatgen.analysis.diffraction.neutron import ATOMIC_SCATTERING_LEN, NDCalculator
import neutron_fingerprints as nf
from database_catalog_builder import process_one_phase, simulate_topM_peaks, resolve_space_group_with_analyzer
from lattice_nudger import NDSticksQ

HEADER = """data_control
_cell_length_a 4.11
_cell_length_b 5.23
_cell_length_c 6.37
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
"""


def motif(isotope="H", mixed=False):
    component = {get_el_sp("H"): .4, get_el_sp(isotope): .6} if mixed else get_el_sp(isotope)
    return Structure(Lattice.orthorhombic(4.11, 5.23, 6.37), [component, "O"], [[.12, .23, .34], [.67, .51, .19]])


def analytic_ratio(structure, b):
    values = []
    for hkl in ([1, 0, 0], [0, 1, 0]):
        length = np.linalg.norm(structure.lattice.reciprocal_lattice_crystallographic.get_cartesian_coords(hkl))
        theta = math.asin(1.54 * length / 2)
        amplitude = b * np.exp(2j * np.pi * np.dot(hkl, structure[0].frac_coords))
        amplitude += ATOMIC_SCATTERING_LEN["O"] * np.exp(2j * np.pi * np.dot(hkl, structure[1].frac_coords))
        values.append(2 * abs(amplitude) ** 2 / (math.sin(theta) ** 2 * math.cos(theta)))
    return values[0] / values[1]


def measured_ratio(pattern):
    values = []
    for spacing in (4.11, 5.23):
        index = np.flatnonzero(np.isclose(pattern.d_hkls, spacing, atol=1e-10, rtol=0))
        assert len(index) == 1
        values.append(pattern.y[index[0]])
    return values[0] / values[1]


@pytest.mark.parametrize("species", ["H", "C", "Na", "Fe3+", "O"])
def test_natural_species_are_exact_upstream_regressions(species):
    structure = Structure(Lattice.cubic(4.2), [species, "O"], [[.1, .2, .3], [.4, .5, .6]])
    before = dict(ATOMIC_SCATTERING_LEN)
    expected = NDCalculator(wavelength=1.54).get_pattern(structure, scaled=False, two_theta_range=(5, 100))
    actual = nf.NeutronFingerprintCalculator(wavelength=1.54).get_pattern(structure, scaled=False, two_theta_range=(5, 100))
    np.testing.assert_array_equal(actual.x, expected.x)
    np.testing.assert_array_equal(actual.y, expected.y)
    assert actual.hkls == expected.hkls
    assert ATOMIC_SCATTERING_LEN == before


@pytest.mark.parametrize("isotope,coefficient", [("D", 6.681), ("T", 4.792)])
def test_isotope_pattern_matches_independent_structure_factor(isotope, coefficient):
    before = dict(ATOMIC_SCATTERING_LEN)
    structure = motif(isotope)
    pattern = nf.NeutronFingerprintCalculator(wavelength=1.54).get_pattern(structure, scaled=False, two_theta_range=(5, 100))
    assert measured_ratio(pattern) == pytest.approx(analytic_ratio(structure, coefficient), rel=1e-12)
    hydrogen = nf.NeutronFingerprintCalculator(wavelength=1.54).get_pattern(motif("H"), scaled=False, two_theta_range=(5, 100))
    assert measured_ratio(pattern) != pytest.approx(measured_ratio(hydrogen), rel=1e-4)
    assert ATOMIC_SCATTERING_LEN == before


def test_mixed_site_uses_coherent_amplitudes_without_combining_isotopes():
    structure = motif("D", mixed=True)
    assert sorted((s.name, n) for s, n in structure[0].species.items()) == [("D", .6), ("H", .4)]
    pattern = nf.NeutronFingerprintCalculator(wavelength=1.54).get_pattern(structure, scaled=False, two_theta_range=(5, 100))
    coefficient = .4 * ATOMIC_SCATTERING_LEN["H"] + .6 * 6.681
    assert measured_ratio(pattern) == pytest.approx(analytic_ratio(structure, coefficient), rel=1e-12)


def test_ordered_isotope_symmetry_is_not_averaged_or_standardized():
    structure = Structure(Lattice.cubic(4), ["H", get_el_sp("D")], [[0, 0, 0], [.5, .5, .5]])
    before = nf.structure_cache_key(structure)
    assert nf.isotope_preserving_space_group_info(structure) == ("Pm-3m", 221)
    number, symbol, _ = resolve_space_group_with_analyzer({}, structure)
    assert (number, symbol) == (221, "Pm-3m")
    assert nf.structure_cache_key(structure) == before
    with pytest.raises(ValueError, match="Symmetry refinement"):
        nf.NeutronFingerprintCalculator(symprec=.01).get_pattern(structure)


def test_unknown_atom_is_not_silently_assigned_hydrogen_scattering():
    structure = Structure(Lattice.cubic(4), [DummySpecies("Xx")], [[0, 0, 0]])
    with pytest.raises(ValueError, match="Unsupported neutron atom/isotope"):
        nf.NeutronFingerprintCalculator().get_pattern(structure)


def test_nudge_cache_and_hkl_table_separate_hydrogen_from_deuterium():
    simulator = NDSticksQ(wavelength_ang=1.54, two_theta_range=(5, 100))
    assert simulator._get_structure_hash(motif("H")) != simulator._get_structure_hash(motif("D"))
    hq, hi = simulator.simulate_QI(motif("H"))
    dq, di = simulator.simulate_QI(motif("D"))
    np.testing.assert_array_equal(hq, dq)
    assert not np.allclose(hi, di)
    _, htable = simulator.simulate_hkl_intensity_table(motif("H"))
    _, dtable = simulator.simulate_hkl_intensity_table(motif("D"))
    assert not np.allclose(htable, dtable)
    assert len(simulator._pattern_cache) == 4


def test_cache_identity_binds_generator_source_and_settings(monkeypatch):
    first = nf.structure_cache_key(motif("D"))
    monkeypatch.setattr(nf, "GENERATOR_VERSION", "different-version")
    assert nf.structure_cache_key(motif("D")) != first
    base = nf.request_metadata(HEADER, "neutron", 1.54, 5, 100, 100)["request_sha256"]
    assert nf.request_metadata(HEADER + "# changed", "neutron", 1.54, 5, 100, 100)["request_sha256"] != base
    assert nf.request_metadata(HEADER, "neutron", 1.55, 5, 100, 100)["request_sha256"] != base


def test_old_disk_fingerprint_is_rebuilt_before_resume(tmp_path):
    (tmp_path / "phases").mkdir()
    source = HEADER + "D1 D .12 .23 .34 1\nO1 O .67 .51 .19 1\n"
    old = tmp_path / "phases" / "source.npz"
    np.savez(old, hkls=np.zeros((1, 3)), I0=np.zeros(1))
    row, failure, _ = process_one_phase("source", {"cif_content": source}, str(tmp_path), 5, 100, 50, False, True, False, False, "neutron", 1.54)
    assert failure is None and row is not None
    with np.load(old) as payload:
        assert np.any(payload["I0"] > 0)
        receipt = json.loads(str(payload["fingerprint_provenance"].item()))
        assert receipt["generator"]["generator"] == nf.GENERATOR_VERSION
        assert str(payload["fingerprint_request_sha256"].item()) == receipt["request_sha256"]


def test_profile_bundle_does_not_relabel_inherited_base_fingerprints(tmp_path):
    import pandas as pd
    from db_pack import get_db_pack_layout
    from db_pack_builder import _write_profiles_bundle, resolve_simulation_settings
    layout = get_db_pack_layout(tmp_path)
    layout.phases_dir.mkdir()
    layout.profiles_dir.mkdir()
    receipt = nf.request_metadata(HEADER, "neutron", 1.54, 5, 100, 50)
    np.savez(layout.phases_dir / "new.npz", fingerprint_provenance=np.asarray(json.dumps(receipt)))
    _write_profiles_bundle(layout, np.zeros((2, 64)), pd.DataFrame({"id": ["legacy", "new"], "row": [0, 1]}),
                           resolve_simulation_settings("neutron"), ["new"])
    with np.load(layout.profiles_npz) as payload:
        provenance = json.loads(str(payload["fingerprint_provenance"].item()))
        assert set(provenance["phases"]) == {"new"}
        assert provenance["phases"]["new"]["request_sha256"] == receipt["request_sha256"]


def test_isotope_source_is_not_discarded_as_a_natural_element_duplicate(tmp_path):
    from db_pack_builder import collect_phase_inputs, _find_matching_base_phase_ids
    path = tmp_path / "deuterated.cif"
    path.write_text(HEADER + "D1 D .12 .23 .34 1\nO1 O .67 .51 .19 1\n")
    class UntouchedBase:
        @property
        def catalog(self):
            raise AssertionError("Isotope source must not enter isotope-blind duplicate matching")
    assert _find_matching_base_phase_ids(collect_phase_inputs([path])[0], base_loader=UntouchedBase()) == []


def test_common_policy_loader_to_actual_builder_preserves_mixed_sites(tmp_path, monkeypatch):
    """Integration dependency: the occupancy branch owns CIF parsing, not this fix."""
    policy_scripts = os.environ.get("RADAR_POLICY_SCRIPTS")
    if not policy_scripts:
        pytest.skip("Set RADAR_POLICY_SCRIPTS for the combined occupancy-loader integration")
    sys.path.append(policy_scripts)
    specification = importlib.util.spec_from_file_location("combined_policy_builder", Path(policy_scripts) / "database_catalog_builder.py")
    builder = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(builder)
    # These are exactly this patch's two builder integration points; all parsing,
    # preprocessing, phase construction and file writing run the shared loader.
    monkeypatch.setattr(builder, "NDCalculator", nf.NeutronFingerprintCalculator)
    monkeypatch.setattr(builder, "resolve_space_group_with_analyzer", resolve_space_group_with_analyzer)
    (tmp_path / "phases").mkdir()
    source = HEADER + "H1 H .12 .23 .34 .4\nD1 D .12 .23 .34 .6\nO1 O .67 .51 .19 1\n"
    row, failure, _ = builder.process_one_phase("mixed", {"cif_content": source}, str(tmp_path), 5, 100, 50, False, False, False, False, "neutron", 1.54)
    assert failure is None and row is not None
    with np.load(tmp_path / "phases" / "mixed.npz") as payload:
        _, _, _, expected = simulate_topM_peaks(motif("D", mixed=True), 5, 100, 50, "neutron", 1.54)
        np.testing.assert_allclose(payload["I0"], expected, rtol=1e-6)
