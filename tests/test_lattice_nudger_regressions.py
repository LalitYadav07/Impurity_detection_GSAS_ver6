import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from pymatgen.core import Lattice, Structure

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.lattice_nudger import (
    LatticeNudger,
    NDSticksQ,
    _apply_params_from_vector,
    _choose_best_lattice_index,
    _clip_q_residual_window,
    _free_param_names,
    _lattice_deviation_metric,
    _lattice_constraint_violations,
    _pack_vector_from_lattice,
    infer_constraints,
)


class Stage4WorkerPolicyTests(unittest.TestCase):
    def test_stage4_seed_defaults_to_deterministic_zero(self):
        nudger = LatticeNudger.__new__(LatticeNudger)
        LatticeNudger.__init__(nudger, db_loader=None, radiation="neutron")
        self.assertEqual(nudger.random_seed, 0)

    def test_lattice_near_tie_prefers_closest_cell(self):
        idx, tie_count = _choose_best_lattice_index(
            scores=[0.90000, 0.90035, 0.89900],
            deviations=[0.1, 5.0, 0.0],
            score_tol=0.0005,
        )
        self.assertEqual(idx, 0)
        self.assertEqual(tie_count, 2)

    def test_lattice_non_tie_prefers_higher_score(self):
        idx, tie_count = _choose_best_lattice_index(
            scores=[0.90000, 0.90200],
            deviations=[0.0, 5.0],
            score_tol=0.0005,
        )
        self.assertEqual(idx, 1)
        self.assertEqual(tie_count, 1)

    def test_lattice_deviation_is_zero_for_starting_cell(self):
        lattice = Lattice.orthorhombic(5.0, 6.0, 7.0)
        self.assertEqual(_lattice_deviation_metric(lattice, lattice, 1.0, 3.0), 0.0)

    def test_outer_workers_are_capped_by_candidate_count(self):
        with patch.dict(os.environ, {}, clear=False), patch("scripts.lattice_nudger.os.cpu_count", return_value=256):
            workers = LatticeNudger._resolve_outer_workers(10)
        self.assertEqual(workers, 10)

    def test_outer_worker_env_override_still_caps_to_real_work(self):
        with patch.dict(os.environ, {"STAGE4_MAX_WORKERS": "64"}, clear=False), patch("scripts.lattice_nudger.os.cpu_count", return_value=256):
            workers = LatticeNudger._resolve_outer_workers(10)
        self.assertEqual(workers, 10)

    def test_inner_parallelism_is_disabled_when_outer_pool_is_active(self):
        with patch.dict(os.environ, {}, clear=False), patch("scripts.lattice_nudger.os.cpu_count", return_value=256):
            workers = LatticeNudger._resolve_inner_workers(50, allow_parallel=False)
        self.assertEqual(workers, 1)

    def test_inner_workers_are_capped_by_rep_count_when_enabled(self):
        with patch.dict(os.environ, {}, clear=False), patch("scripts.lattice_nudger.os.cpu_count", return_value=32):
            workers = LatticeNudger._resolve_inner_workers(3, allow_parallel=True)
        self.assertEqual(workers, 3)

    def test_zero_tolerance_short_circuits_candidate_generation(self):
        nudger = LatticeNudger.__new__(LatticeNudger)
        base_struct = Structure(Lattice.cubic(5.0), ["Si"], [[0.0, 0.0, 0.0]])
        cons = infer_constraints(base_struct, 225)

        with patch("scripts.lattice_nudger._generate_q_targets", side_effect=AssertionError("should not be called")):
            lattices = nudger._make_candidates_qsignature(
                base_struct,
                cons,
                reps=10,
                samples=5000,
                len_tol_pct=0.0,
                ang_tol_deg=0.0,
            )

        self.assertEqual(len(lattices), 1)
        self.assertAlmostEqual(lattices[0].a, 5.0)
        self.assertAlmostEqual(lattices[0].b, 5.0)
        self.assertAlmostEqual(lattices[0].c, 5.0)

    def test_rhombohedral_trigonal_ties_lengths_and_angles(self):
        base_struct = Structure(
            Lattice.from_parameters(5.0, 5.0, 5.0, 75.0, 75.0, 75.0),
            ["Si"],
            [[0.0, 0.0, 0.0]],
        )
        cons = infer_constraints(base_struct, 166)
        names = _free_param_names(cons)
        self.assertEqual(names, ["a", "alpha"])

        lat = _apply_params_from_vector(
            base_struct.lattice,
            cons,
            names,
            [5.2, 76.5],
        )

        self.assertAlmostEqual(lat.a, lat.b)
        self.assertAlmostEqual(lat.a, lat.c)
        self.assertAlmostEqual(lat.alpha, lat.beta)
        self.assertAlmostEqual(lat.alpha, lat.gamma)
        self.assertEqual(_lattice_constraint_violations(cons, lat), [])

    def test_monoclinic_keeps_observed_unique_gamma_axis(self):
        base_struct = Structure(
            Lattice.from_parameters(5.0, 6.0, 7.0, 90.0, 90.0, 105.0),
            ["Si"],
            [[0.0, 0.0, 0.0]],
        )
        cons = infer_constraints(base_struct, 14)
        names = _free_param_names(cons)
        self.assertIn("gamma", names)
        self.assertNotIn("beta", names)

        p0 = _pack_vector_from_lattice(base_struct.lattice, cons, names)
        p0[names.index("gamma")] = 107.0
        lat = _apply_params_from_vector(base_struct.lattice, cons, names, p0)

        self.assertAlmostEqual(lat.alpha, 90.0)
        self.assertAlmostEqual(lat.beta, 90.0)
        self.assertAlmostEqual(lat.gamma, 107.0)
        self.assertEqual(_lattice_constraint_violations(cons, lat), [])


class NDSticksQRegressionTests(unittest.TestCase):
    def test_high_q_window_does_not_shorten_internal_wavelength(self):
        sim = NDSticksQ(wavelength_ang=1.54)
        wl_eff = sim._effective_wavelength_for_q_window((1.0, 15.5))
        self.assertEqual(wl_eff, 1.54)

    def test_residual_window_clips_high_q_by_default_policy(self):
        q, r = _clip_q_residual_window([1.0, 7.9, 8.1, 15.0], [1, 2, 3, 4], q_max=8.0)
        self.assertEqual(q.tolist(), [1.0, 7.9])
        self.assertEqual(r.tolist(), [1.0, 2.0])

    def test_structure_cache_key_includes_chemistry(self):
        sim = NDSticksQ(wavelength_ang=1.54)
        lattice = Lattice.cubic(5.0)
        si = Structure(lattice, ["Si"], [[0.0, 0.0, 0.0]])
        ge = Structure(lattice, ["Ge"], [[0.0, 0.0, 0.0]])
        self.assertNotEqual(sim._get_structure_hash(si), sim._get_structure_hash(ge))

    def test_structure_cache_key_includes_fractional_coordinates(self):
        sim = NDSticksQ(wavelength_ang=1.54)
        lattice = Lattice.cubic(5.0)
        a = Structure(lattice, ["Si"], [[0.0, 0.0, 0.0]])
        b = Structure(lattice, ["Si"], [[0.25, 0.25, 0.25]])
        self.assertNotEqual(sim._get_structure_hash(a), sim._get_structure_hash(b))

    def test_hkl_reposition_table_produces_sorted_q_values(self):
        sim = NDSticksQ(wavelength_ang=1.54, radiation="neutron")
        structure = Structure(Lattice.cubic(3.0), ["Si"], [[0.0, 0.0, 0.0]])
        hkls, intensities = sim.simulate_hkl_intensity_table(structure, q_window=(1.0, 8.0))
        q, i = sim.reposition_hkl_intensities(hkls, intensities, Lattice.cubic(3.1), q_window=(1.0, 8.0))
        self.assertGreater(len(q), 0)
        self.assertEqual(len(q), len(i))
        self.assertTrue(all(q[j] <= q[j + 1] for j in range(len(q) - 1)))

    def test_fast_reposition_defaults_to_neutron_only(self):
        neutron_nudger = LatticeNudger.__new__(LatticeNudger)
        LatticeNudger.__init__(neutron_nudger, db_loader=None, radiation="neutron")
        self.assertTrue(neutron_nudger.fast_peak_reposition)

        xray_nudger = LatticeNudger.__new__(LatticeNudger)
        LatticeNudger.__init__(xray_nudger, db_loader=None, radiation="xray")
        self.assertFalse(xray_nudger.fast_peak_reposition)


if __name__ == "__main__":
    unittest.main()
