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

from scripts.lattice_nudger import LatticeNudger, infer_constraints


class Stage4WorkerPolicyTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
