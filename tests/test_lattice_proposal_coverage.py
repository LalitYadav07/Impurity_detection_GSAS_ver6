"""Physical proposal coverage, independent of diffraction-pattern fitting."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pymatgen.core import Lattice, Structure

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from scripts import lattice_nudger as N


class BoundedProposalTests(unittest.TestCase):
    def propose(self, cell, group, *, reps=10, samples=800, length=4., angle=1.5):
        structure = Structure(Lattice.from_parameters(*cell), ["Si"], [[0, 0, 0]])
        constraints = N.infer_constraints(structure, group)
        nudger = N.LatticeNudger.__new__(N.LatticeNudger)
        nudger.random_seed = 0
        lattices = nudger._make_candidates_qsignature(structure, constraints,
            reps=reps, samples=samples, len_tol_pct=length, ang_tol_deg=angle)
        return structure, constraints, nudger, lattices

    def assert_valid(self, structure, constraints, lattices, length, angle):
        self.assertEqual(lattices[0].parameters, structure.lattice.parameters)
        q0 = N.hkl_Q_signature(structure.lattice)
        for lattice in lattices:
            self.assertTrue(N._within_lattice_tolerances(structure.lattice,
                lattice, length + 1e-10, angle + 1e-10, constraints)[0])
            self.assertEqual(N._lattice_constraint_violations(constraints, lattice), [])
            self.assertGreater(lattice.volume, 0)
            self.assertTrue(np.all(np.isfinite(lattice.matrix)))
            self.assertTrue(np.all(np.abs(N.hkl_Q_signature(lattice) / q0 - 1)
                                   <= N._QWIN_PCT / 100 + 1e-12))

    def test_exact_public_primitive_cells_have_real_diversity(self):
        # Cell values from packaged public MP CIFs; no experimental patterns.
        cells = [
            (5.99596971, 5.99607374, 6.28995714, 98.59360103, 98.70652902, 119.95495998),
            (5.99609119, 5.99612893, 6.89309483, 115.76925673, 90.00158510, 119.99010776),
        ]
        for cell in cells:
            for length in (.6, 4.):
                with self.subTest(cell=cell, length=length):
                    structure, constraints, nudger, lattices = self.propose(cell, 2, length=length)
                    self.assertEqual(len(lattices), 10)
                    self.assertEqual(nudger._last_lattice_sampling['generated'], 800)
                    self.assertGreater(nudger._last_lattice_sampling['accepted'], 700)
                    self.assertFalse(nudger._last_lattice_sampling['only_original_available'])
                    self.assertEqual(lattices[0].parameters, structure.lattice.parameters)
                    self.assertEqual(len({tuple(l.parameters) for l in lattices}), 10)
                    self.assert_valid(structure, constraints, lattices, length, 1.5)

    def test_bounds_and_settings_across_crystal_systems(self):
        examples = [
            (225, (5, 5, 5, 90, 90, 90)),
            (139, (4, 4, 6, 90, 90, 90)),
            (62, (4, 5, 6, 90, 90, 90)),
            (14, (4, 5, 6, 106, 90, 90)),
            (14, (4, 5, 6, 90, 106, 90)),
            (14, (4, 5, 6, 90, 150, 90)),
            (14, (4, 5, 6, 90, 90, 106)),
            (194, (4, 4, 6, 90, 90, 120)),
            (166, (4, 4, 6, 90, 90, 120)),
            (166, (5, 5, 5, 75, 75, 75)),
            (2, (4, 5, 6, 80, 95, 110)),
            (2, (4, 5, 6, 30, 80, 80)),
        ]
        for group, cell in examples:
            with self.subTest(group=group, cell=cell):
                structure, constraints, nudger, lattices = self.propose(cell, group)
                self.assertEqual(len(lattices), 10)
                self.assert_valid(structure, constraints, lattices, 4, 1.5)
                stats = nudger._last_lattice_sampling
                self.assertEqual(stats['generated'], stats['accepted'] + stats['rejected_invalid']
                    + stats['rejected_physical'] + stats['rejected_q_window'])

    def test_representatives_never_exceed_requested_budget(self):
        for reps in (1, 2, 10):
            with self.subTest(reps=reps):
                structure, _, nudger, lattices = self.propose((4, 5, 6, 80, 95, 110), 2, reps=reps)
                self.assertEqual(len(lattices), reps)
                self.assertEqual(lattices[0].parameters, structure.lattice.parameters)
                self.assertEqual(nudger._last_lattice_sampling['selected'], reps)

    def test_seed_and_order_are_reproducible(self):
        first = self.propose((4, 5, 6, 80, 95, 110), 2)[3]
        second = self.propose((4, 5, 6, 80, 95, 110), 2)[3]
        self.assertEqual([l.parameters for l in first], [l.parameters for l in second])

    def test_zero_tolerances_keep_only_source_without_draws(self):
        structure, _, nudger, lattices = self.propose((4, 5, 6, 80, 95, 110), 2, length=0, angle=0)
        self.assertEqual([l.parameters for l in lattices], [structure.lattice.parameters])
        self.assertEqual(nudger._last_lattice_sampling['generated'], 0)

    def test_fixed_length_or_angles_do_not_drift(self):
        for length, angle in ((0, 1.5), (4, 0)):
            with self.subTest(length=length, angle=angle):
                structure, constraints, _, lattices = self.propose((4, 5, 6, 80, 95, 110), 2,
                    length=length, angle=angle)
                self.assert_valid(structure, constraints, lattices, length, angle)

    def test_actual_q_window_rejects_otherwise_physical_proposals(self):
        with patch.object(N, '_QWIN_PCT', .001):
            structure, constraints, nudger, lattices = self.propose((4, 5, 6, 80, 95, 110), 2)
            self.assertGreater(nudger._last_lattice_sampling['rejected_q_window'], 0)
            self.assert_valid(structure, constraints, lattices, 4, 1.5)

    def test_invalid_proposal_is_counted_and_never_returned(self):
        with patch.object(N, '_bounded_lattice_vectors', return_value=[np.array([-1, 5, 6, 80, 95, 110])]):
            structure, _, nudger, lattices = self.propose((4, 5, 6, 80, 95, 110), 2)
            self.assertEqual([l.parameters for l in lattices], [structure.lattice.parameters])
            self.assertEqual(nudger._last_lattice_sampling['rejected_invalid'], 1)

    def test_small_sample_budget_is_not_inflated_by_axis_probes(self):
        _, _, nudger, _ = self.propose((4, 5, 6, 80, 95, 110), 2, samples=7)
        self.assertEqual(nudger._last_lattice_sampling['generated'], 7)

    def test_invalid_budget_and_bounds_are_rejected(self):
        for kwargs in ({'reps': 0}, {'length': -1}, {'angle': float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.propose((4, 5, 6, 80, 95, 110), 2, **kwargs)

    def test_invalid_metric_is_rejected_instead_of_clipped(self):
        source = Structure(Lattice.from_parameters(4, 5, 6, 80, 95, 110), ["Si"], [[0, 0, 0]])
        constraints = N.infer_constraints(source, 2)
        names = N._free_param_names(constraints)
        for values in ([4, 5, 6, 80, 80, 170], [4, 5, 6, 90, 180, 90],
                       [0, 5, 6, 90, 90, 90], [4, 5, 6, 90, float('nan'), 90],
                       [1e200, 1e200, 1e200, 90, 90, 90]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                N._apply_params_from_vector(source.lattice, constraints, names, values)


if __name__ == '__main__':
    unittest.main()
