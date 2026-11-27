import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

try:
    from scripts.rapid_hypothesis_pipeline import _prepare_no_main_rapid_signal
except Exception as exc:  # pragma: no cover - depends on local GSAS-II runtime availability
    if os.environ.get("RADAR_REQUIRE_STRUCTURAL_NATIVE") == "1":
        raise
    _prepare_no_main_rapid_signal = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(_prepare_no_main_rapid_signal is None, f"rapid pipeline import failed: {IMPORT_ERROR}")
class RapidNoMainSignalTests(unittest.TestCase):
    def test_nudge_uses_selected_native_domain_in_a_multi_dataset_config(self):
        import pandas as pd
        import scripts.rapid_hypothesis_pipeline as rapid
        import lattice_nudger

        captured = {}
        class FakeNudger:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
            def optimize_one(self, *args, **kwargs):
                return SimpleNamespace(best_params={}, best_score=.5,
                    nudged_cif_path="synthetic.cif", elapsed_s=0., lattice_deviation=0.)
        rows = pd.DataFrame([dict(phase_ids="phase", formula_keys="Si", formulas="Si")])
        cfg = {"datasets": [{"_native_q_fit_intervals": [(9, 10)]}],
               "rapid_hypothesis": {"parallel_nudge": False}}
        selected_domain = [(1., 2.), (4., 5.)]
        with tempfile.TemporaryDirectory() as folder, patch.object(
            lattice_nudger, "LatticeNudger", FakeNudger
        ), patch.object(rapid, "resolve_xray_doublet_spec", return_value=SimpleNamespace(to_dict=lambda: {})):
            rapid._nudge_unique_phases(rows, SimpleNamespace(get_space_group_number=lambda _: 225),
                np.array([1., 2., 4., 5.]), np.ones(4), cfg, Path(folder),
                valid_q_intervals=selected_domain)
        self.assertEqual(captured["valid_q_intervals"], selected_domain)

    def test_no_main_signal_uses_background_subtracted_residual(self):
        q = np.linspace(1.0, 7.0, 1200)
        background = 80.0 + 14.0 * q + 10.0 * np.sin(q * 1.5)
        peaks = (
            450.0 * np.exp(-0.5 * ((q - 2.2) / 0.025) ** 2)
            + 700.0 * np.exp(-0.5 * ((q - 4.1) / 0.035) ** 2)
            + 300.0 * np.exp(-0.5 * ((q - 5.6) / 0.030) ** 2)
        )
        y = background + peaks
        cfg = {
            "background": {"mode": "auto_fixed_points", "auto_params": {"max_points": 60}},
            "rapid_hypothesis": {"background_subtract_no_main": True},
        }
        with tempfile.TemporaryDirectory() as tmp:
            signal, kind = _prepare_no_main_rapid_signal(q, y, cfg, Path(tmp))
            self.assertEqual(kind, "background-subtracted raw data")
            self.assertTrue((Path(tmp) / "target_background.npz").exists())

        quiet = (q < 1.5) | ((q > 6.2) & (q < 6.8))
        self.assertLess(float(np.median(signal[quiet])), float(np.median(y[quiet])) * 0.15)
        self.assertGreater(float(signal.max()), 250.0)

    def test_no_main_signal_can_be_disabled(self):
        q = np.linspace(1.0, 7.0, 100)
        y = np.linspace(10.0, 50.0, 100)
        cfg = {
            "background": {"mode": "auto_fixed_points"},
            "rapid_hypothesis": {"background_subtract_no_main": False},
        }
        signal, kind = _prepare_no_main_rapid_signal(q, y, cfg, None)
        self.assertEqual(kind, "raw data")
        np.testing.assert_allclose(signal, y)


@unittest.skipIf(_prepare_no_main_rapid_signal is None, f"rapid pipeline import failed: {IMPORT_ERROR}")
class RapidLatticeSamplingReceiptTests(unittest.TestCase):
    def test_sampling_receipt_and_scored_count_survive_csv_roundtrip(self):
        import io
        import json
        import pandas as pd
        import scripts.rapid_hypothesis_pipeline as rapid
        import lattice_nudger
        sampling = dict(method="bounded-cell-sobol", generated=800, accepted=795,
            unique_including_source=796, selected=10, only_original_scored=False)
        class FakeNudger:
            def __init__(self, *args, **kwargs):pass
            def optimize_one(self, *args, **kwargs):
                return SimpleNamespace(best_params={}, best_score=.5,
                    nudged_cif_path="synthetic.cif", elapsed_s=0., lattice_deviation=0.,
                    sampling=sampling, scored_count=10)
        rows = pd.DataFrame([dict(phase_ids="phase", formula_keys="Si", formulas="Si")])
        cfg = {"datasets": [{}], "rapid_hypothesis": {"parallel_nudge": False}}
        with tempfile.TemporaryDirectory() as folder, patch.object(lattice_nudger, "LatticeNudger", FakeNudger), patch.object(
                rapid, "resolve_xray_doublet_spec", return_value=SimpleNamespace(to_dict=lambda: {})):
            result = rapid._nudge_unique_phases(rows, SimpleNamespace(get_space_group_number=lambda _: 225),
                np.array([1., 2., 4., 5.]), np.ones(4), cfg, Path(folder))
        restored = pd.read_csv(io.StringIO(result.to_csv(index=False)))
        self.assertEqual(json.loads(restored.iloc[0]["sampling_json"]), sampling)
        self.assertEqual(restored.iloc[0]["scored_count"], 10)
        self.assertEqual(restored.iloc[0]["best_score"], .5)
        self.assertEqual(result.iloc[0]["error"], "")


    def test_missing_or_unserializable_sampling_does_not_discard_candidate(self):
        import pandas as pd
        import scripts.rapid_hypothesis_pipeline as rapid
        import lattice_nudger
        for sampling in (None, {"bad": {1, 2}}, {"bad": float("nan")}):
            with self.subTest(sampling=repr(sampling)):
                class FakeNudger:
                    def __init__(self, *args, **kwargs):pass
                    def optimize_one(self, *args, **kwargs):
                        return SimpleNamespace(best_params={}, best_score=.5,
                            nudged_cif_path="synthetic.cif", elapsed_s=0., lattice_deviation=0.,
                            sampling=sampling)
                rows = pd.DataFrame([dict(phase_ids="phase", formula_keys="Si", formulas="Si")])
                cfg = {"datasets": [{}], "rapid_hypothesis": {"parallel_nudge": False}}
                with tempfile.TemporaryDirectory() as folder, patch.object(lattice_nudger, "LatticeNudger", FakeNudger), patch.object(
                        rapid, "resolve_xray_doublet_spec", return_value=SimpleNamespace(to_dict=lambda: {})):
                    result = rapid._nudge_unique_phases(rows, SimpleNamespace(get_space_group_number=lambda _: 225),
                        np.array([1., 2., 4., 5.]), np.ones(4), cfg, Path(folder))
                self.assertEqual(result.iloc[0]["sampling_json"], "")
                self.assertTrue(pd.isna(result.iloc[0]["scored_count"]))
                self.assertEqual(result.iloc[0]["best_score"], .5)
                self.assertEqual(result.iloc[0]["error"], "")


if __name__ == "__main__":
    unittest.main()
