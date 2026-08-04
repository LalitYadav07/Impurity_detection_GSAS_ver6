import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from scripts.rapid_hypothesis_pipeline import _prepare_no_main_rapid_signal
except Exception as exc:  # pragma: no cover - depends on local GSAS-II runtime availability
    _prepare_no_main_rapid_signal = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(_prepare_no_main_rapid_signal is None, f"rapid pipeline import failed: {IMPORT_ERROR}")
class RapidNoMainSignalTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
