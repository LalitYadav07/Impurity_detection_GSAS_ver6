import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
GSASII_DIR = REPO_ROOT / "GSAS-II"
for p in (str(REPO_ROOT), str(SCRIPTS_DIR), str(GSASII_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts import gsas_core_infrastructure as gci
from scripts import instprm_presets as ip


class BuiltinInstrumentPresetTests(unittest.TestCase):
    def test_known_lab_xray_preset_metadata_is_available(self):
        preset = ip.get_builtin_instprm_preset(ip.DEFAULT_LAB_XRAY_PRESET_KEY)
        self.assertEqual(preset["gsas_label"], "CuKa lab data")
        self.assertEqual(preset["instrument_mode"], "cw")
        self.assertEqual(preset["radiation_source"], "X-ray")

    def test_unknown_preset_key_raises(self):
        with self.assertRaises(ValueError):
            ip.get_builtin_instprm_preset("does_not_exist")

    def test_write_builtin_preset_creates_valid_instprm_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "generated.instprm"
            written_path = ip.write_builtin_instprm_file(
                ip.DEFAULT_LAB_XRAY_PRESET_KEY,
                output_path,
            )

            self.assertEqual(written_path, output_path)
            text = output_path.read_text(encoding="utf-8")
            self.assertIn("#GSAS-II instrument parameter file for lab CuKa data", text)
            self.assertIn("Type:PXC;Bank:1", text)
            self.assertIn("Lam1:1.5405;Lam2:1.5443", text)

    def test_supported_upload_extensions_include_legacy_gsas_formats(self):
        self.assertEqual(
            ip.SUPPORTED_INSTRUMENT_UPLOAD_EXTENSIONS,
            ["instprm", "prm", "inst", "ins"],
        )

    def test_legacy_prm_normalizes_to_instprm(self):
        legacy_prm = REPO_ROOT / "GSAS-II" / "tests" / "testinp" / "inst_d1a.prm"
        self.assertTrue(legacy_prm.exists(), "Expected bundled GSAS-II legacy .prm sample")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "normalized_from_legacy.instprm"
            written_path = ip.normalize_instrument_profile_to_instprm(
                legacy_prm,
                output_path,
            )

            self.assertEqual(written_path, output_path)
            text = output_path.read_text(encoding="utf-8")
            self.assertIn("#GSAS-II instrument parameter file", text)
            self.assertIn("Type:PNC", text)
            self.assertIn("Lam:1.909", text)

    def test_generated_lab_preset_imports_synthetic_xray_histogram(self):
        if not gci.GSAS_AVAILABLE:
            self.skipTest("GSAS-II is not available in this environment")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            instprm_path = ip.write_builtin_instprm_file(
                ip.DEFAULT_LAB_XRAY_PRESET_KEY,
                tmpdir / "generated_CuKa_lab.instprm",
            )
            data_path = tmpdir / "synthetic_lab_pxrd.xye"

            x = np.linspace(10.0, 80.0, 600)
            y = (
                40.0
                + 220.0 * np.exp(-0.5 * ((x - 24.5) / 0.25) ** 2)
                + 140.0 * np.exp(-0.5 * ((x - 38.2) / 0.33) ** 2)
                + 110.0 * np.exp(-0.5 * ((x - 52.7) / 0.28) ** 2)
            )
            sigma = np.sqrt(np.clip(y, 1.0, None))
            with data_path.open("w", encoding="utf-8") as handle:
                for xpos, ypos, sig in zip(x, y, sigma):
                    handle.write(f"{xpos:.5f} {ypos:.5f} {sig:.5f}\n")

            manager = gci.GSASProjectManager(str(tmpdir), project_name="preset_import")
            self.assertTrue(manager.create_project())
            self.assertTrue(
                manager.add_histogram(
                    str(data_path),
                    str(instprm_path),
                    fmthint="xye",
                    instrument_type="CW",
                )
            )

            hist_data = manager.project.data[manager.main_histogram.name]
            inst = hist_data["Instrument Parameters"][0]
            sample = hist_data["Sample Parameters"]

            self.assertEqual(inst["Type"][0], "PXC")
            self.assertEqual(sample["Type"], "Bragg-Brentano")
            self.assertIn("Transparency", sample)
            self.assertEqual(manager.get_instrument_type(), "CW")


if __name__ == "__main__":
    unittest.main()
