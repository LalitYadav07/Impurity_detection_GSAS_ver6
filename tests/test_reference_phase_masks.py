import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.config_builder import build_pipeline_config
from scripts.reference_phase_masks import (
    REFERENCE_PHASE_PRESETS,
    build_reference_phase_exclusions,
    instrument_model_from_instprm,
    normalize_reference_preset,
)


class ReferencePhaseMaskTests(unittest.TestCase):
    def _write_instprm(self, tmpdir: str, *, wavelength: float = 1.540598, zero: float = 0.0) -> Path:
        path = Path(tmpdir) / "test.instprm"
        path.write_text(f"Lam1: {wavelength}\nZero: {zero}\n", encoding="utf-8")
        return path

    def test_preset_alias_resolution(self):
        self.assertEqual(normalize_reference_preset("Al"), "Al_fcc")
        self.assertEqual(normalize_reference_preset("Cu_fcc"), "Cu_fcc")
        self.assertEqual(normalize_reference_preset("vanadium"), "V_bcc")
        with self.assertRaises(ValueError):
            normalize_reference_preset("Fe_bcc")

    def test_bundled_preset_cifs_exist(self):
        for preset in REFERENCE_PHASE_PRESETS.values():
            self.assertTrue(preset.cif_path.exists(), preset.cif_path)

    def test_al_fcc_cuka_windows_are_generated_in_two_theta_fixed_width(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            instprm = self._write_instprm(tmpdir)
            report = build_reference_phase_exclusions(
                {
                    "enabled": True,
                    "presets": ["Al_fcc"],
                    "window_mode": "fixed",
                    "half_width": 0.30,
                    "include_cu_kbeta": False,
                    "include_secondary_wavelengths": False,
                },
                instprm_path=instprm,
                mode="cw",
                limits=[10.0, 80.0],
            )

        centers = [round(row["center"], 3) for row in report["reflections"]]
        self.assertEqual(centers, [38.473, 44.722, 65.099, 78.232])
        self.assertEqual(len(report["ranges"]), 4)
        self.assertEqual(report["reflections"][0]["hkl"], [1, 1, 1])
        self.assertEqual(report["native_axis"], "2theta_deg")
        self.assertEqual(report["reflections"][0]["half_width"], 0.30)

    def test_cw_lam2_and_cu_kbeta_windows_are_generated_from_instrument(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            instprm = Path(tmpdir) / "test.instprm"
            instprm.write_text(
                "Type:PXC\nLam1:1.540598\nLam2:1.544426\nI(L2)/I(L1):0.5\nZero:-0.007836058285725013\n",
                encoding="utf-8",
            )
            report = build_reference_phase_exclusions(
                {
                    "enabled": True,
                    "references": [{"preset": "Al_fcc"}],
                    "window_mode": "fixed",
                    "half_width": 0.30,
                    "include_cu_kbeta": True,
                },
                instprm_path=instprm,
                mode="cw",
                limits=[10.0, 80.0],
            )

        labels = [row["line"] for row in report["reflections"][:3]]
        self.assertEqual(labels, ["Lam1", "Lam2", "CuKbeta"])
        self.assertEqual(round(report["reflections"][0]["center"], 3), 38.466)
        self.assertEqual(round(report["reflections"][1]["center"], 3), 38.565)
        kbeta_centers = [
            round(row["center"], 3)
            for row in report["reflections"]
            if row["line"] == "CuKbeta"
        ]
        self.assertEqual(kbeta_centers, [34.637, 40.21, 58.177, 69.512, 73.088])
        self.assertTrue(report["include_cu_kbeta"])

    def test_cw_auto_width_uses_instrument_profile_and_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            instprm = Path(tmpdir) / "test.instprm"
            instprm.write_text(
                "Type:PNC\nLam:2.4062686168735197\nZero:-0.009602591470493875\n"
                "U:798.889\nV:-444.367\nW:242.406\nX:0.0\nY:0.0\n",
                encoding="utf-8",
            )
            report = build_reference_phase_exclusions(
                {
                    "enabled": True,
                    "presets": ["Al_fcc"],
                    "window_mode": "auto",
                    "fwhm_factor": 3.0,
                    "fractional_d_tolerance": 0.0,
                    "zero_tolerance_deg": 0.0,
                    "include_secondary_wavelengths": False,
                },
                instprm_path=instprm,
                mode="cw",
                limits=[10.0, 140.0],
            )

        centers = [round(row["center"], 3) for row in report["reflections"]]
        self.assertEqual(centers[:2], [61.932, 72.904])
        self.assertEqual(round(report["instrument"]["zero"], 6), -0.009603)
        self.assertEqual(report["reflections"][0]["width_source"], "auto_profile")
        self.assertGreater(report["reflections"][0]["half_width"], 0.10)

    def test_tof_windows_are_generated_in_native_microseconds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            instprm = Path(tmpdir) / "test.instprm"
            instprm.write_text(
                "Type:PNT\nZero:0.0\ndifA:-5.316\ndifB:-4.125\ndifC:22597.136\n"
                "sig-0:211.81909809832996\nsig-1:-148.75553506596128\n"
                "sig-2:71.43688145043883\nsig-q:-101.23212297704391\n",
                encoding="utf-8",
            )
            report = build_reference_phase_exclusions(
                {
                    "enabled": True,
                    "presets": ["Al_fcc"],
                    "window_mode": "auto",
                    "fwhm_factor": 3.0,
                    "fractional_d_tolerance": 0.0,
                    "zero_tolerance_tof": 0.0,
                },
                instprm_path=instprm,
                mode="tof",
                limits=[1000.0, 60000.0],
            )

        first = report["reflections"][0]
        self.assertEqual(report["native_axis"], "tof_us")
        self.assertEqual(first["hkl"], [1, 1, 1])
        self.assertEqual(round(first["d_spacing"], 4), 2.338)
        self.assertEqual(round(first["center"], 2), 52800.83)
        self.assertEqual(round(first["half_width"], 2), 127.18)
        self.assertEqual(first["width_source"], "auto_profile")

    def test_default_auto_width_includes_position_tolerance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            instprm = Path(tmpdir) / "test.instprm"
            instprm.write_text(
                "Type:PNC\nLam:2.40825\nZero:0.0\n"
                "U:480.78894\nV:-244.82173\nW:201.00528\nX:0.0\nY:7.41090\n",
                encoding="utf-8",
            )
            report = build_reference_phase_exclusions(
                {"enabled": True, "presets": ["Cu_fcc"], "window_mode": "auto"},
                instprm_path=instprm,
                mode="cw",
                limits=[5.0, 126.6],
            )

        rows = {tuple(row["hkl"]): row for row in report["reflections"]}
        self.assertGreater(rows[(1, 1, 1)]["half_width"], 0.75)
        self.assertGreater(rows[(2, 0, 0)]["half_width"], 0.90)
        self.assertEqual(rows[(1, 1, 1)]["width_source"], "auto_profile+tolerance")

    def test_instrument_model_rejects_mode_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            instprm = Path(tmpdir) / "test.instprm"
            instprm.write_text("Type:PNT\nZero:0.0\ndifC:10000.0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                instrument_model_from_instprm(instprm, mode="cw")

    def test_enabled_requires_presets_and_cw_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            instprm = self._write_instprm(tmpdir)
            with self.assertRaises(ValueError):
                build_reference_phase_exclusions(
                    {"enabled": True, "presets": []},
                    instprm_path=instprm,
                    mode="cw",
                )


class ReferencePhaseConfigBuilderTests(unittest.TestCase):
    def test_build_pipeline_config_emits_reference_phase_exclusion_block(self):
        cfg_text = build_pipeline_config(
            run_name="refmask_test",
            data_file="/tmp/example.dat",
            instprm_file="/tmp/example.instprm",
            allowed_elements=["Tb", "Be", "Ge", "O"],
            reference_phase_exclusions={
                "enabled": True,
                "presets": ["Al_fcc", "Cu_fcc"],
                "window_mode": "auto",
                "fwhm_factor": 4.0,
                "include_cu_kbeta": True,
            },
        )
        cfg = yaml.safe_load(cfg_text)

        ref_cfg = cfg["reference_phase_exclusions"]
        self.assertTrue(ref_cfg["enabled"])
        self.assertEqual(ref_cfg["presets"], ["Al_fcc", "Cu_fcc"])
        self.assertEqual(ref_cfg["window_mode"], "auto")
        self.assertEqual(ref_cfg["fwhm_factor"], 4.0)
        self.assertTrue(ref_cfg["include_cu_kbeta"])
        self.assertEqual(cfg["datasets"][0]["name"], "refmask_test")


if __name__ == "__main__":
    unittest.main()
