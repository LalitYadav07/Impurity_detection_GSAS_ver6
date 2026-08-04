import tempfile
import unittest
from pathlib import Path

from scripts.radar_api_server import (
    _query_int,
    _rewrite_config_for_job,
    _safe_artifact_path,
)


class RadarAPIServerTests(unittest.TestCase):
    def test_query_int_falls_back_and_clamps(self):
        self.assertEqual(_query_int({"tail": ["abc"]}, "tail", 80, max_value=500), 80)
        self.assertEqual(_query_int({"tail": ["-10"]}, "tail", 80, min_value=1), 1)
        self.assertEqual(_query_int({"tail": ["9999"]}, "tail", 80, max_value=500), 500)

    def test_artifact_path_cannot_escape_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "result.txt").write_text("ok", encoding="utf-8")
            self.assertEqual(_safe_artifact_path(run_dir, "result.txt"), run_dir / "result.txt")
            with self.assertRaises(ValueError):
                _safe_artifact_path(run_dir, "../secret.txt")

    def test_rewrite_config_requires_uploaded_or_existing_paths(self):
        cfg = {
            "datasets": [
                {
                    "name": "demo",
                    "data_path": "RADAR_PD_API_WILL_REWRITE_DATA",
                    "instprm_path": "RADAR_PD_API_WILL_REWRITE_INSTRUMENT",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data.dat"
            inst = root / "instrument.instprm"
            data.write_text("1 2\n", encoding="utf-8")
            inst.write_text("# inst\n", encoding="utf-8")
            resolved, dataset_name = _rewrite_config_for_job(
                cfg,
                job_dir=root / "job",
                run_dir=root / "job" / "run",
                dataset_name=None,
                run_name="api demo",
                data_path=data,
                instprm_path=inst,
                main_cif_path=None,
                mode="rapid",
            )
        self.assertEqual(dataset_name, "api_demo")
        self.assertEqual(resolved["analysis_mode"], "rapid_hypothesis")
        self.assertTrue(resolved["rapid_hypothesis"]["enabled"])
        self.assertEqual(resolved["datasets"][0]["name"], "api_demo")


if __name__ == "__main__":
    unittest.main()
