import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.ml_ranker_support import discover_ml_ranker_assets, load_first_json_record, write_ranker_status


class MLRankerSupportTests(unittest.TestCase):
    def test_discovers_default_ranker_assets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ranker_dir = root / "ML_ranker" / "mlp_ranker_for_phase_detection-main"
            ranker_dir.mkdir(parents=True)
            (ranker_dir / "infer.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            (ranker_dir / "mlp_ranker.pt").write_text("stub", encoding="utf-8")

            assets = discover_ml_ranker_assets(root)

            self.assertTrue(assets.is_ready)
            self.assertEqual(assets.script_path, ranker_dir / "infer.py")
            self.assertEqual(assets.model_path, ranker_dir / "mlp_ranker.pt")

    def test_reports_missing_ranker_assets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assets = discover_ml_ranker_assets(tmpdir)
            self.assertFalse(assets.is_ready)
            self.assertIn("No usable ML ranker assets found", assets.error)

    def test_env_override_is_used_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script_path = root / "custom_infer.py"
            model_path = root / "custom_ranker.pt"
            script_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            model_path.write_text("stub", encoding="utf-8")

            old_script = os.environ.get("RADAR_ML_RANKER_SCRIPT")
            old_model = os.environ.get("RADAR_ML_RANKER_MODEL")
            try:
                os.environ["RADAR_ML_RANKER_SCRIPT"] = str(script_path)
                os.environ["RADAR_ML_RANKER_MODEL"] = str(model_path)
                assets = discover_ml_ranker_assets(root)
            finally:
                if old_script is None:
                    os.environ.pop("RADAR_ML_RANKER_SCRIPT", None)
                else:
                    os.environ["RADAR_ML_RANKER_SCRIPT"] = old_script
                if old_model is None:
                    os.environ.pop("RADAR_ML_RANKER_MODEL", None)
                else:
                    os.environ["RADAR_ML_RANKER_MODEL"] = old_model

            self.assertTrue(assets.is_ready)
            self.assertEqual(assets.source, "env")

    def test_load_first_json_record_handles_json_and_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            json_path = tmpdir / "one.json"
            jsonl_path = tmpdir / "many.jsonl"
            json_path.write_text(json.dumps({"ranked": [{"mp_id": "mp-1"}]}), encoding="utf-8")
            jsonl_path.write_text("\n" + json.dumps({"ranked": [{"mp_id": "mp-2"}]}) + "\n", encoding="utf-8")

            self.assertEqual(load_first_json_record(json_path)["ranked"][0]["mp_id"], "mp-1")
            self.assertEqual(load_first_json_record(jsonl_path)["ranked"][0]["mp_id"], "mp-2")

    def test_write_ranker_status_creates_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_ranker_status(Path(tmpdir) / "ml_rank_status_pass1.json", status="complete", pass_ix=1)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["pass_ix"], 1)


if __name__ == "__main__":
    unittest.main()
