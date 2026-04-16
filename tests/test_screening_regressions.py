import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.gsas_legacy_bridge import LegacyPipelineBridge


class _FakeDBLoader:
    def __init__(self):
        self.catalog = pd.DataFrame(
            [
                {"id": "accepted1", "space_group": 10, "elements_mask_hi": 1, "elements_mask_lo": 1},
                {"id": "cand1", "space_group": 10, "elements_mask_hi": 1, "elements_mask_lo": 1},
                {"id": "cand2", "space_group": 10, "elements_mask_hi": 1, "elements_mask_lo": 1},
                {"id": "other", "space_group": 20, "elements_mask_hi": 2, "elements_mask_lo": 2},
            ]
        )

    def get_display_name_and_sg(self, pid):
        return pid, "sg"


class AnchorDedupTests(unittest.TestCase):
    def test_anchor_ids_drop_candidates_matching_already_accepted_phase(self):
        bridge = LegacyPipelineBridge(_FakeDBLoader(), "/tmp/fake_profiles")
        shared = np.linspace(0.0, 1.0, 64)
        distinct = np.linspace(1.0, 2.0, 64)
        meta = {
            "profiles": np.vstack([shared, shared, shared, distinct]),
            "pid_to_row": {
                "accepted1": 0,
                "cand1": 1,
                "cand2": 2,
                "other": 3,
            },
        }
        hist_scored = [("cand1", 0.9), ("cand2", 0.8), ("other", 0.7)]

        with patch("scripts.gsas_legacy_bridge._load_profiles64_metadata", return_value=meta):
            no_anchor = bridge.dedup_by_hist_and_elements(hist_scored, corr_threshold=0.95)
            with_anchor = bridge.dedup_by_hist_and_elements(
                hist_scored,
                corr_threshold=0.95,
                anchor_ids=["accepted1"],
            )

        self.assertEqual([pid for pid, _ in no_anchor], ["cand1", "other"])
        self.assertEqual([pid for pid, _ in with_anchor], ["other"])


if __name__ == "__main__":
    unittest.main()
