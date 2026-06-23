import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
ML_COMPONENTS_DIR = REPO_ROOT / "ML_components"
for p in (str(REPO_ROOT), str(SCRIPTS_DIR), str(ML_COMPONENTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.gsas_legacy_bridge import LegacyPipelineBridge
from scripts.ratio_filter import _PROFILE_CACHE, _load_profiles64_metadata, _residual_hist_from_continuous_parts
from models import shortlist_ml_rank


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


class ResidualHistogramGapTests(unittest.TestCase):
    def test_internal_gap_produces_fragmented_observed_mask_without_bridging_area(self):
        Q = np.array([0.10, 0.20, 0.30, 1.10, 1.20, 1.30, 3.10, 3.20, 3.30], dtype=float)
        R = np.ones_like(Q)
        edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)

        H, observed_mask, counts = _residual_hist_from_continuous_parts(
            Q,
            R,
            Q_main_peaks=np.array([], dtype=float),
            edges=edges,
            sigma_bins=0.0,
        )

        self.assertTrue(np.array_equal(observed_mask, np.array([True, True, False, True])))
        self.assertTrue(np.array_equal(counts, np.array([3, 3, 0, 3])))
        self.assertTrue(np.allclose(H, np.array([0.3, 0.3, 0.0, 0.3]), atol=1e-8))

    def test_smoothing_handles_short_fragmented_segments_without_shape_errors(self):
        Q = np.array([0.10, 0.20, 2.10, 2.20], dtype=float)
        R = np.array([1.0, 0.5, 0.75, 0.25], dtype=float)
        edges = np.linspace(0.0, 4.0, 9, dtype=float)

        H, observed_mask, counts = _residual_hist_from_continuous_parts(
            Q,
            R,
            Q_main_peaks=np.array([], dtype=float),
            edges=edges,
            sigma_bins=1.0,
        )

        self.assertEqual(H.shape, (8,))
        self.assertEqual(observed_mask.shape, (8,))
        self.assertEqual(counts.shape, (8,))
        self.assertGreater(np.count_nonzero(observed_mask), 0)
        self.assertTrue(np.all(np.isfinite(H)))


class ProfileMetadataLoadTests(unittest.TestCase):
    def test_profiles64_cache_uses_float32_not_float64(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "profiles64"
            root.mkdir()
            np.savez(
                root / "profiles64.npz",
                profiles=np.ones((3, 64), dtype=np.float16),
                q_min=np.array(0.5),
                q_max=np.array(6.0),
                n_bins=np.array(64),
                sigma_bins=np.array(0.7),
            )
            (root / "index.csv").write_text("id,row\np0,0\np1,1\np2,2\n", encoding="utf-8")

            _PROFILE_CACHE.clear()
            ctx = _load_profiles64_metadata(str(root))

            self.assertEqual(ctx["profiles"].dtype, np.float32)
            self.assertEqual(ctx["profiles"].shape, (3, 64))


class ExplicitMaskMLRankTests(unittest.TestCase):
    def test_shortlist_ml_rank_respects_non_contiguous_mask(self):
        centers = np.arange(64, dtype=float)
        H_res = np.zeros(64, dtype=np.float32)
        H_res[[10, 11, 40, 41]] = np.array([1.0, 0.6, 0.8, 0.4], dtype=np.float32)
        profiles = np.zeros((2, 64), dtype=np.float32)
        profiles[0, [10, 11, 40, 41]] = np.array([0.9, 0.5, 0.7, 0.3], dtype=np.float32)
        profiles[1, [24, 25, 26]] = 1.0
        pid_to_row = {"keep_me": 0, "drop_me": 1}
        mask_bool = np.zeros(64, dtype=bool)
        mask_bool[[10, 11, 40, 41]] = True

        class _FakeModel:
            def __call__(self, xb):
                n = xb.shape[0]
                return (
                    torch.full((n, 1), 0.75, dtype=torch.float32),
                    torch.zeros((n, 1), dtype=torch.float32),
                    None,
                )

        with patch("models.load_ml_model", return_value=(_FakeModel(), "cpu", {"has_cls": False})):
            scored, details, meta = shortlist_ml_rank(
                H_res=H_res,
                centers=centers,
                profiles=profiles,
                pid_to_row=pid_to_row,
                candidate_ids=["keep_me", "drop_me"],
                mask_bool=mask_bool,
                q_active_min=10.0,
                q_active_max=41.0,
                device="cpu",
                topN=None,
                plot=False,
            )

        self.assertEqual([pid for pid, _ in scored], ["keep_me"])
        self.assertEqual([d["phase_id"] for d in details], ["keep_me"])
        self.assertEqual(meta["active_bins"], 4)
        self.assertTrue(meta["fragmented_mask"])


if __name__ == "__main__":
    unittest.main()
