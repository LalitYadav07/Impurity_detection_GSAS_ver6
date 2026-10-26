"""Regression: YAML-selected correlation thresholds reach real profile pruning."""
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts", ROOT / "ML_components"):
    sys.path.insert(0, str(p))

from scripts.gsas_complete_pipeline_nomain import UnifiedPipeline
from scripts.gsas_legacy_bridge import IntegratedCandidateScreener, LegacyPipelineBridge


class FakeDB:
    catalog = pd.DataFrame([
        dict(id=p, space_group=166, elements_mask_hi=0, elements_mask_lo=12)
        for p in ["Li_rich", "Li_poor"]
    ])

    def get_display_name_and_sg(self, pid):
        return pid, "R-3m"


def screened_ids(config_text, *, dataset=None, stage="pass1"):
    pipeline = UnifiedPipeline.__new__(UnifiedPipeline)
    pipeline.top_cfg = yaml.safe_load(config_text) or {}
    screener = IntegratedCandidateScreener(None, FakeDB(), "/unused")
    ids = ["Li_rich", "Li_poor"]
    base = np.linspace(-1, 1, 64)
    other = np.cos(np.linspace(0, 6 * np.pi, 64))
    other -= other.mean()
    other -= np.dot(base, other) / np.dot(base, base) * base
    other *= np.linalg.norm(base) / np.linalg.norm(other)
    profiles = np.vstack([base, base + .15 * other])
    assert .95 < np.corrcoef(profiles)[0, 1] < .999
    with tempfile.TemporaryDirectory() as folder:
        cfg = pipeline._make_hist_plot_cfg(stage, folder, dataset)
        bridge = screener.bridge
        with (
            patch.object(bridge, "set_element_constraints"),
            patch.object(bridge, "filter_candidates_by_elements", return_value=ids),
            patch.object(bridge, "filter_candidates_by_space_group", return_value=ids),
            patch.object(bridge, "screen_by_histogram", return_value=([("Li_rich", .9), ("Li_poor", .8)], {}, {})),
            patch("scripts.gsas_legacy_bridge._load_profiles64_metadata", return_value={"profiles": profiles, "pid_to_row": dict(zip(ids, range(2)))}),
        ):
            result = screener.screen_candidates_comprehensive(
                residual_Q=np.ones(64), Q=np.arange(64),
                residual_native=np.ones(64), x_native=np.arange(64),
                Q_main_peaks=np.array([]), allowed_elements=["Li", "Ni", "O"],
                all_candidate_ids=ids, hist_plot_cfg=cfg,
            )
    return [r.phase_id for r in result]


@pytest.mark.parametrize("stage", ["stage0", "pass1"])
def test_yaml_threshold_preserves_distinct_occupancy_candidates(stage):
    assert screened_ids("{}", stage=stage) == ["Li_rich"]
    assert screened_ids("corr_threshold: 0.95", stage=stage) == ["Li_rich"]
    assert screened_ids("corr_threshold: 0.999", stage=stage) == ["Li_rich", "Li_poor"]


def test_dataset_override_and_precise_threshold_log(caplog):
    with caplog.at_level("INFO"):
        assert screened_ids("corr_threshold: 0.95", dataset={"corr_threshold": .999}) == ["Li_rich", "Li_poor"]
    assert "r>=0.999" in caplog.text
    assert screened_ids("corr_threshold: 0.999", dataset={"corr_threshold": .95}) == ["Li_rich"]


@pytest.mark.parametrize("value", ["-0.01", "1.01", ".nan", ".inf"])
def test_invalid_threshold_is_rejected(value):
    with pytest.raises(ValueError, match="corr_threshold must be finite and between 0 and 1"):
        screened_ids(f"corr_threshold: {value}")
