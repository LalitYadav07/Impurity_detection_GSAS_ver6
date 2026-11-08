"""The deployed full-Q mode reaches the existing linked viewer."""

import numpy as np
import pytest

pytest.importorskip("torch", reason="optional U-Net runtime is not installed")

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.scan_interpretation import PATTERN_SCHEMA
from radar_pd_nova.unet_transition_engine import FullQTransitionEngine


def scan(number, shift):
    q = np.linspace(.2, 12.1, 3000)
    y = 4 + 100 * np.exp(-((q - (2 + shift)) / .05) ** 2)
    y += 65 * np.exp(-((q - 5) / .08) ** 2)
    return {
        "run_id": f"PG3_{number}_BANK1", "run_number": number,
        "instrument": "PG3", "bank": 1, "ipts": "IPTS-test",
        "metadata": {"sample_id": "one-sample", "start_time": f"2026-01-01T00:0{number}:00+00:00"},
        "pattern": {"$schema": PATTERN_SCHEMA, "source_gpx_sha256": "test",
                    "coordinate": {"kind": "q", "unit": "1/angstrom"},
                    "context": {"profile_sha256": "exact-q-input",
                                "reduction_family": "test", "normalization": "pCharge"},
                    "fit_domain": {"limits": [float(q[0]), float(q[-1])], "excluded": []},
                    "x": q.tolist(), "y": y.tolist(), "ycalc": y.tolist(),
                    "sigma": [1.] * len(q), "valid": [True] * len(q),
                    "fit_valid": [True] * len(q)},
    }


def test_full_q_default_keeps_heatmap_fit_and_shows_cause_scores():
    app = RadarPdNovaApp()
    assert isinstance(app._unet_transition_engine, FullQTransitionEngine)
    assert app.server.state.transition_evidence_mode == "unet"
    app._refresh_unet_transition_tracking([scan(1, 0), scan(2, .03)])
    report = app._unet_transition_report
    assert len(report["groups"]) == 1
    assert report["records"][1]["status"] == "valid"
    assert len(report["records"][1]["comparison"]["causes"]) == 4
    state = app.server.state
    assert len(state.unet_transition_causes) == 4
    assert "Joint Q coverage" in state.unet_transition_coverage_note
    assert any(trace["name"] == "Measured heat map" for trace in state.unet_transition_map["data"])
    assert state.unet_transition_reference_has_fit
    assert state.unet_transition_selected_has_fit
