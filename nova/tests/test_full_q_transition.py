"""Exact-Q adapter and profile recovery contracts for the deployed tracker."""

from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from radar_pd_nova.diffraction_coordinates import TofCalibration
from radar_pd_nova.pattern_transition import ObservedPattern, PatternScope
from radar_pd_nova.transition_coordinate_mapping import mapped_calibration
from radar_pd_nova.unet_core.full_q_production import (
    BINS, Q_GRID, FullQTransitionPredictor, prepare_pair, rebin_scan,
)


def pattern(x, y, *, coordinate="q_inv_angstrom", bank="1", valid=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    scope = PatternScope("sample", "PG3", bank, coordinate,
                         "same-geometry", "same-reduction", "pCharge", "density")
    return ObservedPattern(scope, x, y, np.ones(len(x)),
                           np.ones(len(x), dtype=bool) if valid is None else np.asarray(valid, dtype=bool),
                           "scan")


def test_full_q_pair_matches_training_grid_and_shared_affine_transform():
    q = np.linspace(0.2, 12.1, 4500)
    a = pattern(q, 4 + np.exp(-((q - 2) / .05) ** 2) * 100)
    b = pattern(q, 4 + np.exp(-((q - 2.03) / .05) ** 2) * 100)
    x, coverage, stats = prepare_pair(a, b)
    assert x.shape == (3, BINS) and coverage.shape == (BINS,)
    assert stats["covered_bins"] > 900
    np.testing.assert_allclose(x[2], x[1] - x[0], atol=1e-6)
    assert np.all(x[:, ~coverage.astype(bool)] == 0)
    assert float(np.max(np.abs(x[2]))) > .1


def test_rebin_keeps_invalid_q_gap_uncovered():
    q = np.linspace(.2, 12.1, 4500)
    valid = (q < 5) | (q > 7)
    values, coverage = rebin_scan(q, np.ones(len(q)), valid)
    assert not coverage[(Q_GRID >= 5) & (Q_GRID <= 7)].any()
    assert np.all(values[~coverage] == 0)


def test_tof_profile_mapping_and_bank_checks(tmp_path):
    file = tmp_path / "profile.instprm"
    file.write_text("Type:PNT;Bank:7;difC:22000;difA:0;difB:0;Zero:0\n")
    checksum = sha256(file.read_bytes()).hexdigest()
    cal = TofCalibration.from_profile(file, source_bank=2, profile_bank=7,
                                      expected_sha256=checksum)
    d = np.linspace(.4, 3.0, 4500)
    tof = 22000 * d
    a = pattern(tof, 4 + np.exp(-((d - 1.0) / .02) ** 2) * 100,
                coordinate="tof_us", bank="2")
    b = replace(a, intensity=a.intensity * 1.2, observation_id="next")
    x, coverage, stats = prepare_pair(a, b, cal, cal)
    assert x.shape == (3, BINS) and stats["covered_bins"] == int(coverage.sum())
    assert np.max(x[2, coverage.astype(bool)]) > 0
    wrong_a = replace(a, scope=replace(a.scope, bank="3"))
    wrong_b = replace(b, scope=replace(b.scope, bank="3"))
    with pytest.raises(ValueError, match="another physical bank"):
        prepare_pair(wrong_a, wrong_b, cal, cal)


def test_restored_galaxy_input_uses_checked_bytes_and_explicit_bank(tmp_path):
    raw = b"Type:PNT;Bank:1;difC:22000;difA:0;difB:0;Zero:0\n"
    checksum = sha256(raw).hexdigest()
    receipt = {"source_bank": 2, "staged_bank": 1,
               "staged_profile_sha256": checksum,
               "staged_profile_dataset_id": "profile-dataset"}
    record = SimpleNamespace(uid="job", galaxy_job_id="job", input_dataset_ids={"instrument": "profile-dataset"},
                             provenance={"powgen_bank": receipt})
    service = SimpleNamespace(load_text_document=lambda identifier: raw.decode()
                              if identifier == "profile-dataset" else pytest.fail("wrong dataset"))
    scan = {"run_id": "PG3_1_BANK2", "bank": 2, "instrument": "PG3", "galaxy_job_id": "job"}
    cal = mapped_calibration(scan, None, {"job": record}, service)
    assert (cal.source_bank, cal.profile_bank, cal.profile_sha256) == (2, 1, checksum)
    with pytest.raises(ValueError, match="checksum"):
        mapped_calibration(scan, None, {"job": replace_record(record, staged_profile_sha256="0" * 64)}, service)


def replace_record(record, *, staged_profile_sha256):
    updated = dict(record.provenance["powgen_bank"])
    updated["staged_profile_sha256"] = staged_profile_sha256
    return SimpleNamespace(**{**vars(record), "provenance": {"powgen_bank": updated}})


def test_manual_powgen_profile_recovers_only_an_unambiguous_bank():
    single = "Type:PNT;Bank:4;difC:22000;difA:0;difB:0;Zero:0\n"
    receipt = {"source_bank": 2, "manual_profile_dataset_id": "manual",
               "profile_sha256": sha256(single.encode()).hexdigest()}
    record = SimpleNamespace(uid="job", galaxy_job_id="job", input_dataset_ids={},
                             provenance={"powgen_bank": receipt})
    scan = {"run_id": "PG3_1_BANK2", "bank": 2, "instrument": "PG3", "galaxy_job_id": "job"}
    service = SimpleNamespace(load_text_document=lambda _: single)
    calibration = mapped_calibration(scan, None, {"job": record}, service)
    assert (calibration.source_bank, calibration.profile_bank) == (2, 4)
    multi = single + "Type:PNT;Bank:5;difC:21000;difA:0;difB:0;Zero:0\n"
    receipt["profile_sha256"] = sha256(multi.encode()).hexdigest()
    service.load_text_document = lambda _: multi
    with pytest.raises(ValueError, match="unambiguous profile Bank"):
        mapped_calibration(scan, None, {"job": record}, service)


def test_packaged_checkpoint_runs_one_whole_q_pass():
    pytest.importorskip("torch")
    q = np.linspace(.2, 12.1, 4500)
    a = pattern(q, 4 + np.exp(-((q - 2) / .05) ** 2) * 100)
    b = pattern(q, 4 + np.exp(-((q - 2.03) / .05) ** 2) * 100)
    prediction = FullQTransitionPredictor().predict_pair(a, b)
    assert prediction["schema"] == "radar-pd-full-q-transition-pair/v1"
    assert prediction["inference"]["input_bins"] == 1024
    assert len(prediction["causes"]) == 4
