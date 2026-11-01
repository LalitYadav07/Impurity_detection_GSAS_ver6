"""The release tracker preserves scope and local U-Net model behavior."""

from __future__ import annotations

from dataclasses import replace
import gc
import importlib.util
import pytest
import hashlib
import os
from types import SimpleNamespace
import weakref

import numpy as np

from radar_pd_nova.pattern_transition import ObservedPattern, PatternScope
from radar_pd_nova.transition_input import observation
from radar_pd_nova.unet_core.production import (CHECKPOINT, CHECKPOINT_SHA256,
                                                 FocusTransitionPredictor)
from radar_pd_nova.unet_transition_engine import UnetTransitionEngine, available_orders
from radar_pd_nova.unet_core.runner import _quality


def curves(sample="sample-a"):
    x = np.linspace(1., 5., 500)
    y0 = 100. + 400. * np.exp(-((x - 2.) / .04) ** 2)
    y1 = 100. + 400. * np.exp(-((x - 2.04) / .04) ** 2)
    scope = PatternScope(sample, "NOMAD", "1", "q_inv_angstrom", "geometry",
                         "reduction", "counts", "density")
    a = ObservedPattern(scope, x, y0, np.full(len(x), 2.),
                        np.ones(len(x), dtype=bool), "run-a")
    b = ObservedPattern(scope, x, y1, np.full(len(x), 2.),
                        np.ones(len(x), dtype=bool), "run-b")
    return a, b


@pytest.mark.skipif(importlib.util.find_spec("torch") is None,
                    reason="optional U-Net runtime is not installed")
def test_frozen_model_proposes_native_box_for_shift_and_none_for_identity():
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == CHECKPOINT_SHA256
    predictor = FocusTransitionPredictor()
    a, b = curves()
    result = predictor.predict_pair(a, b)
    assert result["status"] == "valid"
    assert result["attention"] == "yes"
    assert any(event["x0"] <= 2. <= event["x1"] and
               "peak_shift" in event["feature_types"] for event in result["events"])
    same = predictor.predict_pair(a, replace(a, observation_id="copy"))
    assert same["attention"] in {"no", "uncertain"} and not same["events"]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None,
                    reason="optional U-Net runtime is not installed")
def test_engine_compares_only_adjacent_compatible_scans_and_rejects_duplicates():
    a, b = curves()
    other = replace(a, scope=replace(a.scope, specimen="sample-b"),
                    observation_id="other")
    items = {"a": observation(a), "b": observation(b), "other": observation(other)}
    engine = UnetTransitionEngine(loader=lambda scan: items[scan["run_id"]])
    rows = [{"run_id": name, "run_number": index}
            for index, name in enumerate(("a", "other", "b"), 1)]
    report = engine.evaluate(rows)
    assert len(report["groups"]) == 2
    assert [r["status"] for r in report["records"]] == ["initial", "initial", "valid"]
    assert report["records"][-1]["previous"] == "a"
    assert report["records"][-1]["events"]
    duplicate = engine.evaluate([rows[0], rows[0]])
    assert all(r["status"] == "unsupported" and not r["events"]
               for r in duplicate["records"])


def test_revised_source_file_invalidates_observation_cache(tmp_path):
    path = tmp_path / "curve.dat"
    path.write_text("first")
    first, second = curves()
    calls = []

    def loader(_):
        calls.append(path.read_text())
        return observation(first if calls[-1] == "first" else second)

    engine = UnetTransitionEngine(loader=loader)
    scan = {"run_id": "one", "source_path": str(path)}
    assert engine.load(scan).pattern.observation_id == "run-a"
    path.write_text("second")
    before = path.stat()
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    assert engine.load(scan).pattern.observation_id == "run-b"
    assert calls == ["first", "second"]


def test_physical_order_and_wider_reference_recompute_the_actual_pairs():
    base, _ = curves()
    temperatures = [3., 1., 2., 4., 5., 6.]
    rows = [{"run_id": f"scan-{i}", "run_number": i + 1,
             "metadata": {"temperature": {"value": value, "unit": "K"}}}
            for i, value in enumerate(temperatures)]
    items = {row["run_id"]: observation(replace(
        base, intensity=base.intensity + i, observation_id=row["run_id"]))
        for i, row in enumerate(rows)}
    calls = []

    def predict(before, after):
        pair = (before.pattern.observation_id, after.pattern.observation_id)
        calls.append(pair)
        return {"status": "valid", "attention": "yes", "events": [{
            "x0": 1., "x1": 1.1, "feature_types": ["peak_shift"],
            "attention_score": .9, "pair": pair}], "reasons": []}

    engine = UnetTransitionEngine(loader=lambda scan: items[scan["run_id"]],
                                  predictor=SimpleNamespace(
                                      metadata={"checkpoint_sha256": "test"},
                                      predict_pair=predict))
    assert "temperature" in {option["value"] for option in available_orders(rows)}
    sorted_report = engine.evaluate(rows, order_by="temperature", reference_gap=2)
    group = sorted_report["groups"][0]
    assert group["run_ids"] == ["scan-1", "scan-2", "scan-0", "scan-3", "scan-4", "scan-5"]
    by_id = {row["run_id"]: row for row in sorted_report["records"]}
    assert by_id["scan-0"]["previous"] == "scan-1"
    assert by_id["scan-0"]["events"][0]["pair"] == ("scan-1", "scan-0")
    assert by_id["scan-5"]["previous"] == "scan-3"
    assert ("scan-3", "scan-5") in calls
    first_report = engine.evaluate(rows, order_by="temperature", reference_gap="first")
    assert next(row for row in first_report["records"] if row["run_id"] == "scan-5")["previous"] == "scan-1"


def test_long_history_does_not_retain_every_native_pattern_or_tile_receipt():
    base, _ = curves()
    living = []

    def loader(scan):
        pattern = replace(base, intensity=base.intensity + scan["run_number"],
                          observation_id=scan["run_id"])
        item = observation(pattern)
        living.append(weakref.ref(item))
        return item

    def predict(_before, _after):
        return {"status": "valid", "attention": "no", "events": [], "reasons": [],
                "inference": {"window_count": 50,
                              "inferred_windows": [{"id": f"tile-{i}"} for i in range(50)],
                              "skipped_windows": []}}

    engine = UnetTransitionEngine(loader=loader, predictor=SimpleNamespace(
        metadata={"checkpoint_sha256": "test"}, predict_pair=predict))
    rows = [{"run_id": f"scan-{i}", "run_number": i}
            for i in range(64)]
    report = engine.evaluate(rows)
    gc.collect()
    assert len(report["records"]) == 64
    assert len(engine._loaded) <= 16
    assert sum(ref() is not None for ref in living) <= 18
    assert all("inferred_windows" not in row["comparison"]["inference"]
               for row in report["records"][1:])


def test_fast_native_tile_quality_preserves_gap_and_edge_decisions():
    regular = np.linspace(1., 10., 1000)
    with_gap = np.r_[regular[:500], regular[600:]]
    assert _quality(regular, (2., 3.), 32)
    assert not _quality(with_gap, (5., 6.5), 32)
    assert not _quality(regular, (1.1, 1.2), 32)


def test_descending_order_keeps_scans_without_timestamps_at_the_end():
    base, _ = curves()
    rows = [
        {"run_id": "early", "run_number": 1,
         "metadata": {"start_time": "2026-01-01T00:00:00+00:00"}},
        {"run_id": "late", "run_number": 2,
         "metadata": {"start_time": "2026-01-02T00:00:00+00:00"}},
        {"run_id": "unknown", "run_number": 3, "metadata": {}},
    ]
    items = {row["run_id"]: observation(replace(base, observation_id=row["run_id"]))
             for row in rows}
    engine = UnetTransitionEngine(
        loader=lambda scan: items[scan["run_id"]],
        predictor=SimpleNamespace(metadata={"checkpoint_sha256": "test"},
                                  predict_pair=lambda *_: {"status": "valid",
                                                           "attention": "no", "events": [],
                                                           "reasons": []}))
    report = engine.evaluate(rows, descending=True)
    assert report["groups"][0]["run_ids"] == ["late", "early", "unknown"]


def test_appending_beyond_old_cache_limit_evaluates_only_one_new_pair():
    base, _ = curves()
    calls, loads = [], []
    def load(scan):
        loads.append(scan["run_id"])
        return observation(replace(base, intensity=base.intensity + scan["run_number"],
                                   observation_id=scan["run_id"]))
    def predict(before, after):
        calls.append((before.fingerprint, after.fingerprint))
        return {"status": "valid", "attention": "no", "events": [], "reasons": []}
    engine = UnetTransitionEngine(loader=load, predictor=SimpleNamespace(
        metadata={"checkpoint_sha256": "test"}, predict_pair=predict))
    rows = [{"run_id": f"scan-{i}", "run_number": i} for i in range(320)]
    engine.evaluate(rows)
    assert len(calls) == 319
    calls.clear(); loads.clear()
    rows.append({"run_id": "scan-320", "run_number": 320})
    engine.evaluate(rows)
    assert len(calls) == 1
    assert len(loads) <= 2
    assert len(engine._loaded) <= 16


def test_revised_middle_observation_recomputes_only_its_two_comparisons():
    base, _ = curves()
    calls = []
    def load(scan):
        return observation(replace(base, intensity=base.intensity + scan["run_number"]
                                   + scan.get("revision", 0), observation_id=scan["run_id"]))
    def predict(*pair):
        calls.append(pair)
        return {"status": "valid", "attention": "no", "events": [], "reasons": []}
    engine = UnetTransitionEngine(loader=load, predictor=SimpleNamespace(
        metadata={"checkpoint_sha256": "test"}, predict_pair=predict))
    rows = [{"run_id": f"scan-{i}", "run_number": i, "metadata": {}} for i in range(40)]
    engine.evaluate(rows)
    calls.clear()
    rows[20]["revision"] = 10
    rows[20]["metadata"] = {"revision": 10}
    engine.evaluate(rows)
    assert len(calls) == 2


def test_cancelled_inference_does_not_load_or_overwrite_evidence():
    import pytest
    from radar_pd_nova.unet_transition_engine import ComparisonSuperseded
    engine = UnetTransitionEngine(loader=lambda *_: pytest.fail("obsolete request loaded a source"))
    with pytest.raises(ComparisonSuperseded):
        engine.evaluate([{"run_id": "a"}], cancelled=lambda: True)


def test_unknown_normalization_never_becomes_a_clear_physical_change_decision():
    from radar_pd_nova.unet_core.production import FEATURE_TYPES
    a, b = curves()
    scope = replace(a.scope, normalization="not_declared_in_product")
    predictor = FocusTransitionPredictor()
    predictor.predict_batch = lambda batch: {
        "attention": np.zeros((len(batch), 156), dtype=np.float32),
        "type_probabilities": {name: np.zeros((len(batch), 156), dtype=np.float32)
                               for name in FEATURE_TYPES}}
    result = predictor.predict_pair(replace(a, scope=scope), replace(b, scope=scope))
    assert result["status"] == "uncertain"
    assert result["attention"] == "uncertain"
    assert any("normalization_unqualified" in reason for reason in result["reasons"])
