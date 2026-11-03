"""Regression coverage for display caching and comparison provenance."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from radar_pd_nova.scan_conditions import _unit, qualified_metadata
from radar_pd_nova.scan_interpretation import PATTERN_SCHEMA
from radar_pd_nova.transition_input import observation, packet_observation, reduced_observation
from radar_pd_nova.unet_transition_engine import (
    UnetTransitionEngine, available_orders, condition_order_value, source_signature,
)
from test_transition_input import product
from test_unet_transition_engine import curves


def fake_engine(loader=None, **kwargs):
    return UnetTransitionEngine(loader=loader, predictor=SimpleNamespace(
        metadata={"checkpoint_sha256": "synthetic"},
        predict_pair=lambda *_: {"status": "valid", "attention": "no",
                                 "events": [], "reasons": []}), **kwargs)


def packet_scan(number):
    x = np.linspace(1., 5., 64)
    return {"run_id": f"NOM_{number}", "run_number": number, "instrument": "NOM",
            "ipts": "IPTS-1", "bank": 2, "source_family": "family-A",
            "comparison_setup_id": "setup-A", "metadata": {"sample_id": "sample-A"},
            "pattern": {"$schema": PATTERN_SCHEMA,
                        "coordinate": {"kind": "d", "unit": "angstrom"},
                        "context": {"profile_sha256": "profile-A", "normalization": "pCharge"},
                        "x": x.tolist(), "y": np.full(len(x), 10. + number).tolist(),
                        "ycalc": np.full(len(x), 10. + number).tolist(),
                        "fit_valid": [True] * len(x),
                        "sigma": [1.] * len(x), "valid": [True] * len(x)}}


def test_default_map_window_and_inspected_pair_are_reused_on_redraw():
    base, _ = curves()
    calls = []

    def loader(scan):
        calls.append(scan["run_id"])
        return observation(replace(base, observation_id=scan["run_id"]))

    engine = fake_engine(loader)
    scans = [{"run_id": f"run-{i}"} for i in range(55)]
    window = scans[1:31]
    displayed = [scans[-1], scans[-2], *window]
    engine.retain_sources(displayed)
    # Same order as the audited map: window first, selected/reference last.
    for _ in range(3):
        for scan in [*window, scans[-1], scans[-2]]:
            engine.load(scan)
    assert len(calls) == len(set(calls)) == 32
    # A new inference working set must not evict the still-displayed sources.
    for scan in scans[31:51]:
        engine.load(scan)
    calls.clear()
    for scan in displayed:
        engine.load(scan)
    assert calls == []
    engine.retain_sources([])
    assert len(engine._loaded) <= 16


def test_raw_cache_byte_budget_prioritizes_selection_and_skips_oversized_arrays():
    base, _ = curves()
    calls = []

    def loader(scan):
        calls.append(scan["run_id"])
        return observation(replace(base, intensity=base.intensity.copy(),
                                   observation_id=scan["run_id"]))

    size = sum(values.nbytes for values in (base.x, base.intensity, base.sigma, base.valid))
    engine = fake_engine(loader, raw_cache_bytes=2 * size)
    scans = [{"run_id": name} for name in ("selected", "reference", "row-1", "row-2")]
    engine.retain_sources(scans)
    for scan in scans:
        engine.load(scan)
        assert engine._loaded_bytes <= 2 * size
    calls.clear()
    assert engine.load(scans[0]).pattern.observation_id == "selected"
    assert engine.load(scans[1]).pattern.observation_id == "reference"
    assert calls == []
    engine.retain_sources([])
    tiny = fake_engine(loader, raw_cache_bytes=size - 1)
    tiny.load(scans[0])
    assert not tiny._loaded and tiny._loaded_bytes == 0


@pytest.mark.parametrize("field,new_value", [
    ("source_family", "family-B"), ("comparison_setup_id", "setup-B"),
    ("ipts", "IPTS-2"), ("instrument", "PG3")])
def test_source_scope_revisions_make_warm_and_fresh_engines_agree(field, new_value):
    scans = [packet_scan(1), packet_scan(2)]
    warm = fake_engine()
    assert len(warm.evaluate(scans)["groups"]) == 1
    revised = deepcopy(scans)
    revised[1][field] = new_value
    assert source_signature(scans[1]) != source_signature(revised[1])
    warm_report = warm.evaluate(revised)
    fresh_report = fake_engine().evaluate(revised)
    assert len(warm_report["groups"]) == len(fresh_report["groups"]) == 2
    assert warm_report["records"][1]["status"] == fresh_report["records"][1]["status"] == "initial"
    assert warm_report["records"][1]["previous"] is None


def test_qualified_canonical_pressure_and_mixed_units_have_identical_ordering():
    scans = [packet_scan(1), packet_scan(2)]
    for scan, value in zip(scans, (1e9, 2e9)):
        scan["metadata"] = qualified_metadata({
            "sample_id": "sample-A", "conditions": {"registry": "SNS-PG3-NOM-conditions/2",
            "channels": [{"quantity": "pressure", "role": "measured", "status": "recorded",
                          "channel": "reviewed gauge", "binding_verified": True,
                          "unit": "Pa", "value": value, "minimum": value, "maximum": value}]}})
    assert [condition_order_value(scan, "pressure")[0] for scan in scans] == [1., 2.]
    assert {option["value"] for option in available_orders(scans)} >= {"pressure"}
    for unit, value in (("Pa", 1e9), ("kPa", 1e6), ("MPa", 1e3),
                        ("GPa", 1.), ("kbar", 10.), ("bar", 1e4)):
        scans[0]["metadata"]["pressure"] = {"value": value, "unit": unit}
        assert condition_order_value(scans[0], "pressure")[0] == pytest.approx(1.)
        assert _unit("pressure", unit) is not None


def test_reduction_warning_survives_raw_and_published_inputs_with_normalization(tmp_path):
    warning = "Reduction records an intensity positivity shift; quantitative intensity interpretation needs review"
    raw, _, _ = product(tmp_path)
    raw["metadata"]["nomad_setup"] = {"warnings": [warning]}
    item = reduced_observation(raw)
    assert item.pattern.scope.normalization == "pCharge"
    assert warning in item.notices
    published = packet_scan(1)
    published["pattern"]["context"]["warnings"] = [{"code": "nomad_reduction_quality", "message": warning}]
    published["scientific_warnings"] = [warning]
    assert packet_observation(published).notices == [warning]
    second = packet_scan(2)
    report = fake_engine().evaluate([published, second])
    assert report["records"][1]["notices"] == [warning]
    assert packet_observation(published).pattern.scope == packet_observation(second).pattern.scope
    # Updating only warning provenance must refresh an already-warm summary.
    engine = fake_engine()
    assert engine.evaluate([second])["records"][0]["notices"] == []
    second["scientific_warnings"] = [warning]
    assert engine.evaluate([second])["records"][0]["notices"] == [warning]
