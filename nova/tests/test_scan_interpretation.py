"""Independent simulated events, qualification failures and online replay."""

import copy
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from radar_pd_nova.scan_interpretation import (
    InterpretationSettings,
    Pattern,
    ScanInterpreter,
    compare_patterns,
    explain_comparison,
    load_pattern,
    PATTERN_SCHEMA,
    _conditions,
)


def packet(
    *,
    centers=(2, 4, 6, 8),
    amplitudes=None,
    width=0.035,
    scale=1,
    offset=0,
    shift=0,
    noise=0,
    seed=12,
    masked=(),
    residual=False,
):
    x = np.linspace(1, 9, 2401)
    amplitudes = amplitudes or [100] * len(centers)
    y = 15 + 0.2 * x
    for center, amplitude in zip(centers, amplitudes):
        y += amplitude * np.exp(-0.5 * ((x - center - shift) / width) ** 2)
    calculated = scale * y + offset
    y = calculated + np.random.default_rng(seed).normal(0, noise, len(x))
    if residual:
        y += 40 * np.exp(-0.5 * ((x - 5) / 0.03) ** 2)
    valid = np.ones(len(x), dtype=bool)
    for a, b in masked:
        valid &= ~((x >= a) & (x <= b))
    return {
        "$schema": PATTERN_SCHEMA,
        "source_gpx_sha256": str(seed),
        "coordinate": {"kind": "d", "unit": "angstrom"},
        "context": {
            "profile_sha256": "profile",
            "reduction_family": "IDL",
            "normalization": "same recipe",
        },
        "fit_domain": {"limits": [1, 9], "excluded": []},
        "x": x.tolist(),
        "y": y.tolist(),
        "ycalc": calculated.tolist(),
        "sigma": [1.0] * len(x),
        "valid": valid.tolist(),
        "fit_valid": valid.tolist(),
    }


def scan(run, *, temperature=300, bank=2, sample="s1", pattern=None, **kwargs):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=run)
    record = {
        "run_id": f"NOM_{run}_BANK{bank}",
        "run_number": run,
        "bank": bank,
        "instrument": "NOM",
        "ipts": "IPTS-1",
        "configuration_id": "cfg1",
        "pattern": pattern or packet(),
        "metadata": {
            "sample_id": sample,
            "start_time": start.isoformat(),
            "end_time": (start + timedelta(seconds=30)).isoformat(),
            "conditions": {
                "channels": [
                    {
                        "quantity": "temperature",
                        "channel": "selectedT",
                        "role": "measured",
                        "status": "recorded",
                        "unit": "K",
                        "value": temperature,
                        "minimum": temperature,
                        "maximum": temperature,
                    }
                ]
            },
        },
        "observables": {
            "source_gpx_sha256": f"gpx{run}",
            "context": {"recipe_key": "full-v1"},
            "fraction_basis": "modeled crystalline mass",
            "phases": [
                {
                    "phase_id": "CIF1",
                    "basis_key": "basis1",
                    "a": 4.0,
                    "weight_percent": 100.0,
                    "fraction_state": "refined",
                    "metric_states": {"a": "refined"},
                }
            ],
        },
    }
    record["pattern"] = copy.deepcopy(record["pattern"])
    record["pattern"]["source_gpx_sha256"] = f"gpx{run}"
    record.update(kwargs)
    return record


def codes(report, index=-1):
    return {f["code"] for f in report["records"][index]["findings"]}


def evaluate(*scans, **kwargs):
    return ScanInterpreter().evaluate(scans, **kwargs)


def test_powgen_readback_jitter_keeps_qualified_series_and_original_provenance():
    from radar_pd_nova.scan_interpretation import _group
    from radar_pd_nova.transition_input import packet_observation
    setup = {"cycle": "2022-1_11A_CAL", "mode": "HR", "frequency": 60,
             "wavelength": 0.8, "calibration_files": ["calibration.h5"],
             "characterization_files": ["PG3_HighRes.txt"], "grouping_files": []}
    groups, transition_scopes = [], []
    for n, guide in enumerate([-54.004, -54.003, -54.007]):
        row = scan(n + 1, instrument="PG3")
        row["pattern"]["context"]["reduction_setup"] = {**setup, "guide": guide}
        groups.append(_group(row, Pattern.parse(row["pattern"])))
        transition_scopes.append(packet_observation(row).pattern.scope)
        assert row["pattern"]["context"]["reduction_setup"]["guide"] == guide
    assert len(set(groups)) == 1
    assert len(set(transition_scopes)) == 1
    for change in [{"cycle": "2021_2_11A_CAL"}, {"mode": "HI", "guide": 54.0},
                   {"calibration_files": ["different.h5"]}, {"grouping_files": ["custom.xml"]}]:
        changed = copy.deepcopy(row)
        changed["pattern"]["context"]["reduction_setup"].update(change)
        assert _group(changed, Pattern.parse(changed["pattern"])) != groups[0]
        assert packet_observation(changed).pattern.scope != transition_scopes[0]
    contradicted = copy.deepcopy(row)
    contradicted["pattern"]["context"]["reduction_setup"]["guide"] = 54
    with pytest.raises(ValueError, match="optical mode"):
        _group(contradicted, Pattern.parse(contradicted["pattern"]))


@pytest.mark.parametrize("instrument,mode", [("NOM", "HR"), ("PG3", "")])
def test_unqualified_or_other_beamline_readback_does_not_get_canonicalized(instrument, mode):
    from radar_pd_nova.scan_interpretation import _group
    first = scan(1, instrument=instrument)
    first["pattern"]["context"]["reduction_setup"] = {
        "cycle": "2022", "mode": mode, "calibration_files": ["cal.h5"],
        "characterization_files": ["char.txt"], "guide": -54.004}
    second = copy.deepcopy(first)
    second["pattern"]["context"]["reduction_setup"]["guide"] = -54.003
    assert _group(first, Pattern.parse(first["pattern"])) != _group(second, Pattern.parse(second["pattern"]))


def compare(a, b):
    return compare_patterns(Pattern.parse(a), Pattern.parse(b), InterpretationSettings())


def test_identical_and_intensity_scale_offset_are_separate_from_local_structure():
    a, b = packet(), packet(scale=1.5, offset=12)
    features = compare(a, b)
    assert features["scale"] == pytest.approx(1.5)
    assert features["offset"] == pytest.approx(12)
    assert features["shape_effect"] < 1e-12
    result = evaluate(scan(1, pattern=a), scan(2, pattern=b))
    assert codes(result) == {"global_scale_change", "background_change"}
    assert codes(evaluate(scan(1), scan(2))) == {"no_resolved_event"}


@pytest.mark.parametrize(
    "event,expected",
    [
        ({"shift": 0.02}, "peak_movement"),
        ({"centers": (2, 4, 5, 6, 8)}, "new_features"),
        ({"centers": (2, 4, 8)}, "lost_features"),
        ({"width": 0.065}, "profile_shape_change"),
        ({"amplitudes": [100, 150, 100, 100]}, "relative_intensity_redistribution"),
        ({"centers": (1.94, 2.06, 4, 6, 8), "width": 0.02}, "profile_shape_change"),
    ],
)
def test_controlled_local_events(event, expected):
    result = evaluate(scan(1), scan(2, pattern=packet(**event)))
    assert expected in codes(result), result["records"][-1]["findings"]
    finding = next(f for f in result["records"][-1]["findings"] if f["code"] == expected)
    assert finding["regions"]


def test_masks_prevent_false_missing_or_added_peaks():
    features = compare(packet(), packet(centers=(2, 4, 8), masked=((5.7, 6.3),)))
    assert features["status"] == "compared"
    assert not features["new_features"] and not features["lost_features"]
    assert features["shape_effect"] < 1e-12


def test_grid_units_flatness_and_coverage_fail_closed():
    changed = packet()
    changed["x"] = (np.asarray(changed["x"]) + 0.0001).tolist()
    assert compare(packet(), changed)["status"] == "unavailable"
    changed = packet()
    changed["coordinate"]["unit"] = "nm"
    assert compare(packet(), changed)["status"] == "unavailable"
    changed = packet()
    changed["y"] = [1.0] * len(changed["x"])
    assert compare(packet(), changed)["status"] == "unavailable"
    changed = packet()
    changed["valid"] = [i < 30 for i in range(len(changed["x"]))]
    changed["fit_valid"] = changed["valid"]
    assert compare(packet(), changed)["status"] == "unavailable"


@pytest.mark.parametrize(
    "key,value",
    [
        ("x", [1] * 20),
        ("y", [float("nan")] * 2401),
        ("valid", [1] * 2401),
        ("fit_valid", [True] * 4),
    ],
)
def test_malformed_pattern_is_rejected(key, value):
    data = packet()
    data[key] = value
    with pytest.raises(ValueError):
        Pattern.parse(data)


def test_legacy_summary_never_overrides_rejected_new_condition():
    a = scan(1)
    a["metadata"]["temperature"] = {"value": 400, "unit": "K"}
    a["metadata"]["conditions"]["channels"][0]["status"] = "conflict"
    assert not _conditions(a)


@pytest.mark.parametrize("change", ["sample", "bank", "profile", "family", "revision", "identity"])
def test_context_boundaries_block_physical_comparison(change):
    a, b = scan(1), scan(2)
    if change == "sample":
        b["metadata"]["sample_id"] = "s2"
    if change == "bank":
        b["bank"] = 5
    if change == "profile":
        b["pattern"]["context"]["profile_sha256"] = "other"
    if change == "family":
        b["pattern"]["context"]["reduction_family"] = "Mantid-live"
    if change == "revision":
        b["metadata"]["source_revision_warning"] = "changed"
    if change == "identity":
        b["metadata"]["identity_status"] = "conflict"
    result = evaluate(a, b)
    assert not result["records"][-1]["comparisons"]


def test_acquisition_order_not_arrival_order_and_deterministic_report():
    a, b, c = scan(1), scan(2), scan(3)
    forward, reverse = evaluate(a, b, c), evaluate(c, a, b)
    assert forward == reverse
    assert [r["run_id"] for r in forward["records"]] == [a["run_id"], b["run_id"], c["run_id"]]
    assert forward["records"][-1]["comparisons"][0]["reference_run_id"] == b["run_id"]
    assert forward["evidence_sha256"]


def test_pending_earlier_scan_prevents_silent_reference_pin_but_unrelated_ipts_does_not():
    a, b = scan(1), scan(2)
    result = evaluate(b, pending=[a])
    assert not result["references"] and result["records"][0]["provisional"]
    a["ipts"] = "IPTS-other"
    assert evaluate(b, pending=[a])["references"]


def test_reference_revision_is_pinned_and_requires_explicit_replacement():
    a, b = scan(1), scan(2)
    result = evaluate(a, b)
    settings = InterpretationSettings.model_validate(result["settings"])
    a["pattern"] = packet(scale=2)
    a["pattern"]["source_gpx_sha256"] = "gpx1"
    replay = evaluate(a, b, settings=settings)
    assert any("reference content changed" in n for n in replay["records"][-1]["notices"])
    assert not any("reference" in c["roles"] for c in replay["records"][-1]["comparisons"])


def test_reductions_of_one_measurement_are_not_independent_confirmations():
    a, revision, b = scan(1), scan(1), scan(2)
    revision["run_id"] += "_r2"
    revision["observables"]["source_gpx_sha256"] = "different"
    revision["pattern"]["source_gpx_sha256"] = "different"
    result = evaluate(a, revision, b)
    assert "revision_conflict" in codes(result, 0)
    assert not result["records"][-1]["comparisons"]
    settings = InterpretationSettings(
        active_revisions={result["records"][0]["measurement_key"]: "different"}
    )
    result = evaluate(a, revision, b, settings=settings)
    assert result["records"][-1]["comparisons"][0]["reference_run_id"] == revision["run_id"]


def test_candidate_churn_is_not_reported_as_phase_appearance_or_fraction_change():
    a, b = scan(1), scan(2)
    b["observables"]["phases"][0]["phase_id"] = "other-stage"
    result = evaluate(a, b)
    assert "candidate_assignment_changed" in codes(result)
    assert not result["records"][-1]["comparisons"][0]["model_changes"]["fraction_basis_compatible"]


def test_eligible_cells_and_fraction_basis_and_recipe_require_real_receipts():
    a, b = scan(1), scan(2)
    b["observables"]["phases"][0].update(a=4.01, weight_percent=95.0)
    result = evaluate(a, b)
    assert {"refined_cell_change", "modeled_fraction_change"} <= codes(result)
    b["configuration_id"] = "changed"
    result = evaluate(a, b)
    assert not {"refined_cell_change", "modeled_fraction_change"} & codes(result)


def test_matched_condition_history_can_have_multiple_roles_and_does_not_imply_hysteresis():
    result = evaluate(scan(1), scan(2, temperature=320), scan(3, pattern=packet(shift=0.02)))
    match = next(
        c for c in result["records"][-1]["comparisons"] if c["reference_run_id"] == "NOM_1_BANK2"
    )
    assert {"reference", "matched_conditions", "similar_history"} <= set(match["roles"])
    assert "history_difference" in codes(result)
    assert "hysteresis" not in json.dumps(result).lower()


def test_sweeps_and_unrecorded_requests_do_not_certify_matching_conditions():
    a, b, c = scan(1), scan(2), scan(3)
    c["metadata"]["conditions"]["channels"][0]["maximum"] = 400
    result = evaluate(a, b, c)
    assert "condition_sweep" in codes(result)
    assert not any("matched_conditions" in r["roles"] for r in result["records"][-1]["comparisons"])


def test_persistence_uses_distinct_nonoverlapping_windows_and_fixed_reference():
    changed = packet(centers=(2, 4, 5, 6, 8))
    result = evaluate(scan(1), scan(2, pattern=changed), scan(3, pattern=changed))
    assert "features_persist" in codes(result)
    a, b, c = scan(1), scan(2, pattern=changed), scan(3, pattern=changed)
    c["metadata"]["start_time"] = b["metadata"]["start_time"]
    assert "features_persist" not in codes(evaluate(a, b, c))


def test_reproducible_positive_residuals_trigger_model_review():
    result = evaluate(
        scan(1, pattern=packet(residual=True)), scan(2, pattern=packet(residual=True))
    )
    assert "persistent_unexplained_features" in codes(result)


def test_repeated_evolution_uses_qualified_recorded_conditions():
    result = evaluate(
        scan(1), scan(2, pattern=packet(shift=0.02)), scan(3, pattern=packet(shift=0.04))
    )
    assert "continuing_pattern_evolution" in codes(result)
    assert "stop" not in {f["code"] for r in result["records"] for f in r["findings"]}


def test_pattern_checksum_bound_and_cache_change_detection(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps(packet()))
    with pytest.raises(ValueError, match="checksum"):
        load_pattern(path, expected_sha256="wrong")
    a = scan(1)
    a.pop("pattern")
    a["pattern_path"] = str(path)
    engine = ScanInterpreter()
    first = engine.pattern(a).key
    path.write_text(json.dumps(packet(scale=2)))
    assert engine.pattern(a).key != first


def test_no_mutation_of_input_or_settings():
    inputs = [scan(1), scan(2)]
    old = copy.deepcopy(inputs)
    settings = InterpretationSettings()
    oldsettings = settings.model_dump()
    result = ScanInterpreter().evaluate(inputs, settings)
    assert inputs == old and settings.model_dump() == oldsettings
    json.dumps(result, allow_nan=False)


def test_long_static_noise_sequence_has_no_local_structure_events():
    # Independent 0.5-intensity noise, reported sigma 1; no future data used.
    scans = [scan(i + 1, pattern=packet(noise=0.5, seed=i)) for i in range(30)]
    report = ScanInterpreter().evaluate(scans)
    local = {
        "new_features",
        "lost_features",
        "peak_movement",
        "profile_shape_change",
        "relative_intensity_redistribution",
    }
    assert not [
        (r["run_id"], codes(report, i) & local)
        for i, r in enumerate(report["records"])
        if codes(report, i) & local
    ]


def test_device_mapping_requires_qualified_units_coverage_and_scope():
    from radar_pd_nova.scan_interpretation import condition_scope

    a = scan(1)
    pressure = {
        "quantity": "pressure",
        "role": "measured",
        "channel": "gauge1",
        "status": "needs_mapping",
        "unit": "Pa",
        "value": 1000,
        "minimum": 1000,
        "maximum": 1000,
        "coverage_fraction": 1,
        "statistic": "time_weighted_step_hold",
    }
    a["metadata"]["conditions"]["channels"].append(pressure)
    settings = InterpretationSettings(
        condition_bindings={condition_scope(a): {"pressure": "gauge1"}}
    )
    assert "pressure" not in _conditions(a)
    assert _conditions(a, settings)["pressure"]["specimen_mapping"]
    pressure["status"] = "unqualified_units"
    assert "pressure" not in _conditions(a, settings)
    pressure["status"] = "needs_mapping"
    pressure["coverage_fraction"] = 0.5
    assert "pressure" not in _conditions(a, settings)
    pressure["coverage_fraction"] = 1
    a["metadata"]["sample_id"] = "another-specimen"
    assert "pressure" not in _conditions(a, settings)


def test_missing_error_coverage_is_explicit_and_does_not_create_an_event():
    a, b = scan(1), scan(2, pattern=packet(shift=0.02))
    a["pattern"]["sigma"] = [None] * len(a["pattern"]["x"])
    result = evaluate(a, b)
    assert "difference_errors_unavailable" in codes(result)
    assert "peak_movement" not in codes(result)


def test_a_curve_from_another_fitted_project_cannot_explain_this_result():
    a, b = scan(1), scan(2)
    b["pattern"]["source_gpx_sha256"] = "stale-project"
    result = evaluate(a, b)
    assert not result["records"][-1]["comparisons"]
    assert "comparison_unresolved" in codes(result)
    assert any("different projects" in n for n in result["records"][-1]["notices"])
