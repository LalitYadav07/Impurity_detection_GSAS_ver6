import json
import zipfile
from copy import deepcopy

import pytest
from hashlib import sha256
from pathlib import Path

import numpy as np

from radar_pd_nova.results import (
    build_result_view,
    complete_experiment_space_groups,
    curate_rapid_rows,
    discover_plot_payloads,
    discover_tables,
    experiment_fit_diagnostics,
    experiment_fit_quality_figure,
    experiment_axis_options,
    experiment_phase_fraction_figure,
    experiment_phase_heatmap_figure,
    experiment_sample_identity,
    experiment_scan_summary,
    figure_for_payload,
    load_plot_with_fallback,
    phase_fraction_rows,
    read_plot_payload,
    run_status_badge,
    total_elapsed_seconds,
)


def test_experiment_scan_summary_uses_best_refined_rapid_hypothesis() -> None:
    payload = {
        "status": "complete",
        "analysis_mode": "rapid",
        "run_name": "IPTS-38000_PG3_63764",
        "summary": {"live_run": {"timings": {"total_seconds": 123.4}}},
        "hypotheses": [
            {
                "gsas_rwp_rank": "2",
                "rwp": "18.0",
                "status": "ok",
                "weights_json": '{"Wrong (SG 1)": 100}',
            },
            {
                "gsas_rwp_rank": "1",
                "rwp": "11.25",
                "status": "ok",
                "weights_json": '{"YFeSi (SG 62)": 70, "Ga (SG 225)": 30}',
            },
        ],
        "warnings": ["review"],
    }

    summary = experiment_scan_summary(payload)

    assert summary["rwp"] == 11.25
    assert summary["elapsed_seconds"] == 123.4
    assert [row["phase"] for row in summary["phases"]] == ["YFeSi", "Ga"]
    assert summary["warning_count"] == 1
    assert summary["final_rwp"] is None


def test_result_status_retains_explicit_scientific_review_requirement(tmp_path):
    view = build_result_view({
        "status": "complete", "analysis_mode": "full",
        "scientific_status": "needs_review", "final_rwp": 0.105,
    }, tmp_path)
    assert view.status == "Complete"
    assert view.metrics[1] == {
        "label": "Status", "value": "Complete · scientific review required",
    }
    assert view.metrics[2]["value"] == "0.105%"
    assert view.review_status == "needs_review"
    assert run_status_badge("Ok", view.review_status) == {
        "label": "Needs scientific review", "color": "#fff0d4",
    }


def test_result_badge_does_not_present_missing_convergence_as_success(tmp_path):
    view = build_result_view({
        "status": "complete", "analysis_mode": "full", "final_rwp": 0.1,
    }, tmp_path)
    assert view.review_status == "unknown"
    assert run_status_badge("Ok", view.review_status)["label"] == "Convergence unknown"
    assert run_status_badge("Error", view.review_status)["label"] == "Error"
    assert run_status_badge("Running", "needs_review")["label"] == "Running"


def test_experiment_summary_preserves_explicit_final_rwp_separately_from_candidate_metrics():
    result=experiment_scan_summary({"status":"complete","analysis_mode":"full","final_rwp":12.5,
        "hypotheses":[{"rwp":8,"status":"ok"}]})
    assert result["final_rwp"]==12.5


def test_refinement_plot_uses_explicit_limits_and_preserves_excluded_gaps():
    payload = {
        "plot_kind": "gsas_fit_with_ticks_v1",
        "fit_limits": [2, 8],
        "excluded_ranges": [[4, 6]],
        "arrays": {
            "x": [1, 2, 3, 7, 8, 9],
            "yobs": [100, 20, 30, 70, 80, 900],
            "ycalc": [0, 19, 29, 69, 79, 0],
            "resid": [100, 1, 1, 1, 1, 900],
            "valid": [True] * 6,
        },
    }
    figure = figure_for_payload(payload)
    assert list(figure.data[0].y) == [None, 20, 30, 70, 80, None]
    assert list(figure.data[1].x) == [1, 2, 3, None, 7, 8, 9]
    assert list(figure.data[1].y) == [None, 19, 29, None, 69, 79, None]
    assert list(figure.data[2].y) == [None, 1, 1, None, 1, 1, None]
    assert list(figure.layout.xaxis.range) == [2, 8]
    assert list(figure.layout.xaxis3.range) == [2, 8]
    assert figure.data[1].connectgaps is False
    assert figure.data[-1].name == "Outside fit (valid observations)"
    assert list(figure.data[-1].y) == [100, None, None, None, None, 900]


def test_refinement_plot_masks_invalid_bins_without_using_intensity_sign():
    payload = {"plot_kind": "gsas_fit_with_ticks_v1", "fit_limits": [1, 6],
               "excluded_ranges": [[5, 5.5]], "arrays": {
        "x": [1, 2, 3, 4, 5, 6], "yobs": [10, 0, -3, 0, 20, 30],
        "ycalc": [9, 1, -2, 8, 19, 29], "resid": [1, -1, -1, -8, 1, 1],
        "valid": [True, True, True, False, True, True]}}
    original = deepcopy(payload)
    figure = figure_for_payload(payload)
    assert list(figure.data[0].y) == [10, 0, -3, None, None, 30]
    difference = dict(zip(figure.data[2].x, figure.data[2].y))
    assert difference[2] == -1 and difference[3] == -1
    assert difference[4] is None and difference[5] is None
    assert figure.data[-1].name == "Outside fit (valid observations)"
    assert list(figure.data[-1].y) == [None, None, None, None, 20, None]
    assert payload == original


def test_refinement_plot_accepts_explicit_weights_and_withholds_unknown_difference():
    payload = {"plot_kind": "gsas_fit_with_ticks_v1", "fit_limits": [1, 3], "arrays": {
        "x": [1, 2, 3], "yobs": [0, -2, 0], "ycalc": [1, -1, 3],
        "resid": [-1, -1, -3], "weights": [1, 2, 0]}}
    figure = figure_for_payload(payload)
    assert list(figure.data[0].y) == [0, -2, None]
    assert list(figure.data[2].y) == [-1, -1, None]
    del payload["arrays"]["weights"]
    legacy = figure_for_payload(payload)
    assert list(legacy.data[0].y) == [0, -2, 0]
    assert all(y is None for y in legacy.data[2].y)
    assert any("validity unavailable" in a.text for a in legacy.layout.annotations)


def _native_fit_plot_payload(instrument="TOF"):
    from gpx_recipe.fit_domain import NATIVE_FIT_CONVENTION
    step, origin = (20., 1000.) if instrument == "TOF" else (.05, 10.)
    x = [origin + step * index for index in range(20)]
    valid = [index not in (10, 11, 12, 13) for index in range(20)]
    fitted = [valid[index] and 3 <= index <= 16 and index not in (6, 7) for index in range(20)]
    observed = [float(index + 20) for index in range(20)]
    observed[4:6] = [-3., 0.]
    return {"schema_version": "1.1", "plot_kind": "gsas_fit_with_ticks_v1",
            "fit_range_convention": NATIVE_FIT_CONVENTION,
            "fit_limits": [x[2] + .2 * step, x[15] + .2 * step],
            "excluded_ranges": [[x[6], x[7]]], "arrays": {
                "x": x, "yobs": observed, "ycalc": [value - .25 for value in observed],
                "resid": [.25] * 20, "weights": [0. if index == 12 else 4. for index in range(20)],
                "valid": valid, "fit_valid": fitted}}


@pytest.mark.parametrize("instrument", ["TOF", "CW"])
@pytest.mark.parametrize("numeric_strings", [False, True])
def test_native_plot_includes_upper_bin_and_masks_excluded_or_invalid_points(instrument, numeric_strings):
    payload = _native_fit_plot_payload(instrument)
    if numeric_strings:
        payload["fit_limits"] = list(map(str, payload["fit_limits"]))
        payload["excluded_ranges"] = [list(map(str, pair)) for pair in payload["excluded_ranges"]]
    before = deepcopy(payload)
    figure = figure_for_payload(payload)
    arrays = payload["arrays"]
    calculated = dict(zip(figure.data[1].x, figure.data[1].y))
    difference = dict(zip(figure.data[2].x, figure.data[2].y))
    for index, x in enumerate(arrays["x"]):
        expected = arrays["fit_valid"][index]
        assert (calculated[x] is not None) is expected
        assert (difference[x] is not None) is expected
        if expected:
            assert calculated[x] == arrays["ycalc"][index]
    assert arrays["x"][16] > float(payload["fit_limits"][1])
    assert figure.layout.xaxis.range[1] == arrays["x"][16]
    assert "N (fit) 8" in figure.layout.title.text
    assert figure.data[0].y[4] == -3 and figure.data[0].y[5] == 0
    assert None in figure.data[1].x and figure.data[1].connectgaps is False
    assert payload == before


@pytest.mark.parametrize("defect", ["missing_valid", "missing_fit", "numeric_mask", "missing_bin", "extra_bin",
                                   "nonfinite_x", "reordered_x", "nonfinite_observation", "zero_weight",
                                   "nonfinite_calculation", "nonfinite_residual", "unknown_convention", "legacy_schema",
                                   "missing_convention", "null_convention", "missing_convention_and_masks",
                                   "missing_limits", "missing_exclusions", "missing_weights", "missing_residual",
                                   "forged_residual"])
def test_native_plot_rejects_bad_masks_and_nonfinite_or_reordered_evidence(defect):
    payload = _native_fit_plot_payload()
    arrays = payload["arrays"]
    if defect == "missing_valid": arrays.pop("valid")
    elif defect == "missing_fit": arrays.pop("fit_valid")
    elif defect == "numeric_mask": arrays["fit_valid"] = list(map(int, arrays["fit_valid"]))
    elif defect == "missing_bin": arrays["fit_valid"][16] = False
    elif defect == "extra_bin": arrays["fit_valid"][17] = True
    elif defect == "nonfinite_x": arrays["x"][0] = float("nan")
    elif defect == "reordered_x": arrays["x"] = arrays["x"][::-1]
    elif defect == "nonfinite_observation": arrays["yobs"][4] = float("inf")
    elif defect == "zero_weight": arrays["weights"][4] = 0.
    elif defect == "nonfinite_calculation": arrays["ycalc"][4] = float("nan")
    elif defect == "nonfinite_residual": arrays["resid"][4] = float("nan")
    elif defect == "legacy_schema": payload["schema_version"] = "1.0"
    elif defect == "missing_convention": payload.pop("fit_range_convention")
    elif defect == "null_convention": payload["fit_range_convention"] = None
    elif defect == "missing_convention_and_masks":
        payload.pop("fit_range_convention"); arrays.pop("valid"); arrays.pop("fit_valid")
    elif defect == "missing_limits": payload.pop("fit_limits")
    elif defect == "missing_exclusions": payload.pop("excluded_ranges")
    elif defect == "missing_weights": arrays.pop("weights")
    elif defect == "missing_residual": arrays.pop("resid")
    elif defect == "forged_residual": arrays["resid"][4] = 1e6
    else: payload["fit_range_convention"] = "unreviewed convention"
    figure = figure_for_payload(payload)
    assert all(value is None for value in figure.data[1].y)
    assert all(value is None for value in figure.data[2].y)
    assert any("unavailable" in annotation.text for annotation in figure.layout.annotations)


def test_native_schema_does_not_fall_back_to_legacy_without_explicit_convention():
    payload = _native_fit_plot_payload()
    payload.pop("fit_range_convention")
    figure = figure_for_payload(payload)
    assert all(value is None for value in figure.data[0].y)
    assert all(value is None for value in figure.data[1].y)
    assert all(value is None for value in figure.data[2].y)
    assert figure.layout.xaxis.range[1] == payload["fit_limits"][1]


def test_legacy_schema_keeps_its_closed_bounds_and_mask_validation():
    payload = _native_fit_plot_payload()
    payload["schema_version"] = "1.0"
    payload.pop("fit_range_convention")
    figure = figure_for_payload(payload)
    assert all(value is None for value in figure.data[2].y)
    assert figure.layout.xaxis.range[1] == payload["fit_limits"][1]


def test_native_residual_allows_roundoff_and_invalid_bin_nan_only():
    payload = _native_fit_plot_payload()
    arrays = payload["arrays"]
    arrays["resid"][4] += 1e-13
    for index in (0, 6, 10):
        arrays["resid"][index] = float("nan")  # Outside, excluded, and invalid bins.
    arrays["yobs"][10] = arrays["ycalc"][10] = float("nan")
    figure = figure_for_payload(payload)
    difference = dict(zip(figure.data[2].x, figure.data[2].y))
    assert difference[arrays["x"][4]] == pytest.approx(.25)
    assert all(difference[arrays["x"][index]] is None for index in (0, 6, 10))
    assert "N (fit) 8" in figure.layout.title.text


@pytest.mark.parametrize("canonical,alias", [("x", "two_theta"), ("yobs", "observed"),
                                             ("ycalc", "calculated"), ("resid", "difference")])
def test_native_canonical_array_cannot_be_rescued_by_a_legacy_alias(canonical, alias):
    payload = _native_fit_plot_payload()
    arrays = payload["arrays"]
    arrays[alias] = arrays[canonical]
    arrays[canonical] = ["corrupt"]
    figure = figure_for_payload(payload)
    assert all(value is None for value in figure.data[1].y)
    assert all(value is None for value in figure.data[2].y)
    assert any("unavailable" in annotation.text for annotation in figure.layout.annotations)


def _bound_plot_fixture(tmp_path):
    x = list(range(1, 17))
    y = [float(v) for v in x]
    y[1], y[2], y[6] = 0.0, -2.0, 0.0
    valid = [i != 6 for i in range(16)]
    packet = {"$schema": "radar-pd-pattern-observation/v1", "x": x, "y": y,
              "sigma": [1.0 if v else None for v in valid], "valid": valid,
              "fit_valid": [v and i >= 9 for i, v in enumerate(valid)],
              "ycalc": [y[i] - .1 if v and i >= 9 else None for i, v in enumerate(valid)],
              "coordinate": {"kind": "tof", "unit": "us"},
              "source_gpx_sha256": "b" * 64,
              "context": {"source_product_sha256": "a" * 64}}
    raw = json.dumps(packet).encode()
    checksum = sha256(raw).hexdigest()
    descriptor = {"name": "Pattern_observation.json", "sha256": checksum,
                  "source_gpx_sha256": "b" * 64}
    summary = {"pattern_observation": descriptor, "observables": {"source_gpx_sha256": "b" * 64},
               "provenance": {"input": {"source_sha256": "a" * 64}}}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / ".radar-pd-cache.json").write_text(json.dumps({"job_id": "job-1"}))
    (tmp_path / "pattern_observation.json").write_bytes(raw)
    binding = {"schema": "radar-pd-plot-observation-binding/v1", "status": "verified", "job_id": "job-1",
               "pattern_path": "pattern_observation.json", "sha256": checksum,
               "source_gpx_sha256": "b" * 64, "source_product_sha256": "a" * 64}
    (tmp_path / "plot_observation_binding.json").write_text(json.dumps(binding))
    path = tmp_path / "plots" / "stage.plotdata.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"plot_kind": "gsas_fit_with_ticks_v1", "fit_limits": [3, 9],
                               "excluded_ranges": [[5, 6]], "arrays": {
        "x": x, "yobs": y, "ycalc": [v - .25 for v in y], "resid": [.25] * len(x)}}))
    return path, binding


def test_existing_plot_uses_bound_observation_but_its_own_stage_domain(tmp_path):
    path, _ = _bound_plot_fixture(tmp_path)
    original = path.read_bytes()
    payload = read_plot_payload(path)
    assert payload["_observation_validity"]["status"] == "verified"
    figure = figure_for_payload(payload)
    difference = dict(zip(figure.data[2].x, figure.data[2].y))
    # The final observation's fit_valid starts at x=10; this earlier stage is
    # explicitly fitted at x=3..9 and must not inherit that final-stage mask.
    assert difference[3] == .25 and difference[8] == .25
    assert difference[5] is None and difference[6] is None and difference[7] is None
    assert difference[10] is None
    assert figure.data[0].y[2] == -2
    assert path.read_bytes() == original


def test_plot_mask_requires_matching_job_source_checksum_and_exact_observations(tmp_path):
    path, binding = _bound_plot_fixture(tmp_path)
    binding_path = tmp_path / "plot_observation_binding.json"
    for key, wrong in [("job_id", "other-job"), ("source_product_sha256", "c" * 64), ("sha256", "d" * 64)]:
        binding_path.write_text(json.dumps({**binding, key: wrong}))
        payload = read_plot_payload(path)
        assert payload["_observation_validity"]["status"] == "unavailable"
        assert all(y is None for y in figure_for_payload(payload).data[2].y)
    binding_path.write_text(json.dumps(binding))
    original = json.loads(path.read_text())
    for key in ("x", "yobs"):
        changed = deepcopy(original)
        changed["arrays"][key][0] += .001
        path.write_text(json.dumps(changed))
        assert read_plot_payload(path)["_observation_validity"]["status"] == "unavailable"


def test_validity_alone_does_not_invent_a_stage_fit_domain():
    payload = {"plot_kind": "gsas_fit_with_ticks_v1", "arrays": {
        "x": [1, 2, 3], "yobs": [0, -2, 4], "ycalc": [1, -1, 3],
        "resid": [-1, -1, 1], "valid": [True, True, True]}}
    figure = figure_for_payload(payload)
    assert all(v is None for v in figure.data[2].y)
    assert any("Fitted-domain metadata unavailable" in a.text for a in figure.layout.annotations)


def test_legacy_refinement_plot_does_not_guess_limits_from_zero_calculated_values():
    payload = {"plot_kind": "gsas_fit_with_ticks_v1", "arrays": {
        "x": [1, 2, 3], "yobs": [3, 2, 1], "ycalc": [0, 2, 0], "resid": [3, 0, 1]}}
    figure = figure_for_payload(payload)
    assert list(figure.data[1].y) == [0, 2, 0]
    assert figure.layout.xaxis.range is None


def test_experiment_figures_preserve_missing_phase_as_gap() -> None:
    scans = [
        {
            "run_number": 10,
            "rwp": 12.0,
            "elapsed_display": "2 min",
            "hypothesis": "A",
            "phases": [{"label": "A (SG 1)", "weight_percent": 100.0}],
        },
        {
            "run_number": 11,
            "rwp": 10.0,
            "elapsed_display": "2 min",
            "hypothesis": "B",
            "phases": [{"label": "B (SG 2)", "weight_percent": 100.0}],
        },
    ]

    phase_figure = experiment_phase_fraction_figure(scans)
    quality_figure = experiment_fit_quality_figure(scans)

    assert list(phase_figure.data[0].y) in ([100.0, None], [None, 100.0])
    assert list(quality_figure.data[0].x) == ["10", "11"]
    assert quality_figure.layout.xaxis.type == "category"


def test_experiment_figures_use_metadata_axis_and_limit_default_phase_traces() -> None:
    scans = [
        {
            "run_number": 10 + scan_index,
            "rwp": 10.0 + scan_index,
            "metadata": {
                "start_time": f"2026-08-21T0{scan_index}:00:00-04:00",
                "temperature": {"value": 300.0 + 50 * scan_index, "unit": "K"},
            },
            "phases": [
                {"label": f"Phase {phase_index} (SG 1)", "weight_percent": 20.0}
                for phase_index in range(7)
            ],
        }
        for scan_index in range(2)
    ]

    options = experiment_axis_options(scans)
    phase_figure = experiment_phase_fraction_figure(scans, x_key="temperature")
    quality_figure = experiment_fit_quality_figure(scans, x_key="start_time")

    assert {row["value"] for row in options} >= {"run_number", "start_time", "temperature"}
    assert len(phase_figure.data) == 5
    assert list(phase_figure.data[0].x) == [300.0, 350.0]
    assert phase_figure.layout.xaxis.title.text == "Sample temperature (K)"
    assert phase_figure.data[0].mode == "markers"
    assert quality_figure.layout.xaxis.type == "date"
    assert quality_figure.layout.xaxis.title.text == "Acquisition time (Eastern)"
    assert list(quality_figure.data[0].x) == [
        "2026-08-21T00:00:00-04:00",
        "2026-08-21T01:00:00-04:00",
    ]


def test_experiment_temperature_axis_does_not_invent_missing_units() -> None:
    scans = [
        {
            "run_number": 10,
            "metadata": {"temperature": {"value": 25.0}},
            "phases": [{"label": "Phase (SG 1)", "weight_percent": 100.0}],
        }
    ]

    options = experiment_axis_options(scans)
    figure = experiment_phase_fraction_figure(scans, x_key="temperature")

    temperature_option = next(row for row in options if row["value"] == "temperature")
    assert temperature_option["title"] == "Sample temperature (unit not reported)"
    assert figure.layout.xaxis.title.text == "Sample temperature (unit not reported)"


def test_experiment_sample_identity_prefers_sample_id_and_labels_fallbacks() -> None:
    identified = experiment_sample_identity(
        {
            "metadata": {
                "sample_id": "118950",
                "sample_name": "Ga flux",
                "sample_formula": "YFeSiGa",
            }
        }
    )
    named = experiment_sample_identity({"metadata": {"sample_name": "PAC blank"}})
    unknown = experiment_sample_identity({})

    assert identified == {
        "key": "id:118950",
        "label": "Sample 118950 - Ga flux | YFeSiGa",
        "id": "118950",
    }
    assert named["key"] == "name:PAC blank"
    assert unknown["key"] == "unassigned"


def test_phase_trend_breaks_chronological_lines_between_samples() -> None:
    scans = [
        {
            "run_number": 10,
            "metadata": {"sample_id": "A"},
            "phases": [{"label": "Phase (SG 1)", "weight_percent": 40.0}],
        },
        {
            "run_number": 11,
            "metadata": {"sample_id": "B"},
            "phases": [{"label": "Phase (SG 1)", "weight_percent": 60.0}],
        },
    ]

    figure = experiment_phase_fraction_figure(scans)

    assert figure.data[0].mode == "lines+markers"
    assert list(figure.data[0].x) == ["10", None, "11"]
    assert list(figure.data[0].y) == [40.0, None, 60.0]


def test_experiment_heatmap_preserves_missing_phases_and_orders_by_abundance() -> None:
    scans = [
        {
            "run_number": 10,
            "phases": [
                {"label": "Minor (SG 2)", "weight_percent": 8.0},
                {"label": "Major (SG 1)", "weight_percent": 92.0},
            ],
        },
        {
            "run_number": 11,
            "phases": [{"label": "Major (SG 1)", "weight_percent": 100.0}],
        },
    ]

    figure = experiment_phase_heatmap_figure(scans)

    points = {
        (x_value, y_value): weight
        for x_value, y_value, weight in zip(
            figure.data[0].x,
            figure.data[0].y,
            figure.data[0].marker.color,
        )
    }
    assert list(figure.layout.yaxis.categoryarray) == ["Major (SG 1)", "Minor (SG 2)"]
    assert points[("10", "Major (SG 1)")] == 92.0
    assert points[("11", "Major (SG 1)")] == 100.0
    assert points[("10", "Minor (SG 2)")] == 8.0
    assert ("11", "Minor (SG 2)") not in points


def test_experiment_heatmap_keeps_low_temperature_trace_phases_visible() -> None:
    scans = [
        {
            "run_number": 12,
            "metadata": {"temperature": {"value": 300.0, "unit": "K"}},
            "phases": [{"label": "High-temperature phase", "weight_percent": 95.0}],
        },
        {
            "run_number": 10,
            "metadata": {"temperature": {"value": 4.0, "unit": "K"}},
            "phases": [{"label": "Low-temperature trace", "weight_percent": 2.0}],
        },
        {
            "run_number": 11,
            "metadata": {"temperature": {"value": 80.0, "unit": "K"}},
            "phases": [{"label": "Low-temperature trace", "weight_percent": 7.0}],
        },
    ]

    figure = experiment_phase_heatmap_figure(scans, x_key="temperature")
    trace = figure.data[0]
    points = {
        (x_value, y_value): weight
        for x_value, y_value, weight in zip(trace.x, trace.y, trace.marker.color)
    }

    assert trace.type == "scatter"
    assert points[(4.0, "Low-temperature trace")] == 2.0
    assert points[(80.0, "Low-temperature trace")] == 7.0
    assert (300.0, "Low-temperature trace") not in points
    assert trace.marker.colorscale[0] == (0.0, "#d6ebe2")
    assert trace.marker.symbol == "square"
    assert figure.layout.xaxis.type == "linear"


def test_phase_map_shortens_unique_descriptions_without_losing_models_or_hover() -> None:
    names = ["ABO3 - long orthorhombic model description (SG Pnma (62))",
             "ABO3 - alternative ordering A (SG P4/mmm (123))",
             "ABO3 - alternative ordering B (SG P4/mmm (123))"]
    figure = experiment_phase_heatmap_figure([
        {"run_number": 1, "phases": [{"label": name, "weight_percent": weight}
                                     for name, weight in zip(names, [70, 20, 10])]}
    ])
    ticks = dict(zip(figure.layout.yaxis.tickvals, figure.layout.yaxis.ticktext))
    assert ticks[names[0]] == "ABO3 (SG Pnma (62))"
    assert ticks[names[1]] == names[1] and ticks[names[2]] == names[2]
    assert set(figure.data[0].y) == set(names)
    assert all(any(name in text for text in figure.data[0].text) for name in names)


def test_experiment_fit_diagnostics_uses_relative_baseline_and_flags_outlier() -> None:
    scans = [
        {"run_id": "PG3_1", "run_number": 1, "rwp": 10.0},
        {"run_id": "PG3_2", "run_number": 2, "rwp": 10.2},
        {"run_id": "PG3_3", "run_number": 3, "rwp": 9.8},
        {"run_id": "PG3_4", "run_number": 4, "rwp": 10.1},
        {"run_id": "PG3_5", "run_number": 5, "rwp": 25.0},
    ]

    diagnostics = experiment_fit_diagnostics(scans)
    figure = experiment_fit_quality_figure(scans)

    assert diagnostics["PG3_1"]["label"] == "No unusual Rwp increase"
    assert diagnostics["PG3_5"]["label"] == "Rwp outlier"
    assert len(figure.layout.shapes) == 1


def test_interleaved_banks_share_one_chronological_axis_and_no_first_bank_median() -> None:
    scans = [
        {"run_number": run, "run_id": f"PG3_{run}", "bank": bank, "rwp": float(run),
         "phases": [{"label": "Main phase", "weight_percent": 100.0}]}
        for run, bank in [(11, 3), (10, 1), (13, 3), (12, 1)]
    ]
    for figure in (experiment_phase_fraction_figure(scans), experiment_fit_quality_figure(scans)):
        assert figure.layout.xaxis.categoryorder == "array"
        assert list(figure.layout.xaxis.categoryarray) == ["10", "11", "12", "13"]
        assert {trace.name.rsplit("Bank ", 1)[-1] for trace in figure.data} == {"1", "3"}
    quality = experiment_fit_quality_figure(scans)
    assert not quality.layout.shapes
    assert not quality.layout.annotations


def test_composition_screen_does_not_create_a_fit_baseline_across_samples() -> None:
    scans = [
        {"run_id": f"PG3_{run}", "run_number": run, "bank": 2, "rwp": 5.0 + run,
         "metadata": {"sample_id": str(1000 + run), "sample_name": f"Composition {run}"}}
        for run in range(1, 10)
    ]
    diagnostics = experiment_fit_diagnostics(scans)
    assert all(row["label"] == "Collecting baseline" for row in diagnostics.values())
    assert not experiment_fit_quality_figure(scans).layout.shapes


def test_fit_baseline_separates_recalibration_of_the_same_sample_and_bank() -> None:
    scans = [
        {"run_id": f"PG3_{run}", "run_number": run, "bank": 2, "rwp": 10.0 + run,
         "metadata": {"sample_id": "sample-a", "powgen_setup": {"cycle": cycle}}}
        for run, cycle in [(1, "old"), (2, "old"), (3, "old"), (4, "new"), (5, "new"), (6, "new")]
    ]
    assert all(row["label"] == "Collecting baseline" for row in experiment_fit_diagnostics(scans).values())
    assert not experiment_fit_quality_figure(scans).layout.shapes


def test_discovers_and_renders_component_results(tmp_path: Path) -> None:
    (tmp_path / "ranking.csv").write_text("rank,hypothesis\n1,Cu + Cu2S\n", encoding="utf-8")
    payload = {
        "plot_kind": "rapid_component_fit_v1",
        "q": [1.0, 2.0, 3.0],
        "target": [0.0, 1.0, 0.0],
        "total_fit": [0.0, 0.8, 0.0],
        "residual": [0.0, 0.2, 0.0],
        "background": [0.0, 0.0, 0.0],
        "components": [{"label": "Cu (SG 225)", "scaled": [0.0, 0.8, 0.0]}],
    }
    plot_path = tmp_path / "fit.plotdata.json"
    plot_path.write_text(json.dumps(payload), encoding="utf-8")
    assert discover_tables(tmp_path)[0]["name"] == "Ranking"
    assert discover_plot_payloads(tmp_path)[0]["kind"] == "rapid_component_fit_v1"
    figure = figure_for_payload(payload)
    assert [trace.name for trace in figure.data] == ["Measured", "Background", "Cu (SG 225)", "Total hypothesis fit", "Difference"]


def test_discover_tables_hides_named_collection_aliases(tmp_path: Path) -> None:
    archive_table = tmp_path / "complete_archive" / "final_phase_fractions.csv"
    collection_table = tmp_path / "named_collection" / "summary_fractions.csv"
    archive_table.parent.mkdir(parents=True)
    collection_table.parent.mkdir(parents=True)
    content = "phase,weight_percent\nFe,100\n"
    archive_table.write_text(content, encoding="utf-8")
    collection_table.write_text(content, encoding="utf-8")

    tables = discover_tables(tmp_path)

    assert [table["name"] for table in tables] == ["Final phase fractions"]


def test_loads_companion_npz_arrays_for_gsas_plot(tmp_path: Path) -> None:
    plot_path = tmp_path / "accepted.png.plotdata.json"
    arrays_path = tmp_path / "accepted.png.plotdata.npz"
    plot_path.write_text(
        json.dumps(
            {
                "plot_kind": "gsas_fit_with_ticks_v1",
                "arrays_npz": arrays_path.name,
                "phase_order": ["main"],
                "phase_ticks": {"main": [20.0]},
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        arrays_path,
        x=np.asarray([10.0, 20.0, 30.0]),
        yobs=np.asarray([1.0, 2.0, 1.0]),
        ycalc=np.asarray([1.0, 1.8, 1.0]),
        resid=np.asarray([0.0, 0.2, 0.0]),
    )

    payload = read_plot_payload(plot_path)
    figure = figure_for_payload(payload)

    assert payload["arrays"]["yobs"] == [1.0, 2.0, 1.0]
    assert [trace.name for trace in figure.data[:3]] == ["Observed", "Calculated", "Difference"]
    assert list(figure.data[0].y) == [1.0, 2.0, 1.0]


def test_discover_plot_payloads_excludes_duplicate_missing_its_arrays(tmp_path: Path) -> None:
    """Galaxy's per-output collection downloads can duplicate a plot's JSON

    under an unrelated filename, without the paired .npz archive that only
    the results-archive extraction keeps alongside it. Such a duplicate must
    never be offered, even when it sorts before the complete copy.
    """

    good_dir = tmp_path / "rapid_results" / "live_run" / "gsas" / "live_rank512_01_rank64_39"
    bad_dir = tmp_path / "diagnostics"
    good_dir.mkdir(parents=True)
    bad_dir.mkdir(parents=True)

    good_json = good_dir / "curve.png.plotdata.json"
    good_json.write_text(
        json.dumps({"plot_kind": "gsas_fit_with_ticks_v1", "arrays_npz": "curve.png.plotdata.npz"}),
        encoding="utf-8",
    )
    np.savez_compressed(
        good_dir / "curve.png.plotdata.npz",
        x=np.asarray([10.0, 20.0]),
        yobs=np.asarray([1.0, 2.0]),
        ycalc=np.asarray([1.0, 1.8]),
        resid=np.asarray([0.0, 0.2]),
    )

    # "diagnostics" sorts before "rapid_results", so without the fix this
    # incomplete duplicate would be picked as the default plot.
    (bad_dir / "rapid_results__live_run__gsas__live_rank512_01_rank64_39__curve.png.plotdata.json").write_text(
        json.dumps({"plot_kind": "gsas_fit_with_ticks_v1", "arrays_npz": "curve.png.plotdata.npz"}),
        encoding="utf-8",
    )

    options = discover_plot_payloads(tmp_path)

    assert len(options) == 1
    assert Path(options[0]["path"]) == good_json


def test_discover_plot_payloads_keeps_distinct_valid_hypothesis_fits_with_the_same_basename(tmp_path: Path) -> None:
    first = tmp_path / "gsas" / "live_rank512_01_rank64_08" / "curve.png.plotdata.json"
    second = tmp_path / "gsas" / "live_rank512_02_rank64_12" / "curve.png.plotdata.json"
    _write_gsas_payload(first, rwp=8.0)
    _write_gsas_payload(second, rwp=9.0)

    options = discover_plot_payloads(tmp_path)

    assert len(options) == 2
    assert {Path(option["path"]).parent.name for option in options} == {
        "live_rank512_01_rank64_08",
        "live_rank512_02_rank64_12",
    }


def test_plot_index_does_not_load_npz_arrays_until_selection(tmp_path: Path, monkeypatch: object) -> None:
    plot_path = tmp_path / "rank01" / "curve.plotdata.json"
    plot_path.parent.mkdir()
    plot_path.write_text(
        json.dumps({"plot_kind": "gsas_fit_with_ticks_v1", "arrays_npz": "curve.npz", "rwp": 4.2}),
        encoding="utf-8",
    )
    (plot_path.parent / "curve.npz").write_bytes(b"present-but-not-read-during-index")

    def fail_if_loaded(*args: object, **kwargs: object) -> object:
        raise AssertionError("np.load must not run while indexing plot metadata")

    monkeypatch.setattr("radar_pd_nova.results.np.load", fail_if_loaded)  # type: ignore[attr-defined]

    options = discover_plot_payloads(tmp_path)

    assert len(options) == 1
    assert options[0]["rwp"] == 4.2


def test_selected_plot_falls_back_when_ranked_npz_is_corrupt(tmp_path: Path) -> None:
    broken = tmp_path / "rank01" / "curve.plotdata.json"
    broken.parent.mkdir()
    broken.write_text(
        json.dumps({"plot_kind": "gsas_fit_with_ticks_v1", "arrays_npz": "curve.npz", "rwp": 3.0}),
        encoding="utf-8",
    )
    (broken.parent / "curve.npz").write_bytes(b"not-an-npz")
    valid = tmp_path / "rank02" / "curve.plotdata.json"
    _write_gsas_payload(valid, rwp=4.0)
    options = discover_plot_payloads(tmp_path)

    selected = load_plot_with_fallback(options, str(broken))

    assert selected is not None
    path, payload, figure = selected
    assert path == str(valid)
    assert payload["rwp"] == 4.0
    assert [trace.name for trace in figure.data[:3]] == ["Observed", "Calculated", "Difference"]


def test_metadata_only_plot_is_excluded_and_falls_back_to_real_curves(tmp_path: Path) -> None:
    empty = tmp_path / "accepted_model.plotdata.json"
    empty.write_text(
        json.dumps({"plot_kind": "gsas_fit_with_ticks_v1", "rwp": 3.0}),
        encoding="utf-8",
    )
    valid = tmp_path / "final_refinement.plotdata.json"
    _write_gsas_payload(valid, rwp=4.0)

    options = discover_plot_payloads(tmp_path)
    selected = load_plot_with_fallback(options, str(empty))

    assert [Path(option["path"]) for option in options] == [valid]
    assert selected is not None
    assert selected[0] == str(valid)


def test_plot_uses_published_static_image_when_interactive_arrays_are_missing(tmp_path: Path) -> None:
    image = tmp_path / "accepted_model.png"
    image.write_bytes(b"published-refinement-image")
    sidecar = tmp_path / "accepted_model.png.plotdata.json"
    sidecar.write_text(
        json.dumps(
            {
                "plot_kind": "gsas_fit_with_ticks_v1",
                "source_plot": image.name,
                "arrays_npz": "missing.plotdata.npz",
                "rwp": 7.5,
            }
        ),
        encoding="utf-8",
    )

    options = discover_plot_payloads(tmp_path)
    selected = load_plot_with_fallback(options, str(sidecar))

    assert len(options) == 1
    assert selected is not None
    assert selected[0] == str(sidecar)
    assert selected[2].layout.title.text == "Published refinement fit / Rwp 7.50%"
    assert len(selected[2].layout.images) == 1


def test_result_archive_uses_orphan_accepted_fit_image(tmp_path: Path) -> None:
    """Older result archives can contain an accepted PNG without a plot sidecar."""

    source = tmp_path / "source" / "gsas_projects" / "seq_pass1_accepted_model.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c63600000020001e221bc330000000049454e44ae426082"
        )
    )
    archive = tmp_path / "results.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(source, "gsas_projects/seq_pass1_accepted_model.png")
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(extracted)

    plots = discover_plot_payloads(extracted)
    assert [plot["name"] for plot in plots] == ["Pass 1 accepted model"]

    selected = load_plot_with_fallback(plots, plots[0]["path"])
    assert selected is not None
    assert selected[0].endswith("seq_pass1_accepted_model.png")
    assert selected[2].layout.title.text == "Published refinement fit"
    assert len(selected[2].layout.images) == 1


def test_result_archive_uses_galaxy_published_accepted_fit_name(tmp_path: Path) -> None:
    """NDIP renames accepted model plots when publishing the Galaxy collection."""

    image = tmp_path / "ndip" / "plots" / "Accepted_fit_after_pass_1.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c63600000020001e221bc330000000049454e44ae426082"
        )
    )

    plots = discover_plot_payloads(tmp_path)
    assert [plot["name"] for plot in plots] == ["Pass 1 accepted model"]

    selected = load_plot_with_fallback(plots, plots[0]["path"])
    assert selected is not None
    assert selected[0].endswith("Accepted_fit_after_pass_1.png")
    assert len(selected[2].layout.images) == 1


def test_galaxy_flattened_plot_companions_keep_interactive_curves(tmp_path: Path) -> None:
    sidecar = tmp_path / "Technical__Plots__main_phase_fit.png.plotdata.json"
    arrays = tmp_path / "Technical__Plots__main_phase_fit.png.plotdata.npz"
    image = tmp_path / "Technical__Plots__main_phase_fit.png"
    sidecar.write_text(
        json.dumps(
            {
                "plot_kind": "gsas_fit_with_ticks_v1",
                "source_plot": "main_phase_fit.png",
                "arrays_npz": "main_phase_fit.png.plotdata.npz",
                "rwp": 6.25,
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        arrays,
        x=np.array([1.0, 2.0, 3.0]),
        yobs=np.array([10.0, 20.0, 30.0]),
        ycalc=np.array([9.0, 19.0, 29.0]),
        resid=np.array([1.0, 1.0, 1.0]),
    )
    image.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c63600000020001e221bc330000000049454e44ae426082"
        )
    )

    plots = discover_plot_payloads(tmp_path)
    selected = load_plot_with_fallback(plots, str(sidecar))

    assert len(plots) == 1
    assert selected is not None
    assert selected[0] == str(sidecar)
    assert [trace.name for trace in selected[2].data[:3]] == ["Observed", "Calculated", "Difference"]
    assert len(selected[2].layout.images) == 0


def test_result_view_deduplicates_manifest_png_represented_by_interactive_sidecar(tmp_path: Path) -> None:
    image = tmp_path / "ndip" / "plots" / "Rapid_final_fit.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"published-fit")
    sidecar = Path(str(image) + ".plotdata.json")
    arrays = Path(str(image) + ".plotdata_arrays.npz")
    sidecar.write_text(
        json.dumps(
            {
                "plot_kind": "gsas_fit_with_ticks_v1",
                "source_plot": image.name,
                "arrays_npz": arrays.name,
                "rwp": 6.25,
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        arrays,
        x=np.array([1.0, 2.0]),
        yobs=np.array([10.0, 20.0]),
        ycalc=np.array([9.0, 19.0]),
        resid=np.array([1.0, 1.0]),
    )
    result = {
        "$schema": "radar-pd-result/v1",
        "analysis_mode": "rapid",
        "status": "complete",
        "phases": [],
        "hypotheses": [],
        "gpx_projects": [],
        "artifacts": {
            "plots": [
                {
                    "name": image.name,
                    "path": f"plots/{image.name}",
                    "source_path": "rapid_results/live_run/curve.png",
                }
            ]
        },
    }

    view = build_result_view(result, tmp_path)

    assert len(view.plots) == 1
    assert Path(view.plots[0].path) == sidecar
    selected = load_plot_with_fallback(view.plots, view.primary_plot_path)
    assert selected is not None
    assert [trace.name for trace in selected[2].data[:3]] == ["Observed", "Calculated", "Difference"]


def test_result_view_resolves_custom_named_plot_from_normalized_manifest(tmp_path: Path) -> None:
    image = tmp_path / "ndip" / "plots" / "custom_database_output_00017.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c63600000020001e221bc330000000049454e44ae426082"
        )
    )
    result = {
        "$schema": "radar-pd-result/v1",
        "analysis_mode": "full",
        "status": "complete",
        "phases": [],
        "hypotheses": [],
        "gpx_projects": [],
        "artifacts": {
            "plots": [
                {
                    "name": image.name,
                    "path": f"plots/{image.name}",
                    "source_path": "unexpected/location/custom_database_output_00017.png",
                }
            ]
        },
    }

    view = build_result_view(result, tmp_path)

    assert len(view.plots) == 1
    assert Path(view.primary_plot_path) == image
    selected = load_plot_with_fallback(view.plots, view.primary_plot_path)
    assert selected is not None
    assert len(selected[2].layout.images) == 1


def test_empty_plot_has_explicit_unavailable_state() -> None:
    figure = figure_for_payload({"plot_kind": "gsas_fit_with_ticks_v1", "rwp": 3.0})

    assert not figure.data
    assert figure.layout.title.text == "Refinement fit unavailable"
    assert "No interactive refinement-fit curves" in figure.layout.annotations[0].text


def test_symbol_only_space_groups_are_completed_from_experiment_peers() -> None:
    scans = [
        {
            "phases": [
                {
                    "phase": "AlFe2V",
                    "space_group": "F m -3 m",
                    "label": "AlFe2V (SG F m -3 m)",
                }
            ]
        },
        {
            "phases": [
                {
                    "phase": "AlVFe2",
                    "space_group": "Fm-3m (225)",
                    "label": "AlVFe2 (SG Fm-3m (225))",
                }
            ]
        },
    ]

    complete_experiment_space_groups(scans)

    assert scans[0]["phases"][0]["space_group"] == "Fm-3m (225)"
    assert scans[0]["phases"][0]["label"] == "AlFe2V (SG Fm-3m (225))"


def test_result_mode_document_is_authoritative_and_warns_on_mismatch(tmp_path: Path) -> None:
    view = build_result_view(
        {"$schema": "radar-pd-result/v1", "analysis_mode": "rapid", "status": "complete"},
        tmp_path,
        submitted_mode="full",
    )

    assert view.mode == "rapid"
    assert view.warnings[0].startswith("Mode mismatch: NOVA submitted Full")


def test_run112_compact_archive_recognizes_five_refinements_and_real_traces(tmp_path: Path) -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "run112_archive_compact.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    for item in fixture["plots"]:
        path = tmp_path / item["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "plot_kind": "gsas_fit_with_ticks_v1",
            "rwp": item["rwp"],
            "phase_order": item["phase_order"],
            "phase_labels": item["phase_labels"],
            "phase_ticks": item["phase_ticks"],
            "arrays": item["arrays"],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
    duplicate = tmp_path / "ndip" / "diagnostics" / "run112_rank01_duplicate.plotdata.json"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_text(
        json.dumps({"plot_kind": "gsas_fit_with_ticks_v1", "arrays_npz": "missing.npz", "rwp": 15.28}),
        encoding="utf-8",
    )

    view = build_result_view(fixture["summary"], tmp_path)

    refinements = [plot for plot in view.plots if plot.stage == "final_refinement"]
    assert len(refinements) == 5
    assert "live_rank512_01_rank64_36" in view.primary_plot_path
    for descriptor in refinements:
        payload = read_plot_payload(descriptor.path)
        figure = figure_for_payload(payload)
        assert [trace.name for trace in figure.data[:3]] == ["Observed", "Calculated", "Difference"]
        assert all(len(trace.x) == 5 for trace in figure.data[:3])
    assert fixture["source"]["source_points_per_curve"] == 2439


def test_phase_fraction_normalization() -> None:
    rows = phase_fraction_rows(
        {
            "phase_fractions": [
                {"formula": "Fe", "space_group": 225, "weight_percent": 91.2},
                {"formula": "Cr", "space_group": 229, "weight_percent": 0.0},
            ]
        }
    )
    assert rows == [
        {"phase": "Fe", "space_group": 225, "weight_percent": 91.2},
        {"phase": "Cr", "space_group": 229, "weight_percent": 0.0},
    ]


def test_environment_fraction_labels_keep_values_and_stable_phase_identity():
    payload={'phases':[dict(phase_id='sample',compound_name='Er oxide',weight_fraction_pct='43.545',
                           component_role='sample_chemistry'),
                      dict(phase_id='can',compound_name='Al',weight_fraction_pct='56.455',
                           component_role='environment_only_chemistry')]}
    original=deepcopy(payload)
    scans=[]
    for run in (21,22):
        scan=experiment_scan_summary(payload);scan.update(run_number=run,sample_key='same',sample_label='Sample')
        scans.append(scan)
    complete_experiment_space_groups(scans)
    figure=experiment_phase_fraction_figure(scans)
    can=next(t for t in figure.data if 'environment chemistry' in t.name)
    assert list(can.y)==[56.455,56.455]
    assert can.customdata[0][3]=='Declared environment-only chemistry'
    assert 'all modeled components' in can.hovertemplate
    assert payload==original
    assert next(p for p in scans[0]['phases'] if p['phase']=='Al')['label']=='Al'
    heatmap=experiment_phase_heatmap_figure(scans)
    assert any('environment chemistry' in label for label in heatmap.layout.yaxis.ticktext)
    assert max(heatmap.data[0].marker.color)==56.455


def test_fraction_labels_use_final_gpx_symmetry_by_id_without_changing_values():
    import copy
    payload={"phases":[
        {"phase_id":"model-a","compound_name":"BaTiO3","space_group":"Pm-3m (221)","weight_fraction_pct":"35.125000"},
        {"phase_id":"model-b","compound_name":"BaTiO3","space_group":"Pm-3m (221)","weight_fraction_pct":"64.875000"}],
        "lattice_parameters":[
            {"phase_id":"model-a","phase":"BaTiO3 (SG Pm-3m, 221)","space_group":"P 4 m m","source_gpx":"final.gpx"},
            {"phase_id":"model-b","phase":"BaTiO3 (SG Pm-3m, 221)","space_group":"P m -3 m","source_gpx":"final.gpx"}]}
    before=copy.deepcopy(payload)
    rows=phase_fraction_rows(payload)
    assert [r['space_group'] for r in rows]==['P4mm','Pm-3m (221)']
    assert [r['weight_percent'] for r in rows]==['35.125000','64.875000']
    assert payload==before


def test_archived_compact_summary_repairs_only_a_unique_final_model_match():
    model={"phase_id":"a","phase":"BaTiO3 (SG Pm-3m, 221)","space_group":"P 4 m m","source_gpx":"final.gpx"}
    payload={"phases":[{"phase":"BaTiO3","space_group":"Pm-3m (221)","weight_percent":100}],
        "lattice_parameters":[model]}
    assert phase_fraction_rows(payload)[0]['space_group']=='P4mm'
    payload['lattice_parameters'].append({**model,'phase_id':'b','space_group':'P m -3 m'})
    assert phase_fraction_rows(payload)[0]['space_group']=='Pm-3m (221)'
    payload['phases'][0]['phase_id']='unmatched-id'
    payload['lattice_parameters']=[model]
    assert phase_fraction_rows(payload)[0]['space_group']=='Pm-3m (221)'


def test_final_symmetry_number_is_retained_only_for_a_matching_symbol():
    payload={"phases":[{"phase_id":"a","phase":"C","space_group":"P6_3/mmc (194)","weight_percent":100}],
        "lattice_parameters":[{"phase_id":"a","phase":"C","space_group":"P 63/m m c","source_gpx":"final.gpx"}]}
    assert phase_fraction_rows(payload)[0]['space_group']=='P6_3/mmc (194)'
    payload['lattice_parameters'][0]['source_gpx']=''
    payload['lattice_parameters'][0]['space_group']='P 1'
    assert phase_fraction_rows(payload)[0]['space_group']=='P6_3/mmc (194)'


def test_phase_fraction_normalization_compacts_formula_and_space_group_not_names() -> None:
    rows = phase_fraction_rows(
        {
            "phases": [
                {"phase": "Al1 Fe2 V1", "space_group": "F m -3 m (225)", "weight_percent": 72.6},
                {"phase": "alpha iron", "space_group": "I m -3 m", "weight_percent": 27.4},
                {"phase": "Al0.5 Fe V0.5 (SG P m -3 m (221))", "weight_percent": 0.0},
            ]
        }
    )

    assert rows == [
        {"phase": "AlFe2V", "space_group": "Fm-3m (225)", "weight_percent": 72.6},
        {"phase": "alpha iron", "space_group": "Im-3m", "weight_percent": 27.4},
        {"phase": "Al0.5FeV0.5", "space_group": "Pm-3m (221)", "weight_percent": 0.0},
    ]


def test_gsas_phase_tick_labels_compact_formula_and_space_group() -> None:
    figure = figure_for_payload(
        {
            "plot_kind": "gsas_fit_with_ticks_v1",
            "phase_order": ["catalog_phase"],
            "phase_labels": {"catalog_phase": "Al Fe2 V (SG F m -3 m (225))"},
            "phase_ticks": {"catalog_phase": [1.5]},
            "arrays": {
                "x": [1.0, 2.0],
                "yobs": [1.0, 2.0],
                "ycalc": [1.0, 2.0],
                "resid": [0.0, 0.0],
            },
        }
    )

    assert figure.layout.yaxis3.ticktext == ("AlFe2V (SG Fm-3m (225))",)


def test_gsas_phase_tick_labels_parse_pipeline_dash_separator() -> None:
    figure = figure_for_payload(
        {
            "plot_kind": "gsas_fit_with_ticks_v1",
            "phase_order": ["catalog_phase"],
            "phase_labels": {"catalog_phase": "Al Fe2 V \u2014 F m -3 m (225)"},
            "phase_ticks": {"catalog_phase": [1.5]},
            "arrays": {
                "x": [1.0, 2.0],
                "yobs": [1.0, 2.0],
                "ycalc": [1.0, 2.0],
                "resid": [0.0, 0.0],
            },
        }
    )

    assert figure.layout.yaxis3.ticktext == ("AlFe2V (SG Fm-3m (225))",)
    assert figure.data[3].name == "AlFe2V (SG Fm-3m (225))"


def test_renders_real_rapid_phase_components_and_zero_contribution() -> None:
    payload = {
        "plot_kind": "rapid_refined_pattern_match",
        "q": [1.0, 2.0, 3.0],
        "target": [0.0, 1.0, 0.0],
        "total_fit": [0.0, 0.9, 0.0],
        "residual": [0.0, 0.1, 0.0],
        "background": [0.0, 0.0, 0.0],
        "phases": [
            {"label": "Cu (SG 225)", "component": [0.0, 0.9, 0.0], "relative_scale": 1.0},
            {"label": "Cu2S (SG 14)", "component": [0.0, 0.0, 0.0], "contribution": 0.0},
        ],
    }

    figure = figure_for_payload(payload)

    assert [trace.name for trace in figure.data] == [
        "Measured",
        "Background",
        "Cu (SG 225) (100.0%)",
        "Cu2S (SG 14) (0.0%)",
        "Total hypothesis fit",
        "Difference",
    ]
    assert list(figure.data[2].y) == [0.0, 0.9, 0.0]


def test_rapid_final_hypothesis_fractions_are_normalized() -> None:
    rows = phase_fraction_rows(
        {
            "analysis_mode": "rapid",
            "phases": [],
            "hypothesis_stage": "final_refinement",
            "hypotheses": [
                {
                    "gsas_rwp_rank": "1",
                    "status": "ok",
                    "weights_json": '{"Cu (SG 225)": 88.5, "Cu2S (SG 14)": 11.5, "CuO (SG 15)": 0.0}',
                }
            ],
        }
    )

    assert rows == [
        {"phase": "Cu", "space_group": "225", "weight_percent": 88.5},
        {"phase": "Cu2S", "space_group": "14", "weight_percent": 11.5},
        {"phase": "CuO", "space_group": "15", "weight_percent": 0.0},
    ]


def test_total_elapsed_seconds_reads_nested_rapid_timing_and_zero() -> None:
    assert total_elapsed_seconds({"summary": {"live_run": {"timings": {"total_seconds": 245.5}}}}) == 245.5
    assert total_elapsed_seconds({"total_seconds": 0.0}) == 0.0


def test_full_result_falls_back_to_manifest_timing_and_presents_known_main_phase(tmp_path: Path) -> None:
    result = {
        "$schema": "radar-pd-result/v1",
        "created_utc": "2026-08-15T03:42:00Z",
        "analysis_mode": "full",
        "status": "complete",
        "hypothesis_stage": "none",
        "phases": [
            {
                "phase_id": "main",
                "compound_name": "main",
                "space_group": "P -4 21 m (113)",
                "weight_fraction_pct": "90.37",
                "is_main": "1",
            },
            {
                "phase_id": "cod-9017775",
                "compound_name": "Al",
                "space_group": "Im-3m (229)",
                "weight_fraction_pct": "9.63",
                "is_main": "0",
            },
        ],
        "provenance": {"source_manifest": {"start_time": "2026-08-15T03:41:37.8"}},
    }

    view = build_result_view(result, tmp_path)

    assert view.phases[0]["phase"] == "Known main phase"
    assert view.phases[1]["phase"] == "Al"
    assert view.metrics[4] == {"label": "Analysis time", "value": "22.2 s"}
    assert view.metrics[5] == {"label": "Result stage", "value": "Final refinement"}


def test_gsas_generated_main_aliases_keep_one_series_label() -> None:
    for name in ("main", "main_", "Main__"):
        row = phase_fraction_rows({"phases": [{"phase_id": "main", "compound_name": name,
            "space_group": "P -4 21 m (113)", "weight_fraction_pct": 100, "is_main": "1"}]})[0]
        assert row["phase"] == "Known main phase"
        assert row["space_group"] == "P-421m (113)"
        assert row["weight_percent"] == 100
    for name, is_main in (("main_", "0"), ("Main_sample", "1"), ("Tb2Be2GeO7", "1")):
        row = phase_fraction_rows({"phases": [{"compound_name": name, "is_main": is_main}]})[0]
        assert row["phase"] == name


def test_gsas_plot_renders_ranked_strongest_bragg_ticks() -> None:
    payload = {
        "plot_kind": "gsas_fit_with_ticks_v1",
        "arrays": {
            "x": [10.0, 20.0, 30.0],
            "yobs": [1.0, 2.0, 1.0],
            "ycalc": [1.0, 1.8, 1.0],
            "resid": [0.0, 0.2, 0.0],
        },
        "phase_order": ["phase_1"],
        "phase_labels": {"phase_1": "Fe (SG 225)"},
        "phase_ticks": {"phase_1": [12.0, 20.0, 28.0]},
        "phase_major_ticks": {"phase_1": [20.0]},
        "phase_major_tick_details": {
            "phase_1": [{"x": 20.0, "rank": 1, "hkl": "1 1 0", "relative_strength": 0.0}]
        },
        "rwp": 0.0,
    }

    figure = figure_for_payload(payload)

    assert [trace.name for trace in figure.data] == [
        "Observed",
        "Calculated",
        "Difference",
        "Fe (SG 225)",
        "Key peaks: Fe (SG 225)",
    ]
    strongest = figure.data[-1]
    assert list(strongest.x) == [20.0]
    assert strongest.marker.line.width == 3
    assert "Relative strength=0.000" in strongest.text[0]
    assert figure.layout.title.text.endswith("Rwp 0.00%")
    assert figure.data[3].showlegend is False
    long_name = "Fe - a long reference description that must not crowd out the fitted curve (SG Fm-3m (225))"
    payload["phase_labels"] = {"phase_1": long_name}
    readable = figure_for_payload(payload)
    assert readable.layout.yaxis3.ticktext == ("Fe (SG 225)",)
    assert readable.data[3].name == long_name
    assert list(readable.data[3].x) == [12.0, 20.0, 28.0]


def test_gsas_plot_hides_internal_custom_catalog_ids() -> None:
    phase_id = "user_00011_yourcustomfilename_collcode258024_bbac164417"
    payload = {
        "plot_kind": "gsas_fit_with_ticks_v1",
        "arrays": {
            "x": [10.0, 20.0],
            "yobs": [1.0, 2.0],
            "ycalc": [1.0, 1.8],
            "resid": [0.0, 0.2],
        },
        "phase_order": [phase_id],
        "phase_labels": {phase_id: f"{phase_id} (Al0.5V0.5Fe1 - Pm-3m (221))"},
        "phase_ticks": {phase_id: [12.0]},
    }

    figure = figure_for_payload(payload)

    assert figure.layout.yaxis3.ticktext == ("Al0.5V0.5Fe1 (SG 221)",)
    assert figure.data[-1].name == "Al0.5V0.5Fe1 - Pm-3m (221)"
    assert figure.data[-1].showlegend is False
    assert figure.layout.margin.l >= 180


def _write_gsas_payload(path: Path, *, rwp: float, phase: str = "TbSSL") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "plot_kind": "gsas_fit_with_ticks_v1",
                "rwp": rwp,
                "arrays": {
                    "x": [1.0, 2.0, 3.0],
                    "yobs": [2.0, 3.0, 2.0],
                    "ycalc": [2.0, 2.8, 2.0],
                    "resid": [0.0, 0.2, 0.0],
                },
                "phase_order": [phase],
                "phase_ticks": {phase: [2.0]},
            }
        ),
        encoding="utf-8",
    )


def test_builds_curated_rapid_result_and_selects_ranked_refinement(tmp_path: Path) -> None:
    best = tmp_path / "rapid_results" / "live_run" / "gsas" / "live_rank512_01_rank64_39" / "curve.png.plotdata.json"
    other = tmp_path / "rapid_results" / "live_run" / "gsas" / "live_rank512_02_rank64_12" / "curve.png.plotdata.json"
    _write_gsas_payload(best, rwp=15.283)
    _write_gsas_payload(other, rwp=22.0)
    (tmp_path / "results.zip").write_bytes(b"zip")
    checkpoint = tmp_path / "gpx" / "rapid_final.gpx"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"gpx")
    nudge = tmp_path / "nudge_results.csv"
    nudge.write_text("formula,space_group,best_score,distance_from_start,a,b,c,alpha,beta,gamma,seconds\nTbSSL,14,0.9,0.01,4,5,6,90,90,90,2\n", encoding="utf-8")
    pattern = tmp_path / "reranked_512_after_radar_nudge.csv"
    pattern.write_text("rank512,rank64,formulas,space_groups,score512,r2_512,sse512,peak_support_summary\n1,39,TbSSL,14,0.95,0.94,0.06,3+1/5\n", encoding="utf-8")
    result = {
        "$schema": "radar-pd-result/v1",
        "analysis_mode": "rapid",
        "status": "complete",
        "hypothesis_stage": "final_refinement",
        "summary": {"live_run": {"timings": {"total_seconds": 61.2}}},
        "phases": [],
        "hypotheses": [
            {
                "gsas_rwp_rank": "1",
                "rank512": "1",
                "rank64": "39",
                "formulas": "TbSSL|Al",
                "space_groups": "14|225",
                "status": "ok",
                "rwp": "15.283",
                "weights_json": '{"TbSSL (SG 14)": 91.0, "Al (SG 225)": 9.0}',
                "curve_png": "/run/rapid_results/live_run/gsas/live_rank512_01_rank64_39/curve.png",
            },
            {
                "gsas_rwp_rank": "2",
                "rank512": "2",
                "formulas": "TbSSL|Be",
                "status": "ok",
                "rwp": "22.0",
                "curve_png": "/run/rapid_results/live_run/gsas/live_rank512_02_rank64_12/curve.png",
            },
        ],
        "gpx_projects": [
            {"label": "rapid_final", "path": "gpx/rapid_final.gpx", "stage": "hypothesis_refinement", "status": "accepted"}
        ],
        "warnings": [],
        "errors": [],
    }

    view = build_result_view(result, tmp_path)

    assert Path(view.primary_plot_path) == best
    assert view.metrics[2] == {"label": "Best Rwp", "value": "15.283%"}
    assert view.phase_total == "100.00% total"
    assert view.phases[0]["weight_display"] == "91.00%"
    assert view.top_refinements[0]["hypothesis"] == "TbSSL (SG 14) + Al (SG 225)"
    assert view.rapid_stages["pattern_scoring"][0]["key_peak_support"] == "3 supported, 1 weak, 1 missing/review"
    assert view.checkpoints[0].handoff_available is True
    assert len(view.plots) == 2
    assert all("/tmp/" not in plot.name for plot in view.plots)


def test_builds_full_result_and_prioritizes_latest_accepted_fit(tmp_path: Path) -> None:
    accepted = tmp_path / "gsas_projects" / "seq_pass2_accepted_model.png.plotdata.json"
    trial = tmp_path / "gsas_projects" / "seq_pass2_trial_blend.png.plotdata.json"
    main = tmp_path / "main_phase_fit.png.plotdata.json"
    _write_gsas_payload(accepted, rwp=8.5, phase="Fe")
    _write_gsas_payload(trial, rwp=10.2, phase="Fe")
    _write_gsas_payload(main, rwp=12.0, phase="Fe")
    result = {
        "$schema": "radar-pd-result/v1",
        "analysis_mode": "full",
        "status": "complete",
        "hypothesis_stage": "phase_refinement",
        "summary": {"final": {"final_rwp": 8.5}, "elapsed_seconds": 120.0},
        "phases": [{"formula": "Fe", "space_group": 225, "weight_percent": 100.0}],
        "hypotheses": [],
        "gpx_projects": [
            {"label": "accepted model", "path": "accepted.gpx", "stage": "full_pipeline_pass", "status": "accepted"}
        ],
    }

    view = build_result_view(result, tmp_path)

    assert Path(view.primary_plot_path) == accepted
    assert view.mode == "full"
    assert view.metrics[0]["value"] == "Full RADAR-PD"
    assert view.phases[0]["phase"] == "Fe"
    assert {plot.name for plot in view.plots} == {
        "Main-phase refinement fit",
        "Best refinement - Pass 2 accepted model / Rwp 8.50%",
        "Pass 2 trial model / Rwp 10.20%",
    }
    assert view.full_progression == [{"stage": "Full pipeline pass", "status": "Pipeline accepted"}]
    assert view.full_models == [
        {
            "model": "Accepted model",
            "stage": "Full pipeline pass",
            "rwp": "-",
            "decision": "Pipeline accepted",
            "note": "-",
        }
    ]


def test_full_result_hides_duplicate_named_plot_collection_aliases(tmp_path: Path) -> None:
    archive_copy = tmp_path / "complete_archive" / "seq_pass1_accepted_model.png.plotdata.json"
    collection_copy = tmp_path / "named_plot_collection" / "seq_pass1_accepted_model.png.plotdata.json"
    _write_gsas_payload(archive_copy, rwp=9.5, phase="Fe")
    _write_gsas_payload(collection_copy, rwp=9.5, phase="Fe")
    result = {
        "$schema": "radar-pd-result/v1",
        "analysis_mode": "full",
        "status": "complete",
        "summary": {"final": {"final_rwp": 9.5}},
        "phases": [{"formula": "Fe", "space_group": 225, "weight_percent": 100.0}],
    }

    view = build_result_view(result, tmp_path)

    assert len(view.plots) == 1
    assert Path(view.primary_plot_path) == archive_copy
    assert view.plots[0].name == "Best refinement - Pass 1 accepted model / Rwp 9.50%"


def test_checkpoint_prefers_galaxy_collection_alias_over_technical_copy(tmp_path: Path) -> None:
    technical = tmp_path / "Technical" / "GSAS_Projects" / "seq_final_main_polished.gpx"
    published = tmp_path / "gpx" / "02_Main_phase_anchor.gpx"
    technical.parent.mkdir(parents=True)
    published.parent.mkdir(parents=True)
    technical.write_bytes(b"technical")
    published.write_bytes(b"published")
    result = {
        "$schema": "radar-pd-result/v1",
        "analysis_mode": "full",
        "status": "complete",
        "phases": [],
        "hypotheses": [],
        "gpx_projects": [
            {
                "label": "seq_final_main_polished",
                "path": "Technical/GSAS_Projects/seq_final_main_polished.gpx",
                "source_path": "Technical/GSAS_Projects/seq_final_main_polished.gpx",
                "collection_path": "gpx/02_Main_phase_anchor.gpx",
                "collection_name": "02_Main_phase_anchor.gpx",
                "stage": "hypothesis_refinement",
                "status": "accepted",
            }
        ],
    }

    view = build_result_view(result, tmp_path)

    assert Path(view.checkpoints[0].path) == published
    assert view.checkpoints[0].galaxy_element_name == "02_Main_phase_anchor"
    assert view.checkpoints[0].handoff_available is True
    assert view.checkpoints[0].local_available is True


def test_checkpoint_can_launch_from_galaxy_without_local_gpx_copy(tmp_path: Path) -> None:
    result = {
        "$schema": "radar-pd-result/v1",
        "analysis_mode": "full",
        "status": "complete",
        "phases": [],
        "hypotheses": [],
        "gpx_projects": [
            {
                "label": "seq_final_main_polished",
                "path": "Technical/GSAS_Projects/seq_final_main_polished.gpx",
                "collection_path": "gpx/02_Main_phase_anchor.gpx",
                "collection_name": "02_Main_phase_anchor.gpx",
                "stage": "final_refinement",
                "status": "accepted",
            }
        ],
    }

    view = build_result_view(result, tmp_path)

    assert view.checkpoints[0].id == "checkpoint-0"
    assert view.checkpoints[0].path == ""
    assert view.checkpoints[0].handoff_available is True
    assert view.checkpoints[0].local_available is False
    assert view.checkpoints[0].name == "Seq final main polished (GPX)"


def test_pattern_only_result_warns_that_coefficients_are_not_phase_fractions(tmp_path: Path) -> None:
    result = {
        "analysis_mode": "rapid",
        "status": "complete",
        "hypothesis_stage": "pattern_scoring",
        "phases": [],
        "hypotheses": [{"rank512": 1, "formulas": "Cu|Cu2S", "phase_coefs512": "1.0|0.2"}],
    }

    view = build_result_view(result, tmp_path)

    assert "not quantitative phase fractions" in view.warnings[0]


def test_curated_rapid_rows_hide_internal_paths_and_json_fields() -> None:
    rows = curate_rapid_rows(
        "final_refinement",
        [
            {
                "gsas_rwp_rank": 1,
                "formulas": "Al Fe2 V|Cu2S",
                "space_groups": "F m -3 m (225)|P 21/c (14)",
                "rwp": 9.25,
                "weights_json": '{"Al Fe2 V": 80, "Cu2S": 20}',
                "cif_paths": "/internal/a.cif|/internal/b.cif",
                "stdout_tail": "technical",
                "status": "ok",
            }
        ],
    )

    assert list(rows[0]) == ["rank", "hypothesis", "rwp", "phase_fractions", "pattern_rank", "status", "time"]
    assert "/internal" not in str(rows[0])
    assert rows[0]["hypothesis"] == "AlFe2V (SG Fm-3m (225)) + Cu2S (SG P21/c (14))"
    assert rows[0]["phase_fractions"] == "AlFe2V: 80.0%; Cu2S: 20.0%"


def test_fixed_result_contract_fixtures_cover_rapid_full_partial_and_failure(tmp_path: Path) -> None:
    fixture_root = Path(__file__).parent / "fixtures" / "results"
    rapid = build_result_view(json.loads((fixture_root / "rapid_final.json").read_text(encoding="utf-8")), tmp_path)
    full = build_result_view(json.loads((fixture_root / "full_final.json").read_text(encoding="utf-8")), tmp_path)
    partial = build_result_view(
        json.loads((fixture_root / "rapid_pattern_only.json").read_text(encoding="utf-8")), tmp_path
    )
    failed = build_result_view(json.loads((fixture_root / "failed.json").read_text(encoding="utf-8")), tmp_path)

    assert rapid.phase_total == "100.00% total"
    assert full.phase_total == "100.00% total"
    assert [row["decision"] for row in full.full_models] == ["Pipeline accepted", "Rejected"]
    assert "not quantitative phase fractions" in partial.warnings[0]
    assert failed.status == "Failed"
    assert failed.warnings == ["No candidate model converged"]


def test_run112_archive_fixture_recovers_five_refinements_and_scientific_traces(tmp_path: Path) -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "run112_archive_compact.json").read_text(encoding="utf-8")
    )
    for plot in fixture["plots"]:
        plot_path = tmp_path / plot["relative_path"]
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path.write_text(
            json.dumps(
                {
                    "plot_kind": "gsas_fit_with_ticks_v1",
                    "rwp": plot["rwp"],
                    "phase_order": plot["phase_order"],
                    "phase_labels": plot["phase_labels"],
                    "phase_ticks": plot["phase_ticks"],
                    "arrays": plot["arrays"],
                }
            ),
            encoding="utf-8",
        )

    # Reproduce the incomplete duplicate that caused the pre-0.1.36 live failure.
    duplicate = tmp_path / "named_collection_copy" / "run112_best.plotdata.json"
    duplicate.parent.mkdir()
    duplicate.write_text(
        json.dumps(
            {
                "plot_kind": "gsas_fit_with_ticks_v1",
                "arrays_npz": "missing.plotdata.npz",
                "rwp": fixture["plots"][0]["rwp"],
            }
        ),
        encoding="utf-8",
    )

    plots = discover_plot_payloads(tmp_path)
    view = build_result_view(fixture["summary"], tmp_path)

    assert fixture["source"]["source_points_per_curve"] == 2439
    assert len(plots) == 5
    assert "rank512_01_rank64_36" in view.primary_plot_path
    assert view.metrics[2]["value"] == "15.283%"
    for plot in plots:
        loaded = load_plot_with_fallback([plot])
        assert loaded is not None
        names = {trace.name: trace for trace in loaded[2].data}
        assert len(names["Observed"].x) == 5
        assert len(names["Calculated"].x) == 5
        assert len(names["Difference"].x) == 5
        assert any(name not in {"Observed", "Calculated", "Difference"} for name in names)


def test_final_joint_plot_is_preferred_to_latest_discovery_pass():
    from radar_pd_nova.results import _primary_plot_index
    plots = [{'path': 'seq_pass10_accepted_model.png', 'stage': 'full_pipeline_pass'},
             {'path': 'seq_final_joint_accepted_model.png', 'stage': 'final_refinement'}]
    assert _primary_plot_index(plots, {}, 'full') == 1
