"""Active transition tab uses local U-Net predictions on measured scan pairs."""

from __future__ import annotations

import asyncio
import pytest

pytest.importorskip("torch", reason="optional U-Net runtime is not installed")
import json
import os
import numpy as np
from types import SimpleNamespace
from unittest.mock import Mock

from radar_pd_nova.app import RadarPdNovaApp as BaseApp
from radar_pd_nova.unet_transition_ui import UnetTransitionMixin

RadarPdNovaApp = BaseApp

from radar_pd_nova.unet_transition_ui import (
    _acquired_label, _condition_label, _display_scale, _fit_figure_for_scan, _map_display_values,
    _map_grid, _matched_display, _score_label, _series_color_limits,
)
from radar_pd_nova.unet_core.production import CHECKPOINT_SHA256
from radar_pd_nova.powgen_watch import WatchedRun, WatchState
from test_scan_interpretation import scan
from test_transition_input import product
from test_transition_tracking import sequence


def scans():
    result = []
    for index, measured in enumerate(sequence()):
        row = scan(index + 1)
        row["pattern"].update(
            coordinate={"kind": "tof", "unit": "us"},
            x=measured.x.tolist(), y=measured.intensity.tolist(),
            sigma=measured.sigma.tolist(), valid=measured.valid.tolist(),
            fit_valid=measured.valid.tolist(), ycalc=measured.intensity.tolist(),
        )
        result.append(row)
    return result


def test_optional_view_preserves_existing_indicator_policy():
    app = RadarPdNovaApp()
    assert hasattr(app, "_transition_engine")
    assert hasattr(app, "_unet_transition_engine")
    assert app._transition_engine is not app._unet_transition_engine
    assert app.server.state.unet_transition_model_version == CHECKPOINT_SHA256[:16]
    assert "Boundary indicators" in app.layout.html
    assert "U-Net local features (experimental)" in app.layout.html
    assert "U-Net feature review" in app.layout.html


def test_scoped_series_and_selected_pair_overlay_use_model_events():
    app = RadarPdNovaApp()
    rows = scans()[:4]
    app._refresh_unet_transition_tracking(rows)
    report = app._unet_transition_report
    assert len(report["groups"]) == 1
    assert [r["status"] for r in report["records"][:2]] == ["initial", "uncertain"]
    assert report["records"][1]["reason"] == "local_model_proposals_rejected_by_measured_evidence"
    assert report["model"]["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert app.server.state.unet_transition_selected == rows[-1]["run_id"]
    assert len(app.server.state.unet_transition_profile["data"]) == 2
    app.select_unet_transition_scan(rows[1]["run_id"])
    assert not app.server.state.unet_transition_follow
    assert "Run 1 · bank 2 → Run 2 · bank 2" in app.server.state.unet_transition_summary
    app.expand_unet_transition_plot()
    assert app.server.state.unet_transition_dialog_figure["data"]
    app.follow_unet_transition_latest()
    assert app.server.state.unet_transition_selected == rows[-1]["run_id"]


def test_same_named_unet_specimens_have_distinct_recorded_labels():
    app = RadarPdNovaApp()
    rows = scans()[:4]
    sample_ids = ('120479', '120480', '120481', '120482')
    for row, sample_id in zip(rows, sample_ids):
        row['metadata'].update(sample_id=sample_id, sample_name='Indium copper vanadate')
    app._refresh_unet_transition_tracking(rows)
    options = app.server.state.unet_transition_series_options
    assert len(options) == 4
    assert len({option['title'] for option in options}) == 4
    for sample_id in sample_ids:
        choice = next(option for option in options if f'Sample {sample_id} ·' in option['title'])
        group = next(group for group in app._unet_transition_report['groups'] if group['id'] == choice['value'])
        assert group['run_ids'] == [rows[sample_ids.index(sample_id)]['run_id']]


def test_revised_measured_curve_replays_without_using_fitted_curve():
    app = RadarPdNovaApp()
    rows = scans()[:3]
    app._refresh_unet_transition_tracking(rows)
    previous = [(r["attention"], r["events"]) for r in app._unet_transition_report["records"]]
    original_sha = app._unet_transition_report["records"][-1]["pattern_sha256"]
    original = app._unet_transition_engine.evaluate
    app._unet_transition_engine.evaluate = Mock(wraps=original)
    for row in rows:
        row["pattern"]["ycalc"] = [999.] * len(row["pattern"]["x"])
        row["rwp"] = 99.
    app._refresh_unet_transition_tracking(rows)
    assert app._unet_transition_engine.evaluate.call_count == 0
    assert [(r["attention"], r["events"]) for r in app._unet_transition_report["records"]] == previous
    rows[-1]["pattern"]["y"] = (np.asarray(rows[-1]["pattern"]["y"]) * 1.25).tolist()
    app._refresh_unet_transition_tracking(rows)
    assert app._unet_transition_engine.evaluate.call_count == 1
    assert app._unet_transition_report["records"][-1]["pattern_sha256"] != original_sha


def test_duplicate_result_revisions_do_not_receive_boxes():
    app = RadarPdNovaApp()
    rows = scans()[:2]
    app._refresh_unet_transition_tracking([rows[0], rows[1], rows[1]])
    duplicates = [r for r in app._unet_transition_report["records"]
                  if r["run_id"] == rows[1]["run_id"]]
    assert len(duplicates) == 2
    assert all(r["status"] == "unsupported" and not r["events"] for r in duplicates)


def test_era_title_condition_is_clearly_nominal_and_scores_do_not_round_to_one():
    assert _condition_label({"metadata": {"run_title": "Er sample, T=300mK"}}) == "0.3 K in run title"
    assert _condition_label({"metadata": {"run_title": "Er sample, T=2 K"}}) == "2 K in run title"
    assert _score_label(.999862) == "0.999862"


def test_scan_condition_labels_use_sort_value_units_and_reviewed_mapping():
    scan_row = {"metadata": {
        "temperature": {"value": 27., "unit": "C"},
        "magnetic_field": {"value": 2500., "unit": "mT"}},
        "axis_overrides": {"temperature": {"value": 310., "unit": "K",
                                           "reviewed_mapping": True}}}
    assert _condition_label(scan_row) == "310 K reviewed mapping"
    assert _condition_label(scan_row, "magnetic_field") == "Field 2.5 T measured"
    assert _condition_label({"metadata": {"temperature": {"value": 27., "unit": "C"}}}) == \
        "300.15 K measured"


def test_live_source_merge_retains_qualified_condition_for_ordering():
    app = RadarPdNovaApp()
    row = scans()[0]
    run = SimpleNamespace(run_id=row["run_id"], run_number=1, instrument="NOM",
                          bank=2, source_path=None, scientific_summary={},
                          scan_metadata=row["metadata"])
    app._post_analysis_scope = "watch:test"
    app._powgen_controller = SimpleNamespace(
        state=SimpleNamespace(discovered={run.run_id: run}, submitted={},
                              completed={}, failed={}),
        settings=SimpleNamespace(ipts="IPTS-1"), source_directory=None,
        records={})
    sources = app._unet_transition_sources([row], [])
    assert sources[0]["metadata"]["temperature"]["value"] == 300


def test_pending_live_raw_banks_are_available_before_refinement(tmp_path):
    app = RadarPdNovaApp()
    watched = {}
    for number, bank in enumerate((3, 2, 3, 2, 2), 1):
        folder = tmp_path / str(number)
        folder.mkdir()
        source, _, _ = product(folder, banks=(2, 3))
        run = WatchedRun(
            run_number=number, bank=bank, instrument="PG3",
            source_path=source["source_path"], fingerprint=f"fixture-{number}",
            scan_metadata={
                "sample_id": "synthetic-sample",
                "start_time": f"2026-05-20T23:0{number}:35.100909667-04:00",
                "temperature": {"value": 2.0 + number, "unit": "K"},
            },
        )
        watched[run.run_id] = run
    app._post_analysis_scope = f"watch:{tmp_path}"
    app._powgen_controller = SimpleNamespace(
        state=WatchState(history_id="test", discovered=watched),
        settings=SimpleNamespace(ipts="IPTS-1"), source_directory=str(tmp_path),
        records={},
    )
    # Match the application's pending rows, which do not yet have a receipt or
    # source path; the watch adapter must restore the confined raw source.
    pending = [{"run_id": run.run_id, "run_number": run.run_number,
                "metadata": dict(run.scan_metadata), "bank": run.bank,
                "instrument": run.instrument, "ipts": "IPTS-1"}
               for run in watched.values()]
    app._refresh_unet_transition_tracking([], pending)
    report = app._unet_transition_report
    assert sorted(len(group["run_ids"]) for group in report["groups"]) == [2, 3]
    assert {group["scope"]["bank"] for group in report["groups"]} == {"2", "3"}
    assert not app.server.state.unet_transition_unavailable
    assert sum(row["previous"] is not None for row in report["records"]) == 3
    assert all(row["source"]["source"] == "read-only reduced bank"
               for row in report["records"])
    assert app.server.state.unet_transition_map["data"]
    assert "Acquisition time unavailable" not in app.server.state.unet_transition_pair_context


@pytest.mark.parametrize("timestamp", [
    "2026-05-20T23:03:35.100909667-04:00",
    " 2026-05-20T23:03:35.100909667-04:00 ",
])
def test_acquisition_label_accepts_facility_nanosecond_timestamps(timestamp):
    row = {"metadata": {"start_time": timestamp}}
    assert _acquired_label(row) == "2026-05-20 23:03 -0400"
    assert row["metadata"]["start_time"] == timestamp


def test_missing_condition_on_new_scan_resets_order_before_showing_boxes():
    app = RadarPdNovaApp()
    rows = scans()[:3]
    for index, row in enumerate(rows[:2]):
        row["metadata"]["temperature"] = {"value": index + 1, "unit": "K"}
    app._refresh_unet_transition_tracking(rows[:2])
    app.select_unet_transition_order("temperature")
    assert app._unet_transition_report["order_by"] == "temperature"
    app._refresh_unet_transition_tracking(rows)
    assert app.server.state.unet_transition_order == "acquired"
    assert app._unet_transition_report["order_by"] == "acquired"
    assert "temperature" not in {item["value"] for item in app.server.state.unet_transition_order_options}


def test_follow_latest_selects_latest_acquisition_after_reverse_physical_sort():
    app = RadarPdNovaApp()
    rows = scans()[:3]
    for index, row in enumerate(rows):
        row["metadata"]["temperature"] = {"value": index + 1, "unit": "K"}
    app._refresh_unet_transition_tracking(rows)
    app.select_unet_transition_order("temperature")
    app.select_unet_transition_direction("descending")
    app.select_unet_transition_scan(rows[0]["run_id"])
    app.follow_unet_transition_latest()
    assert app.server.state.unet_transition_selected == rows[-1]["run_id"]
    assert app._unet_transition_report["groups"][0]["run_ids"][0] == rows[-1]["run_id"]


def test_pair_view_preserves_raw_intensities():
    before = SimpleNamespace(intensity=np.array([1., 2., 3., 9., 100., 1000.]),
                             valid=np.ones(6, dtype=bool))
    after = SimpleNamespace(intensity=np.array([1., 2., 4., 10., 110., 900.]),
                            valid=np.ones(6, dtype=bool))
    original = before.intensity.copy()
    transform, label = _display_scale(before, after, "linear")
    values = transform(np.linspace(1, 1000, 100))
    assert label == "Measured intensity"
    np.testing.assert_allclose(values, np.linspace(1, 1000, 100))
    assert np.all(np.diff(values) > 0)
    np.testing.assert_array_equal(before.intensity, original)


def test_display_difference_does_not_interpolate_across_unsupported_gap():
    before = SimpleNamespace(x=np.array([0., 1., 2., 10., 11.]),
                             intensity=np.array([1., 2., 3., 11., 12.]),
                             sigma=np.ones(5), valid=np.ones(5, dtype=bool))
    after = SimpleNamespace(x=np.arange(12, dtype=float),
                            intensity=np.arange(12, dtype=float) + 1,
                            sigma=np.ones(12), valid=np.ones(12, dtype=bool))
    x, delta, error = _matched_display(before, after, 0, 11)
    np.testing.assert_array_equal(x, np.arange(12))
    assert np.isnan(delta[3:10]).all()
    assert np.isnan(error[3:10]).all()
    assert np.isfinite(delta[[0, 1, 2, 10, 11]]).all()


def test_scan_navigation_and_baseline_preview_are_real_ui_states():
    app = RadarPdNovaApp()
    rows = scans()[:4]
    app._refresh_unet_transition_tracking(rows)
    assert len(app.server.state.unet_transition_scan_cards) == 4
    app.previous_unet_transition_scan()
    assert app.server.state.unet_transition_selected == rows[-2]["run_id"]
    assert not app.server.state.unet_transition_follow
    app.select_unet_transition_scan(rows[0]["run_id"])
    assert app.server.state.unet_transition_prev_disabled
    assert app.server.state.unet_transition_profile["data"]
    assert app.server.state.unet_transition_attention_label == "Baseline"
    app.follow_unet_transition_latest()
    assert app.server.state.unet_transition_selected == rows[-1]["run_id"]


def test_clicking_a_curve_inside_a_model_band_selects_that_region():
    app = RadarPdNovaApp()
    app._refresh_unet_transition_tracking(scans()[:4])
    events = app._unet_transition_report["records"][-1]["events"]
    assert events
    target = len(events) - 1
    center = (events[target]["x0"] + events[target]["x1"]) / 2
    app.select_unet_transition_plot_point({"points": [{"x": center}]})
    assert app.server.state.unet_transition_selected_event == target


def test_incomplete_pair_keeps_review_status_even_with_candidates():
    app = RadarPdNovaApp()
    app._refresh_unet_transition_tracking(scans()[:4])
    row = app._unet_transition_report["records"][-1]
    assert row["events"]
    row["status"] = "uncertain"
    row["reason"] = "incomplete_measured_source_support"
    app._render_unet_transition_tracking()
    assert app.server.state.unet_transition_attention_tone == "pending"
    assert "Review needed" in app.server.state.unet_transition_attention_label
    assert "Comparison incomplete" in app.server.state.unet_transition_notice


def test_linked_map_shows_native_pair_curves_and_heatmap_without_region_panels():
    app = RadarPdNovaApp()
    app._refresh_unet_transition_tracking(scans()[:4])
    app.show_all_unet_transition_overlays()
    state = app.server.state
    data = state.unet_transition_map["data"]
    assert [trace["name"] for trace in data[:3]] == [
        "Reference scan", "Selected scan", "Measured heat map"
    ]
    assert data[2]["z"]["shape"].split(",")[0] == "4"
    assert any(trace["name"] == "U-Net overlays" for trace in data)
    assert "Showing" in state.unet_transition_overlay_summary
    assert "radar-tracker-feature-panel" not in app.layout.html
    assert "radar-tracker-detail" not in app.layout.html


def test_overlay_filter_and_map_navigation_retain_shared_zoom():
    app = RadarPdNovaApp()
    rows = scans()[:4]
    app._refresh_unet_transition_tracking(rows)
    app.show_all_unet_transition_overlays()
    selected_row = app._unet_transition_report["records"][-1]
    named = 0
    overlays = next(trace for trace in app.server.state.unet_transition_map["data"]
                    if trace["name"] == "U-Net overlays")
    app.select_unet_transition_map_click(
        "U-Net overlays", None, None, f"{selected_row['run_id']}|{named}", overlays["meta"]
    )
    state = app.server.state
    assert state.unet_transition_selected_event == named
    assert state.unet_transition_map["layout"]["xaxis"]["range"]
    zoom = list(state.unet_transition_map["layout"]["xaxis"]["range"])
    raster = next(trace for trace in state.unet_transition_map["data"]
                  if trace["name"] == "Measured heat map")
    app.select_unet_transition_map_click("Measured heat map", None, 1,
                                         rows[1]["run_id"], raster["meta"])
    assert state.unet_transition_selected == rows[1]["run_id"]
    assert state.unet_transition_map["layout"]["xaxis"]["range"] == zoom
    app.clear_unet_transition_overlays()
    assert not any(trace["name"] == "U-Net overlays"
                   for trace in state.unet_transition_map["data"])
    app.reset_unet_transition_map_zoom()
    assert "range" not in state.unet_transition_map["layout"]["xaxis"]


def test_global_heatmap_scale_keeps_raw_values_and_uses_all_scans():
    raw = np.array([[0., 1., 2., 3., 7., 13., 100., 500., 1000., 3000.]])
    original = raw.copy()
    distant = SimpleNamespace(intensity=np.array([1., 9000.]),
                              valid=np.array([True, True]))
    near = SimpleNamespace(intensity=raw[0], valid=np.ones(raw.shape[1], dtype=bool))
    limits = _series_color_limits([near, distant], "linear")
    assert limits == (0., 9000.)
    mapped, low, high, ticks, labels = _map_display_values(raw, "linear", limits)
    np.testing.assert_array_equal(mapped, original)
    assert low == 0. and high == 9000.
    assert low < high and len(ticks) == len(labels)
    np.testing.assert_array_equal(raw, original)
    log, _, _, _, _ = _map_display_values(raw, "log", (1., 9000.))
    assert np.isnan(log[0, 0]) and np.all(np.diff(log[0, 1:]) > 0)


def test_heatmap_grid_covers_shifted_scan_ranges():
    left = SimpleNamespace(x=np.linspace(1., 5., 100))
    right = SimpleNamespace(x=np.linspace(3., 7., 100))
    centers, edges = _map_grid([left, right])
    assert edges[0] <= 1. and edges[-1] > 7.
    assert len(centers) == len(edges) - 1


def test_fit_panels_use_published_calculated_curve_and_bragg_ticks():
    row = scans()[0]
    row["pattern"]["phase_ticks"] = {"Er2Be2GeO7": [10., 20.]}
    row["pattern"]["phase_order"] = ["Er2Be2GeO7"]
    figure, note = _fit_figure_for_scan(row)
    assert any(trace["name"] == "Calculated" for trace in figure["data"])
    assert any(trace["name"] == "Er2Be2GeO7" for trace in figure["data"])
    assert "2 Bragg ticks" in note
    row.pop("pattern")
    missing, note = _fit_figure_for_scan(row)
    assert not any(trace.get("name") == "Calculated" for trace in missing["data"])
    assert "unavailable" in note


def test_compact_fit_receipt_does_not_claim_bragg_ticks_were_not_published():
    row = scans()[0]
    row["pattern"].pop("phase_ticks", None)
    row["pattern"].pop("phase_order", None)
    figure, note = _fit_figure_for_scan(row)
    assert any(trace["name"] == "Calculated" for trace in figure["data"])
    assert "Bragg ticks are not included in this compact view" in note
    assert "no Bragg ticks were published" not in note


def test_compact_fit_preserves_verified_observation_and_fit_masks_for_residuals():
    row = scans()[0]
    packet = row["pattern"]
    packet["ycalc"] = (np.asarray(packet["y"]) - 1.5).tolist()
    packet["valid"][:8] = [False] * 8
    packet["fit_valid"][:8] = [False] * 8
    packet["fit_valid"][-12:] = [False] * 12
    figure, _ = _fit_figure_for_scan(row)
    residual = next(trace for trace in figure["data"] if trace.get("name") == "Difference")
    values = np.asarray([np.nan if value is None else value for value in residual["y"]], dtype=float)
    fitted = np.asarray(packet["fit_valid"], dtype=bool)
    assert np.isfinite(values).sum() == fitted.sum()
    assert np.allclose(values[fitted], 1.5)
    assert np.isnan(values[~fitted]).all()
    assert not any("Observation validity unavailable" in str(annotation.get("text"))
                   for annotation in figure["layout"].get("annotations", []))


def test_fit_panel_reads_accepted_refinement_sidecar_for_its_scan(tmp_path):
    plots = tmp_path / "Results" / "Plots"
    plots.mkdir(parents=True)
    (plots / "seq_final_joint_accepted_model.png.plotdata.json").write_text(json.dumps({
        "plot_kind": "gsas_fit_with_ticks_v1", "rwp": 4.2,
        "arrays": {"x": [1., 2., 3.], "yobs": [5., 9., 5.],
                   "ycalc": [5., 8., 5.], "resid": [0., 1., 0.]},
        "phase_order": ["main"], "phase_ticks": {"main": [2.]},
    }))
    figure, note = _fit_figure_for_scan({"run_id": "fit-scan",
                                         "unet_transition_fit_root": str(tmp_path)})
    assert {trace["name"] for trace in figure["data"]} >= {
        "Observed", "Calculated", "Difference", "main"}
    assert "1 Bragg tick" in note


def test_fit_panel_skips_malformed_sidecar_and_prefers_final_accepted_fit(tmp_path):
    plots = tmp_path / "Results" / "Plots"
    plots.mkdir(parents=True)
    malformed = {"plot_kind": "gsas_fit_with_ticks_v1",
                 "arrays": {"x": [1., 2.], "yobs": [1., 2.], "ycalc": []},
                 "phase_ticks": {"bad": [1.5]}}
    (plots / "seq_pass9_accepted_model.png.plotdata.json").write_text(json.dumps(malformed))
    for name, ticks in (("seq_pass1_accepted_model", [1.5]),
                        ("seq_final_joint_accepted_model", [1.5, 2.5])):
        (plots / f"{name}.png.plotdata.json").write_text(json.dumps({
            "plot_kind": "gsas_fit_with_ticks_v1", "rwp": 5. if "final" in name else 1.,
            "arrays": {"x": [1., 2., 3.], "yobs": [2., 3., 2.],
                       "ycalc": [2., 2.5, 2.], "resid": [0., .5, 0.]},
            "phase_ticks": {"main": ticks}, "phase_order": ["main"],
        }))
    _, note = _fit_figure_for_scan({"unet_transition_fit_root": str(tmp_path)})
    assert "2 Bragg ticks" in note
    disconnected, note = _fit_figure_for_scan(
        {"unet_transition_fit_root": str(tmp_path)},
        SimpleNamespace(x=np.array([100., 200.])))
    assert "unavailable" in note
    assert not any(trace.get("name") == "Calculated" for trace in disconnected["data"])


def test_new_fit_sidecar_refreshes_cards_without_repeating_unet_inference(tmp_path):
    app = RadarPdNovaApp()
    row = scans()[0]
    row["pattern"]["ycalc"] = [None] * len(row["pattern"]["x"])
    row["pattern"]["fit_valid"] = [False] * len(row["pattern"]["x"])
    app._refresh_unet_transition_tracking([row])
    assert not app.server.state.unet_transition_selected_has_fit
    original = app._unet_transition_engine.evaluate
    app._unet_transition_engine.evaluate = Mock(wraps=original)
    plots = tmp_path / "Results" / "Plots"
    plots.mkdir(parents=True)
    path = plots / "seq_final_joint_accepted_model.png.plotdata.json"
    x = row["pattern"]["x"][:3]
    payload = {"plot_kind": "gsas_fit_with_ticks_v1",
               "arrays": {"x": x, "yobs": [2., 3., 2.],
                          "ycalc": [2., 2.5, 2.], "resid": [0., .5, 0.]},
               "phase_ticks": {"main": [x[1]]}, "phase_order": ["main"]}
    path.write_text(json.dumps(payload))
    row["unet_transition_fit_root"] = str(tmp_path)
    app._refresh_unet_transition_tracking([row])
    assert app._unet_transition_engine.evaluate.call_count == 0
    assert app.server.state.unet_transition_selected_has_fit
    assert "1 Bragg tick" in app.server.state.unet_transition_selected_fit_note
    payload["phase_ticks"]["main"].append(x[2])
    path.write_text(json.dumps(payload))
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    app._refresh_unet_transition_tracking([row])
    assert app._unet_transition_engine.evaluate.call_count == 0
    assert "2 Bragg ticks" in app.server.state.unet_transition_selected_fit_note


def test_reverting_order_while_recompute_is_queued_keeps_current_report():
    app = RadarPdNovaApp()
    app._refresh_unet_transition_tracking(scans()[:1])
    original_report = app._unet_transition_report

    async def change_and_revert():
        app.select_unet_transition_order("run_number")
        queued = app._unet_transition_task
        assert queued is not None and app.server.state.unet_transition_loading
        app.select_unet_transition_order("acquired")
        assert not app.server.state.unet_transition_loading
        await queued

    asyncio.run(change_and_revert())
    assert app._unet_transition_report is original_report
    assert app.server.state.unet_transition_order == "acquired"
    assert app.server.state.unet_transition_selected == original_report["records"][0]["run_id"]


def test_long_series_map_uses_bounded_window_around_selected_scan():
    app = RadarPdNovaApp()
    pattern = sequence()[0]
    rows = [{"run_id": f"long-{index}",
             "previous": f"long-{index-1}" if index else None,
             "events": []} for index in range(55)]
    app._unet_transition_scans = {
        row["run_id"]: {"run_id": row["run_id"], "run_number": index,
                        "instrument": "POWGEN", "bank": 1, "metadata": {}}
        for index, row in enumerate(rows)
    }
    app._unet_transition_engine = SimpleNamespace(
        load=lambda _scan: SimpleNamespace(pattern=pattern)
    )
    state = app.server.state
    state.unet_transition_series = "long-series"
    app._unet_transition_report = {"records": [{**row, "group": "long-series"} for row in rows],
                              "groups": []}
    state.unet_transition_map_window = 30
    state.unet_transition_selected = rows[-1]["run_id"]
    app._update_unet_transition_map(rows, 54)
    assert app._unet_transition_map_ids == [row["run_id"] for row in rows[25:]]
    assert state.unet_transition_map["data"][2]["z"]["shape"].split(",")[0] == "30"
    assert "map rows 26–55 of 55" in state.unet_transition_overlay_summary
    assert state.unet_transition_map_scroll_max == 26
    assert state.unet_transition_map_scroll_compact == "26–55 / 55"
    app.select_unet_transition_map_start(1)
    assert app._unet_transition_map_ids == [row["run_id"] for row in rows[:30]]
    assert state.unet_transition_map_scroll_label == "Rows 1–30 of 55"
    assert state.unet_transition_map_scroll_compact == "1–30 / 55"


def test_app_computes_unet_only_when_optional_view_is_visible(monkeypatch):
    app = RadarPdNovaApp()
    rows = scans()[:3]
    from radar_pd_nova.unet_transition_engine import UnetTransitionEngine
    calls = []
    evaluate = UnetTransitionEngine.evaluate
    def observed(self, *args, **kwargs):
        calls.append(1)
        return evaluate(self, *args, **kwargs)
    monkeypatch.setattr(UnetTransitionEngine, "evaluate", observed)
    app._refresh_transition_tracking(rows)
    assert app.server.state.transition_evidence_mode == "indicators"
    assert len(calls) == 0
    state = app.server.state
    state.transition_evidence_mode = "unet"
    app._optional_unet_view_changed()
    assert len(calls) == 0
    state.workspace_view = "experiment"
    state.experiment_result_view = "transitions"
    app._optional_unet_view_changed()
    assert len(calls) == 1
    assert app._unet_transition_report["records"]
    state.transition_evidence_mode = "indicators"
    app._optional_unet_view_changed()
    app._refresh_transition_tracking(rows)
    assert len(calls) == 1
    assert app._transition_report["records"]


def test_reset_discards_optional_observations_and_restores_default():
    app = RadarPdNovaApp()
    app._optional_unet_sources = (scans(), [])
    app.server.state.transition_evidence_mode = "unet"
    generation = app._unet_transition_generation
    app._init_unet_transition_tracking()
    assert app._optional_unet_sources == ([], [])
    assert app._unet_transition_generation > generation
    assert app.server.state.transition_evidence_mode == "indicators"
    assert not app._unet_transition_report["records"]


def test_cancelled_old_worker_cannot_clear_new_session_loading_state():
    import threading
    async def run():
        app = RadarPdNovaApp()
        started, release = threading.Event(), threading.Event()
        class Engine:
            def evaluate(self, *_args, **_kwargs):
                started.set()
                release.wait(2)
                return {"records": [], "groups": []}
        generation = app._unet_transition_generation
        app._unet_transition_request = ("old", [], generation, Engine(), "acquired", False, 1)
        old = asyncio.create_task(app._run_unet_transition_tracking())
        app._unet_transition_task = old
        assert await asyncio.to_thread(started.wait, 2)
        try:
            app._init_unet_transition_tracking()
            replacement = asyncio.get_running_loop().create_future()
            app._unet_transition_task = replacement
            app.server.state.unet_transition_loading = True
            await asyncio.gather(old, return_exceptions=True)
            assert app._unet_transition_task is replacement
            assert app.server.state.unet_transition_loading
            replacement.cancel()
        finally:
            release.set()
    asyncio.run(run())
