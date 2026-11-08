"""Regression coverage for linked views across source and request revisions."""

from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
from dataclasses import asdict
import json
import os
from types import SimpleNamespace
import threading

import numpy as np
import pytest

pytest.importorskip("torch", reason="optional U-Net runtime is not installed")

import radar_pd_nova.unet_transition_ui as ui
from radar_pd_nova.scan_interpretation import PATTERN_SCHEMA
from radar_pd_nova.transition_input import load_observation
from radar_pd_nova.unet_transition_engine import ComparisonSuperseded


def scan(number, *, coordinate="d", low=1., high=9.):
    x = np.linspace(low, high, 64)
    y = 10. + np.sin(np.linspace(0, 5, len(x))) + number
    return {
        "run_id": f"synthetic-{number}", "run_number": number,
        "instrument": "NOM", "bank": 2, "ipts": "IPTS-synthetic",
        "configuration_id": "synthetic",
        "metadata": {"sample_id": "synthetic",
                     "start_time": f"2026-01-01T00:{number:02d}:00+00:00"},
        "pattern": {
            "$schema": PATTERN_SCHEMA, "source_gpx_sha256": "synthetic",
            "coordinate": {"kind": coordinate,
                           "unit": "angstrom" if coordinate == "d" else "deg"},
            "context": {"profile_sha256": "synthetic", "reduction_family": "synthetic",
                        "normalization": "synthetic"},
            "fit_domain": {"limits": [low, high], "excluded": []},
            "x": x.tolist(), "y": y.tolist(), "ycalc": y.tolist(),
            "sigma": [1.] * len(x), "valid": [True] * len(x),
            "fit_valid": [True] * len(x),
        },
    }


class FakeEngine:
    """Inert inference; actual input parsing, figures, callbacks and scheduler."""

    def __init__(self):
        self.predictor = SimpleNamespace(metadata={
            "checkpoint_sha256": "synthetic", "name": "inert", "version": "inert"})
        self.calls = 0
        self.failed_ids = set()
        self.retained = []

    def load(self, row):
        if row["run_id"] in self.failed_ids:
            raise OSError("Synthetic source unavailable")
        return load_observation(row)

    def retain_sources(self, sources):
        self.retained = [row["run_id"] for row in sources]

    def evaluate(self, sources, **kwargs):
        self.calls += 1
        ordered = sorted(sources, key=lambda row: row["run_number"],
                         reverse=kwargs.get("descending", False))
        records, groups = [], []
        if ordered:
            scope = asdict(self.load(ordered[0]).pattern.scope)
            group_id = "group-" + scope["coordinate"]
            ids = [row["run_id"] for row in ordered]
            groups = [{"id": group_id, "run_ids": ids, "scope": scope}]
            for index, source in enumerate(ordered):
                records.append({
                    "run_id": source["run_id"], "group": group_id,
                    "previous": ids[index - 1] if index else None,
                    "status": "valid" if index else "initial", "reason": "",
                    "attention": "no", "events": source.get("fake_events", []),
                    "color_limits": {"linear": [0., 100.], "log": [1., 100.]},
                })
        return {"records": records, "groups": groups,
                "order_by": kwargs.get("order_by", "acquired")}


class Harness(ui.UnetTransitionMixin):
    def __init__(self):
        self.server = SimpleNamespace(state=SimpleNamespace(flush=lambda: None))
        self._post_analysis_scope = "synthetic:first"
        self._init_unet_transition_tracking()
        self.server.state.workspace_view = "experiment"
        self.server.state.experiment_result_view = "transitions"
        self.server.state.transition_evidence_mode = "unet"

    def refresh(self, rows):
        self._optional_unet_sources = (rows, [])
        self._refresh_unet_transition_tracking(rows)


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(ui.UnetTransitionMixin, "_new_unet_transition_engine",
                        lambda self: FakeEngine())
    return Harness()


def first_y(figure, name):
    raw = next(trace for trace in figure["data"] if trace["name"] == name)["y"]
    values = (np.frombuffer(base64.b64decode(raw["bdata"]), dtype=raw["dtype"])
              if isinstance(raw, dict) and "bdata" in raw else np.asarray(raw))
    return float(values[0])


def trace(figure, name):
    return next(item for item in figure["data"] if item["name"] == name)


def with_events(rows):
    for row in rows[1:]:
        row["fake_events"] = [{"x0": 2., "x1": 3., "attention_score": .9,
                               "feature_types": ["peak_shift"]}]
    return rows


def test_shared_zoom_rescales_weak_ribbon_without_recomputing_inference(app):
    rows = [scan(i) for i in (1, 2, 3)]
    for row in rows:
        row['pattern']['y'][0] = 10000.
    app.refresh(rows)
    state = app.server.state
    calls = app._unet_transition_engine.calls
    raster = trace(state.unet_transition_map, 'Measured heat map')
    before = deepcopy(raster)
    app.relayout_unet_transition_map({'xaxis2.range[0]': 3., 'xaxis2.range[1]': 5.})
    figure = state.unet_transition_map
    assert app._unet_transition_engine.calls == calls
    assert app._unet_transition_x_range == [3., 5.]
    assert figure['layout']['xaxis']['range'] == figure['layout']['xaxis2']['range'] == [3., 5.]
    assert figure['layout']['yaxis']['range'][1] < 30.
    assert trace(figure, 'Measured heat map') == before
    app.previous_unet_transition_scan()
    assert state.unet_transition_map['layout']['xaxis']['range'] == [3., 5.]
    assert state.unet_transition_map['layout']['yaxis']['range'][1] < 30.
    assert app._unet_transition_engine.calls == calls
    app.relayout_unet_transition_map({'xaxis.autorange': True})
    assert app._unet_transition_x_range is None
    assert 'range' not in state.unet_transition_map['layout']['yaxis']


@pytest.mark.parametrize('event', [None, {'xaxis.range': [1., 1.]},
                                  {'xaxis.range': [float('nan'), 2.]},
                                  {'xaxis.range': ['bad', 2.]}])
def test_invalid_zoom_never_changes_measured_data_or_view(app, event):
    app.refresh([scan(i) for i in (1, 2)])
    before = deepcopy(app.server.state.unet_transition_map)
    app.relayout_unet_transition_map(event)
    assert app.server.state.unet_transition_map == before


def test_temperature_sorted_rows_are_labelled_as_order_not_physical_spacing(app):
    rows = [scan(i) for i in (1, 2, 3)]
    app.refresh(rows)
    patterns = [app._unet_transition_engine.load(row).pattern for row in rows]
    report = app._unet_transition_report['records']
    figure, _, _ = ui._evolution_figure(report, patterns, {r['run_id']: r for r in rows},
        rows[-1]['run_id'], patterns[-1], patterns[-2], selected_types=set(),
        order_label='Temperature (K)', order_by='temperature')
    assert figure['layout']['yaxis2']['title']['text'] == 'Temperature order'


def test_revised_inline_measurement_refreshes_only_affected_linked_fit(app):
    rows = [scan(i) for i in (1, 2, 3)]
    app.refresh(rows)
    untouched_fit = app._cached_unet_transition_fit(rows[0])
    old_calls = app._unet_transition_engine.calls
    old_fit_signature = app._unet_transition_fit_signature
    revised = deepcopy(rows)
    revised[-1]["pattern"]["y"] = (np.asarray(revised[-1]["pattern"]["y"]) + 7.).tolist()
    app.refresh(revised)
    state = app.server.state
    expected = revised[-1]["pattern"]["y"][0]
    assert first_y(state.unet_transition_selected_fit, "Observed") == expected
    assert first_y(state.unet_transition_map, "Selected scan") == expected
    assert app._unet_transition_engine.calls == old_calls + 1
    assert app._unet_transition_fit_signature != old_fit_signature
    assert app._cached_unet_transition_fit(revised[0]) is untouched_fit


def test_fit_only_revision_preserves_other_cached_fits_and_skips_inference(app):
    rows = [scan(i) for i in (1, 2, 3)]
    app.refresh(rows)
    untouched_fit = app._cached_unet_transition_fit(rows[0])
    old_calls = app._unet_transition_engine.calls
    revised = deepcopy(rows)
    revised[-1]["pattern"]["ycalc"] = [4.] * len(revised[-1]["pattern"]["x"])
    app.refresh(revised)
    assert first_y(app.server.state.unet_transition_selected_fit, "Calculated") == 4.
    assert app._unet_transition_engine.calls == old_calls
    assert app._cached_unet_transition_fit(revised[0]) is untouched_fit


def test_arriving_fit_summary_does_not_restart_measured_comparisons(app):
    rows = [scan(i) for i in (1, 2, 3)]
    app.refresh(rows)
    calls = app._unet_transition_engine.calls
    revised = deepcopy(rows)
    revised[-1].update(
        analysis_status="completed", final_rwp=12.,
        scientific_summary={"phases": [{"name": "new fitted candidate"}]},
        output_files=["final.gpx", "summary.json"],
        convergence={"status": "not_converged"},
    )
    app.refresh(revised)
    assert app._unet_transition_engine.calls == calls
    # Physical ordering and revised measured arrays must still be invalidated.
    revised[-1]["axis_overrides"] = {
        "temperature": {"value": 200., "unit": "K", "reviewed_mapping": True}}
    app.refresh(revised)
    assert app._unet_transition_engine.calls == calls + 1


def test_live_raw_pattern_keeps_comparisons_when_fit_receipt_arrives(app, tmp_path):
    from test_transition_input import product
    row, _, _ = product(tmp_path)
    app.refresh([row])
    calls = app._unet_transition_engine.calls
    measured = deepcopy(app.server.state.unet_transition_map)
    revised = deepcopy(row)
    revised.update(pattern=scan(1)["pattern"], pattern_path=str(tmp_path / "later-fit.json"),
                   pattern_observation={"sha256": "later-fit"},
                   unet_transition_fit_root=str(tmp_path / "fit"))
    app.refresh([revised])
    assert app._unet_transition_engine.calls == calls
    for name, axis in (("Selected scan", "y"), ("Measured heat map", "z")):
        assert json.dumps(trace(app.server.state.unet_transition_map, name)[axis]) == \
            json.dumps(trace(measured, name)[axis])
    # Removing the facility source switches to the available published receipt.
    os.unlink(row["source_path"])
    app.refresh([revised])
    assert app._unet_transition_engine.calls == calls + 1
    assert first_y(app.server.state.unet_transition_map, "Selected scan") == revised["pattern"]["y"][0]


def test_revised_packet_path_with_stable_source_path_refreshes_comparison_and_fit(app, tmp_path):
    row = scan(1)
    packet = row.pop("pattern")
    path = tmp_path / "pattern.json"
    path.write_text(json.dumps(packet))
    source = tmp_path / "source.dat"
    source.write_text("unchanged source receipt")
    row.update(pattern_path=str(path), source_path=str(source))
    app.refresh([row])
    calls = app._unet_transition_engine.calls
    packet["y"] = (np.asarray(packet["y"]) + 7.).tolist()
    path.write_text(json.dumps(packet))
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    app.refresh([row])
    state = app.server.state
    assert app._unet_transition_engine.calls == calls + 1
    assert first_y(state.unet_transition_selected_fit, "Observed") == packet["y"][0]
    assert first_y(state.unet_transition_map, "Selected scan") == packet["y"][0]


@pytest.mark.parametrize("change_scope,coordinate", [(True, "d"), (True, "two_theta"),
                                                      (False, "two_theta")])
def test_scope_or_coordinate_replacement_resets_shared_zoom(app, change_scope, coordinate):
    app.refresh([scan(1)])
    app._unet_transition_x_range = [2., 3.]
    app._unet_transition_zoom_revision += 1
    app._render_unet_transition_tracking()
    old_revision = app._unet_transition_zoom_revision
    if change_scope:
        app._post_analysis_scope = "synthetic:second"
    app.refresh([scan(2, coordinate=coordinate, low=100., high=150.)])
    state = app.server.state
    assert app._unet_transition_x_range is None
    assert app._unet_transition_zoom_revision > old_revision
    assert "range" not in state.unet_transition_map["layout"]["xaxis"]
    assert state.unet_transition_map["layout"]["uirevision"].endswith(
        f":{app._unet_transition_zoom_revision}")


@pytest.mark.parametrize("clear_path", ["empty", "map_error", "inference_error"])
def test_empty_and_error_paths_clear_all_linked_and_expanded_scientific_figures(app, clear_path):
    rows = [scan(i) for i in (1, 2)]
    app.refresh(rows)
    app.expand_unet_transition_plot()
    if clear_path == "empty":
        app._apply_unet_transition_tracking({"records": [], "groups": []}, [], "empty")
    elif clear_path == "map_error":
        app._unet_transition_engine.failed_ids.add(rows[-1]["run_id"])
        app._render_unet_transition_tracking()
    else:
        def fail(*_args, **_kwargs):
            raise OSError("Synthetic inference input unavailable")
        app._unet_transition_engine.evaluate = fail
        revised = deepcopy(rows)
        revised[-1]["pattern"]["y"][0] += 1.
        app.refresh(revised)
        assert "Synthetic inference input unavailable" in app.server.state.unet_transition_error
        assert not app.server.state.unet_transition_selected_fit["data"]
        assert not app.server.state.unet_transition_profile["data"]
    state = app.server.state
    assert state.unet_transition_dialog
    assert not state.unet_transition_map["data"]
    assert state.unet_transition_dialog_figure is state.unet_transition_map
    assert not app._unet_transition_map_ids
    assert not app._unet_transition_engine.retained


@pytest.mark.parametrize("change_scope", [True, False])
def test_hidden_worker_reentry_replaces_old_scope_and_restores_progress(app, change_scope):
    rows = [scan(i) for i in (1, 2)]
    app.refresh(rows)
    app.expand_unet_transition_plot()
    started, release = threading.Event(), threading.Event()

    class GatedEngine(FakeEngine):
        def evaluate(self, sources, **kwargs):
            started.set()
            if not release.wait(3):
                raise RuntimeError("test gate timed out")
            if kwargs.get("cancelled", lambda: False)():
                raise ComparisonSuperseded()
            return super().evaluate(sources, **kwargs)

    app._unet_transition_engine = GatedEngine()

    async def run():
        revised = deepcopy(rows)
        revised[-1]["pattern"]["y"][0] += 1.
        app.refresh(revised)
        worker = app._unet_transition_task
        old_request = app._unet_transition_request
        assert await asyncio.to_thread(started.wait, 2)
        state = app.server.state
        try:
            # A queued display action can redraw the applied report while the
            # next input revision is pending. It must not poison that fit key.
            app.select_unet_transition_palette("Cividis")
            state.transition_evidence_mode = "indicators"
            app._optional_unet_view_changed()
            assert not state.unet_transition_loading
            if change_scope:
                app._post_analysis_scope = "synthetic:second"
                new_rows = [scan(3, coordinate="two_theta", low=100., high=150.)]
            else:
                new_rows = revised
            app._optional_unet_sources = (new_rows, [])
            state.transition_evidence_mode = "unet"
            app._optional_unet_view_changed()
            assert app._unet_transition_task is worker
            assert app._unet_transition_request is not None
            assert state.unet_transition_loading
            assert state.unet_transition_loading_message
            if change_scope:
                assert not app._unet_transition_report["records"]
                for name in ("map", "dialog_figure", "profile", "selected_fit", "reference_fit"):
                    assert not getattr(state, "unet_transition_" + name)["data"]
            else:
                assert app._unet_transition_request[0] == old_request[0]
            message = state.unet_transition_loading_message
            app._set_unet_transition_progress(old_request[0], 1, 3, "Old scope",
                                              generation=old_request[2])
            assert state.unet_transition_loading_message == message
            request = app._unet_transition_request
            app._set_unet_transition_progress(request[0], 1, 3, "New scope",
                                              generation=request[2])
            assert state.unet_transition_loading_message == "New scope · 1/3 steps"
        finally:
            release.set()
            await worker
        assert not state.unet_transition_loading
        assert app._unet_transition_request is None
        assert state.unet_transition_selected == new_rows[-1]["run_id"]
        assert app._unet_transition_report["records"][0]["run_id"] == new_rows[0]["run_id"]
        assert first_y(state.unet_transition_selected_fit, "Observed") == new_rows[-1]["pattern"]["y"][0]

    asyncio.run(run())


def test_async_inference_failure_clears_expanded_data_and_loading(app):
    rows = [scan(i) for i in (1, 2)]
    app.refresh(rows)
    app.expand_unet_transition_plot()

    def fail(*_args, **_kwargs):
        raise OSError("Synthetic worker failed")

    app._unet_transition_engine.evaluate = fail

    async def run():
        revised = deepcopy(rows)
        revised[-1]["pattern"]["y"][0] += 1.
        app.refresh(revised)
        await app._unet_transition_task
        state = app.server.state
        assert state.unet_transition_error == "Synthetic worker failed"
        assert not state.unet_transition_loading
        assert app._unet_transition_request is None
        assert not state.unet_transition_map["data"]
        assert state.unet_transition_dialog_figure is state.unet_transition_map

    asyncio.run(run())


def test_delayed_heatmap_click_does_not_select_replacement_scope_row(app):
    app.refresh([scan(i) for i in (1, 2, 3)])
    old_map = trace(app.server.state.unet_transition_map, "Measured heat map")
    app._post_analysis_scope = "synthetic:second"
    rows = [scan(i, coordinate="two_theta", low=100., high=150.) for i in (4, 5, 6)]
    app.refresh(rows)
    state = app.server.state
    app.select_unet_transition_map_click("Measured heat map", None, 1,
                                         "synthetic-2", old_map.get("meta"))
    assert state.unet_transition_selected == rows[-1]["run_id"]
    assert state.unet_transition_follow
    current = trace(state.unet_transition_map, "Measured heat map")
    app.select_unet_transition_map_click("Measured heat map", None, 1,
                                         rows[1]["run_id"], current["meta"])
    assert state.unet_transition_selected == rows[1]["run_id"]
    assert not state.unet_transition_follow


@pytest.mark.parametrize("name", ["U-Net overlays", "Selected scan", "Reference scan"])
def test_delayed_overlay_and_curve_clicks_are_rejected_after_redraw(app, name):
    rows = with_events([scan(i) for i in (1, 2, 3)])
    app.refresh(rows)
    old_trace = trace(app.server.state.unet_transition_map, name)
    token = old_trace.get("customdata", [None])[0]
    app.select_unet_transition_palette("Cividis")
    state = app.server.state
    app.select_unet_transition_map_click(name, 2.5, 2, token, old_trace.get("meta"))
    assert state.unet_transition_selected_event == -1
    assert state.unet_transition_follow


def test_heatmap_click_requires_acquisition_identity_even_with_current_map_revision(app):
    app.refresh([scan(i) for i in (1, 2, 3)])
    state = app.server.state
    meta = trace(state.unet_transition_map, "Measured heat map").get("meta")
    app.select_unet_transition_map_click("Measured heat map", None, 1, "synthetic-1", meta)
    assert state.unet_transition_selected == "synthetic-3"
    app.select_unet_transition_map_click("Measured heat map", None, 1, None)
    assert state.unet_transition_selected == "synthetic-3"


def test_scan_slider_selects_inspected_scan_and_centers_linked_map(app):
    rows = [scan(i) for i in range(1, 56)]
    app.refresh(rows)
    app.select_unet_transition_map_scan(12)
    state = app.server.state
    assert state.unet_transition_selected == rows[11]["run_id"]
    assert not state.unet_transition_follow
    assert state.unet_transition_map_scan_position == 12
    assert state.unet_transition_map_scan_max == 55
    assert state.unet_transition_map_start == 1
    assert rows[11]["run_id"] in app._unet_transition_map_ids
    assert first_y(state.unet_transition_map, "Selected scan") == rows[11]["pattern"]["y"][0]
    assert "Run 11" in state.unet_transition_pair_label
    assert "Run 12" in state.unet_transition_pair_label
    assert app._unet_transition_engine.retained[:2] == [rows[11]["run_id"], rows[10]["run_id"]]


def test_scan_slider_moves_within_a_single_map_page(app):
    app.refresh([scan(i) for i in (1, 2, 3)])
    app.select_unet_transition_map_scan(1)
    state = app.server.state
    assert state.unet_transition_selected == "synthetic-1"
    assert state.unet_transition_map_scan_max == 3
    assert state.unet_transition_map_scroll_max == 1
    assert state.unet_transition_map_height == 470
    assert state.unet_transition_map_scroll_padding == 175
    assert state.unet_transition_map["layout"]["height"] == 470
    assert state.unet_transition_map["layout"]["margin"]["r"] == 90
    heatmap = next(trace for trace in state.unet_transition_map["data"] if trace["type"] == "heatmap")
    assert heatmap["colorbar"]["orientation"] == "v"
    assert state.unet_transition_dialog_figure["layout"]["height"] == 690


def test_return_to_latest_restores_map_window_after_manual_paging(app):
    rows = [scan(i) for i in range(1, 56)]
    app.refresh(rows)
    app.select_unet_transition_map_start(1)
    assert app.server.state.unet_transition_map_manual_scroll
    app.follow_unet_transition_latest()
    state = app.server.state
    assert not state.unet_transition_map_manual_scroll
    assert state.unet_transition_selected == rows[-1]["run_id"]
    assert state.unet_transition_map_start == 26
    assert rows[-1]["run_id"] in app._unet_transition_map_ids


def test_empty_map_resets_both_page_and_inspected_scan_slider_limits(app):
    app.refresh([scan(i) for i in range(1, 56)])
    app.select_unet_transition_map_start(10)
    app._apply_unet_transition_tracking({"records": [], "groups": []}, [], "empty")
    state = app.server.state
    assert state.unet_transition_map_start == 1
    assert state.unet_transition_map_scroll_max == 1
    assert state.unet_transition_map_scan_position == 1
    assert state.unet_transition_map_scan_max == 1
    assert state.unet_transition_map_scroll_compact == "0 scans"
    assert not state.unet_transition_map_manual_scroll
