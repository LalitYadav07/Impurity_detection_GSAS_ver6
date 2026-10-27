import asyncio
import copy
import json
import threading
from types import SimpleNamespace
import numpy as np

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.powgen_watch import WatchedRun, WatchState
from test_scan_interpretation import scan
from test_transition_input import product
from test_transition_tracking import sequence


def scans():
    result = []
    for i, p in enumerate(sequence()):
        row = scan(i + 1)
        row["pattern"].update(
            coordinate={"kind": "tof", "unit": "us"},
            x=p.x.tolist(),
            y=p.intensity.tolist(),
            sigma=p.sigma.tolist(),
            valid=p.valid.tolist(),
            fit_valid=p.valid.tolist(),
            ycalc=p.intensity.tolist(),
        )
        result.append(row)
    return result


def application():
    app = RadarPdNovaApp()
    app._refresh_transition_tracking(scans())
    return app


def test_user_indicator_selection_updates_markers_and_boundaries_without_scientific_jobs():
    app = application()
    jobs = []
    app.service = SimpleNamespace(submit_snapshot=lambda *a,**k:jobs.append(k))
    app.update_transition_policy("selected",["appearance","width"])
    app.update_transition_policy("minimum_signals",2)
    assert not any(r.get("boundary") for r in app._transition_report["records"])
    app.update_transition_policy("minimum_signals",1)
    assert any(r.get("boundary") for r in app._transition_report["records"])
    assert app.server.state.transition_signal_rows
    assert app.server.state.transition_boundary_map["data"]
    lines = [s for s in app.server.state.transition_map["layout"].get("shapes",[]) if s["type"]=="line"]
    assert lines
    assert not jobs


def test_selected_rule_and_assumption_survive_watch_checkpoint_round_trip():
    from radar_pd_nova.transition_tracking import checkpoint
    app = application()
    app.update_transition_policy("selected",["width"])
    app.update_transition_policy("width_percent","5")
    watch = WatchState(history_id="test",instrument="NOM")
    app._powgen_controller = SimpleNamespace(state=watch)
    app._post_analysis_scope = "watch:rule-test"
    app._checkpoint_transition_tracking()
    restored = checkpoint(WatchState.from_json(watch.to_json()).transition_tracking)
    assert restored["policy"]["selected"] == ("width",)
    assert restored["policy"]["width_fraction"] == .05
    assert restored["policy"]["fixed_broadening"] is True


def test_boundary_diagram_click_and_axis_only_change_inspection():
    app = application()
    before = copy.deepcopy(app._transition_report)
    app.update_transition_diagram_axis("run_number")
    app.select_transition_diagram({"points":[{"customdata":["NOM_4_BANK2","new",1]}]})
    assert app.server.state.transition_selected == "NOM_4_BANK2"
    assert not app.server.state.transition_follow
    assert app._transition_report == before


def test_elapsed_acquisition_axis_and_annotations_do_not_change_detection():
    app=application()
    before=copy.deepcopy(app._transition_report)
    state=app.server.state
    assert 'elapsed_time' in [o['value'] for o in state.transition_diagram_axes]
    app.update_transition_diagram_axis('elapsed_time')
    values=[trace['x'][0] for trace in state.transition_boundary_map['data']]
    np.testing.assert_allclose(values,np.arange(len(values)))
    app.update_transition_display('annotation','elapsed_time')
    assert 'min' in state.transition_cursor_label
    assert 'Elapsed acquisition time (min)'==state.transition_boundary_map['layout']['xaxis']['title']['text']
    assert app._transition_report==before


def test_unqualified_acquisition_times_do_not_create_an_elapsed_axis():
    rows=scans()
    for row in rows:row['metadata']['start_time']='2026-01-01T01:00:00'
    app=RadarPdNovaApp();app._refresh_transition_tracking(rows)
    assert 'elapsed_time' not in [o['value'] for o in app.server.state.transition_diagram_axes]


def test_hiding_boundary_lines_preserves_evidence_and_cursor_preview():
    app=application();state=app.server.state
    before=copy.deepcopy(app._transition_report)
    assert any(s['type']=='line' for s in state.transition_map['layout'].get('shapes',[]))
    app.update_transition_display('show_boundaries',False)
    assert not any(s['type']=='line' for s in state.transition_map['layout'].get('shapes',[]))
    app.slide_transition_map(1)
    assert not any(s['type']=='line' for s in state.transition_map['layout'].get('shapes',[]))
    app.update_transition_display('show_boundaries',True)
    assert any(s['type']=='line' for s in state.transition_map['layout'].get('shapes',[]))
    assert app._transition_report==before


def test_invalid_rule_edit_preserves_the_last_valid_rule():
    app = application()
    app.update_transition_policy("selected",["width"])
    app.update_transition_policy("width_percent","nan")
    assert app.server.state.transition_policy["width_fraction"] == .03
    assert app.server.state.transition_error


def test_optional_region_overlay_changes_only_the_selected_scan_display():
    app=application()
    app.select_transition_scan("NOM_4_BANK2")
    before=copy.deepcopy(app._transition_report)
    app.update_transition_display("region_signals",["appearance"])
    app.update_transition_display("overlay","both")
    rectangles=[s for s in app.server.state.transition_map["layout"].get("shapes",[]) if s["type"]=="rect"]
    assert rectangles and all(s["yref"] in {"y domain","y2"} for s in rectangles)
    assert "comparison Run 3 · bank 2 → Run 4 · bank 2" in app.server.state.transition_region_overlay_notice
    assert "both compared scans" in app.server.state.transition_region_overlay_notice
    app.zoom_transition_regions()
    assert app.server.state.transition_map["layout"]["xaxis2"]["range"]
    app.update_transition_display("region_signals",[])
    assert not any(s["type"]=="rect" for s in app.server.state.transition_map["layout"].get("shapes",[]))
    app.update_transition_display("overlay","off")
    assert app._transition_report==before


def test_final_rwp_is_a_selectable_global_indicator_in_the_real_ui():
    from test_refinement_indicator import fitted
    app=RadarPdNovaApp()
    rows=[fitted(1,8),fitted(2,10)]
    app._refresh_transition_tracking(rows)
    app.update_transition_policy("selected",["rwp"])
    state=app.server.state
    assert app._transition_report["records"][1]["boundary"]
    row=next(v for v in state.transition_signal_rows if "Rwp" in v["indicator"])
    assert row["support"]=="Whole pattern" and row["result"]=="Triggered"
    assert "8% → 10%" in state.transition_rwp_notice
    assert all(v["value"]!="rwp" for v in state.transition_region_signal_options)


def test_scan_cursor_preserves_calibrated_zoom_and_does_not_change_inference():
    from radar_pd_nova.diffraction_coordinates import TofCalibration
    app=application()
    c=TofCalibration(22000,-4,1,-7,source_bank=2,profile_sha256="test-profile")
    app._transition_calibration=lambda scan:c
    app.update_transition_display("background",False)
    before=copy.deepcopy(app._transition_report)
    app.relayout_transition_map({"xaxis2.range[0]":2500,"xaxis2.range[1]":4500})
    app.slide_transition_map(2)
    state=app.server.state
    assert state.transition_selected=="NOM_3_BANK2"
    assert state.transition_map["layout"]["xaxis2"]["range"]==[2500,4500]
    native=app._transition_engine.load(app._transition_scans[state.transition_selected]).pattern
    assert state.transition_map["data"][1]["customdata"]==native.intensity.tolist()
    app.update_transition_display("coordinate","d_angstrom")
    np.testing.assert_allclose(state.transition_map["layout"]["xaxis2"]["range"],c.spacing([2500,4500]))
    app.update_transition_display("coordinate","q_inv_angstrom")
    np.testing.assert_allclose(state.transition_map["layout"]["xaxis2"]["range"],sorted(2*np.pi/c.spacing([2500,4500])))
    app.slide_transition_map(4)
    assert state.transition_selected=="NOM_5_BANK2"
    assert app._transition_report==before
    assert "@Relayout" in app.layout.html and "radar-transition-cursor" in app.layout.html


def test_incomplete_calibration_keeps_native_coordinate_and_visible_reason():
    app=application()
    assert [v["value"] for v in app.server.state.transition_coordinate_options]==["tof_us"]
    assert "unavailable" in app.server.state.transition_coordinate_notice.lower()
    app.update_transition_display("coordinate","d_angstrom")
    assert app.server.state.transition_map_coordinate=="tof_us"


def test_reset_zoom_restores_both_axes_and_full_heatmap_after_narrow_zoom_and_cursor_change():
    app=application();state=app.server.state
    original=copy.deepcopy(app._transition_report)
    full=state.transition_map["layout"]["meta"]["full_range"]
    cells=copy.deepcopy(state.transition_map["data"][0])
    app.relayout_transition_map({"xaxis2.range[0]":3950,"xaxis2.range[1]":4050})
    app.slide_transition_map(2)
    revision=state.transition_map["layout"]["uirevision"]
    # The narrow frame itself retains coverage outside that window.
    assert state.transition_map["data"][0]["x"][0]==cells["x"][0]
    assert state.transition_map["data"][0]["x"][-1]==cells["x"][-1]
    app.reset_transition_map_zoom()
    assert all(state.transition_map["layout"][k]["range"]==full for k in ("xaxis","xaxis2"))
    assert state.transition_map["layout"]["uirevision"]!=revision
    assert state.transition_map["data"][0]["x"]==cells["x"]
    assert state.transition_map["data"][0]["z"]==cells["z"]
    assert state.transition_selected=="NOM_3_BANK2"
    assert app._transition_report==original
    assert "Reset zoom" in app.layout.html and "@DoubleClick" in app.layout.html
    assert "@Reset" in app.layout.html


def test_double_click_autorange_request_resets_a_calibrated_q_view():
    from radar_pd_nova.diffraction_coordinates import TofCalibration
    app=application();state=app.server.state
    app._transition_calibration=lambda scan:TofCalibration(22000,-4,1,-7,source_bank=2,profile_sha256="test-profile")
    app.update_transition_display("coordinate","q_inv_angstrom")
    full=state.transition_map["layout"]["meta"]["full_range"]
    middle=sum(full)/2
    app.relayout_transition_map({"xaxis.range":[middle-.1,middle+.1]})
    app.relayout_transition_map({"xaxis.autorange":True,"xaxis2.autorange":True})
    assert state.transition_map["layout"]["xaxis2"]["range"]==full
    assert state.transition_map["data"][0]["x"][0]<middle-.1
    assert state.transition_map["data"][0]["x"][-1]>middle+.1


def test_expanded_fit_zoom_cannot_change_the_diffraction_maps_coordinate_range():
    app=application();state=app.server.state
    before=copy.deepcopy(state.transition_map)
    state.transition_dialog=True;state.transition_dialog_kind="profile"
    app.relayout_transition_map({"xaxis.range":[100,200]})
    app.reset_transition_map_zoom()
    assert state.transition_map==before
    assert app._transition_map_ranges=={}


def test_drag_coalesces_to_the_last_cursor_position_and_discards_a_previous_watch():
    app=application()
    before=copy.deepcopy(app._transition_report)
    calls=[];select=app.select_transition_scan
    def selected(run_id):
        calls.append(run_id);select(run_id)
    app.select_transition_scan=selected
    async def drag():
        app.slide_transition_map(1);app.slide_transition_map(4);app.slide_transition_map(6)
        await app._transition_cursor_task
        assert calls==["NOM_7_BANK2"]
        app.slide_transition_map(0)
        app._transition_generation+=1
        await app._transition_cursor_task
        assert calls==["NOM_7_BANK2"]
    asyncio.run(drag())
    assert app._transition_report==before


def test_local_preview_frames_are_reused_and_full_details_commit_only_on_release(monkeypatch):
    app=application();state=app.server.state
    epoch=state.transition_cursor_epoch
    frames=state.transition_cursor_frames
    assert all(frames)
    selected=state.transition_selected
    report=copy.deepcopy(app._transition_report)
    app.begin_transition_cursor()
    app._render_transition_tracking()
    assert state.transition_selected==selected
    assert app._transition_render_pending
    assert state.transition_cursor_frames is frames
    import radar_pd_nova.transition_tracking_ui as ui
    def unexpected_rebuild(*args,**kwargs):
        raise AssertionError("Cursor-only selection must reuse the map cells and cached frames")
    monkeypatch.setattr(ui,"heatmap_figure",unexpected_rebuild)
    monkeypatch.setattr(ui,"cursor_frames",unexpected_rebuild)
    app.commit_transition_cursor(frames[1]["run_id"],epoch,1)
    assert state.transition_selected==frames[1]["run_id"]
    assert state.transition_cursor_frames is frames
    assert not state.transition_preview["active"]
    assert not app._transition_cursor_dragging
    assert app._transition_report==report
    assert "transition_preview" in state.trame__client_only
    assert "radar-transition-cursor" in app.layout.html and "@Commit" in app.layout.html
    assert ':aria-label="item.aria"' in app.layout.html
    assert ':aria-pressed="item.selected"' in app.layout.html


def test_completed_model_context_reuses_measured_map_and_preview_frames(monkeypatch):
    app = application()
    state = app.server.state
    frames = state.transition_cursor_frames
    epoch = state.transition_cursor_epoch
    import radar_pd_nova.transition_tracking_ui as ui
    def unexpected_rebuild(*args, **kwargs):
        raise AssertionError("A fitted cell update must not retransmit unchanged native ribbons")
    monkeypatch.setattr(ui, "heatmap_figure", unexpected_rebuild)
    monkeypatch.setattr(ui, "cursor_frames", unexpected_rebuild)
    updated = scans()
    for row in updated:
        row["final_rwp"] = 4.2
        row["rwp"] = 4.2
        row["analysis_status"] = "completed"
    app._refresh_transition_tracking(updated)
    assert state.transition_cursor_epoch == epoch
    assert state.transition_cursor_frames is frames
    assert state.transition_rwp == "4.2%"


def test_late_rwp_boundary_decision_invalidates_preview_frames():
    from test_refinement_indicator import fitted
    app = RadarPdNovaApp()
    rows = [fitted(1, 8), fitted(2, 8.1)]
    app._refresh_transition_tracking(rows)
    app.update_transition_policy("selected", ["rwp"])
    epoch = app.server.state.transition_cursor_epoch
    frames = app.server.state.transition_cursor_frames
    changed = copy.deepcopy(rows)
    changed[1]["final_rwp"] = 12
    app._refresh_transition_tracking(changed)
    assert app.server.state.transition_cursor_epoch != epoch
    assert app.server.state.transition_cursor_frames is not frames
    assert any(row.get("boundary") for row in app._transition_report["records"])


def test_old_cursor_epoch_cannot_select_a_row_from_a_different_map():
    app=application();state=app.server.state
    selected=state.transition_selected
    app.begin_transition_cursor()
    app.commit_transition_cursor("NOM_2_BANK2","stale map",1)
    assert state.transition_selected==selected
    assert not app._transition_cursor_dragging


def test_old_cursor_release_cannot_cancel_a_new_drag_or_select_an_older_run():
    app=application();state=app.server.state
    scope=state.transition_cursor_scope;epoch=state.transition_cursor_epoch
    before=state.transition_selected;calls=[];select=app.select_transition_scan
    def selected(run_id):
        calls.append(run_id);select(run_id)
    app.select_transition_scan=selected
    async def rapid_reversal():
        app.begin_transition_cursor(1,scope,"tab-a")
        app.commit_transition_cursor("NOM_1_BANK2",epoch,0,1,scope,"tab-a")
        app.begin_transition_cursor(2,scope,"tab-a")
        app.commit_transition_cursor("NOM_1_BANK2",epoch,0,1,scope,"tab-a")
        assert app._transition_cursor_dragging
        assert state.transition_selected==before
        app.commit_transition_cursor("NOM_5_BANK2",epoch,4,2,scope,"tab-a")
        await app._transition_cursor_task
        assert calls==["NOM_5_BANK2"]
        assert state.transition_cursor_ack=={"client_id":"tab-a","revision":2,"accepted":True,
            "run_id":"NOM_5_BANK2","scope":scope}
    asyncio.run(rapid_reversal())


def test_a_reloaded_browser_can_start_at_revision_one_and_stable_run_ids_survive_new_frames():
    app=application();state=app.server.state
    scope=state.transition_cursor_scope;epoch=state.transition_cursor_epoch
    app.begin_transition_cursor(100,scope,"old-tab")
    app.begin_transition_cursor(1,scope,"reloaded-tab")
    app.commit_transition_cursor("NOM_1_BANK2",epoch,0,100,scope,"old-tab")
    assert app._transition_cursor_dragging and app._transition_cursor_client=="reloaded-tab"
    state.transition_cursor_epoch="new-frame-epoch"
    app.commit_transition_cursor("NOM_3_BANK2",epoch,2,1,scope,"reloaded-tab")
    assert state.transition_selected=="NOM_3_BANK2"
    assert state.transition_cursor_ack["client_id"]=="reloaded-tab"
    assert state.transition_cursor_ack["accepted"]


def test_server_render_never_writes_the_browser_owned_preview_position():
    app=application();state=app.server.state
    preview={"active":True,"label":"latest local gesture","notice":"local pair","evidence":[]}
    state.transition_preview=preview
    app.select_transition_scan("NOM_2_BANK2")
    assert state.transition_preview is preview


def test_a_pending_cursor_commit_cannot_overwrite_a_later_direct_scan_selection():
    app=application();state=app.server.state
    scope=state.transition_cursor_scope;epoch=state.transition_cursor_epoch
    async def direct_selection():
        app.begin_transition_cursor(1,scope,"tab")
        app.commit_transition_cursor("NOM_1_BANK2",epoch,0,1,scope,"tab")
        task=app._transition_cursor_task
        app.select_transition_scan("NOM_4_BANK2")
        await task
        assert state.transition_selected=="NOM_4_BANK2"
        assert not state.transition_cursor_ack["accepted"]
    asyncio.run(direct_selection())


def test_older_multi_select_callbacks_do_not_replace_newer_client_selections():
    app=application();state=app.server.state
    newest=["appearance","splitting","width","rwp"]
    state.transition_boundary_signals=newest
    app.select_transition_boundary_signals(["appearance","rwp"])
    assert state.transition_boundary_signals==newest
    app.select_transition_boundary_signals(newest)
    assert list(state.transition_policy["selected"])==newest
    filters=["splitting","width"]
    state.transition_region_highlights=filters
    app.select_transition_region_signals(["splitting","width","shape"])
    assert state.transition_region_highlights==filters
    app.select_transition_region_signals(filters)
    assert state.transition_region_signals==filters


def test_changing_curve_overlay_keeps_boundary_evidence_tied_to_its_reference():
    app = application()
    app.select_transition_scan("NOM_6_BANK2")
    record = next(r for r in app._transition_report["records"] if r["run_id"] == "NOM_6_BANK2")
    decision = record["boundary_comparison"]
    summary = app.server.state.transition_rule_summary
    assert "Compared with" in summary
    assert str(app._transition_scans[decision["before"]]["run_number"]) in summary
    app.update_transition_display("comparison", "previous")
    assert app.server.state.transition_rule_summary == summary
    assert "Curve overlay compared with" in app.server.state.transition_pattern_evidence[0]
    assert str(app._transition_scans[record["previous"]]["run_number"]) in app.server.state.transition_pattern_evidence[0]
    assert "Boundary rule evidence" in app.layout.html and "Curve overlay evidence" in app.layout.html
    if record['previous'] != decision['before']:
        assert 'Changing the overlay does not change boundary assignments' in app.server.state.transition_comparison_notice


def test_progress_from_an_old_watch_or_superseded_request_never_replaces_current_loading_status():
    app = application()
    state = app.server.state
    state.transition_loading = True
    generation = app._transition_generation
    app._transition_request = ('current', [], generation, None, None)
    app._transition_load_progress('current', generation, 8, 141, 'PG3_50713')
    assert state.transition_loaded_count == 8 and state.transition_load_total == 141
    assert '8 of 141' in state.transition_loading_message and 'PG3_50713' in state.transition_loading_message
    message = state.transition_loading_message
    app._transition_load_progress('old', generation, 140, 141, 'old')
    app._transition_load_progress('current', generation-1, 100, 141, 'other watch')
    assert state.transition_loading_message == message
    state.transition_loading = False
    app._transition_load_progress('current', generation, 141, 141, 'late callback')
    assert state.transition_loading_message == message


def test_series_selector_identifies_the_calibration_and_measured_acquisition_dates():
    app = RadarPdNovaApp()
    rows = scans()
    for row in rows:
        row.setdefault('metadata', {}).update(start_time='2021-08-04T17:00:00-04:00',
            sample_name='Calibration specimen', powgen_setup={'cycle':'2021_2_11A_CAL'})
    rows[-1]['metadata']['start_time'] = '2021-08-05T17:00:00-04:00'
    app._refresh_transition_tracking(rows)
    titles = [r['title'] for r in app.server.state.transition_series_options]
    assert titles and all('2021_2_11A_CAL' in title and '2021-08-04' in title for title in titles)
    assert any('to 2021-08-05' in title for title in titles)


def test_inline_fit_reuses_bragg_rows_only_when_the_published_curve_matches(tmp_path):
    from radar_pd_nova.scan_interpretation import Pattern
    row = scans()[0]
    fit = Pattern.parse(row["pattern"])
    path = tmp_path/"final.plotdata.json"
    payload = {"plot_kind":"gsas_fit_v1","rwp":12,
        "arrays":{"x":fit.x.tolist(),"yobs":fit.y.tolist(),"ycalc":fit.calculated.tolist(),
                  "resid":(fit.y-fit.calculated).tolist()},
        "phase_ticks":{"phase":[float(fit.x[100])]},"phase_order":["phase"]}
    path.write_text(json.dumps(payload))
    app = RadarPdNovaApp()
    app._powgen_controller = SimpleNamespace(records={row["run_id"]:SimpleNamespace(local_output_dir=tmp_path)})
    figure = app._canonical_transition_fit(row,fit)
    assert figure is not None and figure.layout.yaxis3
    app._powgen_controller.records[row["run_id"]] = SimpleNamespace(uid="restored-job",local_output_dir=None)
    app.records["restored-job"] = SimpleNamespace(local_output_dir=tmp_path)
    assert app._canonical_transition_fit(row,fit).layout.yaxis3
    app.records = {"original-submission":SimpleNamespace(galaxy_job_id="restored-job",local_output_dir=tmp_path)}
    assert app._canonical_transition_fit(row,fit).layout.yaxis3
    payload["arrays"]["ycalc"] = (fit.calculated+1).tolist()
    path.write_text(json.dumps(payload))
    assert app._canonical_transition_fit(row,fit) is None


def test_results_are_the_default_and_only_the_new_transition_tab_is_added():
    app = application()
    # Application template, not a separate demonstration page.
    layout = app.layout.html
    assert "Transition tracking" in layout
    assert app.server.state.experiment_result_view == "results"
    assert "Compare scans" not in layout and "Experiment trends" not in layout
    assert "Experiment overview and scan results" not in layout
    assert "Shape difference threshold" not in layout
    assert "radar-scan-evidence-tabs" not in layout
    assert "Refined phase fractions" in layout and "Open full scan result" in layout
    assert "Additional plots and derived quantities" in layout
    assert "radar-transition-workspace" in layout


def test_selected_scan_exposes_integrated_temperature_range_without_opening_provenance():
    rows = scans()
    rows[-1]["metadata"]["conditions"]["channels"][0].update(value=96.83,unit="K",minimum=1.452,
        maximum=280.026,statistic="time_weighted_step_hold")
    app = RadarPdNovaApp()
    app._refresh_transition_tracking(rows)
    assert "time-weighted mean 96.83 K" in app.server.state.transition_condition_summary
    assert "1.452–280.026 K" in app.server.state.transition_condition_summary
    assert "{{ transition_condition_summary }}" in app.layout.html


def test_recovered_series_choices_are_published_to_an_already_connected_client():
    from trame_server.state import State
    app = application()
    values = copy.deepcopy(app.server.state.to_dict())
    values.update(transition_series_options=[], transition_series="")
    updates = []
    state = State(commit_fn=lambda delta: updates.append(copy.deepcopy(delta)), ready=True)
    state.update(values)
    state.flush()
    browser = {"transition_series_options": [], "transition_series": ""}
    updates.clear()
    app.server = SimpleNamespace(state=state)
    app._apply_transition_tracking(app._transition_report, list(app._transition_scans.values()), "recovered")
    state.flush()
    for delta in updates:
        browser.update(delta)
    assert browser["transition_series_options"]
    chosen = next(o for o in browser["transition_series_options"] if o["value"] == browser["transition_series"])
    assert "bank" in chosen["title"] and chosen["title"] != chosen["value"]


def test_completed_refinement_rows_reach_a_client_that_already_received_pending_state():
    from trame_server.state import State
    app = application()
    values = copy.deepcopy(app.server.state.to_dict())
    values["transition_model_rows"] = []
    values["scan_interpretation"] = {}
    updates = []
    state = State(commit_fn=lambda delta: updates.append(copy.deepcopy(delta)), ready=True)
    state.update(values)
    state.flush()
    updates.clear()
    app.server = SimpleNamespace(state=state)
    row = copy.deepcopy(scans()[-1])
    row["pattern"]["source_gpx_sha256"] = row["observables"]["source_gpx_sha256"]
    app._transition_refinement(row)
    state.flush()
    browser = {"transition_model_rows": []}
    for delta in updates:
        browser.update(delta)
    assert browser["transition_model_rows"], "Completed phase/cell context must replace the pending empty table"
    assert browser["transition_model_rows"][0]["phase"]


def test_transition_refinement_uses_actual_symmetry_instead_of_catalog_annotation():
    app = application()
    row = copy.deepcopy(scans()[-1])
    row["pattern"]["source_gpx_sha256"] = row["observables"]["source_gpx_sha256"]
    phase = row["observables"]["phases"][0]
    phase.update(phase="Li0.2C6 (SG P-6m2, 187)", space_group="P 3", source_gpx="final.gpx")
    original = copy.deepcopy(row)
    app._transition_refinement(row)
    labels = [item["phase"] for item in app.server.state.transition_model_rows]
    assert "Li0.2C6 (SG P3)" in labels
    assert all("P-6m2" not in label for label in labels)
    assert row == original


def test_switching_experiment_clears_previous_patterns_before_background_work_finishes():
    app = application()
    assert app.server.state.transition_timeline

    async def run():
        app._post_analysis_scope = "watch:different-experiment"
        app._refresh_transition_tracking([])
        worker = app._transition_task
        try:
            assert app.server.state.transition_loading
            assert not app.server.state.transition_timeline
            assert not app.server.state.transition_model_rows
            assert not app.server.state.transition_map["data"]
            assert not app.server.state.transition_profile["data"]
            assert not app.server.state.transition_can_open
        finally:
            await worker

    asyncio.run(run())


def test_recurring_symbols_selection_and_live_follow_share_one_inspector():
    app = application()
    app.update_transition_policy("selected",["appearance","splitting","width","shape","intensity"])
    state = app.server.state
    assert state.transition_selected == "NOM_8_BANK2"
    assert "Previously seen" in state.transition_summary
    entries = state.transition_timeline
    assert (
        entries[3]["icon"] == entries[7]["icon"]
        and entries[3]["color"] == entries[7]["color"]
    )
    app.select_transition_scan("NOM_4_BANK2")
    assert state.transition_selected == "NOM_4_BANK2" and not state.transition_follow
    assert state.transition_map["layout"]["meta"]["run_ids"][-1] == "NOM_8_BANK2"
    app.select_transition_heatmap({"points": [{"y": "NOM_7_BANK2"}]})
    assert state.transition_selected == "NOM_7_BANK2"
    app.follow_transition_latest()
    assert state.transition_selected == "NOM_8_BANK2" and state.transition_follow


def test_display_changes_and_pin_do_not_recompute_membership_or_launch_jobs():
    app = application()
    app.update_transition_policy("selected",["appearance","splitting","width","shape","intensity"])
    state = app.server.state
    before = copy.deepcopy(app._transition_report)
    app.update_transition_display("comparison", "state")
    assert any("No resolved change" in s for s in state.transition_pattern_evidence)
    app.update_transition_display("comparison", "previous")
    assert state.transition_region_options[1:]
    region = state.transition_region_options[1]["value"]
    app.update_transition_display("region", region)
    assert state.transition_profile["layout"]["xaxis"]["range"]
    app.update_transition_display("scale", "log")
    app.update_transition_display("contrast", 95)
    app.pin_transition_reference()
    assert state.transition_reference_label.endswith("Run 8 · bank 2")
    assert app._transition_report == before
    app.update_transition_display("mode", "fit")
    assert len(state.transition_profile["data"]) == 3
    app.open_transition_plot("map")
    assert state.transition_dialog and state.transition_dialog_figure["data"]


def test_locked_color_limits_and_operator_selection_survive_new_results():
    app = application()
    state = app.server.state
    app.select_transition_scan("NOM_4_BANK2")
    limits = copy.deepcopy(app._transition_saved["limits"])
    rows = scans()
    extra = copy.deepcopy(rows[-1])
    extra["run_id"] = "NOM_9_BANK2"
    extra["run_number"] = 9
    extra["metadata"]["start_time"] = "2026-01-01T00:09:00+00:00"
    app._refresh_transition_tracking([*rows, extra])
    assert state.transition_selected == "NOM_4_BANK2"
    assert app._transition_saved["limits"] == limits


def test_watch_checkpoint_retains_marker_styles_references_and_limits_without_arrays():
    app = application()
    watch = WatchState(history_id="test", instrument="NOM")
    app._powgen_controller = SimpleNamespace(state=watch)
    app._post_analysis_scope = "watch:local-test"
    app._checkpoint_transition_tracking()
    restored = WatchState.from_json(watch.to_json())
    assert restored.transition_tracking["styles"]
    assert restored.transition_tracking["limits"]
    assert '"x": [' not in watch.to_json() and '"y": [' not in watch.to_json()


def test_async_refresh_coalesces_new_scans_and_keeps_one_worker():
    async def run():
        app = RadarPdNovaApp()
        rows = scans()
        app._refresh_transition_tracking(rows[:4])
        worker = app._transition_task
        app._refresh_transition_tracking(rows)
        assert app._transition_task is worker
        await worker
        assert len(app._transition_report["records"]) == 8
        assert app.server.state.transition_selected == "NOM_8_BANK2"
        assert not app.server.state.transition_loading

    asyncio.run(run())


def test_background_display_and_empty_scope_cannot_change_scientific_history_or_leave_a_stale_plot():
    app = application()
    state = app.server.state
    history = copy.deepcopy(app._transition_report)
    measured = copy.deepcopy(
        app._transition_scans[state.transition_selected]["pattern"]["y"]
    )
    app.update_transition_display("background", False)
    assert app._transition_report == history
    assert app._transition_scans[state.transition_selected]["pattern"]["y"] == measured
    assert "local background removed" not in state.transition_map_notice
    app._refresh_transition_tracking([])
    assert not state.transition_timeline and not state.transition_can_pin
    assert not state.transition_map["data"] and not state.transition_profile["data"]
    assert not state.transition_model_rows and not state.transition_peer_labels


def test_short_valid_segments_without_baseline_keep_display_and_inference_separate():
    from radar_pd_nova.transition_tracking import TransitionTrackingEngine
    row = scans()[0]
    valid = np.asarray(row["pattern"]["valid"], dtype=bool)
    valid[90:100] = False
    valid[110:120] = False  # Ten valid samples between gaps cannot estimate a baseline.
    row["pattern"]["valid"] = valid.tolist()
    row["pattern"]["fit_valid"] = valid.tolist()
    app = RadarPdNovaApp()
    app._transition_engine = TransitionTrackingEngine()
    app._transition_scans = {row["run_id"]: row}
    app._transition_calibration = lambda _: None
    item = app._transition_engine.load(row)
    original = item.pattern
    before = original.intensity.copy(), original.valid.copy()
    baseline = app._transition_engine.prepare(item).baseline
    assert np.any(original.valid & ~np.isfinite(baseline))
    app.server.state.transition_background = True
    converted, backgrounds, _ = app._transition_map_display_data([original], [row["run_id"]], original, row["run_id"])
    assert np.any(~np.isfinite(backgrounds[0]))
    np.testing.assert_array_equal(converted[0].intensity, before[0])
    np.testing.assert_array_equal(converted[0].valid, before[1])
    app.server.state.transition_background = False
    raw, backgrounds, _ = app._transition_map_display_data([original], [row["run_id"]], original, row["run_id"])
    assert backgrounds is None
    np.testing.assert_array_equal(raw[0].valid, before[1])


def test_a_discovered_bank_appears_before_refinement_and_survives_an_analysis_failure(
    tmp_path,
):
    source, _, _ = product(tmp_path)
    run = WatchedRun(
        run_number=10,
        source_path=source["source_path"],
        fingerprint="stable-reduction",
        scan_metadata=source["metadata"],
        bank=1,
        instrument="NOM",
    )
    app = RadarPdNovaApp()
    watch = WatchState(
        history_id="local", instrument="NOM", discovered={run.run_id: run}
    )
    app._powgen_controller = SimpleNamespace(
        state=watch,
        source_directory=str(tmp_path),
        settings=SimpleNamespace(ipts="IPTS-10"),
    )
    app._post_analysis_scope = "watch:local-reduced-only"
    app._refresh_transition_tracking([])
    state = app.server.state
    assert state.transition_selected == run.run_id
    assert state.transition_map["data"][0]["y"] == [run.run_id]
    assert state.transition_profile["data"]
    assert (
        not state.transition_model_rows and "pending" in state.transition_model_notice
    )
    before = copy.deepcopy(app._transition_report["records"])
    watch.failed = watch.discovered
    watch.discovered = {}
    app._refresh_transition_tracking([])
    assert app._transition_report["records"] == before
    assert state.transition_map["data"][0]["y"] == [run.run_id]


def test_disappearing_reduction_clears_stale_plots_and_can_recover(monkeypatch):
    app = application()
    state = app.server.state
    selected = state.transition_selected
    load = app._transition_engine.load

    def unavailable(_):
        raise FileNotFoundError("reduced file was replaced")

    monkeypatch.setattr(app._transition_engine, "load", unavailable)
    app.select_transition_scan(selected)
    assert not state.transition_map["data"] and not state.transition_profile["data"]
    assert not state.transition_model_rows and not state.transition_peer_labels
    assert not state.transition_can_pin and "replaced" in state.transition_error
    monkeypatch.setattr(app._transition_engine, "load", load)
    app.select_transition_scan(selected)
    assert state.transition_profile["data"] and not state.transition_error


def test_refinement_context_qualifies_fixed_stale_and_invalid_values():
    app = application()
    row = scans()[-1]
    phase = row["observables"]["phases"][0]
    phase.update(metric_states={"a": "fixed"}, fraction_state="fixed_or_unverified")
    row["observables"]["phases"] = [
        dict(phase, phase_id=f"candidate-{i}", phase=f"Candidate {i}") for i in range(8)
    ]
    app._transition_refinement(row)
    entries = app.server.state.transition_model_rows
    assert len(entries) == 8 and entries[-1]["phase"] == "Candidate 7"
    assert "[fixed]" in entries[0]["cell"] and "unverified" in entries[0]["fraction"]
    row["observables"]["phases"][0].update(
        covariance_fresh=False, weight_percent=float("nan")
    )
    app._transition_refinement(row)
    assert "[stale]" in app.server.state.transition_model_rows[0]["cell"]
    assert app.server.state.transition_model_rows[0]["fraction"] == "—"
    row["rwp"] = 8.2
    row["pattern"]["source_gpx_sha256"] = "different-project"
    app._transition_refinement(row)
    assert not app.server.state.transition_model_rows
    assert app.server.state.transition_rwp == "Not reported"


def test_switching_watches_during_a_checkpoint_saves_the_latest_watch_too():
    async def run():
        app = RadarPdNovaApp()
        app._post_analysis_scope = "watch:old"
        began, release = threading.Event(), threading.Event()
        saved = []

        def old_writer():
            began.set()
            assert release.wait(timeout=5)
            saved.append("old")

        app._powgen_controller = SimpleNamespace(
            state=WatchState(history_id="old"), persist_state=old_writer
        )
        app._checkpoint_transition_tracking()
        worker = app._transition_save_task
        try:
            assert await asyncio.to_thread(began.wait, 5)
            app._post_analysis_scope = "watch:new"
            app._powgen_controller = SimpleNamespace(
                state=WatchState(history_id="new"),
                persist_state=lambda: saved.append("new"),
            )
            app._checkpoint_transition_tracking()
        finally:
            release.set()
        await worker
        assert saved == ["old", "new"]
        assert app._transition_save_task is None

    asyncio.run(run())


def test_finished_result_revisions_are_not_silently_collapsed_by_the_ui_adapter():
    app = RadarPdNovaApp()
    row = scans()[0]
    app._refresh_transition_tracking([row, copy.deepcopy(row)])
    report = app._transition_report["records"]
    assert len(report) == 1 and report[0]["state_id"] is None
    assert report[0]["reason"] == "multiple_result_revisions_require_review"
    assert not app.server.state.transition_series_options
    assert "Multiple results" in app.server.state.transition_unavailable[0]
    # The ordinary pending-to-completed replacement remains a single result.
    app._refresh_transition_tracking([row], pending=[row])
    assert app._transition_report["records"][0]["state_id"]


def test_first_acquisition_does_not_compare_with_itself_as_if_it_were_an_earlier_scan():
    app = RadarPdNovaApp()
    app._refresh_transition_tracking(scans()[:1])
    state = app.server.state
    assert not state.transition_comparison_options
    assert "waiting for a compatible comparison" in state.transition_pattern_evidence[0]
    assert len(state.transition_profile["data"]) == 1


def test_device_mapping_is_scoped_verified_and_cannot_change_pattern_membership():
    rows = scans()
    for row in rows:
        row["metadata"]["conditions"]["channels"].extend(
            [
                {
                    "quantity": "pressure",
                    "channel": "sample-gauge",
                    "role": "measured",
                    "status": "needs_mapping",
                    "value": 1000.0,
                    "unit": "Pa",
                    "statistic": "time_weighted_step_hold",
                    "coverage_fraction": 1,
                },
                {
                    "quantity": "pressure",
                    "channel": "incomplete-gauge",
                    "role": "measured",
                    "status": "needs_mapping",
                    "value": 2000.0,
                    "unit": "Pa",
                    "statistic": "time_weighted_step_hold",
                    "coverage_fraction": 0.4,
                },
            ]
        )
    raw = copy.deepcopy(rows)
    app = RadarPdNovaApp()
    app._refresh_transition_tracking(rows)
    state = app.server.state
    history = copy.deepcopy(app._transition_report)
    assert len(state.transition_binding_options) == 1
    state.transition_binding = state.transition_binding_options[0]["value"]
    app.bind_transition_condition()
    assert not state.scan_interpretation["condition_bindings"]
    state.transition_binding_confirmed = True
    app.bind_transition_condition()
    assert state.scan_interpretation["condition_bindings"] == {
        "NOM:IPTS-1:s1": {"pressure": "sample-gauge"}
    }
    assert any("pressure: time-weighted mean 1000" in s for s in state.transition_conditions)
    assert app._transition_report == history and rows == raw


def test_full_result_navigation_is_available_from_a_completed_hidden_scan_only():
    app = application()
    selected = app.server.state.transition_selected
    completed = {selected: object()}
    app._powgen_controller = SimpleNamespace(
        records={selected: object()}, state=SimpleNamespace(completed=completed)
    )
    opened = []
    app.open_powgen_selected_run = lambda: opened.append(
        app.server.state.powgen_selected_run_id
    )
    app.server.state.powgen_scientific_rows = []  # A separate result display filter.
    app.select_transition_scan(selected)
    assert app.server.state.transition_can_open
    app.open_transition_scan()
    assert opened == [selected]
    completed.clear()
    app.select_transition_scan(selected)
    assert not app.server.state.transition_can_open
    app.open_transition_scan()
    assert opened == [selected]


def test_live_refresh_runs_the_calibrated_tracker_without_the_retired_comparison_engine():
    app = RadarPdNovaApp()

    def forbidden(*_, **__):
        raise AssertionError("Retired heuristic comparison ran during a live refresh")

    app._scan_interpreter.evaluate = forbidden
    app._refresh_post_analysis(scans())
    assert app._transition_report["records"]
    assert (
        not app._interpretation_report["records"] and app._interpretation_task is None
    )


def test_tracking_report_exports_evidence_without_run_secrets_or_large_curve_arrays():
    app = application()
    app._transition_scans[app.server.state.transition_selected][
        "private_service_key"
    ] = "do-not-export"
    downloads = []
    app.download_file = lambda *args: downloads.append(args)
    before = copy.deepcopy(app._transition_report)
    app.download_transition_report()
    name, mime, raw = downloads[0]
    assert name.endswith(".json") and mime == "application/json"
    report = json.loads(raw)
    assert report["records"] == before["records"] and report["settings_sha256"]
    assert b"do-not-export" not in raw and b"private_service_key" not in raw
    assert all("x" not in r and "intensity" not in r for r in report["records"])
    assert app._transition_report == before
