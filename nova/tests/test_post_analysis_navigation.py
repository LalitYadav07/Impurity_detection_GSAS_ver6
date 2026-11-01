"""Single-fit navigation must not mix experiment plots with another editor."""
import copy
import gc
import weakref
from types import SimpleNamespace

import pytest

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.models import AnalysisConfig, RunRecord
from radar_pd_nova.post_analysis import AxisMapping, PostAnalysisSettings, ViewSettings
from radar_pd_nova.powgen_controller import PowgenExperimentSettings
from radar_pd_nova.powgen_watch import WatchedRun, WatchState
from test_post_analysis import scan


class PlotRecorder:
    def __init__(self):
        self.figures = []

    def update(self, figure):
        self.figures.append(figure.to_plotly_json())


class Controller:
    def __init__(self, **values):
        self.__dict__.update(values)


def navigation_app(tmp_path):
    app = RadarPdNovaApp()
    app._refinement_maybe_refresh = lambda: None
    app._refresh_transition_tracking = lambda *args, **kwargs: None
    watch = WatchState(history_id="history")
    for number, temperature, length in [(703, 75, 4.0), (704, 300, 4.004)]:
        packet = scan(number, bank=2, temperature=temperature, a=length)
        run = WatchedRun(number, f"/test/PG3_{number}.gsa", str(number), bank=2,
            scan_metadata=packet["metadata"], scientific_summary={
                "rwp": 5.0, "observables": packet["observables"], "phases": []})
        watch.completed[run.run_id] = run
    app._powgen_controller = Controller(
        state=watch, records={}, source_directory="/SNS/PG3/IPTS-99999/shared/autoreduce",
        settings=PowgenExperimentSettings(ipts="IPTS-99999", history_id="history",
                                         configuration_dataset_id="config", wavelength_angstrom=""))
    widget = PlotRecorder()
    app._post_analysis_widgets = [widget]
    app.server.state.workspace_view = "experiment"
    app._sync_powgen_rows()
    app.select_post_analysis_view("cells")
    app.duplicate_post_analysis_view()
    app.update_post_analysis_setting("axis", "temperature")
    app.update_post_analysis_setting("representation", "permille")
    app.update_post_analysis_setting("lattice_series", [
        {"phase_key": "phase-a", "metric": "a"},
        {"phase_key": "phase-a", "metric": "c"}])
    assert not app.server.state.post_analysis_error
    single_settings = PostAnalysisSettings(views=[ViewSettings(
        id="single-volume", kind="thermal_expansion", metric="volume",
        representation="absolute", axis="run_number")])
    record = RunRecord(uid="single-704", name="PG3_704", mode="full", history_id="history",
        output_dir=str(tmp_path), config=AnalysisConfig(sample_elements=["Cu"], post_analysis=single_settings))
    app.service = SimpleNamespace(result_payload=lambda _: {"summary": {"analysis_mode": "full", "status": "complete"}})
    return app, widget, record


def open_single(app, record):
    app._select_record(record)
    app.server.state.workspace_view = "results"
    app._load_results(record)


def test_experiment_controls_and_figures_survive_single_fit_navigation(tmp_path):
    app, widget, record = navigation_app(tmp_path)
    state = app.server.state
    active = state.post_analysis_active
    settings = copy.deepcopy(state.post_analysis)
    figures = copy.deepcopy(widget.figures[-1])
    references = copy.deepcopy(app._powgen_controller.state.post_analysis["references"])
    acquisitions = copy.deepcopy(app._powgen_controller.state.completed)
    open_single(app, record)
    assert app._post_analysis_scope == "single:single-704"
    assert state.post_analysis_active == "single-volume"
    assert state.post_analysis_editor["representation"] == "absolute"
    assert not state.post_analysis_panels and not widget.figures[-1]["data"]
    app.show_powgen_dashboard()
    assert state.post_analysis_active == active
    assert state.post_analysis == settings
    assert state.post_analysis_editor["representation"] == "permille"
    assert len(state.post_analysis_editor["lattice_series"]) == 2
    assert widget.figures[-1]["data"] == figures["data"]
    assert app._powgen_controller.state.post_analysis["references"] == references
    assert app._powgen_controller.state.completed == acquisitions
    assert next(v for v in state.post_analysis_panels if v["id"] == active)["title"] == "Lattice expansion · 2 signals"


def test_each_scope_retains_its_own_user_changes(tmp_path):
    app, _, record = navigation_app(tmp_path)
    active = app.server.state.post_analysis_active
    watch_settings = copy.deepcopy(app._powgen_controller.state.post_analysis)
    open_single(app, record)
    app.update_post_analysis_setting("metric", "c")
    assert app._powgen_controller.state.post_analysis == watch_settings
    app.show_powgen_dashboard()
    app.update_post_analysis_setting("representation", "normalized")
    changed = copy.deepcopy(app.server.state.post_analysis)
    open_single(app, record)
    assert app.server.state.post_analysis_editor["metric"] == "c"
    assert app.server.state.post_analysis_editor["representation"] == "absolute"
    app.show_powgen_dashboard()
    assert app.server.state.post_analysis_active == active
    assert app.server.state.post_analysis == changed
    assert app.server.state.post_analysis_editor["representation"] == "normalized"


@pytest.mark.parametrize("workspace", ["results", "plots", "files", "monitor"])
def test_background_watch_refresh_does_not_replace_single_scope(tmp_path, workspace):
    app, widget, record = navigation_app(tmp_path)
    open_single(app, record)
    app.server.state.workspace_view = workspace
    before = copy.deepcopy(app.server.state.post_analysis)
    app._sync_powgen_rows()
    assert app._post_analysis_scope == "single:single-704"
    assert app.server.state.post_analysis == before
    assert not widget.figures[-1]["data"]


def test_replaced_watch_with_same_path_uses_its_own_settings(tmp_path):
    app, _, record = navigation_app(tmp_path)
    open_single(app, record)
    previous = app._powgen_controller
    replacement = copy.copy(previous)
    replacement.state = copy.deepcopy(previous.state)
    replacement.state.post_analysis = PostAnalysisSettings(views=[ViewSettings(
        id="new-watch", kind="thermal_expansion", metric="volume", representation="absolute")]).model_dump(mode="json")
    app._powgen_controller = replacement
    app.show_powgen_dashboard()
    assert app.server.state.post_analysis_active == "new-watch"
    assert app.server.state.post_analysis_editor["metric"] == "volume"
    assert app.server.state.post_analysis_editor["representation"] == "absolute"


def test_watch_axis_mapping_is_restored_before_dashboard_row_projection(tmp_path):
    app, _, record = navigation_app(tmp_path)
    settings = app._post_settings()
    settings.axis_mappings = {"temperature": {"PG3_703_BANK2": AxisMapping(
        value=123, unit="K", source="reviewed experiment mapping", kind="measured", verified=True)}}
    app._set_post_settings(settings)
    app._refresh_post_analysis()
    record.config.post_analysis.axis_mappings = {"temperature": {"PG3_703_BANK2": AxisMapping(
        value=999, unit="K", source="different single-fit mapping", kind="measured", verified=True)}}
    open_single(app, record)
    app.show_powgen_dashboard()
    row = next(r for r in app.server.state.powgen_scientific_rows if r["run_number"] == 703)
    assert row["temperature_display"] == "123 K"
    assert row["axis_overrides"]["temperature"]["value"] == 123


def test_settings_cache_does_not_retain_a_replaced_watch_controller(tmp_path):
    app, _, record = navigation_app(tmp_path)
    old = app._powgen_controller
    reference = weakref.ref(old)
    open_single(app, record)
    replacement = copy.copy(old)
    replacement.state = copy.deepcopy(old.state)
    app._powgen_controller = replacement
    del old
    gc.collect()
    assert reference() is None
    app.show_powgen_dashboard()
    assert app._post_analysis_scope_owner is replacement
