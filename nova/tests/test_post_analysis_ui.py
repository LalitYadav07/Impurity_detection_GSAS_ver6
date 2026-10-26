"""Configuration, UI callbacks and watch persistence must not launch refinements."""

import asyncio
import copy
from types import SimpleNamespace

import pytest

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.models import AnalysisConfig, InputSelection, RunRecord
from radar_pd_nova.post_analysis import PostAnalysisSettings, ViewSettings
from radar_pd_nova.powgen_watch import WatchState
from test_post_analysis import scan


def app_with_data():
    app = RadarPdNovaApp()
    app.server.state.sample_elements = "Fe"
    app.server.state.post_analysis_active = "cells"
    app._refresh_post_analysis([scan(1), scan(2, a=4.1)])
    return app


def test_selecting_no_views_removes_all_panels_and_survives_config_round_trip():
    app = app_with_data()
    app.update_post_analysis_selection([])
    assert app._post_settings().views == []
    assert app.server.state.post_analysis_panels == []
    config = app._configuration()
    assert config.post_analysis.views == []


def test_point_inspection_outside_watch_exposes_provenance_without_opening_a_job():
    app = app_with_data()
    app._post_analysis_scope = "single:saved-result"
    view = next(v for v in app._post_analysis_result["views"] if v["id"] == "cells")
    series = view["series"][0]
    point = series["points"][0]
    app.select_post_analysis_point(
        {"points": [{"customdata": [point["run_id"], "", "", "", "", "", series["phase_key"]]}]},
        "cells",
    )
    state = app.server.state
    assert state.post_analysis_point_open
    assert not state.post_analysis_point_can_open
    details = {row["label"]: row["value"] for row in state.post_analysis_point_details}
    assert details["Scan"] == point["run_id"]
    assert details["Phase"] == series["phase"]
    assert details["Source bank"] == str(series["bank"])
    assert "GPX SHA-256" in details


def test_live_point_opens_details_before_explicit_scan_navigation():
    app = app_with_data()
    app._post_analysis_scope = "watch:/SNS/NOM/IPTS-1/shared/autoreduce"
    navigation = []
    opened = []
    app.select_powgen_scan = navigation.append
    app.open_powgen_selected_run = lambda: opened.append(True)
    view = next(v for v in app._post_analysis_result["views"] if v["id"] == "cells")
    series = view["series"][0]
    point = series["points"][0]
    app.select_post_analysis_point(
        {"points": [{"customdata": [point["run_id"], "", "", "", "", "", series["phase_key"]]}]},
        "cells",
    )
    assert app.server.state.post_analysis_point_can_open
    assert navigation == []
    app.open_post_analysis_scan()
    assert navigation == [point["run_id"]]
    assert opened == [True]
    assert not app.server.state.post_analysis_point_open


def test_expanding_a_saved_view_preserves_data_and_scientific_settings():
    app = app_with_data()
    settings = copy.deepcopy(app.server.state.post_analysis)
    data = copy.deepcopy(app._post_analysis_result)
    app.open_plot_inspector("post", "cells")
    assert app.server.state.post_analysis_inspector_open
    assert app.server.state.post_analysis_inspector_figure["data"]
    assert app.server.state.post_analysis == settings
    assert app._post_analysis_result == data


def test_observations_keep_missing_and_excluded_values_with_reasons():
    app = app_with_data()
    app.update_post_analysis_setting("excluded_runs", ["NOM_1_BANK1"])
    rows = app.server.state.post_analysis_observation_rows
    excluded = next(row for row in rows if row["scan"] == "NOM_1_BANK1")
    assert "Excluded" in excluded["status"]
    assert excluded["value"] == "Unavailable"
    assert len(rows) == 2


@pytest.mark.parametrize("field", ["phases", "banks", "samples"])
def test_clearing_a_display_filter_means_all_and_does_not_leave_an_error(field):
    app = app_with_data()
    app.update_post_analysis_setting(field, None)
    assert getattr(next(v for v in app._post_settings().views if v.id == "cells"), field) == []
    assert not app.server.state.post_analysis_error


def test_focus_defaults_to_one_view_and_reference_controls_follow_the_metric():
    app = app_with_data()
    assert app.server.state.post_analysis_show_all is False
    app.select_post_analysis_view("phases")
    assert app.server.state.post_analysis_reference_needed is False
    app.update_post_analysis_setting("representation", "difference")
    assert app.server.state.post_analysis_reference_needed is True


def test_the_picker_offers_only_implemented_recipes_without_discarding_saved_ones():
    app = app_with_data()
    choices = {row["value"]: row for row in app.server.state.post_analysis_recipes}
    for key in ["broadening", "atomic_structure"]:
        assert key not in choices
    assert not choices["phase_evolution"].get("props", {}).get("disabled")
    restored = PostAnalysisSettings(views=[ViewSettings(id="saved", kind="broadening")])
    app._set_post_settings(restored)
    assert app._post_settings() == restored


def test_observation_table_preserves_categorical_preparation_states():
    rows = RadarPdNovaApp._post_observations(
        {
            "series": [
                {
                    "phase": "Fe",
                    "bank": 1,
                    "points": [
                        {
                            "run_id": "scan-1",
                            "x": "annealed",
                            "axis_unit": "preparation state",
                            "value": None,
                            "reason": "missing cell value",
                        }
                    ],
                }
            ]
        }
    )
    assert rows[0]["x"] == "annealed preparation state"
    assert rows[0]["value"] == "Unavailable"
    assert rows[0]["status"] == "missing cell value"


def test_empty_axis_explains_the_missing_input_and_how_to_change_it():
    app = app_with_data()
    app.update_post_analysis_setting("axis", "voltage")
    panel = next(p for p in app.server.state.post_analysis_panels if p["id"] == "cells")
    assert panel["point_count"] == 0
    assert "No qualified voltage values" in panel["empty_reason"]
    assert "Choose another horizontal axis" in panel["empty_reason"]
    assert app.server.state.post_analysis_active_points == 0


def test_duplicate_views_keep_independent_axes_and_references():
    app = app_with_data()
    app.duplicate_post_analysis_view()
    assert len(app._post_settings().views) == 3
    duplicate = app.server.state.post_analysis_active
    app.update_post_analysis_setting("axis", "temperature")
    app.update_post_analysis_setting("reference_run", "NOM_2_BANK1")
    by_id = {v.id: v for v in app._post_settings().views}
    assert (
        by_id[duplicate].axis == "temperature" and by_id[duplicate].reference_run == "NOM_2_BANK1"
    )
    assert by_id["cells"].axis == "shared" and by_id["cells"].reference_run is None


def test_removing_one_duplicate_does_not_remove_the_other_view():
    app = app_with_data()
    app.duplicate_post_analysis_view()
    removed = app.server.state.post_analysis_active
    app.remove_post_analysis_view()
    assert removed not in {v.id for v in app._post_settings().views}
    assert {v.id for v in app._post_settings().views} == {"phases", "cells"}


def test_bad_setting_preserves_last_valid_contract_and_reports_actionable_error():
    app = app_with_data()
    before = copy.deepcopy(app.server.state.post_analysis)
    app.update_post_analysis_setting("metric", "unsupported")
    assert app.server.state.post_analysis == before
    assert "Unsupported post-analysis metric" in app.server.state.post_analysis_error


def test_presentation_callbacks_never_submit_refinements():
    app = app_with_data()

    def forbidden(*_, **__):
        raise AssertionError("A display change submitted a scientific job")

    app.service = SimpleNamespace(submit_snapshot=forbidden, create_submission_snapshot=forbidden)
    for field, value in [
        ("representation", "relative"),
        ("axis", "temperature"),
        ("connect", True),
        ("show_uncertainty", False),
        ("reference_run", "NOM_2_BANK1"),
        ("phases", ["phase-a"]),
    ]:
        app.update_post_analysis_setting(field, value)
        assert not app.server.state.post_analysis_error
    assert app.server.state.post_analysis_panels


def test_compatible_physical_axes_share_zoom_and_ranges_are_saved():
    app = app_with_data()
    app.relayout_post_analysis("cells", {"xaxis.range[0]": 315, "xaxis.range[1]": 325})
    by_id = {v.id: v for v in app._post_settings().views}
    assert by_id["cells"].x_range == (315.0, 325.0)
    assert by_id["phases"].x_range == (315.0, 325.0)
    app.relayout_post_analysis("cells", {"xaxis.autorange": True})
    assert all(v.x_range is None for v in app._post_settings().views)


def test_categorical_run_zoom_does_not_link_to_an_unrelated_axis():
    app = app_with_data()
    app.update_post_analysis_setting("axis", "run_number")
    app.relayout_post_analysis("cells", {"xaxis.range[0]": 0, "xaxis.range[1]": 1})
    by_id = {v.id: v for v in app._post_settings().views}
    assert by_id["cells"].x_range == (0.0, 1.0) and by_id["phases"].x_range is None


def test_watch_settings_and_references_are_checkpointed_without_restarting_jobs():
    app = app_with_data()
    state = WatchState(history_id="h")
    saved = []
    app._powgen_controller = SimpleNamespace(
        state=state, persist_state=lambda: saved.append(state.to_json())
    )
    app._post_analysis_scope = "watch:/SNS/NOM/IPTS-1/shared/autoreduce"
    app.update_post_analysis_setting("representation", "relative")
    asyncio.run(app.save_post_analysis_settings())
    assert saved and state.post_analysis["references"]
    restored = WatchState.from_dict(__import__("json").loads(saved[-1]))
    assert restored.post_analysis == app.server.state.post_analysis


def test_inspecting_a_single_result_cannot_overwrite_live_reference_settings():
    app = app_with_data()
    watch = WatchState(history_id="h", post_analysis=PostAnalysisSettings().model_dump(mode="json"))
    original = copy.deepcopy(watch.post_analysis)
    app._powgen_controller = SimpleNamespace(state=watch)
    app._post_analysis_scope = "single:some-job"
    app.update_post_analysis_setting("representation", "relative")
    assert watch.post_analysis == original


def test_mapping_import_is_atomic_and_conflicts_are_not_silently_replaced(tmp_path):
    app = app_with_data()
    path = tmp_path / "mapping.csv"
    header = "axis,run_id,sample_id,value,unit,source,kind,verified\n"
    path.write_text(header + "voltage,NOM_1_BANK1,,4.1,V,reviewed log,measured,true\n")
    app.server.state.post_analysis_mapping_path = str(path)
    app.import_post_analysis_mapping()
    before = copy.deepcopy(app.server.state.post_analysis)
    path.write_text(
        header
        + "voltage,NOM_2_BANK1,,4.2,V,reviewed log,measured,true\n"
        + "voltage,NOM_1_BANK1,,9.9,V,reviewed log,measured,true\n"
    )
    app.import_post_analysis_mapping()
    assert "conflicting duplicate" in app.server.state.post_analysis_error
    assert app.server.state.post_analysis == before
    app.clear_post_analysis_mapping()
    assert not app._post_settings().axis_mappings


def test_mapping_import_requires_review_and_does_not_assume_naive_times(tmp_path):
    app = app_with_data()
    path = tmp_path / "mapping.csv"
    path.write_text(
        "axis,run_id,value,unit,source,kind,verified\nvoltage,NOM_1_BANK1,4.1,V,log,measured,false\n"
    )
    app.server.state.post_analysis_mapping_path = str(path)
    app.import_post_analysis_mapping()
    assert "verified=true" in app.server.state.post_analysis_error
    assert not app._post_settings().axis_mappings


def test_rechecking_reusable_configuration_does_not_erase_post_analysis_choices():
    app = app_with_data()
    config = AnalysisConfig(sample_elements=["Fe"])
    app.service = SimpleNamespace(
        load_configuration_dataset=lambda _: config,
        load_json_document=lambda _: config.portable_contract(),
    )
    app._set_powgen_configuration_preview("config", config)
    app.update_post_analysis_selection(["distortion"])
    app._set_powgen_configuration_preview("config", config)
    assert [v.kind for v in app._post_settings().views] == ["distortion"]


def test_opening_an_uploaded_single_fit_preserves_the_experiment_history_instead_of_creating_one_point_trends():
    app = app_with_data()
    before = copy.deepcopy(app._post_analysis_result)
    history = copy.deepcopy(app._post_analysis_scans)
    packet = scan(1)["observables"]
    app.service = SimpleNamespace(
        result_payload=lambda _: {
            "summary": {"analysis_mode": "full", "status": "complete", "observables": packet}
        }
    )
    record = RunRecord(
        uid="job",
        name="uploaded-pattern",
        mode="full",
        history_id="h",
        config=AnalysisConfig(sample_elements=["Fe"]),
        inputs=InputSelection(
            source="upload", data_path="scan.gsa", instrument_path="profile.instprm"
        ),
    )
    app._load_results(record)
    assert app._post_analysis_result == before
    assert app._post_analysis_scans == history
    assert app._post_analysis_scope == "single:job"
