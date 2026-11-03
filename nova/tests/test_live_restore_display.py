"""Restore copy, timestamps and server-owned continuation field edits."""
import asyncio
from html.parser import HTMLParser
from types import SimpleNamespace

import pytest

from test_live_restore import app, checkpoint, inspect, restore
from test_live_missing_settings import legacy, open_review
from test_live_paused_refresh import harness
from radar_pd_nova.live_restore import state_recorded_display
from radar_pd_nova.live_settings_review import FIELDS, missing_setting_labels
from radar_pd_nova.app import RadarPdNovaApp


@pytest.mark.parametrize("instrument,conditions,name", [
    ("NOM", "NOM NeXus file", "NOMAD"),
    ("SNAP", "SNAP NeXus file", "SNAP"),
    ("HB2A", "HB2A SPICE acquisition and reviewed channel mapping", "HB-2A"),
])
def test_restored_watch_caption_and_actions_use_saved_beamline(app, instrument, conditions, name):
    checkpoint(app.service, instrument, partial=False)
    asyncio.run(inspect(app))
    restore(app)
    state = app.server.state
    assert state.live_beamline == "PG3"
    assert conditions in state.powgen_dashboard_notice
    assert state.powgen_dashboard_title.startswith(name + " ·")
    assert state.powgen_scan_options[0]["value"].startswith(instrument + "_")
    app.stop_powgen_monitoring()
    assert state.powgen_message.startswith(name + " monitoring stopped.")
    state.powgen_monitoring = True
    app.refresh_powgen_monitor_now()
    assert f"checking the {name} source" in state.powgen_message
    state.powgen_monitoring = False


def test_watch_adapter_uses_legacy_state_then_draft_fallback():
    app = RadarPdNovaApp.__new__(RadarPdNovaApp)
    app.server = SimpleNamespace(state=SimpleNamespace(live_beamline="PG3"))
    app._powgen_controller = SimpleNamespace(state=SimpleNamespace(instrument="NOM"))
    assert app._watch_adapter().instrument == "NOM"
    app._powgen_controller = None
    assert app._watch_adapter().instrument == "PG3"


@pytest.mark.parametrize("stamp,expected", [
    ("2026-11-01T20:59:49.137432+00:00", "2026-11-01 15:59:49 EST"),
    ("2026-07-01T20:59:49Z", "2026-07-01 16:59:49 EDT"),
    ("2026-11-01T20:59:49", "2026-11-01 20:59:49 (timezone not recorded)"),
    ("", "Not recorded"),
])
def test_state_time_is_human_readable_without_inventing_a_timezone(stamp, expected):
    assert state_recorded_display(stamp) == expected


def test_missing_labels_reuse_dialog_fields_and_keep_internal_keys_out_of_copy(legacy):
    controller = legacy._powgen_controller
    labels = missing_setting_labels(controller.restore_missing_settings)
    assert labels == [FIELDS[name][0] for name in controller.restore_missing_settings]
    assert all(name not in legacy.server.state.live_restore_message for name in controller.restore_missing_settings)
    assert all(label in legacy.server.state.live_restore_message for label in labels)
    assert legacy.server.state.powgen_last_checked == "Saved checkpoint; not refreshed yet"


def test_preview_calls_timestamp_recorded_state_not_upload_time(app):
    _, payload = checkpoint(app.service)
    payload["updated_utc"] = "2026-11-01T20:59:49.137432+00:00"
    asyncio.run(inspect(app))
    rows = {row["label"]: row["value"] for row in app.server.state.live_restore_rows}
    assert rows["State recorded"] == "2026-11-01 15:59:49 EST"
    assert "Saved time" not in rows
    restore(app)
    assert "not the dataset upload time" in app.server.state.powgen_message


def test_missing_recorded_timestamp_is_not_replaced_by_restore_time(app):
    _, payload = checkpoint(app.service)
    payload.pop("updated_utc")
    asyncio.run(inspect(app))
    rows = {row["label"]: row["value"] for row in app.server.state.live_restore_rows}
    assert rows["State recorded"] == "Not recorded"


def test_numeric_edit_is_server_owned_and_survives_another_control_update(legacy):
    open_review(legacy)
    state = legacy.server.state
    old_values = state.live_missing_values
    assert old_values["max_active_jobs"] == 5
    legacy.update_missing_live_setting("max_active_jobs", "1")
    assert state.live_missing_values is not old_values
    assert old_values["max_active_jobs"] == 5
    assert state.live_missing_values["max_active_jobs"] == "1"
    assert state.live_missing_acknowledged is False
    # An ordinary top-level dropdown update must not restore stale nested data.
    state.live_missing_cif_choice = "none"
    state.live_missing_frequency_mode = "automatic"
    state.flush()
    assert state.live_missing_values["max_active_jobs"] == "1"
    state.live_missing_acknowledged = True
    asyncio.run(legacy.confirm_missing_live_settings())
    assert legacy._powgen_controller.settings.max_active_jobs == 1
    assert not state.powgen_monitoring


def test_rapid_confirm_after_edit_cannot_commit_the_previous_default(legacy):
    open_review(legacy)
    original = legacy._powgen_controller.settings
    legacy.update_missing_live_setting("max_active_jobs", "1")
    asyncio.run(legacy.confirm_missing_live_settings())
    assert legacy._powgen_controller.settings is original
    assert legacy.server.state.live_missing_values["max_active_jobs"] == "1"
    legacy.server.state.live_missing_acknowledged = True
    asyncio.run(legacy.confirm_missing_live_settings())
    assert legacy._powgen_controller.settings.max_active_jobs == 1


@pytest.mark.parametrize("updates,labels", [
    ({"max_active_jobs": "9"}, ["max_active_jobs"]),
    ({"retry_base_seconds": "30", "retry_max_seconds": "5"}, ["retry_base_seconds", "retry_max_seconds"]),
])
def test_visible_validation_errors_use_the_same_friendly_labels(legacy, updates, labels):
    open_review(legacy)
    original = legacy._powgen_controller.settings
    for field, value in updates.items():
        legacy.update_missing_live_setting(field, value)
    legacy.server.state.live_missing_acknowledged = True
    asyncio.run(legacy.confirm_missing_live_settings())
    message = legacy.server.state.live_missing_message
    assert all(FIELDS[name][0] in message and name not in message for name in labels)
    assert legacy._powgen_controller.settings is original


@pytest.mark.parametrize("text", ["", "-", "1e", "01", None])
def test_partial_input_text_is_preserved_until_confirmation(legacy, text):
    open_review(legacy)
    legacy.update_missing_live_setting("retry_base_seconds", text)
    assert legacy.server.state.live_missing_values["retry_base_seconds"] == text


@pytest.mark.parametrize("condition", ["closed", "busy", "monitoring", "stale", "known_key", "nonstring_key", "invalid_object"])
def test_invalid_or_stale_field_edit_cannot_change_review_values(legacy, condition):
    open_review(legacy)
    state = legacy.server.state
    before = dict(state.live_missing_values)
    name, value = "max_active_jobs", "1"
    if condition == "closed": state.live_missing_review_open = False
    elif condition == "busy": state.live_restore_busy = True
    elif condition == "monitoring": state.powgen_monitoring = True
    elif condition == "stale": state.live_sample_elements = "Different draft"
    elif condition == "known_key": name = "configuration_dataset_id"
    elif condition == "nonstring_key": name = []
    elif condition == "invalid_object": value = {"unsafe": "object"}
    legacy.update_missing_live_setting(name, value)
    assert state.live_missing_values == before


class Tags(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.tags = []
        self.feed(source)
    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def test_rendered_field_uses_explicit_callback_and_outlined_label(app):
    tags = Tags(app.layout.html).tags
    fields = [attrs for tag, attrs in tags if tag == "vtextfield"
              and attrs.get(":modelvalue") == "live_missing_values[row.name]"]
    assert len(fields) == 1
    assert fields[0]["variant"] == "outlined"
    assert "v-model" not in fields[0]
    assert any(key.startswith("@update:") and "row.name" in value for key, value in fields[0].items())


def test_dashboard_refresh_is_stopped_only_and_describes_no_new_scans(app):
    ui = app.layout.html
    assert "Refresh saved results" in ui
    assert "It does not discover or submit scans" in ui
    assert "!powgen_monitoring &amp;&amp; powgen_results_refresh_available" in ui or "!powgen_monitoring && powgen_results_refresh_available" in ui
    assert "The editable setup draft is separate from this restored watch" in ui
    assert "Last status check:" in ui


def test_stopped_refresh_updates_last_check_without_admitting_work():
    application, trace, _, _ = harness()
    application.server.state.powgen_last_checked = "Saved checkpoint; not refreshed yet"
    async def run():
        application.refresh_powgen_monitor_now()
        await application._powgen_stopped_refresh_task
    asyncio.run(run())
    assert trace == ["refresh", "snapshot", ("persist", {"completed": ["scan"]})]
    assert application.server.state.powgen_last_checked != "Saved checkpoint; not refreshed yet"
    assert not application.server.state.powgen_monitoring
