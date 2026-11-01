"""Checkpoint choices and restored-watch actions remain explicit and discoverable."""
import asyncio
from html.parser import HTMLParser
import re
import threading

import pytest

from radar_pd_nova.galaxy_service import GalaxyService
from radar_pd_nova.live_restore import saved_watch_title
from test_live_restore import app, checkpoint, inspect, restore


DATASET_ID = "fb77ae332c9d5aa0"
UPLOAD_ID = "8619bbf067764223b9ee6fad31197f4b"


def history_item(name, **metadata):
    return GalaxyService._history_dataset_item({
        "id": DATASET_ID, "name": name, "state": "ok",
        "create_time": "2026-11-01T21:22:08.123456",
        "update_time": "2026-11-01T21:23:12.123456", **metadata,
    })


@pytest.mark.parametrize("filename,beamline", [
    ("powgen", "POWGEN"), ("nomad", "NOMAD"), ("snap", "SNAP"), ("hb-2a", "HB-2A"),
])
def test_generated_checkpoint_names_become_compact_experiment_choices(filename, beamline):
    name = (f"RADAR-PD {beamline} watch state IPTS-37869 [watch:{'a' * 64}] | "
            f"IPTS-37869_{filename}_watch_state-{UPLOAD_ID}.json")
    row = history_item(name)
    assert saved_watch_title(row) == f"{beamline} · IPTS-37869 · uploaded 2026-11-01 21:22 UTC · fb77ae33"
    assert row["name"] == name  # The durable dataset identity and name are untouched.


def test_old_generated_label_without_filename_is_also_readable():
    row = history_item(f"RADAR-PD POWGEN watch state IPTS-37869 [watch:{'a' * 64}]")
    assert saved_watch_title(row).startswith("POWGEN · IPTS-37869 · uploaded ")


@pytest.mark.parametrize("display", [None, "Copper cooling series, after heater restart"])
def test_meaningful_custom_names_and_display_names_are_preserved(display):
    row = history_item("Copper watch state, first cooling series")
    if display is not None:
        row["display_name"] = display
    assert saved_watch_title(row) == (
        f"{display or 'Copper watch state, first cooling series'} · uploaded 2026-11-01 21:22 UTC · fb77ae33")


@pytest.mark.parametrize("created,updated,expected", [
    ("2026-11-01T16:22:08-05:00", "2026-11-02T00:00:00Z", "uploaded 2026-11-01 21:22 UTC"),
    ("2026-11-01T21:22:08Z", None, "uploaded 2026-11-01 21:22 UTC"),
    (None, "2026-11-01T21:23:12", "updated 2026-11-01 21:23 UTC"),
    ("unavailable", "2026-11-01T21:23:12Z", "updated 2026-11-01 21:23 UTC"),
    (None, None, "upload time unavailable"),
])
def test_dataset_timestamp_is_utc_and_never_mislabeled_as_recorded_state(created, updated, expected):
    row = history_item("POWGEN watch state IPTS-37869", create_time=created, update_time=updated)
    assert saved_watch_title(row) == f"POWGEN · IPTS-37869 · {expected} · fb77ae33"
    assert row["create_time"] == (created or "")


def test_absent_metadata_is_safe_and_does_not_invent_a_time():
    assert saved_watch_title({"name": None, "display_name": None, "id": None}) == (
        "Saved Live watch · upload time unavailable")


def test_listing_formats_choices_without_reading_checkpoints_or_changing_the_draft(app):
    row = history_item("RADAR-PD POWGEN watch state IPTS-37869 | "
                       f"IPTS-37869_powgen_watch_state-{UPLOAD_ID}.json")
    def listing(**kwargs):
        app.service.reads.append(("list", kwargs))
        return [row]
    app.service.search_history_datasets = listing
    state = app.server.state
    state.live_sample_elements = "Cu"
    asyncio.run(app.list_saved_live_watches())
    assert state.live_restore_options[0]["title"] == (
        "POWGEN · IPTS-37869 · uploaded 2026-11-01 21:22 UTC · fb77ae33")
    assert state.live_restore_options[0]["id"] == DATASET_ID
    assert [kind for kind, _ in app.service.reads] == ["list"]
    assert state.live_sample_elements == "Cu" and state.powgen_ipts == ""
    assert app._powgen_controller is None and app._powgen_monitor_task is None
    assert not state.powgen_monitoring


class UiTree(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.nodes, self.stack = [], []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        node = dict(tag=tag, attrs=dict(attrs), text="", parent=self.stack[-1] if self.stack else None)
        self.nodes.append(node)
        self.stack.append(node)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index]["tag"] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        if self.stack:
            self.stack[-1]["text"] += data


def visible(node, values):
    # These rendered visibility conditions use only boolean operators and state.
    expression = node["attrs"]["v-if"].replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", " not ", expression).strip()
    return bool(eval(expression, {"__builtins__": {}}, values))


def test_restored_header_owns_resume_and_draft_retains_only_partial_library_resume(app):
    nodes = UiTree(app.layout.html).nodes
    buttons = [node for node in nodes if node["tag"] == "vbtn"]
    refresh = next(node for node in buttons if node["text"].strip() == "Refresh saved results")
    resumes = [node for node in buttons if node["text"].strip() == "Resume existing watch"]
    assert len(resumes) == 2
    header = next(node for node in resumes if node["parent"] is refresh["parent"])
    draft = next(node for node in resumes if node is not header)
    assert header["attrs"]["@click"] == draft["attrs"]["@click"]
    assert header["attrs"][":disabled"] == draft["attrs"][":disabled"]
    base = dict(live_library_builder_decision_pending=False, live_restored_checkpoint_id="",
                powgen_results_refresh_available=False, live_restore_resume_blocked=False,
                powgen_monitoring=False)
    for changes, expected in [
        ({}, (False, False)),
        ({"live_restored_checkpoint_id": "saved", "powgen_results_refresh_available": True}, (True, False)),
        ({"live_restored_checkpoint_id": "saved", "powgen_results_refresh_available": True,
          "live_restore_resume_blocked": True}, (False, False)),
        ({"live_restored_checkpoint_id": "saved", "powgen_results_refresh_available": True,
          "powgen_monitoring": True}, (False, False)),
        ({"live_library_builder_decision_pending": True, "powgen_results_refresh_available": True}, (False, True)),
        ({"live_restored_checkpoint_id": "saved", "powgen_results_refresh_available": True,
          "live_library_builder_decision_pending": True}, (True, False)),
    ]:
        values = {**base, **changes}
        assert (visible(header, values), visible(draft, values)) == expected
    alert = next(node for node in nodes if node["tag"] == "valert" and node["attrs"].get("v-if") == "live_launch_busy")
    assert "powgen_results_refresh_busy ? 'Refreshing saved results'" in alert["attrs"][":title"]
    assert alert["attrs"][":text"].startswith("powgen_results_refresh_busy ? powgen_message :")


def test_resume_flushes_busy_feedback_before_the_saved_configuration_read(app, monkeypatch):
    original, _ = checkpoint(app.service)
    asyncio.run(inspect(app))
    restore(app)
    state = app.server.state
    state.live_test_message = "Previous one-scan result"
    entered, release = threading.Event(), threading.Event()
    load = app.service.load_configuration_dataset
    def slow_read(key):
        entered.set()
        assert release.wait(5)
        return load(key)
    app.service.load_configuration_dataset = slow_read
    flushed = []
    flush = type(state).flush
    def record_flush(current):
        if current is state:
            flushed.append((state.live_launch_busy, state.powgen_message, state.live_test_message))
        flush(current)
    # State.__setattr__ writes client values; intercept the real flush method.
    monkeypatch.setattr(type(state), "flush", record_flush)
    loops = []
    async def loop():
        loops.append(app._powgen_controller.settings)
    app._powgen_monitor_loop = loop
    async def scenario():
        task = asyncio.create_task(app.resume_existing_powgen_monitoring())
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            assert not state.powgen_monitoring and not loops
            assert any(busy and "Monitoring remains stopped" in message and not test
                       for busy, message, test in flushed)
        finally:
            release.set()
            await task
        await asyncio.sleep(0)
    asyncio.run(scenario())
    assert loops == [original.settings]
    assert state.powgen_monitoring and not state.live_launch_busy
