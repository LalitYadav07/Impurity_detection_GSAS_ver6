"""Saved-watch restoration is explicit, source-bound and never starts science."""
import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import PurePosixPath
import threading

import pytest

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.live_beamlines import beamline_adapter
from radar_pd_nova.live_restore import inspect_checkpoint_payload, read_saved_watch
from radar_pd_nova.models import AnalysisConfig, RunRecord, RunStatus
from radar_pd_nova.powgen_watch import WatchedRun


class SavedHistory:
    history_id = "history-1"
    galaxy_url = "https://local.invalid"
    galaxy_key = "synthetic-key"

    def __init__(self):
        self.payloads = {}
        self.reads = []
        self.config = AnalysisConfig(sample_elements=["Cu"])

    def _dataset_metadata(self, identifier):
        self.reads.append(("metadata", identifier))
        return {"id": identifier, "history_id": self.history_id, "state": "ok",
                "history_content_type": "dataset", "file_size": len(json.dumps(self.payloads[identifier]))}

    def load_text_document(self, identifier, *, max_bytes=None):
        self.reads.append(("checkpoint", identifier))
        if identifier not in self.payloads:raise OSError('Synthetic saved profile is unavailable')
        return json.dumps(self.payloads[identifier])

    def search_history_datasets(self, **kwargs):
        self.reads.append(("list", kwargs))
        return [{"id": key, "name": f"POWGEN watch state IPTS-1234 [{key}]", "state": "ok",
                 "display_name": f"Saved watch {key}"} for key in self.payloads]

    def load_configuration_dataset(self, key):
        self.reads.append(("configuration", key))
        return self.config.model_copy(deep=True)

    def __getattr__(self, name):
        raise AssertionError(f"Unexpected Galaxy operation: {name}")


def checkpoint(service, instrument="PG3", *, partial=True):
    adapter = beamline_adapter(instrument)
    extra = {}
    if instrument in {"NOM", "SNAP"}:
        extra.update(selected_banks=(2,), profile_bank=2, profile_acknowledged=True,
                     allow_exploratory=True, instrument_profile_dataset_id="profile-1")
    if instrument == "SNAP":
        extra["product_family"] = json.dumps([PurePosixPath(adapter.default_folder).name, "2f2708c073568e2a"])
    settings = adapter.settings(ipts="IPTS-1234", history_id=service.history_id,
        configuration_dataset_id="config-1", wavelength_angstrom="" if instrument != "PG3" else "1.5",
        max_active_jobs=2, late_arrival_window=17, retry_base_seconds=7, retry_max_seconds=21,
        main_cif_dataset_id="main-1", **extra)
    controller = adapter.controller(service, settings)
    for index, phase in enumerate(("completed", "submitted", "failed", "discovered") if partial else ("completed",), 101):
        run = WatchedRun(index, f"{controller.source_directory}/{instrument}_{index}.gsa", f"fp-{index}",
            instrument=instrument, bank=2 if instrument in {"NOM", "SNAP"} else None,
            experiment_number=42 if instrument == "HB2A" else None,
            galaxy_job_id=None if phase == "discovered" else f"job-{index}",
            galaxy_result_ids=(f"result-{index}",) if phase == "completed" else (),
            scientific_summary={"rwp": 5.0, "phases": []} if phase == "completed" else {},
            error="saved failure" if phase == "failed" else None,
            submission_attempts=2 if phase == "discovered" else 1,
            scan_metadata={"schema_version": 3, "sample_id": "sample-A"})
        getattr(controller.state, phase)[run.run_id] = run
    payload = json.loads(controller.capture_checkpoint().payload_json)
    service.payloads["checkpoint-1"] = payload
    return controller, payload


@pytest.fixture
def app(monkeypatch):
    instance = RadarPdNovaApp()
    instance.service = SavedHistory()
    instance._sync_activities = lambda: None
    instance.refresh_history = lambda **_: None
    instance._sync_runs = lambda: None
    instance.server.state.powgen_ipts = ""
    instance.server.state.live_beamline = "PG3"
    instance.server.state.flush()
    return instance


async def inspect(app):
    await app.list_saved_live_watches()
    app.server.state.live_restore_dataset_id = "checkpoint-1"
    app.server.state.flush()
    await app.inspect_saved_live_watch()


def restore(app):
    app.restore_saved_live_watch()
    if app.server.state.live_restore_confirm:
        app.confirm_restore_saved_live_watch()


@pytest.mark.parametrize("instrument", ["PG3", "NOM", "SNAP", "HB2A"])
def test_complete_settings_roundtrip_without_source_reads(instrument, monkeypatch):
    service = SavedHistory()
    original, payload = checkpoint(service, instrument)
    def forbidden(*_, **__):
        raise AssertionError("Restoration may not read facility metadata or submit")
    monkeypatch.setattr(type(original), "_hydrate_missing_metadata", forbidden)
    monkeypatch.setattr(type(original), "discover", forbidden)
    monkeypatch.setattr(type(original), "persist_state", forbidden)
    # Dedicated display-input tests cover the one saved-profile read. Keep
    # this state-adoption test independent of any profile fixture.
    monkeypatch.setattr(type(original), "_restore_display_inputs", lambda self:None)
    review = read_saved_watch(service, "checkpoint-1")
    restored = review.controller
    assert not review.missing and not review.blocked
    assert asdict(restored.settings) == asdict(original.settings)
    assert restored.state.as_dict() == original.state.as_dict()
    assert set(restored.records) == set(original.state.completed) | set(original.state.submitted) | set(original.state.failed)
    assert [entry[0] for entry in service.reads] == ["metadata", "checkpoint"]
    assert restored.state_dataset_ids == ["checkpoint-1"]
    assert restored.restore_document_sha256 == review.document_sha256
    assert restored._retry_submission_allowed() is False


@pytest.mark.parametrize("partial", [False, True])
def test_restore_stays_stopped_and_presents_checkpoint_rows(app, partial):
    original, payload = checkpoint(app.service, partial=partial)
    app.server.state.experiment_result_view = "refinement"
    async def forbidden():
        raise AssertionError("No monitor loop during restore")
    app._powgen_monitor_loop = forbidden
    def forbidden_replay(*_, **__):
        raise AssertionError("Restore must not read patterns or refresh/enroll a refinement catalog")
    app._refresh_post_analysis = forbidden_replay
    app._refinement_maybe_refresh = forbidden_replay
    asyncio.run(inspect(app))
    assert app.server.state.live_restore_ready
    assert app._powgen_controller is None
    assert any(row["label"] == "Saved runs" for row in app.server.state.live_restore_rows)
    restore(app)
    assert asdict(app._powgen_controller.settings) == asdict(original.settings)
    for phase in ("discovered", "submitted", "completed", "failed"):
        assert getattr(app._powgen_controller.state, phase) == getattr(original.state, phase)
    assert app._powgen_monitor_task is None and not app.server.state.powgen_monitoring
    assert app.server.state.powgen_next_check == "Stopped"
    assert app.server.state.experiment_result_view == "results"
    assert app.server.state.live_restored_checkpoint_id == "checkpoint-1"
    assert not app.server.state.live_restore_resume_blocked
    assert len(app.server.state.powgen_scientific_rows) == 1
    assert not any(row[0] == "configuration" for row in app.service.reads)


def test_legacy_missing_settings_survive_refresh_and_block_both_start_paths(app):
    _, payload = checkpoint(app.service)
    payload["watch"].pop("settings")
    payload["watch"].pop("settings_schema")
    asyncio.run(inspect(app))
    restore(app)
    controller = app._powgen_controller
    assert "main_cif_dataset_id" in controller.restore_missing_settings
    assert "frequency_hz" in controller.restore_missing_settings
    assert app.server.state.live_restore_resume_blocked
    saved_again = json.loads(controller.capture_checkpoint().payload_json)
    assert "settings" not in saved_again["watch"]
    assert saved_again["watch"]["missing_settings"] == list(controller.restore_missing_settings)
    reopened = inspect_checkpoint_payload(saved_again, app.service, "saved-again", document_sha256="hash")
    assert reopened.missing == controller.restore_missing_settings and reopened.blocked
    before = list(app.service.reads)
    asyncio.run(app.start_powgen_monitoring())
    asyncio.run(app.resume_existing_powgen_monitoring())
    assert not app.server.state.powgen_monitoring and app._powgen_monitor_task is None
    assert app.service.reads == before
    assert "read-only" in app.server.state.error_message


def test_complete_restore_resumes_only_after_explicit_previous_settings_action(app):
    original, _ = checkpoint(app.service)
    asyncio.run(inspect(app))
    restore(app)
    app.server.state.live_sample_elements = "Different current draft"
    loops = []
    async def loop():
        loops.append(app._powgen_controller.settings)
    app._powgen_monitor_loop = loop
    async def resume():
        await app.resume_existing_powgen_monitoring()
        await asyncio.sleep(0)
    asyncio.run(resume())
    assert loops == [original.settings]
    assert app.server.state.powgen_monitoring
    assert [r for r in app.service.reads if r[0] == "configuration"] == [("configuration", "config-1")]


@pytest.mark.parametrize("change", ["history", "service", "selection", "draft", "draft_away_and_back", "controller", "context", "clear", "end"])
def test_slow_inspection_cannot_replace_changed_or_cleared_view(app, change):
    checkpoint(app.service)
    asyncio.run(app.list_saved_live_watches())
    app.server.state.live_restore_dataset_id = "checkpoint-1"
    app.server.state.flush()
    entered, released = threading.Event(), threading.Event()
    original_read = app.service.load_text_document
    def slow(*args, **kwargs):
        entered.set()
        assert released.wait(5)
        return original_read(*args, **kwargs)
    app.service.load_text_document = slow
    async def no_save():
        pass
    app._stop_session_enrollment = no_save
    app._leave_session_page = lambda nonce: app._acknowledge_session_navigation(nonce)
    app.exit_callback = no_save
    async def scenario():
        task = asyncio.create_task(app.inspect_saved_live_watch())
        assert await asyncio.to_thread(entered.wait, 2)
        if change == "history": app.service.history_id = "other-history"
        elif change == "service": app.service = SavedHistory()
        elif change == "selection":
            app.server.state.live_restore_dataset_id = "another-checkpoint"
            app.server.state.flush()
        elif change == "draft": app.server.state.live_sample_elements = "Fe"
        elif change == "draft_away_and_back":
            previous = app.server.state.live_sample_elements
            app.server.state.live_sample_elements = "Fe"
            app.server.state.flush()
            # No connected Trame client marks this unit harness ready;
            # dispatch the same registered input-change callback explicitly.
            app._live_restore_draft_changed()
            app.server.state.live_sample_elements = previous
            app.server.state.flush()
        elif change == "controller": app._powgen_controller = object()
        elif change == "context": app._result_context = object()
        elif change == "clear": await app.clear_workspace()
        elif change == "end": await app.end_session()
        released.set()
        await task
    asyncio.run(scenario())
    assert not app.server.state.live_restore_ready
    assert app._live_restore_review is None
    assert not app.server.state.live_restore_busy
    assert not app.server.state.powgen_monitoring


def test_confirmed_restore_preserves_old_records_and_rejects_changed_confirmation(app):
    checkpoint(app.service)
    old = RunRecord(uid="old-job", name="Old result", mode="full", status=RunStatus.OK, history_id=app.service.history_id)
    app.records[old.uid] = old
    app.server.state.selected_run_uid = old.uid
    asyncio.run(inspect(app))
    app.restore_saved_live_watch()
    assert app.server.state.live_restore_confirm and app._powgen_controller is None
    app.server.state.live_sample_elements = "Edited after confirmation"
    app.confirm_restore_saved_live_watch()
    assert app._powgen_controller is None
    asyncio.run(app.inspect_saved_live_watch())
    restore(app)
    assert app.records[old.uid] is old
    assert app.server.state.selected_run_uid == ""
    assert app._powgen_controller is not None


def test_restore_retains_cached_results_for_the_same_job(app):
    checkpoint(app.service)
    cached = RunRecord(uid="job-101", galaxy_job_id="job-101", history_id=app.service.history_id,
        name="Already downloaded", mode="full", status=RunStatus.OK, summary={"source": "cached"},
        output_dataset_ids={"summary": "summary-101"})
    app.records[cached.uid] = cached
    asyncio.run(inspect(app))
    restore(app)
    assert app.records[cached.uid] is cached
    assert app._powgen_controller.records["PG3_101"] is cached
    assert cached.output_dataset_ids == {"summary": "summary-101"}


def test_double_inspect_and_active_monitor_are_not_concurrent(app):
    checkpoint(app.service)
    app.server.state.powgen_monitoring = True
    asyncio.run(app.list_saved_live_watches())
    assert not app.service.reads
    app.server.state.powgen_monitoring = False
    asyncio.run(app.list_saved_live_watches())
    app.server.state.live_restore_dataset_id = "checkpoint-1"
    app.server.state.flush()
    entered, released = threading.Event(), threading.Event()
    original = app.service.load_text_document
    def slow(*args, **kwargs):
        entered.set(); assert released.wait(5)
        return original(*args, **kwargs)
    app.service.load_text_document = slow
    async def scenario():
        task = asyncio.create_task(app.inspect_saved_live_watch())
        assert await asyncio.to_thread(entered.wait, 2)
        await app.inspect_saved_live_watch()
        await app.start_powgen_monitoring()
        released.set(); await task
    asyncio.run(scenario())
    assert [r[0] for r in app.service.reads].count("checkpoint") == 1
    assert app._powgen_controller is None and not app.server.state.powgen_monitoring


@pytest.mark.parametrize("filter_text,query", [("", "watch state"), ("37869", "watch state IPTS-37869"),
    ("ipts-37869", "watch state IPTS-37869")])
def test_list_filter_uses_literal_checkpoint_name_and_retains_query_on_paging(app, filter_text, query):
    app.server.state.live_restore_search = filter_text
    def page(**kwargs):
        app.service.reads.append(("list", kwargs))
        return [{"id": f"saved-{i}", "name": f"POWGEN watch state IPTS-37869 [watch:{i}]", "state": "ok"} for i in range(100)]
    app.service.search_history_datasets = page
    asyncio.run(app.list_saved_live_watches())
    asyncio.run(app.older_saved_live_watches())
    assert [r[1]["query"] for r in app.service.reads] == [query, query]
    assert [r[1]["offset"] for r in app.service.reads] == [0, 100]
    assert len(app.server.state.live_restore_options) == 100


def test_invalid_saved_watch_filter_does_not_read_or_hide_current_choices(app):
    app.server.state.live_restore_options = [{"id": "known"}]
    app.server.state.live_restore_search = "POWGEN"
    asyncio.run(app.list_saved_live_watches())
    assert not app.service.reads
    assert app.server.state.live_restore_options == [{"id": "known"}]
    assert "IPTS number" in app.server.state.live_restore_message


def test_restore_controls_explain_stopped_inspection_and_draft_separation(app):
    ui = app.layout.html
    assert "Restore a saved Live watch (stopped)" in ui
    assert "Recheck selected checkpoint" in ui
    checkpoint_control = ui[ui.index('<VAutocomplete v-model="live_restore_dataset_id"'):]
    assert '@update:modelValue="trigger(' in checkpoint_control.split('/>', 1)[0]
    assert "Optional IPTS number" in ui
    assert "The editable setup draft remains separate" in ui


def test_selecting_a_saved_watch_previews_it_without_a_second_click(app):
    checkpoint(app.service)

    async def scenario():
        await app.list_saved_live_watches()
        await app.inspect_saved_live_watch("checkpoint-1")

    asyncio.run(scenario())
    assert app.server.state.live_restore_dataset_id == "checkpoint-1"
    assert app.server.state.live_restore_ready
    assert any(row["label"] == "Saved runs" for row in app.server.state.live_restore_rows)
    assert app._powgen_controller is None


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p.update(history_id="other"), "another Galaxy History"),
    (lambda p: p["watch"]["settings"].update(history_id="other"), "History or source"),
    (lambda p: p["watch"]["settings"].update(max_active_jobs=True), "integer"),
    (lambda p: p["watch"]["settings"].update(selected_banks=[False]), "integer"),
    (lambda p: p["watch"].update(database_dataset_id="other"), "Conflicting"),
    (lambda p: p["watch"].update(source_directory="/SNS/PG3/IPTS-999/shared/autoreduce"), "source folder"),
    (lambda p: next(iter(p["completed"].values())).update(source_path="/elsewhere/data.gsa"), "source identity"),
    (lambda p: p.update(completed={"wrong": next(iter(p["completed"].values()))}), "does not match"),
])
def test_invalid_or_cross_scope_checkpoint_is_rejected(mutation, match):
    service = SavedHistory()
    _, payload = checkpoint(service)
    payload = deepcopy(payload)
    mutation(payload)
    with pytest.raises((TypeError, ValueError), match=match):
        inspect_checkpoint_payload(payload, service, "bad", document_sha256="hash")


@pytest.mark.parametrize("field,value", [("resolver", "old-resolver"), ("catalog_sha256", "old-catalog"), ("checkpoint_key", "wrong")])
def test_incompatible_saved_runtime_still_opens_read_only(field, value):
    service = SavedHistory()
    _, payload = checkpoint(service)
    payload["watch"][field] = value
    review = inspect_checkpoint_payload(payload, service, "old-runtime", document_sha256="hash")
    assert review.blocked and review.controller.state.completed


@pytest.mark.parametrize("metadata", [
    {"history_id": "other"}, {"deleted": True}, {"purged": True}, {"state": "running"},
    {"id": "different"}, {"history_content_type": "dataset_collection"}, {"file_size": 100 * 1024 * 1024},
])
def test_only_ready_current_history_metadata_can_be_read(metadata):
    service = SavedHistory()
    checkpoint(service)
    original = service._dataset_metadata
    service._dataset_metadata = lambda key: {**original(key), **metadata}
    with pytest.raises(ValueError):
        read_saved_watch(service, "checkpoint-1")
    assert [r[0] for r in service.reads] == ["metadata"]


@pytest.mark.parametrize("cancel", ["handler", "dialog_closed"])
def test_delayed_confirmation_after_cancel_never_replaces_view(app, cancel):
    checkpoint(app.service)
    app.server.state.powgen_ipts = "1234"
    asyncio.run(inspect(app))
    app.restore_saved_live_watch()
    assert app.server.state.live_restore_confirm
    if cancel == "handler":
        app.cancel_restore_saved_live_watch()
        assert app._live_restore_confirm_signature is None
    else:
        app.server.state.live_restore_confirm = False
    app.confirm_restore_saved_live_watch()
    assert app._powgen_controller is None
