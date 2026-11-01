"""A legacy watch resumes only after an explicit, durable continuation review."""
import asyncio
from copy import deepcopy
from dataclasses import asdict, replace
import json
import threading

import pytest

from test_live_restore import app, checkpoint, inspect, restore, SavedHistory
from radar_pd_nova.checkpoint_uploads import CheckpointUploadPending
from radar_pd_nova.live_restore import inspect_checkpoint_payload
from radar_pd_nova.live_settings_review import (
    json_sha256, lifecycle_sha256, prepare_continuation_settings, validate_continuation_receipt,
)


@pytest.fixture
def legacy(app):
    _, payload = checkpoint(app.service)
    payload["watch"].pop("settings")
    payload["watch"].pop("settings_schema")
    asyncio.run(inspect(app))
    restore(app)
    return app


def open_review(app, *, main="none", acknowledge=True):
    app.review_missing_live_settings()
    assert app.server.state.live_missing_review_open
    app.server.state.live_missing_acknowledged = acknowledge
    app.server.state.live_missing_cif_choice = main
    app.server.state.live_missing_frequency_mode = "automatic"


def uploader(app, *, pending=False, entered=None, release=None):
    calls, uploads, known = [], [], {}
    def save(payload, **kwargs):
        key = json_sha256(payload)
        calls.append(key)
        if entered is not None:
            entered.set()
            assert release.wait(5)
        if key not in known:
            known[key] = f"saved-{len(known) + 1}"
            uploads.append(deepcopy(payload))
            if pending:
                raise CheckpointUploadPending(known[key], "queued")
        return known[key]
    app.service.upload_checkpoint_json_document = save
    app.service.upload_json_document = save
    return calls, uploads


def test_missing_settings_begin_as_explicitly_unconfirmed(legacy):
    legacy.review_missing_live_settings()
    state = legacy.server.state
    assert not state.live_missing_acknowledged
    assert state.live_missing_cif_choice == ""
    assert state.live_missing_frequency_mode == ""
    assert {row["name"] for row in state.live_missing_rows} == {
        "frequency_hz", "late_arrival_window", "max_active_jobs", "max_submission_attempts",
        "retry_base_seconds", "retry_max_seconds"}
    assert "unconfirmed" in state.live_missing_message
    assert legacy.server.state.live_restore_resume_blocked


def test_review_overlays_only_missing_fields_and_preserves_same_controller_and_runs(legacy):
    controller = legacy._powgen_controller
    before, run_hash = asdict(controller.settings), lifecycle_sha256(controller)
    missing = controller.restore_missing_settings
    reads = list(legacy.service.reads)
    open_review(legacy)
    legacy.server.state.live_missing_values = {**legacy.server.state.live_missing_values, "max_active_jobs": "1"}
    asyncio.run(legacy.confirm_missing_live_settings())
    assert legacy._powgen_controller is controller
    assert not legacy.server.state.powgen_monitoring and not legacy.server.state.live_restore_resume_blocked
    assert controller.settings.max_active_jobs == 1 and controller.settings.main_cif_dataset_id is None
    assert controller.settings.frequency_hz is None
    assert all(asdict(controller.settings)[key] == value for key, value in before.items() if key not in missing)
    assert lifecycle_sha256(controller) == run_hash
    assert legacy.service.reads == reads  # None requires no CIF/configuration read.
    assert controller.restore_settings_checkpoint_pending
    assert controller.restore_settings_review["source_checkpoint_dataset_id"] == "checkpoint-1"
    assert controller.restore_settings_review["original_run_state_sha256"] == run_hash
    assert "stay" not in legacy.server.state.error_message


def test_definition_failure_does_not_partially_replace_settings(legacy, monkeypatch):
    controller = legacy._powgen_controller
    original_settings, original_definition = controller.settings, controller.definition
    open_review(legacy)
    def unavailable(*_):
        raise OSError("Profile catalog temporarily unavailable")
    monkeypatch.setattr(type(controller.settings), "definition", unavailable)
    asyncio.run(legacy.confirm_missing_live_settings())
    assert controller.settings is original_settings and controller.definition is original_definition
    assert legacy.server.state.live_restore_resume_blocked
    assert "unavailable" in legacy.server.state.live_missing_message


@pytest.mark.parametrize("ack,choice", [(False, "none"), (True, "")])
def test_no_implicit_acknowledgment_or_main_cif_none(legacy, ack, choice):
    original = legacy._powgen_controller.settings
    open_review(legacy, main=choice, acknowledge=ack)
    asyncio.run(legacy.confirm_missing_live_settings())
    assert legacy._powgen_controller.settings is original
    assert legacy.server.state.live_restore_resume_blocked


@pytest.mark.parametrize("field,value", [
    ("max_active_jobs", 0), ("max_active_jobs", 6), ("max_active_jobs", True),
    ("max_submission_attempts", -1), ("max_submission_attempts", "99999"),
    ("late_arrival_window", -1), ("late_arrival_window", 5001),
    ("retry_base_seconds", 1.5), ("retry_base_seconds", "-2"),
    ("retry_max_seconds", "NaN"), ("retry_max_seconds", 86401),
    ("retry_max_seconds", None), ("retry_base_seconds", None),
    ("frequency_hz", "NaN"), ("frequency_hz", "Infinity"), ("frequency_hz", "0"),
    ("frequency_hz", "1001"), ("frequency_hz", False),
])
def test_invalid_missing_values_fail_without_mutation(legacy, field, value):
    controller = legacy._powgen_controller
    original, runs = controller.settings, lifecycle_sha256(controller)
    values = {key: asdict(original)[key] for key in controller.restore_missing_settings}
    values[field] = value
    with pytest.raises(ValueError):
        prepare_continuation_settings(controller, values, acknowledged=True)
    assert controller.settings is original and lifecycle_sha256(controller) == runs


def test_extra_known_field_and_retry_inversion_are_rejected(legacy):
    controller = legacy._powgen_controller
    values = {key: asdict(controller.settings)[key] for key in controller.restore_missing_settings}
    with pytest.raises(ValueError, match="exact missing"):
        prepare_continuation_settings(controller, {**values, "database_dataset_id": "replacement"}, acknowledged=True)
    with pytest.raises(ValueError, match="at least"):
        prepare_continuation_settings(controller, {**values, "retry_base_seconds": 30, "retry_max_seconds": 5}, acknowledged=True)


def test_automatic_frequency_and_zero_retry_values_are_explicit_and_preserved(legacy):
    controller = legacy._powgen_controller
    values = {key: asdict(controller.settings)[key] for key in controller.restore_missing_settings}
    settings, receipt = prepare_continuation_settings(controller,
        {**values, "frequency_hz": None, "late_arrival_window": 0, "retry_base_seconds": 0, "retry_max_seconds": 0}, acknowledged=True)
    assert settings.frequency_hz is None and settings.late_arrival_window == 0
    assert settings.retry_base_seconds == settings.retry_max_seconds == 0
    assert settings.instrument_profile_dataset_id == controller.settings.instrument_profile_dataset_id
    assert receipt["confirmed_missing_settings"]["frequency_hz"] is None


@pytest.mark.parametrize("metadata,allowed", [
    ({}, True), ({"history_id": "another-history"}, False), ({"state": "running"}, False),
    ({"deleted": True}, False), ({"purged": True}, False), ({"extension": "txt", "name": "not-a-cif"}, False),
])
def test_chosen_main_cif_requires_current_history_metadata(legacy, metadata, allowed):
    controller = legacy._powgen_controller
    original = controller.settings
    open_review(legacy, main="history")
    legacy.server.state.live_missing_cif_id = "cif-1"
    legacy.service._dataset_metadata = lambda key: {
        "id": key, "history_id": legacy.service.history_id, "state": "ok", "extension": "cif", **metadata}
    asyncio.run(legacy.confirm_missing_live_settings())
    assert (controller.settings.main_cif_dataset_id == "cif-1") is allowed
    assert (controller.settings is original) is not allowed


@pytest.mark.parametrize("change", ["history", "draft", "clear", "value", "controller"])
def test_slow_cif_metadata_cannot_apply_a_revoked_review(legacy, change):
    original_controller = legacy._powgen_controller
    original_settings = original_controller.settings
    entered, release = threading.Event(), threading.Event()
    def metadata(key):
        entered.set(); assert release.wait(5)
        return {"id": key, "history_id": "history-1", "state": "ok", "extension": "cif"}
    legacy.service._dataset_metadata = metadata
    async def no_save(): pass
    legacy._stop_session_enrollment = no_save
    open_review(legacy, main="history")
    legacy.server.state.live_missing_cif_id = "cif-1"
    async def scenario():
        task = asyncio.create_task(legacy.confirm_missing_live_settings())
        assert await asyncio.to_thread(entered.wait, 2)
        if change == "history": legacy.service.history_id = "another-history"
        elif change == "draft": legacy.server.state.live_sample_elements = "Ni"
        elif change == "value": legacy.server.state.live_missing_values = {**legacy.server.state.live_missing_values, "max_active_jobs": 1}
        elif change == "clear": await legacy.clear_workspace()
        elif change == "controller": legacy._powgen_controller = object()
        release.set(); await task
    asyncio.run(scenario())
    assert original_controller.settings is original_settings
    assert not legacy.server.state.powgen_monitoring


def test_reviewed_continuation_is_durable_before_same_watch_resumes(legacy):
    controller, run_hash = legacy._powgen_controller, lifecycle_sha256(legacy._powgen_controller)
    calls, uploads = uploader(legacy)
    open_review(legacy)
    asyncio.run(legacy.confirm_missing_live_settings())
    assert calls == []  # Review itself stays local and stopped.
    loops = []
    async def loop():
        assert len(uploads) == 1 and not controller.restore_settings_checkpoint_pending
        loops.append(controller)
    legacy._powgen_monitor_loop = loop
    async def scenario():
        await legacy.resume_existing_powgen_monitoring(); await asyncio.sleep(0)
    asyncio.run(scenario())
    assert loops == [controller] and legacy._powgen_controller is controller
    assert lifecycle_sha256(controller) == run_hash
    assert len(uploads) == 1 and uploads[0]["watch"]["continuation_review"]["reviewed"] is True
    assert uploads[0]["watch"]["settings"]["main_cif_dataset_id"] is None
    assert legacy.server.state.live_missing_saved_checkpoint_id == "saved-1"
    reopened = inspect_checkpoint_payload(uploads[0], legacy.service, "saved-1", document_sha256="b" * 64)
    assert not reopened.missing and not reopened.blocked
    assert reopened.controller.restore_settings_review == controller.restore_settings_review
    assert any(row["label"] == "Settings provenance" for row in reopened.rows)


def test_pending_checkpoint_ack_is_reused_and_no_resume_happens_early(legacy):
    calls, uploads = uploader(legacy, pending=True)
    open_review(legacy)
    asyncio.run(legacy.confirm_missing_live_settings())
    loops = []
    async def loop(): loops.append(True)
    legacy._powgen_monitor_loop = loop
    asyncio.run(legacy.resume_existing_powgen_monitoring())
    assert not loops and not legacy.server.state.powgen_monitoring
    assert legacy._powgen_controller.restore_settings_checkpoint_pending
    assert "No replacement upload" in legacy.server.state.error_message
    async def retry():
        await legacy.resume_existing_powgen_monitoring(); await asyncio.sleep(0)
    asyncio.run(retry())
    assert loops == [True] and len(uploads) == 1 and len(calls) == 2
    assert calls[0] == calls[1]


@pytest.mark.parametrize("change", ["stop", "clear", "history"])
def test_revoked_resume_can_save_receipt_but_never_starts_monitor(legacy, change):
    entered, release = threading.Event(), threading.Event()
    calls, uploads = uploader(legacy, entered=entered, release=release)
    open_review(legacy)
    asyncio.run(legacy.confirm_missing_live_settings())
    loops = []
    async def loop(): loops.append(True)
    async def no_save(): pass
    legacy._powgen_monitor_loop = loop
    legacy._stop_session_enrollment = no_save
    async def scenario():
        task = asyncio.create_task(legacy.resume_existing_powgen_monitoring())
        assert await asyncio.to_thread(entered.wait, 2)
        if change == "stop": legacy.stop_powgen_monitoring()
        elif change == "clear": await legacy.clear_workspace()
        elif change == "history": legacy.service.history_id = "another-history"
        release.set(); await task
    asyncio.run(scenario())
    assert not loops and not legacy.server.state.powgen_monitoring
    assert len(uploads) == 1


def test_settings_only_checkpoint_change_is_not_deduplicated_as_unchanged_runs():
    service = SavedHistory()
    controller, _ = checkpoint(service)
    fake_app = type("App", (), {"service": service})()
    calls, uploads = uploader(fake_app)
    first = controller.capture_checkpoint()
    controller.persist_state(first)
    controller.persist_state()
    assert len(uploads) == 1
    controller.settings = replace(controller.settings, max_active_jobs=1)
    second = controller.capture_checkpoint()
    assert first.state_json == second.state_json
    controller.persist_state(second)
    controller.persist_state(first)
    controller.persist_state()
    assert len(uploads) == 2
    assert uploads[0]["watch"]["settings"]["max_active_jobs"] == 2
    assert uploads[1]["watch"]["settings"]["max_active_jobs"] == 1


def test_receipt_corruption_or_known_setting_replacement_is_rejected(legacy):
    open_review(legacy)
    asyncio.run(legacy.confirm_missing_live_settings())
    settings, receipt = legacy._powgen_controller.settings, legacy._powgen_controller.restore_settings_review
    validate_continuation_receipt(receipt, settings)
    for mutate in (
        lambda r: r.update(resulting_settings_sha256="0" * 64),
        lambda r: r["known_settings"].update(configuration_dataset_id="another-config"),
        lambda r: r["confirmed_missing_settings"].update(max_active_jobs=4),
    ):
        changed = deepcopy(receipt); mutate(changed)
        with pytest.raises(ValueError): validate_continuation_receipt(changed, settings)


def test_runtime_mismatch_cannot_be_waived_by_missing_settings(legacy):
    legacy._powgen_controller.restore_resume_blocked_reasons += ("The saved source resolver differs",)
    legacy.review_missing_live_settings()
    assert not legacy.server.state.live_missing_review_open
    assert "resolver" in legacy.server.state.live_restore_message


def test_automatic_recovery_cannot_relabel_review_under_different_settings(legacy):
    open_review(legacy)
    asyncio.run(legacy.confirm_missing_live_settings())
    controller = legacy._powgen_controller
    payload = json.loads(controller.capture_checkpoint().payload_json)
    legacy.service.load_json_document = lambda _: deepcopy(payload)
    same = type(controller)(legacy.service, controller.settings)
    assert same.restore_latest_state()
    assert same.restore_settings_review == controller.restore_settings_review
    different = type(controller)(legacy.service, replace(controller.settings, max_active_jobs=1))
    with pytest.raises(ValueError, match="matching saved watch needs review"):
        different.restore_latest_state()
    assert different.restore_resume_blocked_reasons
    assert lifecycle_sha256(different) == lifecycle_sha256(controller)


@pytest.mark.parametrize("declaration", [
    {"resume_blocked_reasons": ["Independent profile qualification remains unresolved"]},
    {"missing_settings": ["max_active_jobs"]},
    {"resume_blocked_reasons": "malformed restriction"},
])
def test_automatic_restore_retains_blocked_watch_instead_of_fresh_fallback(legacy, declaration):
    open_review(legacy)
    asyncio.run(legacy.confirm_missing_live_settings())
    approved = legacy._powgen_controller
    payload = json.loads(approved.capture_checkpoint().payload_json)
    payload["watch"].update(declaration)
    legacy.service.load_json_document = lambda _: deepcopy(payload)
    restored = type(approved)(legacy.service, approved.settings)
    with pytest.raises(ValueError, match="Restore a saved Live watch"):
        restored.restore_latest_state()
    assert restored.restore_resume_blocked_reasons and restored.restore_requires_explicit_inspection
    assert lifecycle_sha256(restored) == lifecycle_sha256(approved)
    assert restored.due_submissions() == []
    with pytest.raises(ValueError, match="blocked from new analyses"):
        restored._submit_live_snapshot(next(iter(restored.state.discovered.values())), object())


def test_start_and_resume_cannot_fall_through_a_matching_blocked_checkpoint(legacy):
    open_review(legacy)
    asyncio.run(legacy.confirm_missing_live_settings())
    approved = legacy._powgen_controller
    payload = json.loads(approved.capture_checkpoint().payload_json)
    payload["watch"]["resume_blocked_reasons"] = ["Independent source restriction"]
    legacy.service.load_json_document = lambda _: deepcopy(payload)
    legacy._powgen_controller = None
    legacy._powgen_settings_signature = None
    legacy.server.state.live_restored_checkpoint_id = ""
    legacy._powgen_sample_index = lambda _: None
    async def settings():
        return approved.settings, legacy.service.config.model_copy(deep=True)
    async def forbidden():
        raise AssertionError("A blocked checkpoint cannot start a monitor loop")
    legacy._prepare_live_settings = settings
    legacy._powgen_monitor_loop = forbidden
    asyncio.run(legacy.start_powgen_monitoring())
    assert not legacy.server.state.powgen_monitoring
    assert "Independent source restriction" in legacy.server.state.error_message
    assert lifecycle_sha256(legacy._powgen_controller) == lifecycle_sha256(approved)
    asyncio.run(legacy.start_powgen_monitoring())
    asyncio.run(legacy.resume_existing_powgen_monitoring())
    assert not legacy.server.state.powgen_monitoring and legacy._powgen_monitor_task is None


def test_invalid_run_map_in_a_matching_restricted_checkpoint_does_not_fall_back(legacy):
    open_review(legacy)
    asyncio.run(legacy.confirm_missing_live_settings())
    approved = legacy._powgen_controller
    payload = json.loads(approved.capture_checkpoint().payload_json)
    payload["watch"]["resume_blocked_reasons"] = ["Independent source restriction"]
    payload["completed"] = {"wrong-key": next(iter(payload["completed"].values()))}
    legacy.service.load_json_document = lambda _: deepcopy(payload)
    restored = type(approved)(legacy.service, approved.settings)
    with pytest.raises(ValueError, match="before any new analysis"):
        restored.restore_latest_state()
    assert restored.restore_resume_blocked_reasons
    assert restored.due_submissions() == []
