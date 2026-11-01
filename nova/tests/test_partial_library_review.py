"""Incomplete candidate sets require a current, explicit user decision."""
import asyncio
from copy import deepcopy
import json
import threading
from types import SimpleNamespace
import zipfile

import pytest

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.library_review import (library_build_outcome, library_review_signature,
    partial_library_archive_review, library_failure_is_duplicate)
from radar_pd_nova.library_setup import LIBRARY_FIELDS, LibraryDraft
from radar_pd_nova.models import RunStatus, UtilityActionRecord
from radar_pd_nova.uploads import build_cif_source_archive


CIF = """data_oxide
_chemical_formula_sum 'Al2 O3'
_cell_length_a 4.7
_cell_length_b 4.7
_cell_length_c 13
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
"""


def setup_case(tmp_path, context="live", *, failure=True):
    app = RadarPdNovaApp()
    draft = LibraryDraft(app.server.state, context)
    draft.database_source = "archive"
    draft.library_archive_source = "galaxy"
    draft.history_database_id = "prior-library"
    draft.flush()
    app._library_review_changed()  # Local test state has not entered server.ready().
    path = tmp_path / "candidate.cif"
    path.write_text(CIF)
    draft.database_source = "custom_mini"
    draft.library_builder_local_paths = [str(path)]
    draft.library_builder_name = "selected-phases"
    draft.library_builder_generation = "generation-1"
    draft.flush()
    action = UtilityActionRecord(uid="build-1", tool_id="neutrons_radar_pd_library_builder_prototype",
        name="Candidate library", galaxy_job_id="job-1", status=RunStatus.OK,
        outputs={"library_archive": "partial-library", "library_manifest": "manifest-1"})
    manifest = {"phase_count": 7, "n_phases": 7,
                "phase_ids": [f"phase-{i}" for i in range(7)],
                "failures": [{"source_name": "main-phase.cif", "error": "Invalid site occupancy"}] if failure else []}
    app._library_action_contexts = {action.uid: {"workflow": context,
        "signature": draft.signature(), "source_signature": library_review_signature(draft),
        "generation": "generation-1", "preflight": {"requested_count": 8 if failure else 7,
            "skipped_count": 0, "failure_rows": []}}}
    remembered = []
    app._remember_history_dataset = lambda *args: remembered.append(args)
    app._sync_activities = lambda: None
    app._use_live_element_suggestion = lambda **_: None
    app.refresh_history = lambda **_: None
    app.service = SimpleNamespace(_dataset_document=lambda _: deepcopy(manifest))
    return app, draft, action, manifest, path, remembered


@pytest.mark.parametrize("context", ["single", "live"])
def test_partial_keeps_prior_library_until_explicit_current_acceptance(tmp_path, context):
    app, draft, action, _, _, remembered = setup_case(tmp_path, context)
    asyncio.run(app._utility_completed(action))
    assert draft.database_source == "archive" and draft.history_database_id == "prior-library"
    assert draft.library_builder_built_count == 7 and draft.library_builder_requested_count == 8
    assert draft.library_builder_skipped_count == 1 and draft.library_builder_duplicate_count == 0
    assert draft.library_builder_failure_rows == [{"name": "main-phase.cif", "reason": "Invalid site occupancy"}]
    assert "Previous library selection kept" in draft.library_builder_message
    token = draft.library_builder_partial_review["token"]
    assert remembered[0][0] == "partial-library"
    app.use_partial_candidate_library(context, token)
    assert draft.history_database_id == "partial-library" and draft.library_builder_status == "partial_selected"
    assert not draft.library_builder_partial_review
    other = LibraryDraft(app.server.state, "single" if context == "live" else "live")
    assert other.history_database_id != "partial-library"
    # A repeated event cannot accept any newer review.
    draft.history_database_id = "later-choice"
    app.use_partial_candidate_library(context, token)
    assert draft.history_database_id == "later-choice"


def test_keep_previous_library_declines_only_the_pending_result(tmp_path):
    app, draft, action, _, _, _ = setup_case(tmp_path)
    asyncio.run(app._utility_completed(action))
    token = draft.library_builder_partial_review["token"]
    app.retain_current_candidate_library("live", token)
    assert draft.history_database_id == "prior-library" and draft.database_source == "archive"
    assert draft.library_builder_status == "partial_kept" and not draft.library_builder_partial_review
    assert draft.library_builder_failure_rows[0]["name"] == "main-phase.cif"


def test_incomplete_archive_choice_does_not_erase_last_usable_library(tmp_path):
    app, draft, action, _, _, _ = setup_case(tmp_path)
    draft.database_source = "archive"; draft.library_archive_source = "computer"
    draft.database_archive_path = ""; app._library_review_changed()
    assert draft.library_builder_previous_library["history_database_id"] == "prior-library"
    assert draft.library_builder_previous_library["library_archive_source"] == "galaxy"


def test_all_success_still_auto_selects(tmp_path):
    app, draft, action, _, _, _ = setup_case(tmp_path, failure=False)
    asyncio.run(app._utility_completed(action))
    assert draft.history_database_id == "partial-library" and draft.library_builder_status == "ready"
    assert not draft.library_builder_partial_review


def test_source_preflight_reports_failures_separately_from_exact_duplicates(tmp_path):
    first = tmp_path / "first.cif"; first.write_text(CIF)
    more = tmp_path / "more.zip"
    with zipfile.ZipFile(more, "w") as archive:
        archive.writestr("same.cif", CIF)
        archive.writestr("bad.cif", "data_bad\n")
    _, stats = build_cif_source_archive([str(first), str(more)], "candidates")
    outcome = library_build_outcome({}, stats["cif_count"], {
        "requested_count": stats["source_count"], "skipped_count": stats["skipped_count"],
        "failure_rows": stats["failures"]})
    assert stats["source_count"] == 3 and stats["duplicate_count"] == 1
    assert outcome["built_count"] == 1 and outcome["requested_count"] == 3
    assert outcome["skipped_count"] == 1 and outcome["duplicate_count"] == 1
    assert outcome["duplicate_rows"] == [{"name": "same.cif", "reason": "duplicate structure file"}]
    assert outcome["failure_rows"][0]["name"] == "bad.cif"


@pytest.mark.parametrize("reason", ["duplicate structure file", "phase id already exists in base catalog",
    "phase already exists in base database as known-phase"])
def test_explicit_deduplication_does_not_force_partial_acceptance(tmp_path, reason):
    app, draft, action, manifest, _, _ = setup_case(tmp_path, failure=False)
    manifest["failures"] = [{"source_name": "duplicate.cif", "error": reason}]
    app._library_action_contexts[action.uid]["preflight"]["requested_count"] = 8
    asyncio.run(app._utility_completed(action))
    assert draft.history_database_id == "partial-library"
    assert draft.library_builder_skipped_count == 0 and draft.library_builder_duplicate_count == 1
    assert not draft.library_builder_partial_review


@pytest.mark.parametrize("change", ["name", "radiation", "cif_ids", "paths", "library", "file_bytes"])
def test_stale_partial_review_cannot_replace_a_changed_source(tmp_path, change):
    app, draft, action, _, path, _ = setup_case(tmp_path)
    asyncio.run(app._utility_completed(action))
    token = draft.library_builder_partial_review["token"]
    if change == "name": draft.library_builder_name = "new-name"
    elif change == "radiation": draft.radiation = "xray"
    elif change == "cif_ids": draft.library_builder_cif_ids = ["another-cif"]
    elif change == "paths": draft.library_builder_local_paths = [str(tmp_path / "new.cif")]
    elif change == "library": draft.history_database_id = "chosen-after-review"
    else: path.write_text(CIF + "# modified source\n")
    draft.flush()
    app.use_partial_candidate_library("live", token)
    assert draft.history_database_id != "partial-library" and not draft.library_builder_partial_review


def test_new_build_rejects_delayed_click_from_prior_review(tmp_path):
    app, draft, action, _, _, _ = setup_case(tmp_path)
    asyncio.run(app._utility_completed(action))
    token = draft.library_builder_partial_review["token"]
    pending = []
    app._schedule_utility = lambda coroutine, _: pending.append(coroutine)
    app.build_candidate_library("live")
    assert draft.library_builder_generation != "generation-1" and not draft.library_builder_partial_review
    app.use_partial_candidate_library("live", token)
    assert draft.history_database_id == "prior-library"
    for coroutine in pending: coroutine.close()


@pytest.mark.parametrize("change", ["generation", "preflight"])
def test_delayed_completion_cannot_overwrite_new_generation(tmp_path, change):
    app, draft, action, manifest, _, remembered = setup_case(tmp_path)
    entered, release = threading.Event(), threading.Event()
    def read(_):
        entered.set(); assert release.wait(5)
        return manifest
    app.service._dataset_document = read
    async def scenario():
        task = asyncio.create_task(app._utility_completed(action))
        assert await asyncio.to_thread(entered.wait, 2)
        if change == "generation":
            draft.library_builder_generation = "new-generation"
        else:
            app._library_action_contexts[action.uid]["preflight"]["requested_count"] = 9
        draft.library_builder_status = "building"
        draft.library_builder_message = "New build owns this draft"
        release.set(); await task
    asyncio.run(scenario())
    assert draft.library_builder_status == "building" and draft.library_builder_message == "New build owns this draft"
    assert not draft.library_builder_partial_review and remembered


def test_cancelled_manifest_wait_never_offers_or_activates_partial(tmp_path):
    app, draft, action, manifest, _, _ = setup_case(tmp_path)
    entered, release = threading.Event(), threading.Event()
    def read(_):
        entered.set(); assert release.wait(5)
        return manifest
    app.service._dataset_document = read
    async def scenario():
        task = asyncio.create_task(app._utility_completed(action))
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        release.set()
    asyncio.run(scenario())
    assert draft.history_database_id == "prior-library" and not draft.library_builder_partial_review
    assert draft.library_builder_status == "interrupted"


def test_pending_review_round_trips_as_data_without_rebuilding(tmp_path):
    app, draft, action, _, _, _ = setup_case(tmp_path)
    asyncio.run(app._utility_completed(action))
    saved = json.loads(json.dumps({name: getattr(draft, name) for name in LIBRARY_FIELDS}))
    token = saved["library_builder_partial_review"]["token"]
    restored = RadarPdNovaApp()
    target = LibraryDraft(restored.server.state, "live")
    for name, value in saved.items(): setattr(target, name, value)
    target.history_database_id = "prior-library"; target.flush()
    restored._use_live_element_suggestion = lambda **_: None
    restored.use_partial_candidate_library("live", token)
    assert target.history_database_id == "partial-library" and target.library_builder_status == "partial_selected"


def test_replayed_completion_preserves_pending_review_token(tmp_path):
    app, draft, action, _, _, _ = setup_case(tmp_path, "single")
    asyncio.run(app._utility_completed(action))
    review = deepcopy(draft.library_builder_partial_review)
    status = draft.library_builder_status
    message = draft.library_builder_message
    asyncio.run(app._utility_completed(action))
    assert draft.library_builder_partial_review == review
    assert draft.library_builder_status == status and draft.library_builder_message == message
    assert draft.history_database_id == "prior-library"


def test_unowned_recovered_partial_result_is_not_auto_selected(tmp_path):
    app, draft, action, _, _, remembered = setup_case(tmp_path, "single")
    app._library_action_contexts.clear(); draft.library_builder_generation = ""
    before = draft.signature(), draft.library_builder_status, draft.library_builder_message
    asyncio.run(app._utility_completed(action))
    assert draft.history_database_id == "prior-library"
    assert not draft.library_builder_partial_review
    assert before == (draft.signature(), draft.library_builder_status, draft.library_builder_message)
    assert remembered and remembered[0][0] == "partial-library"


def test_unreported_selected_cif_is_an_explicit_incomplete_result():
    outcome = library_build_outcome({}, 7, {"requested_count": 8})
    assert outcome["skipped_count"] == 1 and outcome["duplicate_count"] == 0
    assert outcome["failure_rows"][0]["name"] == "Unreported source"


def test_all_failure_details_and_review_controls_are_visible():
    html = RadarPdNovaApp().layout.html
    assert "Use partial library" in html and "Keep current library" in html
    assert "library_builder_failure_rows.slice(0, 8)" not in html
    assert "library_builder_requested_count" in html


def test_existing_archive_review_uses_the_same_declared_failure_evidence():
    manifest = {"n_phases": 7, "phase_ids": [str(i) for i in range(7)],
        "failures": [{"source_name": "main.cif", "error": "Invalid occupancy"}]}
    review = partial_library_archive_review(manifest)
    assert review["requires_review"] and review["count_verified"]
    assert review["built_count"] == 7 and review["requested_count"] == 8
    assert review["failure_rows"] == [{"name": "main.cif", "reason": "Invalid occupancy"}]
    assert "earlier upload/preflight omissions" in review["count_scope"]
    manifest["failures"] = [{"source_name": "same.cif", "error": "phase already exists in base database as X"}]
    review = partial_library_archive_review(manifest)
    assert not review["requires_review"] and review["duplicate_count"] == 1


@pytest.mark.parametrize("reason", ["duplicate occupancy is invalid", "near duplicate failed to simulate",
    "phase already exists in base database as ", "Could not build phase"])
def test_unrecognized_failures_are_never_inferred_to_be_deduplication(reason):
    assert not library_failure_is_duplicate({"error": reason})


def test_clear_workspace_invalidates_pending_review_without_activating_it(tmp_path):
    app, draft, action, _, _, _ = setup_case(tmp_path)
    asyncio.run(app._utility_completed(action))
    token = draft.library_builder_partial_review["token"]
    async def stopped(): pass
    app._stop_session_enrollment = stopped
    asyncio.run(app.clear_workspace())
    app.use_partial_candidate_library("live", token)
    assert not draft.library_builder_partial_review and not draft.library_builder_generation
    assert not draft.history_database_id and draft.library_builder_status == "idle"


def test_frozen_preflight_not_mutable_display_state_drives_result(tmp_path):
    app, draft, action, manifest, _, _ = setup_case(tmp_path, failure=False)
    owner = app._library_action_contexts[action.uid]
    owner["preflight"] = {"requested_count": 8, "skipped_count": 1,
        "failure_rows": [{"name": "unreadable.cif", "reason": "Bad CIF"}]}
    draft.library_builder_failure_rows = [{"name": "other.cif", "reason": "other build"}]
    draft.library_builder_skipped_count = 9
    asyncio.run(app._utility_completed(action))
    assert draft.library_builder_requested_count == 8 and draft.library_builder_skipped_count == 1
    assert draft.library_builder_failure_rows == [{"name": "unreadable.cif", "reason": "Bad CIF"}]
    assert draft.history_database_id == "prior-library"


def test_failure_detail_rows_are_not_silently_truncated(tmp_path):
    app, draft, action, manifest, _, _ = setup_case(tmp_path)
    manifest["failures"] = [{"source_name": f"bad-{i}.cif", "error": f"reason-{i}"} for i in range(12)]
    app._library_action_contexts[action.uid]["preflight"]["requested_count"] = 19
    asyncio.run(app._utility_completed(action))
    assert len(draft.library_builder_failure_rows) == 12
    assert draft.library_builder_failure_rows[-1] == {"name": "bad-11.cif", "reason": "reason-11"}


def test_concurrent_workflow_completions_keep_separate_reviews(tmp_path):
    app, live, live_action, manifest, _, _ = setup_case(tmp_path)
    single = LibraryDraft(app.server.state, "single")
    single.database_source = "archive"; single.library_archive_source = "galaxy"
    single.history_database_id = "single-prior"; app._library_review_changed()
    single.database_source = "custom_mini"; single.library_builder_name = "single-set"
    single.library_builder_generation = "single-generation"
    single_action = live_action.model_copy(update={"uid": "single-build", "galaxy_job_id": "single-job",
        "outputs": {"library_archive": "single-partial", "library_manifest": "single-manifest"}})
    app._library_action_contexts[single_action.uid] = {"workflow": "single", "signature": single.signature(),
        "source_signature": library_review_signature(single), "generation": single.library_builder_generation,
        "preflight": {"requested_count": 8, "skipped_count": 0, "failure_rows": []}}
    both = threading.Barrier(2)
    def read(_): both.wait(timeout=5); return deepcopy(manifest)
    app.service._dataset_document = read
    async def finish(): await asyncio.gather(app._utility_completed(live_action), app._utility_completed(single_action))
    asyncio.run(finish())
    assert live.history_database_id == "prior-library" and single.history_database_id == "single-prior"
    live_token = live.library_builder_partial_review["token"]
    single_token = single.library_builder_partial_review["token"]
    assert live_token != single_token
    app.use_partial_candidate_library("single", single_token)
    assert single.history_database_id == "single-partial" and live.history_database_id == "prior-library"


@pytest.mark.parametrize("cancel", [True, False])
def test_cancelled_or_superseded_source_preparation_never_dispatches(tmp_path, monkeypatch, cancel):
    from radar_pd_nova import app as app_module
    app, draft, _, _, _, _ = setup_case(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = app_module.build_cif_source_archive
    def prepare(*args):
        entered.set(); assert release.wait(5)
        return original(*args)
    monkeypatch.setattr(app_module, "build_cif_source_archive", prepare)
    pending = []
    app._schedule_utility = lambda coroutine, _: pending.append(coroutine)
    def unexpected(*args, **kwargs): raise AssertionError("stale preparation uploaded or dispatched")
    app.service._upload_one = unexpected
    app._submit_utility_action = unexpected
    app.build_candidate_library("live")
    async def scenario():
        task = asyncio.create_task(pending[0])
        assert await asyncio.to_thread(entered.wait, 2)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError): await task
        else:
            draft.library_builder_generation = "new-owner"
            draft.library_builder_message = "New preparation"
        release.set()
        if not cancel: await task
    asyncio.run(scenario())
    assert draft.history_database_id == "prior-library"
    if cancel:
        assert not draft.library_builder_active and draft.library_builder_status == "interrupted"
    else:
        assert draft.library_builder_message == "New preparation"
