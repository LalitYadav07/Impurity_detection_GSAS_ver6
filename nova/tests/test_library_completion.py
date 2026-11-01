"""Completed library outputs must be verified without rebuilding the library."""

import asyncio
from types import SimpleNamespace

import pytest

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.library_setup import LibraryDraft, read_library_manifest
from radar_pd_nova.models import RunStatus, UtilityActionRecord


MANIFEST = {
    "$schema": "radar-pd-library/v1",
    "phase_count": 4,
    "n_phases": 4,
    "phase_ids": ["lno", "graphite", "cu", "al"],
    "failures": [],
}


@pytest.fixture
def completion(monkeypatch):
    app = RadarPdNovaApp()
    state = app.server.state
    state.live_database_source = "custom_mini"
    state.live_library_builder_name = "LNO_procedural"
    state.live_library_builder_local_paths = ["source.cif"]
    state.live_library_builder_skipped_count = 0
    state.live_library_builder_failure_rows = []
    draft = LibraryDraft(state, "live")
    action = UtilityActionRecord(
        uid="library-build",
        tool_id="neutrons_radar_pd_library_builder_prototype",
        name="Build LNO",
        galaxy_job_id="completed-job",
        status=RunStatus.OK,
        outputs={"library_archive": "library", "library_manifest": "manifest"},
    )
    app._library_action_contexts = {
        action.uid: {"workflow": "live", "signature": draft.signature()}
    }
    app._sync_activities = lambda: None
    app._remember_history_dataset = lambda *_: None
    app._use_live_element_suggestion = lambda **_: None
    app.refresh_history = lambda **_: None
    replies = [MANIFEST]
    reads = []

    def read(identifier):
        reads.append(identifier)
        result = replies.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    app.service = SimpleNamespace(_dataset_document=read)
    return app, state, action, replies, reads


def test_completed_library_rechecks_transient_manifest_read_without_rebuild(completion):
    app, state, action, replies, reads = completion
    replies[:] = [TimeoutError("temporary publication read timeout"), MANIFEST]

    asyncio.run(app._utility_completed(action))

    assert reads == ["manifest", "manifest"]
    assert state.live_database_source == "archive"
    assert state.powgen_database_dataset_id == "library"
    assert state.live_library_builder_built_count == 4
    assert state.live_library_builder_status == "ready"
    assert "4 usable phase" in state.live_library_builder_message


def test_empty_publication_is_rechecked_but_never_counted_as_zero(completion):
    app, state, action, replies, reads = completion
    replies[:] = [None, MANIFEST]

    asyncio.run(app._utility_completed(action))

    assert reads == ["manifest", "manifest"]
    assert state.powgen_database_dataset_id == "library"
    assert state.live_library_builder_built_count == 4


def test_failed_manifest_read_is_bounded_and_reports_retrieval_failure(completion):
    app, state, action, replies, reads = completion
    replies[:] = [TimeoutError("first"), TimeoutError("second")]

    asyncio.run(app._utility_completed(action))

    assert reads == ["manifest", "manifest"]
    assert state.live_database_source == "custom_mini"
    assert not state.powgen_database_dataset_id
    assert state.live_library_builder_status == "error"
    assert "manifest" in state.live_library_builder_message.lower()
    assert "TimeoutError" in state.live_library_builder_message
    assert "not selected" in state.live_library_builder_message
    assert "0 usable phase" not in state.live_library_builder_message


@pytest.mark.parametrize(
    "manifest",
    [
        {"phase_count": "four"},
        {"phase_count": True},
        {"phase_count": 1.5},
        {"phase_count": -1},
        {"phase_count": 4, "n_phases": 3},
        {"phase_count": 4, "phase_ids": ["same"] * 4},
        {"phase_count": 4, "phase_ids": ["only-one"]},
        {"phase_count": 0, "n_phases": 4},
        {"kind": "mini"},
    ],
)
def test_invalid_count_evidence_is_not_selected_or_retried(completion, manifest):
    app, state, action, replies, reads = completion
    replies[:] = [manifest]

    asyncio.run(app._utility_completed(action))

    assert reads == ["manifest"]
    assert state.live_database_source == "custom_mini"
    assert not state.powgen_database_dataset_id
    assert state.live_library_builder_status == "error"


def test_valid_zero_phase_manifest_is_not_retried_or_selected(completion):
    app, state, action, replies, reads = completion
    replies[:] = [{"phase_count": 0, "n_phases": 0, "phase_ids": [], "failures": []}]

    asyncio.run(app._utility_completed(action))

    assert reads == ["manifest"]
    assert not state.powgen_database_dataset_id
    assert state.live_library_builder_status == "error"
    assert "no usable phases" in state.live_library_builder_message.lower()


def test_selection_change_during_manifest_read_is_preserved(completion):
    app, state, action, _, reads = completion

    def read(identifier):
        reads.append(identifier)
        state.live_library_builder_name = "a different choice"
        return MANIFEST

    app.service._dataset_document = read
    asyncio.run(app._utility_completed(action))

    assert not state.powgen_database_dataset_id
    assert state.live_database_source == "custom_mini"
    assert state.live_library_builder_built_count == 4
    assert "changed selection was kept" in state.live_library_builder_message


def test_legacy_count_only_manifest_still_selects_completed_library(completion):
    app, state, action, replies, _ = completion
    replies[:] = [{"n_phases": 2, "failures": []}]

    asyncio.run(app._utility_completed(action))

    assert state.powgen_database_dataset_id == "library"
    assert state.live_library_builder_built_count == 2


def test_missing_manifest_output_is_not_replaced_by_a_guessed_count(completion):
    app, state, action, _, reads = completion
    action.outputs.pop("library_manifest")

    asyncio.run(app._utility_completed(action))

    assert not reads
    assert not state.powgen_database_dataset_id
    assert state.live_library_builder_status == "error"
    assert "no manifest output" in state.live_library_builder_message


def test_cancelled_manifest_verification_does_not_read_again():
    reads = []

    async def scenario():
        retrying = asyncio.Event()

        def reader(identifier):
            reads.append(identifier)
            return None

        task = asyncio.create_task(
            read_library_manifest(reader, "manifest", on_retry=lambda _: retrying.set())
        )
        await retrying.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert reads == ["manifest"]


@pytest.mark.parametrize("replace_owner", [False, True])
def test_lost_completion_owner_cannot_overwrite_new_draft(completion, replace_owner):
    app, state, action, _, reads = completion

    def read(identifier):
        reads.append(identifier)
        owner = app._library_action_contexts.pop(action.uid)
        if replace_owner:
            app._library_action_contexts[action.uid] = dict(owner)
        state.live_library_builder_status = "idle"
        state.live_library_builder_message = "New workspace draft"
        return MANIFEST

    app.service._dataset_document = read
    asyncio.run(app._utility_completed(action))

    assert reads == ["manifest"]
    assert not state.powgen_database_dataset_id
    assert state.live_database_source == "custom_mini"
    assert state.live_library_builder_status == "idle"
    assert state.live_library_builder_message == "New workspace draft"
