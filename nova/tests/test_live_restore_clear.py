"""Clear removes a saved-watch selection only after its recovery save succeeds."""

import asyncio
from copy import deepcopy
import threading

from test_live_restore import app, checkpoint, inspect, restore


def _restored(app):
    checkpoint(app.service, partial=False)
    asyncio.run(inspect(app))
    restore(app)
    state = app.server.state
    state.live_restore_search = "1234"
    state.live_restore_page = 2
    state.live_restore_has_more = True
    state.flush()
    return app._powgen_controller


def _catalog(state):
    return deepcopy((state.live_restore_options, state.live_restore_search,
                     state.live_restore_page, state.live_restore_has_more))


def _assert_cleared_selection(app):
    state = app.server.state
    assert not state.live_restore_dataset_id
    assert state.live_restore_message == (
        "Choose a saved Live checkpoint in this History to review it before restoring."
    )
    assert not state.live_restored_checkpoint_id
    assert not state.live_restore_resume_blocked
    assert not state.live_restore_ready
    assert not state.live_restore_busy
    assert not state.live_restore_confirm
    assert not state.live_restore_rows
    assert app._live_restore_owner is None
    assert app._live_restore_review is None
    assert app._live_restore_confirm_signature is None


def test_successful_clear_forgets_restoration_selection_but_keeps_catalog(app):
    _restored(app)
    state = app.server.state
    catalog = _catalog(state)
    service_calls = deepcopy(app.service.reads)
    assert state.live_restore_dataset_id == "checkpoint-1"
    assert "Restored PG3" in state.live_restore_message

    asyncio.run(app.clear_workspace())

    assert app._powgen_controller is None and not app.records
    _assert_cleared_selection(app)
    assert _catalog(state) == catalog
    assert app.service.reads == service_calls


def test_failed_clear_save_keeps_restored_watch_selection_and_notice(app):
    controller = _restored(app)
    state = app.server.state
    catalog = _catalog(state)
    message = state.live_restore_message
    records = dict(app.records)

    async def failed_save():
        raise RuntimeError("checkpoint temporarily unavailable")

    app._stop_session_enrollment = failed_save
    asyncio.run(app.clear_workspace())

    assert app._powgen_controller is controller
    assert app.records == records
    assert state.live_restore_dataset_id == "checkpoint-1"
    assert state.live_restored_checkpoint_id == "checkpoint-1"
    assert state.live_restore_message == message
    assert _catalog(state) == catalog
    assert "was not cleared" in state.workspace_notice


def test_late_inspection_cannot_restore_selection_or_notice_after_clear(app, monkeypatch):
    import radar_pd_nova.live_restore as live_restore

    checkpoint(app.service, partial=False)
    original = live_restore.read_saved_watch
    entered, release = threading.Event(), threading.Event()

    def delayed_read(*args, **kwargs):
        entered.set()
        assert release.wait(3), "inspection release timed out"
        return original(*args, **kwargs)

    monkeypatch.setattr(live_restore, "read_saved_watch", delayed_read)

    async def scenario():
        await app.list_saved_live_watches()
        state = app.server.state
        state.live_restore_dataset_id = "checkpoint-1"
        state.flush()
        catalog = _catalog(state)
        pending = asyncio.create_task(app.inspect_saved_live_watch())
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            assert state.live_restore_busy
            await app.clear_workspace()
            _assert_cleared_selection(app)
        finally:
            release.set()
            await asyncio.wait_for(pending, 3)
        _assert_cleared_selection(app)
        assert _catalog(state) == catalog
        assert app._powgen_controller is None and not app.records

    asyncio.run(scenario())
