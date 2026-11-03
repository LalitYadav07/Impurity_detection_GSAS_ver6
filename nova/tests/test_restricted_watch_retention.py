"""Rejected automatic restores retain only validated, currently owned state."""
import asyncio
from copy import deepcopy
import threading

import pytest

from test_live_restore import app, checkpoint
from radar_pd_nova.live_settings_review import lifecycle_sha256


@pytest.fixture
def restricted_start(app):
    original, payload = checkpoint(app.service)
    payload['watch']['resume_blocked_reasons'] = ['Independent source restriction']
    app.service.load_json_document = lambda _: deepcopy(payload)
    app._powgen_sample_index = lambda _: None

    async def settings():
        return original.settings, app.service.config.model_copy(deep=True)

    async def forbidden():
        raise AssertionError('A rejected checkpoint must not start a monitor')

    def forbidden_refresh(*_, **__):
        raise AssertionError('A rejected checkpoint must not refresh science')

    app._prepare_live_settings = settings
    app._powgen_monitor_loop = forbidden
    app._refinement_maybe_refresh = forbidden_refresh
    app._refresh_post_analysis = forbidden_refresh
    return app, original, payload


def test_matching_restricted_watch_retains_read_only_lifecycle(restricted_start):
    app, original, _ = restricted_start
    before = lifecycle_sha256(original)
    asyncio.run(app.start_powgen_monitoring())
    retained = app._powgen_controller
    assert retained is not None and retained is not original
    assert lifecycle_sha256(retained) == before
    assert retained.state_dataset_ids == ['checkpoint-1']
    assert retained.restore_requires_explicit_inspection
    assert app._powgen_controller_configuration is None
    assert app.server.state.live_restored_checkpoint_id == 'checkpoint-1'
    assert app.server.state.live_restore_resume_blocked
    assert app.server.state.powgen_next_check == 'Stopped'
    assert len(app.server.state.powgen_rows) == 4
    assert 'Independent source restriction' in app.server.state.error_message
    assert not app.server.state.powgen_monitoring and app._powgen_monitor_task is None
    asyncio.run(app.start_powgen_monitoring())
    assert 'Restore a saved Live watch' in app.server.state.error_message
    asyncio.run(app.resume_existing_powgen_monitoring())
    assert 'read-only' in app.server.state.error_message
    assert app._powgen_controller is retained and lifecycle_sha256(retained) == before
    assert not app.server.state.powgen_monitoring and app._powgen_monitor_task is None
    assert retained.due_submissions() == []


def test_malformed_restricted_checkpoint_is_not_published(restricted_start):
    app, _, payload = restricted_start
    payload['completed'] = {'wrong-key': next(iter(payload['completed'].values()))}
    asyncio.run(app.start_powgen_monitoring())
    assert app._powgen_controller is None
    assert not app.server.state.live_restored_checkpoint_id
    assert 'could not be validated' in app.server.state.error_message or 'invalid checkpoint' in app.server.state.error_message
    assert not app.server.state.powgen_monitoring and app._powgen_monitor_task is None


def test_ordinary_restore_failure_is_not_published(restricted_start, monkeypatch):
    app, original, _ = restricted_start

    def unavailable(controller):
        raise OSError('Checkpoint unavailable')

    monkeypatch.setattr(type(original), 'restore_latest_state', unavailable)
    asyncio.run(app.start_powgen_monitoring())
    assert app._powgen_controller is None
    assert not app.server.state.live_restored_checkpoint_id
    assert app.server.state.error_message == 'Checkpoint unavailable'
    assert not app.server.state.powgen_monitoring and app._powgen_monitor_task is None


@pytest.mark.parametrize('change', ['stop', 'clear', 'end', 'history', 'origin', 'actor', 'draft', 'context', 'cancel'])
def test_revoked_launch_cannot_publish_restricted_state(restricted_start, monkeypatch, change):
    app, original, _ = restricted_start
    entered, release = threading.Event(), threading.Event()
    restore = type(original).restore_latest_state

    def blocked(controller):
        entered.set()
        assert release.wait(5)
        return restore(controller)

    monkeypatch.setattr(type(original), 'restore_latest_state', blocked)

    async def run():
        task = asyncio.create_task(app.start_powgen_monitoring())
        assert await asyncio.to_thread(entered.wait, 5)
        if change == 'stop':
            app.stop_powgen_monitoring()
        elif change == 'clear':
            await app.clear_workspace()
        elif change == 'end':
            await app.end_session()
        elif change == 'history':
            app.service.history_id = 'replacement-history'
        elif change == 'origin':
            app.service.galaxy_url = 'https://replacement.invalid'
        elif change == 'actor':
            app.service.galaxy_key = 'replacement-actor'
        elif change == 'draft':
            app.server.state.live_sample_elements = 'Fe'
            app.server.state.flush()
        elif change == 'context':
            app._result_context = object()
        else:
            task.cancel()
        release.set()
        if change == 'cancel':
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task

    asyncio.run(run())
    assert app._powgen_controller is None
    assert not app.server.state.live_restored_checkpoint_id
    assert not app.server.state.powgen_monitoring and app._powgen_monitor_task is None
