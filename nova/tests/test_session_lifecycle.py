import asyncio
from types import SimpleNamespace

import pytest

from radar_pd_nova import session_lifecycle as lifecycle
from radar_pd_nova.app import RadarPdNovaApp


def eligible(monkeypatch):
    monkeypatch.setenv(lifecycle.MARKER, '1')
    monkeypatch.setattr(lifecycle, 'CONFIG', SimpleNamespace(is_file=lambda: True))
    monkeypatch.setattr(lifecycle, 'SOCKET', SimpleNamespace(is_socket=lambda: True))
    monkeypatch.setattr(lifecycle.shutil, 'which', lambda name: '/usr/local/bin/supervisorctl')


def test_local_sessions_do_not_touch_a_supervisor(monkeypatch):
    monkeypatch.delenv(lifecycle.MARKER, raising=False)
    monkeypatch.setattr(lifecycle.subprocess, 'Popen', lambda *a, **k: pytest.fail('Unexpected container shutdown'))
    assert lifecycle.schedule_supervised_shutdown() is False


def test_shutdown_helper_is_detached_and_does_not_inherit_credentials(monkeypatch):
    eligible(monkeypatch)
    monkeypatch.setenv('GALAXY_API_KEY', 'test-secret')
    monkeypatch.setenv('AZURE_ILLUMINE_API_KEY', 'another-test-secret')
    calls = []
    monkeypatch.setattr(lifecycle.subprocess, 'Popen', lambda command, **kwargs: calls.append((command, kwargs)))
    assert lifecycle.schedule_supervised_shutdown() is True
    command, options = calls[0]
    assert command[-1] == '/usr/local/bin/supervisorctl'
    assert options['start_new_session'] and options['close_fds']
    assert options['env'] == {lifecycle.MARKER: '1'}
    assert all('secret' not in arg for arg in command)


def test_marked_session_with_missing_supervisor_fails_closed(monkeypatch):
    eligible(monkeypatch)
    monkeypatch.setattr(lifecycle, 'SOCKET', SimpleNamespace(is_socket=lambda: False))
    monkeypatch.setattr(lifecycle.subprocess, 'Popen', lambda *a, **k: pytest.fail('Missing supervisor must not be guessed'))
    with pytest.raises(RuntimeError, match='remains open'):
        lifecycle.schedule_supervised_shutdown()


def test_end_schedules_shutdown_after_checkpoint_and_before_browser_disconnect(monkeypatch):
    app = RadarPdNovaApp()
    events = []
    async def checkpoint():
        events.append('checkpoint')
    async def child_exit():
        pytest.fail('Exiting only the launcher child does not release its container')
    app._stop_session_enrollment = checkpoint
    app._leave_session_page = lambda: events.append('leave')
    app.exit_callback = child_exit
    monkeypatch.setattr(lifecycle, 'schedule_supervised_shutdown', lambda: events.append('detached-shutdown') or True)
    asyncio.run(app.end_session())
    assert events == ['checkpoint', 'detached-shutdown', 'leave']


def test_shutdown_failure_keeps_a_reviewable_page_open(monkeypatch):
    app = RadarPdNovaApp()
    app._leave_session_page = lambda: pytest.fail('Do not redirect after shutdown scheduling failed')
    async def checkpoint():
        pass
    app._stop_session_enrollment = checkpoint
    def fail():
        raise OSError('process launch failed')
    monkeypatch.setattr(lifecycle, 'schedule_supervised_shutdown', fail)
    asyncio.run(app.end_session())
    assert 'shutdown could not be scheduled' in app.server.state.workspace_notice
