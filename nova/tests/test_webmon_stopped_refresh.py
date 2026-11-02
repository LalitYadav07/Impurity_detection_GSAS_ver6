"""A stopped saved watch can inspect acquisition data without resuming work."""
import inspect

import pytest

from radar_pd_nova.webmon.ui import WebmonMixin
from test_webmon_live_scope import view


def test_stopped_refresh_keeps_saved_scope_and_never_starts_jobs(view):
    state = view.server.state
    state.powgen_monitoring = False
    view._powgen_controller.discover = lambda: pytest.fail('No discovery/resume')
    view._powgen_controller.launch_submission = lambda *a, **k: pytest.fail('No analysis')
    view._webmon_poll_task = None
    view.webmon_refresh()
    assert state.powgen_monitoring is False
    assert view._webmon_task is None and view._webmon_poll_task is None
    assert state.webmon_run == '8'
    assert [r['run'] for r in state.webmon_runs] == ['8', '7']
    assert 'sample 100' in state.webmon_scope_notice
    assert 'monitoring remains stopped' in state.webmon_connection
    assert 'checking every' not in state.webmon_connection
    assert not state.webmon_context_ready and not state.webmon_accepted


def test_controller_removed_after_ui_enable_fails_before_read(view, monkeypatch):
    view.server.state.powgen_monitoring = False
    view.server.state.powgen_results_refresh_available = True
    view._powgen_controller = None
    monkeypatch.setattr(view._webmon, 'runs', lambda *a: pytest.fail('No unscoped source read'))
    view.webmon_refresh()
    assert 'Start or restore a saved Live watch' in view.server.state.webmon_message
    assert not view.server.state.webmon_context_ready


def test_refresh_control_includes_saved_controller_availability_and_busy_gate():
    source = inspect.getsource(WebmonMixin._webmon_ui)
    line = next(line for line in source.splitlines() if "VBtn('Refresh source'" in line)
    assert 'webmon_busy' in line
    assert '!powgen_results_refresh_available' in line
    assert '!webmon_local' in line
    assert 'does not resume monitoring or start analyses' in source
