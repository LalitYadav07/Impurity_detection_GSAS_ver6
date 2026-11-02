"""A fresh model search must not reconstruct unrelated historical campaigns."""
import asyncio
from types import SimpleNamespace

import pytest

from radar_pd_nova import symmetry_controller as controller


class State(SimpleNamespace):
    def flush(self):
        pass


def setup_app(tmp_path, monkeypatch):
    state = State(rf_experiment='IPTS-1', sx_result_scope='experiment',
                  sx_start_source='', rf_source='current', sx_campaign='',
                  sx_source='', sx_busy=False, sx_pause_pending=False)
    app = controller.SymmetryMixin()
    app.server = SimpleNamespace(state=state)
    app._symmetry_loading = False
    app._symmetry_refresh_pending = False
    app._symmetry_display_identity = None
    app._model_remote = lambda: object()
    app._symmetry_root = lambda: tmp_path
    app._refinement_store = lambda: object()
    app._symmetry_publish_source = lambda: None
    monkeypatch.setattr(controller, 'start_choices', lambda *a: [
        {'id': 'current', 'title': 'Current published GPX'}])
    return app, state


async def run_action(app, action, *, before_await=None):
    tasks = []
    app._refinement_schedule = lambda coroutine, *a: tasks.append(asyncio.create_task(coroutine))
    action()
    if before_await:
        before_await()
    while tasks:
        pending = list(tasks)
        tasks.clear()
        await asyncio.gather(*pending)


def test_fresh_search_lists_sources_without_restoring_any_old_campaign(tmp_path, monkeypatch):
    app, state = setup_app(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(controller, 'restore_history', lambda *a, **k: calls.append(k['experiment']))
    asyncio.run(run_action(app, app.refresh_symmetry))
    asyncio.run(run_action(app, app.refresh_symmetry))
    assert calls == []
    assert state.sx_start_sources == [{'title': 'Current published GPX', 'value': 'current'}]
    assert state.sx_start_source == 'current'
    assert not state.sx_busy


@pytest.mark.parametrize('scope', ['experiment', 'campaign'])
def test_explicit_load_honors_selected_scope_and_does_not_auto_repeat(tmp_path, monkeypatch, scope):
    app, state = setup_app(tmp_path, monkeypatch)
    state.sx_result_scope = scope
    calls = []
    monkeypatch.setattr(controller, 'restore_history', lambda *a, **k: calls.append(k['experiment']))
    asyncio.run(run_action(app, app.restore_saved_symmetry_campaigns))
    asyncio.run(run_action(app, app.refresh_symmetry))
    assert calls == ['IPTS-1' if scope == 'experiment' else None]
    assert app._model_restore_requests == set()
    state.rf_experiment = 'IPTS-2'
    asyncio.run(run_action(app, app.refresh_symmetry))
    assert len(calls) == 1


def test_failure_requires_another_explicit_load_not_an_automatic_retry(tmp_path, monkeypatch):
    app, state = setup_app(tmp_path, monkeypatch)
    calls = []
    def fail(*a, **k):
        calls.append(k['experiment'])
        raise ValueError('Historical source unavailable')
    monkeypatch.setattr(controller, 'restore_history', fail)
    asyncio.run(run_action(app, app.restore_saved_symmetry_campaigns))
    assert state.sx_error == 'Historical source unavailable'
    asyncio.run(run_action(app, app.refresh_symmetry))
    assert calls == ['IPTS-1']
    asyncio.run(run_action(app, app.restore_saved_symmetry_campaigns))
    assert calls == ['IPTS-1', 'IPTS-1']


def test_selection_changed_before_scheduling_does_not_restore_either_scope(tmp_path, monkeypatch):
    app, state = setup_app(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(controller, 'restore_history', lambda *a, **k: calls.append(k['experiment']))
    asyncio.run(run_action(app, app.restore_saved_symmetry_campaigns,
                           before_await=lambda: setattr(state, 'rf_experiment', 'IPTS-2')))
    state.rf_experiment = 'IPTS-1'
    asyncio.run(run_action(app, app.refresh_symmetry))
    assert calls == []


def test_explicit_reload_after_whole_history_load_is_not_suppressed(tmp_path, monkeypatch):
    app, state = setup_app(tmp_path, monkeypatch)
    app._model_restored_scopes = {None, 'IPTS-1'}
    calls = []
    monkeypatch.setattr(controller, 'restore_history', lambda *a, **k: calls.append(k['experiment']))
    asyncio.run(run_action(app, app.restore_saved_symmetry_campaigns))
    assert calls == ['IPTS-1']


def test_restore_completion_cannot_publish_into_a_changed_experiment(tmp_path, monkeypatch):
    app, state = setup_app(tmp_path, monkeypatch)
    seen = []
    monkeypatch.setattr(controller, 'start_choices', lambda w, scope: seen.append(scope) or [])
    def switch(*a, **k):
        state.rf_experiment = 'IPTS-2'
    monkeypatch.setattr(controller, 'restore_history', switch)
    asyncio.run(run_action(app, app.restore_saved_symmetry_campaigns))
    assert seen == ['IPTS-2']
    assert app._model_restored_scopes == {'IPTS-1'}


@pytest.mark.parametrize('busy', ['loading', 'busy', 'pausing'])
def test_duplicate_load_during_existing_work_does_not_queue_another_restore(tmp_path, monkeypatch, busy):
    app, state = setup_app(tmp_path, monkeypatch)
    app._symmetry_loading = busy == 'loading'
    state.sx_busy = busy == 'busy'
    state.sx_pause_pending = busy == 'pausing'
    app.refresh_symmetry = lambda: pytest.fail('An overlapping restore was queued')
    app.restore_saved_symmetry_campaigns()
    assert not getattr(app, '_model_restore_requests', set())


def test_workspace_clear_discards_old_restore_request_and_scope_cache(tmp_path, monkeypatch):
    app, state = setup_app(tmp_path, monkeypatch)
    app._symmetry_defaults = {}
    app._model_restore_requests = {'IPTS-1'}
    app._model_restored_scopes = {None, 'IPTS-1'}
    app._reset_model_view()
    assert app._model_restore_requests == set()
    assert app._model_restored_scopes == set()


def test_interface_exposes_explicit_recovery_alongside_new_search():
    from radar_pd_nova.app import RadarPdNovaApp
    html = RadarPdNovaApp().layout.html
    assert 'Load saved campaigns' in html
    assert 'this recovers prior work without submitting new refinements' in html
