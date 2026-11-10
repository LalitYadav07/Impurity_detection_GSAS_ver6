"""A restored watch owns acquisition reads without changing the setup draft."""
import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from radar_pd_nova.live_beamlines import beamline_adapter
from test_live_restore import app, inspect, restore
from test_webmon_live_scope import reference, view


def record_reads(monkeypatch, backend):
    calls = []
    for name in ('runs', 'capture', 'checkpoints', 'selected_checkpoint', 'binding'):
        original = getattr(backend, name)

        def call(ipts, *args, _name=name, _original=original, **kwargs):
            calls.append((_name, ipts))
            return _original(ipts, *args, **kwargs)

        monkeypatch.setattr(backend, name, call)
    return calls


@pytest.mark.parametrize('draft', ['', '  ', None])
def test_restored_blank_draft_uses_saved_ipts_for_full_preview_chain(view, monkeypatch, draft):
    state = view.server.state
    state.powgen_ipts = draft
    state.powgen_monitoring = False
    view._webmon_start_scope = None
    calls = record_reads(monkeypatch, view._webmon)
    view._powgen_controller.discover = lambda: pytest.fail('No watch discovery')
    view._powgen_controller.launch_submission = lambda *a, **k: pytest.fail('No scientific submission')

    view.webmon_refresh()
    assert state.webmon_run == '8'
    assert [row['run'] for row in state.webmon_runs] == ['8', '7']
    assert state.powgen_ipts is draft and state.powgen_monitoring is False
    assert view._webmon_task is None
    assert view._webmon_last_preview[:2] == ('IPTS-2767', '8')
    assert {'runs', 'capture', 'checkpoints', 'selected_checkpoint', 'binding'} <= {name for name, _ in calls}
    assert {ipts for _, ipts in calls} == {'IPTS-2767'}

    view.webmon_select('7')
    assert state.webmon_run == '7'
    assert view._webmon_last_preview[:2] == ('IPTS-2767', '7')
    assert {ipts for _, ipts in calls} == {'IPTS-2767'}


def test_actual_saved_watch_restore_leaves_draft_blank_but_can_refresh(app, view, monkeypatch):
    adapter = beamline_adapter('PG3')
    settings = adapter.settings(ipts='IPTS-2767', history_id=app.service.history_id,
        configuration_dataset_id='config-1', selected_sample_ids=('100',),
        selected_banks=(2,), wavelength_angstrom='1.5')
    controller = adapter.controller(app.service, settings)
    app.service.payloads['checkpoint-1'] = json.loads(controller.capture_checkpoint().payload_json)
    app.service.galaxy_key = 'test-key'
    asyncio.run(inspect(app))
    restore(app)
    assert app._powgen_controller is not None
    assert app.server.state.powgen_ipts == ''
    app._webmon = view._webmon
    # This app fixture builds without a production WebMON canvas.
    app._webmon_observed = view._webmon_observed
    calls = record_reads(monkeypatch, app._webmon)

    app.webmon_refresh()
    assert app.server.state.webmon_run == '8', app.server.state.webmon_message
    assert [row['run'] for row in app.server.state.webmon_runs] == ['8', '7']
    assert app.server.state.powgen_ipts == ''
    assert not app.server.state.powgen_monitoring
    assert app._webmon_task is None and app._webmon_poll_task is None
    assert {ipts for _, ipts in calls} == {'IPTS-2767'}


@pytest.mark.parametrize('draft', ['2767', 'IPTS-2767', 'ipts-2767', ' 2767 '])
def test_matching_active_watch_keeps_same_source_and_no_extra_dispatch(view, draft):
    view.server.state.powgen_ipts = draft
    view.server.state.powgen_monitoring = True
    view.webmon_refresh()
    assert view.server.state.webmon_run == '8'
    assert view.server.state.powgen_ipts == draft
    assert view.server.state.powgen_monitoring is True
    assert view._webmon_task is None


@pytest.mark.parametrize('draft', ['9999', 'IPTS-9999', 'IPTS-', '../2767', 0, False])
def test_conflicting_or_invalid_draft_cannot_retarget_saved_watch(view, monkeypatch, draft):
    view.server.state.powgen_ipts = draft
    view._webmon_start_scope = None
    calls = record_reads(monkeypatch, view._webmon)
    view.webmon_refresh()
    assert view.server.state.webmon_run == '8'
    assert {ipts for _, ipts in calls} == {'IPTS-2767'}


def test_different_draft_beamline_does_not_hide_saved_powgen_acquisition(view, monkeypatch):
    view.server.state.live_beamline = 'NOM'
    view.server.state.powgen_ipts = 'IPTS-9999'
    calls = record_reads(monkeypatch, view._webmon)
    view.webmon_refresh()
    assert view.server.state.webmon_run == '8'
    assert {ipts for _, ipts in calls} == {'IPTS-2767'}


def test_wrong_history_is_rejected_even_when_draft_is_blank(view, monkeypatch):
    view.server.state.powgen_ipts = ''
    view.service.history_id = 'another-history'
    view._webmon_start_scope = None
    monkeypatch.setattr(view._webmon, 'runs', lambda *a: pytest.fail('No foreign source read'))
    view.webmon_refresh()
    assert 'another Galaxy History' in view.server.state.webmon_message
    assert not view.server.state.webmon_context_ready


def test_blank_draft_without_saved_controller_never_enables_production_fallback(view, monkeypatch):
    view.server.state.powgen_ipts = ''
    view._powgen_controller = None
    view._webmon_start_scope = None
    monkeypatch.setattr(view._webmon, 'runs', lambda *a: pytest.fail('No unowned production read'))
    view.webmon_refresh()
    assert 'Start or restore a saved Live watch' in view.server.state.webmon_message


@pytest.mark.parametrize('local', [False, True])
def test_standalone_and_local_preview_keep_draft_source(view, monkeypatch, local):
    view._powgen_controller = None
    if local:
        view._webmon.config['local_replay'] = True
    else:
        view.service = None
    calls = record_reads(monkeypatch, view._webmon)
    view.webmon_refresh()
    assert view.server.state.webmon_run == '10'  # Standalone mode has no saved sample filter.
    assert {ipts for _, ipts in calls} == {'2767'}


def test_blank_draft_published_catalog_uses_same_owned_experiment(view):
    view.server.state.powgen_ipts = ''
    view._webmon_start_scope = None
    view.webmon_refresh()
    sources = [reference(), reference('wrong-sample', sample='200'),
               reference('wrong-bank', bank='3'), reference('wrong-history', history='other')]
    view._refinement_store = lambda: SimpleNamespace(state={'sources': {s['id']: s for s in sources}})

    async def refresh():
        pass

    view._refresh_refinement_sources = refresh
    asyncio.run(view.webmon_load_published_sources())
    assert set(view._webmon_published_catalog) == {'source'}


def test_standalone_published_catalog_retains_canonical_experiment_identity(view):
    view._powgen_controller = None
    view.service.galaxy_url = ''
    view.webmon_refresh()
    sources = [reference(), reference('wrong-history', history='other')]
    view._refinement_store = lambda: SimpleNamespace(state={'sources': {s['id']: s for s in sources}})

    async def refresh():
        pass

    view._refresh_refinement_sources = refresh
    asyncio.run(view.webmon_load_published_sources())
    assert set(view._webmon_published_catalog) == {'source'}


def clear_owner(view):
    # The same ownership transitions used by Clear: new context, no controller,
    # blank draft, and cleared displayed data. Retained source files stay intact.
    view._result_context = object()
    view._powgen_controller = None
    view.server.state.powgen_ipts = ''
    view._webmon_clear_scope()
    view.server.state.webmon_runs = []
    view.server.state.webmon_latest = ''
    view.server.state.webmon_message = 'Workspace cleared'


@pytest.mark.parametrize('during', ['runs', 'capture'])
def test_clear_during_preview_read_cannot_republish_old_scope(view, monkeypatch, during):
    view.server.state.powgen_ipts = ''
    view._webmon_start_scope = None
    view.webmon_refresh()
    entered, release = threading.Event(), threading.Event()
    original = getattr(view._webmon, during)

    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(view._webmon, during, delayed)
    updates = []
    view._webmon_observed.update = updates.append

    async def run():
        action = view.webmon_refresh if during == 'runs' else lambda: view.webmon_select('7')
        task = asyncio.create_task(asyncio.to_thread(action))
        assert await asyncio.to_thread(entered.wait, 5)
        clear_owner(view)
        release.set()
        await task

    asyncio.run(run())
    assert view.server.state.webmon_run == ''
    assert view.server.state.webmon_runs == []
    assert view.server.state.webmon_message == 'Workspace cleared'
    assert not view.server.state.webmon_context_ready and not view.server.state.webmon_scope_allowed
    assert len(updates) <= 1  # Clear may send its empty figure; old data never follows it.
    assert view._webmon_last_preview is None


def test_clear_during_published_lookup_cannot_republish_old_catalog(view):
    view.server.state.powgen_ipts = ''
    view._webmon_start_scope = None
    view.webmon_refresh()
    entered, release = asyncio.Event(), asyncio.Event()

    async def refresh():
        entered.set()
        await release.wait()

    view._refresh_refinement_sources = refresh
    view._refinement_store = lambda: SimpleNamespace(state={'sources': {'source': reference()}})

    async def run():
        task = asyncio.create_task(view.webmon_load_published_sources())
        await entered.wait()
        clear_owner(view)
        release.set()
        await task

    asyncio.run(run())
    assert view._webmon_published_catalog == {}
    assert view.server.state.webmon_message == 'Workspace cleared'


def test_draft_change_away_and_back_preserves_saved_watch_owner(view):
    view.server.state.powgen_ipts = ''
    view._webmon_start_scope = None
    view.webmon_refresh()
    owner = view._webmon_scope_snapshot()
    view.server.state.powgen_ipts = '2767'
    view.webmon_scope_changed()
    view.server.state.powgen_ipts = ''
    view.webmon_scope_changed()
    assert view._webmon_scope_current(owner)
    assert view.server.state.webmon_run == '8'
