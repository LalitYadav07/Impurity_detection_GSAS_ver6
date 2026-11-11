"""Clear the current experiment without deleting reusable preparation work."""

import asyncio
import threading
from copy import deepcopy

import pytest

from radar_pd_nova.app import RadarPdNovaApp
from test_preparation_registry import prepared
from test_model_runtime import remote
from test_preparation_workspace import workspace


def test_clear_retires_the_active_preparation_but_keeps_its_library(tmp_path, monkeypatch):
    monkeypatch.setenv('RADAR_PD_PREPARATION_ROOT', str(tmp_path/'app'))
    _, _, archive = prepared(tmp_path)
    app = RadarPdNovaApp()
    store = app._preparation_store()
    key = store.registry.import_archive(archive)
    library = store.registry.get(key)
    before = {str(p.relative_to(library.root)): p.read_bytes()
              for p in library.root.rglob('*') if p.is_file()}
    state = app.server.state
    state.prep_library = key
    state.prep_reference = key
    state.prep_parent_path = 'previous-parent.cif'
    state.prep_profile_path = 'previous-bank.instprm'
    state.prep_context_ready = True
    state.prep_sites = ['Fe1']
    state.prep_selected_k = ['0,0,0']
    app._prep_context = {'parent': 'previous material'}
    app._prep_return = ('refinement', True)

    asyncio.run(app.clear_workspace())

    assert not state.prep_library and not state.prep_reference
    assert not state.prep_parent_path and not state.prep_profile_path
    assert not state.prep_sites and not state.prep_selected_k
    assert not state.prep_context_ready and app._prep_context is None
    assert app._prep_return is None
    assert app._preparation_workspace is None
    reopened = app._preparation_store()
    assert reopened.registry.selected_key() == ''
    retained = reopened.registry.get(key)
    assert before == {str(p.relative_to(retained.root)): p.read_bytes()
                      for p in retained.root.rglob('*') if p.is_file()}
    app.open_preparation_workspace()
    assert not state.prep_library
    assert key in {row['value'] for row in state.prep_library_options}
    app.close_preparation_workspace()
    assert state.workspace_view == 'monitor'
    # Reuse remains an explicit selection, without reconstructing the library.
    state.prep_library = key
    app._preparation_library_changed()
    assert reopened.registry.selected_key() == key


def test_clear_deselects_a_saved_library_before_the_store_is_opened(tmp_path, monkeypatch):
    monkeypatch.setenv('RADAR_PD_PREPARATION_ROOT', str(tmp_path/'app'))
    _, _, archive = prepared(tmp_path)
    app = RadarPdNovaApp()
    registry = app._preparation_store().registry
    key = registry.import_archive(archive)
    app._preparation_workspace = None

    asyncio.run(app.clear_workspace())

    assert app._preparation_workspace is None
    assert registry.selected_key() == ''
    assert registry.get(key).coverage()['all_prepared']


def test_clear_without_preparation_does_not_create_a_store(tmp_path, monkeypatch):
    root = tmp_path/'unused-preparation'
    monkeypatch.setenv('RADAR_PD_PREPARATION_ROOT', str(root))
    app = RadarPdNovaApp()

    asyncio.run(app.clear_workspace())

    assert app._preparation_workspace is None
    assert not root.exists()


def test_failed_deselection_keeps_the_visible_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv('RADAR_PD_PREPARATION_ROOT', str(tmp_path/'app'))
    _, _, archive = prepared(tmp_path)
    app = RadarPdNovaApp()
    store = app._preparation_store()
    key = store.registry.import_archive(archive)
    app.server.state.prep_library = key
    app.records['previous-scan'] = object()

    def failed():
        raise OSError('Selection write failed')

    monkeypatch.setattr(store.registry, 'clear_selection', failed)
    asyncio.run(app.clear_workspace())

    assert 'previous-scan' in app.records
    assert app.server.state.prep_library == store.registry.selected_key() == key
    assert not app.server.state.workspace_clearing
    assert 'was not cleared' in app.server.state.workspace_notice


def test_late_reference_read_cannot_repopulate_cleared_setup(tmp_path, monkeypatch):
    monkeypatch.setenv('RADAR_PD_PREPARATION_ROOT', str(tmp_path/'app'))
    _, _, archive = prepared(tmp_path)
    app = RadarPdNovaApp()
    store = app._preparation_store()
    key = store.registry.import_archive(archive)
    app.server.state.prep_reference = key
    entered, release = threading.Event(), threading.Event()

    def reference(_key):
        entered.set()
        assert release.wait(3)
        return {'parent': {'name': 'Previous material'}}

    monkeypatch.setattr(store, 'references', reference)

    async def scenario():
        pending = app._prep_schedule(app._prep_load_reference(key), 'late-reference')
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            await app.clear_workspace()
            assert pending.cancelled()
        finally:
            release.set()
        assert not app.server.state.prep_reference
        assert not app.server.state.prep_parent_title
        assert app._prep_context is None

    asyncio.run(scenario())


def test_shutdown_alone_keeps_the_selection_for_session_recovery(tmp_path, monkeypatch):
    monkeypatch.setenv('RADAR_PD_PREPARATION_ROOT', str(tmp_path/'app'))
    _, _, archive = prepared(tmp_path)
    app = RadarPdNovaApp()
    store = app._preparation_store()
    key = store.registry.import_archive(archive)

    asyncio.run(app._preparation_shutdown())

    assert store.registry.selected_key() == key


def test_clear_pauses_preparation_and_retains_owned_jobs(workspace):
    from radar_pd_nova.preparation_workspace import PreparationWorkspace

    store, key, service = workspace
    store.start(key)
    store.tick(key)
    queue = store.queue(key)
    before = deepcopy(queue.snapshot()['jobs'])
    submissions = len(service.submissions)
    assert submissions == 1 and before
    app = RadarPdNovaApp()
    app._preparation_workspace = store
    app.server.state.prep_library = key

    asyncio.run(app.clear_workspace())

    assert queue.snapshot()['paused']
    assert queue.snapshot()['jobs'] == before
    assert store.registry.selected_key() == ''
    recovered = PreparationWorkspace(store.root, store.runtime)
    assert recovered.queue(key).snapshot()['jobs'] == before
    assert len(service.submissions) == submissions


@pytest.mark.parametrize('closing_state', ['workspace_clearing', 'session_closing'])
def test_preparation_cannot_start_during_workspace_or_session_shutdown(closing_state):
    app = RadarPdNovaApp()
    setattr(app.server.state, closing_state, True)
    started = []

    async def action():
        started.append(True)

    async def scenario():
        assert app._prep_action(action(), 'Late preparation') is None
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert not started and not app.server.state.prep_busy
    assert 'reset or session shutdown' in app.server.state.prep_error
