"""Late, file-validated preview reads must not undo a cleared workspace."""
import asyncio
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import uuid

import numpy as np
import pytest

from radar_pd_nova.webmon.session import digest
from radar_pd_nova.webmon.stream import LatestQueue
from radar_pd_nova.webmon.ui import WebmonMixin
from test_live_restore import SavedHistory, inspect, restore
from test_webmon_hardening import scientific_request
from test_webmon_live_scope import reference, view
from test_webmon_restored_scope import clear_owner


def reviewed_checkpoints(view, tmp_path):
    project = tmp_path / 'reference.gpx'
    project.write_bytes(b'unchanged reference fixture')
    arrays = tmp_path / 'view.npz'
    np.savez(arrays, x=[1., 2., 3.], y=[10., 20., 30.], calc=[9., 19., 29.],
             sigma=[2., 3., 4.], valid=[True] * 3, live_x=[1., 2., 3.],
             live_scaled=[10., 20., 30.])
    source = reference(history=view.service.history_id)
    source.update(source_path=str(project), source_sha256=digest(project))
    for run in ('7', '8'):
        data = Path(next(r['path'] for r in view._webmon.runs('2767') if r['run'] == run))
        binding = dict(reference_gpx=str(project), reference_sha256=digest(project),
            source_gpx=str(project), source_sha256=digest(project), source_dat=str(data),
            source_dat_sha256=digest(data), live_source_path=str(data),
            description='reviewed reference', review=[], k=['0', '0', '0'],
            ion='Fe+3', maximum_moment=5.)
        view._webmon.config.setdefault('checkpoint_options', {})['IPTS-2767/' + run] = [
            dict(id='reviewed-' + run, title='Reviewed checkpoint ' + run,
                 source_path=str(project), source_sha256=digest(project),
                 refinement_source=source, binding=binding, view_path=str(arrays),
                 view_sha256=digest(arrays), run=run, role='Nuclear reference',
                 verification={'rwp_recomputed': 1., 'peaks': []})]
    view._webmon_checkpoint_plot = SimpleNamespace(update=lambda figure: None)
    view.server.state.webmon_checkpoint_axis = 'tof'
    view.server.state.powgen_ipts = ''
    view._webmon_start_scope = None
    view.webmon_refresh()
    assert view.server.state.webmon_context_ready


def assert_clear(view):
    state = view.server.state
    assert not state.webmon_context_ready
    assert view._webmon_binding is None
    assert state.webmon_reference == ''
    assert state.webmon_reference_rows == []
    assert state.webmon_checkpoints == []
    assert state.webmon_checkpoint == ''
    assert state.webmon_checkpoint_summary == ''
    assert state.webmon_width_rows == []
    assert state.webmon_run == ''
    assert state.webmon_message == 'Workspace cleared'


@pytest.mark.parametrize('action,read', [
    ('select', 'checkpoints'), ('select', 'selected_checkpoint'), ('select', 'binding'),
    ('change', 'checkpoints'), ('change', 'select_checkpoint'), ('change', 'binding'),
    ('axis', 'selected_checkpoint'), ('busy', 'selected_checkpoint'),
])
def test_clear_during_late_checkpoint_read_keeps_all_publication_cleared(view, tmp_path, monkeypatch, action, read):
    reviewed_checkpoints(view, tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = getattr(view._webmon, read)

    def delayed(*args, **kwargs):
        value = original(*args, **kwargs)
        entered.set()
        assert release.wait(5)
        return value

    monkeypatch.setattr(view._webmon, read, delayed)
    calls = {
        'select': lambda: view.webmon_select('7'),
        'change': lambda: view.webmon_checkpoint_changed('reviewed-8'),
        'axis': lambda: view.webmon_checkpoint_axis_changed('tof'),
        'busy': lambda: view.webmon_checkpoint_changed('reviewed-7'),
    }
    if action == 'busy':
        view.server.state.webmon_busy = True

    async def run():
        task = asyncio.create_task(asyncio.to_thread(calls[action]))
        assert await asyncio.to_thread(entered.wait, 5)
        clear_owner(view)
        release.set()
        await task

    asyncio.run(run())
    assert_clear(view)


def test_clear_during_checkpoint_array_read_cannot_repaint_reference(view, tmp_path, monkeypatch):
    reviewed_checkpoints(view, tmp_path)
    updates = []
    view._webmon_checkpoint_plot.update = updates.append
    original = np.load

    def clear_after_open(*args, **kwargs):
        data = original(*args, **kwargs)
        clear_owner(view)
        return data

    monkeypatch.setattr(np, 'load', clear_after_open)
    view.webmon_checkpoint_axis_changed('tof')
    assert_clear(view)
    assert len(updates) == 1 and not updates[0].data  # Only Clear's empty figure.


def test_busy_checkpoint_history_conflict_is_caught_before_source_read(view, tmp_path, monkeypatch):
    reviewed_checkpoints(view, tmp_path)
    view.server.state.webmon_busy = True
    view.service.history_id = 'another-history'
    monkeypatch.setattr(view._webmon, 'selected_checkpoint', lambda *a: pytest.fail('No conflicting source read'))
    view.webmon_checkpoint_changed('other')
    assert not view.server.state.webmon_context_ready
    assert not view.server.state.webmon_accepted
    assert view.server.state.webmon_checkpoint == ''
    assert 'another Galaxy History' in view.server.state.webmon_message


def test_busy_checkpoint_keeps_saved_watch_when_setup_draft_ipts_changes(view, tmp_path, monkeypatch):
    reviewed_checkpoints(view, tmp_path)
    view.server.state.webmon_busy = True
    view.server.state.powgen_ipts = '9999'
    selected=[]
    original=view._webmon.selected_checkpoint
    def read(*args):
        selected.append(args)
        return original(*args)
    monkeypatch.setattr(view._webmon, 'selected_checkpoint', read)
    view.webmon_checkpoint_changed('other')
    assert selected == [('IPTS-2767', '8')]
    assert view.server.state.webmon_context_ready
    assert view.server.state.webmon_checkpoint == 'reviewed-8'


def saved_result(view, tmp_path):
    science = tmp_path / 'science'
    science.mkdir()
    request, binding = scientific_request(science)
    data = Path(next(r['path'] for r in view._webmon.runs('2767') if r['run'] == '8'))
    binding.update(reference_gpx=binding['source_gpx'], reference_sha256=binding['source_sha256'],
        source_dat=str(data), source_dat_sha256=digest(data), live_source_path=str(data),
        description='verified saved reference', review=[])
    request.update(source_revision='PG3_8:' + binding['source_dat_sha256'],
                   reference_revision='PG3_7:' + binding['reference_sha256'])
    out = Path(view._webmon.config['jobs']) / 'saved'
    out.mkdir(parents=True)
    request_path = out / 'request.json'
    request_path.write_text(json.dumps(request))
    binding['request'] = str(request_path)
    source = reference()
    source.update(source_path=binding['reference_gpx'], source_sha256=binding['reference_sha256'])
    view._webmon.config['checkpoint_options'] = {'IPTS-2767/8': [
        dict(id='checkpoint', title='Verified checkpoint', source_path=binding['reference_gpx'],
             source_sha256=binding['reference_sha256'], refinement_source=source, binding=binding)]}
    result = dict(status='complete', request_sha256=digest(request_path),
        source_revision=request['source_revision'], reference_revision=request['reference_revision'],
        forecast_enabled=False, exclusion_authorized=False, rows=[], elapsed_seconds=1.)
    (out / 'result.json').write_text(json.dumps(result))
    view._webmon_review_checkpoint = lambda choice, **_: True
    view.webmon_render = lambda: None
    view._webmon_out = None
    view.server.state.powgen_ipts = ''
    view._webmon_start_scope = None
    view.webmon_select('8')
    assert view.server.state.webmon_context_ready
    view._webmon_saved_paths = {'saved': out}
    view.server.state.webmon_saved_job = 'saved'
    return out


@pytest.mark.parametrize('read', ['restore_binding', 'capture'])
def test_clear_during_saved_result_reads_does_not_restore_old_result(view, tmp_path, monkeypatch, read):
    saved_result(view, tmp_path)
    original = getattr(view._webmon, read)

    def clear_after_read(*args, **kwargs):
        value = original(*args, **kwargs)
        clear_owner(view)
        return value

    monkeypatch.setattr(view._webmon, read, clear_after_read)
    view.webmon_restore()
    assert_clear(view)
    assert view.server.state.webmon_target == ''
    assert view.server.state.webmon_calculation_status == 'Not started'
    assert not view._webmon_request and view._webmon_out is None


def test_clear_during_saved_list_read_does_not_restore_old_choices(view, tmp_path, monkeypatch):
    saved_result(view, tmp_path)
    original = view._webmon.restore_binding

    def clear_after_read(*args, **kwargs):
        value = original(*args, **kwargs)
        clear_owner(view)
        return value

    monkeypatch.setattr(view._webmon, 'restore_binding', clear_after_read)
    view.webmon_saved_list = WebmonMixin.webmon_saved_list.__get__(view)
    view.webmon_saved_list()
    assert_clear(view)
    assert view.server.state.webmon_saved == []
    assert view._webmon_saved_paths == {}


def test_clear_during_quiet_binding_error_preserves_clear_message(view, tmp_path, monkeypatch):
    saved_result(view, tmp_path)
    view.server.state.webmon_follow_latest = False
    view.server.state.webmon_target = '8'

    def clear_and_fail(*args, **kwargs):
        clear_owner(view)
        raise ValueError('old reference changed')

    monkeypatch.setattr(view._webmon, 'restore_binding', clear_and_fail)
    view.webmon_refresh(quiet=True)
    assert_clear(view)


def test_clear_during_stream_capture_keeps_queue_closed_and_message_paused(view, tmp_path):
    view.server.state.powgen_ipts = ''
    view._webmon_start_scope = None
    view.webmon_refresh()

    def capture():
        clear_owner(view)
        return None

    view._webmon_stream = SimpleNamespace(ipts='IPTS-2767', run='8', capture=capture)
    view._webmon_stream_scope = view._webmon_scope_snapshot()
    queue = LatestQueue(tmp_path / 'queue.json')
    view._webmon_queue = queue
    view.webmon_stream_tick()
    assert queue.closed and view._webmon_stream is None
    assert view.server.state.webmon_stream_message.startswith('Automatic updates paused.')
    assert_clear(view)


def test_current_stream_worker_failure_still_pauses_and_reports_error(view, tmp_path):
    view.server.state.powgen_ipts = ''
    view._webmon_start_scope = None
    view.webmon_refresh()
    view._webmon_stream = SimpleNamespace(ipts='IPTS-2767', run='8', capture=lambda: None)
    view._webmon_stream_scope = view._webmon_scope_snapshot()
    view.server.state.webmon_stream_active = True
    queue = LatestQueue(tmp_path / 'failed-worker-queue.json')
    view._webmon_queue = queue
    queue.offer(dict(id='update', binding={}, request={}))

    async def fail(_):
        view.server.state.webmon_calculation_status = 'Failed'
        view.server.state.webmon_message = 'worker failure evidence'

    view._webmon_execute = fail

    async def run():
        view.webmon_stream_tick()
        await view._webmon_task

    asyncio.run(run())
    assert queue.closed and view._webmon_stream is None
    assert not view.server.state.webmon_stream_active
    assert view.server.state.webmon_stream_message == 'Automatic analysis paused: worker failure evidence'


def restored_application(app, view, tmp_path):
    from radar_pd_nova.live_beamlines import beamline_adapter
    app.server.state.ready()
    adapter = beamline_adapter('PG3')
    settings = adapter.settings(ipts='IPTS-2767', history_id=app.service.history_id,
        configuration_dataset_id='config-1', selected_sample_ids=('100',),
        selected_banks=(2,), wavelength_angstrom='1.5')
    controller = adapter.controller(app.service, settings)
    app.service.payloads['checkpoint-1'] = json.loads(controller.capture_checkpoint().payload_json)
    app.service.galaxy_key = 'test-key'
    asyncio.run(inspect(app))
    restore(app)
    app._webmon = view._webmon
    app._webmon_observed = view._webmon_observed
    reviewed_checkpoints(app, tmp_path)
    assert app.server.state.powgen_ipts == ''
    assert app._webmon_binding is not None and app._webmon_pattern is not None


@pytest.fixture
def isolated_app(monkeypatch):
    from radar_pd_nova.app import RadarPdNovaApp
    from trame.app import get_server
    # State.ready() enables real callbacks. Keep these lifecycle tests off the
    # shared default server used by older, deliberately unready app fixtures.
    server = get_server('webmon-lifecycle-' + uuid.uuid4().hex, client_type='vue3')
    monkeypatch.setattr('radar_pd_nova.app.get_server', lambda *args, **kwargs: server)
    app = RadarPdNovaApp()
    app.service = SavedHistory()
    app._sync_activities = lambda: None
    app.refresh_history = lambda **_: None
    app._sync_runs = lambda: None
    return app


def assert_application_cleared(app):
    state = app.server.state
    assert state.workspace_notice.startswith('Workspace cleared.')
    assert app._powgen_controller is None and not state.powgen_monitoring
    assert app._webmon_binding is None and app._webmon_pattern is None
    assert app._webmon_plot_binding is None and app._webmon_display_scope is None
    assert app._webmon_stream_scope is None
    assert not state.webmon_context_ready
    assert state.webmon_run == '' and state.webmon_reference == ''
    assert state.webmon_runs == [] and state.webmon_latest == ''
    assert state.webmon_checkpoints == [] and state.webmon_width_rows == []
    assert state.webmon_checked == 'Not checked'
    assert state.webmon_connection == 'Not connected'
    assert state.webmon_message.startswith('Start or restore a saved Live watch')


def test_real_successful_clear_retires_restored_state_with_unchanged_blank_draft(isolated_app, view, tmp_path):
    app = isolated_app
    restored_application(app, view, tmp_path)
    asyncio.run(app.clear_workspace())
    assert_application_cleared(app)


def test_real_clear_during_late_binding_does_not_repopulate_restored_state(isolated_app, view, tmp_path, monkeypatch):
    app = isolated_app
    restored_application(app, view, tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = app._webmon.binding

    def delayed(*args, **kwargs):
        value = original(*args, **kwargs)
        entered.set()
        assert release.wait(10)
        return value

    monkeypatch.setattr(app._webmon, 'binding', delayed)

    async def run():
        task = asyncio.create_task(asyncio.to_thread(app.webmon_select, '7'))
        assert await asyncio.to_thread(entered.wait, 5)
        await app.clear_workspace()
        assert_application_cleared(app)
        release.set()
        await task

    asyncio.run(run())
    assert_application_cleared(app)


def test_failed_recovery_save_keeps_restored_binding_and_inventory(isolated_app, view, tmp_path):
    app = isolated_app
    restored_application(app, view, tmp_path)
    binding = app._webmon_binding
    pattern = app._webmon_pattern
    runs = app.server.state.webmon_runs
    controller = app._powgen_controller

    async def fail_save():
        raise RuntimeError('recovery save unavailable')

    app._stop_session_enrollment = fail_save
    asyncio.run(app.clear_workspace())
    assert 'was not cleared' in app.server.state.workspace_notice
    assert app._powgen_controller is controller
    assert app._webmon_binding is binding and app._webmon_pattern is pattern
    assert app.server.state.webmon_runs == runs
    assert app.server.state.webmon_context_ready
