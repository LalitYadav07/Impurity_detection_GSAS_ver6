"""First-open measured history reuses saved results without starting science."""
import asyncio
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from radar_pd_nova.post_analysis import PostAnalysisSettings
from test_live_restore import app, checkpoint, inspect, restore
from test_scan_interpretation import packet


def saved_curves(app, tmp_path, *, blocked=False, failing=False):
    controller, _ = checkpoint(app.service, partial=False)
    first = next(iter(controller.state.completed.values()))
    controller.state.completed.clear()
    payloads = {}
    for index in range(6):
        curve = packet(seed=101 + index)
        curve['context'].update(instrument='PG3', source_bank=1)
        raw = json.dumps(curve).encode()
        digest = sha256(raw).hexdigest()
        payloads[digest] = raw
        receipt = {'$schema': curve['$schema'], 'sha256': digest,
                   'source_gpx_sha256': curve['source_gpx_sha256'],
                   'dataset_id': f'curve-{index}', 'local_path': f'/gone/container/curve-{index}.json'}
        run = replace(first, run_number=101 + index,
                      source_path=f'{controller.source_directory}/PG3_{101+index}.gsa',
                      fingerprint=f'fp-{index}', galaxy_job_id=f'job-{index}',
                      scientific_summary={'rwp': 5., 'analysis_mode': 'full', 'phases': [],
                                          'pattern_observation': receipt},
                      scan_metadata={'schema_version': 3, 'sample_id': 'sample-A', 'source_bank': 1})
        controller.state.completed[run.run_id] = run
    settings = PostAnalysisSettings(axis='run_number')
    settings.views[0].temperature_display_unit = 'C'
    controller.state.post_analysis = settings.model_dump(mode='json')
    app.service.payloads['checkpoint-1'] = json.loads(controller.capture_checkpoint().payload_json)
    calls = []
    entered, release = threading.Event(), threading.Event()
    if not blocked:
        release.set()

    def cache(receipt):
        calls.append(receipt['dataset_id'])
        entered.set()
        assert release.wait(5)
        if failing:
            raise OSError('Synthetic measured receipt unavailable')
        path = tmp_path / (receipt['dataset_id'] + '.json')
        path.write_bytes(payloads[receipt['sha256']])
        return {**receipt, 'local_path': str(path)}
    app.service.cache_pattern_observation = cache
    app._refinement_maybe_refresh = lambda: pytest.fail('Tab entry must not enroll refinements')
    asyncio.run(inspect(app))
    restore(app)
    app._test_checkpoint_posts = []
    app._powgen_controller.persist_state = lambda *a, **k: app._test_checkpoint_posts.append(1)
    return calls, entered, release


async def enter(app):
    app.server.state.experiment_result_view = 'transitions'
    app.server.state.flush()
    # Headless Trame states are not marked ready by a browser connection;
    # dispatch the same registered view-change callback explicitly.
    app._watch_transition_view_changed()
    task = getattr(app, '_watch_transition_hydration_task', None)
    assert task is not None, 'Opening a cold restored transition view must load its saved curves'
    await task
    if app._transition_task is not None:
        await app._transition_task
    if getattr(app, '_transition_save_task', None) is not None:
        await app._transition_save_task
    assert app._test_checkpoint_posts == []


def test_first_transition_open_hydrates_all_six_and_warm_toggle_reuses_them(app, tmp_path):
    calls, _, _ = saved_curves(app, tmp_path)
    assert calls == [] and app._transition_report['records'] == []
    assert len(app.server.state.powgen_scientific_rows) == 6
    original_jobs = {key: run.galaxy_job_id for key, run in app._powgen_controller.state.completed.items()}
    async def exercise():
        await enter(app)
        assert len(app._transition_report['records']) == 6
        assert app.server.state.transition_series_options
        before = deepcopy(app._transition_report)
        app.server.state.experiment_result_view = 'results'
        app.server.state.flush()
        app._watch_transition_view_changed()
        app.server.state.experiment_result_view = 'transitions'
        app.server.state.flush()
        app._watch_transition_view_changed()
        assert app._transition_task is None
        assert app._transition_report == before
    asyncio.run(exercise())
    assert sorted(calls) == [f'curve-{i}' for i in range(6)]
    assert app._post_settings().axis == 'run_number'
    assert app._post_settings().views[0].temperature_display_unit == 'C'
    assert not app.server.state.powgen_monitoring
    assert {key: run.galaxy_job_id for key, run in app._powgen_controller.state.completed.items()} == original_jobs


def test_quick_duplicate_entries_share_one_pending_read(app, tmp_path):
    calls, entered, release = saved_curves(app, tmp_path, blocked=True)
    async def exercise():
        state = app.server.state
        state.experiment_result_view = 'transitions'; state.flush()
        app._watch_transition_view_changed()
        task = app._watch_transition_hydration_task
        assert await asyncio.to_thread(entered.wait, 2)
        assert state.transition_hydration_loading
        assert '6 completed scans' in state.transition_hydration_message
        state.experiment_result_view = 'results'; state.flush()
        app._watch_transition_view_changed()
        state.experiment_result_view = 'transitions'; state.flush()
        app._watch_transition_view_changed()
        assert app._watch_transition_hydration_task is task
        release.set(); await task
        if app._transition_task is not None: await app._transition_task
    asyncio.run(exercise())
    assert sorted(calls) == [f'curve-{i}' for i in range(6)]
    assert not app.server.state.transition_hydration_loading


@pytest.mark.parametrize('change', ['controller', 'service', 'context', 'navigation'])
def test_late_old_receipts_do_not_change_a_new_result_scope(app, tmp_path, change):
    _, entered, release = saved_curves(app, tmp_path, blocked=True)
    async def exercise():
        state = app.server.state
        state.experiment_result_view = 'transitions'; state.flush()
        app._watch_transition_view_changed()
        task = app._watch_transition_hydration_task
        assert await asyncio.to_thread(entered.wait, 2)
        state.workspace_view = 'experiment' if change == 'navigation' else 'results'
        state.experiment_result_view = 'refinement'
        if change == 'controller': app._powgen_controller = None
        elif change == 'service': app.service = object()
        elif change == 'context': app._result_context = object()
        owner = SimpleNamespace()
        app._activate_post_analysis_scope('single:new-result', owner=owner)
        before = deepcopy(state.post_analysis)
        state.flush(); release.set(); await task
        assert app._post_analysis_scope == 'single:new-result'
        assert app._post_analysis_scope_owner is owner
        assert state.post_analysis == before
        assert app._post_analysis_scans == []
        assert app._transition_report['records'] == []
    asyncio.run(exercise())


def test_unavailable_saved_receipts_show_retry_without_repeated_gets(app, tmp_path):
    calls, _, _ = saved_curves(app, tmp_path, failing=True)
    async def exercise():
        await enter(app)
        state = app.server.state
        assert not state.transition_hydration_loading
        assert 'Refresh saved results' in state.transition_hydration_error
        state.experiment_result_view = 'results'; state.flush()
        app._watch_transition_view_changed()
        state.experiment_result_view = 'transitions'; state.flush()
        app._watch_transition_view_changed()
        assert app._watch_transition_hydration_task is None
    asyncio.run(exercise())
    assert len(calls) == 6 and not app.server.state.powgen_monitoring


def test_observation_refresh_does_not_poll_unrelated_active_jobs(app, tmp_path):
    calls, _, _ = saved_curves(app, tmp_path)
    controller = app._powgen_controller
    existing = next(iter(controller.state.completed.values()))
    active = replace(existing, run_number=200, galaxy_job_id='active-job', scientific_summary={})
    controller.state.submitted[active.run_id] = active
    # SavedHistory rejects every unspecified service method, including job GETs.
    controller.refresh(checkpoint=False, observations_only=True)
    assert len(calls) == 6
    assert controller.state.submitted == {active.run_id: active}


def test_first_open_fetches_only_the_remaining_missing_curves(app, tmp_path):
    calls, _, _ = saved_curves(app, tmp_path)
    app._powgen_controller.refresh(checkpoint=False)
    assert len(calls) == 2  # Ordinary live-tick fairness remains unchanged.
    already_loaded = set(calls)
    calls.clear()
    asyncio.run(enter(app))
    assert len(calls) == 4
    assert not already_loaded.intersection(calls)
    assert len(app._transition_report['records']) == 6


def test_a_removed_cached_curve_can_be_loaded_again_after_success(app, tmp_path):
    calls, _, _ = saved_curves(app, tmp_path)
    async def exercise():
        await enter(app)
        run = next(iter(app._powgen_controller.state.completed.values()))
        Path(run.scientific_summary['pattern_observation']['local_path']).unlink()
        calls.clear()
        app.server.state.experiment_result_view = 'results'
        app._watch_transition_view_changed()
        await enter(app)
    asyncio.run(exercise())
    assert calls == ['curve-0']


def test_explicit_boundary_edit_still_persists_after_read_only_activation(app, tmp_path):
    saved_curves(app, tmp_path)
    async def exercise():
        await enter(app)
        app.update_transition_policy('width_percent', 5)
        if app._transition_task is not None:
            await app._transition_task
        if app._transition_save_task is not None:
            await app._transition_save_task
    asyncio.run(exercise())
    assert app._test_checkpoint_posts
