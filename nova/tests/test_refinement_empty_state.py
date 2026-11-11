"""Terminal refinement attempts must not look like unfinished calculations."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from radar_pd_nova.refinement_controller import RefinementMixin
from radar_pd_nova.refinement_messages import refinement_empty_message


@pytest.mark.parametrize('status', ['needs_review', 'completed'])
def test_rejected_recipe_without_a_stage_does_not_promise_a_future_fit(status):
    job = {'status': status, 'result': {'status': 'needs_review', 'stages': []}}
    original = deepcopy(job)
    message = refinement_empty_message(job)
    assert 'finished without an accepted refinement stage' in message
    assert 'Results' in message and 'will appear' not in message
    assert job == original


@pytest.mark.parametrize('status', ['failed', 'interrupted', 'needs_recipe'])
def test_stopped_attempt_points_to_status_before_retrying(status):
    assert 'stopped' in refinement_empty_message({'status': status})


def test_pending_paused_and_unreadable_results_have_distinct_guidance():
    assert 'queued' in refinement_empty_message({'status': 'queued'})
    assert 'Waiting' in refinement_empty_message({'status': 'running'})
    assert 'paused' in refinement_empty_message({'status': 'paused'})
    assert 'could not be opened' in refinement_empty_message(
        {'status': 'completed', 'result': {'final_view': 'view.json'}}, unavailable=True)
    assert 'Select a scan' in refinement_empty_message()


def test_selecting_terminal_rejection_then_pending_scan_updates_empty_state(tmp_path):
    class App(RefinementMixin):
        def _refinement_plot_changed(self): pass
        def _refinement_trend_changed(self): pass

    rejected = dict(id='rejected', status='needs_review', recipe_id='recipe', recipes=['recipe'],
                    source=dict(experiment='exp', scan_label='Scan 1', sample_label='sample'),
                    message='Convergence criterion failed',
                    result=dict(directory=str(tmp_path), status='needs_review', stages=[],
                                error='Convergence criterion failed', files={}))
    pending = dict(id='pending', status='queued', recipe_id='recipe', recipes=['recipe'],
                   source=dict(experiment='exp', scan_label='Scan 2', sample_label='sample'),
                   message='Queued')
    app = App()
    app.server = SimpleNamespace(state=SimpleNamespace(
        **app._refinement_state_defaults(), flush=lambda: None))
    app.server.state.rf_experiment = 'exp'
    app._refinement_workspace = SimpleNamespace(state={
        'jobs': {'rejected': rejected, 'pending': pending},
        'recipes': {'recipe': {'version': 1}}})
    asyncio.run(app._select_refinement_job('rejected'))
    assert not app.server.state.rf_has_output
    assert 'finished without an accepted refinement stage' in app.server.state.rf_empty_message
    assert app.server.state.rf_selected_message == 'Convergence criterion failed'
    asyncio.run(app._select_refinement_job('pending'))
    assert not app.server.state.rf_has_output
    assert 'queued' in app.server.state.rf_empty_message
    app._clear_refinement_output()
    assert 'Select a scan' in app.server.state.rf_empty_message
