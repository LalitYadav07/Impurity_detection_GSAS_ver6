"""Completed series updates do not replace a pinned fit or conceal its reference."""
import asyncio
from copy import deepcopy
import threading
from types import SimpleNamespace

import pytest

from radar_pd_nova.refinement_controller import RefinementMixin
from radar_pd_nova.refinement_plots import trend_figure
from test_refinement_plots import view


class State(SimpleNamespace):
    def flush(self): pass


def row(run, temperature, value, *, sample='sample', bank='2', recipe='recipe'):
    final = view(); final['parameters'][0]['value'] = value
    return dict(id=str(run), recipe_id=recipe, recipe_label='Recipe 1',
        source=dict(experiment='exp', sample_key=sample, sample_label=sample, bank=bank,
            run_number=run, metadata={}, axis_overrides={
                'temperature': {'value': temperature, 'unit': 'K', 'reviewed_mapping': True}}),
        baseline=view(), final=final)


def job(value, status):
    return dict(id=value['id'], recipe_id=value['recipe_id'], recipes=[value['recipe_id']],
        source=value['source'], status=status, message=status, attempts=1,
        result={'trend_views': {'baseline': value['baseline'], 'final': value['final']}})


def test_pinned_fit_and_view_choices_survive_completed_arrival_and_failed_scan():
    first = row(64301, 200, 4.)
    second = row(64302, 300, 4.04)
    failed = row(64303, 400, 999.)
    unrelated = row(64304, 500, 999.); unrelated['source']['experiment'] = 'other'
    jobs = {r['id']: job(r, 'completed' if r is first else 'queued') for r in [first, second, failed, unrelated]}
    pending = iter([(second['id'], 'completed'), (failed['id'], 'failed'), (unrelated['id'], 'completed')])
    def run_next():
        item = next(pending, None)
        if item is None: return None
        uid, status = item; jobs[uid]['status'] = status
        return deepcopy(jobs[uid])
    workspace = SimpleNamespace(state={'jobs': jobs, 'recipes': {'recipe': {'version': 1}}},
        lock=threading.RLock(), run_next=run_next, parallel_jobs=1)
    class App(RefinementMixin):
        def _refinement_sync(self): pass
        def _refinement_checkpoint(self): pass
        async def _select_refinement_job(self, uid): raise AssertionError('Pinned fit changed')
        def _read_refinement_view(self, *args): raise AssertionError('Compact trend arrival reloaded heavy views')
        def _refinement_trend_changed(self, **kwargs):
            self.trend_calls += 1
            return super()._refinement_trend_changed(**kwargs)
    app = App(); app._refinement_workspace = workspace; app.trend_calls = 0
    state = State(**app._refinement_state_defaults())
    state.rf_experiment = 'exp'; state.rf_follow = False; state.rf_selected_job = first['id']
    state.rf_trend_metric = 'cell|oxide|c'; state.rf_trend_axis = 'temperature'; state.rf_trend_display = 'relative'
    state.rf_zoom = 17; state.rf_fit = {'pinned': 'unchanged'}; state.rf_stage = 'kept-stage'
    app.server = SimpleNamespace(state=state)
    app._refinement_trend_changed()
    before_fit = deepcopy(state.rf_fit); revision = state.rf_trend['layout']['uirevision']
    assert state.rf_trend_notice.startswith('1 of 1')
    asyncio.run(app._refinement_pump())
    assert state.rf_trend_notice.startswith('2 of 2') and 'scan 64301' in state.rf_trend_notice
    assert '200' in state.rf_trend_notice and 'may change' in state.rf_trend_notice
    assert app.trend_calls == 2  # Initial render and exactly one matching completed arrival.
    assert state.rf_selected_job == first['id'] and state.rf_fit == before_fit
    assert state.rf_zoom == 17 and state.rf_stage == 'kept-stage'
    assert (state.rf_trend_metric, state.rf_trend_axis, state.rf_trend_display) == ('cell|oxide|c', 'temperature', 'relative')
    assert state.rf_trend['layout']['uirevision'] == revision
    assert list(state.rf_trend['data'][0]['x']) == pytest.approx([200, 300])
    assert list(state.rf_trend['data'][0]['y']) == pytest.approx([0, 1])
    assert not state.rf_error


def test_late_earlier_scan_updates_explicit_reference_and_hover():
    rows = [row(64301, 200, 4.04), row(64302, 300, 4.08)]
    before = trend_figure(rows, metric='cell|oxide|c', x_key='temperature', display='relative')
    rows.append(row(64299, 75, 4.))
    after = trend_figure(rows, metric='cell|oxide|c', x_key='temperature', display='relative')
    assert before.layout.meta['references'][0]['run_number'] == 64301
    reference = after.layout.meta['references'][0]
    assert reference['run_number'] == 64299 and reference['coordinate_value'] == 75
    assert list(after.data[0].y) == pytest.approx([0, 1, 2])
    assert 'scan 64299' in after.data[0].hovertext[0] and '75' in after.data[0].hovertext[0]
    assert list(after.data[0].customdata) == ['64299', '64301', '64302']
    assert after.layout.uirevision == before.layout.uirevision


def test_references_are_separate_for_sample_bank_and_recipe_groups():
    records = [row(1, 75, 4.), row(2, 200, 4.04),
               row(3, 90, 8., sample='other', bank='3', recipe='other-recipe')]
    figure = trend_figure(records, metric='cell|oxide|c', x_key='temperature', display='difference')
    references = figure.layout.meta['references']
    assert len(references) == 2
    assert {value['run_number'] for value in references} == {1, 3}
    assert 'sample' in references[0]['group'] and 'bank 2' in references[0]['group']
    assert 'other' in references[1]['group'] and 'bank 3' in references[1]['group']
    assert list(figure.data[1].y) == [0]
