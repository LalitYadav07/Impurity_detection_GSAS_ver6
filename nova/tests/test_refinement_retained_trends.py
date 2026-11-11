"""Stopped recipes and distinct recipe stages remain interpretable in trends."""
from copy import deepcopy
import asyncio
from types import SimpleNamespace

import pytest

from radar_pd_nova.refinement_controller import RefinementMixin
from test_refinement_trend_arrivals import State, row


def build(records):
    jobs={}
    views={}
    recipes={}
    for index,(run,recipe,stage_id,title,accepted) in enumerate(records):
        data=row(run,100.+index,4.+index,recipe=recipe)
        data['source']['scan_label']=f'Run {run}'
        recipes.setdefault(recipe,dict(version=len(recipes)+1,
            plan={'stages':[dict(id=stage_id,title=title)]}))
        result=dict(status='completed' if accepted else 'needs_review',
            baseline_view=f'{run}-baseline',
            stages=[dict(id=stage_id,title=title,view=f'{run}-stage',gpx=f'{run}-stage.gpx')],
            files={f'{run}-stage':'stage-hash',f'{run}-stage.gpx':'gpx-hash'})
        views[result['baseline_view']]=data['baseline']
        views[f'{run}-stage']=data['final']
        if accepted:
            result.update(final_gpx=f'{run}-final.gpx',final_view=f'{run}-final')
            views[result['final_view']]=data['final']
        jobs[str(run)]=dict(id=str(run),source=data['source'],recipe_id=recipe,
            recipes=[recipe],status=result['status'],message='Retained checkpoint',result=result)

    class App(RefinementMixin):
        def _read_refinement_view(self,job,name):
            self.reads.append(name)
            if name not in views:raise FileNotFoundError(name)
            return deepcopy(views[name])
        def _read_refinement_trend_view(self,job,name):
            from gpx_recipe.worker import trend_view
            return trend_view(self._read_refinement_view(job,name))
    app=App();app.reads=[]
    app.server=SimpleNamespace(state=State(**app._refinement_state_defaults()))
    app._refinement_workspace=SimpleNamespace(state={'jobs':jobs,'recipes':recipes})
    app.server.state.rf_experiment='exp'
    return app,views


def message(state):
    return ' '.join(item.get('text','') for item in state.rf_trend['layout'].get('annotations',[]))


def test_stopped_results_offer_retained_stages_without_claiming_missing_artifacts():
    app,_=build([(1,'r1','cells','Cell refinement',False),(2,'r1','cells','Cell refinement',False)])
    before=deepcopy(app._refinement_workspace.state)
    app._refinement_trend_changed();s=app.server.state
    assert 'no accepted final GPX' in s.rf_trend_notice
    assert 'Select a retained stage' in message(s)
    assert 'Recover' not in s.rf_trend_notice and 'Waiting' not in message(s)
    assert app.reads==[]
    s.rf_trend_stage=next(v['value'] for v in s.rf_trend_stage_options if 'Cell refinement' in v['title'])
    s.rf_trend_axis='temperature';s.rf_trend_metric='cell|oxide|c'
    app._refinement_trend_changed()
    assert s.rf_trend_notice.startswith('2 of 2')
    assert 'intermediate checkpoints' in s.rf_trend_notice
    assert list(s.rf_trend['data'][0]['x'])==pytest.approx([100.,101.])
    assert list(s.rf_trend['data'][0]['y'])==pytest.approx([4.,5.])
    assert app._refinement_workspace.state==before


def test_same_stage_id_in_two_recipes_does_not_combine_different_work():
    app,_=build([(1,'r1','stage1','Background only',False),(2,'r2','stage1','Atomic positions',False)])
    app._refinement_trend_changed();s=app.server.state
    retained=[item for item in s.rf_trend_stage_options if item['value']!='final']
    assert len(retained)==2 and retained[0]['value']!=retained[1]['value']
    assert 'Recipe 1' in retained[0]['title'] and 'Background only' in retained[0]['title']
    assert 'Recipe 2' in retained[1]['title'] and 'Atomic positions' in retained[1]['title']
    for choice,run in zip(retained,['1','2']):
        s.rf_trend_stage=choice['value'];app._refinement_trend_changed()
        assert s.rf_trend_notice.startswith('1 of 1')
        assert {point['job_id'] for trace in s.rf_trend['data'] for point in trace['customdata']}=={run}


def test_filter_change_removes_stale_stage_selection_explicitly():
    app,_=build([(1,'r1','cells','Cell refinement',False),(2,'r2','atoms','Atomic positions',True)])
    app._refinement_trend_changed();s=app.server.state
    s.rf_trend_stage=next(v['value'] for v in s.rf_trend_stage_options if 'Cell refinement' in v['title'])
    s.rf_recipe_filter='r2';app._refinement_trend_changed()
    assert s.rf_trend_stage=='final'
    assert 'previous stage is unavailable' in s.rf_trend_notice
    assert s.rf_trend_notice.startswith('1 of 1')
    assert {point['job_id'] for trace in s.rf_trend['data'] for point in trace['customdata']}=={'2'}


def test_legacy_stage_selection_is_migrated_only_when_unambiguous():
    app,_=build([(1,'r1','cells','Cell refinement',False)])
    s=app.server.state;s.rf_trend_stage='cells';app._refinement_trend_changed()
    assert s.rf_trend_stage!='cells' and s.rf_trend_stage!='final'
    assert s.rf_trend_notice.startswith('1 of 1')
    app,_=build([(1,'r1','stage1','Background only',False),(2,'r2','stage1','Atomic positions',False)])
    s=app.server.state;s.rf_trend_stage='stage1';app._refinement_trend_changed()
    assert s.rf_trend_stage=='final' and 'recipe-specific' in s.rf_trend_notice


def test_missing_published_files_are_distinct_from_no_final_output():
    app,views=build([(1,'r1','cells','Cell refinement',True),(2,'r1','cells','Cell refinement',False)])
    del views['1-final']
    app._refinement_trend_changed();s=app.server.state
    assert '1 saved result(s) could not be loaded' in s.rf_trend_notice
    assert '1 result(s) have no accepted final GPX' in s.rf_trend_notice
    assert 'Recover their saved artifacts' in s.rf_trend_notice


def test_only_pending_work_keeps_a_waiting_message():
    app,_=build([(1,'r1','cells','Cell refinement',False)])
    app._refinement_workspace.state['jobs']['1'].update(status='running',result={})
    app._refinement_trend_changed();s=app.server.state
    assert 'Waiting for a completed refinement' in message(s)
    assert 'no accepted' not in s.rf_trend_notice and 'Recover' not in s.rf_trend_notice


def test_absent_stage_is_not_misreported_as_a_missing_file():
    app,_=build([(1,'r1','cells','Cell refinement',False),(2,'r1','cells','Cell refinement',False)])
    app._refinement_workspace.state['jobs']['2']['result']['stages']=[]
    app._refinement_trend_changed();s=app.server.state
    s.rf_trend_stage=next(v['value'] for v in s.rf_trend_stage_options if 'Cell refinement' in v['title'])
    app._refinement_trend_changed()
    assert s.rf_trend_notice.startswith('1 of 2')
    assert '1 result(s) do not contain this stage' in s.rf_trend_notice
    assert 'Recover' not in s.rf_trend_notice


def retained_point(app):
    app._refinement_trend_changed();s=app.server.state
    s.rf_trend_stage=next(v['value'] for v in s.rf_trend_stage_options if v['value']!='final')
    s.rf_result_tab='trends'
    app._refinement_trend_changed()
    return deepcopy(s.rf_trend['data'][0]['customdata'][0])


def test_trend_click_opens_exact_checkpoint_and_matching_gpx_without_changing_jobs():
    app,_=build([(1,'r1','cells','Cell refinement',False)])
    # Keep the real selection/recovery path; the independent fit renderer is
    # covered by the native-view tests.
    app._refinement_plot_changed=lambda **kwargs:None
    app._refinement_schedule=lambda task,name:asyncio.run(task)
    point=retained_point(app);original=deepcopy(app._refinement_workspace.state)
    app.select_refinement_trend_point({'points':[{'customdata':point}]})
    s=app.server.state
    assert s.rf_selected_job=='1' and s.rf_stage=='cells' and s.rf_result_tab=='fit'
    assert s.rf_download=='1-stage.gpx' and s.rf_has_output and not s.rf_follow
    assert not s.rf_error
    assert app._refinement_workspace.state==original


@pytest.mark.parametrize('changed',['trend_stage','source_sha256','view_sha256','recipe_id'])
def test_stale_or_mismatched_trend_points_do_not_open_a_different_fit(changed):
    app,_=build([(1,'r1','cells','Cell refinement',False)])
    point=retained_point(app);point[changed]='different'
    scheduled=[]
    app._refinement_schedule=lambda *args:scheduled.append(args)
    app.select_refinement_trend_point({'points':[{'customdata':point}]})
    assert scheduled==[] and 'trend has changed' in app.server.state.rf_message


def test_removed_requested_checkpoint_never_falls_back_to_another_stage():
    app,_=build([(1,'r1','cells','Cell refinement',False)])
    asyncio.run(app._select_refinement_job('1',requested_stage='missing',show_fit=True))
    assert not app.server.state.rf_has_output
    assert 'checkpoint is no longer available' in app.server.state.rf_error


@pytest.mark.parametrize('event',[None,{}, {'points':7}, {'points':[None]},
    {'points':[{'customdata':[]}]}, {'points':[{'customdata':{'job_id':[]}}]}])
def test_nonpoint_browser_events_do_not_break_the_review(event):
    app,_=build([(1,'r1','cells','Cell refinement',False)])
    retained_point(app)
    app._refinement_schedule=lambda *args:pytest.fail('A nonpoint event scheduled a selection')
    app.select_refinement_trend_point(event)
    assert not app.server.state.rf_error
