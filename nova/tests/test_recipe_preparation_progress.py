"""Slow preparation reports owned phases without scheduling extra science."""
import asyncio
import copy
import threading
from types import SimpleNamespace

import pytest

from radar_pd_nova import refinement_controller as controller
from radar_pd_nova import refinement_workspace as workspace
from radar_pd_nova.refinement_controller import RefinementMixin
from test_refinement_integration import create, source, plan


class Clock:
    def __init__(self):self.value=1000.
    def monotonic(self):return self.value
    def advance(self,seconds):self.value+=seconds


def test_workspace_phase_durations_do_not_change_the_plan_or_dispatches(tmp_path,monkeypatch):
    w,rt=create(tmp_path/'observed');w.observe([source('1')])
    plain,plain_rt=create(tmp_path/'plain');plain.observe([source('1')])
    clock=Clock();monkeypatch.setattr(workspace,'time',clock)
    original=rt.call;events=[];planner_calls=[]
    def call(action,*args,**kwargs):
        clock.advance({'inspect':2,'preview':7}[action])
        return original(action,*args,**kwargs)
    def planner(request,context,previous=None):
        planner_calls.append(request);clock.advance(5)
        return dict(plan=plan(context),calls=[])
    rt.call=call;w._planner=planner
    plain._planner=lambda request,context,previous=None:dict(plan=plan(context),calls=[])
    result=w.prepare('exp','1','background',progress=lambda event:events.append(copy.deepcopy(event)))
    baseline=plain.prepare('exp','1','background')
    assert [(e['phase'],e['status']) for e in events]==[
        ('starting','running'),('inspect','running'),('planning','running'),
        ('preflight','running'),('preflight','ready')]
    assert events[-1]['elapsed_seconds']==14 and events[-1]['phase_seconds']==7
    assert events[-1]['phases']==[
        {'phase':'starting','seconds':0.}, {'phase':'inspect','seconds':2.},
        {'phase':'planning','seconds':5.}]
    assert 'sampled_monotonic' not in result['preparation']
    assert result['plan']==baseline['plan'] and result['preview']==baseline['preview']
    assert planner_calls==['background']
    assert [(a,s,k) for a,s,k in rt.calls]==plain_rt.calls
    assert not w.state['jobs'] and not w.state['recipes']


def test_cached_inspection_and_a_broken_progress_consumer_do_not_repeat_work(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);events=[]
    w.prepare('exp','1','background',progress=events.append)
    def unavailable_ui(event):raise RuntimeError('closed UI')
    second=w.prepare('exp','1','background',progress=unavailable_ui)
    assert second['status']=='ready'
    assert [c[0] for c in rt.calls]==['inspect','preview','preview']


def test_clarification_has_no_preflight_and_no_success_claim(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);events=[]
    def needs_input(request,context,previous=None):
        value=plan(context);value['questions']=['Which atom?']
        return dict(plan=value,calls=[])
    w._planner=needs_input
    result=w.prepare('exp','1','background',progress=events.append)
    assert [c[0] for c in rt.calls]==['inspect']
    assert result['preparation']['status']=='needs_input'
    assert 'preflight' not in [e['phase'] for e in events]
    assert 'native preflight has not run' in controller.recipe_preparation_text(events[-1])


@pytest.mark.parametrize('failure_phase',['inspect','planning','preflight'])
def test_failed_preparation_retains_its_exact_phase(tmp_path,failure_phase):
    w,rt=create(tmp_path);w.observe([source('1')]);events=[];original=rt.call;planner_calls=[]
    def call(action,*args,**kwargs):
        if {'inspect':'inspect','preview':'preflight'}[action]==failure_phase:
            raise ValueError('bounded failure')
        return original(action,*args,**kwargs)
    def planner(request,context,previous=None):
        planner_calls.append(request)
        if failure_phase=='planning':raise ValueError('bounded failure')
        return dict(plan=plan(context),calls=[])
    rt.call=call;w._planner=planner
    with pytest.raises(ValueError,match='bounded failure'):
        w.prepare('exp','1','background',progress=events.append)
    assert events[-1]['phase']==failure_phase and events[-1]['status']=='error'
    assert w.state['drafts']['exp']['preparation']['status']=='error'
    assert len(planner_calls)==(failure_phase!='inspect')
    assert not w.state['jobs'] and not w.state['recipes']


def test_superseded_generation_cannot_publish_ready_progress(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);events=[];original=rt.call
    def call(action,*args,**kwargs):
        result=original(action,*args,**kwargs)
        if action=='preview':w.discard_draft('exp')
        return result
    rt.call=call
    assert w.prepare('exp','1','background',progress=events.append) is None
    assert 'exp' not in w.state['drafts']
    assert events[-1]['status']=='running'
    assert [c[0] for c in rt.calls]==['inspect','preview']


class State(SimpleNamespace):
    def flush(self):pass


class App(RefinementMixin):
    def __init__(self,w):
        self._refinement_workspace=w;self._refinement_preparations={};self.tasks=[];self.checkpoints=0
        self.server=SimpleNamespace(state=State(rf_busy=False,rf_experiment='exp',rf_source='1',
            rf_request='background',rf_setup_mode='expert',rf_error='',rf_message='',
            rf_preparation_progress='',rf_metrics=[{'label':'Queued / running','value':0}]))
    def _refinement_hold_enrollment(self):pass
    def _refinement_sync(self):
        self._sync_recipe_preparation_progress()
        w=self._refinement_workspace;s=self.server.state
        draft=(w.state['drafts'].get(s.rf_experiment) if w else None) or {}
        s.rf_busy=draft.get('status')=='busy'
    def _refinement_checkpoint(self):self.checkpoints+=1
    def _refinement_schedule(self,coroutine,name):
        task=asyncio.create_task(coroutine,name=name);self.tasks.append(task);return task


class Gates:
    def __init__(self,w,rt):
        self.entered={p:threading.Event() for p in ['inspect','planning','preflight']}
        self.release={p:threading.Event() for p in self.entered};self.planner_calls=[]
        original=rt.call
        def wait(phase):
            self.entered[phase].set()
            assert self.release[phase].wait(5),phase
        def call(action,*args,**kwargs):
            wait({'inspect':'inspect','preview':'preflight'}[action])
            return original(action,*args,**kwargs)
        def planner(request,context,previous=None):
            self.planner_calls.append(request);wait('planning')
            return dict(plan=plan(context),calls=[])
        rt.call=call;w._planner=planner
    def finish(self):
        for event in self.release.values():event.set()


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():await asyncio.sleep(.002)


def test_delayed_phases_tick_without_writes_or_polishing_count_changes(tmp_path,monkeypatch):
    w,rt=create(tmp_path);w.observe([source('1')]);app=App(w);gates=Gates(w,rt);clock=Clock()
    monkeypatch.setattr(workspace,'time',clock);monkeypatch.setattr(controller,'time',clock)
    monkeypatch.setattr(controller,'REFINEMENT_PROGRESS_SECONDS',.01)
    saves=[];save=w.save
    def counted_save():saves.append(1);save()
    w.save=counted_save
    async def run():
        app.generate_refinement_recipe()
        try:
            await until(gates.entered['inspect'].is_set)
            app.generate_refinement_recipe()  # A repeated click cannot duplicate preparation.
            assert len(app.tasks)==1
            saved_count=len(saves);clock.advance(65)
            await until(lambda:'1:05 elapsed' in app.server.state.rf_preparation_progress)
            assert len(saves)==saved_count and app.checkpoints==0
            assert 'inspecting native model' in app.server.state.rf_preparation_progress
            gates.release['inspect'].set();await until(gates.entered['planning'].is_set)
            clock.advance(7)
            await until(lambda:'current step 0:07' in app.server.state.rf_preparation_progress)
            assert 'Planning recipe with Luna' in app.server.state.rf_preparation_progress
            gates.release['planning'].set();await until(gates.entered['preflight'].is_set)
            clock.advance(20)
            await until(lambda:'1:32 elapsed' in app.server.state.rf_preparation_progress)
            assert 'native worker' in app.server.state.rf_preparation_progress
        finally:gates.finish()
        await asyncio.gather(*app.tasks)
    asyncio.run(run())
    assert app.server.state.rf_preparation_progress=='Preparation complete · 1:32 elapsed'
    assert app.server.state.rf_metrics==[{'label':'Queued / running','value':0}]
    assert app.checkpoints==1 and gates.planner_calls==['background']
    assert [c[0] for c in rt.calls]==['inspect','preview']
    assert not app._refinement_preparations and not w.state['jobs']


def test_navigation_back_to_active_request_keeps_progress_without_replanning(tmp_path,monkeypatch):
    w,rt=create(tmp_path);w.observe([source('1'),source('2',exp='other')]);app=App(w);gates=Gates(w,rt)
    monkeypatch.setattr(controller,'REFINEMENT_PROGRESS_SECONDS',.01)
    async def run():
        app.generate_refinement_recipe()
        try:
            await until(gates.entered['inspect'].is_set)
            s=app.server.state;s.rf_experiment='other';s.rf_source='2';s.rf_request='B request';s.rf_message='B view'
            app._refinement_sync();assert not s.rf_preparation_progress
            gates.release['inspect'].set();await until(gates.entered['planning'].is_set)
            assert not s.rf_preparation_progress and s.rf_message=='B view'
            s.rf_experiment='exp';s.rf_source='1';s.rf_request='background';app._refinement_sync()
            assert 'Planning recipe with Luna' in s.rf_preparation_progress
        finally:gates.finish()
        await asyncio.gather(*app.tasks)
    asyncio.run(run())
    assert gates.planner_calls==['background'] and [c[0] for c in rt.calls]==['inspect','preview']
    assert not app._refinement_preparations


@pytest.mark.parametrize('change',['source','request','setup','replace_workspace','clear','generation'])
def test_late_progress_and_completion_cannot_replace_new_context(tmp_path,monkeypatch,change):
    w,rt=create(tmp_path);w.observe([source('1'),source('2')]);app=App(w);gates=Gates(w,rt)
    monkeypatch.setattr(controller,'REFINEMENT_PROGRESS_SECONDS',.01)
    async def run():
        app.generate_refinement_recipe()
        try:
            await until(gates.entered['inspect'].is_set)
            s=app.server.state;s.rf_message='Current selection';s.rf_error='Current error'
            if change=='source':s.rf_source='2'
            elif change=='request':s.rf_request='Changed request'
            elif change=='setup':s.rf_setup_mode='guided'
            elif change=='replace_workspace':app._refinement_workspace=copy.copy(w)
            elif change=='clear':app._refinement_workspace=None
            elif change=='generation':w.state['drafts']['exp']={'generation':'new','source_id':'1','request':'background','status':'ready'}
            app._refinement_sync();assert not s.rf_preparation_progress
        finally:gates.finish()
        await asyncio.gather(*app.tasks)
        assert s.rf_message=='Current selection' and s.rf_error=='Current error'
        assert not s.rf_preparation_progress
    asyncio.run(run())
    assert not app._refinement_preparations
    assert gates.planner_calls==['background'] and [c[0] for c in rt.calls]==['inspect','preview']
    assert app.checkpoints==(change not in {'replace_workspace','clear'})


def test_clear_and_cancel_waiter_does_not_cancel_accepted_preparation_or_publish_later(tmp_path,monkeypatch):
    w,rt=create(tmp_path);w.observe([source('1')]);app=App(w);gates=Gates(w,rt)
    monkeypatch.setattr(controller,'REFINEMENT_PROGRESS_SECONDS',.01)
    async def run():
        app.generate_refinement_recipe();await until(gates.entered['inspect'].is_set)
        try:
            app._refinement_workspace=None;s=app.server.state;s.rf_message='Cleared';s.rf_error=''
            app.tasks[0].cancel()
            with pytest.raises(asyncio.CancelledError):await app.tasks[0]
            assert not app._refinement_preparations and not s.rf_preparation_progress
        finally:gates.finish()
        await until(lambda:w.state['drafts']['exp']['status']=='ready')
        await asyncio.sleep(.02)
        assert s.rf_message=='Cleared' and s.rf_error=='' and not s.rf_preparation_progress
    asyncio.run(run())
    assert gates.planner_calls==['background'] and [c[0] for c in rt.calls]==['inspect','preview']
    assert app.checkpoints==0 and not w.state['jobs']


def test_same_source_id_with_new_bytes_rejects_old_inspection_without_stale_ui(tmp_path,monkeypatch):
    w,rt=create(tmp_path);w.observe([source('1')]);app=App(w);gates=Gates(w,rt)
    monkeypatch.setattr(controller,'REFINEMENT_PROGRESS_SECONDS',.01)
    async def run():
        app.generate_refinement_recipe();await until(gates.entered['inspect'].is_set)
        try:
            w.observe([source('1',checksum='new-bytes')])
            s=app.server.state;s.rf_message='New source bytes';s.rf_error='';app._refinement_sync()
            assert not s.rf_preparation_progress
        finally:gates.finish()
        await asyncio.gather(*app.tasks)
        assert s.rf_message=='New source bytes' and not s.rf_error
        assert not s.rf_preparation_progress
    asyncio.run(run())
    assert w.state['drafts']['exp']['status']=='error'
    assert gates.planner_calls==[] and [c[0] for c in rt.calls]==['inspect']
    assert not app._refinement_preparations


def test_completed_progress_follows_automatic_scope_source_selection(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1'),source('2')]);w.prepare('exp','1','background')
    app=RefinementMixin()
    s=State(**app._refinement_state_defaults());app.server=SimpleNamespace(state=s)
    app._refinement_workspace=w;app._refinement_internal=False
    app._refinement_approving=False;app._refinement_inspecting=False;app._refinement_preparations={}
    app._init_guided_refinement()
    s.rf_experiment='exp';s.rf_source='1';s.rf_setup_mode='expert';s.rf_request='background'
    app._refinement_sync()
    assert s.rf_source=='1' and s.rf_plan_ready and s.rf_preparation_progress.startswith('Preparation complete')
    s.rf_scope='scans';s.rf_scans=['scan-2'];app._refinement_scope_changed()
    assert s.rf_source=='2' and not s.rf_draft_status and not s.rf_plan_ready
    assert not s.rf_preparation_progress
    s.rf_scans=['scan-1'];app._refinement_scope_changed()
    assert s.rf_source=='1' and s.rf_plan_ready and s.rf_preparation_progress.startswith('Preparation complete')
    s.rf_scans=['missing-scan'];app._refinement_scope_changed()
    assert not s.rf_source and not s.rf_preparation_progress and not s.rf_plan_ready
    app._refinement_workspace=None;app._refinement_sync()
    assert not s.rf_preparation_progress
    assert [c[0] for c in rt.calls]==['inspect','preview']


def test_preparation_is_labelled_separately_from_real_polishing_counts():
    from radar_pd_nova.app import RadarPdNovaApp
    html=RadarPdNovaApp().layout.html
    assert 'Recipe preparation is separate from polishing.' in html
    assert '{{ rf_preparation_progress }}' in html
    assert 'metric in rf_metrics' in html
