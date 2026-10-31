"""Enrollment scope stays fixed while acquisition and viewer state change."""
import asyncio
import copy
from types import SimpleNamespace
import pytest

from gpx_recipe.symmetry.campaign import Campaign
from radar_pd_nova import model_campaigns as module
from radar_pd_nova.symmetry_controller import SymmetryMixin
from radar_pd_nova.symmetry_workspace import campaign_can_resume
from radar_pd_nova.model_runtime import GalaxyChunkTransport
from radar_pd_nova.model_runtime import CollectOnlyTransport
from radar_pd_nova.model_runtime import PreparationSlotTransport


def setup(tmp_path,monkeypatch):
    c=Campaign(tmp_path/'campaign')
    # Contract and native compatibility have independent worker tests; this
    # fixture isolates the queue's historical/future acquisition policy.
    t=dict(scope=dict(experiment='IPTS-1',sample_key='specimen:A',bank='2'),origin='radar',
        existing_ids=['selected'],seen_ids=['selected','unselected'],maximum_scans=20,future=True)
    c.set_setting('template',t);c.set_setting('following',True)
    monkeypatch.setattr(module,'validate_template',lambda template:template)
    def source(key,**kwargs):
        return dict(id=key,experiment='IPTS-1',sample_key='specimen:A',bank='2',scan_label=key,sample_label='A',**kwargs)
    rows=[(source('selected'),'radar'),(source('unselected'),'radar'),(source('new'),'radar')]
    for field,value in [('experiment','IPTS-2'),('sample_key','specimen:B'),('bank','3')]:
        row=source('wrong-'+field);row[field]=value;rows.append((row,'radar'))
    rows.append((source('other-recipe'),'recipe:other:final'))
    monkeypatch.setattr(module,'choices',lambda *args:rows)
    return c,rows


def test_future_following_adds_only_new_matching_scans_not_old_unselected(tmp_path,monkeypatch):
    c,rows=setup(tmp_path,monkeypatch)
    assert set(module.queue_sources(c,None))=={'selected','new'}
    assert set(module.queue_sources(c,None))=={'selected','new'}
    c.set_setting('following',False)
    rows.append((dict(rows[0][0],id='later'),'radar'))
    assert set(module.queue_sources(c,None))=={'selected','new'}


def test_budget_stops_following_with_visible_reason(tmp_path,monkeypatch):
    c,rows=setup(tmp_path,monkeypatch);t=c.setting('template');t['maximum_scans']=1;c.set_setting('template',t)
    assert set(module.queue_sources(c,None))=={'selected'}
    assert c.setting('following') is False and 'budget reached' in c.setting('budget_message')


def test_completed_inspection_does_not_erase_scans_arriving_during_its_wait(tmp_path,monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    c,rows=setup(tmp_path,monkeypatch)
    entry=module.queue_sources(c,None)['selected']
    module._set_enrollment(c,'selected',dict(entry,status='preparing'))
    entered=threading.Event();release=threading.Event()
    def complete():
        captured=copy.deepcopy(entry);entered.set();assert release.wait(5)
        captured['status']='enrolled';module._set_enrollment(c,'selected',captured)
    with ThreadPoolExecutor(max_workers=1)as pool:
        task=pool.submit(complete);assert entered.wait(5)
        rows.append((dict(rows[0][0],id='arrived-during-inspection'),'radar'))
        assert 'arrived-during-inspection' in module.queue_sources(c,None)
        release.set();task.result(timeout=5)
    saved=c.setting('enrollments')
    assert saved['selected']['status']=='enrolled'
    assert saved['arrived-during-inspection']['status']=='queued' and 'new' in saved


def test_recover_queued_acquisitions_before_the_first_trial_is_created(tmp_path,monkeypatch):
    c,_=setup(tmp_path,monkeypatch);c.set_setting('following',False)
    entries=module.queue_sources(c,None)
    assert not c.trials() and not c.summaries()
    assert campaign_can_resume(c.summaries(),list(entries.values()))
    entries['selected']['status']='preparing'
    assert campaign_can_resume(c.summaries(),list(entries.values()))
    entries['selected']['status']='needs_review'
    assert not campaign_can_resume(c.summaries(),list(entries.values()))


def test_session_close_prevents_dispatch_and_retains_follow_policy_for_explicit_continue(tmp_path,monkeypatch):
    c,rows=setup(tmp_path,monkeypatch)
    app=SymmetryMixin();app._model_preparations={};app._k_cancel=None;app._symmetry_task=None
    app._model_enrollment_task=None;app._model_tick_task=None;app._model_remote=lambda:None
    app._model_active_campaigns={'campaign':(c,SimpleNamespace())}
    asyncio.run(app._pause_model_campaigns())
    assert app._model_stopping and c.setting('paused') and c.setting('following')
    assert module.queue_sources(c,None)=={}


def closing_app(c,t):
    app=SymmetryMixin();app._model_preparations={};app._k_cancel=None
    app._model_enrollment_task=None;app._model_tick_task=None;app._symmetry_task=None;app._model_pause_task=None
    app._model_remote=lambda:None;app._model_active_campaigns={}
    app._model_owned_campaigns={c.root.name:(c,t)};app._model_checkpoint_errors={}
    return app


def test_owned_transport_is_reused_for_refresh_and_resume(tmp_path,monkeypatch):
    monkeypatch.delenv('RADAR_PD_SYMMETRY_PYTHON',raising=False)
    c=SimpleNamespace(root=tmp_path/'owned')
    t=object.__new__(GalaxyChunkTransport);t.max_active_jobs=1;t.state={'unposted-proof':'retained'}
    app=closing_app(c,t);app._model_remote=lambda:SimpleNamespace(max_active_jobs=8)
    assert app._symmetry_transport(c) is t and t.max_active_jobs==8
    assert t.state=={'unposted-proof':'retained'}


def test_refresh_keeps_verified_receipt_cache_with_a_bounded_archive_window(tmp_path,monkeypatch):
    import radar_pd_nova.symmetry_controller as controller
    calls=[]
    def create(root):
        calls.append(root);return SimpleNamespace(root=root,_terminal_checks={})
    monkeypatch.setattr(controller,'Campaign',create)
    app=SymmetryMixin();app._model_owned_campaigns={}
    first=app._symmetry_campaign(tmp_path/'first')
    first._terminal_checks['verified']='stamp-checked'
    assert app._symmetry_campaign(tmp_path/'first') is first
    assert app._symmetry_campaign(tmp_path/'first')._terminal_checks['verified']=='stamp-checked'
    app._symmetry_campaign(tmp_path/'second');app._symmetry_campaign(tmp_path/'third')
    assert len(app._model_review_campaigns)==2
    assert app._symmetry_campaign(tmp_path/'first') is not first and len(calls)==4
    owned=SimpleNamespace(root=tmp_path/'owned',_terminal_checks={})
    app._model_owned_campaigns={'owned':(owned,object())}
    assert app._symmetry_campaign(tmp_path/'owned') is owned and len(calls)==4


def test_close_backs_up_owned_campaign_even_after_its_driver_stopped(tmp_path):
    events=[];settings={}
    c=SimpleNamespace(root=tmp_path/'stopped',set_setting=lambda k,v:settings.update({k:v}))
    class T(GalaxyChunkTransport):
        def __init__(self):pass
        def pause(self):events.append('paused')
        def checkpoint(self,**kwargs):events.append('saved')
    app=closing_app(c,T())
    asyncio.run(app._pause_model_campaigns())
    assert events==['paused','saved'] and settings['paused']


def test_close_waits_for_model_settlement_before_saving_unposted_proof(tmp_path):
    events=[];proof={}
    c=SimpleNamespace(root=tmp_path/'settling',set_setting=lambda *a:None)
    class T(GalaxyChunkTransport):
        def __init__(self):pass
        def pause(self):events.append('paused')
        def checkpoint(self,**kwargs):assert proof.get('not_dispatched');events.append('saved')
    app=closing_app(c,T())
    async def run():
        release=asyncio.Event()
        async def driver():
            await release.wait();proof['not_dispatched']=True;events.append('settled')
        app._symmetry_task=asyncio.create_task(driver())
        end=asyncio.create_task(app._pause_model_campaigns());await asyncio.sleep(0)
        assert not end.done() and events==['paused']
        release.set();await end
    asyncio.run(run())
    assert events==['paused','settled','saved']


@pytest.mark.parametrize('draining_task',['_symmetry_task','_model_pause_task'])
def test_close_does_not_stack_a_second_pending_model_checkpoint_wait(tmp_path,draining_task):
    from radar_pd_nova.checkpoint_uploads import CheckpointUploadPending
    calls=[];c=SimpleNamespace(root=tmp_path/'pending',set_setting=lambda *a:None)
    class T(GalaxyChunkTransport):
        def __init__(self):pass
        def pause(self):pass
        def checkpoint(self,**kwargs):calls.append('explicit retry')
    app=closing_app(c,T())
    async def run():
        release=asyncio.Event()
        async def driver():
            await release.wait()
            app._model_checkpoint_errors[c.root.name]=CheckpointUploadPending('hda1','queued')
        setattr(app,draining_task,asyncio.create_task(driver()))
        end=asyncio.create_task(app._pause_model_campaigns());await asyncio.sleep(0)
        release.set()
        with pytest.raises(CheckpointUploadPending):await end
        assert not calls
        await app._pause_model_campaigns()  # A later explicit retry is permitted.
    asyncio.run(run())
    assert calls==['explicit retry'] and not app._model_checkpoint_errors


def test_switching_exploration_kind_removes_stale_candidate_choices():
    s=SimpleNamespace(sx_kind='structural',sx_parent='Parent',sx_k='0, 0, 0',sx_start_source='scan',
        sx_opd_policy='all',sx_construct_limit=50,sx_discovered=True,sx_candidates=['old'],sx_candidate_options=['old'])
    app=SymmetryMixin();app.server=SimpleNamespace(state=s)
    app._symmetry_draft={'discovery_identity':('magnetic','Parent','0, 0, 0','scan')}
    app._symmetry_unapprove()
    assert not s.sx_discovered and not s.sx_candidates and not s.sx_candidate_options


def test_resource_budget_updates_dispatch_limits_without_changing_scientific_approval():
    runtime=SimpleNamespace(max_active_jobs=1)
    transport=object.__new__(GalaxyChunkTransport);transport.runtime=runtime;transport.max_active_jobs=1
    state=SimpleNamespace(sx_parallel_jobs=2,sx_error='')
    app=SymmetryMixin();app.server=SimpleNamespace(state=state);app._model_runtime_cache=runtime
    app._model_active_campaigns={'campaign':(None,transport)};app._symmetry_approved='sealed-science'
    app._symmetry_resource_changed()
    assert runtime.max_active_jobs==transport.max_active_jobs==2
    assert app._symmetry_approved=='sealed-science'
    state.sx_parallel_jobs=16;app._symmetry_resource_changed()
    assert '1, 2, 4 or 8' in state.sx_error and runtime.max_active_jobs==2
    for count in [4,8]:
        state.sx_parallel_jobs=count;app._symmetry_resource_changed()
        assert runtime.max_active_jobs==transport.max_active_jobs==count
        assert app._symmetry_approved=='sealed-science'


def test_reserved_preparation_slot_tracks_a_budget_lowered_during_a_wave():
    base=SimpleNamespace(max_active_jobs=8)
    view=PreparationSlotTransport(base)
    assert view.max_active_jobs==7
    base.max_active_jobs=2;assert view.max_active_jobs==1
    base.max_active_jobs=1;assert view.max_active_jobs==0


def test_failed_magnetic_setup_keeps_token_cost_and_clears_previous_planning_notes(monkeypatch):
    import radar_pd_nova.symmetry_controller as controller
    state=SimpleNamespace(sx_start_source='scan',sx_request='Find magnetic k',sx_k='',
        sx_planning_notes=['Previous accepted note'],sx_planning_summary='Previous accepted plan',flush=lambda:None)
    app=SymmetryMixin();app.server=SimpleNamespace(state=state)
    app._symmetry_draft=dict(source_id='scan',context={})
    app._refinement_schedule=lambda coroutine,name:asyncio.run(coroutine)
    def failed(*args,**kwargs):
        error=ValueError('Invalid ionic form factor for Fe')
        error.planning_calls=[dict(provider='illumine-luna',model='gpt-6-luna',
            usage=dict(input_tokens=700,output_tokens=100,total_tokens=800))]
        raise error
    monkeypatch.setattr(controller,'make_intent',failed)
    app.plan_symmetry_request()
    assert state.sx_error=='Invalid ionic form factor for Fe'
    assert not state.sx_busy and state.sx_planning_notes==[]
    assert 'no settings accepted' in state.sx_planning_summary
    assert '800 tokens' in state.sx_planning_usage and 'Approximate Luna' in state.sx_planning_cost


def test_continue_reports_recovery_before_waiting_for_remote_results(tmp_path,monkeypatch):
    import radar_pd_nova.symmetry_controller as controller
    root=tmp_path/'campaign';root.mkdir();(root/'campaign.sqlite').touch()
    flushes=[]
    state=SimpleNamespace(sx_campaign='campaign',sx_busy=False,sx_error='',sx_message='',
        flush=lambda:flushes.append((state.sx_busy,state.sx_message)))
    app=SymmetryMixin();app.server=SimpleNamespace(state=state);app._symmetry_task=None
    app._symmetry_root=lambda:tmp_path;app._symmetry_transport=lambda campaign:object()
    app._refinement_store=lambda:object()
    monkeypatch.setattr(controller,'restore_captured_sources',lambda *args:None)
    def recover(transport):
        assert state.sx_busy and flushes
        assert 'before dispatching' in state.sx_message
        raise RuntimeError('Remote result temporarily unavailable')
    monkeypatch.setattr(controller,'Campaign',lambda *args,**kwargs:SimpleNamespace(recover=recover))
    app._refinement_schedule=lambda coroutine,name:asyncio.run(coroutine)
    app.continue_symmetry_campaign()
    assert not state.sx_busy and state.sx_error=='Remote result temporarily unavailable'


def test_queued_source_capture_survives_a_new_viewer_without_trusting_local_paths(tmp_path,monkeypatch):
    c,rows=setup(tmp_path,monkeypatch)
    for source,kind in rows:
        source.update(collection_id='collection-'+source['id'],collection_name='final.gpx',
            internal_source_sha256='original',published_sha256='published',source_path='/old/viewer/source.gpx',
            context={'unsafe':'must re-inspect'})
    entries=module.queue_sources(c,None)
    source=entries['selected']['source']
    assert 'source_path' not in source and 'context' not in source
    # The portable ledger carries the exact source identity across viewers.
    snapshot=c.snapshot();restored=Campaign(tmp_path/'new-viewer');
    for key,value in snapshot['settings'].items():restored.set_setting(key,value)
    workspace=SimpleNamespace(state={'sources':{}},runtime=SimpleNamespace(kind='galaxy'),
        observe=lambda sources:workspace.state['sources'].update({s['id']:s for s in sources}))
    assert module.restore_captured_sources(restored,workspace)==2
    assert set(workspace.state['sources'])=={'selected','new'}
    assert workspace.state['sources']['selected']['collection_id']=='collection-selected'
    bad=copy.deepcopy(entries);bad['selected']['source']['experiment']='IPTS-2';restored.set_setting('enrollments',bad)
    with pytest.raises(ValueError,match='outside the reviewed'):module.restore_captured_sources(restored,workspace)
    unselected=copy.deepcopy(entries['selected']);unselected['id']='unselected';unselected['source']['id']='unselected'
    with pytest.raises(ValueError,match='not selected'):module.captured_source(unselected,c.setting('template'))
    restored.set_setting('enrollments',entries)
    workspace.state['sources']['selected']=dict(source,collection_id='replacement')
    with pytest.raises(ValueError,match='differs'):module.restore_captured_sources(restored,workspace)
    workspace.state['sources']['selected']=dict(source,live_watch=True,sample_key='specimen:B')
    with pytest.raises(ValueError,match='reviewed acquisition mapping changed'):module.restore_captured_sources(restored,workspace)
    workspace.runtime.kind='local'
    assert module.restore_captured_sources(restored,workspace)==0


@pytest.mark.parametrize('slot_frees',[True,False])
def test_waiting_acquisition_gets_a_freed_slot_before_more_old_starts(tmp_path,monkeypatch,slot_frees):
    import json
    import radar_pd_nova.symmetry_controller as controller
    events=[];entry=dict(id='new-scan',scan='New acquisition',status='queued')
    rows=[dict(id=str(i),state='running',handle=json.dumps({'dispatch_key':str(i)})) for i in range(2)]
    class C:
        root=tmp_path/'campaign'
        def __init__(self):self.settings={'template':{'future':False},'enrollments':{'new-scan':entry}};self.polls=0
        def setting(self,key,default=None):return self.settings.get(key,default)
        def set_setting(self,key,value):self.settings[key]=value
        def trials(self):return rows
        def tick(self,transport,**kwargs):
            if isinstance(transport,CollectOnlyTransport):
                self.polls+=1;events.append('collect')
                if slot_frees:rows[0]['state']='terminal'
                elif self.polls>=2:self.settings['paused']=True
            else:
                events.append('admit more starts');self.settings['paused']=True
            return rows
    c=C();transport=SimpleNamespace(max_active_jobs=2)
    monkeypatch.setattr(controller,'queue_sources',lambda *args:{'new-scan':entry})
    def enroll(*args,**kwargs):events.append('prepare new scan');entry['status']='enrolled'
    monkeypatch.setattr(controller,'enroll_source',enroll)
    app=SymmetryMixin();app._model_active_campaigns={};app._model_stopping=False
    app._refinement_store=lambda:None;app._inspect_model_input=lambda *args:None;app.refresh_symmetry=lambda:None
    app.server=SimpleNamespace(state=SimpleNamespace(sx_slots=2,flush=lambda:None))
    asyncio.run(app._drive_symmetry_campaign(c,transport))
    if slot_frees:
        assert events==['collect','prepare new scan','admit more starts']
        assert entry['status']=='enrolled'
    else:
        assert events==['collect','collect'] and entry['status']=='queued'
    assert transport.max_active_jobs==2


@pytest.mark.parametrize('budget',[1,2,4,8])
def test_scan_preparation_does_not_block_other_refinements_or_exceed_job_budget(tmp_path,monkeypatch,budget):
    import json
    import threading
    import radar_pd_nova.symmetry_controller as controller
    entered=threading.Event();release=threading.Event();refilled=threading.Event()
    entry=dict(id='next-scan',scan='Next acquisition',status='queued');limits=[];rows=[]
    class C:
        root=tmp_path/'campaign'
        def __init__(self):self.settings={'template':{},'enrollments':{'next-scan':entry}}
        def setting(self,key,default=None):return self.settings.get(key,default)
        def set_setting(self,key,value):self.settings[key]=value
        def trials(self):return rows
        def tick(self,transport,**kwargs):
            if isinstance(transport,PreparationSlotTransport):
                limits.append(transport.max_active_jobs)
                while len(rows)<transport.max_active_jobs:
                    i=len(rows);rows.append(dict(id=str(i),state='running',handle=json.dumps({'dispatch_key':str(i)})))
                refilled.set()
            elif not isinstance(transport,CollectOnlyTransport) and entry['status']=='enrolled':
                limits.append(transport.max_active_jobs);self.settings['paused']=True
            return rows
    c=C();c.settings['template']={'future':False}
    transport=SimpleNamespace(max_active_jobs=budget,submit_many=lambda *a,**kw:None)
    monkeypatch.setattr(controller,'queue_sources',lambda *a:{'next-scan':entry})
    def inspect(*a,**kw):
        entry['status']='preparing';entered.set();assert release.wait(5)
        entry['status']='enrolled'
    monkeypatch.setattr(controller,'enroll_source',inspect)
    original_sleep=asyncio.sleep
    async def short_sleep(seconds):await original_sleep(min(seconds,.01))
    monkeypatch.setattr(controller.asyncio,'sleep',short_sleep)
    app=SymmetryMixin();app._model_active_campaigns={};app._model_stopping=False;app._model_preparations={}
    app._refinement_store=lambda:None;app._inspect_model_input=lambda *a:None;app.refresh_symmetry=lambda:None
    app.server=SimpleNamespace(state=SimpleNamespace(sx_slots=2,flush=lambda:None))
    async def exercise():
        task=asyncio.create_task(app._drive_symmetry_campaign(c,transport))
        try:
            assert await asyncio.to_thread(entered.wait,3)
            assert await asyncio.to_thread(refilled.wait,3)
            assert entry['status']=='preparing' and len(rows)==budget-1
            assert max(limits)==budget-1
        finally:release.set()
        await asyncio.wait_for(task,5)
    asyncio.run(exercise())
    assert limits[-1]==budget and entry['status']=='enrolled'
    assert transport.max_active_jobs==budget and app._model_enrollment_task is None
    assert not app.server.state.sx_campaign_running


def test_preparation_failure_clears_active_campaign_state(tmp_path,monkeypatch):
    import radar_pd_nova.symmetry_controller as controller
    entry=dict(id='bad',scan='Bad acquisition',status='queued')
    c=SimpleNamespace(root=tmp_path/'campaign',setting=lambda k,d=None:{'template':{'future':False}}.get(k,d),
                      set_setting=lambda *a:None,trials=lambda:[],tick=lambda *a,**kw:[])
    monkeypatch.setattr(controller,'queue_sources',lambda *a:{'bad':entry})
    def fail(*a,**kw):raise ValueError('Invalid captured source')
    monkeypatch.setattr(controller,'enroll_source',fail)
    app=SymmetryMixin();app._model_active_campaigns={};app._model_stopping=False;app._model_preparations={}
    app._refinement_store=lambda:None;app._inspect_model_input=lambda *a:None;app.refresh_symmetry=lambda:None
    app.server=SimpleNamespace(state=SimpleNamespace(sx_slots=2,flush=lambda:None))
    with pytest.raises(ValueError,match='Invalid captured source'):
        asyncio.run(app._drive_symmetry_campaign(c,SimpleNamespace(max_active_jobs=2,submit_many=lambda *a:None)))
    assert not app.server.state.sx_campaign_running and not app._model_active_campaigns
    assert app._model_enrollment_task is None


@pytest.mark.parametrize('budget',[1,8])
def test_preparation_finishing_during_empty_tick_does_not_end_campaign(tmp_path,monkeypatch,budget):
    import threading
    import radar_pd_nova.symmetry_controller as controller
    release=threading.Event();finished=threading.Event();events=[]
    entry=dict(id='first',scan='First acquisition',status='queued')
    class C:
        root=tmp_path/'campaign'
        def __init__(self):self.settings={'template':{'future':False}}
        def setting(self,key,default=None):return self.settings.get(key,default)
        def set_setting(self,key,value):self.settings[key]=value
        def trials(self):return []
        def tick(self,transport,**kwargs):
            if isinstance(transport,PreparationSlotTransport):
                events.append('old empty snapshot');release.set();assert finished.wait(3)
            elif not isinstance(transport,CollectOnlyTransport):
                events.append('newly enrolled work polled');self.settings['paused']=True
            return []
    c=C()
    monkeypatch.setattr(controller,'queue_sources',lambda *a:{'first':entry})
    def inspect(*a,**kw):
        entry['status']='preparing';assert release.wait(3)
        entry['status']='enrolled';finished.set()
    monkeypatch.setattr(controller,'enroll_source',inspect)
    original_sleep=asyncio.sleep
    async def short_sleep(seconds):await original_sleep(0)
    monkeypatch.setattr(controller.asyncio,'sleep',short_sleep)
    app=SymmetryMixin();app._model_active_campaigns={};app._model_stopping=False;app._model_preparations={}
    app._refinement_store=lambda:None;app._inspect_model_input=lambda *a:None;app.refresh_symmetry=lambda:None
    app.server=SimpleNamespace(state=SimpleNamespace(sx_slots=2,flush=lambda:None))
    asyncio.run(app._drive_symmetry_campaign(c,SimpleNamespace(max_active_jobs=budget,submit_many=lambda *a:None)))
    assert events==['old empty snapshot','newly enrolled work polled']
    assert entry['status']=='enrolled' and app._model_enrollment_task is None


def test_driver_fills_all_free_model_slots_in_one_bounded_wave(tmp_path):
    import json
    rows=[dict(id=str(i),state='prepared')for i in range(12)];ticks=[];settings={}
    class C:
        root=tmp_path/'wave'
        def setting(self,k,d=None):return settings.get(k,d)
        def set_setting(self,k,v):settings[k]=v
        def trials(self):return rows
        def tick(self,transport,**kwargs):
            ticks.append(len(ticks))
            active=sum(r['state']=='running'for r in rows)
            if active<transport.max_active_jobs:
                row=next(r for r in rows if r['state']=='prepared')
                row.update(state='running',handle=json.dumps(dict(dispatch_key=row['id'])))
            return rows
    app=SymmetryMixin();app._model_active_campaigns={};app._model_stopping=False
    app._refinement_store=lambda:None
    def refresh():
        if sum(r['state']=='running'for r in rows)==8:settings['paused']=True
    app.refresh_symmetry=refresh
    app.server=SimpleNamespace(state=SimpleNamespace(sx_slots=2,flush=lambda:None))
    asyncio.run(app._drive_symmetry_campaign(C(),SimpleNamespace(max_active_jobs=8)))
    assert len(ticks)==8 and sum(r['state']=='running'for r in rows)==8
    assert sum(r['state']=='prepared'for r in rows)==4


def test_galaxy_driver_flushes_one_durable_wave_before_its_acknowledgment_checkpoint(tmp_path):
    import json
    events=[];rows=[dict(id=str(i),state='prepared') for i in range(12)];settings={}
    class C:
        root=tmp_path/'galaxy-wave'
        def setting(self,k,d=None):return settings.get(k,d)
        def set_setting(self,k,v):settings[k]=v
        def trials(self):return rows
        def tick(self,transport,**kwargs):
            events.append('stage')
            row=next(r for r in rows if r['state']=='prepared')
            row.update(state='running',handle=json.dumps(dict(dispatch_key=row['id'])))
            return rows
    class T(GalaxyChunkTransport):
        max_active_jobs=8
        def __init__(self):pass
        def begin_dispatch_wave(self):events.append('begin')
        def flush_dispatch_wave(self):events.append('durable-before-post');settings['paused']=True
        def checkpoint(self,**kwargs):events.append('ack')
    app=SymmetryMixin();app._model_active_campaigns={};app._model_stopping=False
    app._refinement_store=lambda:None;app.refresh_symmetry=lambda:None
    app.server=SimpleNamespace(state=SimpleNamespace(sx_slots=2,flush=lambda:None))
    asyncio.run(app._drive_symmetry_campaign(C(),T()))
    assert events==['begin']+['stage']*8+['durable-before-post','ack']
    assert sum(r['state']=='running' for r in rows)==8


def test_canceling_driver_waits_for_its_shielded_flush_before_aborting(tmp_path):
    import json
    import threading
    entered=threading.Event();release=threading.Event();events=[]
    rows=[dict(id=str(i),state='prepared') for i in range(2)]
    class C:
        root=tmp_path/'cancel-wave'
        def setting(self,k,d=None):return d
        def set_setting(self,k,v):pass
        def trials(self):return rows
        def tick(self,transport,**kwargs):
            row=next(r for r in rows if r['state']=='prepared')
            row.update(state='running',handle=json.dumps(dict(dispatch_key=row['id'])))
            return rows
    class T(GalaxyChunkTransport):
        max_active_jobs=2
        def __init__(self):self.paused=False;self.flushing=False
        def begin_dispatch_wave(self):pass
        def flush_dispatch_wave(self):
            self.flushing=True;entered.set();assert release.wait(4)
            if not self.paused:events.append('POST')
            self.flushing=False;events.append('settled')
        def pause(self):self.paused=True;events.append('pause')
        def abort_dispatch_wave(self):assert not self.flushing;events.append('abort')
        def checkpoint(self,**kwargs):events.append('ack')
    t=T();app=SymmetryMixin();app._model_active_campaigns={};app._model_stopping=False
    app._refinement_store=lambda:None;app.refresh_symmetry=lambda:None
    app.server=SimpleNamespace(state=SimpleNamespace(sx_slots=2,flush=lambda:None))
    async def run():
        task=asyncio.create_task(app._drive_symmetry_campaign(C(),t))
        assert await asyncio.to_thread(entered.wait,4)
        task.cancel();await asyncio.sleep(0)
        assert t.paused and not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):await task
    asyncio.run(run())
    assert events==['pause','settled','abort']
