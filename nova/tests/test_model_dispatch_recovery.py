import copy
import json
from types import SimpleNamespace

import pytest

from radar_pd_nova.model_dispatch_recovery import inspect_missing, retry_reviewed
from radar_pd_nova.model_runtime import GalaxyModelRuntime,MODEL_TOOL_ID
from radar_pd_nova.models import UtilityActionRecord,RunStatus
from radar_pd_nova.refinement_runtime import RuntimeUnavailable
from test_model_runtime import remote
from test_model_ledger_recovery import saved
from gpx_recipe.symmetry.campaign import Campaign
from radar_pd_nova.model_ledger_recovery import restore_ledger
from radar_pd_nova.model_runtime import GalaxyChunkTransport


@pytest.fixture
def missing(remote,saved,tmp_path,monkeypatch):
    runtime,service,source=remote
    campaign=Campaign(tmp_path/'dispatch',shared_inputs=True)
    snapshot,inputs=saved
    restore_ledger(campaign,snapshot,inputs)
    t=GalaxyChunkTransport(runtime,campaign)
    uid=campaign.trials()[0]['id']
    t.campaign.set_state(uid,'needs_recovery')
    t.state['chunks']['missing']=dict(key='missing',members=[uid],protocol=t.campaign.trials()[0]['protocol'],
        request={'sealed':'request'},pending=True,job=None)
    t.state['members'][uid]='missing';t.save()
    monkeypatch.setattr(t.runtime,'find_dispatches_complete',lambda keys,**kw:{k:None for k in keys})
    return t,service,uid


def test_explicit_retry_returns_only_missing_starts_to_queue_and_preserves_attempt(missing):
    t,service,uid=missing
    old=copy.deepcopy(t.state['chunks']['missing'])
    assert inspect_missing(t)==dict(missing_batches=1,missing_starts=1,recovered_batches=0)
    assert not service.submissions and not t.state['chunks']['missing'].get('not_dispatched')
    with pytest.raises(ValueError,match='acknowledge'):retry_reviewed(t)
    assert retry_reviewed(t,acknowledged=True)==dict(retried_batches=1,recovered_batches=0)
    assert t.campaign.trials()[0]['state']=='prepared'
    assert t.state['chunks']['missing']['request']==old['request']
    assert t.state['members'][uid]=='missing'
    assert t.state['chunks']['missing']['explicit_retry'] and not service.submissions


@pytest.mark.parametrize('failure',['expired','changed','lookup'])
def test_failed_or_stale_review_does_not_reset_missing_job(missing,monkeypatch,failure):
    t,service,uid=missing;inspect_missing(t)
    if failure=='expired':t.state['dispatch_recovery_review']['checked_at']=0
    elif failure=='changed':t.state['chunks']['missing']['request']={'changed':True}
    else:
        def fail(keys,**kw):raise RuntimeUnavailable('Cannot complete job lookup')
        monkeypatch.setattr(t.runtime,'find_dispatches_complete',fail)
    with pytest.raises((ValueError,RuntimeUnavailable)):retry_reviewed(t,acknowledged=True)
    assert t.campaign.trials()[0]['state']=='needs_recovery'
    assert not t.state['chunks']['missing'].get('not_dispatched') and not service.submissions


def test_late_job_is_adopted_instead_of_retried(missing,monkeypatch):
    t,service,uid=missing;inspect_missing(t)
    job=UtilityActionRecord(uid='late',tool_id=MODEL_TOOL_ID,name='late',galaxy_job_id='job1',status=RunStatus.RUNNING)
    monkeypatch.setattr(t.runtime,'find_dispatches_complete',lambda keys,**kw:{k:job for k in keys})
    monkeypatch.setattr(t.runtime,'verify_job',lambda *args:None)
    monkeypatch.setattr(t,'poll',lambda handle,directory:dict(state='running',handle=handle))
    result=retry_reviewed(t,acknowledged=True)
    assert result==dict(retried_batches=0,recovered_batches=1)
    assert t.campaign.trials()[0]['state']=='running'
    assert not t.state['chunks']['missing'].get('not_dispatched') and not service.submissions


@pytest.mark.parametrize('state',['terminal','cancelled','prepared','running'])
def test_recovery_does_not_restart_other_states(missing,state):
    t,service,uid=missing;t.campaign.set_state(uid,state)
    assert inspect_missing(t)['missing_batches']==0
    assert not service.submissions


def test_complete_job_lookup_recovers_unnamed_output_and_refuses_duplicate_or_foreign_jobs(tmp_path,monkeypatch):
    jobs=[dict(id='job1',history_id='history',tool_id=MODEL_TOOL_ID,state='running',params={'dispatch_key':'missing'})]
    service=SimpleNamespace(galaxy_url='https://galaxy.invalid',history_id='history',_headers={},
        _job_details=lambda uid:next(x for x in jobs if x['id']==uid),_job_output_ids=lambda job:{})
    runtime=GalaxyModelRuntime(tmp_path,service)
    class Response:
        def raise_for_status(self):pass
        def json(self):return [{'id':row['id']} for row in jobs]
    monkeypatch.setattr('requests.get',lambda *a,**k:Response())
    assert runtime.find_dispatches_complete({'missing'})['missing'].galaxy_job_id=='job1'
    jobs.append(dict(jobs[0],id='job2'))
    with pytest.raises(RuntimeUnavailable,match='Several'):runtime.find_dispatches_complete({'missing'})
    jobs.pop();jobs[0]['history_id']='other'
    with pytest.raises(RuntimeUnavailable,match='outside'):runtime.find_dispatches_complete({'missing'})


def test_ignored_job_pagination_is_not_absence(tmp_path,monkeypatch):
    service=SimpleNamespace(galaxy_url='https://galaxy.invalid',history_id='history',_headers={},
        _job_details=lambda uid:dict(id=uid,history_id='history',tool_id=MODEL_TOOL_ID,params={}))
    runtime=GalaxyModelRuntime(tmp_path,service)
    class Response:
        def raise_for_status(self):pass
        def json(self):return [{'id':f'job{i}'} for i in range(200)]
    monkeypatch.setattr('requests.get',lambda *a,**k:Response())
    with pytest.raises(RuntimeUnavailable,match='repeated'):runtime.find_dispatches_complete({'missing'})


def test_review_then_continue_submits_one_replacement_and_retains_old_attempt(missing,monkeypatch):
    from gpx_recipe.symmetry.compiler import job_inputs
    t,service,uid=missing
    row=next(row for row in t.campaign.trials() if row['id']==uid)
    directory=t.campaign.trial_directory(row)
    request=json.loads((directory/'job.json').read_text())
    source=dict(id='scan',dataset_id='source-data',source_sha256=request['protocol']['source']['sha256'])
    t.prepare_source(request['protocol'],job_inputs(directory,request),source)
    monkeypatch.setattr(t,'checkpoint',lambda **kw:None)
    assert inspect_missing(t)['missing_batches']==1
    assert retry_reviewed(t,acknowledged=True)['retried_batches']==1
    assert service.submissions==[]
    t.campaign.set_setting('paused',False)
    t.resume()
    handles=t.submit_many([(uid,directory)],workers=1)
    assert len(service.submissions)==1
    assert handles[uid]['dispatch_key']!='missing'
    assert t.state['chunks']['missing']['explicit_retry']
    assert t.state['members'][uid]==handles[uid]['dispatch_key']
    assert t.submit_many([(uid,directory)],workers=1)==handles
    assert len(service.submissions)==1


@pytest.mark.parametrize('change_selection,fail',[(False,False),(False,True),(True,False),(True,True)])
def test_recovery_progress_stays_on_event_loop_and_selected_campaign(missing,monkeypatch,change_selection,fail):
    import asyncio
    import threading
    import radar_pd_nova.model_dispatch_recovery as recovery
    from radar_pd_nova.symmetry_controller import SymmetryMixin
    t,service,_=missing
    t.campaign.set_setting('template',{'scope':{'experiment':'IPTS-1'}})
    flushed=[];main_thread=threading.get_ident()
    class State(SimpleNamespace):
        def flush(self):
            assert threading.get_ident()==main_thread
            flushed.append(self.sx_dispatch_recovery_notice)
    state=State(sx_campaign=t.campaign.root.name,rf_experiment='IPTS-1',sx_busy=False,
                sx_campaign_running=False,sx_pause_pending=False,sx_dispatch_retry_ack=False,
                sx_dispatch_recovery_notice='',sx_error='')
    app=SymmetryMixin();app.server=SimpleNamespace(state=state);app._symmetry_task=None
    app._symmetry_root=lambda:t.campaign.root.parent
    app._symmetry_campaign=lambda root:t.campaign
    app._symmetry_transport=lambda campaign:t
    def finish():state.sx_busy=False;state.flush()
    app._finish_symmetry_preparation=finish
    monkeypatch.setattr(t.campaign,'recover',lambda transport:None)
    async def run():
        loop=asyncio.get_running_loop();tasks=[]
        def lookup(transport,*,progress):
            assert threading.get_ident()!=main_thread
            progress(4)
            if change_selection:
                loop.call_soon_threadsafe(setattr,state,'rf_experiment','IPTS-2')
                loop.call_soon_threadsafe(setattr,state,'sx_dispatch_recovery_notice','New selection')
            progress(8)
            if fail:raise RuntimeUnavailable('Job history unavailable')
            return dict(missing_batches=1,missing_starts=1,recovered_batches=0)
        monkeypatch.setattr(recovery,'inspect_missing',lookup)
        def schedule(coro,*args):
            tasks.append(asyncio.create_task(coro));return tasks[-1]
        app._refinement_schedule=schedule
        app.inspect_missing_model_submissions()
        app.inspect_missing_model_submissions()
        assert len(tasks)==1
        await tasks[0]
    asyncio.run(run())
    assert any('Checked 4 model jobs' in message for message in flushed)
    assert not state.sx_busy and not service.submissions
    if change_selection:
        assert state.sx_dispatch_recovery_notice=='New selection'
        assert state.sx_error=='' and not state.sx_dispatch_retry_ready
        assert not any('Checked 8 model jobs' in message for message in flushed)
    elif fail:
        assert state.sx_error=='Job history unavailable' and not state.sx_dispatch_retry_ready
    else:
        assert state.sx_dispatch_retry_ready and 'No new jobs were submitted' in state.sx_dispatch_recovery_notice
