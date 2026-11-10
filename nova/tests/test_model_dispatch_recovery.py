import copy
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
    monkeypatch.setattr(t.runtime,'find_dispatches_complete',lambda keys:{k:None for k in keys})
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
        def fail(keys):raise RuntimeUnavailable('Cannot complete job lookup')
        monkeypatch.setattr(t.runtime,'find_dispatches_complete',fail)
    with pytest.raises((ValueError,RuntimeUnavailable)):retry_reviewed(t,acknowledged=True)
    assert t.campaign.trials()[0]['state']=='needs_recovery'
    assert not t.state['chunks']['missing'].get('not_dispatched') and not service.submissions


def test_late_job_is_adopted_instead_of_retried(missing,monkeypatch):
    t,service,uid=missing;inspect_missing(t)
    job=UtilityActionRecord(uid='late',tool_id=MODEL_TOOL_ID,name='late',galaxy_job_id='job1',status=RunStatus.RUNNING)
    monkeypatch.setattr(t.runtime,'find_dispatches_complete',lambda keys:{k:job for k in keys})
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
