from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import pytest
from radar_pd_nova.galaxy_service import ANALYZE_TOOL_ID
from radar_pd_nova.live_recovery import failed_input_proof, retire_failed_input
from radar_pd_nova.powgen_controller import PowgenExperimentSettings, PowgenWatchController
from radar_pd_nova.powgen_watch import WatchedRun, build_submission_plan
from radar_pd_nova.checkpoint_uploads import CheckpointUploadPending


class Service:
    history_id='history'
    galaxy_url='https://galaxy.invalid'
    _headers={}
    def __init__(self):
        self.job=dict(id='job-old',history_id='history',tool_id=ANALYZE_TOOL_ID,state='paused',exit_code=None)
        self.metadata=dict(data=dict(state='error',file_size=0,
            misc_info='This job was killed when Galaxy was restarted.  Please retry the job.'),config=dict(state='ok',file_size=50))
        self.saved=[]
    def _job_details(self,key):assert key=='job-old';return deepcopy(self.job)
    def _public_input_dataset_ids(self,job):return {'data':'data','configuration':'config'}
    def _dataset_metadata(self,key):return deepcopy(self.metadata[key])
    def upload_json_document(self,payload,**kwargs):self.saved.append(deepcopy(payload));return 'checkpoint'+str(len(self.saved))


def controller():
    service=Service()
    c=PowgenWatchController(service,PowgenExperimentSettings('IPTS-1','history','config','1.5'))
    run=WatchedRun(123,'/SNS/PG3/IPTS-1/shared/autoreduce/PG3_123.gsa','stable',bank=2,galaxy_job_id='job-old')
    c.state.failed[run.run_id]=run
    c.records[run.run_id]=SimpleNamespace(galaxy_job_id='job-old')
    return c,service,run


def install_retirement(monkeypatch,service):
    posts=[]
    def delete(url,**kwargs):
        assert url=='https://galaxy.invalid/api/jobs/job-old' and kwargs['allow_redirects'] is False
        assert service.saved[-1]['failed']['PG3_123_BANK2']['scan_metadata']['input_retry_intent']['job_id']=='job-old'
        posts.append(url);service.job['state']='deleted'
        return SimpleNamespace(raise_for_status=lambda:None)
    monkeypatch.setattr('requests.delete',delete)
    return posts


def test_explicit_retry_saves_retirement_and_new_generation_before_dispatch(monkeypatch):
    c,service,run=controller();posts=install_retirement(monkeypatch,service)
    original=build_submission_plan(c.definition,replace(run,galaxy_job_id=None)).idempotency_key
    assert c.retry_interrupted_uploads(request=True)==1
    retry=c.state.discovered[run.run_id]
    assert len(posts)==1 and len(service.saved)==2
    assert not c.state.failed and not c.records and not c.interrupted_retries_pending
    assert retry.source_path==run.source_path and retry.fingerprint==run.fingerprint
    assert retry.scan_metadata['previous_attempts'][0]['job_id']=='job-old'
    assert retry.scan_metadata['retry_generation']==1 and c._run_name(retry).endswith('_retry1')
    assert build_submission_plan(c.definition,retry,c.state).idempotency_key!=original
    assert c.due_submissions()==[retry]
    assert c.retry_interrupted_uploads(request=True)==0 and len(posts)==1


@pytest.mark.parametrize('failed_save',[1,2])
def test_pending_checkpoint_blocks_retirement_or_new_native_post_until_ack(monkeypatch,failed_save):
    c,service,run=controller();posts=install_retirement(monkeypatch,service)
    persist=c.persist_state;calls=[]
    def pending(*args,**kwargs):
        calls.append(1)
        if len(calls)==failed_save:raise CheckpointUploadPending('owned-checkpoint','queued')
        return persist(*args,**kwargs)
    c.persist_state=pending
    with pytest.raises(CheckpointUploadPending):c.retry_interrupted_uploads(request=True)
    assert c.interrupted_retries_pending and not c.due_submissions()
    assert len(posts)==(0 if failed_save==1 else 1)
    c.retry_interrupted_uploads()
    assert not c.interrupted_retries_pending and len(posts)==1
    assert c.state.discovered[run.run_id].scan_metadata['retry_generation']==1


@pytest.mark.parametrize('update',[
    dict(state='running'),dict(state='queued'),dict(state='deleted'),dict(history_id='other'),
    dict(tool_id='other'),dict(exit_code=0),dict(command_line='python worker.py'),dict(job_runner_name='runner')])
def test_scientific_work_ordinary_cancellations_and_unrelated_jobs_never_auto_retry(update):
    service=Service();service.job.update(update)
    assert failed_input_proof(service,'job-old') is None


def test_manual_pause_with_valid_or_partially_written_inputs_does_not_qualify():
    service=Service();service.metadata['data']['file_size']=100
    assert failed_input_proof(service,'job-old') is None
    service.metadata['data'].update(state='ok',file_size=100)
    assert failed_input_proof(service,'job-old') is None


def test_explicit_retry_can_handle_cancelled_empty_upload_without_inventing_restart_evidence():
    service=Service();service.metadata['data'].update(state='discarded',misc_info='Upload cancelled')
    proof=failed_input_proof(service,'job-old')
    assert proof and 'Galaxy restart' not in proof['reason']


def test_changed_execution_proof_does_not_cancel_or_restage(monkeypatch):
    service=Service();proof=failed_input_proof(service,'job-old');service.job['state']='running'
    monkeypatch.setattr('requests.delete',lambda *a,**k:pytest.fail('Must not cancel active refinement'))
    with pytest.raises(RuntimeError,match='proof changed'):retire_failed_input(service,proof)
