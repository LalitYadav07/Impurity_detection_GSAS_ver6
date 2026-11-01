"""Explicit terminal-error retries preserve old work and their durable identity."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from radar_pd_nova.checkpoint_uploads import CheckpointUploadPending
from radar_pd_nova.galaxy_service import ANALYZE_TOOL_ID, GalaxyService
from radar_pd_nova.live_recovery import RetrySourceChanged, failed_run_proof
from radar_pd_nova.models import AnalysisConfig, InputSelection, RunRecord, RunStatus
from radar_pd_nova.powgen_controller import PowgenExperimentSettings, PowgenWatchController
from radar_pd_nova.powgen_watch import WatchedRun, WatchState, build_submission_plan


class Service:
    history_id = 'history'
    galaxy_url = 'https://galaxy.invalid'
    _headers = {}
    def __init__(self):
        self.job = dict(id='old', history_id=self.history_id, tool_id=ANALYZE_TOOL_ID,
                        state='error', exit_code=None, external_id=None,
                        command_line=None, stdout='Native pipeline FINAL; outputs may exist',
                        inputs={'data':'data', 'configuration':'config'}, outputs={'summary':'old-summary'})
        self.metadata = {k:dict(history_id=self.history_id,state='ok',file_size=100,deleted=False,purged=False)
                         for k in ('data','config')}
        self.saved = []
        self.recovered_runs = []
        self.posted = []
    def _job_details(self,key):
        assert key == 'old'
        return deepcopy(self.job)
    def _public_input_dataset_ids(self,job): return deepcopy(job['inputs'])
    def _job_output_ids(self,job): return deepcopy(job.get('outputs',{}))
    def _dataset_metadata(self,key): return deepcopy(self.metadata[key])
    def upload_json_document(self,payload,**kwargs):
        self.saved.append(deepcopy(payload)); return f'checkpoint-{len(self.saved)}'
    def recent_runs(self,**kwargs): return deepcopy(self.recovered_runs)
    def submit_snapshot(self,snapshot,*,before_submit=None):
        assert before_submit is not None
        before_submit()
        self.posted.append(snapshot)
        return RunRecord(uid='replacement',name=snapshot.config.run_name,history_id=self.history_id,
                         mode=snapshot.config.mode,galaxy_job_id='replacement',status=RunStatus.QUEUED,
                         analysis_status=RunStatus.QUEUED,config=snapshot.config,inputs=snapshot.inputs)


def controller():
    service=Service()
    c=PowgenWatchController(service,PowgenExperimentSettings('IPTS-1','history','config','1.5'))
    run=WatchedRun(123,'/SNS/PG3/IPTS-1/shared/autoreduce/PG3_123.gsa','stable',bank=2,
                   galaxy_job_id='old',galaxy_result_ids=('old-summary',),
                   scan_metadata={'schema_version':3})
    c.state.failed[run.run_id]=run
    c.records[run.run_id]=RunRecord(uid='old',name=c._run_name(run),mode='full',history_id='history',
                                    galaxy_job_id='old',status='error')
    return c,service,run


def test_error_proof_uses_terminal_state_not_missing_runner_or_native_logs():
    c,s,run=controller()
    proof=failed_run_proof(s,'old')
    assert proof['kind']=='terminal_error' and proof['outputs']=={'summary':'old-summary'}
    assert 'fresh analysis' in proof['reason']
    s.job.update(exit_code=1,external_id='runner-id',command_line='actual worker command')
    assert failed_run_proof(s,'old') is not None
    assert 'no refinement' not in proof['reason'].lower()


@pytest.mark.parametrize('change',[
    {'state':'ok'},{'state':'running'},{'state':'new'},{'state':'queued'},
    {'state':'deleted'},{'state':'deleted_new'},{'state':'cancelled'},{'state':'canceled'},
    {'deleted':True},{'purged':True},{'history_id':'other'}, {'tool_id':'other'}, {'id':'other'},
])
def test_only_owned_authoritative_terminal_errors_qualify(change):
    _,s,_=controller();s.job.update(change)
    assert failed_run_proof(s,'old') is None


@pytest.mark.parametrize('change',[
    {'state':'running'}, {'state':'error'}, {'deleted':True}, {'purged':True}, {'history_id':'other'},
])
def test_unready_or_deleted_inputs_do_not_become_recovered_science(change):
    _,s,_=controller();s.metadata['data'].update(change)
    assert failed_run_proof(s,'old') is None


def test_error_state_is_rechecked_after_input_reads():
    _,s,_=controller();read=s._dataset_metadata
    def changed(key):
        s.job['state']='ok';return read(key)
    s._dataset_metadata=changed
    assert failed_run_proof(s,'old') is None


def test_no_explicit_request_does_not_retry_an_error():
    c,s,run=controller()
    assert c.retry_failed_runs()==0 and not s.saved
    assert c.state.failed[run.run_id].galaxy_job_id=='old'


def test_explicit_error_retry_has_two_durable_stages_and_keeps_artifact_lineage(monkeypatch):
    c,s,run=controller()
    monkeypatch.setattr('requests.delete',lambda *a,**k:pytest.fail('Terminal errors must not be cancelled'))
    original=build_submission_plan(c.definition,replace(run,galaxy_job_id=None)).idempotency_key
    assert c.retry_failed_runs(request=True)==1
    assert len(s.saved)==2
    assert s.saved[0]['failed'][run.run_id]['scan_metadata']['run_retry_intent']['job_id']=='old'
    retry=c.state.discovered[run.run_id]
    assert s.saved[1]['discovered'][run.run_id]['scan_metadata']['retry_generation']==1
    assert retry.galaxy_job_id is None and retry.galaxy_result_ids==()
    assert retry.scan_metadata['previous_attempts'][0]['outputs']=={'summary':'old-summary'}
    assert retry.scan_metadata['retry_source_proof']['job_id']=='old'
    assert c._run_name(retry).endswith('_retry1')
    assert build_submission_plan(c.definition,retry,c.state).idempotency_key!=original
    assert c.retry_failed_runs(request=True)==0
    assert not s.posted


@pytest.mark.parametrize('failed_save',[1,2])
def test_pending_checkpoint_never_allows_a_replacement_post(failed_save):
    c,s,run=controller();persist=c.persist_state;calls=[]
    def pending(*a,**k):
        calls.append(1)
        if len(calls)==failed_save:raise CheckpointUploadPending('pending','queued')
        return persist(*a,**k)
    c.persist_state=pending
    with pytest.raises(CheckpointUploadPending):c.retry_failed_runs(request=True)
    assert c.interrupted_retries_pending and c.due_submissions()==[]
    c.retry_failed_runs()
    assert c.state.discovered[run.run_id].scan_metadata['retry_generation']==1
    assert not c.interrupted_retries_pending and not s.posted


@pytest.mark.parametrize('new_state',['ok','running','deleted','cancelled'])
def test_changed_job_during_checkpoint_is_not_retired_or_resubmitted(new_state):
    c,s,run=controller();save=s.upload_json_document
    def changed(*a,**k):
        value=save(*a,**k);s.job['state']=new_state;return value
    s.upload_json_document=changed
    c.retry_failed_runs(request=True)
    assert run.run_id in c.state.failed and run.run_id not in c.state.discovered
    assert not s.posted


def test_guard_rechecks_state_after_queue_or_staging_wait():
    c,s,run=controller();c.retry_failed_runs(request=True)
    retry=c.state.discovered[run.run_id]
    s.job['state']='ok'
    with pytest.raises(RetrySourceChanged):c._validate_retry_submission(retry)
    assert not s.posted


def test_saved_generation_recovers_an_existing_success_instead_of_rerunning():
    c,s,run=controller();c.retry_failed_runs(request=True)
    retry=c.state.discovered[run.run_id]
    saved=deepcopy(s.saved[-1])
    restored=PowgenWatchController(s,c.settings);restored.state=WatchState.from_dict(saved)
    record=RunRecord(uid='replacement',name=restored._run_name(retry),mode='full',history_id='history',
                     galaxy_job_id='replacement',status='ok',output_dataset_ids={'summary':'new-summary'},
                     input_dataset_ids={'configuration':'config'})
    s.recovered_runs=[record]
    restored._scientific_summary=lambda record:{'rwp':2.0}
    assert restored.reconcile_recent_runs(checkpoint=False)==1
    assert restored.state.completed[run.run_id].galaxy_job_id=='replacement'
    assert restored.due_submissions()==[] and not s.posted


def test_last_guard_runs_after_input_preparation_and_admission(tmp_path,monkeypatch):
    service=GalaxyService('https://galaxy.invalid','test-key','history',output_root=tmp_path)
    snapshot=service.create_submission_snapshot(AnalysisConfig(sample_elements=['Fe','O']),
        InputSelection(data_path='input.gsa',instrument_path='profile.instprm'),idempotency_token='retry-case')
    events=[]
    @contextmanager
    def store():yield object()
    @contextmanager
    def slot(*a,**k):
        events.append('admitted');yield
    monkeypatch.setattr(service,'_store',store)
    monkeypatch.setattr(service,'_prepare_datasets',lambda *a:(events.append('prepared') or {'configuration':'config'}))
    monkeypatch.setattr(service,'_prepared_parameters',lambda *a:(object(),snapshot.inputs))
    monkeypatch.setattr('radar_pd_nova.native_job_admission.HistoryJobAdmission.slot',slot)
    monkeypatch.setattr(service,'_submit_analysis',lambda *a:pytest.fail('Changed source must prevent POST'))
    def guard():
        events.append('guard');raise RetrySourceChanged('old job became OK')
    with pytest.raises(RetrySourceChanged):service.submit_snapshot(snapshot,before_submit=guard)
    assert events==['prepared','admitted','guard']


def test_ui_explains_fresh_science_and_keeps_one_retry_control():
    from radar_pd_nova.app import RadarPdNovaApp
    template=RadarPdNovaApp().layout.html
    assert 'Retry failed runs' in template and 'Retry failed uploads' not in template
    assert 'fresh analyses' in template and 'earlier' in template


def test_saved_intent_can_resume_without_incrementing_generation_twice():
    c,s,run=controller();persist=c.persist_state;calls=[]
    def pending(*a,**k):
        calls.append(1)
        if len(calls)==2:raise CheckpointUploadPending('new-generation','queued')
        return persist(*a,**k)
    c.persist_state=pending
    with pytest.raises(CheckpointUploadPending):c.retry_failed_runs(request=True)
    restored=PowgenWatchController(s,c.settings)
    restored.state=WatchState.from_dict(s.saved[0])
    assert restored.retry_failed_runs()==1
    assert restored.state.discovered[run.run_id].scan_metadata['retry_generation']==1
    assert not s.posted


def test_stop_defers_unsent_retry_even_after_checkpoint_restore():
    c,s,run=controller();c.retry_failed_runs(request=True)
    assert c.pause_failed_run_retries()
    restored=PowgenWatchController(s,c.settings)
    restored.state=WatchState.from_dict(c.state.as_dict())
    assert restored.retry_failed_runs()==0 and restored.due_submissions()==[]
    assert restored.retry_failed_runs(request=True)==1
    restored.reconcile_recent_runs(checkpoint=False)
    assert len(restored.due_submissions())==1


def test_revoked_retry_stays_failed_until_another_explicit_action():
    c,s,run=controller();c.retry_failed_runs(request=True)
    retry=c.state.discovered[run.run_id]
    c.hold_failed_retry(retry,'Previous job was cancelled')
    assert c.due_submissions()==[] and c.retry_failed_runs()==0
    assert c.state.failed[run.run_id].galaxy_job_id=='old'
    assert c.state.failed[run.run_id].galaxy_result_ids==('old-summary',)


def test_retry_transport_rechecks_after_a_lost_acknowledgement_delay(monkeypatch,tmp_path):
    service=GalaxyService('https://galaxy.invalid','test-key','history',output_root=tmp_path)
    allowed=[True];posts=[]
    def guard():
        if not allowed[0]:raise RetrySourceChanged('Previous job changed while awaiting acknowledgement')
    def post(*args):
        posts.append(1);raise OSError('Acknowledgement lost')
    monkeypatch.setattr(service,'_submit_analysis',post)
    monkeypatch.setattr(service,'_find_acknowledged_job',lambda *a:None)
    monkeypatch.setattr('radar_pd_nova.galaxy_service.time.sleep',lambda *a:allowed.__setitem__(0,False))
    with pytest.raises(RetrySourceChanged):
        service._submit_with_retry(object(),object(),run_name='retry1',config_dataset_id='config',before_submit=guard)
    assert posts==[1]
