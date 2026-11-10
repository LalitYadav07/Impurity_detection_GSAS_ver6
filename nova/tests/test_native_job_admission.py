from types import SimpleNamespace
import threading

import pytest

from radar_pd_nova.native_job_admission import (
    AdmissionUnavailable, HistoryJobAdmission, configured_limit,
)


def service(history='h'):
    return SimpleNamespace(galaxy_url='https://galaxy.invalid', history_id=history, _headers={})


class Response:
    def __init__(self, rows): self.rows = rows
    def raise_for_status(self): pass
    def json(self): return self.rows


def job(uid, tool='neutrons_radar_pd_model_campaign_prototype', state='running'):
    return dict(id=uid, tool_id=tool, state=state, history_id='h')


@pytest.mark.parametrize('value', ['0', '-1', '33', '1.5', 'x', 'True', '01'])
def test_invalid_deployment_limits(value):
    with pytest.raises(ValueError): configured_limit({'RADAR_PD_MAX_ACTIVE_JOBS': value})


def test_capacity_counts_all_workflows_without_viewer_or_duplicates():
    rows = [job('m'), job('m'), job('r', 'neutrons_radar_pd_refine_recipe_prototype'),
            job('v', 'neutrons_radar_pd_interactive_prototype'), job('o', 'other'), job('done', state='ok')]
    gate = HistoryJobAdmission(service(), fetch=lambda *a, **k: Response(rows))
    assert {row['id'] for row in gate.active_jobs()} == {'m', 'r'}


@pytest.mark.parametrize('rows', [{}, [None], [job('a')]*1001, [{**job('a'), 'history_id':'other'}]])
def test_incomplete_or_wrong_history_queue_cannot_authorize_submission(rows):
    gate = HistoryJobAdmission(service(), fetch=lambda *a, **k: Response(rows))
    with pytest.raises(AdmissionUnavailable):
        with gate.slot('neutrons_radar_pd_refine_recipe_prototype', family_limit=1):
            pytest.fail('must not submit')


def test_full_shared_capacity_waits_for_other_workflow_to_finish():
    ticks = [0]
    queues = iter([[job('m')], []])
    def wait(seconds): ticks[0] += seconds
    gate = HistoryJobAdmission(service(), total_limit=1, poll_seconds=.01,
        fetch=lambda *a, **k: Response(next(queues)), clock=lambda: ticks[0], wait=wait)
    with gate.slot('neutrons_radar_pd_refine_recipe_prototype', family_limit=4):
        assert ticks[0] == .01


def test_per_family_capacity_is_preserved_under_larger_shared_budget():
    gate = HistoryJobAdmission(service(), total_limit=8, timeout=0,
        fetch=lambda *a, **k: Response([job('m')]))
    with pytest.raises(AdmissionUnavailable):
        with gate.slot('neutrons_radar_pd_model_campaign_prototype', family_limit=1): pass


def test_cancelled_work_never_posts_and_locks_are_history_scoped():
    event = threading.Event(); event.set()
    gate = HistoryJobAdmission(service(), fetch=lambda *a, **k: pytest.fail('must not fetch'))
    assert gate.lock is HistoryJobAdmission(service()).lock
    assert gate.lock is not HistoryJobAdmission(service('another')).lock
    with pytest.raises(AdmissionUnavailable):
        with gate.slot('neutrons_radar_pd_refine_recipe_prototype', family_limit=1, cancel=event): pass


def test_failed_post_releases_lock_without_retrying():
    gate = HistoryJobAdmission(service(), fetch=lambda *a, **k: Response([]))
    with pytest.raises(RuntimeError, match='ambiguous'):
        with gate.slot('neutrons_radar_pd_refine_recipe_prototype', family_limit=1):
            raise RuntimeError('ambiguous POST')
    assert gate.lock.acquire(blocking=False)
    gate.lock.release()


def test_pause_during_capacity_request_never_admits_the_job():
    event=threading.Event()
    def fetch(*args,**kwargs):
        event.set()
        return Response([])
    gate=HistoryJobAdmission(service(),fetch=fetch)
    with pytest.raises(AdmissionUnavailable,match='paused'):
        with gate.slot('neutrons_radar_pd_refine_recipe_prototype',family_limit=1,cancel=event):
            pytest.fail('A paused job must not be submitted')


def test_family_at_its_limit_does_not_block_another_workflow():
    rows=[job('model')];started=threading.Event();refined=threading.Event();errors=[]
    gate=HistoryJobAdmission(service('fairness'),total_limit=2,poll_seconds=.01,timeout=2,
        fetch=lambda *a,**k:Response([{**r,'history_id':'fairness'} for r in rows]))
    def model():
        try:
            started.set()
            with gate.slot('neutrons_radar_pd_model_campaign_prototype',family_limit=1):pass
        except Exception as exc:errors.append(exc)
    worker=threading.Thread(target=model);worker.start();assert started.wait(1)
    with gate.slot('neutrons_radar_pd_refine_recipe_prototype',family_limit=1):refined.set()
    assert refined.is_set()
    rows.clear();worker.join(2)
    assert not worker.is_alive() and not errors and not gate.waiters


def test_failed_capacity_check_does_not_leave_a_waiter():
    gate=HistoryJobAdmission(service('failure-cleanup'),fetch=lambda *a,**k:Response({}))
    with pytest.raises(AdmissionUnavailable):
        with gate.slot('neutrons_radar_pd_refine_recipe_prototype',family_limit=1):pass
    assert not gate.waiters


def test_analyze_uses_same_admission_and_honors_pause(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from radar_pd_nova.galaxy_service import GalaxyService
    from radar_pd_nova.models import AnalysisConfig,InputSelection,RunStatus
    service=GalaxyService('https://galaxy.invalid','test-key','history',output_root=tmp_path)
    snapshot=service.create_submission_snapshot(
        AnalysisConfig(sample_elements=['Al','O']), InputSelection(data_path='scan.gsa',instrument_path='profile.instprm'),
        idempotency_token='capacity-test')
    @contextmanager
    def store():yield object()
    monkeypatch.setattr(service,'_store',store)
    monkeypatch.setattr(service,'_prepare_datasets',lambda *args:{'configuration':'config'})
    monkeypatch.setattr(service,'_prepared_parameters',lambda *args:(object(),snapshot.inputs))
    monkeypatch.setattr(service,'_submit_with_retry',lambda *a,**kw:pytest.fail('Paused Analyze must not POST'))
    def active(self):
        service.cancel_pending_submission('capacity-test')
        return []
    monkeypatch.setattr(HistoryJobAdmission,'active_jobs',active)
    result=service.submit_snapshot(snapshot)
    assert result.status==RunStatus.CANCELLED and result.galaxy_job_id is None


def native_inputs(operation='preparation-structural-training'):
    import base64,json
    request=dict(schema='radar-model-native/v1',operation=operation,payload={},dispatch_key='abc')
    return dict(action='native',dispatch_key='abc',model_request_b64=
        base64.b64encode(json.dumps(request).encode()).decode())


def test_training_admission_classifies_actual_inline_and_legacy_request_bytes():
    import base64
    from radar_pd_nova.native_job_admission import operation_from_inputs
    inputs=native_inputs()
    assert operation_from_inputs(service(),inputs)=='preparation-structural-training'
    raw=base64.b64decode(inputs.pop('model_request_b64'))
    inputs['model_request']={'dataset_id':'request-data'}
    response=Response(None);response.iter_content=lambda chunk_size:[raw];response.close=lambda:None
    assert operation_from_inputs(service(),inputs,fetch=lambda *a,**k:response)=='preparation-structural-training'
    inputs['admission_operation']='inspect'  # Unused labels cannot hide a trainer.
    assert operation_from_inputs(service(),inputs,fetch=lambda *a,**k:response)=='preparation-structural-training'


@pytest.mark.parametrize('damage',['base64','identity','oversized','two_transports'])
def test_invalid_request_cannot_bypass_training_admission(damage):
    from radar_pd_nova.native_job_admission import operation_from_inputs,MAX_REQUEST_BYTES
    inputs=native_inputs()
    if damage=='base64':inputs['model_request_b64']='%%'
    elif damage=='identity':inputs['dispatch_key']='different'
    elif damage=='oversized':inputs['model_request_b64']='a'*(2*MAX_REQUEST_BYTES)
    else:inputs['model_request']={'dataset_id':'another-request'}
    with pytest.raises(AdmissionUnavailable):operation_from_inputs(service(),inputs)


@pytest.mark.parametrize('active_operation,capacity',[
    ('preparation-structural-examples',1),('inspect',1),('preparation-structural-training',0)])
def test_cpu_training_does_not_consume_the_whole_model_family(active_operation,capacity):
    from radar_pd_nova.preparation_limits import CPU_TRAINING_TOOL_ID,TRAINING_OPERATION
    calls=[]
    def fetch(url,**kwargs):
        calls.append(url)
        return Response([job('active')] if url.endswith('/api/jobs') else
            dict(job('active'),params=native_inputs(active_operation)))
    gate=HistoryJobAdmission(service(),fetch=fetch)
    assert gate.available_capacity(CPU_TRAINING_TOOL_ID,family_limit=1,operation=TRAINING_OPERATION)==capacity
    # An immutable job's classification is reused inside this admission pass.
    assert gate.available_capacity(CPU_TRAINING_TOOL_ID,family_limit=1,operation=TRAINING_OPERATION)==capacity
    assert sum(url.endswith('/active') for url in calls)==1
    with gate.slot(CPU_TRAINING_TOOL_ID,family_limit=8):pass


def test_saved_cuda_training_uses_the_same_training_slot_without_device_probing():
    from radar_pd_nova.preparation_limits import CPU_TRAINING_TOOL_ID,CUDA_TRAINING_TOOL_ID,TRAINING_OPERATION
    gate=HistoryJobAdmission(service(),fetch=lambda *a,**k:Response([job('gpu',CUDA_TRAINING_TOOL_ID)]))
    assert gate.available_capacity(CPU_TRAINING_TOOL_ID,family_limit=1,operation=TRAINING_OPERATION)==0
    with gate.slot(CPU_TRAINING_TOOL_ID,family_limit=8):pass


def test_unverifiable_active_cpu_operation_blocks_training_without_a_post():
    from radar_pd_nova.preparation_limits import CPU_TRAINING_TOOL_ID,TRAINING_OPERATION
    gate=HistoryJobAdmission(service(),fetch=lambda url,**k:
        Response([job('unknown')] if url.endswith('/api/jobs') else job('unknown')))
    with pytest.raises(AdmissionUnavailable,match='verifiable operation'):
        gate.available_capacity(CPU_TRAINING_TOOL_ID,family_limit=1,operation=TRAINING_OPERATION)
    with gate.slot(CPU_TRAINING_TOOL_ID,family_limit=8):pass


@pytest.mark.parametrize('operation,limit',[(None,8),('inspect',8),('preparation-structural-training',1)])
def test_utility_admission_derives_cpu_training_limit_from_the_posted_request(tmp_path,monkeypatch,operation,limit):
    from contextlib import contextmanager
    from radar_pd_nova.galaxy_service import GalaxyService
    from radar_pd_nova.preparation_limits import CPU_TRAINING_TOOL_ID
    seen=[]
    @contextmanager
    def slot(self,tool_id,*,family_limit,cancel=None,operation=None):
        seen.append((tool_id,family_limit,operation));yield
    monkeypatch.setattr(HistoryJobAdmission,'slot',slot)
    monkeypatch.setenv('RADAR_PD_MAX_ACTIVE_JOBS','8')
    monkeypatch.setattr('radar_pd_nova.galaxy_service.requests.post',lambda *a,**k:
        SimpleNamespace(raise_for_status=lambda:None,json=lambda:dict(jobs=[dict(id='accepted')])) )
    connected=GalaxyService('https://galaxy.invalid','test','h',output_root=tmp_path)
    connected.submit_utility(tool_id=CPU_TRAINING_TOOL_ID,name='test',
        inputs=native_inputs(operation) if operation else {'action':'chunk'})
    assert seen==[(CPU_TRAINING_TOOL_ID,limit,operation)]
