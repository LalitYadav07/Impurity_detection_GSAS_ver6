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
