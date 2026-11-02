"""A queued retry drains real pre-POST admission before changing watch state."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import threading
from types import SimpleNamespace
from uuid import uuid4

import pytest
from trame.app import get_server

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.galaxy_service import GalaxyService, SubmissionDeferred, SubmissionReceiptPending
from radar_pd_nova.models import AnalysisConfig, InputSelection, RunRecord, RunStatus
from radar_pd_nova.native_job_admission import HistoryJobAdmission
from radar_pd_nova.powgen_controller import PowgenWatchController
from radar_pd_nova.powgen_watch import WatchState, WatchedRun


@pytest.fixture
def transport(tmp_path, monkeypatch):
    service = GalaxyService('https://galaxy.invalid', 'placeholder', 'h', output_root=tmp_path)
    rows = {f'old-{n}': dict(id=f'old-{n}', history_id='h', tool_id='neutrons_radar_pd_analyze_prototype',
                            state='paused' if n < 3 else 'queued') for n in range(7)}
    rows['campaign'] = dict(id='campaign', history_id='h', tool_id='neutrons_radar_pd_model_campaign_prototype', state='queued')
    uploads, posts = [], []
    waiting = threading.Event()
    prepared = threading.Event()
    lock = threading.RLock()

    @contextmanager
    def store():
        yield object()

    def upload(key, path, label):
        with lock:
            uploads.append((key, path))
        return key, 'fresh-' + path.replace('/', '_') + '-' + key

    def active(self):
        with lock:
            current = [dict(r) for r in rows.values() if r['state'] in {'paused', 'queued', 'running', 'new'}]
        assert len(current) <= 8
        if len(current) == 8:
            waiting.set()
        return current

    def parameters(store, snapshot, inputs):
        prepared.set()
        return snapshot, snapshot.inputs

    def post(store, snapshot):
        with lock:
            uid = 'accepted-' + snapshot.idempotency_token
            assert uid not in rows, 'Duplicate POST for the same token'
            posts.append(snapshot.idempotency_token)
            rows[uid] = dict(id=uid, history_id='h', tool_id='neutrons_radar_pd_analyze_prototype', state='queued')
            assert sum(r['state'] in {'paused', 'queued', 'new', 'running'} for r in rows.values()) <= 8
        return uid, object()

    monkeypatch.setattr(service, '_store', store)
    monkeypatch.setattr(service, '_upload_one', upload)
    monkeypatch.setattr(service, '_prepared_parameters', parameters)
    monkeypatch.setattr(service, '_submit_analysis', post)
    monkeypatch.setattr(service, '_find_acknowledged_job', lambda *a: None)
    monkeypatch.setattr(HistoryJobAdmission, 'active_jobs', active)
    return SimpleNamespace(service=service, rows=rows, uploads=uploads, posts=posts,
                           waiting=waiting, prepared=prepared, lock=lock)


def snapshot(service, token='same-token'):
    return service.create_submission_snapshot(AnalysisConfig(run_name=token, sample_elements=['Fe', 'O']),
        InputSelection(data_path='scan.gsa', instrument_path='bank2.instprm'), idempotency_token=token)


def test_full_eight_including_paused_yields_and_reuses_exact_inputs(transport):
    t = transport
    request = snapshot(t.service)
    pause = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(t.service.submit_snapshot, request, admission_pause=pause)
        assert t.waiting.wait(2)
        record = t.service.pending_record(request)
        inputs = dict(record.prepared_dataset_ids)
        pause.set()
        with pytest.raises(SubmissionDeferred):
            pending.result(timeout=1)
    assert not t.posts and not record.cancel_requested
    assert record.analysis_status == RunStatus.UPLOADING
    assert not t.service._submission_cancel_events[request.idempotency_token].is_set()
    with t.lock:
        t.rows['old-0']['state'] = 'deleted'
    pause.clear()
    result = t.service.submit_snapshot(request, admission_pause=pause)
    assert result is record and result.galaxy_job_id == 'accepted-same-token'
    assert result.prepared_dataset_ids == inputs
    assert t.posts == ['same-token'] and len(t.uploads) == 3


@pytest.mark.parametrize('ambiguous', [False, True])
def test_pause_after_post_still_acknowledges_without_duplicate(transport, monkeypatch, ambiguous):
    t = transport
    with t.lock:
        t.rows['old-0']['state'] = 'deleted'
    pause = threading.Event()
    request = snapshot(t.service, 'late-ack')
    posts = []
    class Tool:
        def __init__(self, id): pass
        def assign_id(self, uid, store): self.uid = uid
    monkeypatch.setattr('nova.galaxy.Tool', Tool)
    def post(*args):
        posts.append(1)
        pause.set()
        if ambiguous:
            raise TimeoutError('acknowledgement lost after acceptance')
        return 'late-accepted', object()
    monkeypatch.setattr(t.service, '_submit_analysis', post)
    monkeypatch.setattr(t.service, '_find_acknowledged_job', lambda *a: 'late-accepted')
    result = t.service.submit_snapshot(request, admission_pause=pause)
    assert result.galaxy_job_id == 'late-accepted' and len(posts) == 1


def test_unresolved_post_is_held_across_pause_and_resume_without_reposting(transport, monkeypatch):
    t = transport
    t.rows['old-0']['state'] = 'deleted'
    pause = threading.Event()
    request = snapshot(t.service, 'unresolved-ack')
    posts = []
    def post(*args):
        posts.append(1)
        pause.set()
        raise TimeoutError('request may have reached Galaxy')
    monkeypatch.setattr(t.service, '_submit_analysis', post)
    with pytest.raises(SubmissionReceiptPending):
        t.service.submit_snapshot(request, admission_pause=pause)
    pause.clear()
    with pytest.raises(SubmissionReceiptPending):
        t.service.submit_snapshot(request, admission_pause=pause)
    assert posts == [1]
    assert t.service.pending_record(request).provenance['submission_receipt_pending'] is True


class DrainController:
    source_directory = '/read-only/source'
    settings = SimpleNamespace(max_active_jobs=5, ipts='IPTS-1', configuration_dataset_id='config')
    refresh_errors = {}
    interrupted_retries_pending = PowgenWatchController.interrupted_retries_pending
    _submit_live_snapshot = PowgenWatchController._submit_live_snapshot
    _validate_retry_submission = PowgenWatchController._validate_retry_submission
    _run_name = PowgenWatchController._run_name
    hold_failed_retry = PowgenWatchController.hold_failed_retry
    pause_failed_run_retries = PowgenWatchController.pause_failed_run_retries

    def __init__(self, transport):
        self.t = transport
        self.service = transport.service
        self.state = WatchState(history_id='h', instrument='NOM')
        self.records = {}
        self._before_live_submit = None
        self._retry_checkpoint_pending = False
        self._recent_runs_reconciled = True
        self.working = 0
        self.lock = threading.Lock()
        self.retry_done = threading.Event()
        self.repair_checkpointed = threading.Event()
        self.checkpoints = []
        for n in range(47):
            run = self.run(100000 + n, f'complete-{n}')
            self.state.completed[run.run_id] = run
        for n in range(3):
            run = self.run(220637 + n, f'old-{n}')
            self.state.failed[run.run_id] = run
            transport.rows[f'old-{n}']['inputs'] = {'data_inputs|input_source|diffraction_pattern': {'id': f'bad-{n}', 'src': 'hda'}}
            transport.rows[f'old-{n}']['outputs'] = {}
        for n in range(4):
            run = self.run(220650 + n, f'old-{n+3}')
            self.state.submitted[run.run_id] = run
        run = self.run(220660)
        self.state.discovered[run.run_id] = run

    @staticmethod
    def run(n, job=None):
        return WatchedRun(run_number=n, source_path=f'/source/{n}.gsa', fingerprint=str(n), instrument='NOM', bank=2, galaxy_job_id=job)

    def discover(self, **kwargs): pass
    def reconcile_recent_runs(self, **kwargs): return 0
    due_submissions = PowgenWatchController.due_submissions

    def launch_submission(self, run):
        with self.lock: self.working += 1
        try:
            generation = run.scan_metadata.get('retry_generation', 0)
            return self._submit_live_snapshot(run, snapshot(self.service, f'{run.run_id}-g{generation}'))
        finally:
            with self.lock: self.working -= 1

    def acknowledge_submission(self, run, record, *, persist):
        self.state.mark_submitted(run, record.galaxy_job_id)
        self.records[run.run_id] = record
        return record

    def retry_failed_runs(self, *, request=False):
        with self.lock: assert self.working == 0, 'Retry mutated state before workers drained'
        result = PowgenWatchController.retry_failed_runs(self, request=request)
        self.retry_done.set()
        return result

    def capture_checkpoint(self): return self.state.to_json()
    def persist_state(self, snapshot=None):
        payload = json.loads(snapshot or self.capture_checkpoint())
        self.checkpoints.append(payload)
        if sum(r.get('scan_metadata', {}).get('retry_generation', 0) == 1 for r in payload['discovered'].values()) == 3:
            self.repair_checkpointed.set()
        return 'checkpoint'
    def refresh(self, **kwargs): return {}


def app_controller(t, monkeypatch):
    server = get_server('retry-drain-' + uuid4().hex, client_type='vue3')
    monkeypatch.setattr('radar_pd_nova.app.get_server', lambda *a, **k: server)
    app = RadarPdNovaApp()
    controller = DrainController(t)
    app._powgen_controller = controller
    app.service = t.service
    app._sync_powgen_rows = lambda **kwargs: None
    app._sync_runs = lambda: None
    app.server.state.powgen_monitoring = True
    return app, controller


def test_explicit_retry_drains_blocked_launch_before_real_proof_repair(transport, monkeypatch):
    t = transport
    app, controller = app_controller(t, monkeypatch)
    monkeypatch.setattr(t.service, '_job_details', lambda uid: dict(t.rows[uid]))
    monkeypatch.setattr(t.service, '_dataset_metadata', lambda uid: dict(history_id='h', state='error', file_size=0, deleted=False, purged=False))
    def delete(url, **kwargs):
        uid = url.rsplit('/', 1)[-1]
        assert uid in {'old-0', 'old-1', 'old-2'}
        t.rows[uid]['state'] = 'deleted'
        return SimpleNamespace(raise_for_status=lambda: None)
    monkeypatch.setattr('requests.delete', delete)
    originals = {key: run.galaxy_job_id for key, run in controller.state.completed.items()}

    async def run():
        task = asyncio.create_task(app._powgen_monitor_loop())
        app._powgen_monitor_task = task
        try:
            assert await asyncio.to_thread(t.waiting.wait, 2)
            app.retry_powgen_failed_runs()
            assert await asyncio.to_thread(controller.repair_checkpointed.wait, 3)
            app.stop_powgen_monitoring()
            await asyncio.wait_for(task, 3)
        finally:
            app.stop_powgen_monitoring()
            await asyncio.wait_for(task, 4)
        assert controller.retry_done.is_set()
        assert all(t.rows[f'old-{n}']['state'] == 'deleted' for n in range(3))
        assert {key: run.galaxy_job_id for key, run in controller.state.completed.items()} == originals
        assert len(originals) == 47
        for n in range(3):
            recovered = controller.state.discovered[f'NOM_{220637+n}_BANK2']
            assert recovered.scan_metadata['retry_generation'] == 1
            assert recovered.scan_metadata['previous_attempts'][0]['job_id'] == f'old-{n}'
        assert not any('1000' in token for token in t.posts)
        assert controller.working == 0
    asyncio.run(run())


def test_unresolved_ack_is_checkpointed_and_excluded_from_fresh_admission(transport, monkeypatch):
    t = transport
    t.rows['old-0']['state'] = 'deleted'
    app, controller = app_controller(t, monkeypatch)
    posted = threading.Event()
    attempts = []
    def post(*args):
        attempts.append(1)
        posted.set()
        raise TimeoutError('acknowledgement unavailable')
    monkeypatch.setattr(t.service, '_submit_analysis', post)

    async def run():
        task = asyncio.create_task(app._powgen_monitor_loop())
        app._powgen_monitor_task = task
        try:
            assert await asyncio.to_thread(posted.wait, 2)
            for _ in range(100):
                run = controller.state.discovered['NOM_220660_BANK2']
                if run.scan_metadata.get('submission_receipt_pending'):
                    break
                await asyncio.sleep(.01)
            assert run.scan_metadata['submission_receipt_pending'] is True
            assert not controller.due_submissions()
            assert controller._recent_runs_reconciled is False
        finally:
            app.stop_powgen_monitoring()
            await asyncio.wait_for(task, 3)
        saved = WatchState.from_json(json.dumps(controller.checkpoints[-1]))
        assert saved.discovered['NOM_220660_BANK2'].scan_metadata['submission_receipt_pending']
        assert attempts == [1] and len(saved.completed) == 47
    asyncio.run(run())


def test_unresolved_ack_keeps_reconciling_until_late_job_appears(transport, monkeypatch):
    t = transport
    controller = DrainController(t)
    run = controller.state.discovered['NOM_220660_BANK2']
    run.scan_metadata['submission_receipt_pending'] = True
    visible, calls = [], []
    monkeypatch.setattr(t.service, 'recent_runs', lambda **kwargs: calls.append(kwargs) or list(visible))
    monkeypatch.setattr(controller, '_record_matches_watch', lambda record, run: True, raising=False)
    assert controller._recent_runs_reconciled is True
    assert PowgenWatchController.reconcile_recent_runs(controller, checkpoint=False) == 0
    visible.append(RunRecord(uid='late', galaxy_job_id='late', name=controller._run_name(run),
                             history_id='h', mode='full', status=RunStatus.QUEUED))
    assert PowgenWatchController.reconcile_recent_runs(controller, checkpoint=False) == 1
    assert controller.state.submitted[run.run_id].galaxy_job_id == 'late'
    assert run.run_id not in controller.state.discovered and not t.posts
    assert len(calls) == 2


@pytest.mark.parametrize('revoke', ['stop', 'owner'])
def test_stop_or_owner_change_drains_admission_without_repair_or_post(transport, monkeypatch, revoke):
    t = transport
    app, controller = app_controller(t, monkeypatch)
    async def run():
        task = asyncio.create_task(app._powgen_monitor_loop())
        app._powgen_monitor_task = task
        try:
            assert await asyncio.to_thread(t.waiting.wait, 2)
            if revoke == 'stop':
                app.stop_powgen_monitoring()
            else:
                app._result_context = object()
                app.server.state.powgen_message = 'new owner message'
            await asyncio.wait_for(task, 7)
        finally:
            controller._submission_pause_event and controller._submission_pause_event.set()
            if not task.done(): await asyncio.wait_for(task, 4)
        assert controller.working == 0 and not t.posts and not controller.retry_done.is_set()
        assert len(controller.state.completed) == 47 and len(controller.state.failed) == 3
        assert len(controller.state.discovered) == 1
        if revoke == 'owner': assert app.server.state.powgen_message == 'new owner message'
    asyncio.run(run())
