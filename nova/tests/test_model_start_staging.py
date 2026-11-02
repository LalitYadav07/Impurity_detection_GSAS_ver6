"""Delayed source backups must overlap staging without admitting science early."""
import asyncio
import copy
import threading
from types import SimpleNamespace

import pytest

import radar_pd_nova.symmetry_controller as module
from gpx_recipe.symmetry.contracts import digest
from radar_pd_nova.checkpoint_uploads import CheckpointUploadPending
from radar_pd_nova.model_runtime import GalaxyChunkTransport


class Gate:
    def __init__(self, name, events):
        self.name, self.events = name, events
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.error = None

    def __call__(self):
        self.events.append(self.name + '-started')
        self.entered.set()
        assert self.release.wait(5), 'test must release its blocking operation'
        self.events.append(self.name + '-finished')
        self.finished.set()
        if self.error is not None:
            raise self.error


@pytest.fixture
def start_app(tmp_path, monkeypatch):
    events = []
    checkpoint, packet = Gate('checkpoint', events), Gate('packet', events)
    capture=Gate('capture', [])
    capture.release.set()
    state = SimpleNamespace(sx_error='', sx_busy=False, sx_message='', sx_plan_ready=True,
                            sx_approved=True, sx_start_source='scan1', rf_experiment='exp1',
                            sx_campaign='', sx_setup=True, sx_pause_pending=False,
                            sx_queue_notice='', flush=lambda: None)
    service = SimpleNamespace(history_id='history1')
    source = dict(id='scan1', source_sha256='a'*64, scan_label='Scan 1',
                  sample_label='Sample 1', bank='1', dataset_id='source-hda')
    template = dict(existing_ids=['scan1'], future=False, maximum_scans=1,
                    scope=dict(experiment='exp1', sample_key='sample1', bank='1'))
    draft = dict(source_id='scan1', experiment='exp1', root=tmp_path/'input',
                 protocol={'source': {'sha256': 'a'*64}, 'candidate': 'unchanged'},
                 template=template, form_signature='form1',
                 scope_sources={'scan1': source}, scope_origins={'scan1': 'radar'})
    workspace = SimpleNamespace(checkpoint=checkpoint, runtime=SimpleNamespace(resolve=lambda row: copy.deepcopy(row)))
    campaigns = {}

    class Campaign:
        def __init__(self, root, **kwargs):
            self.root, self.settings, self.enrolled = root, {}, False

        def trials(self):
            return ['existing-trial'] if self.enrolled else []

        def set_setting(self, key, value):
            self.settings[key] = copy.deepcopy(value)

    def campaign(root, **kwargs):
        return campaigns.setdefault(root, Campaign(root, **kwargs))

    monkeypatch.setattr(module, 'Campaign', campaign)
    monkeypatch.setattr(module, 'resolve_start', lambda *_: copy.deepcopy(source))
    def prepare(protocol,root,target):
        capture()
        return target/'inputs'
    monkeypatch.setattr(module, 'prepare_shared_inputs', prepare)
    monkeypatch.setattr(module, 'portable_source', copy.deepcopy)
    captured = []
    transport = object.__new__(GalaxyChunkTransport)
    transport.state = {'inputs': {}}

    def stage(protocol, root, row):
        captured.append((copy.deepcopy(protocol), root, copy.deepcopy(row)))
        packet()
        # Same existing contract: a captured HDA reference; no new OK poll.
        transport.state['inputs'][digest(protocol)] = 'acknowledged-packet-hda'

    transport.prepare_source = stage
    app = module.SymmetryMixin()
    app.server, app.service = SimpleNamespace(state=state), service
    app._symmetry_task = None
    app._symmetry_draft, app._symmetry_approved = draft, digest(template)
    app._model_start_cancel = None
    app._model_form = lambda payload=None: 'form1'
    app._refinement_store = lambda: workspace
    app._symmetry_root = lambda: tmp_path/'campaigns'
    app._symmetry_transport = lambda c: transport
    app._model_active_campaigns = {}; app._model_owned_campaigns = {}
    app._model_preparations = {}; app._model_checkpoint_errors = {}
    app._model_enrollment_task = app._model_tick_task = app._model_pause_task = None
    app._k_cancel = None
    app._model_remote = lambda: None
    app.refresh_symmetry = lambda: None
    scheduled = []

    def schedule(work, name):
        task = asyncio.create_task(work, name=name)
        scheduled.append(task)
        return task

    app._refinement_schedule = schedule

    def queue(c, w):
        assert w is workspace
        assert checkpoint.finished.is_set() and packet.finished.is_set()
        events.append('queue')

    monkeypatch.setattr(module, 'queue_sources', queue)

    async def drive(c, t):
        assert checkpoint.finished.is_set() and packet.finished.is_set()
        assert t is transport
        assert t.state['inputs'][digest(draft['protocol'])] == 'acknowledged-packet-hda'
        assert c.settings['template'] == template
        events.append('driver')
        c.enrolled = True

    app._drive_symmetry_campaign = drive
    yield SimpleNamespace(app=app, state=state, service=service, workspace=workspace,
                          draft=draft, source=source, events=events, captured=captured,
                          packet=packet, checkpoint=checkpoint, capture=capture, tasks=scheduled)
    checkpoint.release.set(); packet.release.set();capture.release.set()


async def entered(gate):
    assert await asyncio.to_thread(gate.entered.wait, 2), gate.name+' was not started'


@pytest.mark.parametrize('first', ['checkpoint', 'packet'])
def test_staging_and_backup_overlap_and_both_gate_the_original_driver(start_app, first):
    f = start_app

    async def exercise():
        f.app.start_symmetry_campaign(); task=f.app._symmetry_task
        try:
            await entered(f.checkpoint); await entered(f.packet)
            assert f.events == ['checkpoint-started', 'packet-started'] or set(f.events)=={'checkpoint-started','packet-started'}
            gate=getattr(f,first); gate.release.set()
            assert await asyncio.to_thread(gate.finished.wait,2)
            await asyncio.sleep(0)
            assert not task.done() and 'queue' not in f.events and 'driver' not in f.events
        finally:
            f.checkpoint.release.set(); f.packet.release.set()
        await task
        assert f.events[-2:] == ['queue','driver']
        assert len(f.captured)==1 and f.captured[0][0]==f.draft['protocol']
        assert f.captured[0][2]['dataset_id']=='source-hda'
        assert not f.state.sx_error and not f.state.sx_busy

    asyncio.run(exercise())


@pytest.mark.parametrize('failed', ['checkpoint','packet'])
def test_failure_waits_for_other_owned_operation_and_never_enters_driver(start_app, failed):
    f=start_app; getattr(f,failed).error=RuntimeError(failed+' rejected')

    async def exercise():
        f.app.start_symmetry_campaign();task=f.app._symmetry_task
        try:
            await entered(f.checkpoint);await entered(f.packet)
            getattr(f,failed).release.set()
            assert await asyncio.to_thread(getattr(f,failed).finished.wait,2)
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            f.checkpoint.release.set();f.packet.release.set()
        await task
        assert 'queue' not in f.events and 'driver' not in f.events
        assert f.state.sx_error==failed+' rejected' and not f.state.sx_busy
        assert f.checkpoint.finished.is_set() and f.packet.finished.is_set()

    asyncio.run(exercise())


def test_pending_checkpoint_is_not_retried_or_treated_as_success(start_app):
    f=start_app;f.checkpoint.error=CheckpointUploadPending('same-hda','queued')
    async def exercise():
        f.checkpoint.release.set();f.packet.release.set()
        f.app.start_symmetry_campaign();await f.app._symmetry_task
        assert f.events.count('checkpoint-started')==1 and f.events.count('packet-started')==1
        assert 'same-hda' in f.state.sx_error and 'driver' not in f.events
    asyncio.run(exercise())


@pytest.mark.parametrize('cancelled', ['checkpoint','packet'])
def test_cancelled_operation_is_not_a_successful_barrier(start_app,cancelled):
    f=start_app;getattr(f,cancelled).error=asyncio.CancelledError()
    async def exercise():
        f.checkpoint.release.set();f.packet.release.set()
        f.app.start_symmetry_campaign()
        with pytest.raises(asyncio.CancelledError):await f.app._symmetry_task
        assert 'driver' not in f.events and 'queue' not in f.events
        assert not f.state.sx_busy and f.app._model_start_cancel is None
    asyncio.run(exercise())


def test_task_cancellation_settles_threads_and_keeps_duplicate_start_blocked(start_app):
    f=start_app
    async def exercise():
        f.app.start_symmetry_campaign();task=f.app._symmetry_task
        try:
            await entered(f.checkpoint);await entered(f.packet)
            task.cancel();await asyncio.sleep(0)
            f.app.start_symmetry_campaign()
            assert f.app._symmetry_task is task and len(f.tasks)==1 and not task.done()
            task.cancel();await asyncio.sleep(0)
        finally:
            f.checkpoint.release.set();f.packet.release.set()
        with pytest.raises(asyncio.CancelledError):await task
        assert f.checkpoint.finished.is_set() and f.packet.finished.is_set()
        assert 'driver' not in f.events and 'queue' not in f.events
        assert not f.state.sx_busy and f.app._model_start_cancel is None
    asyncio.run(exercise())


@pytest.mark.parametrize('change', ['experiment','source','service','history','workspace','draft','template','protocol','approval'])
def test_changed_scope_cannot_claim_or_dispatch_after_the_barrier(start_app,change):
    f=start_app
    async def exercise():
        f.app.start_symmetry_campaign();task=f.app._symmetry_task
        try:
            await entered(f.checkpoint);await entered(f.packet)
            if change=='experiment':f.state.rf_experiment='exp2'
            elif change=='source':f.state.sx_start_source='scan2'
            elif change=='service':f.app.service=SimpleNamespace(history_id='history1')
            elif change=='history':f.service.history_id='history2'
            elif change=='workspace':f.app._refinement_store=lambda:object()
            elif change=='draft':f.app._symmetry_draft=copy.deepcopy(f.draft)
            elif change=='template':f.draft['template']['existing_ids'].append('scan2')
            elif change=='protocol':f.draft['protocol']['candidate']='changed'
            elif change=='approval':f.app._symmetry_approved='different'
        finally:
            f.checkpoint.release.set();f.packet.release.set()
        await task
        assert 'queue' not in f.events and 'driver' not in f.events
        assert 'scope or source context changed' in f.state.sx_error
    asyncio.run(exercise())


def test_cancellation_during_local_capture_cannot_start_a_late_upload(start_app):
    f=start_app;f.capture.release.clear()
    async def exercise():
        f.app.start_symmetry_campaign();task=f.app._symmetry_task
        try:
            await entered(f.checkpoint);await entered(f.capture)
            task.cancel();await asyncio.sleep(0)
        finally:
            f.capture.release.set();f.checkpoint.release.set();f.packet.release.set()
        with pytest.raises(asyncio.CancelledError):await task
        assert not f.packet.entered.is_set() and 'queue' not in f.events
    asyncio.run(exercise())


@pytest.mark.parametrize('raises',[False,True])
def test_unavailable_scheduler_does_not_retain_a_pending_start(start_app,raises):
    f=start_app
    def unavailable(work,name):
        if raises:raise RuntimeError('scheduler unavailable')
    f.app._refinement_schedule=unavailable
    if raises:
        with pytest.raises(RuntimeError,match='scheduler unavailable'):f.app.start_symmetry_campaign()
    else:f.app.start_symmetry_campaign()
    assert f.app._model_start_cancel is None and f.events==[]


def test_staging_does_not_mutate_the_reviewed_draft_or_source(start_app):
    f=start_app
    original_draft=copy.deepcopy(f.draft);original_source=copy.deepcopy(f.source)
    f.workspace.runtime.resolve=lambda _:f.source
    prepare=f.app._symmetry_transport(None).prepare_source
    def mutate_captured_arguments(protocol,root,source):
        protocol['transport_local_note']='not part of the reviewed object'
        source['transport_local_note']='not part of workspace state'
        prepare(protocol,root,source)
    f.app._symmetry_transport(None).prepare_source=mutate_captured_arguments
    async def exercise():
        f.checkpoint.release.set();f.packet.release.set()
        f.app.start_symmetry_campaign();await f.app._symmetry_task
    asyncio.run(exercise())
    assert f.draft==original_draft and f.source==original_source


@pytest.mark.parametrize('action', ['pause','end'])
def test_explicit_pause_or_end_waits_for_owned_staging_and_forbids_dispatch(start_app,action):
    f=start_app
    async def exercise():
        f.app.start_symmetry_campaign();task=f.app._symmetry_task
        try:
            await entered(f.checkpoint);await entered(f.packet)
            if action=='pause':
                f.app.pause_model_campaign();await f.app._model_pause_task
            else:
                ending=asyncio.create_task(f.app._pause_model_campaigns())
                await asyncio.sleep(0)
                assert not ending.done()
        finally:
            f.checkpoint.release.set();f.packet.release.set()
        await task
        if action=='end':await ending
        assert 'queue' not in f.events and 'driver' not in f.events
        assert 'paused or stopped' in f.state.sx_error
    asyncio.run(exercise())


def test_duplicate_click_and_completed_campaign_do_not_reupload_or_dispatch(start_app):
    f=start_app
    async def exercise():
        f.app.start_symmetry_campaign();first=f.app._symmetry_task
        try:
            await entered(f.checkpoint);await entered(f.packet)
            f.app.start_symmetry_campaign();assert f.app._symmetry_task is first
        finally:
            f.checkpoint.release.set();f.packet.release.set()
        await first
        f.app.start_symmetry_campaign();await f.app._symmetry_task
        assert f.events.count('driver')==f.events.count('queue')==1
        assert f.events.count('packet-started')==f.events.count('checkpoint-started')==1
        assert 'already exists' in f.state.sx_error
    asyncio.run(exercise())
