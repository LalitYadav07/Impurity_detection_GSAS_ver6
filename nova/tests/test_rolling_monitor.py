"""Exercise admission, upload draining and checkpoints through the real UI loop."""
import asyncio
import threading
from types import SimpleNamespace

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.models import RunRecord, RunStatus


class Controller:
    source_directory = '/read-only/experiment'
    settings = SimpleNamespace(max_active_jobs=5)
    refresh_errors = {}

    def __init__(self, count=12, blocked=(2,)):
        runs = [SimpleNamespace(run_id=f'scan-{n}', n=n) for n in range(1, count + 1)]
        self.state = SimpleNamespace(discovered={r.run_id:r for r in runs}, submitted={}, completed={}, failed={})
        self.blocked = set(blocked)
        self.release = threading.Event()
        self.two_entered = threading.Event()
        self.sixth_entered = threading.Event()
        self.checkpointed = threading.Event()
        self.launched = []
        self.checkpoints = []
        self.working = self.max_working = 0
        self.max_reserved_and_active = 0
        self.lock = threading.Lock()

    def discover(self, *, checkpoint): assert not checkpoint
    def reconcile_recent_runs(self, *, checkpoint):
        assert not checkpoint
        return 0

    def due_submissions(self, *, reserved_run_ids=None):
        reserved = set(reserved_run_ids or ())
        available = max(0, 5-len(self.state.submitted)-len(reserved))
        ready = [r for r in self.state.discovered.values() if r.run_id not in reserved][:available]
        self.max_reserved_and_active = max(self.max_reserved_and_active,
            len(self.state.submitted)+len(reserved)+len(ready))
        return ready

    def launch_submission(self, run):
        with self.lock:
            self.working += 1
            self.max_working = max(self.max_working, self.working)
            self.launched.append(run.n)
            if len(self.launched) >= 2: self.two_entered.set()
            if run.n == 6: self.sixth_entered.set()
        try:
            if run.n in self.blocked:
                assert self.release.wait(12), 'Test did not release the blocked upload'
            return RunRecord(uid=run.run_id, name=run.run_id, mode='full', history_id='local', galaxy_job_id=run.run_id)
        finally:
            with self.lock: self.working -= 1

    def acknowledge_submission(self, run, record, *, persist):
        assert not persist
        self.state.submitted[run.run_id] = self.state.discovered.pop(run.run_id)
        return record

    def refresh(self, *, checkpoint):
        assert not checkpoint
        result = {}
        for identifier in list(self.state.submitted):
            self.state.completed[identifier] = self.state.submitted.pop(identifier)
            result[identifier] = RunRecord(uid=identifier, name=identifier, mode='full', history_id='local', status=RunStatus.OK)
        return result

    def capture_checkpoint(self):
        return len(self.state.completed)

    def persist_state(self, snapshot):
        self.checkpoints.append(snapshot)
        if snapshot >= 5:
            self.checkpointed.set()


def app_with_controller(controller):
    app = RadarPdNovaApp()
    app._powgen_controller = controller
    app._sync_powgen_rows = lambda: None
    app._sync_runs = lambda: None
    app.server.state.powgen_monitoring = True
    return app


def test_slow_upload_does_not_block_slot_refill_or_durable_checkpoints():
    controller = Controller()
    app = app_with_controller(controller)

    async def run():
        task = asyncio.create_task(app._powgen_monitor_loop())
        app._powgen_monitor_task = task
        try:
            assert await asyncio.to_thread(controller.sixth_entered.wait, 8)
            assert not controller.release.is_set()
            assert await asyncio.to_thread(controller.checkpointed.wait, 5)
            assert 'scan-2' not in controller.state.completed
            assert max(controller.checkpoints) >= 5
        finally:
            app.stop_powgen_monitoring()
            controller.release.set()
            await asyncio.wait_for(task, 7)
        assert len(controller.state.completed) >= 6
        assert controller.max_working <= 2
        assert controller.max_reserved_and_active <= 5
        assert not app.server.state.live_launch_busy
        assert not app._powgen_submissions_in_flight

    asyncio.run(run())


def test_stop_drains_in_flight_acknowledgements_but_never_starts_waiting_uploads():
    controller = Controller(blocked=(1, 2))
    app = app_with_controller(controller)

    async def run():
        task = asyncio.create_task(app._powgen_monitor_loop())
        app._powgen_monitor_task = task
        try:
            assert await asyncio.to_thread(controller.two_entered.wait, 4)
            app.stop_powgen_monitoring()
            assert not task.cancelled() and app.server.state.live_launch_busy
            controller.release.set()
            await asyncio.wait_for(task, 7)
        finally:
            controller.release.set()
        assert controller.launched == [1, 2]
        assert set(controller.state.completed) == {'scan-1', 'scan-2'}
        assert len(controller.state.discovered) == 10
        assert controller.checkpoints[-1] == 2
        assert not app.server.state.live_launch_busy
        assert app._powgen_monitor_task is None

    asyncio.run(run())


def test_slow_checkpoint_upload_does_not_hold_analysis_admission():
    class SlowCheckpoint(Controller):
        def __init__(self):
            super().__init__(count=8,blocked=())
            self.checkpoint_entered=threading.Event()
            self.checkpoint_release=threading.Event()
        def persist_state(self,snapshot):
            if not self.checkpoints:
                self.checkpoint_entered.set()
                assert self.checkpoint_release.wait(10)
            super().persist_state(snapshot)
    controller=SlowCheckpoint();app=app_with_controller(controller)
    async def run():
        task=asyncio.create_task(app._powgen_monitor_loop());app._powgen_monitor_task=task
        try:
            assert await asyncio.to_thread(controller.checkpoint_entered.wait,2)
            assert await asyncio.to_thread(controller.sixth_entered.wait,7)
            assert not controller.checkpoint_release.is_set()
        finally:
            app.stop_powgen_monitoring();controller.checkpoint_release.set()
            await asyncio.wait_for(task,7)
        assert controller.checkpoints[-1]==len(controller.state.completed)
        # Refill may start scan 6 before staggered scans 2-5. Stop must retain
        # every actually started job, without starting those waiting uploads.
        assert 'scan-6' in controller.state.completed
        assert set(controller.state.completed)=={f'scan-{n}' for n in controller.launched}
    asyncio.run(run())
