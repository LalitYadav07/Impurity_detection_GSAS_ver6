"""Coalesce completion backups without weakening claim/ack crash recovery."""
import asyncio
from copy import deepcopy
import threading
from types import SimpleNamespace

import pytest

from radar_pd_nova.refinement_controller import RefinementMixin
from radar_pd_nova.checkpoint_uploads import CheckpointUploadPending
from radar_pd_nova.refinement_sources import RefinementScope
from test_refinement_integration import create, source, approved


class State(SimpleNamespace):
    def flush(self): pass


class App(RefinementMixin):
    def __init__(self, workspace):
        self._refinement_workspace = workspace
        self._refinement_checkpoint_task = None
        self._refinement_checkpoint_dirty = False
        self.server = SimpleNamespace(state=State(rf_experiment='exp', rf_follow=True,
            rf_selected_job='', rf_error='', rf_queue_running=False))
        self.selected = []
    def _refinement_sync(self): pass
    def _refinement_schedule(self, work, name): return asyncio.create_task(work, name=name)
    async def _select_refinement_job(self, uid): self.selected.append(uid)


def queued(tmp_path, count=3):
    w, rt = create(tmp_path)
    w.observe([source(str(i)) for i in range(1, count+1)])
    recipe = approved(w, RefinementScope('exp'))
    w.start(recipe['id'], recipe['selection_token'])
    rt.kind = 'galaxy'
    snapshots, posts = [], []
    def save(document):
        snapshots.append(deepcopy(document)); return 'backup-' + str(len(snapshots))
    rt.save_workspace = save
    def run(action, src, directory, **kwargs):
        assert action == 'run'
        checkpoint_jobs = snapshots[-1]['jobs']
        claim = checkpoint_jobs[kwargs['dispatch_key']]
        assert claim['status'] == 'running' and claim['attempts'] == 1
        for previous in posts: assert checkpoint_jobs[previous]['status'] == 'completed'
        posts.append(kwargs['dispatch_key'])
        kwargs['on_submitted']({'uid': 'native-'+src['id'], 'galaxy_job_id': 'native-'+src['id']})
        assert snapshots[-1]['jobs'][kwargs['dispatch_key']]['remote_action']['galaxy_job_id'] == 'native-'+src['id']
        return {'status': 'completed', 'directory': str(directory)}
    rt.call = run
    return w, rt, snapshots, posts


async def finish(app):
    await asyncio.gather(app._refinement_pump(), return_exceptions=True)
    if app._refinement_checkpoint_task:
        await app._refinement_checkpoint_task


def test_completion_is_in_next_claim_checkpoint_with_only_one_final_backup(tmp_path):
    w, _, snapshots, posts = queued(tmp_path)
    app = App(w)
    asyncio.run(finish(app))
    assert len(posts) == 3 and len(set(posts)) == 3
    assert len(snapshots) == 7  # three claims, three acknowledgments, one final completion
    assert all(j['status'] == 'completed' for j in snapshots[-1]['jobs'].values())


@pytest.mark.parametrize('stop', ['pause', 'cancel', 'view_error'])
def test_deferred_completion_is_saved_when_next_claim_never_happens(tmp_path, stop):
    w, _, snapshots, posts = queued(tmp_path)
    app = App(w)
    async def stop_after_first(uid):
        if stop == 'pause':
            w.pause('exp'); app._refinement_checkpoint()
        elif stop == 'cancel': raise asyncio.CancelledError()
        else: raise ValueError('Result display stopped')
    app._select_refinement_job = stop_after_first
    asyncio.run(finish(app))
    assert len(posts) == 1
    final = snapshots[-1]['jobs']
    assert final[posts[0]]['status'] == 'completed'
    assert final[posts[0]]['remote_action']['galaxy_job_id']
    assert sorted(j['status'] for key, j in final.items() if key != posts[0]) == (['paused', 'paused'] if stop=='pause' else ['queued', 'queued'])


def test_required_claim_upload_failure_prevents_native_post(tmp_path):
    w, rt, snapshots, posts = queued(tmp_path, count=1)
    def unavailable(_): raise RuntimeError('Checkpoint upload unavailable')
    rt.save_workspace = unavailable
    app = App(w)
    asyncio.run(finish(app))
    assert not posts
    assert next(iter(w.state['jobs'].values()))['status'] == 'failed'
    assert 'backup failed' in app.server.state.rf_error


def test_pending_checkpoint_pauses_untouched_queue_and_resumes_same_claim(tmp_path):
    w, rt, snapshots, posts = queued(tmp_path)
    ready_save = rt.save_workspace
    rt.save_workspace = lambda _: (_ for _ in ()).throw(CheckpointUploadPending('existing-upload', 'queued'))
    first = w.run_next()
    assert first['status'] == 'paused' and not posts
    assert {j['status'] for j in w.state['jobs'].values()} == {'paused'}
    assert sum(j['attempts'] for j in w.state['jobs'].values()) == 1
    assert w.run_next() is None  # No cascade of failures or replacement submissions.
    assert all(not r['armed'] for r in w.state['recipes'].values())
    rt.save_workspace = ready_save
    def resumed(action, src, directory, **kwargs):
        claim = snapshots[-1]['jobs'][kwargs['dispatch_key']]
        assert claim['id'] == first['id'] and claim['attempts'] == 2
        posts.append(claim['id'])
        kwargs['on_submitted']({'galaxy_job_id': 'one-accepted-job'})
        return {'status': 'completed', 'directory': str(directory)}
    rt.call = resumed
    w.resume_queue('exp')
    result = w.run_next()
    assert result['id'] == first['id'] and result['status'] == 'completed'
    assert posts == [first['id']]
    assert sum(j['status'] == 'queued' for j in w.state['jobs'].values()) == 2


def test_pending_ack_backup_preserves_accepted_job_and_pauses_remaining_work(tmp_path):
    w, rt, snapshots, posts = queued(tmp_path)
    ready_save = rt.save_workspace
    def save(document):
        if any(j.get('remote_action') for j in document['jobs'].values()):
            raise CheckpointUploadPending('ack-upload', 'running')
        return ready_save(document)
    rt.save_workspace = save
    first = w.run_next()
    assert first['status'] == 'interrupted'
    remote = first['remote_action']
    assert remote['galaxy_job_id'] and posts == [first['id']]
    assert all(j['status'] == 'paused' for j in w.state['jobs'].values() if j['id'] != first['id'])
    assert w.run_next() is None
    rt.save_workspace = ready_save
    def recover(action, src, directory, **kwargs):
        assert kwargs['resume'] == remote and kwargs['dispatch_key'] == first['id']
        return {'status': 'completed', 'directory': str(directory)}
    rt.call = recover
    w.retry(first['id'])
    recovered = w.run_next()
    assert recovered['status'] == 'completed' and posts == [first['id']]


def test_old_checkpoint_failure_cannot_overwrite_a_replacement_workspace():
    entered, release = threading.Event(), threading.Event()
    calls = []
    def old_checkpoint():
        calls.append('old'); entered.set(); assert release.wait(3)
        raise ValueError('Old history unavailable')
    old = SimpleNamespace(checkpoint=old_checkpoint)
    new = SimpleNamespace(checkpoint=lambda: calls.append('new'))
    app = App(old)
    async def scenario():
        app._refinement_checkpoint()
        old_task = app._refinement_checkpoint_task
        assert await asyncio.to_thread(entered.wait, 2)
        app._refinement_workspace = new
        app.server.state.rf_error = 'New workspace status'
        app._refinement_checkpoint()
        new_task = app._refinement_checkpoint_task
        await new_task
        release.set(); await old_task
    asyncio.run(scenario())
    assert calls == ['old', 'new']
    assert app.server.state.rf_error == 'New workspace status'
