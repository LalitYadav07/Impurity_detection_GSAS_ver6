"""Restart and explicit Continue cannot bypass an incomplete Start barrier."""
import asyncio
import copy
import threading

import pytest

from test_model_start_staging import start_app, entered
import radar_pd_nova.symmetry_controller as controller
import radar_pd_nova.model_campaigns as enrollment
from gpx_recipe.symmetry.campaign import Campaign
from gpx_recipe.symmetry.contracts import digest
from radar_pd_nova.checkpoint_uploads import CheckpointUploadPending


def following_case(f):
    f.draft['template'].update(future=True,maximum_scans=2,origin='radar',seen_ids=['scan1'])
    f.source.update(experiment='exp1',sample_key='sample1')
    f.app._symmetry_approved=digest(f.draft['template'])


def saved_campaign(f,tmp_path):
    return Campaign(tmp_path/'campaigns'/('campaign-'+f.app._symmetry_approved[:12]))


@pytest.mark.parametrize('stop',['checkpoint_failure','checkpoint_pending','cancel','pause','end','source_change'])
def test_interrupted_start_and_saved_bootstrap_remain_unarmed_after_restore(start_app,monkeypatch,tmp_path,stop):
    f=start_app;following_case(f)
    if stop=='checkpoint_failure':f.checkpoint.error=RuntimeError('source backup failed')
    if stop=='checkpoint_pending':f.checkpoint.error=CheckpointUploadPending('same-source-hda','queued')
    async def exercise():
        f.app.start_symmetry_campaign();task=f.app._symmetry_task
        ending=None
        try:
            await entered(f.checkpoint);await entered(f.packet)
            # A prepare_source side effect could have uploaded this bootstrap.
            bootstrap=saved_campaign(f,tmp_path).snapshot()
            assert bootstrap['settings']['paused'] is True
            assert bootstrap['settings']['following'] is False
            assert bootstrap['settings']['start_barrier']['state']=='incomplete'
            if stop=='cancel':task.cancel();await asyncio.sleep(0)
            elif stop=='pause':f.app.pause_model_campaign();await f.app._model_pause_task
            elif stop=='end':ending=asyncio.create_task(f.app._pause_model_campaigns());await asyncio.sleep(0)
            elif stop=='source_change':f.state.sx_start_source='replacement'
        finally:f.checkpoint.release.set();f.packet.release.set()
        try:await task
        except asyncio.CancelledError:assert stop=='cancel'
        if ending:await ending
        assert 'driver' not in f.events and 'queue' not in f.events
        return bootstrap
    bootstrap=asyncio.run(exercise())
    for name,snapshot in [('bootstrap',bootstrap),('final',saved_campaign(f,tmp_path).snapshot())]:
        restored=Campaign(tmp_path/name);restored.restore(snapshot,{})
        assert restored.setting('paused') is True and restored.setting('following') is False
        assert not enrollment.start_barrier_ready(restored)
        later={**f.source,'id':'later','scan_label':'Later'}
        monkeypatch.setattr(enrollment,'choices',lambda *_:[(later,'radar')])
        # Even a stray unpause cannot defeat the incomplete marker.
        restored.set_setting('paused',False)
        assert 'later' not in enrollment.queue_sources(restored,None)
        with pytest.raises(ValueError,match='Initial model preparation'):
            enrollment.enroll_source(restored,None,'scan1',lambda *_:None)


def test_continue_refuses_failed_start_before_recovery_or_any_io(start_app,monkeypatch,tmp_path):
    f=start_app;f.checkpoint.error=RuntimeError('source checkpoint unavailable')
    async def exercise():
        f.checkpoint.release.set();f.packet.release.set()
        f.app.start_symmetry_campaign();await f.app._symmetry_task
        f.state.sx_campaign=saved_campaign(f,tmp_path).root.name
        monkeypatch.setattr(controller,'restore_captured_sources',lambda *_:pytest.fail('no recovery before Start barrier'))
        f.app._symmetry_transport=lambda *_:pytest.fail('no transport before Start barrier')
        f.app.continue_symmetry_campaign();await f.app._symmetry_task
        assert 'Continue cannot skip' in f.state.sx_error
        assert f.events.count('checkpoint-started')==1 and 'driver' not in f.events
    asyncio.run(exercise())


def test_successful_start_retry_rechecks_both_operations_then_arms_exact_scope(start_app,tmp_path):
    f=start_app;following_case(f)
    original=copy.deepcopy(f.draft['template'])
    f.checkpoint.error=RuntimeError('first backup failed')
    async def exercise():
        f.checkpoint.release.set();f.packet.release.set()
        f.app.start_symmetry_campaign();await f.app._symmetry_task
        saved=saved_campaign(f,tmp_path);first=saved.setting('start_barrier')
        assert first['state']=='incomplete' and saved.setting('paused') is True
        f.checkpoint.error=None
        f.app.start_symmetry_campaign();await f.app._symmetry_task
        marker=saved.setting('start_barrier')
        assert marker['attempt']!=first['attempt'] and marker['state']=='complete'
        assert saved.setting('paused') is False and saved.setting('following') is True
        assert saved.setting('template')==original and enrollment.start_barrier_ready(saved)
        assert f.events.count('checkpoint-started')==f.events.count('packet-started')==2
        assert f.events.count('driver')==1 and f.events.count('queue')==1
    asyncio.run(exercise())


def test_pause_after_barrier_before_driver_keeps_saved_start_unarmed(start_app,monkeypatch,tmp_path):
    f=start_app;following_case(f)
    entered_queue=threading.Event();release_queue=threading.Event()
    def queue(*_):
        entered_queue.set();assert release_queue.wait(5)
    monkeypatch.setattr(controller,'queue_sources',queue)
    async def exercise():
        f.checkpoint.release.set();f.packet.release.set()
        f.app.start_symmetry_campaign();task=f.app._symmetry_task
        try:
            assert await asyncio.to_thread(entered_queue.wait,2)
            f.app.pause_model_campaign();await f.app._model_pause_task
            saved=saved_campaign(f,tmp_path)
            assert saved.setting('paused') is True
            assert saved.setting('following') is False
            assert not enrollment.start_barrier_ready(saved)
        finally:release_queue.set()
        await task
        assert 'driver' not in f.events and not enrollment.start_barrier_ready(saved)
    asyncio.run(exercise())


def test_stale_start_cleanup_cannot_disarm_a_newer_attempt(tmp_path):
    c=Campaign(tmp_path/'campaign')
    old={'schema':enrollment.START_BARRIER_SCHEMA,'attempt':'old','template_sha256':'a'*64}
    new=dict(old,attempt='new')
    enrollment.store_start_barrier(c,old)
    enrollment.store_start_barrier(c,new)
    assert not enrollment.store_start_barrier(c,old,complete=True,current_only=True)
    assert not enrollment.store_start_barrier(c,old,current_only=True)
    assert c.setting('start_barrier')['attempt']=='new'


def test_legacy_no_marker_and_completed_matching_marker_remain_compatible(tmp_path):
    c=Campaign(tmp_path/'legacy');template={'existing_ids':['one']}
    c.set_setting('template',template)
    assert enrollment.start_barrier_ready(c)
    marker={'schema':enrollment.START_BARRIER_SCHEMA,'attempt':'one','template_sha256':digest(template)}
    enrollment.store_start_barrier(c,marker,complete=True,following=True)
    assert enrollment.start_barrier_ready(c)
    c.set_setting('template',dict(template,existing_ids=['different']))
    assert not enrollment.start_barrier_ready(c)
