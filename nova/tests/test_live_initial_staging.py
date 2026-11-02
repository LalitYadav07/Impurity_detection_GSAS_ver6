"""Bounded initial staging preserves inputs and never launches on failed ownership."""
import asyncio
from contextlib import contextmanager
from copy import copy
import threading
from types import SimpleNamespace
import zipfile

import pytest

from test_live_setup import live, LocalGalaxy, CIF, TrialController
from radar_pd_nova.live_setup import configuration_digest
from radar_pd_nova.galaxy_service import GalaxyService
from radar_pd_nova.models import AnalysisConfig
from radar_pd_nova.powgen_controller import PowgenWatchController


LABELS=('configuration','main-phase CIF','instrument profile','candidate library')

class TrackingGalaxy(LocalGalaxy):
    galaxy_key='synthetic-key'
    def __init__(self,blocks=()):
        super().__init__()
        self.shared=SimpleNamespace(lock=threading.Lock(),started=[],finished=[],connections=[],
            active=0,peak=0,events={k:threading.Event() for k in LABELS},
            releases={k:threading.Event() for k in LABELS},blocks=set(blocks),failure=None)
    def staging_client(self):
        return copy(self)  # Shared fake server; frozen URL/key/History attributes.
    def _work(self,label):
        x=self.shared
        with x.lock:
            x.started.append(label);x.connections.append((self.galaxy_url,self.galaxy_key,self.history_id))
            x.active+=1;x.peak=max(x.peak,x.active)
        x.events[label].set()
        try:
            if label in x.blocks:assert x.releases[label].wait(5),'test did not release upload'
            if x.failure==label:raise RuntimeError('synthetic upload failure')
            with x.lock:x.finished.append(label)
        finally:
            with x.lock:x.active-=1
    def save_configuration(self,config):
        self._work('configuration')
        return super().save_configuration(config)
    def upload_document(self,path,*,label):
        kind=('main-phase CIF' if 'main phase' in label else
              'instrument profile' if 'instrument profile' in label else 'candidate library')
        self._work(kind)
        self.uploads.append((str(path),label))
        return self.history_id+'-'+path.name

def setup(live,tmp_path,blocks=()):
    service=TrackingGalaxy(blocks);live.service=service
    main=tmp_path/'main.cif';main.write_text(CIF)
    profile=tmp_path/'profile.instprm';profile.write_text('synthetic profile')
    library=tmp_path/'library.zip'
    with zipfile.ZipFile(library,'w') as z:
        z.writestr('pack/catalog_deduplicated.csv','id\n')
        z.writestr('pack/mp_experimental_stable.csv','id\n')
        z.writestr('pack/profiles64/example.npy','profile')
    s=live.server.state
    s.powgen_main_cif_source='computer';s.powgen_main_cif_path=str(main)
    s.powgen_instrument_source='computer';s.powgen_instrument_path=str(profile)
    s.live_database_source='archive';s.live_library_archive_source='computer'
    s.live_database_archive_path=str(library);s.flush()
    return service

async def wait(event):
    assert await asyncio.to_thread(event.wait,5)

def test_two_active_uploads_overlap_and_exact_inputs_are_preserved(live,tmp_path):
    service=setup(live,tmp_path,LABELS);x=service.shared
    async def run():
        task=asyncio.create_task(live._prepare_live_settings())
        await wait(x.events['configuration']);await wait(x.events['main-phase CIF'])
        assert x.peak==2 and set(x.started)=={'configuration','main-phase CIF'}
        x.releases['configuration'].set();await wait(x.events['instrument profile'])
        assert 'candidate library' not in x.started
        x.releases['instrument profile'].set();await wait(x.events['candidate library'])
        x.releases['candidate library'].set();x.releases['main-phase CIF'].set()
        settings,config=await task
        assert settings.main_cif_dataset_id=='local-history-main.cif'
        assert settings.instrument_profile_dataset_id=='local-history-profile.instprm'
        assert settings.database_dataset_id=='local-history-library.zip'
        assert configuration_digest(config)==configuration_digest(service.configs[settings.configuration_dataset_id])
        assert settings.history_id=='local-history' and settings.selected_banks==(2,)
    asyncio.run(run())
    assert x.peak==2 and len(x.finished)==4 and not service.jobs

@pytest.mark.parametrize('kind',['stop','clear','history','origin','actor','draft'])
def test_ownership_change_during_upload_prevents_later_uploads_or_submit(live,tmp_path,kind):
    service=setup(live,tmp_path,['configuration','main-phase CIF']);x=service.shared
    async def run():
        task=asyncio.create_task(live.test_live_scan())
        await wait(x.events['configuration']);await wait(x.events['main-phase CIF'])
        if kind=='stop':live.stop_powgen_monitoring()
        elif kind=='clear':live._result_context=object()
        elif kind=='history':service.history_id='new-history'
        elif kind=='origin':service.galaxy_url='https://other.invalid'
        elif kind=='actor':service.galaxy_key='new-actor'
        else:live.server.state.live_sample_elements='Fe';live.server.state.flush()
        x.releases['configuration'].set();x.releases['main-phase CIF'].set()
        await task
    asyncio.run(run())
    assert set(x.started)=={'configuration','main-phase CIF'}
    assert len(x.finished)==2 and not service.jobs
    assert all(c==('https://local.invalid','synthetic-key','local-history') for c in x.connections)
    assert not live._live_uploaded_assets and not live._live_saved_configurations

def test_failed_upload_drains_other_ack_and_never_admits_a_scan(live,tmp_path):
    service=setup(live,tmp_path,['configuration','main-phase CIF']);x=service.shared
    x.failure='main-phase CIF'
    async def run():
        task=asyncio.create_task(live.test_live_scan())
        await wait(x.events['configuration']);await wait(x.events['main-phase CIF'])
        x.releases['main-phase CIF'].set()
        await asyncio.sleep(.03)
        x.releases['configuration'].set()
        await task
    asyncio.run(run())
    assert set(x.started)=={'configuration','main-phase CIF'} and x.finished==['configuration']
    assert not service.jobs and 'synthetic upload failure' in live.server.state.error_message

def test_coroutine_cancellation_drains_acknowledgements_without_more_posts(live,tmp_path):
    service=setup(live,tmp_path,['configuration','main-phase CIF']);x=service.shared
    async def run():
        task=asyncio.create_task(live._prepare_live_settings())
        await wait(x.events['configuration']);await wait(x.events['main-phase CIF'])
        task.cancel();await asyncio.sleep(0)
        assert not task.done()
        x.releases['configuration'].set();x.releases['main-phase CIF'].set()
        with pytest.raises(asyncio.CancelledError):await task
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    asyncio.run(run())
    assert set(x.started)=={'configuration','main-phase CIF'} and len(x.finished)==2
    assert not service.jobs and not live._live_uploaded_assets

@pytest.mark.parametrize('changed', ['history_id', 'galaxy_url', 'galaxy_key'])
def test_ready_caches_are_reused_only_in_the_same_connection(live,tmp_path,changed):
    service=setup(live,tmp_path)
    async def run():
        first,_=await live._prepare_live_settings()
        second,_=await live._prepare_live_settings()
        assert first==second and len(service.shared.started)==4
        setattr(service,changed,{'history_id':'different-history',
            'galaxy_url':'https://different.invalid','galaxy_key':'different-actor'}[changed])
        third,_=await live._prepare_live_settings()
        assert third.history_id==service.history_id
        assert third.main_cif_dataset_id==service.history_id+'-main.cif'
    asyncio.run(run())
    assert len(service.shared.started)==8 and not service.jobs


def test_new_workspace_shares_two_upload_slots_while_old_acknowledgements_drain(live,tmp_path):
    service=setup(live,tmp_path,LABELS);x=service.shared
    async def run():
        old=asyncio.create_task(live._prepare_live_settings())
        await wait(x.events['configuration']);await wait(x.events['main-phase CIF'])
        old_owner=live._live_staging_owner
        live._result_context=object()
        new=asyncio.create_task(live._prepare_live_settings())
        await asyncio.sleep(.03)
        assert live._live_staging_owner is not old_owner
        assert x.started==['configuration','main-phase CIF']
        x.releases['configuration'].set();x.releases['main-phase CIF'].set()
        assert await old is None
        await wait(x.events['instrument profile']);await wait(x.events['candidate library'])
        x.releases['instrument profile'].set();x.releases['candidate library'].set()
        assert await new is not None
    asyncio.run(run())
    assert x.peak==2 and len(x.started)==6 and len(x.finished)==6
    assert not service.jobs

def test_real_staging_client_retains_ready_upload_contract_and_frozen_connection(tmp_path,monkeypatch):
    service=GalaxyService('https://original.invalid','synthetic','history',output_root=tmp_path)
    client=service.staging_client()
    service.galaxy_url='https://new.invalid';service.galaxy_key='new';service.history_id='new'
    assert (client.galaxy_url,client.galaxy_key,client.history_id)==('https://original.invalid','synthetic','history')
    stores=[];waits=[]
    @contextmanager
    def store():
        obj=object();stores.append(obj);yield obj
    def upload(path,connection,label,*,wait_ready=True):
        assert connection in stores
        waits.append(wait_ready)
        return SimpleNamespace(id='ready-hda')
    monkeypatch.setattr(client,'_store',store)
    monkeypatch.setattr(client,'_upload_dataset',upload)
    asset=tmp_path/'main.cif';asset.write_text(CIF)
    assert client.upload_document(asset,label='main')=='ready-hda'
    action=client.save_configuration(AnalysisConfig(sample_elements=['Al','O']))
    assert action.outputs['config_output']=='ready-hda'
    assert waits==[True,True] and len(stores)==2


def test_changed_file_waiting_for_a_slot_is_not_uploaded(live,tmp_path):
    service=setup(live,tmp_path,['configuration','main-phase CIF']);x=service.shared
    async def run():
        task=asyncio.create_task(live._prepare_live_settings())
        await wait(x.events['configuration']);await wait(x.events['main-phase CIF'])
        (tmp_path/'profile.instprm').write_text('different profile bytes')
        x.releases['configuration'].set();x.releases['main-phase CIF'].set()
        with pytest.raises(ValueError,match='changed'):await task
    asyncio.run(run())
    assert set(x.started)=={'configuration','main-phase CIF'} and not service.jobs

def test_old_staging_finally_does_not_release_new_context_busy_flag(live,tmp_path):
    service=setup(live,tmp_path,['configuration','main-phase CIF']);x=service.shared
    async def run():
        task=asyncio.create_task(live.test_live_scan())
        await wait(x.events['configuration']);await wait(x.events['main-phase CIF'])
        live._result_context=object()
        live.server.state.live_launch_busy=True
        live.server.state.live_test_message='New workspace owns this progress'
        x.releases['configuration'].set();x.releases['main-phase CIF'].set()
        await task
        assert live.server.state.live_launch_busy
        assert live.server.state.live_test_message=='New workspace owns this progress'
    asyncio.run(run())
    assert not service.jobs


def revoke_launch(live, kind):
    if kind == 'stop':
        live.stop_powgen_monitoring()
    elif kind == 'clear':
        live._result_context = object()
        live._powgen_controller = None
        live.records.clear()
    elif kind == 'end':
        live._invalidate_powgen_resume()
        live.server.state.session_closing = True
    elif kind == 'history':
        live.service.history_id = 'replacement-history'
    elif kind == 'origin':
        live.service.galaxy_url = 'https://replacement.invalid'
    elif kind == 'actor':
        live.service.galaxy_key = 'replacement-actor'
    else:
        live.server.state.live_sample_elements = 'Fe'
        live.server.state.flush()


@pytest.mark.parametrize('kind', ['stop', 'clear', 'end', 'history', 'origin', 'actor', 'draft'])
def test_trial_discovery_cannot_submit_after_launch_is_revoked(live, monkeypatch, kind):
    entered, release = threading.Event(), threading.Event()
    original = TrialController.discover
    def discover(controller, **kwargs):
        entered.set()
        assert release.wait(5)
        original(controller, **kwargs)
    monkeypatch.setattr(TrialController, 'discover', discover)
    async def run():
        task = asyncio.create_task(live.test_live_scan())
        await wait(entered)
        revoke_launch(live, kind)
        release.set()
        await task
    asyncio.run(run())
    assert not live.service.jobs


@pytest.mark.parametrize('kind', ['stop', 'clear', 'end', 'history', 'origin', 'actor', 'draft'])
def test_watch_restore_cannot_publish_or_start_after_launch_is_revoked(live, monkeypatch, kind):
    entered, release = threading.Event(), threading.Event()
    def restore(controller):
        entered.set()
        assert release.wait(5)
        return False
    monkeypatch.setattr(TrialController, 'restore_latest_state', restore)
    async def run():
        task = asyncio.create_task(live.start_powgen_monitoring())
        await wait(entered)
        revoke_launch(live, kind)
        release.set()
        await task
    asyncio.run(run())
    assert not live.service.jobs
    assert not live.server.state.powgen_monitoring
    assert live._powgen_monitor_task is None
    assert live._powgen_controller is None


@pytest.mark.parametrize('kind', ['stop', 'clear', 'end', 'history', 'origin', 'actor', 'draft'])
def test_trial_guard_is_rechecked_after_submission_input_preparation(live, monkeypatch, kind):
    entered, release = threading.Event(), threading.Event()
    original = TrialController.launch_submission
    context = {}
    def submit(snapshot, *, before_submit):
        entered.set()
        assert release.wait(5)
        before_submit()
        return original(context['controller'], context['run'])
    def launch(controller, run):
        context.update(controller=controller, run=run)
        return PowgenWatchController._submit_live_snapshot(
            controller, SimpleNamespace(scan_metadata={}), object()
        )
    live.service.submit_snapshot = submit
    monkeypatch.setattr(TrialController, 'launch_submission', launch)
    async def run():
        task = asyncio.create_task(live.test_live_scan())
        await wait(entered)
        revoke_launch(live, kind)
        release.set()
        await task
    asyncio.run(run())
    assert not live.service.jobs


@pytest.mark.parametrize('retry', [False, True])
def test_final_post_hook_composes_existing_retry_guard(retry):
    calls = []
    controller = object.__new__(PowgenWatchController)
    controller._before_live_submit = lambda: calls.append('launch')
    controller._validate_retry_submission = lambda run: calls.append('retry')
    def submit(snapshot, *, before_submit):
        before_submit()
        calls.append('post')
        return 'accepted'
    controller.service = SimpleNamespace(submit_snapshot=submit)
    run = SimpleNamespace(scan_metadata={'retry_source_proof': {'job': 'old'}} if retry else {})
    assert controller._submit_live_snapshot(run, object()) == 'accepted'
    assert calls == (['launch', 'retry', 'post'] if retry else ['launch', 'post'])


def test_normal_watch_has_no_initial_launch_guard_lifetime():
    controller = object.__new__(PowgenWatchController)
    calls = []
    controller.service = SimpleNamespace(submit_snapshot=lambda snapshot: calls.append(snapshot))
    token = object()
    controller._submit_live_snapshot(SimpleNamespace(scan_metadata={}), token)
    assert calls == [token]


@pytest.mark.parametrize('kind', ['stop', 'clear'])
def test_accepted_trial_receipt_survives_owner_loss_without_reopening_workspace(live, monkeypatch, kind):
    entered, release = threading.Event(), threading.Event()
    original = TrialController.launch_submission
    def launch(controller, run):
        record = original(controller, run)  # Galaxy accepted before revocation.
        entered.set()
        assert release.wait(5)
        return record
    monkeypatch.setattr(TrialController, 'launch_submission', launch)
    async def run():
        task = asyncio.create_task(live.test_live_scan())
        await wait(entered)
        revoke_launch(live, kind)
        live.server.state.workspace_view = 'refinement'
        release.set()
        await task
    asyncio.run(run())
    assert len(live.service.jobs) == 1
    record = live.service.jobs[0]
    assert record.galaxy_job_id == 'job-1' and record.provenance['live_trial']['run'] == 100
    assert live.server.state.workspace_view == 'refinement'
    assert live.server.state.selected_run_uid != record.uid
    assert (record.uid in live.records) is (kind == 'stop')


def test_trial_retains_existing_submission_service_and_acknowledgement_cache(live, tmp_path):
    service = setup(live, tmp_path)
    asyncio.run(live.test_live_scan())
    assert TrialController.instances[0].service is service
    assert len(service.jobs) == 1


@pytest.mark.parametrize('mode', ['trial', 'watch'])
def test_late_launch_error_does_not_replace_stop_feedback(live, monkeypatch, mode):
    entered, release = threading.Event(), threading.Event()
    def delayed_failure(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        raise RuntimeError('late error from old launch')
    monkeypatch.setattr(TrialController,
        'discover' if mode == 'trial' else 'restore_latest_state', delayed_failure)
    async def run():
        action = live.test_live_scan if mode == 'trial' else live.start_powgen_monitoring
        task = asyncio.create_task(action())
        await wait(entered)
        live.stop_powgen_monitoring()
        stopped = live.server.state.powgen_message
        release.set()
        await task
        assert live.server.state.powgen_message == stopped
        assert 'late error' not in live.server.state.error_message
        assert not live.server.state.live_launch_busy
    asyncio.run(run())
    assert not live.service.jobs


