"""Bounded initial staging preserves inputs and never launches on failed ownership."""
import asyncio
from contextlib import contextmanager
from copy import copy
import threading
from types import SimpleNamespace
import zipfile

import pytest

from test_live_setup import live, LocalGalaxy, CIF
from radar_pd_nova.live_setup import configuration_digest
from radar_pd_nova.galaxy_service import GalaxyService
from radar_pd_nova.models import AnalysisConfig


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


