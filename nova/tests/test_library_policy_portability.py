import asyncio
from copy import deepcopy
from pathlib import Path
import shutil
import threading
import json
import zipfile
from types import SimpleNamespace

import pytest

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.galaxy_service import GalaxyService
from radar_pd_nova.library_setup import LibraryDraft,activation_for_review,inspect_library_review,library_source_key,require_library_activation
from radar_pd_nova.models import AnalysisConfig,RunStatus
from test_library_occupancy_activation import archive


@pytest.mark.parametrize('new_archive',[False,True])
def test_portable_history_configuration_uses_current_explicit_review(tmp_path,new_archive):
    app=RadarPdNovaApp();state=app.server.state
    current=activation_for_review(inspect_library_review(archive(tmp_path)))
    stored=AnalysisConfig(sample_elements=['C','Li'],instrument_mode='tof',library_review={**current,'source_key':'old-history-selection'})
    saves=[]
    def save(config):
        saves.append(deepcopy(config))
        return SimpleNamespace(status=RunStatus.OK,outputs={'config_output':'derived-config'})
    app.service=SimpleNamespace(load_configuration_dataset=lambda _:stored.model_copy(deep=True),save_configuration=save)
    app._schedule_utility=lambda work,*_:work.close()
    app._register_utility=lambda *_:None;app._remember_history_dataset=lambda *_:None
    state.live_configuration_source='history';state.powgen_configuration_dataset_id='old-config'
    state.live_database_source='archive';state.live_library_archive_source='galaxy';state.powgen_database_dataset_id='new-history-library'
    state.flush()
    with pytest.raises(ValueError,match='review'):
        app._live_configuration()
    if new_archive:
        current={**current,'archive_sha256':'e'*64}
    state.live_library_review={**current,'source_key':library_source_key(LibraryDraft(state,'live'))}
    state.live_library_review_source=state.live_library_review['source_key']
    resolved=app._live_configuration()
    assert resolved.library_review==state.live_library_review
    assert stored.library_review['source_key']=='old-history-selection'
    expected='derived-config' if new_archive else 'old-config'
    assert asyncio.run(app._save_live_configuration(resolved))==expected
    assert len(saves)==int(new_archive)
    if new_archive:assert saves[0].library_review==state.live_library_review
    assert asyncio.run(app._save_live_configuration(resolved))==expected
    assert len(saves)==int(new_archive)
    assert 'source_key' not in resolved.portable_contract()['library_review']


def test_reactivation_never_substitutes_for_actual_archive_hash_check(tmp_path):
    preview=inspect_library_review(archive(tmp_path));permission=activation_for_review(preview)
    with pytest.raises(ValueError,match='Review the selected'):
        require_library_activation(preview,{**permission,'archive_sha256':'f'*64})
    with pytest.raises(ValueError,match='Review the selected'):
        require_library_activation(preview,{**permission,'policy_set_sha256':'f'*64})


def test_explicit_builtin_selection_removes_inapplicable_saved_permission(tmp_path):
    permission=activation_for_review(inspect_library_review(archive(tmp_path)))
    stored=AnalysisConfig(sample_elements=['C','Li'],instrument_mode='tof',library_review={**permission,'source_key':'old archive'})
    app=RadarPdNovaApp();state=app.server.state;saves=[]
    app.service=SimpleNamespace(load_configuration_dataset=lambda _:stored.model_copy(deep=True),
        save_configuration=lambda config:(saves.append(config) or SimpleNamespace(status=RunStatus.OK,outputs={'config_output':'derived'})))
    app._schedule_utility=lambda work,*_:work.close()
    app._register_utility=lambda *_:None;app._remember_history_dataset=lambda *_:None
    state.live_configuration_source='history';state.powgen_configuration_dataset_id='stored'
    state.live_database_source='builtin'
    resolved=app._live_configuration()
    assert resolved.library_review=={} and stored.library_review
    assert asyncio.run(app._save_live_configuration(resolved))=='derived'
    assert saves[0].library_review=={}


def test_library_archive_hashing_is_chunked_not_path_read_bytes(tmp_path,monkeypatch):
    path=archive(tmp_path)
    def forbidden(_):raise AssertionError('Whole-file read_bytes is not allowed for library archives')
    monkeypatch.setattr(Path,'read_bytes',forbidden)
    assert inspect_library_review(path)['requires_review']


def test_review_cache_does_not_share_nested_lists_or_dicts(tmp_path):
    path=archive(tmp_path);service=object.__new__(GalaxyService)
    service._library_review_lock=threading.RLock();service._library_review_cache={}
    service._dataset_metadata=lambda _:dict(state='ok',update_time='fixed',file_size=path.stat().st_size)
    downloads=[]
    def download(_,target):downloads.append(target);shutil.copyfile(path,target)
    service._download_dataset=download
    first=service.inspect_library_review('library')
    first['reviews'][0]['components']['C']=999
    first['manifest']['occupancy_reviews'].clear()
    second=service.inspect_library_review('library')
    assert second['reviews'][0]['components']['C']==.6
    assert second['manifest']['occupancy_reviews']
    assert len(downloads)==1


@pytest.mark.parametrize('failed',[False,True])
def test_legacy_augmented_archive_retains_declared_counts_and_failures(tmp_path,failed):
    path=tmp_path/'legacy.zip'
    manifest={'kind':'augmented','n_base_phases':100,'n_added_phases':1,'phase_ids':['added'],
              'failures':[{'id':'missing','error':'unreadable CIF'}] if failed else []}
    with zipfile.ZipFile(path,'w') as output:output.writestr('pack/manifest.json',json.dumps(manifest))
    preview=inspect_library_review(path)
    assert preview['phase_count'] is None
    assert preview['declared_augmented_counts']=={'base':100,'added':1,'total':101}
    assert preview['requires_review'] is failed
    assert preview['failures']==manifest['failures']


def test_legacy_augmented_added_ids_are_still_validated(tmp_path):
    path=tmp_path/'bad-legacy.zip'
    with zipfile.ZipFile(path,'w') as output:
        output.writestr('pack/manifest.json',json.dumps({'kind':'augmented','n_base_phases':100,'n_added_phases':2,'phase_ids':['only-one']}))
    with pytest.raises(ValueError,match='phase IDs'):inspect_library_review(path)
