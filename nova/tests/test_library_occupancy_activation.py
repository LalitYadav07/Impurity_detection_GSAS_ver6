"""Host review is explicit, hash-bound permission; it is not a crystal oracle."""
import asyncio
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import zipfile

import pytest

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.configuration import config_from_contract
from radar_pd_nova.galaxy_service import GalaxyService
from radar_pd_nova.library_setup import (
    LibraryDraft, activation_for_review, inspect_library_review,
    library_source_key, require_library_activation,
)
from radar_pd_nova.models import AnalysisConfig, InputSelection, SubmissionSnapshot
from radar_pd_nova.setup_forms import configuration_from_form


def archive(tmp_path, *, empty_sites=False):
    raw=b'data_synthetic\n# synthetic host declaration; runtime must validate\n'
    review={
        'schema':'radar-cif-occupancy-review/v1','reviewed':True,
        'source_sha256':hashlib.sha256(raw).hexdigest(),
        'canonical_sha256':hashlib.sha256(raw).hexdigest(),
        'maximum_site_excess':.001,'normalization_applied':False,
        'review_note':'Synthetic reviewed input',
        'composition':{'C':.6,'Li':.40066},
        'affected_sites':[] if empty_sites else [{
            'labels':['C1','Li1'],'fractional_coordinates':[0,0,0],
            'total':1.00066,'components':{'C':.6,'Li':.40066}}],
    }
    review['policy_sha256']=hashlib.sha256(json.dumps(review,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    path=tmp_path/'library.zip'
    with zipfile.ZipFile(path,'w') as output:
        output.writestr('pack/manifest.json',json.dumps({'n_phases':1,'phase_ids':['test'], 'occupancy_reviews':{'test':review},'failures':[]}))
        output.writestr('pack/cif_map.json',json.dumps({'test':'cifs/test.cif'}))
        output.writestr('pack/cifs/test.cif',raw)
    return path


def test_imported_review_flag_is_not_permission_and_display_is_unverified(tmp_path):
    preview=inspect_library_review(archive(tmp_path))
    assert preview['requires_review']
    assert preview['actual_affected_site_count'] is None
    assert preview['evidence_verification']=='unverified archive declaration'
    with pytest.raises(ValueError,match='Review the selected'):require_library_activation(preview,{})
    activation=activation_for_review(preview)
    require_library_activation(preview,activation)
    changed={**activation,'archive_sha256':'f'*64}
    with pytest.raises(ValueError):require_library_activation(preview,changed)


def test_empty_affected_declaration_does_not_masquerade_as_zero(tmp_path):
    with pytest.raises(ValueError,match='empty declaration'):inspect_library_review(archive(tmp_path,empty_sites=True))


def test_exact_activation_survives_config_contract_and_library_change_invalidates_form(tmp_path):
    app=RadarPdNovaApp();state=app.server.state
    state.sample_elements='C, Li';state.database_source='archive';state.library_archive_source='galaxy';state.history_database_id='library'
    state.flush()
    permission={**activation_for_review(inspect_library_review(archive(tmp_path))), 'source_key':library_source_key(LibraryDraft(state))}
    state.library_review=permission
    config=configuration_from_form(state)
    assert config_from_contract(config.portable_contract()).library_review==permission
    state.history_database_id='replacement'
    with pytest.raises(ValueError,match='changed'):configuration_from_form(state)
    app._invalidate_library_review()
    assert state.library_review=={}


def test_slow_review_cannot_authorize_replacement_library(tmp_path):
    app=RadarPdNovaApp();state=app.server.state
    state.database_source='archive';state.library_archive_source='galaxy';state.history_database_id='old'
    state.flush()
    started=threading.Event();release=threading.Event();tasks=[]
    preview=inspect_library_review(archive(tmp_path))
    def inspect(_):
        started.set();assert release.wait(3);return preview
    app.service=SimpleNamespace(inspect_library_review=inspect)
    async def exercise():
        app._schedule_utility=lambda work,*_:tasks.append(asyncio.create_task(work))
        app.review_candidate_library()
        assert await asyncio.to_thread(started.wait,2)
        state.history_database_id='new';app._invalidate_library_review()
        release.set();await tasks[0]
    asyncio.run(exercise())
    app.approve_candidate_library_review()
    assert state.library_review=={}
    assert state.library_review_preview=={}
    assert not state.library_review_loading


def test_galaxy_wire_parameters_include_same_explicit_permission(tmp_path):
    path=archive(tmp_path);permission=activation_for_review(inspect_library_review(path))
    config=AnalysisConfig(sample_elements=['C','Li'],library_review=permission)
    inputs=InputSelection(source='galaxy',data_dataset_id='data',instrument_dataset_id='profile',database_archive_path=str(path))
    snapshot=SubmissionSnapshot(config=config,inputs=inputs,idempotency_token='test-permission')
    service=object.__new__(GalaxyService)
    service.tool_has_input=lambda *_:True
    prepared={'configuration':'config','database_archive':'library','data':'data','instrument':'profile'}
    params,_=service._prepared_parameters(None,snapshot,prepared)
    assert json.loads(params.inputs['library|database|library_review_activation'])==permission
    with pytest.raises(ValueError,match='Review the selected'):
        service._prepared_parameters(None,snapshot.model_copy(update={'config':AnalysisConfig(sample_elements=['C','Li'])}),prepared)
    service.tool_has_input=lambda *_:False
    with pytest.raises(ValueError,match='does not support'):
        service._prepared_parameters(None,snapshot,prepared)


@pytest.mark.parametrize('edit',[{'acknowledged':False},{'archive_sha256':'not-a-hash'}, {'allow_incomplete':'yes'}, {'extra':True}])
def test_config_rejects_malformed_permission(edit):
    permission={'acknowledged':True,'archive_sha256':'a'*64,'policy_set_sha256':'b'*64}
    with pytest.raises(ValueError):AnalysisConfig(sample_elements=['C'],library_review={**permission,**edit})
