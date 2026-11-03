"""Explicit watch restoration loads only its captured display profile."""
import asyncio
from dataclasses import asdict,replace
from hashlib import sha256
import json
from pathlib import Path
import threading

import numpy as np
import pytest

from radar_pd_nova.live_beamlines import beamline_adapter
from radar_pd_nova.live_restore import read_saved_watch
from radar_pd_nova.powgen_watch import WatchedRun
from radar_pd_nova.transition_coordinate_mapping import mapped_calibration
from radar_pd_nova.diffraction_coordinates import coordinate_values
from test_live_restore import SavedHistory,app as live_app
from test_nomad import profile as nomad_profile,source as nomad_source
from test_snap import profile as snap_profile,write_source
from radar_pd_nova import nomad,snap,hb2a


class ProfileHistory(SavedHistory):
    def __init__(self,text):
        super().__init__();self.text=text;self.profile_threads=[]

    def load_text_document(self,identifier,*,max_bytes=None):
        if identifier in self.payloads:return super().load_text_document(identifier,max_bytes=max_bytes)
        assert identifier=='saved-profile'
        self.reads.append(('profile',identifier));self.profile_threads.append(threading.get_ident())
        if isinstance(self.text,Exception):raise self.text
        return self.text


def captured_watch(tmp_path,monkeypatch,instrument,*,text=None):
    adapter=beamline_adapter(instrument)
    if instrument=='NOM':
        text=text if text is not None else nomad_profile(2,2837.58510543,31.).replace('difA:0','difA:4.2').replace('Zero:0','Zero:-3')
        local=nomad_source.__wrapped__(tmp_path)
        setup=nomad.inspect_reduced_setup(local,selected_bank=2)
        extra=dict(selected_banks=(2,),profile_bank=2,profile_acknowledged=True,allow_exploratory=True)
        receipt_key='nomad_bank_provenance'
        provenance=dict(source_bank=2,profile_bank=2,profile_sha256=sha256(text.encode()).hexdigest(),
            profile=dict(path='/old/viewer/staged-bank1.instprm',profile_bank=1,sha256='0'*64)) if isinstance(text,str) else {}
        metadata={receipt_key:provenance}
    elif instrument=='SNAP':
        text=text if text is not None else snap_profile(2)
        local=write_source(tmp_path/'SNAP000042_bank_test.gsa');setup=snap.inspect_reduced_setup(local)
        extra=dict(selected_banks=(2,),profile_bank=2,profile_acknowledged=True,allow_exploratory=True,
            product_family=json.dumps([Path(adapter.default_folder).name,'2f2708c073568e2a']))
        receipt_key='snap_bank_provenance'
        provenance=dict(source_bank=2,profile_bank=2,profile_sha256=sha256(text.encode()).hexdigest(),
            profile=dict(path='/old/viewer/staged-bank1.instprm',profile_bank=1,sha256='0'*64)) if isinstance(text,str) else {}
        metadata={receipt_key:provenance}
    else:
        text=text if text is not None else '# selected profile\nType:PNC\nBank:1\nLam:2.4063\nZero:-0.01\nU:800\nV:-440\nW:240\nX:0\nY:0\n'
        setup=hb2a.Setup(42,101,'time','a'*64,(1,2),10,20.,100.,0,{'nominal_wavelength_angstrom':2.4063})
        extra=dict(profile_acknowledged=True)
        metadata={'hb2a_setup':asdict(setup),'hb2a_pattern_provenance':dict(source_bank=1,
            profile_sha256=sha256(text.encode()).hexdigest())} if isinstance(text,str) else {'hb2a_setup':asdict(setup)}
    service=ProfileHistory(text)
    settings=adapter.settings(ipts='IPTS-1234',history_id=service.history_id,
        configuration_dataset_id='saved-config',instrument_profile_dataset_id='saved-profile',**extra)
    controller=adapter.controller(service,settings)
    run=WatchedRun(101,controller.source_directory+'/'+instrument+'_101.gsa','captured-fingerprint',
        instrument=instrument,bank=1 if instrument=='HB2A' else 2,
        experiment_number=42 if instrument=='HB2A' else None,galaxy_job_id='saved-job',
        scientific_summary={'rwp':8.,'phases':[]},scan_metadata=metadata)
    controller.state.completed[run.run_id]=run
    payload=json.loads(controller.capture_checkpoint().payload_json)
    service.payloads['selected-checkpoint']=payload
    # These tests use a synthetic, already captured acquisition setup. No
    # facility source is read merely by relabeling a restored profile's axis.
    monkeypatch.setattr(type(controller),'_setup',lambda self,run:setup)
    monkeypatch.setattr(type(controller),'discover',lambda *args,**kwargs:pytest.fail('No facility scan'))
    monkeypatch.setattr(type(controller),'persist_state',lambda *args,**kwargs:pytest.fail('Read-only restore'))
    return service,settings,run,setup,payload


@pytest.mark.parametrize('instrument',['NOM','SNAP','HB2A'])
def test_selected_checkpoint_loads_saved_profile_off_callback_thread_and_maps_axes(tmp_path,monkeypatch,instrument):
    service,settings,run,setup,payload=captured_watch(tmp_path,monkeypatch,instrument)
    main_thread=threading.get_ident()
    review=asyncio.run(asyncio.to_thread(read_saved_watch,service,'selected-checkpoint'))
    restored=review.controller
    assert asdict(restored.settings)==asdict(settings)
    assert not review.blocked and not review.missing
    assert restored._supplied_profile.is_file() and not restored.display_profile_error
    assert service.profile_threads and all(thread!=main_thread for thread in service.profile_threads)
    assert service.reads==[('metadata','selected-checkpoint'),('checkpoint','selected-checkpoint'),('profile','saved-profile')]
    calibration=mapped_calibration({'run_id':run.run_id,'bank':run.bank,'instrument':instrument},restored,{})
    d=np.array([1.5,2.,3.,4.])
    if instrument=='HB2A':
        assert calibration.wavelength==2.4063 and calibration.zero==-0.01
        native=calibration.two_theta(d);unit='two_theta_deg'
    else:
        assert calibration.source_bank==2 and calibration.profile_bank==2
        expected=(2837.58510543,4.2,0.,-3.) if instrument=='NOM' else (6200.,2.,.4,1.)
        assert (calibration.difc,calibration.difa,calibration.difb,calibration.zero)==expected
        native=expected[0]*d+expected[1]*d*d+expected[2]/d+expected[3];unit='tof_us'
    np.testing.assert_allclose(coordinate_values(native,unit,'d_angstrom',calibration),d,rtol=1e-10)
    np.testing.assert_allclose(coordinate_values(native,unit,'q_inv_angstrom',calibration),2*np.pi/d,rtol=1e-10)
    # Display callbacks use the already loaded profile, never a new transfer.
    count=len(service.profile_threads)
    mapped_calibration({'run_id':run.run_id,'bank':run.bank,'instrument':instrument},restored,{})
    assert len(service.profile_threads)==count


@pytest.mark.parametrize('instrument',['NOM','SNAP','HB2A'])
@pytest.mark.parametrize('failure',['missing','invalid'])
def test_unavailable_profile_keeps_readonly_state_and_reports_display_error(tmp_path,monkeypatch,instrument,failure):
    text=OSError('saved profile missing') if failure=='missing' else 'not a neutron profile'
    service,settings,run,setup,payload=captured_watch(tmp_path,monkeypatch,instrument,text=text)
    review=read_saved_watch(service,'selected-checkpoint');restored=review.controller
    assert len(restored.state.completed)==1 and asdict(restored.settings)==asdict(settings)
    assert restored._supplied_profile is None and 'profile unavailable' in restored.display_profile_error
    assert not review.blocked and not review.missing
    with pytest.raises(ValueError,match='profile unavailable'):
        mapped_calibration({'run_id':run.run_id,'bank':run.bank,'instrument':instrument},restored,{})


@pytest.mark.parametrize('instrument',['NOM','SNAP','HB2A'])
def test_changed_profile_bytes_never_enable_the_old_receipt_mapping(tmp_path,monkeypatch,instrument):
    service,settings,run,setup,payload=captured_watch(tmp_path,monkeypatch,instrument)
    service.text=service.text.replace('Zero:', '# changed\nZero:')
    review=read_saved_watch(service,'selected-checkpoint')
    with pytest.raises(ValueError,match='analysis receipt|differs|checksum'):
        mapped_calibration({'run_id':run.run_id,'bank':run.bank,'instrument':instrument},review.controller,{})


@pytest.mark.parametrize('instrument',['NOM','SNAP','HB2A'])
def test_latest_restore_uses_the_same_hook_and_replaces_stale_cached_profile(tmp_path,monkeypatch,instrument):
    service,settings,run,setup,payload=captured_watch(tmp_path,monkeypatch,instrument)
    controller=beamline_adapter(instrument).controller(service,settings)
    wrong=tmp_path/'another-draft.instprm';wrong.write_text('wrong old cached input')
    controller._supplied_profile=wrong
    monkeypatch.setattr(controller,'_checkpoint_candidates',lambda:[{'id':'selected-checkpoint'}])
    monkeypatch.setattr(controller,'_hydrate_missing_metadata',lambda:None)
    service.load_json_document=lambda key:payload
    assert asyncio.run(asyncio.to_thread(controller.restore_latest_state))
    assert controller._supplied_profile!=wrong
    assert controller._supplied_profile.read_text()==service.text
    assert not controller.display_profile_error
    assert [r for r in service.reads if r[0]=='profile']==[('profile','saved-profile')]


def test_explicit_restore_uses_saved_nomad_profile_while_editable_draft_is_pg3(tmp_path,monkeypatch,live_app):
    service,settings,run,setup,payload=captured_watch(tmp_path,monkeypatch,'NOM')
    live_app.service=service
    state=live_app.server.state
    state.live_beamline='PG3';state.powgen_instrument_dataset_id='wrong-draft-profile'
    state.nomad_profile_bank=1;state.flush()
    async def review():
        await live_app.list_saved_live_watches()
        state.live_restore_dataset_id='selected-checkpoint';state.flush()
        await live_app.inspect_saved_live_watch()
    asyncio.run(review())
    assert state.live_restore_ready
    controller=live_app._live_restore_review[1].controller
    assert controller.settings.instrument=='NOM' and controller.settings.profile_bank==2
    assert controller.settings.instrument_profile_dataset_id=='saved-profile'
    assert state.live_beamline=='PG3' and state.powgen_instrument_dataset_id=='wrong-draft-profile'
    calibration=mapped_calibration({'run_id':run.run_id,'bank':2,'instrument':'NOM'},controller,{})
    assert calibration.profile_bank==2 and calibration.difc==2837.58510543


@pytest.mark.parametrize('instrument',['NOM','SNAP'])
def test_original_profile_bank_disagreement_stays_blocked(tmp_path,monkeypatch,instrument):
    service,settings,run,setup,payload=captured_watch(tmp_path,monkeypatch,instrument)
    key='nomad_bank_provenance' if instrument=='NOM' else 'snap_bank_provenance'
    metadata=dict(run.scan_metadata)
    metadata[key]=dict(metadata[key],profile_bank=1)
    review=read_saved_watch(service,'selected-checkpoint')
    restored=review.controller
    # Use an immutable replacement to avoid depending on checkpoint layout.
    restored.state.completed[run.run_id]=replace(restored.state.completed[run.run_id],scan_metadata=metadata)
    with pytest.raises(ValueError,match='Bank label'):
        mapped_calibration({'run_id':run.run_id,'bank':2,'instrument':instrument},restored,{})


def test_cw_missing_wavelength_is_a_display_error_not_failed_restoration(tmp_path,monkeypatch):
    service,settings,run,setup,payload=captured_watch(tmp_path,monkeypatch,'HB2A',
        text='Type:PNC\nBank:1\nZero:0\n')
    restored=read_saved_watch(service,'selected-checkpoint').controller
    assert len(restored.state.completed)==1 and restored._supplied_profile is None
    assert 'profile unavailable' in restored.display_profile_error
