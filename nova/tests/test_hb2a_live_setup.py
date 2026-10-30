import asyncio
from dataclasses import replace
import json

from test_live_setup import live
from test_hb2a import archive,profile,add,_he3_spice
from radar_pd_nova.hb2a_controller import Hb2aSampleIndex
from radar_pd_nova.scan_interpretation_ui import scan_label


def test_live_cw_draft_profile_check_and_saved_recipe_are_independent(live,archive,profile):
    add(archive)
    state=live.server.state
    state.live_beamline='HB2A';state.flush()
    state.powgen_ipts='1';state.flush()
    state.live_sample_elements='Mn, Ge, N'
    state.powgen_processing_scope='all'
    state.powgen_instrument_source='computer'
    state.powgen_instrument_path=str(profile)
    state.hb2a_profile_acknowledged=True
    state.hb2a_temperature_channel='sample_temp'
    state.hb2a_temperature_unit='K'
    state.hb2a_condition_reviewed=True
    state.flush()
    live._powgen_sample_index=lambda _:Hb2aSampleIndex('IPTS-1')
    assert live._live_configuration().instrument_mode=='cw'
    assert state.instrument_mode=='auto'  # The Single-pattern draft is independent.
    assert asyncio.run(live.check_powgen_experiment())
    prepared=asyncio.run(live._prepare_live_settings())
    settings,config=prepared
    assert config.instrument_mode=='cw'
    assert settings.instrument=='HB2A' and settings.selected_banks==(1,)
    assert json.loads(settings.condition_bindings_json)['temperature']['channel']=='sample_temp'
    assert settings.definition().facility=='HFIR'


def test_changing_ipts_clears_all_reviewed_bindings_and_profile_ack(live):
    state=live.server.state;state.live_beamline='HB2A';state.flush()
    live._live_beamline_changed()
    state.powgen_ipts='1';live._powgen_ipts_changed();state.flush()
    state.hb2a_profile_acknowledged=True
    state.hb2a_temperature_channel='sample_temp';state.hb2a_condition_reviewed=True
    state.hb2a_magnetic_field_channel='active_field';state.hb2a_magnetic_field_reviewed=True
    state.hb2a_pressure_channel='hi_pressure';state.hb2a_pressure_reviewed=True
    state.hb2a_collimator_override='OUT';state.hb2a_allow_stopped=True
    state.flush();state.powgen_ipts='2';live._powgen_ipts_changed();state.flush()
    assert not state.hb2a_profile_acknowledged and not state.hb2a_collimator_override
    assert not state.hb2a_temperature_channel and not state.hb2a_magnetic_field_channel and not state.hb2a_pressure_channel
    assert not state.hb2a_allow_stopped


def test_he3_discovery_suggests_reviewed_pair_without_overwriting_manual_sensor(live,archive):
    from radar_pd_nova.hb2a_temperature import HE3_AUTOMATIC_CHANNEL
    _,raw=add(archive);raw.write_text(_he3_spice(.28,1.2005))
    state=live.server.state;state.live_beamline='HB2A';state.flush()
    live._live_beamline_changed();state.powgen_ipts='1';live._powgen_ipts_changed();state.flush()
    live._powgen_sample_index=lambda _:Hb2aSampleIndex('IPTS-1')
    asyncio.run(live.refresh_powgen_samples('1'))
    assert state.hb2a_temperature_channel==HE3_AUTOMATIC_CHANNEL
    assert state.hb2a_temperature_unit=='K' and not state.hb2a_condition_reviewed
    state.hb2a_temperature_channel='he3potl_temp';state.hb2a_condition_reviewed=True;state.flush()
    asyncio.run(live.refresh_powgen_samples('1'))
    assert state.hb2a_temperature_channel=='he3potl_temp' and state.hb2a_condition_reviewed


def test_normalization_change_invalidates_inflight_sample_scope(live):
    state=live.server.state;state.live_beamline='HB2A';state.flush()
    live._live_beamline_changed()
    state.powgen_discovered_sample_options=[{'value':'old','title':'old'}]
    state.powgen_processing_sample_ids=['old']
    token=live._powgen_sample_request
    state.hb2a_normalization='monitor';live._hb2a_normalization_changed();state.flush()
    assert live._powgen_sample_request is not token
    assert not state.powgen_processing_sample_ids and not state.powgen_discovered_sample_options


def test_hb2a_acquisition_labels_do_not_collapse_equal_scan_numbers():
    assert scan_label({'instrument':'HB2A','run_number':1,'metadata':{'experiment_number':1000}})=='Exp 1000 · Scan 1'
    assert scan_label({'instrument':'HB2A','run_number':1,'metadata':{'experiment_number':1001}})=='Exp 1001 · Scan 1'


def test_switching_four_beamlines_keeps_facility_and_geometry_distinct(live):
    state=live.server.state
    labels={row['value']:row['title'] for row in state.live_beamline_options}
    assert labels['HB2A']=='HB-2A (preview)' and labels['SNAP']=='SNAP (preview)'
    assert labels['PG3']=='POWGEN' and labels['NOM']=='NOMAD'
    for instrument,facility,mode in [('HB2A','HFIR','cw'),('SNAP','SNS','tof'),
                                     ('NOM','SNS','tof'),('PG3','SNS','tof'),('HB2A','HFIR','cw')]:
        state.live_beamline=instrument;live._live_beamline_changed();state.flush()
        state.powgen_ipts='1';live._powgen_ipts_changed();state.flush()
        live._powgen_source_changed();state.flush()
        assert state.powgen_source_directory.startswith(f'/{facility}/{instrument}/IPTS-1/')
        assert live._live_configuration().instrument_mode==mode
        if instrument=='HB2A':
            assert live._powgen_processing_bank_ids()==(1,)
            assert not state.powgen_bank_options
            assert state.powgen_dashboard_wavelength=='Calibrated CW profile'
        elif instrument in {'NOM','SNAP'}:
            assert state.powgen_dashboard_wavelength=='Broadband neutron TOF'
