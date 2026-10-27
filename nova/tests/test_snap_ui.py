"""Live form/immutable scientific-scope contracts for the SNAP adapter."""
import asyncio
from dataclasses import replace

from radar_pd_nova.live_beamlines import BEAMLINES
from radar_pd_nova.snap import parse_filename
from test_live_setup import live, TrialController


def test_snap_live_setup_keeps_export_variant_and_profile_mapping(live,tmp_path,monkeypatch):
    state=live.server.state
    state.live_beamline='SNAP';live._live_beamline_changed()
    path=tmp_path/'reviewed.instprm';path.write_text('# reviewed test profile\n')
    state.powgen_ipts='38203';state.powgen_processing_scope='all'
    state.powgen_source_subfolder='shared/SNAPRed/export/gsa/bank';live._powgen_source_changed()
    state.live_sample_elements='Si';state.nomad_physical_bank=2;state.nomad_profile_bank=7
    state.snap_product_family=parse_filename('SNAP73470_bank_autoreduce.gsa').family
    state.nomad_profile_acknowledged=True;state.nomad_allow_exploratory=True
    state.powgen_instrument_path=str(path)
    captured=[]
    def preflight(*args,**kwargs):
        captured.append(kwargs)
        return {'ready':True,'source_directory':'/SNS/SNAP/IPTS-38203/shared/SNAPRed/export/gsa/bank',
                'scan_count':1,'available_banks':[2],'message':'Reviewed SNAP mapping'}
    monkeypatch.setitem(BEAMLINES,'SNAP',replace(BEAMLINES['SNAP'],preflight=preflight,controller=TrialController))
    state.flush()
    async def start():
        await live.start_powgen_monitoring();await asyncio.sleep(0)
    asyncio.run(start())
    assert live._powgen_controller is not None,state.error_message
    settings=live._powgen_controller.settings
    assert settings.instrument=='SNAP' and settings.selected_banks==(2,) and settings.profile_bank==7
    assert settings.product_family==state.snap_product_family and settings.profile_acknowledged
    assert captured[0]['product_family']==state.snap_product_family
    assert live.service.saved[0].instrument_mode=='tof'
    assert 'SNAP grouping / export variant' in live.layout.html


def test_snap_export_variant_change_invalidates_sample_selection_and_preflight(live):
    state=live.server.state;state.live_beamline='SNAP';live._live_beamline_changed()
    state.powgen_processing_sample_ids=['old'];state.powgen_preflight_ready=True
    state.snap_product_family='["all","reRed"]';live._snap_product_changed()
    assert not state.powgen_processing_sample_ids and not state.powgen_preflight_ready
    assert 'Refresh sample IDs' in state.powgen_sample_message
