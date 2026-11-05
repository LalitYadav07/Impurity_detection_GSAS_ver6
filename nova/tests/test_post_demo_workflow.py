from copy import deepcopy
import pytest
from radar_pd_nova.symmetry_workspace import model_progress
from radar_pd_nova.models import AnalysisConfig
from radar_pd_nova.configuration import config_from_contract


def test_profile_setup_survives_portable_roundtrip_and_legacy_defaults():
    c=AnalysisConfig(sample_elements=['Fe','O'],tof_profile_mode='per_scan',tof_profile_terms=['X','Z'],instrument_mode='tof')
    assert config_from_contract(c.portable_contract()).tof_profile_terms==['X','Z']
    assert config_from_contract({'chemistry':{'sample_elements':['Fe','O']}}).tof_profile_mode=='fixed'
    with pytest.raises(ValueError):AnalysisConfig(sample_elements=['Fe'],tof_profile_mode='per_scan',tof_profile_terms=[])


def test_clearing_profile_selection_does_not_silently_release_x():
    from radar_pd_nova.app import RadarPdNovaApp
    from radar_pd_nova.setup_forms import configuration_from_form
    app=RadarPdNovaApp();s=app.server.state
    s.sample_elements=['Fe','O'];s.main_cif_source='upload';s.main_cif_path='main.cif'
    s.tof_profile_mode='per_scan';s.tof_profile_terms=[]
    with pytest.raises(ValueError,match='selected terms'):configuration_from_form(s)


def test_progress_counts_models_and_never_calls_an_inadequate_leader_qualified():
    rows=[dict(protocol='one',candidate='a',name='A',execution_state='terminal',native_verified=True,
               has_view=True,rwp=27.,status='inadequate_fit'),
          dict(protocol='one',candidate='a',name='A',execution_state='terminal',has_view=False,status='profile_equivalent'),
          dict(protocol='one',candidate='b',name='B',execution_state='prepared',has_view=False,status='pending')]
    before=deepcopy(rows)
    text=model_progress(rows)
    assert '1 / 2' in text and 'Best audited, needs review' in text and '27.000%' in text
    assert 'Best qualified' not in text and rows==before
    rows[0]['status']='valid'
    assert 'Best qualified' in model_progress(rows)
    assert 'First checked fit pending' in model_progress(rows[2:])


def test_failed_model_progress_does_not_look_like_a_waiting_first_fit():
    rows=[dict(protocol='one',candidate='a',name='A',execution_state='terminal',
               native_verified=False,has_view=False,status='technical_failure')]
    text=model_progress(rows)
    assert '1 / 1' in text and 'No checked fit returned' in text
    assert 'First checked fit pending' not in text
