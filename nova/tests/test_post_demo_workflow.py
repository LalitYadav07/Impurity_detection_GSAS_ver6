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
    assert '1 / 2' in text and 'Lowest checked Rwp, needs review' in text and '27.000%' in text
    assert 'passing fit checks' not in text and rows==before
    rows[0]['status']='valid'
    assert 'Lowest Rwp passing fit checks' in model_progress(rows)
    assert 'First checked fit pending' in model_progress(rows[2:])


def test_magnetic_progress_shows_small_improvement_over_its_nuclear_baseline():
    row=dict(protocol='warm',candidate='a',name='P21',kind='magnetic',execution_state='terminal',
             native_verified=True,has_view=True,rwp=6.585531,status='valid',baseline_rwp=6.616386)
    text=model_progress([row])
    assert 'nuclear baseline 6.616% (change -0.031 percentage points)' in text
    assert 'magnetic order' not in text and 'Best qualified' not in text
    for baseline in (None,float('nan'),float('inf'),True,-1):
        assert 'nuclear baseline' not in model_progress([dict(row,baseline_rwp=baseline)])


def test_progress_only_compares_the_selected_leader_to_its_own_baseline():
    leader=dict(protocol='cold',candidate='a',name='A',kind='magnetic',execution_state='terminal',
                native_verified=True,has_view=True,rwp=12.,status='valid',baseline_rwp=33.)
    other=dict(leader,candidate='b',name='B',rwp=11.,status='optimization_unresolved',baseline_rwp=20.)
    text=model_progress([leader,other])
    assert 'A · Rwp 12.000%' in text and 'nuclear baseline 33.000%' in text
    assert 'change -21.000' in text and 'baseline 20.000' not in text


def test_failed_model_progress_does_not_look_like_a_waiting_first_fit():
    rows=[dict(protocol='one',candidate='a',name='A',execution_state='terminal',
               native_verified=False,has_view=False,status='technical_failure')]
    text=model_progress(rows)
    assert '1 / 1' in text and 'No checked fit returned' in text
    assert 'First checked fit pending' not in text
