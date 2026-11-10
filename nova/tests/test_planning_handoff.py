"""Planner handoffs are reviewed input drafts, never submitted observations."""
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from radar_pd_nova.planning_controller import PlanningMixin
from radar_pd_nova.planning_handoff import PlanningHandoffMixin,SCHEMA,_snapshot
from radar_pd_nova.preparation_controller import PreparationMixin
from radar_pd_nova.battery_preparation import checked_upload
from radar_pd_nova.uploads import read_browser_upload,store_browser_upload
from radar_pd_nova import planning_bundle


class State(SimpleNamespace):
    def change(self,*keys):return lambda callback:callback
    def flush(self):pass


class App(PlanningHandoffMixin,PlanningMixin,PreparationMixin):
    def _preparation_store(self):
        return SimpleNamespace(root=self.test_root/'preparation',
            registry=SimpleNamespace(summaries=lambda:[],selected_key=lambda:None))
    def download_file(self,name,kind,raw):self.download=(name,kind,raw)


def app(tmp_path,*,live=False):
    instance=App();instance.test_root=tmp_path
    instance.server=SimpleNamespace(state=State())
    instance.service=SimpleNamespace(output_root=tmp_path,history_id='owned-history')
    instance._init_planning();instance._init_preparation();instance._init_planning_handoff()
    instance.load_planning_example();s=instance.server.state
    s.workflow_mode='powgen' if live else 'single';s.workspace_view='planning';s.setup_collapsed=True
    s.radiation='neutron';s.main_cif_path='';s.main_cif_source='none';s.instrument_path='';s.instrument_source='upload'
    s.fit_start=None;s.fit_end=None;s.bank='';s.busy=False
    s.selected_run_uid='original-result';s.data_path='/existing/measured.gsa';s.input_source='upload'
    s.live_beamline='PG3';s.powgen_monitoring=False;s.live_launch_busy=False
    s.live_configuration_source='inline';s.powgen_main_cif_source='none';s.powgen_main_cif_path=''
    s.powgen_instrument_source='automatic';s.powgen_instrument_path='';s.powgen_processing_banks=[]
    s.live_fit_start=None;s.live_fit_end=None;s.powgen_ipts='IPTS-123';s.powgen_processing_sample_ids=['original']
    instance._prep_parent_upload=SimpleNamespace(current_path=None)
    instance._prep_profile_upload=SimpleNamespace(current_path=None)
    return instance,s


def test_preparation_review_has_no_effect_until_confirmed_then_preserves_owned_files_and_banks(tmp_path):
    instance,s=app(tmp_path);original=_snapshot(instance,'preparation')
    instance.use_planning_inputs_for_preparation()
    assert s.plan_handoff_open and s.plan_handoff_can_apply and not s.plan_handoff_error
    assert _snapshot(instance,'preparation')==original
    pending=instance._planning_pending_handoff;expected=instance._planning_handoff_mapping(pending)
    instance.confirm_planning_handoff()
    assert not s.plan_handoff_open and not s.plan_handoff_error
    parent=checked_upload(instance._prep_parent_upload,s.prep_parent_path,maximum=10*1024*1024)
    profile=checked_upload(instance._prep_profile_upload,s.prep_profile_path)
    assert parent.read_bytes()==Path(s.plan_cif).read_bytes()
    assert hashlib.sha256(profile.read_bytes()).hexdigest()==expected['profile_sha256']
    assert read_browser_upload(s.prep_parent_path,max_bytes=10*1024*1024,suffix='.cif')==parent.read_bytes()
    assert s.prep_profile_bank=='1' and expected['source_bank']==2
    bank=instance._prep_banks['POWGEN source bank 2']
    assert bank['grid']['bank']==1 and bank['grid']['bins']==expected['bins']
    assert [bank['grid']['lower'],bank['grid']['upper']]==expected['tof_range_microseconds']
    assert not s.prep_context_ready and s.workspace_view=='preparation'
    assert s.selected_run_uid=='original-result' and s.data_path=='/existing/measured.gsa'


def test_single_analysis_draft_preserves_measurement_and_result_while_owning_exact_inputs(tmp_path):
    instance,s=app(tmp_path);s.main_cif_path='/old/main.cif';s.instrument_path='/old/profile.instprm'
    instance.use_planning_inputs_for_analysis()
    assert s.main_cif_path=='/old/main.cif' and s.instrument_path=='/old/profile.instprm'
    instance.confirm_planning_handoff()
    assert not s.plan_handoff_error and s.main_cif_source=='upload' and s.instrument_source=='upload'
    assert s.instrument_mode=='tof' and s.fit_start<s.fit_end
    assert read_browser_upload(s.main_cif_path,max_bytes=10*1024*1024,suffix='.cif')==Path(s.plan_cif).read_bytes()
    assert (s.data_path,s.input_source,s.selected_run_uid)==('/existing/measured.gsa','upload','original-result')
    assert s.bank=='2' and not s.setup_collapsed
    assert s.workspace_notice==s.notice and s.workspace_notice_kind=='info'
    assert 'real data' in s.workspace_notice


def test_live_draft_uses_computer_sources_and_keeps_single_setup_and_ipts(tmp_path):
    instance,s=app(tmp_path,live=True);s.main_cif_path='/single/main.cif';s.instrument_path='/single/profile.instprm'
    instance.use_planning_inputs_for_analysis();instance.confirm_planning_handoff()
    assert not s.plan_handoff_error and s.powgen_main_cif_source==s.powgen_instrument_source=='computer'
    assert s.live_fit_start<s.live_fit_end and s.powgen_processing_banks==[2]
    assert s.main_cif_path=='/single/main.cif' and s.instrument_path=='/single/profile.instprm'
    assert s.powgen_ipts=='IPTS-123' and s.powgen_processing_sample_ids==['original']
    assert s.selected_run_uid=='original-result'


@pytest.mark.parametrize('change',['planner','target','source','captured'])
def test_stale_review_or_tampered_file_never_replaces_target(tmp_path,change):
    instance,s=app(tmp_path);instance.use_planning_inputs_for_analysis()
    if change=='planner':s.plan_d_min=.6
    elif change=='target':s.fit_start=888.
    elif change=='source':
        uploaded=store_browser_upload(Path(s.plan_cif).read_bytes(),'copy.cif');s.plan_cif=str(uploaded)
    else:instance._planning_pending_handoff['profile_path'].write_text('changed')
    before=_snapshot(instance,'single');instance.confirm_planning_handoff()
    assert s.plan_handoff_error and _snapshot(instance,'single')==before


def test_cancel_keeps_existing_preparation_draft(tmp_path):
    instance,s=app(tmp_path);s.prep_parent_path='/old';old=_snapshot(instance,'preparation')
    instance.use_planning_inputs_for_preparation();instance.cancel_planning_handoff()
    assert not s.plan_handoff_open and _snapshot(instance,'preparation')==old


@pytest.mark.parametrize('target,flag',[('single','busy'),('live','powgen_monitoring'),('live','live_launch_busy'),('preparation','prep_busy')])
def test_active_target_is_protected(tmp_path,target,flag):
    instance,s=app(tmp_path,live=target=='live');setattr(s,flag,True)
    if target=='preparation':instance.use_planning_inputs_for_preparation()
    else:instance.use_planning_inputs_for_analysis()
    before=_snapshot(instance,target);instance.confirm_planning_handoff()
    assert s.plan_handoff_error and not s.plan_handoff_can_apply and _snapshot(instance,target)==before


@pytest.mark.parametrize('choice,allowed',[
    ({'Si':'Nat. Abund.'},True),({'Pr':'141 '},True),({'Pr3+':'141 '},True),({'D':'2'},True),
    ({'Si':'28'},False),({'H':'2'},False),({'Si':{'28':1.}},False),({'Si':{'28':.25,'29':.75}},False)])
def test_isotope_support_distinguishes_native_overrides_from_unchanged_cif_support(tmp_path,choice,allowed):
    instance,s=app(tmp_path)
    import re
    symbol=re.match(r'[A-Z][a-z]?',next(iter(choice)))[0]
    if symbol!='Si':
        raw=Path(s.plan_cif).read_text().replace('Si',symbol).encode()
        s.plan_cif=str(store_browser_upload(raw,'element.cif'))
    s.plan_isotopes=json.dumps(choice)
    instance.use_planning_inputs_for_analysis()
    assert s.plan_handoff_can_apply is allowed
    if not allowed:
        assert 'Isotope' in s.plan_handoff_error
        old=_snapshot(instance,'single');instance.confirm_planning_handoff();assert _snapshot(instance,'single')==old


@pytest.mark.parametrize('kind',['structural','magnetic','comparison','mcif','promoted'])
def test_transition_assumptions_are_retained_as_intent_without_warm_parent_substitution(tmp_path,kind):
    instance,s=app(tmp_path)
    if kind=='promoted':s.plan_promoted_parent=True
    else:s.plan_transition=kind
    s.plan_structural_parameters='{"mode-a":0.125}';s.plan_k='0, 0, 1/2'
    instance.use_planning_inputs_for_preparation()
    assert s.plan_handoff_open and not s.plan_handoff_can_apply
    assert 'native CIF export' in s.plan_handoff_error if kind=='promoted' else 'Explicitly choose' in s.plan_handoff_error
    before=_snapshot(instance,'preparation');instance.confirm_planning_handoff();assert _snapshot(instance,'preparation')==before
    instance.export_planning_handoff()
    with zipfile.ZipFile(io.BytesIO(instance.download[2])) as archive:
        intent=json.loads(archive.read('intent.json'));document,files=planning_bundle.read_archive(archive.read('plan.zip'))
    assert intent['schema']==SCHEMA and intent['requires_native_reinspection']
    assert not intent['measured_observations_included'] and not intent['scientific_result']
    assert document['settings']['plan_k']=='0, 0, 1/2'
    assert document['settings']['plan_structural_parameters']=='{"mode-a":0.125}'


def test_custom_profile_needs_an_explicit_physical_bank_and_preserves_profile_bank(tmp_path):
    instance,s=app(tmp_path)
    # Pick a known modern Bank 1 file; it remains a custom upload.
    profile=Path(next(row['profile'] for row in instance._planning_profiles if row['id']==s.plan_configuration))
    s.plan_custom_profile=str(store_browser_upload(profile.read_bytes(),'custom.instprm'))
    instance.use_planning_inputs_for_preparation()
    assert s.plan_handoff_source_bank_required and not s.plan_handoff_can_apply
    s.plan_handoff_source_bank='7';instance._planning_handoff_bank_changed()
    assert s.plan_handoff_can_apply
    instance.confirm_planning_handoff()
    assert not s.plan_handoff_error and s.prep_profile_bank=='1'
    assert 'POWGEN source bank 7' in instance._prep_banks


def test_arbitrary_local_input_path_is_not_authority(tmp_path):
    instance,s=app(tmp_path)
    path=tmp_path/'unowned.cif';path.write_bytes(Path(s.plan_cif).read_bytes());s.plan_cif=str(path)
    instance.use_planning_inputs_for_analysis()
    assert s.plan_handoff_error and instance._planning_pending_handoff is None
    assert s.main_cif_path==''


def test_synthetic_result_stays_inside_intent_and_is_never_a_measured_input(tmp_path):
    from test_planning_bundle import result
    instance,s=app(tmp_path);result(instance)
    instance.use_planning_inputs_for_analysis();instance.confirm_planning_handoff();instance.export_planning_handoff()
    assert not s.plan_handoff_error and s.data_path=='/existing/measured.gsa'
    with zipfile.ZipFile(io.BytesIO(instance.download[2])) as archive:
        document,files=planning_bundle.read_archive(archive.read('plan.zip'))
    assert any(name.endswith('curves.npz') for name in files)
    assert all('radar-planning' not in str(value) for value in document['settings'].values())


def test_dense_planner_grid_cannot_be_silently_coarsened_for_preparation(tmp_path):
    instance,s=app(tmp_path);s.plan_step=.00008
    instance.use_planning_inputs_for_preparation()
    assert not s.plan_handoff_can_apply and s.prep_parent_path==''


def test_known_mismatched_measured_bank_is_not_given_the_planner_profile(tmp_path):
    instance,s=app(tmp_path);s.bank='bank1'
    instance.use_planning_inputs_for_analysis()
    assert not s.plan_handoff_can_apply and 'source bank differs' in s.plan_handoff_error
    instance.confirm_planning_handoff()
    assert not s.instrument_path and s.bank=='bank1'


def test_cancelled_review_token_cannot_later_apply(tmp_path):
    instance,s=app(tmp_path);instance.use_planning_inputs_for_analysis();instance.cancel_planning_handoff()
    instance.confirm_planning_handoff()
    assert s.main_cif_path==s.instrument_path==''


def test_unknown_isotope_atom_type_requires_native_inspection(tmp_path):
    instance,s=app(tmp_path);s.plan_isotopes='{"Pr":"141 "}'
    instance.use_planning_inputs_for_analysis()
    assert not s.plan_handoff_can_apply and 'absent from the uploaded CIF' in s.plan_handoff_error


@pytest.mark.parametrize('transition',['structural','magnetic','comparison','mcif'])
def test_explicit_original_parent_choice_connects_transition_plans_without_importing_their_response(tmp_path,transition):
    instance,s=app(tmp_path);s.plan_transition=transition;s.plan_k='0, 0, 1/2'
    s.plan_structural_parameters='{"mode-a":0.125}'
    instance.use_planning_inputs_for_preparation()
    assert s.plan_handoff_parent_choice_required and not s.plan_handoff_can_apply
    s.plan_handoff_use_original_parent=True;instance._planning_handoff_bank_changed()
    assert s.plan_handoff_can_apply
    instance.confirm_planning_handoff()
    assert not s.plan_handoff_error and Path(s.prep_parent_path).read_bytes()==Path(s.plan_cif).read_bytes()
    assert s.prep_custom_k=='0, 0, 1/2' and not s.prep_context_ready
    instance.export_planning_handoff()
    with zipfile.ZipFile(io.BytesIO(instance.download[2])) as archive:manifest=json.loads(archive.read('intent.json'))
    assert manifest['structure_scope']=='original_nuclear_parent' and not manifest['transition_response_imported']
    assert manifest['transition_context']['plan_structural_parameters']=='{"mode-a":0.125}'


def test_promoted_child_cannot_be_substituted_by_selecting_original_parent(tmp_path):
    instance,s=app(tmp_path);s.plan_promoted_parent=True
    instance.use_planning_inputs_for_analysis();s.plan_handoff_use_original_parent=True;instance._planning_handoff_bank_changed()
    assert not s.plan_handoff_can_apply and 'qualified native CIF export' in s.plan_handoff_error


def test_native_reinspection_hook_restores_only_matched_pending_k_and_ionic_site_inputs(tmp_path):
    instance,s=app(tmp_path)
    s.plan_cif=str(store_browser_upload(Path(s.plan_cif).read_text().replace('Si','Fe').encode(),'iron.cif'))
    s.plan_transition='magnetic';s.plan_k='0, 0, 1/2';s.plan_sites=['Fe1'];s.plan_form_factors='{"Fe1":"Fe3+"}'
    instance.use_planning_inputs_for_preparation();s.plan_handoff_use_original_parent=True
    instance._planning_handoff_bank_changed();instance.confirm_planning_handoff()
    bank=next(iter(instance._prep_banks.values()))
    context=dict(source_sha256=hashlib.sha256(Path(s.prep_parent_path).read_bytes()).hexdigest(),
        banks=[bank],parent=dict(atoms=[dict(label='Fe1',type='Fe',occupancy=1.)]))
    s.prep_k_options=[];instance._prep_site_inputs={'Fe1':dict(label='Fe1',element='Fe',form_factor='',lande_g='2')}
    assert instance.apply_planning_preparation_intent(context)
    assert s.prep_selected_k==['0,0,1/2'] and s.prep_sites==['Fe1']
    assert instance._prep_site_inputs['Fe1']['form_factor']=='Fe3+'
    assert not s.prep_discovery_ready
    before=copy.deepcopy(s.prep_selected_k)
    assert not instance.apply_planning_preparation_intent(dict(context,source_sha256='f'*64))
    assert s.prep_selected_k==before


def custom_bank_profile(instance,state,*,legacy=False,multibank=True,header=False):
    """Use distinct instrument coefficients so a wrong bank is observable."""
    if legacy:
        raw=('INS   BANK  '+('2' if multibank else '1')+'\n'
             'INS   HTYPE PNT\n'
             'INS  1 ICONS 22000 0 0\n'+
             ('INS  2 ICONS 27000 0 0\n' if multibank else '')).encode()
        name='mapped.prm'
    else:
        profile=Path(next(row['profile'] for row in instance._planning_profiles
            if row['id']==state.plan_configuration))
        lines=profile.read_text().splitlines()
        first='\n'.join(lines[1:])+'\n'
        second='\n'.join('Bank:2.0' if line.startswith('Bank:') else
            'difC:27000.0' if line.startswith('difC:') else line for line in lines[1:])+'\n'
        if multibank:
            raw=('#Bank 1: GSAS-II instrument parameter file\n'+first+
                '#Bank 2: GSAS-II instrument parameter file\n'+second).encode()
        else:
            prefix='#Bank 2: GSAS-II instrument parameter file\n' if header else lines[0]+'\n'
            raw=(prefix+second).encode()
        name='mapped.instprm'
    state.plan_custom_profile=str(store_browser_upload(raw,name))
    state.plan_profile_bank=2 if multibank or not legacy else 1
    return raw


@pytest.mark.parametrize('live',[False,True])
@pytest.mark.parametrize('legacy',[False,True])
def test_analysis_handoff_refuses_unpreserved_instrument_bank_selection(tmp_path,live,legacy):
    instance,s=app(tmp_path,live=live)
    custom_bank_profile(instance,s,legacy=legacy)
    instance.use_planning_inputs_for_analysis()
    s.plan_handoff_source_bank='1';instance._planning_handoff_bank_changed()
    mapping=instance._planning_handoff_mapping(instance._planning_pending_handoff)
    assert mapping['source_bank']==1 and mapping['profile_bank']==2
    before=_snapshot(instance,'live' if live else 'single')
    assert not s.plan_handoff_can_apply and 'standalone selected-bank profile' in s.plan_handoff_error
    instance.confirm_planning_handoff()
    assert _snapshot(instance,'live' if live else 'single')==before


@pytest.mark.parametrize('live',[False,True])
@pytest.mark.parametrize('legacy',[False,True])
def test_analysis_single_bank_profile_keeps_every_byte_with_a_distinct_physical_bank(tmp_path,live,legacy):
    instance,s=app(tmp_path,live=live)
    raw=custom_bank_profile(instance,s,legacy=legacy,multibank=False)
    instance.use_planning_inputs_for_analysis()
    s.plan_handoff_source_bank='7';instance._planning_handoff_bank_changed()
    assert s.plan_handoff_can_apply
    instance.confirm_planning_handoff()
    assert not s.plan_handoff_error
    destination=s.powgen_instrument_path if live else s.instrument_path
    assert Path(destination).read_bytes()==raw
    assert s.powgen_processing_banks==[7] if live else s.bank=='7'


@pytest.mark.parametrize('live',[False,True])
def test_analysis_handoff_refuses_an_explicit_bank_header_that_needs_remapping(tmp_path,live):
    instance,s=app(tmp_path,live=live)
    custom_bank_profile(instance,s,multibank=False,header=True)
    instance.use_planning_inputs_for_analysis()
    s.plan_handoff_source_bank='1';instance._planning_handoff_bank_changed()
    assert not s.plan_handoff_can_apply and 'explicit bank header cannot be remapped' in s.plan_handoff_error


@pytest.mark.parametrize('legacy',[False,True])
def test_preparation_preserves_the_explicit_instrument_bank_for_multibank_profiles(tmp_path,legacy):
    instance,s=app(tmp_path)
    raw=custom_bank_profile(instance,s,legacy=legacy)
    instance.use_planning_inputs_for_preparation()
    s.plan_handoff_source_bank='1';instance._planning_handoff_bank_changed()
    assert s.plan_handoff_can_apply
    instance.confirm_planning_handoff()
    assert not s.plan_handoff_error
    bank=instance._prep_banks['POWGEN source bank 1']
    assert bank['grid']['bank']==2 and Path(bank['instrument']).read_bytes()==raw
