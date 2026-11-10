"""Planning lifecycle controls: stale physics and changed files cannot be exported."""
import asyncio
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import zipfile

import numpy as np

from radar_pd_nova.planning_controller import PlanningMixin


class State(SimpleNamespace):
    def change(self,*keys):
        return lambda callback: callback

    def flush(self):
        pass


def app():
    instance=PlanningMixin()
    instance.server=SimpleNamespace(state=State())
    instance._init_planning()
    return instance,instance.server.state


def test_only_the_current_calculation_can_be_cancelled():
    instance,s=app();s.plan_busy=True
    first=instance._planning_begin_calculation()
    second=instance._planning_begin_calculation()
    instance._planning_finish_job(first)
    assert s.plan_busy and s.plan_can_cancel and instance._planning_cancel is second
    generation=instance._planning_generation
    instance.cancel_planning_simulation()
    assert second.is_set() and not s.plan_can_cancel and not first.is_set()
    assert instance._planning_generation==generation+1
    instance._planning_finish_job(second)
    assert not s.plan_busy and not s.plan_can_cancel


def test_history_storage_cannot_offer_a_calculation_cancel():
    instance,s=app();pending=[]
    instance._refinement_schedule=lambda coroutine,name:pending.append(coroutine)
    s.plan_can_cancel=True
    instance._planning_storage_action(lambda:None,'Saving the plan…')
    assert s.plan_busy and not s.plan_can_cancel
    generation=instance._planning_generation
    instance.cancel_planning_simulation()
    assert s.plan_save_message=='Saving the plan…'
    assert instance._planning_generation==generation
    asyncio.run(pending.pop())
    assert not s.plan_busy and not s.plan_can_cancel


def test_editing_invalidates_saved_notice_but_preserves_saved_plan_identity():
    instance,s=app()
    s.plan_saved_id='history:existing-plan'
    s.plan_save_message='Saved to Galaxy History.'
    s.plan_current=True;s.plan_result_ready=True
    instance._planning_edited()
    assert not s.plan_save_message and s.plan_saved_id=='history:existing-plan'
    assert not s.plan_current and not s.plan_result_ready
    s.plan_save_message='Saved to Galaxy History.'
    instance._planning_name_edited()
    assert not s.plan_save_message and s.plan_saved_id=='history:existing-plan'


def result(instance,root):
    root.mkdir()
    np.savez(root/'curves.npz',d=np.arange(3.)+1,expected=np.ones(3),nuclear=np.ones(3),magnetic=np.zeros(3))
    receipt={'looks':[],'intensity_unit':'unit scale','artifacts':{
        'curves.npz':hashlib.sha256((root/'curves.npz').read_bytes()).hexdigest()}}
    (root/'result.json').write_text(json.dumps(receipt))
    instance._planning_receipt_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest()
    instance._planning_output=root
    instance.server.state.plan_current=True
    instance.server.state.plan_result_ready=True


def test_planning_has_no_ipts_dependency_and_legacy_profiles_are_explicit():
    instance,s=app()
    instance.open_planning_workspace()
    assert s.workspace_view=='planning'
    assert not any('ipts' in key for key in vars(s))
    assert sum(row['props']['disabled'] for row in s.plan_configurations)==14
    assert s.plan_response_mode=='current_cycle' and s.plan_resolution_model=='cycle_table'
    assert (s.plan_d_min,s.plan_d_max)==(.55,7.24)
    assert all('Current count response' in row['title'] for row in s.plan_configurations[:3])


def test_configuration_changes_select_working_count_ranges_and_custom_preview():
    instance,s=app()
    s.plan_configuration='2026B_HighRes_60HzB1_CWL0p8.instprm'
    instance._planning_context_edited()
    assert (s.plan_d_min,s.plan_d_max)==(.3,3.1)
    s.plan_d_min=.5;s.plan_d_max=2.5
    # Editing chemistry does not discard the user's requested range.
    s.plan_isotopes='{"Si":"28"}';instance._planning_context_edited()
    assert (s.plan_d_min,s.plan_d_max)==(.5,2.5)
    s.plan_custom_profile='/uploaded/custom.instprm';instance._planning_context_edited()
    assert s.plan_response_mode=='profile_only' and s.plan_resolution_model=='profile'
    s.plan_custom_profile='';s.plan_configuration='2026B_HighRes_60HzB3_CWL2p665.instprm'
    instance._planning_context_edited()
    assert s.plan_response_mode=='current_cycle' and (s.plan_d_min,s.plan_d_max)==(1.1,7.24)


def test_numerical_profile_clears_incompatible_standard_normalization():
    instance,s=app()
    s.plan_intensity_calibration='silicon_63584';s.plan_resolution_model='cycle_table'
    instance._planning_resolution_edited()
    assert s.plan_intensity_calibration=='vanadium'


def test_packaged_silicon_example_clears_previous_scenario_and_is_runnable():
    instance,s=app()
    s.plan_mcif='old.mcif';s.plan_phase_1_cif='old.cif';s.plan_phase_1_percent=20.
    s.plan_current=True;s.plan_result_ready=True
    instance.load_planning_example()
    request=instance._planning_request()
    assert Path(request['cif']).is_file() and request['sample']['mass_g']==.95
    assert request['sample']['beam_width_cm']==1. and not request['additional_phases']
    assert not request.get('mcif') and request['intensity_calibration']=='vanadium'
    assert request['seconds']==[60.,300.,1800.] and request['response_mode']=='current_cycle'
    assert not s.plan_current and not s.plan_result_ready


def test_charge_request_does_not_require_beam_power_and_current_time_is_explicit(tmp_path):
    instance,s=app();cif=tmp_path/'sample.cif';cif.write_text('data_test\n')
    s.plan_cif=str(cif);s.plan_response_mode='current_cycle';s.plan_exposure_basis='charge'
    s.plan_intensity_calibration='silicon_63584'
    s.plan_resolution_model='profile'
    s.plan_charge='25, 125, 750';s.plan_power='';s.plan_energy=''
    request=instance._planning_request()
    assert request['charge_uah']==[25.,125.,750.] and 'seconds' not in request
    assert request['beam']=={}
    assert request['intensity_calibration']=='silicon_63584'
    s.plan_exposure_basis='current_time';s.plan_current_ua=1500.
    request=instance._planning_request()
    assert request['seconds']==[60.,300.,1800.]
    assert request['beam']=={'average_current_ua':1500.}
    s.plan_exposure_basis='power_time';s.plan_power=1.95
    assert instance._planning_request()['beam']=={'power_mw':1.95,'proton_energy_gev':1.3}


def test_current_plot_switches_counts_and_normalized_units_without_resimulation(tmp_path):
    instance,s=app();root=tmp_path/'current';root.mkdir();(root/'reduced').mkdir()
    np.savez(root/'curves.npz',d=[1.,2.,3.],expected=[2.,4.,6.],nuclear=[2.,4.,6.],magnetic=[0.,0.,0.],
             raw_rate=[5.,10.,15.],raw_nuclear_rate=[4.,8.,12.],raw_magnetic_rate=[0.,0.,0.])
    np.savez(root/'reduced/look-0001.npz',y=[2.1,4.2,6.3],sigma=[.1,.1,.1],
             counting_variance=[.004,.006,.008],expected_counts=[125.,250.,375.],
             counts=[126,251,376],raw_predictive_variance=[150.,300.,450.])
    receipt={'mode':'current_cycle','looks':[{'cumulative_charge_uah':25.,'sufficient_statistics':'look-0001.npz'}],
             'intensity_unit':'ratio','artifacts':{}}
    for name in ['curves.npz','reduced/look-0001.npz']:
        receipt['artifacts'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    (root/'result.json').write_text(json.dumps(receipt))
    instance._planning_receipt_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest()
    instance._planning_output=root;s.plan_current=True;s.plan_result_ready=True
    instance._planning_plot()
    assert s.plan_figure['data'][0]['y']==[2.1,4.2,6.3]
    np.testing.assert_allclose(s.plan_figure['data'][0]['error_y']['array'],.1)
    s.plan_error_model='counting';instance._planning_plot()
    np.testing.assert_allclose(s.plan_figure['data'][0]['error_y']['array'],np.sqrt([.004,.006,.008]))
    assert s.plan_figure['layout']['yaxis']['title']['text']=='ratio'
    s.plan_pattern_view='counts';instance._planning_plot()
    assert s.plan_figure['data'][0]['y']==[126,251,376]
    assert s.plan_figure['data'][1]['y']==[125.,250.,375.]
    assert s.plan_figure['data'][0]['name']=='Synthetic · 25 μAh'
    assert 'counts' in s.plan_figure['layout']['yaxis']['title']['text']
    np.testing.assert_allclose(s.plan_figure['data'][0]['error_y']['array'],np.sqrt([125,250,375]))
    s.plan_error_model='predictive';instance._planning_plot()
    np.testing.assert_allclose(s.plan_figure['data'][0]['error_y']['array'],np.sqrt([150,300,450]))
    s.plan_plot_mode='uncertainty';instance._planning_plot()
    assert len(s.plan_figure['data'])==2
    np.testing.assert_allclose(s.plan_figure['data'][0]['y'],np.sqrt([125,250,375]))
    np.testing.assert_allclose(s.plan_figure['data'][1]['y'],np.sqrt([150,300,450]))


def test_site_readouts_follow_native_linked_parameters_and_clear_on_subgroup_change():
    instance,s=app()
    instance._planning_prepared={'controls':{'parameters':[
        {'name':'::constr0','symbol':'q1','label':'Linked moment q1 (μB)','description':'Controls Er1 Mx and My'}],
        'sites':[{'label':'Er1','parent_label':'Er1','ion':'Er+3','position':[0,0,0],
            'metric':[[1,-.5,0],[-.5,1,0],[0,0,1]],'offset':[0,0,0],'basis':[[1],[1],[0]]}]}}
    instance._planning_parameter_rows('magnetic',{'::constr0':2.})
    assert s.plan_moment_sites[0]['magnitude']=='2'  # unit-axis metric, not Cartesian sqrt(8)
    assert [c['value'] for c in s.plan_moment_sites[0]['components']]==['2','2','0']
    instance.update_planning_magnetic_parameter('::constr0','3')
    assert s.plan_parameter_rows[0]['label']=='Linked moment q1 (μB)'
    assert s.plan_moment_sites[0]['magnitude']=='3'
    instance._planning_magnetic_edited()
    assert not s.plan_moment_sites and not s.plan_parameter_rows


def test_magnetic_ion_is_inherited_or_unambiguous_but_never_guessed():
    instance,s=app()
    s.plan_form_factor_options={'Er1':['Er+3'],'Fe1':['Fe+2','Fe+3']}
    assert instance._planning_form_factor('Er1','Er',{})=='Er+3'
    assert instance._planning_form_factor('Fe1','Fe+2',{})=='Fe+2'
    assert instance._planning_form_factor('Fe1','Fe',{})==''
    assert instance._planning_form_factor('Fe1','Fe',{'Fe1':'Fe+3'})=='Fe+3'


def test_changing_band_or_custom_profile_clears_band_specific_intensity_reference():
    instance,s=app()
    s.plan_intensity_calibration='silicon_63584'
    instance._planning_context_edited()
    assert s.plan_intensity_calibration=='silicon_63584'
    s.plan_configuration='2026B_HighRes_60HzB1_CWL0p8.instprm'
    instance._planning_context_edited()
    assert s.plan_intensity_calibration=='vanadium'
    s.plan_configuration='2026B_HighRes_60HzB2_CWL1p5.instprm'
    s.plan_intensity_calibration='silicon_63584';s.plan_custom_profile='/some/custom.instprm'
    instance._planning_context_edited()
    assert s.plan_intensity_calibration=='vanadium'


def test_edit_cancels_in_flight_work_and_clears_plots_and_download():
    instance,s=app()
    event=instance._planning_cancel=threading.Event()
    s.plan_current=True;s.plan_result_ready=True;s.plan_figure={'old':'data'}
    generation=instance._planning_generation
    instance._planning_edited()
    assert event.is_set() and instance._planning_generation>generation
    assert not s.plan_current and not s.plan_result_ready and s.plan_figure=={'data':[],'layout':{}}


def test_coordinate_controls_change_only_discovered_values_and_reject_invalid_edits():
    instance,s=app();instance._planning_parameter_rows('magnetic',{'1::AMx:0':1.})
    s.plan_current=True
    instance.update_planning_magnetic_parameter('1::AMx:0','2.5')
    assert json.loads(s.plan_parameters)=={'1::AMx:0':2.5}
    assert s.plan_parameter_rows[0]['value']==2.5 and not s.plan_current
    for name,value in [('unknown','1'),('1::AMx:0','nan'),('1::AMx:0','31')]:
        instance.update_planning_magnetic_parameter(name,value)
        assert s.plan_error and json.loads(s.plan_parameters)=={'1::AMx:0':2.5}
    import pytest
    with pytest.raises(ValueError,match='invalid transition coordinates'):instance._planning_request()


def test_isotope_picker_preserves_exact_native_labels():
    instance,s=app();s.plan_isotope_options=[{'element':'Y','options':['Nat. Abund.','89 '],'selected':'Nat. Abund.'}]
    instance.update_planning_isotope('Y','89 ')
    assert json.loads(s.plan_isotopes)=={'Y':'89 '}
    assert s.plan_isotope_options[0]['selected']=='89 '
    instance.update_planning_isotope('Y','89')
    assert s.plan_error and json.loads(s.plan_isotopes)=={'Y':'89 '}


def test_isotope_atom_percentages_preserve_invalid_drafts_and_block_simulation():
    import pytest
    instance,s=app()
    instance._planning_isotope_rows({'isotope_options':{'H':['Nat. Abund.','1','2']},
                                    'isotope_selection':{'H':'Nat. Abund.'}})
    instance.update_planning_isotope('H','Atom fractions')
    assert instance._planning_isotopes()=={'H':{'Nat. Abund.':1.}}
    instance.update_planning_isotope_fraction('H','Nat. Abund.','0')
    instance.update_planning_isotope_fraction('H','1','25')
    assert s.plan_error and not s.plan_current
    # A subgroup edit clears coordinate errors, but must not make a bad isotope
    # draft fall back silently to the last valid abundance dictionary.
    instance._planning_k_edited()
    with pytest.raises(ValueError,match='sum to one'):instance._planning_isotopes()
    instance.update_planning_isotope_fraction('H','2','75')
    assert not s.plan_error and instance._planning_isotopes()=={'H':{'1':.25,'2':.75}}
    assert json.loads(s.plan_isotopes)=={'H':{'1':.25,'2':.75}}
    instance.update_planning_isotope_fraction('H','2','nan')
    with pytest.raises(ValueError):instance._planning_isotopes()
    instance.update_planning_isotope('H','2')
    assert instance._planning_isotopes()=={'H':'2'} and not s.plan_error


def test_isotope_mixture_receipt_restores_percent_controls_without_changing_abundances():
    instance,s=app()
    choice={'1':1/3,'2':2/3}
    instance._planning_isotope_rows({'isotope_options':{'H':['Nat. Abund.','1','2']},
                                    'isotope_selection':{'H':choice}})
    assert s.plan_isotope_options[0]['selected']=='Atom fractions'
    assert instance._planning_isotopes()=={'H':choice}
    instance.update_planning_isotope('H','Atom fractions')
    assert instance._planning_isotopes()=={'H':choice}


def test_parent_edit_discards_subgroup_and_mode_bindings():
    instance,s=app()
    for name in ('_planning_parent_output','_planning_catalog','_planning_prepared',
                 '_planning_magnetic_request','_planning_structural_catalog'):
        setattr(instance,name,object())
    s.plan_candidate='old';s.plan_structural_candidate='old'
    instance._planning_context_edited()
    assert instance._planning_parent_output is None and instance._planning_catalog is None
    assert instance._planning_prepared is None and instance._planning_structural_catalog is None
    assert not s.plan_candidate and not s.plan_structural_candidate


def structural_result(instance, root):
    result(instance,root)
    parent=root/'parent.gpx';parent.write_bytes(b'opaque native structural child')
    receipt=json.loads((root/'result.json').read_text())
    receipt.update(parent_model_artifact='parent.gpx',
        structural_state={'schema':'powgen-planning-structural-state/v1',
            'gpx':'old-preparation.gpx','gpx_sha256':'old-preparation-hash',
            'nuclear_cell_multiplier':3.,'modes':{'mode-a':.025},'cell':[6.,6.,11.,90.,90.,120.]})
    receipt['artifacts']['parent.gpx']=hashlib.sha256(parent.read_bytes()).hexdigest()
    (root/'result.json').write_text(json.dumps(receipt))
    instance._planning_receipt_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest()
    instance.server.state.plan_can_promote_parent=True
    return receipt


def test_structural_parent_promotion_keeps_exact_native_model_and_resets_k_basis(tmp_path):
    instance,s=app();s.plan_cif='original-parent.cif';s.plan_k='1/3, 1/3, 0'
    structural_result(instance,tmp_path/'structural')
    instance._planning_prepared=object();instance._planning_catalog=object()
    instance.promote_planning_structural_parent()
    assert not s.plan_error and s.plan_promoted_parent and s.plan_k=='0, 0, 0'
    assert s.plan_cif=='original-parent.cif' and not s.plan_current and not s.plan_can_promote_parent
    assert instance._planning_prepared is None and instance._planning_catalog is None
    receipt=instance._planning_parent_receipt()
    request=instance._planning_parent_request(receipt)
    assert request['source_gpx']==str(tmp_path/'structural/parent.gpx')
    reference=request['nuclear_reference']
    assert reference['source_gpx_sha256']==request['source_gpx_sha256']
    assert reference['nuclear_cell_multiplier']==3. and reference['structural_state']['modes']=={'mode-a':.025}
    assert 'gpx' not in reference['structural_state']
    # A new magnetic k invalidates subgroup bindings while retaining the chosen child.
    instance._planning_k_edited()
    assert instance._planning_parent_receipt()==receipt and instance._planning_parent_state is not None
    instance.reset_planning_nuclear_parent()
    assert instance._planning_parent_state is None and instance._planning_parent_output is None
    assert not s.plan_promoted_parent and not s.plan_parent_note


def test_structural_parent_promotion_rejects_stale_or_modified_native_model(tmp_path):
    instance,s=app();structural_result(instance,tmp_path/'structural')
    s.plan_current=False;instance.promote_planning_structural_parent()
    assert 'current' in s.plan_error and instance._planning_parent_state is None
    s.plan_current=True
    (tmp_path/'structural/parent.gpx').write_bytes(b'changed')
    instance.promote_planning_structural_parent()
    assert 'changed' in s.plan_error and instance._planning_parent_state is None


def test_parent_context_edit_discards_promoted_structure_and_its_basis_note(tmp_path):
    instance,s=app();structural_result(instance,tmp_path/'structural')
    instance.promote_planning_structural_parent();assert s.plan_promoted_parent
    s.plan_configuration='different-profile'
    instance._planning_context_edited()
    assert instance._planning_parent_state is None and instance._planning_parent_output is None
    assert not s.plan_promoted_parent and not s.plan_parent_note and not s.plan_can_promote_parent


def test_export_captures_sealed_artifacts_and_rejects_changed_bytes(tmp_path):
    instance,s=app();instance.service=SimpleNamespace(output_root=tmp_path)
    instance.load_planning_example();result(instance,instance._planning_root()/'campaign')
    downloads=[]
    instance.download_file=lambda *args:downloads.append(args)
    instance.export_planning_simulation()
    assert len(downloads)==1
    with zipfile.ZipFile(io.BytesIO(downloads[0][2])) as archive:
        assert {'plan.json','results/output/curves.npz','results/output/result.json'}<=set(archive.namelist())
        assert 'inputs/plan_cif/silicon.cif' in archive.namelist()
    (instance._planning_output/'curves.npz').write_bytes(b'changed')
    instance.export_planning_simulation()
    assert len(downloads)==1 and 'changed' in s.plan_error
    assert not s.plan_result_ready and not s.plan_current


def test_plot_rejects_changed_receipt_even_with_unchanged_curves(tmp_path):
    instance,s=app();result(instance,tmp_path/'campaign')
    root=instance._planning_output
    external=tmp_path/'outside.npz';external.write_bytes((root/'curves.npz').read_bytes())
    receipt=json.loads((root/'result.json').read_text())
    receipt['artifacts']['curves.npz']='wrong-hash'
    (root/'result.json').write_text(json.dumps(receipt))
    instance._planning_plot()
    assert not s.plan_current and not s.plan_result_ready and 'changed' in s.plan_error


def test_plot_keeps_scientific_axis_units_and_qualification(tmp_path):
    instance,s=app();result(instance,tmp_path/'campaign')
    s.plan_qualification='Unqualified synthetic preview'
    instance._planning_plot()
    layout=s.plan_figure['layout']
    assert layout['xaxis']['title']['text']=='d spacing (Å)'
    assert layout['yaxis']['title']['text']=='unit scale'
    assert layout['title']['text']=='Unqualified synthetic preview'


def test_parent_model_cannot_change_after_preview(tmp_path):
    import pytest
    instance,s=app();root=tmp_path/'parent';root.mkdir();(root/'model').mkdir()
    model=root/'model/forward.gpx';model.write_bytes(b'original model')
    receipt={'artifacts':{'model/forward.gpx':hashlib.sha256(model.read_bytes()).hexdigest()}}
    (root/'result.json').write_text(json.dumps(receipt))
    instance._planning_parent_output=root
    instance._planning_parent_receipt_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest()
    assert instance._planning_parent_receipt()==receipt
    model.write_bytes(b'changed model')
    with pytest.raises(ValueError,match='parent model changed'):instance._planning_parent_receipt()


def test_mixture_symmetry_uses_sealed_primary_model_and_not_the_multiphase_project(tmp_path):
    instance,s=app();root=tmp_path/'parent';root.mkdir();(root/'model').mkdir()
    primary=root/'model/reference-parent.gpx';primary.write_bytes(b'primary phase only')
    combined=root/'model/forward.gpx';combined.write_bytes(b'primary plus impurity')
    receipt={'parent_model_artifact':'model/reference-parent.gpx','artifacts':{
        str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (primary,combined)}}
    (root/'result.json').write_text(json.dumps(receipt))
    instance._planning_parent_output=root
    instance._planning_parent_receipt_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest()
    assert instance._planning_parent_receipt()==receipt
    primary.write_bytes(b'changed primary phase')
    import pytest
    with pytest.raises(ValueError,match='parent model changed'):instance._planning_parent_receipt()


def test_mixture_picker_converts_mass_percent_and_rejects_settings_without_a_cif(tmp_path):
    instance,s=app();first=tmp_path/'first.cif';first.write_text('data_primary\n')
    second=tmp_path/'second.cif';second.write_text('data_secondary\n')
    s.plan_cif=str(first);s.plan_phase_1_cif=str(second);s.plan_phase_1_percent='12.5'
    s.plan_phase_1_size='2';s.plan_phase_1_microstrain='50'
    request=instance._planning_request()
    assert request['additional_phases']==[{'cif':str(second),'mass_fraction':.125,'crystallite_size_um':2.,'microstrain':50.}]
    s.plan_phase_1_cif=''
    import pytest
    with pytest.raises(ValueError,match='Upload the CIF'):instance._planning_request()


def test_stale_native_completion_cannot_restore_results(tmp_path,monkeypatch):
    instance,s=app()
    instance.service=SimpleNamespace(output_root=tmp_path)
    instance._planning_request=lambda:{'cif':'unused'}
    calls=[]
    def execute(request,directory,event):
        calls.append(request)
        output=Path(directory)/'simulation';output.mkdir()
        (output/'result.json').write_text('{}')
        instance._planning_edited()
        return {}
    monkeypatch.setattr(instance,'_planning_execute',execute)
    asyncio.run(instance._run_planning_simulation())
    assert calls==[{'cif':'unused'}]
    assert not s.plan_current and not s.plan_result_ready and not s.plan_busy
    assert instance._planning_output is None and instance._planning_cancel is None
