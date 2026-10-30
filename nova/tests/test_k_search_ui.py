import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from gpx_recipe.symmetry.compiler import file_sha
from radar_pd_nova.symmetry_controller import SymmetryMixin
from radar_pd_nova.k_search_plot import evidence


def app_and_result(tmp_path):
    class State(SimpleNamespace):
        def flush(self):pass
    state=State(sx_start_source='original',rf_experiment='IPTS-1',sx_kind='magnetic',sx_parent='parent',sx_histogram='bank',
        sx_sites=['Mn1'],sx_form_factors='Mn1=Mn',sx_lande=2.,sx_directions=['x'],sx_mag_d_min=0.,sx_max_moment=6.,sx_sign_starts=True,
        sx_k_budget=60,sx_k_limit=6,sx_k_selected='0,0,1/2',sx_k_axis='q',sx_k_reset=0,sx_k='0,0,0',sx_approved=True,sx_plan_ready=True,sx_error='')
    app=SymmetryMixin();app.server=SimpleNamespace(state=state);app._symmetry_approved='old-approval'
    app._symmetry_draft=dict(root=tmp_path,source_id='original',context=dict(sha256='b'*64,phases=[dict(name='parent',setting_sha256='a'*64)]),protocol={'old':True})
    candidate=dict(id='mag:one',magnetic_space_group="Pn'ma'",irrep=None)
    catalog=dict(parent_setting_sha256='a'*64,k=['0','0','1/2'],candidates=[candidate])
    x=np.linspace(10,100,100);q=np.linspace(1,4,100);y=np.exp(-((q-2)/.1)**2)
    baseline=tmp_path/'baseline.npz';profile=tmp_path/'profile.npz'
    mask=np.ones(100,dtype=bool);mask[40:45]=False
    np.savez(baseline,x=x,q=q,observed=10+y,nuclear=np.full(100,10.),mask=mask)
    np.savez(profile,intensity=y)
    result=dict(source_sha256='b'*64,parent='parent',histogram='bank',coordinate='tof',baseline=str(baseline),baseline_sha256=file_sha(baseline),
        families=[dict(k=['0','0','1/2'],best_template='one',simulated=1,planned=1,available_starts=6,discovered=1,status='sampled',compatible_candidate_ids=['mag:one'])],
        templates={'one':dict(path=str(profile),sha256=file_sha(profile))},ranking={'templates':[dict(id='one',amplitude=1.)]},catalogs={'0,0,1/2':catalog})
    app._k_result=result;app._k_result_identity=app._k_identity();app._k_cancel=None
    return app,result


def test_selected_k_reuses_catalog_but_requires_a_new_refinement_plan(tmp_path):
    app,result=app_and_result(tmp_path);app.use_candidate_k();s=app.server.state
    assert s.sx_k=='0, 0, 1/2' and s.sx_discovered
    assert not s.sx_approved and not s.sx_plan_ready and app._symmetry_approved==''
    assert 'protocol' not in app._symmetry_draft
    assert app._symmetry_draft['catalog']==result['catalogs']['0,0,1/2']
    assert app._symmetry_draft['discovery_identity']==('magnetic','parent','0, 0, 1/2','original')
    assert 'sampled only a subset' in s.sx_discovery_scope


def test_k_handoff_defaults_to_translation_and_site_compatible_embeddings(tmp_path):
    app,result=app_and_result(tmp_path)
    catalog=result['catalogs']['0,0,1/2'];catalog['candidates'].append(dict(id='mag:other',magnetic_space_group='P1',irrep=None))
    result['families'][0]['compatible_candidate_ids']=['mag:one']
    app.use_candidate_k()
    assert app.server.state.sx_candidates==['mag:one']
    assert len(app.server.state.sx_candidate_options)==2  # full discovery remains inspectable


def test_canted_search_does_not_drop_models_that_only_lack_a_collinear_survey_start(tmp_path):
    app,result=app_and_result(tmp_path);app.server.state.sx_moment_model='symmetry_allowed'
    result['catalogs']['0,0,1/2']['candidates'].append(dict(id='mag:canted',magnetic_space_group='P1',irrep=None))
    result['families'][0]['compatible_candidate_ids']=['mag:one']
    app.use_candidate_k()
    assert app.server.state.sx_candidates==['mag:one','mag:canted']
    assert 'All catalog embeddings remain selected' in app.server.state.sx_discovery_scope
    assert not app.server.state.sx_approved


def test_unfinished_compatibility_inventory_cannot_be_claimed_verified(tmp_path):
    app,result=app_and_result(tmp_path);result['families'][0].pop('compatible_candidate_ids')
    app.use_candidate_k()
    assert 'inventory is unfinished' in app.server.state.sx_error
    assert app.server.state.sx_k=='0,0,0' and app._symmetry_approved=='old-approval'


@pytest.mark.parametrize('handler',['_symmetry_parent_changed','_symmetry_start_changed'])
def test_new_parent_or_checkpoint_does_not_keep_another_atoms_ion_mapping(tmp_path,handler):
    app,_=app_and_result(tmp_path)
    app._symmetry_draft['context']['phases'][0]['atoms']=[]
    getattr(app,handler)()
    state=app.server.state
    assert state.sx_sites==[] and state.sx_form_factors=='' and state.sx_lande==''


@pytest.mark.parametrize('field,value',[('sx_start_source','another'),('sx_parent','another'),('sx_histogram','another'),('rf_experiment','another'),('sx_form_factors','Mn1=Fe')])
def test_changed_inputs_cannot_apply_a_stale_k_result(tmp_path,field,value):
    app,_=app_and_result(tmp_path);setattr(app.server.state,field,value);app.use_candidate_k()
    assert 'changed' in app.server.state.sx_error
    assert app.server.state.sx_k=='0,0,0' and app._symmetry_approved=='old-approval'


def test_missing_or_foreign_catalog_does_not_become_an_executable_plan(tmp_path):
    app,result=app_and_result(tmp_path);result['catalogs'].clear();app.use_candidate_k()
    assert 'no complete subgroup catalog' in app.server.state.sx_error
    app,result=app_and_result(tmp_path);result['catalogs']['0,0,1/2']['parent_setting_sha256']='c'*64;app.use_candidate_k()
    assert 'another parent' in app.server.state.sx_error


@pytest.mark.parametrize('axis,label',[('q','Q (Å⁻¹)'),('d','d (Å)'),('native','Time of flight (µs)')])
def test_native_calibrated_evidence_axes_and_masked_gaps(tmp_path,axis,label):
    _,result=app_and_result(tmp_path);plot=evidence(result,'0,0,1/2',axis=axis)
    assert plot['layout']['xaxis2']['title']['text']==label
    assert any(t['name']=='Added magnetic template' for t in plot['data'])
    assert all(t['connectgaps'] is False for t in plot['data'])
    assert all(None in t['y'] for t in plot['data'])


def test_tampered_profiles_cannot_be_displayed_as_valid_evidence(tmp_path):
    _,result=app_and_result(tmp_path);Path(result['templates']['one']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):evidence(result,'0,0,1/2')


def test_inputs_changed_clear_results_and_cancel_inflight_work(tmp_path):
    import threading
    app,_=app_and_result(tmp_path);app._k_cancel=threading.Event();app.server.state.sx_k_busy=True
    app._k_inputs_changed()
    assert app._k_cancel.is_set() and app._k_result is None
    assert not app.server.state.sx_k_has_result and not app.server.state.sx_k_rows


def test_actual_app_exposes_survey_inside_existing_model_view():
    from radar_pd_nova.app import RadarPdNovaApp
    app=RadarPdNovaApp();html=app.layout.html
    assert 'Find candidate k-vectors' in html and 'Use selected k' in html
    assert 'Candidate propagation vectors' in html and 'No Rietveld refinement runs in this survey' in html
    assert 'radar-user-guide' in html
    assert app.server.state.sx_k_budget==60 and app.server.state.sx_k_limit==6
    assert app.server.state.sx_k=='' and app.server.state.sx_lande=='' and app.server.state.sx_max_moment==''
    assert app.server.state.sx_refine_scale is False
    assert app.server.state.sx_refine_background is False
    assert 'Refine shared nuclear/magnetic intensity scale' in html
    assert 'Refine existing background coefficients' in html


def test_incomplete_survey_explains_global_and_per_family_failures():
    from radar_pd_nova.k_search_controller import coverage_notes
    result=dict(errors=['Inventory exceeded the time budget'],families=[
        dict(k=['0','0','1/2'],reason='Catalog unavailable'),
        dict(k=['0','0','0'],simulation_failures=[dict(reason='Unsupported native representation')])])
    notes=coverage_notes(result)
    assert notes==['Inventory exceeded the time budget','(0, 0, 1/2): Catalog unavailable',
        '(0, 0, 0): 1 native starts failed; first reason: Unsupported native representation']
def test_failed_simulations_are_visible_as_failures_instead_of_a_time_budget_suggestion():
    from radar_pd_nova.k_search_controller import survey_status_message
    failed=dict(status='incomplete',families=[dict(simulated=0,simulation_failures=[{'reason':'unsupported ADP'}]*12)])
    assert '12 native simulations failed' in survey_status_message(failed)
    assert 'No usable magnetic profiles' in survey_status_message(failed)
    assert 'time budget' not in survey_status_message(failed)
    failed['families'][0]['simulated']=2
    assert 'Partial survey' in survey_status_message(failed)
