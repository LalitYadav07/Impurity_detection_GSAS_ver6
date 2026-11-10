"""Exercise deferred upload callbacks against a connected Trame state."""
import copy
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

from trame_server.state import State

from radar_pd_nova.uploads import store_browser_upload
from test_planning_handoff import App


def reactive_app(tmp_path):
    app=App();app.test_root=tmp_path
    commits=[]
    state=State(commit_fn=lambda delta:commits.append(copy.deepcopy(delta)),ready=True)
    app.server=SimpleNamespace(state=state)
    app.service=SimpleNamespace(output_root=tmp_path,history_id='reactive-history')
    app._init_planning();app._init_preparation();app._init_planning_handoff()
    state.update(dict(workflow_mode='single',workspace_view='planning',setup_collapsed=True,
        radiation='neutron',main_cif_path='',main_cif_source='none',instrument_path='',instrument_source='upload',
        fit_start=None,fit_end=None,bank='',busy=False,selected_run_uid='',data_path='',input_source='upload'))
    app._prep_parent_upload=SimpleNamespace(current_path=None)
    app._prep_profile_upload=SimpleNamespace(current_path=None)
    state.flush()
    app.load_planning_example();state.flush()
    return app,state,commits


def handoff(tmp_path):
    app,state,commits=reactive_app(tmp_path)
    app.use_planning_inputs_for_preparation();state.flush()
    assert state.plan_handoff_can_apply, state.plan_handoff_error
    expected=app._planning_handoff_mapping(app._planning_pending_handoff)
    app.confirm_planning_handoff();state.flush()
    assert not state.plan_handoff_error
    return app,state,commits,expected


def test_reviewed_tof_range_and_native_bank_grid_stay_equal_after_reactive_flush(tmp_path):
    app,state,commits,expected=handoff(tmp_path)
    bank=next(iter(app._prep_banks.values()))['grid']
    assert state.prep_grid_kind==bank['kind']=='PNT'
    assert [float(state.prep_lower),float(state.prep_upper)]==expected['tof_range_microseconds']
    assert [bank['lower'],bank['upper']]==expected['tof_range_microseconds']
    visible={}
    for delta in commits:visible.update(delta)
    assert [float(visible['prep_lower']),float(visible['prep_upper'])]==expected['tof_range_microseconds']
    # The guard lasts until the reviewed upload changes, not just one flush.
    app._preparation_profile_changed();state.flush()
    assert [float(state.prep_lower),float(state.prep_upper)]==[bank['lower'],bank['upper']]


def test_new_upload_with_identical_bytes_still_uses_ordinary_profile_defaults(tmp_path):
    app,state,_,_=handoff(tmp_path)
    old=app._prep_profile_upload.current_path
    new=store_browser_upload(old.read_bytes(),'selected-again.instprm')
    app._prep_profile_upload.current_path=new;state.prep_profile_path=str(new);state.flush()
    assert (state.prep_grid_kind,state.prep_lower,state.prep_upper)==('PNT','3000','180000')
    assert app._prep_profile_handoff is None


def test_replacement_bytes_in_same_path_invalidate_guard_and_use_actual_cw_type(tmp_path):
    app,state,_,_=handoff(tmp_path)
    path=app._prep_profile_upload.current_path
    path.write_text('Type: PNC\nBank: 1\nLam: 1.5406\nZero: 0\n')
    app._preparation_profile_changed();state.flush()
    assert (state.prep_grid_kind,state.prep_lower,state.prep_upper)==('PNC','10','160')
    assert app._prep_profile_handoff is None


def test_changed_checksum_with_unchanged_native_type_also_invalidates_guard(tmp_path):
    app,state,_,_=handoff(tmp_path)
    path=app._prep_profile_upload.current_path
    with path.open('a') as stream:stream.write('\n# changed profile bytes\n')
    app._preparation_profile_changed();state.flush()
    assert (state.prep_grid_kind,state.prep_lower,state.prep_upper)==('PNT','3000','180000')
    assert app._prep_profile_handoff is None


def test_legacy_native_tof_profile_preserves_its_reviewed_range_on_flush(tmp_path):
    app,state,_=reactive_app(tmp_path)
    original=files('radar_pd_nova').joinpath('data/powgen_calibrations/PGHR_60-2016A.prm')
    state.plan_custom_profile=str(store_browser_upload(original.read_bytes(),original.name))
    state.flush()
    app.use_planning_inputs_for_preparation();state.flush()
    state.plan_handoff_source_bank='1';state.flush()
    assert state.plan_handoff_can_apply,state.plan_handoff_error
    expected=app._planning_handoff_mapping(app._planning_pending_handoff)
    app.confirm_planning_handoff();state.flush()
    grid=next(iter(app._prep_banks.values()))['grid']
    assert not state.plan_handoff_error and state.prep_grid_kind==grid['kind']=='PNT'
    assert [float(state.prep_lower),float(state.prep_upper)]==expected['tof_range_microseconds']
    assert [grid['lower'],grid['upper']]==expected['tof_range_microseconds']
