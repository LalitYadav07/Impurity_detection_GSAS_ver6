"""Behavioral checks for the integrated, session-owned planning workflow."""
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from radar_pd_nova.planning_controller import PlanningMixin


class State(SimpleNamespace):
    def change(self,*keys):return lambda callback:callback
    def flush(self):pass


def planner(tmp_path):
    app=PlanningMixin()
    app.server=SimpleNamespace(state=State(workspace_view='monitor',setup_collapsed=False))
    app.service=SimpleNamespace(output_root=tmp_path)
    app._init_planning()
    return app,app.server.state


@pytest.mark.parametrize('view', ['monitor','experiment','refinement','battery','plots'])
def test_navigation_preserves_exact_analysis_context(tmp_path,view):
    app,s=planner(tmp_path)
    s.workspace_view=view;s.setup_collapsed=False
    s.powgen_ipts='35915';s.selected_run_uid='existing-run';s.live_assistant_open=True
    app.select_planning_tab('planning');app.select_planning_tab('planning')
    s.plan_mass=2.75
    app.select_planning_tab('analysis')
    assert (s.workspace_view,s.setup_collapsed)==(view,False)
    assert (s.powgen_ipts,s.selected_run_uid,s.live_assistant_open)==('35915','existing-run',True)
    app.select_planning_tab('planning')
    assert s.plan_mass==2.75


def test_custom_profile_cannot_offer_measured_count_response(tmp_path):
    app,s=planner(tmp_path)
    assert any(row['value']=='current_cycle' for row in s.plan_response_options)
    s.plan_custom_profile='/uploaded/custom.instprm';app._planning_context_edited()
    assert s.plan_response_mode=='profile_only'
    assert s.plan_response_options==[{'title':'Pattern shape preview','value':'profile_only'}]
    assert all(row['value']!='cycle_table' for row in s.plan_resolution_options)


def test_cycle_switch_selects_a_matching_profile(tmp_path):
    app,s=planner(tmp_path)
    s.plan_cycle='2025-2_11A_CAL';app._planning_cycle_edited();app._planning_context_edited()
    assert s.plan_configuration.startswith('2025B_')
    assert s.plan_response_mode=='profile_only'
    assert all(row['value'].startswith('2025B_') for row in s.plan_visible_configurations)


def test_saved_plan_survives_new_controller_and_captures_source(tmp_path):
    app,s=planner(tmp_path);app.load_planning_example()
    s.plan_name='Silicon · long exposure';s.plan_mass=1.25;s.plan_seconds='60, 600'
    app.save_planning_plan()
    assert not s.plan_error
    uid=s.plan_saved_id
    original=s.plan_cif
    another,t=planner(tmp_path)
    assert any(row['value']==uid for row in t.plan_saved_options)
    t.plan_saved_id=uid;another.restore_planning_plan()
    assert not t.plan_error
    assert (t.plan_mass,t.plan_seconds,t.plan_name)==(1.25,'60, 600','Silicon · long exposure')
    assert t.plan_cif!=original
    assert Path(t.plan_cif).read_bytes()==Path(original).read_bytes()
    assert another._planning_input(t.plan_cif)==Path(t.plan_cif)


def test_changed_saved_input_cannot_be_restored(tmp_path):
    app,s=planner(tmp_path);app.load_planning_example();app.save_planning_plan()
    saved=app._planning_root()/'saved'/s.plan_saved_id
    record=json.loads((saved/'plan.json').read_text())
    (saved/record['settings']['plan_cif']).write_text('changed')
    app.restore_planning_plan()
    assert 'missing or changed' in s.plan_error
    assert not s.plan_current and not s.plan_result_ready


def test_client_path_cannot_be_used_as_scientific_input_or_saved(tmp_path):
    app,s=planner(tmp_path)
    private=tmp_path/'private.cif';private.write_text('not an uploaded structure')
    with pytest.raises(ValueError,match='Upload'):app._planning_input(str(private))
    s.plan_cif=str(private);app.save_planning_plan()
    assert 'Upload' in s.plan_error
    assert not s.plan_saved_options


def test_saved_plan_selector_is_not_a_filesystem_path(tmp_path):
    app,s=planner(tmp_path);s.plan_saved_id='../../another-session'
    app.restore_planning_plan()
    assert 'Select a saved plan' in s.plan_error


def test_invalid_saved_result_does_not_replace_current_inputs(tmp_path):
    app,s=planner(tmp_path);app.load_planning_example();app.save_planning_plan()
    path=app._planning_root()/'saved'/s.plan_saved_id/'plan.json'
    data=json.loads(path.read_text());data['backend']['_planning_output']='/outside/workspace'
    path.write_text(json.dumps(data));s.plan_mass=2.5
    app.restore_planning_plan()
    assert 'outside this workspace' in s.plan_error
    assert s.plan_mass==2.5


def test_cancel_only_releases_its_own_job(tmp_path):
    app,s=planner(tmp_path);old=threading.Event();current=threading.Event()
    app._planning_cancel=current;s.plan_busy=True
    app._planning_finish_job(old)
    assert s.plan_busy and app._planning_cancel is current
    app.cancel_planning_simulation();app._planning_finish_job(current)
    assert not s.plan_busy and app._planning_cancel is None
    assert 'cancelled' in s.plan_message


def test_return_to_analysis_refreshes_its_selected_view_without_losing_rail(tmp_path):
    app,s=planner(tmp_path);s.workspace_view='plots';s.setup_collapsed=False
    def select(view):
        assert view=='plots'
        s.setup_collapsed=True
    app.select_workspace_view=select
    app.open_planning_workspace();app.close_planning_workspace()
    assert s.workspace_view=='plots' and s.setup_collapsed is False


def test_changed_engine_restores_inputs_without_publishing_old_physics(tmp_path,monkeypatch):
    app,s=planner(tmp_path);app.load_planning_example();app.save_planning_plan()
    monkeypatch.setattr('radar_pd_nova.planning_storage.implementation_signature',lambda:'new-engine')
    s.plan_current=True;app.restore_planning_plan()
    assert not s.plan_error
    assert not s.plan_current and not s.plan_result_ready
    assert app._planning_parent_output is None and app._planning_prepared is None
    assert 'engine changed' in s.plan_message


@pytest.mark.parametrize('kind', ['magnetic','structural'])
def test_selected_transition_requires_its_prepared_state(tmp_path,kind):
    app,s=planner(tmp_path);s.plan_transition=kind
    with pytest.raises(ValueError,match='Prepare'):app._planning_validate_transition()
    app._planning_prepared_kind=kind
    app._planning_validate_transition()


def test_partial_child_cell_blocks_simulation_until_completed_or_cleared(tmp_path):
    app,s=planner(tmp_path);app.load_planning_example()
    app.update_planning_cell(0,'5.5')
    assert 'cell' in app._planning_input_errors
    with pytest.raises(ValueError,match='invalid transition'):app._planning_request()
    app.update_planning_cell(0,'')
    assert not app._planning_input_errors and s.plan_structural_cell==''
    for i,value in enumerate([5.5,5.5,5.5,90,90,90]):app.update_planning_cell(i,str(value))
    assert not app._planning_input_errors
    assert json.loads(s.plan_structural_cell)==[5.5,5.5,5.5,90,90,90]


def test_parent_preparation_does_not_publish_a_changed_request(tmp_path,monkeypatch):
    app,s=planner(tmp_path);app.load_planning_example()
    event=threading.Event();generation=app._planning_generation
    def execute(request,directory,cancel):
        assert request['response_mode']=='profile_only'
        app._planning_generation+=1
        return {}
    monkeypatch.setattr(app,'_planning_execute',execute)
    asyncio.run(app._ensure_planning_parent(generation,event))
    assert app._planning_parent_output is None
    assert not s.plan_current


def test_cancelled_discovery_cannot_publish_a_parent(tmp_path,monkeypatch):
    app,s=planner(tmp_path);app.load_planning_example()
    event=threading.Event()
    def execute(request,directory,cancel):
        cancel.set();return {}
    monkeypatch.setattr(app,'_planning_execute',execute)
    asyncio.run(app._ensure_planning_parent(app._planning_generation,event))
    assert app._planning_parent_output is None
