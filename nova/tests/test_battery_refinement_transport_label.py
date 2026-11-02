"""Battery handoffs must describe the runtime that will execute the recipe."""
from types import SimpleNamespace

import pytest

from radar_pd_nova.battery_controller import BatteryMixin
from radar_pd_nova import battery_preparation, refinement_controller
from test_battery_preparation_controller import State


@pytest.mark.parametrize('kind,label', [('local','Local scientific worker'),('galaxy','Galaxy scientific jobs')])
def test_prepared_model_handoff_keeps_configured_runtime(kind,label,monkeypatch,tmp_path):
    app=BatteryMixin();app.server=SimpleNamespace(state=State());app._init_battery()
    workspace=SimpleNamespace(runtime=SimpleNamespace(kind=kind))
    app._refinement_store=lambda:workspace
    app._battery_prepared=(tmp_path,{'prepared':True})
    source={'id':'exact-publication','experiment':'experiment'}
    observed=[]
    def prepared(w,path,result):
        assert w is workspace and path==tmp_path and result=={'prepared':True}
        observed.append('source');return source
    monkeypatch.setattr(battery_preparation,'prepared_source',prepared)
    monkeypatch.setattr(refinement_controller,'refinement_experiment_options',lambda w:{'experiment':'Battery'})
    app._refinement_sync=lambda:observed.append('sync')
    app.open_battery_refinement()
    state=app.server.state
    assert observed==['source','sync'] and not state.battery_error
    assert state.rf_transport==label and workspace.runtime.kind==kind
    assert state.rf_source=='exact-publication' and state.rf_scans==['exact-publication']
    assert state.rf_scope=='scans' and state.rf_existing is True and state.rf_future is False
    assert state.workspace_view=='refinement' and state.rf_mode=='recipe'


@pytest.mark.parametrize('kind,label', [('local','Local scientific worker'),('galaxy','Galaxy scientific jobs')])
def test_portfolio_result_handoff_keeps_configured_runtime(kind,label,monkeypatch):
    app=BatteryMixin();app.server=SimpleNamespace(state=State());app._init_battery()
    workspace=SimpleNamespace(runtime=SimpleNamespace(kind=kind),state={
        'battery_portfolios':{'portfolio':{'jobs':{'trial':'job'}}},
        'jobs':{'job':{'source':{'experiment':'experiment'}}}})
    app._refinement_store=lambda:workspace;app._battery_portfolio_token='portfolio'
    monkeypatch.setattr(refinement_controller,'refinement_experiment_options',lambda w:{'experiment':'Battery'})
    observed=[]
    app._refinement_sync=lambda:observed.append('sync')
    app._select_refinement_job=lambda uid:('selected',uid)
    app._refinement_schedule=lambda value,name:observed.append((value,name))
    app.open_battery_trial_result('job')
    state=app.server.state
    assert not state.battery_error and state.rf_transport==label and workspace.runtime.kind==kind
    assert state.workspace_view=='refinement' and state.rf_mode=='results'
    assert state.rf_experiment=='experiment'
    assert observed==['sync',(('selected','job'),'battery-trial-result')]
