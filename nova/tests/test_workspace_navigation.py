"""User journeys between analysis and the two before-measurement tools."""
from types import SimpleNamespace

import pytest

from radar_pd_nova.planning_controller import PlanningMixin
from radar_pd_nova.preparation_controller import PreparationMixin
from radar_pd_nova.workspace_navigation import WorkspaceNavigationMixin


class State(SimpleNamespace):
    def flush(self):
        pass


class App(WorkspaceNavigationMixin, PreparationMixin, PlanningMixin):
    def _prep_refresh(self):
        self.refreshes += 1

    def _prepared_selection_caption(self):
        return 'Selected reusable calculations'


def app(view='experiment', collapsed=False):
    instance = App()
    instance.server = SimpleNamespace(state=State(
        workspace_view=view, setup_collapsed=collapsed, live_assistant_open=False,
        selected_run_uid='run-58786', powgen_ipts='34411', prep_training_record_open=False,
        plan_mass=2.5, plan_current=True, prep_library='validated-library',
    ))
    instance.refreshes = 0
    instance._prep_return = None
    instance._planning_previous_workspace = None
    instance.active_jobs = ['training', 'simulation', 'analysis']
    return instance, instance.server.state


@pytest.mark.parametrize('first,second', [('planning','preparation'), ('preparation','planning')])
@pytest.mark.parametrize('view,collapsed', [('experiment',True),('plots',False),('refinement',True)])
def test_switch_tools_returns_to_exact_analysis_view(first, second, view, collapsed):
    instance, state = app(view, collapsed)
    for value in (first, second, first, 'analysis'):
        instance.select_primary_workspace(value)
    assert (state.workspace_view, state.setup_collapsed) == (view, collapsed)
    assert state.selected_run_uid == 'run-58786'
    assert state.powgen_ipts == '34411'
    assert (state.plan_mass, state.plan_current, state.prep_library) == (2.5, True, 'validated-library')
    assert instance.active_jobs == ['training', 'simulation', 'analysis']


def test_direct_open_buttons_and_back_use_same_return_context():
    instance, state = app('plots', False)
    instance.open_planning_workspace()
    instance.open_preparation_workspace()
    instance.close_preparation_workspace()
    assert (state.workspace_view, state.setup_collapsed) == ('plots', False)


def test_new_analysis_selection_becomes_next_return_point():
    instance, state = app()
    instance.select_primary_workspace('planning')
    instance.select_primary_workspace('analysis')
    state.workspace_view = 'refinement'
    state.setup_collapsed = False
    instance.select_primary_workspace('preparation')
    instance.select_primary_workspace('analysis')
    assert (state.workspace_view, state.setup_collapsed) == ('refinement', False)


def test_current_tool_click_is_idempotent_and_analysis_click_does_not_reset_view():
    instance, state = app('plots', False)
    instance.select_primary_workspace('analysis')
    assert state.workspace_view == 'plots'
    instance.select_primary_workspace('preparation')
    count = instance.refreshes
    instance.select_primary_workspace('preparation')
    assert instance.refreshes == count


def test_unknown_workspace_cannot_mutate_current_selection():
    instance, state = app()
    with pytest.raises(ValueError, match='Choose Analysis'):
        instance.select_primary_workspace('../../another-session')
    assert state.workspace_view == 'experiment'
