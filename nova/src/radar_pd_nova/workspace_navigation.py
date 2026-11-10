"""Keep independent planning workspaces attached to one analysis context.

Switching between the two tools must not replace the user's return point with
the other tool. Navigation does not cancel or reset any scientific operation.
"""
from __future__ import annotations


TOOLS = frozenset({'planning', 'preparation'})


class WorkspaceNavigationMixin:
    def _remember_analysis_context(self):
        state = self.server.state
        current = (state.workspace_view, state.setup_collapsed)
        if current[0] not in TOOLS:
            self._analysis_workspace_return = current
        if getattr(self, '_analysis_workspace_return', None) is None:
            candidates = [getattr(self, '_prep_return', None),
                          getattr(self, '_planning_previous_workspace', None)]
            self._analysis_workspace_return = next(
                (value for value in candidates if value and value[0] not in TOOLS),
                ('monitor', False),
            )
        return self._analysis_workspace_return

    def open_preparation_workspace(self, **kwargs):
        return_to = self._remember_analysis_context()
        result = super().open_preparation_workspace(**kwargs)
        self._prep_return = return_to
        return result

    def open_planning_workspace(self, **kwargs):
        return_to = self._remember_analysis_context()
        result = super().open_planning_workspace(**kwargs)
        self._planning_previous_workspace = return_to
        return result

    def select_primary_workspace(self, value, **_):
        if value not in {'analysis', *TOOLS}:
            raise ValueError('Choose Analysis, Experiment Planner or Experiment preparation')
        state = self.server.state
        if value == state.workspace_view:
            return
        if value == 'planning':
            self.open_planning_workspace()
        elif value == 'preparation':
            self.open_preparation_workspace()
        elif state.workspace_view == 'planning':
            self.close_planning_workspace()
        elif state.workspace_view == 'preparation':
            self.close_preparation_workspace()
