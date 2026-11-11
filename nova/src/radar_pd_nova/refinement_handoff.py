"""Explicit entry into recipe review from another analysis workspace."""


def open_source(app, source, request):
    from .refinement_controller import refinement_experiment_options
    from .refinement_sources import published_selection_key

    state = app.server.state
    workspace = app._refinement_store()
    workspace.observe([source], enroll_future=False, manual=True)
    previous = getattr(app, '_refinement_internal', False)
    app._refinement_internal = True
    try:
        state.rf_experiments = [dict(title=v, value=k)
                               for k, v in refinement_experiment_options(workspace).items()]
        state.rf_experiment = source['experiment']
        state.rf_source = source['id']
        state.rf_scope = 'scans'
        state.rf_scans = [published_selection_key(source)]
        state.rf_samples = []
        state.rf_banks = []
        state.rf_existing = True
        state.rf_future = False
        state.rf_mode = 'recipe'
        state.rf_selected_job = ''
        state.rf_setup_mode = 'expert'
        state.rf_request = request
        state.experiment_result_view = 'refinement'
        state.flush()
    finally:
        app._refinement_internal = previous
    app._refinement_sync()
    app.inspect_refinement_project()
