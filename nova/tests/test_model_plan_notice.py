"""A new plan owns its status message without erasing scientific warnings."""
from types import SimpleNamespace

import pytest

from radar_pd_nova import symmetry_controller as module


@pytest.fixture
def plan_app(tmp_path, monkeypatch):
    state=SimpleNamespace(sx_error='This exact campaign already exists',
        sx_message='Campaign stopped; review the error above.', sx_adequate=10.,
        sx_kind='structural', sx_parent='parent', sx_histogram='histogram', sx_k='0,0,0',
        sx_candidates=['new-candidate'], sx_modes=[], sx_directions=['x','y','z'],
        sx_sign_starts=True, sx_structural_starts=4, sx_structural_seconds=120.,
        sx_structural_radius=.05, sx_structural_cells=.01, sx_structural_recovery=0.,
        sx_parent_reference='', sx_magnetic_optimizer='profiled_quadratic',
        sx_optimizer_iterations=120, rf_experiment='experiment', sx_existing='selected',
        sx_follow_new=False, sx_max_scans=1, sx_selected_scans=['new-source'],
        sx_moment_model='symmetry_allowed', sx_plan_ready=False, sx_approved=False,
        sx_preparation_notice='Native context checked; optimization requires review',
        sx_source_notices=['Calibration is unverified'], sx_fit_diagnostics=['Old fit unresolved'])
    app=module.SymmetryMixin();app.server=SimpleNamespace(state=state)
    app._symmetry_draft=dict(discovery_identity='new-identity',source_id='new-source',
        source={'id':'new-source'}, context={}, catalog={}, root=tmp_path)
    app._discovery_identity=lambda:'new-identity'
    app._model_form=lambda payload=None:'new-form'
    app._refinement_store=lambda:object()
    candidate=dict(id='new-candidate',bns_number='',transformation_matrix=[],origin_shift=[],modes=[])
    monkeypatch.setattr(module,'build_plan',lambda *a:dict(candidates=[candidate],policy={}))
    monkeypatch.setattr(module,'verify_inputs',lambda *a:None)
    monkeypatch.setattr(module,'make_template',lambda *a,**kw:dict(existing_ids=['new-source'],future=False,maximum_scans=1))
    monkeypatch.setattr(module,'model_choices',lambda *a:[({'id':'new-source'},'radar')])
    monkeypatch.setattr(module,'model_origin',lambda *a:'radar')
    monkeypatch.setattr(module,'iter_trials',lambda *a:range(4))
    monkeypatch.setattr(module,'trial_count',lambda *a:4)
    monkeypatch.setattr(module,'candidate_display_name',lambda *a:'New candidate')
    monkeypatch.setattr(module,'optimizer_budget_label',lambda *a:'Four starts')
    monkeypatch.setattr(module,'selected_optimizer_settings',lambda *a:{})
    return app,state


def test_distinct_successful_plan_replaces_old_duplicate_rejection_notice(plan_app):
    app,state=plan_app
    warnings=(state.sx_preparation_notice,list(state.sx_source_notices),list(state.sx_fit_diagnostics))
    app.build_symmetry_plan()
    assert state.sx_plan_ready and not state.sx_approved and not state.sx_error
    assert state.sx_message=='Finite plan prepared. Review this plan and approve it before starting the campaign.'
    assert app._symmetry_draft['settings']['candidates']==['new-candidate']
    assert warnings==(state.sx_preparation_notice,state.sx_source_notices,state.sx_fit_diagnostics)


def test_failed_rebuild_does_not_claim_a_prepared_plan(plan_app):
    app,state=plan_app;state.sx_adequate=0
    app.build_symmetry_plan()
    assert not state.sx_plan_ready and not state.sx_approved
    assert 'adequate Rwp' in state.sx_error and 'Finite plan prepared' not in state.sx_message
    assert state.sx_source_notices==['Calibration is unverified']
