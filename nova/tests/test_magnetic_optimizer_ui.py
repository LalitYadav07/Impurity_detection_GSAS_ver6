"""A new optimizer is an explicit scientific choice, not a hidden retry."""
import asyncio
from types import SimpleNamespace

import pytest

from radar_pd_nova.magnetic_optimizer import selected_optimizer_settings, optimizer_budget_label
from radar_pd_nova.model_campaigns import scope_progress
from radar_pd_nova.symmetry_controller import MODEL_FIELDS, SymmetryMixin
from radar_pd_nova.symmetry_workspace import fit_caption, fit_diagnostics, moment_precision_note, parameter_rows


def test_legacy_and_structural_choices_do_not_add_protocol_keys():
    assert selected_optimizer_settings('magnetic', 'gsas_hessian', None) == {}
    assert selected_optimizer_settings('structural', 'profiled_quadratic', 60) == {}


def test_profiled_choice_preserves_its_explicit_iteration_budget():
    assert selected_optimizer_settings('magnetic', 'profiled_quadratic', '60') == {
        'magnetic_optimizer': 'profiled_quadratic', 'optimizer_iterations': 60}


@pytest.mark.parametrize('value', [0, 201, 1.5, True, None, 'nan', ''])
def test_invalid_profiled_iteration_budget_is_rejected(value):
    with pytest.raises(ValueError):
        selected_optimizer_settings('magnetic', 'profiled_quadratic', value)


def test_unknown_optimizer_cannot_silently_use_the_default():
    with pytest.raises(ValueError):
        selected_optimizer_settings('magnetic', 'unreviewed', 60)


def test_legacy_budget_label_is_unchanged():
    assert optimizer_budget_label({'cycles': 30}) == 'Up to 30 refinement cycles per start.'
    assert optimizer_budget_label({'cycles': 1}) == 'Up to 1 refinement cycles per start. Preliminary one-cycle screening.'


def test_profiled_progress_reports_iterations_not_unused_native_cycles():
    policy = {'cycles': 30, 'magnetic_optimizer': 'profiled_quadratic', 'optimizer_iterations': 60}
    label = scope_progress({'protocol': {'policy': policy, 'candidates': []}}, [{'status': 'enrolled'}])
    assert '60 profiled optimizer iterations per start' in label
    assert '30 refinement cycles' not in label
    assert 'GSAS-II evaluates' in label


def test_optimizer_changes_invalidate_an_inflight_setup_reply():
    state = SimpleNamespace(rf_experiment='experiment', sx_start_source='source', sx_request='search',
                            sx_magnetic_optimizer='gsas_hessian', sx_optimizer_iterations=60)
    app = SymmetryMixin()
    app.server = SimpleNamespace(state=state)
    app._symmetry_draft = {'context': {'sha256': 'sourcehash'}}
    before = app._symmetry_setup_identity()
    state.sx_magnetic_optimizer = 'profiled_quadratic'
    after = app._symmetry_setup_identity()
    assert before != after
    state.sx_optimizer_iterations = 80
    assert after != app._symmetry_setup_identity()


def test_optimizer_and_iteration_budget_are_in_the_approved_form_identity():
    assert 'sx_magnetic_optimizer' in MODEL_FIELDS
    assert 'sx_optimizer_iterations' in MODEL_FIELDS


def planner_app(tmp_path, optimizer):
    state = SimpleNamespace(sx_start_source='scan', sx_request='Find k and refine', sx_k='',
                            sx_kind='magnetic', sx_magnetic_optimizer=optimizer,
                            sx_optimizer_iterations=60, sx_existing='selected',
                            sx_max_scans=1, sx_follow_new=False, flush=lambda: None)
    app = SymmetryMixin()
    app.server = SimpleNamespace(state=state)
    app._symmetry_draft = dict(source_id='scan', context={}, root=tmp_path)
    app._refinement_schedule = lambda coroutine, name: asyncio.run(coroutine)
    app._symmetry_unapprove = lambda: None
    settings = dict(kind='magnetic', parent='Hematite', histogram='PWDR scan', k=None,
                    sites=[dict(label='Fe', form_factor='Fe+3', lande_g=2.)],
                    moment_model='symmetry_allowed', refine_histogram_scale=True,
                    refine_background=True, directions=['x', 'y', 'z'], adequate_rwp=15.,
                    maximum_moment=6., magnetic_d_min=0., cycles=20, relative_sign_starts=True)
    return app, state, settings


@pytest.mark.parametrize('optimizer', ['gsas_hessian', 'profiled_quadratic'])
def test_visible_optimizer_is_passed_as_a_reviewed_luna_declaration(tmp_path, monkeypatch, optimizer):
    from radar_pd_nova import symmetry_controller as controller
    app, state, settings = planner_app(tmp_path, optimizer)
    expected = selected_optimizer_settings('magnetic', optimizer, 60)
    seen = []
    def draft(*args, declared):
        seen.append(declared)
        return dict(status='ready_for_k_survey', calls=[], intent=dict(
            settings={**settings, **expected}, enrollment=None,
            assumptions=[], questions=[], unsupported=[]))
    monkeypatch.setattr(controller, 'make_intent', draft)
    app.plan_symmetry_request()
    assert seen == [expected]
    assert not state.sx_error and not state.sx_busy
    assert state.sx_magnetic_optimizer == optimizer
    assert state.sx_optimizer_iterations == 60


def test_luna_reply_is_not_applied_after_the_user_changes_optimizer_budget(tmp_path, monkeypatch):
    from radar_pd_nova import symmetry_controller as controller
    app, state, settings = planner_app(tmp_path, 'profiled_quadratic')
    def draft(*args, declared):
        state.sx_optimizer_iterations = 80
        return dict(status='ready_for_k_survey', calls=[], intent=dict(
            settings={**settings, **declared}, enrollment=None,
            assumptions=[], questions=[], unsupported=[]))
    monkeypatch.setattr(controller, 'make_intent', draft)
    app.plan_symmetry_request()
    assert 'changed during planning' in state.sx_error
    assert state.sx_optimizer_iterations == 80
    assert 'planning' not in app._symmetry_draft


def profiled_row(**changes):
    row = dict(kind='magnetic', name='C2/c', status='valid', rwp=10.8, adequate_rwp_percent=15.,
               native_convergence={'converged': None, 'Aborted': None, 'SVD0': None},
               optimizer=dict(schema='radar-profiled-magnetic-optimizer/v1', name='profiled_quadratic',
                              success=True, iterations=32, maximum_iterations=60,
                              qualification={'passed': True}, uncertainty_available=False))
    row.update(changes)
    return row


def test_profiled_fit_never_claims_native_optimizer_convergence_or_covariance():
    row = profiled_row()
    assert 'profiled optimization checks passed' in fit_caption(row)
    diagnostics = ' '.join(item['value'] for item in fit_diagnostics(row))
    assert '32 / 60 iterations' in diagnostics
    assert 'GSAS-II convergence and covariance are not claimed' in diagnostics
    note = moment_precision_note(row)
    assert 'independent native-objective checks passed' in note
    assert 'does not establish canting' in note
    assert 'uncertainties are unavailable' in note


def test_profiled_solver_failure_is_not_hidden_by_a_low_residual():
    row = profiled_row(status='optimization_unresolved')
    row['optimizer']['success'] = False
    row['optimizer']['qualification']['passed'] = False
    assert 'checks passed' not in fit_caption(row)
    assert 'unqualified checkpoint' in moment_precision_note(row)
    assert 'Independent checks did not pass' in ' '.join(item['value'] for item in fit_diagnostics(row))


def test_profiled_parameters_do_not_reuse_stale_native_error_bars():
    row = profiled_row(refined_moments=[dict(label='Fe', magnitude_esd=.1,
                       component_esd={'mx': .1, 'my': .1, 'mz': .1})])
    view = {'phases': [dict(name='Magnetic', moments=[dict(label='Fe', magnitude=3.7,
                                                   components=[.5, 0., 3.6])])]}
    values = parameter_rows(row, view)
    assert len(values) == 4 and all(item['uncertainty'] == '—' for item in values)
    assert values[0]['value'] == 3.7
