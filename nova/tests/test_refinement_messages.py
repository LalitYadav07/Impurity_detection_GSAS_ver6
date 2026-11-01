from radar_pd_nova.refinement_messages import sample_width_failure_message
from radar_pd_nova.refinement_messages import convergence_notice
from gpx_recipe.convergence import SCHEMA


def test_failed_width_uses_native_ids_and_physical_bank_not_list_order():
    source=dict(context=dict(phases=[dict(native_id=7,name='Graphite')],
        histograms=[dict(native_id=9,name='PWDR Bank 1')]),metadata={'physical_banks':{'PWDR Bank 1':2}})
    result=dict(status='needs_review',stages=[{},{}],
        error="Native dependent parameters violated physical/requested bounds before save-time clipping: [('7:9:Size;i', -2.14)].")
    message=sample_width_failure_message(result,source)
    assert 'Graphite · physical bank 2 · sample size = -2.14 µm' in message
    assert '2 earlier verified stages' in message and 'no final GPX was accepted' in message
    assert sample_width_failure_message(result,{}) is None
    result['error']=result['error'].replace("('7:9:Size;i', -2.14)","('7:9:Size;i', -2.14), ('7::AUiso:0', -0.01)")
    assert sample_width_failure_message(result,source) is None  # Never hide a second, unmapped violation.


def test_saved_width_mismatch_shows_both_values_and_keeps_the_failed_stage_unaccepted():
    source=dict(context=dict(phases=[dict(native_id=7,name='oxide')],
        histograms=[dict(native_id=9,name='PWDR Bank 1')]),metadata={'physical_banks':{'PWDR Bank 1':2}})
    result=dict(status='needs_review',stages=[{},{}],
        error="Native saved parameters differ from solver values (variable, solver, saved): [('7:9:Size;i', 0.0003594, 0.001)].")
    message=sample_width_failure_message(result,source)
    assert 'oxide · physical bank 2' in message
    assert '0.0003594 µm in the solver, 0.001 µm in the saved model' in message
    assert 'The fitted and stored values disagree' in message
    assert 'Stage 3 stopped' in message and '2 earlier verified stages' in message
    result['error']=result['error'].replace('0.0003594, 0.001','0.0003594, None')
    assert 'unavailable in the saved model' in sample_width_failure_message(result,source)
    result['error']=result['error'].replace("('7:9:Size;i', 0.0003594, None)","('7:9:Size;i', 0.0003594, None), ('7:9:Scale', 0.2, 0.1)")
    assert sample_width_failure_message(result,source) is None


def test_convergence_notice_keeps_solver_stopping_and_final_cycle_movement_distinct():
    evidence=dict(schema=SCHEMA,native_converged=True,last_cycle_complete=True,
                  last_cycle_max_shift_over_sigma=1.907342,total_run_max_shift_over_sigma=.82)
    message=convergence_notice(dict(convergence=evidence))
    assert 'solver reported convergence' in message
    assert 'final-cycle shift / conditional s.u.: 1.9073' in message
    assert '0.82' not in message and 'passed' not in message
    assert 'reviewed convergence criteria' in message and 'do not establish model adequacy' in message


def test_missing_or_partial_convergence_does_not_use_total_run_movement():
    assert 'unavailable' in convergence_notice({})
    evidence=dict(schema=SCHEMA,native_converged=False,last_cycle_complete=False,
                  last_cycle_max_shift_over_sigma=.01,total_run_max_shift_over_sigma=9.)
    message=convergence_notice(dict(convergence=evidence))
    assert 'did not report convergence' in message and 'incomplete' in message
    assert 'may omit other parameters' in message and '9' not in message
    evidence.update(last_cycle_max_shift_over_sigma=True,aborted=True,relative_objective_change=-.2)
    message=convergence_notice(dict(convergence=evidence))
    assert 'reported an abort' in message and 'objective increased' in message
    assert 'Largest available' not in message


def test_objective_notice_distinguishes_native_tolerance_from_material_increase():
    evidence=dict(schema=SCHEMA,native_converged=True,last_cycle_complete=True,
                  last_cycle_max_shift_over_sigma=.02,relative_objective_change=-1e-7,
                  relative_objective_tolerance=.001)
    assert 'within the recorded native tolerance 0.001' in convergence_notice(dict(convergence=evidence))
    evidence['relative_objective_change']=-.002
    assert 'beyond the recorded native tolerance 0.001' in convergence_notice(dict(convergence=evidence))
