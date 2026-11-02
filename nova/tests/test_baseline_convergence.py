from copy import deepcopy

import pytest

from radar_pd_nova.results import baseline_convergence_summary, build_result_view, experiment_scan_summary, experiment_convergence_notice


def payload():
    return {'analysis_mode': 'full', 'status': 'complete', 'scientific_status': 'checks_passed',
        'final_rwp': 12.5, 'lattice_parameters': [{'a': 4., 'a_esd': .001}],
        'gpx_projects': [{'source_path': 'Technical/final.gpx', 'sha256': 'a' * 64,
                          'published_sha256': 'b' * 64, 'status': 'accepted'}],
        'native_convergence': {'schema': 'radar-baseline-convergence/v1',
            'source_gpx': 'Technical/final.gpx', 'source_sha256': 'a' * 64,
            'published_sha256': 'b' * 64, 'native_converged': True, 'aborted': False,
            'covariance_fresh': True, 'independent_parameters': 2,
            'last_cycle_complete': True, 'missing_last_cycle_parameters': [],
            'last_cycle_ratios': {'0::A0': 1.6, ':0:Scale': -.25},
            'last_cycle_max_shift_over_sigma': 1.6, 'total_run_max_shift_over_sigma': 147.89}}


def test_native_true_above_one_sigma_is_presented_without_a_new_gate(tmp_path):
    data = payload(); before = deepcopy(data); view = build_result_view(data, tmp_path)
    assert view.metrics[1]['value'] == 'Complete'
    assert {'label': 'Native convergence', 'value': 'Reported'} in view.metrics
    assert {'label': 'Final-cycle max shift / s.u.', 'value': '1.6'} in view.metrics
    assert not any('convergence' in text.lower() for text in view.warnings)
    assert view.full_models[0]['decision'] == 'Pipeline accepted'
    assert view.checkpoints[0].status == 'Pipeline accepted'
    assert view.full_progression[0]['status'] == 'Pipeline accepted'
    assert data == before
    assert experiment_scan_summary(data)['lattice_parameters'] == data['lattice_parameters']


@pytest.mark.parametrize('update,label', [({'native_converged': False}, 'Not reached'),
                                         ({'aborted': True}, 'Aborted'),
                                         ({'covariance_fresh': False}, 'Unverified')])
def test_failure_remains_review_with_rwp_and_uncertainties_preserved(tmp_path, update, label):
    data = payload(); data['native_convergence'].update(update); view = build_result_view(data, tmp_path)
    assert view.metrics[1]['value'] == 'Complete · scientific review required'
    assert view.metrics[2]['value'] == '12.500%'
    assert {'label': 'Native convergence', 'value': label} in view.metrics
    assert view.warnings
    assert experiment_scan_summary(data)['lattice_parameters'][0]['a_esd'] == .001


@pytest.mark.parametrize('corrupt', [
    lambda d: d.pop('native_convergence'),
    lambda d: d['native_convergence'].update(schema='unknown'),
    lambda d: d['native_convergence'].update(source_sha256='c' * 64),
    lambda d: d['native_convergence'].update(published_sha256='c' * 64),
    lambda d: d['native_convergence'].update(source_gpx='another.gpx'),
    lambda d: d['gpx_projects'].append(deepcopy(d['gpx_projects'][0])),
    lambda d: d['native_convergence'].update(last_cycle_max_shift_over_sigma=147.89),
])
def test_legacy_or_unbound_evidence_is_unknown_not_passed(tmp_path, corrupt):
    data = payload(); corrupt(data); view = build_result_view(data, tmp_path)
    assert view.metrics[1]['value'] == 'Complete · convergence unknown'
    assert {'label': 'Native convergence', 'value': 'Unknown'} in view.metrics
    assert any('unknown' in text.lower() for text in view.warnings)
    assert data['scientific_status'] == 'checks_passed'


@pytest.mark.parametrize('edit', [
    {'last_cycle_complete': False, 'last_cycle_max_shift_over_sigma': None},
    {'last_cycle_ratios': {'0::A0': float('nan'), ':0:Scale': -.25}},
    {'last_cycle_ratios': {'0::A0': True, ':0:Scale': -.25}},
    {'independent_parameters': True},
    {'missing_last_cycle_parameters': ['missing']},
])
def test_incomplete_diagnostics_cannot_display_historical_value_as_final(tmp_path, edit):
    data = payload(); data['native_convergence'].update(edit); view = build_result_view(data, tmp_path)
    assert {'label': 'Final-cycle max shift / s.u.', 'value': 'Unavailable'} in view.metrics
    assert {'label': 'Native convergence', 'value': 'Unverified'} in view.metrics
    assert any('do not cover every independent parameter' in text for text in view.warnings)


def test_untrusted_native_messages_are_not_rendered(tmp_path):
    data = payload(); data['native_convergence'].update(native_converged=False,
        native_message='<script>alert(1)</script>', method='<img src=x onerror=alert(1)>')
    view = build_result_view(data, tmp_path)
    assert '<script>' not in str(view.to_state()) and 'onerror=' not in str(view.to_state())


def test_rapid_presentation_keeps_legacy_behavior(tmp_path):
    data = payload(); data['analysis_mode'] = 'rapid'; data.pop('native_convergence')
    view = build_result_view(data, tmp_path)
    assert view.metrics[1]['value'] == 'Complete'
    assert not any(row['label'] == 'Native convergence' for row in view.metrics)
    assert experiment_scan_summary(data)['native_convergence'] == {}


def test_failure_before_any_baseline_keeps_original_error_prominent(tmp_path):
    view = build_result_view({'analysis_mode': 'full', 'status': 'failed',
        'errors': [{'message': 'No candidate model converged'}]}, tmp_path)
    assert view.warnings == ['No candidate model converged']
    assert not any(row['label'] == 'Native convergence' for row in view.metrics)


def test_experiment_projection_carries_unknown_and_false_native_evidence():
    data = payload(); data.pop('native_convergence')
    summary = experiment_scan_summary(data)
    assert summary['native_convergence']['status'] == 'unknown'
    assert summary['scientific_status'] == 'needs_review'
    assert any('unknown' in message.lower() for message in summary['scientific_warnings'])
    data = payload(); data['native_convergence']['native_converged'] = False
    data['warnings'] = [{'code': 'native_convergence_review', 'message': 'Old exported convergence wording'}]
    summary = experiment_scan_summary(data)
    assert summary['native_convergence']['status'] == 'needs_review'
    assert any('did not reach convergence' in message for message in summary['scientific_warnings'])
    assert summary['warning_count'] == 1


def test_experiment_notice_counts_visible_fits_without_changing_values():
    good = experiment_scan_summary(payload())
    data = payload(); data['native_convergence']['native_converged'] = False
    bad = experiment_scan_summary(data)
    legacy = deepcopy(good); legacy.pop('native_convergence')
    incomplete = deepcopy(good); incomplete['native_convergence']['last_cycle_complete'] = False
    rows = [good, bad, legacy, incomplete, {'analysis_mode': 'rapid', 'rwp': 2.}, {'analysis_mode': 'full'}]
    original = deepcopy(rows); notice = experiment_convergence_notice(rows)
    assert '1 not converged or aborted' in notice
    assert '2 with unknown or incomplete' in notice
    assert '4 total' in notice and 'All values remain displayed' in notice
    assert rows == original
    assert experiment_convergence_notice([good]) == ''
    assert experiment_convergence_notice([]) == ''
    assert '1 total' in experiment_convergence_notice([legacy])
