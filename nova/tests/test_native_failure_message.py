"""A native fatal cause must survive later generic runner stderr."""
import pytest

from radar_pd_nova.galaxy_service import GalaxyService
from radar_pd_nova.models import AnalysisMode, RunRecord, RunStatus, UtilityActionRecord


def refresh_error(tmp_path, monkeypatch, stdout, stderr):
    service = GalaxyService('https://galaxy.example', 'key', 'history', output_root=tmp_path)
    job = {'id':'failed', 'state':'error', 'stdout':stdout, 'stderr':stderr, 'params':{}}
    monkeypatch.setattr(service, '_job_details', lambda _uid: job)
    monkeypatch.setattr(service, '_job_output_ids', lambda _job: {})
    monkeypatch.setattr(service, '_public_input_dataset_ids', lambda _job: {})
    monkeypatch.setattr(service, '_config_from_job', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(service, '_inputs_from_parameters', lambda *_args, **_kwargs: None)
    record = RunRecord(uid='failed', name='failed', mode=AnalysisMode.FULL, history_id='history')
    return service.refresh(record)


def test_native_fatal_precedes_missing_output_stderr_and_preserves_log(tmp_path, monkeypatch):
    stdout = ('Screening complete\n[FATAL] Candidate import failed: source occupancy was changed\n'
              'Traceback: native call path retained\n')
    stderr = 'Pulsar metadata processing failed: missing pattern_observation.json'
    record = refresh_error(tmp_path, monkeypatch, stdout, stderr)
    assert record.status is RunStatus.ERROR
    assert record.message.startswith('[FATAL] Candidate import failed: source occupancy was changed')
    assert stderr in record.message
    assert record.console_tail == stdout


def test_long_native_error_remains_bounded_without_losing_fatal_prefix(tmp_path, monkeypatch):
    stdout = '[FATAL] scientific cause ' + 'x'*12000 + '\ntraceback\n'
    stderr = 'runner context '*3000
    record = refresh_error(tmp_path, monkeypatch, stdout, stderr)
    assert record.message.startswith('[FATAL] scientific cause ')
    assert len(record.message) <= 4000
    assert record.console_tail == stdout[-12000:]


def test_only_a_line_start_fatal_marker_overrides_stderr(tmp_path, monkeypatch):
    record = refresh_error(tmp_path, monkeypatch, 'Example text containing [FATAL] in a quotation', 'actual runner failure')
    assert record.message == 'actual runner failure'


def test_last_native_fatal_is_used_and_plain_stderr_fallback_unchanged(tmp_path, monkeypatch):
    record = refresh_error(tmp_path, monkeypatch, '[FATAL] first\nprogress\n[FATAL] final cause\n', 'runner failure')
    assert record.message.startswith('[FATAL] final cause')
    fallback = refresh_error(tmp_path, monkeypatch, 'ordinary output', 'x'*5000)
    assert fallback.message == 'x'*4000


def test_failed_input_proof_keeps_priority_over_stdout(tmp_path, monkeypatch):
    service = GalaxyService('https://galaxy.example', 'key', 'history', output_root=tmp_path)
    job = {'id':'failed', 'state':'paused', 'stdout':'[FATAL] stale log', 'stderr':'runner detail', 'params':{}}
    monkeypatch.setattr(service, '_job_details', lambda _uid: job)
    monkeypatch.setattr(service, '_job_output_ids', lambda _job: {})
    monkeypatch.setattr(service, '_public_input_dataset_ids', lambda _job: {})
    monkeypatch.setattr(service, '_failed_input_message', lambda _job: 'Input upload failed; review it in History')
    monkeypatch.setattr(service, '_config_from_job', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(service, '_inputs_from_parameters', lambda *_args, **_kwargs: None)
    record = RunRecord(uid='failed', name='failed', mode=AnalysisMode.FULL, history_id='history')
    assert service.refresh(record).message == 'Input upload failed; review it in History'


def test_utility_error_uses_native_cause_and_keeps_its_smaller_bound(tmp_path, monkeypatch):
    service = GalaxyService('https://galaxy.example', 'key', 'history', output_root=tmp_path)
    job = {'id':'utility', 'state':'error', 'stdout':'[FATAL] builder validation failed', 'stderr':'Runner missing output'}
    monkeypatch.setattr(service, '_job_details', lambda _uid: job)
    monkeypatch.setattr(service, '_job_output_ids', lambda _job: {})
    action = UtilityActionRecord(uid='utility', galaxy_job_id='utility', tool_id='builder', name='builder')
    result = service.refresh_utility(action)
    assert result.message.startswith('[FATAL] builder validation failed')
    assert 'Runner missing output' in result.message
    assert len(result.message) <= 2000


def test_long_utility_fatal_still_retains_runner_context(tmp_path, monkeypatch):
    service = GalaxyService('https://galaxy.example', 'key', 'history', output_root=tmp_path)
    job = {'id':'utility', 'state':'error', 'stdout':'[FATAL] '+'x'*6000, 'stderr':'Runner missing output'}
    monkeypatch.setattr(service, '_job_details', lambda _uid: job)
    monkeypatch.setattr(service, '_job_output_ids', lambda _job: {})
    action = UtilityActionRecord(uid='utility', galaxy_job_id='utility', tool_id='builder', name='builder')
    result = service.refresh_utility(action)
    assert result.message.startswith('[FATAL]')
    assert result.message.endswith('Runner missing output')
    assert len(result.message) <= 2000


@pytest.fixture
def refresh_library_error(tmp_path, monkeypatch):
    service = GalaxyService('https://galaxy.example', 'key', 'history', output_root=tmp_path)
    monkeypatch.setattr(service, '_job_output_ids', lambda _job: {})

    def refresh(**diagnostics):
        job = {'id': 'failed-library-job', 'state': 'error', **diagnostics}
        monkeypatch.setattr(service, '_job_details', lambda _uid: job)
        action = UtilityActionRecord(uid='library', galaxy_job_id=job['id'],
                                     tool_id='neutrons_radar_pd_library_builder_prototype', name='Build library')
        return service.refresh_utility(action)

    return refresh


@pytest.mark.parametrize('diagnostics', [
    {},
    {'stdout': '', 'stderr': '', 'exit_code': None},
    {'stdout': ' \n', 'stderr': '\t', 'exit_code': None},
    {'tool_stdout': None, 'job_stderr': '', 'exit_code': None},
])
def test_library_failure_without_diagnostics_names_job_and_missing_evidence(refresh_library_error, diagnostics):
    result = refresh_library_error(**diagnostics)
    assert result.status is RunStatus.ERROR
    assert result.message == (
        'Galaxy job failed-library-job ended with an error. '
        'No stdout, stderr, or exit code was available. Open the job details in Galaxy.'
    )


@pytest.mark.parametrize('exit_code', [0, 1, -9])
def test_library_failure_reports_available_exit_code_without_guessing_cause(refresh_library_error, exit_code):
    result = refresh_library_error(exit_code=exit_code)
    assert result.message == (
        f'Galaxy job failed-library-job ended with an error (exit code {exit_code}). '
        'No stdout or stderr was available. Open the job details in Galaxy.'
    )


@pytest.mark.parametrize('diagnostics, expected', [
    ({'stderr': 'CIF validation failed\n'}, 'CIF validation failed\n'),
    ({'tool_stderr': 'Native builder detail'}, 'Native builder detail'),
    ({'stderr': ' \n', 'tool_stderr': 'Native builder detail'}, 'Native builder detail'),
    ({'stdout': '\t', 'job_stdout': 'Useful builder failure'}, 'Useful builder failure'),
    ({'stdout': 'Useful builder failure'}, 'Useful builder failure'),
])
def test_library_failure_preserves_available_diagnostics(refresh_library_error, diagnostics, expected):
    assert refresh_library_error(**diagnostics).message == expected
