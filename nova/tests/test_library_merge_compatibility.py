"""Merged partial-library and occupancy controls preserve strict old transports."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import zipfile

import pytest

from radar_pd_nova.galaxy_service import GalaxyService, LIBRARY_BUILDER_TOOL_ID
from radar_pd_nova.library_setup import activation_for_review, inspect_library_review
from radar_pd_nova.models import AnalysisConfig, InputSelection, SubmissionSnapshot


@pytest.fixture
def transport(tmp_path, monkeypatch):
    posts = []
    class Gate:
        total_limit = 8
        def __init__(self, _): pass
        def slot(self, *args, **kwargs): return nullcontext()
    monkeypatch.setattr('radar_pd_nova.native_job_admission.HistoryJobAdmission', Gate)
    class Response:
        def raise_for_status(self): pass
        def json(self): return {'jobs': [{'id': 'accepted'}]}
    def post(url, **kwargs):
        posts.append(kwargs['json']); return Response()
    monkeypatch.setattr('radar_pd_nova.galaxy_service.requests.post', post)
    service = GalaxyService('https://example.invalid', 'test-key', 'history', output_root=tmp_path)
    return service, posts


@pytest.mark.parametrize('zero', [None, '0', '0.0', 0, 0.0])
def test_strict_builder_uses_only_original_wrapper_inputs(transport, zero):
    service, posts = transport
    def no_probe(*args): raise AssertionError('Strict mode needs no new wrapper capability')
    service.tool_has_input = no_probe
    inputs = {'cif_archive': {'dataset_id': 'cifs'}, 'library_name': 'Strict library',
              'library_mode': 'mini', 'radiation': 'neutron', 'overwrite': ''}
    if zero is not None:
        inputs.update(reviewed_occupancy_excess=zero, occupancy_review_note='not activated')
    original = deepcopy(inputs)
    service.submit_utility(tool_id=LIBRARY_BUILDER_TOOL_ID, name='Build', inputs=inputs)
    assert inputs == original
    assert set(posts[0]['inputs']) == {'cif_archive', 'library_name', 'library_mode', 'radiation', 'overwrite'}
    assert posts[0]['inputs']['cif_archive'] == {'src': 'hda', 'id': 'cifs'}


@pytest.mark.parametrize('supported', [False, True])
def test_opt_in_rounding_still_requires_support_and_keeps_exact_note(transport, supported):
    service, posts = transport
    service.tool_has_input = lambda *args: supported
    inputs = {'cif_archive': {'dataset_id': 'cifs'}, 'reviewed_occupancy_excess': '0.001',
              'occupancy_review_note': "Reviewed O'Brien source"}
    if not supported:
        with pytest.raises(ValueError, match='does not support'):
            service.submit_utility(tool_id=LIBRARY_BUILDER_TOOL_ID, name='Build', inputs=inputs)
        assert not posts
    else:
        service.submit_utility(tool_id=LIBRARY_BUILDER_TOOL_ID, name='Build', inputs=inputs)
        assert posts[0]['inputs']['reviewed_occupancy_excess'] == '0.001'
        assert posts[0]['inputs']['occupancy_review_note'] == inputs['occupancy_review_note']


def library(tmp_path: Path, failure: bool):
    path = tmp_path / 'library.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('pack/manifest.json', json.dumps({'n_phases': 1, 'phase_ids': ['kept'],
            'failures': [{'source_name': 'missing-main.cif', 'error': 'Invalid structure'}] if failure else []}))
    return path


@pytest.mark.parametrize('failure', [False, True])
def test_combined_archive_gate_is_strict_by_default_and_partial_requires_permission(tmp_path, failure):
    path = library(tmp_path, failure)
    inputs = InputSelection(source='galaxy', data_dataset_id='data', instrument_dataset_id='profile',
                            database_archive_path=str(path))
    service = object.__new__(GalaxyService)
    def old_wrapper(*args): raise AssertionError('No new control is needed for strict complete archives')
    service.tool_has_input = old_wrapper
    prepared = {'configuration': 'config', 'database_archive': 'library', 'data': 'data', 'instrument': 'profile'}
    snapshot = SubmissionSnapshot(config=AnalysisConfig(sample_elements=['Al', 'O']),
                                  inputs=inputs, idempotency_token='strict-compat')
    if not failure:
        params, _ = service._prepared_parameters(None, snapshot, prepared)
        assert 'library|database|library_review_activation' not in params.inputs
        return
    with pytest.raises(ValueError, match='Review the selected'):
        service._prepared_parameters(None, snapshot, prepared)
    permission = activation_for_review(inspect_library_review(path))
    snapshot = snapshot.model_copy(update={'config': AnalysisConfig(sample_elements=['Al', 'O'], library_review=permission)})
    service.tool_has_input = lambda *args: False
    with pytest.raises(ValueError, match='does not support'):
        service._prepared_parameters(None, snapshot, prepared)
    service.tool_has_input = lambda *args: True
    params, _ = service._prepared_parameters(None, snapshot, prepared)
    assert json.loads(params.inputs['library|database|library_review_activation']) == permission
