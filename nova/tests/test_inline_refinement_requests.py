"""Inline request transport keeps the existing approval and dispatch contract."""

import base64
from copy import deepcopy
import json
import threading
import zipfile

import pytest

from radar_pd_nova.refinement_runtime import REFINEMENT_TOOL_ID, RuntimeUnavailable
from test_refinement_transport import runtime


def _inline(service):
    calls = []

    def capability(tool, name):
        calls.append((tool, name))
        return tool == REFINEMENT_TOOL_ID and name == 'recipe_request_b64'

    service.tool_has_input = capability
    return calls


@pytest.mark.parametrize('action', ['prepare', 'inspect', 'edit', 'preview', 'run'])
def test_inline_closed_request_is_literal_and_uses_no_upload(tmp_path, monkeypatch, action):
    rt, service, source = runtime(tmp_path)
    capabilities = _inline(service)
    monkeypatch.setenv('AZURE_ILLUMINE_API_KEY', 'SECRET-NEVER-IN-WORKER')
    plan = {'request': "La'\n$(touch NEVER); `touch ALSO_NEVER` α", 'stages': []}
    approval = {'source': source['source_sha256'], 'reviewed': True}
    # A failed native result is sufficient here: transport must preserve it too.
    with zipfile.ZipFile(service.archive, 'w') as archive:
        archive.writestr('response.json', json.dumps({
            'schema': 'radar-refinement-worker/v1', 'action': action, 'status': 'failed',
            'source_sha256': source['source_sha256'], 'files': {},
        }))

    result = rt.call(action, source, tmp_path / 'output', plan=plan, approval=approval,
                     dispatch_key='1' * 24)

    assert result['status'] == 'failed'
    assert capabilities == [(REFINEMENT_TOOL_ID, 'recipe_request_b64')]
    assert not service.requests and len(service.submissions) == 1
    inputs = service.submissions[0]['inputs']
    assert set(inputs) == {'action', 'dispatch_key', 'source_gpx', 'recipe_request_b64'}
    assert inputs['dispatch_key'] == '1' * 24 and inputs['action'] == action
    assert inputs['source_gpx'] == {'dataset_id': 'original-data'}
    expected = dict(plan=plan, approval=approval, expected_source_sha256=source['source_sha256'],
                    dispatch_key='1' * 24)
    decoded = base64.b64decode(inputs['recipe_request_b64'], validate=True)
    assert decoded == (json.dumps(expected, indent=2, sort_keys=True) + '\n').encode()
    assert json.loads(decoded) == expected and b'SECRET-NEVER-IN-WORKER' not in decoded


@pytest.mark.parametrize('advertised', [False, None])
def test_legacy_wrapper_keeps_the_existing_hda_dependency(tmp_path, advertised):
    rt, service, source = runtime(tmp_path)
    if advertised is not None:
        service.tool_has_input = lambda *_: advertised
    uploads = []
    service.upload_json_document = lambda payload, **kwargs: (uploads.append((payload, kwargs)) or 'request-data')

    rt.call('inspect', source, tmp_path / 'output', dispatch_key='2' * 24)

    assert len(uploads) == 1 and uploads[0][1]['wait_ready'] is False
    assert service.submissions[0]['inputs']['recipe_request'] == {'dataset_id': 'request-data'}
    assert 'recipe_request_b64' not in service.submissions[0]['inputs']


def test_acknowledged_inline_dispatch_does_not_repeat_upload_lookup_or_post(tmp_path, monkeypatch):
    rt, service, source = runtime(tmp_path)
    _inline(service)
    original_job = service.job
    service.job = lambda: original_job().model_copy(update={'inputs': deepcopy(service.submissions[-1]['inputs'])})
    seen = []
    rt.call('inspect', source, tmp_path / 'first', dispatch_key='3' * 24, on_submitted=seen.append)
    saved = json.loads((rt.root / 'dispatches' / ('3' * 24 + '.json')).read_text())
    assert 'recipe_request_b64' not in saved['inputs']
    assert saved['inputs']['source_gpx'] == {'dataset_id': 'original-data'}
    assert seen == [saved]
    service.tool_has_input = lambda *_: pytest.fail('an acknowledged job must not rediscover capabilities')
    monkeypatch.setattr(rt, 'find_dispatch', lambda *_: pytest.fail('no History search for a saved acknowledgement'))

    rt.call('inspect', source, tmp_path / 'again', dispatch_key='3' * 24, on_submitted=seen.append)

    assert len(service.submissions) == 1 and not service.requests
    assert seen == [saved, saved]


@pytest.mark.parametrize('found', [True, False])
def test_inline_lost_ack_is_recovered_or_held_without_second_post(tmp_path, monkeypatch, found):
    rt, service, source = runtime(tmp_path)
    _inline(service)
    service.error = True
    with pytest.raises(RuntimeUnavailable, match='did not acknowledge'):
        rt.call('inspect', source, tmp_path / 'first', dispatch_key='4' * 24)
    assert (rt.root / 'dispatches' / ('4' * 24 + '.pending')).exists()
    monkeypatch.setattr(rt, 'find_dispatch', lambda *_: service.job() if found else None)
    if found:
        assert rt.call('inspect', source, tmp_path / 'again', dispatch_key='4' * 24)['galaxy_job_id'] == 'Galaxy-job'
    else:
        with pytest.raises(RuntimeUnavailable, match='No duplicate'):
            rt.call('inspect', source, tmp_path / 'again', dispatch_key='4' * 24)
    assert len(service.submissions) == 1 and not service.requests


@pytest.mark.parametrize('plan', [{'bad': float('nan')}, {'large': 'x' * (4 * 1024 * 1024)}], ids=['nonfinite', 'oversized'])
def test_invalid_inline_payload_never_claims_or_submits(tmp_path, plan):
    rt, service, source = runtime(tmp_path)
    _inline(service)
    with pytest.raises(ValueError):
        rt.call('inspect', source, tmp_path / 'output', plan=plan, dispatch_key='5' * 24)
    assert not service.submissions and not service.requests
    assert not list((rt.root / 'dispatches').iterdir())


def test_failed_capability_discovery_is_not_an_ambiguous_post(tmp_path):
    rt, service, source = runtime(tmp_path)
    def failed(*_):
        raise RuntimeError('tool input contract unavailable')
    service.tool_has_input = failed
    with pytest.raises(RuntimeError, match='input contract'):
        rt.call('inspect', source, tmp_path / 'output', dispatch_key='6' * 24)
    assert not service.submissions and not service.requests
    assert not list((rt.root / 'dispatches').iterdir())


def test_pause_during_capability_read_prevents_the_scientific_post(tmp_path):
    rt, service, source = runtime(tmp_path)
    cancelled = threading.Event()
    def capability(*_):
        cancelled.set()
        return True
    service.tool_has_input = capability
    with pytest.raises(RuntimeUnavailable, match='paused before dispatch'):
        rt.call('inspect', source, tmp_path / 'output', dispatch_key='7' * 24, cancel_event=cancelled)
    assert not service.submissions and not service.requests
    assert not list((rt.root / 'dispatches').iterdir())


@pytest.mark.parametrize('inline', [False, True])
def test_saved_remote_action_restores_without_new_request_or_post(tmp_path, inline):
    from radar_pd_nova.refinement_runtime import GalaxyRefinementRuntime

    rt, service, source = runtime(tmp_path)
    if inline:
        _inline(service)
    actions = []
    rt.call('inspect', source, tmp_path / 'first', dispatch_key='8' * 24,
            on_submitted=actions.append)
    service.tool_has_input = lambda *_: pytest.fail('restoring a saved job must not renegotiate transport')
    service.upload_json_document = lambda *a, **kw: pytest.fail('restoration must not upload another request')
    service.submit_utility = lambda **kw: pytest.fail('restoration must not submit another job')
    restored = GalaxyRefinementRuntime(tmp_path / 'restored', service, poll_seconds=0)

    result = restored.call('inspect', source, tmp_path / 'restored-output',
                           dispatch_key='8' * 24, resume=actions[0])

    assert result['source_sha256'] == source['source_sha256']
    assert result['galaxy_job_id'] == 'Galaxy-job'


@pytest.mark.parametrize('failed_checkpoint', [False, True])
def test_inline_dispatch_still_requires_claim_and_ack_checkpoints(tmp_path, failed_checkpoint):
    from radar_pd_nova.refinement_sources import RefinementScope
    from test_refinement_integration import approved, create, source as source_record

    rt, service, transport_source = runtime(tmp_path)
    capabilities = _inline(service)
    workspace, _ = create(tmp_path / 'scientific-workspace')
    source = source_record('1')
    source.update(transport_source, id='1')
    workspace.observe([source])
    recipe = approved(workspace, RefinementScope('exp'))
    workspace.start(recipe['id'], recipe['selection_token'])
    workspace.runtime = rt
    snapshots = []

    def checkpoint(document):
        if failed_checkpoint:
            raise RuntimeError('checkpoint not durable')
        snapshots.append(deepcopy(document))
        return 'checkpoint'

    rt.save_workspace = checkpoint
    with zipfile.ZipFile(service.archive, 'w') as archive:
        archive.writestr('response.json', json.dumps({
            'schema': 'radar-refinement-worker/v1', 'action': 'run', 'status': 'failed',
            'source_sha256': transport_source['source_sha256'], 'files': {},
        }))
    original_submit = service.submit_utility

    def submit(**kwargs):
        saved = snapshots[-1]['jobs'][kwargs['inputs']['dispatch_key']]
        assert saved['status'] == 'running' and saved['attempts'] == 1
        assert not saved.get('remote_action')
        return original_submit(**kwargs)

    service.submit_utility = submit
    result = workspace.run_next()
    if failed_checkpoint:
        assert not snapshots and not capabilities and not service.submissions
        assert 'checkpoint not durable' in result['message']
    else:
        assert len(service.submissions) == 1 and not service.requests
        assert len(snapshots) == 2
        assert snapshots[-1]['jobs'][result['id']]['remote_action']['galaxy_job_id'] == 'Galaxy-job'
        assert 'recipe_request_b64' not in json.dumps(snapshots)
