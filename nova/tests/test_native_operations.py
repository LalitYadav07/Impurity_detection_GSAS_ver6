"""Durable native operations across queued work and viewer loss."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import threading
import zipfile

import pytest

from gpx_recipe.symmetry.contracts import digest
from radar_pd_nova.model_runtime import GalaxyModelRuntime
from radar_pd_nova.models import RunStatus
from radar_pd_nova.native_operations import handle_path, validate_handle
from radar_pd_nova.refinement_runtime import RuntimeUnavailable
from test_model_runtime import remote


def inputs(tmp_path, source):
    root = tmp_path/'native-work'
    root.mkdir()
    shutil.copyfile(source['source_path'], root/'source.gpx')
    return root, dict(source=str(root/'source.gpx'))


def reply(service, tmp_path, handle, *, files=None, seconds=1.25, mutate=None):
    files = files or {'native/nested/value.json': b'{"checked":true}'}
    response = dict(schema='radar-model-native-result/v1', dispatch_key=handle['key'],
        request_sha256=digest(handle['request']), source_sha256=handle['source']['source_sha256'],
        worker_root='/worker/native', input_root='/worker/inputs-packet',
        result=dict(path='/worker/native/nested/value.json'),
        files={name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        native_seconds=seconds)
    if mutate:
        mutate(response)
    archive = tmp_path/'reply.zip'
    with zipfile.ZipFile(archive, 'w') as output:
        output.writestr('response.json', json.dumps(response))
        for name, data in files.items():
            output.writestr(name, data)
    job = service.jobs[handle['key']]
    job.outputs = {'result_bundle': service._put(archive.read_bytes(), 'native results')}
    job.status = RunStatus.OK


def test_queued_submit_returns_durable_handle_and_collects_after_viewer_loss(remote, tmp_path, monkeypatch):
    runtime, service, source = remote
    root, payload = inputs(tmp_path, source)
    submit = service.submit_utility
    def queued(**kwargs):
        job = submit(**kwargs)
        job.status = RunStatus.QUEUED
        return job
    monkeypatch.setattr(service, 'submit_utility', queued)
    monkeypatch.setattr(runtime, 'wait_job', lambda *a, **k: pytest.fail('Async submission must not wait'))
    intents = []
    def checkpoint(value):
        assert not service.submissions
        assert value['submission_started'] and value['job'] is None
        assert json.loads(handle_path(root, value['key']).read_text()) == value
        intents.append(value)
    handle = runtime.submit_native('inspect', payload, root, source=source, on_intent=checkpoint)
    assert len(intents) == len(service.submissions) == 1
    assert runtime.poll_native(handle)['state'] == 'queued'
    assert 'source_path' not in handle['source'] and 'inputs' not in handle['job']
    assert 'PRIVATE-' not in json.dumps(handle)
    # Both the dispatch ledger and the original local files may be gone. The
    # captured handle is sufficient to authenticate and collect the old job.
    fresh = GalaxyModelRuntime(tmp_path/'another-runtime', service)
    monkeypatch.setattr(fresh, 'find_dispatch', lambda key: pytest.fail('Acknowledged job needs no name search'))
    reply(service, tmp_path, handle)
    destination = tmp_path/'restored-attempt'
    result, seconds = fresh.collect_native(handle, destination)
    assert seconds == 1.25
    assert result['path'] == str(destination/'nested/value.json')
    assert json.loads(Path(result['path']).read_text()) == {'checked': True}
    assert len(service.submissions) == 1
    # Collection is idempotent even if its checkpoint acknowledgment was lost.
    assert fresh.collect_native(handle, destination) == (result, seconds)


def test_lost_submission_ack_cannot_become_a_duplicate_in_a_new_viewer(remote, tmp_path, monkeypatch):
    runtime, service, source = remote
    root, payload = inputs(tmp_path, source)
    service.lose_ack = True
    with pytest.raises(RuntimeUnavailable, match='acknowledge'):
        runtime.submit_native('inspect', payload, root, source=source)
    saved = json.loads(next((root/'native-dispatches').glob('*.json')).read_text())
    assert saved['submission_started'] and saved['job'] is None
    fresh = GalaxyModelRuntime(tmp_path/'fresh-runtime', service)
    monkeypatch.setattr(fresh, 'find_dispatch', lambda key: None)
    assert fresh.poll_native(saved)['state'] == 'needs_recovery'
    assert fresh.submit_native('inspect', payload, root, source=source) == saved
    assert len(service.submissions) == 1
    monkeypatch.setattr(fresh, 'find_dispatch', lambda key: service.jobs.get(key))
    recovered = fresh.poll_native(saved)
    assert recovered['state'] == 'finished' and recovered['handle']['job']
    assert len(service.submissions) == 1


def test_failed_intent_checkpoint_does_not_post_and_same_attempt_can_resume(remote, tmp_path):
    runtime, service, source = remote
    root, payload = inputs(tmp_path, source)
    def unavailable(_):
        raise TimeoutError('Checkpoint did not reach durable storage')
    with pytest.raises(TimeoutError):
        runtime.submit_native('inspect', payload, root, source=source, on_intent=unavailable)
    assert not service.submissions
    saved = json.loads(next((root/'native-dispatches').glob('*.json')).read_text())
    assert not saved['submission_started']
    handle = runtime.submit_native('inspect', payload, root, source=source)
    assert handle['key'] == saved['key'] and len(service.submissions) == 1


@pytest.mark.parametrize('status,expected', [(RunStatus.ERROR, 'failed'), (RunStatus.CANCELLED, 'cancelled')])
def test_poll_terminal_failure_never_resubmits_or_collects(remote, tmp_path, status, expected):
    runtime, service, source = remote
    root, payload = inputs(tmp_path, source)
    handle = runtime.submit_native('inspect', payload, root, source=source)
    service.jobs[handle['key']].status = status
    assert runtime.poll_native(handle)['state'] == expected
    with pytest.raises(RuntimeUnavailable, match='completes successfully'):
        runtime.collect_native(handle, root)
    assert len(service.submissions) == 1


def test_immutable_collection_rejects_changed_file_before_copying_other_outputs(remote, tmp_path):
    runtime, service, source = remote
    root, payload = inputs(tmp_path, source)
    handle = runtime.submit_native('inspect', payload, root, source=source)
    files = {'native/new.json': b'new', 'native/old.json': b'old'}
    reply(service, tmp_path, handle, files=files)
    (root/'old.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='immutable attempt'):
        runtime.collect_native(handle, root)
    assert not (root/'new.json').exists()
    assert (root/'old.json').read_bytes() == b'changed'


def test_relocated_output_collisions_are_rejected(remote, tmp_path):
    runtime, service, source = remote
    root, payload = inputs(tmp_path, source)
    handle = runtime.submit_native('inspect', payload, root, source=source)
    reply(service, tmp_path, handle, files={'native/result.json': b'first', 'inputs-packet/result.json': b'second'})
    with pytest.raises(ValueError, match='collide'):
        runtime.collect_native(handle, root)
    assert not (root/'result.json').exists()


@pytest.mark.parametrize('mutation', ['history', 'source', 'extra_source', 'extra_request', 'job_inputs'])
def test_handle_rejects_foreign_or_non_scientific_context(remote, tmp_path, mutation):
    runtime, _, source = remote
    root, payload = inputs(tmp_path, source)
    handle = copy.deepcopy(runtime.submit_native('inspect', payload, root, source=source))
    if mutation == 'history':
        handle['history_id'] = 'other-history'
    elif mutation == 'source':
        handle['source']['source_sha256'] = '0'*64
    elif mutation == 'extra_source':
        handle['source']['environment'] = {'key': 'private'}
    elif mutation == 'extra_request':
        handle['request']['environment'] = {'key': 'private'}
    else:
        handle['job']['inputs'] = {'unreviewed': 'private'}
    handle['identity'] = digest(dict(history_id=handle['history_id'], source=handle['source'], request=handle['request']))
    with pytest.raises(ValueError):
        validate_handle(runtime, handle)


def test_pre_dispatch_cancellation_posts_nothing(remote, tmp_path):
    runtime, service, source = remote
    root, payload = inputs(tmp_path, source)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RuntimeUnavailable, match='before dispatch'):
        runtime.submit_native('inspect', payload, root, source=source, cancel=cancel)
    assert not service.submissions


def test_fresh_runtime_revalidates_saved_job_owner(remote, tmp_path):
    runtime, service, source = remote
    root, payload = inputs(tmp_path, source)
    handle = runtime.submit_native('inspect', payload, root, source=source)
    fresh = GalaxyModelRuntime(tmp_path/'new-runtime', service)
    service._job_details = lambda uid: dict(id=uid, history_id='someone-else', tool_id=fresh.tool_id,
        params={'dispatch_key': handle['key']})
    with pytest.raises(RuntimeUnavailable, match='does not belong'):
        fresh.poll_native(handle)
