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
from gpx_recipe.symmetry.compiler import file_sha
from radar_pd_nova.model_runtime import GalaxyModelRuntime
from radar_pd_nova.models import RunStatus
from radar_pd_nova.native_operations import handle_path, validate_handle
from radar_pd_nova.refinement_runtime import RuntimeUnavailable
from test_model_runtime import remote


def test_simultaneous_viewers_share_one_submission_when_file_locks_are_process_scoped(remote,tmp_path,monkeypatch):
    import fcntl
    import time
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setattr(fcntl,'flock',lambda *args:None)
    runtime,service,source=remote
    root,payload=inputs(tmp_path,source)
    viewers=[GalaxyModelRuntime(runtime.root,service,poll_seconds=.001) for _ in range(4)]
    for viewer in viewers:
        monkeypatch.setattr(viewer,'find_dispatch',lambda key:service.jobs.get(key))
        monkeypatch.setattr(viewer,'wait_capacity',lambda cancel=None:None)
    original=service.submit_utility
    def slow_post(**kwargs):
        time.sleep(.01);return original(**kwargs)
    monkeypatch.setattr(service,'submit_utility',slow_post)
    barrier=threading.Barrier(4)
    def submit(viewer):
        barrier.wait(timeout=10)
        return viewer.submit_native('inspect',payload,root,source=source)
    with ThreadPoolExecutor(max_workers=4) as pool:handles=list(pool.map(submit,viewers))
    assert len({h['key'] for h in handles})==1 and len(service.submissions)==1
    assert len({h['job']['galaxy_job_id'] for h in handles})==1


def test_structural_training_capture_includes_all_declared_blocks_without_loading_arrays(tmp_path):
    from types import SimpleNamespace
    from radar_pd_nova.native_operations import capture
    from gpx_recipe.preparation.structural_jobs import PLAN_SCHEMA
    from gpx_recipe.symmetry.compiler import file_sha
    root=tmp_path/'training';root.mkdir()
    source=root/'parent.cif';source.write_text('data_public_contract\n')
    instrument=root/'profile.instprm';instrument.write_text('# transport profile\n')
    directory=root/'blocks/one';directory.mkdir(parents=True)
    # Opaque transport bytes: numerical validation belongs to the worker,
    # never a full-array load on the interactive application's capture path.
    (directory/'examples.npz').write_bytes(b'opaque transport control')
    (directory/'examples.json').write_text(json.dumps(dict(schema='radar-structural-ranking-examples/v3',source_sha256=file_sha(source),
        instrument_sha256=file_sha(instrument),grid=dict(kind='PNT',lower=3000.,upper=40000.,bins=701,bank=1))))
    plan=dict(schema=PLAN_SCHEMA,parent_sha256=file_sha(source),instrument_sha256=file_sha(instrument),
        blocks=[dict(directory='blocks/one',manifest_sha256=file_sha(directory/'examples.json'),
            examples_sha256=file_sha(directory/'examples.npz'))])
    (root/'plan.json').write_text(json.dumps(plan))
    descriptor=dict(id='parent-control',source_sha256=file_sha(source),dataset_id='parent-dataset')
    runtime=SimpleNamespace(service=SimpleNamespace(history_id='history-control'),
        resolve=lambda value:descriptor,dataset_id=lambda value:value['dataset_id'])
    payload=dict(source=str(source),instrument=str(instrument),
        grid=dict(kind='PNT',lower=3000.,upper=40000.,bins=701,bank=1),
        plan=str(root/'plan.json'),destination=str(root/'result'),settings={})
    handle,files=capture(runtime,'preparation-structural-training',payload,root,descriptor,600)
    assert set(files)=={'parent.cif','profile.instprm','plan.json','blocks/one/examples.json','blocks/one/examples.npz'}
    assert set(handle['request']['input_hashes'])==set(files)
    long_handle,_=capture(runtime,'preparation-structural-training',payload,root,descriptor,7200)
    assert long_handle['request']['timeout']==7200
    validate_handle(runtime,long_handle)
    changed=copy.deepcopy(payload);changed['grid']['bank']=2
    with pytest.raises(ValueError,match='physical bank grid'):
        capture(runtime,'preparation-structural-training',changed,root,descriptor,7200)
    changed=copy.deepcopy(payload);changed['settings']['representation']='reference-current-signed-difference/shared-Q-asinh/v1'
    with pytest.raises(ValueError,match='representation differs'):
        capture(runtime,'preparation-structural-training',changed,root,descriptor,7200)
    descriptor['source_sha256']='f'*64
    with pytest.raises(ValueError,match='another Galaxy source'):
        capture(runtime,'preparation-structural-training',payload,root,descriptor,600)


def inputs(tmp_path, source):
    root = tmp_path/'native-work'
    root.mkdir()
    shutil.copyfile(source['source_path'], root/'source.gpx')
    return root, dict(source=str(root/'source.gpx'))


def test_paired_training_capture_uses_typed_plan_without_loading_native_arrays(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from test_structural_paired_preparation import inputs as paired_inputs
    from gpx_recipe.preparation import structural_jobs
    from radar_pd_nova.native_operations import capture
    root=tmp_path/'paired-capture';example,directory=paired_inputs(root)
    structural_jobs.create_plan(example['source'],[directory],root/'plan.json')
    descriptor=dict(id='paired-parent',source_sha256=file_sha(root/'parent.cif'),dataset_id='paired-dataset')
    runtime=SimpleNamespace(service=SimpleNamespace(history_id='paired-history'),
        resolve=lambda value:descriptor,dataset_id=lambda value:value['dataset_id'])
    payload=dict(source=example['source'],instrument=example['instrument'],grid=example['grid'],
        plan=str(root/'plan.json'),destination=str(root/'trained'),
        settings=dict(representation='reference-current-signed-difference/shared-Q-asinh/v1'))
    monkeypatch.setattr(structural_jobs,'read_training_block',lambda *a:pytest.fail('Interactive capture must not load raw arrays'))
    handle,files=capture(runtime,'preparation-structural-training',payload,root,descriptor,7200)
    assert handle['request']['timeout']==7200
    assert set(files)=={'parent.cif','profile.instprm','plan.json','examples/examples.json','examples/examples.npz'}
    validate_handle(runtime,handle)


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


def test_batch_transport_deduplicates_inputs_and_captures_typed_plan(remote,tmp_path):
    import io
    from gpx_recipe.preparation.batch import stage
    from test_preparation_queue import setup
    queue,_,_=setup(remote,tmp_path,count=2);runtime,service,_=remote
    entries=[(r['id'],queue.library.start(r['id'])) for r in list(queue.library.document['entries'])]
    payload=stage(queue.library,entries,queue.root/'batch',workers=4,seconds=60)
    # This test checks packet capture only; the service's fake native job does
    # not construct real diffraction arrays or provide numerical validation.
    service.submit_utility=type(service).submit_utility.__get__(service)
    handle=runtime.submit_native('preparation-batch',payload,queue.root,source=queue.source,timeout=120)
    assert len(handle['request']['input_hashes'])==3
    packet=service.data[handle['asset_id']]
    with zipfile.ZipFile(io.BytesIO(packet)) as archive:
        assert len(archive.namelist())==3
        assert 'batch/plan.json' in archive.namelist()
        assert len([p for p in archive.namelist() if p.endswith('.cif')])==1
        assert len([p for p in archive.namelist() if p.endswith('.instprm')])==1
    assert len(service.submissions)==1 and b'PRIVATE-' not in packet
    destination=tmp_path/'new-viewer-inputs'
    restored=runtime.restore_native_inputs(handle,destination)
    assert set(restored)==set(handle['request']['input_hashes'])
    assert (destination/'batch/plan.json').read_bytes()==Path(payload['plan']).read_bytes()
    assert len(service.submissions)==1


def test_input_recovery_verifies_packet_and_preserves_existing_files(remote,tmp_path):
    runtime,service,source=remote
    root,payload=inputs(tmp_path,source)
    handle=runtime.submit_native('inspect',payload,root,source=source)
    # The thin GPX packet reconstructs its declared source from the separately
    # captured dataset, with the same marker and checksums as the worker.
    service.data['original-data']=Path(source['source_path']).read_bytes()
    destination=tmp_path/'restored-inputs'
    restored=runtime.restore_native_inputs(handle,destination)
    assert (destination/'source.gpx').read_bytes()==Path(source['source_path']).read_bytes()
    assert set(restored)==set(handle['request']['input_hashes'])
    (destination/'source.gpx').write_bytes(b'changed')
    with pytest.raises(ValueError,match='immutable native attempt'):
        runtime.restore_native_inputs(handle,destination)
    assert (destination/'source.gpx').read_bytes()==b'changed'
    assert len(service.submissions)==1


def test_packet_byte_budget_is_checked_before_upload(remote,tmp_path,monkeypatch):
    from gpx_recipe.symmetry import bundle
    runtime,service,source=remote
    monkeypatch.setattr(bundle,'MAX_BYTES',2)
    with pytest.raises(ValueError,match='packet byte/file budget'):
        runtime.packet(tmp_path,['source.gpx'],source_sha256=source['source_sha256'])
    assert not service.data and not service.submissions
