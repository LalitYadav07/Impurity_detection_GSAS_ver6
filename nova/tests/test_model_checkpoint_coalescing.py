"""Checkpoint coalescing keeps exact remote intent and final recovery evidence."""
import asyncio
import copy
import gzip
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

import pytest

from gpx_recipe.symmetry.campaign import Campaign
from gpx_recipe.symmetry.compiler import file_sha, package_signature
from gpx_recipe.symmetry.contracts import digest, validate_protocol
from gpx_recipe.symmetry.templates import make_template
from radar_pd_nova.checkpoint_uploads import CheckpointUploadPending, CheckpointUploads
from radar_pd_nova.model_campaigns import restore_history
from radar_pd_nova.model_runtime import GalaxyChunkTransport, GalaxyModelRuntime
from radar_pd_nova.models import RunStatus
from radar_pd_nova.refinement_runtime import RuntimeUnavailable
from radar_pd_nova.symmetry_controller import SymmetryMixin

from test_model_runtime import remote


def running_campaign(remote, tmp_path, monkeypatch):
    """Real ledger, compilation, ZIP ingestion and transport; no native fits."""
    import radar_pd_nova.symmetry_controller as controller
    runtime, service, source = remote
    runtime.max_active_jobs = 8
    service.tool_has_input = lambda *args: True
    source.update(experiment='IPTS-1', sample_key='specimen:A', bank='1')
    service.data['original-data'] = Path(source['source_path']).read_bytes()
    inputs = tmp_path/'inputs'
    inputs.mkdir()
    shutil.copyfile(source['source_path'], inputs/'source.gpx')
    candidate = dict(id='mag:fixture', magnetic_space_group="Pn'ma'",
                     transformation_matrix=[[1,0,0],[0,1,0],[0,0,1]], origin_shift=[0,0,0])
    (inputs/'catalog.json').write_text(json.dumps(dict(schema='radar-symmetry-catalog/v1',
        parent_setting_sha256='a'*64, k=['0','0','0'], candidates=[candidate])))
    protocol = validate_protocol(dict(kind='magnetic', acquisition_id=source['id'],
        source=dict(path='source.gpx', sha256=file_sha(inputs/'source.gpx')),
        catalog=dict(path='catalog.json', sha256=file_sha(inputs/'catalog.json')),
        parent_phase='Parent', histogram='Bank', parent_setting_sha256='a'*64,
        magnetic_function_sha256='b'*64, implementation_sha256=package_signature(),
        k=['0','0','0'], candidates=[candidate],
        seeds=[dict(id='start'+str(i), axis='x', magnitude=3.) for i in range(12)],
        sites=[dict(label='Mn', form_factor='Mn', lande_g=2.)],
        policy=dict(cycles=12, timeout_seconds=60, maximum_trials=12,
                    maximum_cell_multiplier=8, maximum_moment=6., fit_band_percent=.25,
                    maximum_mode=.5, adequate_rwp_percent=12.)))
    campaign = Campaign(tmp_path/'original/campaign-aaaaaaaaaaaa', shared_inputs=True)
    transport = GalaxyChunkTransport(runtime, campaign)
    transport.prepare_source(protocol, inputs, source)
    campaign.enroll(protocol, inputs, approved_hash=digest(protocol))
    campaign.set_setting('template', make_template(protocol, dict(sha256=source['source_sha256']),
        source, [(source,'radar')], origin='radar'))
    campaign.set_setting('enrollments', {source['id']:dict(id=source['id'], status='enrolled', origin='radar',
        source_sha256=source['source_sha256'], protocol=digest(protocol))})
    # Small batches expose multiple dispatch waves without running any science.
    monkeypatch.setattr(transport, 'chunk_size_for', lambda protocol: 1)
    checkpoints, post_snapshots = [], []
    original_save = transport.checkpointer.save_workspace
    def save(document):
        result = original_save(document)
        checkpoints.append(copy.deepcopy(document))
        return result
    monkeypatch.setattr(transport.checkpointer, 'save_workspace', save)
    original_submit = service.submit_utility
    def submit(**kwargs):
        key = kwargs['inputs']['dispatch_key']
        saved = checkpoints[-1]['transport']['chunks'][key]
        assert saved['key'] == key and saved['job'] is None
        assert saved['members'] == transport.state['chunks'][key]['members']
        post_snapshots.append(copy.deepcopy(checkpoints[-1]))
        job = original_submit(**kwargs)
        job.status = RunStatus.RUNNING
        chunk = transport.state['chunks'][key]
        response = dict(schema='radar-model-chunk-result/v1', dispatch_key=key,
            protocol_sha256=chunk['protocol'], pending=[], allocation=dict(workers=2),
            trials=[dict(id=uid, state='finished', status='unsupported') for uid in chunk['members']])
        archive = tmp_path/(key+'.zip')
        with zipfile.ZipFile(archive, 'w') as output:
            output.writestr('response.json', json.dumps(response))
            for uid in chunk['members']:
                output.writestr('trials/'+uid+'/result.json', json.dumps(dict(
                    schema='radar-symmetry-trial/v1', id=uid, protocol_sha256=chunk['protocol'],
                    status='unsupported', reason='Synthetic transport test; no native fit')))
        job.outputs = {'result_bundle':service._put(archive.read_bytes(), 'model results')}
        return job
    monkeypatch.setattr(service, 'submit_utility', submit)
    app = SymmetryMixin()
    app._model_active_campaigns = {}
    app._model_stopping = False
    app._refinement_store = lambda: None
    app.refresh_symmetry = lambda: None
    app.server = SimpleNamespace(state=SimpleNamespace(sx_slots=2, sx_parallel_jobs=8,
                                                       flush=lambda: None))
    monkeypatch.setattr(controller, 'queue_sources', lambda *args: campaign.setting('enrollments'))
    original_sleep = asyncio.sleep
    sleeps = []
    async def step(seconds):
        sleeps.append(seconds)
        for job in service.jobs.values():
            job.status = RunStatus.OK
        await original_sleep(0)
    monkeypatch.setattr(controller.asyncio, 'sleep', step)
    return SimpleNamespace(app=app, campaign=campaign, transport=transport, service=service,
        source=source, checkpoints=checkpoints, post_snapshots=post_snapshots, sleeps=sleeps,
        step=step)


def test_interim_ack_backup_is_coalesced_with_next_ready_intent(remote, tmp_path, monkeypatch):
    fixture = running_campaign(remote, tmp_path, monkeypatch)
    asyncio.run(fixture.app._drive_symmetry_campaign(fixture.campaign, fixture.transport))
    assert len(fixture.service.submissions) == len(fixture.post_snapshots) == 12
    assert len({row['inputs']['dispatch_key'] for row in fixture.service.submissions}) == 12
    first, second = fixture.post_snapshots[0], fixture.post_snapshots[8]
    assert len(first['transport']['chunks']) == 8
    assert all(not chunk['job'] for chunk in first['transport']['chunks'].values())
    assert len(second['transport']['chunks']) == 12
    assert sum(bool(chunk['job']) for chunk in second['transport']['chunks'].values()) == 8
    # No separate eight-ACK backup can block admission of the second wave.
    assert fixture.checkpoints[:2] == [first, second]
    final = fixture.checkpoints[-1]
    assert all(row['state'] == 'terminal' for row in final['ledger']['trials'])
    assert len(final['transport']['receipt_hashes']) == 12
    assert all(row['state'] == 'terminal' for row in fixture.campaign.trials())


@pytest.mark.parametrize('view_other_campaign', [False, True])
def test_background_driver_does_not_restart_another_campaign_review(remote, tmp_path, monkeypatch, view_other_campaign):
    fixture = running_campaign(remote, tmp_path, monkeypatch)
    selected = 'saved-campaign-under-review' if view_other_campaign else fixture.campaign.root.name
    fixture.app.server.state.sx_campaign = selected
    refreshes = []
    fixture.app.refresh_symmetry = lambda: refreshes.append(selected)

    asyncio.run(fixture.app._drive_symmetry_campaign(fixture.campaign, fixture.transport))

    # Navigation must not pause/cancel the independent scientific driver or
    # prevent its exact recovery checkpoints from being published.
    assert len(fixture.service.submissions) == 12
    assert all(row['state'] == 'terminal' for row in fixture.campaign.trials())
    assert len(fixture.checkpoints[-1]['transport']['receipt_hashes']) == 12
    assert bool(refreshes) is not view_other_campaign
    assert fixture.app.server.state.sx_campaign == selected


@pytest.mark.parametrize('lost_ack', [False, True])
def test_lost_viewer_without_interim_backup_recovers_keys_and_never_reposts(remote, tmp_path, monkeypatch, lost_ack):
    import radar_pd_nova.symmetry_controller as controller
    fixture = running_campaign(remote, tmp_path, monkeypatch)
    if lost_ack:
        submit = fixture.service.submit_utility
        def lose_ninth(**kwargs):
            job = submit(**kwargs)
            if len(fixture.service.submissions) == 9:
                raise TimeoutError('Lost viewer after ninth accepted POST')
            return job
        monkeypatch.setattr(fixture.service, 'submit_utility', lose_ninth)
    else:
        async def lose_before_next_intent(seconds):
            raise RuntimeError('Lost viewer before the next intent backup')
        monkeypatch.setattr(controller.asyncio, 'sleep', lose_before_next_intent)
    with pytest.raises((RuntimeError, RuntimeUnavailable)):
        asyncio.run(fixture.app._drive_symmetry_campaign(fixture.campaign, fixture.transport))
    expected = 9 if lost_ack else 8
    assert len(fixture.service.submissions) == expected
    # A new process has none of the original viewer's dispatch/ACK caches.
    recovered_runtime = GalaxyModelRuntime(tmp_path/'new-runtime', fixture.service)
    monkeypatch.setattr(recovered_runtime, 'find_dispatch', lambda key:fixture.service.jobs.get(key))
    restore_history(tmp_path/'new-viewer', recovered_runtime, experiment='IPTS-1')
    recovered = Campaign(tmp_path/'new-viewer/campaign-aaaaaaaaaaaa', shared_inputs=True)
    transport = GalaxyChunkTransport(recovered_runtime, recovered)
    for job in fixture.service.jobs.values():job.status = RunStatus.OK
    recovered.recover(transport)
    assert len(fixture.service.submissions) == expected
    assert sum(row['state'] == 'terminal' for row in recovered.trials()) == expected
    # The never-attempted suffix of a lost wave lacks remote absence proof.
    # It stays held; it must not be confused with a new finite wave.
    if lost_ack:
        assert sum(row['state'] == 'running' for row in recovered.trials()) == 3
    else:
        assert sum(row['state'] == 'prepared' for row in recovered.trials()) == 4
    assert len(recovered.trials()) == 12


def test_stop_between_waves_flushes_backup_and_leaves_unsubmitted_starts(remote, tmp_path, monkeypatch):
    import radar_pd_nova.symmetry_controller as controller
    fixture = running_campaign(remote, tmp_path, monkeypatch)
    async def stop(seconds):
        fixture.app._model_stopping = True
        await fixture.step(seconds)
    monkeypatch.setattr(controller.asyncio, 'sleep', stop)
    asyncio.run(fixture.app._drive_symmetry_campaign(fixture.campaign, fixture.transport))
    assert len(fixture.service.submissions) == 8
    final = fixture.checkpoints[-1]
    assert final['ledger']['settings']['paused'] is True
    assert sum(row['state'] == 'terminal' for row in final['ledger']['trials']) == 8
    assert sum(row['state'] == 'prepared' for row in final['ledger']['trials']) == 4


def test_final_pending_checkpoint_cannot_be_reported_as_finished(remote, tmp_path, monkeypatch):
    fixture = running_campaign(remote, tmp_path, monkeypatch)
    save = fixture.transport.checkpointer.save_workspace
    pending = []
    def final_pending(document):
        if not pending and all(row['state'] == 'terminal' for row in document['ledger']['trials']):
            pending.append(copy.deepcopy(document))
            raise CheckpointUploadPending('final-owned-hda', 'queued')
        return save(document)
    monkeypatch.setattr(fixture.transport.checkpointer, 'save_workspace', final_pending)
    asyncio.run(fixture.app._drive_symmetry_campaign(fixture.campaign, fixture.transport))
    assert len(pending) == 1
    assert len(fixture.service.submissions) == 12
    assert all(row['state'] == 'terminal' for row in fixture.checkpoints[-1]['ledger']['trials'])
    assert fixture.campaign.root.name not in fixture.app._model_checkpoint_errors
    assert not fixture.campaign.setting('checkpoint_notice')


def test_real_pending_final_upload_keeps_same_hda_until_ready(remote, tmp_path, monkeypatch):
    import radar_pd_nova.symmetry_controller as controller
    fixture = running_campaign(remote, tmp_path, monkeypatch)
    clock, uploads, states, pending = [0.], [], {}, []
    def upload(path, *, label):
        document = json.loads(gzip.decompress(Path(path).read_bytes()))
        key = fixture.service.upload_document(path, label=label)
        uploads.append(key)
        is_final = all(row['state'] == 'terminal' for row in document['ledger']['trials'])
        states[key] = 'queued' if is_final and not pending else 'ok'
        if states[key] == 'queued':pending.append(key)
        return key
    queue = CheckpointUploads(upload, lambda key, seconds:dict(id=key, state=states[key]),
        clock=lambda:clock[0], sleep=lambda seconds:clock.__setitem__(0, clock[0]+seconds))
    def save(path, **kwargs):return queue.save(path, timeout=.01, **kwargs)
    monkeypatch.setattr(fixture.service, 'upload_checkpoint_document', save, raising=False)
    async def release_final(seconds):
        if pending and states[pending[0]] == 'queued':
            assert fixture.campaign.root.name in fixture.app._model_active_campaigns
            assert fixture.campaign.root.name in fixture.app._model_checkpoint_errors
            assert uploads[-1] == pending[0]
            assert len(fixture.service.submissions) == 12
            states[pending[0]] = 'ok'
        await fixture.step(seconds)
    monkeypatch.setattr(controller.asyncio, 'sleep', release_final)
    asyncio.run(fixture.app._drive_symmetry_campaign(fixture.campaign, fixture.transport))
    assert len(pending) == 1 and uploads.count(pending[0]) == 1
    assert not fixture.app._model_checkpoint_errors
    assert len(fixture.service.submissions) == 12


def test_stop_during_coalesced_intent_wait_prevents_every_new_post(remote, tmp_path, monkeypatch):
    fixture = running_campaign(remote, tmp_path, monkeypatch)
    save = fixture.transport.checkpointer.save_workspace
    def stop_at_second_intent(document):
        if len(document['transport']['chunks']) == 12 and any(
                chunk['job'] is None for chunk in document['transport']['chunks'].values()):
            fixture.app._model_stopping = True
        return save(document)
    monkeypatch.setattr(fixture.transport.checkpointer, 'save_workspace', stop_at_second_intent)
    asyncio.run(fixture.app._drive_symmetry_campaign(fixture.campaign, fixture.transport))
    assert len(fixture.service.submissions) == 8
    assert fixture.checkpoints[-1]['ledger']['settings']['paused'] is True
    assert sum(row['state'] == 'prepared' for row in fixture.campaign.trials()) == 4
    unposted = [chunk for chunk in fixture.transport.state['chunks'].values() if chunk.get('not_dispatched')]
    assert len(unposted) == 4 and all(chunk['job'] is None for chunk in unposted)


@pytest.mark.parametrize('state', ['prepared', 'queued', 'preparing', 'task', 'enrolled', 'terminal'])
@pytest.mark.parametrize('paused', [False, True])
def test_checkpoint_is_deferred_only_for_known_approved_work(tmp_path, state, paused):
    campaign = Campaign(tmp_path/'decision')
    campaign.set_setting('paused', paused)
    campaign.set_setting('enrollments', {'scan':dict(status=state)})
    campaign.trials = lambda:[dict(state=state)]
    app = SymmetryMixin()
    app._model_stopping = False
    preparation = object() if state == 'task' else None
    assert app._model_checkpoint_due(campaign, preparation) is (paused or state in {'enrolled','terminal'})
    app._model_stopping = True
    assert app._model_checkpoint_due(campaign, preparation) is True
