"""Performance paths must preserve dependencies, provenance, and recovery."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from types import SimpleNamespace
import threading
import time

import pytest

from radar_pd_nova.galaxy_service import GalaxyService
from radar_pd_nova.models import AnalysisConfig, InputSelection, RunRecord, RunStatus, UtilityActionRecord


def service(tmp_path):
    result = GalaxyService('https://galaxy.example', 'key', 'history', output_root=tmp_path)
    @contextmanager
    def store():
        yield SimpleNamespace(history_id='history')
    result._store = store
    return result


def test_scientific_inputs_queue_but_documents_wait_for_durable_storage(tmp_path):
    s = service(tmp_path); calls = []
    def upload(path, store, label, *, wait_ready=True):
        calls.append((label, wait_ready)); return SimpleNamespace(id=label)
    s._upload_dataset = upload
    assert s._upload_one('data', 'pattern.gsa', 'pattern') == ('data', 'pattern')
    assert s.upload_document('state.json', label='checkpoint') == 'checkpoint'
    assert s.upload_json_document({'revision': 1}, name='request.json', label='request', wait_ready=False) == 'request'
    assert calls == [('pattern', False), ('checkpoint', True), ('request', False)]
    assert not list((tmp_path / 'provenance').iterdir())


def test_remote_input_import_queues_dependency_without_client_wait(tmp_path):
    s = service(tmp_path); calls = []
    s._import_remote_dataset = lambda uri, label, **kw: calls.append(kw) or 'pending-hda'
    assert s._import_one('data', 'gxfiles://source/file', 'pattern') == ('data', 'pending-hda')
    assert calls == [{'wait_ready': False}]


def test_pending_upload_returns_native_hda_without_readiness_wait(tmp_path, monkeypatch):
    from radar_pd_nova import galaxy_service
    import nova.galaxy
    source = tmp_path / 'pattern.gsa'; source.write_text('pending pattern')
    waited = []
    store = SimpleNamespace(history_id='history', nova_connection=SimpleNamespace(galaxy_instance=SimpleNamespace(
        tools=SimpleNamespace(upload_file=lambda **kw: {'outputs': [{'id': 'pending'}]}),
        datasets=SimpleNamespace(wait_for_dataset=lambda identifier: waited.append(identifier)))))
    dataset = galaxy_service.GalaxyService._upload_dataset(str(source), store, 'diffraction data', wait_ready=False)
    assert dataset.id == 'pending' and waited == []
    galaxy_service.GalaxyService._upload_dataset(str(source), store, 'checkpoint')
    assert waited == ['pending']


def test_analysis_acknowledgement_does_not_start_a_second_polling_thread(tmp_path, monkeypatch):
    import nova.galaxy
    s = service(tmp_path); posted = []; assigned = []
    class Tool:
        def __init__(self, id): pass
        def run(self, *a, **kw): pytest.fail('A duplicate background polling thread was started')
        def assign_id(self, identifier, store): assigned.append(identifier)
    monkeypatch.setattr(nova.galaxy, 'Tool', Tool)
    def post(url, **kwargs):
        posted.append(kwargs)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'jobs': [{'id':'accepted'}]})
    monkeypatch.setattr('radar_pd_nova.galaxy_service.requests.post', post)
    params = SimpleNamespace(inputs={'pattern': {'src':'hda','id':'pending'}})
    identifier, _ = s._submit_analysis(object(), params)
    assert identifier == 'accepted' and assigned == ['accepted']
    assert posted[0]['json']['inputs'] == params.inputs
    assert posted[0]['json']['history_id'] == 'history'
    assert posted[0]['timeout'] > 0
    assert 'key' not in posted[0]['json']


def test_all_prepared_analysis_inputs_are_native_dependencies(tmp_path, monkeypatch):
    import nova.galaxy
    class Parameters:
        def __init__(self): self.values = {}
        def add_input(self, name, value): self.values[name] = value
    monkeypatch.setattr(nova.galaxy, 'Parameters', Parameters)
    s = service(tmp_path)
    config = AnalysisConfig(sample_elements=['Al','Mg','O'])
    inputs = InputSelection(data_dataset_id='data', instrument_dataset_id='profile', main_cif_dataset_id='main', database_dataset_id='library')
    snapshot = s.create_submission_snapshot(config, inputs)
    prepared = dict(configuration='config', data='data', instrument='profile', main_cif='main', database_archive='library')
    params, _ = s._prepared_parameters(object(), snapshot, prepared)
    refs = [value for value in params.values.values() if isinstance(value, dict)]
    assert sorted(item['id'] for item in refs) == sorted(prepared.values())
    assert all(item['src'] == 'hda' for item in refs)


def test_polling_keeps_submitted_config_and_delivery_without_downloading_them(tmp_path):
    s = service(tmp_path); reads = []
    s._job_details = lambda _: dict(state='running', params={'config_file': {'id':'config'}}, inputs={
        'reproducibility|configuration_override|config_file': {'id':'config', 'src':'hda'}})
    s._dataset_document = lambda identifier: reads.append(identifier) or {}
    record = RunRecord(uid='job', name='scan', mode='full', history_id='history',
        config=AnalysisConfig(sample_elements=['Al'], full_max_passes=7), idempotency_token='submitted',
        inputs=InputSelection(data_dataset_id='data', instrument_dataset_id='profile'))
    for _ in range(4): s.refresh(record)
    assert reads == [] and record.config.full_max_passes == 7


def test_final_configuration_read_is_once_and_failures_remain_retryable(tmp_path):
    s = service(tmp_path); reads = []
    s._job_details = lambda _: dict(state='ok', outputs={'resolved_config': {'id':'resolved'}})
    config = AnalysisConfig(sample_elements=['Al'], full_max_passes=7, run_name='scan')
    def document(identifier):
        reads.append(identifier)
        if len(reads) == 1: raise TimeoutError('temporary download failure')
        return config.portable_contract()
    s._dataset_document = document
    record = RunRecord(uid='job', name='scan', mode='full', history_id='history', config=config, idempotency_token='submitted',
        inputs=InputSelection(data_dataset_id='data', instrument_dataset_id='profile'))
    for _ in range(4): s.refresh(record)
    assert reads == ['resolved', 'resolved']
    assert record.config.full_max_passes == 7 and record.config.run_name == 'scan'


def test_recovery_retries_provisional_settings_until_configuration_is_readable(tmp_path):
    s = service(tmp_path); readable = False; reads = []
    parameters = {'reproducibility|configuration_override|config_file': {'id':'config'},
        'data_inputs|input_source|diffraction_pattern': {'id':'data'},
        'data_inputs|input_source|instrument_source|instrument_file': {'id':'profile'}}
    s._job_details = lambda _: dict(state='running', params=parameters,
        inputs={'reproducibility|configuration_override|config_file': {'id':'config','src':'hda'}})
    exact = AnalysisConfig(sample_elements=['Al'], full_max_passes=7)
    def document(identifier):
        reads.append(identifier)
        if not readable: raise TimeoutError('The input is not ready yet')
        return exact.portable_contract()
    s._dataset_document = document
    record = RunRecord(uid='job', name='scan', mode='full', history_id='history')
    s.refresh(record)
    assert not record.provenance.get('configuration_dataset_id')
    readable = True
    s.refresh(record)
    assert record.config.full_max_passes == 7
    assert record.provenance['configuration_dataset_id'] == 'config'
    count = len(reads)
    s.refresh(record)
    assert len(reads) == count


def test_configuration_recovery_uses_encoded_public_input_identity(tmp_path):
    s = service(tmp_path); reads = []; provenance = {}
    config = AnalysisConfig(sample_elements=['Al'], full_max_passes=7)
    s._dataset_document = lambda identifier: reads.append(identifier) or config.portable_contract()
    job = {'inputs': {'reproducibility|configuration_override|config_file': {'id':'encoded-config','src':'hda'}}}
    recovered = s._config_from_job(job, {'config_file': {'id':12345}}, {}, provenance=provenance)
    assert reads == ['encoded-config'] and recovered.full_max_passes == 7
    assert provenance['configuration_dataset_id'] == 'encoded-config'


@pytest.mark.parametrize('utility', [False, True])
@pytest.mark.parametrize('metadata', [{'state':'error'}, {'state':'discarded','deleted':True}])
def test_failed_upload_is_not_an_indefinitely_queued_analysis(tmp_path, utility, metadata):
    s = service(tmp_path)
    s._job_details = lambda _: dict(state='paused', inputs={'recipe_request': {'id':'bad', 'src':'hda'}})
    s._dataset_metadata = lambda _: metadata
    if utility:
        record = UtilityActionRecord(uid='action', name='recipe', tool_id='refine', galaxy_job_id='job')
        s.refresh_utility(record)
    else:
        record = RunRecord(uid='job', name='scan', mode='full', history_id='history')
        s.refresh(record)
    assert record.status is RunStatus.ERROR and 'input' in record.message and 'History' in record.message


def test_manually_paused_job_with_healthy_inputs_remains_recoverable(tmp_path):
    s = service(tmp_path)
    s._job_details = lambda _: dict(state='paused', inputs={'data': {'id':'ok', 'src':'hda'}})
    s._dataset_metadata = lambda _: {'state':'ok'}
    record = RunRecord(uid='job', name='scan', mode='full', history_id='history')
    s.refresh(record)
    assert record.status is RunStatus.QUEUED and record.stage == 'Galaxy job paused'


def test_profile_reuse_requires_exact_ready_undeleted_content(tmp_path):
    s = service(tmp_path); profile = tmp_path / 'profile.instprm'; profile.write_text('DIFC: 12345\n')
    rows = [dict(id=name, state=state) for name,state in [('failed','error'),('pending','running'),('deleted','ok'),('wrong','ok'),('exact','ok')]]
    queries = []
    def search(**kwargs): queries.append(kwargs['query']); return rows
    s.search_history_datasets = search
    s._dataset_metadata = lambda identifier: {'state':'ok','deleted':identifier=='deleted'}
    s.load_text_document = lambda identifier, **kw: profile.read_text() if identifier=='exact' else 'DIFC: 99999\n'
    s.upload_document = lambda *a, **kw: pytest.fail('An exact verified profile was reuploaded')
    assert s.stage_instrument_profile(profile, label='profile') == 'exact'
    assert sha256(profile.read_bytes()).hexdigest() in queries[0]


def test_profile_search_failure_falls_back_to_a_pending_input(tmp_path):
    s = service(tmp_path); profile = tmp_path / 'profile.instprm'; profile.write_text('DIFC: 12345\n')
    s.search_history_datasets = lambda **kw: (_ for _ in ()).throw(TimeoutError())
    calls = []
    s.upload_document = lambda path, **kw: calls.append(kw) or 'pending-profile'
    assert s.stage_instrument_profile(profile, label='profile') == 'pending-profile'
    assert calls[0]['wait_ready'] is False


def test_parallel_result_downloads_share_one_limit_and_report_each_failure(tmp_path):
    s = service(tmp_path); lock = threading.Lock(); active = peak = 0
    def download(identifier, path):
        nonlocal active, peak
        with lock:
            active += 1; peak = max(peak, active)
        try:
            time.sleep(0.03)
            if identifier == 'bad': raise OSError('transfer failed')
            path.write_text(identifier)
        finally:
            with lock: active -= 1
    s._download_dataset = download
    batches = [{str(n):(str(n),tmp_path/str(n)) for n in range(i,i+8)} for i in (0,8)]
    batches[0]['bad'] = ('bad',tmp_path/'bad')
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(s._prefetch_results, batches))
    assert 1 < peak <= 4
    assert isinstance(outcomes[0]['bad'], OSError)
    assert sum(p.is_file() for p in tmp_path.iterdir()) == 16
