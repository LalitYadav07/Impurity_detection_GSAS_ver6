import asyncio,copy,json,os,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from radar_pd_nova.webmon.contracts import validate_request,verify_result,seal_posterior,verify_posterior,file_hash
from radar_pd_nova.webmon.snapshot import SnapshotArchive,parse_live_ascii
from radar_pd_nova.webmon.session import WebmonSession
from radar_pd_nova.webmon.statistics import QuadraticObservation,sample_posterior
from radar_pd_nova.webmon.registry import CheckpointRegistry
from radar_pd_nova.webmon.processes import spawn_process,stop_process,wait_process


def scientific_request(tmp_path):
    source=tmp_path/'source.gpx';source.write_bytes(b'fixture GPX bytes; never unpickled')
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps(dict(k=['0','0','0'],candidates=[dict(id='mag:test')])) )
    preflight=tmp_path/'preflight.json';preflight.write_text(json.dumps([dict(id='mag:test')]))
    r=dict(source_gpx=str(source),source_sha256=file_hash(source),catalog=str(catalog),catalog_sha256=file_hash(catalog),preflight=str(preflight),preflight_sha256=file_hash(preflight),k=['0','0','0'],sites=[dict(label='Er1',form_factor='Er+3')],lower=5,upper=7,particles=64,workers=1,source_revision='PG3_1:livehash',reference_revision='PG3_2:refhash')
    b=dict(source_gpx=str(source),source_sha256=r['source_sha256'],source_dat_sha256='livehash',reference_sha256='refhash',k=r['k'],ion='Er+3',maximum_moment=9)
    return r,b


def test_request_rejects_broken_scientific_binding(tmp_path):
    r,b=scientific_request(tmp_path);validate_request(r,b)
    r['reference_revision']='PG3_2:wrong'
    with pytest.raises(ValueError,match='revision'):validate_request(r,b)


def test_reviewed_catalogue_cannot_be_replaced_with_self_consistent_subset(tmp_path):
    r,b=scientific_request(tmp_path)
    template=tmp_path/'reviewed.json';template.write_text(json.dumps(r));b['request']=str(template)
    replacement=tmp_path/'subset.json'
    replacement.write_text(Path(r['catalog']).read_text())
    r['catalog']=str(replacement);r['catalog_sha256']=file_hash(replacement)
    with pytest.raises(ValueError,match='reviewed candidate set'):validate_request(r,b)

@pytest.mark.parametrize('key,value',[('particles',64.5),('workers',99),('candidate_seconds',float('nan')),('upper',float('inf')),('lower',7)])
def test_bad_resource_or_moment_bounds_fail_before_native_launch(tmp_path,key,value):
    r,b=scientific_request(tmp_path);r[key]=value
    with pytest.raises(ValueError):validate_request(r,b)


def test_catalogue_path_traversal_and_duplicate_preflight_rejected(tmp_path):
    r,b=scientific_request(tmp_path);p=Path(r['catalog']);v=json.loads(p.read_text());v['candidates'][0]['id']='../../escape';p.write_text(json.dumps(v));r['catalog_sha256']=file_hash(p)
    with pytest.raises(ValueError,match='identifiers'):validate_request(r,b)


def test_finished_result_requires_exact_reference_and_closed_scientific_gates(tmp_path):
    r,b=scientific_request(tmp_path);p=tmp_path/'request.json';p.write_text(json.dumps(r))
    result=dict(status='complete',request_sha256=file_hash(p),source_revision=r['source_revision'],reference_revision=r['reference_revision'],forecast_enabled=False,exclusion_authorized=False)
    q=tmp_path/'result.json';q.write_text(json.dumps(result));verify_result(tmp_path,binding=b,complete=True)
    result['reference_revision']='wrong';q.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='revision'):verify_result(tmp_path,binding=b,complete=True)
    result['reference_revision']=r['reference_revision'];result['forecast_enabled']=True;q.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='authorize'):verify_result(tmp_path,binding=b,complete=True)


def test_posterior_tamper_and_unsealed_archive_rejected(tmp_path):
    for name in ('kernel.json','kernel.npz','posterior.json','posterior.npz','inference-request.json'):(tmp_path/name).write_bytes(name.encode())
    receipt=seal_posterior(tmp_path);verify_posterior(tmp_path/'posterior.json',receipt)
    (tmp_path/'posterior.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):verify_posterior(tmp_path/'posterior.json',receipt)
    with pytest.raises(ValueError,match='no artifact'):verify_posterior(tmp_path/'posterior.json')


def test_preview_uses_captured_bytes_even_if_live_file_changes(tmp_path,monkeypatch):
    root=tmp_path/'PG3/IPTS-1/shared/livereduce';root.mkdir(parents=True);p=root/'PG3_2.dat'
    raw=b'# X , Y , E Distribution=true\n0\n1,2,1\n2,3,1\n';p.write_bytes(raw)
    manifest=tmp_path/'context.json';manifest.write_text(json.dumps(dict(facility_root=str(tmp_path),archive=str(tmp_path/'archive'))));s=WebmonSession(manifest)
    capture=s.archive.capture
    def switching(*args,**kwargs):
        result=capture(*args,**kwargs);p.write_bytes(raw.replace(b'1,2,1',b'1,9,1'));return result
    monkeypatch.setattr(s.archive,'capture',switching)
    _,receipt,pattern=s.capture('1','2');assert pattern.y[0]==2
    assert receipt['content_sha256']!=file_hash(p)


def test_runs_are_ordered_numerically(tmp_path):
    folder=tmp_path/'PG3/IPTS-1/shared/livereduce';folder.mkdir(parents=True)
    for run in (99,100):(folder/f'PG3_{run}.dat').touch()
    m=tmp_path/'m.json';m.write_text(json.dumps(dict(facility_root=str(tmp_path),archive=str(tmp_path/'a'))))
    assert [r['run'] for r in WebmonSession(m).runs('1')]==['100','99']


def test_registry_preserves_other_writers_and_immutable_revisions(tmp_path):
    a=CheckpointRegistry(tmp_path/'registry.json');b=CheckpointRegistry(a.path)
    a.add('IPTS-1/2',dict(id='a',parent_checkpoint='base'));b.add('IPTS-1/2',dict(id='b',parent_checkpoint='base'))
    assert {v['id'] for v in a.read()['IPTS-1/2']}=={'a','b'}
    with pytest.raises(ValueError,match='immutable'):a.add('IPTS-1/2',dict(id='a',parent_checkpoint='wrong'))

@pytest.mark.parametrize('forms',[np.zeros((1,1,1)),np.array([[[-1.]]])])
def test_unbounded_or_nonphysical_moment_forms_are_explicitly_unresolved(forms):
    with pytest.raises(ValueError):QuadraticObservation(np.zeros(2),np.ones((1,1,2)),np.ones(2),np.ones(2),forms)


def test_sampler_reports_local_and_sign_proposals_separately():
    m=QuadraticObservation(np.zeros(2),np.ones((1,1,2)),np.ones(2),np.ones(2),np.ones((1,1,1)))
    p=sample_posterior(m,particles=64,max_seconds=5)
    for h in p['history']:
        assert h['local_proposals']+h['sign_proposals']==64*12
        assert 0<=h['local_acceptance']<=1 and 0<=h['sign_acceptance']<=1


def test_native_process_deadline_terminates_child():
    async def scenario():
        p=await spawn_process(sys.executable,'-c','import time;time.sleep(30)')
        with pytest.raises(asyncio.TimeoutError):await wait_process(p,.05)
        assert p.returncode is not None
    asyncio.run(scenario())


def test_native_process_cancellation_terminates_process_group(tmp_path):
    async def scenario():
        child_pid=tmp_path/'pid';heartbeat=tmp_path/'heartbeat'
        child_code='import time,sys;from pathlib import Path\nfor i in range(3000):\n Path(sys.argv[1]).write_text(str(i));time.sleep(.01)'
        code='import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,"-c",sys.argv[2],sys.argv[3]]);Path(sys.argv[1]).write_text(str(p.pid));time.sleep(30)'
        p=await spawn_process(sys.executable,'-c',code,str(child_pid),child_code,str(heartbeat))
        for _ in range(100):
            if child_pid.exists() and heartbeat.exists():break
            await asyncio.sleep(.01)
        assert heartbeat.exists();await stop_process(p)
        assert p.returncode is not None
        last=heartbeat.read_text();await asyncio.sleep(.1)
        assert heartbeat.read_text()==last
    asyncio.run(scenario())


def test_worker_rejects_a_second_writer_before_loading_inputs(tmp_path):
    import fcntl
    from radar_pd_nova.webmon.worker import run
    with (tmp_path/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='active writer'):run(tmp_path/'request.json')


def test_invalid_manifest_budgets_fail_closed():
    from radar_pd_nova.webmon.doctor import validate_config
    c=dict(facility_root='/SNS',archive='/tmp/snapshots',jobs='/tmp/jobs',worker_command=['python','-m','radar_pd_nova.webmon.worker'])
    validate_config(c)
    with pytest.raises(ValueError):validate_config(dict(c,poll_seconds=float('nan')))
    with pytest.raises(ValueError):validate_config(dict(c,worker_command='python -m bad'))


def test_cancelled_calculation_has_explicit_status():
    from radar_pd_nova.webmon.ui import WebmonMixin
    obj=WebmonMixin();obj.server=SimpleNamespace(state=SimpleNamespace())
    obj._webmon_process=None;obj._webmon=SimpleNamespace(config={'local_replay':True})
    asyncio.run(obj.webmon_cancel())
    assert obj._webmon_cancelled
    assert obj.server.state.webmon_calculation_status=='Cancelled'


def test_stopping_remote_wait_is_not_reported_as_job_cancellation():
    from radar_pd_nova.webmon.ui import WebmonMixin
    obj=WebmonMixin();obj.server=SimpleNamespace(state=SimpleNamespace())
    obj._webmon_process=None;obj._webmon=SimpleNamespace(config={'local_replay':False})
    asyncio.run(obj.webmon_cancel())
    assert obj.server.state.webmon_calculation_status=='Stopped waiting'
    assert 'Galaxy job continues' in obj.server.state.webmon_message
