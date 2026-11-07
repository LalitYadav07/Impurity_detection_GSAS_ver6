"""Batch allocation, fairness, artifact and terminal-state contracts."""
import copy
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from gpx_recipe.execution import write_json
from gpx_recipe.preparation import batch
from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.symmetry.native_protocol import validate_native_payload


def plan(root,count=4):
    inputs=root/'inputs';inputs.mkdir()
    (inputs/'parent.cif').write_text('data_public_batch_control\n')
    (inputs/'profile.instprm').write_text('# profile contract placeholder\n')
    tasks=[]
    for i in range(count):
        entry=format(i,'024x')
        request=dict(source='parent.cif',instrument='profile.instprm',
            grid=dict(kind='PNT',lower=3000.,upper=40000.,bins=701,bank=1),
            settings=dict(maximum_denominator=8),destination='tasks/'+entry+'/attempt-1/compiled')
        tasks.append(dict(entry=entry,attempt=1,kind='reference',request=request))
    path=inputs/'plan.json';write_json(path,dict(schema=batch.SCHEMA,tasks=tasks))
    return dict(source=str(inputs/'parent.cif'),plan=str(path),destination=str(root/'outputs'),workers=4,seconds=30)


def prepared(task,request,scratch,**kwargs):
    root=Path(request['destination']);root.mkdir(parents=True)
    manifest=dict(schema='radar-experiment-preparation-entry/v1',kind=task['kind'],
        input_sha256=[file_sha(request[k]) for k in ('source','instrument')],
        settings=request['settings'],grid=request['grid'],observations_included=False,scientific_result=False,artifacts={})
    write_json(root/'entry.json',manifest)
    return dict(state='prepared',reason='',result=dict(schema='radar-experiment-preparation-worker/v1',
        status='prepared',kind=task['kind'],output=str(root),observations_included=False,scientific_result=False,
        artifacts=['entry.json'],entry_sha256=file_sha(root/'entry.json')))


@pytest.mark.parametrize('scattering',[False,True])
def test_batch_accepts_legacy_and_factor_bearing_magnetic_artifacts(tmp_path,scattering):
    request=plan(tmp_path,count=1)
    task=json.loads(Path(request['plan']).read_text())['tasks'][0]
    task['kind']='magnetic'
    payload=dict(task['request'],destination=request['destination'])
    for role in ('source','instrument'):
        payload[role]=str(Path(request['plan']).parent/payload[role])
    row=prepared(task,payload,tmp_path/'scratch')
    output=Path(payload['destination']);manifest=json.loads((output/'entry.json').read_text())
    names=['response.npz']+(['scattering.npz'] if scattering else [])
    for name in names:
        np.savez(output/name,control=np.ones(3))
        manifest['artifacts'][name]=file_sha(output/name)
    if scattering:
        manifest['detail']=dict(scattering=dict(status='prepared',artifact='scattering.npz',identity='a'*64))
    write_json(output/'entry.json',manifest)
    row['result'].update(artifacts=['entry.json']+names,entry_sha256=file_sha(output/'entry.json'))
    assert {path.name for path in batch.verify_result(task,payload,row)}==set(names+['entry.json'])
    if scattering:
        manifest['detail']['scattering']['identity']='unbound'
        write_json(output/'entry.json',manifest)
        row['result']['entry_sha256']=file_sha(output/'entry.json')
        with pytest.raises(ValueError,match='physical identity'):
            batch.verify_result(task,payload,row)


@pytest.mark.parametrize('result',[None,[],{},dict(artifacts=['entry.json','entry.json'])])
def test_batch_rejects_malformed_receipts_with_a_checked_error(tmp_path,result):
    with pytest.raises(ValueError,match='portable artifact inventory'):
        batch.verify_result(dict(kind='magnetic'),dict(destination=str(tmp_path)),dict(result=result))


def test_batch_uses_granted_workers_with_separate_attempt_paths(tmp_path,monkeypatch):
    from gpx_recipe.symmetry.resources import allocation
    monkeypatch.setattr(batch,'allocation',lambda requested,**kwargs:
        allocation(requested,cgroup=tmp_path/'no-cgroup-report',**kwargs))
    request=plan(tmp_path);guard=threading.Lock();active=0;peak=0;paths=[]
    def runner(task,payload,scratch,**kwargs):
        nonlocal active,peak
        with guard:
            active+=1;peak=max(active,peak);paths.append(payload['destination'])
        try:
            time.sleep(.02)
            return prepared(task,payload,scratch,**kwargs)
        finally:
            with guard:active-=1
    result=batch.execute(request,runner=runner,environ={'GALAXY_SLOTS':'2','GALAXY_MEMORY_MB':'8192'})
    assert result['status']=='complete' and all(row['state']=='prepared' for row in result['entries'])
    assert result['all_prepared'] and result['counts']==dict(prepared=4,failed=0,cancelled=0,pending=0)
    assert result['allocation']['workers']==2 and peak==2
    assert len(paths)==len(set(paths))==4
    assert result['observations_included'] is False and result['scientific_result'] is False
    progress=json.loads((Path(result['output'])/'progress.json').read_text())
    assert progress['terminal']==progress['prepared']==4 and not progress['active'] and not progress['pending']


def test_unreported_allocation_remains_one_worker(tmp_path,monkeypatch):
    from gpx_recipe.symmetry.resources import allocation
    # An empty environment does not remove a real container CPU quota. This
    # unit test explicitly models both absent scheduler and cgroup reports.
    monkeypatch.setattr(batch,'allocation',lambda requested,**kwargs:
        allocation(requested,cgroup=tmp_path/'no-cgroup-report',**kwargs))
    result=batch.execute(plan(tmp_path),runner=prepared,environ={})
    assert result['allocation']['workers']==1


def test_one_failed_task_does_not_block_other_entries(tmp_path):
    request=plan(tmp_path)
    def runner(task,*args,**kwargs):
        if task['entry']=='0'*24:raise RuntimeError('private diagnostic must not be copied into status')
        return prepared(task,*args,**kwargs)
    result=batch.execute(request,runner=runner,environ={'GALAXY_SLOTS':'2'})
    assert [row['state'] for row in result['entries']]==['failed','prepared','prepared','prepared']
    assert not result['all_prepared'] and result['counts']['failed']==1
    assert 'private diagnostic must not be copied' not in json.dumps(result)


def test_soft_deadline_keeps_unstarted_entries_pending(tmp_path):
    request=plan(tmp_path,count=3);now=[0.]
    def runner(*args,**kwargs):
        result=prepared(*args,**kwargs);now[0]+=31.;return result
    result=batch.execute(request,runner=runner,clock=lambda:now[0],environ={'GALAXY_SLOTS':'1'})
    assert result['status']=='partial'
    assert [row['state'] for row in result['entries']]==['prepared','pending','pending']
    assert all(row['reason']=='batch_time_budget' for row in result['entries'][1:])
    progress=json.loads((Path(result['output'])/'progress.json').read_text())
    assert progress['terminal']==1 and progress['pending']==2


def test_cancel_before_start_preserves_all_entries_as_pending(tmp_path):
    event=threading.Event();event.set()
    result=batch.execute(plan(tmp_path),runner=lambda *a,**k:pytest.fail('No cancelled work may start'),cancel=event,environ={})
    assert all(row['state']=='pending' and row['reason']=='cancelled_before_start' for row in result['entries'])


def test_artifact_budget_defers_work_instead_of_discarding_finished_entries(tmp_path,monkeypatch):
    request=plan(tmp_path,count=3)
    monkeypatch.setattr(batch,'RESULT_BUDGET',batch.RESERVES['reference']+512*1024)
    def runner(task,payload,scratch,**kwargs):
        result=prepared(task,payload,scratch,**kwargs)
        path=Path(payload['destination'])/'entry.json';manifest=json.loads(path.read_text())
        manifest['padding']='x'*(1024*1024);write_json(path,manifest)
        result['result']['entry_sha256']=file_sha(path)
        return result
    result=batch.execute(request,runner=runner,environ={'GALAXY_SLOTS':'1'})
    assert [row['state'] for row in result['entries']]==['prepared','pending','pending']
    assert all(row['reason']=='batch_artifact_budget' for row in result['entries'][1:])
    assert (Path(result['output'])/result['artifacts'][0]).is_file()


@pytest.mark.parametrize('change',['extra','duplicate','escape','output','attempt','kind'])
def test_invalid_or_ambiguous_plan_is_rejected_before_native_work(tmp_path,change):
    request=plan(tmp_path);path=Path(request['plan']);document=json.loads(path.read_text())
    first=document['tasks'][0]
    if change=='extra':first['request']['api_key']='not allowed'
    elif change=='duplicate':document['tasks'][1]=copy.deepcopy(first)
    elif change=='escape':first['request']['source']='../private.cif'
    elif change=='output':first['request']['destination']='someone-else/compiled'
    elif change=='attempt':first['attempt']=True
    else:first['kind']='python'
    write_json(path,document)
    with pytest.raises(ValueError):batch.execute(request,runner=lambda *a,**k:pytest.fail('No invalid work'),environ={})


def test_batch_payload_is_closed_to_environment_or_executable_fields(tmp_path):
    request=plan(tmp_path)
    validate_native_payload('preparation-batch',request)
    with pytest.raises(ValueError):validate_native_payload('preparation-batch',dict(request,environment={'key':'secret'}))


def test_changed_common_input_invalidates_the_batch(tmp_path):
    request=plan(tmp_path,count=1)
    def runner(task,payload,*args,**kwargs):
        result=prepared(task,payload,*args,**kwargs)
        Path(payload['instrument']).write_text('changed calibration')
        return result
    with pytest.raises(ValueError,match='inputs changed'):
        batch.execute(request,runner=runner,environ={})


def test_nested_native_environment_retains_reviewed_adapters_and_drops_service_keys(monkeypatch):
    import os
    from gpx_recipe.symmetry.compiler import native_environment
    monkeypatch.setenv('AZURE_ILLUMINE_API_KEY','private-model-sentinel')
    monkeypatch.setenv('BCS_API_KEY','private-bilbao-sentinel')
    monkeypatch.setenv('PYTHONPATH','/unreviewed/arbitrary')
    environment=native_environment()
    first=Path(environment['PYTHONPATH'].split(os.pathsep)[0])
    assert (first/'gsas_magnetic_preflight.py').is_file()
    assert '/unreviewed' not in environment['PYTHONPATH']
    assert 'PRIVATE' not in repr(environment).upper()
    assert environment['OPENBLAS_NUM_THREADS']=='1'


def test_termination_signals_are_scoped_and_relay_to_cancellation_event():
    import signal
    from gpx_recipe.process import cancellation_signals
    event=threading.Event();before=signal.getsignal(signal.SIGTERM)
    with cancellation_signals(event):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
        assert event.is_set()
    assert signal.getsignal(signal.SIGTERM)==before
