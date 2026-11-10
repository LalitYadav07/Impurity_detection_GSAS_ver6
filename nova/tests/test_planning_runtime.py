"""Managed planner transport uses real durable native handles, without GSAS-II."""
import copy
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import threading
import zipfile

import numpy as np
import pytest

from gpx_recipe.planning.managed import RESULT_SCHEMA, OPERATIONS, read_plan
from gpx_recipe.symmetry.contracts import digest
from gpx_recipe.symmetry.native_protocol import validate_native_payload
from radar_pd_nova.models import RunStatus
from radar_pd_nova.planning_runtime import GalaxyPlanningRuntime, capture
from radar_pd_nova.refinement_runtime import RuntimeUnavailable
from test_model_runtime import remote


def scenario(tmp_path):
    source=tmp_path/'sample.cif';source.write_text('data_transport_control\n')
    profile=tmp_path/'profile.instprm';profile.write_text('# instrument transport control\n')
    return dict(schema='powgen-planning-scenario/v1',cif=str(source),custom_profile=str(profile),additional_phases=[])


def reply(service, request, job, tmp_path):
    data=io.BytesIO();np.savez(data,d=[1.,2.,3.],expected=[0.,1.,0.],nuclear=[0.,1.,0.],magnetic=[0.,0.,0.])
    files={'native/result/curves.npz':data.getvalue(),'native/result/model/forward.gpx':b'opaque native model',
           'native/result/scenario.json':json.dumps(dict(schema='powgen-planning-scenario/v1',cif='inputs/cif.cif')).encode()}
    receipt=dict(schema='powgen-planning-result/v1',mode='profile_only',configuration='test',
        intensity_unit='shape',looks=[],warnings=[],isotope_options={},isotope_selection={},phases=[],
        artifacts={name.removeprefix('native/result/'):hashlib.sha256(raw).hexdigest() for name,raw in files.items()})
    files['native/result/result.json']=json.dumps(receipt).encode()
    wrapper=dict(schema=RESULT_SCHEMA,operation=request['operation'],output='/worker/native/result',result=receipt)
    outer=dict(schema='radar-model-native-result/v1',dispatch_key=request['dispatch_key'],
        request_sha256=digest(request),source_sha256=request['expected_source_sha256'],
        worker_root='/worker/native',input_root='/worker/inputs-packet',result=wrapper,
        files={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()},native_seconds=.5)
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w') as archive:
        archive.writestr('response.json',json.dumps(outer))
        for name,raw in files.items():archive.writestr(name,raw)
    job.outputs={'result_bundle':service._put(output.getvalue(),'managed planning result')}
    job.status=RunStatus.OK


def adapter(native, root, **kwargs):
    return GalaxyPlanningRuntime(native,root,check_text=lambda p:Path(p),
        check_native=lambda p:(_ for _ in ()).throw(ValueError('untrusted GPX')),**kwargs)


def test_every_planning_action_has_a_closed_transport_payload():
    for operation in OPERATIONS:
        payload=dict(source='source.cif',plan='plan.json',children=[],destination='result')
        validate_native_payload(operation,payload)
        with pytest.raises(ValueError):validate_native_payload(operation,{**payload,'python':'unsafe'})


def test_capture_uses_relative_inventory_and_rejects_changed_bytes(tmp_path):
    value=scenario(tmp_path);root=tmp_path/'packet'
    document,payload=capture('planning-parent',value,root,check_text=lambda p:Path(p),check_native=lambda p:Path(p))
    assert all(not Path(name).is_absolute() for name in document['files'])
    assert set(document['files'])=={'assets/00.cif','assets/01.instprm'}
    decoded=read_plan(root/'plan.json','planning-parent')
    assert decoded['cif']==str(root/'assets/00.cif')
    (root/'assets/00.cif').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):read_plan(root/'plan.json','planning-parent')


def test_capture_rejects_uploaded_gpx_and_escape_before_dispatch(tmp_path):
    value=scenario(tmp_path);gpx=tmp_path/'uploaded.gpx';gpx.write_bytes(b'not executable authority')
    value['prepared_magnetic']={'gpx':str(gpx)}
    with pytest.raises(ValueError,match='untrusted'):
        capture('planning-simulate',value,tmp_path/'packet',check_text=lambda p:Path(p),
            check_native=lambda p:(_ for _ in ()).throw(ValueError('untrusted native upload')))
    value=scenario(tmp_path);document,_=capture('planning-parent',value,tmp_path/'safe',
        check_text=lambda p:Path(p),check_native=lambda p:Path(p))
    document['request']['cif']='../sample.cif'
    (tmp_path/'safe/plan.json').write_text(json.dumps(document))
    with pytest.raises(ValueError,match='relative'):read_plan(tmp_path/'safe/plan.json','planning-parent')


def test_galaxy_parent_publishes_verified_artifacts_without_local_science(remote,tmp_path,monkeypatch):
    native,service,_=remote;value=scenario(tmp_path);intents=[];progress=[]
    original=service.submit_utility
    def submit(**kwargs):
        assert intents and not service.submissions
        job=original(**kwargs)
        request=service.requests[kwargs['inputs']['model_request']['dataset_id']]
        reply(service,request,job,tmp_path)
        return job
    monkeypatch.setattr(service,'submit_utility',submit)
    monkeypatch.setattr('gpx_recipe.process.run_scientific',lambda *a,**k:pytest.fail('NOVA must not run native science'))
    runtime=adapter(native,tmp_path/'runtime-planner',checkpoint=intents.append,progress=lambda **row:progress.append(row))
    result=runtime.execute('planning-parent',value,tmp_path/'parent',threading.Event())
    assert result['mode']=='profile_only' and len(service.submissions)==1
    assert (tmp_path/'parent/simulation/curves.npz').is_file()
    model=tmp_path/'parent/simulation/model/forward.gpx'
    assert str(model) in runtime.artifacts and runtime.artifacts[str(model)]['handle']['job']
    assert progress[-1]['state']=='finished' and progress[-1]['job_id']
    assert all(b'PRIVATE-' not in raw for raw in service.data.values())
    # A fresh adapter proves the GPX against its original job without loading it.
    fresh=adapter(native,tmp_path/'fresh')
    verified=fresh.verify_artifacts(runtime.artifacts,tmp_path/'reverified')
    assert str(model) in verified and len(service.submissions)==1
    original,_=fresh.verify_result(tmp_path/'parent/simulation',verified)
    assert original['cif']=='inputs/cif.cif'
    path=tmp_path/'parent/simulation/result.json';changed=json.loads(path.read_text());changed['mode']='current_cycle'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='metadata differs'):
        fresh.verify_result(tmp_path/'parent/simulation',verified)


def test_ambiguous_post_cannot_duplicate_and_recovery_only_collects(remote,tmp_path,monkeypatch):
    native,service,_=remote;value=scenario(tmp_path);service.lose_ack=True
    runtime=adapter(native,tmp_path/'planner')
    with pytest.raises(RuntimeUnavailable,match='acknowledge'):
        runtime.execute('planning-parent',value,tmp_path/'pending',threading.Event())
    handle=json.loads((tmp_path/'pending/operation.json').read_text())['handle']
    assert handle['submission_started'] and handle['job'] is None and len(service.submissions)==1
    # With no acknowledgement found, explicit recovery stops. No submit path.
    monkeypatch.setattr(native,'find_dispatch',lambda _:None)
    with pytest.raises(RuntimeError,match='duplicate'):
        runtime.recover(handle,tmp_path/'missing',threading.Event())
    assert len(service.submissions)==1
    key=handle['key'];reply(service,handle['request'],service.jobs[key],tmp_path)
    monkeypatch.setattr(native,'find_dispatch',lambda key:service.jobs[key])
    result=runtime.recover(handle,tmp_path/'recovered',threading.Event())
    assert result['mode']=='profile_only' and len(service.submissions)==1


def test_cancel_requests_exact_job_and_never_collects(remote,tmp_path,monkeypatch):
    native,service,_=remote;value=scenario(tmp_path);event=threading.Event();cancelled=[]
    original=service.submit_utility
    def queued(**kwargs):
        job=original(**kwargs);job.status=RunStatus.QUEUED;return job
    monkeypatch.setattr(service,'submit_utility',queued)
    monkeypatch.setattr(native,'cancel_job',lambda job,key:cancelled.append((job.galaxy_job_id,key)))
    monkeypatch.setattr(native,'result',lambda *a:pytest.fail('Cancelled output must not be collected'))
    runtime=adapter(native,tmp_path/'planner',progress=lambda **row:event.set() if row['state']=='queued' else None)
    with pytest.raises(RuntimeError,match='cancelled'):
        runtime.execute('planning-parent',value,tmp_path/'cancelled',event)
    assert len(cancelled)==len(service.submissions)==1
    assert not runtime.artifacts


def test_worker_adapter_stages_real_closed_packet_and_routes_existing_engine(remote,tmp_path,monkeypatch):
    from radar_pd_nova.native_operations import capture as native_capture
    from gpx_recipe.planning import managed_worker,worker
    from gpx_recipe.execution import write_json
    native,_,_=remote;request=scenario(tmp_path)
    packet=tmp_path/'packet'
    _,payload=capture('planning-parent',request,packet,check_text=lambda p:Path(p),check_native=lambda p:Path(p))
    source=dict(id='parent',dataset_id='test-dataset',source_path=payload['source'],
        source_sha256=hashlib.sha256(Path(payload['source']).read_bytes()).hexdigest())
    handle,_=native_capture(native,'planning-parent',payload,packet,source,300)
    entrypoint=os.environ.get('RADAR_PD_TEST_MODEL_ENTRYPOINT')
    if entrypoint is None:
        entrypoint=Path(__file__).resolve().parents[2]/'scripts/ndip_model_campaign.py'
    entrypoint=Path(entrypoint)
    assert entrypoint.is_file(),'Provide the reviewed worker entrypoint fixture for a relocated test suite'
    spec=importlib.util.spec_from_file_location('planning_worker_test',entrypoint)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'require_adapters',lambda:None)
    called=[]
    def forward(value,output):
        called.append(value)
        assert Path(value['cif']).read_text()=='data_transport_control\n'
        assert Path(value['custom_profile']).read_text().startswith('# instrument')
        receipt=dict(schema='powgen-planning-result/v1',mode='profile_only',artifacts={})
        write_json(output/'result.json',receipt)
        return receipt
    monkeypatch.setattr(worker,'run',forward)
    def child(command,**kwargs):
        assert command[1:3]==['-m','gpx_recipe.planning.managed_worker']
        value=managed_worker.execute(command[3],json.loads(Path(command[4]).read_text()))
        write_json(Path(command[5]),value)
        return type('Result',(),{'returncode':0})()
    monkeypatch.setattr(module,'run_scientific',child)
    root=tmp_path/'native';root.mkdir()
    result,_=module.native(handle['request'],packet,root,threading.Event())
    assert result['schema']==RESULT_SCHEMA and len(called)==1
    assert result['operation']=='planning-parent'
