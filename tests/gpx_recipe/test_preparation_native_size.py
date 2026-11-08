"""Owned native sizing, conservative fallbacks and unchanged output guards."""
import copy
import io
import json
from pathlib import Path
import shutil
import threading
import time

import numpy as np
import pytest

from gpx_recipe.execution import write_json
from gpx_recipe.preparation import batch, native_size
from gpx_recipe.preparation.structural_jobs import publish_result
from gpx_recipe.preparation.structural_paired_blocks import read
from gpx_recipe.preparation.structural_pair_arrays import encode
from gpx_recipe.symmetry.compiler import package_signature
from gpx_recipe.symmetry.contracts import digest
from test_structural_paired_preparation import inputs


def measured(request, *, modes=None):
    # Public artificial fixture, not a claim about a facility experiment.
    rows=1+(request['settings']['trajectory_slice'][1]-request['settings']['trajectory_slice'][0])*(request['settings']['steps']+1)
    bins=request['grid']['bins'];modes=1 if modes is None else modes;cells=1
    return dict(rows=rows,bins=bins,mode_dimensions=modes,cell_dimensions=cells,
        numeric_bytes=native_size.paired_array_bytes(rows,bins,modes,cells))


def test_native_measurement_binds_request_attempt_and_installed_code(tmp_path):
    request,_=inputs(tmp_path);Path(request['destination']).mkdir()
    native_size.publish(request,**measured(request))
    record=native_size.read(request,package_signature())
    assert record['reservation_bytes']<32*1024**2
    for change in ['instrument','destination']:
        other=copy.deepcopy(request);other[change]+='-other'
        with pytest.raises((FileNotFoundError,ValueError)):native_size.read(other,package_signature())
    with pytest.raises(ValueError,match='implementation'):
        native_size.read(request,'f'*64)
    with pytest.raises(ValueError,match='fresh'):
        native_size.publish(request,**measured(request))


def test_unpublished_native_size_is_pending_instead_of_an_invalid_record(tmp_path):
    request,_=inputs(tmp_path)
    with pytest.raises(FileNotFoundError):native_size.read(request,package_signature())
    Path(request['destination']).mkdir()
    with pytest.raises(FileNotFoundError):native_size.read(request,package_signature())
    # A dangling link is an invalid publication, not a pending producer.
    (Path(request['destination'])/native_size.NAME).symlink_to(tmp_path/'missing.json')
    with pytest.raises(ValueError):native_size.read(request,package_signature())


@pytest.mark.parametrize('case',['extra','rows','bins','numeric','reserve','implementation','request'])
def test_changed_or_foreign_sizing_cannot_reduce_reservations(tmp_path,case):
    request,_=inputs(tmp_path);Path(request['destination']).mkdir()
    native_size.publish(request,**measured(request));path=Path(request['destination'])/native_size.NAME
    value=json.loads(path.read_text())
    if case=='extra':value['claimed_authority']=True
    elif case=='rows':value['rows']-=1
    elif case=='bins':value['bins']=request['grid']['bins']+2
    elif case=='numeric':value['numeric_bytes']-=1
    elif case=='reserve':value['reservation_bytes']=1
    else:value[case+'_sha256']='f'*64
    write_json(path,value)
    with pytest.raises(ValueError):native_size.read(request,package_signature())


def test_size_record_cannot_arrive_through_a_symlink_or_oversized_file(tmp_path):
    request,_=inputs(tmp_path);root=Path(request['destination']);root.mkdir()
    native_size.publish(request,**measured(request));path=root/native_size.NAME
    elsewhere=tmp_path/'elsewhere.json';path.replace(elsewhere);path.symlink_to(elsewhere)
    with pytest.raises((OSError,ValueError)):native_size.read(request,package_signature())
    path.unlink();path.write_bytes(b'x'*(native_size.MAX_RECORD_BYTES+1))
    with pytest.raises(ValueError):native_size.read(request,package_signature())


@pytest.mark.parametrize('rows,bins,modes,cells',[(17,101,1,1),(41,1000,8000,6)])
def test_incompressible_array_inventory_and_maximum_manifests_fit_reservation(rows,bins,modes,cells):
    rng=np.random.default_rng(913)
    a=dict(q=np.arange(bins,dtype=np.float64),theory_intensity=rng.normal(size=(rows,bins)),
        background=rng.normal(size=bins),parent_intensity=rng.normal(size=(rows,bins)),
        parent_background=rng.normal(size=(rows,bins)),parent_a=rng.normal(size=(rows,6)),
        parameters=rng.normal(size=(rows,modes+cells)),trajectory=np.arange(rows,dtype=np.int64),
        role=np.ones(rows,np.int8),geometry_groups=np.ones((rows,2),np.int32),
        ambiguous_geometry=np.zeros(rows,bool),displacement_rms=rng.random(rows),relative_metric_norm=rng.random(rows))
    size=native_size.paired_array_bytes(rows,bins,modes,cells)
    assert sum(v.nbytes for v in a.values())==size
    stream=io.BytesIO();np.savez_compressed(stream,**encode(a))
    # Two existing 8 MiB metadata limits plus the complete 64 KiB task log.
    assert len(stream.getvalue())+2*batch.MAX_METADATA+65536<min(native_size.EXAMPLE_RESERVE,size+native_size.OVERHEAD)


def example_plan(root):
    example,template=inputs(root/'fixture');captured=root/'capture';captured.mkdir()
    for role in ('source','instrument','child'):shutil.copy2(example[role],captured/Path(example[role]).name)
    tasks=[]
    for i in range(4):
        key=format(i,'024x');payload=copy.deepcopy(example)
        for role in ('source','instrument','child'):payload[role]=Path(example[role]).name
        payload['destination']='tasks/'+key+'/attempt-1/compiled'
        tasks.append(dict(entry=key,attempt=1,kind='structural-examples',request=payload))
    write_json(captured/'plan.json',dict(schema=batch.SCHEMA,tasks=tasks))
    return dict(source=str(captured/'parent.cif'),plan=str(captured/'plan.json'),
        destination=str(root/'outputs'),workers=4,seconds=30),template


@pytest.mark.parametrize('report',['valid','absent','foreign','large','uploaded'])
def test_only_valid_owned_small_sizes_unlock_granted_workers(tmp_path,monkeypatch,report):
    from gpx_recipe.symmetry.resources import allocation
    monkeypatch.setattr(batch,'allocation',lambda requested,**kwargs:
        allocation(requested,cgroup=tmp_path/'no-cgroup',**kwargs))
    request,template=example_plan(tmp_path);guard=threading.Lock();active=peak=0;barrier=threading.Barrier(4)
    def runner(task,payload,scratch,**kwargs):
        nonlocal active,peak
        root=Path(payload['destination']);root.mkdir(parents=True)
        with guard:active+=1;peak=max(peak,active)
        try:
            if report=='uploaded':
                dimensions=measured(payload)
                record=dict(schema=native_size.SCHEMA,request_sha256=digest(payload),
                    implementation_sha256=package_signature(),**dimensions,
                    reservation_bytes=dimensions['numeric_bytes']+native_size.OVERHEAD)
                # Even a byte-valid, matching input hint is not a measurement
                # from the newly created native output namespace.
                write_json(Path(request['plan']).parent/'uploaded-hints'/task['entry']/native_size.NAME,record)
            elif report!='absent':
                native_size.publish(payload,**measured(payload,modes=1100000 if report=='large' else None))
                if report=='foreign':
                    path=root/native_size.NAME;record=json.loads(path.read_text());record['request_sha256']='f'*64;write_json(path,record)
            if report=='valid':barrier.wait(timeout=5)
            else:time.sleep(.25)
            for name in ('examples.json','examples.npz'):shutil.copy2(template/name,root/name)
            result=publish_result('structural-examples',payload,dict(seconds=.1),['examples.json','examples.npz'])
            return dict(state='prepared',reason='',result=result)
        finally:
            with guard:active-=1
    result=batch.execute(request,runner=runner,environ={'GALAXY_SLOTS':'4','GALAXY_MEMORY_MB':'8192'})
    assert result['all_prepared'] and result['counts']['prepared']==4
    assert result['allocation']['workers']==4
    assert peak==(4 if report=='valid' else 2)
    if report=='valid':assert result['allocation']['native_sized_tasks']>=2
    elif report=='foreign':assert result['allocation']['native_size_reports_rejected']>=1
    if report in {'valid','absent','uploaded'}:
        assert result['allocation']['native_size_reports_rejected']==0
    assert result['allocation']['peak_reserved_result_bytes']<=batch.RESULT_BUDGET
    assert not any(native_size.NAME in name for name in result['artifacts'])


def test_final_artifact_check_also_enforces_the_admitted_bound(tmp_path):
    request,template=example_plan(tmp_path);task=json.loads(Path(request['plan']).read_text())['tasks'][0]
    payload=task['request']
    for role in ('source','instrument','child'):payload[role]=str(Path(request['plan']).parent/payload[role])
    payload['destination']=str(tmp_path/'compiled');root=Path(payload['destination']);root.mkdir()
    for name in ('examples.json','examples.npz'):shutil.copy2(template/name,root/name)
    result=publish_result('structural-examples',payload,dict(seconds=.1),['examples.json','examples.npz'])
    with pytest.raises(ValueError,match='artifact budget'):
        batch.verify_result(task,payload,dict(result=result),maximum_bytes=65537)


def test_sizing_record_cannot_change_after_it_granted_concurrency(tmp_path,monkeypatch):
    from gpx_recipe.symmetry.resources import allocation
    monkeypatch.setattr(batch,'allocation',lambda requested,**kwargs:
        allocation(requested,cgroup=tmp_path/'no-cgroup',**kwargs))
    request,template=example_plan(tmp_path);barrier=threading.Barrier(4)
    def runner(task,payload,scratch,**kwargs):
        root=Path(payload['destination']);root.mkdir(parents=True)
        native_size.publish(payload,**measured(payload))
        barrier.wait(timeout=5)
        # The first pair must have released space for the second pair. Even
        # a whitespace-only replacement invalidates that owned measurement.
        path=root/native_size.NAME;path.write_bytes(path.read_bytes()+b' ')
        for name in ('examples.json','examples.npz'):shutil.copy2(template/name,root/name)
        result=publish_result('structural-examples',payload,dict(seconds=.1),['examples.json','examples.npz'])
        return dict(state='prepared',reason='',result=result)
    result=batch.execute(request,runner=runner,environ={'GALAXY_SLOTS':'4','GALAXY_MEMORY_MB':'8192'})
    assert result['allocation']['peak_admitted_tasks']==4
    assert result['counts']['failed']>=2 and not result['all_prepared']
    assert sum(result['counts'].values())==4 and result['counts']['pending']==0
    assert all('verification' in row['reason'] for row in result['entries'] if row['state']=='failed')
