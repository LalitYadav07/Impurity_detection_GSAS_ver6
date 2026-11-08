"""Closed structural dispatch and immutable complete-model capture, no GSAS."""
import copy
import asyncio
import hashlib
import io
import json
from pathlib import Path
import zipfile
import time
from types import SimpleNamespace

import pytest

from gpx_recipe.execution import write_json
from gpx_recipe.structural_live.job import validate_payload,reviewed_job
from gpx_recipe.symmetry.compiler import compile_native_bundle,file_sha,package_signature
from gpx_recipe.symmetry.contracts import digest,validate_candidate,validate_protocol
from gpx_recipe.symmetry.native_protocol import validate_native_payload
from radar_pd_nova.webmon.structural_runtime import StructuralSnapshotRuntime
from test_model_runtime import remote

RAW=b'# X , Y , E Distribution=true\n0\n1,-10,2\n2,20,3\n3,30,4\n'


def captured(tmp_path,source):
    root=tmp_path/'handoff';inputs=root/'model-inputs';inputs.mkdir(parents=True)
    (inputs/'source.gpx').write_bytes(Path(source['source_path']).read_bytes())
    (inputs/'reference.gpx').write_bytes(b'opaque separately reviewed parent structure')
    (inputs/'child.cif').write_bytes(b'public fixture for capture only')
    candidate=validate_candidate(dict(id='struct:public',child=dict(path='child.cif',sha256=file_sha(inputs/'child.cif')),
        modes=['mode-a','mode-b'],transformation_matrix=[[-1.,0,0],[0,-1.,0],[0,0,1.]],origin_shift=[0,0,0]))
    write_json(inputs/'catalog.json',dict(schema='radar-symmetry-catalog/v1',parent_setting_sha256='a'*64,
        k=['0','0','0'],candidates=[candidate]))
    protocol=validate_protocol(dict(kind='structural',source=dict(path='source.gpx',sha256=file_sha(inputs/'source.gpx')),
        structural_reference=dict(path='reference.gpx',sha256=file_sha(inputs/'reference.gpx')),
        catalog=dict(path='catalog.json',sha256=file_sha(inputs/'catalog.json')),parent_phase='Public parent',
        histogram='PWDR public',parent_setting_sha256='a'*64,magnetic_function_sha256='b'*64,
        implementation_sha256=package_signature(),k=['0','0','0'],sites=[],candidates=[candidate],
        seeds=[dict(id='prepared')],policy=dict(cycles=20,timeout_seconds=180,maximum_trials=1,maximum_cell_multiplier=8,
            maximum_moment=6.,fit_band_percent=.25,maximum_mode=.5,adequate_rwp_percent=12.,refine_cell=True,
            refine_background=True,refine_histogram_scale=True,structural_optimizer='prepared_nonlinear',
            structural_starts=2,structural_seconds=4.,structural_mode_radius=.05,structural_cell_radius=.005)))
    model=root/'model'
    compile_native_bundle(protocol,inputs,model,candidate_id=candidate['id'],seed_id='prepared',approved_hash=digest(protocol))
    (root/'anchor.dat').write_bytes(RAW);raw=RAW.replace(b'-10',b'-12');(root/'current.dat').write_bytes(raw)
    identity=dict(facility='SNS',instrument='PG3',ipts=1,run=2,epoch='public-control')
    mapping=dict(mapping_kind='user_reviewed',source_gpx_sha256=source['source_sha256'],
        snapshot_sha256=file_sha(root/'anchor.dat'),physical_bank=1,identity=identity,
        export_geometry=dict(kind='linear_tof_from_d',microseconds_per_angstrom=100.),
        normalization='Reviewed public density',producer_config_sha256='c'*64)
    payload=dict(source=str(model/'source.gpx'),directory=str(model),anchor_snapshot=str(root/'anchor.dat'),
        snapshot=str(root/'current.dat'),receipt=dict(identity=identity,content_sha256=hashlib.sha256(raw).hexdigest()),
        mapping=mapping,source_scope=dict(run='2',sample_id='public',sample='Public control',wavelength='1.5'),
        revision=2,source_revision='PG3_2:'+mapping['snapshot_sha256'],
        reference_revision='PG3_ref:'+source['source_sha256'],refine_transfer=False,destination=str(root/'output'))
    return root,payload,protocol


def test_structural_dispatch_uses_existing_durable_model_transport_and_captures_all_assets(remote,tmp_path):
    runtime,service,source=remote;root,payload,protocol=captured(tmp_path,source)
    adapter=StructuralSnapshotRuntime(runtime)
    handle=adapter.submit(payload,root,source=source,timeout=60)
    assert handle['request']['operation']=='structural-snapshot'
    expected={'model/job.json','model/source.gpx','model/reference.gpx','model/catalog.json','model/child.cif',
              'anchor.dat','current.dat'}
    assert set(handle['request']['input_hashes'])==expected
    assert handle['request']['payload']['directory']=='model'
    assert adapter.submit(payload,root,source=source,timeout=60)==handle
    assert len(service.submissions)==1
    packet=service.data[handle['asset_id']]
    with zipfile.ZipFile(io.BytesIO(packet)) as archive:
        assert 'current.dat' in archive.namelist() and 'model/reference.gpx' in archive.namelist()
        assert 'model/source.gpx' not in archive.namelist()  # separately supplied immutable HDA
    assert 'PRIVATE-' not in json.dumps(handle)
    assert reviewed_job(root/'model')[1]==protocol


def test_structural_dispatch_carries_the_sealed_optional_preparation_snapshot(remote,tmp_path):
    from gpx_recipe.preparation import structural_hints
    runtime,_,source=remote;root,payload,protocol=captured(tmp_path,source)
    document=dict(schema=structural_hints.SCHEMA,protocol_sha256=digest(protocol),library_id='a'*24,
        library_name='Public',entries=[],eligible_entries=0,omitted_entries=0,observations_included=False)
    document['snapshot_sha256']=digest(document)
    path=root/'model'/structural_hints.MANIFEST;path.parent.mkdir();write_json(path,document)
    handle=StructuralSnapshotRuntime(runtime).submit(payload,root,source=source,timeout=60)
    assert handle['request']['payload']['prepared_scattering']==document
    assert 'model/'+structural_hints.MANIFEST in handle['request']['input_hashes']


@pytest.mark.parametrize('problem',[OSError('Result download interrupted'),ValueError('Publication needs verification')])
def test_finished_snapshot_with_uncollected_result_recovers_the_same_galaxy_job(remote,tmp_path,monkeypatch,problem):
    from radar_pd_nova.webmon.structural_stream import analyse_snapshot
    from radar_pd_nova.webmon.stream import ReviewRequired
    runtime,service,source=remote;root,payload,_=captured(tmp_path,source)
    folder=root/'revision';folder.mkdir()
    item=dict(payloads=[payload],revision=payload['revision'],receipt=payload['receipt'],
        captured_monotonic=time.monotonic(),folder=str(folder))
    adapter=StructuralSnapshotRuntime(runtime);collections=[]
    def collect(handle,directory):
        collections.append(handle['key'])
        if len(collections)==1:raise problem
        return dict(result=dict(metrics=dict(rwp_native=2.))),1.
    monkeypatch.setattr(adapter,'collect',collect)
    with pytest.raises(ReviewRequired,match='unresolved'):
        asyncio.run(analyse_snapshot(adapter,item,root,source=source,poll_seconds=.001))
    saved=json.loads((folder/'progress.json').read_text())
    assert saved['terminal']==0 and saved['rows'][0]['state']=='needs_recovery'
    assert 'Galaxy job finished; result collection needs recovery' in saved['rows'][0]['reason']
    assert len(service.submissions)==1
    rows=asyncio.run(analyse_snapshot(adapter,item,root,source=source,poll_seconds=.001))
    assert rows[0]['state']=='verified_point_fit'
    assert collections[0]==collections[1] and len(service.submissions)==1
    assert json.loads((folder/'progress.json').read_text())['terminal']==1


def test_structural_native_dispatch_uses_the_existing_shared_model_input_layout(remote,tmp_path):
    from gpx_recipe.symmetry.compiler import prepare_shared_inputs,compile_shared_job
    from gpx_recipe.symmetry.contracts import trial_id
    runtime,_,source=remote;root,payload,protocol=captured(tmp_path,source)
    shared=prepare_shared_inputs(protocol,root/'model-inputs',root/'models')
    child=protocol['candidates'][0];directory=root/'models/trials'/trial_id(protocol,child['id'],'prepared')
    compile_shared_job(protocol,directory,candidate_id=child['id'],seed_id='prepared',approved_hash=digest(protocol))
    payload.update(directory=str(directory),source=str(shared/'source.gpx'))
    handle=StructuralSnapshotRuntime(runtime).submit(payload,root,source=source,timeout=60)
    names=set(handle['request']['input_hashes'])
    assert str((shared/'source.gpx').relative_to(root)) in names
    assert str((shared/'reference.gpx').relative_to(root)) in names
    assert str((directory/'job.json').relative_to(root)) in names
    assert not any(name.endswith('.py') for name in names)
    assert not any(name.startswith(str(directory.relative_to(root))+'/') and name.endswith('.gpx') for name in names)


@pytest.mark.parametrize('change',['extra','revision','transfer','mapping','anchor_revision','reference_revision','receipt'])
def test_closed_structural_payload_rejects_changed_review_or_arbitrary_controls(remote,tmp_path,change):
    _,_,source=remote;_,payload,_=captured(tmp_path,source)
    if change=='extra':payload['environment']={'arbitrary':'code'}
    if change=='revision':payload['revision']=True
    if change=='transfer':payload['refine_transfer']='yes'
    if change=='mapping':payload['mapping']['mapping_kind']='guessed'
    if change=='anchor_revision':payload['source_revision']='PG3_2:'+'a'*64
    if change=='reference_revision':payload['reference_revision']='PG3_1:'+'a'*64
    if change=='receipt':payload['receipt']['identity']=dict(payload['receipt']['identity'],run=3)
    with pytest.raises(ValueError):validate_native_payload('structural-snapshot',payload)


@pytest.mark.parametrize('change',['snapshot','anchor','child','scope','source','timeout'])
def test_structural_capture_rejects_invalid_inputs_before_upload(remote,tmp_path,change):
    runtime,service,source=remote;root,payload,_=captured(tmp_path,source);timeout=60
    if change in ('snapshot','anchor'):
        Path(payload['snapshot' if change=='snapshot' else 'anchor_snapshot']).write_bytes(b'changed')
    if change=='child':(root/'model/child.cif').write_bytes(b'changed')
    if change=='scope':
        job=json.loads((root/'model/job.json').read_text());job['protocol']['policy']['structural_seconds']=5.
        write_json(root/'model/job.json',job)
    if change=='source':source=dict(source,source_sha256='d'*64)
    if change=='timeout':timeout=5
    with pytest.raises(ValueError):StructuralSnapshotRuntime(runtime).submit(payload,root,source=source,timeout=timeout)
    assert not service.submissions and not service.data


@pytest.mark.parametrize('change',[None,'revision','candidate','settings','likelihood','gpx','histogram','output'])
def test_structural_collection_checks_source_revision_artifacts_and_display(remote,tmp_path,change):
    runtime,_,source=remote;root,payload,protocol=captured(tmp_path,source)
    from radar_pd_nova.native_operations import capture
    handle,_=capture(runtime,'structural-snapshot',payload,root,source,60)
    job=json.loads((root/'model/job.json').read_text());output=root/'output';output.mkdir()
    fitted=dict(snapshot_sha256=payload['receipt']['content_sha256'],snapshot_revision=2,
        previous_likelihood_accumulated=False,all_physical_modes_retained=True,optimization_qualified=False)
    settings=dict(mode_radius=.05,cell_radius=.005,seconds=4.,starts=2,recovery_seconds=0.,background_scale=None)
    fitted['settings']=copy.deepcopy(settings)
    for role in ('frozen','continuation'):
        path=output/(role+'.gpx');path.write_bytes(b'opaque verified-publication fixture')
        fitted[role]=dict(path=str(path),sha256=file_sha(path))
    view=output/'view.json';write_json(view,dict(schema='radar-symmetry-view/v1',histogram=protocol['histogram']))
    from gpx_recipe.structural_live.acquisition import _mapping
    result=dict(schema='radar-structural-snapshot-result/v1',kind='structural_point_fit',state='verified_point_fit',
        optimization_status='optimization_unresolved',protocol_sha256=digest(protocol),trial_id=job['id'],
        candidate_id=job['candidate_id'],implementation_sha256=package_signature(),source_sha256=source['source_sha256'],
        child_sha256=protocol['candidates'][0]['child']['sha256'],reference_sha256=protocol['structural_reference']['sha256'],
        acquisition=digest([_mapping(payload['mapping']),payload['source_scope']]),anchor_sha256=payload['mapping']['snapshot_sha256'],
        snapshot_sha256=payload['receipt']['content_sha256'],revision=2,forecast_enabled=False,
        model_exclusion_authorized=False,statistical_policy_qualified=False,settings=settings,
        result=fitted,output=str(output),view=str(view))
    if change=='revision':result['revision']=1
    if change=='candidate':result['candidate_id']='struct:other'
    if change=='settings':fitted['settings']['starts']=3
    if change=='likelihood':fitted['previous_likelihood_accumulated']=True
    if change=='gpx':Path(fitted['continuation']['path']).write_bytes(b'changed')
    if change=='histogram':write_json(view,dict(schema='radar-symmetry-view/v1',histogram='PWDR other bank'))
    if change=='output':result['output']=str(root.parent)
    restored=[]
    adapter=StructuralSnapshotRuntime(SimpleNamespace(restore_native_inputs=lambda h,r:restored.append(h),
        collect_native=lambda h,r:(result,3.5)))
    if change:
        with pytest.raises(ValueError):adapter.collect(handle,root)
    else:
        assert adapter.collect(handle,root)==(result,3.5)
        assert restored==[handle]
