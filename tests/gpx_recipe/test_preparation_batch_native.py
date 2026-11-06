"""Actual batched CIF-to-native preparation through the Galaxy worker wrapper."""
import json
from pathlib import Path
import subprocess
import sys
import copy

import numpy as np

import pytest

pytest.importorskip('GSASII.GSASIIscriptable')
from test_preparation_worker_native import request
from gpx_recipe.execution import write_json
from gpx_recipe.preparation.batch import stage, read_plan, inputs, RESULT_SCHEMA
from gpx_recipe.preparation.operations import create_library
from gpx_recipe.preparation.magnetic import load, ARRAYS
from gpx_recipe.symmetry.bundle import pack, unpack
from gpx_recipe.symmetry.compiler import file_sha, package_signature
from gpx_recipe.symmetry.contracts import digest


def test_two_bank_native_preparation_shares_one_job_and_exports_no_construction_gpx(tmp_path):
    banks={};models=[];parent=None
    for kind in ('PNC','PNT'):
        payload,protocol,candidate=request(tmp_path/kind,kind)
        parent=parent or payload['source']
        banks[kind]=dict(instrument=payload['instrument'],grid=payload['grid'])
        settings=dict(candidate=candidate,k=['0','0','0'],sites=protocol['sites'],maximum_moment=6.,
            refine_histogram_scale=True,refine_background=True)
        models.append(dict(kind='magnetic',bank=kind,settings=settings))
    library=create_library(tmp_path/'library',name='Public two-bank batch',source=parent,banks=banks,models=models)
    entries=[(row['id'],library.start(row['id'])) for row in list(library.document['entries'])]
    captured=tmp_path/'captured'
    payload=stage(library,entries,captured,workers=4,seconds=120)
    plan=read_plan(captured/'plan.json')
    assert len(inputs(plan))==3  # one parent plus two independent profiles
    payload['source']=str(Path(payload['source']).relative_to(captured))
    payload['plan']='plan.json';payload['destination']='prepared'
    names=inputs(plan)+['plan.json'];pack(captured,tmp_path/'inputs.zip',names)
    source=captured/payload['source']
    message=dict(schema='radar-model-native/v1',operation='preparation-batch',payload=payload,
        implementation_sha256=package_signature(),expected_source_sha256=file_sha(source),
        dispatch_key='d'*24,timeout=150,input_hashes={name:file_sha(captured/name) for name in names})
    write_json(tmp_path/'request.json',message)
    script=Path(__file__).resolve().parents[2]/'scripts/ndip_model_campaign.py'
    with (tmp_path/'wrapper.log').open('w') as stream:
        subprocess.run([sys.executable,str(script),'native',str(source),str(tmp_path/'inputs.zip'),
            str(tmp_path/'request.json'),str(tmp_path/'job'),str(tmp_path/'result.zip')],
            check=True,timeout=180,stdout=stream,stderr=subprocess.STDOUT)
    received=unpack(tmp_path/'result.zip',tmp_path/'received')
    envelope=json.loads((received/'response.json').read_text());result=envelope['result']
    assert result['schema']==RESULT_SCHEMA and result['status']=='complete'
    assert result['plan_sha256']==digest(plan) and len(result['entries'])==2
    assert result['allocation']['requested_workers']==4
    assert 1<=result['allocation']['workers']<=2
    assert all(row['state']=='prepared' for row in result['entries'])
    assert result['observations_included'] is False and result['scientific_result'] is False
    assert not any(name.endswith('.gpx') or '/.work/' in name for name in envelope['files'])
    assert all(name.startswith('native/prepared/') for name in envelope['files'])
    parallel_responses={}
    for row in result['entries']:
        native_result=row['result']
        relative=plan['tasks'][next(i for i,t in enumerate(plan['tasks']) if t['entry']==row['entry'])]['request']['destination']
        output=received/'native/prepared'/relative
        manifest=json.loads((output/'entry.json').read_text())
        response=load(output/'response.npz',physics_sha256=manifest['detail']['physics_sha256'])
        parallel_responses[row['entry']]=response
        assert response._native_seal is None and response.quadratic.shape[1]==3
        assert file_sha(output/'entry.json')==native_result['entry_sha256']
    # Compare the identical finite work at one and two allocated process slots.
    # Timing is reported, not asserted: host load can change the speed ratio.
    serial=copy.deepcopy(message);serial['payload']['workers']=1;serial['dispatch_key']='e'*24
    write_json(tmp_path/'serial-request.json',serial)
    with (tmp_path/'serial-wrapper.log').open('w') as stream:
        subprocess.run([sys.executable,str(script),'native',str(source),str(tmp_path/'inputs.zip'),
            str(tmp_path/'serial-request.json'),str(tmp_path/'serial-job'),str(tmp_path/'serial-result.zip')],
            check=True,timeout=180,stdout=stream,stderr=subprocess.STDOUT)
    restored=unpack(tmp_path/'serial-result.zip',tmp_path/'serial-received')
    serial_result=json.loads((restored/'response.json').read_text())['result']
    assert serial_result['all_prepared'] and serial_result['allocation']['workers']==1
    for row in serial_result['entries']:
        relative=next(t['request']['destination'] for t in plan['tasks'] if t['entry']==row['entry'])
        output=restored/'native/prepared'/relative
        manifest=json.loads((output/'entry.json').read_text())
        response=load(output/'response.npz',physics_sha256=manifest['detail']['physics_sha256'])
        previous=parallel_responses[row['entry']]
        assert response.physics_sha256==previous.physics_sha256
        for name in ARRAYS:np.testing.assert_allclose(getattr(response,name),getattr(previous,name),rtol=1e-12,atol=1e-12)
    write_json(tmp_path/'batch-scaling.json',dict(
        scope='Two public artificial PNC/PNT preparation contexts, excluding Galaxy scheduling.',
        serial_seconds=serial_result['seconds'],parallel_seconds=result['seconds'],
        serial_workers=serial_result['allocation']['workers'],parallel_workers=result['allocation']['workers'],
        ratio=serial_result['seconds']/result['seconds'],identical_scientific_arrays=True))
