"""Four allocated CPUs through the actual Live snapshot worker boundary."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

G = pytest.importorskip('GSASII.GSASIIscriptable')
from test_structural_snapshot_job_native import snapshot_case
from gpx_recipe.execution import write_json
from gpx_recipe.live.snapshot import parse_live_ascii, uniform_density_to_tof
from gpx_recipe.structural_live.native import sha
from gpx_recipe.symmetry.bundle import pack, unpack
from gpx_recipe.symmetry.compiler import package_signature
from gpx_recipe.symmetry.contracts import digest


def test_live_snapshot_uses_four_native_starts_and_preserves_its_publication(tmp_path,monkeypatch):
    monkeypatch.setenv('GALAXY_SLOTS','4')
    monkeypatch.setenv('GALAXY_MEMORY_MB','8192')
    case=snapshot_case(tmp_path,True,starts=4)
    hashes={str(path.relative_to(case.packet)):sha(path)
            for path in case.packet.rglob('*') if path.is_file() and path.suffix!='.py'}
    request=dict(schema='radar-model-native/v1',operation='structural-snapshot',
        payload=case.payload,implementation_sha256=package_signature(),
        expected_source_sha256=case.source_sha,input_hashes=hashes,
        timeout=180,dispatch_key='d'*24)
    pack(case.packet,tmp_path/'assets.zip',
         [name for name in hashes if name!=case.payload['source']])
    write_json(tmp_path/'request.json',request)
    script=Path(__file__).resolve().parents[2]/'scripts/ndip_model_campaign.py'
    with (tmp_path/'wrapper.log').open('w') as log:
        process=subprocess.run([sys.executable,str(script),'native',str(case.source),
            str(tmp_path/'assets.zip'),str(tmp_path/'request.json'),str(tmp_path/'worker'),
            str(tmp_path/'result.zip'),'--preparation-certificate',str(tmp_path/'certificate.json')],
            stdout=log,stderr=subprocess.STDOUT,timeout=210)
    assert process.returncode==0,(tmp_path/'wrapper.log').read_text()[-16000:]
    received=unpack(tmp_path/'result.zip',tmp_path/'received')
    envelope=json.loads((received/'response.json').read_text())
    result=envelope['result'];fit=result['result'];parallel=fit['result']['parallel_execution']
    assert result['protocol_sha256']==digest(case.protocol)
    assert result['snapshot_sha256']==hashlib.sha256(case.raw).hexdigest()
    assert result['reference_sha256']==case.warm_sha
    assert parallel['workers']==parallel['peak_active_workers']==4
    assert len(parallel['processes'])==4
    assert len({row['pid'] for row in parallel['processes']})==4
    assert all(row['cpu_seconds']>0 and row['allocated_slots']==1 for row in parallel['processes'])
    assert parallel['total_start_allocation_seconds']<=case.protocol['policy']['structural_seconds']
    assert fit['all_physical_modes_retained'] and not fit['optimization_qualified']
    assert fit['verification']['relative_error']<1e-8
    assert fit['verification']['max_sigma_error']<.01
    measured=uniform_density_to_tof(parse_live_ascii(case.raw),export_difc=case.factor)
    published=list((tmp_path/'worker/reviewable_projects').glob('*.gpx'))
    assert len(published)==2
    assert {sha(path) for path in published}=={fit[role]['sha256'] for role in ('frozen','continuation')}
    for path in published:
        arrays=G.G2Project(gpxfile=str(path)).histogram(0).data['data'][1]
        np.testing.assert_array_equal(arrays[0],measured['x'])
        np.testing.assert_array_equal(arrays[1],measured['y'])
        np.testing.assert_allclose(arrays[2],measured['weights'],rtol=1e-12,atol=0)
    assert sha(case.source)==case.source_sha and sha(case.warm)==case.warm_sha
    write_json(tmp_path/'acceptance.json',dict(native_workers=4,parallel=parallel,
        workflow_timings=result['workflow_timings'],rwp_percent=fit['metrics']['rwp_native'],
        source_unchanged=True,reference_unchanged=True,snapshot_arrays_preserved=True,
        scope='Public artificial Live snapshot control; no material identification claim'))
