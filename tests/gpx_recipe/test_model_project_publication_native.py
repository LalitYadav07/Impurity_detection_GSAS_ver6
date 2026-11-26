"""Actual model wrapper exports reviewable structural GPXs without promotion."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from test_structural_snapshot_job_native import snapshot_case
from gpx_recipe.execution import write_json
from gpx_recipe.symmetry.bundle import pack,unpack,DATA_SUFFIXES
from gpx_recipe.symmetry.compiler import file_sha,job_inputs
from gpx_recipe.symmetry.contracts import digest,trial_id


def test_structural_model_wrapper_publishes_the_checked_continuation_separately(tmp_path,monkeypatch):
    monkeypatch.setenv('GALAXY_SLOTS','4');monkeypatch.setenv('GALAXY_MEMORY_MB','8192')
    case=snapshot_case(tmp_path,True,starts=4)
    uid=trial_id(case.protocol,case.candidate['id'],'prepared')
    model=case.packet/'models/trials'/uid
    captured=job_inputs(model,json.loads((model/'job.json').read_text()))
    paths=[str(p.relative_to(captured)) for p in captured.rglob('*')
           if p.is_file() and p.suffix in DATA_SUFFIXES and p.name!='source.gpx']
    pack(captured,tmp_path/'assets.zip',paths)
    request=dict(schema='radar-model-chunk/v1',protocol=case.protocol,
        protocol_sha256=digest(case.protocol),trials=[uid],workers=4,seconds=240,dispatch_key='e'*24)
    write_json(tmp_path/'request.json',request)
    script=Path(__file__).resolve().parents[2]/'scripts/ndip_model_campaign.py'
    with (tmp_path/'wrapper.log').open('w') as log:
        process=subprocess.run([sys.executable,str(script),'chunk',str(case.source),
            str(tmp_path/'assets.zip'),str(tmp_path/'request.json'),str(tmp_path/'worker'),
            str(tmp_path/'result.zip')],stdout=log,stderr=subprocess.STDOUT,timeout=250)
    assert process.returncode==0,(tmp_path/'wrapper.log').read_text()[-15000:]
    received=unpack(tmp_path/'result.zip',tmp_path/'received')
    result=json.loads((received/'response.json').read_text())
    assert result['trials'][0]['status']=='optimization_unresolved'
    receipt=json.loads((received/'trials'/uid/'result.json').read_text())
    assert receipt['effects_verified'] and receipt['audit']['passed'] and receipt['reviewable_gpx']
    assert not list((tmp_path/'worker/verified_projects').glob('*.gpx'))
    published=list((tmp_path/'worker/reviewable_projects').glob('*.gpx'))
    assert [p.name for p in published]==['reviewable_'+uid+'.gpx']
    expected=receipt['files']['results/'+receipt['gpx']]
    assert file_sha(published[0])==expected==file_sha(received/'trials'/uid/'results'/receipt['gpx'])
    assert G.G2Project(gpxfile=str(published[0])).histogram(0).name==case.protocol['histogram']
    assert file_sha(case.source)==case.source_sha
