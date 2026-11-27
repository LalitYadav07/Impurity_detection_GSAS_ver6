"""Exercise the actual Galaxy Recipe command against the installed package.

The only structure and pattern are generated here; no experimental inputs.
Run with only this test file mounted, so a source checkout cannot repair imports.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np

from gpx_recipe.compiler import approve
from gpx_recipe.execution import freeze
from gpx_recipe.symmetry.compiler import package_signature

CLI=Path('/opt/radar-pd/scripts/ndip_refinement.py')
CANONICAL=Path('/opt/radar-symmetry/gpx_recipe')


def invoke(action,source,request,root):
    request_path=root/(action+'-request.json');request_path.write_text(json.dumps(request))
    directory=root/action;archive=root/(action+'.zip')
    command=[sys.executable,str(CLI),action,str(source),str(request_path),str(directory),str(archive)]
    started=time.monotonic()
    completed=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=90)
    assert completed.returncode==0,completed.stdout+completed.stderr
    with zipfile.ZipFile(archive) as bundle:
        response=json.loads(bundle.read('response.json'))
        assert response['action']==action and response['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
        if action=='run':assert response['final_gpx'] in bundle.namelist()
    return response,time.monotonic()-started


def test_galaxy_absolute_recipe_cli_uses_canonical_source_and_native_stages(tmp_path):
    assert CLI.is_file() and CANONICAL.is_dir()
    assert hashlib.sha256(CLI.read_bytes()).hexdigest()==os.environ['RADAR_EXPECTED_RECIPE_LAUNCHER_SHA']
    # Python puts the absolute script's directory first. Check that exact
    # precedence independently of the test process's own module imports.
    code="""import sys,json
sys.path.insert(0,'/opt/radar-pd/scripts')
import gpx_recipe.worker as worker
import gpx_recipe.contracts as contracts
from gpx_recipe.symmetry.compiler import package_signature
print(json.dumps({'worker':worker.__file__,'contracts':contracts.__file__,'signature':package_signature()}))
"""
    probe=subprocess.run([sys.executable,'-c',code],cwd=tmp_path,capture_output=True,text=True,check=True,timeout=30)
    identity=json.loads(probe.stdout.strip().splitlines()[-1])
    for key in ('worker','contracts'):assert Path(identity[key]).resolve().is_relative_to(CANONICAL.resolve())
    assert identity['signature']==package_signature()==os.environ['RADAR_EXPECTED_RECIPE_SIGNATURE']

    from GSASII import GSASIIscriptable as G
    from GSASII import GSASIIstrMain as main
    instrument=tmp_path/'synthetic.instprm'
    instrument.write_text('#GSAS-II instrument parameter file\nType:PNC\nBank:1\nLam:1.54\nZero:0\nU:.12\nV:0\nW:.08\nX:0\nY:0\nZ:0\nSH/L:.002\nPolariz.:0\nAzimuth:0\n')
    source=tmp_path/'source.gpx';project=G.G2Project(newgpx=str(source))
    phase=project.add_phase(phasename='Public artificial oxide',cell=[4.7,5.6,6.8,79,93,104])
    phase.add_atom(.37,.29,.19,'O','O1');phase.add_atom(.13,.21,.34,'Fe','Fe1')
    histogram=project.add_simulated_powder_histogram('public synthetic control',str(instrument),12,125,Npoints=601,scale=120,phases=[phase])
    freeze(project);histogram.data['data'][1][1][:]=1.;project.save()
    truth=np.asarray(main.DoNoFit(str(source),histogram.name),float)
    histogram.data['data'][1][1]=truth+.02*np.sin(np.arange(len(truth)))
    histogram.data['data'][1][2][:]=2500.;histogram.data['data'][0]['Dummy']=False
    histogram.data['Sample Parameters']['Scale'][0]*=.8;project.save()
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    inspected,inspect_seconds=invoke('inspect',source,{'expected_source_sha256':before},tmp_path)
    assert inspected['status']=='inspected';context=inspected['context']
    plan=dict(schema='radar-gpx-recipe/v1',model_signature=context['model_signature'],source_sha256=before,
        failure_policy='continue_independent',request='Public CLI runtime alignment control',assumptions=[],questions=[],unsupported=[],
        stages=[dict(id='scale',title='Scale only',cycles=20,constraints=[],free=[dict(kind='histogram_scale',histogram=context['histograms'][0]['key'])])])
    request=dict(plan=plan,expected_source_sha256=before)
    preview,preview_seconds=invoke('preview',source,request,tmp_path)
    assert preview['status']=='preview_passed'
    approval=approve(plan,context,preview['preview'],approver='Public image CLI regression')
    result,run_seconds=invoke('run',source,dict(request,approval=approval),tmp_path)
    assert result['status']=='completed' and result['rejected_stages']==[]
    assert len(result['stages'])==1 and result['stages'][0]['id']=='scale'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    print(json.dumps(dict(schema='radar-public-recipe-cli-check/v1',identity=identity,
        launcher_sha256=hashlib.sha256(CLI.read_bytes()).hexdigest(),public_source_sha256=before,
        source_unchanged=True,inspect_status=inspected['status'],preview_status=preview['status'],
        run_status=result['status'],seconds=dict(inspect=inspect_seconds,preview=preview_seconds,run=run_seconds))))


if __name__=='__main__':
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix='radar-public-recipe-cli-') as folder:
        test_galaxy_absolute_recipe_cli_uses_canonical_source_and_native_stages(Path(folder))
    print('Verified absolute Galaxy Recipe CLI: canonical source, inspect, preview, approved run, unchanged public input')
