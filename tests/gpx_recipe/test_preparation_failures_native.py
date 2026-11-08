"""Real scope rejection stays actionable through batch and Galaxy wrappers."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from gpx_recipe.execution import write_json
from gpx_recipe.preparation import batch,failures
from gpx_recipe.preparation.structural_training_blocks import REPRESENTATION
from gpx_recipe.symmetry.bundle import pack,unpack
from gpx_recipe.symmetry.compiler import file_sha,package_signature
from test_native_profile_capture import native_source


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_scope_conflict_preserves_all_cases_and_reaches_the_actual_native_wrapper(tmp_path,kind):
    source,_=native_source(tmp_path/'native',kind)
    captured=tmp_path/'captured';captured.mkdir()
    parent=captured/'parent.cif';G.G2Project(gpxfile=str(source)).phase(0).export_CIF(str(parent))
    import shutil
    shutil.copy2(tmp_path/'native/profile.instprm',captured/'profile.instprm')
    shutil.copy2(Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif',captured/'child.cif')
    child=captured/'child.cif'
    grid=dict(kind=kind,lower=12. if kind=='PNC' else 3000.,
        upper=125. if kind=='PNC' else 40000.,bins=701,bank=1)
    request=dict(source='parent.cif',instrument='profile.instprm',child='child.cif',grid=grid,
        settings=dict(candidate=dict(id='public-ersi',transformation_matrix=[[-1.,0,0],[0,-1.,0],[0,0,1.]],
            origin_shift=[0,0,0],child=dict(path='child.cif',sha256=file_sha(child))),
            mode_radius=.05,cell_radius=.00001,seed=1947,parent_seed=8341,
            parent_relative_metric_range=[.01,.02],representation=REPRESENTATION,
            trajectories=3,steps=2,trajectory_slice=[0,3]),
        destination='tasks/'+'a'*24+'/attempt-1/compiled')
    reference=dict(source='parent.cif',instrument='profile.instprm',grid=grid,
        settings=dict(maximum_denominator=8),destination='tasks/'+'b'*24+'/attempt-1/compiled')
    plan=batch.validate_plan(dict(schema=batch.SCHEMA,tasks=[
        dict(entry='a'*24,attempt=1,kind='structural-examples',request=request),
        dict(entry='b'*24,attempt=1,kind='reference',request=reference)]))
    write_json(captured/'plan.json',plan)
    names=batch.inputs(plan)+['plan.json'];pack(captured,tmp_path/'inputs.zip',names)
    payload=dict(source='parent.cif',plan='plan.json',destination='prepared',workers=2,seconds=120)
    def message(operation,payload):
        return dict(schema='radar-model-native/v1',operation=operation,payload=payload,
            implementation_sha256=package_signature(),expected_source_sha256=file_sha(parent),
            dispatch_key='e'*24,timeout=150,input_hashes={name:file_sha(captured/name) for name in names})
    wrapper=Path(__file__).resolve().parents[2]/'scripts/ndip_model_campaign.py'
    write_json(tmp_path/'batch-request.json',message('preparation-batch',payload))
    with (tmp_path/'batch-wrapper.log').open('w') as log:
        subprocess.run([sys.executable,str(wrapper),'native',str(parent),str(tmp_path/'inputs.zip'),
            str(tmp_path/'batch-request.json'),str(tmp_path/'batch-job'),str(tmp_path/'batch-result.zip')],
            check=True,timeout=180,stdout=log,stderr=subprocess.STDOUT)
    received=unpack(tmp_path/'batch-result.zip',tmp_path/'received')
    result=json.loads((received/'response.json').read_text())['result']
    assert result['status']=='complete' and not result['all_prepared']
    assert result['counts']==dict(prepared=1,failed=1,cancelled=0,pending=0)
    assert result['entries'][0]['reason']==failures.REASONS['parent_cell_domain_conflict']
    assert result['entries'][1]['state']=='prepared'
    assert [row['entry'] for row in result['entries']]==['a'*24,'b'*24]
    assert not list(received.rglob('*.gpx'))
    single=copy.deepcopy(request);single['destination']='rejected'
    write_json(tmp_path/'single-request.json',message('preparation-structural-examples',single))
    with (tmp_path/'single-wrapper.log').open('w') as log:
        process=subprocess.run([sys.executable,str(wrapper),'native',str(parent),str(tmp_path/'inputs.zip'),
            str(tmp_path/'single-request.json'),str(tmp_path/'single-job'),str(tmp_path/'single-result.zip')],
            timeout=180,stdout=log,stderr=subprocess.STDOUT)
    assert process.returncode!=0
    assert failures.REASONS['parent_cell_domain_conflict'] in (tmp_path/'single-wrapper.log').read_text()
    assert json.loads((captured/'plan.json').read_text())==plan
