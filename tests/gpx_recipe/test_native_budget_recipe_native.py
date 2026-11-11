"""Real CW/TOF recipes stop bounded MINPACK work and retain prior stages."""
import os
from pathlib import Path
import numpy as np
import pytest

if os.environ.get('RADAR_REQUIRE_STRUCTURAL_NATIVE') == '1':
    import GSASII.GSASIIscriptable
else:
    pytest.importorskip('GSASII.GSASIIscriptable')

from test_native_profile_capture import native_source
from gpx_recipe.compiler import approve
from gpx_recipe.snapshot import gsas, snapshot
from gpx_recipe.structural_live.native import NativeCandidate
from gpx_recipe.structural_live.optimization.refinable_gpx import export_refinable
from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.worker import work


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_budget_rejection_rolls_back_and_independent_native_stage_finishes(tmp_path,kind):
    native_source(tmp_path/'source',kind)
    native=NativeCandidate(tmp_path/'source/prepared',tmp_path/'point')
    point=native.save_verified_prediction(native.evaluate(np.zeros(native.basis['physical_basis'].shape[1])))
    exported=export_refinable(point['path'],tmp_path/'export',expected_sha256=point['sha256'])
    source=Path(exported['path']);before=file_sha(source);context=snapshot(source)
    base=[dict(kind='background',histogram='h0'),dict(kind='histogram_scale',histogram='h0')]
    def stage(name,limit,free):
        return dict(id=name,title=name,cycles=1,max_function_evaluations=limit,free=free,constraints=[])
    plan=dict(schema='radar-gpx-recipe/v1',model_signature=context['model_signature'],
        request='Bound the numerical work and recover independent stages.',assumptions=[],questions=[],unsupported=[],
        failure_policy='continue_independent',stages=[stage('before',1000,base),
            stage('limited',1,base+[dict(kind='cell',phase='p0')]),stage('after',1000,base)])
    preview=work('preview',source,tmp_path/'preview',plan=plan)
    assert preview['status']=='preview_passed',preview
    approval=approve(plan,context,preview['preview'],approver='Native budget regression')
    result=work('run',source,tmp_path/'run',plan=plan,approval=approval)
    assert result['status']=='completed',result
    assert result['rejected_stages']==['limited']
    outcomes=result['stage_outcomes']
    assert [s['status'] for s in outcomes]==['completed','rejected','completed']
    assert 'budget' in outcomes[1]['reason']
    assert outcomes[1]['native_evaluation_budget']['evaluations']==1
    assert outcomes[1]['native_evaluation_budget']['exhausted']
    assert outcomes[0]['native_solver_termination']['final']['solver_success']
    G=gsas();first=G.G2Project(gpxfile=str(tmp_path/'run/verified_before.gpx'))
    final=G.G2Project(gpxfile=str(tmp_path/'run'/result['final_gpx']))
    np.testing.assert_allclose(first.phases()[0].data['General']['Cell'][1:],final.phases()[0].data['General']['Cell'][1:],atol=1e-12,rtol=0)
    assert final.data['Covariance']['data']['Rvals']['radar_native_solver_termination']['state']=='observed'
    assert file_sha(source)==before and not (tmp_path/'run/verified_limited.gpx').exists()
