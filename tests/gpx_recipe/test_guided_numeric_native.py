"""A complete guided recipe on public CW/TOF structural checkpoints."""
import os
from pathlib import Path
import numpy as np
import pytest

if os.environ.get('RADAR_REQUIRE_STRUCTURAL_NATIVE') == '1':
    import GSASII.GSASIIscriptable
else:
    pytest.importorskip('GSASII.GSASIIscriptable')

from test_native_profile_capture import native_source
from gpx_recipe import guided
from gpx_recipe.compiler import approve
from gpx_recipe.snapshot import snapshot
from gpx_recipe.structural_live.native import NativeCandidate
from gpx_recipe.structural_live.optimization.refinable_gpx import export_refinable
from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.worker import work


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_guided_numerical_recipe_preserves_modes_and_runs_bounded_stages(tmp_path,kind):
    native_source(tmp_path/'source',kind)
    native=NativeCandidate(tmp_path/'source/prepared',tmp_path/'point')
    point=native.save_verified_prediction(native.evaluate(np.zeros(native.basis['physical_basis'].shape[1])))
    exported=export_refinable(point['path'],tmp_path/'export',expected_sha256=point['sha256'])
    source=Path(exported['path']);before=file_sha(source);context=snapshot(source)
    setup=guided.start(context)
    answers={'recipe':'mixture','model_review':'reviewed','cell_phases':['p0'],'solver_review':'accepted'}
    while (question:=guided.next_question(context,setup)) is not None:
        setup=guided.answer(context,setup,question['id'],answers[question['id']])
    plan=guided.build_plan(context,setup)
    preview=work('preview',source,tmp_path/'preview',plan=plan)
    assert preview['status']=='preview_passed',preview
    approval=approve(plan,context,preview['preview'],approver='Public guided numerical regression')
    result=work('run',source,tmp_path/'run',plan=plan,approval=approval)
    assert result['status']=='completed',result
    assert len(result['stage_outcomes'])==2 and not result['rejected_stages']
    for row in result['stage_outcomes']:
        assert row['native_solver_termination']['final']['solver_success']
        assert 0<row['native_evaluation_budget']['evaluations']<=1000
        assert row['convergence']['maximum_cycles'] is None
    after=snapshot(tmp_path/'run'/result['final_gpx'])
    for old,new in zip(context['phases'][0]['atoms'],after['phases'][0]['atoms']):
        assert old['label']==new['label'] and old['element']==new['element']
        for field in ('coordinates','occupancy','adp','multiplicity'):
            np.testing.assert_allclose(old[field],new[field],atol=1e-12,rtol=0)
    assert file_sha(source)==before
