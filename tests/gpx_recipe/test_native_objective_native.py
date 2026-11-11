"""Native stopping controls survive review, execution and checkpoint handoff."""
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
from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.worker import work


def noisy_source(root, kind, method='analytic Hessian'):
    source, _ = native_source(root, kind)
    project = gsas().G2Project(gpxfile=str(source))
    project.data['Controls']['data'].update({'deriv type':method, 'min dM/M':.001})
    histogram = project.histograms()[0]
    values = np.asarray(histogram.data['data'][1][1], dtype=float)
    sigma = .1 * np.sqrt(np.maximum(np.abs(values), 1.))
    histogram.data['data'][1][1] = values + np.random.default_rng(2839).normal(0, sigma)
    histogram.data['data'][1][2] = 1. / sigma**2
    project.phases()[0].data['General']['Cell'][1] *= 1.003
    histogram.SampleParameters['Scale'][0] *= .8
    histogram.data['Background'][0][3] += 2.
    project.save()
    return source


def stage(name, tolerance=None, *, numeric=False):
    result = dict(id=name, title=name, cycles=1 if numeric else 12,
        free=[dict(kind='background', histogram='h0'),
              dict(kind='histogram_scale', histogram='h0'), dict(kind='cell', phase='p0')],
        constraints=[])
    if numeric:
        result['max_function_evaluations'] = 500
    else:
        result['convergence'] = {'max_shift_over_sigma':.5}
    if tolerance is not None:
        result['relative_objective_tolerance'] = tolerance
    return result


def run(source, root, stages, *, plan=None):
    before = file_sha(source)
    context = snapshot(source)
    plan = plan or dict(schema='radar-gpx-recipe/v1', model_signature=context['model_signature'],
        request='Public synthetic stopping-control regression.', assumptions=[],
        questions=[], unsupported=[], stages=stages)
    preview = work('preview', source, root/'preview', plan=plan)
    assert preview['status'] == 'preview_passed', preview
    approval = approve(plan, context, preview['preview'], approver='Native stopping-control regression')
    result = work('run', source, root/'run', plan=plan, approval=approval)
    assert result['status'] == 'completed', result
    final = snapshot(root/'run'/result['final_gpx'])
    assert final['histograms'][0]['instrument'] == context['histograms'][0]['instrument']
    assert final['histograms'][0]['limits'] == context['histograms'][0]['limits']
    for old, new in zip(context['phases'][0]['atoms'], final['phases'][0]['atoms']):
        for key in ('coordinates', 'occupancy', 'adp'):
            np.testing.assert_array_equal(old[key], new[key])
    assert file_sha(source) == before
    return result, final


@pytest.mark.parametrize('kind', ['PNC', 'PNT'])
def test_explicit_hessian_tolerance_controls_real_fit_without_relaxing_acceptance(tmp_path, kind):
    source = noisy_source(tmp_path/'input', kind)
    ordinary, _ = run(source, tmp_path/'ordinary', [stage('fit')])
    tighter, final = run(source, tmp_path/'tighter', [stage('fit', 1e-5)])
    first = ordinary['stage_outcomes'][0]['convergence']
    second = tighter['stage_outcomes'][0]['convergence']
    assert first['relative_objective_tolerance'] == .001
    assert second['relative_objective_tolerance'] == 1e-5
    assert final['controls']['min dM/M'] == 1e-5
    assert second['last_cycle_max_shift_over_sigma'] < first['last_cycle_max_shift_over_sigma']
    assert second['last_cycle_max_shift_over_sigma'] <= .5
    assert tighter['stage_outcomes'][0]['convergence_screen']['status'] == 'passed'


@pytest.mark.parametrize('kind', ['PNC', 'PNT'])
def test_omitted_later_tolerance_restores_original_control_in_native_checkpoint(tmp_path, kind):
    source = noisy_source(tmp_path/'input', kind)
    result, final = run(source, tmp_path/'case', [stage('tighter', 1e-5), stage('original')])
    assert [r['convergence']['relative_objective_tolerance'] for r in result['stage_outcomes']] == [1e-5,.001]
    assert final['controls']['min dM/M'] == .001


@pytest.mark.parametrize('kind', ['PNC', 'PNT'])
def test_numerical_tolerance_preserves_bounded_residual_work(tmp_path, kind):
    source = noisy_source(tmp_path/'input', kind, 'numeric')
    result, final = run(source, tmp_path/'case', [stage('numeric', 1e-5, numeric=True)])
    row = result['stage_outcomes'][0]
    assert final['controls']['min dM/M'] == 1e-5
    assert row['native_solver_termination']['final']['solver_success']
    assert 0 < row['native_evaluation_budget']['evaluations'] <= 500
    assert row['convergence']['last_cycle_max_shift_over_sigma'] is None


@pytest.mark.parametrize('kind', ['PNC', 'PNT'])
def test_guided_controls_reach_real_recipe_execution(tmp_path, kind):
    from gpx_recipe import guided
    source = noisy_source(tmp_path/'input', kind)
    context = snapshot(source)
    state = guided.start(context)
    answers = dict(recipe='mixture', model_review='reviewed', cell_phases=['p0'])
    while (question := guided.next_question(context, state)) is not None:
        state = guided.answer(context, state, question['id'], answers[question['id']])
    state = guided.with_solver_settings(context, state, maximum=20, relative_objective_tolerance=1e-5)
    plan = guided.build_plan(context, state)
    result, final = run(source, tmp_path/'case', [], plan=plan)
    assert len(result['stage_outcomes']) == 2
    assert all(r['convergence']['relative_objective_tolerance'] == 1e-5 for r in result['stage_outcomes'])
    assert all(r['convergence']['maximum_cycles'] == 20 for r in result['stage_outcomes'])
    assert final['controls']['min dM/M'] == 1e-5
