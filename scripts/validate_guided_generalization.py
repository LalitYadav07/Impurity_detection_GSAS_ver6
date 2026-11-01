"""Exercise an ordinary-material guide on supplied real GPX projects.

No phase name, bank number, atom label or input path is built into this runner.
It verifies the native outputs and the plot/parameter display from those outputs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from gpx_recipe import guided
from gpx_recipe.jobs import scientific_environment
from radar_pd_nova.refinement_controller import recipe_cards, refinement_parameter_warnings
from radar_pd_nova.refinement_plots import fit_figure, trend_figure
from radar_pd_nova.refinement_runtime import LocalRefinementRuntime
from radar_pd_nova.refinement_sources import RefinementScope, file_sha
from radar_pd_nova.refinement_workspace import RefinementWorkspace


def main(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    input_hashes={path.resolve():file_sha(path) for path in args.gpx}
    runtime=LocalRefinementRuntime(root/'runtime',python=args.python,gsasii_root=os.getenv('GSASII_ROOT'))
    def no_planner(*a,**k):raise AssertionError('Ordinary guided workflow must not call a planner')
    w=RefinementWorkspace(root/'workspace',history_id='generalization-check',runtime=runtime,planner=no_planner)
    sources=[dict(id=f'input-{i}',experiment='ordinary',experiment_label='Ordinary material validation',
                  scan_key=f'input-{i}',scan_label=path.parent.name+' / '+path.name,
                  run_number=args.run_number[i-1] if args.run_number else None,
                  source_path=str(path.resolve()),source_sha256=file_sha(path),sample_key='ordinary',sample_label='Supplied ordinary-material series',bank='',project={})
             for i,path in enumerate(args.gpx,1)]
    w.observe(sources[:2]);ctx=w.inspect(sources[0]['id'])['context']
    assert not any(guided.species(a) in {'Li','Na'} for p in ctx['phases'] for a in p['atoms']), 'Use a non-battery control with no Li/Na sites'
    setup=guided.start(ctx)
    options=guided.next_question(ctx,setup)['options'];assert [o['value'] for o in options]==['mixture']
    answers=dict(recipe='mixture',model_review='reviewed',cell_phases=[p['key'] for p in ctx['phases'] if p['histograms']])
    while (question:=guided.next_question(ctx,setup)) is not None:
        setup=guided.answer(ctx,setup,question['id'],answers[question['id']])
    plan=guided.build_plan(ctx,setup);draft=w.prepare('ordinary',sources[0]['id'],plan['request'],guided=setup)
    scope=RefinementScope('ordinary');_,token=w.selection(scope,future=True)
    recipe=w.approve('ordinary',draft['generation'],scope,existing=True,future=True,selection_token=token,approver='Local ordinary-material engineering audit')
    w.start(recipe['id'],token)
    rows=[];report=[]
    def drain():
        while (job:=w.run_next()) is not None:
            print(json.dumps(dict(scan=job['source']['id'],status=job['status'],message=job['message'])),flush=True)
            assert job['status']=='completed', job['message']
            directory=Path(job['directory']);result=job['result']
            baseline=json.loads((directory/result['baseline_view']).read_text());final=json.loads((directory/result['final_view']).read_text())
            for h in final['histograms']:
                for axis in ['native','d','Q']:
                    figure,notice=fit_figure(baseline,final,histogram=h['name'],axis=axis)
                    assert len(figure.data)>2 and notice['polished_rwp']==h['rwp']
                    assert not any('Li content' in n or 'Na content' in n for n in refinement_parameter_warnings(final))
            rows.append(dict(id=job['id'],source=job['source'],recipe_id=recipe['id'],recipe_label='Recipe 1',baseline=baseline,final=final))
            report.append(dict(source=job['source']['source_path'],source_sha256=job['source']['source_sha256'],run_number=job['source'].get('run_number'),directory=str(directory),
                               phases=[p['name'] for p in final['phases']],histograms=[h['name'] for h in final['histograms']],status=job['status']))
    drain()
    w.observe(sources[2:]);drain()
    for metric in ['rwp','phase_fractions']+[p['key'] for p in rows[0]['final']['parameters'] if p['kind']=='cell']:
        figure=trend_figure(rows,metric=metric)
        assert figure.data and all(len(trace.x)==len(sources) for trace in figure.data)
    # Exercise the actual Galaxy command wrapper and archive boundary locally,
    # using only the scripts/ package bridge, not the NOVA development path.
    wrapper=root/'wrapper';wrapper.mkdir()
    request=wrapper/'request.json';request.write_text(json.dumps(dict(plan=plan,approval=recipe['approval'],expected_source_sha256=sources[-1]['source_sha256'])))
    scripts=Path(__file__).resolve().parent;environment=scientific_environment(os.getenv('GSASII_ROOT'));environment['PYTHONPATH']=str(scripts)
    with (wrapper/'console.log').open('w') as log:
        subprocess.run([args.python,str(scripts/'ndip_refinement.py'),'run',sources[-1]['source_path'],str(request),str(wrapper/'worker'),str(wrapper/'results.zip')],
                       cwd=wrapper,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    response=json.loads((wrapper/'worker/response.json').read_text());assert response['status']=='completed',response
    assert all(file_sha(path)==checksum for path,checksum in input_hashes.items())
    summary=dict(checks_passed=True,scope='Real ordinary-material inputs; engineering execution/display check, not scientific acceptance',
                 inputs=[dict(path=str(path),sha256=checksum) for path,checksum in input_hashes.items()],
                 sources=report,recipe_cards=recipe_cards(plan,ctx),decisions=guided.decision_summary(ctx,setup),
                 native_wrapper='completed',plot_axes=['native','d','Q'],trends='Rwp, phase fractions, all cell metrics',model_calls=0)
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(checks_passed=True,summary=str(root/'summary.json'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpx',type=Path,action='append',required=True)
    parser.add_argument('--run-number',type=int,action='append',help='Actual run number for each --gpx, in the same order; omitted metadata stays unknown')
    parser.add_argument('--python',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if len(args.gpx)<2:parser.error('Supply at least two GPX projects')
    if args.run_number and len(args.run_number)!=len(args.gpx):parser.error('Supply one --run-number for each --gpx')
    main(args)
