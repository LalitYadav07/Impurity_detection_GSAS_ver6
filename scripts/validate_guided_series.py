"""Native series-transfer controls using explicitly labelled derived fixtures.

The mixture controls reuse one measured exposure with perturbed starts. The
occupancy controls use synthetic observations; these are not new battery data.
Run with the NOVA Python, --python pointing at the GSAS-II scientific Python.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def fixtures(args):
    from gpx_recipe.snapshot import gsas
    G = gsas()
    root = args.output / 'fixtures'
    root.mkdir(parents=True)
    for family, source in [('mixture', args.real), ('vacancy', args.synthetic), ('redistribution', args.redistribution)]:
        for variant in ['first', 'second', 'bad', 'future']:
            project = G.G2Project(gpxfile=str(source))
            phase = project.phase('LNO_H')
            if family == 'mixture':
                for hap in phase.data['Histograms'].values():
                    hap['Scale'][0] *= 1.02 if variant != 'first' else 1.
                if variant == 'bad':
                    # Alter instrument calibration: the native signature
                    # must reject this even with otherwise matching phase names.
                    for h in project.histograms():
                        h.InstrumentParameters['difC'][1] *= 1.001
            else:
                phase.atom('Li1').occupancy = .45 if variant == 'first' else .5
                if family == 'redistribution':
                    phase.atom('Li_extra').occupancy = (.55*3 + .12*6 - phase.atom('Li1').occupancy*3)/6
                if variant == 'bad':
                    if family == 'vacancy':
                        phase.atom('Li1').ADP = .025
                    else:
                        phase.atom('Li_extra').occupancy += .01
            # Use distinct scan labels, mutable histogram IDs and bytes while
            # preserving bank order and links. These are derived controls.
            for h in list(project.histograms()):
                old = h.name; new = 'PWDR ' + variant + '_' + old[5:]
                project.data[new] = project.data.pop(old)
                for names in project.names:
                    if names[0] == old:
                        names[0] = new
                for p in project.phases():
                    if old in p.data['Histograms']:
                        p.data['Histograms'][new] = p.data['Histograms'].pop(old)
                        p.data['Histograms'][new]['Histogram'] = new
            project.save(str(root / (family + '-' + variant + '.gpx')))


def main(args):
    from gpx_recipe import guided
    from gpx_recipe.jobs import scientific_environment
    from radar_pd_nova.refinement_runtime import LocalRefinementRuntime
    from radar_pd_nova.refinement_sources import RefinementScope, file_sha
    from radar_pd_nova.refinement_workspace import RefinementWorkspace
    args.output.mkdir(parents=True, exist_ok=False)
    command = [args.python, __file__, '--make-fixtures', '--output', str(args.output), '--real', str(args.real),
               '--synthetic', str(args.synthetic), '--redistribution', str(args.redistribution)]
    with (args.output / 'fixtures.log').open('w') as log:
        subprocess.run(command, env=scientific_environment(os.getenv('GSASII_ROOT')), stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = dict(scope=__doc__, cases={})
    for family in ['mixture', 'vacancy', 'redistribution']:
        root = args.output / family
        runtime = LocalRefinementRuntime(root / 'runtime', python=args.python, gsasii_root=os.getenv('GSASII_ROOT'))
        def no_planner(*a, **k):
            raise AssertionError('Series execution must not call a planner')
        w = RefinementWorkspace(root / 'workspace', history_id='local-series-control', runtime=runtime, planner=no_planner)
        def source(variant):
            path = args.output / 'fixtures' / (family + '-' + variant + '.gpx')
            return dict(id=variant, experiment=family, scan_key=variant, source_path=str(path), source_sha256=file_sha(path),
                        sample_key='control', sample_label='Derived series control', bank='', scan_label=variant, project={})
        initial = [source('first'), source('second')]
        w.observe(initial)
        ctx = w.inspect('first')['context'];p = next(p for p in ctx['phases'] if p['name'] == 'LNO_H')
        answers = dict(recipe=family, model_review='reviewed', cell_phases=[p['key']] if family=='mixture' else [],
                       phase=p['key'], sites=['Li1','Li_extra'] if family=='redistribution' else 'Li1',
                       total_ions=.55*3+.12*6, positions=[], conditional='accepted', reference_scope='series',
                       reference='Derived numerical controls only. Held parameters are identical, vacancies allowed; synthetic conserved total is 2.37 ions/unit cell.')
        state=guided.start(ctx)
        while (q:=guided.next_question(ctx,state)) is not None:
            state=guided.answer(ctx,state,q['id'],answers[q['id']])
        plan=guided.build_plan(ctx,state)
        draft=w.prepare(family,'first',plan['request'],guided=state)
        scope=RefinementScope(family,'samples',samples=('control',));_,token=w.selection(scope,future=True)
        recipe=w.approve(family,draft['generation'],scope,existing=True,future=True,selection_token=token,approver='Local engineering series check')
        w.start(recipe['id'],token)
        rows=[]
        def drain():
            while (job:=w.run_next()) is not None:
                row=dict(scan=job['source']['id'],status=job['status'],message=job['message'],directory=job['directory'])
                result=job.get('result') or {}
                if result.get('final_gpx'):
                    parameters=json.loads((Path(job['directory'])/'final.view.json').read_text())['parameters']
                    row['occupancies']=[p for p in parameters if p.get('parameter')=='occupancy']
                rows.append(row)
                print(json.dumps(dict(family=family,scan=row['scan'],status=row['status'])),flush=True)
        drain()
        # Exercise resumed future enrollment and continuing after rejection.
        w=RefinementWorkspace(root/'workspace',history_id='local-series-control',runtime=runtime,planner=no_planner)
        arrivals=[source('bad'),source('future')]
        w.observe(arrivals);drain()
        assert [r['status'] for r in rows]==['completed','completed','needs_recipe','completed'], rows
        assert len({j['source']['source_sha256'] for j in w.state['jobs'].values()})==4
        assert all(w.state['recipes'][j['recipe_id']]['plan']==plan for j in w.state['jobs'].values())
        assert all(file_sha(Path(s['source_path']))==s['source_sha256'] for s in initial+arrivals)
        summary['cases'][family]=rows
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    summary['checks_passed']=True
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(checks_passed=True,summary=str(args.output/'summary.json'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['real','synthetic','redistribution','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--make-fixtures',action='store_true')
    args=parser.parse_args()
    fixtures(args) if args.make_fixtures else main(args)
