"""Bounded native acceptance exercise for the guided recipes.

Run with the GSAS-II Python and nova/src on PYTHONPATH. The supplied single-site
fixture must be the declared synthetic LNO control (truth Li1=.65, fixed Ni=.08).
The redistribution fixture generated here is an artificial two-site numerical
control, not a proposed battery structure or a scientific benchmark.
"""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np

from gpx_recipe import guided
from gpx_recipe.compiler import approve
from gpx_recipe.execution import freeze, write_json
from gpx_recipe.model_edits import SCHEMA
from gpx_recipe.snapshot import gsas, snapshot
from gpx_recipe.worker import work


def checksum(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def decisions(context,recipe,**overrides):
    phase=next(p for p in context['phases'] if p['name']=='LNO_H')
    values=dict(recipe=recipe,model_review='reviewed',cell_phases=[phase['key']],phase=phase['key'],sites='Li1',positions=['O3'],
        reference='SYNTHETIC numerical control: fixed input mixing and ADPs, vacancies allowed. No experimental composition or reference claim.',conditional='accepted',reference_scope='single')
    values.update(overrides);state=guided.start(context)
    while (q:=guided.next_question(context,state)) is not None:
        state=guided.answer(context,state,q['id'],values[q['id']])
    return state


def run(source,output,recipe,**overrides):
    context=snapshot(source);state=decisions(context,recipe,**overrides);plan=guided.build_plan(context,state)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'decisions.json',state);write_json(output/'plan.json',plan)
    preview=work('preview',source,output/'preview',plan=plan,expected_source_sha256=context['source']['sha256'])
    if preview['status']!='preview_passed':return dict(status=preview['status'],error=preview.get('error'),directory=str(output))
    approval=approve(plan,context,preview['preview'],approver='Authorized local engineering validation; not scientific acceptance')
    write_json(output/'approval.json',approval)
    result=work('run',source,output/'run',plan=plan,approval=approval,expected_source_sha256=context['source']['sha256'])
    row=dict(status=result['status'],error=result.get('error'),directory=str(output),source_sha256=checksum(source),
             effective_variable_counts=[len(s['effective_vary']) for s in preview['preview']['stages']])
    if result.get('final_gpx'):
        final=snapshot(output/'run'/result['final_gpx']);phase=next(p for p in final['phases'] if p['name']=='LNO_H')
        row['occupancies']={a['label']:a['occupancy'] for a in phase['atoms']}
        view=json.loads((output/'run'/'final.view.json').read_text())
        row['rwp']={h['name']:h['rwp'] for h in view['histograms']}
        row['occupancy_parameters']=[p for p in view['parameters'] if p.get('parameter')=='occupancy']
    return row


def make_redistribution(source,output):
    output.mkdir(parents=True,exist_ok=False);truth=output/'truth.gpx';shutil.copy2(source,truth)
    G=gsas();project=G.G2Project(gpxfile=str(truth));phase=project.phase('LNO_H')
    phase.atom('Li1').occupancy=.55
    phase.add_atom(0,0,.22,'Li','Li_extra',occ=.12,uiso=.012)
    freeze(project);project.data['Controls']['data']['max cyc']=0
    project.save();project.do_refinements([{}])
    rng=np.random.default_rng(20261031)
    for h in project.histograms():
        arrays=h.data['data'][1];mask=np.ma.getmaskarray(arrays[1]);valid=np.ma.getdata(arrays[2])>0
        arrays[1]=np.ma.array(np.asarray(arrays[3])+rng.normal(0,.5,len(arrays[0])),mask=mask)
        arrays[2]=np.ma.array(np.where(valid,4.,0.),mask=np.ma.getmaskarray(arrays[2]))
    project.data['Covariance']['data']={};project.save()
    ctx=snapshot(truth);p=next(p for p in ctx['phases'] if p['name']=='LNO_H')
    atoms={a['label']:a for a in p['atoms']};m1,m2=atoms['Li1']['multiplicity'],atoms['Li_extra']['multiplicity']
    assert m1!=m2
    total=.55*m1+.12*m2
    starts=output/'start.gpx';project.save(str(starts));phase=project.phase('LNO_H')
    phase.atom('Li1').occupancy=.45;phase.atom('Li_extra').occupancy=(total-.45*m1)/m2
    project.save()
    record=dict(scope='Artificial distinct-site numerical control; not a battery structural hypothesis',seed=20261031,sigma=.5,
        truth=dict(Li1=.55,Li_extra=.12),multiplicities=dict(Li1=m1,Li_extra=m2),total_ions=total,truth_sha256=checksum(truth),starting_sha256=checksum(starts))
    write_json(output/'truth.json',record)
    return starts,record


def corrections(source,output,cif_dir):
    ctx=snapshot(source);base=dict(schema=SCHEMA,source_sha256=ctx['source']['sha256'],model_signature=ctx['model_signature'],reason='Engineering phase-correction round trip')
    original_hash=checksum(source);rows=[]
    for operation in ['add','replace','remove']:
        request=dict(base,operation=operation)
        if operation=='add':
            request.update(name='Al_second_component',links=[dict(histogram=h['key'],scale=.01) for h in ctx['histograms']])
        else:
            target_name='LNO_H' if operation=='replace' else 'Al_second_component'
            request['phase']=next(p['key'] for p in ctx['phases'] if p['name']==target_name)
        if operation!='remove':
            text=(cif_dir/('Al.cif' if operation=='add' else 'LNO_H.cif')).read_text()
            request.update(cif=text,cif_sha256=sha256(text.encode()).hexdigest())
        directory=output/operation
        result=work('edit',source,directory,plan=request)
        rows.append(dict(operation=operation,status=result['status'],error=result.get('error'),directory=str(directory)))
        if result['status']!='model_edited':break
        assert checksum(source)==original_hash
        view=json.loads((directory/'baseline.view.json').read_text())
        assert all(h['rwp'] is None and h['fit_statistics']['calculation_status']=='not_yet_evaluated' for h in view['histograms'])
        rows[-1]['stale_fit_scores_cleared']=True
        edited=directory/result['final_gpx']
        if operation=='add':
            # Same crystallographic phase deliberately has a distinct identity.
            added=snapshot(edited)
            assert {'Al','Al_second_component'}<={p['name'] for p in added['phases']}
            rows[-1]['separate_same_structure_components']=True
            # Hold this input for a rank-deficient scale-refinement check.
            rows[-1]['duplicate_source']=str(edited)
        if operation=='replace':
            old=next(p for p in ctx['phases'] if p['name']=='LNO_H')
            new=next(p for p in result['context']['phases'] if p['name']=='LNO_H')
            assert old['histograms']==new['histograms'] and old['phase_scales']==new['phase_scales']
        # Continue corrections on the new model revision.
        source=edited;original_hash=checksum(source);ctx=result['context']
        base.update(source_sha256=original_hash,model_signature=ctx['model_signature'])
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('real','synthetic','cifs','output'):parser.add_argument('--'+arg,required=True,type=Path)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    before={str(p):checksum(p) for p in [args.real,args.synthetic]}
    summary=dict(scope='Engineering execution checks; real-data model is unqualified, synthetic tests are not experimental battery validation',cases={})
    write_json(out/'summary.json',summary)
    def record(key,value):
        summary['cases'][key]=value;write_json(out/'summary.json',summary)
        print(json.dumps(dict(case=key,status=value.get('status') if isinstance(value,dict) else [r['status'] for r in value])),flush=True)
    record('phase_corrections',corrections(args.synthetic,out/'edits',args.cifs))
    record('real_fixed_composition',run(args.real,out/'real_mixture','mixture'))
    record('synthetic_single_site',run(args.synthetic,out/'single_site','vacancy',cell_phases=[],positions=[]))
    multi,truth=make_redistribution(args.synthetic,out/'redistribution_fixture')
    record('synthetic_redistribution',run(multi,out/'redistribution','redistribution',sites=['Li1','Li_extra'],total_ions=truth['total_ions'],cell_phases=[],positions=[]))
    record('aborted_structure_stage',run(args.synthetic,out/'aborted_structure','vacancy'))
    single=summary['cases']['synthetic_single_site'];multiple=summary['cases']['synthetic_redistribution']
    if single['status']=='completed':single['truth_error']=single['occupancies']['Li1']-.65
    if multiple['status']=='completed':
        multiple['truth_errors']={a:multiple['occupancies'][a]-x for a,x in truth['truth'].items()}
        multiple['conservation_error']=sum(multiple['occupancies'][a]*m for a,m in truth['multiplicities'].items())-truth['total_ions']
    added=summary['cases']['phase_corrections'][0]
    if added.get('duplicate_source'):
        record('nonidentifiable_duplicate_phases',run(Path(added['duplicate_source']),out/'duplicate_phases','mixture'))
    summary['source_unchanged']=all(checksum(p)==h for p,h in before.items())
    summary['checks_passed']=bool(summary['source_unchanged'] and all(r['status']=='model_edited' for r in summary['cases']['phase_corrections'])
        and len(summary['cases']['phase_corrections'])==3 and single['status']=='completed' and abs(single['truth_error'])<.03
        and multiple['status']=='completed' and max(abs(x) for x in multiple['truth_errors'].values())<.03
        and abs(multiple['conservation_error'])<1e-6 and summary['cases'].get('nonidentifiable_duplicate_phases',{}).get('status')!='completed'
        and summary['cases']['aborted_structure_stage']['status']=='needs_review')
    write_json(out/'summary.json',summary)
    print(json.dumps(dict(checks_passed=summary['checks_passed'],summary=str(out/'summary.json'))),flush=True)


if __name__=='__main__':main()
