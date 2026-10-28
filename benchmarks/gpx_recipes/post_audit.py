"""Independent saved-GPX checks beyond the execution receipt's vary audit."""
import argparse
import json
import hashlib
from pathlib import Path
import shutil
import tempfile

import numpy as np
from gpx_recipe.snapshot import gsas
from gpx_recipe.advanced import lattice
from gpx_recipe.studio_state import extract_gsasii_refinement_state


def equal(a,b):
    return bool(np.allclose(np.asarray(a,dtype=float),np.asarray(b,dtype=float),rtol=1e-9,atol=1e-9))


def native_to_unit(project,hist,value,unit):
    if unit in ('native','tof','two_theta'):return float(value)
    d=float(lattice(project).Pos2dsp(hist.InstrumentParameters,value))
    return d if unit=='d' else 2*np.pi/d


def inspect(root,mode,row,cases):
    work=Path(row['artifact_directory']);case=cases[row['case']]
    source=root/'inputs'/(case['project']+'.gpx')
    plan=json.loads((work/'plan.json').read_text())
    checkpoints=[work/('refined_'+s['id']+'.gpx') for s in plan['stages']]
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,*checkpoints]}
    try:
        with tempfile.TemporaryDirectory(prefix='gpx-post-audit-') as directory:
            local=Path(directory);shutil.copy2(source,local/'source.gpx')
            for path in checkpoints:shutil.copy2(path,local/path.name)
            return _inspect_copies(root,mode,row,cases,local)
    finally:
        assert all(hashlib.sha256(p.read_bytes()).hexdigest()==sha for p,sha in before.items()),'Post-audit changed an original GPX'


def _inspect_copies(root,mode,row,cases,local):
    G=gsas();case=cases[row['case']];work=Path(row['artifact_directory'])
    plan=json.loads((work/'plan.json').read_text());context=json.loads((root/'inputs'/(case['project']+'.context.json')).read_text())
    receipt=json.loads((work/'refined.execution.json').read_text())
    stage_receipts={s['id']:s for s in receipt['stages']}
    previous=G.G2Project(gpxfile=str(local/'source.gpx'));checks=0
    for stage in plan['stages']:
        path=local/('refined_'+stage['id']+'.gpx');current=G.G2Project(gpxfile=str(path))
        stage_result=stage_receipts[stage['id']];ledger=stage_result['parameter_changes']
        assert ledger['released_parameters'],'An executed stage has an empty released-parameter audit'
        assert not ledger.get('unobserved_released_parameters') and not ledger.get('unobserved_fixed_parameters'),'Parameter coverage is incomplete'
        extracted=extract_gsasii_refinement_state(path,current,G,max_correlations=0)
        saved={p['name']:p for p in extracted['parameter_space']['parameters']}
        covered=set()
        for parameter in ledger['released_parameters']:
            name=parameter['variable'];assert name in saved,'Audited parameter absent from reloaded GPX: '+name
            assert np.isclose(float(saved[name]['current_value']),parameter['after'],rtol=1e-10,atol=1e-12),'Audited final value differs from reloaded GPX: '+name
            covered.update([name,parameter.get('refinement_variable',name)]);checks+=1
        for name in stage_result['actual_vary']:
            if name in covered:continue
            origins=stage_result.get('constraint_parameter_provenance',{}).get(name,[])
            assert origins and all(p['variable'] in covered for p in origins),'An actual independent variable has no audited direct/dependent provenance: '+name
            checks+=1
        for p in context['phases']:
            free=any(t['kind']=='cell' and t['phase']==p['key'] for t in stage['free'])
            if not free:
                assert equal(current.phase(p['name']).data['General']['Cell'][1:],previous.phase(p['name']).data['General']['Cell'][1:]),'A fixed cell changed'
                checks+=1
        for h in context['histograms']:
            hist=current.histogram(h['name']);prior=previous.histogram(h['name'])
            bg_free=any(t['kind']=='background' and t['histogram']==h['key'] for t in stage['free'])
            if not bg_free and not (stage.get('background') or {}).get('histogram')==h['key']:
                a,b=hist.data['Background'][0],prior.data['Background'][0]
                assert a[0]==b[0] and equal(a[2:],b[2:]),'A fixed background changed'
                checks+=1
        edit=stage.get('background')
        if edit:
            hist=current.histogram(next(h['name'] for h in context['histograms'] if h['key']==edit['histogram']))
            prior=previous.histogram(hist.name);record=hist.data['Background']
            assert record[0][0]==edit['function'] and int(record[0][2])==edit['coefficients'],'Saved background model/count differs from request'
            if edit['fixed_points']=='remove':assert not record[1].get('FixedPoints'),'Background support points were not removed'
            else:assert record[1].get('FixedPoints')==prior.data['Background'][1].get('FixedPoints'),'Background support points were not preserved'
            checks+=1
        for edit in stage.get('fit_ranges',[]):
            hist=current.histogram(next(h['name'] for h in context['histograms'] if h['key']==edit['histogram']))
            endpoints=sorted(native_to_unit(current,hist,x,edit['unit']) for x in hist.data['Limits'][1])
            assert np.allclose(endpoints,[edit['lower'],edit['upper']],rtol=1e-7,atol=1e-7),'Saved fit limits fail inverse calibration'
            checks+=1
        for edit in stage.get('exclusions',[]):
            hist=current.histogram(next(h['name'] for h in context['histograms'] if h['key']==edit['histogram']))
            saved=[sorted(native_to_unit(current,hist,x,edit['unit']) for x in region) for region in hist.Excluded()]
            prior=previous.histogram(hist.name)
            old=[sorted(native_to_unit(current,hist,x,edit['unit']) for x in region) for region in prior.Excluded()]
            expected=(old if edit['mode']=='add' else [])+edit['regions']
            assert len(saved)==len(expected),'Excluded-region count differs from exact add/replace/clear request'
            assert all(np.allclose(x,y,rtol=1e-7,atol=1e-7) for x,y in zip(sorted(saved),sorted(expected))),'Saved exclusions fail exact inverse-calibration history check'
            checks+=1
        previous=current
    return checks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('campaign');args=parser.parse_args();root=Path(args.campaign)
    cases={c['id']:c for c in json.loads((root/'catalog.json').read_text())};results=[]
    for mode in ['oracle','model']:
        if not (root/(mode+'-status.json')).exists():continue
        state=json.loads((root/(mode+'-status.json')).read_text())
        for row in state['results'].values():
            if row.get('source_sha256')!=state['current_code'] or row['status']!='verified_execution':continue
            result={'case':row['case'],'mode':mode}
            try:result.update(status='verified',checks=inspect(root,mode,row,cases))
            except Exception as exc:result.update(status='failed',error=str(exc))
            results.append(result)
    output={'attempts':len(results),'checks':sum(r.get('checks',0) for r in results),'failures':[r for r in results if r['status']=='failed'],'results':results}
    (root/'saved-gpx-post-audit.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({k:v for k,v in output.items() if k!='results'}))
    if output['failures']:raise SystemExit(1)


if __name__=='__main__':main()
