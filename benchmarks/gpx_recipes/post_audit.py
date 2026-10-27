"""Independent saved-GPX checks beyond the execution receipt's vary audit."""
import argparse
import json
from pathlib import Path

import numpy as np
from gpx_recipe.snapshot import gsas
from gpx_recipe.advanced import lattice


def equal(a,b):
    return bool(np.allclose(np.asarray(a,dtype=float),np.asarray(b,dtype=float),rtol=1e-9,atol=1e-9))


def native_to_unit(project,hist,value,unit):
    if unit in ('native','tof','two_theta'):return float(value)
    d=float(lattice(project).Pos2dsp(hist.InstrumentParameters,value))
    return d if unit=='d' else 2*np.pi/d


def inspect(root,mode,row,cases):
    G=gsas();case=cases[row['case']];work=Path(row['artifact_directory'])
    plan=json.loads((work/'plan.json').read_text());context=json.loads((root/'inputs'/(case['project']+'.context.json')).read_text())
    previous=G.G2Project(gpxfile=str(root/'inputs'/(case['project']+'.gpx')));checks=0
    for stage in plan['stages']:
        path=work/('refined_'+stage['id']+'.gpx');current=G.G2Project(gpxfile=str(path))
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
        for edit in stage.get('fit_ranges',[]):
            hist=current.histogram(next(h['name'] for h in context['histograms'] if h['key']==edit['histogram']))
            endpoints=sorted(native_to_unit(current,hist,x,edit['unit']) for x in hist.data['Limits'][1])
            assert np.allclose(endpoints,[edit['lower'],edit['upper']],rtol=1e-7,atol=1e-7),'Saved fit limits fail inverse calibration'
            checks+=1
        for edit in stage.get('exclusions',[]):
            hist=current.histogram(next(h['name'] for h in context['histograms'] if h['key']==edit['histogram']))
            saved=[sorted(native_to_unit(current,hist,x,edit['unit']) for x in region) for region in hist.Excluded()]
            if edit['mode']=='clear':assert not saved,'Excluded regions were not cleared'
            else:
                for region in edit['regions']:assert any(np.allclose(x,region,rtol=1e-7,atol=1e-7) for x in saved),'Saved exclusion fails inverse calibration'
            checks+=1
        previous=current
    return checks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('campaign');args=parser.parse_args();root=Path(args.campaign)
    cases={c['id']:c for c in json.loads((root/'catalog.json').read_text())};results=[]
    for mode in ['oracle','model']:
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
