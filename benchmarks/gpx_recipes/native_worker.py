"""Isolated native execution and independent checks for one benchmark recipe."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import tempfile
import time
import traceback

import numpy as np
from gpx_recipe.compiler import approve,compile_script
from gpx_recipe.execution import run_plan,write_json
from gpx_recipe.snapshot import gsas,json_safe
from gpx_recipe.identity import explicit_constraints,constraint_binding


def raw_arrays(g):
    result={}
    for h in g.histograms():
        result[h.name]=[hashlib.sha256(np.asarray(np.ma.getdata(a)).tobytes()).hexdigest() for a in h.data['data'][1][:3]]
    return result


def close(a,b):return bool(np.allclose(np.asarray(a,dtype=float),np.asarray(b,dtype=float),rtol=1e-10,atol=1e-10))


def verify(source, output, plan, context, receipt):
    # GSAS-II may save while resolving IDs for get_Controls(). An independent
    # audit must never change the source or the delivered checkpoint bytes.
    source,output=Path(source),Path(output)
    paths=[source,output,*[output.with_name(output.stem+'_'+s['id']+'.gpx') for s in plan['stages']]]
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}
    try:
        with tempfile.TemporaryDirectory(prefix='gpx-independent-audit-') as directory:
            root=Path(directory);local_source=root/'source'/'input.gpx';local_output=root/'outputs'/output.name
            local_source.parent.mkdir();local_output.parent.mkdir()
            shutil.copy2(source,local_source);shutil.copy2(output,local_output)
            for path in paths[2:]:
                if path.exists():shutil.copy2(path,local_output.parent/path.name)
            return _verify_copies(local_source,local_output,plan,context,receipt)
    finally:
        assert all(hashlib.sha256(p.read_bytes()).hexdigest()==digest for p,digest in before.items()),'Independent audit changed an original GPX'


def _verify_copies(source, output, plan, context, receipt):
    G=gsas(); original=G.G2Project(gpxfile=str(source));original.index_ids();checks=[];warnings=[]
    original_arrays=raw_arrays(original)
    original_constraints=constraint_binding(explicit_constraints(original,json_safe),context['phases'],context['histograms'])
    original_restraints=json_safe(original.data.get('Restraints',{}))
    stage_files=[output.with_name(output.stem+'_'+s['id']+'.gpx') for s in plan['stages']]
    previous=original; normalized=set(); equations=[]; bound_history={}
    for stage,row,path in zip(plan['stages'],receipt['stages'],stage_files):
        if not path.exists():path=output
        current=G.G2Project(gpxfile=str(path));current.index_ids();cfg=row['configured_state'];free=stage['free']
        saved_constraints=constraint_binding(explicit_constraints(current,json_safe),context['phases'],context['histograms'])
        assert all(record in saved_constraints for record in original_constraints),'Original explicit constraint removed or modified'
        assert json_safe(current.data.get('Restraints',{}))==original_restraints,'Original restraint removed or modified'
        normalized.update(stage.get('phase_normalization',[]));equations.extend(c for c in stage['constraints'] if c not in equations)
        assert current.get_Controls('cycles')==stage['cycles'],'Refinement cycle control mismatch'
        expected_atoms={(t['phase'],t['atom']):set(t['flags']) for t in free if t['kind']=='atom'}
        if raw_arrays(current)!=original_arrays:raise AssertionError('Observed x/y/weight data changed')
        for hrecord in context['histograms']:
            h=current.histogram(hrecord['name']);before=original.histogram(hrecord['name'])
            for key,val in before.InstrumentParameters.items():
                now=h.InstrumentParameters[key]
                if isinstance(val,(list,tuple)) and len(val)>1:
                    if isinstance(val[1],str):assert now[1]==val[1]
                    else:assert close(now[1],val[1]),'Instrument value changed: '+key
            expected_global=any(t['kind']=='histogram_scale' and t['histogram']==hrecord['key'] for t in free)
            assert bool(h.SampleParameters['Scale'][1])==expected_global,'Histogram scale flag mismatch'
            cfg_h=next(x for x in cfg['histograms'] if x['key']==hrecord['key'])
            assert close(h.data['Limits'],cfg_h['limits']),'Fit limits/exclusions changed during refinement'
            if not expected_global:assert close(h.SampleParameters['Scale'][0],cfg_h['histogram_scale'][0]),'Fixed histogram scale changed'
            if hrecord['key'] in normalized:
                linked=[p for p in current.phases() if h.name in p.data['Histograms'] and p.data['Histograms'][h.name].get('Use',True)]
                assert abs(sum(p.data['Histograms'][h.name]['Scale'][0] for p in linked)-1)<1e-7,'Phase scales not normalized'
            if hrecord['key'] in stage.get('phase_normalization',[]):
                linked=[p for p in current.phases() if h.name in p.data['Histograms'] and p.data['Histograms'][h.name].get('Use',True)]
                before_h=previous.histogram(h.name)
                for p in linked:
                    prior=previous.phase(p.name)
                    old=before_h.SampleParameters['Scale'][0]*prior.data['Histograms'][h.name]['Scale'][0]
                    cp=next(x for x in cfg['phases'] if context['phases'][int(x['key'][1:])]['name']==p.name)
                    new=cfg_h['histogram_scale'][0]*cp['hap'][h.name]['scale'][0]
                    assert close(old,new),'Normalization changed initial phase intensity product'
        for precord in context['phases']:
            p=current.phase(precord['name']);cp=next(x for x in cfg['phases'] if x['key']==precord['key'])
            old_texture=original.phase(precord['name']).data['General'].get('SH Texture',{})
            texture=p.data['General'].get('SH Texture',{})
            if old_texture.get('SH Coeff'):
                assert not texture['SH Coeff'][0],'Unrequested phase-global texture coefficients are free'
                assert set(texture['SH Coeff'][1])==set(old_texture['SH Coeff'][1]),'Original global texture coefficient set changed'
                for name,value in old_texture['SH Coeff'][1].items():assert close(texture['SH Coeff'][1][name],value),'Fixed phase-global texture coefficient changed'
            for name in ['Order','Model']:
                if name in old_texture:assert texture[name]==old_texture[name],'Global texture model/order changed'
            for name in ['Sample omega','Sample chi','Sample phi']:
                if name in old_texture:
                    assert not texture[name][0],'Unrequested global texture sample angle is free'
                    assert close(texture[name][1],old_texture[name][1]),'Fixed global texture sample angle value changed'
            assert bool(p.data['General']['Cell'][0])==any(t['kind']=='cell' and t['phase']==precord['key'] for t in free)
            for a in p.atoms():
                expected=expected_atoms.get((precord['key'],a.label),set());assert set(a.refinement_flags.strip())==expected,'Atom flags mismatch: '+a.label
                before=next(x for x in cp['atoms'] if x['label']==a.label)
                for flag,value,key in [('F',a.occupancy,'occupancy'),('X',list(a.coordinates),'coordinates'),('U',a.ADP,'adp')]:
                    if flag not in expected:assert close(value,before[key]),'Fixed atom parameter changed: '+a.label+' '+flag
                if 'F' in expected and not -.000001<=a.occupancy<=1.000001:warnings.append('Occupancy outside physical interval')
                if 'U' in expected and float(a.ADP)<0:warnings.append('Negative Uiso')
            for hname,hap in p.data['Histograms'].items():
                hkey=next(x['key'] for x in context['histograms'] if x['name']==hname)
                phase_free=any(t['kind']=='phase_scale' and t['phase']==precord['key'] and t['histogram']==hkey for t in free)
                assert bool(hap['Scale'][1])==phase_free,'Phase-scale flag mismatch'
                tex_free=any(t['kind']=='preferred_orientation' and t['phase']==precord['key'] and t['histogram']==hkey for t in free)
                assert bool(hap['Pref.Ori.'][2])==tex_free,'Texture flag mismatch'
                if not phase_free:assert close(hap['Scale'][0],cp['hap'][hname]['scale'][0]),'Fixed phase scale changed'
                if hap['Scale'][0]<0:warnings.append('Negative phase scale')
        for constraint in equations:
            if constraint['kind']=='uiso_equal':
                values=[float(current.phase(next(p['name'] for p in context['phases'] if p['key']==t['phase'])).atom(t['atom']).ADP) for t in constraint['atoms']]
                assert max(values)-min(values)<1e-9,'Uiso equality not satisfied'
                continue
            value=sum(t['coefficient']*current.phase(next(p['name'] for p in context['phases'] if p['key']==t['phase'])).atom(t['atom']).occupancy for t in constraint['terms'])
            assert abs(value-constraint['total'])<1e-7,'Occupancy equation not satisfied'
        for bound in stage.get('bounds',[]):bound_history[(bound['phase'],bound['atom'],bound['parameter'])]=bound
        for bound in bound_history.values():
            p=current.phase(next(p['name'] for p in context['phases'] if p['key']==bound['phase']));a=p.atom(bound['atom']);value=a.occupancy if bound['parameter']=='F' else a.ADP
            assert bound['lower']-1e-7<=value<=bound['upper']+1e-7,'Custom bounds not respected'
            index=next(i for i,a in enumerate(p.atoms()) if a.label==bound['atom']);variable=f'{p.id}::'+('Afrac' if bound['parameter']=='F' else 'AUiso')+f':{index}'
            assert close(current.get_Controls('parmMin',variable=variable),bound['lower']),'Lower control bound was not set'
            assert close(current.get_Controls('parmMax',variable=variable),bound['upper']),'Upper control bound was not set'
            if bound in stage.get('bounds',[]) and 'value' in bound:
                cp=next(x for x in cfg['phases'] if x['key']==bound['phase']);ca=next(a for a in cp['atoms'] if a['label']==bound['atom'])
                assert close(ca['occupancy' if bound['parameter']=='F' else 'adp'],bound['value']),'Explicit starting value not set'
        for edit in stage.get('texture',[]):
            p=current.phase(next(p['name'] for p in context['phases'] if p['key']==edit['phase']));hname=next(h['name'] for h in context['histograms'] if h['key']==edit['histogram']);record=p.data['Histograms'][hname]['Pref.Ori.']
            assert record[0]==('MD' if edit['model']=='march-dollase' else 'SH')
            if record[0]=='MD':assert record[3]==edit['direction']
            else:assert record[4]==edit['order'] and record[5]
        assert set(row['actual_vary'])==set(row['effective_vary']),'Actual covariance vary list mismatch'
        assert current.data['Covariance']['data'].get('freshCOV'),'No fresh native covariance'
        if row['covariance'].get('rank') is not None and row['covariance']['rank']<row['covariance']['variable_count']:warnings.append('Rank-deficient covariance')
        checks.append(dict(stage=stage['id'],independent_variables=len(row['actual_vary']),rwp=row['rwp'],observations_unchanged=True,flags_exact=True,fixed_values_preserved=True,physical_constraints_checked=True))
        previous=current
    return dict(checks=checks,scientific_warnings=sorted(set(warnings)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('campaign');parser.add_argument('case');parser.add_argument('kind',choices=['oracle','model']);parser.add_argument('--workdir');args=parser.parse_args()
    root=Path(args.campaign);case=next(c for c in json.loads((root/'catalog.json').read_text()) if c['id']==args.case)
    work=Path(args.workdir) if args.workdir else root/'cases'/case['id']/args.kind;work.mkdir(parents=True,exist_ok=True)
    plan=json.loads((work/'plan.json').read_text());context=json.loads((root/'inputs'/(case['project']+'.context.json')).read_text());source=root/'inputs'/(case['project']+'.gpx')
    started=time.monotonic();cpu=time.process_time();source_hash=hashlib.sha256(source.read_bytes()).hexdigest();result={'case':case['id'],'kind':args.kind,'project':case['project']}
    try:
        preview=run_plan(source,work/'preview.gpx',plan,preview=True)
        approval=approve(plan,context,preview,approver='automated-recipe-benchmark');write_json(work/'approval.json',approval)
        script=work/'refine.py';compile_script(plan,approval,script)
        # Execute the actual compiled wrapper, not a substitute direct recipe path.
        namespace={'__name__':'__main__'}
        import sys
        sys.argv=[str(script),'--source-gpx',str(source),'--output-gpx',str(work/'refined.gpx')]
        exec(compile(script.read_text(),str(script),'exec'),namespace)
        receipt=json.loads((work/'refined.execution.json').read_text())
        audit=verify(source,work/'refined.gpx',plan,context,receipt)
        result.update(status='verified_execution',**audit)
    except Exception as exc:
        receipt_path=work/'refined.execution.json';receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        error=str(exc);scientific=any(s in error.lower() for s in ['froze a variable','singular','fresh refinement covariance','no effective independent','svd','diverged','negative phase','rank','mixed-site occupancy','overfilling','fully constrained','before save-time clipping','normalization drifted','requested uiso_equal relation','requested occupancy_sum relation','already frozen by preserved source']) or any(s.get('blocking_frozen_variables',s.get('frozen_variables')) for s in receipt.get('stages',[]))
        result.update(status='scientific_stop' if scientific else 'runtime_defect',error=error,error_type=type(exc).__name__,stages_completed=sum(s.get('status')=='completed' for s in receipt.get('stages',[])),traceback=traceback.format_exc())
    result.update(source_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==source_hash,elapsed_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if not result['source_unchanged']:result['status']='runtime_defect';result['error']='Original GPX changed'
    write_json(work/'result.json',result);print(json.dumps({k:result[k] for k in ['case','kind','status','elapsed_seconds','cpu_seconds','peak_rss_kib']}))


if __name__=='__main__':main()
