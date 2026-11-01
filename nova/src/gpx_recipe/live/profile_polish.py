"""Bounded diagnostic sample-broadening trials, with the instrument held fixed.
No automatic physical interpretation or scientific acceptance of the winner.
"""
import copy,json,sys
from pathlib import Path
import numpy as np
from .contracts import file_hash
from .io import atomic_json


def polish(receipt_path):
    from GSASII import GSASIIscriptable as G
    from .gpx_index import install_fresh_gpx_index
    from .native_validation import checked_native_refine,refresh_saved_profiles
    from gpx_recipe.execution import freeze
    from gpx_recipe.symmetry.native_preparation import normalize_flags
    install_fresh_gpx_index()
    receipt_path=Path(receipt_path).resolve();r=json.loads(receipt_path.read_text());out=receipt_path.parent
    choice=r['base_option'];source=Path(choice['source_path'])
    if file_hash(source)!=choice['source_sha256']:raise ValueError('Reviewed GPX changed')
    original=G.G2Project(str(source))
    if len(original.histograms())!=1 or len(original.phases())!=1:raise ValueError('Diagnostic broadening currently requires one nuclear phase and one histogram; use explicit multiphase refinement otherwise')
    if original.phase(0).data['General']['Type']!='nuclear':raise ValueError('Use a nuclear-only checkpoint')
    results=[]
    trials=[('unchanged',False,False,1.,1000.),('size_003',True,False,.03,0.),('size_010',True,False,.1,0.),('size_030',True,False,.3,0.),('strain',False,True,100.,100.),('size_strain',True,True,.1,100.)]
    for name,ref_size,ref_strain,size,strain in trials:
        try:
            g=G.G2Project(str(source),newgpx=str(out/(name+'.gpx')));h=g.histogram(0);p=g.phase(0)
            normalize_flags(g);freeze(g);instrument=copy.deepcopy(h.data['Instrument Parameters'])
            p.set_HAP_refinements({'Scale':True});h.set_refinements({'Background':{'refine':True}});p.set_refinements({'Cell':True})
            hap=p.data['Histograms'][h.name]
            if name!='unchanged':
                hap['Size'][0]='isotropic';hap['Mustrain'][0]='isotropic'
                hap['Size'][1][0]=size;hap['Mustrain'][1][0]=strain
            for kind,flag,lo,hi in [('Size',ref_size,.001,100.),('Mustrain',ref_strain,0.,10000.)]:
                p.set_HAP_refinements({kind:{'type':'isotropic','refine':flag}})
                variable=f'{p.id}:{h.id}:{kind};i'
                g.set_Controls('parmMin',lo,variable=variable);g.set_Controls('parmMax',hi,variable=variable)
            g.set_Controls('parmMin',0.,variable=f'{p.id}:{h.id}:Scale');g.set_Controls('cycles',20)
            checks=checked_native_refine(g);refresh_saved_profiles(g);g.save()
            if h.data['Instrument Parameters']!=instrument:raise ValueError('Instrument parameters changed')
            for i in (0,1,2):
                if not np.array_equal(h.data['data'][1][i],original.histogram(0).data['data'][1][i]):raise ValueError('Original observations changed')
            physical=bool(.001<=hap['Size'][1][0]<=100 and 0<=hap['Mustrain'][1][0]<=10000 and hap['Scale'][0]>=0)
            x,y,w,calc,*_=h.data['data'][1];valid=(w>0)&(x>=h.data['Limits'][1][0])&(x<=h.data['Limits'][1][1])
            for lo,hi in h.data['Limits'][2:]:valid&=~((x>=lo)&(x<=hi))
            residual=(y[valid]-calc[valid])*np.sqrt(w[valid]);rwp=float(100*np.linalg.norm(residual)/np.linalg.norm(y[valid]*np.sqrt(w[valid])))
            results.append(dict(name=name,path=g.filename,rwp=rwp,size_um=float(hap['Size'][1][0]),microstrain=float(hap['Mustrain'][1][0]),physical=physical,numerical_checks=checks,
                reduced_residual_square=float(np.mean(residual**2)),residual_lag1=float(np.corrcoef(residual[:-1],residual[1:])[0,1]),instrument_preserved=True,scientifically_accepted=False))
        except Exception as exc:results.append(dict(name=name,error=str(exc),physical=False))
    valid=[v for v in results if v.get('physical') and v.get('numerical_checks',{}).get('qualified') and np.isfinite(v.get('rwp',np.nan))]
    if not valid:atomic_json(out/'profile-trials.json',results);raise ValueError('No physically bounded numerically successful diagnostic trial')
    best=min(valid,key=lambda v:v['rwp']);atomic_json(out/'profile-trials.json',dict(trials=results,selected_for_review=best['name'],scientifically_qualified=False,interpretation='Effective broadening only; compare a same-configuration standard before attributing it to the sample'))
    if file_hash(source)!=choice['source_sha256']:raise ValueError('Original source changed')
    prep=out/'preparation-input.json';atomic_json(prep,dict(base_option=choice,job_id='bounded-profile-diagnostic',source_path=best['path'],source_sha256=file_hash(best['path'])))
    from .prepare_reference import prepare
    result=prepare(prep);result['profile_diagnostic']=str(out/'profile-trials.json')
    result['title']='Profile diagnostic · '+result['title'];atomic_json(out/'option.json',result)
    return result

if __name__=='__main__':polish(sys.argv[1])
