"""Reusable bounded sample-width controls; calibration stays fixed.

The numerical winner is a review candidate, never a scientifically accepted
instrument calibration or proof of a sample's size/strain mechanism.
"""
import copy
from pathlib import Path

import numpy as np

from .contracts import file_hash
from .io import atomic_json

TRIALS=(('unchanged',False,False,1.,1000.),('size_003',True,False,.03,0.),
        ('size_010',True,False,.1,0.),('size_030',True,False,.3,0.),
        ('strain',False,True,100.,100.),('size_strain',True,True,.1,100.))


def residual_metrics(histogram):
    """Score fresh saved profiles on the exact usable native observations."""
    from ..fit_domain import native_powder_fit_mask
    arrays=histogram.data['data'][1]
    x,y,w,calculated=(np.asarray(np.ma.getdata(a),float) for a in arrays[:4])
    valid=native_powder_fit_mask(x,histogram.data['Limits'][1],histogram.data['Limits'][2:])
    for a in arrays[:4]:valid &= ~np.ma.getmaskarray(a)
    valid &= np.isfinite(y)&np.isfinite(w)&np.isfinite(calculated)&(w>0)
    if not valid.any():raise ValueError('Sample-width trial has no usable native observations')
    weighted=y[valid]*np.sqrt(w[valid]);denominator=float(weighted@weighted)
    if not np.isfinite(denominator) or denominator<=0:
        raise ValueError('Sample-width Rwp is undefined for zero weighted intensity')
    residual=(y[valid]-calculated[valid])*np.sqrt(w[valid])
    if not np.isfinite(residual).all():raise ValueError('Sample-width residuals are not finite')
    lag=None;adjacent=np.diff(np.flatnonzero(valid))==1
    left,right=residual[:-1][adjacent],residual[1:][adjacent]
    if len(left)>1 and np.std(left)>0 and np.std(right)>0:
        value=float(np.corrcoef(left,right)[0,1])
        if np.isfinite(value):lag=value
    return dict(rwp=float(100*np.sqrt((residual@residual)/denominator)),
        reduced_residual_square=float(np.mean(residual**2)),residual_lag1=lag,
        usable_observations=int(valid.sum()),residual_adjacent_pairs=int(adjacent.sum()))


def sample_width_trials(source,destination,*,cycles=20):
    """Run the same complete controls for Model, Live or diagnostic callers.

    Every trial starts from the unchanged source. Each has its own GPX, full
    native refinement, refreshed saved curve, physical bounds and receipt.
    This function does not require a magnetic catalog or transfer a snapshot.
    """
    from GSASII import GSASIIscriptable as G
    from .gpx_index import install_fresh_gpx_index
    from .native_validation import checked_native_refine,refresh_saved_profiles
    from ..execution import freeze
    from ..symmetry.native_preparation import normalize_flags
    if type(cycles) is not int or not 1<=cycles<=100:
        raise ValueError('Use one to one hundred native cycles per sample-width control')
    install_fresh_gpx_index()
    source=Path(source).resolve();out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=True)
    if any((out/(name+'.gpx')).exists() for name,*_ in TRIALS):
        raise ValueError('Sample-width controls require fresh output GPX paths')
    source_sha=file_hash(source);original=G.G2Project(str(source))
    if len(original.histograms())!=1 or len(original.phases())!=1:
        raise ValueError('Sample-width controls require one nuclear phase and one histogram')
    if original.phase(0).data['General']['Type']!='nuclear':raise ValueError('Use a nuclear-only checkpoint')
    results=[]
    for name,ref_size,ref_strain,size,strain in TRIALS:
        try:
            g=G.G2Project(str(source),newgpx=str(out/(name+'.gpx')));h=g.histogram(0);p=g.phase(0)
            normalize_flags(g);freeze(g);instrument=copy.deepcopy(h.data['Instrument Parameters'])
            p.set_HAP_refinements({'Scale':True});h.set_refinements({'Background':{'refine':True}})
            p.set_refinements({'Cell':True});hap=p.data['Histograms'][h.name]
            if name!='unchanged':
                hap['Size'][0]='isotropic';hap['Mustrain'][0]='isotropic'
                hap['Size'][1][0]=size;hap['Mustrain'][1][0]=strain
            for kind,flag,lo,hi in [('Size',ref_size,.001,100.),('Mustrain',ref_strain,0.,10000.)]:
                p.set_HAP_refinements({kind:{'type':'isotropic','refine':flag}})
                variable=f'{p.id}:{h.id}:{kind};i'
                g.set_Controls('parmMin',lo,variable=variable);g.set_Controls('parmMax',hi,variable=variable)
            g.set_Controls('parmMin',0.,variable=f'{p.id}:{h.id}:Scale');g.set_Controls('cycles',cycles)
            checks=checked_native_refine(g);refresh_saved_profiles(g);g.save()
            if h.data['Instrument Parameters']!=instrument:raise ValueError('Instrument parameters changed')
            for index in (0,1,2):
                current=h.data['data'][1][index];before=original.histogram(0).data['data'][1][index]
                if (not np.array_equal(np.ma.getdata(current),np.ma.getdata(before))
                        or not np.array_equal(np.ma.getmaskarray(current),np.ma.getmaskarray(before))):
                    raise ValueError('Original observations, weights or masks changed')
            physical=bool(.001<=hap['Size'][1][0]<=100 and 0<=hap['Mustrain'][1][0]<=10000 and hap['Scale'][0]>=0)
            results.append(dict(name=name,path=g.filename,size_um=float(hap['Size'][1][0]),
                microstrain=float(hap['Mustrain'][1][0]),physical=physical,numerical_checks=checks,
                instrument_preserved=True,observations_and_masks_preserved=True,
                scientifically_accepted=False,**residual_metrics(h)))
        except Exception as exc:results.append(dict(name=name,error=str(exc),physical=False))
    if file_hash(source)!=source_sha:raise ValueError('Original sample-width source changed')
    valid=[v for v in results if v.get('physical') and v.get('numerical_checks',{}).get('qualified')
           and np.isfinite(v.get('rwp',np.nan))]
    best=min(valid,key=lambda v:v['rwp']) if valid else None
    report=dict(trials=results,selected_for_review=None if best is None else best['name'],
        source_sha256=source_sha,scientifically_qualified=False,
        interpretation='Effective broadening only; compare a same-configuration standard before attributing it to the sample')
    atomic_json(out/'profile-trials.json',report)
    return report
