"""Native no-fit calculation with the reflection tables from that exact call.

GSAS-II's DoNoFit returns only its calculated array. This adapter uses the same
native setup and errRefine call, retaining the tables for trustworthy Bragg
annotations. It neither refines nor changes the input GPX.
"""
import numpy as np


def forward_profile(path, histogram):
    from GSASII import GSASIIstrMain as main, GSASIIstrIO as io
    from GSASII import GSASIIstrMath as math, GSASIImpsubs as mp
    mp.InitMP();main.ptx.pyqlmninit()
    controls=io.GetControls(str(path));calc=dict(controls)
    histograms,phases=io.GetUsedHistogramsAndPhases(str(path))
    if not phases or histogram not in histograms:
        raise ValueError('Native forward profile requires the selected histogram and linked phases')
    bodies=io.GetRigidBodies(str(path));ids=bodies.get('RBIds',{'Vector':[],'Residue':[]})
    _,body_parameters=io.GetRigidBodyModels(bodies,Print=False)
    (count,indices,_,phase_parameters,pawley,ff,ef,orb,bl,mf,wave)=io.GetPhaseData(phases,{},ids,Print=False)
    calc.update(atomIndx=indices,Natoms=count,FFtables=ff,EFtables=ef,ORBtables=orb,BLtables=bl,MFtables=mf,maxSSwave=wave)
    _,hap_parameters,hap_controls=io.GetHistogramPhaseData(phases,histograms,Controls=calc,Print=False)
    calc.update(hap_controls)
    _,hist_parameters,_,hist_controls=io.GetHistogramData(histograms,Print=False);calc.update(hist_controls)
    parameters={**body_parameters,**phase_parameters,**hap_parameters,**hist_parameters}
    io.GetFprime(calc,histograms)
    math.errRefine([],[histograms,phases,{},bodies],parameters,{},[],calc,pawley,None)
    result=histograms[histogram]
    ticks=[]
    for name,reflections in result.get('Reflection Lists',{}).items():
        phase=phases.get(name,{})
        if phase.get('General',{}).get('Modulated'):continue
        kind=phase.get('General',{}).get('Type','nuclear')
        for row in reflections.get('RefList',[]):
            if len(row)<6 or not np.isfinite(row[:6]).all() or row[4]<=0:continue
            ticks.append(dict(phase=name,kind=kind,hkl=[int(v) for v in row[:3]],d=float(row[4]),native=float(row[5])))
    return np.asarray(result['Data'][3],float).copy(),ticks


def tick_arrays(ticks):
    return dict(tick_phase=np.asarray([r['phase'] for r in ticks],dtype='U'),
                tick_kind=np.asarray([r['kind'] for r in ticks],dtype='U'),
                tick_hkl=np.asarray([r['hkl'] for r in ticks],dtype=int).reshape(-1,3),
                tick_d=np.asarray([r['d'] for r in ticks],float),
                tick_native=np.asarray([r['native'] for r in ticks],float))
