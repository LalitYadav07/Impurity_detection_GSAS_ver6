"""Native nuclear displacement-mode trial with an exact zero-mode control.

Adapted scientific procedure from the MIT-licensed reference project; see
../STUDIO-LICENSE. No reference-project imports or runtime are used.
"""
from __future__ import annotations

import copy
import importlib
import numpy as np

from ..snapshot import gsas
from ..execution import freeze,write_json
from .compiler import confined
from .worker import TrialDisposition


def _mapped_parent_sites(parent,child,transform,origin,S):
    """Use exact parent symmetry/positions, not a guessed split-label prefix."""
    pcx,pct,_,_=parent['General']['AtomPtrs'];ccx,cct,_,_=child['General']['AtomPtrs']
    orbit=[]
    for atom in parent['Atoms']:
        positions=[np.asarray(a[0]) for a in S.GenAtom(atom[pcx:pcx+3],parent['General']['SGData'],All=False)]
        orbit.append((atom,positions))
    mapping=[]
    for atom,xyz in zip(child['Atoms'],child['ISODISTORT']['G2parentCoords']):
        in_parent=np.asarray(transform).T@np.asarray(xyz)+origin
        matches=[]
        for old,positions in orbit:
            if atom[cct].casefold()!=old[pct].casefold():continue
            if any(np.max(np.abs((in_parent-p+.5)%1.-.5))<1e-6 for p in positions):matches.append(old)
        if len(matches)!=1:raise TrialDisposition('unsupported','Child-to-parent site mapping is missing or ambiguous; explicit mapping is required')
        mapping.append(matches[0])
    return mapping


def run(source_gpx,ctx,protocol,candidate,seed):
    G=gsas();prefix=G.__package__+'.' if G.__package__ else ''
    L=importlib.import_module(prefix+'GSASIIlattice');M=importlib.import_module(prefix+'GSASIImath');S=importlib.import_module(prefix+'GSASIIspc');O=importlib.import_module(prefix+'GSASIIobj');R=importlib.import_module(prefix+'GSASIIstrMain')
    from gsas_magnetic_preflight import setting_fingerprint
    from gsas_isodistort_import import prepare_isodistort_cif,add_isodistort_phase
    output=ctx.run_dir/'child-refinement.gpx'
    p=G.G2Project(gpxfile=str(source_gpx),newgpx=str(output));parent=p.phase(protocol['parent_phase']);h=p.histogram(protocol['histogram'])
    from .native_preparation import normalize_flags
    ctx.save_json('legacy-flag-normalization.json',normalize_flags(p))
    if parent.data['General']['Type']!='nuclear' or parent.data['General'].get('Modulated'):raise TrialDisposition('unsupported','A nonmodulated nuclear parent is required')
    if setting_fingerprint(parent.data['General']['SGData'])!=protocol['parent_setting_sha256']:raise TrialDisposition('integrity_failure','The native parent setting changed')
    if any(ph.data['General']['Type']!='nuclear' for ph in p.phases()):raise TrialDisposition('unsupported','Magnetic/nuclear coexistence needs a separately declared executor')
    p.set_Controls('sequential',[]);p.set_Controls('seqCopy',False);p.set_Controls('Reverse Seq',False)
    for phase in p.phases():
        for name,hap in phase.data['Histograms'].items():hap['Use']=name==h.name
    freeze(p)
    # Parent-coordinate and material constraints need exact substitution of
    # dependent variables. Refuse them until a transformation is implemented.
    if any(p.data['Constraints']['data'].get(key) for key in ['Phase','HAP']):raise TrialDisposition('unsupported','Source phase/HAP constraints require an explicit child-variable transfer')
    if p.data.get('Restraints',{}).get('data',{}).get(parent.name):raise TrialDisposition('unsupported','Parent restraints need a child-site mapping before replacement')
    p.save();baseline=np.asarray(R.DoNoFit(str(output),h.name),dtype=float)
    old=copy.deepcopy(parent.data);hap=copy.deepcopy(old['Histograms'][h.name]);trans=np.asarray(candidate['transformation_matrix']);origin=np.asarray(candidate['origin_shift'])
    det=abs(np.linalg.det(trans))
    if det>protocol['policy']['maximum_cell_multiplier']:raise TrialDisposition('unsupported','Candidate exceeds the declared cell-expansion budget')
    effective=L.CellDijCorr(old['General']['Cell'][1:7],old['General']['SGData'],old['Histograms'],h.name)
    child_cell=list(L.TransformCell(effective,trans))
    authoritative=confined(ctx.run_dir.parent,candidate['child']['path'])
    native_cif,import_receipt=prepare_isodistort_cif(authoritative,ctx.run_dir/'cif-import')
    ctx.save_json('cif-import.json',import_receipt)
    old_name=parent.name;p.data['Phases'].pop(old_name)
    for names in p.names:
        if names and names[0]=='Phases' and old_name in names:names.remove(old_name)
    h.data.get('Reflection Lists',{}).pop(old_name,None);p.data['Covariance']['data']={}
    child,import_receipt=add_isodistort_phase(p,authoritative,phasename=old_name+' distortion '+candidate['id'][-8:],histograms=[h.name],fmthint='CIF')
    ctx.save_json('cif-import.json',import_receipt)
    child.data['General']['Cell'][1:]=child_cell
    iso=child.data.get('ISODISTORT',{});native_names=[g.name for g in iso.get('G2ModeList',[])]
    if not native_names or not set(candidate['modes']).issubset(native_names):raise TrialDisposition('integrity_failure','The discovered native mode identities changed')
    if set(seed['mode_values'])!=set(candidate['modes']):raise TrialDisposition('unsupported','Provide a nonzero signed start for every selected native mode')
    mapping=_mapped_parent_sites(old,child.data,trans,origin,S)
    pcx,pct,_,pcia=old['General']['AtomPtrs'];cx,ct,_,cia=child.data['General']['AtomPtrs']
    site_mapping=[]
    for atom,parent_atom,xyz in zip(child.data['Atoms'],mapping,iso['G2parentCoords']):
        if parent_atom[pcia]!='I':raise TrialDisposition('unsupported','Anisotropic parent ADPs need an exact tensor transfer')
        atom[cx:cx+3]=list(xyz);atom[cx+3]=parent_atom[pcx+3];atom[cia]='I';atom[cia+1]=parent_atom[pcia+1]
        atom[cia+2:cia+8]=[0.]*6
        site_mapping.append(dict(child=atom[ct-1],parent=parent_atom[pct-1]))
    represented={a[pct-1]:0. for a in old['Atoms']}
    for atom,parent_atom in zip(child.data['Atoms'],mapping):represented[parent_atom[pct-1]]+=S.SytSym(atom[cx:cx+3],child.data['General']['SGData'])[1]*atom[cx+3]
    for atom in old['Atoms']:
        expected=S.SytSym(atom[pcx:pcx+3],old['General']['SGData'])[1]*atom[pcx+3]*det
        if not np.isclose(represented[atom[pct-1]],expected,rtol=0,atol=1e-6):raise TrialDisposition('effect_mismatch','The child does not preserve parent-site number density')
    ch=child.data['Histograms'][h.name]
    for family in ['Size','Mustrain']:
        if str(hap[family][0]).lower()!='isotropic':raise TrialDisposition('unsupported','Non-isotropic '+family+' needs a verified tensor transfer')
        ch[family][1][0]=hap[family][1][0]
    if abs(hap['Pref.Ori.'][1]-1.)>1e-12:raise TrialDisposition('unsupported','Nontrivial texture needs an exact orientation transfer')
    for family in ['Extinction','Babinet']:ch[family]=copy.deepcopy(hap[family])
    ch['Scale']=[hap['Scale'][0]/det,False];ch['HStrain']=[[0.]*len(S.HStrainNames(child.data['General']['SGData'])),[False]*len(S.HStrainNames(child.data['General']['SGData']))]
    iso['modeDispl']=[0.]*len(native_names)
    p.update_ids();freeze(p);p.save()
    zero=np.asarray(R.DoNoFit(str(output),h.name),dtype=float)
    weight=np.asarray(h.data['data'][1][2]);x=np.asarray(h.data['data'][1][0]);lo,hi=h.data['Limits'][1]
    mask=(weight>0)&(x>=lo)&(x<=hi)
    for a,b in h.data['Limits'][2:]:mask&=(x<a)|(x>b)
    error=float(np.sqrt(np.sum(weight[mask]*(zero[mask]-baseline[mask])**2)/max(np.sum(weight[mask]*baseline[mask]**2),1e-30)))
    ctx.save_json('zero-control.json',dict(relative_profile_error=error,threshold=1e-5,site_mapping=site_mapping,exact_parent_cell_transferred=True,phase_scale=ch['Scale'][0],determinant=det))
    if not np.isfinite(error) or error>1e-5:raise TrialDisposition('effect_mismatch','The zero-mode child does not reproduce the native parent calculation')
    iso['modeDispl']=[seed['mode_values'].get(name,0.) for name in native_names]
    drawing=child.data.setdefault('Drawing',{});drawing['atomPtrs']=[2,1,6,17];drawing['Atoms']=[M.MakeDrawAtom(child.data,a) for a in child.data['Atoms']]
    if M.ApplyModeDisp(child.data):raise TrialDisposition('effect_mismatch','Native mode displacement application failed')
    mode_constraints=[c for c in p.data['Constraints']['data']['Phase'] if c[-1]=='f']
    if len(mode_constraints)!=len(native_names):raise TrialDisposition('integrity_failure','Native mode-constraint inventory changed')
    for constraint,name in zip(mode_constraints,native_names):constraint[-2]=name in candidate['modes']
    active={(g.atom,g.name[-1]) for g in iso['G2VarList']}
    for atom in child.data['Atoms']:
        if any((atom[-1],axis) in active for axis in 'xyz'):
            atom[ct+1]='X'
            for axis in 'xyz':
                if (atom[-1],axis) not in active:p.data['Constraints']['data']['Phase'].append([[1.,O.G2VarObj([child.data['ranId'],None,'dA'+axis,atom[-1]])],None,None,'h'])
    if protocol['policy']['refine_cell']:child.set_refinements({'Cell':True})
    if protocol['policy']['refine_background']:h.set_refinements({'Background':True})
    if protocol['policy']['refine_histogram_scale']:h.set_refinements({'Sample Parameters':['Scale']})
    for name in candidate['modes']:
        # Native nv-* values are changes from the nonzero start, not absolute Q.
        variable=f'{child.id}::nv-{name}';q0=seed['mode_values'][name];bound=protocol['policy']['maximum_mode']
        p.set_Controls('parmMin',-bound-q0,variable=variable);p.set_Controls('parmMax',bound-q0,variable=variable)
    p.set_Controls('cycles',protocol['policy']['cycles']);p.save();p.do_refinements([{}]);p.save()
    cov=p.data.get('Covariance',{}).get('data',{});rvals=cov.get('Rvals',{})
    _,_,modes,_=M.CalcIsoDisp(child.data);moments=dict(zip(native_names,map(float,modes)))
    physical=all(np.isfinite(v) and abs(v)<=protocol['policy']['maximum_mode']+1e-7 for v in moments.values())
    actual_cell=np.asarray(child.data['General']['Cell'][1:7]);initial_cell=np.asarray(child_cell[:6])
    physical &= bool(np.all(np.abs(actual_cell[:3]/initial_cell[:3]-1)<=protocol['policy']['maximum_cell_relative_change']) and np.all(np.abs(actual_cell[3:]-initial_cell[3:])<=protocol['policy']['maximum_angle_change_degrees']))
    status='physical_rejection' if not physical else 'optimization_unresolved' if not rvals.get('converged') or rvals.get('Aborted') or rvals.get('SVD0',0) else 'valid'
    return dict(status=status,gpx=output.name,rwp=float(rvals.get('Rwp',h.get_wR())),nvars=int(rvals.get('Nvars',len(cov.get('varyList',[])))),modes=moments,zero_control_error=error)
