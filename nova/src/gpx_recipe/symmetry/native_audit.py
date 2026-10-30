"""Independent native output checks and bounded browser-safe curve exports."""
from __future__ import annotations

import copy
import importlib
from pathlib import Path
import numpy as np

from ..snapshot import gsas,json_safe
from ..execution import write_json
from .worker import TrialDisposition


def values(value):
    if isinstance(value,dict):return {k:values(v) for k,v in value.items() if k!='Use'}
    if isinstance(value,(list,tuple,np.ndarray)):return [values(v) for v in value if not isinstance(v,(bool,np.bool_))]
    return json_safe(value)


def same(a,b):
    a,b=values(a),values(b)
    if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):return bool(np.isclose(a,b,rtol=0,atol=1e-10,equal_nan=True))
    return a==b


def nuclear_atoms(phase):
    cx,ct,_,cia=phase.data['General']['AtomPtrs']
    return {a[ct-1]:[a[ct],*a[cx:cx+4],a[cia],*a[cia+1:cia+8]] for a in phase.data['Atoms']}


def inspect_source(path):
    from gsas_magnetic_preflight import setting_fingerprint
    from gsas_cif_import import _gsas_operations
    from .compiler import file_sha
    G=gsas();p=G.G2Project(gpxfile=str(path));phases=[]
    for phase in p.phases():
        cx,ct,_,cia=phase.data['General']['AtomPtrs'];general=phase.data['General']
        phases.append(dict(name=phase.name,type=general['Type'],setting_sha256=setting_fingerprint(general['SGData']),space_group=general['SGData']['SpGrp'],
            cell=general['Cell'][1:7],operations=[dict(rotation=np.asarray(m).tolist(),translation=np.asarray(t).tolist()) for m,t in _gsas_operations(general['SGData'])],
            atoms=[dict(label=a[ct-1],type=a[ct],xyz=a[cx:cx+3],occupancy=a[cx+3],adp_type=a[cia],uiso=a[cia+1]) for a in phase.data['Atoms']],
            histograms=list(phase.data['Histograms'])))
    from .worker import runtime_identity
    return json_safe(dict(schema='radar-symmetry-source/v1',sha256=file_sha(path),phases=phases,
        histograms=[dict(name=h.name,points=len(h.data['data'][1][0]),limits=h.data['Limits'],instrument=h.InstrumentParameters) for h in p.histograms()],runtime=runtime_identity()))


def audit_trial(source,output,protocol,candidate):
    G=gsas();a=G.G2Project(gpxfile=str(source));b=G.G2Project(gpxfile=str(output))
    ah=a.histogram(protocol['histogram']);bh=b.histogram(protocol['histogram']);parent=a.phase(protocol['parent_phase'])
    from .native_preparation import normalize_flags
    normalize_flags(a);normalize_flags(b)
    # A native abort can restore an old calculated array while retaining
    # attempted-cycle Rvals. Always export the curve of the actual checkpoint.
    R=importlib.import_module((G.__package__+'.' if G.__package__ else '')+'GSASIIstrMain')
    fresh=np.asarray(R.DoNoFit(str(output),bh.name),dtype=float)
    if fresh.shape!=np.shape(bh.data['data'][1][0]) or not np.all(np.isfinite(fresh)):raise TrialDisposition('effect_mismatch','The final checkpoint could not be forward calculated')
    bh.data['data'][1][3]=fresh;bh.data['data'][1][5]=np.asarray(bh.data['data'][1][1])-fresh
    b.save()
    checks={
        'observations_weights_preserved':all(np.array_equal(x,y,equal_nan=True) and np.array_equal(np.ma.getmaskarray(x),np.ma.getmaskarray(y)) for x,y in zip(ah.data['data'][1][:3],bh.data['data'][1][:3])),
        'limits_exclusions_preserved':same(ah.data['Limits'],bh.data['Limits']),
        'instrument_preserved':same(ah.data['Instrument Parameters'],bh.data['Instrument Parameters']),
        'sample_state_preserved':same({k:v for k,v in ah.data['Sample Parameters'].items() if not (k=='Scale' and protocol['policy']['refine_histogram_scale'])},{k:v for k,v in bh.data['Sample Parameters'].items() if not (k=='Scale' and protocol['policy']['refine_histogram_scale'])}),
    }
    if not protocol['policy']['refine_background']:checks['background_preserved']=same(ah.data['Background'],bh.data['Background'])
    checks['other_phases_preserved']=all(any(q.name==phase.name and same(nuclear_atoms(phase),nuclear_atoms(q)) and same(phase.data['General']['Cell'][1:],q.data['General']['Cell'][1:]) and same(phase.data['Histograms'][ah.name],q.data['Histograms'][bh.name]) for q in b.phases()) for phase in a.phases() if phase.name!=protocol['parent_phase'])
    cov=b.data.get('Covariance',{}).get('data',{});vary=[str(x) for x in cov.get('varyList',[])]
    if protocol['kind']=='magnetic':
        import GSASII.GSASIIlattice as L
        nuclear=b.phase(protocol['parent_phase']);magnets=[p for p in b.phases() if p.data['General']['Type']=='magnetic']
        checks.update(nuclear_atoms_preserved=same(nuclear_atoms(parent),nuclear_atoms(nuclear)),nuclear_cell_preserved=same(parent.data['General']['Cell'][1:],nuclear.data['General']['Cell'][1:]),
            nuclear_hap_values_preserved=same(parent.data['Histograms'][ah.name],nuclear.data['Histograms'][bh.name]),
            one_magnetic_phase=len(magnets)==1,
            moment_only_vary=bool(vary) and all('::AM' in name or name.startswith('::constr') or (protocol['policy']['refine_histogram_scale'] and name==f':{bh.id}:Scale') or (protocol['policy']['refine_background'] and name in {f':{bh.id}:Back;{i}' for i in range(ah.data['Background'][0][2])}) for name in vary))
        if protocol['policy']['refine_histogram_scale']:
            checks['declared_common_scale_varied']=f':{bh.id}:Scale' in vary
            checks['positive_common_scale']=bool(np.isfinite(bh.data['Sample Parameters']['Scale'][0]) and bh.data['Sample Parameters']['Scale'][0]>0)
        if protocol['policy']['refine_background']:
            checks['declared_background_coefficients_varied']={f':{bh.id}:Back;{i}' for i in range(ah.data['Background'][0][2])}.issubset(set(vary))
            checks['background_basis_and_extra_terms_preserved']=same(ah.data['Background'][0][:3],bh.data['Background'][0][:3]) and len(ah.data['Background'][0])==len(bh.data['Background'][0]) and same(ah.data['Background'][1:],bh.data['Background'][1:])
        if len(magnets)==1:
            magnetic=magnets[0];det=abs(np.linalg.det(candidate['transformation_matrix']))
            from .displacements import verify_fixed_displacements
            import json
            displacement_receipt=json.loads(Path(output).with_name('fixed-displacements.json').read_text())
            checks['fixed_magnetic_displacement_ellipsoids_preserved']=verify_fixed_displacements(parent.data,magnetic.data,displacement_receipt,candidate['transformation_matrix'])
            from .magnetic_identity import verify_native_operators,verify_collinear_structure,verify_symmetry_moments
            try:verify_native_operators(candidate,magnetic.data['General']['SGData']);checks['complete_magnetic_operators_preserved']=True
            except ValueError:checks['complete_magnetic_operators_preserved']=False
            if protocol['policy'].get('moment_model','collinear')=='symmetry_allowed':
                try:verify_symmetry_moments(magnetic.data);checks['site_symmetry_moments_preserved']=True
                except ValueError:checks['site_symmetry_moments_preserved']=False
            else:
                try:verify_collinear_structure(magnetic.data);checks['full_structure_collinear']=True
                except ValueError:checks['full_structure_collinear']=False
            if 'magnetic_d_min' in protocol['policy']:
                declared=protocol['policy']['magnetic_d_min'];actual=magnetic.data['General'].get('MagDmin')
                if declared:
                    checks['magnetic_reflection_domain_preserved']=bool(actual is not None and np.isclose(actual,declared,rtol=0,atol=1e-10))
                else:
                    source_x=np.asarray(ah.data['data'][1][0],dtype=float)
                    source_d=np.asarray(L.Pos2dsp(ah.InstrumentParameters,source_x),dtype=float)
                    low,high=ah.data['Limits'][1]
                    domain=(source_x>=low)&(source_x<=high)&np.isfinite(source_d)&(source_d>0)
                    checks['magnetic_reflection_domain_preserved']=bool(domain.any() and actual is not None and 0<actual<float(source_d[domain].min()))
            checks['magnetic_cell_linked']=bool(np.allclose(L.TransformCell(nuclear.data['General']['Cell'][1:7],candidate['transformation_matrix'])[:6],magnetic.data['General']['Cell'][1:7],atol=1e-8,rtol=0))
            checks['material_amount_linked']=bool(np.isclose(magnetic.data['Histograms'][bh.name]['Scale'][0]*det,nuclear.data['Histograms'][bh.name]['Scale'][0],rtol=0,atol=1e-10))
    else:
        children=[p for p in b.phases() if p.data.get('ISODISTORT')]
        if len(children)!=1:raise TrialDisposition('effect_mismatch','The output has no unique displacement-mode child')
        child=children[0];cx,ct,_,cia=child.data['General']['AtomPtrs']
        checks['structural_vary_closed']=bool(vary) and all(('::nv-' in name and name.split('::nv-',1)[1] in candidate['modes']) or (protocol['policy']['refine_cell'] and name.split('::')[-1] in {'A0','A1','A2','A3','A4','A5'}) or (protocol['policy']['refine_background'] and ':Back;' in name) or (protocol['policy']['refine_histogram_scale'] and name==f':{bh.id}:Scale') for name in vary)
        checks['native_mode_inventory_present']=bool(child.data.get('ISODISTORT',{}).get('G2ModeList'))
        required={f'{child.id}::nv-{name}' for name in candidate['modes']}
        checks['selected_modes_actually_varied']=required.issubset(set(vary))
        iso=child.data['ISODISTORT'];allowed={(g.atom,g.name[-1]) for g in iso['G2VarList']}
        checks['non_mode_coordinates_fixed']=all((atom[-1],axis) in allowed or np.isclose(atom[cx+i],zero[i],rtol=0,atol=1e-10) for atom,zero in zip(child.data['Atoms'],iso['G2parentCoords']) for i,axis in enumerate('xyz'))
        mapping=__import__('json').loads(Path(output).with_name('zero-control.json').read_text())['site_mapping']
        old={a[parent.data['General']['AtomPtrs'][1]-1]:a for a in parent.data['Atoms']};pcx,pct,_,pcia=parent.data['General']['AtomPtrs']
        checks['occupancies_and_adps_preserved']=all(np.isclose(atom[cx+3],old[row['parent']][pcx+3],rtol=0,atol=1e-10) and atom[cia]==old[row['parent']][pcia] and np.isclose(atom[cia+1],old[row['parent']][pcia+1],rtol=0,atol=1e-10) for atom,row in zip(child.data['Atoms'],mapping))
    if not all(checks.values()):raise TrialDisposition('effect_mismatch','Native output violates the declared parameter/data contract: '+str({k:v for k,v in checks.items() if not v}))
    try:metrics=profile_residual_metrics(bh.data['data'][1][:3],fresh,bh.data['Limits'])
    except ValueError as exc:raise TrialDisposition('effect_mismatch',str(exc))
    result=dict(checks=checks,actual_vary=vary,passed=True,rwp_recomputed=metrics['rwp_native'],
        residual_metrics=metrics,calculated_profile_refreshed=True)
    write_json(Path(output).with_suffix('.audit.json'),result)
    write_view(b,protocol['histogram'],Path(output).with_suffix('.view.json'))
    return result


def profile_residual_metrics(arrays,calculated,limits):
    """Independently reproduce GSAS-II errRefine's powder Rwp convention.

    GSAS-II masks excluded residuals but keeps their observed intensity in its
    normalization. Also retain a fit-points-only metric with an explicit label.
    Neither changes observations, weights, masks or the objective being fitted.
    """
    x,observed,weight=[np.asarray(np.ma.getdata(a),dtype=float) for a in arrays]
    calculated=np.asarray(calculated,dtype=float)
    lo,hi=limits[1];domain=np.zeros(len(x),dtype=bool)
    domain[np.searchsorted(x,lo):np.searchsorted(x,hi)+1]=True
    valid=np.isfinite(observed)&np.isfinite(weight)&(weight>0)
    for array in arrays[1:]:valid &= ~np.ma.getmaskarray(array)
    normalization=domain&valid
    fit=normalization&np.isfinite(x)&np.isfinite(calculated)&~np.ma.getmaskarray(arrays[0])
    for aa,bb in limits[2:]:fit &= (x<aa)|(x>bb)
    denominator=float(np.sum(weight[normalization]*observed[normalization]**2))
    fitted_denominator=float(np.sum(weight[fit]*observed[fit]**2))
    if not fit.any() or min(denominator,fitted_denominator)<=0:
        raise ValueError('There is no finite, nonempty common fit domain')
    numerator=float(np.sum(weight[fit]*(observed[fit]-calculated[fit])**2))
    return dict(rwp_native=100*float(np.sqrt(numerator/denominator)),
        rwp_fit_points_only=100*float(np.sqrt(numerator/fitted_denominator)),
        definition='GSAS-II powder errRefine; excluded residuals omitted; full native fit-range observed normalization',
        fitted_points=int(fit.sum()),normalization_points=int(normalization.sum()))


def write_view(project,histogram,destination):
    """Export directly from native data; ordinary nuclear snapshot rejects magnetism."""
    G=gsas();prefix=G.__package__+'.' if G.__package__ else ''
    L=importlib.import_module(prefix+'GSASIIlattice');M=importlib.import_module(prefix+'GSASIImath')
    h=project.histogram(histogram);arrays=h.data['data'][1]
    x,y,w,calc,bg=map(lambda a:np.asarray(np.ma.getdata(a),dtype=float),arrays[:5])
    valid=np.isfinite(x)&np.isfinite(y)&np.isfinite(w)&(w>0)
    for array in arrays[:3]:valid &= ~np.ma.getmaskarray(array)
    low,high=h.data['Limits'][1];fitted=valid&(x>=low)&(x<=high)
    for a,b in h.data['Limits'][2:]:fitted &= (x<a)|(x>b)
    d=np.asarray(L.Pos2dsp(h.InstrumentParameters,x));phases=[];ticks=[]
    from gsas_magnetic_preflight import magnetic_magnitude
    for phase in project.phases():
        general=phase.data['General'];row=dict(name=phase.name,type=general['Type'],space_group=general['SGData'].get('MagSpGrp') or general['SGData']['SpGrp'],cell=general['Cell'][1:7])
        if general['Type']=='magnetic':
            cx,ct,_,_=general['AtomPtrs'];row['moments']=[dict(label=a[ct-1],components=a[cx+4:cx+7],magnitude=magnetic_magnitude(a[cx+4:cx+7],general['Cell'][1:7])) for a in phase.data['Atoms']]
        if phase.data.get('ISODISTORT'):
            _,_,modes,_=M.CalcIsoDisp(phase.data);row['modes']=[dict(label=label,value=float(value)) for label,value in zip(phase.data['ISODISTORT']['IsoModeList'],modes)]
        phases.append(row)
        if phase.data.get('Histograms',{}).get(histogram,{}).get('Use'):
            refs=np.asarray(h.data.get('Reflection Lists',{}).get(phase.name,{}).get('RefList',[]))
            if refs.ndim==2 and refs.shape[1]>5:ticks.append(dict(phase=phase.name,native=refs[:,5].tolist(),d=refs[:,4].tolist()))
    view=json_safe(dict(schema='radar-symmetry-view/v1',histogram=histogram,coordinate='tof' if 'T' in str(h.InstrumentParameters['Type'][0]) else 'two_theta',x=x,d=d,observed=y,calculated=calc,background=bg,weights=w,fitted=fitted,limits=[low,high],phases=phases,ticks=ticks))
    write_json(destination,view);return view
