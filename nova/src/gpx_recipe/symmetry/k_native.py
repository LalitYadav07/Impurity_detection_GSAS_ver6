"""Native-only preparation, compatibility inventory and sealed forward jobs.

This module runs in scientific Python, never in the web process. It accepts
typed project/catalog data, not executable model output, and needs no secrets.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import resource
import numpy as np

from ..snapshot import gsas,json_safe
from ..execution import write_json
from .compiler import file_sha,confined,verify_inputs,package_signature
from .contracts import digest,trial_id,validate_protocol,validate_candidate
from .worker import runtime_identity,require_runtime,Artifacts,TrialDisposition


def _open(source):
    return gsas().G2Project(gpxfile=str(source))


def prepare(request,root):
    from .native_audit import inspect_source
    from ..commensurate_k import recover_points
    from GSASII import GSASIIlattice as L,GSASIIstrMain as M
    import spglib
    source=Path(request['source']).resolve();before=file_sha(source)
    if before!=request['source_sha256']:raise ValueError('Starting GPX changed before k preparation')
    context=inspect_source(source);require_runtime(context['runtime'],context['runtime']['magnetic_function_sha256'])
    p=_open(source);phase=p.phase(request['parent']);h=p.histogram(request['histogram'])
    if any(q.data['General'].get('Type')!='nuclear' or q.data['General'].get('Modulated') for q in p.phases()):
        raise ValueError('Start k screening from a nuclear-only, unmodulated checkpoint')
    if h.name not in phase.data['Histograms'] or not phase.data['Histograms'][h.name].get('Use',True):raise ValueError('The parent is not actively linked to this bank')
    if not str(h.InstrumentParameters['Type'][0]).startswith('PN'):raise ValueError('Magnetic k screening requires neutron powder data')
    parent=next(q for q in context['phases'] if q['name']==phase.name)
    # Expand exact native operations. Preserve mixed-site chemistry as one
    # composite spglib type, rather than duplicating atoms at the same position.
    positions=[];motifs=[]
    for atom in parent['atoms']:
        if atom['occupancy']<=0:continue
        if not np.isfinite(atom['occupancy']) or atom['occupancy']>1+1e-7:raise ValueError('Review nonphysical parent occupancies before k screening')
        orbit=[]
        for op in parent['operations']:
            pos=(np.asarray(op['rotation'])@atom['xyz']+op['translation'])%1
            if any(np.max(np.abs((pos-old+.5)%1-.5))<1e-7 for old in orbit):continue
            orbit.append(pos)
            found=next((i for i,old in enumerate(positions) if np.max(np.abs((pos-old+.5)%1-.5))<1e-7),None)
            if found is None:found=len(positions);positions.append(pos);motifs.append([])
            motifs[found].append((atom['type'],round(atom['occupancy'],7)))
    if not positions or len(positions)>5000:raise ValueError('Expanded parent exceeds the k-survey size budget')
    types={};numbers=[]
    for motif in motifs:
        token=tuple(sorted(motif));types.setdefault(token,len(types)+1);numbers.append(types[token])
    structure=(L.cell2AB(parent['cell'])[0].T,np.asarray(positions),numbers)
    points=recover_points(structure,parent['operations'],maximum_denominator=int(request.get('maximum_denominator',8)))
    x,observed,weights=[np.ma.asarray(v,dtype=float) for v in h.data['data'][1][:3]]
    nuclear=np.asarray(M.DoNoFit(str(source),h.name),dtype=float)
    ds=np.asarray(L.Pos2dsp(h.InstrumentParameters,np.ma.getdata(x)),dtype=float)
    lo,hi=h.data['Limits'][1]
    mask=(np.ma.getdata(x)>=lo)&(np.ma.getdata(x)<=hi)&np.isfinite(ds)&(ds>0)
    for values in (x,observed,weights):mask&=~np.ma.getmaskarray(values)&np.isfinite(np.ma.getdata(values))
    mask&=np.ma.getdata(weights)>0
    for low,high in h.data['Limits'][2:]:mask&=~((x>=min(low,high))&(x<=max(low,high)))
    if not mask.any():raise ValueError('No calibrated fitted bins are available')
    profile=root/'baseline.npz'
    np.savez_compressed(profile,x=np.ma.getdata(x),q=2*np.pi/ds,observed=np.ma.getdata(observed),
                        nuclear=nuclear,weights=np.ma.getdata(weights),mask=mask)
    if file_sha(source)!=before:raise ValueError('Native preparation changed the original GPX')
    return dict(schema='radar-k-native-prepare/v1',source_sha256=before,context=context,
                points=points,profile=profile.name,profile_sha256=file_sha(profile),
                parent=request['parent'],histogram=h.name,input_unchanged=True)


def inventory(request,root):
    from GSASII import GSASIIlattice as L,GSASIIspc as S
    from gsas_magnetic_preflight import preflight_magnetic_sites,NativeRepresentationError,direction_in_basis
    from .magnetic_identity import authoritative_operations,verify_native_operators,require_collinear_line
    from .chemistry import validate_sites
    from .k_periodicity import compatible_arms,star_vectors
    source=Path(request['source']).resolve()
    if file_sha(source)!=request['source_sha256']:raise ValueError('Inventory source changed')
    p=_open(source);nuclear=p.phase(request['parent'])
    cx,ct,_,_=nuclear.data['General']['AtomPtrs']
    validate_sites([dict(label=a[ct-1],type=a[ct],occupancy=a[cx+3]) for a in nuclear.data['Atoms']],request['sites'])
    axes=request.get('directions',['x','y','z'])
    if not axes or set(axes)-set('xyz'):raise ValueError('Invalid finite magnetic directions')
    results=[]
    parent_ops=[dict(rotation=r,translation=t) for r,t in S.AllOps(nuclear.data['General']['SGData'])[2]]
    for catalog in request['catalogs']:
        if catalog['parent_setting_sha256']!=request['parent_setting_sha256']:raise ValueError('Catalog belongs to another parent setting')
        compatible=[];dispositions=[];star=star_vectors(catalog['k'],parent_ops)
        for raw in catalog['candidates']:
            candidate=validate_candidate(raw)
            try:
                authority=authoritative_operations(candidate)
                definition=L.makeBilbaoPhase([candidate['magnetic_space_group'],candidate.get('bns_lattice','')],candidate['origin_shift'],candidate['transformation_matrix'],True)
                verify_native_operators(candidate,definition['SGData'],authority=authority)
                if not compatible_arms(catalog['k'],parent_ops,candidate['transformation_matrix'],S.AllOps(definition['SGData'])[2],definition['SGData']['SpnFlp'],star=star):
                    raise ValueError('The subgroup translation lattice cannot carry the requested k star; the service also returned higher-symmetry alternatives')
                cell=L.TransformCell(nuclear.data['General']['Cell'][1:7],candidate['transformation_matrix'])
                _,report=preflight_magnetic_sites(nuclear.data,{**candidate,'candidate_id':candidate['id']},
                    magnetic_sites=[s['label'] for s in request['sites']],site_policy='all',
                    expected_parent_setting=request['parent_setting_sha256'],maximum_cell_multiplier=8)
                if not report['compatible']:raise ValueError('Selected sites do not support nonzero moments in this subgroup')
                for axis in axes:
                    try:
                        direction=np.eye(3)['xyz'.index(axis)]
                        require_collinear_line([m for m,_ in S.AllOps(definition['SGData'])[2]],L.cell2AB(cell[:6])[0],cell[:3],direction)
                        if not all(direction_in_basis(site['basis'],direction) for site in report['sites']):raise ValueError('Selected sites do not support this subgroup/direction')
                        compatible.append(dict(candidate_id=candidate['id'],axis=axis,
                            family=candidate.get('bns_number') or candidate['magnetic_space_group'],
                            operation_count=len(authority),cell_multiplier=float(abs(np.linalg.det(candidate['transformation_matrix'])))))
                    except (ValueError,NativeRepresentationError) as exc:
                        dispositions.append(dict(candidate_id=candidate['id'],axis=axis,status='incompatible_or_unsupported',reason=str(exc)))
            except (ValueError,NativeRepresentationError) as exc:
                dispositions.append(dict(candidate_id=candidate['id'],status='incompatible_or_unsupported',reason=str(exc)))
        results.append(dict(k=catalog['k'],discovered=len(catalog['candidates']),eligible=compatible,dispositions=dispositions))
    return dict(schema='radar-k-native-inventory/v1',families=results,source_sha256=request['source_sha256'])


def audit_simulation(source,output,protocol,candidate):
    from .native_audit import same,nuclear_atoms
    from .magnetic_identity import verify_native_operators,verify_collinear_structure,verify_symmetry_moments
    from GSASII import GSASIIlattice as L
    a,b=_open(source),_open(output)
    from .native_preparation import normalize_flags
    normalize_flags(a);normalize_flags(b)
    h=a.histogram(protocol['histogram']);out=b.histogram(h.name)
    checks=dict(observations_weights_preserved=all(np.array_equal(x,y,equal_nan=True) and np.array_equal(np.ma.getmaskarray(x),np.ma.getmaskarray(y)) for x,y in zip(h.data['data'][1][:3],out.data['data'][1][:3])),
        limits_exclusions_preserved=same(h.data['Limits'],out.data['Limits']),instrument_preserved=same(h.data['Instrument Parameters'],out.data['Instrument Parameters']),
        background_preserved=same(h.data['Background'],out.data['Background']),sample_preserved=same(h.data['Sample Parameters'],out.data['Sample Parameters']))
    checks['all_nuclear_phases_preserved']=all(same(nuclear_atoms(phase),nuclear_atoms(b.phase(phase.name))) and same(phase.data['General']['Cell'][1:],b.phase(phase.name).data['General']['Cell'][1:]) and same(phase.data['Histograms'][h.name],b.phase(phase.name).data['Histograms'][h.name]) for phase in a.phases() if h.name in phase.data['Histograms'])
    magnetic=[p for p in b.phases() if p.data['General']['Type']=='magnetic']
    checks['one_magnetic_phase']=len(magnetic)==1
    if len(magnetic)==1:
        m=magnetic[0];parent=a.phase(protocol['parent_phase']);det=abs(np.linalg.det(candidate['transformation_matrix']))
        verify_native_operators(candidate,m.data['General']['SGData'])
        if protocol['policy'].get('moment_model','collinear')=='symmetry_allowed':
            verify_symmetry_moments(m.data);checks['site_symmetry_moments_verified']=True
        else:
            verify_collinear_structure(m.data);checks['collinearity_verified']=True
        checks.update(bns_operators_verified=True,
            nuclear_cell_linked=bool(np.allclose(L.TransformCell(parent.data['General']['Cell'][1:7],candidate['transformation_matrix'])[:6],m.data['General']['Cell'][1:7],atol=1e-8,rtol=0)),
            material_amount_linked=bool(np.isclose(m.data['Histograms'][h.name]['Scale'][0]*det,parent.data['Histograms'][h.name]['Scale'][0],atol=1e-10,rtol=0)))
        from gsas_magnetic_preflight import magnetic_magnitude
        cx=m.data['General']['AtomPtrs'][0]
        maximum=max(magnetic_magnitude(atom[cx+4:cx+7],m.data['General']['Cell'][1:7]) for atom in m.data['Atoms'])
        checks['declared_moment_bound']=bool(0<maximum<=protocol['policy']['maximum_moment']+1e-8)
    else:maximum=None
    return dict(checks=checks,passed=all(checks.values()),maximum_moment=maximum)


def template(request,root):
    directory=Path(request['directory']).resolve();job=json.loads((directory/'job.json').read_text())
    protocol=validate_protocol(job['protocol'])
    if job['schema']!='radar-symmetry-job/v1' or job['protocol_sha256']!=digest(protocol) or job['id']!=trial_id(protocol,job['candidate_id'],job['seed_id']):raise ValueError('Simulation job identity changed')
    if job['package_sha256']!=package_signature() or protocol['implementation_sha256']!=package_signature():raise ValueError('Simulation implementation changed')
    verify_inputs(protocol,directory);runtime=runtime_identity();require_runtime(runtime,protocol['magnetic_function_sha256'])
    if protocol['kind']!='magnetic':raise ValueError('The k survey only simulates magnetic hypotheses')
    candidate=next(c for c in protocol['candidates'] if c['id']==job['candidate_id'])
    seed=next(s for s in protocol['seeds'] if s['id']==job['seed_id'])
    output=directory/'simulation';output.mkdir(exist_ok=False)
    from .native_magnetic import simulate
    result=simulate(confined(directory,protocol['source']['path']),Artifacts(output),protocol,candidate,seed)
    verify_inputs(protocol,directory)
    result.update(id=job['id'],protocol_sha256=job['protocol_sha256'],runtime=runtime,
        profile=str((output/result['profile']).relative_to(root)),profile_sha256=file_sha(output/result['profile']),
        source_sha256=protocol['source']['sha256'])
    return result


def _portfolio_child(directory):
    """One disposable Linux child; the parent never opens a GPX."""
    directory=Path(directory);started=time.monotonic()
    try:result=template(dict(directory=str(directory)),directory)
    except Exception as exc:result=dict(status=getattr(exc,'status','failed'),reason=str(exc),error_type=type(exc).__name__)
    usage=resource.getrusage(resource.RUSAGE_SELF)
    result.update(elapsed_seconds=time.monotonic()-started,llm_calls=0,
        resources=dict(cpu_seconds=usage.ru_utime+usage.ru_stime,peak_rss_bytes=int(usage.ru_maxrss*1024)))
    write_json(directory/'portfolio-response.json',json_safe(result))


def portfolio(request,root):
    import multiprocessing
    import os
    import uuid
    from collections import deque
    if not os.sys.platform.startswith('linux'):raise ValueError('The fork-isolated native portfolio runs in the Linux scientific image')
    workers=request['workers'];seconds=request['seconds']
    if type(workers) is not int or not 1<=workers<=4 or not 0<float(seconds)<=600:raise ValueError('Invalid native portfolio bounds')
    directories=[confined(root,value) for value in request['directories']]
    if len(directories)>512 or len(set(directories))!=len(directories):raise ValueError('Invalid finite portfolio')
    # Warm modules only. No project is opened in the parent, so each child
    # starts from pristine GSAS-II project globals and is never reused.
    runtime=runtime_identity();require_runtime(runtime,runtime['magnetic_function_sha256'])
    from . import native_magnetic
    context=multiprocessing.get_context('fork');todo=deque(directories);active={};results=[]
    progress_path=root/('portfolio-progress-'+uuid.uuid4().hex+'.json')
    deadline=time.monotonic()+float(seconds)
    try:
        while (todo or active) and time.monotonic()<deadline:
            while todo and len(active)<workers and time.monotonic()<deadline:
                directory=todo.popleft();child=context.Process(target=_portfolio_child,args=(str(directory),));child.start();active[child.pid]=(child,directory)
            for pid,(child,directory) in list(active.items()):
                if child.is_alive():continue
                child.join();del active[pid]
                receipt=directory/'portfolio-response.json'
                response=json.loads(receipt.read_text()) if receipt.exists() else dict(status='failed',reason='Native child exited without a terminal response')
                results.append(dict(directory=str(directory.relative_to(root)),response=response))
            write_json(progress_path,dict(completed=len(results),total=len(directories),active=len(active)))
            if active:time.sleep(.03)
    finally:
        for child,_ in active.values():child.terminate()
        for child,_ in active.values():
            child.join(timeout=2)
            if child.is_alive():child.kill();child.join()
    return dict(schema='radar-k-native-portfolio/v1',results=results,
                pending=[str(p.relative_to(root)) for p in list(todo)+[d for _,d in active.values()]],
                runtime=runtime,workers=workers,fresh_process_per_template=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','inventory','template','portfolio'])
    parser.add_argument('request');parser.add_argument('output');args=parser.parse_args()
    output=Path(args.output).resolve();root=output.parent
    request=json.loads(Path(args.request).read_text());started=time.monotonic()
    try:result=globals()[args.action](request,root)
    except Exception as exc:
        result=dict(status=getattr(exc,'status','failed'),reason=str(exc),error_type=type(exc).__name__)
    result.update(elapsed_seconds=time.monotonic()-started,llm_calls=0)
    write_json(output,json_safe(result))


if __name__=='__main__':main()
