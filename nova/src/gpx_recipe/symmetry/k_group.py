"""One checked current magnetic response for a finite k-survey seed group.

No optimization or model elimination takes place here. Each original seed is
constructed in the shared constraint basis and retains its own operator,
fixed-input and propagation-periodicity audit. Unsupported contexts use the
ordinary native simulation path through the caller.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import numpy as np

from ..execution import write_json
from ..native_forward import forward_profile,tick_arrays
from ..preparation.hints import cache_for
from ..preparation.native_magnetic import render_current_response,moment_basis,physics_identity,current_native_profiles
from ..live.native_profile import NativeReducedProfile
from .compiler import confined,file_sha,package_signature,verify_inputs
from .contracts import digest,trial_id,validate_protocol
from .native_magnetic import prepare
from .profiled_groups import native_seed,seed_moments
from .worker import Artifacts,TrialDisposition,require_runtime,runtime_identity


def checked_job(directory):
    directory=Path(directory).resolve();job=json.loads((directory/'job.json').read_text())
    protocol=validate_protocol(job['protocol']);sha=digest(protocol)
    if (job['schema']!='radar-symmetry-job/v1' or job['protocol_sha256']!=sha
            or job['id']!=trial_id(protocol,job['candidate_id'],job['seed_id'])
            or job['package_sha256']!=package_signature()
            or protocol['implementation_sha256']!=package_signature()):
        raise ValueError('K-survey job differs from its reviewed inputs or implementation')
    verify_inputs(protocol,directory)
    candidate=next(c for c in protocol['candidates'] if c['id']==job['candidate_id'])
    seed=next(s for s in protocol['seeds'] if s['id']==job['seed_id'])
    if 'candidate_ids' in seed and candidate['id'] not in seed['candidate_ids']:
        raise ValueError('K-survey seed is outside the candidate scope')
    return job,protocol,candidate,seed


def simulate_group(directories):
    """Return per-directory responses after preparing one shared subgroup."""
    from GSASII import GSASIIscriptable as G
    from .k_native import audit_simulation
    from .k_periodicity import audit_structure
    directories=[Path(p).resolve() for p in directories]
    if not 1<=len(directories)<=512 or len(set(directories))!=len(directories):
        raise ValueError('Declare a bounded unique k-survey group')
    entries=[checked_job(p) for p in directories]
    first,protocol,candidate,first_seed=entries[0];sha=digest(protocol)
    if (protocol['kind']!='magnetic' or protocol['policy'].get('moment_model')!='symmetry_allowed'
            or protocol['policy'].get('refine_cell')):
        raise ValueError('Grouped k templates require fixed-geometry symmetry-allowed moments')
    if any(job['protocol_sha256']!=sha or c!=candidate for job,p,c,s in entries):
        raise ValueError('K-survey group changes its source context or subgroup')
    runtime=runtime_identity();require_runtime(runtime,protocol['magnetic_function_sha256'])
    source=confined(directories[0],protocol['source']['path'])
    # This derived protocol selects the existing construction-only API. The
    # complete scientific scope, seed choices and all returned IDs stay bound
    # to the original reviewed protocol; no fit is performed.
    construction=copy.deepcopy(protocol)
    construction['policy'].update(magnetic_optimizer='profiled_quadratic',optimizer_iterations=1)
    construction=validate_protocol(construction)
    scratch=directories[0]/'shared-k-physics';scratch.mkdir(exist_ok=False)
    began=time.monotonic()
    built=prepare(source,Artifacts(scratch,input_root=directories[0]),construction,candidate,first_seed)
    path=scratch/built['gpx']
    original=G.G2Project(gpxfile=str(path))
    magnetic=original.phase(built['seed_template']['magnetic_phase'])
    ct=magnetic.data['General']['AtomPtrs'][1]
    labels=[a[ct-1] for a in magnetic.data['Atoms']]
    x=np.asarray(original.histogram(protocol['histogram']).data['data'][1][0],float).copy()
    state=json.loads((scratch/'prepared-state.json').read_text())
    baseline=np.asarray(state['baseline_target_profiles'][protocol['histogram']]['arrays'][3],float)
    native=NativeReducedProfile(path)
    factor_cache,preparation=cache_for(directories[0],protocol,candidate=candidate['id'])
    basis=moment_basis(native);components=basis.components.copy()
    starts=np.asarray([native_seed(built['seed_template'],seed,components,labels) for _,_,_,seed in entries])
    identity=physics_identity(native,basis)
    dimension=starts.shape[1]
    # Full coefficient qualification is intentionally not free. For short
    # finite surveys, shared exact native evaluation is cheaper than building
    # a tensor from factor-only hints. A selected rendered response must reach
    # the same current-context compiler as Model/Live even for a short group;
    # the compiler still checks its identity, provenance and native probes.
    threshold=2*dimension**2+2*dimension+8
    if factor_cache is not None and (len(entries)>=threshold or preparation.get('rendered_responses_attached',0)):
        response=render_current_response(native,factor_cache=factor_cache)
        histogram=native.histo_phases[0][native.parts[0][0]]
        scale=float(native.parms[f":{histogram['hId']}:Scale"])
        physical_x=response.x
        curves=scale*np.einsum('si,nij,sj->sn',starts,response.quadratic,starts)
        compiler=dict(method='qualified_quadratic_response',proof=response.proof)
        reuse=response.proof['compiler'].get('native_reflection_diagnostics') or {}
        preparation['scattering_factors_reused']=bool(reuse.get('scattering_factors_reused',False))
        preparation['rendered_response_reused']=bool(reuse.get('prepared_rendered_response_reused',False))
        preparation['used']=preparation['scattering_factors_reused'] or preparation['rendered_response_reused']
    else:
        physical_x,curves,compiler=current_native_profiles(native,starts)
        preparation.update(used=False,scattering_factors_reused=False,rendered_response_reused=False)
    indices=np.searchsorted(x,physical_x)
    if (np.any(indices>=len(x)) or not np.array_equal(x[indices],physical_x)
            or baseline.shape!=x.shape):
        raise ValueError('K-survey response differs from the actual calibrated native grid')
    # Native rendering zeros excluded intervals and out-of-range bins. Bins
    # masked only in observation weights still retain their physical profile;
    # the survey applies the actual observation mask separately when scoring.
    source_histogram=original.histogram(protocol['histogram'])
    valid=np.zeros(len(x),dtype=bool);valid[indices]=True
    lo,hi=source_histogram.data['Limits'][1];valid&=(x>=lo)&(x<=hi)
    for lo,hi in source_histogram.data['Limits'][2:]:valid&=~((x>=lo)&(x<=hi))
    if not valid.any():raise ValueError('K-survey has no original fitted physical bins')
    proof=dict(schema='radar-k-group-physics/v1',protocol_sha256=sha,
        construction_protocol_sha256=digest(construction),candidate_id=candidate['id'],
        prepared_scattering=preparation,physics_sha256=identity,
        native_constructions=1,declared_starts=len(entries),native_compiler=compiler,
        preparation_seconds=time.monotonic()-began,no_refinement=True)
    write_json(scratch/'k-physics.json',proof)
    results=[];ticks=None;independent_checked=False
    for directory,(job,p,c,seed),curve in zip(directories,entries,curves):
        output=directory/'simulation';output.mkdir(exist_ok=False)
        try:
            moments=seed_moments(built['seed_template'],seed,labels)
            project=G.G2Project(gpxfile=str(path),newgpx=str(output/'magnetic-refinement.gpx'))
            phase=project.phase(built['seed_template']['magnetic_phase'])
            cx=phase.data['General']['AtomPtrs'][0]
            for atom,value in zip(phase.data['Atoms'],moments):atom[cx+4:cx+7]=value.tolist()
            gpx=output/'magnetic-refinement.gpx';project.save(str(gpx))
            audit=audit_simulation(confined(directory,p['source']['path']),gpx,p,c)
            if not audit['passed']:raise TrialDisposition('effect_mismatch','Grouped k construction changed fixed inputs')
            parent=project.phase(p['parent_phase']).data
            try:periodicity=audit_structure(parent,phase.data,c,p['k'])
            except ValueError as exc:periodicity=dict(passed=False,reason=str(exc))
            if not periodicity['passed']:
                results.append(dict(directory=directory,response=dict(status='k_periodicity_mismatch',
                    reason=periodicity['reason'],candidate_id=c['id'],seed_id=seed['id'])))
                continue
            intensity=np.zeros(len(x))
            intensity[indices]=curve
            intensity[~valid]=0.
            tolerance=1e-8*max(1.,float(np.max(abs(baseline+intensity))))
            if not np.isfinite(intensity).all() or np.min(intensity)<-tolerance:
                raise ValueError('Grouped magnetic contribution is nonfinite or negative')
            if not independent_checked:
                direct,ticks=forward_profile(gpx,p['histogram'])
                # Native forward arrays are zero outside the refined range.
                # Compare on this exact response domain, with original masks.
                error=float(np.max(abs((baseline+intensity-direct)[valid]))/max(float(np.max(abs(intensity[valid]))),1e-12))
                if not np.isfinite(error) or error>1e-8:
                    raise ValueError('Grouped k response differs from the independent native magnetic contribution')
                independent_checked=True;proof['independent_seed_error']=error
                proof['independent_seed_id']=job['id'];write_json(scratch/'k-physics.json',proof)
            target=output/'magnetic-profile.npz'
            np.savez_compressed(target,intensity=np.maximum(intensity,0.),**tick_arrays(ticks))
            write_json(output/'simulation-audit.json',audit)
            write_json(output/'propagation-periodicity-audit.json',periodicity)
            results.append(dict(directory=directory,response=dict(status='simulated',id=job['id'],
                protocol_sha256=sha,source_sha256=p['source']['sha256'],runtime=runtime,
                profile=str(target.relative_to(directory)),profile_sha256=file_sha(target),
                gpx=str(gpx.relative_to(directory)),audit=audit,periodicity=periodicity,
                no_refinement=True,initial_maximum_moment=audit['maximum_moment'],
                magnetic_atoms=len(phase.data['Atoms']),shared_physics=proof)))
        except Exception as error:
            results.append(dict(directory=directory,response=dict(status=getattr(error,'status','failed'),
                reason=str(error),error_type=type(error).__name__)))
        finally:verify_inputs(p,directory)
    return results
