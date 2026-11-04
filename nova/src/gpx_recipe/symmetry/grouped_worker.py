"""One native preparation, then a complete surrogate search for one model.

Native verification runs only for distinct calculated-profile solutions.
Equivalent starts retain their own screened magnetic parameters and a checked
reference to the native representative. They are not native-verified GPXs.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
import traceback

import numpy as np

from ..execution import write_json
from .compiler import confined, file_sha, job_inputs, package_signature, verify_inputs
from .contracts import digest, iter_trials, trial_id, validate_protocol
from .native_magnetic import prepare
from .native_profiled import NativeMagneticStartingPoint
from .profiled_groups import native_seed, screen_group, calculated_residual
from .group_receipts import native_rwp_for_profile
from .worker import Artifacts, TrialDisposition, execute, require_runtime, runtime_identity


SCHEMA = 'radar-profiled-group-job/v1'
EQUIVALENT = 'profile_equivalent'


def _write_screen_result(directory, job, record, *, status, reason, runtime,
                         reference=None, kernel=None, moments=None,nvars=None,rwp=None):
    output=directory/'results';output.mkdir(exist_ok=True)
    write_json(output/'surrogate-screening.json',record)
    files={'results/surrogate-screening.json':file_sha(output/'surrogate-screening.json')}
    result=dict(schema='radar-symmetry-trial/v1',id=job['id'],
        protocol_sha256=job['protocol_sha256'],candidate_id=job['candidate_id'],
        seed_id=job['seed_id'],status=status,reason=reason,llm_calls=0,
        runtime=runtime,inputs_verified=True,effects_verified=False,
        rwp=record.get('rwp') if rwp is None else rwp,files=files,screening_only=True,
        native_reference=reference,group_kernel=kernel,moments=moments or [],nvars=nvars,
        screen_seconds=record.get('wall_seconds'))
    write_json(directory/'result.json',result)
    return result


def execute_group(directories, *, cancelled=None):
    directories=[Path(p).resolve() for p in directories]
    if not directories or len(directories)>256 or len(set(directories))!=len(directories):
        raise ValueError('A group needs one to 256 unique declared trial directories')
    if len({p.parent for p in directories})!=1:
        raise ValueError('Grouped trials must belong to one captured dispatch directory')
    jobs=[json.loads((p/'job.json').read_text()) for p in directories]
    protocol=validate_protocol(jobs[0]['protocol']);sha=digest(protocol)
    if (protocol['kind']!='magnetic' or protocol['policy'].get('moment_model')!='symmetry_allowed'
            or protocol['policy'].get('magnetic_optimizer')!='profiled_quadratic'):
        raise ValueError('Grouped worker only supports fixed-geometry full-basis profiled magnetism')
    candidate_id=jobs[0]['candidate_id']
    allowed={trial_id(protocol,c['id'],s['id']) for c,s in iter_trials(protocol)}
    for job,directory in zip(jobs,directories):
        if (job['protocol_sha256']!=sha or digest(job['protocol'])!=sha
                or job['candidate_id']!=candidate_id
                or job['id']!=trial_id(protocol,candidate_id,job['seed_id'])
                or job['id'] not in allowed
                or job['package_sha256']!=package_signature()
                or directory.name!=job['id']):
            raise ValueError('Grouped jobs do not share the same sealed model and implementation')
        if (directory/'result.json').exists():
            raise ValueError('A completed trial cannot be silently rerun in a group')
    input_root=job_inputs(directories[0],jobs[0]);verify_inputs(protocol,input_root)
    if any(job_inputs(d,j)!=input_root for d,j in zip(directories,jobs)):
        raise ValueError('Grouped jobs must share their immutable input capture')
    runtime=runtime_identity();require_runtime(runtime,protocol['magnetic_function_sha256'])
    candidate=next(c for c in protocol['candidates'] if c['id']==candidate_id)
    seeds={s['id']:s for s in protocol['seeds']}
    owner=directories[0];data=owner/'group-data';data.mkdir(exist_ok=False)
    prepared=data/'prepared';prepared.mkdir()
    cancelled=cancelled or (lambda:False)
    began=time.monotonic()
    try:
        built=prepare(confined(input_root,protocol['source']['path']),
            Artifacts(prepared,input_root=input_root),protocol,candidate,seeds[jobs[0]['seed_id']])
        try:
            native=NativeMagneticStartingPoint(prepared/built['gpx'],
                refine_scale=protocol['policy']['refine_histogram_scale'],
                refine_background=protocol['policy']['refine_background'],
                maximum_moment=protocol['policy']['maximum_moment'])
            identity=digest(dict(protocol=protocol,candidate=candidate,variables=native.moment_names,
                maps=native.cartesian_maps.tolist()))
            cache=input_root/'_profiled_cache'/f'{identity}.npz'
            model,proof=native.build(cache_path=cache,cache_identity=identity)
        except ValueError as exc:
            raise TrialDisposition('unsupported',str(exc)) from exc
        shutil.copyfile(cache,data/'kernel.npz')
        labels=[native.phases[name]['Atoms'][i][native.phases[name]['General']['AtomPtrs'][1]-1]
            for name,i,_ in native.atoms]
        start_vectors=[(j['id'],native_seed(built['seed_template'],seeds[j['seed_id']],
            native.component_maps,labels)) for j in jobs]
        kernel=dict(owner=owner.name,identity=identity,sha256=file_sha(data/'kernel.npz'))
        write_json(data/'kernel.json',dict(schema='radar-profiled-group-kernel/v1',
            protocol_sha256=sha,candidate_id=candidate_id,kernel=kernel,
            normalization=model.normalization,maximum_moment=model.maximum_moment,
            fixed_scale=model.fixed_scale,magnetic_variables=native.moment_names,
            component_maps=native.component_maps.tolist(),labels=labels,compilation=proof))
        kernel=dict(kernel,metadata_sha256=file_sha(data/'kernel.json'))
        report,fits=screen_group(model,start_vectors,
            max_iterations=protocol['policy']['optimizer_iterations'],cancelled=cancelled,
            on_progress=lambda state:write_json(data/'progress.json',dict(stage='screening',**state)))
        write_json(data/'screen.json',report)
        records={r['id']:r for r in report['records']}
        directories_by_id={j['id']:p for j,p in zip(jobs,directories)}
        jobs_by_id={j['id']:j for j in jobs}
        verification_started=time.monotonic()
        native_results={}
        for group in report['groups']:
            if cancelled():break
            uid=group['representative']
            native_results[uid]=execute(directories_by_id[uid],screened=dict(
                magnetic_variables=native.moment_names,fit=fits[uid]))
            write_json(data/'progress.json',dict(stage='native_verification',
                screened=len(records),total=len(jobs),
                verified_profiles=len(native_results),distinct_profiles=len(report['groups'])))
            # Verify parameters against the same kernel as well as the native
            # worker's independent checks. A different numerical outcome may
            # not serve as the reference for any screened alternatives.
            result=native_results[uid]
            if result.get('status') in {'valid','inadequate_fit'}:
                proof_path=directories_by_id[uid]/'results'/'optimizer-receipt.json'
                actual=json.loads(proof_path.read_text())
                if (actual.get('objective_identity',{}).get('cache_identity')!=identity or
                    not actual.get('qualification',{}).get('passed')):
                    raise ValueError('Native representative lost its checked group objective')
        for job,directory in zip(jobs,directories):
            uid=job['id']
            if (directory/'result.json').exists() or uid not in records:
                continue
            record=records[uid]
            group=next((g for g in report['groups'] if uid in g['members']),None)
            representative=group['representative'] if group else None
            reference_result=native_results.get(representative)
            if group and reference_result is None:
                continue  # cancellation: keep unverified work pending
            status='optimization_unresolved'
            reason=record.get('reason') or 'The surrogate optimizer did not qualify this start.'
            reference=None;moments=[];rwp=None
            if reference_result and reference_result.get('status') in {'valid','inadequate_fit'}:
                status=EQUIVALENT
                reason=('Surrogate-screened start with the same calculated profile as a '
                    'native-verified representative. Its magnetic parameters are retained; '
                    'it is not a separately native-verified GPX.')
                reference=dict(id=representative,receipt_sha256=digest(reference_result))
                view_files=[p for p in reference_result['files'] if p.endswith('.view.json')]
                if len(view_files)!=1:raise ValueError('Native representative has no unique profile view')
                fit=fits[uid]
                rwp=native_rwp_for_profile(reference_result['rwp'],calculated_residual(model,fit),
                    calculated_residual(model,fits[representative]))
                components=np.einsum('aij,j->ai',native.component_maps,fit.coefficients)
                cartesian=np.einsum('aij,j->ai',native.cartesian_maps,fit.coefficients)
                moments=[dict(label=label,moment=c.tolist(),magnitude=float(np.linalg.norm(v)),
                              component_esd={},magnitude_esd=None)
                         for label,c,v in zip(labels,components,cartesian)]
            elif reference_result:
                reason='Native verification of this calculated-profile group requires review: '+str(reference_result.get('reason',''))
            _write_screen_result(directory,job,record,status=status,reason=reason,
                runtime=runtime,reference=reference,kernel=kernel,moments=moments,
                nvars=reference_result.get('nvars') if reference_result else None,rwp=rwp)
        write_json(data/'summary.json',dict(schema='radar-profiled-group-result/v1',
            protocol_sha256=sha,candidate_id=candidate_id,declared=len(jobs),
            screened=len(records),distinct_profiles=len(report['groups']),
            native_attempted_profiles=len(native_results),
            native_verified_profiles=sum(r.get('effects_verified') is True and r.get('audit',{}).get('passed') is True for r in native_results.values()),
            wall_seconds=time.monotonic()-began,
            native_verification_seconds=time.monotonic()-verification_started,
            preparation_and_compilation_seconds=verification_started-began-report['wall_seconds'],
            screen_seconds=report['wall_seconds'],kernel=kernel))
    except Exception as exc:
        # An unsupported group is checked once, not once for every seed. Other
        # failures remain technical errors and do not become scientific rejections.
        status=exc.status if isinstance(exc,TrialDisposition) else 'technical_failure'
        (data/'failure.log').write_text(traceback.format_exc())
        for job,directory in zip(jobs,directories):
            if not (directory/'result.json').exists():
                _write_screen_result(directory,job,dict(id=job['id'],group_preparation_failed=True),
                    status=status,reason=str(exc),runtime=runtime)
    verify_inputs(protocol,input_root)


def main(path):
    request=json.loads(Path(path).read_text())
    if request.get('schema')!=SCHEMA or set(request)!={'schema','directories'}:
        raise ValueError('Invalid grouped-worker request')
    event=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:event.set())
    execute_group(request['directories'],cancelled=event.is_set)


if __name__=='__main__':main(sys.argv[1])
