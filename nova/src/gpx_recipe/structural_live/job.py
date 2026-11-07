"""Closed Galaxy handoff for one reviewed structural acquisition revision.

The existing acquisition owner captures bytes and chooses when to dispatch.
This operation uses the same sealed model protocol and nonlinear service as
Model exploration. Imported theory banks keep the existing producer-authority
policy; uploaded native contexts are never treated as trusted executables or
qualified construction state. A point fit is not a magnetic posterior.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import time

from ..execution import write_json
from ..live.snapshot import stable_read
from ..symmetry.compiler import confined, file_sha, package_signature, verify_inputs,job_inputs
from ..symmetry.contracts import digest, trial_id, validate_protocol
from .acquisition import _mapping, _scope, prepare_acquisition, StructuralAcquisition
from .service import SearchSettings, workflow_timings

FIELDS={'source','directory','anchor_snapshot','snapshot','receipt','mapping',
        'source_scope','revision','source_revision','reference_revision',
        'refine_transfer','destination'}


def validate_payload(payload):
    if not isinstance(payload,dict) or FIELDS-set(payload) or set(payload)-FIELDS-{'prepared_scattering'}:
        raise ValueError('Use the complete closed structural snapshot request')
    for field in ('source','directory','anchor_snapshot','snapshot','destination',
                  'source_revision','reference_revision'):
        value=payload[field]
        if not isinstance(value,str) or not 1<=len(value)<=1000 or '\x00' in value:
            raise ValueError('Bounded structural input paths and revisions are required')
    mapping=_mapping(payload['mapping']);_scope(payload['source_scope'],mapping['identity'])
    anchor=payload['mapping'].get('snapshot_sha256')
    if not isinstance(anchor,str) or not re.fullmatch('[a-f0-9]{64}',anchor):
        raise ValueError('Review an exact structural anchor snapshot')
    if (payload['source_revision'].split(':')[-1]!=anchor or
            payload['reference_revision'].split(':')[-1]!=mapping['source_gpx_sha256']):
        raise ValueError('Structural handoff revisions differ from the reviewed mapping')
    if type(payload['revision']) is not int or payload['revision']<0 or type(payload['refine_transfer']) is not bool:
        raise ValueError('An explicit snapshot revision and transfer freedom are required')
    receipt=payload['receipt']
    if (not isinstance(receipt,dict) or receipt.get('identity')!=mapping['identity']
            or not isinstance(receipt.get('content_sha256'),str)
            or not re.fullmatch('[a-f0-9]{64}',receipt['content_sha256'])
            or not isinstance(receipt.get('metadata',{}),dict)):
        raise ValueError('The snapshot receipt must bind this exact acquisition and bytes')
    if len(json.dumps(payload,allow_nan=False))>4*1024*1024:
        raise ValueError('Structural snapshot metadata exceeds its transport budget')
    return payload


def reviewed_job(directory):
    """Read one self-contained approved child, with all original model assets."""
    root=Path(directory).resolve();path=confined(root,'job.json')
    if path.stat().st_size>8*1024*1024:raise ValueError('Structural job metadata is oversized')
    job=json.loads(path.read_text())
    if (not isinstance(job,dict) or set(job)!={'schema','protocol','protocol_sha256',
            'candidate_id','seed_id','id','package_sha256'} or job['schema'] not in {'radar-symmetry-job/v1','radar-symmetry-job/v2'}):
        raise ValueError('Capture a reviewed structural model job')
    protocol=validate_protocol(job['protocol'])
    if (protocol['kind']!='structural' or protocol['policy'].get('structural_optimizer')!='prepared_nonlinear'
            or job['protocol_sha256']!=digest(protocol)
            or job['id']!=trial_id(protocol,job['candidate_id'],job['seed_id'])
            or job['package_sha256']!=protocol['implementation_sha256']
            or job['package_sha256']!=package_signature()):
        raise ValueError('Structural model scope, implementation or approval changed')
    candidate=next((c for c in protocol['candidates'] if c['id']==job['candidate_id']),None)
    seed=next((s for s in protocol['seeds'] if s['id']==job['seed_id']),None)
    if candidate is None or seed is None or candidate['id'] not in seed.get('candidate_ids',[candidate['id']]):
        raise ValueError('Structural child is outside its reviewed finite portfolio')
    verify_inputs(protocol,job_inputs(root,job))
    return job,protocol,candidate


def execute(payload):
    """Own one scientific job; do not poll, discover models or infer settings."""
    validate_payload(payload)
    directory=Path(payload['directory']).resolve()
    job,protocol,candidate=reviewed_job(directory)
    inputs=job_inputs(directory,job)
    if 'prepared_scattering' in payload:
        from ..preparation.structural_hints import read as read_preparation
        if read_preparation(inputs,protocol)!=payload['prepared_scattering']:
            raise ValueError('Structural preparation metadata differs from the captured model inputs')
    source=Path(payload['source']).resolve()
    if file_sha(source)!=protocol['source']['sha256']:
        raise ValueError('Galaxy source differs from the reviewed structural parent')
    mapping=payload['mapping'];snapshot=stable_read(payload['snapshot'])
    if hashlib.sha256(snapshot).hexdigest()!=payload['receipt']['content_sha256']:
        raise ValueError('Captured structural snapshot changed before dispatch')
    anchor=stable_read(payload['anchor_snapshot'])
    if hashlib.sha256(anchor).hexdigest()!=mapping['snapshot_sha256']:
        raise ValueError('The reviewed structural anchor changed before dispatch')
    # Bind the actual native parent/runtime, independently of uploaded labels.
    from ..symmetry.native_audit import inspect_source,write_view
    from ..symmetry.worker import require_runtime
    actual=inspect_source(source)
    require_runtime(actual['runtime'],protocol['magnetic_function_sha256'])
    parent=next((p for p in actual['phases'] if p['name']==protocol['parent_phase']),None)
    if parent is None or parent['setting_sha256']!=protocol['parent_setting_sha256']:
        raise ValueError('The captured nuclear parent setting differs from its reviewed model')
    scope=protocol['policy']
    settings=SearchSettings(mode_radius=scope['structural_mode_radius'],cell_radius=scope['structural_cell_radius'],
        seconds=scope['structural_seconds'],starts=scope['structural_starts'],
        recovery_seconds=scope['structural_recovery_seconds'])
    from ..preparation.structural_hints import candidates as prepared_candidates
    banks,attachment=prepared_candidates(inputs,protocol,candidate['id'])
    from ..preparation.tof_profile_hints import candidates as profile_candidates
    profiles,profile_attachment=profile_candidates(inputs,protocol)
    attachment['profile_tables']=profile_attachment
    reference=confined(inputs,protocol['structural_reference']['path']) if 'structural_reference' in protocol else None
    root=Path(payload['destination']);root.mkdir(parents=True,exist_ok=False)
    request=dict(source_gpx=str(source),source_sha256=protocol['source']['sha256'],
        observation_source=payload['anchor_snapshot'],observation_source_sha256=mapping['snapshot_sha256'],
        source_revision=payload['source_revision'],reference_revision=payload['reference_revision'],
        parent_phase=protocol['parent_phase'],mapping=mapping,refine_transfer=payload['refine_transfer'])
    preparation_began=time.monotonic()
    contract=prepare_acquisition(request,confined(inputs,candidate['child']['path']),
        candidate['transformation_matrix'],candidate['origin_shift'],root/'acquisition',
        source_scope=payload['source_scope'],settings=settings,prepared_banks=banks,prepared_profiles=profiles,reference_gpx=reference)
    metadata=json.loads((root/'acquisition/model/prepared.json').read_text())
    if set(metadata['inventory']['native_mode_names'])!=set(candidate['modes']):
        raise ValueError('Structural acquisition requires every reviewed physical mode')
    if metadata['histogram']!=protocol['histogram']:
        raise ValueError('Structural acquisition changed its reviewed histogram')
    owner=StructuralAcquisition(root/'acquisition',root/'session')
    fitting_began=time.monotonic()
    fitted=owner.update(snapshot,payload['receipt'],revision=payload['revision'],
        current_scope=payload['source_scope'],current_mapping=mapping)
    stages=workflow_timings(fitting_began-preparation_began,time.monotonic()-fitting_began,scope='acquisition')
    from .native import modules
    G,*_=modules();project=G.G2Project(gpxfile=fitted['continuation']['path'])
    view=root/'view.json';write_view(project,metadata['histogram'],view)
    result=dict(schema='radar-structural-snapshot-result/v1',state='verified_point_fit',
        kind='structural_point_fit',optimization_status='optimization_unresolved',
        protocol_sha256=job['protocol_sha256'],candidate_id=candidate['id'],trial_id=job['id'],
        implementation_sha256=package_signature(),source_sha256=protocol['source']['sha256'],
        reference_sha256=protocol.get('structural_reference',{}).get('sha256'),
        child_sha256=candidate['child']['sha256'],acquisition=contract['acquisition'],
        anchor_sha256=mapping['snapshot_sha256'],snapshot_sha256=payload['receipt']['content_sha256'],
        revision=payload['revision'],settings=asdict(settings),preparation_attachment=attachment,
        result=fitted,view=str(view),output=str(root),workflow_timings=stages,
        forecast_enabled=False,model_exclusion_authorized=False,statistical_policy_qualified=False)
    # Only independently checked publications and their receipts leave this
    # job. Native construction/verification scratch is not a portable library.
    paths=[view,root/'acquisition/acquisition.json',root/'acquisition/transfer.json',
           root/'acquisition/model/prepared.json',root/'acquisition/model/starting-curve-contract.json']
    paths += [Path(fitted[role]['path']) for role in ('frozen','continuation')]
    result['artifacts']=sorted([str(p.relative_to(root)) for p in paths]+['result.json'])
    write_json(root/'result.json',result)
    if (file_sha(source)!=protocol['source']['sha256'] or stable_read(payload['snapshot'])!=snapshot
            or stable_read(payload['anchor_snapshot'])!=anchor):
        raise ValueError('Structural handoff inputs changed during native calculation')
    verify_inputs(protocol,inputs)
    return result


if __name__=='__main__':
    import sys
    request=json.loads(Path(sys.argv[1]).read_text())
    write_json(sys.argv[2],execute(request))
