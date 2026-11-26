"""Reuse a model GPX only after binding its published bytes to the exact job."""
import base64
import copy
import json
from pathlib import Path
import tempfile

from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.symmetry.contracts import digest
from .galaxy_service import _parameter_value
from .model_checkpoint import archived_chunk
from .model_runtime import identifier


def published_project(transport, trial_id, protocol_sha256, captured):
    """Return a checked HDA, or None for older jobs without this publication.

    Saved workspace labels and collection names cannot authorize a handoff.
    The authenticated producing job must match the exact captured request,
    and the selected collection member must match the already audited GPX.
    This function never uploads, submits a job, or treats review as acceptance.
    """
    captured=Path(captured);expected=file_sha(captured)
    with transport._state_lock:
        key=transport.state.get('members',{}).get(trial_id)
        chunk=copy.deepcopy(transport.state.get('chunks',{}).get(key))
    if not chunk or not chunk.get('complete') or not chunk.get('job'):
        return None
    request=chunk.get('request')
    if (chunk.get('key')!=key or chunk.get('protocol')!=protocol_sha256
            or (request is not None and (not isinstance(request,dict)
                or request.get('protocol_sha256')!=protocol_sha256
                or trial_id not in request.get('trials',[])))):
        raise ValueError('Published model lookup differs from the selected trial')
    if request is None and not archived_chunk(chunk):
        raise ValueError('Published model lookup lacks its completed archive witness')
    job_id=identifier(chunk['job']['galaxy_job_id'])
    service=transport.runtime.service
    details=service._job_details(job_id)
    transport.runtime._verify_job_details(details,job_id,key)
    if details.get('state')!='ok':return None
    params=details.get('params') or details.get('parameters') or {}
    encoded=_parameter_value(params,'model_request_b64')
    if not encoded:return None  # Legacy file-input jobs retain the upload path.
    if not isinstance(encoded,str) or len(encoded)>8*1024*1024:
        raise ValueError('Published model request exceeds its verification bound')
    try:actual=json.loads(base64.b64decode(encoded,validate=True))
    except (ValueError,TypeError):raise ValueError('Published model request could not be verified') from None
    if not isinstance(actual,dict) or _parameter_value(params,'action')!='chunk':
        raise ValueError('Published model job differs from its captured request')
    if request is not None:
        if actual!=request:
            raise ValueError('Published model job differs from its captured request')
    else:
        # Completed transport checkpoints retire the bulky request. Recover it
        # from the authenticated producing job, keeping the local protocol,
        # dispatch, member list and finished result as independent witnesses.
        # Older compact checkpoints lack the new full-request fingerprint;
        # their exact GPX bytes are still checked below before any handoff.
        members=chunk.get('members')
        finished={row.get('id') for row in chunk['result'].get('trials',[])
                  if isinstance(row,dict) and row.get('state')=='finished'}
        trials=actual.get('trials')
        if (not isinstance(members,list) or not isinstance(trials,list)
                or not 1<=len(trials)<=256
                or any(not isinstance(uid,str) for uid in members+trials)
                or len(set(trials))!=len(trials)
                or set(trials)!=set(members) or trial_id not in finished
                or actual.get('protocol_sha256')!=protocol_sha256
                or actual.get('dispatch_key')!=key
                or not isinstance(actual.get('protocol'),dict)
                or digest(actual['protocol'])!=protocol_sha256
                or (chunk.get('request_sha256') is not None
                    and digest(actual)!=chunk['request_sha256'])):
            raise ValueError('Published model job differs from its completed checkpoint')
    outputs=service._job_output_ids(details)
    labels={'reviewable_projects':'Structural point-fit GPXs (review required)',
            'verified_projects':'Qualified model GPXs'}
    missing={name:label for name,label in labels.items() if name not in outputs}
    if missing:
        outputs.update(service._history_job_collection_ids(job_id,expected=missing))
    matches=[]
    for role in labels:
        if role not in outputs:continue
        for member in service.collection_elements(identifier(outputs[role])):
            if member.get('name')==trial_id:
                matches.append((role,identifier(member['id'])))
    if not matches:return None  # Older worker or no independently checked GPX.
    if len(matches)!=1:raise ValueError('Several published GPXs claim this model trial')
    role,dataset=matches[0];metadata=service._dataset_metadata(dataset)
    if metadata.get('deleted') or metadata.get('purged') or metadata.get('state')!='ok':
        return None
    size=captured.stat().st_size
    if metadata.get('file_size')!=size:
        raise ValueError('Published model GPX differs from the audited file size')
    with tempfile.TemporaryDirectory(prefix='published-gpx-',dir=transport.campaign.root) as temporary:
        target=Path(temporary)/'model.gpx'
        service._download_dataset(dataset,target,maximum_bytes=size)
        if file_sha(target)!=expected:
            raise ValueError('Published model GPX differs from the audited bytes')
    return dict(dataset_id=dataset,job_id=job_id,role=role,sha256=expected)
