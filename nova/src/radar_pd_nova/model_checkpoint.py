"""Compact transport caches while retaining exact scientific recovery proofs."""
import copy
import json
import re

from gpx_recipe.symmetry.contracts import digest


def job_record(job):
    # The authenticated job API revalidates History/tool/dispatch on recovery.
    # Its inline base64 request duplicates the sealed chunk request and hash.
    return job.model_dump(mode='json',exclude={'inputs'})


def _identity(value):
    return isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_-]{1,128}',value) is not None


def archived_chunk(chunk):
    job=chunk.get('job') or {};result=chunk.get('result') or {}
    return (chunk.get('complete') is True and _identity(job.get('galaxy_job_id'))
            and _identity((job.get('outputs') or {}).get('result_bundle'))
            and result.get('schema')=='radar-model-chunk-result/v1'
            and result.get('dispatch_key')==chunk.get('key')
            and result.get('protocol_sha256')==chunk.get('protocol'))


def compact_transport(state):
    """Only a completed, acknowledged result can retire its request cache."""
    for chunk in state.get('chunks',{}).values():
        job=chunk.get('job')
        if isinstance(job,dict) and _identity(job.get('galaxy_job_id')):
            job.pop('inputs',None)
        if archived_chunk(chunk):chunk.pop('request',None)


def portable_checkpoint(document):
    """Keep finite trials, ambiguous intents, HDA references and receipt hashes.

    Full receipt JSON is an audit cache, not the only copy of a result. Drop it
    only for a terminal member of its acknowledged immutable Galaxy archive;
    recovery re-downloads and verifies that exact receipt and all its artifacts.
    """
    result=copy.deepcopy(document);transport=result['transport']
    compact_transport(transport)
    hashes=transport.setdefault('receipt_hashes',{})
    for row in result['ledger']['trials']:
        if row.get('state')!='terminal' or not row.get('receipt') or not row.get('handle'):
            continue
        handle=json.loads(row['handle']);key=transport.get('members',{}).get(row['id'])
        chunk=transport.get('chunks',{}).get(key,{})
        if handle.get('key')!=row['id'] or handle.get('dispatch_key')!=key or not archived_chunk(chunk):
            continue
        finished={trial['id'] for trial in chunk['result'].get('trials',[]) if trial.get('state')=='finished'}
        if row['id'] not in finished:continue
        sha=digest(json.loads(row['receipt']))
        if row['id'] in hashes and hashes[row['id']]!=sha:
            raise ValueError('A saved terminal receipt changed before checkpointing')
        hashes[row['id']]=sha
        row['receipt']=None
    return result


def verify_archived_receipts(state, terminal, received):
    """Validate compact checkpoint witnesses before publishing any local result."""
    hashes=state.get('receipt_hashes',{})
    for uid,row in terminal.items():
        if row.get('state')=='finished' and uid in hashes:
            path=received/'trials'/uid/'result.json'
            if digest(json.loads(path.read_text()))!=hashes[uid]:
                raise ValueError('The recovered terminal receipt differs from its saved checkpoint')
