"""Viewer-side recovery; scientific jobs and their sealed implementation stay fixed."""
import json
import time
import hashlib
from pathlib import Path

from gpx_recipe.symmetry.compiler import verify_inputs
from gpx_recipe.symmetry.contracts import digest, iter_trials, validate_protocol


def restore_ledger(campaign, snapshot, input_roots, *, progress=None):
    """Validate the entire finite plan, then restore states in one transaction.

    The shared compiler still reconstructs every original sealed job. A viewer
    restart must not require one synchronous SQLite commit per planned start.
    The marker permits recovery after an interruption during reconstruction;
    nothing in this function dispatches a scientific job.
    """
    if snapshot.get('schema') != 'radar-model-ledger/v1':
        raise ValueError('Restore a valid model ledger')
    existing=campaign.trials()
    if existing and not campaign.setting('restore_in_progress',False):
        raise ValueError('Restore into an empty or interrupted recovery ledger')
    expected={};protocols={}
    for row in snapshot['protocols']:
        protocol=validate_protocol(json.loads(row['body']));sha=row['hash']
        if digest(protocol)!=sha or sha in protocols:
            raise ValueError('Saved protocol identity changed')
        verify_inputs(protocol,input_roots[sha])
        metadata=json.loads(row['metadata'])
        if set(metadata)-{'scan','sample','bank','condition','condition_unit','series'}:
            raise ValueError('Unknown source provenance fields')
        protocols[sha]=(protocol,metadata)
        for candidate,seed in iter_trials(protocol):
            uid=digest(dict(protocol=sha,candidate=candidate['id'],seed=seed['id']))[:24]
            expected[uid]=(sha,candidate['id'],seed['id'])
    seen=set();updates=[]
    for row in snapshot['trials']:
        if row['id'] in seen or expected.get(row['id'])!=tuple(row[k] for k in ['protocol','candidate','seed']):
            raise ValueError('Saved model trial differs from its approved plan')
        if row['state'] not in {'prepared','submitting','running','cancelling','terminal','cancelled','needs_recovery'}:
            raise ValueError('Unknown saved trial state')
        for field in ['handle','receipt']:
            if row.get(field) and not isinstance(json.loads(row[field]),dict):
                raise ValueError('Malformed saved trial receipt or handle')
        state=row['state']
        if state=='terminal':state='running' if row.get('handle') else 'needs_recovery'
        updates.append((state,row.get('handle') or None,row.get('receipt') or None,row.get('error'),time.time(),row['id']))
        seen.add(row['id'])
    if seen!=set(expected):raise ValueError('Saved model checkpoint omits declared starts')
    if any(expected.get(row['id'])!=tuple(row[k] for k in ['protocol','candidate','seed']) for row in existing):
        raise ValueError('Interrupted recovery contains a different finite plan')
    campaign.set_setting('restore_in_progress',True)
    for index, (sha,(protocol,metadata)) in enumerate(protocols.items(), 1):
        if progress:
            progress(f'Rebuilding approved input {index} of {len(protocols)}; {len(expected):,} declared starts')
        campaign.enroll(protocol,input_roots[sha],approved_hash=sha,metadata=metadata,_restoring=True)
    rebuilt={row['id']:tuple(row[k] for k in ['protocol','candidate','seed']) for row in campaign.trials()}
    if rebuilt!=expected:raise ValueError('Reconstructed ledger omits or changes declared starts')
    with campaign.connect() as db:
        db.executemany('UPDATE trials SET state=?,handle=?,receipt=?,error=?,updated=? WHERE id=?',updates)
        db.executemany('INSERT INTO events(trial,state,detail,at) VALUES(?,?,?,?)',
                       [(uid,state,error,stamp) for state,_,_,error,stamp,uid in updates])
        db.executemany('INSERT OR REPLACE INTO settings VALUES(?,?)',
                       [(key,json.dumps(value,allow_nan=False)) for key,value in snapshot.get('settings',{}).items() if key!='restore_in_progress'])


def verify_continuation(campaign, *, progress=None):
    """Verify each sealed start; hash shared inputs once per actual location.

    Native workers still independently rehash their inputs before/after every
    refinement. The viewer checks all job identities and entry points, plus
    content hashes and change witnesses for every common input and package.
    Legacy self-contained bundles retain their original full verifier.
    """
    from gpx_recipe.symmetry.compiler import SHARED_ENTRYPOINT,assets,confined,file_sha,package_signature
    pending=[row for row in campaign.trials() if row['state'] not in {'terminal','cancelled'}]
    if not pending:raise ValueError('This campaign has no pending trials to continue')
    if all(row['state']=='needs_recovery' and not (campaign.trial_directory(row)/'result.json').is_file() for row in pending):
        raise ValueError('These trials need explicit transport recovery; no duplicate job was submitted')
    implementation=package_signature();entrypoint=hashlib.sha256(SHARED_ENTRYPOINT.encode()).hexdigest()
    protocols={};witnesses={}
    if progress:
        progress(0, len(pending))
    for index, row in enumerate(pending, 1):
        directory=Path(campaign.trial_directory(row)).resolve()
        job=json.loads(confined(directory,'job.json').read_text())
        if job.get('schema')!='radar-symmetry-job/v2':
            return campaign.verify_sealed_continuation()
        sha=row['protocol'];raw=job.get('protocol')
        if sha not in protocols:
            protocol=validate_protocol(raw)
            if digest(protocol)!=sha:raise ValueError('A sealed job identity changed; no replacement was submitted')
            protocols[sha]=protocol
        protocol=protocols[sha]
        if (digest(raw)!=sha or job.get('protocol_sha256')!=sha or job.get('id')!=row['id']
                or job.get('candidate_id')!=row['candidate'] or job.get('seed_id')!=row['seed']
                or digest(dict(protocol=sha,candidate=row['candidate'],seed=row['seed']))[:24]!=row['id']):
            raise ValueError('A sealed job identity changed; no replacement was submitted')
        if job.get('package_sha256')!=protocol['implementation_sha256'] or implementation!=protocol['implementation_sha256']:
            raise ValueError('The shared campaign requires its original pinned worker implementation')
        if directory.name!=row['id'] or directory.parent.name not in {'trials','preparing'}:
            raise ValueError('Invalid shared trial layout')
        if file_sha(confined(directory,'worker.py'))!=entrypoint:
            raise ValueError('A shared trial entry point changed')
        root=confined(directory.parent.parent,'inputs/'+sha);key=(sha,str(root))
        if key not in witnesses:
            paths=[asset['path'] for asset in assets(protocol)]
            before=campaign._artifact_stamps(root,paths)
            verify_inputs(protocol,root)
            if before!=campaign._artifact_stamps(root,paths):raise ValueError('A shared input changed during verification')
            witnesses[key]=(root,paths,before)
        if progress and index % 256 == 0 and index < len(pending):
            progress(index, len(pending))
    for root,paths,before in witnesses.values():
        if before!=campaign._artifact_stamps(root,paths):raise ValueError('A shared input changed during verification')
    if package_signature()!=implementation:raise ValueError('The pinned worker changed during verification')
    if progress:
        progress(len(pending), len(pending))
    return [row['id'] for row in pending]

