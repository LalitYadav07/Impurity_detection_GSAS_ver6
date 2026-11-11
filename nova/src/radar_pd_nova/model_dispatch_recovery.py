"""Explicit recovery of interrupted Galaxy submissions, never automatic retry."""
from __future__ import annotations

import time

from gpx_recipe.symmetry.contracts import digest
from .model_checkpoint import job_record


def missing_chunks(transport):
    """Only unresolved current member assignments qualify for review."""
    trials={row['id']:row for row in transport.campaign.trials()}
    result={}
    for key,chunk in transport.state.get('chunks',{}).items():
        if chunk.get('job') or chunk.get('complete') or chunk.get('not_dispatched'):
            continue
        members=chunk.get('members') or []
        if not members or any(transport.state['members'].get(uid)!=key for uid in members):
            continue
        if any(uid not in trials or trials[uid]['state']!='needs_recovery'
               or (transport.campaign.trial_directory(trials[uid])/'result.json').is_file()
               for uid in members):
            continue
        result[key]=chunk
    return result


def _fingerprint(chunks):
    return digest({key:{'protocol':row['protocol'],'members':row['members'],
                        'request':row.get('request')} for key,row in chunks.items()})


def inspect_missing(transport,*,progress=None):
    """Reconcile with a complete job lookup; save an expiring review, not a POST."""
    chunks=missing_chunks(transport)
    transport.state.pop('dispatch_recovery_review',None)
    if not chunks:
        transport.save()
        return dict(missing_batches=0,missing_starts=0,recovered_batches=0)
    found=transport.runtime.find_dispatches_complete(set(chunks),progress=progress)
    recovered=0
    for key,job in found.items():
        if job is not None:
            chunks[key]['job']=job_record(job)
            recovered+=1
    missing={key:row for key,row in chunks.items() if found[key] is None}
    if missing:
        transport.state['dispatch_recovery_review']={
            'keys':sorted(missing),'fingerprint':_fingerprint(missing),
            'checked_at':time.time(),
        }
    transport.save()
    return dict(missing_batches=len(missing),
                missing_starts=sum(len(row['members']) for row in missing.values()),
                recovered_batches=recovered)


def retry_reviewed(transport,*,acknowledged=False,progress=None):
    """Retire reviewed absent attempts. Continue remains a separate action.

    Galaxy has no client-enforced exactly-once POST. This explicit action
    acknowledges that a late server acceptance is still possible. We retain
    every prior attempt and repeat the complete lookup before changing state.
    Existing jobs are adopted; scientific failures and cancellations are not
    reset by this path.
    """
    if not acknowledged:
        raise ValueError('Review and acknowledge the missing-submission retry first')
    review=transport.state.get('dispatch_recovery_review') or {}
    if not review or not 0<=time.time()-review.get('checked_at',0)<=300:
        raise ValueError('Check missing submissions again; the review has expired')
    chunks=missing_chunks(transport)
    selected={key:chunks[key] for key in review['keys'] if key in chunks}
    if len(selected)!=len(review['keys']) or _fingerprint(selected)!=review['fingerprint']:
        raise ValueError('The missing submissions changed; check them again')
    found=transport.runtime.find_dispatches_complete(set(selected),progress=progress)
    retried=recovered=0
    for key,chunk in selected.items():
        if found[key] is not None:
            chunk['job']=job_record(found[key]);recovered+=1
        else:
            chunk.update(not_dispatched=True,complete=True,
                         explicit_retry={'checked_at':time.time(),
                                         'reason':'No matching job in complete History lookup'})
            retried+=1
    transport.state.pop('dispatch_recovery_review',None)
    transport.save()
    transport.campaign.recover(transport)
    return dict(retried_batches=retried,recovered_batches=recovered)
