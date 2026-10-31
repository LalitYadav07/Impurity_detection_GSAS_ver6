"""Read native campaign evidence without loading GSAS-II in the UI process."""
from __future__ import annotations

import json
import time
import threading
from collections import OrderedDict
from pathlib import Path
from functools import lru_cache

import numpy as np

from gpx_recipe.symmetry.campaign import Campaign,verify_receipt
from gpx_recipe.symmetry.compiler import confined,assets,job_inputs
from gpx_recipe.symmetry.contracts import digest

_verified_receipts=OrderedDict()
_evidence_lock=threading.RLock()


def clear_evidence_cache():
    with _evidence_lock:
        _verified_receipts.clear();cached_view.cache_clear();comparison_profile.cache_clear()


def checked_receipt(directory,job,expected_receipt=None):
    """Rehash changed evidence immediately; bound unchanged hash reuse to 5 min.

    Polling hundreds of completed starts must not reread every common GPX and
    output on each UI refresh. Inode/ctime/mtime/size changes invalidate reuse.
    A new viewer always starts with a full cryptographic verification.
    """
    with _evidence_lock:return _checked_receipt(directory,job,expected_receipt)


def _checked_receipt(directory,job,expected_receipt):
    key=str(directory);cached=_verified_receipts.get(key)
    if cached and cached['expected']==expected_receipt and time.monotonic()-cached['at']<300:
        if Campaign._artifact_stamps(directory,cached['paths'])==cached['stamps']:
            _verified_receipts.move_to_end(key);return cached['result']
    raw=json.loads((directory/'result.json').read_text())
    if expected_receipt and digest(raw)!=digest(json.loads(expected_receipt)):raise ValueError('A terminal receipt changed after collection')
    paths=[(directory,p) for p in set(['job.json','result.json',*raw.get('files',{})])]
    paths.extend((job_inputs(directory,job),a['path']) for a in assets(job['protocol']))
    before=Campaign._artifact_stamps(directory,paths)
    result=verify_receipt(directory,dict(id=job['id'],protocol_sha256=job['protocol_sha256']))
    if expected_receipt and digest(result)!=digest(json.loads(expected_receipt)):raise ValueError('A terminal receipt changed after collection')
    after=Campaign._artifact_stamps(directory,paths)
    if before!=after:raise ValueError('Evidence changed during verification')
    _verified_receipts[key]=dict(expected=expected_receipt,at=time.monotonic(),paths=paths,stamps=after,result=result)
    _verified_receipts.move_to_end(key)
    # Common canted campaigns contain 600--840 starts per acquisition and
    # several acquisitions. A 512-entry LRU thrashes on every ordered refresh.
    # Keep this bounded; changed artifact stamps and the 5-minute TTL still
    # force the same full cryptographic verification.
    while len(_verified_receipts)>4096:_verified_receipts.popitem(last=False)
    return result


def read_trial(directory,metadata=None,*,expected_receipt=None):
    directory=Path(directory).resolve();job=json.loads((directory/'job.json').read_text());p=job['protocol']
    candidate=next(c for c in p['candidates'] if c['id']==job['candidate_id'])
    row=dict(id=job['id'],protocol=job['protocol_sha256'],candidate=job['candidate_id'],seed=job['seed_id'],directory=str(directory),
        name=candidate['magnetic_space_group'] or candidate['irrep'] or candidate['id'],bns=candidate.get('bns_number'),
        embedding=dict(basis=candidate['transformation_matrix'],origin=candidate['origin_shift']),kind=p['kind'],source=p.get('acquisition_id') or p['source']['sha256'],source_sha256=p['source']['sha256'],sites=p.get('sites',[]),
        metadata=metadata or {},parent=p['parent_phase'],histogram=p['histogram'],k=p['k'],worker_revision=job.get('package_sha256',''),status='pending',rwp=None,nvars=None,reason='',has_view=False,has_gpx=False)
    if (directory/'result.json').is_file():
        try:
            result=checked_receipt(directory,job,expected_receipt)
            row.update({k:result.get(k) for k in ['status','rwp','nvars','reason']})
            row['refined_moments']=result.get('moments',[])
            row['resources']=result.get('resources',{})
            row['timings']=result.get('timings',{})
            row['receipt_sha256']=digest(result);row['native_version']=result.get('runtime',{}).get('version','unknown')
            row['bns_audited']=result.get('audit',{}).get('checks',{}).get('complete_magnetic_operators_preserved') is True
            gpx='results/'+str(result.get('gpx') or '')
            if gpx in result.get('files',{}):row.update(has_gpx=True,gpx_relative=gpx)
            views=[k for k in result.get('files',{}) if k.endswith('.view.json')]
            row['has_view']=len(views)==1
            if row['has_view']:row['view_path']=str(confined(directory,views[0]))
            native='results/'+str(result.get('native_receipt') or '')
            if native in result.get('files',{}):
                evidence=json.loads(confined(directory,native).read_text())
                row['baseline_rwp']=(evidence.get('baseline_residuals',{}).get(p['histogram']) or {}).get('wR')
        except Exception as exc:
            row.update(status='integrity_failure',reason=str(exc),has_view=False,has_gpx=False,rwp=None,nvars=None)
            for key in ['receipt_sha256','view_path','gpx_relative']:row.pop(key,None)
    elif expected_receipt:
        row.update(status='integrity_failure',reason='The collected terminal receipt is missing')
    return row


@lru_cache(maxsize=16)
def cached_view(path,mtime_ns,size,ctime_ns=None):
    return json.loads(Path(path).read_text())


def view_for(row):
    path=Path(row['view_path']);stat=path.stat();return cached_view(str(path),stat.st_mtime_ns,stat.st_size,stat.st_ctime_ns)


@lru_cache(maxsize=1024)
def comparison_profile(path,mtime_ns,size,ctime_ns):
    """Cache compact display comparisons, separately from large full views.

    This is reached only after receipt verification. The complete observed
    domain participates in the identity; calculated curves never merge scans
    or GPX revisions merely because their residuals happen to agree.
    """
    view=cached_view(path,mtime_ns,size,ctime_ns)
    mask=np.asarray(view['fitted'],dtype=bool)
    observed=np.asarray(view['observed'],dtype=float)[mask]
    calculated=np.asarray(view['calculated'],dtype=float)[mask]
    calculated.setflags(write=False)
    return (digest([view['x'],view['observed'],view['weights'],view['fitted']]),
            calculated,max(float(np.max(np.abs(observed))),1e-12))


def choose_group(groups,selected,*,explicit=False):
    """Prefer a qualified fit until a user chooses a particular trial.

    A trial can move from a pending group to its final pattern group. Preserve
    that explicit selection by trial identity rather than the old group ID.
    """
    if not groups:return ''
    if explicit:
        for group in groups:
            if any(row['id']==selected for row in group['members']):return group['id']
    for group in groups:
        if any(row.get('has_view') and row.get('status')=='valid' for row in group['members']):return group['id']
    return groups[0]['id']


def choose_member(members,selected,*,explicit=False):
    if explicit and any(row['id']==selected for row in members):return selected
    qualified=next((row for row in members if row.get('has_view') and row.get('status')=='valid'),None)
    return (qualified or (members[0] if members else {})).get('id','')


def display_groups(groups):
    """Keep qualified choices visible without discarding other dispositions."""
    displayed=[]
    for group in groups:
        qualified=[row for row in group['members'] if row.get('has_view') and row.get('status')=='valid']
        representative=min(qualified,key=lambda row:(row['rwp'] is None,row['rwp'] or 0,row['id'])) if qualified else group['representative']
        displayed.append(dict(group,representative=representative))
    return sorted(displayed,key=lambda group:(group['representative'].get('status')!='valid',
        group['representative'].get('rwp') is None,group['representative'].get('rwp') or 0,group['id']))


def disposition_reason(row):
    if row.get('reason'):return row['reason']
    if row['status']=='optimization_unresolved':return 'Native convergence or rank checks did not qualify this checkpoint.'
    if row['status']=='physical_rejection':return 'The refined model violates the declared physical bounds.'
    if row['status']=='valid':return 'Native parameter/data audits and the declared fit criterion passed.'
    return 'Native parameter/data audits retained.' if row['has_view'] else 'Awaiting an independently audited native result.'


def display_status(status):
    return {'prepared':'awaiting dispatch','running':'assigned to a batch',
            'submitting':'submission pending'}.get(status,status.replace('_',' '))


def fit_caption(row):
    """Keep the selected acquisition/model identifiable in a full-screen fit."""
    if not row:return ''
    metadata=row.get('metadata') or {}
    parts=[str(metadata.get('scan') or 'Selected scan'),str(row.get('name') or 'Selected model')]
    if row.get('bns'):parts.append('BNS '+str(row['bns']))
    rwp=row.get('rwp')
    if isinstance(rwp,(int,float)) and not isinstance(rwp,bool) and np.isfinite(rwp):
        parts.append(f'Rwp {rwp:.3f}%')
    status=row.get('status')
    if status:parts.append('qualified fit' if status=='valid' else display_status(status))
    return ' · '.join(parts)


def embedding_evidence(rows,selected):
    """Compare exact model embeddings within one approved source protocol.

    Preserve all native dispositions. Start counts describe optimization and
    coverage, never independent measurements or a structure probability.
    """
    if not selected:return [],''
    keys=('protocol','source','source_sha256')
    if not all(selected.get(key) for key in keys):return [],''
    same_source=[row for row in rows if all(row.get(key)==selected.get(key) for key in keys)]
    partitions={}
    for row in same_source:partitions.setdefault(row['candidate'],[]).append(row)
    table=[];summary=''
    for candidate,members in partitions.items():
        qualified=[row for row in members if row.get('status')=='valid' and row.get('has_view') and
                   isinstance(row.get('rwp'),(int,float)) and np.isfinite(row['rwp']) and row['rwp']>=0]
        unresolved=[row['rwp'] for row in members if row.get('status')=='optimization_unresolved' and
                    row.get('has_view') and isinstance(row.get('rwp'),(int,float)) and np.isfinite(row['rwp']) and row['rwp']>=0]
        terminal=sum(row.get('execution_state')=='terminal' for row in members)
        best=min(qualified,key=lambda row:row['rwp']) if qualified else None
        low,high=(min(row['rwp'] for row in qualified),max(row['rwp'] for row in qualified)) if qualified else (None,None)
        label=members[0]['name']+(' · BNS '+members[0]['bns'] if members[0].get('bns') else '')+' · '+candidate[-6:]
        table.append(dict(id=candidate,model=label,coverage=f'{terminal} / {len(members)}',
            qualified=len(qualified),parameters=best.get('nvars') if best else None,
            best_rwp=round(low,5) if best else None,
            rwp_range=f'{low:.5f}–{high:.5f}%' if best else '—',
            unresolved_rwp=round(min(unresolved),5) if unresolved else None,
            selected='Current' if candidate==selected['candidate'] else '',_sort=(best is None,low if best else float('inf'),candidate)))
        if candidate==selected['candidate']:
            summary=f'{terminal} / {len(members)} starts terminal; {len(qualified)} qualified for this exact embedding.'
            if best:summary+=f' Qualified Rwp range {low:.5f}–{high:.5f}%.'
            if unresolved:summary+=f' Lowest unresolved checkpoint {min(unresolved):.5f}%.'
    table.sort(key=lambda row:row['_sort'])
    for row in table:row.pop('_sort')
    return table,summary


def campaign_can_resume(summaries,enrollments,*,following=False):
    """A queued acquisition is recoverable even before its first trial exists."""
    return (any(s['pending_count'] for s in summaries) or following or
            any(e['status'] in {'queued','preparing'} for e in enrollments))


def moment_precision_note(row):
    """Flag weak conditional estimates without turning a screen into a test.

    This does not change a sealed trial's disposition, infer a norm error,
    combine correlated components, or assign a probability of magnetic order.
    """
    if row.get('kind') != 'magnetic':return ''
    estimates=row.get('refined_moments') or []
    if not estimates:return 'Moment uncertainties are unavailable in this receipt. An audited fit alone does not establish magnetic order.'
    weak=[];missing=[]
    for item in estimates:
        ratios=[];sigmas=item.get('component_esd') or {};values=item.get('moment') or []
        for index,axis in enumerate(('mx','my','mz')):
            if index>=len(values):continue
            value,sigma=values[index],sigmas.get(axis)
            if isinstance(value,(int,float)) and isinstance(sigma,(int,float)) and np.isfinite(value) and np.isfinite(sigma) and sigma>0:
                ratios.append(abs(value)/sigma)
        # A legacy collinear norm may have an actual native uncertainty.
        if not ratios:
            value,sigma=item.get('magnitude'),item.get('magnitude_esd')
            if isinstance(value,(int,float)) and isinstance(sigma,(int,float)) and np.isfinite(value) and np.isfinite(sigma) and sigma>0:
                ratios.append(abs(value)/sigma)
        label=item.get('label','Magnetic atom')
        if not ratios:missing.append(label)
        elif max(ratios)<3:weak.append(f'{label} (largest component / s.u. {max(ratios):.2f})')
    messages=[]
    if weak:messages.append('Weak conditional moment estimates: '+', '.join(weak)+'. No independently varied component exceeds 3 conditional standard uncertainties for these atoms.')
    if missing:messages.append('Moment uncertainty unavailable for '+', '.join(missing)+'.')
    if messages:messages.append('This component screen is not a joint significance test or proof of magnetic order; uncertainties depend on the selected model and fixed nuclear parameters.')
    return ' '.join(messages)


def parameter_rows(row,view):
    """Conditional moment uncertainties come from the audited native receipt.

    A nonzero best-fit moment can remain entirely unresolved by the data. Do
    not infer an uncertainty from its magnitude or hide a missing uncertainty.
    """
    estimates={a['label']:a for a in row.get('refined_moments',[])}
    rows=[]
    for phase in view['phases']:
        for value in phase.get('moments',phase.get('modes',[])):
            magnetic='magnitude' in value
            uncertainty=estimates.get(value['label'],{}).get('magnitude_esd') if magnetic else value.get('esd')
            rows.append(dict(phase=phase['name'],parameter=value['label'],
                             value=round(value.get('magnitude',value.get('value')),6),
                             uncertainty=round(uncertainty,6) if isinstance(uncertainty,(int,float)) else '—',
                             unit='µB' if magnetic else 'Å'))
            if magnetic and 'components' in value:
                component_errors=estimates.get(value['label'],{}).get('component_esd') or {}
                for axis,component in zip(('mx','my','mz'),value['components']):
                    error=component_errors.get(axis)
                    rows.append(dict(phase=phase['name'],parameter=value['label']+' · '+axis,
                        value=round(component,6),uncertainty=round(error,6) if isinstance(error,(int,float)) else '—',unit='µB'))
    return rows


def group_models(rows,*,relative_tolerance=1e-6):
    """Only group equivalent calculated profiles on identical observed fit domains.

    Keep every member/disposition visible. This is display compression and does
    not equate space groups, count phase boundaries or choose a physical model.
    """
    groups=[];partitions={};no_curve={}
    for row in sorted(rows,key=lambda r:(r['rwp'] is None,r['rwp'] or 0,r['id'])):
        if not row['has_view']:
            key=(row['status'],row.get('reason',''))
            if key not in no_curve:
                no_curve[key]=dict(id=row['id'],members=[],representative=row);groups.append(no_curve[key])
            no_curve[key]['members'].append(row);continue
        path=Path(row['view_path']);stat=path.stat()
        domain,calc,scale=comparison_profile(str(path),stat.st_mtime_ns,stat.st_size,stat.st_ctime_ns)
        key=(row['source'],row.get('source_sha256'),domain);chosen=None
        for previous,profile in partitions.get(key,[]):
            if len(profile)==len(calc) and np.max(np.abs(profile-calc))<=relative_tolerance*scale:chosen=previous;break
        if chosen is None:
            chosen=dict(id=row['id'],members=[],representative=row);groups.append(chosen);partitions.setdefault(key,[]).append((chosen,calc))
        chosen['members'].append(row)
    return groups


def campaign_rows(root,*,experiment=None):
    campaign=Campaign(root)
    with campaign.connect() as db:metadata={r['hash']:json.loads(r['metadata']) for r in db.execute('SELECT hash,metadata FROM protocols')}
    rows=[]
    trials=campaign.trials()
    for trial in trials:
        # Restrict before opening any GPX/job/result files. Large portfolios in
        # another experiment must not delay the selected experiment's view.
        if experiment is not None and metadata[trial['protocol']].get('series')!=experiment:continue
        row=read_trial(campaign.trial_directory(trial),metadata[trial['protocol']],expected_receipt=trial.get('receipt'))
        row['execution_state']=trial['state']
        if trial['state']!='terminal' and row['status']=='pending':row.update(status=trial['state'],reason=trial['error'] or '')
        rows.append(row)
    summaries=campaign.summaries()
    for summary in summaries:
        summary['pending_count']=sum(t['protocol']==summary['protocol'] and t['state'] not in {'terminal','cancelled'} for t in trials)
    return rows,summaries


def coverage_rows(rows):
    """One compact row per exact acquisition and declared hypothesis."""
    partitions={}
    for row in rows:partitions.setdefault(row['protocol'],[]).append(row)
    result=[]
    for protocol,members in partitions.items():
        first=members[0];metadata=first.get('metadata') or {};condition=metadata.get('condition')
        received=[r for r in members if r.get('execution_state')=='terminal']
        fits=[r['rwp'] for r in members if r.get('rwp') is not None and r.get('has_view')]
        qualified=[r['rwp'] for r in members if r.get('status')=='valid' and r.get('rwp') is not None]
        result.append(dict(id=protocol,source=first['source'],scan=metadata.get('scan') or first['source'],
            condition=(f"{condition:g} {metadata.get('condition_unit','')}".strip() if isinstance(condition,(int,float)) else '—'),
            bank=str(metadata.get('bank') or '—'),
            sites=', '.join(s['label']+'='+s['form_factor'] for s in first.get('sites',[])) or first.get('kind',''),
            coverage=f'{len(received)} / {len(members)}',qualified=len(qualified),
            best_qualified=f'{min(qualified):.3f}%' if qualified else '—',
            best_audited=f'{min(fits):.3f}%' if fits else '—',
            status='In progress' if len(received)!=len(members) else 'Review qualified alternatives' if qualified else 'No qualified fit'))
    return result


def scope_rows(rows, experiment, scope='experiment'):
    if scope=='campaign':return rows
    if scope!='experiment':raise ValueError('Unknown model-result scope')
    return [row for row in rows if row.get('metadata',{}).get('series')==experiment]
