"""Read native campaign evidence without loading GSAS-II in the UI process."""
from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache

import numpy as np

from gpx_recipe.symmetry.campaign import Campaign,verify_receipt
from gpx_recipe.symmetry.compiler import confined
from gpx_recipe.symmetry.contracts import digest


def read_trial(directory,metadata=None,*,expected_receipt=None):
    directory=Path(directory).resolve();job=json.loads((directory/'job.json').read_text());p=job['protocol']
    candidate=next(c for c in p['candidates'] if c['id']==job['candidate_id'])
    row=dict(id=job['id'],protocol=job['protocol_sha256'],candidate=job['candidate_id'],seed=job['seed_id'],directory=str(directory),
        name=candidate['magnetic_space_group'] or candidate['irrep'] or candidate['id'],bns=candidate.get('bns_number'),
        embedding=dict(basis=candidate['transformation_matrix'],origin=candidate['origin_shift']),kind=p['kind'],source=p.get('acquisition_id') or p['source']['sha256'],source_sha256=p['source']['sha256'],sites=p.get('sites',[]),
        metadata=metadata or {},parent=p['parent_phase'],histogram=p['histogram'],k=p['k'],worker_revision=job.get('package_sha256',''),status='pending',rwp=None,nvars=None,reason='',has_view=False,has_gpx=False)
    if (directory/'result.json').is_file():
        try:
            result=verify_receipt(directory,dict(id=row['id'],protocol_sha256=row['protocol']))
            if expected_receipt and digest(result)!=digest(json.loads(expected_receipt)):
                raise ValueError('A terminal receipt changed after collection')
            row.update({k:result.get(k) for k in ['status','rwp','nvars','reason']})
            row['refined_moments']=result.get('moments',[])
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
def cached_view(path,mtime_ns,size):
    return json.loads(Path(path).read_text())


def view_for(row):
    path=Path(row['view_path']);stat=path.stat();return cached_view(str(path),stat.st_mtime_ns,stat.st_size)


def choose_group(groups,selected,*,explicit=False):
    """Follow the first available fit until a user chooses a particular trial.

    A trial can move from a pending group to its final pattern group. Preserve
    that explicit selection by trial identity rather than the old group ID.
    """
    if not groups:return ''
    if explicit:
        for group in groups:
            if any(row['id']==selected for row in group['members']):return group['id']
    return groups[0]['id']


def disposition_reason(row):
    if row.get('reason'):return row['reason']
    if row['status']=='optimization_unresolved':return 'Native convergence or rank checks did not qualify this checkpoint.'
    if row['status']=='physical_rejection':return 'The refined model violates the declared physical bounds.'
    if row['status']=='valid':return 'Native parameter/data audits and the declared fit criterion passed.'
    return 'Native parameter/data audits retained.' if row['has_view'] else 'Awaiting an independently audited native result.'


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
        view=view_for(row);mask=np.asarray(view['fitted'],dtype=bool);calc=np.asarray(view['calculated'],dtype=float)[mask]
        key=digest([row['source'],row.get('source_sha256'),view['x'],view['observed'],view['weights'],view['fitted']]);chosen=None
        scale=max(float(np.max(np.abs(np.asarray(view['observed'],dtype=float)[mask]))),1e-12)
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
