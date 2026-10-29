"""Read native campaign evidence without loading GSAS-II in the UI process."""
from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache

import numpy as np

from gpx_recipe.symmetry.campaign import Campaign,verify_receipt
from gpx_recipe.symmetry.compiler import confined
from gpx_recipe.symmetry.contracts import digest


def read_trial(directory,metadata=None):
    directory=Path(directory).resolve();job=json.loads((directory/'job.json').read_text());p=job['protocol']
    candidate=next(c for c in p['candidates'] if c['id']==job['candidate_id'])
    row=dict(id=job['id'],protocol=job['protocol_sha256'],candidate=job['candidate_id'],seed=job['seed_id'],directory=str(directory),
        name=candidate['magnetic_space_group'] or candidate['irrep'] or candidate['id'],bns=candidate.get('bns_number'),
        embedding=dict(basis=candidate['transformation_matrix'],origin=candidate['origin_shift']),kind=p['kind'],source=p.get('acquisition_id') or p['source']['sha256'],source_sha256=p['source']['sha256'],
        metadata=metadata or {},parent=p['parent_phase'],histogram=p['histogram'],worker_revision=job.get('package_sha256',''),status='pending',rwp=None,nvars=None,reason='',has_view=False,has_gpx=False)
    if (directory/'result.json').is_file():
        try:
            result=verify_receipt(directory,dict(id=row['id'],protocol_sha256=row['protocol']))
            row.update({k:result.get(k) for k in ['status','rwp','nvars','reason']})
            row['receipt_sha256']=digest(result);row['native_version']=result.get('runtime',{}).get('version','unknown')
            row['bns_audited']=result.get('audit',{}).get('checks',{}).get('complete_magnetic_operators_preserved') is True
            gpx='results/'+str(result.get('gpx') or '')
            if gpx in result.get('files',{}):row.update(has_gpx=True,gpx_relative=gpx)
            views=[k for k in result.get('files',{}) if k.endswith('.view.json')]
            row['has_view']=len(views)==1
            if row['has_view']:row['view_path']=str(confined(directory,views[0]))
        except Exception as exc:row.update(status='integrity_failure',reason=str(exc))
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
        key=digest([row['source'],view['x'],view['observed'],view['weights'],view['fitted']]);chosen=None
        scale=max(float(np.max(np.abs(np.asarray(view['observed'],dtype=float)[mask]))),1e-12)
        for previous,profile in partitions.get(key,[]):
            if len(profile)==len(calc) and np.max(np.abs(profile-calc))<=relative_tolerance*scale:chosen=previous;break
        if chosen is None:
            chosen=dict(id=row['id'],members=[],representative=row);groups.append(chosen);partitions.setdefault(key,[]).append((chosen,calc))
        chosen['members'].append(row)
    return groups


def campaign_rows(root):
    campaign=Campaign(root)
    with campaign.connect() as db:metadata={r['hash']:json.loads(r['metadata']) for r in db.execute('SELECT * FROM protocols')}
    rows=[]
    for trial in campaign.trials():
        row=read_trial(campaign.trial_directory(trial),metadata[trial['protocol']])
        row['execution_state']=trial['state']
        if trial.get('receipt'):
            try:
                actual=json.loads((campaign.trial_directory(trial)/'result.json').read_text())
                if digest(actual)!=digest(json.loads(trial['receipt'])):row.update(status='integrity_failure',reason='A terminal receipt changed after collection',has_view=False,rwp=None)
            except Exception as exc:row.update(status='integrity_failure',reason=str(exc),has_view=False,rwp=None)
        if trial['state']!='terminal' and row['status']=='pending':row.update(status=trial['state'],reason=trial['error'] or '')
        rows.append(row)
    return rows,campaign.summaries()
