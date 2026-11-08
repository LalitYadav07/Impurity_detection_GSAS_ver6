"""One presentation of advisory structural priorities in campaigns and Live."""
import json

from gpx_recipe.preparation.structural_ranking_result import validate
from gpx_recipe.preparation.structural_adaptive_sets import validate_work_set


def describe(evidence):
    if not evidence.get('ranking_available'):return evidence.get('reason','Prepared ranking unavailable')
    groups=[r['space_group'] for r in evidence['families']]
    work=evidence.get('adaptive_work_set')
    if work is None:
        return f'Prepared scores order {len(groups)} symmetry classes. No calibrated first work set is available; all native models remain scheduled.'
    work=validate_work_set(work,groups);retained=work['retained_space_groups'];deferred=work['deferred_space_groups']
    labels=', '.join('SG '+str(c) for c in retained)
    return (f'Adaptive first work set: {len(retained)} / {len(groups)} symmetry classes · {labels}. '
        f'{len(deferred)} classes deferred in the priority order; every declared native model remains scheduled.')


def checked_receipts(campaign,rows):
    """Validate once per visible protocol, never once per model/start or click."""
    saved=campaign.setting('structural_rankings',{})
    if not isinstance(saved,dict):return {}
    result={}
    with campaign.connect() as db:
        for sha in dict.fromkeys(r['protocol'] for r in rows):
            value=saved.get(sha)
            if value is None:continue
            stored=db.execute('SELECT body FROM protocols WHERE hash=?',(sha,)).fetchone()
            if stored is None:continue
            try:
                if value.get('protocol_sha256')!=sha:continue
                validate(value,json.loads(stored['body']))
                describe(value)
            except (ValueError,TypeError,KeyError):continue
            result[sha]=value
    return result


def visible_rows(evidence,rows):
    """Use only the selected scan's exact protocols, retaining every class."""
    first={}
    for row in rows:first.setdefault(row['protocol'],row)
    result=[]
    for sha,row in first.items():
        if sha not in evidence:continue
        result.append(dict(id=sha,scan=(row.get('metadata') or {}).get('scan') or row.get('source',''),
            summary=describe(evidence[sha])))
    return result
