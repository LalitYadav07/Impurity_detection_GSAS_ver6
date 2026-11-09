"""One presentation of advisory structural priorities in campaigns and Live."""
import json

from gpx_recipe.preparation.structural_ranking_result import validate
from gpx_recipe.preparation.structural_adaptive_sets import validate_work_set


def describe(evidence):
    if not evidence.get('ranking_available'):return evidence.get('reason','Prepared ranking unavailable')
    groups=[r['space_group'] for r in evidence['families']]
    work=evidence.get('adaptive_work_set')
    if work is None:
        labels=', '.join('SG '+str(c) for c in groups)
        return (f'Prepared scores order {len(groups)} symmetry classes · {labels}. '
            'No calibrated first work set is available. Every reviewed model remains scheduled for native fitting.')
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
        for sha in dict.fromkeys(r.get('protocol') for r in rows
                if isinstance(r.get('protocol'),str)):
            value=saved.get(sha)
            if not isinstance(value,dict):continue
            stored=db.execute('SELECT body FROM protocols WHERE hash=?',(sha,)).fetchone()
            if stored is None:continue
            try:
                if value.get('protocol_sha256')!=sha:continue
                validate(value,json.loads(stored['body']))
                describe(value)
            except (ValueError,TypeError,KeyError):continue
            result[sha]=value
    return result


def visible_rows(evidence,rows,*,previous=()):
    """Use only the selected scan's exact protocols, retaining every class."""
    first={}
    for row in rows:
        if isinstance(row.get('protocol'),str):first.setdefault(row['protocol'],row)
    result=[];selected={r['id']:r.get('mode_group') for r in previous}
    for sha,row in first.items():
        if sha not in evidence:continue
        value=dict(id=sha,scan=(row.get('metadata') or {}).get('scan') or row.get('source',''),summary=describe(evidence[sha]))
        if 'mode_estimates' in evidence[sha]:
            value.update(mode_view(mode_rows(evidence[sha]),selected.get(sha)))
        result.append(value)
    return result


def mode_rows(evidence):
    """One bounded presentation for campaign and live-snapshot consumers."""
    from gpx_recipe.preparation.structural_joint_modes import validate_estimates
    try:value=validate_estimates(evidence.get('mode_estimates'))
    except (ValueError,TypeError,KeyError):return []
    if not value['available']:return []
    result=[]
    for row in value['rows']:
        family=value['families'][row['family_index']];lower=row['lower_estimate'];upper=row['upper_estimate']
        strain=family.get('quantity')=='lattice_strain';unit='dimensionless' if strain else 'Å RMS'
        result.append(dict(space_group='SG '+str(row['space_group']),family=family['irrep']+(' · strain' if strain else ''),
            k=', '.join(family['k']),amplitude='Not learned' if lower is None else f'{lower:.5g} – {upper:.5g} {unit}',
            learned=lower is not None,
            status='Unresolved · uncalibrated' if lower is not None else 'Unresolved · no fitting support',
            reason=row['reason']))
    order={'SG '+str(r['space_group']):i for i,r in enumerate(evidence.get('families',[]))}
    result.sort(key=lambda row:order.get(row['space_group'],len(order)))
    return result


def mode_view(rows,selected=None):
    """Filter only the presentation; keep every conditional row and native job."""
    groups=list(dict.fromkeys(row['space_group'] for row in rows))
    group=selected if selected in groups else next(iter(groups),'')
    visible=[row for row in rows if row['space_group']==group]
    learned=sum(row['learned'] for row in visible)
    notes=list(dict.fromkeys(row['reason'] for row in visible if row.get('reason')))
    summary=(f'{group} · {len(visible)} distortion families · {learned} learned ranges · '
             f'{len(visible)-learned} without fitting support' if visible else '')
    return dict(mode_rows=rows,mode_groups=groups,mode_group=group,
                mode_visible_rows=visible,mode_summary=summary,mode_notes=notes)
