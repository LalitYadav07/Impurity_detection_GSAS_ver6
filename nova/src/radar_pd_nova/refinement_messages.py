"""Scientific names for native failures, without interpreting them as fits."""
import copy
import re

from gpx_recipe.convergence import SCHEMA as CONVERGENCE_SCHEMA, assess_convergence, finite

REVIEW_SCHEMA = 'radar-refinement-review/v1'


def review_notices(values):
    """Keep exact notices once; a longer verbatim notice retains its prefix."""
    unique=list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    return [value for value in unique if not any(value!=other and value in other for other in unique)]


def galaxy_refinement_activity(action):
    """Report observed transport state, separately from scientific completion."""
    return {
        'uploading': 'Preparing inputs on NDIP',
        'queued': 'Queued on NDIP',
        'running': 'Running on NDIP',
        'ok': 'Collecting and verifying outputs',
        'error': 'Galaxy job failed',
        'cancelled': 'Galaxy job cancelled',
    }.get((action or {}).get('status'), 'Submitted to Galaxy')


def retain_review_evidence(result, view, criteria=None):
    """Retain small diagnostics from the verified final view, not its curves."""
    if (not isinstance(view,dict) or result.get('status')!='completed' or not result.get('output_sha256')
            or view.get('source_sha256')!=result['output_sha256']):return False
    evidence=view.get('convergence')
    if not isinstance(evidence,dict) or evidence.get('schema')!=CONVERGENCE_SCHEMA:return False
    retained=dict(schema=REVIEW_SCHEMA,source_sha256=result.get('source_sha256'),
        output_sha256=result['output_sha256'],convergence=copy.deepcopy(evidence),criteria=copy.deepcopy(criteria))
    if result.get('review_evidence')==retained:return False
    result['review_evidence']=retained
    return True


def refinement_review_status(result):
    """Flag recorded diagnostic failures without inventing a scientific cutoff."""
    if result.get('rejected_stages'):
        return dict(status='needs_review',needs_review=True,
            message='Recovered recipe: rejected or skipped stages '+', '.join(result['rejected_stages'])+
            '. Previous parameter values were retained. Final output requires scientific review.')
    if result.get('status') == 'needs_review':
        reason = str(result.get('error') or 'The recipe stopped before accepting a final project.').strip()
        return dict(status='needs_review', needs_review=True,
            message=reason + ' No final GPX was accepted; open the scan to review any retained stages.')
    retained=result.get('review_evidence') or {}
    if not isinstance(retained,dict):retained={}
    bound=(bool(result.get('source_sha256')) and bool(result.get('output_sha256')) and retained.get('schema')==REVIEW_SCHEMA
        and retained.get('source_sha256')==result.get('source_sha256')
        and retained.get('output_sha256')==result.get('output_sha256'))
    evidence=retained.get('convergence') if bound else None
    if not isinstance(evidence,dict) or evidence.get('schema')!=CONVERGENCE_SCHEMA:
        return dict(status='unavailable',needs_review=False,
            message='Convergence diagnostics unavailable. Final GPXs require scientific review.')
    reasons=[]
    if evidence.get('native_converged') is False:reasons.append('Native solver did not report convergence.')
    if evidence.get('aborted') is True:reasons.append('Native solver reported an abort.')
    if evidence.get('last_cycle_complete') is False or evidence.get('missing_last_cycle_parameters'):
        reasons.append('Final-cycle evidence is incomplete.')
    value=evidence.get('last_cycle_max_shift_over_sigma')
    if (value is not None and (not finite(value) or value<0)) or (evidence.get('last_cycle_complete') is True and value is None):
        reasons.append('Final-cycle shift evidence is invalid or unavailable.')
    for key in ('relative_objective_change','relative_objective_tolerance'):
        if evidence.get(key) is not None and not finite(evidence[key]):
            reasons.append('Native objective diagnostic is nonfinite.');break
    criteria=retained.get('criteria')
    if isinstance(criteria,dict) and criteria.get('max_shift_over_sigma') is not None:
        try:
            assessment=assess_convergence(evidence,criteria['max_shift_over_sigma'])
        except ValueError:
            reasons.append('The recorded convergence criterion is invalid.')
        else:
            if assessment['status']!='passed':reasons.append(assessment['message'])
    missing=(type(evidence.get('native_converged')) is not bool
             or evidence.get('last_cycle_complete') is not True or not finite(value))
    status='needs_review' if reasons else 'unavailable' if missing else 'recorded'
    message=convergence_notice({'convergence':evidence})
    if reasons:message=' '.join(dict.fromkeys(reasons))+' '+message
    return dict(status=status,needs_review=bool(reasons),message=message)


def convergence_notice(view):
    """Expose recorded diagnostics without inventing an unreviewed cutoff."""
    evidence = view.get('convergence')
    if not isinstance(evidence, dict) or evidence.get('schema') != CONVERGENCE_SCHEMA:
        return ('Final-cycle convergence evidence is unavailable for this saved result. '
                'Job completion does not establish convergence.')
    if evidence.get('aborted') is True:
        parts = ['The native solver reported an abort.']
    elif evidence.get('native_converged') is True:
        parts = ['The native solver reported convergence.']
    elif evidence.get('native_converged') is False:
        parts = ['The native solver did not report convergence.']
    else:
        parts = ['The native convergence status is unavailable.']
    value = evidence.get('last_cycle_max_shift_over_sigma')
    usable = finite(value) and value >= 0
    if evidence.get('last_cycle_complete') is True and usable:
        parts.append(f'Largest final-cycle shift / conditional s.u.: {value:.5g}.')
    else:
        parts.append('Final-cycle shift evidence is incomplete.')
        if usable:
            parts.append(f'Largest available shift / conditional s.u.: {value:.5g}; this may omit other parameters.')
    change = evidence.get('relative_objective_change')
    if finite(change) and change < -1e-12:
        tolerance = evidence.get('relative_objective_tolerance')
        detail = f'The last-cycle objective increased by {-change:.5g} relative'
        if finite(tolerance) and tolerance > 0:
            relation = 'within' if -change <= tolerance else 'beyond'
            detail += f', {relation} the recorded native tolerance {tolerance:.5g}'
        parts.append(detail+'.')
    parts.append('Compare these diagnostics with the reviewed convergence criteria. '
                 'Conditional uncertainties do not establish model adequacy or calibration accuracy.')
    return ' '.join(parts)


def sample_width_failure_message(result, source):
    message=str(result.get('error') or '')
    if result.get('status')!='needs_review':
        return None
    clipped='Native saved parameters differ from solver values' in message
    if clipped:
        pairs=re.findall(r"\('([^']+)',\s*([-+\d.eE]+),\s*([-+\d.eE]+|None)\)",message)
    elif 'violated physical/requested bounds' in message:
        pairs=re.findall(r"\('([^']+)',\s*([-+\d.eE]+)\)",message)
    else:return None
    if not pairs:return None
    context=(source or {}).get('context') or {}
    banks=(source or {}).get('metadata',{}).get('physical_banks',{})
    labels=[]
    for pair in pairs:
        variable,value=pair[:2]
        parts=variable.split(':')
        if len(parts)!=3 or parts[2] not in {'Size;i','Mustrain;i'}:return None
        phase=next((p for p in context.get('phases',[]) if str(p.get('native_id'))==parts[0]),None)
        histogram=next((h for h in context.get('histograms',[]) if str(h.get('native_id'))==parts[1]),None)
        if phase is None or histogram is None or histogram['name'] not in banks:return None
        parameter,unit=('sample size','µm') if parts[2]=='Size;i' else ('sample microstrain','µstrain')
        label=f"{phase['name']} · physical bank {banks[histogram['name']]} · {parameter} = {float(value):.5g} {unit}"
        if clipped:
            label+=' in the solver, '+('unavailable in the saved model' if pair[2]=='None' else f"{float(pair[2]):.5g} {unit} in the saved model")
        labels.append(label)
    completed=len(result.get('stages') or [])
    cause='The fitted and stored values disagree. ' if clipped else 'The value fell outside the requested physical bounds. '
    return (f"Stage {completed+1} stopped: {'; '.join(labels)}. "+cause+
            f"{completed} earlier verified stage{'s' if completed!=1 else ''} remain available; no final GPX was accepted. "
            'Review the sample-width model and its starting values before another trial.')
