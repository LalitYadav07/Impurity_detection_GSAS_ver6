"""Scientific names for native failures, without interpreting them as fits."""
import re

from gpx_recipe.convergence import SCHEMA as CONVERGENCE_SCHEMA, finite


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
