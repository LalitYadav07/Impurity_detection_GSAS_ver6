"""Native applicability check for a reviewed series occupancy recipe.

This compares held model values, not acquisition conditions. A user's scope
review is still needed; GPX sample metadata is not an authoritative run log.
"""
from copy import deepcopy

from .snapshot import digest

SCHEMA = 'radar-held-reference/v1'


def _values(value):
    """Ignore refinement flags; each stage explicitly sets the complete free set."""
    if isinstance(value, bool):
        return None
    if isinstance(value, dict):
        return {key: _values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_values(item) for item in value]
    return value


def held_values(context, plan):
    free = [target for stage in plan['stages'] for target in stage['free']]
    cells = {target['phase'] for target in free if target['kind'] == 'cell'}
    atoms = {}
    for target in free:
        if target['kind'] == 'atom':
            atoms.setdefault((target['phase'], target['atom']), set()).update(target['flags'])
    scales = {(t['phase'], t['histogram']) for t in free if t['kind'] == 'phase_scale'}
    result = {}
    for phase in context['phases']:
        key = phase['key']
        if key not in cells:
            result[key + '.cell'] = phase['cell']
        for atom in phase['atoms']:
            prefix = key + '.' + atom['label']
            flags = atoms.get((key, atom['label']), set())
            for flag, field in [('F', 'occupancy'), ('X', 'coordinates'), ('U', 'adp')]:
                if flag not in flags:
                    result[prefix + '.' + field] = atom[field]
        for field in ['neutron_isotopes', 'neutron_scattering']:
            result[key + '.' + field] = phase.get(field)
        texture = {k: v for k, v in phase.get('global_texture', {}).items()
                   if k not in {'SHShow', 'PFhkl', 'PFxyz', 'PlotType'}}
        result[key + '.texture'] = _values(texture)
        for histogram, hap in phase.get('histogram_phase_models', {}).items():
            excluded = {'Histogram', 'Show', 'hId', 'Use', 'LeBail'}
            if (key, histogram) in scales:
                excluded.add('Scale')
            result[key + '.' + histogram + '.profile'] = _values({k: v for k, v in hap.items() if k not in excluded})
    for histogram in context['histograms']:
        key = histogram['key']
        # Scan labels, IDs and condition fields are not fitted corrections.
        # Conditions are explicitly reviewed by the user, not inferred here.
        excluded = {'ranId', 'InstrName', 'Temperature', 'Pressure', 'Time', 'FreePrm1', 'FreePrm2', 'FreePrm3'}
        result[key + '.sample_corrections'] = _values({k: v for k, v in histogram.get('sample_parameters', {}).items() if k not in excluded})
        result[key + '.limits'] = histogram.get('limits')
        result[key + '.background_auxiliary'] = _values(histogram.get('background_auxiliary'))
    return deepcopy(result)


def make_reference_policy(context, plan, answers):
    return dict(schema=SCHEMA, recipe=answers['recipe'], phase=answers['phase'],
                sites=[answers['sites']] if answers['recipe'] == 'vacancy' else list(answers['sites']),
                total_ions=answers.get('total_ions') if answers['recipe'] == 'redistribution' else None,
                held=held_values(context, plan))


def validate_reference_policy(context, plan):
    from .contracts import exact_keys
    from .guided import occupancy_spec, species
    policy = plan['reference_policy']
    fields = {'schema', 'recipe', 'phase', 'sites', 'total_ions', 'held'}
    exact_keys(policy, fields, fields)
    if policy['schema'] != SCHEMA or policy['recipe'] not in {'vacancy', 'redistribution'}:
        raise ValueError('Unknown held-reference policy')
    if (not isinstance(policy['phase'], str) or not isinstance(policy['sites'], list)
            or not policy['sites'] or any(not isinstance(site, str) for site in policy['sites'])
            or len(set(policy['sites'])) != len(policy['sites'])
            or not isinstance(policy['held'], dict) or not policy['held']):
        raise ValueError('Invalid held-reference targets or values')
    if (policy['recipe'] == 'vacancy' and (len(policy['sites']) != 1 or policy['total_ions'] is not None)
            or policy['recipe'] == 'redistribution' and (len(policy['sites']) < 2 or type(policy['total_ions']) not in {int, float})):
        raise ValueError('Invalid held-reference occupancy model')
    phase = next((p for p in context['phases'] if p['key'] == policy['phase']), None)
    eligible = {a['label'] for a in phase['atoms'] if species(a) in {'Li', 'Na'} and a['adp_type'] == 'I'} if phase else set()
    if not set(policy['sites']) <= eligible:
        raise ValueError('Held-reference sites require existing isotropic Li/Na atoms')
    current = held_values(context, plan)
    changed = [key for key in sorted(set(current) | set(policy['held']))
               if key not in current or key not in policy['held'] or digest(current[key]) != digest(policy['held'][key])]
    if changed:
        raise ValueError('Held reference model does not match this scan: ' + ', '.join(changed[:4]) + '. Review a new reference recipe.')
    # Recheck capacities and the multiplicity-weighted total on every arrival;
    # compatible topology alone says nothing about the current occupancies.
    answers = dict(recipe=policy['recipe'], phase=policy['phase'],
                   sites=policy['sites'][0] if policy['recipe'] == 'vacancy' else policy['sites'])
    if policy['recipe'] == 'redistribution':
        answers['total_ions'] = policy['total_ions']
    try:
        spec = occupancy_spec(context, answers)
    except (ValueError, StopIteration) as exc:
        raise ValueError('Occupancy reference model does not match this scan: ' + str(exc)) from exc
    targets = {(t['phase'], t['atom']) for stage in plan['stages'] for t in stage['free']
               if t['kind'] == 'atom' and 'F' in t['flags']}
    if targets != {(policy['phase'], site) for site in policy['sites']}:
        raise ValueError('Held-reference sites must match the released occupancies')
    last = plan['stages'][-1]
    if last.get('bounds') != spec['bounds'] or last['constraints'] != ([spec['constraint']] if spec['constraint'] else []):
        raise ValueError('Held-reference bounds and conservation equation must match the approved occupancy model')
