"""Closed held-Li grids and conditional nuisance policies, never composition fits."""
from copy import deepcopy
import math

from ..advanced import occupancy_groups
from ..contracts import validate_plan
from .refinement_policy import policy_recipe

SCHEMA = 'radar-held-li-profile/v1'
LI_ELEMENTS = {'Li', 'Li_6', 'Li_7'}


def validate_profile(profile):
    if profile is None:
        return None
    keys = {'schema', 'phase', 'atom', 'mixing_atom', 'refine_cells'}
    if not isinstance(profile, dict) or set(profile) != keys or profile['schema'] != SCHEMA:
        raise ValueError('Declare the held-Li phase, atom, mixing choice and cell policy')
    for key in ('phase', 'atom'):
        if not isinstance(profile[key], str) or not profile[key].strip() or len(profile[key]) > 128:
            raise ValueError('Choose an exact Li phase and atom label')
    other = profile['mixing_atom']
    if other is not None and (not isinstance(other, str) or not other.strip() or len(other) > 128 or other == profile['atom']):
        raise ValueError('Choose a distinct co-located mixing atom, or leave mixing fixed')
    if type(profile['refine_cells']) is not bool:
        raise ValueError('Declare whether shared cells are refined or held')
    return deepcopy(profile)


def profile_target(context, profile):
    profile = validate_profile(profile)
    phases = [p for p in context['phases'] if p['name'] == profile['phase'] and p['histograms']]
    if len(phases) != 1:
        raise ValueError('The held-Li diagnostic needs the same active phase in every trial')
    phase = phases[0]
    atoms = {a['label']: a for a in phase['atoms']}
    target = atoms.get(profile['atom'])
    if not target or target['element'] not in LI_ELEMENTS:
        raise ValueError('The held occupancy target must be an explicitly identified Li atom')
    value = target['occupancy']
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Held Li site occupancy must be finite and between zero and one')
    mixing = atoms.get(profile['mixing_atom']) if profile['mixing_atom'] else None
    capacity = None
    if profile['mixing_atom']:
        groups = occupancy_groups(phase['atoms'])
        group = next((g for g in groups if target['label'] in g and profile['mixing_atom'] in g), None)
        if not mixing or mixing['element'] in LI_ELEMENTS or group is None:
            raise ValueError('Mixing refinement requires a non-Li atom at the same coordinates and multiplicity; symmetry-equivalent sites are not guessed')
        capacity = 1 - sum(atoms[label]['occupancy'] for label in group if label != mixing['label'])
        if not 0 < capacity <= 1 or not 0 <= mixing['occupancy'] <= capacity + 1e-12:
            raise ValueError('The held Li and other occupants leave no valid interval for the mixing atom')
    return phase, target, mixing, capacity


def occupancy_variable(atom):
    names = [v['name'] for v in atom.get('variables', []) if v['group'] == 'occupancy']
    if len(names) != 1:
        raise ValueError('The held-Li diagnostic requires a unique native occupancy variable')
    return names[0]


def profile_recipe(context, physical_banks, policy, profile):
    phase, target, mixing, capacity = profile_target(context, profile)
    plan = policy_recipe(context, physical_banks, policy)
    if not profile['refine_cells']:
        # Retain bank-specific background setup stages and any reviewed width
        # stage, but remove the otherwise redundant cell-only extension.
        plan['stages'] = [s for s in plan['stages'] if s['id'] != 'cells']
        for stage in plan['stages']:
            stage['free'] = [t for t in stage['free'] if t['kind'] != 'cell']
            if stage['id'] == 'sample_widths':
                stage['title'] = 'Selected sample widths with backgrounds and phase scales; cells held'
    if mixing:
        stage = deepcopy(plan['stages'][-1])
        stage.update(id='li_nuisance', title='Co-located mixing occupancy with the reviewed nuisance parameters')
        stage['free'].append(dict(kind='atom', phase=phase['key'], atom=mixing['label'], flags='F'))
        stage['bounds'] = [dict(phase=phase['key'], atom=mixing['label'], parameter='F', lower=0., upper=capacity)]
        plan['stages'].append(stage)
        # Native objective stopping can precede a small final shift/sigma.
        # One explicit, bounded continuation uses exactly the same freedoms;
        # its final convergence evidence still has to pass the reviewed screen.
        continuation = deepcopy(stage)
        continuation.update(id='li_nuisance_recheck', title='Continue unchanged nuisance model and recheck final-cycle convergence')
        plan['stages'].append(continuation)
    plan['request'] = (f"Hold {phase['name']} / {target['label']} site occupancy at {target['occupancy']:.17g}. "
        'Optimize the explicitly reviewed backgrounds, phase scales, cell/width choices and mixing choice. '
        'This finite grid is a conditional sensitivity diagnostic, not a Li-content determination.')
    plan['assumptions'] = [
        'Instrument profile, texture, positions, displacement parameters and all unselected occupancies remain fixed.',
        'Shared cells are refined.' if profile['refine_cells'] else 'All cells remain at the reviewed starting values.',
        (f"Only {mixing['label']} occupancy is additionally released, from zero to {capacity:.17g}; vacancies are allowed."
         if mixing else 'All non-Li occupancies remain fixed; mixing sensitivity is untested.'),
        'Existing constraints and bounds remain subject to native preflight; no atom is silently released.',
        'Different ADP, texture, background and structural assumptions need separately reviewed comparisons.',
        'A grid minimum is conditional on this model. Screening bands are not confidence intervals.'
    ]
    if mixing:
        plan['assumptions'].append('One explicit nuisance continuation is included with the same free parameters and bounds; the native stopping rule and final-cycle screening ceiling are unchanged.')
    return validate_plan(plan, context)


def profile_binding(context, fixed_hypothesis, profile):
    """Compare all fixed inputs while allowing only reviewed starting freedoms."""
    phase, target, mixing, _ = profile_target(context, profile)
    fixed = deepcopy(fixed_hypothesis)
    for record in fixed:
        for atom in record['atoms']:
            if record['name'] == phase['name'] and atom['label'] == target['label']:
                atom['occupancy'] = 'held_grid_value'
            if mixing and record['name'] == phase['name'] and atom['label'] == mixing['label']:
                atom['occupancy'] = 'released_mixing_start'
    histograms = []
    for h in context['histograms']:
        if 'background_auxiliary' not in h or 'sample_parameters' not in h:
            raise ValueError('Held-Li comparison needs a fresh native inspection of background and sample inputs')
        # Native random IDs identify separate GPX objects; they are not sample
        # corrections. All other sample fields remain part of the comparison.
        sample = {k: deepcopy(v) for k, v in h['sample_parameters'].items() if k != 'ranId'}
        histograms.append(dict(key=h['key'], auxiliary=h['background_auxiliary'], sample=sample))
    controls = context.get('controls', {})
    solver = {k: controls.get(k) for k in ('deriv type', 'min dM/M', 'shift factor', 'SVDtol', 'Marquardt', 'F**2', 'HatomFix', 'newLeBail')}
    scattering = []
    for p in context['phases']:
        if (not isinstance(p.get('neutron_isotopes'), dict) or not p['neutron_isotopes']
                or not isinstance(p.get('neutron_scattering'), dict) or not p['neutron_scattering']
                or any(value is None for value in p['neutron_scattering'].values())):
            raise ValueError('Held-Li comparison needs native isotope choices and their scattering coefficients')
        scattering.append(dict(phase=p['name'], isotopes=p['neutron_isotopes'], coefficients=p['neutron_scattering']))
    return dict(topology=context['model_signature'], fixed=fixed, histograms=histograms, solver=solver,
                neutron_scattering=scattering,
                cells=None if profile['refine_cells'] else [dict(phase=p['name'], cell=p['cell']) for p in context['phases']])


def verify_profile_preview(context, preview, profile):
    phase, target, mixing, capacity = profile_target(context, profile)
    held = occupancy_variable(target)
    for stage in preview['stages']:
        if held in stage['effective_vary']:
            raise ValueError('Native preflight released the held Li occupancy')
        atoms = [a for p in stage.get('configured_state', {}).get('phases', []) if p['key'] == phase['key']
                 for a in p['atoms'] if a['label'] == target['label']]
        if len(atoms) != 1 or atoms[0]['occupancy'] != target['occupancy'] or 'F' in atoms[0]['flags']:
            raise ValueError('Native preflight does not preserve the declared held Li value')
    if mixing and occupancy_variable(mixing) not in preview['stages'][-1]['effective_vary']:
        raise ValueError('The selected mixing occupancy is not an independent nuisance parameter under the preserved constraints')
    if mixing:
        bounds = [b for b in preview['stages'][-1]['configured_state'].get('effective_bounds', [])
                  if b['phase'] == phase['key'] and b['atom'] == mixing['label'] and b['parameter'] == 'F']
        if len(bounds) != 1 or bounds[0]['lower'] != 0 or bounds[0]['upper'] != capacity:
            raise ValueError('Native preflight does not retain the reviewed vacancy-permitting mixing bounds')


def profile_description(profile):
    text = f"Held-Li target: {profile['phase']} / {profile['atom']}; occupancy values come from the prepared models. "
    text += ('Shared phase cells refined. ' if profile['refine_cells'] else 'All phase cells held. ')
    text += (f"{profile['mixing_atom']} occupancy refined from zero to the remaining same-site capacity; vacancies allowed. "
             if profile['mixing_atom'] else 'All mixing occupancies fixed. ')
    if profile['mixing_atom']:
        text += 'One unchanged-nuisance continuation stage rechecks final-cycle convergence. '
    return text + 'Positions, ADPs, other occupancies, texture and instrument profile fixed. This is conditional site sensitivity; Li composition remains unidentified.'


def verify_profile_result(view, profile, held_value, mixing_bounds=None):
    """A saved final model, not a family label, must prove Li remained held."""
    phases = [p for p in view.get('phases', []) if p['name'] == profile['phase']]
    atoms = [a for p in phases for a in p['atoms'] if a['label'] == profile['atom']]
    metrics = [p for p in view.get('parameters', []) if p.get('kind') == 'atom'
               and p.get('phase') == profile['phase'] and p.get('atom') == profile['atom'] and p.get('parameter') == 'occupancy']
    if (len(atoms) != 1 or atoms[0]['element'] not in LI_ELEMENTS or atoms[0]['occupancy'] != held_value
            or 'F' in atoms[0].get('flags', '') or len(metrics) != 1 or metrics[0]['status'] != 'fixed'
            or metrics[0]['value'] != held_value or metrics[0].get('sigma') is not None):
        raise ValueError('The final model does not preserve the reviewed fixed Li occupancy')
    if not profile['mixing_atom']:
        return dict(status='fixed_mixing_assumption')
    mixing = [a for p in phases for a in p['atoms'] if a['label'] == profile['mixing_atom']]
    fitted = [p for p in view.get('parameters', []) if p.get('kind') == 'atom'
              and p.get('phase') == profile['phase'] and p.get('atom') == profile['mixing_atom'] and p.get('parameter') == 'occupancy']
    if len(mixing) != 1 or len(fitted) != 1 or not mixing_bounds or len(mixing_bounds) != 2:
        raise ValueError('The final mixing parameter or its reviewed bounds are missing')
    value = mixing[0]['occupancy']
    lower, upper = mixing_bounds
    if not math.isfinite(value) or not lower-1e-9 <= value <= upper+1e-9 or fitted[0]['value'] != value:
        raise ValueError('The final mixing parameter violates its reviewed bounds')
    boundary = min(abs(value-lower), abs(value-upper)) <= 1e-8
    qualified = fitted[0].get('native_status') == 'refined_independent' and 'F' in mixing[0].get('flags', '')
    return dict(status='bound_or_unresolved' if boundary or not qualified else 'optimized', occupancy=value, bounds=mixing_bounds)


def summarize_profile(profile, rows, band):
    """Retain all failures; distinguish curve shape from chemical identification."""
    points = []
    for value in sorted({r['li_occupancy'] for r in rows}):
        trials = [r for r in rows if r['li_occupancy'] == value]
        completed = [r for r in trials if r['status'] == 'completed' and r['wss_per_point'] is not None]
        values = [r['wss_per_point'] for r in completed]
        spread = max(values)-min(values) if values else None
        reasons = []
        if len(completed) != len(trials): reasons.append('incomplete_trials')
        if any(r.get('convergence', {}).get('status') != 'passed' for r in completed): reasons.append('convergence_unresolved')
        if len(trials) < 2: reasons.append('single_start_only')
        if spread is not None and spread > band: reasons.append('start_sensitive')
        if any(r.get('adequacy') == 'threshold_exceeded' for r in completed): reasons.append('residual_screen_exceeded')
        if profile['mixing_atom'] and any(r.get('li_nuisance', {}).get('status') != 'optimized' for r in completed):
            reasons.append('mixing_bound_or_unresolved')
        points.append(dict(occupancy=value, declared_starts=len(trials), completed_starts=len(completed),
            minimum_wss_per_point=min(values) if values else None, start_spread=spread,
            trial_ids=[r['id'] for r in trials], reasons=reasons))
    available = [p for p in points if p['minimum_wss_per_point'] is not None]
    minimum = min((p['minimum_wss_per_point'] for p in available), default=None)
    for point in points:
        point['delta_wss_per_point'] = (point['minimum_wss_per_point']-minimum
            if point['minimum_wss_per_point'] is not None and minimum is not None else None)
    eligible = len(points) >= 3 and not any(p['reasons'] for p in points)
    near = [p['occupancy'] for p in points if p['delta_wss_per_point'] is not None and p['delta_wss_per_point'] <= band]
    status = 'optimization_unresolved'
    if eligible:
        if len(near) == len(points): status = 'flat_over_tested_grid'
        elif points[0]['occupancy'] in near or points[-1]['occupancy'] in near: status = 'grid_boundary_unresolved'
        else:
            indices = [i for i, p in enumerate(points) if p['occupancy'] in near]
            status = 'separated_low_residual_regions' if any(b-a > 1 for a, b in zip(indices, indices[1:])) else 'interior_grid_preference'
    return dict(schema='radar-held-li-summary/v1', target=deepcopy(profile), status=status, points=points,
        grid_preference=near if eligible else [], screening_band=band, composition_status='not_identified',
        interpretation='Li is held at each input value. This curve is conditional on the selected mixing, cell and width freedoms; '
            'ADPs, positions, texture, unselected occupancies and instrument profile remain fixed. '
            'A tested grid preference is not a Li composition, confidence interval or proof of a unique minimum. '
            'All declared starts, failures and convergence screens remain visible.')
