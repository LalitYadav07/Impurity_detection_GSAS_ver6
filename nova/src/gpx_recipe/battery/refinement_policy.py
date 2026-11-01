"""Explicit background and sample-width policies for battery comparisons."""
from copy import deepcopy

from ..contracts import validate_plan
from .recipes import baseline_recipe

SCHEMA = 'radar-battery-refinement-policy/v1'
WIDTH_KINDS = {'sample_size': 'Size', 'sample_microstrain': 'Mustrain'}


def validate_policy(policy=None):
    if policy is None:
        return dict(schema=SCHEMA, background=None, sample_widths=[], cycles_per_stage=6)
    required = {'schema', 'background', 'sample_widths'}
    if not isinstance(policy, dict) or not required <= set(policy) or set(policy) - (required | {'cycles_per_stage'}) or policy['schema'] != SCHEMA:
        raise ValueError('Choose an explicit battery refinement policy')
    cycles = policy.get('cycles_per_stage', 6)
    if type(cycles) is not int or not 1 <= cycles <= 30:
        raise ValueError('The reviewed cycle limit must be an integer from 1 to 30')
    background = policy['background']
    if background is not None:
        if (not isinstance(background, dict) or set(background) != {'function', 'coefficients'}
                or background['function'] != 'chebyschev-1'
                or type(background['coefficients']) is not int or not 1 <= background['coefficients'] <= 36):
            raise ValueError('The reviewed background needs one to 36 Chebyshev coefficients')
    widths = policy['sample_widths']
    if not isinstance(widths, list) or len(widths) > 48:
        raise ValueError('Declare at most 48 phase/bank sample-width targets')
    seen = set()
    for row in widths:
        if not isinstance(row, dict) or set(row) != {'phase', 'physical_bank', 'kind'}:
            raise ValueError('Each sample-width target needs its phase, physical bank and kind')
        if (not isinstance(row['phase'], str) or not row['phase'].strip() or len(row['phase']) > 128
                or type(row['physical_bank']) is not int or row['physical_bank'] not in range(1, 7)
                or not isinstance(row['kind'], str) or row['kind'] not in WIDTH_KINDS):
            raise ValueError('Choose a named phase, physical bank and isotropic sample width')
        key = (row['phase'], row['physical_bank'], row['kind'])
        if key in seen:
            raise ValueError('Duplicate sample-width target')
        seen.add(key)
    result = deepcopy(policy)
    result['cycles_per_stage'] = cycles
    result['sample_widths'].sort(key=lambda row: (row['phase'], row['physical_bank'], row['kind']))
    return result


def width_targets(context, physical_banks, policy):
    result = []
    for row in validate_policy(policy)['sample_widths']:
        phases = [p for p in context['phases'] if p['name'] == row['phase']]
        histograms = [h for h in context['histograms'] if physical_banks.get(h['name']) == row['physical_bank']]
        if len(phases) != 1 or len(histograms) != 1 or histograms[0]['name'] not in phases[0]['histograms']:
            raise ValueError(f"Sample width requires the same linked phase {row['phase']} and physical bank {row['physical_bank']} in every trial")
        result.append(dict(kind=row['kind'], phase=phases[0]['key'], histogram=histograms[0]['key']))
    return result


def policy_recipe(context, physical_banks, policy=None):
    policy = validate_policy(policy)
    plan = baseline_recipe(context)
    if policy['background'] is not None:
        # The existing closed recipe supports one background configuration per
        # stage. Configure each bank explicitly while refining all backgrounds
        # and scales; later stages retain those numerical models and values.
        scales, cells = plan['stages']
        stages = []
        for index, histogram in enumerate(context['histograms'], 1):
            bank = physical_banks.get(histogram['name'])
            if type(bank) is not int:
                raise ValueError('Background review requires a physical bank for every histogram')
            stage = deepcopy(scales)
            stage.update(id=f'background_{index}', title=f'Physical bank {bank} background model with phase scales',
                background=dict(histogram=histogram['key'], **policy['background'], fixed_points='preserve'))
            stages.append(stage)
        plan['stages'] = stages + [cells]
    targets = width_targets(context, physical_banks, policy)
    if targets:
        stage = deepcopy(plan['stages'][-1])
        stage.update(id='sample_widths', title='Selected sample widths with cells, backgrounds and phase scales')
        stage['free'].extend(targets)
        plan['stages'].append(stage)
    for stage in plan['stages']:
        stage['cycles'] = policy['cycles_per_stage']
    if policy['background'] is not None or targets:
        plan['request'] = ('Battery comparison with an explicitly reviewed background and sample-width policy. '
            'Retain the instrument profile, texture and all atomic positions, occupancies and displacement parameters. '
            'This comparison does not measure Li or certify calibration.')
        plan['assumptions'] = [
            'Use the same declared policy in every family/start trial.',
            'Retain atomic positions, occupancies, displacement parameters, texture and instrument parameters.',
            'Only explicitly selected isotropic sample widths are released; other shape inputs remain fixed.',
            'Sample widths are phase/bank parameters, not automatically a shared physical size across banks.',
            'Retain existing background support points; more background terms can absorb diffraction signal.',
            'Lower residuals with extra parameters do not establish a preferred physical model or Li content.'
        ]
    if policy['cycles_per_stage'] != 6:
        plan['request'] += f" Allow up to {policy['cycles_per_stage']} cycles in every stage; retain native stopping criteria."
    return validate_plan(plan, context)


def policy_description(policy=None, *, include_atomic_summary=True):
    policy = validate_policy(policy)
    background = policy['background']
    text = ('Preserve each starting background model. ' if background is None else
            f"Chebyshev background: {background['coefficients']} coefficients per bank; existing support points preserved. ")
    text += f"Maximum {policy['cycles_per_stage']} cycles per stage; native stopping criteria retained. "
    if policy['sample_widths']:
        labels = {'sample_size': 'sample size', 'sample_microstrain': 'sample microstrain'}
        text += 'Release ' + '; '.join(f"{r['phase']} / physical bank {r['physical_bank']} / {labels[r['kind']]}"
                                     for r in policy['sample_widths']) + '. '
    else:
        text += 'All sample widths remain fixed. '
    return text + ('Atomic inputs, texture and instrument profile remain fixed.' if include_atomic_summary else '')
