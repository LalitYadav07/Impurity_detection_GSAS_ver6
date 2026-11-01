from copy import deepcopy

import pytest

from gpx_recipe.battery.recipes import baseline_recipe
from gpx_recipe.battery.refinement_policy import validate_policy, policy_recipe, policy_description
from test_contracts import context


def prepared(context):
    context['histograms'].append(dict(key='h1',name='second'))
    phase=context['phases'][0]
    phase['histograms'].append('second')
    phase['histogram_phase_models']={h:{'Size':['isotropic',[1.,1.,1.],[False]*3],
        'Mustrain':['isotropic',[1000.,1000.,1000.],[False]*3]} for h in ('h0','h1')}
    return {'PWDR test':2,'second':5}


def test_default_policy_is_exactly_the_existing_baseline(context):
    assert policy_recipe(context,{'PWDR test':2})==baseline_recipe(context)
    legacy=validate_policy();legacy.pop('cycles_per_stage')
    assert policy_recipe(context,{'PWDR test':2},legacy)==baseline_recipe(context)


def test_reviewed_cycle_limit_is_applied_without_changing_the_free_parameter_set(context):
    banks=prepared(context);policy=validate_policy()
    baseline=policy_recipe(context,banks,policy)
    policy['cycles_per_stage']=30
    revised=policy_recipe(context,banks,policy)
    assert all(s['cycles']==30 for s in revised['stages'])
    assert [s['free'] for s in revised['stages']]==[s['free'] for s in baseline['stages']]
    assert '30 cycles' in policy_description(policy)


@pytest.mark.parametrize('value',[0,31,True,6.5,'30',float('nan')])
def test_bad_cycle_limits_are_not_silently_rounded(value):
    policy=validate_policy();policy['cycles_per_stage']=value
    with pytest.raises(ValueError,match='cycle limit'):validate_policy(policy)


def test_joint_bank_backgrounds_and_only_explicit_widths_are_released(context):
    banks=prepared(context);before=deepcopy(context)
    policy=validate_policy()
    policy['background']=dict(function='chebyschev-1',coefficients=6)
    policy['sample_widths']=[dict(phase='phase',physical_bank=5,kind='sample_microstrain')]
    plan=policy_recipe(context,banks,policy)
    assert len(plan['stages'])==4
    assert [s['background']['histogram'] for s in plan['stages'] if 'background' in s]==['h0','h1']
    assert all(s['background']['fixed_points']=='preserve' for s in plan['stages'] if 'background' in s)
    widths=[t for s in plan['stages'] for t in s['free'] if t['kind'].startswith('sample_')]
    assert widths==[dict(kind='sample_microstrain',phase='p0',histogram='h1')]
    assert not any(t['kind'] in {'atom','preferred_orientation','histogram_scale'} for s in plan['stages'] for t in s['free'])
    assert 'physical bank 5 / sample microstrain' in policy_description(policy)
    assert context==before


@pytest.mark.parametrize('change',[ 'missing_phase','missing_bank','unlinked','anisotropic' ])
def test_width_selection_never_substitutes_another_phase_bank_or_model(context,change):
    banks=prepared(context);policy=validate_policy()
    policy['sample_widths']=[dict(phase='phase',physical_bank=5,kind='sample_size')]
    if change=='missing_phase':policy['sample_widths'][0]['phase']='missing'
    elif change=='missing_bank':policy['sample_widths'][0]['physical_bank']=3
    elif change=='unlinked':context['phases'][0]['histograms'].remove('second')
    else:context['phases'][0]['histogram_phase_models']['h1']['Size'][0]='uniaxial'
    with pytest.raises(ValueError):policy_recipe(context,banks,policy)


@pytest.mark.parametrize('value',[0,37,True,6.5,float('nan')])
def test_background_parameter_count_is_finite_bounded_and_integer(value):
    policy=validate_policy();policy['background']=dict(function='chebyschev-1',coefficients=value)
    with pytest.raises(ValueError):validate_policy(policy)


def test_unknown_and_duplicate_targets_cannot_expand_the_active_set():
    policy=validate_policy();target=dict(phase='phase',physical_bank=2,kind='sample_size')
    policy['sample_widths']=[target,target]
    with pytest.raises(ValueError,match='Duplicate'):validate_policy(policy)
    policy['sample_widths']=[dict(target,kind='instrument_width')]
    with pytest.raises(ValueError,match='isotropic'):validate_policy(policy)
    policy=validate_policy();policy['release_lithium']=True
    with pytest.raises(ValueError,match='explicit'):validate_policy(policy)
