import copy
import pytest

from gpx_recipe.symmetry.contracts import policy,digest,validate_protocol
from gpx_recipe.symmetry.planner import validate_intent,requested_optimizer,response_schema
from gpx_recipe.symmetry.plans import build_plan
from test_symmetry_contracts import make_protocol
from test_symmetry_planner import source,intent
from test_symmetry_plans import inputs


def legacy_policy():
    return dict(cycles=30,timeout_seconds=120,maximum_trials=2000,maximum_cell_multiplier=8,
        maximum_moment=6.,fit_band_percent=.25,maximum_mode=.5,adequate_rwp_percent=15.)


def test_old_policy_normalization_and_hash_stay_exact():
    value=policy(legacy_policy())
    assert not {'magnetic_optimizer','optimizer_iterations'}&value.keys()
    # Frozen with contracts.py from source1720495, before this feature.
    assert digest(value)=='270abb3333620a38274a2813e64a4b283da306a13221b29d5aa02e6c5592fa1e'


@pytest.mark.parametrize('value',[
    dict(magnetic_optimizer='other'),dict(optimizer_iterations=60),
    dict(magnetic_optimizer='profiled_quadratic'),
    dict(magnetic_optimizer='profiled_quadratic',optimizer_iterations=True,moment_model='symmetry_allowed'),
    dict(magnetic_optimizer='profiled_quadratic',optimizer_iterations=0,moment_model='symmetry_allowed'),
    dict(magnetic_optimizer='profiled_quadratic',optimizer_iterations=201,moment_model='symmetry_allowed'),
    dict(magnetic_optimizer='profiled_quadratic',optimizer_iterations=60,moment_model='collinear'),
    dict(magnetic_optimizer='gsas_hessian',optimizer_iterations=60),
])
def test_bad_optimizer_contracts_fail_closed(value):
    with pytest.raises(ValueError):policy(dict(legacy_policy(),**value))


@pytest.mark.parametrize('kind,k',[('structural',['0','0','0']),('magnetic',['1','0','0']),('magnetic',['1/2','0','0'])])
def test_profiled_optimizer_scope_is_explicit(tmp_path,kind,k):
    p=make_protocol(tmp_path);p.update(kind=kind,k=k)
    p['policy'].update(magnetic_optimizer='profiled_quadratic',optimizer_iterations=60,moment_model='symmetry_allowed')
    with pytest.raises(ValueError,match='optimizer'):validate_protocol(p)


def test_new_policy_and_budget_change_approval_hash(tmp_path):
    old=make_protocol(tmp_path);old['policy']['moment_model']='symmetry_allowed'
    new=copy.deepcopy(old);new['policy'].update(magnetic_optimizer='profiled_quadratic',optimizer_iterations=60)
    assert digest(validate_protocol(old))!=digest(validate_protocol(new))
    other=copy.deepcopy(new);other['policy']['optimizer_iterations']=61
    assert digest(validate_protocol(new))!=digest(validate_protocol(other))


def test_plan_propagates_only_explicit_optimizer_fields(tmp_path):
    catalog,context,settings=inputs(tmp_path)
    old=build_plan(context,catalog,tmp_path,settings)
    assert 'magnetic_optimizer' not in old['policy']
    new=build_plan(context,catalog,tmp_path,dict(settings,moment_model='symmetry_allowed',
        magnetic_optimizer='profiled_quadratic',optimizer_iterations=60))
    assert new['policy']['magnetic_optimizer']=='profiled_quadratic'
    assert new['policy']['optimizer_iterations']==60


def test_llm_cannot_silently_choose_an_optimizer(source,intent):
    intent['settings'].update(k=['0','0','0'],moment_model='symmetry_allowed',
        magnetic_optimizer='profiled_quadratic',optimizer_iterations=60)
    with pytest.raises(ValueError,match='Unknown fields'):
        validate_intent(intent,source,request='Find magnetic order')
    declared=dict(magnetic_optimizer='profiled_quadratic',optimizer_iterations=60)
    result=validate_intent(intent,source,request='Find magnetic order',declared=declared)
    assert result['settings']['optimizer_iterations']==60
    assert declared==dict(magnetic_optimizer='profiled_quadratic',optimizer_iterations=60)


def test_exact_positive_optimizer_assignments_are_binding(source,intent):
    request='Find magnetic order. Use magnetic_optimizer=profiled_quadratic, optimizer_iterations=60.'
    intent['settings'].update(k=['0','0','0'],moment_model='symmetry_allowed',
        magnetic_optimizer='profiled_quadratic',optimizer_iterations=60)
    assert validate_intent(intent,source,request=request)['settings']['optimizer_iterations']==60
    intent['settings']['optimizer_iterations']=30
    with pytest.raises(ValueError,match='declared optimizer_iterations'):validate_intent(intent,source,request=request)


@pytest.mark.parametrize('wording',[
    'Do not use magnetic_optimizer=profiled_quadratic.',
    'The previous run used magnetic_optimizer=profiled_quadratic.',
    'Use magnetic_optimizer=profiled_quadratic or magnetic_optimizer=gsas_hessian.',
    'Use magnetic_optimizer=profiled_quadratic; use magnetic_optimizer=gsas_hessian.',
    'Use optimizer_iterations=60.',
])
def test_ambiguous_or_forbidden_optimizer_assignments_do_not_opt_in(wording):
    with pytest.raises(ValueError):requested_optimizer(wording)


def test_legacy_response_schema_does_not_offer_an_optimizer(source):
    props=response_schema(source)['properties']['settings']['anyOf'][0]['properties']
    assert 'magnetic_optimizer' not in props
    selected=requested_optimizer('Use magnetic_optimizer=profiled_quadratic, optimizer_iterations=60.')
    props=response_schema(source,selected)['properties']['settings']['anyOf'][0]['properties']
    assert props['magnetic_optimizer']['enum']==['profiled_quadratic']
    assert props['optimizer_iterations']['enum']==[60]
