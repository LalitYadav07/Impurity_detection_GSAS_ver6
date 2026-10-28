"""Scientific boundary regression tests for extended recipe operations."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from gpx_recipe.contracts import validate_plan
from gpx_recipe.planner import compact_context,expand_context
from gpx_recipe.advanced import audit, occupancy_groups, site_capacity
from gpx_recipe.execution import resolve_constraint_provenance, parameter_matches_target
from gpx_recipe.providers import duration
from test_contracts import context,plan


def test_histogram_phase_scale_degeneracy_is_rejected(context,plan):
    context['phases'][0]['histograms']=['PWDR test']
    plan['stages']=[dict(id='scales',title='Scales',cycles=4,free=[dict(kind='histogram_scale',histogram='h0'),dict(kind='phase_scale',phase='p0',histogram='h0')],constraints=[])]
    with pytest.raises(ValueError,match='sum-to-one'):validate_plan(plan,context)
    plan['stages'][0].update(free=[dict(kind='histogram_scale',histogram='h0')],phase_normalization=['h0'])
    validate_plan(plan,context) # A single phase is held at one; only global scale varies.


def test_legacy_phase_scale_convention_remains_valid(context,plan):
    plan['stages']=[dict(id='scales',title='Scales',cycles=4,free=[dict(kind='phase_scale',phase='p0',histogram='h0')],constraints=[])]
    validate_plan(plan,context)


@pytest.mark.parametrize('bounds',[(-.1,1,None),(0,1.2,None),(.8,.2,None),(.1,.8,.9)])
def test_unphysical_or_outside_start_bounds_blocked(context,plan,bounds):
    lo,hi,start=bounds;plan['stages']=plan['stages'][:1];plan['stages'][0]['free'][0]['flags']='F'
    bound=dict(phase='p0',atom='O1',parameter='F',lower=lo,upper=hi)
    if start is not None:bound['value']=start
    plan['stages'][0]['bounds']=[bound]
    with pytest.raises(ValueError):validate_plan(plan,context)


def test_zero_texture_axis_rejected(context,plan):
    plan['stages'][0]['texture']=[dict(phase='p0',histogram='h0',model='march-dollase',direction=[0,0,0])]
    with pytest.raises(ValueError,match='nonzero'):validate_plan(plan,context)


def test_ranges_require_positive_ordered_explicit_units(context,plan):
    plan['stages'][0]['fit_ranges']=[dict(histogram='h0',unit='Q',lower=3,upper=1)]
    with pytest.raises(ValueError,match='increasing'):validate_plan(plan,context)


def test_lossless_parameter_tables_preserve_all_values(context):
    context['parameter_records']=[dict(name='a',value=.3,flags=False),dict(name='b',value=None,flags=True)]
    context['phases'][0]['atoms'][0]['variables']=[dict(name='a',value=.3),dict(name='b',value=None)]
    original=copy.deepcopy(context);packed=compact_context(context)
    assert packed['parameter_records']['columns']==['name','value','flags']
    assert expand_context(packed)==original and context==original


def test_audit_is_frozen_not_a_reference_to_live_gsas_lists():
    scale=[3.0,True];limits=[[1,20],[2,19]]
    hist=SimpleNamespace(data={'Limits':limits},SampleParameters={'Scale':scale})
    project=SimpleNamespace(histogram=lambda name:hist)
    state=audit(project,dict(histograms=[dict(key='h0',name='test')],phases=[]))
    scale[0]=9.;limits[1][0]=4
    assert state['histograms'][0]['histogram_scale'][0]==3.
    assert state['histograms'][0]['limits'][1][0]==2


def test_rate_reset_duration_units():
    assert duration('1m19s')==79 and duration('2500ms')==2.5 and duration('79')==79
    assert duration('untrusted string') is None


def test_constraint_independent_variable_requires_allowed_configured_origins():
    records=[dict(name='0::Afrac:0',group='occupancy',configured_now=True),dict(name='0::AUiso:0',group='atomic_displacement',configured_now=False)]
    space=dict(parameters=records,effective_vary=['::constr0'])
    inverse=lambda name,vary: (['::constr0'],[1.])
    assert resolve_constraint_provenance(space,['::constr0'],['0::Afrac:0'],inverse,{'occupancy'})
    assert not resolve_constraint_provenance(space,['::constr0'],['0::Afrac:0','0::AUiso:0'],inverse,{'occupancy','atomic_displacement'})
    assert not resolve_constraint_provenance(space,['::constr0'],['0::Afrac:0'],inverse,{'lattice'})


def test_mixed_site_capacity_accounts_for_held_peer():
    atoms=[dict(label='A',coordinates=[0,0,0],multiplicity=4,occupancy=.4),dict(label='B',coordinates=[1,0,0],multiplicity=4,occupancy=.6)]
    assert occupancy_groups(atoms)==[['A','B']]
    context=dict(phases=[dict(key='p0',atoms=atoms)])
    phase=SimpleNamespace(atom=lambda label:SimpleNamespace(occupancy={'A':.4,'B':.6}[label]))
    free=[dict(kind='atom',phase='p0',atom='A',flags='F')]
    assert site_capacity(context,'p0','A',phase,free)==.4
    free.append(dict(kind='atom',phase='p0',atom='B',flags='F'))
    assert site_capacity(context,'p0','A',phase,free)==1.


def test_explicit_bounds_must_not_permit_overfilled_mixed_site(context,plan):
    context['phases'][0]['occupancy_groups']=[['O1','Mg1']]
    context['phases'][0]['atoms'][0]['occupancy']=.4;context['phases'][0]['atoms'][1]['occupancy']=.6
    plan['stages']=plan['stages'][:1];plan['stages'][0]['free'][0]['flags']='F'
    plan['stages'][0]['bounds']=[dict(phase='p0',atom='O1',parameter='F',lower=.1,upper=.8)]
    with pytest.raises(ValueError,match='overfilling'):validate_plan(plan,context)


def test_parameter_scope_uses_native_ids_not_sorted_phase_indices(context):
    context['phases'][0]['native_id']=5;context['histograms'][0]['native_id']=9
    context['phases'][0]['atoms'][0]['native_id']=2
    context['phases'][0]['atoms'][1]['native_id']=3
    row=dict(group='atomic_displacement',phase_id=5,histogram_id=None,atom_id=2)
    target=dict(kind='atom',phase='p0',atom='O1',flags='U')
    assert parameter_matches_target(row,target,context)
    assert not parameter_matches_target({**row,'atom_id':3},target,context)
    assert not parameter_matches_target({**row,'phase_id':0},target,context)
    hap=dict(group='phase_fraction',phase_id=5,histogram_id=9,atom_id=None)
    assert parameter_matches_target(hap,dict(kind='phase_scale',phase='p0',histogram='h0'),context)
    assert not parameter_matches_target({**hap,'histogram_id':8},dict(kind='phase_scale',phase='p0',histogram='h0'),context)


def test_explicit_stage_layout_and_optional_nulls_are_preserved():
    from gpx_recipe.planner import explicit_stage_count
    from gpx_recipe.schema import normalize_optional_nulls
    assert explicit_stage_count('Create exactly three stages. Refine O1.')==3
    assert explicit_stage_count('Apply these stages in order.\n1. Refine background and O1 jointly.')==1
    assert explicit_stage_count('Do not use two stages. Refine O1 and Uiso together.') is None
    assert explicit_stage_count('Refine O1 and Uiso together.') is None
    value={'stages':[{'bounds':[{'value':None,'lower':0.0}], 'texture':[{'value':None,'order':4,'direction':None}]}]}
    clean=normalize_optional_nulls(value)
    assert clean['stages'][0]['bounds']==[{'lower':0.0}]
    assert clean['stages'][0]['texture']==[{'order':4}]
    assert value['stages'][0]['bounds'][0]['value'] is None


def test_normalized_subset_keeps_explicitly_held_phases_and_needs_real_freedom(context,plan):
    original=copy.deepcopy(context['phases'][0]);context['phases']=[{**copy.deepcopy(original),'key':f'p{i}','name':f'phase {i}'} for i in range(3)]
    stage=dict(id='s1',title='Selected fractions',cycles=4,free=[dict(kind='histogram_scale',histogram='h0')]+[dict(kind='phase_scale',phase=p,histogram='h0') for p in ['p0','p1']],constraints=[],phase_normalization=['h0']);plan['stages']=[stage]
    validate_plan(plan,context) # p2 is deliberately held at its normalized start.
    stage['free']=stage['free'][:2]
    with pytest.raises(ValueError,match='fully determined by sum-to-one'):validate_plan(plan,context)
    stage['free']=stage['free'][:1];validate_plan(plan,context) # All fractions held; only global scale refines.


def test_later_single_fraction_release_is_also_blocked_by_persistent_normalization(context,plan):
    original=copy.deepcopy(context['phases'][0]);context['phases']=[{**copy.deepcopy(original),'key':f'p{i}','name':f'phase {i}'} for i in range(2)]
    stage=dict(id='s1',title='Normalize all',cycles=4,free=[dict(kind='histogram_scale',histogram='h0')]+[dict(kind='phase_scale',phase=p,histogram='h0') for p in ['p0','p1']],constraints=[],phase_normalization=['h0'])
    later=dict(id='s2',title='Only p0 fraction',cycles=4,free=[dict(kind='phase_scale',phase='p0',histogram='h0')],constraints=[]);plan['stages']=[stage,later]
    with pytest.raises(ValueError,match='fully determined by sum-to-one'):validate_plan(plan,context)


def test_normalization_keeps_tighter_source_scale_bounds():
    from gpx_recipe.advanced import configure
    hist=SimpleNamespace(name='h',id=0,SampleParameters={'Scale':[2.,True]})
    phases={f'p{i}':SimpleNamespace(id=i,data={'Histograms':{'h':{'Use':True,'Scale':[value,False]}}}) for i,value in enumerate([3.,1.])}
    controls={('parmMin','0:0:Scale'):.6,('parmMax','0:0:Scale'):.9}
    project=SimpleNamespace(data={'Constraints':{'data':{}}},get_Controls=lambda key,variable:controls.get((key,variable)),set_Controls=lambda key,value,variable:controls.__setitem__((key,variable),value),add_EqnConstr=lambda *a,**k:None)
    stage={'free':[{'kind':'histogram_scale','histogram':'h0'}],'phase_normalization':['h0']}
    configure(project,stage,{},phases,{'h0':hist},set())
    assert controls[('parmMin','0:0:Scale')]==.6 and controls[('parmMax','0:0:Scale')]==.9
    assert phases['p0'].data['Histograms']['h']['Scale']==[.75,False] and hist.SampleParameters['Scale'][0]==8.
    controls[('parmMin','0:0:Scale')]=.8
    with pytest.raises(ValueError,match='preserved source bounds'):configure(project,stage,{},phases,{'h0':hist},set())
