from copy import deepcopy
import pytest
from gpx_recipe.tof_profile import capabilities, gamma_domain
from gpx_recipe.contracts import validate_plan, stage_table
from gpx_recipe.execution import parameter_matches_target
from gpx_recipe.schema import response_schema
from test_contracts import context, plan


def test_release_is_available_native_tof_term_and_exact_histogram(context, plan):
    context['histograms'][0].update(native_id=3,instrument={'Type':'PNT','X':0,'Y':-1})
    t={'kind':'instrument_profile','histogram':'h0','parameter':'X'}
    plan['stages']=[dict(id='width',title='TOF width',cycles=20,free=[t],constraints=[])]
    validate_plan(plan,context)
    assert stage_table(plan)[0]['free']==['instrument_profile:h0:X']
    row={'name':':3:X','group':'instrument_profile','phase_id':None,'histogram_id':3,'family':'X'}
    assert parameter_matches_target(row,t,context)
    for wrong in ({'family':'Y'},{'histogram_id':4},{'phase_id':2}):
        assert not parameter_matches_target({**row,**wrong},t,context)
    t['parameter']='Z'
    with pytest.raises(ValueError,match='native TOF'):validate_plan(plan,context)
    t['parameter']=['X']
    with pytest.raises(ValueError,match='native TOF'):validate_plan(plan,context)
    t['parameter']='X';context['histograms'][0]['instrument']['Type']='PNC'
    with pytest.raises(ValueError):validate_plan(plan,context)


def test_no_joint_sample_width_release(context,plan):
    context['histograms'][0]['instrument']={'Type':'PNT','X':0}
    context['phases'][0]['histogram_phase_models']={'h0':{'Size':['isotropic']}}
    plan['stages'][0]['free']=[dict(kind='instrument_profile',histogram='h0',parameter='X'),dict(kind='sample_size',phase='p0',histogram='h0')]
    with pytest.raises(ValueError,match='confounded'):validate_plan(plan,context)


def test_signed_coefficients_and_interior_width_minimum():
    assert gamma_domain({'X':5.25,'Y':-.2},.2,4)['gamma_min']>0
    with pytest.raises(ValueError,match='negative'):gamma_domain({'Z':3,'X':-4,'Y':1},1,3)
    assert capabilities({'instrument':{'Type':'PNT','X':0,'Y':-1}})=={'X':0.,'Y':-1.}
    assert not capabilities({'instrument':{'Type':'PNC','X':1}})


def test_profile_and_atom_x_are_different_schema_targets():
    kinds=response_schema()['properties']['stages']['items']['properties']['free']['items']['anyOf']
    assert any(t['properties']['kind']['enum']==['instrument_profile'] for t in kinds)
    assert any(t['properties']['kind']['enum']==['atom'] for t in kinds)


def test_literal_profile_terms_preserve_each_coefficient_and_bank():
    from gpx_recipe.intent import literal_stage_cards
    ctx={'histograms':[{'key':'h0','name':'bank1'},{'key':'h1','name':'bank2'}], 'phases':[]}
    cards=literal_stage_cards('Complete free set.\nStage 1: Free: h0 TOF instrument profile X; h0 TOF instrument profile Y; h1 TOF instrument profile Z.',ctx)
    assert cards[0]['literal_free']==[
        {'kind':'instrument_profile','histogram':'h0','parameter':'X'},
        {'kind':'instrument_profile','histogram':'h0','parameter':'Y'},
        {'kind':'instrument_profile','histogram':'h1','parameter':'Z'}]
