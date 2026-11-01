from copy import deepcopy

import pytest

from gpx_recipe.contracts import validate_plan
from gpx_recipe.execution import parameter_matches_target
from gpx_recipe.intent import parse_free_item
from gpx_recipe.planner import build_payload
from test_contracts import context, plan


@pytest.mark.parametrize('kind,family,group',[('sample_size','Size','crystallite_size'),('sample_microstrain','Mustrain','microstrain')])
def test_width_release_is_scoped_to_exact_coefficient(context,plan,kind,family,group):
    phase=context['phases'][0]
    phase['native_id']=3;context['histograms'][0]['native_id']=2
    phase['histograms']=[context['histograms'][0]['name']]
    phase['histogram_phase_models']={'h0':{family:['isotropic',[1,1,.5],[False]*3]}}
    target=dict(kind=kind,phase='p0',histogram='h0')
    plan['stages']=[dict(id='width',title='Reviewed sample width',cycles=3,free=[target],constraints=[])]
    validate_plan(plan,context)
    row=dict(name=f"{phase['native_id']}:{context['histograms'][0]['native_id']}:{family};i",group=group,
             phase_id=phase['native_id'],histogram_id=context['histograms'][0]['native_id'],family=family+';i')
    assert parameter_matches_target(row,target,context)
    row['family']=family+';mx'
    assert not parameter_matches_target(row,target,context)
    row['family']=family+';i';row['histogram_id']=999
    assert not parameter_matches_target(row,target,context)
    phase['histogram_phase_models']['h0'][family][0]='uniaxial'
    with pytest.raises(ValueError,match='existing isotropic'):validate_plan(plan,context)


def test_literal_width_request_keeps_phase_and_bank_scope(context):
    assert parse_free_item('p0 h0 sample microstrain',context)==dict(kind='sample_microstrain',phase='p0',histogram='h0')
    assert parse_free_item('p0 h0 isotropic crystallite size',context)==dict(kind='sample_size',phase='p0',histogram='h0')
    assert parse_free_item('p0 h0 anisotropic crystallite size',context) is None
    assert parse_free_item('p0 h0 instrument size',context) is None


def test_width_capabilities_do_not_claim_anisotropic_release(context):
    context['phases'][0]['histogram_phase_models']={'h0':{'Size':['isotropic'],'Mustrain':['generalized']}}
    payload=build_payload('Inspect supported widths',context)
    capability=payload['target_capabilities'][0]['sample_widths']['h0']
    assert capability['sample_size']['supported']
    assert not capability['sample_microstrain']['supported']
