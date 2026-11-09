import copy
import numpy as np
import pytest
from gpx_recipe.preparation.structural_label_retention import LabelRetention,validate,validate_bundle

def example():
    allowed=np.zeros((4,7),bool)
    allowed[0,6]=True;allowed[1,0]=True;allowed[2,5:]=True;allowed[3]=True
    order=np.broadcast_to(np.arange(7),allowed.shape)
    return allowed,order

def report():
    y,order=example();stats=LabelRetention(list(range(1,8)),5);stats.add(y,order)
    return stats.finish()

def test_a_hit_on_one_compatible_alias_cannot_give_credit_to_a_lost_space_group():
    result=report()
    assert result['any_compatible_hit_rate']==.5
    assert result['all_compatible_retention']==.25
    assert result['per_class']['7']==dict(target_rows=3,retained_target_rows=0,retention=0.)
    assert result['per_class']['6']==dict(target_rows=2,retained_target_rows=0,retention=0.)
    assert result['macro_label_retention']==pytest.approx(5/7)
    assert result['worst_class_label_retention']==0.
    assert result['compatible_labels']==11 and result['retained_compatible_labels']==6

def test_batched_counts_are_identical_and_missing_subgroup_support_is_not_zero_retention():
    y,order=example();stats=LabelRetention(list(range(1,8)),5)
    stats.add(y[:2],order[:2]);stats.add(y[2:],order[2:]);assert stats.finish()==report()
    subset=LabelRetention([1,2,3],5);subset.add(np.array([[True,False,False]]),np.array([[0,1,2]]))
    value=subset.finish();assert value['top_k']==3 and value['requested_top_k']==5
    assert value['all_compatible_retention']==1.
    assert value['per_class']['2']['retention'] is None
    assert not value['experimental_coverage_qualified'] and not value['candidate_pruning_authorized']

def test_empty_observation_subset_keeps_explicit_missing_rates():
    result=LabelRetention([1,2],1).finish()
    assert result['rows']==0 and result['any_compatible_hit_rate'] is None
    assert result['macro_label_retention'] is None
    assert all(v['retention'] is None for v in result['per_class'].values())

@pytest.mark.parametrize('case',['duplicate','floating','missing_target','truncated','oversized'])
def test_invalid_or_incomplete_rankings_are_rejected(case):
    y,order=example();order=order.copy()
    if case=='duplicate':order[0,0]=1
    elif case=='floating':order=order.astype(float)
    elif case=='missing_target':y[0]=False
    elif case=='truncated':order=order[:,:5]
    else:y=np.tile(y,(65,1));order=np.tile(order,(65,1))
    with pytest.raises(ValueError):LabelRetention(list(range(1,8)),5).add(y,order)

@pytest.mark.parametrize('case',['rare_denominator','rare_retained','macro','total','any_count','all_count','depth','authority'])
def test_inconsistent_label_denominators_cannot_be_imported(case):
    value=report()
    if case=='rare_denominator':value['per_class']['7']['target_rows']=2
    elif case=='rare_retained':value['per_class']['7']['retained_target_rows']=1
    elif case=='macro':value['macro_label_retention']=1.
    elif case=='total':value['retained_compatible_labels']+=1
    elif case=='any_count':value['any_compatible_rows']=0
    elif case=='all_count':value['all_compatible_rows']=4
    elif case=='depth':value['top_k']=7
    else:value['candidate_pruning_authorized']=True
    with pytest.raises(ValueError):validate(value)

def test_both_depths_must_retain_the_native_role_denominators():
    y,order=example();one=LabelRetention(list(range(1,8)),1);one.add(y,order)
    bundle=dict(top1=one.finish(),top5=report())
    validate_bundle(bundle,list(range(1,8)),rows=4,class_rows=y.sum(0).tolist(),any_top1=.5,any_top5=.5)
    with pytest.raises(ValueError):validate_bundle(bundle,list(range(1,8)),class_rows=[1]*7)
