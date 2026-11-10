"""Independent calibration is reserved before scores exist, never reshuffled."""
import copy

import numpy as np
import pytest

from gpx_recipe.preparation import structural_calibration_partition as partition
from gpx_recipe.preparation import structural_training_counts as counts


def cohort():
    groups=[f'group-{i:02d}' for i in range(60) for _ in range(400)]
    split=np.repeat([0]*40+[1]*10+[2]*10,400).astype(np.int8)
    clean=np.arange(len(groups))%4!=0
    y=np.zeros((len(groups),2),bool);y[np.arange(len(groups)),np.arange(len(groups))%2]=True
    return groups,split,clean,y


def describe(groups,split,clean,y,*,calibration):
    return counts.describe(groups,split.tolist(),clean.tolist(),generated_pairs=int(clean.sum()),
        simulated_observations=int((~clean).sum()),compatible_labels=y.tolist(),classes=[12,166],calibration=calibration)


def test_fixed_reservation_preserves_original_holdouts_and_every_observation_sibling():
    groups,base,clean,y=cohort();before=base.copy();result=partition.reserve(groups,base,clean,y)
    d=result.metadata;assert d['available'] and len(d['calibration_groups'])==20
    assert d['fitting_clean_pairs']==6000
    np.testing.assert_array_equal(before,base)
    np.testing.assert_array_equal(result.split[base!=0],base[base!=0])
    assert not result.split.flags.writeable
    for group in set(groups):assert len(set(result.split[np.array(groups)==group]))==1
    saved=describe(groups,result.split,clean,y,calibration=True)
    assert saved['schema']==counts.CALIBRATION_SCHEMA
    assert counts.partition_names(saved)==('train','validation','test','calibration')
    assert saved['partitions']['calibration']['clean_pairs']==6000
    assert partition.validate(d,dataset=saved)==d


def test_noise_duplication_does_not_change_the_reserved_groups_or_clean_pair_denominator():
    groups,base,clean,y=cohort();first=partition.reserve(groups,base,clean,y)
    other=partition.reserve(groups+groups,np.r_[base,base],np.r_[clean,np.zeros(len(clean),bool)],np.r_[y,y])
    assert other.metadata['calibration_groups']==first.metadata['calibration_groups']
    assert other.metadata['assignment_sha256']==first.metadata['assignment_sha256']
    assert other.metadata['fitting_clean_pairs']==first.metadata['fitting_clean_pairs']


def test_order_changes_do_not_select_new_groups():
    groups,base,clean,y=cohort();first=partition.reserve(groups,base,clean,y)
    order=np.random.default_rng(9).permutation(len(groups))
    other=partition.reserve([groups[i] for i in order],base[order],clean[order],y[order])
    assert other.metadata['calibration_groups']==first.metadata['calibration_groups']
    assert other.metadata['assignment_sha256']==first.metadata['assignment_sha256']
    np.testing.assert_array_equal(other.split,first.split[order])


def test_missing_fit_family_support_disables_calibration_without_trying_another_split():
    groups,base,clean,y=cohort();first=partition.reserve(groups,base,clean,y)
    rare=np.array([g in first.metadata['calibration_groups'][:1] for g in groups])
    labels=np.c_[y,rare|(base!=0)]
    result=partition.reserve(groups,base,clean,labels)
    assert not result.metadata['available'] and result.metadata['calibration_groups']==[]
    assert result.metadata['proposed_groups']==first.metadata['proposed_groups']
    assert 'support' in result.metadata['reason']
    np.testing.assert_array_equal(result.split,base)


@pytest.mark.parametrize('case',['few_groups','few_clean_pairs'])
def test_small_preparations_stay_usable_without_pretending_to_have_confidence(case):
    groups,base,clean,y=cohort()
    if case=='few_groups':
        chosen=np.r_[np.arange(400),np.arange(16000,16400),np.arange(20000,20400)]
        groups=[groups[i] for i in chosen];base=base[chosen];clean=clean[chosen];y=y[chosen]
    else:clean[:]=True;clean[::2]=False
    result=partition.reserve(groups,base,clean,y)
    assert not result.metadata['available']
    np.testing.assert_array_equal(result.split,base)
    legacy=describe(groups,result.split,clean,y,calibration=False)
    assert legacy['schema']==counts.COVERAGE_SCHEMA and partition.validate(result.metadata,dataset=legacy)


@pytest.mark.parametrize('case',['cross_group','new_fold','integer_labels','unknown_group','no_clean'])
def test_invalid_original_partitions_cannot_claim_an_independent_reservation(case):
    groups,base,clean,y=cohort()
    if case=='cross_group':base[0]=1
    elif case=='new_fold':base[0]=3
    elif case=='integer_labels':y=y.astype(np.int8)
    elif case=='unknown_group':groups[0]=''
    else:clean[:]=False
    with pytest.raises(ValueError):partition.reserve(groups,base,clean,y)


@pytest.mark.parametrize('case',['trained_count','calibration_count','groups','flag','minimum','extra','schema'])
def test_saved_counts_cannot_claim_a_different_fitting_or_calibration_partition(case):
    groups,base,clean,y=cohort();result=partition.reserve(groups,base,clean,y)
    d=copy.deepcopy(result.metadata);saved=describe(groups,result.split,clean,y,calibration=True)
    if case=='trained_count':d['fitting_clean_pairs']+=1
    elif case=='calibration_count':saved['partitions']['calibration']['rows']+=1
    elif case=='groups':d['calibration_groups'].pop()
    elif case=='flag':d['experimental_coverage_qualified']=True
    elif case=='minimum':d['minimum_clean_fitting_pairs']=1
    elif case=='extra':d['prune_all_other_families']=True
    else:saved['schema']=[]
    with pytest.raises(ValueError):partition.validate(d,dataset=saved)


def test_ui_reports_actual_fitting_pairs_separately_from_calibration(monkeypatch):
    from radar_pd_nova import preparation_labels
    groups,base,clean,y=cohort();result=partition.reserve(groups,base,clean,y)
    saved=describe(groups,result.split,clean,y,calibration=True)
    monkeypatch.setattr(preparation_labels,'_entry_detail',lambda *a:dict(dataset=saved))
    text=preparation_labels.training_description(None,{'model':{'kind':'structural-training'}})
    assert 'train / validation / test / calibration: 6,000 / 3,000 / 3,000 / 6,000' in text
