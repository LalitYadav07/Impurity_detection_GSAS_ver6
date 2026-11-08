"""Adaptive sets preserve ties, physical groups and ambiguous alternatives."""
import copy
import math

import numpy as np
import pytest

from gpx_recipe.preparation import structural_adaptive_sets as sets

IDENTITIES={k:chr(97+i)*64 for i,k in enumerate(['weights_sha256','parent_sha256','instrument_sha256','dataset_sha256'])}


def calibration(*,groups=19,gap=.7,duplicate=1):
    scores=np.tile([gap,0.,-2.],(groups*duplicate,1)).astype(np.float64)
    allowed=np.tile([False,True,False],(len(scores),1))
    # Include all three classes without altering the worst-case score gap.
    allowed[0,0]=True;scores[-1]=[0.,-2.,gap];allowed[-1]=[True,False,True]
    return sets.calibrate(scores,allowed,[f'g{i}' for i in range(groups) for _ in range(duplicate)],
        classes=[12,166,194],identities=IDENTITIES,used_groups=['fit','validation','test'])


def test_adaptive_size_changes_with_the_scores_without_a_top_five_cap():
    c=calibration()
    assert c['quantile_rank']==19 and c['maximum_gap']==.7
    narrow=sets.select(np.array([[2.,0.,-2.]]),c,identities=IDENTITIES)
    wide=sets.select(np.array([[2.,1.4,1.35]]),c,identities=IDENTITIES)
    assert narrow['retained_space_groups']==[12]
    assert wide['retained_space_groups']==[12,166,194]
    assert narrow['deferred_space_groups']==[166,194]
    assert not narrow['candidate_pruning_authorized'] and not narrow['model_scores_are_probabilities']


def test_ties_and_a_chain_of_similar_scores_are_never_split():
    c=calibration(gap=0.)
    assert c['maximum_gap']==0.
    tied=sets.select(np.zeros((1,3)),c,identities=IDENTITIES)
    close=sets.select(np.array([[0.,-.2,-.4]]),c,identities=IDENTITIES)
    assert tied['retained_space_groups']==close['retained_space_groups']==[12,166,194]
    gap=sets.select(np.array([[0.,-.2,-.5]]),c,identities=IDENTITIES)
    assert gap['retained_space_groups']==[12,166]


def test_group_score_keeps_every_compatible_label_and_every_observation_variant():
    logits=np.array([[3.,1.,0.],[3.,0.,1.],[0.,2.,1.]])
    labels=np.array([[True,True,False],[True,False,True],[False,True,False]])
    c=sets.calibrate(logits,labels,['one','one','two'],classes=[1,2,3],identities=IDENTITIES,
        used_groups=['held-out'],error_rate=.5)
    assert c['groups']==[dict(id='one',maximum_gap=2.),dict(id='two',maximum_gap=0.)]
    assert c['maximum_gap']==2.
    selected=sets.select(logits[:1],c,identities=IDENTITIES)
    assert selected['retained_space_groups']==[1,2]


def test_correlated_augmentations_cannot_manufacture_independent_calibration():
    one=calibration(groups=1,duplicate=100)
    assert one['rows']==100 and len(one['groups'])==1
    assert one['maximum_gap'] is None
    r=sets.select(np.array([[100.,0.,-100.]]),one,identities=IDENTITIES)
    assert r['retained_space_groups']==[12,166,194]
    assert len(calibration(groups=19,duplicate=2)['groups'])==19


def test_missing_class_support_or_domain_mismatch_retains_every_family():
    c=sets.calibrate(np.tile([1.,0.,-1.],(19,1)),np.tile([True,False,False],(19,1)),
        [f'g{i}' for i in range(19)],classes=[1,2,3],identities=IDENTITIES,used_groups=[])
    assert c['maximum_gap'] is None and c['class_target_coverage']==[True,False,False]
    r=sets.select(np.array([[2.,1.,0.]]),calibration(),identities=IDENTITIES,within_support=False)
    assert r['retained_space_groups']==[12,166,194] and not r['deferred_space_groups']


def test_shifted_logits_do_not_change_the_candidate_set():
    c=calibration();a=np.array([[1.,.5,-2.]])
    before=sets.select(a,c,identities=IDENTITIES)
    assert sets.select(a+1024,c,identities=IDENTITIES)==before


def test_test_metrics_expose_false_exclusions_aliases_and_correlated_rows():
    c=calibration()
    scores=np.array([[3.,0.,-1.],[0.,3.,-1.],[0.,0.,0.]])
    labels=np.array([[True,True,False],[False,True,False],[False,False,True]])
    result=sets.evaluate(scores,labels,['test-1','test-1','test-2'],c,identities=IDENTITIES)
    assert result['any_compatible_retention']==1.
    assert result['all_compatible_retention']==pytest.approx(2/3)
    assert result['all_compatible_group_retention']==.5
    assert result['excluded_compatible_labels']==1
    assert result['per_class']['166']['retention']==.5
    assert result['retained_size']['maximum']==3
    with pytest.raises(ValueError,match='own calibration'):
        sets.evaluate(scores,labels,['g0','g0','new'],c,identities=IDENTITIES)


@pytest.mark.parametrize('change',['weights','parent','instrument','dataset'])
def test_calibration_cannot_be_borrowed_for_other_physics_or_weights(change):
    c=calibration();identities=dict(IDENTITIES)
    identities[{'weights':'weights_sha256','parent':'parent_sha256','instrument':'instrument_sha256',
        'dataset':'dataset_sha256'}[change]]='f'*64
    with pytest.raises(ValueError,match='different weights'):
        sets.select(np.zeros((1,3)),c,identities=identities)


@pytest.mark.parametrize('case',['group_leak','missing_label','integer_logits','nan','overflow','class_order'])
def test_invalid_or_leaking_inputs_cannot_produce_a_calibration(case):
    logits=np.array([[1.,0.,-1.]]);labels=np.array([[True,False,False]]);groups=['one'];used=['fit'];classes=[1,2,3]
    if case=='group_leak':used=['one']
    elif case=='missing_label':labels[:]=False
    elif case=='integer_logits':logits=logits.astype(np.int64)
    elif case=='nan':logits[0,0]=math.nan
    elif case=='overflow':logits[0,:2]=[np.finfo(float).max,-np.finfo(float).max]
    else:classes=[2,1,3]
    with pytest.raises(ValueError):sets.calibrate(logits,labels,groups,classes=classes,identities=IDENTITIES,used_groups=used)


@pytest.mark.parametrize('case',['threshold','denominator','rank','qualified','pruning','extra','coverage','ratio'])
def test_saved_flags_and_changed_quantiles_cannot_authorize_exclusion(case):
    c=copy.deepcopy(calibration())
    if case=='threshold':c['maximum_gap']=0.
    elif case=='denominator':c['groups'].pop()
    elif case=='rank':c['quantile_rank']-=1
    elif case=='qualified':c['experimental_coverage_qualified']=True
    elif case=='pruning':c['candidate_pruning_authorized']=True
    elif case=='extra':c['absolute_absence']=True
    elif case=='ratio':c['adjacent_score_ratio']=.9
    else:c['class_target_coverage'][0]=False
    with pytest.raises(ValueError):sets.validate(c)
