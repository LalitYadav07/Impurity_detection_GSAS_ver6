"""Generated pairs and augmented/aliased feature rows have distinct counts."""
import copy

import pytest

from gpx_recipe.preparation.structural_training_counts import describe,validate


def counts():
    return describe(['a','a','a','b','b','c'],[0,0,0,1,1,2],[True,True,False,True,False,True],
        generated_pairs=8,simulated_observations=5)


def test_counts_keep_deduplication_and_augmentation_denominators_separate():
    result=counts()
    assert result['generated_pairs']==8 and result['unique_clean_pairs']==4
    assert result['simulated_observations_generated']==5 and result['unique_simulated_only_rows']==2
    assert result['unique_rows']==6 and result['trajectory_groups']==3
    assert result['partitions']['train']==dict(rows=3,clean_pairs=2,simulated_only_rows=1,trajectory_groups=1)
    assert result['partitions']['validation']==dict(rows=2,clean_pairs=1,simulated_only_rows=1,trajectory_groups=1)


def test_a_noisy_sibling_cannot_inflate_the_number_of_independent_holdout_groups():
    with pytest.raises(ValueError,match='cannot cross'):
        describe(['a','a','b'],[0,1,2],[True,False,True],generated_pairs=2,simulated_observations=1)


@pytest.mark.parametrize('change',['bool','missing_partition','fake_clean','fake_group','missing_rows','raw_too_small','noisy_too_small'])
def test_inconsistent_saved_counts_are_not_displayed_as_dataset_evidence(change):
    value=copy.deepcopy(counts())
    if change=='bool':value['unique_clean_pairs']=True
    elif change=='missing_partition':del value['partitions']['test']
    elif change=='fake_clean':value['partitions']['train']['clean_pairs']=3
    elif change=='fake_group':value['partitions']['test']['trajectory_groups']=2
    elif change=='missing_rows':value['unique_rows']=5
    elif change=='raw_too_small':value['generated_pairs']=3
    else:value['simulated_observations_generated']=1
    with pytest.raises(ValueError):validate(value)


def coverage():
    return describe(['a','a','a','b','b','c'],[0,0,0,1,1,2],[True,True,False,True,False,True],
        generated_pairs=8,simulated_observations=5,classes=[6,8],
        compatible_labels=iter([[True,True],[True,False],[False,True],
            [True,True],[True,True],[True,False]]))


def test_family_membership_counts_clean_pairs_and_trajectories_without_double_counting_totals():
    value=coverage();part=value['family_coverage']['partitions']['train']
    assert value['schema']=='radar-structural-training-counts/v2'
    assert value['unique_clean_pairs']==4 and value['partitions']['train']['clean_pairs']==2
    assert part['6']==dict(rows=2,clean_pairs=2,simulated_only_rows=0,
        trajectory_groups=1,clean_trajectory_groups=1)
    assert part['8']==dict(rows=2,clean_pairs=1,simulated_only_rows=1,
        trajectory_groups=1,clean_trajectory_groups=1)
    assert sum(c['clean_pairs'] for c in part.values())==3  # Compatible labels overlap.
    assert value['family_coverage']['partitions']['test']['8']['rows']==0
    assert validate(counts())['schema']=='radar-structural-training-counts/v1'


@pytest.mark.parametrize('change',['unknown_schema','missing_family','wrong_family','bool',
    'inflated_clean','inflated_groups','all_empty','extra','missing_coverage'])
def test_family_coverage_cannot_inflate_or_omit_its_native_membership(change):
    value=copy.deepcopy(coverage());part=value['family_coverage']['partitions']['train']
    if change=='unknown_schema':value['schema']=[]
    elif change=='missing_family':del part['6']
    elif change=='wrong_family':value['family_coverage']['classes']=[6,9]
    elif change=='bool':part['6']['clean_pairs']=True
    elif change=='inflated_clean':part['6']['clean_pairs']=3
    elif change=='inflated_groups':part['6']['clean_trajectory_groups']=2
    elif change=='all_empty':
        for cell in part.values():
            for key in cell:cell[key]=0
    elif change=='extra':value['family_coverage']['pruning_authorized']=True
    else:del value['family_coverage']
    with pytest.raises(ValueError):validate(value)


@pytest.mark.parametrize('classes,labels',[
    ([6,6],[[True,True]]*6),([6,8],[[True,False]]*5),
    ([6,8],[[True,False]]*7),([6,8],[[False,False]]*6),
    ([6,8],[[1,False]]*6),([6,8],None),
])
def test_exact_label_inventory_is_required_for_family_coverage(classes,labels):
    with pytest.raises(ValueError):
        describe(['a','a','a','b','b','c'],[0,0,0,1,1,2],[True,True,False,True,False,True],
            generated_pairs=8,simulated_observations=5,classes=classes,compatible_labels=labels)
