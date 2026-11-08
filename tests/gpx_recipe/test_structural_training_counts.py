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
