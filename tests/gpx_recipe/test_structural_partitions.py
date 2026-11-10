import numpy as np
import pytest

from gpx_recipe.preparation.structural_partitions import partition


def example():
    # Each group is a different physical trajectory. Some pure lattice
    # trajectories lack the low-symmetry labels of atomic distortions.
    coverage=np.array([[1,1,0,0],[1,0,1,0],[1,0,0,1],[1,1,1,1],
        [1,1,0,1],[1,0,1,1],[1,1,1,0],[1,1,0,0],
        [1,0,1,0],[1,0,0,1],[1,1,1,1],[1,1,1,1],
        [1,1,0,1],[1,0,1,1],[1,1,1,0],[1,1,1,1]],bool)
    return [f'physical-{i}' for i in range(len(coverage))],coverage


def test_all_classes_have_independent_train_validation_and_test_support():
    groups,labels=example();result=partition(groups,labels)
    assert {name:sum(v==name for v in result.values()) for name in ('train','validation','test')}==dict(train=12,validation=2,test=2)
    for name in ('train','validation','test'):
        selected=[result[g]==name for g in groups]
        assert labels[selected].any(0).all()
    assert result==partition(groups,labels)


def test_noisy_siblings_and_input_row_order_cannot_change_group_assignments():
    groups,labels=example();expected=partition(groups,labels)
    extended_groups=np.repeat(groups,5).tolist();extended_labels=np.repeat(labels,5,axis=0)
    rng=np.random.default_rng(7318);order=rng.permutation(len(extended_groups))
    actual=partition([extended_groups[i] for i in order],extended_labels[order])
    assert actual==expected


def test_noise_variants_do_not_substitute_for_independent_structures():
    groups=['same-trajectory']*20+['another-trajectory']*20+['third-trajectory']*20
    labels=np.zeros((60,2),bool);labels[:,0]=True;labels[:20,1]=True
    with pytest.raises(ValueError,match='three independent'):partition(groups,labels)


def test_failed_coverage_never_drops_a_class_or_changes_input_labels():
    # Three groups per class, but a one-group holdout cannot cover two
    # disjoint classes. More independent data or a separately declared split
    # design is required; do not publish a partial-class validation result.
    groups=[f'g{i}' for i in range(6)];labels=np.array([[1,0]]*3+[[0,1]]*3,bool);before=labels.copy()
    with pytest.raises(ValueError,match='class-complete'):partition(groups,labels)
    np.testing.assert_array_equal(labels,before)
