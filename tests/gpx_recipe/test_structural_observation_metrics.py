import numpy as np
import pytest

from gpx_recipe.preparation.structural_observation_metrics import ObservationMetrics


def test_clean_failure_cannot_hide_behind_noisy_success_or_duplicate_batches():
    # Three noise siblings must not make their failed clean source look good.
    clean = np.array([True, False, False, False, True, False])
    labels = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1]], bool)
    hit = np.array([False, True, True, True, False, True])
    metrics = ObservationMetrics(clean, 6, 2)
    for selected in (np.array([4, 2, 0]), np.array([5, 1, 3])):
        metrics.add(selected, labels[selected], hit[selected], hit[selected])
    parts = metrics.finish()['partitions']
    assert parts['clean_native'] == dict(rows=2, represented_classes=2,
        top1=0., top5=0., macro_top5=0., worst_class_top5=0.)
    assert parts['simulated_observation'] == dict(rows=4, represented_classes=2,
        top1=1., top5=1., macro_top5=1., worst_class_top5=1.)


def test_subset_does_not_invent_scores_for_unrepresented_classes_or_missing_noise():
    metrics = ObservationMetrics(np.ones(3, bool), 3, 4)
    labels = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0]], bool)
    first = np.array([True, False, False]); five = np.array([True, True, False])
    metrics.add(np.arange(3), labels, first, five)
    parts = metrics.finish()['partitions']
    assert parts['clean_native']['represented_classes'] == 2
    assert parts['clean_native']['top5'] == pytest.approx(2/3)
    assert parts['clean_native']['macro_top5'] == .75
    assert parts['clean_native']['worst_class_top5'] == .5
    assert parts['simulated_observation'] == dict(rows=0, represented_classes=0,
        top1=None, top5=None, macro_top5=None, worst_class_top5=None)


@pytest.mark.parametrize('flags', [[0, 1], np.array([True]), np.array([[True, False]])])
def test_provenance_must_be_exact_boolean_per_row(flags):
    with pytest.raises(ValueError, match='provenance'):
        ObservationMetrics(flags, 2, 3)
