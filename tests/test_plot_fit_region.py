from types import SimpleNamespace

from scripts.plot_payload import histogram_fit_region


def test_active_gsas_limits_are_distinct_from_original_measurement_limits():
    histogram = SimpleNamespace(data={"Limits": [(1, 100), [10, 80], [20, 25], [50, 55]]})
    assert histogram_fit_region(histogram) == ([10.0, 80.0], [[20.0, 25.0], [50.0, 55.0]])


def test_missing_or_nonfinite_limits_do_not_create_an_inferred_fit_range():
    for limits in [[], [(1, 100)], [(1, 100), [float("nan"), 80]], [(1, 100), [80, 10]]]:
        assert histogram_fit_region(SimpleNamespace(data={"Limits": limits}))[0] is None
