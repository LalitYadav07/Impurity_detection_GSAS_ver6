"""Readable summaries must preserve their exact scientific source values."""
from copy import deepcopy

import pytest

from radar_pd_nova.results import build_result_view, phase_fraction_rows


@pytest.mark.parametrize(('formula', 'display'), [
    ('LiNi0.333333333333Co0.333333333333Mn0.333333333334O2',
     'LiNi0.3333Co0.3333Mn0.3333O2'),
    ('Fe0.000000001234567O', 'Fe0.000000001235O'),
    ('Li0.9999999999CoO2', 'LiCoO2'),
    ('Al2O3', 'Al2O3'),
    ('alpha iron 0.123456789', 'alpha iron 0.123456789'),
    ('K7.32Ni10Te6O32 - published partial-K model',
     'K7.32Ni10Te6O32 - published partial-K model'),
])
def test_readable_formula_does_not_change_source_or_fraction(tmp_path, formula, display):
    source = {'phases': [{'phase': formula, 'weight_percent': 61.23456789}]}
    original = deepcopy(source)
    view = build_result_view(source, tmp_path)
    assert view.phases[0]['phase_display'] == display
    assert view.phases[0]['phase'] == phase_fraction_rows(source)[0]['phase'] == formula
    assert view.phases[0]['weight_percent'] == 61.23456789
    assert source == original


def test_acquisition_warning_explains_missing_temperature_without_rewriting_source(tmp_path):
    message = 'GPX acquisition metadata requires review: product_interval_needs_review'
    source = {'scientific_status': 'needs_review',
              'warnings': [{'code': 'acquisition_handoff_review', 'message': message}]}
    original = deepcopy(source)
    view = build_result_view(source, tmp_path)
    assert view.review_status == 'needs_review'
    assert 'acquisition interval is unverified' in view.warnings[0]
    assert 'temperature was not assigned to the GPX' in view.warnings[0]
    assert 'start and stop times' in view.warnings[0]
    assert source == original


def test_unknown_acquisition_reason_remains_visible(tmp_path):
    source = {'warnings': [{'code': 'acquisition_handoff_review',
                            'message': 'GPX acquisition metadata requires review: future_review_reason'}]}
    assert 'future_review_reason' in build_result_view(source, tmp_path).warnings[0]
