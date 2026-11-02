import pytest

from radar_pd_nova.sample_labels import recorded_sample_title


@pytest.mark.parametrize('metadata, expected', [
    ({'sample_id': ' 120479 ', 'sample_name': ' Indium copper vanadate '},
     'Sample 120479 · Indium copper vanadate'),
    ({'sample_id': 120480, 'sample_name': 'Indium copper vanadate'},
     'Sample 120480 · Indium copper vanadate'),
    ({'sample_id': '12', 'sample_name': '12'}, 'Sample 12'),
    ({'sample_name': 'Unnumbered specimen'}, 'Unnumbered specimen'),
    ({}, 'Sample identity unavailable'),
    ({'sample_id': '4', 'sample_name': '<specimen>'}, 'Sample 4 · <specimen>'),
])
def test_recorded_sample_title_keeps_identity_and_honest_fallback(metadata, expected):
    assert recorded_sample_title(metadata) == expected
