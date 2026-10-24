"""Facility timestamp behavior must agree on native and container Python."""
from datetime import datetime
import re

import pytest

from radar_pd_nova import facility_metadata
from radar_pd_nova.powgen_metadata import _duration_seconds
from radar_pd_nova.post_analysis import timestamp


@pytest.mark.parametrize("suffix", ["-04:00", "+00:00", "Z"])
def test_nexus_nanoseconds_are_normalized_before_the_python_parser(monkeypatch, suffix):
    class Python310Parser:
        @staticmethod
        def fromisoformat(text):
            assert not re.search(r"\.\d{7}", text)
            return datetime.fromisoformat(text)

    monkeypatch.setattr(facility_metadata, "datetime", Python310Parser)
    start = "2014-05-12T09:38:22.519596666" + suffix
    end = "2014-05-12T09:38:23.136252666" + suffix
    assert facility_metadata.parse_nexus_timestamp(start).microsecond == 519596
    assert _duration_seconds(start, end) == 0.617
    assert timestamp(start) is not None
    assert timestamp(end) - timestamp(start) == pytest.approx(0.616656, abs=1e-6)


def test_missing_offset_is_not_invented_for_a_scientific_time_axis():
    assert timestamp("2014-05-12T09:38:22.519596666") is None
    assert timestamp("invalid timestamp") is None
    assert _duration_seconds("invalid", "invalid") is None
    assert _duration_seconds("2014-05-12T09:38:22", "2014-05-12T09:38:23Z") is None
