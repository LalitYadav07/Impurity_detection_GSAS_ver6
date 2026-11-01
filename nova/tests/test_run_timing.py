from datetime import datetime, timezone

import pytest

from radar_pd_nova.models import AnalysisMode, ResultStatus, RunRecord, RunStatus
from radar_pd_nova.run_timing import galaxy_terminal_timestamp, run_elapsed_label


def record(**kwargs):
    return RunRecord(uid="run", name="example", mode=AnalysisMode.FULL, history_id="history",
                     created_utc="2026-11-01T12:00:00Z", **kwargs)


def test_finished_runtime_does_not_increase_when_a_saved_run_is_reopened():
    run = record(status=RunStatus.OK, galaxy_terminal_utc="2026-11-01T12:03:21Z",
                 updated_utc="2026-11-02T19:00:00Z")
    assert run_elapsed_label(run, now=datetime(2026, 11, 3, tzinfo=timezone.utc)) == "201 s from submission"
    assert run_elapsed_label(RunRecord.model_validate(run.model_dump())) == "201 s from submission"


def test_active_elapsed_includes_queue_and_uses_utc_for_naive_galaxy_timestamps():
    run = record(status=RunStatus.QUEUED)
    run.created_utc = "2026-11-01T12:00:00"
    assert run_elapsed_label(run, now=datetime(2026, 11, 1, 12, 2, tzinfo=timezone.utc)) == "120 s elapsed"


@pytest.mark.parametrize("status", [RunStatus.OK, RunStatus.ERROR, RunStatus.CANCELLED])
def test_terminal_legacy_record_does_not_claim_local_updated_time_is_completion(status):
    run = record(status=status, updated_utc="2026-11-02T19:00:00Z")
    assert run_elapsed_label(run) == "Finished · duration unavailable"


def test_result_loading_does_not_extend_analysis_runtime():
    run = record(status=RunStatus.OK, analysis_status=RunStatus.OK, result_status=ResultStatus.COLLECTING,
                 galaxy_terminal_utc="2026-11-01T12:00:30Z")
    assert run_elapsed_label(run) == "30 s from submission"


def test_only_terminal_galaxy_job_metadata_provides_a_finished_timestamp():
    metadata = {"state": "ok", "update_time": "2026-11-01T12:03:21"}
    assert galaxy_terminal_timestamp(metadata) == "2026-11-01T12:03:21+00:00"
    assert galaxy_terminal_timestamp({**metadata, "state": "running"}) is None
    assert galaxy_terminal_timestamp({**metadata, "update_time": "bad"}) is None
    assert galaxy_terminal_timestamp({"state": "ok"}) is None


def test_invalid_or_reversed_times_are_unavailable():
    run = record(status=RunStatus.OK, galaxy_terminal_utc="2026-11-01T11:00:00Z")
    assert run_elapsed_label(run) == "Duration unavailable"
    run.galaxy_terminal_utc = "bad"
    assert run_elapsed_label(run) == "Duration unavailable"
