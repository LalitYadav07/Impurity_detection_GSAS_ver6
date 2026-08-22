from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from radar_pd_nova.powgen_watch import (
    ANALYZE_TOOL_ID,
    WatchDefinition,
    WatchState,
    build_submission_plan,
    discover_from_listing,
    load_watch_state,
    save_watch_state,
    validate_output_path,
)


def _definition() -> WatchDefinition:
    return WatchDefinition(
        facility="sns",
        instrument="pg3",
        ipts="ipts-321",
        subfolder="shared/autoreduce",
        history_id="galaxy-history",
        config_refs={
            "configuration": "configuration-hda",
            "instrument_profile": "instrument-hda",
        },
    )


def test_definition_is_confined_to_sns_powgen() -> None:
    definition = _definition()
    assert definition.source_directory == "/SNS/PG3/IPTS-321/shared/autoreduce"
    assert definition.configuration_ref == "configuration-hda"

    with pytest.raises(ValueError, match="SNS"):
        WatchDefinition(
            facility="HFIR",
            instrument="PG3",
            ipts="IPTS-321",
            subfolder="shared",
            history_id="history",
            configuration_ref="config",
            instrument_profile_ref="profile",
        )


def test_discovery_uses_supplied_listing_and_is_idempotent() -> None:
    definition = _definition()
    state = WatchState(history_id=definition.history_id)
    listing = [
        {"path": "/SNS/PG3/IPTS-321/shared/autoreduce/PG3_100.dat", "size": 10},
        {"path": "/SNS/PG3/IPTS-321/shared/autoreduce/PG3_100.gsa", "size": 20},
        {"name": "PG3_101.xye", "size": 30, "modified_ns": 40},
        {"path": "/SNS/PG3/IPTS-999/shared/autoreduce/PG3_999.gsa"},
        {"name": "notes.txt"},
    ]

    runs = discover_from_listing(definition, listing, state)

    assert [run.run_id for run in runs] == ["PG3_100", "PG3_101"]
    assert runs[0].source_path.endswith("PG3_100.gsa")
    assert set(state.discovered) == {"PG3_100", "PG3_101"}
    assert discover_from_listing(definition, listing, state) == []


def test_discovery_can_limit_initial_history_and_continue_from_cursor() -> None:
    definition = _definition()
    state = WatchState(history_id=definition.history_id)
    listing = [f"PG3_{run}.gsa" for run in (100, 101, 102)]

    initial = discover_from_listing(definition, listing, state, newest_limit=1)
    later = discover_from_listing(
        definition,
        listing + ["PG3_103.gsa", "PG3_104.gsa"],
        state,
        minimum_run_number=103,
    )

    assert [run.run_number for run in initial] == [102]
    assert [run.run_number for run in later] == [103, 104]


def test_state_roundtrip_preserves_galaxy_job_and_result_ids(tmp_path: Path) -> None:
    definition = _definition()
    state = WatchState(history_id=definition.history_id)
    run = discover_from_listing(definition, ["PG3_100.gsa"], state)[0]
    state.mark_submitted(run, "galaxy-job-1")
    state.mark_completed(run, ["result-archive", "result-report"])

    state_path = tmp_path / "powgen-watch" / "history.json"
    save_watch_state(state_path, state)
    restored = load_watch_state(state_path)

    assert restored.as_dict() == state.as_dict()
    completed = restored.completed["PG3_100"]
    assert completed.galaxy_job_id == "galaxy-job-1"
    assert completed.galaxy_result_ids == ("result-archive", "result-report")
    assert json.loads(state_path.read_text(encoding="utf-8"))["history_id"] == "galaxy-history"


def test_submission_retry_is_durable_and_only_due_after_delay() -> None:
    definition = _definition()
    state = WatchState(history_id=definition.history_id)
    run = discover_from_listing(definition, ["PG3_100.gsa"], state)[0]

    pending = state.defer_submission(run, "temporary Galaxy failure", retry_after_seconds=30)
    before_retry = datetime.fromisoformat(pending.last_attempt_utc) + timedelta(seconds=10)
    after_retry = datetime.fromisoformat(pending.next_retry_utc) + timedelta(seconds=1)
    restored = WatchState.from_dict(state.as_dict())

    assert pending.submission_attempts == 1
    assert pending.error == "temporary Galaxy failure"
    assert restored.retryable_runs(now=before_retry.astimezone(timezone.utc)) == []
    assert [item.run_id for item in restored.retryable_runs(now=after_retry.astimezone(timezone.utc))] == [
        "PG3_100"
    ]


@pytest.mark.parametrize(
    "path",
    [
        "/SNS/PG3/IPTS-321/shared/state.json",
        "/HFIR/HB2A/IPTS-321/shared/state.json",
        r"\SNS\PG3\IPTS-321\shared\state.json",
    ],
)
def test_experiment_trees_are_never_output_destinations(path: str) -> None:
    with pytest.raises(ValueError, match="must not be under"):
        validate_output_path(path)


def test_submission_plan_only_targets_existing_galaxy_analyze() -> None:
    definition = _definition()
    state = WatchState(history_id=definition.history_id)
    run = discover_from_listing(definition, ["PG3_100.gsa"], state)[0]

    first = build_submission_plan(definition, run, state)
    second = build_submission_plan(definition, run, state)

    assert first == second
    assert first is not None
    assert first.tool_id == ANALYZE_TOOL_ID == "neutrons_radar_pd_analyze_prototype"
    assert first.history_id == definition.history_id
    assert first.imports["diffraction_pattern"]["read_only"] is True
    assert first.inputs["reproducibility|configuration_override|config_file"] == "configuration-hda"
    serialized = json.dumps(first.as_dict(), sort_keys=True)
    assert "ndip_runner" not in serialized
    assert "subprocess" not in serialized

    state.mark_submitted(run, "galaxy-job-1")
    assert build_submission_plan(definition, run, state) is None
    assert state.mark_submitted(run, "galaxy-job-1").galaxy_job_id == "galaxy-job-1"
    with pytest.raises(ValueError, match="another Galaxy job"):
        state.mark_submitted(run, "galaxy-job-2")
