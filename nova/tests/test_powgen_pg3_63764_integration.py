from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from radar_pd_nova.powgen import (
    resolve_packaged_powgen_profile_path,
    resolve_powgen_profile,
)
from radar_pd_nova.powgen_watch import (
    ANALYZE_TOOL_ID,
    WatchDefinition,
    WatchState,
    build_submission_plan,
    discover_from_listing,
    validate_output_path,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "powgen" / "pg3_63764.json"


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_pg3_63764_read_only_powgen_to_galaxy_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the complete read-only POWGEN discovery/submission contract."""

    fixture = _fixture()
    expected = fixture["expected"]

    def reject_recursive_scan(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("POWGEN discovery must use only the bounded supplied listing")

    def reject_filesystem_write(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("POWGEN watch state and outputs must remain Galaxy-owned")

    monkeypatch.setattr(os, "walk", reject_recursive_scan)
    monkeypatch.setattr(Path, "rglob", reject_recursive_scan)
    monkeypatch.setattr(Path, "write_text", reject_filesystem_write)

    definition = WatchDefinition(
        facility=fixture["facility"],
        instrument=fixture["instrument"],
        ipts=fixture["ipts"],
        subfolder=fixture["subfolder"],
        history_id=fixture["history_id"],
        configuration_ref=fixture["configuration_ref"],
        instrument_profile_ref=fixture["instrument_profile_ref"],
    )
    state = WatchState(history_id=definition.history_id)

    discovered = discover_from_listing(definition, fixture["listing"], state)
    run = next(item for item in discovered if item.run_number == fixture["run_number"])

    assert run.source_path == expected["selected_path"]
    assert run.source_path.endswith("PG3_63764.gsa")
    assert state.discovered[run.run_id] == run

    profile = resolve_powgen_profile(
        run_number=run.run_number,
        wavelength_angstrom=fixture["wavelength_angstrom"],
        frequency_hz=fixture["frequency_hz"],
    )
    profile_path = resolve_packaged_powgen_profile_path(profile)

    assert profile.provenance.cycle == "2026B"
    assert profile.provenance.rule_id == expected["profile_rule_id"]
    assert profile.provenance.wavelength_angstrom == "1.5"
    assert profile.provenance.frequency_hz == "60"
    assert profile.profile_filename == expected["profile_filename"]
    assert profile.profile_sha256 == expected["profile_sha256"]
    assert profile_path.name == expected["profile_filename"]

    plan = build_submission_plan(definition, run, state)
    assert plan is not None
    assert plan.tool_id == ANALYZE_TOOL_ID == "neutrons_radar_pd_analyze_prototype"
    assert plan.history_id == fixture["history_id"]
    assert plan.run_id == "PG3_63764"
    assert plan.imports == {
        "diffraction_pattern": {
            "source_path": expected["selected_path"],
            "destination": "galaxy_history",
            "read_only": True,
        }
    }
    assert plan.inputs == {
        "data_inputs|input_source|source_kind": "history",
        "data_inputs|input_source|diffraction_pattern": {
            "dataset_ref": "diffraction_pattern"
        },
        "data_inputs|input_source|instrument_source|kind": "uploaded",
        "data_inputs|input_source|instrument_source|instrument_file": fixture[
            "instrument_profile_ref"
        ],
        "reproducibility|configuration_override|config_kind": "existing",
        "reproducibility|configuration_override|config_file": fixture[
            "configuration_ref"
        ],
        "library|database|database_kind": "builtin",
        "reproducibility|run_name": "PG3_63764",
    }

    state.mark_submitted(run, expected["galaxy_job_id"])
    state.mark_completed(run, expected["galaxy_result_ids"])

    # This string represents the contents of a Galaxy JSON dataset. No local
    # state file is needed, and the complete lifecycle survives the roundtrip.
    galaxy_state_dataset = state.to_json()
    restored = WatchState.from_json(galaxy_state_dataset)
    assert restored.as_dict() == state.as_dict()
    assert restored.completed["PG3_63764"].galaxy_result_ids == tuple(
        expected["galaxy_result_ids"]
    )

    serialized_plan = json.dumps(plan.as_dict(), sort_keys=True)
    assert "output_path" not in serialized_plan
    assert "output_directory" not in serialized_plan
    assert "destination\": \"galaxy_history" in serialized_plan
    with pytest.raises(ValueError, match="must not be under"):
        validate_output_path(
            f"{definition.source_directory}/radar-pd-results/PG3_63764/state.json"
        )
