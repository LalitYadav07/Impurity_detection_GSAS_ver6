from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from radar_pd_nova.facility import FacilityBrowser, WatchCandidate, WatchRecipe


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ndip_ipts_watch.py"
SPEC = importlib.util.spec_from_file_location("ndip_ipts_watch", SCRIPT)
assert SPEC and SPEC.loader
WATCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WATCH
SPEC.loader.exec_module(WATCH)


def _candidate(fingerprint: str = "fingerprint") -> WatchCandidate:
    return WatchCandidate("shared/incoming/scan.dat", 100, 1234, fingerprint)


def test_new_file_must_remain_stable_before_processing() -> None:
    state = WATCH.load_state(Path("missing-state.json"))
    candidate = _candidate()

    assert WATCH.ready_candidates([candidate], state, settle_seconds=60, process_existing=True, now=100) == []
    assert WATCH.ready_candidates([candidate], state, settle_seconds=60, process_existing=True, now=159) == []
    assert WATCH.ready_candidates([candidate], state, settle_seconds=60, process_existing=True, now=160) == [candidate]


def test_existing_files_can_be_baselined_without_running() -> None:
    state = WATCH.load_state(Path("missing-state.json"))
    candidate = _candidate()

    assert WATCH.ready_candidates([candidate], state, settle_seconds=60, process_existing=False, now=100) == []
    assert candidate.fingerprint in state["completed"]
    assert state["completed"][candidate.fingerprint]["status"] == "ignored_existing"


def test_failed_file_retries_after_delay_and_stops_at_limit() -> None:
    candidate = _candidate()
    state = {
        "$schema": WATCH.STATE_SCHEMA,
        "initialized": True,
        "observations": {
            candidate.relative_path: {
                "fingerprint": candidate.fingerprint,
                "first_seen_epoch": 1,
            }
        },
        "completed": {},
        "failed": {
            candidate.fingerprint: {
                "attempts": 1,
                "max_attempts": 3,
                "next_retry_epoch": 200,
            }
        },
    }

    assert WATCH.ready_candidates([candidate], state, settle_seconds=10, process_existing=True, now=199) == []
    assert WATCH.ready_candidates([candidate], state, settle_seconds=10, process_existing=True, now=200) == [candidate]
    state["failed"][candidate.fingerprint]["attempts"] = 3
    assert WATCH.ready_candidates([candidate], state, settle_seconds=10, process_existing=True, now=300) == []


def test_scan_once_records_failure_for_retry(tmp_path: Path, monkeypatch) -> None:
    shared = tmp_path / "HB2A" / "IPTS-123" / "shared"
    incoming = shared / "incoming"
    output = shared / "results"
    incoming.mkdir(parents=True)
    output.mkdir()
    (incoming / "scan.dat").write_text("1 2\n", encoding="utf-8")
    (shared / "config.yaml").write_text("mode: rapid\n", encoding="utf-8")
    (shared / "profile.instprm").write_text("profile\n", encoding="utf-8")
    browser = FacilityBrowser(tmp_path)
    recipe = WatchRecipe(
        instrument="HB2A",
        ipts="IPTS-123",
        source_directory="shared/incoming",
        output_directory="shared/results",
        configuration="shared/config.yaml",
        instrument_profile="shared/profile.instprm",
        settle_seconds=10,
        retry_delay_seconds=30,
    )
    state_path = tmp_path / "state.json"
    candidate = browser.discover_watch_candidates(recipe)[0]
    state = WATCH.load_state(state_path)
    state["initialized"] = True
    state["observations"][candidate.relative_path] = {
        "fingerprint": candidate.fingerprint,
        "first_seen_epoch": 0,
    }
    WATCH.save_state(state_path, state)

    monkeypatch.setattr(WATCH.time, "time", lambda: 100.0)
    monkeypatch.setattr(WATCH, "analyze_candidate", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    summary = WATCH.scan_once(browser, recipe, state_path)

    assert summary["failed"] == 1
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    failure = saved["failed"][candidate.fingerprint]
    assert failure["attempts"] == 1
    assert failure["next_retry_epoch"] == 130.0


def test_failed_artifact_directories_never_overwrite(tmp_path: Path) -> None:
    first = tmp_path / "run-failed"
    first.mkdir()
    assert WATCH.next_available_path(first) == tmp_path / "run-failed-2"


def test_worker_lock_rejects_a_live_lease(tmp_path: Path) -> None:
    lock = tmp_path / "watch.lock"
    with WATCH.worker_lock(lock, stale_after_seconds=60, heartbeat_seconds=1):
        with pytest.raises(RuntimeError, match="Another watcher"):
            with WATCH.worker_lock(lock, stale_after_seconds=60, heartbeat_seconds=1):
                pass
    assert not lock.exists()


def test_worker_lock_recovers_a_stale_lease(tmp_path: Path) -> None:
    lock = tmp_path / "watch.lock"
    lock.write_text("abandoned\n", encoding="utf-8")
    os.utime(lock, (1, 1))

    with WATCH.worker_lock(lock, stale_after_seconds=1, heartbeat_seconds=1):
        assert lock.is_file()
    assert not lock.exists()
