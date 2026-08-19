"""Persistent, restart-safe RADAR-PD processing for an SNS IPTS watch recipe.

This worker is intentionally separate from the NOVA browser process. It can be
run once by a scheduler or continuously by an NDIP-managed service. Every
selected path is validated by ``FacilityBrowser`` before processing.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
NOVA_SRC = ROOT / "nova" / "src"
if str(NOVA_SRC) not in sys.path:
    sys.path.insert(0, str(NOVA_SRC))

from radar_pd_nova.facility import FacilityBrowser, WatchCandidate, WatchRecipe  # noqa: E402


STATE_SCHEMA = "radar-pd-watch-state/v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_recipe(path: Path, browser: FacilityBrowser) -> WatchRecipe:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("$schema") != "radar-pd-watch/v1":
        raise ValueError("Expected a radar-pd-watch/v1 JSON recipe")
    return WatchRecipe.from_dict(payload).validate(browser)


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "$schema": STATE_SCHEMA,
            "initialized": False,
            "observations": {},
            "completed": {},
            "failed": {},
            "updated_utc": utc_now(),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("$schema") != STATE_SCHEMA:
        raise ValueError("Watch state has an unsupported schema")
    for key in ("observations", "completed", "failed"):
        payload.setdefault(key, {})
    return payload


def save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_utc"] = utc_now()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def worker_lock(
    path: Path,
    *,
    stale_after_seconds: int = 300,
    heartbeat_seconds: int = 30,
) -> Iterator[None]:
    """Hold a recoverable exclusive lease on a watch state file.

    ``O_EXCL`` prevents concurrent acquisition. A heartbeat keeps the lease
    fresh while the worker is alive; a replacement may remove only a lock that
    has not been refreshed for ``stale_after_seconds``.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            break
        except FileExistsError as exc:
            try:
                age = max(0.0, time.time() - path.stat().st_mtime)
            except FileNotFoundError:
                continue
            if attempt == 0 and age >= stale_after_seconds:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                continue
            raise RuntimeError(f"Another watcher owns {path}; lease age is {age:.0f}s") from exc
    else:  # pragma: no cover - defensive, the loop either breaks or raises
        raise RuntimeError(f"Could not acquire watcher lease {path}")

    stopped = threading.Event()

    def heartbeat() -> None:
        while not stopped.wait(max(1, heartbeat_seconds)):
            try:
                path.touch(exist_ok=True)
            except OSError:
                return

    try:
        os.write(descriptor, f"pid={os.getpid()} started={utc_now()}\n".encode())
        os.close(descriptor)
        thread = threading.Thread(target=heartbeat, name="radar-watch-lock", daemon=True)
        thread.start()
        yield
    finally:
        stopped.set()
        if "thread" in locals():
            thread.join(timeout=max(2, heartbeat_seconds + 1))
        path.unlink(missing_ok=True)


def ready_candidates(
    candidates: list[WatchCandidate],
    state: dict[str, Any],
    *,
    settle_seconds: int,
    process_existing: bool,
    now: float | None = None,
) -> list[WatchCandidate]:
    current_time = time.time() if now is None else now
    observations = state["observations"]
    completed = state["completed"]
    failed = state["failed"]
    initial = not bool(state.get("initialized"))
    ready: list[WatchCandidate] = []
    for candidate in candidates:
        if candidate.fingerprint in completed:
            continue
        failure = failed.get(candidate.fingerprint) or {}
        if int(failure.get("attempts", 0)) >= int(failure.get("max_attempts", 1)):
            continue
        if float(failure.get("next_retry_epoch", 0)) > current_time:
            continue
        if initial and not process_existing:
            completed[candidate.fingerprint] = {
                "source": candidate.relative_path,
                "status": "ignored_existing",
                "recorded_utc": utc_now(),
            }
            continue
        observed = observations.get(candidate.relative_path)
        if not observed or observed.get("fingerprint") != candidate.fingerprint:
            observations[candidate.relative_path] = {
                "fingerprint": candidate.fingerprint,
                "first_seen_epoch": current_time,
                "size": candidate.size,
                "modified_ns": candidate.modified_ns,
            }
            continue
        if current_time - float(observed.get("first_seen_epoch", current_time)) >= settle_seconds:
            ready.append(candidate)
    state["initialized"] = True
    return ready


def safe_run_name(path: str, fingerprint: str) -> str:
    stem = Path(path).stem
    clean = "".join(character if character.isalnum() or character in "_.-" else "-" for character in stem)
    clean = clean.strip(".-") or "pattern"
    return f"{clean}-{fingerprint[:8]}"


def next_available_path(path: Path) -> Path:
    """Return ``path`` or a numbered sibling without overwriting evidence."""

    if not path.exists():
        return path
    for index in range(2, 10_000):
        candidate = path.with_name(f"{path.name}-{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a unique path beside {path}")


def analyze_candidate(
    browser: FacilityBrowser,
    recipe: WatchRecipe,
    candidate: WatchCandidate,
) -> Path:
    data = browser.resolve_file(recipe.instrument, recipe.ipts, candidate.relative_path, role="data")
    config = browser.resolve_file(recipe.instrument, recipe.ipts, recipe.configuration, role="config")
    instrument = (
        browser.resolve_file(recipe.instrument, recipe.ipts, recipe.instrument_profile, role="instrument")
        if recipe.instrument_profile
        else None
    )
    main_cif = (
        browser.resolve_file(recipe.instrument, recipe.ipts, recipe.main_cif, role="cif")
        if recipe.main_cif
        else None
    )
    output_parent = browser.resolve_directory(recipe.instrument, recipe.ipts, recipe.output_directory)
    run_name = safe_run_name(candidate.relative_path, candidate.fingerprint)
    destination = output_parent / run_name
    if destination.exists():
        raise FileExistsError(f"Watch result already exists: {destination}")
    staging = output_parent / f".{run_name}.processing"
    if staging.exists():
        raise RuntimeError(f"Incomplete staging directory already exists: {staging}")
    work = staging / "work"
    portal = staging / "results"
    staging.mkdir(mode=0o770)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "ndip_runner.py"),
        "analyze",
        "--config",
        str(config),
        "--data",
        str(data),
        "--mode",
        recipe.analysis_mode,
        "--run-name",
        run_name,
        "--input-source",
        "ipts",
        "--work-dir",
        str(work),
        "--output-dir",
        str(portal),
    ]
    if instrument:
        command.extend(("--instrument", str(instrument)))
    elif recipe.use_builtin_cuka:
        command.extend(("--instrument-preset", "xray_lab_cuka"))
    if main_cif:
        command.extend(("--main-cif", str(main_cif)))
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        (staging / "watch-console.log").write_text(
            completed.stdout + ("\n[stderr]\n" + completed.stderr if completed.stderr else ""),
            encoding="utf-8",
        )
        if completed.returncode:
            raise RuntimeError(f"RADAR-PD exited with code {completed.returncode}")
        shutil.rmtree(work, ignore_errors=True)
        manifest = {
            "$schema": "radar-pd-watch-result/v1",
            "source": candidate.relative_path,
            "source_fingerprint": candidate.fingerprint,
            "recipe": recipe.as_dict(),
            "completed_utc": utc_now(),
        }
        (staging / "watch-result.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.replace(destination)
        return destination
    except Exception:
        failed = next_available_path(output_parent / f"{run_name}-failed")
        if staging.exists():
            staging.replace(failed)
        raise


def scan_once(browser: FacilityBrowser, recipe: WatchRecipe, state_path: Path) -> dict[str, int]:
    state = load_state(state_path)
    candidates = browser.discover_watch_candidates(recipe)
    ready = ready_candidates(
        candidates,
        state,
        settle_seconds=recipe.settle_seconds,
        process_existing=recipe.process_existing,
    )
    summary = {"discovered": len(candidates), "ready": len(ready), "completed": 0, "failed": 0}
    for candidate in ready:
        prior_failure = state["failed"].get(candidate.fingerprint) or {}
        attempt = int(prior_failure.get("attempts", 0)) + 1
        try:
            destination = analyze_candidate(browser, recipe, candidate)
            state["completed"][candidate.fingerprint] = {
                "source": candidate.relative_path,
                "destination": str(destination),
                "completed_utc": utc_now(),
            }
            state["failed"].pop(candidate.fingerprint, None)
            summary["completed"] += 1
        except Exception as exc:
            state["failed"][candidate.fingerprint] = {
                "source": candidate.relative_path,
                "error": str(exc),
                "attempts": attempt,
                "max_attempts": recipe.max_attempts,
                "next_retry_epoch": time.time() + recipe.retry_delay_seconds,
                "failed_utc": utc_now(),
            }
            summary["failed"] += 1
        save_state(state_path, state)
    save_state(state_path, state)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--facility-root", default=os.getenv("RADAR_PD_FACILITY_ROOT", "/SNS"))
    parser.add_argument("--state", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    browser = FacilityBrowser(args.facility_root)
    recipe = load_recipe(args.recipe.resolve(), browser)
    source = browser.resolve_directory(recipe.instrument, recipe.ipts, recipe.source_directory)
    state_path = args.state.resolve() if args.state else source / ".radar-pd-watch-state.json"
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    with worker_lock(lock_path):
        while True:
            summary = scan_once(browser, recipe, state_path)
            print(json.dumps({"time": utc_now(), **summary}), flush=True)
            if args.once:
                return 1 if summary["failed"] else 0
            time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
