"""Versioned contracts and small utilities for the NDIP/Galaxy adapter.

This module deliberately has no Galaxy dependency.  The hosted application,
the script API and NDIP can all consume the same documents.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


INPUT_SCHEMA = "radar-pd-input/v1"
CONFIG_SCHEMA = "radar-pd-config/v1"
STATE_SCHEMA = "radar-pd-state/v1"
RESULT_SCHEMA = "radar-pd-result/v1"
GPX_INDEX_SCHEMA = "radar-pd-gpx-index/v1"
LIBRARY_SCHEMA = "radar-pd-library/v1"

STAGE_STATES = {"pending", "running", "complete", "partial", "failed", "skipped"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_name(value: object, fallback: str = "radar_run") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._-")
    return text or fallback


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.write("\n")
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return destination


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def validate_stage_state(value: object) -> str:
    state = str(value or "").strip().lower()
    if state not in STAGE_STATES:
        raise ValueError(f"Unsupported NDIP stage state: {value!r}")
    return state


def file_record(path: str | Path, *, role: str, root: str | Path | None = None) -> dict[str, Any]:
    item = Path(path)
    if root is not None:
        try:
            display_path = item.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            display_path = item.name
    else:
        display_path = item.name
    record: dict[str, Any] = {
        "role": role,
        "name": item.name,
        "path": display_path,
        "size_bytes": item.stat().st_size,
        "sha256": sha256_file(item),
    }
    return record


def input_manifest(
    *,
    run_name: str,
    mode: str,
    data: str | Path,
    instrument: str | Path,
    main_cif: str | Path | None = None,
    source: str = "history",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    inputs = [file_record(data, role="diffraction_data"), file_record(instrument, role="instrument_profile")]
    if main_cif:
        inputs.append(file_record(main_cif, role="main_phase_cif"))
    return {
        "$schema": INPUT_SCHEMA,
        "created_utc": utc_now(),
        "run_name": safe_name(run_name),
        "analysis_mode": mode,
        "source": source,
        "inputs": inputs,
        "metadata": dict(metadata or {}),
    }


def initial_state(*, run_name: str, mode: str, input_doc: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "$schema": STATE_SCHEMA,
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "run_name": safe_name(run_name),
        "analysis_mode": mode,
        "status": "pending",
        "input": dict(input_doc),
        "stages": {},
        "artifacts": {},
        "warnings": [],
        "errors": [],
    }


def update_state(
    state: Mapping[str, Any],
    *,
    status: str | None = None,
    stage: str | None = None,
    stage_status: str | None = None,
    message: str | None = None,
    error: str | None = None,
    artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    updated = json.loads(json.dumps(state, default=str))
    updated["updated_utc"] = utc_now()
    if status is not None:
        updated["status"] = validate_stage_state(status)
    if stage is not None:
        stages = updated.setdefault("stages", {})
        block = stages.setdefault(stage, {})
        if stage_status is not None:
            block["status"] = validate_stage_state(stage_status)
        if message:
            block["message"] = message
        block["updated_utc"] = utc_now()
    if error:
        updated.setdefault("errors", []).append({"time": utc_now(), "message": error, "stage": stage})
    if artifacts:
        updated.setdefault("artifacts", {}).update(dict(artifacts))
    return updated


def relative_artifact_records(paths: Iterable[Path], *, root: Path, role: str) -> list[dict[str, Any]]:
    return [file_record(path, role=role, root=root) for path in sorted(paths)]


__all__ = [
    "CONFIG_SCHEMA",
    "GPX_INDEX_SCHEMA",
    "INPUT_SCHEMA",
    "LIBRARY_SCHEMA",
    "RESULT_SCHEMA",
    "STATE_SCHEMA",
    "atomic_write_json",
    "file_record",
    "initial_state",
    "input_manifest",
    "read_json",
    "relative_artifact_records",
    "safe_name",
    "sha256_file",
    "update_state",
    "utc_now",
]
