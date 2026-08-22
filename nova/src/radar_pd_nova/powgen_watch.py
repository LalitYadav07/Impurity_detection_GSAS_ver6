"""Read-only POWGEN discovery and Galaxy submission planning.

This module deliberately stops at a declarative submission plan.  The caller
is responsible for importing the selected read-only facility file into Galaxy
and submitting the existing Analyze tool.  Scientific code is never executed
by the watcher process.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping, Sequence

from .powgen import PowgenReducedFile, select_preferred_reduced_file


WATCH_STATE_SCHEMA = "radar-pd-powgen-watch-state/v1"
ANALYZE_TOOL_ID = "neutrons_radar_pd_analyze_prototype"
_IPTS_RE = re.compile(r"^IPTS-\d+$", re.IGNORECASE)
_FACILITY_ROOTS = frozenset({"SNS", "HFIR"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _nonempty(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{label} is required")
    return result


def _relative_subfolder(value: str) -> str:
    normalized = str(value or "").strip().replace("\\", "/").strip("/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("POWGEN watch subfolder must be a relative path without traversal")
    return path.as_posix()


@dataclass(frozen=True)
class WatchDefinition:
    """One read-only SNS POWGEN folder watched into one Galaxy history.

    Configuration references are existing Galaxy dataset identifiers.  The
    canonical ``config_refs`` mapping is also exposed so additional immutable
    Analyze references can be carried without expanding this contract.
    """

    ipts: str
    subfolder: str
    history_id: str
    configuration_ref: str | None = None
    instrument_profile_ref: str | None = None
    config_refs: Mapping[str, str] = field(default_factory=dict)
    facility: str = "SNS"
    instrument: str = "PG3"
    main_cif_ref: str | None = None

    def __post_init__(self) -> None:
        facility = str(self.facility).strip().upper()
        instrument = str(self.instrument).strip().upper()
        ipts = str(self.ipts).strip().upper()
        if facility != "SNS":
            raise ValueError("POWGEN watches are limited to the SNS facility")
        if instrument != "PG3":
            raise ValueError("POWGEN watches require the PG3 instrument")
        if not _IPTS_RE.fullmatch(ipts):
            raise ValueError("POWGEN watch IPTS must have the form IPTS-<number>")

        refs = {str(key): _nonempty(value, f"config_refs[{key!r}]") for key, value in self.config_refs.items()}
        configuration = self.configuration_ref or refs.get("configuration") or refs.get("config")
        instrument_profile = (
            self.instrument_profile_ref
            or refs.get("instrument_profile")
            or refs.get("instrument")
        )
        configuration = _nonempty(configuration, "configuration_ref")
        instrument_profile = _nonempty(instrument_profile, "instrument_profile_ref")
        canonical_refs = dict(refs)
        canonical_refs["configuration"] = configuration
        canonical_refs["instrument_profile"] = instrument_profile
        if self.main_cif_ref:
            canonical_refs["main_cif"] = _nonempty(self.main_cif_ref, "main_cif_ref")

        object.__setattr__(self, "facility", facility)
        object.__setattr__(self, "instrument", instrument)
        object.__setattr__(self, "ipts", ipts)
        object.__setattr__(self, "subfolder", _relative_subfolder(self.subfolder))
        object.__setattr__(self, "history_id", _nonempty(self.history_id, "history_id"))
        object.__setattr__(self, "configuration_ref", configuration)
        object.__setattr__(self, "instrument_profile_ref", instrument_profile)
        object.__setattr__(self, "config_refs", canonical_refs)

    @property
    def source_directory(self) -> str:
        return f"/SNS/PG3/{self.ipts}/{self.subfolder}"


@dataclass(frozen=True)
class WatchedRun:
    """One discovered PG3 reduced pattern and its stable identity."""

    run_number: int
    source_path: str
    fingerprint: str
    discovered_utc: str = field(default_factory=_utc_now)
    galaxy_job_id: str | None = None
    galaxy_result_ids: tuple[str, ...] = ()
    error: str | None = None
    submission_attempts: int = 0
    last_attempt_utc: str | None = None
    next_retry_utc: str | None = None
    scientific_summary: Mapping[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return f"PG3_{self.run_number}"

    @property
    def result_dataset_ids(self) -> tuple[str, ...]:
        """Compatibility name for the persisted Galaxy result identifiers."""

        return self.galaxy_result_ids

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["galaxy_result_ids"] = list(self.galaxy_result_ids)
        payload["scientific_summary"] = dict(self.scientific_summary)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WatchedRun":
        return cls(
            run_number=int(value["run_number"]),
            source_path=str(value["source_path"]),
            fingerprint=str(value["fingerprint"]),
            discovered_utc=str(value.get("discovered_utc") or _utc_now()),
            galaxy_job_id=(str(value["galaxy_job_id"]) if value.get("galaxy_job_id") else None),
            galaxy_result_ids=tuple(str(item) for item in value.get("galaxy_result_ids", ())),
            error=(str(value["error"]) if value.get("error") else None),
            submission_attempts=max(0, int(value.get("submission_attempts", 0))),
            last_attempt_utc=(str(value["last_attempt_utc"]) if value.get("last_attempt_utc") else None),
            next_retry_utc=(str(value["next_retry_utc"]) if value.get("next_retry_utc") else None),
            scientific_summary=(
                dict(value.get("scientific_summary") or {})
                if isinstance(value.get("scientific_summary") or {}, Mapping)
                else {}
            ),
        )


def _run_map_from_dict(value: Any, label: str) -> dict[str, WatchedRun]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Watch state {label!r} must be a JSON object")
    result: dict[str, WatchedRun] = {}
    for key, item in value.items():
        if not isinstance(item, Mapping):
            raise ValueError(f"Watch state run {key!r} must be a JSON object")
        run = WatchedRun.from_dict(item)
        if str(key) != run.run_id:
            raise ValueError(f"Watch state key {key!r} does not match {run.run_id!r}")
        result[run.run_id] = run
    return result


@dataclass
class WatchState:
    """Restart-safe lifecycle state whose identifiers point into Galaxy."""

    history_id: str
    discovered: dict[str, WatchedRun] = field(default_factory=dict)
    submitted: dict[str, WatchedRun] = field(default_factory=dict)
    completed: dict[str, WatchedRun] = field(default_factory=dict)
    failed: dict[str, WatchedRun] = field(default_factory=dict)
    initial_backfill_complete: bool = False
    updated_utc: str = field(default_factory=_utc_now)
    schema: str = WATCH_STATE_SCHEMA

    def __post_init__(self) -> None:
        self.history_id = _nonempty(self.history_id, "history_id")
        if self.schema != WATCH_STATE_SCHEMA:
            raise ValueError(f"Unsupported POWGEN watch state schema: {self.schema!r}")
        seen: set[str] = set()
        for phase in (self.discovered, self.submitted, self.completed, self.failed):
            overlap = seen.intersection(phase)
            if overlap:
                raise ValueError(f"Runs occur in more than one watch phase: {sorted(overlap)}")
            seen.update(phase)

    def contains(self, run: int | str | WatchedRun) -> bool:
        run_id = _coerce_run_id(run)
        return any(run_id in phase for phase in (self.discovered, self.submitted, self.completed, self.failed))

    def as_dict(self) -> dict[str, Any]:
        return {
            "$schema": self.schema,
            "history_id": self.history_id,
            "discovered": {key: run.as_dict() for key, run in sorted(self.discovered.items())},
            "submitted": {key: run.as_dict() for key, run in sorted(self.submitted.items())},
            "completed": {key: run.as_dict() for key, run in sorted(self.completed.items())},
            "failed": {key: run.as_dict() for key, run in sorted(self.failed.items())},
            "initial_backfill_complete": self.initial_backfill_complete,
            "updated_utc": self.updated_utc,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WatchState":
        if value.get("$schema") != WATCH_STATE_SCHEMA:
            raise ValueError("Expected a radar-pd-powgen-watch-state/v1 document")
        return cls(
            history_id=str(value.get("history_id") or ""),
            discovered=_run_map_from_dict(value.get("discovered", {}), "discovered"),
            submitted=_run_map_from_dict(value.get("submitted", {}), "submitted"),
            completed=_run_map_from_dict(value.get("completed", {}), "completed"),
            failed=_run_map_from_dict(value.get("failed", {}), "failed"),
            # Checkpoints written before full initial backfill was introduced
            # deliberately omitted this field. Treat them as incomplete so a
            # restored watch discovers historical scans that were skipped by
            # the old newest-only startup policy.
            initial_backfill_complete=bool(value.get("initial_backfill_complete", False)),
            updated_utc=str(value.get("updated_utc") or _utc_now()),
            schema=str(value["$schema"]),
        )

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "WatchState":
        payload = json.loads(value)
        if not isinstance(payload, Mapping):
            raise ValueError("POWGEN watch state must be a JSON object")
        return cls.from_dict(payload)

    def mark_submitted(self, run: int | str | WatchedRun, galaxy_job_id: str) -> WatchedRun:
        run_id = _coerce_run_id(run)
        job_id = _nonempty(galaxy_job_id, "galaxy_job_id")
        existing = self.submitted.get(run_id) or self.completed.get(run_id)
        if existing is not None:
            if existing.galaxy_job_id != job_id:
                raise ValueError(f"{run_id} is already associated with another Galaxy job")
            return existing
        if run_id in self.failed:
            raise ValueError(f"{run_id} is terminally failed")
        discovered = self.discovered.pop(run_id, None)
        if discovered is None:
            raise KeyError(f"{run_id} has not been discovered")
        submitted = _replace_run(
            discovered,
            galaxy_job_id=job_id,
            error=None,
            next_retry_utc=None,
        )
        self.submitted[run_id] = submitted
        self.updated_utc = _utc_now()
        return submitted

    def defer_submission(
        self,
        run: int | str | WatchedRun,
        error: str,
        *,
        retry_after_seconds: int,
    ) -> WatchedRun:
        """Keep a pre-acknowledgement failure retryable and checkpoint it."""

        run_id = _coerce_run_id(run)
        message = _nonempty(error, "error")
        if retry_after_seconds < 0:
            raise ValueError("retry_after_seconds must not be negative")
        discovered = self.discovered.get(run_id)
        if discovered is None:
            raise KeyError(f"{run_id} is not awaiting submission")
        attempted_at = datetime.now(timezone.utc)
        retry_at = attempted_at.timestamp() + retry_after_seconds
        pending = _replace_run(
            discovered,
            error=message,
            submission_attempts=discovered.submission_attempts + 1,
            last_attempt_utc=attempted_at.isoformat(),
            next_retry_utc=datetime.fromtimestamp(retry_at, timezone.utc).isoformat(),
        )
        self.discovered[run_id] = pending
        self.updated_utc = _utc_now()
        return pending

    def retryable_runs(self, *, now: datetime | None = None) -> list[WatchedRun]:
        """Return pending discoveries whose retry delay has elapsed."""

        current = now or datetime.now(timezone.utc)
        due: list[WatchedRun] = []
        for run in self.discovered.values():
            if not run.next_retry_utc:
                due.append(run)
                continue
            try:
                retry_at = datetime.fromisoformat(run.next_retry_utc.replace("Z", "+00:00"))
            except ValueError:
                due.append(run)
                continue
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            if retry_at <= current:
                due.append(run)
        return sorted(due, key=lambda item: item.run_number)

    def mark_completed(
        self,
        run: int | str | WatchedRun,
        galaxy_result_ids: Sequence[str],
        *,
        galaxy_job_id: str | None = None,
        scientific_summary: Mapping[str, Any] | None = None,
    ) -> WatchedRun:
        run_id = _coerce_run_id(run)
        result_ids = tuple(_nonempty(item, "galaxy_result_id") for item in galaxy_result_ids)
        if not result_ids:
            raise ValueError("At least one Galaxy result identifier is required")
        existing = self.completed.get(run_id)
        if existing is not None:
            expected_job = galaxy_job_id or existing.galaxy_job_id
            if existing.galaxy_job_id != expected_job or existing.galaxy_result_ids != result_ids:
                raise ValueError(f"{run_id} is already completed with different Galaxy identifiers")
            if scientific_summary and dict(existing.scientific_summary) != dict(scientific_summary):
                return self.set_scientific_summary(run_id, scientific_summary)
            return existing
        submitted = self.submitted.pop(run_id, None)
        if submitted is None:
            raise KeyError(f"{run_id} has not been submitted")
        if galaxy_job_id and submitted.galaxy_job_id != galaxy_job_id:
            raise ValueError(f"{run_id} Galaxy job identifier does not match its submission")
        completed = _replace_run(
            submitted,
            galaxy_result_ids=result_ids,
            scientific_summary=dict(scientific_summary or {}),
        )
        self.completed[run_id] = completed
        self.updated_utc = _utc_now()
        return completed

    def set_scientific_summary(
        self,
        run: int | str | WatchedRun,
        summary: Mapping[str, Any],
    ) -> WatchedRun:
        """Attach a compact, JSON-safe result summary to a completed scan."""

        run_id = _coerce_run_id(run)
        completed = self.completed.get(run_id)
        if completed is None:
            raise KeyError(f"{run_id} is not completed")
        updated = _replace_run(completed, scientific_summary=dict(summary))
        self.completed[run_id] = updated
        self.updated_utc = _utc_now()
        return updated

    def mark_failed(
        self,
        run: int | str | WatchedRun,
        error: str,
        *,
        galaxy_job_id: str | None = None,
    ) -> WatchedRun:
        run_id = _coerce_run_id(run)
        message = _nonempty(error, "error")
        existing = self.failed.get(run_id)
        if existing is not None:
            if existing.error != message or (galaxy_job_id and existing.galaxy_job_id != galaxy_job_id):
                raise ValueError(f"{run_id} is already failed with different details")
            return existing
        source = self.submitted.pop(run_id, None) or self.discovered.pop(run_id, None)
        if source is None:
            raise KeyError(f"{run_id} is not active")
        if galaxy_job_id and source.galaxy_job_id and source.galaxy_job_id != galaxy_job_id:
            raise ValueError(f"{run_id} Galaxy job identifier does not match its submission")
        failed = _replace_run(source, galaxy_job_id=galaxy_job_id or source.galaxy_job_id, error=message)
        self.failed[run_id] = failed
        self.updated_utc = _utc_now()
        return failed


def _replace_run(run: WatchedRun, **changes: Any) -> WatchedRun:
    payload = run.as_dict()
    payload.update(changes)
    return WatchedRun.from_dict(payload)


def _coerce_run_id(value: int | str | WatchedRun) -> str:
    if isinstance(value, WatchedRun):
        return value.run_id
    if isinstance(value, int):
        return f"PG3_{value}"
    text = str(value).strip().upper()
    return text if text.startswith("PG3_") else f"PG3_{int(text)}"


def _listing_path(entry: str | Path | Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    if isinstance(entry, Mapping):
        if entry.get("is_dir") is True or str(entry.get("kind", "")).lower() == "directory":
            return "", entry
        raw = entry.get("path") or entry.get("absolute_path") or entry.get("relative_path") or entry.get("name")
        return str(raw or ""), entry
    return str(entry), {}


def _path_in_watch(definition: WatchDefinition, raw_path: str) -> str | None:
    text = raw_path.strip().replace("\\", "/")
    if not text:
        return None
    source = PurePosixPath(definition.source_directory)
    path = PurePosixPath(text)
    if path.is_absolute():
        candidate = path
    elif len(path.parts) == 1:
        candidate = source / path
    else:
        relative_source = PurePosixPath(definition.subfolder)
        try:
            relative = path.relative_to(relative_source)
        except ValueError:
            return None
        candidate = source / relative
    if any(part in {"", ".", ".."} for part in candidate.parts[1:]):
        return None
    try:
        candidate.relative_to(source)
    except ValueError:
        return None
    return candidate.as_posix()


def _fingerprint(path: str, metadata: Mapping[str, Any]) -> str:
    observations = {
        str(key): metadata[key]
        for key in ("id", "size", "mtime", "modified", "modified_ns", "etag", "sha256")
        if key in metadata
    }
    encoded = json.dumps({"path": path, "observations": observations}, sort_keys=True, default=str).encode()
    return sha256(encoded).hexdigest()


def discover_from_listing(
    definition: WatchDefinition,
    file_listing: Iterable[str | Path | Mapping[str, Any]],
    state: WatchState | None = None,
    *,
    minimum_run_number: int | None = None,
    newest_limit: int | None = None,
) -> list[WatchedRun]:
    """Discover supported PG3 files using only the caller-supplied listing.

    At most one preferred reduced file is selected per run.  Runs already in
    any state phase are omitted, making repeated discovery idempotent.
    """

    if state is not None and state.history_id != definition.history_id:
        raise ValueError("Watch definition and state refer to different Galaxy histories")
    if minimum_run_number is not None and minimum_run_number < 0:
        raise ValueError("minimum_run_number must be non-negative")
    if newest_limit is not None and newest_limit < 1:
        raise ValueError("newest_limit must be positive")

    by_run: dict[int, list[tuple[PowgenReducedFile, Mapping[str, Any]]]] = {}
    for entry in file_listing:
        raw_path, metadata = _listing_path(entry)
        candidate_path = _path_in_watch(definition, raw_path)
        if candidate_path is None:
            continue
        try:
            parsed = select_preferred_reduced_file([candidate_path])
        except ValueError:
            continue
        if minimum_run_number is not None and parsed.run_number < minimum_run_number:
            continue
        by_run.setdefault(parsed.run_number, []).append((parsed, metadata))

    discovered: list[WatchedRun] = []
    run_numbers = sorted(by_run)
    if newest_limit is not None:
        run_numbers = run_numbers[-newest_limit:]
    for run_number in run_numbers:
        candidates = by_run[run_number]
        preferred = select_preferred_reduced_file([item.path for item, _ in candidates], run_number=run_number)
        metadata = next(metadata for item, metadata in candidates if item.path == preferred.path)
        run = WatchedRun(
            run_number=run_number,
            source_path=preferred.path,
            fingerprint=_fingerprint(preferred.path, metadata),
        )
        if state is not None:
            if state.contains(run):
                continue
            state.discovered[run.run_id] = run
            state.updated_utc = _utc_now()
        discovered.append(run)
    return discovered


discover_runs = discover_from_listing


@dataclass(frozen=True)
class GalaxySubmissionPlan:
    """Declarative request for the deployed Galaxy Analyze tool."""

    history_id: str
    run_id: str
    idempotency_key: str
    source_path: str
    tool_id: str
    imports: Mapping[str, Mapping[str, Any]]
    inputs: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_submission_plan(
    definition: WatchDefinition,
    run: WatchedRun,
    state: WatchState | None = None,
) -> GalaxySubmissionPlan | None:
    """Build an Analyze plan, or return ``None`` after prior submission.

    The plan imports the read-only facility object into Galaxy History first.
    It intentionally contains no command, executable, work directory, or
    experiment output path.
    """

    if state is not None:
        if state.history_id != definition.history_id:
            raise ValueError("Watch definition and state refer to different Galaxy histories")
        if run.run_id in state.submitted or run.run_id in state.completed or run.run_id in state.failed:
            return None
        known = state.discovered.get(run.run_id)
        if known is None or known.fingerprint != run.fingerprint:
            raise ValueError(f"{run.run_id} is not the discovered watch input")
    source = _path_in_watch(definition, run.source_path)
    if source != run.source_path:
        raise ValueError("Run source is outside the configured read-only POWGEN folder")

    key_material = f"{definition.history_id}\0{run.run_id}\0{run.fingerprint}".encode()
    idempotency_key = sha256(key_material).hexdigest()
    inputs: dict[str, Any] = {
        "data_inputs|input_source|source_kind": "history",
        "data_inputs|input_source|diffraction_pattern": {"dataset_ref": "diffraction_pattern"},
        "data_inputs|input_source|instrument_source|kind": "uploaded",
        "data_inputs|input_source|instrument_source|instrument_file": definition.instrument_profile_ref,
        "reproducibility|configuration_override|config_kind": "existing",
        "reproducibility|configuration_override|config_file": definition.configuration_ref,
        "reproducibility|run_name": run.run_id,
    }
    if definition.main_cif_ref:
        inputs["data_inputs|main_cif"] = definition.main_cif_ref
    return GalaxySubmissionPlan(
        history_id=definition.history_id,
        run_id=run.run_id,
        idempotency_key=idempotency_key,
        source_path=run.source_path,
        tool_id=ANALYZE_TOOL_ID,
        imports={
            "diffraction_pattern": {
                "source_path": run.source_path,
                "destination": "galaxy_history",
                "read_only": True,
            }
        },
        inputs=inputs,
    )


submission_plan = build_submission_plan


def validate_output_path(path: str | os.PathLike[str]) -> Path:
    """Reject watcher-owned files in SNS/HFIR experiment trees."""

    raw = os.fspath(path).strip().replace("\\", "/")
    normalized = "/" + raw.lstrip("/")
    first = PurePosixPath(normalized).parts[1].upper() if len(PurePosixPath(normalized).parts) > 1 else ""
    if first in _FACILITY_ROOTS:
        raise ValueError("Watcher state/output paths must not be under /SNS or /HFIR")
    resolved = Path(path).expanduser().resolve(strict=False)
    parts = [part.upper() for part in resolved.parts]
    if len(parts) > 1 and parts[1] in _FACILITY_ROOTS:
        raise ValueError("Watcher state/output paths must not be under /SNS or /HFIR")
    return resolved


def save_watch_state(path: str | os.PathLike[str], state: WatchState) -> Path:
    destination = validate_output_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(state.to_json(), encoding="utf-8")
    temporary.replace(destination)
    return destination


def load_watch_state(path: str | os.PathLike[str], *, history_id: str | None = None) -> WatchState:
    source = validate_output_path(path)
    if not source.is_file():
        if history_id is None:
            raise FileNotFoundError(source)
        return WatchState(history_id=history_id)
    return WatchState.from_json(source.read_text(encoding="utf-8"))


save_state = save_watch_state
load_state = load_watch_state


__all__ = [
    "ANALYZE_TOOL_ID",
    "GalaxySubmissionPlan",
    "WATCH_STATE_SCHEMA",
    "WatchDefinition",
    "WatchState",
    "WatchedRun",
    "build_submission_plan",
    "discover_from_listing",
    "discover_runs",
    "load_state",
    "load_watch_state",
    "save_state",
    "save_watch_state",
    "submission_plan",
    "validate_output_path",
]
