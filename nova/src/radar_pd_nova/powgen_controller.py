"""POWGEN live-experiment orchestration around the existing Analyze tool.

The controller owns no scientific implementation.  It discovers reduced PG3
patterns from a read-only experiment directory, resolves a verified official
instrument profile, and submits the same ``AnalysisConfig`` and
``InputSelection`` objects used by an ordinary NOVA run.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import threading
import time
from typing import Any, Iterable, Mapping

from .galaxy_service import GalaxyService
from .models import AnalysisConfig, AnalysisMode, InputSelection, InputSource, RunRecord, RunStatus
from .powgen import (
    PowgenProfileResolution,
    resolve_packaged_powgen_profile_path,
    resolve_powgen_profile,
)
from .powgen_watch import (
    WatchDefinition,
    WatchState,
    WatchedRun,
    build_submission_plan,
    discover_from_listing,
)
from .powgen_metadata import read_powgen_scan_metadata
from .results import experiment_scan_summary


@dataclass(frozen=True)
class PowgenExperimentSettings:
    """User-controlled settings for one read-only POWGEN experiment watch."""

    ipts: str
    history_id: str
    configuration_dataset_id: str
    wavelength_angstrom: str
    frequency_hz: str = "60"
    subfolder: str = "shared/autoreduce"
    main_cif_dataset_id: str | None = None
    database_dataset_id: str | None = None
    initial_backfill_limit: int | None = None
    late_arrival_window: int = 100
    max_active_jobs: int = 5
    max_submission_attempts: int = 5
    retry_base_seconds: int = 15
    retry_max_seconds: int = 300

    def __post_init__(self) -> None:
        if self.initial_backfill_limit is not None and self.initial_backfill_limit < 0:
            raise ValueError("initial_backfill_limit must not be negative")
        if self.late_arrival_window < 0:
            raise ValueError("late_arrival_window must not be negative")
        if self.max_active_jobs < 1:
            raise ValueError("max_active_jobs must be positive")
        if self.max_submission_attempts < 1:
            raise ValueError("max_submission_attempts must be positive")
        if self.retry_base_seconds < 0:
            raise ValueError("retry_base_seconds must not be negative")
        if self.retry_max_seconds < self.retry_base_seconds:
            raise ValueError("retry_max_seconds must be at least retry_base_seconds")

    def definition(self, instrument_profile_ref: str = "packaged-official-profile") -> WatchDefinition:
        config_refs = {}
        if self.database_dataset_id:
            config_refs["candidate_library"] = self.database_dataset_id
        return WatchDefinition(
            facility="SNS",
            instrument="PG3",
            ipts=self.ipts,
            subfolder=self.subfolder,
            history_id=self.history_id,
            configuration_ref=self.configuration_dataset_id,
            instrument_profile_ref=instrument_profile_ref,
            main_cif_ref=self.main_cif_dataset_id,
            config_refs=config_refs,
        )


def bounded_directory_listing(directory: str | Path, *, max_entries: int = 5000) -> list[dict[str, Any]]:
    """List one directory without recursion, following links, or file reads."""

    if max_entries < 1:
        raise ValueError("max_entries must be positive")
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"POWGEN reduced-data directory does not exist: {root}")
    rows: list[dict[str, Any]] = []
    with _scandir(root) as entries:
        for index, entry in enumerate(entries):
            if index >= max_entries:
                raise RuntimeError(
                    f"POWGEN directory contains more than {max_entries} entries; narrow the watch folder"
                )
            if not entry.is_file(follow_symlinks=False):
                continue
            stat = entry.stat(follow_symlinks=False)
            rows.append(
                {
                    "path": str(root / entry.name).replace("\\", "/"),
                    "size": int(stat.st_size),
                    "modified_ns": int(stat.st_mtime_ns),
                }
            )
    return rows


_SUPPORTED_POWGEN_WAVELENGTHS = (0.8, 1.5, 2.665)


def _metadata_wavelength(metadata: Mapping[str, Any]) -> str | None:
    metric = metadata.get("wavelength")
    if not isinstance(metric, Mapping) or metric.get("value") is None:
        return None
    unit = re.sub(r"[^a-z]", "", str(metric.get("unit") or "").lower())
    if unit not in {"a", "angstrom", "angstroms"}:
        return None
    observed = float(metric["value"])
    nearest = min(_SUPPORTED_POWGEN_WAVELENGTHS, key=lambda value: abs(value - observed))
    return f"{nearest:g}" if abs(nearest - observed) <= 0.05 else None


def preflight_powgen_experiment(
    ipts: str,
    *,
    requested_wavelength: str | None = None,
) -> dict[str, Any]:
    """Inspect one POWGEN experiment without submitting jobs or writing files."""

    definition = WatchDefinition(
        ipts=ipts,
        subfolder="shared/autoreduce",
        history_id="preflight",
        configuration_ref="preflight-configuration",
        instrument_profile_ref="preflight-profile",
    )
    listing = bounded_directory_listing(definition.source_directory)
    completed_rows = [
        row
        for row in listing
        if str(row.get("path") or "").lower().endswith(".gsa")
    ]
    runs = discover_from_listing(definition, completed_rows)
    if not runs:
        return {
            "ready": False,
            "source_directory": definition.source_directory,
            "scan_count": 0,
            "message": "The directory is readable, but it contains no completed POWGEN .gsa reductions.",
        }

    latest = runs[-1]
    metadata = read_powgen_scan_metadata(definition.ipts, latest.run_number)
    detected_wavelength = _metadata_wavelength(metadata)
    selected_wavelength = detected_wavelength or str(requested_wavelength or "").strip()
    profile_filename = ""
    profile_rule = ""
    profile_error = ""
    if not selected_wavelength:
        profile_error = "Select a POWGEN wavelength because the latest NeXus file does not report one."
    else:
        try:
            resolution = resolve_powgen_profile(
                run_number=latest.run_number,
                wavelength_angstrom=selected_wavelength,
                frequency_hz="60",
            )
            resolve_packaged_powgen_profile_path(resolution)
            profile_filename = resolution.profile_filename
            profile_rule = resolution.provenance.rule_id
        except (OSError, ValueError) as exc:
            profile_error = str(exc)

    sample_values = [
        value
        for value in (
            f"ID {metadata.get('sample_id')}" if metadata.get("sample_id") else "",
            str(metadata.get("sample_name") or "").strip(),
            str(metadata.get("sample_formula") or "").strip(),
        )
        if value
    ]
    temperature = metadata.get("temperature")
    temperature_display = ""
    if isinstance(temperature, Mapping) and temperature.get("value") is not None:
        unit = str(temperature.get("unit") or "").strip()
        temperature_display = f"{float(temperature['value']):g} {unit}".strip()

    return {
        "ready": not profile_error,
        "source_directory": definition.source_directory,
        "scan_count": len(runs),
        "first_run": runs[0].run_id,
        "latest_run": latest.run_id,
        "latest_file": Path(latest.source_path).name,
        "latest_start_time": str(metadata.get("start_time") or ""),
        "sample_display": " | ".join(sample_values) or "Not reported",
        "temperature_display": temperature_display or "Not reported",
        "metadata_available": bool(metadata.get("available")),
        "detected_wavelength": detected_wavelength or "",
        "selected_wavelength": selected_wavelength,
        "wavelength_source": (
            f"Detected from {latest.run_id} NeXus metadata"
            if detected_wavelength
            else "Selected manually; wavelength was not available from NeXus metadata"
        ),
        "profile_filename": profile_filename,
        "profile_rule": profile_rule,
        "profile_error": profile_error,
        "message": (
            f"Ready: {len(runs)} completed reductions from {runs[0].run_id} through {latest.run_id}."
            if not profile_error
            else f"Experiment found, but instrument setup is not ready: {profile_error}"
        ),
    }


def _scandir(path: Path):
    import os

    return os.scandir(path)


class PowgenWatchController:
    """Coordinate discovery, submission, and Galaxy-owned watch state."""

    def __init__(self, service: GalaxyService, settings: PowgenExperimentSettings) -> None:
        if service.history_id != settings.history_id:
            raise ValueError("POWGEN settings must target the active Galaxy history")
        self.service = service
        self.settings = settings
        self.definition = settings.definition()
        self.state = WatchState(history_id=settings.history_id)
        self.records: dict[str, RunRecord] = {}
        self.state_dataset_ids: list[str] = []
        self._primed = False
        self._last_seen_run_number: int | None = None
        self._startup_run_floor: int | None = None
        self._last_persisted_state: str | None = None
        self._configuration: AnalysisConfig | None = None
        self._configuration_lock = threading.Lock()
        self._profile_dataset_ids: dict[str, str] = {}
        self._profile_upload_lock = threading.Lock()
        self._persist_lock = threading.Lock()
        self._recent_runs_reconciled = False
        self.refresh_errors: dict[str, str] = {}

    def restore_latest_state(self) -> bool:
        """Recover the newest matching checkpoint and acknowledged Galaxy jobs.

        Checkpoints are ordinary Galaxy datasets, so this remains recoverable
        after a NOVA browser refresh without writing anything into the IPTS.
        Pending discoveries and acknowledged Galaxy jobs are both restored.
        A stored Galaxy job ID is sufficient to poll a job directly, so
        recovery does not depend on the job still appearing in a bounded list
        of recent History jobs.
        """

        query = f"POWGEN watch state {self.settings.ipts}"
        datasets = self.service.search_history_datasets(
            query=query,
            limit=25,
            include_generated=True,
        )
        restored_dataset_id = ""
        restored: WatchState | None = None
        for row in datasets:
            dataset_id = str(row.get("id") or "")
            if not dataset_id:
                continue
            try:
                payload = self.service.load_json_document(dataset_id)
                watch = payload.get("watch") if isinstance(payload, Mapping) else None
                if isinstance(watch, Mapping):
                    if str(watch.get("ipts") or "").upper() != self.settings.ipts.upper():
                        continue
                    if str(watch.get("configuration_dataset_id") or "") != self.settings.configuration_dataset_id:
                        continue
                    if str(watch.get("wavelength_angstrom") or "") != self.settings.wavelength_angstrom:
                        continue
                    if str(watch.get("database_dataset_id") or "") != str(
                        self.settings.database_dataset_id or ""
                    ):
                        continue
                    stored_limit = (
                        watch.get("initial_backfill_limit")
                        if "initial_backfill_limit" in watch
                        else None
                    )
                    if stored_limit != self.settings.initial_backfill_limit:
                        continue
                candidate = WatchState.from_dict(payload)
                if candidate.history_id != self.settings.history_id:
                    continue
            except (OSError, TypeError, ValueError):
                continue
            restored = candidate
            restored_dataset_id = dataset_id
            break
        if restored is None:
            return False

        self.state = restored
        self._hydrate_missing_metadata()
        all_runs = [
            *restored.discovered.values(),
            *restored.submitted.values(),
            *restored.completed.values(),
            *restored.failed.values(),
        ]
        if all_runs:
            self._last_seen_run_number = max(run.run_number for run in all_runs)
            self._startup_run_floor = self._last_seen_run_number
            self._primed = True

        for run_id, run in {**restored.submitted, **restored.completed}.items():
            if run.galaxy_job_id:
                self.records[run_id] = self._minimal_record(run)

        self.state_dataset_ids = [restored_dataset_id]
        self._last_persisted_state = self.state.to_json()
        return True

    @property
    def source_directory(self) -> str:
        return self.definition.source_directory

    def discover(self, listing: Iterable[str | Path | Mapping[str, Any]] | None = None) -> list[WatchedRun]:
        rows = list(listing) if listing is not None else bounded_directory_listing(self.source_directory)
        # POWGEN autoreduction can expose auxiliary text products before the
        # canonical GSAS file. Waiting for .gsa prevents submitting a partial
        # representation and gives GSAS-II the format it handles directly.
        completed_rows = [
            row
            for row in rows
            if str(row.get("path") or row.get("name") or row).lower().endswith(".gsa")
        ]
        if not self.state.initial_backfill_complete:
            available = discover_from_listing(self.definition, completed_rows)
            if available:
                self._startup_run_floor = max(run.run_number for run in available)
            limit = self.settings.initial_backfill_limit
            discovered = []
            if limit != 0:
                discovered = discover_from_listing(
                    self.definition,
                    completed_rows,
                    self.state,
                    newest_limit=limit,
                )
            discovered = self._attach_scan_metadata(discovered)
            known_runs = [
                *self.state.discovered.values(),
                *self.state.submitted.values(),
                *self.state.completed.values(),
                *self.state.failed.values(),
            ]
            if known_runs:
                self._last_seen_run_number = max(run.run_number for run in known_runs)
            elif self._startup_run_floor is not None:
                self._last_seen_run_number = self._startup_run_floor
            self.state.initial_backfill_complete = True
            self._primed = True
            self.persist_state()
            return discovered

        discovered = discover_from_listing(
            self.definition,
            completed_rows,
            self.state,
            minimum_run_number=max(
                1,
                (self._last_seen_run_number or 0) - self.settings.late_arrival_window,
                (
                    (self._startup_run_floor or 0) + 1
                    if self.settings.initial_backfill_limit is not None
                    else 1
                ),
            ),
        )
        discovered = self._attach_scan_metadata(discovered)
        if discovered:
            self._last_seen_run_number = max(
                self._last_seen_run_number or 0,
                *(run.run_number for run in discovered),
            )
        return discovered

    def _read_scan_metadata(self, run: WatchedRun) -> dict[str, Any]:
        """Read optional NeXus conditions without making discovery fragile."""

        try:
            return read_powgen_scan_metadata(self.settings.ipts, run.run_number)
        except (OSError, TypeError, ValueError):
            return {}

    def _attach_scan_metadata(self, runs: Iterable[WatchedRun]) -> list[WatchedRun]:
        enriched: list[WatchedRun] = []
        for run in runs:
            metadata = dict(run.scan_metadata or {}) or self._read_scan_metadata(run)
            updated = replace(run, scan_metadata=metadata) if metadata else run
            if run.run_id in self.state.discovered:
                self.state.discovered[run.run_id] = updated
            enriched.append(updated)
        return enriched

    def _hydrate_missing_metadata(self) -> None:
        """Backfill old checkpoints once when their matching NeXus is readable."""

        for phase in (
            self.state.discovered,
            self.state.submitted,
            self.state.completed,
            self.state.failed,
        ):
            for run_id, run in list(phase.items()):
                current = dict(run.scan_metadata or {})
                if int(current.get("schema_version") or 0) >= 3:
                    continue
                metadata = self._read_scan_metadata(run)
                if metadata:
                    phase[run_id] = replace(run, scan_metadata={**current, **metadata})

    def _minimal_record(self, run: WatchedRun) -> RunRecord:
        """Rebuild enough state to poll an acknowledged Galaxy job by ID."""

        job_id = str(run.galaxy_job_id or "").strip()
        if not job_id:
            raise ValueError(f"{run.run_id} has no acknowledged Galaxy job ID")
        return RunRecord(
            uid=job_id,
            galaxy_job_id=job_id,
            name=f"{self.settings.ipts}_{run.run_id}",
            mode=AnalysisMode.FULL,
            history_id=self.settings.history_id,
            status=RunStatus.QUEUED,
            analysis_status=RunStatus.QUEUED,
            stage="Recovering Galaxy job",
            progress=0,
        )

    def due_submissions(self) -> list[WatchedRun]:
        """Return ready runs without exceeding the active Galaxy-job cap."""

        available = max(0, self.settings.max_active_jobs - len(self.state.submitted))
        return self.state.retryable_runs()[:available]

    def defer_submission(
        self,
        run: WatchedRun,
        error: str,
        *,
        persist: bool = True,
    ) -> WatchedRun:
        """Retry a pre-acknowledgement failure with bounded backoff."""

        current = self.state.discovered.get(run.run_id)
        if current is None:
            raise KeyError(f"{run.run_id} is not awaiting submission")
        next_attempt = current.submission_attempts + 1
        if next_attempt >= self.settings.max_submission_attempts:
            failed = self.state.mark_failed(
                current,
                f"Submission failed after {next_attempt} attempts: {error}",
            )
            if persist:
                self.persist_state()
            return failed
        delay = min(
            self.settings.retry_max_seconds,
            self.settings.retry_base_seconds * (2 ** current.submission_attempts),
        )
        pending = self.state.defer_submission(
            current,
            error,
            retry_after_seconds=delay,
        )
        if persist:
            self.persist_state()
        return pending

    def resolve_profile(self, run: WatchedRun) -> tuple[PowgenProfileResolution, Path]:
        resolution = resolve_powgen_profile(
            run_number=run.run_number,
            wavelength_angstrom=self.settings.wavelength_angstrom,
            frequency_hz=self.settings.frequency_hz,
        )
        return resolution, resolve_packaged_powgen_profile_path(resolution)

    def _profile_dataset_id(self, profile_path: Path) -> str:
        """Upload each packaged profile once and reuse it for every matching scan."""

        key = str(profile_path.resolve())
        cached = self._profile_dataset_ids.get(key)
        if cached:
            return cached
        with self._profile_upload_lock:
            cached = self._profile_dataset_ids.get(key)
            if cached:
                return cached
            dataset_id = self.service.upload_document(profile_path, label="POWGEN instrument profile")
            self._profile_dataset_ids[key] = dataset_id
            return dataset_id

    def launch_submission(self, run: WatchedRun) -> RunRecord:
        """Create one Galaxy job without moving the watch-state checkpoint.

        This split lets the NOVA monitor launch independent scans concurrently
        while keeping the state transition and immutable Galaxy checkpoint on
        its event-loop thread.  ``submit`` remains the synchronous convenience
        wrapper used by tests and non-interactive callers.
        """

        plan = build_submission_plan(self.definition, run, self.state)
        if plan is None:
            existing = self.records.get(run.run_id)
            if existing is None:
                raise ValueError(f"{run.run_id} was already submitted outside this controller instance")
            return existing

        if self._configuration is None:
            with self._configuration_lock:
                if self._configuration is None:
                    self._configuration = self.service.load_configuration_dataset(
                        self.settings.configuration_dataset_id
                    )
        config = self._configuration
        config = config.model_copy(update={"run_name": f"{self.settings.ipts}_{run.run_id}"})
        _, profile_path = self.resolve_profile(run)
        profile_dataset_id = self._profile_dataset_id(profile_path)
        inputs = InputSelection(
            source=InputSource.UPLOAD,
            data_path=run.source_path,
            instrument_dataset_id=profile_dataset_id,
            main_cif_dataset_id=self.settings.main_cif_dataset_id,
            database_dataset_id=self.settings.database_dataset_id,
            facility_root="/SNS",
            instrument="PG3",
            ipts=self.settings.ipts,
            data_relative_path=f"{self.settings.subfolder}/{Path(run.source_path).name}",
        )
        snapshot = self.service.create_submission_snapshot(
            config,
            inputs,
            idempotency_token=plan.idempotency_key,
            output_profile="monitor",
            prepared_dataset_ids={"configuration": self.settings.configuration_dataset_id},
        )
        record = self.service.submit_snapshot(snapshot)
        if not record.galaxy_job_id:
            raise RuntimeError(f"Galaxy did not acknowledge {run.run_id}")
        return record

    def acknowledge_submission(
        self,
        run: WatchedRun,
        record: RunRecord,
        *,
        persist: bool = True,
    ) -> RunRecord:
        """Checkpoint one Galaxy-acknowledged scan in the watch state."""

        if not record.galaxy_job_id:
            raise RuntimeError(f"Galaxy did not acknowledge {run.run_id}")
        self.state.mark_submitted(run, record.galaxy_job_id)
        self.records[run.run_id] = record
        if persist:
            self.persist_state()
        return record

    @staticmethod
    def _parse_timestamp(value: str) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)

    def _record_matches_watch(self, record: RunRecord, run: WatchedRun) -> bool:
        """Reject similarly named jobs belonging to another monitor configuration."""

        expected_name = f"{self.settings.ipts}_{run.run_id}"
        if record.name.strip().upper() != expected_name.upper():
            return False

        created = self._parse_timestamp(record.created_utc)
        discovered = self._parse_timestamp(run.discovered_utc)
        if created is not None and discovered is not None and created < discovered - timedelta(minutes=5):
            return False

        inputs = record.inputs
        candidate_database = (
            str(inputs.database_dataset_id or "")
            if inputs is not None
            else str(record.input_dataset_ids.get("database_archive") or "")
        )
        candidate_main = (
            str(inputs.main_cif_dataset_id or "")
            if inputs is not None
            else str(record.input_dataset_ids.get("main_cif") or "")
        )
        return candidate_database == str(self.settings.database_dataset_id or "") and candidate_main == str(
            self.settings.main_cif_dataset_id or ""
        )

    def reconcile_recent_runs(self) -> int:
        """Recover jobs that Galaxy accepted after a checkpoint write was lost.

        Galaxy jobs and their outputs are authoritative. The immutable watch
        checkpoint is a restart accelerator, but a transient checkpoint upload
        must never make a completed scan disappear from the experiment view.
        """

        if self._recent_runs_reconciled:
            return 0
        active = {**self.state.discovered, **self.state.submitted}
        if not active:
            self._recent_runs_reconciled = True
            return 0

        recovered = self.service.recent_runs(limit=max(50, min(250, len(active) * 4)))
        candidates: dict[str, RunRecord] = {}
        pattern = re.compile(rf"^{re.escape(self.settings.ipts)}_(PG3_\d+)$", re.IGNORECASE)
        for record in recovered:
            match = pattern.fullmatch(record.name.strip())
            if match is None:
                continue
            run_id = match.group(1).upper()
            run = active.get(run_id)
            if run is None or not self._record_matches_watch(record, run):
                continue
            acknowledged = self.state.submitted.get(run_id)
            if acknowledged is not None and acknowledged.galaxy_job_id != record.galaxy_job_id:
                continue
            previous = candidates.get(run_id)
            if previous is None or record.updated_utc > previous.updated_utc:
                candidates[run_id] = record

        recovered_count = 0
        state_changed = False
        for run_id, record in candidates.items():
            run = self.state.discovered.get(run_id) or self.state.submitted.get(run_id)
            if run is None or not record.galaxy_job_id:
                continue
            run_recovered = False
            if run_id in self.state.discovered:
                self.state.mark_submitted(run, record.galaxy_job_id)
                run_recovered = True
                state_changed = True
            self.records[run_id] = record

            if record.status == RunStatus.OK:
                result_ids = tuple(str(value) for value in record.output_dataset_ids.values() if value)
                if result_ids:
                    try:
                        summary = self._scientific_summary(record)
                    except Exception as exc:
                        summary = {}
                        self.refresh_errors[run_id] = f"Scientific summary pending: {exc}"
                    self.state.mark_completed(
                        run_id,
                        result_ids,
                        galaxy_job_id=record.galaxy_job_id,
                        scientific_summary=summary,
                    )
                    run_recovered = True
                    state_changed = True
            elif record.status in {RunStatus.ERROR, RunStatus.CANCELLED}:
                self.state.mark_failed(
                    run_id,
                    record.message or f"Galaxy ended with status {record.status.value}",
                    galaxy_job_id=record.galaxy_job_id,
                )
                run_recovered = True
                state_changed = True
            if run_recovered:
                recovered_count += 1

        self._recent_runs_reconciled = True
        if state_changed:
            self.persist_state()
        return recovered_count

    def submit(self, run: WatchedRun) -> RunRecord:
        """Launch and durably acknowledge one scan synchronously."""

        return self.acknowledge_submission(run, self.launch_submission(run))

    def refresh(self) -> dict[str, RunRecord]:
        state_changed = False
        self.refresh_errors = {}
        for run_id, run in list(self.state.submitted.items()):
            record = self.records.get(run_id)
            if record is None and run.galaxy_job_id:
                record = self._minimal_record(run)
                self.records[run_id] = record
            if record is None:
                self.refresh_errors[run_id] = "Acknowledged run has no Galaxy job ID"
                continue
            try:
                refreshed = self.service.refresh(record)
            except Exception as exc:
                self.refresh_errors[run_id] = str(exc)
                continue
            self.records[run_id] = refreshed
            if refreshed.status.value == "ok" and run_id in self.state.submitted:
                # Galaxy completion is authoritative. Do not make the live
                # watcher download and unpack every result archive before it
                # can acknowledge a finished scan. The NOVA results view
                # collects the archive lazily when a user opens this record.
                result_ids = tuple(str(value) for value in refreshed.output_dataset_ids.values() if value)
                if result_ids:
                    summary = self._scientific_summary(refreshed)
                    self.state.mark_completed(
                        run_id,
                        result_ids,
                        galaxy_job_id=refreshed.galaxy_job_id,
                        scientific_summary=summary,
                    )
                    state_changed = True
                else:
                    refreshed.stage = "Finalizing Galaxy outputs"
                    refreshed.progress = max(refreshed.progress, 98)
            elif refreshed.status.value in {"error", "cancelled"} and run_id in self.state.submitted:
                self.state.mark_failed(
                    run_id,
                    refreshed.message or f"Galaxy ended with status {refreshed.status.value}",
                    galaxy_job_id=refreshed.galaxy_job_id,
                )
                state_changed = True
        for run_id, run in list(self.state.completed.items()):
            if run.scientific_summary or not run.galaxy_job_id:
                continue
            record = self.records.get(run_id) or self._minimal_record(run)
            try:
                refreshed = self.service.refresh(record)
                self.records[run_id] = refreshed
                summary = self._scientific_summary(refreshed)
                if summary:
                    self.state.set_scientific_summary(run_id, summary)
                    state_changed = True
            except Exception as exc:
                self.refresh_errors.setdefault(run_id, f"Scientific summary pending: {exc}")
        if state_changed:
            self.persist_state()
        return dict(self.records)

    def _scientific_summary(self, record: RunRecord) -> dict[str, Any]:
        """Load only the small normalized result JSON, never the result archive."""

        dataset_id = str(record.output_dataset_ids.get("summary") or "").strip()
        if not dataset_id:
            return {}
        payload = self.service.load_json_document(dataset_id)
        return experiment_scan_summary(payload)

    def persist_state(self) -> str:
        """Write an immutable watch-state checkpoint to Galaxy History."""

        with self._persist_lock:
            serialized = self.state.to_json()
            if serialized == self._last_persisted_state and self.state_dataset_ids:
                return self.state_dataset_ids[-1]
            payload = self.state.as_dict()
            payload["watch"] = {
                "facility": "SNS",
                "instrument": "PG3",
                "ipts": self.settings.ipts,
                "source_directory": self.source_directory,
                "configuration_dataset_id": self.settings.configuration_dataset_id,
                "database_dataset_id": self.settings.database_dataset_id,
                "wavelength_angstrom": self.settings.wavelength_angstrom,
                "initial_backfill_limit": self.settings.initial_backfill_limit,
            }
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    dataset_id = self.service.upload_json_document(
                        payload,
                        name=f"{self.settings.ipts}_powgen_watch_state.json",
                        label=f"POWGEN watch state {self.settings.ipts}",
                    )
                    self.state_dataset_ids.append(dataset_id)
                    self._last_persisted_state = serialized
                    return dataset_id
                except Exception as exc:
                    last_error = exc
                    if attempt < 2:
                        time.sleep(0.5 * (2**attempt))
            raise RuntimeError(f"Galaxy could not save the POWGEN watch checkpoint: {last_error}") from last_error


__all__ = [
    "PowgenExperimentSettings",
    "PowgenWatchController",
    "bounded_directory_listing",
    "preflight_powgen_experiment",
]
