"""POWGEN live-experiment orchestration around the existing Analyze tool.

The controller owns no scientific implementation.  It discovers reduced PG3
patterns from a read-only experiment directory, resolves a verified official
instrument profile, and submits the same ``AnalysisConfig`` and
``InputSelection`` objects used by an ordinary NOVA run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    initial_scan_count: int = 1
    late_arrival_window: int = 100
    max_submission_attempts: int = 5
    retry_base_seconds: int = 15
    retry_max_seconds: int = 300

    def __post_init__(self) -> None:
        if self.initial_scan_count < 1:
            raise ValueError("initial_scan_count must be positive")
        if self.late_arrival_window < 0:
            raise ValueError("late_arrival_window must not be negative")
        if self.max_submission_attempts < 1:
            raise ValueError("max_submission_attempts must be positive")
        if self.retry_base_seconds < 0:
            raise ValueError("retry_base_seconds must not be negative")
        if self.retry_max_seconds < self.retry_base_seconds:
            raise ValueError("retry_max_seconds must be at least retry_base_seconds")

    def definition(self, instrument_profile_ref: str = "packaged-official-profile") -> WatchDefinition:
        return WatchDefinition(
            facility="SNS",
            instrument="PG3",
            ipts=self.ipts,
            subfolder=self.subfolder,
            history_id=self.history_id,
            configuration_ref=self.configuration_dataset_id,
            instrument_profile_ref=instrument_profile_ref,
            main_cif_ref=self.main_cif_dataset_id,
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
        self._last_persisted_state: str | None = None
        self._configuration: AnalysisConfig | None = None
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
        all_runs = [
            *restored.discovered.values(),
            *restored.submitted.values(),
            *restored.completed.values(),
            *restored.failed.values(),
        ]
        if all_runs:
            self._last_seen_run_number = max(run.run_number for run in all_runs)
            self._primed = True

        for run_id, run in restored.submitted.items():
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
        if not self._primed:
            discovered = discover_from_listing(
                self.definition,
                completed_rows,
                self.state,
                newest_limit=self.settings.initial_scan_count,
            )
            if discovered:
                self._last_seen_run_number = max(run.run_number for run in discovered)
                self._primed = True
            return discovered

        discovered = discover_from_listing(
            self.definition,
            completed_rows,
            self.state,
            minimum_run_number=max(
                1,
                (self._last_seen_run_number or 0) - self.settings.late_arrival_window,
            ),
        )
        if discovered:
            self._last_seen_run_number = max(
                self._last_seen_run_number or 0,
                *(run.run_number for run in discovered),
            )
        return discovered

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
        """Return newly discovered or delayed runs ready for submission."""

        return self.state.retryable_runs()

    def defer_submission(self, run: WatchedRun, error: str) -> WatchedRun:
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
        self.persist_state()
        return pending

    def resolve_profile(self, run: WatchedRun) -> tuple[PowgenProfileResolution, Path]:
        resolution = resolve_powgen_profile(
            run_number=run.run_number,
            wavelength_angstrom=self.settings.wavelength_angstrom,
            frequency_hz=self.settings.frequency_hz,
        )
        return resolution, resolve_packaged_powgen_profile_path(resolution)

    def submit(self, run: WatchedRun) -> RunRecord:
        plan = build_submission_plan(self.definition, run, self.state)
        if plan is None:
            existing = self.records.get(run.run_id)
            if existing is None:
                raise ValueError(f"{run.run_id} was already submitted outside this controller instance")
            return existing

        if self._configuration is None:
            self._configuration = self.service.load_configuration_dataset(
                self.settings.configuration_dataset_id
            )
        config = self._configuration
        config = config.model_copy(update={"run_name": f"{self.settings.ipts}_{run.run_id}"})
        _, profile_path = self.resolve_profile(run)
        inputs = InputSelection(
            source=InputSource.UPLOAD,
            data_path=run.source_path,
            instrument_path=str(profile_path),
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
        )
        record = self.service.submit_snapshot(snapshot)
        if not record.galaxy_job_id:
            raise RuntimeError(f"Galaxy did not acknowledge {run.run_id}")
        self.state.mark_submitted(run, record.galaxy_job_id)
        self.records[run.run_id] = record
        self.persist_state()
        return record

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
                    self.state.mark_completed(run_id, result_ids, galaxy_job_id=refreshed.galaxy_job_id)
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
        if state_changed:
            self.persist_state()
        return dict(self.records)

    def persist_state(self) -> str:
        """Write an immutable watch-state checkpoint to Galaxy History."""

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
            "wavelength_angstrom": self.settings.wavelength_angstrom,
        }
        dataset_id = self.service.upload_json_document(
            payload,
            name=f"{self.settings.ipts}_powgen_watch_state.json",
            label=f"POWGEN watch state {self.settings.ipts}",
        )
        self.state_dataset_ids.append(dataset_id)
        self._last_persisted_state = serialized
        return dataset_id


__all__ = [
    "PowgenExperimentSettings",
    "PowgenWatchController",
    "bounded_directory_listing",
]
