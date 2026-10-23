"""POWGEN live-experiment orchestration around the existing Analyze tool.

The controller owns no scientific implementation.  It discovers reduced PG3
patterns from a read-only experiment directory, resolves a verified official
instrument profile, and submits the same ``AnalysisConfig`` and
``InputSelection`` objects used by an ordinary NOVA run.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import json
import threading
import time
import tempfile
from hashlib import sha256
from typing import Any, Iterable, Mapping

from .galaxy_service import GalaxyService
from .models import AnalysisConfig, AnalysisMode, InputSelection, InputSource, RunRecord, RunStatus
from .powgen_setup import (
    RESOLVER_VERSION, CATALOG_PATH, BankProfile, inspect_reduced_setup, resolve_bank_profile, stage_bank, text_hash,
)
from .powgen_watch import (
    WatchDefinition,
    WatchState,
    WatchedRun,
    build_submission_plan,
    discover_from_listing,
)
from .powgen_metadata import read_powgen_scan_metadata
from .powgen_samples import PowgenSampleIndex, include_run, run_role
from .results import experiment_scan_summary


@dataclass(frozen=True)
class PowgenExperimentSettings:
    """User-controlled settings for one read-only POWGEN experiment watch."""

    ipts: str
    history_id: str
    configuration_dataset_id: str
    wavelength_angstrom: str
    instrument_profile_dataset_id: str | None = None
    instrument_profile_name: str | None = None
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
    selected_sample_ids: tuple[str, ...] | None = None
    selected_banks: tuple[int, ...] | None = None
    excluded_run_numbers: tuple[int, ...] = ()
    include_background_runs: bool = False
    include_ramp_runs: bool = False

    def __post_init__(self) -> None:
        if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in self.excluded_run_numbers):
            raise ValueError("Excluded run numbers must be positive integers")
        object.__setattr__(self, "excluded_run_numbers", tuple(sorted(set(self.excluded_run_numbers))))
        if self.selected_banks is not None:
            banks = tuple(sorted(set(self.selected_banks)))
            if not banks or any(isinstance(b, bool) or not isinstance(b, int) or b <= 0 for b in banks):
                raise ValueError("Choose at least one positive diffraction bank, or all banks")
            object.__setattr__(self, "selected_banks", banks)
        if self.selected_sample_ids is not None:
            selected = tuple(sorted({str(value).strip() for value in self.selected_sample_ids if str(value).strip()}))
            if not selected:
                raise ValueError("Choose at least one sample ID, or process all samples")
            object.__setattr__(self, "selected_sample_ids", selected)
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

    def definition(self, instrument_profile_ref: str | None = None) -> WatchDefinition:
        config_refs = {}
        if self.database_dataset_id:
            config_refs["candidate_library"] = self.database_dataset_id
        profile_ref = (
            instrument_profile_ref
            or self.instrument_profile_dataset_id
            or (RESOLVER_VERSION + ":" + sha256(CATALOG_PATH.read_bytes()).hexdigest())
        )
        return WatchDefinition(
            facility="SNS",
            instrument="PG3",
            ipts=self.ipts,
            subfolder=self.subfolder,
            history_id=self.history_id,
            configuration_ref=self.configuration_dataset_id,
            instrument_profile_ref=profile_ref,
            main_cif_ref=self.main_cif_dataset_id,
            config_refs=config_refs,
        )


def bounded_directory_listing(directory: str | Path) -> list[dict[str, Any]]:
    """Stream one folder, stat'ing only completed PG3 reductions; never recurse.

    Large autoreduce folders contain many ancillary files per scan. A limit on
    raw directory entries blocks normal experiments and cannot safely truncate
    an unordered listing. Filter filenames before any filesystem metadata call.
    """

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"POWGEN reduced-data directory does not exist: {root}")
    rows: list[dict[str, Any]] = []
    with _scandir(root) as entries:
        for entry in entries:
            if not re.fullmatch(r"PG3_\d+\.gsa", entry.name, re.IGNORECASE):
                continue
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                stat = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue  # Autoreduction may atomically replace a file mid-listing.
            rows.append(
                {
                    "path": str(root / entry.name).replace("\\", "/"),
                    "size": int(stat.st_size),
                    "modified_ns": int(stat.st_mtime_ns),
                }
            )
    return rows


def discover_powgen_samples(ipts: str, sample_index: PowgenSampleIndex,
                            subfolder: str = "shared/autoreduce") -> dict[str, Any]:
    definition = PowgenExperimentSettings(ipts, "samples", "samples", "", subfolder=subfolder).definition()
    runs = discover_from_listing(definition, bounded_directory_listing(definition.source_directory))
    inventory = sample_index.inventory(runs)
    options = [{"title": "Current autoreduce folder", "value": "shared/autoreduce"}]
    # Historical cycle folders are explicit choices, never recursively scanned
    # or merged with current reductions that may contain reprocessed copies.
    root = Path(f"/SNS/PG3/{definition.ipts}/shared/autoreduce")
    if root.is_dir():
        with _scandir(root) as entries:
            names = sorted([e.name for e in entries if re.fullmatch(r"\d{4}[-_][AB12]", e.name, re.I)
                            and e.is_dir(follow_symlinks=False)], reverse=True)
        options += [{"title": f"Archived cycle {n}", "value": f"shared/autoreduce/{n}"} for n in names]
    inventory["source_options"] = options
    return inventory


_POWGEN_PROFILE_SUFFIXES = {".instprm", ".prm", ".inst", ".ins"}


def preflight_powgen_experiment(
    ipts: str, *, requested_wavelength: str | None = None,
    instrument_profile_name: str | None = None,
    selected_sample_ids: tuple[str, ...] | None = None,
    selected_banks: tuple[int, ...] | None = None,
    initial_backfill_limit: int | None = None,
    sample_index: PowgenSampleIndex | None = None,
    subfolder: str = "shared/autoreduce",
    excluded_run_numbers: tuple[int, ...] = (),
    include_background_runs: bool = False,
    include_ramp_runs: bool = False,
) -> dict[str, Any]:
    """Check every selected initial run/bank without submitting scientific jobs."""
    settings = PowgenExperimentSettings(ipts, "preflight", "preflight", "",
        selected_sample_ids=selected_sample_ids, selected_banks=selected_banks,
        initial_backfill_limit=initial_backfill_limit, subfolder=subfolder)
    definition = settings.definition()
    runs = discover_from_listing(definition, bounded_directory_listing(definition.source_directory))
    total = len(runs)
    if selected_sample_ids is not None:
        index = sample_index or PowgenSampleIndex(definition.ipts)
        with ThreadPoolExecutor(max_workers=4) as pool:
            identities = list(pool.map(index.identity, runs))
        runs = [run for run, identity in zip(runs, identities) if identity.get("sample_id") in selected_sample_ids]
    selected_count = len(runs)
    runs = [run for run in runs if include_run(run, excluded_run_numbers=excluded_run_numbers,
            include_background_runs=include_background_runs, include_ramp_runs=include_ramp_runs)]
    excluded_count = selected_count - len(runs)
    if initial_backfill_limit is not None:
        # New-only monitoring still checks the latest matching setup as a
        # startup preview. Every arriving run is checked again at submission.
        runs = runs[-max(1, initial_backfill_limit):]
    if not runs:
        return {"ready": False, "source_directory": definition.source_directory,
                "scan_count": 0, "total_scan_count": total,
                "message": "No completed reductions match this experiment/sample selection."}
    override = str(instrument_profile_name or "").strip()
    errors, setups, profiles, available_banks = [], [], [], set()
    def inspect(run):
        try:
            return run, inspect_reduced_setup(run.source_path), ""
        except (OSError, ValueError, UnicodeError) as exc:
            return run, None, str(exc)
    with ThreadPoolExecutor(max_workers=4) as pool:
        inspected = list(pool.map(inspect, runs))
    analysis_count = 0
    for run, setup, error in inspected:
        if error:
            errors.append(f"{run.run_id}: {error}")
            continue
        setups.append(setup)
        available_banks.update(b.bank for b in setup.banks)
        banks = [b.bank for b in setup.banks if selected_banks is None or b.bank in selected_banks]
        for bank in banks:
            analysis_count += 1
            try:
                descriptor = next(b for b in setup.banks if b.bank == bank)
                if not override and (descriptor.encoding != "FXYE" or descriptor.channels != descriptor.records):
                    raise ValueError("Automatic bank staging requires explicit GSAS FXYE records")
                if requested_wavelength and abs(float(requested_wavelength) - setup.wavelength) > 0.0005:
                    raise ValueError("Selected wavelength disagrees with this reduced file; use automatic per-run detection")
                if override:
                    if Path(override).name != override or Path(override).suffix.lower() not in _POWGEN_PROFILE_SUFFIXES:
                        raise ValueError("The supplied POWGEN instrument profile name is not valid")
                    if len(setup.banks) != 1:
                        raise ValueError("A manual fallback cannot be applied to multiple banks; use validated bank mappings")
                else:
                    profiles.append(resolve_bank_profile(setup, bank))
            except (OSError, ValueError) as exc:
                errors.append(f"{run.run_id} BANK {bank}: {exc}")
    if override and len({(v.cycle, v.mode, v.wavelength, v.frequency, v.banks[0].bank) for v in setups}) > 1:
        errors.append("One manual profile cannot be reused across mixed calibration, mode, wavelength, frequency or bank settings")
    if not analysis_count and not errors:
        errors.append("None of the selected banks occurs in the selected scans")
    latest = runs[-1]
    metadata = read_powgen_scan_metadata(definition.ipts, latest.run_number)
    wave_values = sorted({v.wavelength for v in setups})
    wave = f"{wave_values[0]:g}" if len(wave_values) == 1 else ""
    names = sorted({p.profile_filename for p in profiles})
    profile_error = "; ".join(errors[:3]) + (f"; and {len(errors)-3} more" if len(errors) > 3 else "")
    sample = " | ".join(str(metadata[k]) for k in ["sample_id", "sample_name", "sample_formula"] if metadata.get(k)) or "Not reported"
    temperature = metadata.get("temperature") or {}
    return {
        "ready": not errors, "source_directory": definition.source_directory,
        "scan_count": len(runs), "analysis_count": analysis_count, "total_scan_count": total,
        "excluded_count": excluded_count,
        "review_runs": [{"run_number": r.run_number, **run_role(r)} for r in runs],
        "first_run": runs[0].run_id, "latest_run": latest.run_id, "latest_file": Path(latest.source_path).name,
        "latest_start_time": str(metadata.get("start_time") or ""), "sample_display": sample,
        "temperature_display": f"{temperature.get('value', '')} {temperature.get('unit', '')}".strip() or "Not reported",
        "metadata_available": bool(metadata.get("available")), "detected_wavelength": wave,
        "selected_wavelength": wave or "Mixed; resolved separately for each run",
        "wavelength_source": "Read independently from each reduced file",
        "available_banks": sorted(available_banks), "blocked_count": len(errors), "blocked_examples": errors[:20],
        "profile_filename": override or (names[0] if len(names) == 1 else f"{len(names)} calibrated profiles"),
        "profile_rule": "user-supplied" if override else RESOLVER_VERSION,
        "profile_source": "User-supplied GSAS-II profile" if override else "Verified reduction cycle, mode, frequency, wavelength and bank",
        "profile_error": profile_error,
        "message": (f"Ready: {len(runs)} scans, {analysis_count} separate bank analyses. Every arriving scan is checked again."
                    if not errors else f"Instrument setup needs attention: {profile_error}"),
    }


def _scandir(path: Path):
    import os

    return os.scandir(path)


class PowgenWatchController:
    """Coordinate discovery, submission, and Galaxy-owned watch state."""

    def __init__(self, service: GalaxyService, settings: PowgenExperimentSettings,
                 *, sample_index: PowgenSampleIndex | None = None) -> None:
        if service.history_id != settings.history_id:
            raise ValueError("POWGEN settings must target the active Galaxy history")
        self.service = service
        self.settings = settings
        self.sample_index = sample_index or PowgenSampleIndex(settings.ipts)
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
        self._staging = tempfile.TemporaryDirectory(prefix="radar-powgen-banks-")
        self._setup_cache: dict[tuple, Any] = {}
        self._manual_setup_signature = None
        self._manual_setup_lock = threading.Lock()
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
                if not isinstance(watch, Mapping) and self.settings.selected_sample_ids is not None:
                    continue
                if isinstance(watch, Mapping):
                    if watch.get("catalog_sha256") != sha256(CATALOG_PATH.read_bytes()).hexdigest():
                        continue
                    if watch.get("resolver") != RESOLVER_VERSION:
                        continue
                    if tuple(watch.get("excluded_run_numbers") or ()) != self.settings.excluded_run_numbers:
                        continue
                    if watch.get("include_background_runs", True) != self.settings.include_background_runs:
                        continue
                    if watch.get("include_ramp_runs", True) != self.settings.include_ramp_runs:
                        continue
                    stored_banks = watch.get("selected_banks")
                    if (tuple(stored_banks) if stored_banks is not None else None) != self.settings.selected_banks:
                        continue
                    if watch.get("source_directory") != self.source_directory:
                        continue
                    if str(watch.get("ipts") or "").upper() != self.settings.ipts.upper():
                        continue
                    if str(watch.get("configuration_dataset_id") or "") != self.settings.configuration_dataset_id:
                        continue
                    if str(watch.get("wavelength_angstrom") or "") != self.settings.wavelength_angstrom:
                        continue
                    if str(watch.get("instrument_profile_dataset_id") or "") != str(
                        self.settings.instrument_profile_dataset_id or ""
                    ):
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
                    stored_samples = watch.get("selected_sample_ids")
                    if (tuple(sorted(stored_samples)) if stored_samples is not None else None) != self.settings.selected_sample_ids:
                        continue
                candidate = WatchState.from_dict(payload)
                if candidate.history_id != self.settings.history_id:
                    continue
            except (OSError, TypeError, ValueError):
                continue
            if isinstance(watch, Mapping) and watch.get("manual_setup_signature"):
                self._manual_setup_signature = tuple(watch["manual_setup_signature"])
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
            discovered = self._matching_samples(available)
            if limit == 0:
                discovered = []
            elif limit is not None:
                discovered = discovered[-limit:]
            discovered = self._remember_discoveries(self._expand_banks(discovered))
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
        discovered = self._remember_discoveries(self._expand_banks(self._matching_samples(discovered)))
        discovered = self._attach_scan_metadata(discovered)
        if discovered:
            self._last_seen_run_number = max(
                self._last_seen_run_number or 0,
                *(run.run_number for run in discovered),
            )
        return discovered

    def _setup(self, run: WatchedRun):
        path = Path(run.source_path)
        stat = path.stat()
        sidecar = path.with_suffix(".py")
        side_stat = sidecar.stat() if sidecar.is_file() else None
        key = (str(path), stat.st_size, stat.st_mtime_ns,
               (side_stat.st_size, side_stat.st_mtime_ns) if side_stat else None)
        if key not in self._setup_cache:
            self._setup_cache[key] = inspect_reduced_setup(path)
        return self._setup_cache[key]

    def _expand_banks(self, runs: list[WatchedRun]) -> list[WatchedRun]:
        expanded = []
        for run in runs:
            try:
                setup = self._setup(run)
                for bank in setup.banks:
                    if self.settings.selected_banks is not None and bank.bank not in self.settings.selected_banks:
                        continue
                    # Keep the familiar run identifier for single-bank files;
                    # multiple banks have distinct lifecycle and Galaxy jobs.
                    selected = replace(run, bank=bank.bank if len(setup.banks) > 1 else None,
                        scan_metadata={**run.scan_metadata, "source_bank": bank.bank,
                                       "powgen_setup": setup.as_dict()})
                    expanded.append(selected)
            except (OSError, ValueError, UnicodeError) as exc:
                expanded.append(replace(run, scan_metadata={**run.scan_metadata, "setup_error": str(exc)}))
        return expanded

    def _matching_samples(self, runs: list[WatchedRun]) -> list[WatchedRun]:
        runs = [run for run in runs if include_run(run,
            excluded_run_numbers=self.settings.excluded_run_numbers,
            include_background_runs=self.settings.include_background_runs,
            include_ramp_runs=self.settings.include_ramp_runs)]
        if self.settings.selected_sample_ids is None:
            return runs
        selected = set(self.settings.selected_sample_ids)
        with ThreadPoolExecutor(max_workers=4) as pool:
            identities = list(pool.map(self.sample_index.identity, runs))
        return [run for run, metadata in zip(runs, identities) if metadata.get("sample_id") in selected]

    def _remember_discoveries(self, runs: list[WatchedRun]) -> list[WatchedRun]:
        new = [run for run in runs if not self.state.contains(run)]
        self.state.discovered.update({run.run_id: run for run in new})
        return new

    def _read_scan_metadata(self, run: WatchedRun) -> dict[str, Any]:
        """Read optional NeXus conditions without making discovery fragile."""

        try:
            return read_powgen_scan_metadata(self.settings.ipts, run.run_number)
        except (OSError, TypeError, ValueError):
            return {}

    def _attach_scan_metadata(self, runs: Iterable[WatchedRun]) -> list[WatchedRun]:
        enriched: list[WatchedRun] = []
        by_source: dict[str, dict[str, Any]] = {}
        for run in runs:
            if run.source_path not in by_source:
                by_source[run.source_path] = self._read_scan_metadata(run)
            metadata = {**by_source[run.source_path], **dict(run.scan_metadata or {})}
            metadata.update(run_role(run))
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
            output_profile="monitor",
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

    def resolve_profile(self, run: WatchedRun) -> tuple[BankProfile, Path]:
        setup = self._setup(run)
        if self.settings.wavelength_angstrom and abs(float(self.settings.wavelength_angstrom) - setup.wavelength) > 0.0005:
            raise ValueError("Selected wavelength conflicts with this run; use automatic detection")
        bank = run.bank or int(run.scan_metadata.get("source_bank") or setup.banks[0].bank)
        resolution = resolve_bank_profile(setup, bank)
        return resolution, resolution.path

    def _profile_dataset_id(self, profile_path: Path) -> str:
        """Upload each packaged profile once and reuse it for every matching scan."""

        key = text_hash(profile_path.read_bytes())
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
        if config.radiation.value != "neutron" or config.instrument_mode == "cw":
            raise ValueError("POWGEN requires a neutron time-of-flight configuration")
        config = config.model_copy(update={"run_name": f"{self.settings.ipts}_{run.run_id}"})
        setup = self._setup(run)
        data_path = run.source_path
        bank_provenance = {"resolver": RESOLVER_VERSION, "source_path": run.source_path,
                           "source_signature": setup.source_signature,
                           "source_bank": run.bank or run.scan_metadata.get("source_bank") or setup.banks[0].bank}
        if self.settings.instrument_profile_dataset_id:
            if len(setup.banks) != 1:
                raise ValueError("Manual fallback cannot be applied to a multiple-bank reduction")
            signature = (setup.cycle, setup.mode, setup.wavelength, setup.frequency, setup.banks[0].bank)
            with self._manual_setup_lock:
                if self._manual_setup_signature is not None and self._manual_setup_signature != signature:
                    raise ValueError("This run's calibration, mode, wavelength, frequency or bank changed; the manual profile cannot be reused")
                self._manual_setup_signature = signature
            profile_dataset_id = self.settings.instrument_profile_dataset_id
            bank_provenance["manual_profile_dataset_id"] = profile_dataset_id
        else:
            resolution, _ = self.resolve_profile(run)
            destination = Path(self._staging.name) / run.fingerprint / run.run_id
            data, profile_path = stage_bank(Path(run.source_path), setup, resolution.source_bank, resolution, destination)
            data_path = str(data)
            profile_dataset_id = self._profile_dataset_id(profile_path)
            bank_provenance["profile"] = resolution.as_dict()
            receipt_path = destination / f"PG3_{run.run_number}_bank{resolution.source_bank}.json"
            if receipt_path.is_file():
                bank_provenance.update(json.loads(receipt_path.read_text()))
        inputs = InputSelection(
            source=InputSource.UPLOAD,
            data_path=data_path,
            bank=str(run.bank or run.scan_metadata.get("source_bank") or setup.banks[0].bank),
            run_number=str(run.run_number),
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
        record.provenance["powgen_bank"] = bank_provenance
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
        if record.provenance.get("powgen_bank") and run.run_id in self.state.discovered:
            current = self.state.discovered[run.run_id]
            self.state.discovered[run.run_id] = replace(current, scan_metadata={
                **current.scan_metadata, "powgen_bank_provenance": record.provenance["powgen_bank"]})
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
        candidate_instrument = (
            str(inputs.instrument_dataset_id or "")
            if inputs is not None
            else str(record.input_dataset_ids.get("instrument") or "")
        )
        matches_custom_profile = (
            not self.settings.instrument_profile_dataset_id
            or candidate_instrument == self.settings.instrument_profile_dataset_id
        )
        return (
            candidate_database == str(self.settings.database_dataset_id or "")
            and candidate_main == str(self.settings.main_cif_dataset_id or "")
            and matches_custom_profile
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
        pattern = re.compile(rf"^{re.escape(self.settings.ipts)}_(PG3_\d+(?:_BANK\d+)?)$", re.IGNORECASE)
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
                "instrument_profile_dataset_id": self.settings.instrument_profile_dataset_id,
                "instrument_profile_name": self.settings.instrument_profile_name,
                "wavelength_angstrom": self.settings.wavelength_angstrom,
                "initial_backfill_limit": self.settings.initial_backfill_limit,
                "selected_sample_ids": list(self.settings.selected_sample_ids) if self.settings.selected_sample_ids is not None else None,
                "selected_banks": list(self.settings.selected_banks) if self.settings.selected_banks is not None else None,
                "excluded_run_numbers": list(self.settings.excluded_run_numbers),
                "include_background_runs": self.settings.include_background_runs,
                "include_ramp_runs": self.settings.include_ramp_runs,
                "resolver": RESOLVER_VERSION,
                "catalog_sha256": sha256(CATALOG_PATH.read_bytes()).hexdigest(),
                "manual_setup_signature": self._manual_setup_signature,
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
