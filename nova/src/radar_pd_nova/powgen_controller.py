"""POWGEN live-experiment orchestration around the existing Analyze tool.

The controller owns no scientific implementation.  It discovers reduced PG3
patterns from a read-only experiment directory, resolves a verified official
instrument profile, and submits the same ``AnalysisConfig`` and
``InputSelection`` objects used by an ordinary NOVA run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import json
import threading
import time
import tempfile
import inspect
from hashlib import sha256
from typing import Any, Iterable, Mapping

from .galaxy_service import GalaxyService
from .checkpoint_uploads import CheckpointUploadPending
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
from .powgen_samples import PowgenSampleIndex, include_run, run_role, identity_readiness, select_runs
from .results import experiment_scan_summary
from .acquisition import acquisition_context, stage_manual_acquisition


@dataclass(frozen=True)
class WatchCheckpoint:
    generation: int
    state_json: str
    payload_json: str


@dataclass(frozen=True)
class _MetadataRead:
    run_id: str
    job_id: str
    fingerprint: str
    kind: str
    record_json: str = ""
    receipt_json: str = ""
    result_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _MetadataResult:
    request: _MetadataRead
    payload_json: str = ""
    error: str = ""


@dataclass(frozen=True)
class PowgenExperimentSettings:
    """User-controlled settings for one read-only POWGEN experiment watch."""

    ipts: str
    history_id: str
    configuration_dataset_id: str
    wavelength_angstrom: str
    instrument_profile_dataset_id: str | None = None
    instrument_profile_name: str | None = None
    frequency_hz: str | None = "60"
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
    excluded_run_numbers: tuple[int | str, ...] = ()
    include_background_runs: bool = False
    include_ramp_runs: bool = False
    instrument: str = "PG3"

    def __post_init__(self) -> None:
        from .powgen_watch import normalize_exclusions
        object.__setattr__(self, "excluded_run_numbers", normalize_exclusions(self.excluded_run_numbers,self.instrument))
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
                            subfolder: str = "shared/autoreduce", *, progress=None) -> dict[str, Any]:
    definition = PowgenExperimentSettings(ipts, "samples", "samples", "", subfolder=subfolder).definition()
    runs = discover_from_listing(definition, bounded_directory_listing(definition.source_directory))
    kwargs={"progress":progress} if progress is not None else {}
    inventory = sample_index.inventory(runs, catalog=True, **kwargs)
    options = [{"title": "Current autoreduce folder", "value": "shared/autoreduce"}]
    # Historical cycle folders are explicit choices, never recursively scanned
    # or merged with current reductions that may contain reprocessed copies.
    root = Path(f"/SNS/PG3/{definition.ipts}/shared/autoreduce")
    if root.is_dir():
        with _scandir(root) as entries:
            names = sorted([e.name for e in entries if re.fullmatch(r"(?:19|20)\d{2}(?:[-_]?[AB12])?", e.name, re.I)
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
    index = sample_index or PowgenSampleIndex(definition.ipts)
    selection_stats = {}
    runs = select_runs(runs, index, selected_sample_ids, limit=initial_backfill_limit,
        excluded_run_numbers=excluded_run_numbers, include_background_runs=include_background_runs,
        include_ramp_runs=include_ramp_runs, selection_stats=selection_stats)
    excluded_count = selection_stats.get("excluded_count", 0)
    if not runs:
        review = selection_stats.get("identity_review") or {}
        return {"ready": False, "source_directory": definition.source_directory,
                "scan_count": 0, "total_scan_count": total,
                "metadata_resolution": [{"run_number": n, "status": "needs_review", "sample_id": "",
                    "actions": [{"message": reason}]} for n, reason in review.items()],
                "metadata_notices": (["Candidate runs were excluded because their source identities need review."] if review else []),
                "message": ("No completed reductions have a verified matching sample identity; review the recorded sources."
                            if review else "No completed reductions match this experiment/sample selection.")}
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
    identity_rows = identity_readiness(index, runs)
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
        "metadata_resolution": identity_rows,
        "metadata_notices": [*metadata.get('metadata_notices', []), *(
            [f"{len(selection_stats['identity_review'])} candidate run(s) were not selected because source identity needs review: "
             + ", ".join(str(n) for n in list(selection_stats['identity_review'])[:10])]
            if selection_stats.get('identity_review') else [])],
        "message": (f"Ready: {len(runs)} scans, {analysis_count} separate bank analyses. Every arriving scan is checked again."
                    if not errors else f"Instrument setup needs attention: {profile_error}"),
    }


def _scandir(path: Path):
    import os

    return os.scandir(path)


class PowgenWatchController:
    """Coordinate discovery, submission, and Galaxy-owned watch state."""

    resolver_version = RESOLVER_VERSION
    bank_provenance_key = "powgen_bank"
    setup_metadata_key = "powgen_setup"

    @property
    def beamline_label(self):
        return "NOMAD" if self.settings.instrument == "NOM" else "POWGEN"

    def _catalog_digest(self):
        return sha256(CATALOG_PATH.read_bytes()).hexdigest()

    def _watch_extension(self):
        return {}

    def _checkpoint_key(self) -> str:
        """Index compatible settings before downloading a History checkpoint.

        This is a search hint, not a replacement for validating the stored
        watch and source/job identities during restoration.
        """
        identity = {
            "history": self.settings.history_id,
            "instrument": self.settings.instrument,
            "source": self.source_directory,
            "references": dict(self.definition.config_refs),
            "resolver": self.resolver_version,
            "catalog": self._catalog_digest(),
            "wavelength": self.settings.wavelength_angstrom,
            "initial_backfill_limit": self.settings.initial_backfill_limit,
            "selected_sample_ids": self.settings.selected_sample_ids,
            "selected_banks": self.settings.selected_banks,
            "excluded_run_numbers": self.settings.excluded_run_numbers,
            "include_background_runs": self.settings.include_background_runs,
            "include_ramp_runs": self.settings.include_ramp_runs,
            **self._watch_extension(),
        }
        return sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:32]

    def _checkpoint_candidates(self):
        label = f"{self.beamline_label} watch state {self.settings.ipts}"
        marker = f"[watch:{self._checkpoint_key()}]"
        seen = set()
        # Prefer the matching settings index. The second query preserves
        # recovery of pre-index checkpoints without re-reading keyed states
        # belonging to another library/configuration/sample selection.
        for query in (f"{label} {marker}", label):
            for row in self.service.search_history_datasets(
                query=query, limit=25, include_generated=True,
            ):
                dataset_id = str(row.get("id") or "")
                if not dataset_id or dataset_id in seen:
                    continue
                seen.add(dataset_id)
                stored_marker = re.search(r"\[watch:[0-9a-f]{32}\]", str(row.get("name") or ""))
                if stored_marker and stored_marker[0] != marker:
                    continue
                yield row

    def _run_name(self, run):
        generation=run.scan_metadata.get('retry_generation',0)
        suffix=f'_retry{generation}' if generation else ''
        return f"{self.settings.ipts}_{run.run_id}{suffix}"

    def _list_source_directory(self):
        return bounded_directory_listing(self.source_directory)

    def __init__(self, service: GalaxyService, settings: PowgenExperimentSettings,
                 *, sample_index: PowgenSampleIndex | None = None) -> None:
        if service.history_id != settings.history_id:
            raise ValueError("POWGEN settings must target the active Galaxy history")
        self.service = service
        self.settings = settings
        self.sample_index = sample_index or PowgenSampleIndex(settings.ipts)
        self.definition = settings.definition()
        self.state = WatchState(history_id=settings.history_id, instrument=settings.instrument)
        self.records: dict[str, RunRecord] = {}
        self.state_dataset_ids: list[str] = []
        self._primed = False
        self._last_seen_run_number: int | None = None
        self._startup_run_floor: int | None = None
        self._last_persisted_state: str | None = None
        self._last_persisted_payload_sha256: str | None = None
        self._configuration: AnalysisConfig | None = None
        self._configuration_lock = threading.Lock()
        self._profile_dataset_ids: dict[str, str] = {}
        self._profile_upload_lock = threading.Lock()
        self._persist_lock = threading.Lock()
        self._checkpoint_lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._metadata_refresh_cursor = ""
        self._curve_cache_stamps: dict[tuple[str, str], tuple] = {}
        self._checkpoint_generation = 0
        self._persisted_checkpoint_generation = 0
        self._staging = tempfile.TemporaryDirectory(prefix="radar-powgen-banks-")
        self._setup_cache: dict[tuple, Any] = {}
        self._manual_setup_signature = None
        self._manual_setup_lock = threading.Lock()
        self._recent_runs_reconciled = False
        self.refresh_errors: dict[str, str] = {}
        self._retry_checkpoint_pending = False
        self._retry_submission_allowed = lambda: True
        self._before_live_submit = None

    @property
    def interrupted_retries_pending(self):
        return self._retry_checkpoint_pending or any(
            (run.scan_metadata.get('input_retry_intent') or run.scan_metadata.get('run_retry_intent'))
            and not run.scan_metadata.get('retry_deferred_by_stop')
            for run in self.state.failed.values())

    def retry_interrupted_uploads(self, *, request=False):
        """Compatibility name for the explicit failed-run recovery flow."""
        return self.retry_failed_runs(request=request)

    def retry_failed_runs(self, *, request=False):
        """Save explicit intent and generation before admitting fresh science.

        Terminal errors retain their jobs and artifacts. Only the separately
        proven paused-empty-input path may retire an unstarted placeholder.
        request=False resumes already saved intent; it never enrolls errors.
        """
        from .live_recovery import (RetrySourceChanged, failed_input_proof, failed_run_proof,
                                    retire_failed_input, validate_retry_source)
        resumed=0
        if request:
            for phase in (self.state.failed,self.state.discovered):
                for key,run in list(phase.items()):
                    if run.scan_metadata.get('retry_deferred_by_stop'):
                        metadata=dict(run.scan_metadata);metadata.pop('retry_deferred_by_stop')
                        phase[key]=replace(run,scan_metadata=metadata)
                        self._retry_checkpoint_pending=True
                        resumed+=int(phase is self.state.discovered)
                        if phase is self.state.discovered:
                            self._recent_runs_reconciled=False
            intents = []
            for key,run in list(self.state.failed.items()):
                if (not run.galaxy_job_id or run.scan_metadata.get('input_retry_intent')
                        or run.scan_metadata.get('run_retry_intent')):
                    continue
                generation=run.scan_metadata.get('retry_generation',0)
                if type(generation) is not int or not 0<=generation<20:
                    continue
                proof=failed_run_proof(self.service,run.galaxy_job_id) or failed_input_proof(self.service,run.galaxy_job_id)
                if proof:
                    intents.append((key,run,proof))
            for key,run,proof in intents:
                if self.state.failed.get(key) is not run:
                    continue
                field='run_retry_intent' if proof.get('kind')=='terminal_error' else 'input_retry_intent'
                self.state.failed[key]=replace(run,scan_metadata={**run.scan_metadata,field:proof})
                self._retry_checkpoint_pending=True
        if not self.interrupted_retries_pending:
            return 0
        self._retry_checkpoint_pending=True
        self.persist_state()
        retried=resumed
        for key,run in list(self.state.failed.items()):
            proof=run.scan_metadata.get('run_retry_intent') or run.scan_metadata.get('input_retry_intent')
            if not proof or self.state.failed.get(key) is not run:
                continue
            generation=run.scan_metadata.get('retry_generation',0)
            if type(generation) is not int or not 0<=generation<20:
                raise RetrySourceChanged('Invalid saved retry generation; review the checkpoint before retrying')
            if proof.get('kind')=='terminal_error':
                try:
                    validate_retry_source(self.service,proof)
                except RetrySourceChanged as exc:
                    metadata=dict(run.scan_metadata);metadata.pop('run_retry_intent',None)
                    metadata['retry_hold']=str(exc)
                    self.state.failed[key]=replace(run,scan_metadata=metadata)
                    continue
            else:
                retire_failed_input(self.service,proof)
            if self.state.failed.get(key) is not run:
                raise RetrySourceChanged('The selected failed run changed while checking its retry; no replacement was submitted')
            metadata=dict(run.scan_metadata)
            metadata.pop('input_retry_intent',None);metadata.pop('run_retry_intent',None)
            metadata.pop('retry_hold',None)
            metadata.pop('submission_receipt_pending', None)
            metadata['retry_generation']=metadata.get('retry_generation',0)+1
            metadata['previous_attempts']=[*metadata.get('previous_attempts',[]),proof]
            metadata['retry_source_proof']=proof
            self.state.failed.pop(key)
            self.state.discovered[key]=replace(run,galaxy_job_id=None,error=None,
                galaxy_result_ids=(),scientific_summary={},next_retry_utc=None,
                submission_attempts=0,scan_metadata=metadata)
            self.records.pop(key,None)
            self._recent_runs_reconciled=False
            retried+=1
        self.persist_state()
        # The generation checkpoint itself may wait on Galaxy. Recheck the
        # old job before allowing this saved generation to reach submission.
        held=False
        for run in list(self.state.discovered.values()):
            proof=run.scan_metadata.get('retry_source_proof')
            if proof:
                try:
                    validate_retry_source(self.service,proof)
                except RetrySourceChanged as exc:
                    self.hold_failed_retry(run,str(exc))
                    held=True
                    retried=max(0,retried-1)
        if held:
            self.persist_state()
        self._retry_checkpoint_pending=False
        return retried

    def pause_failed_run_retries(self):
        """Stop/End defers unsubmitted retry intent until another explicit click."""
        changed=False
        for phase in (self.state.failed,self.state.discovered):
            for key,run in list(phase.items()):
                names=(('input_retry_intent','run_retry_intent') if phase is self.state.failed
                       else ('retry_source_proof',))
                if any(run.scan_metadata.get(name) for name in names):
                    phase[key]=replace(run,scan_metadata={**run.scan_metadata,'retry_deferred_by_stop':True})
                    changed=True
        if changed:
            self._retry_checkpoint_pending=False
        return changed

    def hold_failed_retry(self, run, message):
        """Keep a revoked retry out of automatic submission and keep its past."""
        current=self.state.discovered.get(run.run_id)
        if current is None or current.fingerprint!=run.fingerprint:
            return
        metadata=dict(current.scan_metadata)
        proof=metadata.pop('retry_source_proof',None)
        if not proof:
            return
        metadata['retry_hold']=message
        self.state.discovered.pop(run.run_id)
        self.state.failed[run.run_id]=replace(current,galaxy_job_id=proof['job_id'],
            error=message,scan_metadata=metadata,galaxy_result_ids=tuple(proof.get('outputs',{}).values()))

    def _validate_retry_submission(self, run):
        from .live_recovery import RetrySourceChanged, validate_retry_source
        proof=run.scan_metadata.get('retry_source_proof')
        if not proof:
            return
        def current():
            known=self.state.discovered.get(run.run_id)
            return (known is not None and known.fingerprint==run.fingerprint and known.source_path==run.source_path
                    and known.scan_metadata.get('retry_generation')==run.scan_metadata.get('retry_generation')
                    and known.scan_metadata.get('retry_source_proof')==proof
                    and not known.scan_metadata.get('source_revision_warning')
                    and not known.scan_metadata.get('setup_error')
                    and not known.scan_metadata.get('retry_deferred_by_stop')
                    and not self.interrupted_retries_pending and self._retry_submission_allowed())
        if not current():
            raise RetrySourceChanged('This explicit retry is no longer active or its checkpoint is pending; no fresh analysis was submitted')
        if proof.get('kind')=='terminal_error' and not self._recent_runs_reconciled:
            raise RetrySourceChanged('The saved retry generation has not reconciled with Galaxy; no fresh analysis was submitted')
        validate_retry_source(self.service,proof)
        if not current():
            raise RetrySourceChanged('The Live session or retry changed while checking; no fresh analysis was submitted')

    def _submit_live_snapshot(self, run, snapshot):
        if getattr(self, "restore_resume_blocked_reasons", ()):
            raise ValueError("This saved watch is blocked from new analyses until its saved restrictions are reviewed")
        launch_check = getattr(self, "_before_live_submit", None)
        admission_pause = getattr(self, "_submission_pause_event", None)
        admission_allowed = self._retry_submission_allowed
        retry = bool(run.scan_metadata.get('retry_source_proof'))
        if launch_check is None and not retry and admission_pause is None:
            return self.service.submit_snapshot(snapshot)

        def before_submit():
            if admission_pause is not None and (admission_pause.is_set() or not admission_allowed()):
                from .galaxy_service import SubmissionDeferred
                raise SubmissionDeferred("Live owner no longer admits a new Analyze job")
            if launch_check is not None:
                launch_check()
            if retry:
                self._validate_retry_submission(run)

        return self.service.submit_snapshot(snapshot, before_submit=before_submit,
            **({'admission_pause': admission_pause} if admission_pause is not None else {}))

    def restore_latest_state(self) -> bool:
        """Recover the newest matching checkpoint and acknowledged Galaxy jobs.

        Checkpoints are ordinary Galaxy datasets, so this remains recoverable
        after a NOVA browser refresh without writing anything into the IPTS.
        Pending discoveries and acknowledged Galaxy jobs are both restored.
        A stored Galaxy job ID is sufficient to poll a job directly, so
        recovery does not depend on the job still appearing in a bounded list
        of recent History jobs.
        """

        restored_dataset_id = ""
        restored: WatchState | None = None
        for row in self._checkpoint_candidates():
            dataset_id = str(row.get("id") or "")
            if not dataset_id:
                continue
            restricted_match = False
            try:
                payload = self.service.load_json_document(dataset_id)
                watch = payload.get("watch") if isinstance(payload, Mapping) else None
                if not isinstance(watch, Mapping) and self.settings.selected_sample_ids is not None:
                    continue
                if isinstance(watch, Mapping):
                    if watch.get("checkpoint_key") and watch["checkpoint_key"] != self._checkpoint_key():
                        continue
                    if watch.get("instrument", "PG3") != self.settings.instrument:
                        continue
                    if any(watch.get(key) != value for key,value in self._watch_extension().items()):
                        continue
                    if watch.get("catalog_sha256") != self._catalog_digest():
                        continue
                    if watch.get("resolver") != self.resolver_version:
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
                    restricted_match = bool((watch.get("resume_blocked_reasons") or watch.get("missing_settings"))
                        and payload.get("history_id") == self.settings.history_id
                        and payload.get("instrument", "PG3") == self.settings.instrument)
                candidate = WatchState.from_dict(payload)
                if candidate.history_id != self.settings.history_id or candidate.instrument != self.settings.instrument:
                    continue
            except (OSError, TypeError, ValueError) as exc:
                if restricted_match:
                    self.restore_checkpoint_dataset_id = dataset_id
                    self.restore_requires_explicit_inspection = True
                    self.restore_resume_blocked_reasons = ("The matching restricted checkpoint could not be validated",)
                    raise ValueError("The matching saved watch needs review before any new analysis. "
                        "Use Restore a saved Live watch to inspect its invalid checkpoint.") from exc
                continue
            # A matching saved watch with unresolved restrictions is not an
            # absent checkpoint. Returning False would let Start make a fresh
            # controller and duplicate its work. Retain it read-only and stop.
            blocked = []
            missing = ()
            if isinstance(watch, Mapping):
                declared = watch.get("resume_blocked_reasons", [])
                if not isinstance(declared, list) or any(not isinstance(v, str) or not v.strip() for v in declared):
                    blocked.append("Invalid saved resume restriction")
                else:
                    blocked.extend(declared)
                declared_missing = watch.get("missing_settings", [])
                if not isinstance(declared_missing, list) or any(not isinstance(v, str) or not v.strip() for v in declared_missing):
                    blocked.append("Invalid saved missing-settings declaration")
                elif declared_missing:
                    missing = tuple(declared_missing)
                    blocked.append("Original settings are missing: " + ", ".join(missing))
                if watch.get("continuation_review"):
                    from .live_settings_review import validate_continuation_receipt
                    try:
                        validate_continuation_receipt(watch["continuation_review"], self.settings)
                    except (TypeError, ValueError) as exc:
                        blocked.append(str(exc))
            if blocked:
                self._adopt_restored_state(candidate, dataset_id, hydrate_metadata=False)
                self.restore_checkpoint_dataset_id = dataset_id
                self.restore_missing_settings = missing
                self.restore_resume_blocked_reasons = tuple(dict.fromkeys(blocked))
                self.restore_requires_explicit_inspection = True
                self._last_persisted_payload_sha256 = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                raise ValueError("The matching saved watch needs review: " + "; ".join(self.restore_resume_blocked_reasons)
                    + ". Use Restore a saved Live watch to inspect this checkpoint; monitoring remains stopped.")
            if isinstance(watch, Mapping) and watch.get("manual_setup_signature"):
                self._manual_setup_signature = tuple(watch["manual_setup_signature"])
            restored = candidate
            restored_dataset_id = dataset_id
            break
        if restored is None:
            return False

        self._adopt_restored_state(restored, restored_dataset_id, hydrate_metadata=True)
        if isinstance(watch, Mapping) and isinstance(watch.get("continuation_review"), Mapping):
            self.restore_settings_review = json.loads(json.dumps(watch["continuation_review"]))
        self._last_persisted_payload_sha256 = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return True

    def _adopt_restored_state(self, restored: WatchState, dataset_id: str, *, hydrate_metadata: bool) -> None:
        """Adopt already validated state without starting or submitting work."""
        if restored.history_id != self.settings.history_id or restored.instrument != self.settings.instrument:
            raise ValueError("The checkpoint belongs to another History or beamline")
        self.state = restored
        if hydrate_metadata:
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

        self.records = {}
        recoverable = {**restored.submitted, **restored.completed}
        if not hydrate_metadata:
            recoverable.update(restored.failed)
        for run_id, run in recoverable.items():
            if run.galaxy_job_id:
                record = self._minimal_record(run)
                if not hydrate_metadata:
                    # These are saved statuses, not a new Galaxy status query.
                    if run_id in restored.completed:
                        record.status = record.analysis_status = RunStatus.OK
                        record.stage = "Saved completed job; refresh results to check Galaxy"
                    elif run_id in restored.failed:
                        record.status = record.analysis_status = RunStatus.ERROR
                        record.message = run.error or "Saved failed job"
                        record.stage = "Saved failed job; refresh results to check Galaxy"
                    else:
                        record.stage = "Saved in-flight job; refresh results to check Galaxy"
                self.records[run_id] = record

        self.state_dataset_ids = [dataset_id]
        self._last_persisted_state = self.state.to_json()

    @property
    def source_directory(self) -> str:
        return self.definition.source_directory

    def discover(self, listing: Iterable[str | Path | Mapping[str, Any]] | None = None, *, checkpoint: bool = True) -> list[WatchedRun]:
        rows = list(listing) if listing is not None else self._list_source_directory()
        # POWGEN autoreduction can expose auxiliary text products before the
        # canonical GSAS file. Waiting for .gsa prevents submitting a partial
        # representation and gives GSAS-II the format it handles directly.
        completed_rows = [
            row
            for row in rows
            if str(row.get("path") or row.get("name") or row).lower().endswith(".gsa")
        ]
        self._refresh_incomplete_discoveries(completed_rows, checkpoint=checkpoint)
        if self.settings.instrument == "PG3":
            self._flag_powgen_revisions(completed_rows, checkpoint=checkpoint)
        if not self.state.initial_backfill_complete:
            available = discover_from_listing(self.definition, completed_rows)
            self._remember_startup_sources(available)
            if available:
                self._startup_run_floor = max(run.run_number for run in available)
            limit = self.settings.initial_backfill_limit
            if limit == 0:
                discovered = []
            else:
                discovered = self._matching_samples(available, limit=limit)
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
            if checkpoint:
                self.persist_state()
            return discovered

        discovered = self._subsequent_discoveries(completed_rows)
        discovered = self._remember_discoveries(self._expand_banks(self._matching_samples(discovered)))
        discovered = self._attach_scan_metadata(discovered)
        if discovered:
            self._last_seen_run_number = max(
                self._last_seen_run_number or 0,
                *(run.run_number for run in discovered),
            )
        return discovered

    def _remember_startup_sources(self, runs):
        """Beamlines with derived-product IDs may persist their startup baseline."""

    def _subsequent_discoveries(self, rows):
        return discover_from_listing(
            self.definition,
            rows,
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

    def _refresh_incomplete_discoveries(self, rows, *, checkpoint=True):
        """Replace an unsubmitted parse-error placeholder when writing finishes.

        A partial multi-bank file initially has no known bank labels. Keeping
        that placeholder beside the completed bank-specific discoveries would
        submit the same run twice. Check pending paths independently of the
        startup backfill floor, which already includes the partial run number.
        Acknowledged jobs and completed results are never replaced here.
        """
        pending = {r.source_path: r for r in self.state.discovered.values()
                   if r.scan_metadata.get("setup_error")}
        if not pending:
            return
        observed = discover_from_listing(self.definition, rows)
        changed = False
        for run in observed:
            if run.source_path not in pending:
                continue
            expanded = self._expand_banks(self._matching_samples([run]))
            if any(r.scan_metadata.get("setup_error") for r in expanded):
                continue
            for key, old in list(self.state.discovered.items()):
                if old.source_path == run.source_path:
                    del self.state.discovered[key]
            self._attach_scan_metadata(self._remember_discoveries(expanded))
            changed = True
        if changed and checkpoint:
            self.persist_state()

    def _flag_powgen_revisions(self, rows, *, checkpoint=True):
        """Keep result provenance when a POWGEN reduction/receipt is replaced."""
        observed = {r.source_path: r for r in discover_from_listing(self.definition, rows)}
        changed = False
        for phase in (self.state.discovered, self.state.submitted, self.state.completed, self.state.failed):
            for key, old in list(phase.items()):
                current = observed.get(old.source_path)
                setup = old.scan_metadata.get(self.setup_metadata_key) or {}
                if current is None or not setup or old.scan_metadata.get("setup_error"):
                    continue
                path = Path(old.source_path)
                signatures = []
                for suffix in (".py", ".xml"):
                    try:
                        stat = path.with_suffix(suffix).stat()
                        signatures.append([stat.st_size, stat.st_mtime_ns])
                    except FileNotFoundError:
                        signatures.append(None)
                expected = [list(setup[k]) if setup.get(k) is not None else None
                            for k in ("reduction_script_signature", "reduction_xml_signature")]
                revision = [current.fingerprint, *signatures]
                if ((current.fingerprint == old.fingerprint and signatures == expected)
                        or old.scan_metadata.get("replacement_revision") == revision):
                    continue
                # An atomic rewrite with identical content is not a new input.
                provenance = old.scan_metadata.get("powgen_bank_provenance") or {}
                known_hash = provenance.get("source_sha256")
                def bounded_digest(candidate, limit):
                    try:
                        with candidate.open("rb") as stream:
                            raw = stream.read(limit + 1)
                    except FileNotFoundError:
                        return ""
                    return sha256(raw).hexdigest() if len(raw) <= limit else None
                same_source = known_hash and bounded_digest(path, 64 * 1024 * 1024) == known_hash
                same_receipts = all(
                    bounded_digest(path.with_suffix(suffix), limit) == setup.get(field, "")
                    for suffix, field, limit in ((".py", "reduction_script_sha256", 262144),
                                                 (".xml", "reduction_xml_sha256", 1024 * 1024))
                )
                if same_source and same_receipts:
                    continue
                notice = ("Source reduction or reduction receipt was replaced after discovery; retained results "
                          "belong to the original input. Start a separately checked watch to reprocess.")
                summary = dict(old.scientific_summary)
                if summary:
                    summary["scientific_status"] = "needs_review"
                    summary["scientific_warnings"] = list(dict.fromkeys([
                        *summary.get("scientific_warnings", []), notice]))
                phase[key] = replace(old, scan_metadata={**old.scan_metadata,
                    "replacement_revision": revision, "source_revision_warning": notice}, scientific_summary=summary)
                changed = True
        if changed and checkpoint:
            self.persist_state()

    def _setup(self, run: WatchedRun):
        path = Path(run.source_path)
        stat = path.stat()
        sidecar = path.with_suffix(".py")
        side_stat = sidecar.stat() if sidecar.is_file() else None
        xml = path.with_suffix('.xml')
        xml_stat = xml.stat() if xml.is_file() else None
        key = (str(path), stat.st_size, stat.st_mtime_ns,
               (side_stat.st_size, side_stat.st_mtime_ns) if side_stat else None,
               (xml_stat.st_size, xml_stat.st_mtime_ns) if xml_stat else None)
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
                                       self.setup_metadata_key: setup.as_dict()})
                    expanded.append(selected)
            except (OSError, ValueError, UnicodeError) as exc:
                expanded.append(replace(run, scan_metadata={**run.scan_metadata, "setup_error": str(exc)}))
        return expanded

    def _matching_samples(self, runs: list[WatchedRun], *, limit=None) -> list[WatchedRun]:
        return select_runs(runs, self.sample_index, self.settings.selected_sample_ids, limit=limit,
            excluded_run_numbers=self.settings.excluded_run_numbers,
            include_background_runs=self.settings.include_background_runs,
            include_ramp_runs=self.settings.include_ramp_runs)

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
            if hasattr(self.sample_index, 'metadata_index') and not run.product_id:
                metadata.update(self.sample_index.metadata_index.enrich(run.run_number, metadata))
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

    def due_submissions(self, *, reserved_run_ids: set[str] | None = None) -> list[WatchedRun]:
        """Return ready runs without exceeding the active Galaxy-job cap."""

        if self.interrupted_retries_pending or getattr(self, "restore_resume_blocked_reasons", ()):
            return []
        reserved = set(reserved_run_ids or ())
        available = max(0, self.settings.max_active_jobs - len(self.state.submitted) - len(reserved))
        ready = [run for run in self.state.retryable_runs()
                 if run.run_id not in reserved
                 and not run.scan_metadata.get('retry_deferred_by_stop')
                 and not run.scan_metadata.get('submission_receipt_pending')
                 and not run.scan_metadata.get("setup_error")
                 and not run.scan_metadata.get("source_revision_warning")
                 and not (run.scan_metadata.get('retry_source_proof',{}).get('kind')=='terminal_error'
                          and not self._recent_runs_reconciled)]
        return ready[:available]

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
            stage = getattr(self.service, "stage_instrument_profile", self.service.upload_document)
            dataset_id = stage(profile_path, label=f"{self.beamline_label} instrument profile")
            self._profile_dataset_ids[key] = dataset_id
            return dataset_id

    def launch_submission(self, run: WatchedRun) -> RunRecord:
        """Create one Galaxy job without moving the watch-state checkpoint.

        This split lets the NOVA monitor launch independent scans concurrently
        while keeping the state transition and immutable Galaxy checkpoint on
        its event-loop thread.  ``submit`` remains the synchronous convenience
        wrapper used by tests and non-interactive callers.
        """

        if run.scan_metadata.get("source_revision_warning"):
            raise ValueError("POWGEN input changed after discovery; check a new watch before reprocessing")
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
        config = config.model_copy(update={"run_name": self._run_name(run)})
        setup = self._setup(run)
        data_path = run.source_path
        acquisition = acquisition_context(run.scan_metadata, instrument='PG3', run_number=run.run_number,
            source=run.source_path, script_sha256=setup.reduction_script_sha256)
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
            if acquisition is not None:
                destination = Path(self._staging.name) / run.fingerprint / run.run_id / Path(run.source_path).name
                original_hash = sha256(Path(run.source_path).read_bytes()).hexdigest()
                data_path = str(stage_manual_acquisition(run.source_path, destination, acquisition,
                    {'schema':'radar-pd-input-provenance/v1', 'instrument':'PG3',
                     'source_bank':bank_provenance['source_bank'], 'source_sha256':original_hash,
                     'warnings':[]}))
                bank_provenance.update(source_sha256=original_hash,
                    data_sha256=sha256(Path(data_path).read_bytes()).hexdigest())
        else:
            resolution, _ = self.resolve_profile(run)
            destination = Path(self._staging.name) / run.fingerprint / run.run_id
            data, profile_path = stage_bank(Path(run.source_path), setup, resolution.source_bank, resolution, destination,
                acquisition=acquisition)
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
        record = self._submit_live_snapshot(run,snapshot)
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
        if record.provenance.get(self.bank_provenance_key) and run.run_id in self.state.discovered:
            current = self.state.discovered[run.run_id]
            self.state.discovered[run.run_id] = replace(current, scan_metadata={
                **current.scan_metadata, self.bank_provenance_key + "_provenance": record.provenance[self.bank_provenance_key]})
        submitted = self.state.mark_submitted(run, record.galaxy_job_id)
        if record.timings_seconds:
            submitted.scan_metadata['orchestration_seconds'] = dict(record.timings_seconds)
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

        expected_name = self._run_name(run)
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
        matches_custom_profile = self._profile_matches_watch(record, run, candidate_instrument)
        return (
            candidate_database == str(self.settings.database_dataset_id or "")
            and candidate_main == str(self.settings.main_cif_dataset_id or "")
            and matches_custom_profile
        )

    def _profile_matches_watch(self, record, run, candidate_instrument):
        return (
            not self.settings.instrument_profile_dataset_id
            or candidate_instrument == self.settings.instrument_profile_dataset_id
        )

    def reconcile_recent_runs(self, *, checkpoint: bool = True) -> int:
        """Recover jobs that Galaxy accepted after a checkpoint write was lost.

        Galaxy jobs and their outputs are authoritative. The immutable watch
        checkpoint is a restart accelerator, but a transient checkpoint upload
        must never make a completed scan disappear from the experiment view.
        """

        if self._recent_runs_reconciled and not any(
            run.scan_metadata.get('submission_receipt_pending') for run in self.state.discovered.values()
        ):
            return 0
        active = {**self.state.discovered, **self.state.submitted}
        if not active:
            self._recent_runs_reconciled = True
            return 0

        recovery_options = {"limit": max(50, min(250, len(active) * 4))}
        # Local transports from older simulator versions may implement the
        # original unfiltered method. Production can avoid hydrating unrelated
        # configurations while retaining the full identity checks below.
        parameters = inspect.signature(self.service.recent_runs).parameters
        if "configuration_dataset_id" in parameters:
            recovery_options["configuration_dataset_id"] = self.settings.configuration_dataset_id
        if "known_terminal_job_states" in parameters:
            active_job_ids = {run.galaxy_job_id for run in active.values() if run.galaxy_job_id}
            terminal_job_states = {}
            for phase, status in ((self.state.completed, RunStatus.OK), (self.state.failed, RunStatus.ERROR)):
                for run_id, run in phase.items():
                    record = self.records.get(run_id)
                    if (run.galaxy_job_id and run.galaxy_job_id not in active_job_ids and record is not None
                            and record.galaxy_job_id == run.galaxy_job_id
                            and record.history_id == self.settings.history_id
                            and record.status == status and record.analysis_status == status):
                        terminal_job_states[run.galaxy_job_id] = status.value
            recovery_options["known_terminal_job_states"] = terminal_job_states
        recovered = self.service.recent_runs(**recovery_options)
        candidates: dict[str, RunRecord] = {}
        runs_by_name = {self._run_name(run).upper(): run_id for run_id, run in active.items()}
        for record in recovered:
            run_id = runs_by_name.get(record.name.strip().upper())
            if run_id is None:
                continue
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
        if state_changed and checkpoint:
            self.persist_state()
        return recovered_count

    def submit(self, run: WatchedRun) -> RunRecord:
        """Launch and durably acknowledge one scan synchronously."""

        return self.acknowledge_submission(run, self.launch_submission(run))

    def refresh(self, *, checkpoint: bool = True, observations_only: bool = False) -> dict[str, RunRecord]:
        """Refresh metadata, or hydrate missing completed observations for a view.

        View activation skips submitted-job polling and resolves the completed
        batch with the same two-worker bound. Ordinary ticks still read only
        two older pending receipts for fairness.
        """
        with self._refresh_lock:
            return self._refresh_metadata_batch(checkpoint=checkpoint, observations_only=observations_only)

    @staticmethod
    def _same_run(run: WatchedRun | None, request: _MetadataRead) -> bool:
        return (run is not None and run.galaxy_job_id == request.job_id
                and run.fingerprint == request.fingerprint)

    @staticmethod
    def _curve_cache_stamp(receipt: Mapping[str, Any]) -> tuple | None:
        try:
            path = Path(str(receipt.get("local_path") or ""))
            if not path.is_file():
                return None
            stat = path.stat()
            return (str(path), receipt.get("sha256"), receipt.get("source_gpx_sha256"),
                    stat.st_size, stat.st_mtime_ns)
        except (OSError, TypeError, ValueError):
            return None

    def _fetch_metadata(self, request: _MetadataRead) -> _MetadataResult:
        """Worker I/O only: no watch-state or shared RunRecord mutation."""
        try:
            if request.kind == "curve":
                payload = self.service.cache_pattern_observation(json.loads(request.receipt_json))
            else:
                record = RunRecord.model_validate_json(request.record_json)
                payload = self._read_scientific_summary(record)
            return _MetadataResult(request, json.dumps(payload, allow_nan=False))
        except Exception as exc:
            return _MetadataResult(request, error=str(exc))

    def _refresh_metadata_batch(self, *, checkpoint: bool, observations_only: bool = False) -> dict[str, RunRecord]:
        state_changed = False
        self.refresh_errors = {}
        new_reads: list[_MetadataRead] = []
        old_reads: list[_MetadataRead] = []
        # Status refresh is intentionally separate from slow summary/curve
        # downloads, so one unreadable result cannot hide later job states.
        for run_id, run in (() if observations_only else list(self.state.submitted.items())):
            record = self.records.get(run_id)
            if run.galaxy_job_id and (record is None or record.galaxy_job_id != run.galaxy_job_id):
                record = self._minimal_record(run)
                self.records[run_id] = record
            if record is None:
                self.refresh_errors[run_id] = "Acknowledged run has no Galaxy job ID"
                continue
            try:
                refreshed = self.service.refresh(record.model_copy(deep=True))
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
                    new_reads.append(_MetadataRead(run_id, str(refreshed.galaxy_job_id or ""), run.fingerprint,
                        "complete", record_json=refreshed.model_dump_json(), result_ids=result_ids))
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
            if run.scientific_summary:
                receipt = run.scientific_summary.get("pattern_observation") or {}
                # Restore just this small artifact after a NOVA session restart.
                if receipt.get("sha256") and hasattr(self.service, "cache_pattern_observation"):
                    stamp = self._curve_cache_stamp(receipt)
                    key = (run_id, str(run.galaxy_job_id or ""))
                    if stamp is None or self._curve_cache_stamps.get(key) != stamp:
                        old_reads.append(_MetadataRead(run_id, key[1], run.fingerprint, "curve",
                            receipt_json=json.dumps(dict(receipt), allow_nan=False)))
                continue
            if not run.galaxy_job_id:
                continue
            record = self.records.get(run_id)
            if record is None or record.galaxy_job_id != run.galaxy_job_id:
                record = self._minimal_record(run)
                self.records[run_id] = record
            try:
                refreshed = self.service.refresh(record.model_copy(deep=True))
                self.records[run_id] = refreshed
                old_reads.append(_MetadataRead(run_id, str(run.galaxy_job_id), run.fingerprint, "summary",
                    record_json=refreshed.model_dump_json()))
            except Exception as exc:
                self.refresh_errors.setdefault(run_id, f"Scientific summary pending: {exc}")
        # New completions take priority. Only two older pending receipts are
        # read per ordinary tick. Explicit view hydration reads the missing
        # completed batch once, still using just two I/O workers below.
        old_reads.sort(key=lambda item: item.run_id)
        if old_reads and not observations_only:
            split = next((i for i, item in enumerate(old_reads) if item.run_id > self._metadata_refresh_cursor), 0)
            old_reads = old_reads[split:] + old_reads[:split]
            old_reads = old_reads[:2]
            self._metadata_refresh_cursor = old_reads[-1].run_id
        reads = new_reads + old_reads
        if reads:
            with ThreadPoolExecutor(max_workers=2) as workers:
                pending = [workers.submit(self._fetch_metadata, request) for request in reads]
                # Preserve request order in checkpoints and summary consumers;
                # completion timing of parallel reads must not reorder scans.
                for future in pending:
                    result = future.result()
                    request = result.request
                    current = (self.state.submitted if request.kind == "complete" else self.state.completed).get(request.run_id)
                    if not self._same_run(current, request):
                        self.refresh_errors[request.run_id] = "Result identity changed while metadata was loading; retrying current state"
                        continue
                    if request.kind == "complete" and (
                            self.records.get(request.run_id) is None
                            or self.records[request.run_id].galaxy_job_id != request.job_id
                            or self.records[request.run_id].status != RunStatus.OK):
                        self.refresh_errors[request.run_id] = "Job state changed while metadata was loading; retrying current state"
                        continue
                    if result.error:
                        label = "Curve observation" if request.kind == "curve" else "Scientific summary"
                        self.refresh_errors[request.run_id] = f"{label} pending: {result.error}"
                        continue
                    try:
                        payload = json.loads(result.payload_json)
                        if request.kind == "curve":
                            receipt = current.scientific_summary.get("pattern_observation") or {}
                            if dict(receipt) != json.loads(request.receipt_json):
                                self.refresh_errors[request.run_id] = "Curve receipt changed while metadata was loading; retrying current state"
                                continue
                            summary = {**current.scientific_summary, "pattern_observation": payload}
                        else:
                            record = RunRecord.model_validate_json(request.record_json)
                            summary = self._decorate_scientific_summary(record, payload)
                        if request.kind == "complete":
                            self.state.mark_completed(request.run_id, request.result_ids,
                                galaxy_job_id=request.job_id, scientific_summary=summary)
                            state_changed = True
                        elif summary and dict(summary) != dict(current.scientific_summary):
                            self.state.set_scientific_summary(request.run_id, summary)
                            state_changed = True
                        stamp = self._curve_cache_stamp(summary.get("pattern_observation") or {})
                        if stamp is not None:
                            self._curve_cache_stamps[(request.run_id, request.job_id)] = stamp
                    except Exception as exc:
                        self.refresh_errors[request.run_id] = f"Scientific summary pending: {exc}"
        if state_changed and checkpoint:
            self.persist_state()
        return dict(self.records)

    def _scientific_summary(self, record: RunRecord) -> dict[str, Any]:
        return self._decorate_scientific_summary(record, self._read_scientific_summary(record))

    def _decorate_scientific_summary(self, record: RunRecord, summary: dict[str, Any]) -> dict[str, Any]:
        """Apply beamline warnings on the serial controller thread."""
        return summary

    def _read_scientific_summary(self, record: RunRecord) -> dict[str, Any]:
        """Load only the small normalized result JSON, never the result archive."""

        dataset_id = str(record.output_dataset_ids.get("summary") or "").strip()
        if not dataset_id:
            return {}
        payload = self.service.load_json_document(dataset_id)
        summary = experiment_scan_summary(payload)
        receipt = summary.get("pattern_observation") or {}
        # Monitor jobs publish the small curve directly, even when the full
        # tables collection is intentionally omitted. Older full jobs still
        # resolve through their original tables collection.
        if receipt.get("sha256") and record.output_dataset_ids.get("pattern_observation"):
            receipt = {**receipt, "dataset_id": record.output_dataset_ids["pattern_observation"]}
        if receipt.get("sha256") and hasattr(self.service, "cache_pattern_observation"):
            try:
                summary["pattern_observation"] = self.service.cache_pattern_observation(
                    receipt, collection_id=record.output_dataset_ids.get("tables"))
            except Exception as exc:
                summary["pattern_observation"] = {**receipt, "collection_id":record.output_dataset_ids.get("tables"), "notice": str(exc)}
        return summary

    def capture_checkpoint(self) -> WatchCheckpoint:
        """Detach the current state before an asynchronous upload starts."""
        with self._checkpoint_lock:
            serialized = self.state.to_json()
            payload = json.loads(serialized)
            payload["watch"] = {
                "settings_schema": "radar-pd-live-settings/v1",
                "settings": asdict(self.settings),
                "facility": self.definition.facility,
                "instrument": self.settings.instrument,
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
                "resolver": self.resolver_version,
                "catalog_sha256": self._catalog_digest(),
                "manual_setup_signature": self._manual_setup_signature,
                "checkpoint_key": self._checkpoint_key(),
                **self._watch_extension(),
            }
            # Refreshing an old read-only checkpoint must not turn defaults
            # used for display into a claim that the original settings exist.
            if getattr(self, "restore_missing_settings", ()):
                payload["watch"].pop("settings", None)
                payload["watch"].pop("settings_schema", None)
                payload["watch"]["missing_settings"] = list(self.restore_missing_settings)
            if getattr(self, "restore_resume_blocked_reasons", ()):
                payload["watch"]["resume_blocked_reasons"] = list(self.restore_resume_blocked_reasons)
            if getattr(self, "restore_settings_review", None):
                payload["watch"]["continuation_review"] = json.loads(json.dumps(self.restore_settings_review))
            self._checkpoint_generation += 1
            return WatchCheckpoint(self._checkpoint_generation, serialized, json.dumps(payload))

    def persist_state(self, snapshot: WatchCheckpoint | None = None) -> str:
        """Write one detached checkpoint; stale writers cannot replace newer state."""
        snapshot = snapshot or self.capture_checkpoint()
        with self._persist_lock:
            payload = json.loads(snapshot.payload_json)
            payload_sha256 = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if self.state_dataset_ids and (
                snapshot.generation <= self._persisted_checkpoint_generation
                or payload_sha256 == getattr(self, "_last_persisted_payload_sha256", None)
            ):
                self._persisted_checkpoint_generation = max(self._persisted_checkpoint_generation, snapshot.generation)
                return self.state_dataset_ids[-1]
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    uploader = getattr(self.service, 'upload_checkpoint_json_document', self.service.upload_json_document)
                    dataset_id = uploader(
                        payload,
                        name=f"{self.settings.ipts}_{self.beamline_label.lower()}_watch_state.json",
                        label=f"{self.beamline_label} watch state {self.settings.ipts} [watch:{self._checkpoint_key()}]",
                    )
                    self.state_dataset_ids.append(dataset_id)
                    self._last_persisted_state = snapshot.state_json
                    self._last_persisted_payload_sha256 = payload_sha256
                    self._persisted_checkpoint_generation = snapshot.generation
                    return dataset_id
                except CheckpointUploadPending:
                    # Its POST was acknowledged. Retain that HDA and let a
                    # later checkpoint offer poll it; do not duplicate uploads.
                    raise
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
