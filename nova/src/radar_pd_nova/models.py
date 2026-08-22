"""Typed state shared by the NOVA UI and Galaxy service."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AnalysisMode(str, Enum):
    FULL = "full"
    RAPID = "rapid"


class Radiation(str, Enum):
    NEUTRON = "neutron"
    XRAY = "xray"


class InputSource(str, Enum):
    UPLOAD = "upload"
    GALAXY = "galaxy"
    GALAXY_REMOTE = "galaxy_remote"
    IPTS_BROWSER = "ipts_browser"
    IPTS_EVENT = "ipts_event"
    IPTS_MANUAL = "ipts_manual"


class RunStatus(str, Enum):
    NEW = "new"
    UPLOADING = "uploading"
    QUEUED = "queued"
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"
    CANCELLED = "cancelled"


class ResultStatus(str, Enum):
    """State of result collection, independent of the Galaxy analysis job."""

    NOT_REQUESTED = "not_requested"
    COLLECTING = "collecting"
    READY = "ready"
    STALE = "stale"
    ERROR = "error"


class SubmissionPhase(str, Enum):
    """User-visible phases before Galaxy assigns an Analyze job identifier."""

    VALIDATING = "validating"
    UPLOADING_CONFIGURATION = "uploading_configuration"
    UPLOADING_DATA = "uploading_data"
    UPLOADING_INSTRUMENT = "uploading_instrument"
    UPLOADING_OPTIONAL = "uploading_optional"
    WAITING_FOR_GALAXY = "waiting_for_galaxy"
    ACKNOWLEDGED = "acknowledged"
    CANCELLED = "cancelled"
    ERROR = "error"


class InputSelection(BaseModel):
    """File or IPTS inputs selected by the user."""

    source: InputSource = InputSource.UPLOAD
    instrument_source: Literal["upload", "galaxy", "galaxy_remote", "ipts", "builtin"] = "upload"
    data_path: str | None = None
    data_dataset_id: str | None = None
    data_remote_uri: str | None = None
    instrument_path: str | None = None
    instrument_dataset_id: str | None = None
    instrument_remote_uri: str | None = None
    main_cif_path: str | None = None
    main_cif_dataset_id: str | None = None
    main_cif_remote_uri: str | None = None
    database_archive_path: str | None = None
    database_dataset_id: str | None = None
    use_builtin_cuka: bool = False
    event_file_path: str | None = None
    event_dataset_id: str | None = None
    facility_root: str = "/SNS"
    instrument: str | None = None
    ipts: str | None = None
    run_number: int | None = None
    bank: str | None = None
    data_relative_path: str | None = None
    instrument_relative_path: str | None = None
    main_cif_relative_path: str | None = None
    publish_results_to_ipts: bool = False
    publish_directory: str | None = None
    publish_subfolder: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "InputSelection":
        if self.source in {
            InputSource.UPLOAD,
            InputSource.GALAXY,
            InputSource.GALAXY_REMOTE,
            InputSource.IPTS_BROWSER,
        }:
            if not (self.data_path or self.data_dataset_id or self.data_remote_uri):
                raise ValueError("Select a diffraction pattern")
            if not self.use_builtin_cuka and not (
                self.instrument_path or self.instrument_dataset_id or self.instrument_remote_uri
            ):
                raise ValueError("Select an instrument profile or use the built-in Cu K-alpha profile")
            if self.source == InputSource.GALAXY_REMOTE and not self.data_remote_uri:
                raise ValueError("Select a diffraction pattern from a Galaxy remote file source")
            if self.source == InputSource.IPTS_BROWSER:
                if not all((self.instrument, self.ipts, self.data_relative_path)):
                    raise ValueError("Select a facility instrument, IPTS, and diffraction file")
        elif self.source == InputSource.IPTS_EVENT:
            if not (self.event_file_path or self.event_dataset_id):
                raise ValueError("Select a NeXus event file")
            if not self.bank:
                raise ValueError("Select a detector bank")
        elif self.source == InputSource.IPTS_MANUAL:
            if not all((self.instrument, self.ipts, self.run_number, self.bank)):
                raise ValueError("Instrument, IPTS, run number, and bank are required")
        if self.publish_results_to_ipts and not all(
            (self.instrument, self.ipts, self.publish_directory)
        ):
            raise ValueError("Select an IPTS result destination before enabling result publishing")
        return self


class AnalysisConfig(BaseModel):
    """Complete user-facing configuration for Full and Rapid RADAR-PD."""

    run_name: str = Field(default_factory=lambda: datetime.now().strftime("radar_%Y%m%d_%H%M%S"))
    mode: AnalysisMode = AnalysisMode.RAPID
    radiation: Radiation = Radiation.NEUTRON
    instrument_mode: Literal["auto", "cw", "tof"] = "auto"
    sample_elements: list[str] = Field(min_length=1)
    environment_elements: list[str] = Field(default_factory=list)
    limits: tuple[float, float] | None = None
    exclude_regions: list[tuple[float, float]] = Field(default_factory=list)
    reference_masks_enabled: bool = False
    reference_mask_presets: list[Literal["Al_fcc", "Cu_fcc", "V_bcc"]] = Field(default_factory=list)
    reference_window_mode: Literal["auto", "fixed"] = "auto"
    include_cu_kbeta: bool = False
    background_mode: str = "auto_fixed_points"
    background_type: str = "chebyschev-1"
    background_terms: int = Field(default=6, ge=1, le=36)
    main_prenudge: bool = True
    main_shadow_filter: bool = True
    cleanup_enabled: bool = False
    refine_u_iso: bool = False
    refine_positions: bool = False
    magnetic_precheck: bool = False
    magnetic_q_max: float = Field(default=4.0, gt=0)
    magnetic_denominators: list[int] = Field(default_factory=lambda: [2, 3, 4])
    full_profile: Literal["quick", "balanced", "thorough", "custom"] = "balanced"
    full_max_passes: int = Field(default=2, ge=1, le=20)
    full_min_phase_percent: float = Field(default=0.5, ge=0)
    full_top_n_ml: int = Field(default=35, ge=1)
    full_nudge_candidates: int = Field(default=7, ge=1)
    full_nudge_samples: int = Field(default=5000, ge=1)
    full_nudge_representatives: int = Field(default=50, ge=1)
    full_compare_candidates: int = Field(default=2, ge=1)
    full_compare_cycles: int = Field(default=6, ge=1)
    full_cell_length_tolerance_pct: float = Field(default=1.0, gt=0)
    full_cell_angle_tolerance_deg: float = Field(default=3.0, gt=0)
    full_rwp_improvement_threshold: float = Field(default=0.06, ge=0)
    rapid_phases_per_hypothesis: int = Field(default=3, ge=1, le=5)
    rapid_stage_output_limit: int = Field(default=10, ge=3, le=50)
    rapid_gsas_validation_limit: int = Field(default=5, ge=0)
    rapid_parallel_workers: int = Field(default=4, ge=1, le=16)
    rapid_show_family_variants: bool = True
    rapid_final_polish_enabled: bool = False

    @field_validator("sample_elements", "environment_elements", mode="before")
    @classmethod
    def normalize_elements(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = value.replace(",", " ").split()
        normalized: list[str] = []
        for raw in value:
            token = str(raw).strip().capitalize()
            if token and token not in normalized:
                normalized.append(token)
        return normalized

    @model_validator(mode="after")
    def validate_ranges(self) -> "AnalysisConfig":
        if self.limits and self.limits[0] >= self.limits[1]:
            raise ValueError("Fit range start must be less than fit range end")
        for start, end in self.exclude_regions:
            if start >= end:
                raise ValueError("Ignored-region start must be less than end")
        denominators = list(dict.fromkeys(int(value) for value in self.magnetic_denominators))
        if not denominators or any(value < 2 for value in denominators):
            raise ValueError("Magnetic denominators must contain integers greater than or equal to 2")
        self.magnetic_denominators = denominators
        return self

    def portable_contract(self) -> dict[str, Any]:
        """Return the exact `radar-pd-config/v1` contract consumed by NDIP."""

        return {
            "$schema": "radar-pd-config/v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "analysis": {
                "mode": self.mode.value,
                "radiation": self.radiation.value,
                "instrument_mode": self.instrument_mode,
            },
            "chemistry": {
                "sample_elements": self.sample_elements,
                "environment_elements": self.environment_elements,
            },
            "pattern": {
                "limits": list(self.limits) if self.limits else None,
                "exclude_regions": [list(region) for region in self.exclude_regions],
                "reference_phase_exclusions": {
                    "enabled": self.reference_masks_enabled,
                    "presets": self.reference_mask_presets,
                    "window_mode": self.reference_window_mode,
                    "include_cu_kbeta": self.include_cu_kbeta,
                },
            },
            "background": {
                "mode": self.background_mode,
                "type": self.background_type,
                "terms": self.background_terms,
            },
            "main_phase": {
                "prenudge": self.main_prenudge,
                "shadow_filter": self.main_shadow_filter,
                "cleanup": {
                    "enabled": self.cleanup_enabled,
                    "refine_u_iso": self.refine_u_iso,
                    "refine_positions": self.refine_positions,
                },
            },
            "magnetic_precheck": {
                "enabled": self.magnetic_precheck,
                "q_max": self.magnetic_q_max,
                "denominators": self.magnetic_denominators,
            },
            "full": {
                "profile": self.full_profile,
                "max_passes": self.full_max_passes,
                "min_phase_percent": self.full_min_phase_percent,
                "top_n_ml": self.full_top_n_ml,
                "nudge_candidates": self.full_nudge_candidates,
                "cell_length_tolerance_pct": self.full_cell_length_tolerance_pct,
                "cell_angle_tolerance_deg": self.full_cell_angle_tolerance_deg,
                "nudge_samples": self.full_nudge_samples,
                "nudge_representatives": self.full_nudge_representatives,
                "compare_candidates": self.full_compare_candidates,
                "compare_cycles": self.full_compare_cycles,
                "rwp_improvement_threshold": self.full_rwp_improvement_threshold,
            },
            "rapid": {
                "phases_per_hypothesis": self.rapid_phases_per_hypothesis,
                "stage_output_limit": self.rapid_stage_output_limit,
                "gsas_validation_limit": self.rapid_gsas_validation_limit,
                "parallel_workers": self.rapid_parallel_workers,
                "show_family_variants": self.rapid_show_family_variants,
                "final_polish_enabled": self.rapid_final_polish_enabled,
            },
        }


class SubmissionSnapshot(BaseModel):
    """Immutable, button-time input used for one idempotent submission."""

    model_config = ConfigDict(frozen=True)

    config: AnalysisConfig
    inputs: InputSelection
    client_revision: int = Field(default=0, ge=0)
    display_summary: dict[str, str] = Field(default_factory=dict)
    idempotency_token: str = Field(min_length=8)
    created_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @model_validator(mode="after")
    def detach_mutable_models(self) -> "SubmissionSnapshot":
        # Frozen prevents attribute replacement. Deep copies ensure subsequent
        # Trame state changes cannot mutate nested values captured by a click.
        object.__setattr__(self, "config", self.config.model_copy(deep=True))
        object.__setattr__(self, "inputs", self.inputs.model_copy(deep=True))
        object.__setattr__(self, "display_summary", dict(self.display_summary))
        return self


class SubmissionProgress(BaseModel):
    phase: SubmissionPhase = SubmissionPhase.VALIDATING
    label: str = "Validating run plan"
    completed_items: int = Field(default=0, ge=0)
    total_items: int = Field(default=0, ge=0)
    started_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    elapsed_seconds: float = Field(default=0.0, ge=0)
    galaxy_job_id: str | None = None


class CacheManifest(BaseModel):
    job_id: str
    archive_dataset_id: str
    archive_size: int | None = Field(default=None, ge=0)
    archive_update_time: str | None = None
    adapter_version: str = "0.3.27"
    collected_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class GalaxyArtifactRef(BaseModel):
    artifact_id: str
    source_type: Literal["dataset", "collection_element", "archive"] = "dataset"
    scientific_role: str
    display_name: str
    archive_path: str | None = None
    dataset_id: str | None = None
    collection_id: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class UtilityActionRecord(BaseModel):
    uid: str
    tool_id: str
    name: str
    associated_run_uid: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    galaxy_job_id: str | None = None
    entrypoint_id: str | None = None
    status: RunStatus = RunStatus.NEW
    outputs: dict[str, str] = Field(default_factory=dict)
    message: str = ""
    created_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RunRecord(BaseModel):
    """Recoverable reference to one Galaxy-backed RADAR-PD run."""

    uid: str
    galaxy_job_id: str | None = None
    name: str
    mode: AnalysisMode
    history_id: str
    status: RunStatus = RunStatus.NEW
    analysis_status: RunStatus | None = None
    result_status: ResultStatus = ResultStatus.NOT_REQUESTED
    stage: str = "Preparing submission"
    progress: int = Field(default=0, ge=0, le=100)
    created_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    input_dataset_ids: dict[str, str] = Field(default_factory=dict)
    output_dataset_ids: dict[str, str] = Field(default_factory=dict)
    config: AnalysisConfig | None = None
    inputs: InputSelection | None = None
    output_dir: str | None = None
    message: str = ""
    console_tail: str = ""
    submission: SubmissionProgress | None = None
    idempotency_token: str | None = None
    prepared_dataset_ids: dict[str, str] = Field(default_factory=dict)
    cancel_requested: bool = False
    cache_manifest: CacheManifest | None = None
    recovery_diagnostics: list[str] = Field(default_factory=list)
    published_output_dir: str | None = None
    publish_message: str = ""
    publication_job_id: str | None = None
    publication_status: RunStatus | None = None
    publication_target: str | None = None

    @model_validator(mode="after")
    def adapt_legacy_record(self) -> "RunRecord":
        if self.galaxy_job_id is None and not self.uid.startswith("pending-"):
            self.galaxy_job_id = self.uid
        if self.analysis_status is None:
            self.analysis_status = self.status
        return self

    def as_row(self) -> dict[str, Any]:
        timestamp = self.created_utc.replace("T", " ")[:16]
        short_job = (self.galaxy_job_id or "pending")[:8]
        return {
            "name": self.name,
            "mode": self.mode.value.title(),
            "status": self.status.value.title(),
            "stage": self.stage,
            "progress": f"{self.progress}%",
            "uid": self.uid,
            "job_id": self.galaxy_job_id or "Pending",
            "created": self.created_utc,
            "display_name": f"{self.name} / {self.mode.value.title()} / {timestamp} / {self.status.value.title()} / {short_job}",
        }

    @property
    def local_output_dir(self) -> Path | None:
        return Path(self.output_dir) if self.output_dir else None


def selected_run_uid(value: Any) -> str:
    """Normalize Vuetify selection payloads across client versions."""

    selected = value
    if isinstance(selected, dict):
        selected = selected.get("value", selected.get("modelValue", selected.get("item", selected)))
    if isinstance(selected, (list, tuple)):
        selected = selected[0] if selected else ""
    if isinstance(selected, dict):
        raw = selected.get("raw", selected)
        selected = raw.get("uid", raw.get("value", "")) if isinstance(raw, dict) else raw
    return str(selected or "")
