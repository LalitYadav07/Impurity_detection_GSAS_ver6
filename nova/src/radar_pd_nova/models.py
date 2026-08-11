"""Typed state shared by the NOVA UI and Galaxy service."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class AnalysisMode(str, Enum):
    FULL = "full"
    RAPID = "rapid"


class Radiation(str, Enum):
    NEUTRON = "neutron"
    XRAY = "xray"


class InputSource(str, Enum):
    UPLOAD = "upload"
    GALAXY = "galaxy"
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


class InputSelection(BaseModel):
    """File or IPTS inputs selected by the user."""

    source: InputSource = InputSource.UPLOAD
    data_path: str | None = None
    data_dataset_id: str | None = None
    instrument_path: str | None = None
    instrument_dataset_id: str | None = None
    main_cif_path: str | None = None
    main_cif_dataset_id: str | None = None
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

    @model_validator(mode="after")
    def validate_source(self) -> "InputSelection":
        if self.source in {InputSource.UPLOAD, InputSource.GALAXY}:
            if not (self.data_path or self.data_dataset_id):
                raise ValueError("Select a diffraction pattern")
            if not self.use_builtin_cuka and not (self.instrument_path or self.instrument_dataset_id):
                raise ValueError("Select an instrument profile or use the built-in Cu K-alpha profile")
        elif self.source == InputSource.IPTS_EVENT:
            if not (self.event_file_path or self.event_dataset_id):
                raise ValueError("Select a NeXus event file")
            if not self.bank:
                raise ValueError("Select a detector bank")
        elif self.source == InputSource.IPTS_MANUAL:
            if not all((self.instrument, self.ipts, self.run_number, self.bank)):
                raise ValueError("Instrument, IPTS, run number, and bank are required")
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
                "denominators": [2, 3, 4],
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


class RunRecord(BaseModel):
    """Recoverable reference to one Galaxy-backed RADAR-PD run."""

    uid: str
    name: str
    mode: AnalysisMode
    history_id: str
    status: RunStatus = RunStatus.NEW
    stage: str = "Preparing submission"
    progress: int = Field(default=0, ge=0, le=100)
    created_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    input_dataset_ids: dict[str, str] = Field(default_factory=dict)
    output_dataset_ids: dict[str, str] = Field(default_factory=dict)
    output_dir: str | None = None
    message: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode.value.title(),
            "status": self.status.value.title(),
            "stage": self.stage,
            "progress": f"{self.progress}%",
            "uid": self.uid,
        }

    @property
    def local_output_dir(self) -> Path | None:
        return Path(self.output_dir) if self.output_dir else None
