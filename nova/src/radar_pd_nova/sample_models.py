"""Explicit sample-to-model routing, independent of names and fit quality.

Each library is the allowed candidate scope for that sample. A null known main
means blind discovery inside that library; it never inherits another sample's
main phase. Galaxy references are bound to their endpoint and History.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .facility_metadata import sample_identifier


class SampleModelAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    instrument: Literal["PG3", "NOM", "HB2A", "SNAP"]
    ipts: str
    sample_id: str
    candidate_library_dataset_id: str | None
    known_main_cif_dataset_id: str | None
    acquisition_ids: tuple[str, ...] = Field(default=(), max_length=10000)

    @field_validator("acquisition_ids")
    @classmethod
    def valid_acquisitions(cls, value):
        if any(not re.fullmatch(r"HB2A_EXP[1-9]\d*_SCAN\d{4,}", item) for item in value):
            raise ValueError("Use observed HB-2A experiment/scan identities for acquisition-scoped models")
        return tuple(sorted(set(value)))

    @field_validator("ipts")
    @classmethod
    def valid_ipts(cls, value):
        if not re.fullmatch(r"IPTS-[1-9]\d*", value):
            raise ValueError("Use the exact IPTS identifier for each sample model assignment")
        return value

    @field_validator("sample_id")
    @classmethod
    def valid_sample(cls, value):
        if not value or sample_identifier(value) != value:
            raise ValueError("Use a recorded sample ID, preserving its exact text")
        return value

    @field_validator("candidate_library_dataset_id", "known_main_cif_dataset_id")
    @classmethod
    def valid_dataset(cls, value):
        if value is not None and (not value or len(value) > 256 or re.search(r"\s|[/\\]", value)):
            raise ValueError("Model references must be Galaxy dataset IDs, or null")
        return value

    @property
    def key(self):
        return self.instrument, self.ipts, self.sample_id

    @model_validator(mode="after")
    def acquisition_beamline(self):
        if self.acquisition_ids and self.instrument != "HB2A":
            raise ValueError("Explicit acquisition scopes currently apply to HB-2A experiment/scan identities")
        return self


class SampleModelAssignments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    galaxy_url: str
    history_id: str = Field(min_length=1, max_length=256)
    assignments: tuple[SampleModelAssignment, ...] = Field(min_length=1, max_length=1000)

    @field_validator("galaxy_url")
    @classmethod
    def valid_url(cls, value):
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.netloc
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Sample model references require a Galaxy endpoint without credentials")
        return value.rstrip("/")

    @model_validator(mode="after")
    def unique_samples(self):
        groups = {}
        acquisitions = set()
        for entry in self.assignments:
            for acquisition in entry.acquisition_ids:
                key = (entry.instrument, entry.ipts, acquisition)
                if key in acquisitions:
                    raise ValueError("Acquisition model assignments must not overlap, including across sample labels")
                acquisitions.add(key)
            for prior in groups.setdefault(entry.key, []):
                if (not prior.acquisition_ids or not entry.acquisition_ids
                        or set(prior.acquisition_ids).intersection(entry.acquisition_ids)):
                    raise ValueError("A sample may have only one model assignment per acquisition; acquisition scopes must not overlap")
            groups[entry.key].append(entry)
        return self


@dataclass(frozen=True)
class SampleModelBinding:
    instrument: str
    ipts: str
    sample_id: str
    candidate_library_dataset_id: str | None
    known_main_cif_dataset_id: str | None
    mode: str
    acquisition_id: str | None = None

    def receipt(self):
        value = asdict(self)
        value["binding_sha256"] = sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        return value


def name_only_identity(metadata):
    return (":NAME:" in str(metadata.get("sample_id") or "")
            or "no sample ID recorded" in str(metadata.get("identity_basis") or ""))


def assigned_entry(routing, instrument, ipts, metadata, acquisition_id=None):
    entries = [entry for entry in routing.assignments
               if entry.key == (instrument, ipts, metadata.get("sample_id"))
               and (not entry.acquisition_ids or acquisition_id in entry.acquisition_ids)]
    if not entries:
        raise ValueError(f"No model assignment for sample {metadata.get('sample_id')} and acquisition {acquisition_id or 'not recorded'}; review its allowed library and known main CIF")
    if name_only_identity(metadata) and not entries[0].acquisition_ids:
        raise ValueError("A name-only sample label requires an explicit reviewed acquisition selection; it is not a verified physical sample ID")
    return entries[0]


def single_sample_cohort(metadata):
    """Reduce a completely source-verified cohort, without using its names."""
    rows = list(metadata)
    if not rows or any((row.get("identity_status") or row.get("status")) != "resolved"
            or not row.get("sample_id") or row.get("identity_verification") == "catalog_preview"
            for row in rows):
        raise ValueError("The shared model needs a verified sample ID for every selected scan; review unknown or conflicting identities")
    if any(name_only_identity(row) for row in rows):
        raise ValueError("Name-only sample labels require reviewed acquisition-scoped model assignments")
    samples = {row["sample_id"] for row in rows}
    if len(samples) != 1:
        raise ValueError("The selected cohort contains multiple sample IDs; assign each sample its allowed library and known main CIF")
    return (next(iter(samples)),)


def check_model_selection(config, *, instrument, ipts, selected_sample_ids,
                          galaxy_url, history_id, cohort_metadata=()):
    """Validate a draft before staging; later arrivals are checked individually."""
    routing = config.sample_models
    if routing is None:
        if selected_sample_ids is None:
            selected_sample_ids = single_sample_cohort(cohort_metadata)
        if len(selected_sample_ids) != 1:
            raise ValueError("Select one sample for the shared library/main CIF, or import explicit sample model assignments for every sample")
        if any(":NAME:" in sample for sample in selected_sample_ids):
            raise ValueError("Name-only sample labels require reviewed acquisition-scoped model assignments")
        if cohort_metadata and single_sample_cohort(cohort_metadata) != tuple(selected_sample_ids):
            raise ValueError("Verified sample identity no longer matches the selected sample scope")
        return "Shared library and main CIF apply only to sample " + selected_sample_ids[0]
    if routing.galaxy_url != str(galaxy_url).rstrip("/") or routing.history_id != history_id:
        raise ValueError("Sample model dataset references belong to another Galaxy endpoint or History; review and rebind them")
    assignments = {entry.key: entry for entry in routing.assignments}
    selected = selected_sample_ids or tuple(entry.sample_id for entry in routing.assignments
        if (entry.instrument, entry.ipts) == (instrument, ipts))
    if not selected:
        raise ValueError("No sample model assignments match this beamline and IPTS")
    missing = [sample for sample in selected if (instrument, ipts, sample) not in assignments]
    if missing:
        raise ValueError("No model assignment for sample(s): " + ", ".join(missing))
    if config.library_review:
        raise ValueError("A global library policy review cannot authorize per-sample libraries; use qualified libraries without a global policy override")
    if config.tof_profile_mode == "per_scan" and any(
            entry.known_main_cif_dataset_id is None for entry in routing.assignments
            if entry.instrument == instrument and entry.ipts == ipts and entry.sample_id in selected):
        raise ValueError("Per-scan profile refinement requires a supplied main CIF for every assigned sample; use a fixed profile for blind discovery")
    for row in cohort_metadata:
        if (row.get("identity_status") or row.get("status")) != "resolved" or not row.get("sample_id"):
            raise ValueError("Review unknown or conflicting source identities before assigning models")
        assigned_entry(routing, instrument, ipts, row, row.get("run_id"))
    return f"{len(selected)} explicit sample model assignment(s); unassigned arrivals stop for review"


def bind_sample_models(config, *, instrument, ipts, metadata, selected_sample_ids,
                       galaxy_url, history_id, shared_library=None, shared_main=None, acquisition_id=None):
    """Resolve only authoritative identity; formula/name/Rwp are never keys."""
    check_model_selection(config, instrument=instrument, ipts=ipts,
        selected_sample_ids=selected_sample_ids, galaxy_url=galaxy_url, history_id=history_id)
    sample_id = metadata.get("sample_id")
    if (metadata.get("identity_status") != "resolved" or not isinstance(sample_id, str)
            or not sample_id or metadata.get("identity_verification") == "catalog_preview"):
        raise ValueError("A verified source-file sample identity is required before choosing its models")
    if selected_sample_ids is not None and sample_id not in selected_sample_ids:
        raise ValueError("Verified sample identity no longer matches the selected sample scope")
    if config.sample_models is None:
        if name_only_identity(metadata):
            raise ValueError("Name-only sample labels require reviewed acquisition-scoped model assignments")
        return SampleModelBinding(instrument, ipts, sample_id, shared_library, shared_main, "single_sample", acquisition_id)
    entry = assigned_entry(config.sample_models, instrument, ipts, metadata, acquisition_id)
    return SampleModelBinding(instrument, ipts, sample_id,
        entry.candidate_library_dataset_id, entry.known_main_cif_dataset_id, "explicit_assignment", acquisition_id)
