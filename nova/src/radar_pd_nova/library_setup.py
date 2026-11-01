"""Independent workflow drafts for the common candidate-library builder."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Callable

LIBRARY_FIELDS = (
    "database_source",
    "library_archive_source",
    "database_archive_path",
    "library_builder_cif_ids",
    "library_builder_local_paths",
    "library_builder_name",
    "library_builder_mode",
    "library_builder_active",
    "library_builder_status",
    "library_builder_progress",
    "library_builder_built_count",
    "library_builder_skipped_count",
    "library_builder_failure_rows",
    "library_builder_message",
    "library_builder_file_rows",
    "library_builder_elements",
    "library_builder_unknown_elements",
    "library_builder_requested_count",
    "library_builder_duplicate_count",
    "library_builder_duplicate_rows",
    "library_builder_generation",
    "library_builder_partial_review",
    "library_builder_previous_library",
)


class LibraryManifestReadError(ValueError):
    """The completed build's manifest could not be retrieved and verified."""


async def read_library_manifest(
    reader: Callable[[str], Any],
    dataset_id: str | None,
    *,
    on_retry: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Re-read publication once; never resubmit the completed scientific job."""
    if not dataset_id:
        raise LibraryManifestReadError("The completed build has no manifest output.")
    failure = "empty or non-object document"
    for attempt in range(2):
        try:
            document = await asyncio.to_thread(reader, dataset_id)
        except Exception as exc:
            # Preserve the failure category without exposing response bodies or
            # credential-bearing request details in the UI.
            failure = type(exc).__name__
        else:
            if isinstance(document, dict) and document:
                return document
            failure = "empty or non-object document"
        if attempt == 0:
            if on_retry:
                on_retry(failure)
            await asyncio.sleep(1.0)
    raise LibraryManifestReadError(f"Manifest retrieval failed after two reads ({failure}).")


def library_manifest_phase_count(manifest: dict[str, Any]) -> int:
    """Require consistent integral counts, including IDs when supplied."""
    counts = []
    for key in ("phase_count", "n_phases"):
        if key not in manifest:
            continue
        value = manifest[key]
        if isinstance(value, bool):
            raise ValueError(f"Manifest {key} must be a non-negative integer.")
        if isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
            value = int(value)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Manifest {key} must be a non-negative integer.")
        counts.append(value)
    if not counts:
        raise ValueError("Manifest contains no usable phase count.")
    if len(set(counts)) != 1:
        raise ValueError("Manifest phase counts disagree.")
    count = counts[0]
    if "phase_ids" in manifest:
        identifiers = manifest["phase_ids"]
        if (
            not isinstance(identifiers, list)
            or any(not isinstance(item, str) or not item.strip() for item in identifiers)
            or len(set(identifiers)) != len(identifiers)
            or len(identifiers) != count
        ):
            raise ValueError("Manifest phase IDs do not match its usable phase count.")
    return count


class LibraryDraft:
    """Give the existing builder one explicit destination for all state changes."""

    def __init__(self, state: Any, workflow: str = "single"):
        if workflow not in {"single", "live"}:
            raise ValueError("Unknown candidate-library workflow")
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "workflow", workflow)

    def _key(self, name: str) -> str:
        if self.workflow == "live":
            if name == "history_database_id":
                return "powgen_database_dataset_id"
            if name in LIBRARY_FIELDS or name == "radiation":
                return "live_" + name
        return name

    def __getattr__(self, name: str) -> Any:
        return getattr(self._state, self._key(name))

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._state, self._key(name), value)

    def signature(self) -> str:
        return json.dumps(
            {
                name: getattr(self, name, None)
                for name in (
                    "database_source",
                    "library_builder_name",
                    "library_builder_cif_ids",
                    "library_builder_local_paths",
                    "radiation",
                )
            },
            sort_keys=True,
        )
