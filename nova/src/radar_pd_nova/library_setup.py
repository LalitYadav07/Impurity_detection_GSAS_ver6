"""Independent workflow drafts for the common candidate-library builder."""

from __future__ import annotations

import json
from typing import Any

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
)


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
