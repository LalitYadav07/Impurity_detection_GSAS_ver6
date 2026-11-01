"""Explicit, source-bound acceptance of incomplete candidate libraries."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
import uuid

from .library_setup import LibraryDraft, library_manifest_phase_count


SELECTION_FIELDS = (
    "database_source", "library_archive_source", "database_archive_path", "history_database_id",
)
EXACT_DUPLICATE_REASONS = frozenset({
    "duplicate structure file", "phase id already exists in base catalog",
})
MATCHING_STRUCTURE_PREFIX = "phase already exists in base database as "


def library_selection(draft: LibraryDraft) -> dict[str, Any]:
    return {name: getattr(draft, name, None) for name in SELECTION_FIELDS}


def library_review_signature(draft: LibraryDraft) -> str:
    """Bind a review to its form and local source identities without remote reads."""
    sources = []
    for value in getattr(draft, "library_builder_local_paths", None) or []:
        path = Path(str(value))
        try:
            stat = path.stat()
            sources.append((str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size,
                            stat.st_mtime_ns, stat.st_ctime_ns))
        except OSError:
            sources.append((str(path), "unavailable"))
    return json.dumps({"form": draft.signature(), "selection": library_selection(draft),
                       "local_sources": sources}, sort_keys=True)


def _failure_row(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {"name": str(value.get("source_name") or value.get("name") or value.get("id") or "CIF"),
                "reason": str(value.get("error") or value.get("reason") or "Could not be added")}
    text = str(value)
    name, separator, reason = text.partition(": ")
    return {"name": name if separator else "CIF", "reason": reason if separator else text}


def library_failure_is_duplicate(value: Any) -> bool:
    """Recognize only the existing producers' explicit deduplication receipts."""
    row = _failure_row(value)
    reason = row["reason"]
    return (reason in EXACT_DUPLICATE_REASONS
            or (reason.startswith(MATCHING_STRUCTURE_PREFIX)
                and bool(reason[len(MATCHING_STRUCTURE_PREFIX):].strip())))


def library_build_outcome(manifest: dict[str, Any], built: int,
                          preflight: dict[str, Any]) -> dict[str, Any]:
    """Separate explicit deduplication from failed or unexplained omissions."""
    preflight_rows = [_failure_row(row) for row in preflight.get("failure_rows") or []]
    native_rows = [_failure_row(row) for row in manifest.get("failures") or []]
    rows = [*preflight_rows, *native_rows]
    unexplained = max(0, int(preflight.get("skipped_count") or 0) - len(preflight_rows))
    rows.extend({"name": "Source preflight", "reason": "A selected source was skipped; its reason was not reported."}
                for _ in range(unexplained))
    duplicate_rows = [row for row in rows if library_failure_is_duplicate(row)]
    failure_rows = [row for row in rows if not library_failure_is_duplicate(row)]
    requested = int(preflight.get("requested_count") or 0)
    requested = max(requested, built + len(rows))
    # A verified count gap is itself incomplete, even if a producer omitted
    # its per-source failure records. Never call it deduplication by inference.
    missing = max(0, requested - built - len(rows))
    failure_rows.extend({"name": "Unreported source", "reason": "The build did not account for this selected CIF; inspect the saved manifest and log."}
                        for _ in range(missing))
    return dict(built_count=built, requested_count=requested,
                skipped_count=len(failure_rows), failure_rows=failure_rows,
                duplicate_count=len(duplicate_rows), duplicate_rows=duplicate_rows)


def partial_library_archive_review(manifest: dict[str, Any]) -> dict[str, Any]:
    """Use the already-read archive manifest; no second read or job is needed.

    Historical manifests describe native builder inputs only. They cannot prove
    that an earlier host upload/preflight retained every originally selected CIF.
    """
    counts = dict(manifest)
    if not any(key in counts for key in ("phase_count", "n_phases")) and "n_added_phases" in counts:
        counts["phase_count"] = counts["n_added_phases"]
    try:
        built = library_manifest_phase_count(counts)
        count_verified = True
    except ValueError:
        built = 0
        count_verified = False
    outcome = library_build_outcome(manifest, built, {})
    return {**outcome, "count_verified": count_verified,
            "requires_review": bool(outcome["skipped_count"]),
            "count_scope": "Declared native builder inputs; earlier upload/preflight omissions are not recorded in historical archives."}


class LibraryReviewMixin:
    def _interrupt_candidate_library(self, draft: LibraryDraft, generation: str) -> None:
        if draft.library_builder_generation != generation:
            return
        draft.library_builder_active = False
        draft.library_builder_partial_review = {}
        draft.library_builder_status = "interrupted"
        draft.library_builder_message = "Library build monitoring stopped. Already accepted work remains in History; no new library was selected."
        draft.flush()

    def _library_build_owner_current(self, action: Any, owner_context: Any, owner: dict[str, Any]) -> bool:
        if owner_context is None:
            return False  # Recovered/unowned output belongs in History, never in another draft.
        contexts = getattr(self, "_library_action_contexts", {})
        if contexts.get(action.uid) is not owner_context or owner_context != owner:
            return False
        draft = LibraryDraft(self.server.state, owner.get("workflow", "single"))
        return not owner.get("generation") or owner["generation"] == draft.library_builder_generation

    def _complete_candidate_library(self, action: Any, owner_context: Any, owner: dict[str, Any],
                                    manifest: dict[str, Any], built: int, manifest_error: str) -> bool:
        if not self._library_build_owner_current(action, owner_context, owner):
            return False
        draft = LibraryDraft(self.server.state, owner.get("workflow", "single"))
        selection_current = (not owner.get("signature") or owner["signature"] == draft.signature())
        selection_current = selection_current and (
            not owner.get("source_signature") or owner["source_signature"] == library_review_signature(draft))
        preflight = owner.get("preflight") or {
            "failure_rows": list(draft.library_builder_failure_rows or []),
            "skipped_count": int(draft.library_builder_skipped_count or 0),
        }
        outcome = library_build_outcome(manifest, built, preflight)
        self._library_action_contexts.pop(action.uid, None)
        draft.library_builder_partial_review = {}
        draft.library_builder_built_count = built
        draft.library_builder_requested_count = outcome["requested_count"]
        draft.library_builder_skipped_count = outcome["skipped_count"]
        draft.library_builder_failure_rows = outcome["failure_rows"]
        draft.library_builder_duplicate_count = outcome["duplicate_count"]
        draft.library_builder_duplicate_rows = outcome["duplicate_rows"]
        draft.library_builder_progress = 100
        partial = bool(outcome["skipped_count"])
        draft.library_builder_status = "partial" if partial else "ready"
        if built <= 0:
            draft.library_builder_status = "error"
            draft.library_builder_message = manifest_error or (
                "The build completed with no usable phases. The library was not selected; inspect the build log and manifest.")
        elif not selection_current:
            draft.library_builder_message = f"Library ready with {built} usable phase(s) in History. Your changed selection was kept."
        elif partial:
            draft.library_builder_message = (
                f"Partial build: {built} of {outcome['requested_count']} supplied CIF(s) built; "
                f"{outcome['skipped_count']} failed source(s). Previous library selection kept. "
                "Review every skipped source below before choosing Use partial library.")
            if not draft.library_builder_generation:
                draft.library_builder_generation = uuid.uuid4().hex
            self._offer_partial_library(draft, action, outcome, draft.library_builder_generation)
        else:
            draft.database_source = "archive"
            draft.library_archive_source = "galaxy"
            draft.history_database_id = action.outputs["library_archive"]
            if draft.workflow == "live":
                self._use_live_element_suggestion(only_if_empty=True)
            draft.library_builder_message = f"Library ready with {built} usable phase(s) and selected for the next analysis."
            if outcome["duplicate_count"]:
                draft.library_builder_message += f" {outcome['duplicate_count']} duplicate/already-present source(s) were accounted for separately."
        draft.notice = draft.library_builder_message
        return True

    def _init_library_reviews(self) -> None:
        self._library_review_internal = False
        for workflow in ("single", "live"):
            draft = LibraryDraft(self.server.state, workflow)
            draft.library_builder_previous_library = library_selection(draft)
        fields = []
        for workflow in ("single", "live"):
            draft = LibraryDraft(self.server.state, workflow)
            fields.extend(draft._key(name) for name in (*SELECTION_FIELDS,
                "library_builder_name", "library_builder_cif_ids", "library_builder_local_paths", "radiation"))
        self.server.state.change(*dict.fromkeys(fields))(self._library_review_changed)

    def _library_review_changed(self, **_: Any) -> None:
        if self._library_review_internal:
            return
        for workflow in ("single", "live"):
            draft = LibraryDraft(self.server.state, workflow)
            review = getattr(draft, "library_builder_partial_review", None) or {}
            if review and not self._partial_library_review_current(draft, review):
                draft.library_builder_partial_review = {}
                draft.library_builder_status = "partial_stale"
                draft.library_builder_message = "The inputs or library selection changed. The partial build remains in History and is not selected."
            usable_archive = draft.database_source == "archive" and (
                (draft.library_archive_source == "galaxy" and bool(draft.history_database_id))
                or (draft.library_archive_source == "computer" and bool(draft.database_archive_path)))
            if draft.database_source == "builtin" or usable_archive:
                draft.library_builder_previous_library = library_selection(draft)

    def _partial_library_review_current(self, draft: LibraryDraft, review: dict[str, Any]) -> bool:
        return bool(review.get("token") and review.get("workflow") == draft.workflow
                    and review.get("generation") == getattr(draft, "library_builder_generation", None)
                    and review.get("signature") == library_review_signature(draft))

    def _offer_partial_library(self, draft: LibraryDraft, action: Any,
                               outcome: dict[str, Any], generation: str) -> None:
        previous = deepcopy(getattr(draft, "library_builder_previous_library", None) or {})
        self._library_review_internal = True
        try:
            for key, value in previous.items():
                if key in SELECTION_FIELDS:
                    setattr(draft, key, value)
            draft.library_builder_partial_review = {
                "token": uuid.uuid4().hex, "workflow": draft.workflow, "generation": generation,
                "signature": library_review_signature(draft), "archive_id": action.outputs["library_archive"],
                "manifest_id": action.outputs.get("library_manifest"), "job_id": action.galaxy_job_id,
                "action_uid": action.uid, "outcome": deepcopy(outcome),
            }
            draft.flush()
        finally:
            self._library_review_internal = False

    def _partial_library_choice(self, context: str, review_token: str | None) -> tuple[LibraryDraft, dict[str, Any]] | None:
        draft = LibraryDraft(self.server.state, context)
        review = getattr(draft, "library_builder_partial_review", None) or {}
        if (draft.library_builder_active or (context == "live" and
                (self.server.state.powgen_monitoring or self.server.state.live_launch_busy))):
            return None
        if not review_token or review.get("token") != review_token:
            return None  # A delayed click must not accept a newer review.
        if not self._partial_library_review_current(draft, review):
            draft.library_builder_partial_review = {}
            draft.library_builder_status = "partial_stale"
            draft.library_builder_message = "The inputs or library selection changed. The partial build remains in History and is not selected."
            draft.flush()
            return None
        return draft, review

    def use_partial_candidate_library(self, context: str = "single", review_token: str | None = None, **_: Any) -> None:
        choice = self._partial_library_choice(context, review_token)
        if choice is None:
            return
        draft, review = choice
        draft.library_builder_partial_review = {}
        draft.database_source = "archive"
        draft.library_archive_source = "galaxy"
        draft.history_database_id = review["archive_id"]
        draft.library_builder_status = "partial_selected"
        outcome = review["outcome"]
        draft.library_builder_message = (f"Partial library selected after review: {outcome['built_count']} of "
            f"{outcome['requested_count']} supplied CIF(s) built; {outcome['skipped_count']} failed source(s) remain excluded.")
        if context == "live":
            self._use_live_element_suggestion(only_if_empty=True)
        draft.notice = draft.library_builder_message
        draft.flush()

    def retain_current_candidate_library(self, context: str = "single", review_token: str | None = None, **_: Any) -> None:
        choice = self._partial_library_choice(context, review_token)
        if choice is None:
            return
        draft, _ = choice
        draft.library_builder_partial_review = {}
        draft.library_builder_status = "partial_kept"
        draft.library_builder_message = "Previous library selection kept. The unselected partial build and its failure details remain in History."
        draft.notice = draft.library_builder_message
        draft.flush()
