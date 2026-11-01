"""Explicit, stopped restoration of one History-owned Live checkpoint.

Inspection reads only the selected small provenance document. It never invokes
the controller's automatic restore, source discovery, profile staging or submit
paths. Missing legacy settings remain missing, including after a results refresh.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from hashlib import sha256
import json
from pathlib import PurePosixPath
import re
from typing import Any, Mapping, get_type_hints

from dateutil.tz import tzstr

from pydantic import TypeAdapter
from trame.widgets import html
from trame.widgets import vuetify3 as vuetify

from .library_setup import LibraryDraft
from .live_beamlines import beamline_adapter
from .powgen_watch import WatchState
from .setup_forms import CONFIG_FORM_FIELDS


SETTINGS_SCHEMA = "radar-pd-live-settings/v1"
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
_PHASES = ("discovered", "submitted", "completed", "failed")
_COMMON_WATCH_FIELDS = (
    "configuration_dataset_id", "database_dataset_id", "instrument_profile_dataset_id",
    "instrument_profile_name", "wavelength_angstrom", "initial_backfill_limit",
    "selected_sample_ids", "selected_banks", "excluded_run_numbers",
    "include_background_runs", "include_ramp_runs",
)
_EXTENSIONS = {
    "NOM": {"profile_bank": "nomad_profile_bank", "profile_acknowledged": "nomad_profile_acknowledged",
            "allow_exploratory": "nomad_allow_exploratory", "main_cif_dataset_id": "nomad_main_cif_dataset_id"},
    "SNAP": {"product_family": "snap_product_family", "profile_bank": "snap_profile_bank",
             "profile_acknowledged": "snap_profile_acknowledged", "allow_exploratory": "snap_allow_exploratory",
             "condition_bindings_json": "snap_condition_bindings"},
    "HB2A": {"normalization": "hb2a_normalization", "profile_acknowledged": "hb2a_profile_acknowledged",
             "condition_bindings_json": "hb2a_condition_bindings", "collimator_override": "hb2a_collimator_override",
             "allow_stopped": "hb2a_allow_stopped", "excluded_acquisitions": "hb2a_excluded_acquisitions"},
}


def state_recorded_display(value):
    """Label the state timestamp, never imply it is the dataset upload time."""
    try:
        stamp = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return "Not recorded"
    if stamp.tzinfo is None:
        return stamp.strftime("%Y-%m-%d %H:%M:%S") + " (timezone not recorded)"
    return stamp.astimezone(tzstr("EST5EDT,M3.2.0/2,M11.1.0/2")).strftime("%Y-%m-%d %H:%M:%S %Z")


def restore_restriction_display(reasons, missing):
    from .live_settings_review import missing_setting_labels
    readable = []
    for reason in reasons:
        if reason.startswith("Original settings are missing:") and missing:
            reason = "Settings to review: " + ", ".join(missing_setting_labels(missing))
        if reason not in readable:
            readable.append(reason)
    return "; ".join(readable)


@dataclass(frozen=True)
class SavedWatchReview:
    controller: Any
    dataset_id: str
    document_sha256: str
    saved_utc: str
    rows: list[dict[str, str]]
    missing: tuple[str, ...]
    blocked: tuple[str, ...]


def _typed_settings(settings_class, values: dict) -> Any:
    """Use the registered settings contract, without JSON type coercion."""
    hints = get_type_hints(settings_class)
    typed = {name: TypeAdapter(hints[name]).validate_json(json.dumps(value), strict=True)
             for name, value in values.items()}
    return settings_class(**typed)


def inspect_checkpoint_payload(payload: Mapping, service, dataset_id: str, *, document_sha256: str) -> SavedWatchReview:
    """Validate the existing WatchState schema and reconstruct its exact scope."""
    if not isinstance(payload, Mapping) or not isinstance(payload.get("watch"), Mapping):
        raise ValueError("This dataset has no saved Live watch settings")
    saved = WatchState.from_dict(payload)
    if saved.history_id != service.history_id:
        raise ValueError("The checkpoint belongs to another Galaxy History")
    watch = payload["watch"]
    if watch.get("instrument", "PG3") != saved.instrument:
        raise ValueError("Checkpoint settings and run state name different beamlines")
    adapter = beamline_adapter(saved.instrument)
    ipts = str(watch.get("ipts") or "")
    source = str(watch.get("source_directory") or "")
    if saved.instrument == "HB2A":
        from .hb2a import proposal_root
        root = PurePosixPath(proposal_root(ipts))
    else:
        root = PurePosixPath(f"/{adapter.facility}/{saved.instrument}/{ipts}")
    try:
        subfolder = PurePosixPath(source).relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("The saved source folder does not belong to the saved experiment") from exc
    if not source or ".." in PurePosixPath(source).parts:
        raise ValueError("The saved source folder is invalid")
    names = {field.name for field in fields(adapter.settings)}
    complete = "settings" in watch or "settings_schema" in watch
    if complete:
        if watch.get("settings_schema") != SETTINGS_SCHEMA or not isinstance(watch.get("settings"), dict):
            raise ValueError("Unsupported saved Live settings version")
        values = dict(watch["settings"])
        unknown = set(values) - names
        if unknown:
            raise ValueError("Unsupported saved Live settings: " + ", ".join(sorted(unknown)))
    else:
        values = {name: watch[name] for name in _COMMON_WATCH_FIELDS if name in watch}
        values.update({name: watch[key] for name, key in _EXTENSIONS.get(saved.instrument, {}).items() if key in watch})
        values.update(history_id=saved.history_id, instrument=saved.instrument, ipts=ipts, subfolder=subfolder)
    missing = sorted(names - set(values))
    declared_missing = watch.get("missing_settings", [])
    if not isinstance(declared_missing, list) or any(not isinstance(v, str) or v not in names for v in declared_missing):
        raise ValueError("Invalid missing-settings declaration")
    missing = sorted(set(missing) | set(declared_missing))
    settings = _typed_settings(adapter.settings, values)
    if (settings.history_id != saved.history_id or settings.instrument != saved.instrument
            or settings.ipts != ipts or settings.subfolder != subfolder):
        raise ValueError("The saved settings disagree with the checkpoint's History or source scope")
    controller = adapter.controller(service, settings)
    if controller.source_directory != source:
        raise ValueError("The saved source folder is not a supported exact source scope")
    if watch.get("facility", adapter.facility) != adapter.facility:
        raise ValueError("The saved facility does not match this beamline")
    # A second, conflicting settings copy must never silently win.
    serialized = json.loads(json.dumps(asdict(settings)))
    for name in _COMMON_WATCH_FIELDS:
        if name in watch and serialized[name] != watch[name]:
            raise ValueError(f"Conflicting saved Live setting: {name}")
    for name, key in _EXTENSIONS.get(saved.instrument, {}).items():
        if key in watch and serialized[name] != watch[key]:
            raise ValueError(f"Conflicting saved Live setting: {name}")
    for phase in _PHASES:
        for run in getattr(saved, phase).values():
            path = PurePosixPath(run.source_path)
            if ".." in path.parts or not path.is_relative_to(PurePosixPath(source)) or not run.fingerprint:
                raise ValueError(f"{run.run_id} has no valid source identity in the saved folder")
    blocked = []
    if missing:
        blocked.append("Original settings are missing: " + ", ".join(missing))
    if watch.get("resolver") != controller.resolver_version:
        blocked.append("The saved source resolver differs from this NOVA version")
    if watch.get("catalog_sha256") != controller._catalog_digest():
        blocked.append("The saved instrument/profile catalog differs from this NOVA version")
    if not missing and watch.get("checkpoint_key") != controller._checkpoint_key():
        blocked.append("The saved watch identity cannot be reproduced from its settings")
    previous_blocks = watch.get("resume_blocked_reasons", [])
    if not isinstance(previous_blocks, list) or any(not isinstance(v, str) for v in previous_blocks):
        raise ValueError("Invalid saved resume restriction")
    blocked = list(dict.fromkeys([*blocked, *previous_blocks]))
    controller._adopt_restored_state(saved, dataset_id, hydrate_metadata=False)
    controller.restore_checkpoint_dataset_id = dataset_id
    controller.restore_document_sha256 = document_sha256
    controller.restore_missing_settings = tuple(missing)
    controller.restore_resume_blocked_reasons = tuple(blocked)
    controller._last_persisted_payload_sha256 = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if watch.get("continuation_review"):
        from .live_settings_review import validate_continuation_receipt
        validate_continuation_receipt(watch["continuation_review"], settings)
        controller.restore_settings_review = json.loads(json.dumps(watch["continuation_review"]))
    controller._retry_submission_allowed = lambda: False
    if watch.get("manual_setup_signature"):
        controller._manual_setup_signature = tuple(watch["manual_setup_signature"])
    from .live_settings_review import missing_setting_labels

    def displayed(name, value):
        return "Unknown (not saved)" if name in missing else str(value)
    limit = settings.initial_backfill_limit
    recorded_utc = str(payload.get("updated_utc") or "")
    rows = [
        {"label": "Checkpoint", "value": dataset_id},
        {"label": "State recorded", "value": state_recorded_display(recorded_utc)},
        {"label": "Experiment", "value": f"{adapter.name} · {ipts}"},
        {"label": "Source folder", "value": source},
        {"label": "Saved runs", "value": " · ".join(f"{len(getattr(saved, p))} {p}" for p in _PHASES)},
        {"label": "Configuration", "value": settings.configuration_dataset_id},
        {"label": "Sample IDs", "value": displayed("selected_sample_ids", ", ".join(settings.selected_sample_ids or ()) or "All samples")},
        {"label": "Banks", "value": displayed("selected_banks", ", ".join(map(str, settings.selected_banks or ())) or "All banks")},
        {"label": "Initial scan selection", "value": displayed("initial_backfill_limit", "All existing matching scans" if limit is None else "Future matching scans only" if limit == 0 else f"Latest {limit} matching scans")},
        {"label": "Excluded runs", "value": displayed("excluded_run_numbers", ", ".join(map(str, settings.excluded_run_numbers)) or "None")},
        {"label": "Instrument profile", "value": displayed("instrument_profile_dataset_id", settings.instrument_profile_dataset_id or "Saved automatic mapping")},
        {"label": "Candidate library", "value": displayed("database_dataset_id", settings.database_dataset_id or "Built-in catalog")},
        {"label": "Main-phase CIF", "value": displayed("main_cif_dataset_id", settings.main_cif_dataset_id or "None")},
        {"label": "Concurrent analyses", "value": displayed("max_active_jobs", settings.max_active_jobs)},
        {"label": "Late-arrival window", "value": displayed("late_arrival_window", settings.late_arrival_window)},
        {"label": "Settings still requiring review", "value": ", ".join(missing_setting_labels(missing)) or "None"},
    ]
    if getattr(controller, "restore_settings_review", None):
        rows.append({"label": "Settings provenance", "value": "Includes explicitly reviewed continuation settings; not all values were recovered from the original checkpoint"})
    return SavedWatchReview(controller, dataset_id, document_sha256, recorded_utc, rows, tuple(missing), tuple(blocked))


def read_saved_watch(service, dataset_id: str) -> SavedWatchReview:
    """Read exactly one ready, current-History HDA, with a bounded JSON body."""
    history_id = service.history_id
    metadata = service._dataset_metadata(dataset_id)
    if (str(metadata.get("id") or "") != dataset_id or metadata.get("history_id") != history_id
            or metadata.get("deleted") or metadata.get("purged") or metadata.get("state") != "ok"
            or metadata.get("history_content_type", "dataset") != "dataset"):
        raise ValueError("Choose a ready, undeleted checkpoint in the current Galaxy History")
    size = metadata.get("file_size")
    if size is not None and (int(size) < 0 or int(size) > MAX_CHECKPOINT_BYTES):
        raise ValueError("The checkpoint exceeds the supported metadata size")
    text = service.load_text_document(dataset_id, max_bytes=MAX_CHECKPOINT_BYTES)
    if service.history_id != history_id:
        raise ValueError("The active History changed while reading the checkpoint")
    payload = json.loads(text)
    return inspect_checkpoint_payload(payload, service, dataset_id, document_sha256=sha256(text.encode("utf-8")).hexdigest())


class LiveRestoreMixin:
    def _init_live_restore(self):
        state = self.server.state
        state.live_restore_options = []
        state.live_restore_dataset_id = ""
        state.live_restore_rows = []
        state.live_restore_message = "Choose a saved Live checkpoint in this History to review it before restoring."
        state.live_restore_busy = False
        state.live_restore_ready = False
        state.live_restore_confirm = False
        state.live_restore_search = ""
        state.live_restore_page = 0
        state.live_restore_has_more = False
        state.live_restore_resume_blocked = False
        state.live_restored_checkpoint_id = ""
        self._live_restore_owner = None
        self._live_restore_review = None
        self._live_restore_confirm_signature = None
        state.change("live_restore_dataset_id")(self._live_restore_selection_changed)
        draft_fields = ["live_" + field for field in CONFIG_FORM_FIELDS]
        draft_fields += ["live_" + field for field in ("configuration_source", "database_source",
            "library_archive_source", "database_archive_path", "library_review")]
        draft_fields += ["live_beamline", "powgen_ipts", "powgen_source_subfolder", "powgen_configuration_dataset_id",
            "powgen_backfill_mode", "powgen_processing_scope", "powgen_processing_sample_ids", "powgen_bank_scope",
            "powgen_processing_banks", "powgen_wavelength", "powgen_instrument_source", "powgen_instrument_path",
            "powgen_instrument_dataset_id", "powgen_main_cif_source", "powgen_main_cif_path", "powgen_main_cif_dataset_id",
            "powgen_database_dataset_id", "powgen_excluded_runs", "powgen_include_background", "powgen_include_ramps",
            "nomad_physical_bank", "nomad_profile_bank", "nomad_profile_acknowledged", "nomad_allow_exploratory",
            "snap_product_family", "hb2a_normalization", "hb2a_profile_acknowledged"]
        state.change(*dict.fromkeys(draft_fields))(self._live_restore_draft_changed)

    def _invalidate_live_restore(self):
        self._live_restore_owner = None
        self._live_restore_review = None
        self._live_restore_confirm_signature = None
        state = self.server.state
        state.live_restore_busy = False
        state.live_restore_ready = False
        state.live_restore_confirm = False
        state.live_restore_rows = []
        if hasattr(self, "_live_missing_owner"):
            self._invalidate_live_settings_review()

    def _live_restore_selection_changed(self, **_):
        self._invalidate_live_restore()

    def _live_restore_draft_changed(self, **_):
        if (self._live_restore_owner is not None or self._live_restore_review is not None
                or getattr(self, "_live_missing_owner", None) is not None):
            self._invalidate_live_restore()
            self.server.state.live_restore_message = "The Live draft changed. Inspect the selected checkpoint again before restoring."

    def _live_restore_context(self):
        state = self.server.state
        return (self.service, self.service.history_id, self._result_context,
                state.live_restore_dataset_id, state.live_restore_search, state.live_beamline, state.powgen_ipts,
                self._powgen_preflight_signature(), LibraryDraft(state, "live").signature(),
                self._powgen_controller)

    def _live_restore_can_read(self):
        state = self.server.state
        return not (state.powgen_monitoring or state.live_launch_busy or state.workspace_clearing or state.session_closing
                    or getattr(self, "_powgen_resume_owner", None) is not None
                    or (self._powgen_monitor_task is not None and not self._powgen_monitor_task.done()))

    async def list_saved_live_watches(self, *, older=False, **_):
        if not self._live_restore_can_read() or self.server.state.live_restore_busy:
            return
        state = self.server.state
        requested_ipts = str(state.live_restore_search or "").strip().upper()
        if requested_ipts and not re.fullmatch(r"(?:IPTS-)?[1-9][0-9]*", requested_ipts):
            state.live_restore_message = "Enter an IPTS number such as 37869 or IPTS-37869, or leave the filter empty."
            return
        if requested_ipts and not requested_ipts.startswith("IPTS-"):
            requested_ipts = "IPTS-" + requested_ipts
        owner, context = object(), self._live_restore_context()
        self._live_restore_owner = owner
        state.live_restore_busy = True
        state.live_restore_ready = False
        self._live_restore_review = None
        page = int(state.live_restore_page or 0) + 1 if older else 0
        query = "watch state" + (" " + requested_ipts if requested_ipts else "")
        state.live_restore_message = "Listing saved Live checkpoints in this History…"
        state.flush()
        try:
            rows = await asyncio.to_thread(self.service.search_history_datasets,
                query=query, limit=100, offset=page * 100, include_generated=True)
            if self._live_restore_owner is not owner or context != self._live_restore_context() or not self._live_restore_can_read():
                return
            state.live_restore_options = [dict(row, title=row.get("display_name") or row.get("name") or row["id"])
                for row in rows if row.get("state") == "ok" and "watch state" in str(row.get("name") or "").lower()]
            state.live_restore_page = page
            state.live_restore_has_more = len(rows) == 100
            state.live_restore_message = (f"Page {page + 1}: choose one checkpoint, then inspect its saved scope."
                if state.live_restore_options else "No ready Live checkpoints on this page in the current History.")
        except Exception as exc:
            if self._live_restore_owner is owner and context == self._live_restore_context():
                state.live_restore_message = f"Could not list saved Live checkpoints: {exc}"
        finally:
            if self._live_restore_owner is owner:
                self._live_restore_owner = None
                state.live_restore_busy = False
                state.flush()

    async def older_saved_live_watches(self, **_):
        await self.list_saved_live_watches(older=True)

    async def inspect_saved_live_watch(self, **_):
        state = self.server.state
        if not self._live_restore_can_read() or state.live_restore_busy:
            return
        dataset_id = str(state.live_restore_dataset_id or "")
        if not dataset_id or not any(row.get("id") == dataset_id for row in state.live_restore_options):
            state.live_restore_message = "Select a checkpoint from this History's saved-watch list."
            return
        owner, context = object(), self._live_restore_context()
        self._live_restore_owner = owner
        self._live_restore_review = None
        state.live_restore_busy, state.live_restore_ready = True, False
        state.live_restore_rows = []
        state.live_restore_message = "Reading the selected checkpoint; no scans or configurations are being submitted…"
        state.flush()
        try:
            review = await asyncio.to_thread(read_saved_watch, self.service, dataset_id)
            if self._live_restore_owner is not owner or context != self._live_restore_context() or not self._live_restore_can_read():
                return
            self._live_restore_review = (context, review)
            state.live_restore_rows = review.rows
            state.live_restore_ready = True
            state.live_restore_message = ("Read-only restoration. " + restore_restriction_display(review.blocked, review.missing) + ". Resume stays disabled."
                if review.blocked else "Saved user-reviewed continuation settings are available. Restore remains stopped; resuming requires a separate action."
                if getattr(review.controller, "restore_settings_review", None) else
                "The complete saved settings are available. Restore remains stopped; resuming requires a separate action.")
        except Exception as exc:
            if self._live_restore_owner is owner and context == self._live_restore_context():
                state.live_restore_message = f"Checkpoint could not be restored: {exc}"
        finally:
            if self._live_restore_owner is owner:
                self._live_restore_owner = None
                state.live_restore_busy = False
                state.flush()

    def restore_saved_live_watch(self, *, confirmed=False, **_):
        state = self.server.state
        entry = self._live_restore_review
        if not self._live_restore_can_read() or state.live_restore_busy or not entry:
            return
        context, review = entry
        if context != self._live_restore_context():
            self._invalidate_live_restore()
            state.live_restore_message = "The Live context changed. Inspect this checkpoint again before restoring."
            return
        replacement = self._powgen_controller is not None or bool(self.records) or bool(state.powgen_ipts)
        if replacement and not confirmed:
            self._live_restore_confirm_signature = context
            state.live_restore_confirm = True
            return
        if confirmed and (not state.live_restore_confirm or self._live_restore_confirm_signature != context):
            state.live_restore_confirm = False
            return
        controller = review.controller
        settings = controller.settings
        # Reopening a checkpoint must retain already downloaded results for
        # the same immutable Galaxy job. Never overwrite a different owner.
        for run_id, record in list(controller.records.items()):
            cached = self.records.get(record.uid)
            if cached is not None:
                if cached.history_id != record.history_id or cached.galaxy_job_id != record.galaxy_job_id:
                    state.live_restore_message = "A cached record has a different History/job identity; the saved watch was not restored."
                    state.live_restore_confirm = False
                    return
                controller.records[run_id] = cached
        # Keep the current editable draft separate. The restored controller and
        # displayed provenance hold every original setting; Resume uses those
        # settings rather than reconstructing a new watch from the draft.
        self._powgen_controller = controller
        self._powgen_controller_configuration = None
        self._powgen_settings_signature = (settings.instrument, json.dumps(asdict(settings), sort_keys=True))
        self._powgen_monitor_task = None
        self._result_context = object()
        self._result_open_owner = None
        state.selected_run_uid = ""
        state.selected_run_name = "No run selected"
        self._reset_result_state()
        state.powgen_monitoring = False
        state.powgen_retry_requested = state.powgen_retry_busy = False
        state.powgen_next_check = "Stopped"
        state.powgen_last_checked = "Saved checkpoint; not refreshed yet"
        state.workflow_mode = "powgen"
        state.experiment_result_view = "results"
        state.workspace_view = "experiment"
        state.live_restore_confirm = False
        state.live_restored_checkpoint_id = review.dataset_id
        state.live_restore_resume_blocked = bool(review.blocked)
        self.records.update({record.uid: record for record in controller.records.values()})
        self._sync_powgen_rows(refresh_dependents=False, load_patterns=False)
        self._sync_runs()
        self._sync_workspace_options("full")
        state.powgen_message = (f"Restored {settings.instrument} {settings.ipts} from checkpoint {review.dataset_id}. "
            f"State recorded: {state_recorded_display(review.saved_utc)}. Monitoring is stopped. "
            "This is the recorded state time, not the dataset upload time. "
            "Refresh results checks existing jobs only. The editable setup draft is separate.")
        state.live_restore_message = (state.powgen_message + (" Read-only: " + restore_restriction_display(review.blocked, review.missing) if review.blocked else
            " Use Resume existing watch to continue its saved settings. Clear workspace to configure a separate watch."))
        self._live_restore_review = None
        state.live_restore_ready = False
        state.flush()

    def confirm_restore_saved_live_watch(self, **_):
        self.restore_saved_live_watch(confirmed=True)

    def cancel_restore_saved_live_watch(self, **_):
        self._live_restore_confirm_signature = None
        self.server.state.live_restore_confirm = False
        self.server.state.flush()

    def _live_restore_ui(self):
        with html.Details(v_show="workflow_mode === 'powgen'", classes="radar-soft-panel mb-3"):
            html.Summary("Restore a saved Live watch (stopped)")
            html.P("Select one checkpoint from this History and review its time, source and counts. Restoration starts no monitoring, scans or refinement jobs.", classes="radar-help-copy")
            vuetify.VTextField(label="Optional IPTS number", placeholder="37869 or IPTS-37869", v_model="live_restore_search", density="compact", disabled=("live_restore_busy || powgen_monitoring || live_launch_busy",))
            with html.Div(classes="radar-button-row"):
                vuetify.VBtn("List saved watches", click=self.list_saved_live_watches, loading=("live_restore_busy",), disabled=("live_restore_busy || powgen_monitoring || live_launch_busy",), size="small", variant="outlined")
                vuetify.VBtn("Older checkpoints", click=self.older_saved_live_watches, v_if="live_restore_has_more", disabled=("live_restore_busy || powgen_monitoring || live_launch_busy",), size="small", variant="text")
            vuetify.VAutocomplete(label="Saved Live checkpoint", v_model="live_restore_dataset_id", items=("live_restore_options",), item_title="title", item_value="id", density="compact", clearable=True, disabled=("live_restore_busy || powgen_monitoring || live_launch_busy",))
            vuetify.VBtn("Inspect selected checkpoint", click=self.inspect_saved_live_watch, disabled=("!live_restore_dataset_id || live_restore_busy || powgen_monitoring || live_launch_busy",), size="small", variant="outlined")
            with html.Div(v_for="row in live_restore_rows", key="row.label", classes="radar-preflight-row"):
                html.Span("{{ row.label }}", classes="radar-preflight-label")
                html.Span("{{ row.value }}", classes="radar-preflight-value")
            html.P("{{ live_restore_message }}", classes="radar-help-copy")
            vuetify.VBtn("Restore stopped watch", click=self.restore_saved_live_watch, disabled=("!live_restore_ready || live_restore_busy || powgen_monitoring || live_launch_busy",), size="small", variant="outlined")
        with vuetify.VDialog(v_model="live_restore_confirm", max_width=620, persistent=True):
            with vuetify.VCard():
                vuetify.VCardTitle("Replace the displayed Live watch?")
                vuetify.VCardText("The selected checkpoint replaces the displayed watch and its result selection. Existing Galaxy jobs, checkpoints and cached results are preserved. The editable setup draft remains separate. Monitoring stays stopped.")
                with vuetify.VCardActions():
                    vuetify.VBtn("Keep current view", click=self.cancel_restore_saved_live_watch, variant="text")
                    vuetify.VBtn("Restore stopped watch", click=self.confirm_restore_saved_live_watch, color="primary", variant="flat")
