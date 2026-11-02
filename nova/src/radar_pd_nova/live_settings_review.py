"""Explicit continuation settings for otherwise read-only legacy Live watches."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import math
import re

from trame.widgets import html
from trame.widgets import vuetify3 as vuetify

from .live_restore import _typed_settings


REVIEW_SCHEMA = "radar-pd-live-continuation-review/v1"
# These are bounded monitor controls, not a way to replace a known profile,
# library, source selection, composition, or scientific configuration.
FIELDS = {
    "frequency_hz": ("Frequency setting (Hz; profiles still use each run's metadata)", "frequency", 0, 1000),
    "late_arrival_window": ("Late-arriving run window", "integer", 0, 5000),
    "max_active_jobs": ("Maximum active analyses", "integer", 1, 5),
    "max_submission_attempts": ("Maximum submission attempts", "integer", 1, 5),
    "retry_base_seconds": ("Initial retry wait (seconds)", "integer", 0, 3600),
    "retry_max_seconds": ("Maximum retry wait (seconds)", "integer", 0, 86400),
    "main_cif_dataset_id": ("Main-phase CIF", "cif", None, None),
}


def missing_setting_labels(names):
    """Use the same field labels in the dialog and saved-watch status."""
    return [FIELDS[name][0] if name in FIELDS else str(name).replace("_", " ").capitalize()
            for name in names]


def setting_error_display(error):
    return re.sub(r"\b(" + "|".join(map(re.escape, FIELDS)) + r")\b",
                  lambda match: FIELDS[match.group(0)][0], str(error))


def json_sha256(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def lifecycle_sha256(controller):
    return json_sha256({phase: {key: run.as_dict() for key, run in getattr(controller.state, phase).items()}
                        for phase in ("discovered", "submitted", "completed", "failed")})


def reviewable_missing_settings(controller):
    if getattr(controller, "restore_requires_explicit_inspection", False):
        raise ValueError("Inspect this exact checkpoint with Restore a saved Live watch before reviewing missing settings")
    missing = tuple(getattr(controller, "restore_missing_settings", ()))
    if not missing:
        raise ValueError("This watch has no missing monitor settings to review")
    if set(missing) - set(FIELDS):
        raise ValueError("This checkpoint needs settings outside the supported continuation review: "
                         + ", ".join(sorted(set(missing) - set(FIELDS))))
    if any(not reason.startswith("Original settings are missing:")
           for reason in getattr(controller, "restore_resume_blocked_reasons", ())):
        raise ValueError("The saved resolver or profile catalog also needs review; missing-setting confirmation cannot authorize it")
    return missing


def prepare_continuation_settings(controller, values, *, acknowledged):
    """Validate a missing-fields-only overlay; do not mutate the watch."""
    missing = reviewable_missing_settings(controller)
    if acknowledged is not True:
        raise ValueError("Explicitly confirm all listed continuation settings before using them")
    if not isinstance(values, dict) or set(values) != set(missing):
        raise ValueError("Only the exact missing monitor settings may be supplied")
    normalized = {}
    for name in missing:
        value = values[name]
        _, kind, lower, upper = FIELDS[name]
        if kind == "cif":
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("Choose None or one ready main-phase CIF in this History")
            normalized[name] = value
        elif kind == "integer":
            if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
                raise ValueError(f"{name} must be an integer between {lower} and {upper}")
            number = int(value)
            if not lower <= number <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}")
            normalized[name] = number
        elif value is None:
            normalized[name] = None  # Explicit automatic per-run metadata.
        else:
            try:
                number = Decimal(str(value))
            except (InvalidOperation, ValueError):
                raise ValueError(f"{name} must be a finite positive frequency") from None
            if isinstance(value, bool) or not number.is_finite() or not lower < number <= upper:
                raise ValueError(f"{name} must be greater than {lower} and no greater than {upper}")
            normalized[name] = str(value)
    original = asdict(controller.settings)
    settings = _typed_settings(type(controller.settings), {**original, **normalized})
    # The registered settings constructor enforces relationships such as
    # retry_max_seconds >= retry_base_seconds. All known fields stay exact.
    result = asdict(settings)
    known = {name: value for name, value in original.items() if name not in missing}
    if any(result[name] != value for name, value in known.items()):
        raise ValueError("Continuation review changed an already known setting")
    receipt = {
        "schema": REVIEW_SCHEMA, "reviewed": True,
        "source_checkpoint_dataset_id": controller.restore_checkpoint_dataset_id,
        "source_checkpoint_sha256": controller.restore_document_sha256,
        "confirmed_missing_settings": normalized,
        "known_settings": known,
        "known_settings_sha256": json_sha256(known),
        "resulting_settings_sha256": json_sha256(result),
        "original_run_state_sha256": lifecycle_sha256(controller),
        "reviewed_utc": datetime.now(timezone.utc).isoformat(),
    }
    return settings, json.loads(json.dumps(receipt))


def validate_continuation_receipt(receipt, settings):
    """Check internal provenance binding when reopening a reviewed checkpoint."""
    values = asdict(settings)
    if (not isinstance(receipt, dict) or receipt.get("schema") != REVIEW_SCHEMA or receipt.get("reviewed") is not True
            or not str(receipt.get("source_checkpoint_dataset_id") or "").strip()
            or not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("source_checkpoint_sha256") or ""))
            or receipt.get("resulting_settings_sha256") != json_sha256(values)):
        raise ValueError("The continuation review does not match these saved settings")
    missing = receipt.get("confirmed_missing_settings")
    known = receipt.get("known_settings")
    if (not isinstance(missing, dict) or not missing or set(missing) - set(FIELDS)
            or not isinstance(known, dict) or set(known).intersection(missing)
            or set(known) | set(missing) != set(values)
            or receipt.get("known_settings_sha256") != json_sha256(known)
            or any(json_sha256(value) != json_sha256(values[name]) for name, value in {**known, **missing}.items())):
        raise ValueError("The continuation review changed known settings or omits its confirmed values")


class LiveSettingsReviewMixin:
    def _init_live_settings_review(self):
        state = self.server.state
        state.live_missing_review_open = False
        state.live_missing_rows = []
        state.live_missing_values = {}
        state.live_missing_acknowledged = False
        state.live_missing_cif_choice = ""
        state.live_missing_cif_options = [{"title": "None — no main-phase CIF", "value": "none"},
                                         {"title": "Choose a CIF from this History", "value": "history"}]
        state.live_missing_frequency_mode = ""
        state.live_missing_frequency_options = [
            {"title": "Automatic from each run (None)", "value": "automatic"},
            {"title": "Record a nominal frequency; still resolve each run's profile", "value": "recorded"}]
        state.live_missing_cif_id = ""
        state.live_missing_has_cif = False
        state.live_missing_message = ""
        state.live_missing_saved_checkpoint_id = ""
        self._live_missing_owner = None
        self._live_missing_io_owner = None
        self._live_missing_context = None

    def _invalidate_live_settings_review(self):
        self._live_missing_owner = None
        if self._live_missing_io_owner is not None:
            self._live_missing_io_owner = None
            self.server.state.live_restore_busy = False
        self._live_missing_context = None
        self.server.state.live_missing_review_open = False

    def _missing_settings_context(self, controller):
        return (self._live_restore_context(), json_sha256(asdict(controller.settings)),
                lifecycle_sha256(controller), tuple(getattr(controller, "restore_missing_settings", ())))

    def review_missing_live_settings(self, **_):
        state = self.server.state
        controller = self._powgen_controller
        if controller is None or not self._live_restore_can_read() or state.live_restore_busy:
            return
        try:
            missing = reviewable_missing_settings(controller)
        except ValueError as exc:
            state.live_restore_message = setting_error_display(exc)
            return
        self._live_missing_owner = object()
        self._live_missing_context = self._missing_settings_context(controller)
        values = asdict(controller.settings)
        state.live_missing_rows = [{"name": name, "label": FIELDS[name][0], "kind": FIELDS[name][1],
            "minimum": FIELDS[name][2], "maximum": FIELDS[name][3]}
            for name in missing if name != "main_cif_dataset_id"]
        state.live_missing_values = {name: values[name] for name in missing if name != "main_cif_dataset_id"}
        state.live_missing_has_cif = "main_cif_dataset_id" in missing
        state.live_missing_cif_choice = ""
        state.live_missing_cif_id = ""
        state.live_missing_frequency_mode = ""
        state.live_missing_acknowledged = False
        state.live_missing_message = "Suggested values are unconfirmed. These will be new continuation settings, not recovered facts."
        state.live_missing_review_open = True
        state.flush()

    def update_missing_live_setting(self, name, value, **_):
        """Commit a nested field edit to Trame's top-level synchronized state."""
        state = self.server.state
        controller = self._powgen_controller
        if (controller is None or self._live_missing_owner is None or not state.live_missing_review_open
                or state.live_restore_busy or not self._live_restore_can_read()
                or not isinstance(name, str) or name not in FIELDS or name == "main_cif_dataset_id"
                or name not in getattr(controller, "restore_missing_settings", ())):
            return
        if self._missing_settings_context(controller) != self._live_missing_context:
            self._invalidate_live_settings_review()
            state.live_restore_message = "The watch changed. Open the missing-settings review again."
            state.flush()
            return
        if (value is not None and not isinstance(value, (str, int, float))) or isinstance(value, bool):
            return
        if isinstance(value, float) and not math.isfinite(value):
            return
        # Keep partial input text intact. Confirmation owns type/range checks.
        state.live_missing_values = {**dict(state.live_missing_values or {}), name: value}
        state.live_missing_acknowledged = False
        state.flush()

    async def confirm_missing_live_settings(self, **_):
        state = self.server.state
        controller, owner = self._powgen_controller, self._live_missing_owner
        if (controller is None or owner is None or not self._live_restore_can_read()
                or state.live_restore_busy or not state.live_missing_review_open):
            return
        context = self._missing_settings_context(controller)
        if context != self._live_missing_context:
            self._invalidate_live_settings_review()
            state.live_restore_message = "The watch or its settings changed; open the missing-settings review again."
            return
        values = dict(state.live_missing_values or {})
        if "frequency_hz" in values:
            if state.live_missing_frequency_mode == "automatic":
                values["frequency_hz"] = None
            elif state.live_missing_frequency_mode != "recorded":
                state.live_missing_message = "Explicitly choose automatic per-run frequency or a recorded nominal value."
                return
        if state.live_missing_has_cif:
            if state.live_missing_cif_choice == "none":
                values["main_cif_dataset_id"] = None
            elif state.live_missing_cif_choice == "history" and state.live_missing_cif_id:
                values["main_cif_dataset_id"] = str(state.live_missing_cif_id)
            else:
                state.live_missing_message = "Explicitly choose None or a main-phase CIF from this History."
                return
        try:
            submitted = json_sha256([dict(state.live_missing_values or {}), state.live_missing_acknowledged,
                state.live_missing_cif_choice, state.live_missing_cif_id, state.live_missing_frequency_mode])
        except (TypeError, ValueError):
            state.live_missing_message = "Use finite, typed values for every missing setting."
            return
        service = self.service
        self._live_missing_io_owner = owner
        state.live_restore_busy = True
        state.flush()
        def current():
            return (self._live_missing_owner is owner and self._live_missing_io_owner is owner and controller is self._powgen_controller
                    and context == self._missing_settings_context(controller) and self._live_restore_can_read()
                    and service is self.service and state.live_missing_review_open
                    and submitted == json_sha256([dict(state.live_missing_values or {}), state.live_missing_acknowledged,
                        state.live_missing_cif_choice, state.live_missing_cif_id, state.live_missing_frequency_mode]))
        try:
            settings, receipt = prepare_continuation_settings(controller, values, acknowledged=state.live_missing_acknowledged)
            main = values.get("main_cif_dataset_id")
            if main:
                metadata = await asyncio.to_thread(service._dataset_metadata, main)
                if not current():
                    return
                if (metadata.get("id") != main or metadata.get("history_id") != controller.settings.history_id
                        or metadata.get("state") != "ok" or metadata.get("deleted") or metadata.get("purged")
                        or metadata.get("history_content_type", "dataset") != "dataset"
                        or not (str(metadata.get("extension") or metadata.get("file_ext") or "").lower() == "cif"
                                or str(metadata.get("name") or "").lower().endswith(".cif"))):
                    raise ValueError("Choose a ready, undeleted CIF dataset from this watch's History")
            if not current():
                return
            definition = settings.definition()
            controller.settings = settings
            controller.definition = definition
            controller.restore_settings_review = receipt
            controller.restore_settings_checkpoint_pending = True
            controller.restore_missing_settings = ()
            controller.restore_resume_blocked_reasons = ()
            self._powgen_settings_signature = (settings.instrument, json.dumps(asdict(settings), sort_keys=True))
            state.live_restore_resume_blocked = False
            self._invalidate_live_settings_review()
            state.live_restore_message = ("Continuation settings were explicitly reviewed for this saved watch. "
                "Resuming requires saving their checkpoint and provenance before continuing the same run maps. "
                "The monitor status above shows whether the watch is currently running or stopped.")
            state.flush()
        except (TypeError, ValueError, OSError) as exc:
            if self._live_missing_owner is owner:
                state.live_missing_message = setting_error_display(exc)
        finally:
            if self._live_missing_io_owner is owner:
                self._live_missing_io_owner = None
                state.live_restore_busy = False
                state.flush()

    async def _checkpoint_live_continuation(self, controller, service, *, current):
        if not getattr(controller, "restore_settings_checkpoint_pending", False):
            return True
        self.server.state.powgen_message = "Saving the reviewed continuation settings before monitoring can resume…"
        self.server.state.flush()
        snapshot = controller.capture_checkpoint()
        dataset_id = await asyncio.to_thread(controller.persist_state, snapshot)
        if not current():
            return False
        controller.restore_settings_checkpoint_pending = False
        controller.restore_continuation_checkpoint_id = dataset_id
        self.server.state.live_missing_saved_checkpoint_id = dataset_id
        return True

    def _live_settings_review_ui(self):
        vuetify.VBtn("Review missing monitor settings", click=self.review_missing_live_settings,
            v_if="!!live_restored_checkpoint_id && live_restore_resume_blocked", variant="outlined", size="small",
            disabled=("powgen_monitoring || live_launch_busy || live_restore_busy",), classes="mb-3")
        with vuetify.VDialog(v_model="live_missing_review_open", max_width=700, max_height="90vh", scrollable=True, persistent=True):
            with vuetify.VCard():
                vuetify.VCardTitle("Review missing continuation settings")
                with vuetify.VCardText():
                    html.P("The saved checkpoint omitted these settings. Suggested defaults below are not confirmed. Known inputs, sample selection and saved run maps remain unchanged; each run still undergoes its normal profile checks.")
                    with html.Div(v_for="row in live_missing_rows", key="row.name"):
                        vuetify.VSelect(v_if="row.kind === 'frequency'", label="Frequency source: choose explicitly",
                            v_model="live_missing_frequency_mode", items=("live_missing_frequency_options",),
                            item_title="title", item_value="value", density="compact", variant="outlined", disabled=("live_restore_busy",))
                        vuetify.VTextField(label=("row.label",), model_value=("live_missing_values[row.name]",),
                            update_modelValue=(self.update_missing_live_setting, "[row.name, $event]"),
                            v_if="row.kind !== 'frequency' || live_missing_frequency_mode === 'recorded'",
                            type=("row.kind === 'integer' ? 'number' : 'text'",),
                            min=("row.minimum",), max=("row.maximum",), density="compact", variant="outlined",
                            disabled=("live_restore_busy",))
                    with html.Div(v_if="live_missing_has_cif"):
                        vuetify.VSelect(label="Main-phase CIF: choose explicitly", v_model="live_missing_cif_choice",
                            items=("live_missing_cif_options",),
                            item_title="title", item_value="value", density="compact", variant="outlined", disabled=("live_restore_busy",))
                        vuetify.VAutocomplete(label="Main-phase CIF in this History", v_model="live_missing_cif_id",
                            v_if="live_missing_cif_choice === 'history'", items=("history_cif_datasets",),
                            item_title="display_name", item_value="id", density="compact", variant="outlined", disabled=("live_restore_busy",))
                    vuetify.VCheckbox(label="I reviewed every listed value as new continuation settings; they are not recovered facts.",
                        v_model="live_missing_acknowledged", disabled=("live_restore_busy",))
                    html.P("{{ live_missing_message }}", classes="radar-help-copy")
                with vuetify.VCardActions():
                    vuetify.VBtn("Keep read-only", click="live_missing_review_open = false", variant="text", disabled=("live_restore_busy",))
                    vuetify.VBtn("Use reviewed settings — stay stopped", click=self.confirm_missing_live_settings,
                        disabled=("!live_missing_acknowledged || live_restore_busy",), loading=("live_restore_busy",), variant="flat")
