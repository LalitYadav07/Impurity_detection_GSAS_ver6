"""Live-only conversation controller over the existing scientific draft."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from secrets import token_hex

from .live_assistant import (
    HELP, PROMPTS, STEPS, STEP_FIELDS, TITLES, WATCH_FIELDS,
    EDIT_FIELDS, SetupConversation, goal_patch, parse_elements, signature,
)
from .live_assistant_language import (
    MAX_REPLY, MAX_SHOWN_SAMPLES, ParsedReply, ReplyContext, SampleOption,
    interpret_reply, pending_answer,
)
from .live_assistant_ui import LiveAssistantUi
from .live_setup_preview import NoPreviewData, inspect_inventory, preview_pattern
from .setup_forms import full_profile_settings, parse_regions
from .uploads import cif_element_suggestion, inspect_cif_upload
from .live_beamlines import BEAMLINES, beamline_adapter


class LiveAssistantMixin(LiveAssistantUi):
    def _init_live_assistant(self):
        self._reset_live_assistant()
        self.server.state.change(*WATCH_FIELDS)(self._live_assistant_changed)

    def _reset_live_assistant(self):
        """Forget experiment-scoped conversation state without registering watchers twice."""
        state = self.server.state
        task = getattr(self, "_live_assistant_action_task", None)
        self._assistant_generation = object()
        self._live_assistant_action_task = None
        if task is not None and not task.done():
            task.cancel()
        self._live_conversation = SetupConversation()
        self._assistant_request = None
        self._assistant_inventory = None
        self._assistant_inventory_signature = None
        self._assistant_checked_signature = None
        self._assistant_preview_signature = None
        self._assistant_preview_inventory = None
        state.live_assistant_open = False
        state.live_assistant_step = "experiment"
        state.live_assistant_text = ""
        state.live_assistant_busy = False
        state.live_assistant_error = ""
        state.live_assistant_preview = {}
        state.live_assistant_plot = {"data": [], "layout": {"height": 320}}
        state.live_assistant_inventory_note = ""
        state.live_assistant_summary = []
        state.live_assistant_messages = []
        state.live_assistant_prompt = PROMPTS["experiment"]
        state.live_assistant_steps = [{"value": s, "title": TITLES[s], "confirmed": False} for s in STEPS]
        state.live_assistant_complete = False
        state.live_assistant_revision = 0
        state.live_assistant_focus = ""
        state.live_assistant_reply_choices = []
        state.live_assistant_sample_choices = []
        state.live_assistant_context_token = token_hex(12)
        self._assistant_visible_signature = None
        self._assistant_shown_samples = ()
        self._assistant_refresh_sample_choices()

    def _assistant_snapshots(self):
        state = self.server.state
        return {step: {name: deepcopy(getattr(state, name, None)) for name in fields}
                for step, fields in STEP_FIELDS.items()}

    def _live_assistant_changed(self, **_):
        if not hasattr(self, "_live_conversation"):
            return
        state = self.server.state
        if (self._assistant_inventory_signature is not None
                and self._assistant_inventory_signature != self._assistant_inventory_scope()):
            self._assistant_inventory_signature = None
            self._assistant_inventory = None
            state.powgen_bank_options = []
            state.live_assistant_inventory_note = "Sample or run selection changed. Refresh banks for these samples before choosing a bank."
        self._assistant_refresh_sample_choices()
        pending = self._live_conversation.pending
        if pending and pending["signature"] != self._assistant_language_signature():
            self._assistant_clear_reply_choices()
        changed = self._live_conversation.invalidate(self._assistant_snapshots())
        state.live_assistant_revision += 1
        if changed:
            self._assistant_checked_signature = None
            state.live_assistant_complete = False
            state.live_assistant_summary = []
        if self._assistant_preview_signature is not None and self._assistant_preview_signature != self._preview_scope():
            state.live_assistant_preview = {}
            state.live_assistant_plot = {"data": [], "layout": {"height": 320}}
            self._assistant_preview_signature = None
            self._assistant_preview_inventory = None
        state.live_assistant_steps = [
            {"value": s, "title": TITLES[s], "confirmed": s in self._live_conversation.confirmed}
            for s in STEPS
        ]
        state.live_assistant_prompt = PROMPTS.get(state.live_assistant_step, PROMPTS["experiment"])

    def _assistant_locked(self):
        state = self.server.state
        return bool(state.powgen_monitoring or state.live_launch_busy or state.live_library_builder_active)

    def _assistant_say(self, role, message):
        self._live_conversation.say(role, message)
        self.server.state.live_assistant_messages = list(self._live_conversation.messages[-6:])

    def _assistant_step(self, step):
        state = self.server.state
        self._assistant_clear_reply_choices()
        state.live_assistant_step = step
        state.live_assistant_focus = ""
        self._assistant_refresh_sample_choices()
        state.live_assistant_prompt = PROMPTS[step]
        state.live_assistant_error = ""

    def open_live_assistant(self, **_):
        state = self.server.state
        if self._assistant_locked():
            return
        state.live_assistant_open = True
        self._assistant_step(next((s for s in STEPS[:-1] if s not in self._live_conversation.confirmed), "review"))
        state.flush()

    def live_assistant_go(self, step="experiment", **_):
        if self._assistant_locked() or self.server.state.live_assistant_busy:
            return
        if step in STEPS:
            self._assistant_step(step)
        self.server.state.flush()

    def live_assistant_manual(self, section=2, **_):
        self._assistant_clear_reply_choices()
        self.server.state.live_setup_panels = [int(section)]
        self.server.state.live_assistant_open = False
        self.server.state.flush()

    def _queue_live_assistant_action(self, action, *args, **kwargs):
        """Release the browser RPC while read-only discovery/inspection awaits I/O.

        Awaiting the whole operation in a wslink event serializes later browser
        events, including navigation. Keep a single tracked task instead; the
        existing action owns its scientific validation and busy state.
        """
        task = getattr(self, "_live_assistant_action_task", None)
        if self._assistant_locked() or self.server.state.live_assistant_busy or (task and not task.done()):
            return

        async def run():
            try:
                await action(*args, **kwargs)
            except Exception as exc:
                self.server.state.live_assistant_error = str(exc)
                self.server.state.flush()

        self._live_assistant_action_task = asyncio.create_task(run())

    def start_live_assistant_continue(self, payload=None, **kwargs):
        self._queue_live_assistant_action(self.live_assistant_continue, deepcopy(payload), **kwargs)

    def start_live_assistant_send(self, text="", context_token=None, **kwargs):
        self._queue_live_assistant_action(self.live_assistant_send, text, context_token=context_token, **kwargs)

    def start_live_assistant_choose_reply(self, choice_id="", **kwargs):
        self._queue_live_assistant_action(self.live_assistant_choose_reply, choice_id, **kwargs)

    def start_live_assistant_preview(self, **kwargs):
        self._queue_live_assistant_action(self.live_assistant_preview, **kwargs)

    def start_live_assistant_inventory(self, **kwargs):
        self._queue_live_assistant_action(self.live_assistant_inventory, **kwargs)

    def start_live_assistant_suggest_elements(self, **kwargs):
        self._queue_live_assistant_action(self.live_assistant_suggest_elements, **kwargs)

    async def live_assistant_inventory(self, **_):
        state = self.server.state
        if self._assistant_locked() or state.live_assistant_busy:
            return
        state.live_assistant_busy = True
        generation = self._assistant_generation
        state.live_assistant_error = ""
        state.flush()
        try:
            await self._assistant_inspect()
        except (ValueError, OSError) as exc:
            state.live_assistant_error = str(exc)
        finally:
            if generation is self._assistant_generation:
                state.live_assistant_busy = False
                state.flush()

    def _assistant_ingest(self, payload):
        """Use button-time visible values, including debounced text fields."""
        if not isinstance(payload, dict):
            return
        state = self.server.state
        fields = STEP_FIELDS.get(state.live_assistant_step, ())
        for name in fields:
            if name in payload:
                setattr(state, name, deepcopy(payload[name]))
        # Resolve context callbacks before discovery instead of allowing a
        # delayed beamline/IPTS callback to clear the newly discovered choices.
        if state.live_assistant_step == "experiment":
            self._live_beamline_changed()
            self._powgen_ipts_changed()
            if payload.get("powgen_source_subfolder"):
                state.powgen_source_subfolder = payload["powgen_source_subfolder"]
            self._powgen_source_changed()
        state.flush()

    def _assistant_confirm(self, step, summary):
        state = self.server.state
        state.flush()
        self._live_conversation.confirmed[step] = signature(self._assistant_snapshots()[step])
        self._assistant_say("user", summary)
        self._live_assistant_changed()

    def _assistant_context(self):
        state = self.server.state
        return (state.live_beamline, self._normalize_powgen_ipts(state.powgen_ipts), state.powgen_source_subfolder)

    def _assistant_inventory_scope(self):
        state = self.server.state
        fields = ("live_beamline", "powgen_ipts", "powgen_source_subfolder",
                  "powgen_processing_scope", "powgen_processing_sample_ids", "powgen_excluded_runs",
                  "powgen_include_background", "powgen_include_ramps", "hb2a_normalization",
                  "snap_product_family", "hb2a_collimator_override", "hb2a_allow_stopped")
        return signature({key: getattr(state, key, None) for key in fields})

    async def _assistant_inspect(self, *, allow_waiting=True, proposed=None):
        state = self.server.state
        context = self._assistant_context()
        inventory_scope = self._assistant_inventory_scope()
        proposed = proposed or {}
        def value(name):
            return proposed.get(name, getattr(state, name))
        selected = (tuple(value("powgen_processing_sample_ids"))
                    if value("powgen_processing_scope") == "selected" else None)
        selection = signature(self._assistant_snapshots()["selection"])
        # Discovery precedes the sensor-review controls. Unreviewed suggested
        # channels must not prevent seeing those controls or become qualified
        # temperatures in a preview. The launch preflight still validates the
        # full configured binding, including its explicit acknowledgement.
        condition_bindings = None
        if context[0] == "HB2A":
            condition_bindings = {key: value for key, value in self._hb2a_condition_bindings(state).items()
                                  if value.get("reviewed") is True}
        try:
            inventory = await asyncio.to_thread(
                inspect_inventory, *context, self._powgen_sample_index(context[1]),
                sample_ids=selected, excluded_runs=tuple(value("powgen_excluded_runs") or []),
                include_background=bool(value("powgen_include_background")),
                include_ramps=bool(value("powgen_include_ramps")),
                normalization=state.hb2a_normalization,
                product_family=state.snap_product_family,
                condition_bindings=condition_bindings,
                collimator_override=state.hb2a_collimator_override,
                allow_stopped=bool(state.hb2a_allow_stopped),
            )
        except NoPreviewData as exc:
            if not allow_waiting:
                raise
            inventory = None
            waiting_message = str(exc)
        if (context != self._assistant_context()
                or inventory_scope != self._assistant_inventory_scope()
                or selection != signature(self._assistant_snapshots()["selection"])):
            raise ValueError("The experiment or sample selection changed during discovery. Continue with the current selection again.")
        if proposed:
            return inventory
        self._assistant_publish_inventory(inventory, waiting_message if inventory is None else "")
        return inventory

    def _assistant_publish_inventory(self, inventory, waiting_message=""):
        state = self.server.state
        self._assistant_inventory = inventory
        self._assistant_inventory_signature = self._assistant_inventory_scope()
        if inventory is None:
            state.powgen_bank_options = []
            state.live_assistant_inventory_note = waiting_message
            return None
        state.powgen_bank_options = inventory["banks"]
        state.live_assistant_inventory_note = (
            f"Bank preview: {inventory['run_id']} · sample {inventory['sample_id'] or 'unidentified'}. "
            "The final experiment check validates the selected initial scans separately."
        )
        return inventory

    def _assistant_validate_selection(self):
        state = self.server.state
        selected = self._powgen_processing_ids()
        allowed = {str(item["value"]) for item in state.powgen_discovered_sample_options}
        if selected and set(selected) - allowed:
            raise ValueError("A selected sample is no longer in this experiment. Refresh the samples and select it again.")
        selected_banks = self._powgen_processing_bank_ids()
        available_banks = {int(item["value"]) for item in state.powgen_bank_options}
        if selected_banks and set(selected_banks) - available_banks:
            raise ValueError("A selected bank is not available for these samples. Refresh banks and choose a displayed bank.")
        if state.powgen_backfill_mode not in {"new_only", "latest_1", "latest_5", "latest_20", "all"}:
            raise ValueError("Choose the initial monitoring scope.")
        source = state.powgen_instrument_source
        adapter = beamline_adapter(state.live_beamline)
        if not adapter.automatic_profiles and source == "automatic":
            raise ValueError(adapter.name + " needs a supplied .instprm and its physical/profile bank mapping.")
        if source == "computer":
            path = Path(str(state.powgen_instrument_path or ""))
            suffixes = {".instprm"} if state.live_beamline != "PG3" else {".instprm", ".prm", ".inst", ".ins"}
            if not path.is_file() or not path.stat().st_size or path.suffix.lower() not in suffixes:
                raise ValueError("Choose a nonempty instrument profile matching this beamline and bank.")
        elif source == "galaxy":
            if not state.powgen_instrument_dataset_id:
                raise ValueError("Select an instrument profile from Galaxy History.")
        elif source != "automatic":
            raise ValueError("Choose the instrument profile source.")
        if state.live_beamline in {'NOM','SNAP'} and not state.nomad_profile_acknowledged:
            raise ValueError("Review and confirm the " + adapter.name + " physical-bank/profile-bank mapping.")
        if state.live_beamline == 'HB2A' and not state.hb2a_profile_acknowledged:
            raise ValueError('Review the HB-2A profile, normalization and reduction corrections before continuing.')

    def _assistant_validate_chemistry(self, config=None):
        state = self.server.state
        sample = parse_elements(", ".join(config.sample_elements) if config else state.live_sample_elements)
        environment = parse_elements(", ".join(config.environment_elements) if config else state.live_environment_elements, optional=True)
        overlap = sorted(set(sample) & set(environment))
        if overlap:
            raise ValueError("Review " + ", ".join(overlap) + ": it is in both sample and environment lists. The current environment filter can exclude mixed sample phases; keep the sample chemistry complete and review the environment assignment.")

    async def _assistant_configuration(self):
        checked = self._powgen_preflight_signature()
        config = await asyncio.to_thread(self._live_configuration)
        try:
            json.dumps(config.model_dump(mode="json"), allow_nan=False)
        except ValueError as exc:
            raise ValueError("Use finite numbers for the Live settings and fit regions.") from exc
        if checked != self._powgen_preflight_signature():
            raise ValueError("The Live settings changed while loading the configuration. Review the current settings again.")
        return config

    async def live_assistant_continue(self, payload=None, **_):
        state = self.server.state
        if self._assistant_locked() or state.live_assistant_busy:
            return
        self._assistant_clear_reply_choices()
        step = state.live_assistant_step
        token = object()
        self._assistant_request = token
        state.live_assistant_busy = True
        state.live_assistant_error = ""
        try:
            self._assistant_ingest(payload)
            if step == "experiment":
                state.powgen_ipts = self._normalize_powgen_ipts(state.powgen_ipts)
                discovery = await self.refresh_powgen_samples(state.powgen_ipts)
                if state.powgen_sample_discovery_ipts != state.powgen_ipts:
                    raise ValueError(state.powgen_sample_message)
                if discovery and discovery.get("source_unavailable"):
                    # The default family is only a starting choice. Archived
                    # experiments may have other valid families; let the user
                    # choose one without claiming that the experiment failed.
                    self._assistant_say("assistant", state.powgen_sample_message)
                    self._assistant_step("experiment")
                    return
                # Inventory before chemistry or a candidate library is required.
                await self._assistant_inspect()
                self._assistant_confirm(step, f"{state.live_beamline_label} · {state.powgen_ipts} · {state.powgen_source_subfolder}")
                self._assistant_step("selection")
            elif step == "selection":
                await self._assistant_inspect()
                self._assistant_validate_selection()
                samples = ", ".join(self._powgen_processing_ids() or ()) or "All samples, including unidentified scans"
                banks = ", ".join(map(str, self._powgen_processing_bank_ids() or ())) or "All banks, separately"
                self._assistant_confirm(step, f"Samples: {samples}. Banks: {banks}.")
                self._assistant_step("structures")
            elif step == "structures":
                library = self._live_library_selection()
                if state.powgen_main_cif_source not in {"none", "computer", "galaxy"}:
                    raise ValueError("Choose the dominant-phase CIF source, or no supplied main phase.")
                if state.powgen_main_cif_source == "computer":
                    path = Path(str(state.powgen_main_cif_path or ""))
                    if not path.is_file() or path.stat().st_size > 20 * 1024 * 1024:
                        raise ValueError("Select a main-phase CIF of at most 20 MB, or choose no supplied main phase.")
                    inspect_cif_upload(path.read_bytes(), path.name)
                elif state.powgen_main_cif_source == "galaxy" and not state.powgen_main_cif_dataset_id:
                    raise ValueError("Choose the main-phase CIF from History, or choose no supplied main phase.")
                self._assistant_confirm(step, library["label"] + ("; supplied dominant phase" if state.powgen_main_cif_source != "none" else "; no supplied dominant phase"))
                self._assistant_step("chemistry")
            elif step == "chemistry":
                if state.live_configuration_source == "history":
                    config = await self._assistant_configuration()
                    self._assistant_validate_chemistry(config)
                    self._assistant_confirm(step, "Chemistry from the selected saved configuration.")
                else:
                    self._assistant_validate_chemistry()
                    self._assistant_confirm(step, "Sample: " + state.live_sample_elements + "; environment: " + (state.live_environment_elements or "none"))
                self._assistant_step("settings")
            elif step == "settings":
                config = await self._assistant_configuration()
                self._assistant_validate_chemistry(config)
                state.live_assistant_summary = self._configuration_summary_rows(config)
                self._assistant_confirm(step, f"{config.mode.value.title()} analysis; {config.background_terms} background terms; fit limits use the input pattern's units.")
                self._assistant_step("review")
            else:
                missing = next((s for s in STEPS[:-1] if s not in self._live_conversation.confirmed), None)
                if missing:
                    self._assistant_step(missing)
                    raise ValueError("Confirm " + TITLES[missing].lower() + " for the current inputs first.")
                config = await self._assistant_configuration()
                self._assistant_validate_chemistry(config)
                state.live_assistant_summary = self._configuration_summary_rows(config)
                ready = await self.check_powgen_experiment()
                if not ready:
                    raise ValueError(state.powgen_preflight_message)
                self._assistant_checked_signature = self._powgen_preflight_signature()
                state.live_assistant_complete = True
                self._assistant_say("assistant", "The Live configuration is filled and the experiment check passed. Finish setup when ready; Start monitoring remains your next action.")
        except Exception as exc:
            state.live_assistant_error = str(exc)
        finally:
            if self._assistant_request is token:
                state.live_assistant_busy = False
                state.flush()

    def finish_live_assistant(self, **_):
        state = self.server.state
        if self._assistant_locked() or state.live_assistant_busy:
            return
        if self._library_decision_pending("live"):
            state.live_assistant_error = "Resolve the partial candidate-library review before finishing this setup."
            state.flush()
            return
        if not state.powgen_preflight_ready or self._assistant_checked_signature != self._powgen_preflight_signature():
            state.live_assistant_complete = False
            state.live_assistant_error = "Settings changed. Check the current Live configuration again before finishing."
        else:
            state.live_assistant_open = False
            state.live_setup_panels = [3]
            state.notice = "Live setup completed. Use Start monitoring when ready."
        state.flush()

    def live_assistant_goal(self, goal, **_):
        if self._assistant_locked() or self.server.state.live_assistant_busy:
            return
        self._assistant_clear_reply_choices()
        try:
            if self.server.state.live_configuration_source != "inline":
                raise ValueError("Edit a copy of the saved configuration before changing its settings.")
            self._live_conversation.apply(self.server.state, goal_patch(goal), "Apply " + goal + " analysis goal")
            self._assistant_say("assistant", "Applied " + goal + ". The actual settings are shown in the Live panel and the final review.")
            self.server.state.live_assistant_error = ""
        except ValueError as exc:
            self.server.state.live_assistant_error = str(exc)
        self.server.state.flush()

    async def live_assistant_suggest_elements(self, **_):
        from .library_setup import (
            LibraryDraft, inspect_library_review_bytes, require_library_activation,
        )
        from .uploads import read_browser_upload

        state = self.server.state
        if self._assistant_locked() or state.live_assistant_busy:
            return
        service = self.service
        fields = STEP_FIELDS['structures'] + STEP_FIELDS['chemistry'] + ('live_library_review',)
        def current_scope():
            return signature({**{name: getattr(state, name, None) for name in fields},
                              'history': getattr(self.service, 'history_id', None),
                              'galaxy': getattr(self.service, 'galaxy_url', None)})
        scope = current_scope()
        generation = self._assistant_generation
        state.live_assistant_busy = True
        state.live_assistant_error = ""
        state.flush()
        try:
            if state.live_configuration_source != "inline":
                raise ValueError("Edit a copy of the saved configuration to change its chemistry.")
            self._require_library_decision('live')
            draft = LibraryDraft(state, 'live')
            if draft.database_source == 'archive':
                if draft.library_archive_source == 'galaxy':
                    identifier = str(draft.history_database_id or '')
                    if not identifier:
                        raise ValueError('Choose a candidate library archive first.')
                    # Use the server's read/hash inspection, never editable
                    # preview fields copied back from the browser.
                    preview = await asyncio.to_thread(service.inspect_library_review, identifier)
                elif draft.library_archive_source == 'computer':
                    contents = await asyncio.to_thread(read_browser_upload,
                        str(draft.database_archive_path or ''), max_bytes=500 * 1024 * 1024, suffix='.zip')
                    preview = await asyncio.to_thread(inspect_library_review_bytes, contents)
                else:
                    raise ValueError('Choose a library archive from Galaxy History or upload it here.')
                if generation is not self._assistant_generation:
                    return
                if self._assistant_locked() or self.service is not service or current_scope() != scope:
                    raise ValueError('The selected structures or chemistry changed. Request the element suggestion again.')
                require_library_activation(preview, draft.library_review or {})
                suggestion = preview.get('element_suggestion') or {}
                if not suggestion.get('available'):
                    raise ValueError('The selected library has no complete CIF element suggestion: '
                                     + str(suggestion.get('reason') or 'readable mapped source CIFs are unavailable.')
                                     + ' Review the source CIFs or enter the sample elements directly.')
                elements = set(suggestion['elements'])
            else:
                elements = set(state.live_library_builder_elements or [])
            if state.powgen_main_cif_source == "computer" and state.powgen_main_cif_path:
                path = Path(state.powgen_main_cif_path)
                if path.stat().st_size > 20 * 1024 * 1024:
                    raise ValueError("The selected main CIF exceeds the preview size limit.")
                elements.update(cif_element_suggestion(path.read_text(encoding="utf-8-sig")))
            if not elements:
                raise ValueError("For CIFs selected from History, build the candidate library first, then use its element suggestion. You can also upload a main-phase CIF or enter the sample elements directly.")
            parse_elements(", ".join(sorted(elements)))
            self._live_conversation.apply(state, {"live_sample_elements": ", ".join(sorted(elements))}, "Use CIF element suggestion")
            self._assistant_say("assistant", "Copied the CIF element suggestion. Review it for missing impurity elements and assign environment contributions separately.")
            state.live_assistant_error = ""
        except (OSError, ValueError) as exc:
            if generation is self._assistant_generation:
                state.live_assistant_error = str(exc)
        finally:
            if generation is self._assistant_generation:
                state.live_assistant_busy = False
                state.flush()

    def _preview_scope(self):
        state = self.server.state
        names = STEP_FIELDS["experiment"] + STEP_FIELDS["selection"] + ("live_fit_start", "live_fit_end", "live_ignore_regions")
        return signature({name: getattr(state, name, None) for name in names})

    async def live_assistant_preview(self, **_):
        state = self.server.state
        if self._assistant_locked() or state.live_assistant_busy:
            return
        state.live_assistant_busy = True
        state.live_assistant_error = ""
        state.flush()
        try:
            scope = self._preview_scope()
            inventory = await self._assistant_inspect(allow_waiting=False)
            banks = self._powgen_processing_bank_ids() or tuple(b["value"] for b in inventory["banks"])
            bank = next((b for b in banks if b in {v["value"] for v in inventory["banks"]}), None)
            if bank is None:
                raise ValueError("The preview reduction has none of the selected banks.")
            start, end = state.live_fit_start, state.live_fit_end
            if (start in (None, "")) != (end in (None, "")):
                raise ValueError("Enter both fit limits or leave both blank.")
            limits = None if start in (None, "") else (float(start), float(end))
            if limits and limits[0] >= limits[1]:
                raise ValueError("Fit start must be below fit end.")
            preview = await asyncio.to_thread(preview_pattern, inventory, bank, limits, parse_regions(state.live_ignore_regions))
            if scope != self._preview_scope():
                raise ValueError("The selection or fit region changed. Inspect the current curve again.")
            self._assistant_preview_signature = scope
            self._assistant_preview_inventory = deepcopy(inventory)
            state.live_assistant_preview = preview
            state.live_assistant_plot = {
                "data": [
                    {"x": preview["x"], "y": preview["y"], "type": "scatter", "mode": "lines",
                     "name": "Measured", "connectgaps": False, "line": {"color": "#15543c"}},
                    {"x": preview["x"], "y": preview["background"], "type": "scatter", "mode": "lines",
                     "name": "Estimated envelope", "connectgaps": False, "line": {"color": "#a66a00", "dash": "dash"}},
                ],
                "layout": {"height": 320, "margin": {"l": 65, "r": 16, "t": 35, "b": 55},
                           "xaxis": {"title": {"text": "Native coordinate (" + preview.get('unit', 'degrees 2θ' if state.live_beamline == 'HB2A' else 'µs') + ")"}},
                           "yaxis": {"title": {"text": "Reduced intensity (native basis)"}},
                           "legend": {"orientation": "h", "y": 1.2}},
            }
        except Exception as exc:
            state.live_assistant_preview = {}
            state.live_assistant_error = str(exc)
        finally:
            state.live_assistant_busy = False
            state.flush()

    def live_assistant_apply_background(self, **_):
        state = self.server.state
        if self._assistant_locked() or state.live_assistant_busy:
            return
        try:
            if state.live_configuration_source != "inline":
                raise ValueError("Edit a copy of the saved configuration before changing the background.")
            inventory = self._assistant_preview_inventory
            if not inventory or self._assistant_preview_signature != self._preview_scope():
                raise ValueError("Inspect the current curve and fit region before applying a suggestion.")
            stat = Path(inventory["source_path"]).stat()
            if [stat.st_size, stat.st_mtime_ns] != inventory["source_signature"]:
                raise ValueError("The inspected reduction changed. Refresh the preview.")
            terms = (state.live_assistant_preview.get("recommendation") or {}).get("terms")
            if terms is None:
                raise ValueError("This curve has no background recommendation; review the controls manually.")
            self._live_conversation.apply(state, {
                "live_background_mode": "auto_fixed_points", "live_background_type": "chebyschev-1",
                "live_background_terms": int(terms),
            }, "Use inspected-bank background suggestion")
            self._assistant_say("assistant", f"Applied automatic fixed points / Chebyshev-1 / {terms} terms to the Live setup. The suggestion came from the preview bank; review its suitability for all selected samples and banks.")
            state.live_assistant_error = ""
        except (OSError, ValueError) as exc:
            state.live_assistant_error = str(exc)
        state.flush()

    def _assistant_refresh_sample_choices(self):
        state = self.server.state
        samples = tuple(SampleOption(str(s["value"]), str(s["title"]))
                        for s in state.powgen_discovered_sample_options)
        self._assistant_shown_samples = samples[:MAX_SHOWN_SAMPLES]
        state.live_assistant_sample_choices = [
            {"number": i + 1, "title": s.title} for i, s in enumerate(self._assistant_shown_samples)
        ]
        visible = signature({"step": state.live_assistant_step, "focus": state.live_assistant_focus,
                             "samples": state.powgen_discovered_sample_options,
                             "banks": state.powgen_bank_options})
        if visible != self._assistant_visible_signature:
            self._assistant_visible_signature = visible
            state.live_assistant_context_token = token_hex(12)

    def _assistant_language_context(self):
        state = self.server.state
        return ReplyContext(
            step=state.live_assistant_step, beamline=state.live_beamline,
            samples=tuple(SampleOption(str(s["value"]), str(s["title"]))
                          for s in state.powgen_discovered_sample_options),
            shown_samples=self._assistant_shown_samples,
            banks=tuple(int(b["value"]) for b in state.powgen_bank_options),
            focus=state.live_assistant_focus,
            reference_presets=tuple(state.live_reference_mask_presets or []),
            defer_sample_bank_membership=True,
            wavelengths=tuple(str(option['value']) for option in (getattr(state, 'powgen_wavelength_options', None) or []) if option.get('value')),
        )

    def _assistant_language_signature(self):
        state = self.server.state
        return signature({
            "draft": self._assistant_snapshots(), "step": state.live_assistant_step,
            "focus": state.live_assistant_focus, "samples": state.powgen_discovered_sample_options,
            "banks": state.powgen_bank_options, "shown": state.live_assistant_sample_choices,
        })

    def _assistant_clear_reply_choices(self):
        self._live_conversation.pending = None
        self.server.state.live_assistant_reply_choices = []

    def _assistant_offer_reply_choices(self, reply):
        self._assistant_say("assistant", reply.message)
        if not reply.choices:
            self._assistant_clear_reply_choices()
            return
        self.server.state.flush()  # Settle field watchers before taking a receipt.
        token = token_hex(12)
        self._live_conversation.pending = {
            "signature": self._assistant_language_signature(),
            "token": token, "choices": deepcopy(reply.choices),
        }
        self.server.state.live_assistant_reply_choices = [
            {"id": f"{token}:{i}", "label": f"{i + 1}. {choice.label}"}
            for i, choice in enumerate(reply.choices)
        ]

    async def live_assistant_choose_reply(self, choice_id="", **_):
        state = self.server.state
        if self._assistant_locked() or state.live_assistant_busy:
            return
        try:
            pending = self._live_conversation.pending
            if not pending or pending["signature"] != self._assistant_language_signature():
                raise ValueError("Those choices have changed. Repeat the answer using the current setup.")
            ids = {f"{pending['token']}:{i}": c for i, c in enumerate(pending["choices"])}
            if choice_id not in ids:
                raise ValueError("That choice is no longer available. Use the current buttons.")
            choice = ids[choice_id]
            self._assistant_clear_reply_choices()
            self._assistant_say("user", choice.label)
            await self._assistant_dispatch_reply(ParsedReply(choice.action, choice.value))
        except (ValueError, OSError) as exc:
            state.live_assistant_error = str(exc)
        state.flush()

    async def _assistant_apply_reply_patch(self, values, description):
        state = self.server.state
        if not isinstance(values, dict) or set(values) - EDIT_FIELDS:
            raise ValueError("Only Live setup fields can be changed.")
        context_keys = {"live_beamline", "powgen_ipts"}
        if set(values) & context_keys:
            if "live_beamline" in values and values["live_beamline"] not in BEAMLINES:
                raise ValueError("Choose a registered Live beamline.")
            ipts = self._normalize_powgen_ipts(values["powgen_ipts"]) if "powgen_ipts" in values else None
            if set(values) - context_keys:
                if (values.get("live_beamline", state.live_beamline) != state.live_beamline
                        or (ipts is not None and ipts != self._normalize_powgen_ipts(state.powgen_ipts))):
                    raise ValueError("Find the experiment first, then choose its samples and settings. No part of this request was applied.")
                # An explicit current header is context, not a request to reset
                # its discovered sample list or user-reviewed profile.
                values = {key: value for key, value in values.items() if key not in context_keys}
            else:
                if "live_beamline" in values:
                    if values["live_beamline"] != state.live_beamline and ipts is None:
                        state.powgen_ipts = ""
                    state.live_beamline = values["live_beamline"]
                    self._live_beamline_changed()
                if ipts is not None:
                    state.powgen_ipts = ipts
                    self._powgen_ipts_changed()
                # Undo cannot restore inputs from another experiment without discovery.
                self._live_conversation.undo.clear()
                self._assistant_step("experiment")
                if ipts is not None:
                    await self.live_assistant_continue()
                else:
                    self._assistant_say("assistant", "Beamline updated. What is the IPTS? Enter its digits or fill the IPTS field.")
                return
        scientific = any(k.startswith("live_") and k not in {"live_database_source"} for k in values)
        if scientific and state.live_configuration_source != "inline":
            raise ValueError("Edit a copy of the saved configuration before changing its chemistry or settings.")
        if "live_full_max_passes" in values:
            rounds = values["live_full_max_passes"]
            if type(rounds) is not int or not 1 <= rounds <= 20:
                raise ValueError("Choose an integer maximum of 1 to 20 discovery rounds per pattern.")
            if values.get("live_analysis_mode", state.live_analysis_mode) != "full":
                raise ValueError("Discovery rounds apply to Full analysis. Say 'Full analysis; one discovery round' or edit the Rapid budget.")
            profile = values.get("live_full_profile", state.live_full_profile)
            if profile not in {"quick", "balanced", "thorough", "custom"}:
                raise ValueError("Choose a valid Full search preset before changing its discovery limit.")
            # Presets override stored custom values when the final configuration
            # is resolved. Materialize that same preset, then change only the
            # requested limit; do not silently switch to a smaller preset.
            values = {**{"live_" + key: value for key, value in full_profile_settings(profile).items()},
                      **values, "live_full_profile": "custom"}
        if "powgen_wavelength" in values and values["powgen_wavelength"]:
            if state.live_beamline != 'PG3' or values["powgen_wavelength"] not in {
                    str(option['value']) for option in (getattr(state, 'powgen_wavelength_options', None) or [])}:
                raise ValueError("Choose an exact wavelength from the current POWGEN wavelength policy list. No setup values changed.")
        for key in ("live_sample_elements", "live_environment_elements"):
            if key in values:
                parse_elements(values[key], optional=key == "live_environment_elements")
        if "powgen_processing_sample_ids" in values:
            allowed = {str(s["value"]) for s in state.powgen_discovered_sample_options}
            if set(values["powgen_processing_sample_ids"]) - allowed:
                raise ValueError("A sample is no longer in the current experiment. Refresh the list.")
        scope_fields = {"powgen_processing_scope", "powgen_processing_sample_ids", "powgen_excluded_runs",
                        "powgen_include_background", "powgen_include_ramps"}
        inventory = None
        request_signature = self._assistant_language_signature()
        if set(values) & scope_fields:
            generation = self._assistant_generation
            state.live_assistant_busy = True
            state.flush()
            try:
                inventory = await self._assistant_inspect(proposed=values, allow_waiting=False)
            finally:
                if generation is self._assistant_generation:
                    state.live_assistant_busy = False
        if request_signature != self._assistant_language_signature():
            raise ValueError("The setup changed while checking this request. No part was applied; repeat it for the current settings.")
        banks = values.get("powgen_processing_banks", [])
        if "nomad_physical_bank" in values:
            banks = [values["nomad_physical_bank"]]
        available = inventory["banks"] if inventory is not None else state.powgen_bank_options
        if set(banks) - {int(b["value"]) for b in available}:
            raise ValueError("A bank is no longer in the current experiment. Refresh the list.")
        self._live_conversation.apply(state, values, description or "Use typed Live setup values")
        if inventory is not None:
            state.flush()
            self._assistant_publish_inventory(inventory)
        self._assistant_say("assistant", (description or "Updated the Live setup fields") + ". Review the values before continuing.")

    async def _assistant_dispatch_reply(self, reply):
        state = self.server.state
        action, value = reply.action, reply.value
        if action == "help":
            topic = value or {"experiment": "profile", "selection": "banks", "structures": "library", "chemistry": "chemistry", "settings": "goal", "review": "goal"}[state.live_assistant_step]
            self._assistant_say("assistant", (reply.message + " " if reply.message else "") + HELP[topic])
        elif action == "clarify":
            self._assistant_offer_reply_choices(reply)
        elif action == "goal":
            self.live_assistant_goal(value)
        elif action == "back":
            self._assistant_step(STEPS[max(0, STEPS.index(state.live_assistant_step) - 1)])
        elif action == "navigate":
            if value not in STEPS:
                raise ValueError("Choose a Live setup section.")
            self._assistant_step(value)
        elif action in {"next", "continue"}:
            await self.live_assistant_continue()
        elif action == "undo":
            self._assistant_say("assistant", self._live_conversation.undo_last(state))
        elif action == "patch":
            await self._assistant_apply_reply_patch(deepcopy(value), reply.message)
        elif action == "focus":
            self._assistant_step("selection")
            state.live_assistant_focus = value
            self._assistant_refresh_sample_choices()
            self._assistant_say("assistant", "Choose " + value + " using the displayed list or exact bank numbers. The other selections stay as they are.")
        elif action == "keep":
            self._assistant_say("assistant", "Kept the current settings. Continue when you have reviewed them.")
        elif action == "cancel":
            self._assistant_clear_reply_choices()
            self._assistant_say("assistant", "Cancelled that suggestion. Your setup values are unchanged.")
        elif action != "empty":
            self._assistant_say("assistant", reply.message)

    async def live_assistant_send(self, text="", context_token=None, **_):
        state = self.server.state
        if self._assistant_locked() or state.live_assistant_busy:
            return
        state.live_assistant_text = ""
        state.live_assistant_error = ""
        try:
            self._assistant_refresh_sample_choices()
            if context_token is not None and context_token != state.live_assistant_context_token:
                raise ValueError("The question or displayed choices changed. Repeat your reply for the current question.")
            text = str(text or "").strip()
            if not text:
                return
            self._assistant_say("user", text[:MAX_REPLY])
            pending = self._live_conversation.pending
            reply = None
            if pending and len(text) <= MAX_REPLY:
                if pending["signature"] != self._assistant_language_signature():
                    self._assistant_clear_reply_choices()
                    if pending_answer(text, pending["choices"]) is not None:
                        raise ValueError("The setup changed after those choices were shown. Repeat your answer for the current setup.")
                else:
                    reply = pending_answer(text, pending["choices"])
            if reply is None:
                reply = interpret_reply(text, self._assistant_language_context())
            # Explanations may interrupt a pending decision without losing it.
            if reply.action not in {"help", "clarify", "unknown"}:
                self._assistant_clear_reply_choices()
            await self._assistant_dispatch_reply(reply)
        except (ValueError, OSError) as exc:
            state.live_assistant_error = str(exc)
        state.flush()
