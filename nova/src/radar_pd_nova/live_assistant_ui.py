"""Conversation cards bound directly to the ordinary Live configuration."""

from trame.widgets import html, plotly, vuetify3 as vuetify

from .live_assistant import HELP, STEP_FIELDS
from .uploads import NamedFileUpload


LOCK = "powgen_monitoring || live_launch_busy || live_assistant_busy || live_library_builder_active"


class LiveAssistantUi:
    def _assistant_select(self, label, model, items, **kwargs):
        return vuetify.VSelect(label=label, v_model=(model,), items=(items,),
                               item_title="title", item_value="value", density="compact",
                               variant="outlined", disabled=(LOCK,), **kwargs)

    def _assistant_history(self, label, model, items):
        vuetify.VAutocomplete(label=label, v_model=(model,), items=(items,),
                             item_title="display_name", item_value="id", clearable=True,
                             density="compact", variant="outlined", disabled=(LOCK,))

    def _assistant_next(self, step, label="Continue"):
        fields = STEP_FIELDS.get(step, ())
        payload = "{" + ",".join(f"{name}:{name}" for name in fields) + "}"
        vuetify.VBtn(label, click=(self.live_assistant_continue, f"[{payload}]"),
                     disabled=(LOCK,), loading=("live_assistant_busy",),
                     color="#15543c", variant="flat", prepend_icon="mdi-arrow-right")

    def _live_assistant_ui(self):
        with html.Section(v_show="workflow_mode === 'powgen' && live_assistant_open && !powgen_monitoring",
                          classes="radar-assistant", raw_attrs=['aria-label="Live setup assistant"']):
            with html.Div(classes="radar-assistant-heading"):
                with html.Div():
                    html.Div("LIVE BEAMLINE SETUP", classes="radar-kicker")
                    html.H1("Let’s configure your experiment")
                    html.P("A guided setup using experiment metadata and scientific rules. Your choices fill the Live panel as you go.")
                with html.Div(classes="radar-assistant-actions"):
                    vuetify.VBtn("Refresh History inputs", click=self.refresh_history,
                                 id="radar-assistant-refresh-inputs", prepend_icon="mdi-refresh",
                                 variant="text", size="small", disabled=(LOCK,))
                    vuetify.VBtn("Use the Live panel", click=(self.live_assistant_manual, "[0]"),
                                 variant="outlined", size="small", disabled=(LOCK,))
            html.P("Added files in Galaxy? Refresh History inputs to update the profile, CIF and library choices. "
                   "{{ history_instrument_datasets.length }} profiles and {{ history_cif_datasets.length }} CIFs currently available.",
                   v_show="live_assistant_step === 'selection' || live_assistant_step === 'structures'",
                   classes="radar-help-copy", raw_attrs=['aria-live="polite"'])
            with html.Nav(classes="radar-assistant-steps", raw_attrs=['aria-label="Setup questions"']):
                vuetify.VBtn(v_for="item in live_assistant_steps", key="item.value",
                             text=("item.title",),
                             prepend_icon=("item.confirmed ? 'mdi-check-circle-outline' : 'mdi-circle-outline'",),
                             variant=("live_assistant_step === item.value ? 'tonal' : 'text'",),
                             color="#15543c", size="small", disabled=(LOCK,),
                             click=(self.live_assistant_go, "[item.value]"))
            with html.Div(classes="radar-assistant-conversation", raw_attrs=['aria-live="polite"', 'aria-label="Recent conversation"']):
                with html.Div(v_for="(message, index) in live_assistant_messages", key="index",
                              classes=("'radar-assistant-message ' + (message.role === 'user' ? 'is-user' : '')",)):
                    html.Span("{{ message.role === 'user' ? 'You' : 'Setup assistant' }}", classes="radar-assistant-speaker")
                    html.P("{{ message.text }}")
            with html.Div(v_if="live_assistant_reply_choices.length", classes="radar-assistant-clarification",
                          raw_attrs=['role="group"', 'aria-label="Clarify your reply"']):
                html.Strong("Choose an interpretation")
                html.P("Nothing from this suggestion is applied until you choose. You can also type an option number, or yes for a single choice.")
                with html.Div(classes="radar-assistant-actions"):
                    vuetify.VBtn(v_for="choice in live_assistant_reply_choices", key="choice.id",
                                 text=("choice.label",), click=(self.live_assistant_choose_reply, "[choice.id]"),
                                 variant="outlined", color="#15543c", disabled=(LOCK,))
                    vuetify.VBtn("Cancel suggestion", click=(self.live_assistant_send, "['cancel']"),
                                 variant="text", disabled=(LOCK,))
            with html.Div(classes="radar-assistant-question"):
                html.Div("SETUP ASSISTANT", classes="radar-micro-label")
                html.H2("{{ live_assistant_prompt }}")
                vuetify.VAlert(v_show="!!live_assistant_error", text=("live_assistant_error",),
                               role="alert", type="warning", variant="tonal", density="compact", classes="mb-3")
                with html.Div(v_if="live_assistant_busy", raw_attrs=['role="status"', 'aria-live="polite"']):
                    vuetify.VProgressLinear(indeterminate=True, color="#15543c", height=3, classes="mb-2")
                    html.P("{{ powgen_sample_loading ? powgen_sample_message : 'Checking the selected experiment inputs and configuration…' }}",
                           classes="radar-help-copy")
                with html.Fieldset(disabled=(LOCK,), classes="radar-plain-fieldset"):
                    self._assistant_experiment_card()
                    self._assistant_selection_card()
                    self._assistant_structure_card()
                    self._assistant_chemistry_card()
                    self._assistant_settings_card()
                    self._assistant_review_card()
            with html.Div(classes="radar-assistant-composer"):
                vuetify.VTextField(v_model=("live_assistant_text",), label="Reply or ask about a setting",
                                   placeholder="I'm on POWGEN, IPTS 12345 · the second one · use balanced",
                                   density="compact", variant="outlined", hide_details=True,
                                   maxlength=1000, disabled=(LOCK,),
                                   __properties=["maxlength"],
                                   __events=[("keydown_enter", "keydown.enter.prevent")],
                                   keydown_enter=(self.live_assistant_send, "[live_assistant_text, live_assistant_context_token]"))
                vuetify.VBtn("Send", click=(self.live_assistant_send, "[live_assistant_text, live_assistant_context_token]"), disabled=(LOCK,),
                             variant="outlined", color="#15543c")
            with html.Details(classes="radar-assistant-help"):
                html.Summary("What can I ask?")
                html.P("Reply with a beamline/IPTS, sample name, displayed sample option number, element list, or analysis goal. Ask what a setting means, say back or undo, or change an earlier choice. Spelling suggestions need confirmation; identifier digits and bank mappings are never guessed.")
                with html.Div(classes="radar-assistant-actions"):
                    for topic in HELP:
                        vuetify.VBtn(topic.capitalize(), click=(self.live_assistant_send, f"['why {topic}']"),
                                     variant="text", size="small", disabled=(LOCK,))

    def _assistant_experiment_card(self):
        with html.Div(v_show="live_assistant_step === 'experiment'"):
            with html.Div(classes="radar-field-pair"):
                self._assistant_select("Beamline", "live_beamline", "live_beamline_options")
                vuetify.VTextField(label="Experiment IPTS", v_model=("powgen_ipts",), placeholder="IPTS-12345",
                                   density="compact", variant="outlined", disabled=(LOCK,))
            self._assistant_select("Reduced-data folder", "powgen_source_subfolder", "powgen_source_options")
            self._assistant_select('Reduced-product normalization', 'hb2a_normalization',
                "[{title:'Time · norm_time.dat',value:'time'},{title:'Monitor · autoreduced.dat',value:'monitor'},{title:'Legacy export; review normalization',value:'legacy'}]",
                v_show="live_beamline === 'HB2A'")
            self._assistant_select('Grouping / export variant', 'snap_product_family', 'snap_product_family_options',
                v_show="live_beamline === 'SNAP'")
            html.P("I read reduced-file and sample metadata from the selected experiment folder. Choose one reduction version; archived copies are kept separate.", classes="radar-help-copy")
            html.P('Monitor normalization uses the measured beam monitor; time normalization uses counting time. Match your reference analysis and review monitor reliability.',v_if="live_beamline === 'HB2A'",classes='radar-help-copy')
            self._assistant_next("experiment", "Find samples and banks")

    def _assistant_selection_card(self):
        with html.Div(v_show="live_assistant_step === 'selection'"):
            html.P("{{ powgen_sample_message }}", classes="radar-help-copy")
            html.P("Reply with exact bank numbers for this question, or say ‘samples’ to return to sample choices.",
                   v_if="live_assistant_focus === 'banks'", classes="radar-help-copy")
            with html.Div(v_if="live_assistant_sample_choices.length && live_assistant_focus !== 'banks'", classes="radar-assistant-sample-options"):
                html.Strong("Numbered sample choices")
                html.P("You can reply with a name or say ‘the second one’. These numbers refer to this sample list; say ‘bank 2’ to choose a bank.", classes="radar-help-copy")
                with html.Ol():
                    html.Li("{{ option.title }}", v_for="option in live_assistant_sample_choices", key="option.number")
                html.P("The first 12 samples are numbered here. The sample control below contains the complete list.",
                       v_if="powgen_discovered_sample_options.length > live_assistant_sample_choices.length", classes="radar-help-copy")
            self._assistant_select("Which samples should share this setup?", "powgen_processing_scope", "powgen_processing_options")
            vuetify.VAutocomplete(v_show="powgen_processing_scope === 'selected'", label="Sample IDs and names",
                                  v_model=("powgen_processing_sample_ids",), items=("powgen_discovered_sample_options",),
                                  item_title="title", item_value="value", multiple=True, chips=True,
                                  closable_chips=True, clearable=True, density="compact", variant="outlined", disabled=(LOCK,))
            with html.Div(v_show="live_beamline === 'PG3'"):
                self._assistant_select("Banks to process", "powgen_bank_scope",
                                       "[{title:'All available banks, separately',value:'all'},{title:'Selected banks',value:'selected'}]")
                self._assistant_select("Selected banks", "powgen_processing_banks", "powgen_bank_options",
                                       v_show="powgen_bank_scope === 'selected'", multiple=True, chips=True)
                self._assistant_select("Wavelength policy", "powgen_wavelength", "powgen_wavelength_options")
            self._assistant_select("Source bank / focused histogram", "nomad_physical_bank", "powgen_bank_options",
                                   v_show="live_beamline === 'NOM' || live_beamline === 'SNAP'")
            html.P("{{ live_assistant_inventory_note }}", classes="radar-help-copy")
            self._assistant_select("Instrument profile", "powgen_instrument_source", "powgen_instrument_source_options")
            with html.Div(v_show="powgen_instrument_source === 'computer'"):
                NamedFileUpload("powgen_instrument_path", label="Matching instrument profile",
                                extensions=[".instprm", ".prm", ".inst", ".ins"],
                                help_text="NOMAD requires .instprm. POWGEN also accepts the supported legacy GSAS profiles.",
                                key="assistant-live-instrument")
            with html.Div(v_show="powgen_instrument_source === 'galaxy'"):
                self._assistant_history("Profile from Galaxy History", "powgen_instrument_dataset_id", "history_instrument_datasets")
            self._live_beamline_review_controls()
            with html.Details(classes="radar-soft-panel mb-3"):
                html.Summary("Run titles and initial monitoring scope")
                self._assistant_select("When monitoring starts", "powgen_backfill_mode", "powgen_backfill_options")
                vuetify.VCheckbox(label="Include background / empty scans", v_model=("powgen_include_background",), density="compact", disabled=(LOCK,))
                vuetify.VCheckbox(label="Include ramp / cooling scans", v_model=("powgen_include_ramps",), density="compact", disabled=(LOCK,))
                vuetify.VAutocomplete(label="Exclude individual runs", v_model=("powgen_excluded_runs",),
                                      items=("powgen_run_options.filter(r => powgen_processing_scope === 'all' || powgen_processing_sample_ids.includes(r.sample_id))",),
                                      item_title="title", item_value="value", multiple=True, chips=True,
                                      density="compact", variant="outlined", disabled=(LOCK,))
                html.P("Sample IDs do not prove sample presence. Review run titles. Later matching scans use the same selection.", classes="radar-help-copy")
            self._assistant_next("selection", "Use these samples and banks")

    def _assistant_structure_card(self):
        with html.Div(v_show="live_assistant_step === 'structures'"):
            self._candidate_library_controls("live", instance="assistant")
            html.H3("Known dominant phase (optional)")
            html.P("Possible or minor phases belong in the candidate library above. A supplied dominant phase is protected by the refinement pipeline.", classes="radar-help-copy")
            self._assistant_select("Dominant-phase CIF", "powgen_main_cif_source", "powgen_main_cif_source_options")
            with html.Div(v_show="powgen_main_cif_source === 'computer'"):
                NamedFileUpload("powgen_main_cif_path", label="Known dominant-phase CIF", extensions=[".cif"],
                                optional=True, key="assistant-live-main-cif")
            with html.Div(v_show="powgen_main_cif_source === 'galaxy'"):
                self._assistant_history("Dominant-phase CIF from History", "powgen_main_cif_dataset_id", "history_cif_datasets")
            self._assistant_next("structures", "Use these structures")

    def _assistant_chemistry_card(self):
        with html.Div(v_show="live_assistant_step === 'chemistry'"):
            self._assistant_select("Analysis configuration", "live_configuration_source", "live_configuration_source_options")
            with html.Div(v_show="live_configuration_source === 'history'"):
                self._assistant_history("Saved Live configuration", "powgen_configuration_dataset_id", "history_configuration_datasets")
                html.P("The saved chemistry and analysis settings will be retained. Sample selection, instrument and structures still come from this Live setup.", classes="radar-help-copy")
                vuetify.VBtn("Edit a copy", click=self.edit_live_configuration, variant="outlined", size="small", disabled=(LOCK,))
            with html.Div(v_show="live_configuration_source === 'inline'"):
                with html.Div(classes="radar-field-pair"):
                    vuetify.VTextField(label="Sample element symbols", v_model=("live_sample_elements",),
                                       placeholder="Fe, O", density="compact", variant="outlined", disabled=(LOCK,))
                    vuetify.VTextField(label="Can / environment symbols", v_model=("live_environment_elements",),
                                       placeholder="Al, V (optional)", density="compact", variant="outlined", disabled=(LOCK,))
                vuetify.VBtn("Use CIF element suggestion", click=self.live_assistant_suggest_elements,
                             variant="outlined", size="small", disabled=(LOCK,))
                html.P("The sample list permits candidate chemistry, including impurities. It is not a formula. CIF suggestions may omit contaminants; environment filtering does not automatically mask peaks.", classes="radar-help-copy")
            self._assistant_next("chemistry", "Confirm chemistry")

    def _assistant_settings_card(self):
        with html.Div(v_show="live_assistant_step === 'settings'"):
            with html.Div(v_show="live_configuration_source === 'inline'"):
                with html.Div(classes="radar-assistant-goals"):
                    for goal, title, description in (
                        ("balanced", "Balanced discovery", "A first Live setup using the existing Balanced search preset."),
                        ("rapid", "Rapid hypotheses", "Rank possible phase combinations with focused final refinements."),
                        ("thorough", "Thorough discovery", "A broader search when weak unexplained peaks matter."),
                    ):
                        with html.Div(classes="radar-assistant-goal"):
                            html.Strong(title)
                            html.P(description)
                            vuetify.VBtn("Apply " + goal, click=(self.live_assistant_goal, f"['{goal}']"),
                                         variant="outlined", size="small", disabled=(LOCK,))
                html.P("Current: {{ live_analysis_mode === 'full' ? 'Full / ' + live_full_profile : 'Rapid / ' + live_rapid_phases_per_hypothesis + ' candidate phases per hypothesis' }}. Continue to keep these settings.", classes="radar-help-copy")
                with html.Div(v_show="live_analysis_mode === 'rapid'", classes="radar-field-pair"):
                    self._number_field(label="Candidate phases per hypothesis", v_model=("live_rapid_phases_per_hypothesis",), density="compact", variant="outlined", disabled=(LOCK,))
                    self._number_field(label="Number of final refinements", v_model=("live_rapid_gsas_validation_limit",), density="compact", variant="outlined", disabled=(LOCK,))
                with html.Details(classes="radar-soft-panel mb-3"):
                    html.Summary("Fit region, background and curve inspection")
                    self._regions_controls("live_", locked=LOCK)
                    self._background_controls("live_", locked=LOCK)
                    vuetify.VBtn("Inspect a representative curve", click=self.live_assistant_preview,
                                 prepend_icon="mdi-chart-line", variant="outlined", size="small", disabled=(LOCK,))
                    self._assistant_curve_preview()
                with html.Details(classes="radar-soft-panel mb-3"):
                    html.Summary("Reuse a configuration file")
                    NamedFileUpload("live_config_import_path", label="Import Live settings", extensions=[".yaml", ".yml", ".json"],
                                    optional=True, key="assistant-live-config")
                    vuetify.VBtn("Apply imported settings", click=self.apply_live_uploaded_configuration,
                                 variant="outlined", size="small", disabled=(LOCK + " || !live_config_import_path",))
            html.P("Magnetic checks, reference masks, refinement safeguards, custom budgets and additional result plots remain available in the Live panel.", classes="radar-help-copy")
            with html.Div(classes="radar-assistant-actions"):
                vuetify.VBtn("Edit advanced Live settings", click=(self.live_assistant_manual, "[2]"),
                             variant="text", size="small", disabled=(LOCK,))
                self._assistant_next("settings", "Review this setup")

    def _assistant_curve_preview(self):
        with html.Div(v_if="live_assistant_preview.run_id", classes="radar-assistant-preview"):
            html.Strong("{{ live_assistant_preview.run_id }} · bank {{ live_assistant_preview.bank }} · sample {{ live_assistant_preview.sample_id }}")
            html.P("{{ live_assistant_preview.valid_points }} valid points · {{ live_assistant_preview.negative_points }} negative intensities retained · {{ live_assistant_preview.unit }}", classes="radar-help-copy")
            plotly.Figure(state_variable_name="live_assistant_plot", responsive=True,
                          display_logo=False, style="width:100%;height:320px;")
            html.P("{{ live_assistant_preview.recommendation.reason }}")
            html.P("{{ live_assistant_preview.recommendation.limitation }}", classes="radar-help-copy")
            html.P("{{ live_assistant_preview.display_note }}", classes="radar-help-copy")
            html.P("{{ (live_assistant_preview.notices || []).join('; ') }}", classes="radar-help-copy")
            vuetify.VBtn("Apply background suggestion", v_show="live_assistant_preview.recommendation.terms != null",
                         click=self.live_assistant_apply_background, variant="outlined", size="small", disabled=(LOCK,))

    def _assistant_review_card(self):
        with html.Div(v_show="live_assistant_step === 'review'"):
            with html.Div(classes="radar-preflight-summary"):
                with html.Div(v_for="row in live_assistant_summary", key="row.label", classes="radar-preflight-row"):
                    html.Span("{{ row.label }}", classes="radar-preflight-label")
                    html.Span("{{ row.value }}", classes="radar-preflight-value")
            html.P("The check reads experiment inputs. Your existing Start monitoring action remains separate.", classes="radar-help-copy")
            vuetify.VAlert(v_show="powgen_metadata_notices.length || nomad_quality_notices.length",
                           text=("powgen_metadata_notices.concat(nomad_quality_notices).join('; ')",),
                           type="warning", variant="tonal", density="compact")
            with html.Div(v_if="live_assistant_complete"):
                with html.Div(v_for="row in powgen_preflight_details", key="row.label", classes="radar-preflight-row"):
                    html.Span("{{ row.label }}", classes="radar-preflight-label")
                    html.Span("{{ row.value }}", classes="radar-preflight-value")
            with html.Div(classes="radar-assistant-actions mt-3"):
                self._assistant_next("review", "Check Live configuration")
                vuetify.VBtn("Finish setup", v_show="live_assistant_complete", click=self.finish_live_assistant,
                             color="#15543c", variant="flat", disabled=(LOCK,))

    @staticmethod
    def _live_assistant_css():
        return """
        .radar-assistant { max-width: 1050px; margin: 0 auto; padding: 26px 30px 38px; color: var(--radar-ink); }
        .radar-assistant-heading { display:flex; align-items:flex-start; justify-content:space-between; gap:18px; margin-bottom:18px; }
        .radar-assistant-heading h1 { font-size:26px; line-height:1.25; margin:7px 0 9px; color:var(--radar-brand-900); }
        .radar-assistant-heading p { color:var(--radar-muted); font-size:14px; line-height:1.5; max-width:620px; }
        .radar-assistant-steps { display:flex; flex-wrap:wrap; gap:4px; margin:12px 0 20px; }
        .radar-assistant-conversation { display:flex; flex-direction:column; gap:10px; margin-bottom:16px; }
        .radar-assistant-message { padding:10px 14px; max-width:90%; background:var(--radar-surface); border:1px solid var(--radar-line); border-radius:12px; }
        .radar-assistant-message.is-user { align-self:flex-end; background:#e0efe6; border-color:#c5dfcf; }
        .radar-assistant-speaker { font-size:11px; color:var(--radar-brand-700); font-weight:700; }
        .radar-assistant-message p { margin:4px 0 0; font-size:13px; line-height:1.5; overflow-wrap:anywhere; }
        .radar-assistant-clarification { border:1px solid #8fb7a0; background:#edf5ef; border-radius:12px; padding:16px; margin:16px 0; }
        .radar-assistant-clarification p { font-size:13px; margin:8px 0 12px; }
        .radar-assistant-clarification .v-btn { height:auto; min-height:36px; max-width:100%; }
        .radar-assistant-clarification .v-btn__content { white-space:normal; overflow-wrap:anywhere; padding:6px 0; }
        .radar-assistant-sample-options { padding:12px 14px; margin-bottom:16px; background:var(--radar-surface-muted); border-radius:8px; font-size:13px; }
        .radar-assistant-sample-options ol { padding-left:24px; }
        .radar-assistant-sample-options li { padding:3px 0; overflow-wrap:anywhere; }
        .radar-assistant-question { padding:23px; background:var(--radar-surface); border:1px solid var(--radar-line); border-radius:14px; }
        .radar-assistant-question h2 { margin:8px 0 22px; font-size:19px; line-height:1.5; font-weight:600; }
        .radar-assistant-question h3 { font-size:16px; margin:20px 0 8px; }
        .radar-assistant-question .radar-help-copy { font-size:13px; line-height:1.5; }
        .radar-assistant-question .radar-field-pair { gap:14px; }
        .radar-assistant-question .radar-preflight-label { font-size:12px; }
        .radar-assistant-question .radar-preflight-value { font-size:13px; }
        .radar-assistant-question .radar-preflight-row { grid-template-columns:minmax(125px, .4fr) minmax(0, 1fr); padding:9px 2px; }
        .radar-assistant-goals { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-bottom:16px; }
        .radar-assistant-goal { padding:15px; border:1px solid var(--radar-line); border-radius:10px; background:var(--radar-surface-muted); }
        .radar-assistant-goal p { font-size:13px; line-height:1.5; margin:8px 0 16px; }
        .radar-assistant-actions { display:flex; flex-wrap:wrap; align-items:center; gap:10px; }
        .radar-assistant-composer { display:grid; grid-template-columns:minmax(0,1fr) auto; align-items:center; gap:10px; margin-top:20px; }
        .radar-assistant-help { margin:14px 0; color:var(--radar-muted); font-size:13px; }
        .radar-assistant-help p { margin:10px 0; }
        .radar-assistant-preview { margin-top:16px; min-width:0; overflow-wrap:anywhere; }
        @media (max-width:1100px) { .radar-assistant { padding:20px 16px; } .radar-assistant-goals { grid-template-columns:1fr; } }
        .radar-layout.is-guided-live { grid-template-columns:minmax(0,1fr) !important; }
        .radar-layout.is-guided-live > .radar-setup-rail { display:none; }
        @media (max-width:620px) { .radar-assistant-heading { flex-direction:column; } .radar-assistant-question { padding:16px; } .radar-assistant-question .radar-field-pair { grid-template-columns:1fr; } .radar-assistant-composer { grid-template-columns:1fr; } .radar-assistant-message { max-width:100%; } }
        """
