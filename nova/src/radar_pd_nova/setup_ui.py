"""Editors shared by the two independent analysis workflows."""

from trame.widgets import html
from trame.widgets import vuetify3 as vuetify

from .uploads import NamedFileUpload, NamedMultiCifUpload


class SetupEditorMixin:
    def _chemistry_controls(self, prefix="", locked=None):
        vuetify.VTextField(
            label="Sample elements",
            v_model=(f"{prefix}sample_elements",),
            placeholder="Tb, Be, Ge, O",
            density="compact",
            variant="outlined",
            hint="Comma- or space-separated symbols",
            persistent_hint=True,
            disabled=(locked,) if locked else False,
        )

    def _regions_controls(self, prefix="", locked=None):
        with html.Div(classes="radar-field-pair"):
            self._number_field(
                label="Fit start (pattern x-axis)",
                v_model=(f"{prefix}fit_start",),
                density="compact",
                variant="outlined",
                clearable=True,
                disabled=(locked,) if locked else False,
            )
            self._number_field(
                label="Fit end (pattern x-axis)",
                v_model=(f"{prefix}fit_end",),
                density="compact",
                variant="outlined",
                clearable=True,
                disabled=(locked,) if locked else False,
            )
        vuetify.VTextarea(
            label="Ignored regions",
            v_model=(f"{prefix}ignore_regions",),
            placeholder="One start,end pair per line\n2.0, 3.2",
            density="compact",
            variant="outlined",
            rows=2,
            auto_grow=True,
            disabled=(locked,) if locked else False,
        )
        html.P(
            "Use the same coordinate and units as the measured pattern: TOF in microseconds for GSAS/POWGEN data, or 2-theta in degrees for constant-wavelength data.",
            classes="radar-field-note",
        )

    def _background_controls(self, prefix="", locked=None):
        vuetify.VSelect(
            label="Background correction",
            v_model=(f"{prefix}background_mode",),
            items=(
                "[{title:'Automatic fixed points',value:'auto_fixed_points'},{title:'Manual / pipeline default',value:'manual'}]",
            ),
            item_title="title",
            item_value="value",
            density="compact",
            variant="outlined",
            disabled=(locked,) if locked else False,
        )
        with html.Div(classes="radar-field-pair"):
            vuetify.VSelect(
                label="Function",
                v_model=(f"{prefix}background_type",),
                items=("['chebyschev-1','chebyschev','cosine','Q^2 power series']",),
                density="compact",
                variant="outlined",
                disabled=(locked,) if locked else False,
            )
            self._number_field(
                label="Terms",
                v_model=(f"{prefix}background_terms",),
                min=1,
                max=36,
                density="compact",
                variant="outlined",
                disabled=(locked,) if locked else False,
            )

    def _budget_controls(self, prefix="", locked=None, compact=False):
        with (
            html.Div(v_show=f"{prefix}analysis_mode === 'rapid'"),
            html.Div(classes="radar-field-pair"),
        ):
            self._number_field(
                label="Phases / hypothesis",
                v_model=(f"{prefix}rapid_phases_per_hypothesis",),
                min=1,
                max=5,
                density="compact",
                variant="outlined",
                disabled=(locked,) if locked else False,
            )
            self._number_field(
                label="Retained / stage",
                v_model=(f"{prefix}rapid_stage_output_limit",),
                min=3,
                max=50,
                density="compact",
                variant="outlined",
                disabled=(locked,) if locked else False,
            )
            self._number_field(
                label="Final refinements",
                v_model=(f"{prefix}rapid_gsas_validation_limit",),
                min=0,
                density="compact",
                variant="outlined",
                disabled=(locked,) if locked else False,
            )
            self._number_field(
                label="Parallel workers",
                v_model=(f"{prefix}rapid_parallel_workers",),
                min=1,
                max=16,
                density="compact",
                variant="outlined",
                disabled=(locked,) if locked else False,
            )
        with html.Div(v_show=f"{prefix}analysis_mode === 'full'"):
            vuetify.VSelect(
                label="Search profile",
                v_model=(f"{prefix}full_profile",),
                items=(
                    "[{title:'Quick',value:'quick'},{title:'Balanced',value:'balanced'},{title:'Thorough',value:'thorough'},{title:'Custom',value:'custom'}]",
                ),
                item_title="title",
                item_value="value",
                density="compact",
                variant="outlined",
                disabled=(locked,) if locked else False,
            )
            vuetify.VAlert(
                v_show=f"{prefix}full_profile === 'quick'",
                text="One discovery pass for a fast first assessment.",
                type="info",
                variant="tonal",
                density="compact",
            )
            vuetify.VAlert(
                v_show=f"{prefix}full_profile === 'balanced'",
                text="Up to two discovery passes; the search may stop early when the accepted model no longer improves.",
                type="info",
                variant="tonal",
                density="compact",
            )
            vuetify.VAlert(
                v_show=f"{prefix}full_profile === 'thorough'",
                text="Up to three discovery passes. Small Rwp changes alone do not stop the residual search, but scientific safety checks still apply.",
                type="info",
                variant="tonal",
                density="compact",
            )
            with html.Div(
                v_show=f"{prefix}full_profile === 'custom'",
                classes="radar-custom-budget-controls",
            ):
                if compact:
                    with html.Div(classes="radar-custom-budget-grid"):
                        self._number_field(
                            label="Discovery rounds",
                            v_model=(f"{prefix}full_max_passes",),
                            min=1,
                            max=20,
                            density="compact",
                            variant="outlined",
                            disabled=(locked,) if locked else False,
                        )
                        self._number_field(
                            label="Minimum phase wt%",
                            v_model=(f"{prefix}full_min_phase_percent",),
                            min=0,
                            density="compact",
                            variant="outlined",
                            disabled=(locked,) if locked else False,
                        )
                    with html.Details(classes="radar-soft-panel mb-2"):
                        html.Summary("Candidate search budget")
                        self._custom_budget_controls(
                            prefix, locked, include_basic=False
                        )
                else:
                    self._custom_budget_controls(prefix, locked)

    def _expert_controls(self, prefix="", main_phase_ready="false", locked=None):
        vuetify.VSwitch(
            v_model=(f"{prefix}reference_masks_enabled",),
            label="Mask reference/can peaks",
            color="#15543c",
            density="compact",
            inset=True,
            disabled=(locked,) if locked else False,
        )
        vuetify.VSelect(
            v_show=f"{prefix}reference_masks_enabled",
            label="Reference structures",
            v_model=(f"{prefix}reference_mask_presets",),
            items=("['Al_fcc','Cu_fcc','V_bcc']",),
            multiple=True,
            chips=True,
            density="compact",
            variant="outlined",
            disabled=(locked,) if locked else False,
        )
        vuetify.VSelect(
            v_show=f"{prefix}reference_masks_enabled",
            label="Reference-mask window",
            v_model=(f"{prefix}reference_window_mode",),
            items=(
                "[{title:'Automatic from resolution',value:'auto'},{title:'Fixed window',value:'fixed'}]",
            ),
            item_title="title",
            item_value="value",
            density="compact",
            variant="outlined",
            disabled=(locked,) if locked else False,
        )
        self._number_field(
            v_show=f"{prefix}reference_masks_enabled && {prefix}reference_window_mode === 'fixed'",
            label="Fixed half-width (pattern x-axis units)",
            v_model=(f"{prefix}reference_fixed_half_width",),
            min=0,
            clearable=True,
            density="compact",
            variant="outlined",
            hint="Blank uses the instrument-aware default",
            persistent_hint=True,
            disabled=(locked,) if locked else False,
        )
        with html.Div(
            v_show=f"{prefix}reference_masks_enabled && {prefix}reference_window_mode === 'auto'",
            classes="radar-field-pair",
        ):
            self._number_field(
                label="FWHM multiplier",
                v_model=(f"{prefix}reference_fwhm_factor",),
                min=0,
                step="0.1",
                density="compact",
                variant="outlined",
                disabled=(locked,) if locked else False,
            )
            self._number_field(
                label="Fractional d tolerance",
                v_model=(f"{prefix}reference_fractional_d_tolerance",),
                min=0,
                step="0.001",
                density="compact",
                variant="outlined",
                disabled=(locked,) if locked else False,
            )
            self._number_field(
                label="Zero-offset tolerance",
                v_model=(f"{prefix}reference_zero_tolerance",),
                min=0,
                clearable=True,
                density="compact",
                variant="outlined",
                hint="Pattern x-axis units; blank uses the instrument default",
                persistent_hint=True,
                disabled=(locked,) if locked else False,
            )
            self._number_field(
                label="Minimum half-width",
                v_model=(f"{prefix}reference_min_half_width",),
                min=0,
                clearable=True,
                density="compact",
                variant="outlined",
                hint="Pattern x-axis units",
                persistent_hint=True,
                disabled=(locked,) if locked else False,
            )
            self._number_field(
                label="Maximum half-width",
                v_model=(f"{prefix}reference_max_half_width",),
                min=0,
                clearable=True,
                density="compact",
                variant="outlined",
                hint="Pattern x-axis units",
                persistent_hint=True,
                disabled=(locked,) if locked else False,
            )
        vuetify.VSwitch(
            v_show=f"{prefix}radiation === 'xray' && {prefix}reference_masks_enabled",
            v_model=(f"{prefix}include_cu_kbeta",),
            label="Mask Cu K-beta companions",
            color="#15543c",
            density="compact",
            inset=True,
            disabled=(locked,) if locked else False,
        )
        vuetify.VDivider(classes="my-3")
        vuetify.VAlert(
            v_show=f"!({main_phase_ready})",
            text="Main-phase safeguards become available after a CIF is actually selected.",
            type="info",
            variant="tonal",
            density="compact",
        )
        vuetify.VSwitch(
            v_show=main_phase_ready,
            v_model=(f"{prefix}main_prenudge",),
            label="Anchor supplied main-phase cell",
            color="#15543c",
            density="compact",
            inset=True,
            disabled=(locked,) if locked else False,
        )
        vuetify.VSwitch(
            v_show=main_phase_ready,
            v_model=(f"{prefix}main_shadow_filter",),
            label="Filter main-shadow-only candidates",
            color="#15543c",
            density="compact",
            inset=True,
            disabled=(locked,) if locked else False,
        )
        vuetify.VSwitch(
            v_show=main_phase_ready,
            v_model=(f"{prefix}cleanup_enabled",),
            label="Clean up main-CIF internal parameters",
            color="#15543c",
            density="compact",
            inset=True,
            disabled=(locked,) if locked else False,
        )
        vuetify.VCheckbox(
            v_show=f"({main_phase_ready}) && {prefix}cleanup_enabled",
            v_model=(f"{prefix}refine_u_iso",),
            label="Refine isotropic displacement parameters",
            density="compact",
            disabled=(locked,) if locked else False,
        )
        vuetify.VCheckbox(
            v_show=f"({main_phase_ready}) && {prefix}cleanup_enabled",
            v_model=(f"{prefix}refine_positions",),
            label="Refine atomic positions",
            density="compact",
            disabled=(locked,) if locked else False,
        )
        vuetify.VSwitch(
            v_show=f"{prefix}radiation === 'xray' && ({main_phase_ready})",
            v_model=(f"{prefix}light_calibration_enabled",),
            label="Calibrate Zero and U/V/W from the supplied main phase",
            color="#15543c",
            density="compact",
            inset=True,
            disabled=(locked,) if locked else False,
        )
        with html.Div(v_show=f"{prefix}analysis_mode === 'rapid'"):
            vuetify.VDivider(classes="my-3")
            vuetify.VSwitch(
                v_model=(f"{prefix}rapid_show_family_variants",),
                label="Keep family variants in Solution Inspector",
                color="#15543c",
                density="compact",
                inset=True,
                disabled=(locked,) if locked else False,
            )
            vuetify.VSwitch(
                v_model=(f"{prefix}rapid_final_polish_enabled",),
                label="Run final polish after ranking",
                color="#15543c",
                density="compact",
                inset=True,
                disabled=(locked,) if locked else False,
            )
        vuetify.VTextField(
            label="Excluded space groups",
            v_model=(f"{prefix}excluded_space_groups",),
            density="compact",
            variant="outlined",
            hint="Comma-separated space-group numbers; leave blank to allow all",
            persistent_hint=True,
            disabled=(locked,) if locked else False,
        )

    def _candidate_library_controls(self, context="single"):
        """Render the same builder with independent uploads and selections."""
        prefix = "live_" if context == "live" else ""

        def key(name):
            return (
                "powgen_database_dataset_id"
                if context == "live" and name == "history_database_id"
                else prefix + name
            )

        def value(name):
            return (key(name),)

        locked = f"{prefix}library_builder_active" + (
            " || powgen_monitoring || live_launch_busy" if prefix else ""
        )
        scope = key("database_source")
        archive_source = key("library_archive_source")
        upload_key = "live" if prefix else "radar"
        vuetify.VSelect(
            label="Which candidate library should RADAR-PD search?",
            v_model=value("database_source"),
            items=("database_source_options",),
            item_title="title",
            item_value="value",
            density="compact",
            variant="outlined",
            disabled=(locked,),
            hint="CIFs and ZIP files build a reusable candidate library. Existing libraries can be selected directly.",
            persistent_hint=True,
        )
        with html.Div(
            v_show=f"{scope} === 'archive'",
            key=f"'{upload_key}-library-archive-panel'",
            classes="radar-library-panel",
        ):
            vuetify.VSelect(
                label="Where is the reusable library archive?",
                v_model=value("library_archive_source"),
                items=("library_archive_source_options",),
                item_title="title",
                item_value="value",
                disabled=(locked,),
                density="compact",
                variant="outlined",
            )
            with html.Fieldset(
                v_show=f"{archive_source} === 'computer'",
                disabled=(locked,),
                classes="radar-plain-fieldset",
            ):
                NamedFileUpload(
                    key("database_archive_path"),
                    label="RADAR-PD library archive",
                    help_text="A library previously produced by the RADAR-PD library builder (.zip)",
                    extensions=[".zip"],
                    optional=False,
                    key=f"{upload_key}-database-upload",
                )
            vuetify.VAutocomplete(
                v_show=f"{archive_source} === 'galaxy'",
                label="Galaxy library archive",
                v_model=value("history_database_id"),
                items=("history_archive_datasets",),
                item_title="display_name",
                item_value="id",
                disabled=(locked,),
                density="compact",
                variant="outlined",
                no_data_text="No library archives are in this history",
            )
        with html.Fieldset(
            v_show=f"{scope} === 'custom_mini' || {scope} === 'custom_augmented'",
            disabled=(locked,),
            key=f"'{upload_key}-library-builder-panel'",
            classes="radar-library-panel radar-plain-fieldset",
        ):
            with html.Div(classes="radar-library-scope-callout"):
                html.Strong(
                    "{{ "
                    + scope
                    + " === 'custom_augmented' ? 'Built-in MP/COD + my structures' : 'Only my structures' }}"
                )
                html.Span(
                    "{{ "
                    + scope
                    + " === 'custom_augmented' ? 'Your CIFs will augment the full built-in catalog.' : 'RADAR-PD will search only the compact library built from your CIFs.' }}"
                )
            vuetify.VTextField(
                label="Library name",
                v_model=value("library_builder_name"),
                density="compact",
                variant="outlined",
                disabled=(locked,),
                hint="A short scientific name, for example Y_Fe_Si_candidate_phases.",
                persistent_hint=True,
            )
            NamedMultiCifUpload(
                key("library_builder_local_paths"),
                key("library_builder_file_rows"),
                key=f"{upload_key}-cif-library-upload",
            )
            html.Div(
                "Optional: add CIF datasets already saved in Galaxy",
                classes="radar-inline-divider",
            )
            vuetify.VAutocomplete(
                label="CIFs from Galaxy History",
                v_model=value("library_builder_cif_ids"),
                items=("history_cif_datasets",),
                item_title="display_name",
                item_value="id",
                multiple=True,
                chips=True,
                closable_chips=True,
                disabled=(locked,),
                density="compact",
                variant="outlined",
            )
            vuetify.VBtn(
                text=(
                    f"{prefix}library_builder_active ? 'Building library...' : 'Build and use this library'",
                ),
                click=(self.build_candidate_library, f"['{context}']"),
                disabled=(
                    f"{locked} || (!{prefix}library_builder_cif_ids.length && !{prefix}library_builder_local_paths.length) || !{prefix}library_builder_name.trim()",
                ),
                loading=value("library_builder_active"),
                prepend_icon="mdi-bookshelf",
                color="#15543c",
                variant="flat",
                block=True,
            )
            html.A(
                "Open the standalone Galaxy library utility",
                href="/?tool_id=neutrons_radar_pd_library_builder_prototype&version=latest",
                target="_blank",
                classes="radar-secondary-link mt-2",
            )
        with html.Div(
            classes="radar-library-build-status",
            v_show=f"{prefix}library_builder_status !== 'idle' || {prefix}library_builder_file_rows.length > 0",
        ):
            pending_revision = (
                f"({scope} === 'custom_mini' || {scope} === 'custom_augmented') && "
                f"({prefix}library_builder_status === 'ready' || {prefix}library_builder_status === 'partial')"
            )
            html.Strong(
                "{{ (" + pending_revision + ") ? 'Build the current CIF selection before using it. Previous builds remain in History.' : "
                + prefix
                + "library_builder_status === 'idle' ? "
                + prefix
                + "library_builder_file_rows.filter(item => item.status === 'Ready').reduce((sum,item) => sum + (item.cif_count || 0), 0) + "
                + prefix
                + "library_builder_cif_ids.length + ' structure source(s) ready to build.' : "
                + prefix
                + "library_builder_message }}"
            )
            vuetify.VProgressLinear(
                model_value=value("library_builder_progress"),
                color="#15543c",
                height=6,
                rounded=True,
                classes="mt-2",
                v_show=f"!({pending_revision})",
            )
            html.Span(
                "{{ "
                + prefix
                + "library_builder_built_count }} usable / {{ "
                + prefix
                + "library_builder_skipped_count }} skipped",
                v_show=f"!({pending_revision}) && ({prefix}library_builder_status === 'ready' || {prefix}library_builder_status === 'partial')",
            )
            with html.Div(
                v_for=f"failure in {prefix}library_builder_failure_rows.slice(0, 8)",
                key="failure.name + failure.reason",
                classes="radar-library-failure-row",
                v_show=f"!({pending_revision})",
            ):
                html.Strong("{{ failure.name }}")
                html.Span("{{ failure.reason }}")

    def _custom_budget_controls(self, prefix="", locked=None, include_basic=True):
        with html.Div(classes="radar-custom-budget-grid"):
            for label, model in (
                ("Discovery rounds", f"{prefix}full_max_passes"),
                ("Minimum phase wt%", f"{prefix}full_min_phase_percent"),
                ("ML candidates", f"{prefix}full_top_n_ml"),
                ("Nudge candidates", f"{prefix}full_nudge_candidates"),
                ("Nudge samples", f"{prefix}full_nudge_samples"),
                ("Nudge representatives", f"{prefix}full_nudge_representatives"),
                ("Comparison candidates", f"{prefix}full_compare_candidates"),
                ("Comparison cycles", f"{prefix}full_compare_cycles"),
                (
                    "Final joint-cell cycles (TOF; 0 disables)",
                    f"{prefix}full_final_joint_cell_cycles",
                ),
                ("Cell length tolerance %", f"{prefix}full_cell_length_tolerance_pct"),
                (
                    "Cell angle tolerance (deg)",
                    f"{prefix}full_cell_angle_tolerance_deg",
                ),
                ("Minimum Rwp improvement", f"{prefix}full_rwp_improvement_threshold"),
                ("Nudge scoring Q max", f"{prefix}full_score_q_max"),
                ("Pearson cell-refine cutoff", f"{prefix}full_pearson_cell_min_r"),
                (
                    "Nudge near-tie score tolerance",
                    f"{prefix}full_lattice_tiebreak_score_tol",
                ),
            ):
                if not include_basic and model in {
                    prefix + "full_max_passes",
                    prefix + "full_min_phase_percent",
                }:
                    continue
                self._number_field(
                    label=label,
                    v_model=(model,),
                    min=0,
                    density="compact",
                    variant="outlined",
                    disabled=(locked,) if locked else False,
                )
        self._number_field(
            label="Duplicate-candidate threshold",
            v_model=(f"{prefix}full_dedup_threshold",),
            min=0,
            max=1,
            step="0.0001",
            density="compact",
            variant="outlined",
            hint="Higher values retain more similar structures and Li-occupancy alternatives. 1.0 can still merge perfectly correlated profiles.",
            persistent_hint=True,
            disabled=(locked,) if locked else False,
        )
        vuetify.VSwitch(
            v_model=(f"{prefix}full_candidate_pruning",),
            label="Use automatic candidate pruning",
            color="#15543c",
            density="compact",
            inset=True,
            disabled=(locked,) if locked else False,
        )
        with html.Div(
            v_show=f"{prefix}full_candidate_pruning", classes="radar-custom-budget-grid"
        ):
            self._number_field(
                label="Knee minimum points",
                v_model=(f"{prefix}full_knee_min_points_hist",),
                min=1,
                density="compact",
                variant="outlined",
                disabled=(locked,) if locked else False,
            )
            self._number_field(
                label="Knee minimum relative span",
                v_model=(f"{prefix}full_knee_min_relative_span",),
                min=0,
                max=1,
                step="0.01",
                density="compact",
                variant="outlined",
                disabled=(locked,) if locked else False,
            )
            self._number_field(
                label="Keep when no knee is found",
                v_model=(f"{prefix}full_knee_keep_if_no_knee",),
                min=0,
                density="compact",
                variant="outlined",
                hint="0 keeps every candidate",
                persistent_hint=True,
                disabled=(locked,) if locked else False,
            )
            self._number_field(
                label="Maximum candidates after pruning",
                v_model=(f"{prefix}full_knee_keep_at_most",),
                min=0,
                density="compact",
                variant="outlined",
                hint="0 removes the cap",
                persistent_hint=True,
                disabled=(locked,) if locked else False,
            )
