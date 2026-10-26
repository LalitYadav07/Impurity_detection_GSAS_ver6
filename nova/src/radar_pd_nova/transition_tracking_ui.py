"""One additional live-results tab, sharing the existing experiment and jobs."""
# Trame's context nesting mirrors the component hierarchy.
# ruff: noqa: SIM117

from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np

from trame.widgets import html, plotly
from trame.widgets import vuetify3 as vuetify

from .post_analysis import number
from .results import discover_plot_payloads, read_plot_payload, figure_for_payload, read_json
from .scan_conditions import condition_display
from .scan_interpretation import InterpretationSettings, _conditions, condition_scope
from .scan_interpretation_ui import phase_labels, scan_label
from .transition_input import digest
from .transition_tracking import TransitionTrackingEngine, acquisition_key, checkpoint
from .transition_signals import BoundaryPolicy, SIGNALS, LOCAL_SIGNALS
from .transition_regions import localized_regions
from .transition_cursor import cursor_frames, selection_figure
from .transition_cursor_widget import TransitionCursor
from .diffraction_coordinates import coordinate_values, display_pattern
from .transition_coordinate_mapping import mapped_calibration, cached_record
from .transition_tracking_plot import (
    COORDINATES,
    empty_figure,
    heatmap_figure,
    locked_limits,
    marker_style,
    profile_figure,
    boundary_figure,
)

EVENTS = {
    "initial": "Initial pattern",
    "new": "New pattern detected",
    "reference": "Comparison reference established",
    "continued": "Matches the selected pattern reference",
    "returned": "Previously seen pattern returns",
    "unassigned": "Pattern comparison needs review",
}
REASONS = {
    "insufficient_reproducible_feature_support": "Insufficient reproducible peak support; no pattern group assigned.",
    "ambiguous_or_nontransitive_state_membership": "Earlier pattern groups do not provide one unambiguous match.",
    "ambiguous_between_existing_pattern_references": "This scan matches more than one earlier reference; its pattern group remains unresolved.",
    "state_memory_point_capacity_reached": "The pattern-memory limit was reached; split the experiment into a narrower watch.",
    "state_memory_capacity_reached": "The pattern-memory limit was reached; split the experiment into a narrower watch.",
    "multiple_result_revisions_require_review": "Multiple results refer to this acquisition; review the active revision.",
}
FINDINGS = {
    "resolved_component_count_change": "Resolved splitting or merging",
    "relative_intensity_redistribution": "Relative intensities changed",
    "local_shape_or_component_change": "Local peak shape changed",
    "new_feature": "Additional resolved signal",
    "lost_feature": "Lost resolved signal",
}


class TransitionTrackingMixin:
    def _init_transition_tracking(self):
        self._transition_engine = TransitionTrackingEngine(policy=BoundaryPolicy())
        self._transition_report = {"records": [], "groups": []}
        self._transition_scans = {}
        self._transition_signature = None
        self._transition_task = None
        self._transition_generation = getattr(self, "_transition_generation", 0) + 1
        self._transition_watch_scope = None
        self._transition_saved = {"references": {}, "limits": {}, "styles": {}}
        self._transition_regions = {}
        self._transition_map_ids = []
        self._transition_map_ranges = {}
        self._transition_map_reset_revisions = {}
        self._transition_cursor_request = None
        self._transition_cursor_task = None
        self._transition_cursor_dragging = False
        self._transition_cursor_client = None
        self._transition_cursor_revision = 0
        self._transition_cursor_applying = False
        self._transition_cursor_frames_key = None
        self._transition_map_stats = {}
        self._transition_render_pending = False
        self._transition_final_summary_cache={}
        for key, value in {
            "transition_series": "",
            "transition_series_options": [],
            "transition_scans": [],
            "transition_selected": "",
            "transition_timeline": [],
            "transition_follow": True,
            "transition_summary": "Waiting for measured patterns",
            "transition_context": "",
            "transition_error": "",
            "transition_loading": False,
            "transition_history_notice": "",
            "transition_scale": "asinh",
            "transition_contrast": 50,
            "transition_background": True,
            "transition_window": 30,
            "transition_window_options": [30, 60, 100],
            "transition_scale_options": [
                {"title": "Asinh · weak and strong peaks", "value": "asinh"},
                {"title": "Linear", "value": "linear"},
                {"title": "Log · positive values", "value": "log"},
            ],
            "transition_model_headers": [
                {"title": "Phase / candidate", "key": "phase"},
                {"title": "Fraction", "key": "fraction"},
                {"title": "Cell", "key": "cell"},
            ],
            "transition_annotation": "temperature",
            "transition_annotation_options": [],
            "transition_comparison": "previous",
            "transition_comparison_options": [],
            "transition_mode": "observed",
            "transition_region": "full",
            "transition_region_options": [],
            "transition_map": plotly.Figure.to_data(
                empty_figure("Waiting for measured curves")
            ),
            "transition_profile": plotly.Figure.to_data(empty_figure("Select a scan")),
            "transition_map_notice": "",
            "transition_map_coordinate": "tof_us",
            "transition_coordinate_options": [{"title":"Time of flight (μs)","value":"tof_us"}],
            "transition_coordinate_notice": "",
            "transition_palette": "Turbo",
            "transition_cursor": 0,
            "transition_cursor_max": 0,
            "transition_cursor_label": "Waiting for scans",
            "transition_cursor_top": 140,
            "transition_cursor_height": 190,
            "transition_cursor_frames": [],
            "transition_cursor_epoch": "",
            "transition_cursor_scope": "",
            "transition_cursor_run_ids": [],
            "transition_cursor_ack": {},
            "transition_preview": {"active":False},
            "transition_pattern_evidence": [],
            "transition_notices": [],
            "transition_model_rows": [],
            "transition_model_notice": "",
            "transition_rwp": "—",
            "transition_conditions": [],
            "transition_condition_summary": "",
            "transition_peer_labels": "",
            "transition_source": "",
            "transition_reference_label": "",
            "transition_can_pin": False,
            "transition_can_open": False,
            "transition_binding": "",
            "transition_binding_options": [],
            "transition_binding_confirmed": False,
            "transition_binding_notice": "",
            "transition_dialog": False,
            "transition_dialog_kind": "profile",
            "transition_dialog_figure": {"data": [], "layout": {}},
            "transition_unavailable": [],
            "transition_policy": asdict(BoundaryPolicy()),
            "transition_boundary_signals": list(BoundaryPolicy().selected),
            "transition_signal_options": [{"title":v,"value":k} for k,v in SIGNALS.items()],
            "transition_vote_options": list(range(1,len(BoundaryPolicy().selected)+1)),
            "transition_width_percent": 3.0,
            "transition_rule_summary": "The first eligible scan establishes a reference.",
            "transition_signal_rows": [],
            "transition_rwp_relative_percent": 10.0,
            "transition_rwp_notice": "",
            "transition_region_overlay": "off",
            "transition_region_signals": list(LOCAL_SIGNALS),
            "transition_region_highlights": list(LOCAL_SIGNALS),
            "transition_region_signal_options": [],
            "transition_region_overlay_notice": "",
            "transition_region_overlay_can_zoom": False,
            "transition_region_reference": "previous",
            "transition_show_split_features": True,
            "transition_region_evidence_rows": [],
            "transition_diagram_axis": "temperature",
            "transition_diagram_axes": [{"title":"Run number","value":"run_number"}],
            "transition_boundary_map": plotly.Figure.to_data(empty_figure("Waiting for measured patterns")),
        }.items():
            setattr(self.server.state, key, value)
        self.server.state.client_only("transition_preview",
            "transition_boundary_signals","transition_region_highlights")

    def _transition_tracking_dashboard(self):
        with html.Section(classes="radar-result-card radar-transition-workspace my-3"):
            with html.Div(classes="radar-scan-heading"):
                with html.Div():
                    html.H3("Pattern history", classes="radar-card-title")
                    html.P("{{ transition_context }}", classes="radar-section-help")
                vuetify.VBtn(
                    text=("transition_follow ? 'Following latest' : 'Follow latest'",),
                    variant=("transition_follow ? 'tonal' : 'outlined'",),
                    size="small",
                    click=self.follow_transition_latest,
                )
            vuetify.VProgressLinear(
                v_if="transition_loading", indeterminate=True, color="#15543c", height=3
            )
            vuetify.VAlert(
                v_if="!!transition_error",
                text=("transition_error",),
                type="error",
                variant="tonal",
                density="compact",
            )
            html.P(
                "{{ transition_history_notice }}",
                v_if="transition_history_notice",
                classes="radar-section-help",
            )
            with html.Div(classes="radar-transition-selectors"):
                vuetify.VSelect(
                    label="Sample / bank / reduction series",
                    v_model=("transition_series",),
                    items=("transition_series_options",),
                    density="compact",
                    variant="outlined",
                    hide_details=True,
                    update_modelValue=(self.select_transition_series, "[$event]"),
                )
                vuetify.VAutocomplete(
                    label="Inspect scan",
                    v_model=("transition_selected",),
                    items=("transition_scans",),
                    density="compact",
                    variant="outlined",
                    hide_details=True,
                    update_modelValue=(self.select_transition_scan, "[$event]"),
                )
                vuetify.VSelect(
                    label="Scans in view",
                    v_model=("transition_window",),
                    items=("transition_window_options",),
                    density="compact",
                    variant="outlined",
                    hide_details=True,
                    update_modelValue=(
                        self.update_transition_display,
                        "['window',$event]",
                    ),
                )
            with html.Div(classes="radar-result-card my-2"):
                html.H4("Boundary indicators")
                with html.Div(classes="radar-transition-policy-controls"):
                    vuetify.VAutocomplete(label="Indicators used for boundaries", multiple=True, chips=True,
                        closable_chips=True, v_model=("transition_boundary_signals",),
                        items=("transition_signal_options",), density="compact", variant="outlined", hide_details=True,
                        update_modelValue=(self.select_transition_boundary_signals,"[$event]"))
                    vuetify.VSelect(label="Indicators required", v_model=("transition_policy.minimum_signals",),
                        items=("transition_vote_options",), density="compact", variant="outlined", hide_details=True,
                        update_modelValue=(self.update_transition_policy,"['minimum_signals',$event]"))
                    vuetify.VSelect(label="Boundary comparison", v_model=("transition_policy.comparison",),
                        items=([{"title":"Current group reference","value":"reference"},
                               {"title":"Preceding eligible scan","value":"previous"}],),
                        density="compact", variant="outlined", hide_details=True,
                        update_modelValue=(self.update_transition_policy,"['comparison',$event]"))
                vuetify.VCheckbox(label="Assume fixed instrumental and sample broadening",density="compact",hide_details=True,
                    v_model=("transition_policy.fixed_broadening",),
                    update_modelValue=(self.update_transition_policy,"['fixed_broadening',$event]"))
                html.P("Each arriving scan updates the map. The selected count is a boundary rule; indicators can share the same physical origin and are not independent probabilities.",classes="radar-section-help")
                html.P("Final Rwp is an optional whole-pattern, model-dependent signal. Pending fits or incompatible recipes/domains are unavailable; they do not count as unchanged.",
                    v_if="transition_policy.selected.includes('rwp')",classes="radar-section-help")
                with html.Details():
                    html.Summary("Indicator sensitivity")
                    with html.Div(classes="radar-transition-selectors"):
                        self._number_field(label="Minimum relative width change (%)",min=.5,max=50,step=.5,
                            v_model=("transition_width_percent",),density="compact",variant="outlined",
                            blur=(self.update_transition_policy,"['width_percent',transition_width_percent]"))
                        self._number_field(label="Width / intensity error ratio",min=2,max=10,step=.5,
                            v_model=("transition_policy.error_ratio",),density="compact",variant="outlined",
                            blur=(self.update_transition_policy,"['error_ratio',transition_policy.error_ratio]"))
                        vuetify.VSelect(label="Supporting peak regions",items=(list(range(1,11)),),
                            v_model=("transition_policy.minimum_features",),density="compact",variant="outlined",
                            update_modelValue=(self.update_transition_policy,"['minimum_features',$event]"))
                    html.P("Width is local FWHM divided by peak position, measured after independent peak matching. Errors are conditional profile-fit diagnostics. Setup/profile changes start a separate comparison series.",classes="radar-section-help")
                    self._number_field(label="Minimum split-separation change (component FWHM)",min=.02,max=2,step=.02,
                        v_model=("transition_policy.split_separation_fwhm",),density="compact",variant="outlined",
                        blur=(self.update_transition_policy,"['split_separation_fwhm',transition_policy.split_separation_fwhm]"))
                    with html.Div(v_if="transition_policy.selected.includes('rwp')",classes="radar-post-controls"):
                        self._number_field(label="Final Rwp change (percentage points)",min=.01,max=100,step=.1,
                            v_model=("transition_policy.rwp_delta_pp",),density="compact",variant="outlined",
                            blur=(self.update_transition_policy,"['rwp_delta_pp',transition_policy.rwp_delta_pp]"))
                        self._number_field(label="Final Rwp relative change (%)",min=1,max=1000,step=1,
                            v_model=("transition_rwp_relative_percent",),density="compact",variant="outlined",
                            blur=(self.update_transition_policy,"['rwp_relative_percent',transition_rwp_relative_percent]"))
                        vuetify.VSelect(label="Rwp direction",v_model=("transition_policy.rwp_direction",),
                            items=([{"title":"Either direction","value":"either"},{"title":"Increase only","value":"increase"}],),
                            density="compact",variant="outlined",
                            update_modelValue=(self.update_transition_policy,"['rwp_direction',$event]"))
                    html.P("Rwp must cross both thresholds. Its change can reflect model adequacy, counting statistics or sample changes; it has no localized peak region.",
                        v_if="transition_policy.selected.includes('rwp')",classes="radar-section-help")
            with html.Ol(
                classes="radar-transition-timeline",
                aria_label="Select a scan by its recurring pattern symbol",
            ):
                with html.Li(v_for="item in transition_timeline", key="item.run_id"):
                    with html.Button(
                        type="button",
                        classes=(
                            "'radar-transition-marker ' + (item.selected ? 'is-selected' : item.related ? 'is-related' : '')",
                        ),
                        aria_label=("item.aria",),
                        aria_pressed=("item.selected",),
                        click=(self.select_transition_scan, "[item.run_id]"),
                    ):
                        vuetify.VIcon(
                            icon=("item.icon",),
                            color=("item.color",),
                            size=25,
                            aria_hidden="true",
                        )
                        html.Span("{{ item.label }}")
            html.P(
                "Matching symbols follow the selected boundary rule and comparison reference. They are pattern groups, not independently identified chemical phases.",
                classes="radar-section-help",
            )
            html.P(
                "{{ transition_summary }}",
                classes="radar-transition-summary",
                role="status",
                aria_live="polite",
            )
            with html.Div(classes="radar-card-heading"):
                html.H3("Boundary indicator history")
                vuetify.VSelect(label="Map horizontal axis",v_model=("transition_diagram_axis",),
                    items=("transition_diagram_axes",),density="compact",variant="outlined",hide_details=True,
                    update_modelValue=(self.update_transition_diagram_axis,"[$event]"))
            with html.Div(style="height:230px"):
                diagram=plotly.Figure(responsive=True,style="height:100%;width:100%",
                    click=(self.select_transition_diagram,"[{points:($event.points || []).map(p=>({customdata:p.customdata}))}]"))
                diagram.data=("transition_boundary_map.data",)
                diagram.layout=("transition_boundary_map.layout",)
            html.P("The vertical axis counts selected indicators that triggered. The dashed line is your required count. Gray error bars show how unavailable indicators could change that count; initial references are not compared. Symbols retain the pattern groups above. Click a point to inspect its reference and evidence.",classes="radar-section-help")
            with html.Div(classes="radar-card-heading"):
                html.H3("Diffraction evolution")
                with html.Div(classes="radar-button-row"):
                    vuetify.VBtn("Reset zoom",variant="text",size="small",
                        click=self.reset_transition_map_zoom)
                    vuetify.VBtn("Expand map",variant="text",size="small",
                        click=(self.open_transition_plot, "['map']"))
            with html.Details(classes="radar-transition-map-options my-2", open=True):
                html.Summary("Map display")
                with html.Div(classes="radar-post-controls my-3"):
                    vuetify.VSelect(label="Diffraction horizontal axis",v_model=("transition_map_coordinate",),
                        items=("transition_coordinate_options",),density="compact",variant="outlined",hide_details=True,
                        update_modelValue=(self.update_transition_display,"['coordinate',$event]"))
                    vuetify.VSelect(label="Color palette",v_model=("transition_palette",),
                        items=([{"title":"Rainbow · high contrast","value":"Turbo"},{"title":"Cividis","value":"Cividis"}],),
                        density="compact",variant="outlined",hide_details=True,
                        update_modelValue=(self.update_transition_display,"['palette',$event]"))
                    vuetify.VSelect(label="Changed-region overlay",v_model=("transition_region_overlay",),
                        items=([{"title":"Off","value":"off"},{"title":"Ribbon only","value":"ribbon"},
                                {"title":"Ribbon + compared scan pair","value":"both"}],),
                        density="compact",variant="outlined",hide_details=True,
                        update_modelValue=(self.update_transition_display,"['overlay',$event]"))
                html.P("{{ transition_coordinate_notice }}",classes="radar-section-help")
                vuetify.VSelect(label="Region comparison",v_if="transition_region_overlay !== 'off'",
                    v_model=("transition_region_reference",),
                    items=([{"title":"Previous scan","value":"previous"},{"title":"Boundary reference","value":"boundary"},
                            {"title":"First scan","value":"first"}],),density="compact",variant="outlined",hide_details=True,
                    update_modelValue=(self.update_transition_display,"['region_reference',$event]"))
                vuetify.VSelect(label="Local indicators to highlight",multiple=True,chips=True,
                    v_if="transition_region_overlay !== 'off'",v_model=("transition_region_highlights",),
                    items=("transition_region_signal_options",),density="compact",variant="outlined",hide_details=True,
                    update_modelValue=(self.select_transition_region_signals,"[$event]"))
                html.P("{{ transition_preview.active ? transition_preview.notice : transition_region_overlay_notice }}",v_if="transition_region_overlay !== 'off'",classes="radar-section-help")
                vuetify.VCheckbox(label="Include recognized split features below the change threshold",
                    v_if="transition_region_overlay !== 'off'",v_model=("transition_show_split_features",),density="compact",hide_details=True,
                    update_modelValue=(self.update_transition_display,"['split_features',$event]"))
                with html.Details(v_if="transition_region_overlay !== 'off' && transition_region_evidence_rows.length"):
                    html.Summary("Local component measurements")
                    vuetify.VDataTable(items=("transition_preview.active ? transition_preview.evidence : transition_region_evidence_rows",),items_per_page=16,density="compact",hide_default_footer=True,
                        headers=([{"title":"Region","key":"region"},{"title":"Evidence","key":"status"},
                                 {"title":"Earlier separation","key":"before"},{"title":"Later separation","key":"after"},
                                 {"title":"Change / FWHM","key":"effect"},{"title":"Conditional error ratio","key":"error"}],))
                vuetify.VBtn("Zoom to highlighted regions",v_if="transition_region_overlay !== 'off'",
                    disabled=("!transition_region_overlay_can_zoom",),variant="text",size="small",
                    click=self.zoom_transition_regions)
                with html.Div(classes="radar-transition-map-controls"):
                    vuetify.VSelect(
                        label="Intensity scale",
                        v_model=("transition_scale",),
                        items=("transition_scale_options",),
                        density="compact",
                        variant="outlined",
                        hide_details=True,
                        update_modelValue=(
                            self.update_transition_display,
                            "['scale',$event]",
                        ),
                    )
                    vuetify.VSlider(
                        label="Weak-intensity contrast",
                        v_model=("transition_contrast",),
                        min=0,
                        max=100,
                        step=1,
                        hide_details=True,
                        color="#15543c",
                        disabled=("transition_scale !== 'asinh'",),
                        end=(self.update_transition_display, "['contrast',$event]"),
                    )
                    vuetify.VSelect(
                        label="Row annotation",
                        v_model=("transition_annotation",),
                        items=("transition_annotation_options",),
                        density="compact",
                        variant="outlined",
                        hide_details=True,
                        update_modelValue=(
                            self.update_transition_display,
                            "['annotation',$event]",
                        ),
                    )
                vuetify.VCheckbox(
                    label="Remove estimated local background in the map",
                    v_model=("transition_background",),
                    density="compact",
                    hide_details=True,
                    update_modelValue=(
                        self.update_transition_display,
                        "['background',$event]",
                    ),
                )
                vuetify.VBtn(
                    "Rescale colors to visible scans",
                    variant="text",
                    size="small",
                    click=self.rescale_transition_map,
                )
            TransitionCursor(
                figure=("transition_map",),frames=("transition_cursor_frames",),
                run_ids=("transition_cursor_run_ids",),cursor=("transition_cursor",),
                scope=("transition_cursor_scope",),epoch=("transition_cursor_epoch",),
                ack=("transition_cursor_ack",),top=("transition_cursor_top",),height=("transition_cursor_height",),
                begin=(self.begin_transition_cursor,"[$event.revision,$event.scope,$event.client_id]"),
                commit=(self.commit_transition_cursor,"[$event.run_id,$event.epoch,$event.index,$event.revision,$event.scope,$event.client_id]"),
                click=(self.select_transition_heatmap,"[$event]"),
                relayout=(self.relayout_transition_map,"[$event]"),reset=self.reset_transition_map_zoom,
                preview="transition_preview = $event")
            html.P("{{ transition_preview.active ? transition_preview.label : transition_cursor_label }} · White dashed lines: boundary candidates. Green pointer: selected scan. Outlined boxes span the compared scan pair. Move the cursor to preview a scan; release to load its full details. The ribbon follows the map's horizontal zoom and intensity treatment.",classes="radar-section-help")
            html.P("{{ transition_map_notice }}", classes="radar-section-help")
            with html.Div(classes="radar-transition-inspector-heading"):
                html.H3("{{ transition_summary }}")
                with html.Div(classes="radar-button-row"):
                    vuetify.VBtn(
                        "Pin selected scan as reference",
                        variant="outlined",
                        size="small",
                        disabled=("!transition_can_pin",),
                        click=self.pin_transition_reference,
                    )
                    vuetify.VBtn(
                        "Open full scan result",
                        variant="outlined",
                        size="small",
                        disabled=("!transition_can_open",),
                        click=self.open_transition_scan,
                    )
            html.P("{{ transition_condition_summary }}", v_if="transition_condition_summary", classes="radar-section-help")
            with html.Div(classes="radar-scan-evidence-grid"):
                with html.Div(classes="radar-scan-curve"):
                    with vuetify.VBtnToggle(
                        v_model=("transition_mode",),
                        mandatory=True,
                        color="#15543c",
                        density="compact",
                        update_modelValue=(
                            self.update_transition_display,
                            "['mode',$event]",
                        ),
                    ):
                        vuetify.VBtn(
                            "Observed comparison", value="observed", size="small"
                        )
                        vuetify.VBtn("RADAR-PD fit", value="fit", size="small")
                    with html.Div(classes="radar-transition-plot-controls"):
                        vuetify.VSelect(
                            v_show="transition_mode === 'observed'",
                            label="Compare with",
                            v_model=("transition_comparison",),
                            items=("transition_comparison_options",),
                            density="compact",
                            variant="outlined",
                            hide_details=True,
                            update_modelValue=(
                                self.update_transition_display,
                                "['comparison',$event]",
                            ),
                        )
                        vuetify.VSelect(
                            label="Pattern region",
                            v_model=("transition_region",),
                            items=("transition_region_options",),
                            density="compact",
                            variant="outlined",
                            hide_details=True,
                            update_modelValue=(
                                self.update_transition_display,
                                "['region',$event]",
                            ),
                        )
                        vuetify.VBtn(
                            "Expand plot",
                            variant="text",
                            size="small",
                            click=(self.open_transition_plot, "['profile']"),
                        )
                    with html.Div(classes="radar-scan-chart"):
                        figure = plotly.Figure(
                            display_mode_bar=True,
                            responsive=True,
                            style="height:100%;width:100%",
                        )
                        figure.data = ("transition_profile.data",)
                        figure.layout = ("transition_profile.layout",)
                    html.P(
                        "Original peak positions are retained. Display contrast and zoom do not change pattern membership.",
                        classes="radar-section-help",
                    )
                with html.Aside(classes="radar-scan-evidence"):
                    html.H4("Boundary evidence")
                    html.P("{{ transition_rule_summary }}",classes="radar-section-help")
                    vuetify.VDataTable(items=("transition_signal_rows",),
                        headers=([{"title":"Indicator","key":"indicator"},{"title":"Result","key":"result"},
                                 {"title":"Support","key":"support"}],),
                        density="compact",items_per_page=-1,hide_default_footer=True)
                    html.P("{{ transition_rwp_notice }}",v_if="transition_rwp_notice",classes="radar-section-help")
                    with html.Div(v_if="transition_mode === 'observed'", classes="mt-3"):
                        html.H4("Selected curve comparison")
                        with html.Ul():
                            html.Li(
                                "{{ finding }}",
                                v_for="finding in transition_pattern_evidence",
                                key="finding",
                            )
                    html.P("{{ transition_peer_labels }}", classes="radar-section-help")
                    html.H4("RADAR-PD refinement", classes="mt-4")
                    html.P("Rwp {{ transition_rwp }}")
                    html.P(
                        "{{ transition_model_notice }}", classes="radar-section-help"
                    )
                    vuetify.VDataTable(
                        v_if="transition_model_rows.length",
                        items=("transition_model_rows",),
                        headers=("transition_model_headers",),
                        density="compact",
                        items_per_page=-1,
                        hide_default_footer=True,
                    )
                    with html.Details(classes="mt-3"):
                        html.Summary("Conditions and provenance")
                        html.P("{{ transition_source }}", classes="radar-section-help")
                        html.P(
                            "{{ transition_reference_label }}",
                            classes="radar-section-help",
                        )
                        with html.Ul():
                            html.Li(
                                "{{ condition }}",
                                v_for="condition in transition_conditions",
                                key="condition",
                            )
                            html.Li(
                                "{{ notice }}",
                                v_for="notice in transition_notices",
                                key="notice",
                            )
                        with html.Div(
                            v_if="transition_binding_options.length", classes="mt-3"
                        ):
                            vuetify.VSelect(
                                label="Specimen-connected readback",
                                v_model=("transition_binding",),
                                items=("transition_binding_options",),
                                density="compact",
                                variant="outlined",
                                hide_details=True,
                            )
                            vuetify.VCheckbox(
                                label="I verified that this device measured this specimen",
                                v_model=("transition_binding_confirmed",),
                                density="compact",
                                hide_details=True,
                            )
                            vuetify.VBtn(
                                "Apply device mapping",
                                variant="outlined",
                                size="small",
                                disabled=(
                                    "!transition_binding || !transition_binding_confirmed",
                                ),
                                click=self.bind_transition_condition,
                            )
                        html.P(
                            "{{ transition_binding_notice }}",
                            v_if="transition_binding_notice",
                            classes="radar-section-help",
                        )
                        vuetify.VBtn(
                            "Download experiment tracking report",
                            variant="text",
                            size="small",
                            disabled=("!transition_scans.length",),
                            click=self.download_transition_report,
                        )
            with html.Details(v_if="transition_unavailable.length", classes="mt-3"):
                html.Summary("Scans awaiting comparable curve evidence")
                with html.Ul():
                    html.Li(
                        "{{ item }}", v_for="item in transition_unavailable", key="item"
                    )

    def _transition_tracking_dialog(self):
        with vuetify.VDialog(v_model=("transition_dialog",), fullscreen=True):
            with vuetify.VCard(classes="radar-inspector-card"):
                with html.Div(classes="radar-inspector-toolbar"):
                    html.H3(
                        "{{ transition_dialog_kind === 'map' ? 'Diffraction evolution' : 'Pattern inspector' }}"
                    )
                    vuetify.VBtn("Reset zoom",v_if="transition_dialog_kind === 'map'",variant="text",
                        click=self.reset_transition_map_zoom)
                    vuetify.VBtn(
                        "Close expanded plot",
                        click="transition_dialog = false",
                        variant="outlined",
                    )
                with html.Div(
                    v_if="transition_dialog", classes="radar-inspector-frame"
                ):
                    figure = plotly.Figure(
                        display_mode_bar=True,
                        responsive=True,
                        style="height:100%;width:100%",
                        raw_attrs=[":double-click=\"transition_dialog_kind === 'map' ? false : 'reset+autosize'\""],
                        click=(
                            self.select_transition_heatmap,
                            "[$event?.points?.[0]?.data?.type === 'heatmap' ? $event.points[0].y : null]",
                        ),
                        relayout=(self.relayout_transition_map,"[$event]"),
                        double_click=self.reset_transition_map_zoom,
                    )
                    figure.data = ("transition_dialog_figure.data",)
                    figure.layout = ("transition_dialog_figure.layout",)

    def _transition_sources(self, scans, pending):
        sources = {
            str(s["run_id"]): dict(s) for s in [*pending, *scans] if s.get("run_id")
        }
        seen, duplicates = set(), set()
        for scan in scans:
            run_id = str(scan.get("run_id") or "")
            if run_id in seen:
                duplicates.add(run_id)
            seen.add(run_id)
        controller = getattr(self, "_powgen_controller", None)
        if controller is not None and str(
            getattr(self, "_post_analysis_scope", "")
        ).startswith("watch:"):
            root = getattr(controller, "source_directory", None)
            settings = getattr(controller, "settings", None)
            for stage in ("discovered", "submitted", "completed", "failed"):
                for run in getattr(controller.state, stage, {}).values():
                    row = sources.setdefault(run.run_id, {"run_id": run.run_id})
                    meta = dict(run.scan_metadata or {})
                    row.update(
                        run_number=run.run_number,
                        instrument=run.instrument,
                        ipts=getattr(settings, "ipts", row.get("ipts")),
                        bank=run.bank or meta.get("source_bank"),
                        metadata=meta,
                        source_path=run.source_path,
                        transition_source_root=root,
                        analysis_status=stage,
                        configuration_id=getattr(settings,"configuration_dataset_id",row.get("configuration_id")),
                        candidate_library_id=getattr(settings,"database_dataset_id",row.get("candidate_library_id")),
                    )
                    for key in ("final_rwp","analysis_mode","observables","pattern_observation","result_status"):
                        value=(run.scientific_summary or {}).get("status" if key=="result_status" else key)
                        if (key not in row or row[key] is None) and value is not None:
                            row[key]=value
                    if not row.get("pattern_path"):
                        row["pattern_path"]=(row.get("pattern_observation") or {}).get("local_path")
                    # Older checkpoints retain a generic/best Rwp only. Migrate
                    # from this exact cached job's explicit final field, never
                    # guess that the generic metric is a final-project value.
                    if "rwp" in self.server.state.transition_policy.get("selected",[]) and row.get("final_rwp") is None:
                        record=cached_record(row,controller,getattr(self,"records",{}))
                        output=getattr(record,"local_output_dir",None)
                        path=Path(output)/"summary.json" if output else None
                        try:
                            stat=path.stat() if path else None
                            if stat and stat.st_size<4*1024*1024:
                                key=(str(path),stat.st_size,stat.st_mtime_ns)
                                if key not in self._transition_final_summary_cache:
                                    self._transition_final_summary_cache[key]=read_json(path)
                                    if len(self._transition_final_summary_cache)>128:
                                        self._transition_final_summary_cache.pop(next(iter(self._transition_final_summary_cache)))
                                payload=self._transition_final_summary_cache[key]
                                project=(row.get("observables") or {}).get("source_gpx_sha256")
                                if project and (payload.get("observables") or {}).get("source_gpx_sha256")==project:
                                    row["final_rwp"]=payload.get("final_rwp")
                                    row["result_status"]=payload.get("status")
                        except (OSError,TypeError):
                            pass
        # A pending placeholder and its finished result represent one run;
        # multiple finished revisions require the engine's explicit review
        # outcome instead of silently retaining the final dictionary entry.
        return sorted(
            [
                *sources.values(),
                *(dict(sources[k]) for k in duplicates if k in sources),
            ],
            key=acquisition_key,
        )

    def _refresh_transition_tracking(self, scans, pending=()):
        if not hasattr(self, "_transition_engine"):
            return
        scope = str(getattr(self, "_post_analysis_scope", ""))
        if scope != self._transition_watch_scope:
            self._cancel_transition_cursor()
            self._transition_watch_scope = scope
            self._transition_generation += 1
            controller = getattr(self, "_powgen_controller", None)
            saved = (
                getattr(getattr(controller, "state", None), "transition_tracking", {})
                if scope.startswith("watch:")
                else {}
            )
            self._transition_saved = checkpoint(saved)
            restored_policy = BoundaryPolicy.restore(self._transition_saved.get("policy"))
            self.server.state.transition_policy = asdict(restored_policy)
            self.server.state.transition_boundary_signals=list(restored_policy.selected)
            self.server.state.dirty("transition_boundary_signals")
            self.server.state.transition_width_percent = restored_policy.width_fraction*100
            self.server.state.transition_rwp_relative_percent = restored_policy.rwp_relative*100
            self.server.state.transition_vote_options = list(range(1,max(1,len(restored_policy.selected))+1))
            self._transition_engine = TransitionTrackingEngine(
                styles=self._transition_saved["styles"], policy=restored_policy,
            )
            self._transition_signature = None
            self.server.state.transition_series = ""
            self.server.state.transition_follow = True
            # A new watch's heading is published immediately. Retire the prior
            # watch's inspector and plots before its asynchronous replay starts.
            self._transition_report = {"records": [], "groups": []}
            self._transition_scans = {}
            self.server.state.transition_series_options = []
            self.server.state.transition_history_notice = ""
            self._render_transition_tracking()
        sources = self._transition_sources(scans, pending)
        signatures = []
        for row in sources:
            signature = None
            if row.get("source_path"):
                try:
                    st = Path(row["source_path"]).stat()
                    signature = [st.st_size, st.st_mtime_ns]
                except OSError:
                    signature = "unavailable"
            signatures.append([row, signature])
        policy = BoundaryPolicy.restore(self.server.state.transition_policy)
        signature = digest([scope, signatures, policy.fingerprint])
        if signature == self._transition_signature:
            return
        self._transition_request = (
            signature,
            sources,
            self._transition_generation,
            self._transition_engine,
            policy,
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            if self._transition_task is None:
                self.server.state.transition_loading = True
                self._transition_task = loop.create_task(
                    self._run_transition_tracking()
                )
            return
        try:
            report = self._transition_engine.evaluate_with_policy(sources, policy)
            self._apply_transition_tracking(report, sources, signature)
        except (ValueError, TypeError, OSError) as exc:
            self.server.state.transition_error = str(exc)

    async def _run_transition_tracking(self):
        try:
            while True:
                signature, sources, generation, engine, policy = self._transition_request
                try:
                    report = await asyncio.to_thread(engine.evaluate_with_policy, sources, policy)
                except (ValueError, TypeError, OSError) as exc:
                    if self._transition_request[0] != signature:
                        continue
                    if generation == self._transition_generation:
                        self.server.state.transition_error = str(exc)
                    break
                if self._transition_request[0] != signature:
                    continue
                if generation == self._transition_generation:
                    self._apply_transition_tracking(report, sources, signature)
                    self.server.state.flush()
                break
        except (ValueError, TypeError, OSError) as exc:
            self.server.state.transition_error = str(exc)
            self.server.state.flush()
        finally:
            self._transition_task = None
            self.server.state.transition_loading = False
            self.server.state.flush()

    def _apply_transition_tracking(self, report, scans, signature):
        state = self.server.state
        previous_policy = self._transition_report.get("policy_sha256")
        previous = [
            (r["run_id"], r.get("pattern_sha256"))
            for r in self._transition_report["records"]
        ]
        current = [(r["run_id"], r.get("pattern_sha256")) for r in report["records"]]
        state.transition_history_notice = (
            "Boundary rule changed; received scans were re-evaluated. Refinements were not rerun."
            if previous_policy and previous_policy != report.get("policy_sha256")
            else "Acquisition history changed; pattern membership has been replayed in acquisition order."
            if previous and current[: len(previous)] != previous
            else "Final refinement evidence changed; received boundary assignments were updated under your Rwp rule."
            if previous and report.get("refinement_evidence_updated")
            else ""
        )
        self._transition_report = report
        self._transition_scans = {str(s["run_id"]): s for s in scans}
        self._transition_signature = signature
        self._transition_saved["styles"] = report["styles"]
        if report.get("boundary_policy"):
            self._transition_saved["policy"] = report["boundary_policy"]
        series_options = []
        for group in report["groups"]:
            if not any(r.get("group") == group["id"] for r in report["records"]):
                continue
            scope = group["scope"]
            first = self._transition_scans[group["run_ids"][0]]
            meta = first.get("metadata") or {}
            title = f"{meta.get('sample_name') or meta.get('sample_id')} · {scope['instrument']} bank {scope['bank']}"
            source = next(
                (
                    r.get("source", {}).get("source")
                    for r in report["records"]
                    if r.get("group") == group["id"]
                ),
                None,
            )
            duplicates = [
                o
                for o in series_options
                if o["title"].startswith(title)
            ]
            if duplicates:
                title += f" · reduction {len(duplicates) + 1}"
            if source == "published measured histogram":
                title += " · saved histogram"
            series_options.append(
                {"title": title, "value": group["id"]}
            )
        # Build locally, then assign once: mutating a list already marked clean
        # by Trame can leave a connected browser with no choices after recovery.
        state.transition_series_options = series_options
        options = [o["value"] for o in series_options]
        if state.transition_series not in options:
            state.transition_series = options[0] if options else ""
        state.transition_unavailable = [
            f"{scan_label(self._transition_scans.get(r['run_id'], {}))}: {REASONS.get(r.get('reason'), r.get('reason'))}"
            for r in report["records"]
            if r.get("group") is None
        ]
        state.transition_error = ""
        self._render_transition_tracking()
        self._checkpoint_transition_tracking()

    def _series_rows(self):
        return [
            r
            for r in self._transition_report["records"]
            if r.get("group") == self.server.state.transition_series
        ]

    def _window_rows(self, rows):
        limit = int(self.server.state.transition_window)
        ids = [r["run_id"] for r in rows]
        selected = (
            ids.index(self.server.state.transition_selected)
            if self.server.state.transition_selected in ids
            else len(rows) - 1
        )
        start = (
            max(0, len(rows) - limit)
            if self.server.state.transition_follow
            else max(0, min(selected - limit // 2, len(rows) - limit))
        )
        return rows[start : start + limit]

    def _render_transition_tracking(self):
        if self._transition_cursor_dragging:
            # Keep the map's indices stable during a local preview even if a
            # new acquisition arrives. Received evidence continues to append.
            self._transition_render_pending = True
            return
        self._transition_render_pending = False
        try:
            self._render_transition_tracking_data()
        except (OSError, ValueError, TypeError) as exc:
            # Facility files may disappear or change between a completed
            # evaluation and an operator click. Do not leave an old plot
            # appearing to describe the newly selected acquisition.
            self._transition_unavailable_plot(exc)

    def _transition_unavailable_plot(self, exc):
        state = self.server.state
        state.transition_error = str(exc)
        unavailable = plotly.Figure.to_data(
            empty_figure(
                "Measured curve unavailable; waiting for a stable result receipt"
            )
        )
        state.transition_map = unavailable
        state.transition_profile = unavailable
        state.transition_boundary_map = unavailable
        state.transition_signal_rows = []
        state.transition_rwp_notice=""
        state.transition_region_overlay_notice=""
        state.transition_region_overlay_can_zoom=False
        state.transition_cursor=0
        state.transition_cursor_max=0
        state.transition_cursor_label="No comparable scan"
        state.transition_cursor_frames=[]
        state.transition_cursor_scope=""
        state.transition_cursor_run_ids=[]
        state.transition_region_evidence_rows=[]
        self._transition_cursor_frames_key=None
        state.transition_map_notice = (
            "Refresh when the reduced curve is available again."
        )
        state.transition_pattern_evidence = [str(exc)]
        state.transition_source = "Selected measured curve is unavailable"
        state.transition_peer_labels = ""
        state.transition_reference_label = ""
        state.transition_notices = []
        state.transition_model_rows = []
        state.transition_model_notice = (
            "Measured evidence is unavailable for this selection."
        )
        state.transition_rwp = "—"
        state.transition_conditions = []
        state.transition_condition_summary = ""
        state.transition_can_pin = False
        state.transition_can_open = False
        state.transition_binding_options = []
        state.transition_binding_notice = ""
        state.transition_binding = ""
        state.transition_binding_confirmed = False
        self._transition_binding_candidates = {}
        self._transition_map_ids = []
        if state.transition_dialog:
            self._update_transition_dialog()

    def _render_transition_tracking_data(self):
        state = self.server.state
        rows = self._series_rows()
        if not rows:
            state.transition_boundary_map = plotly.Figure.to_data(empty_figure("Waiting for measured patterns"))
            state.transition_signal_rows = []
            state.transition_rwp_notice=""
            state.transition_region_overlay_notice=""
            state.transition_region_overlay_can_zoom=False
            state.transition_cursor=0
            state.transition_cursor_max=0
            state.transition_cursor_label="Waiting for scans"
            state.transition_cursor_frames=[]
            state.transition_cursor_scope=""
            state.transition_cursor_run_ids=[]
            state.transition_region_evidence_rows=[]
            self._transition_cursor_frames_key=None
            state.transition_timeline = []
            state.transition_scans = []
            state.transition_selected = ""
            state.transition_summary = "Waiting for compatible measured patterns"
            state.transition_context = "Live reductions and published curve receipts are read without running a new refinement."
            state.transition_map = plotly.Figure.to_data(
                empty_figure("Waiting for a known specimen, bank and stable curve")
            )
            state.transition_profile = plotly.Figure.to_data(
                empty_figure("Select a comparable acquisition")
            )
            state.transition_pattern_evidence = []
            state.transition_model_rows = []
            state.transition_rwp = "—"
            state.transition_model_notice = (
                "Refinement context will appear with a selected scan."
            )
            state.transition_peer_labels = ""
            state.transition_source = ""
            state.transition_reference_label = ""
            state.transition_conditions = []
            state.transition_condition_summary = ""
            state.transition_notices = []
            state.transition_can_pin = False
            state.transition_can_open = False
            state.transition_binding_options = []
            state.transition_binding_notice = ""
            state.transition_binding = ""
            state.transition_binding_confirmed = False
            self._transition_binding_candidates = {}
            state.transition_region_options = [
                {"title": "Full pattern", "value": "full"}
            ]
            state.transition_region = "full"
            state.transition_comparison_options = []
            state.transition_map_notice = (
                "No compatible measured series is available yet."
            )
            self._transition_regions = {}
            self._transition_map_ids = []
            return
        ids = [r["run_id"] for r in rows]
        if state.transition_follow or state.transition_selected not in ids:
            state.transition_selected = ids[-1]
        selected = next(r for r in rows if r["run_id"] == state.transition_selected)
        scan = self._transition_scans[selected["run_id"]]
        state.transition_scans = [
            {
                "title": scan_label(self._transition_scans[r["run_id"]]),
                "value": r["run_id"],
            }
            for r in rows
        ]
        window = self._window_rows(rows)
        state.transition_timeline = [
            dict(
                run_id=r["run_id"],
                label=str(
                    self._transition_scans[r["run_id"]].get("run_number") or r["run_id"]
                ),
                aria=scan_label(self._transition_scans[r["run_id"]])
                + " · "
                + EVENTS[r["event"]],
                selected=r is selected,
                related=bool(
                    r.get("state_id") and r.get("state_id") == selected.get("state_id")
                ),
                **marker_style(r.get("style_index")),
            )
            for r in window
        ]
        state.transition_context = f"{len(rows)} acquisitions in this compatible series · showing {len(window)} · {scan_label(scan)}"
        state.transition_summary = scan_label(scan) + " · " + EVENTS[selected["event"]]
        all_conditions = {}
        settings = InterpretationSettings.model_validate(state.scan_interpretation)
        for r in rows:
            for k, v in _conditions(
                self._transition_scans[r["run_id"]], settings
            ).items():
                if v.get("status") == "recorded":
                    all_conditions[k] = v
        state.transition_annotation_options = [
            {"title": "Run number only", "value": "none"}
        ] + [{"title": k.replace("_", " ").title(), "value": k} for k in all_conditions]
        if state.transition_annotation not in [
            o["value"] for o in state.transition_annotation_options
        ]:
            state.transition_annotation = "none"
        self._render_boundary_map(rows, all_conditions)
        decision = selected.get("boundary_comparison") or {}
        baseline_id = decision.get("before")
        baseline = self._transition_scans.get(baseline_id)
        baseline_label = scan_label(baseline) if baseline else str(baseline_id or "reference")
        rule_outcome = (
            "Boundary candidate under the chosen rule." if selected.get("boundary") else
            "Insufficient available indicators to decide this rule." if decision.get("status") == "uncertain" else
            "The chosen boundary rule is not met."
        )
        state.transition_rule_summary = (
            f"Compared with {baseline_label}: {decision.get('votes',0)} selected indicator(s) triggered; "
            f"{decision.get('required',1)} required. " + rule_outcome
            if decision else "First eligible scan establishes the initial comparison reference."
        )
        state.transition_signal_rows = [{"indicator":v["title"] + (" · selected" if v.get("selected") else ""),
            "result":"Select to compare" if k=="rwp" and not v.get("selected") else
                "Triggered" if v["detected"] else "Below setting" if v["available"] else "Unavailable",
            "support":"Whole pattern" if not v.get("localizable",True) else f"{v['support']} / {v['required_features']}"}
            for k,v in decision.get("indicators",{}).items()]
        rwp=(decision.get("indicators") or {}).get("rwp") or {}
        details=rwp.get("details") or {}
        state.transition_rwp_notice=(f"Final Rwp: {details['before']:.3g}% → {details['after']:.3g}% "
            f"({details['delta_pp']:+.3g} percentage points). Whole-pattern evidence; no region box."
            if rwp.get("selected") and details.get("available") else rwp.get("reason","") if rwp.get("selected") else "")
        patterns = []
        labels = []
        map_ids = []
        for r in window:
            try:
                item = self._transition_engine.load(self._transition_scans[r["run_id"]])
            except (OSError, ValueError, TypeError):
                continue
            patterns.append(item.pattern)
            map_ids.append(r["run_id"])
            conditions = _conditions(self._transition_scans[r["run_id"]], settings)
            channel = conditions.get(state.transition_annotation, {})
            row_scan = self._transition_scans[r["run_id"]]
            label = str(row_scan.get("run_number") or r["run_id"][-22:])
            if channel.get("status") == "recorded" and isinstance(
                channel.get("value"), (float, int)
            ):
                label += f" · {channel['value']:.4g} {channel.get('unit', '')}"
            labels.append(label)
        self._transition_map_ids = map_ids
        group = state.transition_series
        first = None
        color_key = group + (":local" if state.transition_background else ":observed")
        if patterns:
            # Freeze to the first eligible acquisition in the series, not the
            # currently selected/displayed subset or the latest bright peak.
            eligible = next((r for r in rows if r.get("state_id")), rows[0])
            first = self._transition_engine.load(
                self._transition_scans[eligible["run_id"]]
            ).pattern
            if (
                color_key not in self._transition_saved["limits"]
                or self._transition_saved["limits"][color_key].get("source")
                == "no valid reference bins"
            ):
                candidates = []
                for candidate in window:
                    if not candidate.get("state_id"):
                        continue
                    item = self._transition_engine.load(
                        self._transition_scans[candidate["run_id"]]
                    )
                    candidates.append(
                        locked_limits(
                            item.pattern,
                            background=self._transition_engine.prepare(item).baseline
                            if state.transition_background
                            else None,
                        )
                    )
                if not candidates and first.valid.any():
                    first_item = self._transition_engine.load(
                        self._transition_scans[eligible["run_id"]]
                    )
                    candidates = [
                        locked_limits(
                            first,
                            background=self._transition_engine.prepare(
                                first_item
                            ).baseline
                            if state.transition_background
                            else None,
                        )
                    ]
                self._transition_saved["limits"][color_key] = (
                    {
                        "low": min(v["low"] for v in candidates),
                        "high": max(v["high"] for v in candidates),
                        "soft": candidates[0]["soft"],
                        "source": "initial received window",
                    }
                    if candidates
                    else {
                        "low": 0,
                        "high": 1,
                        "soft": 0.01,
                        "source": "no valid reference bins",
                    }
                )
        try:
            current = self._transition_engine.load(scan)
            position = rows.index(selected)
            comparison_ids = {
                "previous": rows[position - 1]["run_id"] if position else None,
                "state": selected.get("representative"),
                "reference": self._transition_saved["references"].get(group),
            }
            if comparison_ids["reference"] not in ids:
                comparison_ids["reference"] = next(
                    (r["run_id"] for r in rows if r.get("state_id")), None
                )
                if comparison_ids["reference"]:
                    self._transition_saved["references"][group] = comparison_ids[
                        "reference"
                    ]
            # A scan is not evidence of agreement with an earlier observation
            # when the only available representative/reference is itself.
            comparison_ids = {
                k: v if v != selected["run_id"] else None
                for k, v in comparison_ids.items()
            }
            titles = {
                "previous": "Previous scan",
                "state": "First matching pattern",
                "reference": "Pinned reference",
            }
            state.transition_comparison_options = [
                {
                    "title": title
                    + " · "
                    + scan_label(self._transition_scans[comparison_ids[k]]),
                    "value": k,
                }
                for k, title in titles.items()
                if comparison_ids.get(k)
            ]
            available = [o["value"] for o in state.transition_comparison_options]
            if state.transition_comparison not in available:
                state.transition_comparison = available[0] if available else "previous"
            compare_id = comparison_ids.get(state.transition_comparison)
            before = (
                self._transition_engine.load(self._transition_scans[compare_id])
                if compare_id
                else None
            )
            pair = (
                self._transition_engine.compare_evidence(
                    self._transition_engine.prepare(before),
                    self._transition_engine.prepare(current),
                )
                if before
                else None
            )
            self._transition_regions = {}
            for i, region in enumerate((pair or {}).get("regions", [])):
                lo, hi = region["interval"]
                pad = max(
                    (hi - lo) * 1.5,
                    float(current.pattern.x[-1] - current.pattern.x[0]) * 0.0005,
                )
                self._transition_regions[str(i)] = (
                    max(current.pattern.x[0], lo - pad),
                    min(current.pattern.x[-1], hi + pad),
                )
            state.transition_region_options = [
                {"title": "Full pattern", "value": "full"}
            ] + [
                {"title": f"Changed region {i + 1} · {v[0]:.5g}–{v[1]:.5g}", "value": k}
                for i, (k, v) in enumerate(self._transition_regions.items())
            ]
            if state.transition_region not in self._transition_regions:
                state.transition_region = "full"
            region = self._transition_regions.get(state.transition_region)
            fit = None
            if state.transition_mode == "fit":
                try:
                    fit = self._scan_interpreter.pattern(scan)
                except (OSError, ValueError, TypeError):
                    pass
                model_hash = (scan.get("observables") or {}).get("source_gpx_sha256")
                if (
                    fit is not None
                    and model_hash
                    and fit.packet.get("source_gpx_sha256") != model_hash
                ):
                    fit = None
            state.transition_profile = plotly.Figure.to_data(
                profile_figure(
                    current.pattern,
                    comparison=before.pattern if before else None,
                    comparison_label=scan_label(self._transition_scans[compare_id])
                    if compare_id
                    else "Earlier pattern",
                    pair=pair,
                    region=region,
                    fit=fit,
                    mode=state.transition_mode,
                )
            )
            if state.transition_mode == "fit" and fit is not None:
                canonical = self._canonical_transition_fit(scan, fit)
                if canonical is not None:
                    if region:
                        canonical.update_xaxes(range=list(region))
                    state.transition_profile = plotly.Figure.to_data(canonical)
            state.transition_pattern_evidence = self._transition_evidence(
                selected, pair, compare_id
            )
            if compare_id:
                state.transition_pattern_evidence.insert(0,
                    "Curve overlay compared with " + scan_label(self._transition_scans[compare_id]) + ".")
            state.transition_source = (
                current.provenance.get("source", "")
                + " · "
                + COORDINATES[current.pattern.scope.coordinate]
            )
            state.transition_notices = (
                current.notices
                + self._transition_report.get("qualifications", [])
                + (pair or {}).get("qualifications", [])
            )
            peers = [
                scan_label(self._transition_scans[r["run_id"]])
                for r in rows
                if r.get("state_id")
                and r.get("state_id") == selected.get("state_id")
                and r is not selected
            ]
            state.transition_peer_labels = (
                "Matching patterns: "
                + ", ".join(peers[:12])
                + (f" and {len(peers) - 12} more" if len(peers) > 12 else "")
                if peers
                else "No earlier matching pattern assigned."
            )
            reference = self._transition_saved["references"].get(group)
            state.transition_reference_label = (
                "Pinned reference: " + scan_label(self._transition_scans[reference])
                if reference
                else "Reference unavailable"
            )
            state.transition_can_pin = bool(
                current.pattern.valid.mean() >= 0.75 and reference != selected["run_id"]
            )
            display_patterns,display_backgrounds,display_first = self._transition_map_display_data(
                patterns,map_ids,first,eligible["run_id"] if first is not None else None)
            cursor=map_ids.index(state.transition_selected) if state.transition_selected in map_ids else max(0,len(map_ids)-1)
            state.transition_cursor=cursor
            state.transition_cursor_max=max(0,len(map_ids)-1)
            state.transition_cursor_label=labels[cursor] if labels else "Waiting for scans"
            display_range=self._transition_map_ranges.get((group,state.transition_map_coordinate))
            state.transition_region_signal_options=[{"title":SIGNALS[k],"value":k} for k in LOCAL_SIGNALS]
            overlay_data=self._transition_overlay_data(selected,rows,current.pattern)
            change_regions=overlay_data["regions"]
            comparison_run=overlay_data["comparison_run"]
            state.transition_region_overlay_notice=overlay_data["notice"]
            state.transition_region_evidence_rows=overlay_data["evidence"]
            state.transition_region_overlay_can_zoom=bool(change_regions)
            self._transition_overlay_range=None
            if change_regions:
                low=min(r["interval"][0] for r in change_regions)
                high=max(r["interval"][1] for r in change_regions)
                padding=(high-low)*.08
                self._transition_overlay_range=[max(float(display_patterns[cursor].x[0]),low-padding),
                    min(float(display_patterns[cursor].x[-1]),high+padding)]
            limits=self._transition_saved["limits"][color_key]
            map_key=self._transition_cursor_key(map_ids,labels,limits)
            frames=state.transition_cursor_frames
            if map_key==self._transition_cursor_frames_key and cursor<len(frames) and frames[cursor]:
                state.transition_map=selection_figure(state.transition_map,frames[cursor],display_range)
                stats=self._transition_map_stats
            else:
                mapfig, stats = heatmap_figure(
                    display_patterns,
                    labels=labels,
                    run_ids=map_ids,
                    limits=self._transition_saved["limits"][color_key],
                    scale=state.transition_scale,
                    contrast=state.transition_contrast,
                    selected=state.transition_selected,
                    region=display_range,
                    grid_reference=display_first,
                    boundaries=window,
                    backgrounds=display_backgrounds,
                    palette=state.transition_palette,
                    ribbon=display_patterns[cursor] if display_patterns else None,
                    ribbon_background=display_backgrounds[cursor] if display_backgrounds else None,
                    ribbon_label=state.transition_cursor_label,
                    change_regions=change_regions,
                    overlay=state.transition_region_overlay,
                    comparison_run=comparison_run,
                    view_revision=self._transition_map_reset_revisions.get((group,state.transition_map_coordinate),0),
                )
                state.transition_map = plotly.Figure.to_data(mapfig)
                self._transition_map_stats=stats
            self._cache_transition_cursor(display_patterns,display_backgrounds,map_ids,labels,rows,
                self._transition_saved["limits"][color_key])
            state.transition_cursor_scope=digest([self._transition_generation,group])
            state.transition_cursor_run_ids=list(map_ids)
            if display_patterns:
                layout=state.transition_map["layout"]
                domain=layout["yaxis2"]["domain"]
                inner=layout["height"]-layout["margin"]["t"]-layout["margin"]["b"]
                map_height=inner*(domain[1]-domain[0])
                row_height=map_height/len(map_ids)
                state.transition_cursor_top=layout["margin"]["t"]+(1-domain[1])*inner+row_height/2-9
                state.transition_cursor_height=max(18,map_height-row_height+18)
            state.transition_map_notice = "Acquisition order · shared locked scale · peak-preserving display bins · masked gaps retained."
            if state.transition_background:
                state.transition_map_notice += (
                    " Estimated local background removed for display only."
                )
            if stats.get("saturated_cells"):
                state.transition_map_notice += f" {stats['saturated_cells']} display cells exceed the upper color limit."
            if stats.get("log_masked_cells"):
                state.transition_map_notice += (
                    f" Log view masks {stats['log_masked_cells']} non-positive cells."
                )
            if stats.get("reference_range_exceeded"):
                state.transition_map_notice += (
                    " Some scans extend beyond the reference grid shown."
                )
        except (OSError, ValueError, TypeError) as exc:
            self._transition_unavailable_plot(exc)
            return
        state.transition_error = ""
        self._transition_refinement(scan)
        if state.transition_dialog:
            self._update_transition_dialog()

    def _transition_calibration(self,scan):
        return mapped_calibration(scan,getattr(self,"_powgen_controller",None),getattr(self,"records",{}))

    def _transition_overlay_data(self,selected,rows,pattern):
        state=self.server.state
        result={"regions":[],"comparison_run":None,"notice":"","evidence":[]}
        if state.transition_region_overlay=="off":
            return result
        target=state.transition_map_coordinate
        scan=self._transition_scans[selected["run_id"]]
        calibration=(self._transition_calibration(scan)
            if target!=pattern.scope.coordinate and "tof_us" in {target,pattern.scope.coordinate} else None)
        decision=self._transition_region_decision(selected,rows)
        other=decision.get("before")
        if other==selected["run_id"]:
            other=decision.get("after")
        comparison=(self._transition_engine.load(self._transition_scans[other]).pattern
            if other in self._transition_scans else None)
        regions,stats=localized_regions(decision,state.transition_region_signals,pattern,target,calibration,
            comparison_pattern=comparison,include_split_features=state.transition_show_split_features)
        pair_label=" → ".join(scan_label(self._transition_scans[k])
            for k in [decision.get("before"),decision.get("after")] if k in self._transition_scans)
        recognized=sum(r.get("status")=="recognized_split" for r in regions)
        result.update(regions=regions,comparison_run=other,
            notice=(f"{stats['shown']} of {stats['total']} localized regions · comparison "
                f"{pair_label or 'awaiting a second scan'}. Boxes span both compared scans. "
                f"{recognized} dotted split-feature box(es) are recognized features below the change rule; "
                "solid boxes passed a local rule. Masked gaps are excluded. Rwp gets no region box."))
        for region in regions:
            for diagnostic in region.get("diagnostics",[]):
                profiles=[diagnostic["before_profile"],diagnostic["after_profile"]]
                separations=[]
                for profile in profiles:
                    positions=coordinate_values(profile["positions"],pattern.scope.coordinate,target,calibration)
                    separations.append(f"{abs(positions[1]-positions[0]):.5g}"
                        if profile["two_component_qualified"] else "Single component")
                result["evidence"].append({
                    "region":f"{region['interval'][0]:.5g}–{region['interval'][1]:.5g} {COORDINATES[target]}",
                    "status":"Local change" if diagnostic.get("passes_rule") else "Recognized split; below change rule",
                    "before":separations[0],"after":separations[1],
                    "effect":f"{diagnostic['separation_change_fwhm']:.3g}" if "separation_change_fwhm" in diagnostic else "Component count changed",
                    "error":f"{diagnostic.get('error_ratio',0):.3g}"})
        return result

    def _transition_cursor_key(self,ids,labels,limits):
        state=self.server.state
        return digest([self._transition_generation,self._transition_signature,ids,labels,limits,
            state.transition_scale,state.transition_contrast,state.transition_background,
            state.transition_map_coordinate,state.transition_coordinate_notice,state.transition_palette,
            state.transition_policy,state.transition_region_overlay,state.transition_region_signals,
            state.transition_region_reference,state.transition_show_split_features,
            self._transition_map_ranges.get((state.transition_series,state.transition_map_coordinate)),
            self._transition_map_reset_revisions.get((state.transition_series,state.transition_map_coordinate),0)])

    def _cache_transition_cursor(self,patterns,backgrounds,ids,labels,rows,limits):
        state=self.server.state
        key=self._transition_cursor_key(ids,labels,limits)
        cursor=ids.index(state.transition_selected) if state.transition_selected in ids else len(ids)-1
        cached=state.transition_cursor_frames
        if key==self._transition_cursor_frames_key and 0<=cursor<len(cached) and cached[cursor]:
            return
        by_id={row["run_id"]:row for row in rows}
        def regions(run_id):
            pattern=self._transition_engine.load(self._transition_scans[run_id]).pattern
            return self._transition_overlay_data(by_id[run_id],rows,pattern)
        state.transition_cursor_frames=cursor_frames(patterns,backgrounds,ids,labels,
            selected=state.transition_selected,limits=limits,scale=state.transition_scale,
            contrast=state.transition_contrast,overlay=state.transition_region_overlay,regions=regions,
            base_shapes=[s for s in state.transition_map["layout"].get("shapes",[]) if s["type"]=="line"])
        state.transition_cursor_epoch=key
        self._transition_cursor_frames_key=key

    def _transition_region_decision(self,selected,rows):
        mode=self.server.state.transition_region_reference
        key={"previous":"previous_comparison","boundary":"boundary_comparison","first":"anchor_comparison"}[mode]
        result=selected.get(key)
        if not result and rows and selected["run_id"]==rows[0]["run_id"] and len(rows)>1:
            result=rows[1].get("previous_comparison")
        return result or {}

    def _transition_map_display_data(self,patterns,ids,first,first_id):
        """Convert display copies only; native detector arrays and evidence stay unchanged."""
        state=self.server.state
        if not patterns:
            return [],None,None
        native=patterns[0].scope.coordinate
        calibrations={};errors=[]
        for run_id in dict.fromkeys([*ids,first_id]):
            if run_id is None:
                continue
            try:
                calibration=self._transition_calibration(self._transition_scans[run_id])
                item=first if run_id==first_id else patterns[ids.index(run_id)]
                # Validate the entire visible domain, not just its center.
                display_pattern(item,"d_angstrom" if native=="tof_us" else "tof_us",calibration)
                calibrations[run_id]=calibration
            except (OSError,ValueError,TypeError,KeyError,UnicodeError) as exc:
                calibrations[run_id]=None
                errors.append(str(exc))
        available=(["tof_us","d_angstrom","q_inv_angstrom"] if not errors else
            ["d_angstrom","q_inv_angstrom"] if native!="tof_us" else [native])
        state.transition_coordinate_options=[{"title":COORDINATES[k],"value":k} for k in available]
        if state.transition_map_coordinate not in available:
            state.transition_map_coordinate=native
        target=state.transition_map_coordinate
        if not errors:
            labels=sorted({f"{c.label} · profile Bank {c.profile_bank} → physical bank {c.source_bank} · {c.profile_sha256[:10]}"
                for c in calibrations.values() if c is not None})
            state.transition_coordinate_notice="Calibration: "+"; ".join(labels[:3])+". Intensity keeps its original measurement scale."
        else:
            state.transition_coordinate_notice=("Calibrated TOF conversion unavailable: "+errors[0]
                +". Available axes retain their measured coordinate calibration; no nominal geometry is substituted.")
        converted=[display_pattern(p,target,calibrations[k]) for p,k in zip(patterns,ids)]
        backgrounds=None
        if state.transition_background:
            backgrounds=[]
            for p,k in zip(patterns,ids):
                baseline=self._transition_engine.prepare(self._transition_engine.load(self._transition_scans[k])).baseline
                backgrounds.append(display_pattern(replace(p,intensity=baseline),target,calibrations[k]).intensity)
        reference=display_pattern(first,target,calibrations[first_id]) if first is not None else None
        return converted,backgrounds,reference

    def _canonical_transition_fit(self, scan, fit):
        """Reuse the full-result fit renderer only for matching measured/model arrays."""
        controller = getattr(self,"_powgen_controller",None)
        record = cached_record(scan,controller,getattr(self,"records",{}))
        root = getattr(record,"local_output_dir",None)
        if root is None or not Path(root).is_dir():
            # Recovered watches initially hold a minimal polling record. The
            # application may already have collected this exact job's outputs.
            records = getattr(self,"records",{})
            cached = records.get(getattr(record,"uid",None))
            job_id = getattr(record,"galaxy_job_id",None) or getattr(record,"uid",None)
            if cached is None and job_id:
                cached = next((r for r in records.values() if getattr(r,"galaxy_job_id",None) == job_id),None)
            root = getattr(cached,"local_output_dir",None)
        if root is None or not Path(root).is_dir():
            return None
        for descriptor in discover_plot_payloads(root):
            if "gsas" not in str(descriptor.get("kind","")).lower():
                continue
            payload = read_plot_payload(descriptor["path"])
            arrays = payload.get("arrays",payload)
            x = np.asarray(arrays.get("x",[]),dtype=float)
            observed = np.asarray(arrays.get("yobs",[]),dtype=float)
            calculated = np.asarray(arrays.get("ycalc",[]),dtype=float)
            if (x.shape == fit.x.shape and observed.shape == fit.y.shape and calculated.shape == fit.calculated.shape
                    and np.allclose(x,fit.x,equal_nan=True) and np.allclose(observed,fit.y,equal_nan=True)
                    and np.allclose(calculated[fit.fitted],fit.calculated[fit.fitted],equal_nan=True)):
                return figure_for_payload(payload)
        return None

    def _render_boundary_map(self, rows, conditions):
        state = self.server.state
        state.transition_diagram_axes = [{"title":"Run number","value":"run_number"}] + [
            {"title":key.replace("_"," ").title()+f" ({value.get('unit','')})", "value":key}
            for key,value in conditions.items() if isinstance(value.get("value"),(int,float))]
        if state.transition_diagram_axis not in [v["value"] for v in state.transition_diagram_axes]:
            state.transition_diagram_axis = "run_number"
        axis = state.transition_diagram_axis
        values = []
        settings = InterpretationSettings.model_validate(state.scan_interpretation)
        for row in rows:
            scan = self._transition_scans[row["run_id"]]
            value = scan.get("run_number") if axis == "run_number" else _conditions(scan,settings).get(axis,{})
            if isinstance(value,dict):
                value = value.get("value") if value.get("status") == "recorded" else None
            values.append(number(value))
        title = next(v["title"] for v in state.transition_diagram_axes if v["value"]==axis)
        state.transition_boundary_map = plotly.Figure.to_data(boundary_figure(rows,values,title))

    def update_transition_diagram_axis(self, value, **_):
        if value in [v["value"] for v in self.server.state.transition_diagram_axes]:
            self.server.state.transition_diagram_axis = value
            self._render_transition_tracking()

    def select_transition_diagram(self, event, **_):
        for point in (event or {}).get("points",[]):
            value = point.get("customdata")
            if isinstance(value,list) and value:
                self.select_transition_scan(value[0])
                break

    def update_transition_policy(self, field, value, **_):
        state = self.server.state
        previous = asdict(BoundaryPolicy.restore(self._transition_saved.get("policy")))
        payload = dict(state.transition_policy)
        try:
            if field == "selected":
                payload[field] = value
                payload["minimum_signals"] = min(int(payload.get("minimum_signals",1)),max(1,len(value)))
            elif field in {"minimum_signals","minimum_features"}:
                payload[field] = int(value)
            elif field == "width_percent":
                payload["width_fraction"] = float(value)/100
            elif field == "error_ratio":
                payload[field] = float(value)
            elif field == "rwp_delta_pp":
                payload[field]=float(value)
            elif field == "split_separation_fwhm":
                payload[field]=float(value)
            elif field == "rwp_relative_percent":
                payload["rwp_relative"]=float(value)/100
            elif field in {"comparison","fixed_broadening","rwp_direction"}:
                payload[field] = value
            else:
                return
            policy = BoundaryPolicy(**payload)
            state.transition_policy = asdict(policy)
            if field=="selected" and not _.get("client_selection"):
                state.transition_boundary_signals=list(policy.selected)
                state.dirty("transition_boundary_signals")
            state.transition_width_percent = policy.width_fraction*100
            state.transition_rwp_relative_percent=policy.rwp_relative*100
            state.transition_vote_options = list(range(1,max(1,len(policy.selected))+1))
            self._transition_saved["policy"] = asdict(policy)
            self._transition_signature = None
            self._refresh_transition_tracking(list(self._transition_scans.values()))
            self._checkpoint_transition_tracking()
        except (TypeError,ValueError) as exc:
            state.transition_policy = previous
            if field=="selected":
                state.transition_boundary_signals=list(previous["selected"])
                state.dirty("transition_boundary_signals")
            state.transition_width_percent = previous["width_fraction"]*100
            state.transition_rwp_relative_percent=previous["rwp_relative"]*100
            state.transition_error = str(exc)
        state.flush()

    def select_transition_boundary_signals(self,value,**_):
        # The browser keeps its optimistic multi-select value. Echoing each
        # older acknowledgement can erase clicks made while a replay runs.
        self.update_transition_policy("selected",value,client_selection=True)

    def select_transition_region_signals(self,value,**_):
        self.update_transition_display("region_signals",value,client_selection=True)

    def _transition_evidence(self, record, pair, compare_id):
        if pair and "indicators" in pair:
            prefix = f"{pair['votes']} of {len(pair['selected'])} selected indicators triggered; {pair['required']} required."
            if pair["status"] == "no_resolved_change":
                prefix = "No resolved change meets the selected boundary rule. " + prefix
            return [prefix] + [v["title"]+(f": {v['support']} supporting peak region(s)." if v.get("localizable",True)
                else f": {v['details']['before']:.3g}% → {v['details']['after']:.3g}%; whole-pattern evidence without a peak region.")
                for v in pair["indicators"].values() if v["detected"]] + (
                ["Required measured indicators are unavailable; no boundary assignment."] if pair["status"]=="uncertain" else [])
        reference_notes = []
        if record.get("event") == "reference":
            reference_notes.append(
                "Two compatible acquisitions established this comparison reference. Its relation to earlier patterns remains unresolved; this is not a transition alert."
            )
        elif record.get("state_id") and record.get("relation_to_other_references") == "unresolved":
            reference_notes.append(
                "This scan matches its pattern group, but that group's relation to some earlier references remains unresolved."
            )
        if record.get("reason"):
            return reference_notes + [REASONS.get(record["reason"], record["reason"])]
        if pair is None:
            return reference_notes + [
                "First eligible measured pattern; waiting for a compatible comparison."
            ]
        if pair["status"] == "uncertain":
            return reference_notes + [
                "Comparison uncertain: " + "; ".join(pair["reasons"]).replace("_", " ")
            ]
        if pair["status"] == "no_resolved_change":
            return reference_notes + [
                "No resolved change from the selected comparator after independent local peak movement."
            ]
        return reference_notes + [
            FINDINGS.get(code, code.replace("_", " ").capitalize())
            for code in pair.get("reasons", [])
        ]

    def _transition_refinement(self, scan):
        state = self.server.state
        model_rows = []
        phases = (scan.get("observables") or {}).get("phases", []) or scan.get(
            "phases", []
        )
        names = phase_labels(scan)
        model_hash = (scan.get("observables") or {}).get("source_gpx_sha256")
        curve_hash = (scan.get("pattern_observation") or scan.get("pattern") or {}).get(
            "source_gpx_sha256"
        )
        mismatched = bool(model_hash and curve_hash and model_hash != curve_hash)
        if mismatched:
            phases = []
            state.transition_model_notice = "Measured curve and refinement refer to different projects; model context is withheld."
        for p in phases:
            fresh = p.get("covariance_fresh") is not False
            states = p.get("metric_states") or {}
            cells = [
                f"{k}={p[k]:.6g}"
                + (
                    f" ± {p[k + '_esd']:.2g}"
                    if fresh and number(p.get(k + "_esd")) is not None
                    else ""
                )
                + " Å"
                + (
                    " [stale]"
                    if not fresh
                    else f" [{states.get(k, 'unverified')}]"
                    if states.get(k) not in {"refined", "constraint-derived"}
                    else ""
                )
                for k in ("a", "b", "c")
                if isinstance(p.get(k), (float, int)) and number(p[k]) is not None
            ]
            fraction = number(p.get("weight_percent"))
            fraction_state = p.get("fraction_state", "unverified")
            phase = str(
                p.get("phase")
                or p.get("label")
                or p.get("phase_id")
                or "Unnamed candidate"
            )
            if p.get("phase_id"):
                phase = names.get(str(p["phase_id"]), phase)
            model_rows.append(
                {
                    "phase": phase,
                    "fraction": f"{fraction:.3g}%"
                    + (
                        f" [{fraction_state.replace('_', ' ')}]"
                        if fraction_state not in {"refined", "constraint-derived"}
                        else ""
                    )
                    if fraction is not None and 0 <= fraction <= 100
                    else "—",
                    "cell": ", ".join(cells) or "Not reported",
                }
            )
        # Publish the complete replacement once. Mutating a state list that is
        # already clean can leave a connected client displaying the pending table.
        state.transition_model_rows = model_rows
        value = None if mismatched else number(scan.get("rwp"))
        state.transition_rwp = (
            f"{value:.3g}%" if value is not None and value >= 0 else "Not reported"
        )
        if not mismatched:
            state.transition_model_notice = (
                "Modeled crystalline fractions exclude omitted or amorphous material. A candidate switch alone does not change the pattern symbol."
                if phases
                else "Refinement pending or unavailable; observed-pattern tracking is independent of fit completion."
            )
        conditions = _conditions(
            scan, InterpretationSettings.model_validate(state.scan_interpretation)
        )
        state.transition_conditions = [
            condition_display(v, k.replace('_', ' ')) + f" · {v.get('status', 'unavailable')}"
            for k, v in conditions.items()
        ]
        state.transition_condition_summary = condition_display(conditions.get("temperature"), "Sample temperature during this acquisition")
        controller = getattr(self, "_powgen_controller", None)
        state.transition_can_open = bool(
            controller is not None
            and str(scan.get("run_id")) in getattr(controller, "records", {})
            and str(scan.get("run_id")) in getattr(controller.state, "completed", {})
        )
        self._transition_condition_options(scan)

    def _transition_condition_options(self, scan):
        """Expose only complete readbacks that can support an explicit mapping."""
        state = self.server.state
        channels = ((scan.get("metadata") or {}).get("conditions") or {}).get(
            "channels", []
        )
        allowed = {"temperature", "magnetic_field", "voltage", "pressure", "gas_flow"}
        self._transition_binding_candidates = {
            digest([condition_scope(scan), c.get("quantity"), c.get("channel")]): c
            for c in channels
            if c.get("quantity") in allowed
            and c.get("channel")
            and c.get("role") == "measured"
            and c.get("status") == "needs_mapping"
            and c.get("statistic") == "time_weighted_step_hold"
            and (number(c.get("coverage_fraction")) or 0) >= 0.95
        }
        state.transition_binding_options = [
            {
                "title": f"{c['quantity'].replace('_', ' ').title()} · {c['channel']}",
                "value": key,
            }
            for key, c in self._transition_binding_candidates.items()
        ]
        scope = (condition_scope(scan), str(scan.get("run_id")))
        if scope != getattr(self, "_transition_binding_scope", None):
            state.transition_binding = ""
            state.transition_binding_confirmed = False
            state.transition_binding_notice = ""
            self._transition_binding_scope = scope

    def bind_transition_condition(self, **_):
        state = self.server.state
        scan = self._transition_scans.get(state.transition_selected, {})
        metadata = scan.get("metadata") or {}
        try:
            if not state.transition_binding_confirmed:
                raise ValueError(
                    "Verify the specimen/device connection before applying a mapping."
                )
            if (
                not metadata.get("sample_id")
                or not scan.get("ipts")
                or not scan.get("instrument")
            ):
                raise ValueError("A known instrument, IPTS and specimen are required.")
            channel = self._transition_binding_candidates.get(state.transition_binding)
            if channel is None:
                raise ValueError("Choose a complete measured readback for this scan.")
            payload = InterpretationSettings.model_validate(
                state.scan_interpretation
            ).model_dump(mode="json")
            scope = condition_scope(scan)
            payload["condition_bindings"][scope] = {
                **payload["condition_bindings"].get(scope, {}),
                channel["quantity"]: channel["channel"],
            }
            state.scan_interpretation = InterpretationSettings.model_validate(
                payload
            ).model_dump(mode="json")
            controller = getattr(self, "_powgen_controller", None)
            if controller is not None and str(
                getattr(self, "_post_analysis_scope", "")
            ).startswith("watch:"):
                controller.state.scan_interpretation = dict(state.scan_interpretation)
                self._checkpoint_transition_tracking()
            self._render_transition_tracking()
            state.transition_binding_notice = "Applied to this instrument, IPTS and specimen. Raw metadata is unchanged."
        except (ValueError, TypeError) as exc:
            state.transition_binding_notice = str(exc)
        state.flush()

    def open_transition_scan(self, **_):
        if self.server.state.transition_can_open:
            # The transition history includes scans hidden by result display
            # filters. Open the recorded run directly without applying filters.
            self.server.state.powgen_selected_run_id = (
                self.server.state.transition_selected
            )
            self.open_powgen_selected_run()

    def download_transition_report(self, **_):
        """Export the evidence index, without credential-bearing run objects."""
        payload = {
            **self._transition_report,
            "presentation": checkpoint(self._transition_saved),
            "selected_run_id": self.server.state.transition_selected,
            "contents": "Pattern compatibility evidence index. Native curves remain in their recorded source products; this file does not contain curve arrays.",
        }
        self.download_file(
            "radar-pd-transition-tracking.json",
            "application/json",
            (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode(),
        )

    def select_transition_series(self, value, **_):
        self._cancel_transition_cursor()
        if value not in [
            o["value"] for o in self.server.state.transition_series_options
        ]:
            return
        self.server.state.transition_series = value
        self.server.state.transition_follow = True
        self.server.state.transition_region = "full"
        self._render_transition_tracking()

    def select_transition_scan(self, value, **_):
        ids = [r["run_id"] for r in self._series_rows()]
        if value not in ids:
            return
        if not self._transition_cursor_applying:
            self._cancel_transition_cursor()
        self.server.state.transition_selected = value
        self.server.state.transition_follow = value == ids[-1]
        self.server.state.transition_region = "full"
        self._render_transition_tracking()

    def select_transition_heatmap(self, payload, **_):
        if (
            self.server.state.transition_dialog
            and self.server.state.transition_dialog_kind != "map"
        ):
            return
        if isinstance(payload, (str, int, float)):
            y = payload
        else:
            points = payload.get("points", []) if isinstance(payload, dict) else []
            if not points:
                return
            y = points[0].get("y")
        if isinstance(y, str) and y in self._transition_map_ids:
            self.select_transition_scan(y)
        elif isinstance(y, (int, float)) and 0 <= int(y) < len(
            self._transition_map_ids
        ):
            self.select_transition_scan(self._transition_map_ids[int(y)])

    def follow_transition_latest(self, **_):
        state = self.server.state
        state.transition_follow = not state.transition_follow
        self._render_transition_tracking()

    def update_transition_display(self, field, value, **_):
        options = {
            "scale": {"asinh", "linear", "log"},
            "palette": {"Turbo","Cividis"},
            "mode": {"observed", "fit"},
            "comparison": {"previous", "state", "reference"},
            "window": {30, 60, 100},
        }
        if field in options and value not in options[field]:
            return
        if field=="overlay":
            if value not in {"off","ribbon","both"}:
                return
            self.server.state.transition_region_overlay=value
            self._render_transition_tracking()
            return
        if field=="region_signals":
            if not isinstance(value,list) or any(v not in LOCAL_SIGNALS for v in value):
                return
            self.server.state.transition_region_signals=list(dict.fromkeys(value))
            if not _.get("client_selection"):
                self.server.state.transition_region_highlights=list(dict.fromkeys(value))
                self.server.state.dirty("transition_region_highlights")
            self._render_transition_tracking()
            return
        if field=="region_reference":
            if value not in {"previous","boundary","first"}:
                return
            self.server.state.transition_region_reference=value
            self._render_transition_tracking()
            return
        if field=="split_features":
            if not isinstance(value,bool):
                return
            self.server.state.transition_show_split_features=value
            self._render_transition_tracking()
            return
        if field=="coordinate":
            state=self.server.state
            if value not in {o["value"] for o in state.transition_coordinate_options}:
                return
            group=state.transition_series
            old=(state.transition_map.get("layout",{}).get("meta") or {}).get("coordinate",state.transition_map_coordinate)
            limits=self._transition_map_ranges.get((group,old))
            if limits and value!=old:
                try:
                    scan=self._transition_scans[state.transition_selected]
                    calibration=self._transition_calibration(scan) if "tof_us" in {old,value} else None
                    self._transition_map_ranges[(group,value)]=sorted(coordinate_values(limits,old,value,calibration).tolist())
                except (OSError,ValueError,KeyError):
                    self._transition_map_ranges.pop((group,value),None)
            state.transition_map_coordinate=value
            self._render_transition_tracking()
            return
        if field == "contrast":
            try:
                value = max(0, min(100, float(value)))
            except (ValueError, TypeError):
                return
        if field == "region" and value not in {"full", *self._transition_regions}:
            return
        if field == "annotation" and value not in [
            v["value"] for v in self.server.state.transition_annotation_options
        ]:
            return
        if field == "background" and not isinstance(value, bool):
            return
        if field not in {*options, "contrast", "region", "annotation", "background"}:
            return
        setattr(self.server.state, "transition_" + field, value)
        self._render_transition_tracking()

    def relayout_transition_map(self,event,**_):
        """Plotly links both panels immediately; retain that range across scan arrivals."""
        if not isinstance(event,dict):
            return
        state=self.server.state
        if state.transition_dialog and state.transition_dialog_kind!="map":
            return
        key=(state.transition_series,state.transition_map_coordinate)
        for axis in ("xaxis2","xaxis"):
            if event.get(axis+".autorange") is True:
                self.reset_transition_map_zoom()
                return
            limits=event.get(axis+".range")
            if axis+".range[0]" in event and axis+".range[1]" in event:
                limits=[event[axis+".range[0]"],event[axis+".range[1]"]]
            if isinstance(limits,(list,tuple)) and len(limits)==2:
                values=[number(v) for v in limits]
                if all(v is not None for v in values) and values[0]!=values[1]:
                    values=sorted(values)
                    if self._transition_map_ranges.get(key)!=values:
                        self._transition_map_ranges[key]=values
                        # Refresh display bins in the visible domain so a
                        # native doublet is not hidden in a full-range bin.
                        self._render_transition_tracking()
                        state.flush()
                    return

    def reset_transition_map_zoom(self,**_):
        """Restore full data and both linked axes, overriding Plotly's old view."""
        state=self.server.state
        if state.transition_dialog and state.transition_dialog_kind!="map":
            return
        full=(state.transition_map.get("layout",{}).get("meta") or {}).get("full_range")
        if not isinstance(full,(list,tuple)) or len(full)!=2:
            return
        key=(state.transition_series,state.transition_map_coordinate)
        self._transition_cursor_dragging=False
        self._transition_map_ranges[key]=list(full)
        self._transition_map_reset_revisions[key]=self._transition_map_reset_revisions.get(key,0)+1
        self._render_transition_tracking()
        state.flush()

    def zoom_transition_regions(self,**_):
        if self.server.state.transition_region_overlay_can_zoom and getattr(self,"_transition_overlay_range",None):
            state=self.server.state
            self._transition_map_ranges[(state.transition_series,state.transition_map_coordinate)]=list(self._transition_overlay_range)
            self._render_transition_tracking()

    def slide_transition_map(self,value,**_):
        try:
            index=int(value)
        except (ValueError,TypeError):
            return
        if not 0<=index<len(self._transition_map_ids):
            return
        self._queue_transition_cursor(self._transition_map_ids[index])

    def _acknowledge_transition_cursor(self,client_id,revision,accepted):
        self.server.state.transition_cursor_ack={
            "client_id":client_id,"revision":revision,"accepted":bool(accepted),
            "run_id":self.server.state.transition_selected,
            "scope":self.server.state.transition_cursor_scope}

    def _cancel_transition_cursor(self):
        if self._transition_cursor_client is not None:
            self._acknowledge_transition_cursor(self._transition_cursor_client,self._transition_cursor_revision,False)
        self._transition_cursor_request=None
        self._transition_cursor_dragging=False
        self._transition_cursor_client=None
        self._transition_cursor_revision=0

    @staticmethod
    def _valid_cursor_revision(client_id,revision):
        return (isinstance(client_id,str) and 0<len(client_id)<=128
            and isinstance(revision,int) and not isinstance(revision,bool) and 0<=revision<2**53)

    def begin_transition_cursor(self,revision=None,scope=None,client_id=None,**_):
        if client_id is not None:
            if not self._valid_cursor_revision(client_id,revision):
                return
            expected=digest([self._transition_generation,self.server.state.transition_series])
            if scope!=expected:
                return
            if client_id==self._transition_cursor_client and revision<self._transition_cursor_revision:
                return
            if client_id!=self._transition_cursor_client:
                self._cancel_transition_cursor()
            self._transition_cursor_client=client_id
            self._transition_cursor_revision=revision
            self._transition_cursor_request=None
        self._transition_cursor_dragging=True

    def commit_transition_cursor(self,run_id=None,epoch=None,index=None,revision=None,scope=None,client_id=None,**_):
        protocol=client_id is not None
        if protocol:
            if not self._valid_cursor_revision(client_id,revision):
                return
            if client_id!=self._transition_cursor_client or revision<self._transition_cursor_revision:
                self._acknowledge_transition_cursor(client_id,revision,False)
                self.server.state.flush()
                return
            self._transition_cursor_revision=revision
        self._transition_cursor_dragging=False
        expected=digest([self._transition_generation,self.server.state.transition_series])
        if (protocol and scope!=expected) or (not protocol and epoch!=self.server.state.transition_cursor_epoch):
            self._render_transition_tracking()
            if protocol:
                self._acknowledge_transition_cursor(client_id,revision,False)
            self.server.state.flush()
            return
        if run_id is None and epoch==self.server.state.transition_cursor_epoch:
            try:
                cursor=int(index)
                run_id=self._transition_map_ids[cursor] if 0<=cursor<len(self._transition_map_ids) else None
            except (TypeError,ValueError):
                run_id=None
        eligible={r["run_id"] for r in self._series_rows()} if protocol else set(self._transition_map_ids)
        if run_id not in eligible:
            if protocol:
                self._acknowledge_transition_cursor(client_id,revision,False)
                self.server.state.flush()
            return
        if run_id==self.server.state.transition_selected and not self._transition_render_pending:
            if protocol:
                self._acknowledge_transition_cursor(client_id,revision,True)
            self.server.state.flush()
            return
        self._queue_transition_cursor(run_id,client_id,revision)

    def _apply_transition_cursor_request(self,generation,run_id,client_id,revision):
        if generation!=self._transition_generation:
            return
        if client_id is not None and (client_id!=self._transition_cursor_client or revision!=self._transition_cursor_revision):
            return
        self._transition_cursor_applying=True
        try:
            self.select_transition_scan(run_id)
        finally:
            self._transition_cursor_applying=False
        if client_id is not None:
            self._acknowledge_transition_cursor(client_id,revision,True)
        self.server.state.flush()

    def _queue_transition_cursor(self,run_id,client_id=None,revision=None):
        self._transition_cursor_request=(self._transition_generation,run_id,client_id,revision)
        try:
            loop=asyncio.get_running_loop()
        except RuntimeError:
            request=self._transition_cursor_request
            self._transition_cursor_request=None
            self._apply_transition_cursor_request(*request)
            return
        if self._transition_cursor_task is None:
            self._transition_cursor_task=loop.create_task(self._slide_transition_map())

    async def _slide_transition_map(self):
        try:
            while self._transition_cursor_request is not None:
                await asyncio.sleep(.06)
                request=self._transition_cursor_request
                self._transition_cursor_request=None
                if request is not None:
                    self._apply_transition_cursor_request(*request)
        finally:
            self._transition_cursor_task=None

    def pin_transition_reference(self, **_):
        if self.server.state.transition_can_pin:
            self._transition_saved["references"][
                self.server.state.transition_series
            ] = self.server.state.transition_selected
            self.server.state.transition_comparison = "reference"
            self._render_transition_tracking()
            self._checkpoint_transition_tracking()

    def rescale_transition_map(self, **_):
        rows = self._window_rows(self._series_rows())
        if not rows:
            return
        limits = [
            locked_limits(
                self._transition_engine.load(
                    self._transition_scans[r["run_id"]]
                ).pattern,
                background=self._transition_engine.prepare(
                    self._transition_engine.load(self._transition_scans[r["run_id"]])
                ).baseline
                if self.server.state.transition_background
                else None,
            )
            for r in rows
            if r.get("state_id")
        ]
        if not limits:
            return
        color_key = self.server.state.transition_series + (
            ":local" if self.server.state.transition_background else ":observed"
        )
        self._transition_saved["limits"][color_key] = {
            "low": min(v["low"] for v in limits),
            "high": max(v["high"] for v in limits),
            "soft": limits[0]["soft"],
            "source": "visible eligible scans at explicit rescale",
        }
        self._render_transition_tracking()
        self._checkpoint_transition_tracking()

    def open_transition_plot(self, kind, **_):
        if kind not in {"map", "profile"}:
            return
        self.server.state.transition_dialog_kind = kind
        self.server.state.transition_dialog = True
        self._update_transition_dialog()

    def _update_transition_dialog(self):
        state = self.server.state
        figure = dict(
            state.transition_map
            if state.transition_dialog_kind == "map"
            else state.transition_profile
        )
        figure["layout"] = {**figure.get("layout", {}), "height": 800, "autosize": True}
        if state.transition_dialog_kind == "map" and figure.get("data"):
            figure["data"] = [dict(t) for t in figure["data"]]
            figure["data"][0]["colorbar"] = {
                **figure["data"][0].get("colorbar", {}),
                "y": -58 / (800 - 145),
            }
        state.transition_dialog_figure = figure

    def _checkpoint_transition_tracking(self):
        controller = getattr(self, "_powgen_controller", None)
        if controller is None or not str(
            getattr(self, "_post_analysis_scope", "")
        ).startswith("watch:"):
            return
        self._transition_saved["styles"] = dict(
            self._transition_report.get("styles", {})
        )
        controller.state.transition_tracking = checkpoint(self._transition_saved)
        # Persist references and color identities without curve arrays. The
        # snapshot is replaced as a whole so background JSON writes cannot
        # observe a dictionary being mutated by a later UI action.
        if not callable(getattr(controller, "persist_state", None)):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._transition_save_request = controller
        if getattr(self, "_transition_save_task", None) is None:
            self._transition_save_task = loop.create_task(
                self._persist_transition_tracking()
            )

    async def _persist_transition_tracking(self):
        try:
            while True:
                controller = self._transition_save_request
                before = digest(controller.state.transition_tracking)
                try:
                    await asyncio.to_thread(controller.persist_state)
                except Exception as exc:  # noqa: BLE001 - isolate an external checkpoint failure from the live UI
                    if controller is not self._transition_save_request:
                        continue
                    self.server.state.transition_error = f"Tracking checkpoint could not be saved ({type(exc).__name__}); check the Galaxy connection."
                    break
                if controller is self._transition_save_request and before == digest(
                    controller.state.transition_tracking
                ):
                    break
        finally:
            self._transition_save_task = None
            self.server.state.flush()
