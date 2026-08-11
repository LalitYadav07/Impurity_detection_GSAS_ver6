"""RADAR-PD interactive NOVA/Trame application."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from nova.trame import ThemedApp
from nova.trame.view.components import FileUpload
from trame.app import get_server
from trame.widgets import html, plotly, vuetify3 as vuetify

from .galaxy_service import GalaxyService
from .models import AnalysisConfig, AnalysisMode, InputSelection, InputSource, RunRecord, RunStatus
from .results import discover_plot_payloads, discover_tables, figure_for_payload, phase_fraction_rows, read_json, read_table


def _list_value(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    return [str(value)]


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


class RadarPdNovaApp(ThemedApp):
    """Interactive orchestration and results workspace backed by Galaxy."""

    def __init__(self) -> None:
        self.server = get_server(None, client_type="vue3")
        super().__init__(server=self.server)
        self.service = GalaxyService()
        self.records: dict[str, RunRecord] = {}
        self._plot_widget: Any | None = None
        self._initialize_state()
        self.create_ui()
        self.server.controller.on_server_ready.add(self._recover_runs)

    def _initialize_state(self) -> None:
        state = self.server.state
        state.trame__title = "RADAR-PD Interactive"
        state.active_page = "setup"
        state.connection_status = "Checking NDIP connection"
        state.connection_ok = False
        state.busy = False
        state.notice = ""
        state.error_message = ""
        state.analysis_mode = "rapid"
        state.radiation = "neutron"
        state.instrument_mode = "auto"
        state.input_source = "upload"
        state.data_path = ""
        state.instrument_path = ""
        state.main_cif_path = ""
        state.database_archive_path = ""
        state.event_file_path = ""
        state.use_builtin_cuka = False
        state.ipts_instrument = ""
        state.ipts = ""
        state.run_number = None
        state.bank = ""
        state.sample_elements = ""
        state.environment_elements = ""
        state.run_name = ""
        state.fit_start = None
        state.fit_end = None
        state.ignore_regions = ""
        state.reference_masks_enabled = False
        state.reference_mask_presets = []
        state.include_cu_kbeta = False
        state.background_mode = "auto_fixed_points"
        state.background_type = "chebyschev-1"
        state.background_terms = 6
        state.main_prenudge = True
        state.main_shadow_filter = True
        state.cleanup_enabled = False
        state.refine_u_iso = False
        state.refine_positions = False
        state.magnetic_precheck = False
        state.magnetic_q_max = 4.0
        state.full_profile = "balanced"
        state.full_max_passes = 2
        state.full_min_phase_percent = 0.5
        state.full_top_n_ml = 35
        state.full_nudge_candidates = 7
        state.full_nudge_samples = 5000
        state.full_nudge_representatives = 50
        state.full_compare_candidates = 2
        state.full_compare_cycles = 6
        state.full_cell_length_tolerance_pct = 1.0
        state.full_cell_angle_tolerance_deg = 3.0
        state.rapid_phases_per_hypothesis = 3
        state.rapid_stage_output_limit = 10
        state.rapid_gsas_validation_limit = 5
        state.rapid_parallel_workers = 4
        state.rapid_show_family_variants = True
        state.rapid_final_polish_enabled = False
        state.history_datasets = []
        state.history_data_id = ""
        state.history_instrument_id = ""
        state.history_main_cif_id = ""
        state.history_database_id = ""
        state.run_rows = []
        state.selected_run_uid = ""
        state.selected_run_name = "No run selected"
        state.selected_run_status = "-"
        state.selected_run_stage = "-"
        state.selected_run_progress = 0
        state.selected_run_message = ""
        state.summary_cards = []
        state.phase_rows = []
        state.table_options = []
        state.selected_table = ""
        state.table_rows = []
        state.table_headers = []
        state.plot_options = []
        state.selected_plot = ""
        state.artifact_options = []
        state.selected_artifact = ""
        state.gpx_rows = []
        state.solution_rows = []
        state.last_log = ""
        state.result_tab = "overview"
        state.mode_options = [
            {"title": "Rapid Hypothesis Mode", "value": "rapid"},
            {"title": "Full RADAR-PD", "value": "full"},
        ]
        state.source_options = [
            {"title": "Upload from this computer or select a server file", "value": "upload"},
            {"title": "Reuse datasets already in this Galaxy history", "value": "galaxy"},
            {"title": "SNS IPTS: resolve from a NeXus event file", "value": "ipts_event"},
            {"title": "SNS IPTS: instrument, IPTS, run, and bank", "value": "ipts_manual"},
        ]
        state.radiation_options = [
            {"title": "Neutron powder diffraction", "value": "neutron"},
            {"title": "X-ray powder diffraction", "value": "xray"},
        ]

    def create_ui(self) -> None:
        self.set_theme("CompactTheme")
        with super().create_ui() as layout:
            layout.toolbar_title.set_text("RADAR-PD Interactive")
            with layout.pre_content:
                self._navigation()
            with layout.content:
                html.Style(self._css())
                with html.Div(classes="radar-shell"):
                    self._header()
                    with html.Div(v_show="active_page === 'setup'"):
                        self._setup_page()
                    with html.Div(v_show="active_page === 'runs'"):
                        self._runs_page()
                    with html.Div(v_show="active_page === 'results'"):
                        self._results_page()
                    with html.Div(v_show="active_page === 'artifacts'"):
                        self._artifacts_page()
            return layout

    def _navigation(self) -> None:
        with vuetify.VTabs(v_model=("active_page", "setup"), color="#0b5d46", density="compact"):
            vuetify.VTab("New analysis", value="setup", prepend_icon="mdi-flask-outline")
            vuetify.VTab("Runs", value="runs", prepend_icon="mdi-progress-clock")
            vuetify.VTab("Results", value="results", prepend_icon="mdi-chart-line")
            vuetify.VTab("Artifacts", value="artifacts", prepend_icon="mdi-folder-outline")

    def _header(self) -> None:
        with html.Section(classes="radar-hero"):
            html.Div("RADAR-PD SCIENTIFIC AI WORKSPACE", classes="radar-kicker")
            html.H1("Phase detection for powder diffraction")
            html.P(
                "Configure Full or Rapid analysis, submit it to NDIP compute, leave this page safely, and return to inspect the same Galaxy-backed run.",
                classes="radar-subtitle",
            )
            with html.Div(classes="radar-status-row"):
                vuetify.VChip(
                    text=("connection_status",),
                    color=("connection_ok ? '#d8f3e6' : '#fff0d4'",),
                    prepend_icon=("connection_ok ? 'mdi-cloud-check-outline' : 'mdi-cloud-alert-outline'",),
                    variant="flat",
                    size="small",
                )
                vuetify.VChip("Neutron and X-ray", size="small", variant="outlined")
                vuetify.VChip("Full and Rapid", size="small", variant="outlined")
                vuetify.VChip("Galaxy-backed recovery", size="small", variant="outlined")

    def _section(self, title: str, subtitle: str, number: int) -> Any:
        card = vuetify.VCard(classes="radar-card", variant="flat")
        card.__enter__()
        with vuetify.VCardTitle(classes="radar-card-title"):
            html.Span(str(number), classes="step-number")
            html.Span(title)
        vuetify.VCardSubtitle(subtitle, classes="radar-card-subtitle")
        return card

    def _close_section(self, card: Any) -> None:
        card.__exit__(None, None, None)

    def _setup_page(self) -> None:
        with vuetify.VRow(classes="mt-4", align="start"):
            with vuetify.VCol(cols=12, lg=7):
                card = self._section("Analysis path", "Both modes use the same inputs and scientific safeguards.", 1)
                with vuetify.VCardText():
                    vuetify.VSelect(
                        label="Analysis mode",
                        v_model=("analysis_mode",),
                        items=("mode_options",),
                        item_title="title",
                        item_value="value",
                        variant="outlined",
                        density="comfortable",
                    )
                    with vuetify.VRow():
                        with vuetify.VCol(cols=12, md=6):
                            vuetify.VSelect(
                                label="Measurement type",
                                v_model=("radiation",),
                                items=("radiation_options",),
                                item_title="title",
                                item_value="value",
                                variant="outlined",
                            )
                        with vuetify.VCol(cols=12, md=6):
                            vuetify.VSelect(
                                label="Pattern geometry",
                                v_model=("instrument_mode",),
                                items=("[{title:'Auto detect',value:'auto'},{title:'Constant wavelength',value:'cw'},{title:'Time of flight',value:'tof'}]",),
                                item_title="title",
                                item_value="value",
                                variant="outlined",
                            )
                self._close_section(card)

                card = self._section("Diffraction data", "Upload from the laptop, reuse Galaxy data, or resolve an SNS IPTS run.", 2)
                with vuetify.VCardText():
                    vuetify.VSelect(
                        label="Where is the diffraction pattern?",
                        v_model=("input_source",),
                        items=("source_options",),
                        item_title="title",
                        item_value="value",
                        variant="outlined",
                    )
                    with html.Div(v_if="input_source === 'upload'", classes="upload-grid"):
                        FileUpload(
                            "data_path",
                            label="Choose diffraction data",
                            extensions=[".dat", ".xye", ".xy", ".csv", ".txt", ".fxye", ".xrdml", ".xml"],
                            return_contents=False,
                            show_server_files=True,
                            color="#0b5d46",
                        )
                        FileUpload(
                            "instrument_path",
                            label="Choose GSAS-II instrument profile",
                            extensions=[".instprm", ".prm", ".inst", ".ins"],
                            return_contents=False,
                            show_server_files=True,
                            color="#0b5d46",
                        )
                        vuetify.VSwitch(
                            v_if="radiation === 'xray' && instrument_mode !== 'tof'",
                            v_model=("use_builtin_cuka",),
                            label="Use built-in Cu K-alpha laboratory profile",
                            color="#0b5d46",
                            inset=True,
                        )
                    with html.Div(v_if="input_source === 'galaxy'"):
                        html.P("Select durable inputs from the current Galaxy history.", classes="field-help")
                        vuetify.VSelect(label="Diffraction data", v_model=("history_data_id",), items=("history_datasets",), item_title="name", item_value="id", variant="outlined")
                        vuetify.VSelect(label="Instrument profile", v_model=("history_instrument_id",), items=("history_datasets",), item_title="name", item_value="id", variant="outlined")
                        vuetify.VSwitch(v_if="radiation === 'xray'", v_model=("use_builtin_cuka",), label="Use built-in Cu K-alpha laboratory profile", color="#0b5d46")
                    with html.Div(v_if="input_source === 'ipts_event'"):
                        FileUpload("event_file_path", label="Choose NeXus event file", extensions=[".nxs", ".h5", ".hdf5"], return_contents=False, show_server_files=True, color="#0b5d46")
                        vuetify.VTextField(label="Detector bank", v_model=("bank",), variant="outlined", placeholder="for example bank1")
                    with html.Div(v_if="input_source === 'ipts_manual'"):
                        with vuetify.VRow():
                            with vuetify.VCol(cols=12, sm=6):
                                vuetify.VTextField(label="SNS instrument", v_model=("ipts_instrument",), variant="outlined", placeholder="HB2A")
                            with vuetify.VCol(cols=12, sm=6):
                                vuetify.VTextField(label="IPTS", v_model=("ipts",), variant="outlined", placeholder="IPTS-12345")
                            with vuetify.VCol(cols=12, sm=6):
                                vuetify.VTextField(label="Run number", v_model=("run_number",), type="number", variant="outlined")
                            with vuetify.VCol(cols=12, sm=6):
                                vuetify.VTextField(label="Detector bank", v_model=("bank",), variant="outlined")
                    vuetify.VDivider(classes="my-4")
                    html.H3("Known main phase and candidate library", classes="subsection-title")
                    with html.Div(v_if="input_source === 'galaxy'"):
                        vuetify.VSelect(label="Known/main phase CIF (optional)", v_model=("history_main_cif_id",), items=("history_datasets",), item_title="name", item_value="id", clearable=True, variant="outlined")
                        vuetify.VSelect(label="Custom candidate-library archive (optional)", v_model=("history_database_id",), items=("history_datasets",), item_title="name", item_value="id", clearable=True, variant="outlined")
                    with html.Div(v_if="input_source !== 'galaxy'", classes="upload-grid"):
                        FileUpload("main_cif_path", label="Choose known/main phase CIF (optional)", extensions=[".cif"], return_contents=False, show_server_files=True, color="#0b5d46")
                        FileUpload("database_archive_path", label="Choose candidate-library ZIP (optional)", extensions=[".zip"], return_contents=False, show_server_files=True, color="#0b5d46")
                self._close_section(card)

                card = self._section("Chemistry and pattern policy", "Define sample chemistry separately from container or environment chemistry.", 3)
                with vuetify.VCardText():
                    with vuetify.VRow():
                        with vuetify.VCol(cols=12, md=6):
                            vuetify.VTextField(label="Sample elements", v_model=("sample_elements",), placeholder="Cu, P, Pb, O, S", variant="outlined", hint="Comma- or space-separated element symbols", persistent_hint=True)
                        with vuetify.VCol(cols=12, md=6):
                            vuetify.VTextField(label="Sample can / environment elements", v_model=("environment_elements",), placeholder="Al, V", variant="outlined", hint="Allowed as pure environment phases, not mixed freely into sample chemistry", persistent_hint=True)
                    with vuetify.VRow():
                        with vuetify.VCol(cols=12, sm=6):
                            vuetify.VTextField(label="Fit range start (optional)", v_model=("fit_start",), type="number", variant="outlined")
                        with vuetify.VCol(cols=12, sm=6):
                            vuetify.VTextField(label="Fit range end (optional)", v_model=("fit_end",), type="number", variant="outlined")
                    vuetify.VTextarea(label="Ignored regions", v_model=("ignore_regions",), placeholder="One start,end pair per line\n2.0, 3.2", variant="outlined", rows=2, auto_grow=True)
                    vuetify.VSwitch(v_model=("reference_masks_enabled",), label="Mask selected can/reference peak positions", color="#0b5d46", inset=True)
                    vuetify.VSelect(v_if="reference_masks_enabled", label="Reference structures", v_model=("reference_mask_presets",), items=("['Al_fcc','Cu_fcc','V_bcc']",), multiple=True, chips=True, variant="outlined")
                    vuetify.VSwitch(v_if="radiation === 'xray' && reference_masks_enabled", v_model=("include_cu_kbeta",), label="Also mask Cu K-beta companion positions", color="#0b5d46", inset=True)
                self._close_section(card)

                card = self._section("Refinement safeguards", "These options are shared by Full and Rapid analysis.", 4)
                with vuetify.VCardText():
                    with vuetify.VRow():
                        with vuetify.VCol(cols=12, md=4):
                            vuetify.VSelect(label="Background correction", v_model=("background_mode",), items=("[{title:'Automatic fixed points',value:'auto_fixed_points'},{title:'Manual / pipeline default',value:'manual'}]",), item_title="title", item_value="value", variant="outlined")
                        with vuetify.VCol(cols=12, md=4):
                            vuetify.VSelect(label="Background function", v_model=("background_type",), items=("['chebyschev-1','chebyschev','cosine','Q^2 power series']",), variant="outlined")
                        with vuetify.VCol(cols=12, md=4):
                            vuetify.VTextField(label="Background terms", v_model=("background_terms",), type="number", min=1, max=36, variant="outlined")
                    vuetify.VSwitch(v_model=("main_prenudge",), label="Automatically anchor a supplied main-phase cell", color="#0b5d46", inset=True)
                    vuetify.VSwitch(v_model=("main_shadow_filter",), label="Filter candidates supported only by strong main-phase peaks", color="#0b5d46", inset=True)
                    vuetify.VSwitch(v_model=("magnetic_precheck",), label="Check whether residual peaks can be indexed by a commensurate magnetic propagation vector", color="#0b5d46", inset=True)
                    with html.Div(v_if="magnetic_precheck"):
                        vuetify.VTextField(label="Magnetic precheck Q maximum", v_model=("magnetic_q_max",), type="number", variant="outlined")
                    vuetify.VSwitch(v_model=("cleanup_enabled",), label="Refine supplied main-CIF internal parameters after lattice anchoring", color="#0b5d46", inset=True)
                    with vuetify.VRow(v_if="cleanup_enabled"):
                        with vuetify.VCol(cols=12, sm=6):
                            vuetify.VCheckbox(v_model=("refine_u_iso",), label="Refine isotropic displacement parameters")
                        with vuetify.VCol(cols=12, sm=6):
                            vuetify.VCheckbox(v_model=("refine_positions",), label="Refine atomic positions")
                self._close_section(card)

                card = self._section("Mode-specific search controls", "Only controls relevant to the selected analysis path are shown.", 5)
                with vuetify.VCardText():
                    with html.Div(v_if="analysis_mode === 'rapid'"):
                        with vuetify.VRow():
                            with vuetify.VCol(cols=12, md=6):
                                vuetify.VTextField(label="Phases per hypothesis", v_model=("rapid_phases_per_hypothesis",), type="number", min=1, max=5, variant="outlined")
                            with vuetify.VCol(cols=12, md=6):
                                vuetify.VTextField(label="Hypotheses retained per stage", v_model=("rapid_stage_output_limit",), type="number", min=3, max=50, variant="outlined")
                            with vuetify.VCol(cols=12, md=6):
                                vuetify.VTextField(label="Final refinements", v_model=("rapid_gsas_validation_limit",), type="number", min=0, variant="outlined")
                            with vuetify.VCol(cols=12, md=6):
                                vuetify.VTextField(label="Parallel workers", v_model=("rapid_parallel_workers",), type="number", min=1, max=16, variant="outlined")
                        vuetify.VSwitch(v_model=("rapid_show_family_variants",), label="Keep same-family variants available in Solution Inspector", color="#0b5d46", inset=True)
                        vuetify.VSwitch(v_model=("rapid_final_polish_enabled",), label="Run final polish after ranking", color="#0b5d46", inset=True)
                    with html.Div(v_if="analysis_mode === 'full'"):
                        vuetify.VSelect(label="Search profile", v_model=("full_profile",), items=("['quick','balanced','thorough','custom']",), variant="outlined")
                        with vuetify.VRow():
                            with vuetify.VCol(cols=12, md=6):
                                vuetify.VTextField(label="Impurity discovery rounds", v_model=("full_max_passes",), type="number", min=1, variant="outlined")
                            with vuetify.VCol(cols=12, md=6):
                                vuetify.VTextField(label="Stop below phase fraction (wt%)", v_model=("full_min_phase_percent",), type="number", min=0, variant="outlined")
                        with vuetify.VExpansionPanels(variant="accordion", classes="mt-2"):
                            with vuetify.VExpansionPanel(title="Custom search breadth and lattice nudge"):
                                with vuetify.VExpansionPanelText():
                                    for label, model in (
                                        ("ML candidates", "full_top_n_ml"),
                                        ("Lattice-nudge candidates", "full_nudge_candidates"),
                                        ("Nudge samples", "full_nudge_samples"),
                                        ("Nudge representatives", "full_nudge_representatives"),
                                        ("Joint comparison candidates", "full_compare_candidates"),
                                        ("Joint comparison cycles", "full_compare_cycles"),
                                    ):
                                        vuetify.VTextField(label=label, v_model=(model,), type="number", min=1, variant="outlined", density="compact")
                self._close_section(card)

            with vuetify.VCol(cols=12, lg=5):
                with vuetify.VCard(classes="review-card sticky-review", variant="flat"):
                    vuetify.VCardTitle("Review and submit")
                    with vuetify.VCardText():
                        html.P("A portable configuration is generated from these controls and saved with the Galaxy run.", classes="field-help")
                        vuetify.VTextField(label="Run name (optional)", v_model=("run_name",), variant="outlined", placeholder="Generated automatically when left blank")
                        with html.Div(classes="review-summary"):
                            html.Div("Mode", classes="review-label")
                            html.Div("{{ analysis_mode === 'rapid' ? 'Rapid Hypothesis Mode' : 'Full RADAR-PD' }}", classes="review-value")
                            html.Div("Measurement", classes="review-label")
                            html.Div("{{ radiation === 'neutron' ? 'Neutron' : 'X-ray' }} / {{ instrument_mode.toUpperCase() }}", classes="review-value")
                            html.Div("Source", classes="review-label")
                            html.Div("{{ source_options.find(x => x.value === input_source)?.title }}", classes="review-value")
                            html.Div("Sample", classes="review-label")
                            html.Div("{{ sample_elements || 'Enter sample elements' }}", classes="review-value")
                        vuetify.VAlert(v_if="error_message", text=("error_message",), type="error", variant="tonal", classes="mb-3")
                        vuetify.VAlert(v_if="notice", text=("notice",), type="success", variant="tonal", classes="mb-3")
                        vuetify.VBtn(
                            "Submit analysis to NDIP",
                            click=self.submit_run,
                            loading=("busy",),
                            disabled=("!connection_ok || busy",),
                            color="#0b5d46",
                            size="large",
                            block=True,
                            prepend_icon="mdi-rocket-launch-outline",
                        )
                        vuetify.VBtn("Refresh Galaxy inputs", click=self.refresh_history, variant="outlined", block=True, classes="mt-2", prepend_icon="mdi-refresh")

    def _runs_page(self) -> None:
        with html.Div(classes="page-section"):
            with html.Div(classes="section-heading-row"):
                with html.Div():
                    html.H2("Galaxy-backed runs")
                    html.P("Close this NOVA session and return later; the analysis and its outputs remain in Galaxy.")
                vuetify.VBtn("Recover from history", click=self._recover_runs, prepend_icon="mdi-history", color="#0b5d46", variant="outlined")
            with vuetify.VCard(classes="radar-card", variant="flat"):
                vuetify.VDataTable(
                    headers=("[{title:'Run',key:'name'},{title:'Mode',key:'mode'},{title:'Status',key:'status'},{title:'Current stage',key:'stage'},{title:'Progress',key:'progress'}]",),
                    items=("run_rows",),
                    item_value="uid",
                    hover=True,
                    density="comfortable",
                    click_row=self._row_selected,
                    no_data_text="No RADAR-PD runs are present in this history yet.",
                )
            with vuetify.VCard(classes="radar-card mt-4", variant="flat", v_if="selected_run_uid"):
                with vuetify.VCardTitle(classes="d-flex align-center justify-space-between"):
                    html.Span("{{ selected_run_name }}")
                    vuetify.VChip(text=("selected_run_status",), color="#d8f3e6", variant="flat")
                with vuetify.VCardText():
                    html.Div("{{ selected_run_stage }}", classes="stage-label")
                    vuetify.VProgressLinear(model_value=("selected_run_progress",), color="#0b5d46", height=10, rounded=True, classes="mb-3")
                    vuetify.VAlert(v_if="selected_run_message", text=("selected_run_message",), type="warning", variant="tonal")
                    with vuetify.VBtnGroup(variant="outlined", divided=True):
                        vuetify.VBtn("Refresh", click=self.refresh_selected_run, prepend_icon="mdi-refresh")
                        vuetify.VBtn("Inspect results", click=self.open_selected_results, prepend_icon="mdi-chart-box-outline", disabled=("selected_run_status !== 'Ok'",))
                        vuetify.VBtn("Stop", click=self.cancel_selected_run, prepend_icon="mdi-stop-circle-outline", color="#a33131", disabled=("selected_run_status !== 'Running' && selected_run_status !== 'Queued'",))

    def _results_page(self) -> None:
        with html.Div(classes="page-section"):
            html.H2("Scientific results")
            html.P("Tables and interactive plots are reconstructed from RADAR-PD's normalized Galaxy outputs.")
            vuetify.VAlert(v_if="!selected_run_uid", text="Select a completed run from the Runs page.", type="info", variant="tonal")
            with html.Div(v_if="selected_run_uid"):
                with html.Div(classes="metric-grid"):
                    with vuetify.VCard(v_for="card in summary_cards", key="card.label", classes="metric-card", variant="flat"):
                        html.Div("{{ card.label }}", classes="metric-label")
                        html.Div("{{ card.value }}", classes="metric-value")
                with vuetify.VTabs(v_model=("result_tab", "overview"), color="#0b5d46", classes="mt-4"):
                    vuetify.VTab("Overview", value="overview")
                    vuetify.VTab("Rankings and tables", value="tables")
                    vuetify.VTab("Interactive plots", value="plots")
                    vuetify.VTab(v_if="analysis_mode === 'rapid'", text="Solution Inspector", value="inspector")
                with html.Div(v_show="result_tab === 'overview'", classes="result-panel"):
                    html.H3("Phase fractions")
                    vuetify.VDataTable(
                        headers=("[{title:'Phase',key:'phase'},{title:'Space group',key:'space_group'},{title:'Weight (%)',key:'weight_percent'}]",),
                        items=("phase_rows",),
                        density="compact",
                        no_data_text="Phase fractions are not available in the normalized summary.",
                    )
                with html.Div(v_show="result_tab === 'tables'", classes="result-panel"):
                    vuetify.VSelect(label="Result table", v_model=("selected_table",), items=("table_options",), item_title="name", item_value="path", variant="outlined", change=self._table_changed)
                    vuetify.VDataTable(headers=("table_headers",), items=("table_rows",), density="compact", fixed_header=True, height=520, no_data_text="Select a result table.")
                with html.Div(v_show="result_tab === 'plots'", classes="result-panel"):
                    vuetify.VSelect(label="Interactive plot", v_model=("selected_plot",), items=("plot_options",), item_title="name", item_value="path", variant="outlined", change=self._plot_changed)
                    self._plot_widget = plotly.Figure(display_mode_bar=True)
                with html.Div(v_show="result_tab === 'inspector'", classes="result-panel"):
                    html.H3("Rapid hypothesis inspector")
                    html.P("Compare pattern-ranked hypotheses and same-family variants before choosing a refinement checkpoint. Follow-up refinement is submitted as a separate Galaxy job so the original result remains unchanged.", classes="field-help")
                    vuetify.VDataTable(headers=("table_headers",), items=("solution_rows",), density="compact", no_data_text="No rapid hypothesis table was published for this run.")
                    vuetify.VAlert(text="Targeted refinement and GPX handoff remain separate Galaxy actions. Select the source hypothesis or checkpoint under Artifacts, then launch the appropriate follow-up tool without modifying this run.", type="info", variant="tonal", classes="mt-3")

    def _artifacts_page(self) -> None:
        with html.Div(classes="page-section"):
            html.H2("Artifacts and downstream handoffs")
            html.P("Download the complete archive, open the HTML report, or preserve a GPX checkpoint for hosted GSAS-II.")
            with vuetify.VCard(classes="radar-card", variant="flat"):
                with vuetify.VCardText():
                    vuetify.VSelect(label="Artifact", v_model=("selected_artifact",), items=("artifact_options",), item_title="title", item_value="path", variant="outlined")
                    vuetify.VBtn("Download selected artifact", click=self.download_artifact, color="#0b5d46", prepend_icon="mdi-download", disabled=("!selected_artifact",))
            html.H3("GSAS-II refinement checkpoints", classes="mt-6")
            vuetify.VDataTable(
                headers=("[{title:'Checkpoint',key:'name'},{title:'Stage',key:'stage'},{title:'Status',key:'status'},{title:'File',key:'path'}]",),
                items=("gpx_rows",),
                density="comfortable",
                no_data_text="No GPX checkpoint index is available for this run.",
            )

    def _parse_regions(self) -> list[tuple[float, float]]:
        regions: list[tuple[float, float]] = []
        for line in str(self.server.state.ignore_regions or "").splitlines():
            if not line.strip():
                continue
            parts = [part.strip() for part in line.replace(";", ",").split(",")]
            if len(parts) != 2:
                raise ValueError(f"Ignored region must be 'start,end': {line}")
            regions.append((float(parts[0]), float(parts[1])))
        return regions

    def _configuration(self) -> AnalysisConfig:
        state = self.server.state
        run_name = str(state.run_name or "").strip()
        fit_start = _optional_float(state.fit_start)
        fit_end = _optional_float(state.fit_end)
        if (fit_start is None) != (fit_end is None):
            raise ValueError("Fit range requires both a start and an end value")
        values: dict[str, Any] = {
            "mode": state.analysis_mode,
            "radiation": state.radiation,
            "instrument_mode": state.instrument_mode,
            "sample_elements": state.sample_elements,
            "environment_elements": state.environment_elements,
            "limits": None if fit_start is None else (fit_start, fit_end),
            "exclude_regions": self._parse_regions(),
            "reference_masks_enabled": bool(state.reference_masks_enabled),
            "reference_mask_presets": _list_value(state.reference_mask_presets),
            "include_cu_kbeta": bool(state.include_cu_kbeta),
            "background_mode": state.background_mode,
            "background_type": state.background_type,
            "background_terms": int(state.background_terms),
            "main_prenudge": bool(state.main_prenudge),
            "main_shadow_filter": bool(state.main_shadow_filter),
            "cleanup_enabled": bool(state.cleanup_enabled),
            "refine_u_iso": bool(state.refine_u_iso),
            "refine_positions": bool(state.refine_positions),
            "magnetic_precheck": bool(state.magnetic_precheck),
            "magnetic_q_max": float(state.magnetic_q_max),
            "full_profile": state.full_profile,
            "full_max_passes": int(state.full_max_passes),
            "full_min_phase_percent": float(state.full_min_phase_percent),
            "full_top_n_ml": int(state.full_top_n_ml),
            "full_nudge_candidates": int(state.full_nudge_candidates),
            "full_nudge_samples": int(state.full_nudge_samples),
            "full_nudge_representatives": int(state.full_nudge_representatives),
            "full_compare_candidates": int(state.full_compare_candidates),
            "full_compare_cycles": int(state.full_compare_cycles),
            "full_cell_length_tolerance_pct": float(state.full_cell_length_tolerance_pct),
            "full_cell_angle_tolerance_deg": float(state.full_cell_angle_tolerance_deg),
            "rapid_phases_per_hypothesis": int(state.rapid_phases_per_hypothesis),
            "rapid_stage_output_limit": int(state.rapid_stage_output_limit),
            "rapid_gsas_validation_limit": int(state.rapid_gsas_validation_limit),
            "rapid_parallel_workers": int(state.rapid_parallel_workers),
            "rapid_show_family_variants": bool(state.rapid_show_family_variants),
            "rapid_final_polish_enabled": bool(state.rapid_final_polish_enabled),
        }
        if run_name:
            values["run_name"] = run_name
        return AnalysisConfig(**values)

    def _inputs(self) -> InputSelection:
        state = self.server.state
        source = InputSource(state.input_source)
        return InputSelection(
            source=source,
            data_path=(state.data_path or None) if source == InputSource.UPLOAD else None,
            data_dataset_id=(state.history_data_id or None) if source == InputSource.GALAXY else None,
            instrument_path=(state.instrument_path or None) if source == InputSource.UPLOAD else None,
            instrument_dataset_id=(state.history_instrument_id or None) if source == InputSource.GALAXY else None,
            main_cif_path=(state.main_cif_path or None) if source != InputSource.GALAXY else None,
            main_cif_dataset_id=(state.history_main_cif_id or None) if source == InputSource.GALAXY else None,
            database_archive_path=(state.database_archive_path or None) if source != InputSource.GALAXY else None,
            database_dataset_id=(state.history_database_id or None) if source == InputSource.GALAXY else None,
            use_builtin_cuka=bool(state.use_builtin_cuka),
            event_file_path=state.event_file_path or None,
            instrument=state.ipts_instrument or None,
            ipts=state.ipts or None,
            run_number=int(state.run_number) if state.run_number not in (None, "") else None,
            bank=state.bank or None,
        )

    def submit_run(self, **_: Any) -> None:
        state = self.server.state
        state.error_message = ""
        state.notice = ""
        state.busy = True
        state.flush()
        try:
            config = self._configuration()
            inputs = self._inputs()
            record = self.service.submit(config, inputs)
            self.records[record.uid] = record
            state.run_name = config.run_name
            state.selected_run_uid = record.uid
            state.notice = f"{config.run_name} was submitted to NDIP. You may close this NOVA session safely."
            state.active_page = "runs"
            self._sync_runs()
            self._select_record(record)
            threading.Thread(target=self.service.monitor, args=(record, self._monitor_update), daemon=True).start()
        except Exception as exc:
            state.error_message = str(exc)
        finally:
            state.busy = False
            state.flush()

    def _monitor_update(self, record: RunRecord) -> None:
        self.records[record.uid] = record
        self._sync_runs()
        if self.server.state.selected_run_uid == record.uid:
            self._select_record(record)
            if record.status == RunStatus.OK:
                self._load_results(record)
        self.server.state.flush()

    def _sync_runs(self) -> None:
        self.server.state.run_rows = [record.as_row() for record in sorted(self.records.values(), key=lambda item: item.created_utc, reverse=True)]

    def _recover_runs(self, **_: Any) -> None:
        state = self.server.state
        try:
            self.service.validate_connection()
            state.connection_ok = True
            state.connection_status = "Connected to NDIP"
            self.refresh_history()
            for recovered in self.service.recent_runs():
                current = self.records.get(recovered.uid)
                self.records[recovered.uid] = current or recovered
            self._sync_runs()
        except Exception as exc:
            state.connection_ok = False
            state.connection_status = "NDIP connection unavailable"
            state.error_message = str(exc)
        state.flush()

    def refresh_history(self, **_: Any) -> None:
        state = self.server.state
        try:
            state.history_datasets = self.service.list_history_datasets()
            state.notice = f"Loaded {len(state.history_datasets)} reusable datasets from this history."
        except Exception as exc:
            state.error_message = f"Could not load Galaxy datasets: {exc}"
        state.flush()

    def _row_selected(self, _event: Any, item: Any, **__: Any) -> None:
        row = item.get("item", item) if isinstance(item, dict) else {}
        uid = str(row.get("uid") or "")
        if uid and uid in self.records:
            self._select_record(self.records[uid])

    def _select_record(self, record: RunRecord) -> None:
        state = self.server.state
        state.selected_run_uid = record.uid
        state.selected_run_name = record.name
        state.selected_run_status = record.status.value.title()
        state.selected_run_stage = record.stage
        state.selected_run_progress = record.progress
        state.selected_run_message = record.message

    def refresh_selected_run(self, **_: Any) -> None:
        uid = self.server.state.selected_run_uid
        if not uid or uid not in self.records:
            return
        try:
            record = self.service.refresh(self.records[uid])
            if record.status == RunStatus.OK and not record.output_dir:
                record = self.service.collect_results(record)
            self.records[uid] = record
            self._select_record(record)
            self._sync_runs()
        except Exception as exc:
            self.server.state.selected_run_message = str(exc)
        self.server.state.flush()

    def cancel_selected_run(self, **_: Any) -> None:
        uid = self.server.state.selected_run_uid
        if not uid:
            return
        try:
            self.service.cancel(uid)
            self.records[uid].status = RunStatus.CANCELLED
            self.records[uid].stage = "Stopped by user"
            self._select_record(self.records[uid])
            self._sync_runs()
        except Exception as exc:
            self.server.state.selected_run_message = str(exc)
        self.server.state.flush()

    def open_selected_results(self, **_: Any) -> None:
        uid = self.server.state.selected_run_uid
        if uid and uid in self.records:
            record = self.records[uid]
            if not record.output_dir:
                record = self.service.collect_results(record)
            self._load_results(record)
            self.server.state.active_page = "results"
            self.server.state.flush()

    def _load_results(self, record: RunRecord) -> None:
        state = self.server.state
        payload = self.service.result_payload(record)
        summary = payload.get("summary") or {}
        state.analysis_mode = record.mode.value
        state.phase_rows = phase_fraction_rows(summary)
        total_seconds = summary.get("elapsed_seconds") or summary.get("total_seconds") or summary.get("timing", {}).get("total")
        state.summary_cards = [
            {"label": "Run", "value": record.name},
            {"label": "Analysis", "value": "Rapid Hypothesis" if record.mode == AnalysisMode.RAPID else "Full RADAR-PD"},
            {"label": "Status", "value": record.status.value.title()},
            {"label": "Total time", "value": f"{float(total_seconds):.1f} s" if total_seconds is not None else "See stage timing"},
        ]
        root = record.local_output_dir
        if root is None:
            return
        state.table_options = discover_tables(root)
        state.plot_options = discover_plot_payloads(root)
        state.artifact_options = [
            {"title": f"{item['kind'].title()} / {item['name']} ({item['size'] / 1024:.1f} KB)", "path": item["path"]}
            for item in payload.get("artifacts", [])
        ]
        state.gpx_rows = [
            {
                "name": item.get("label") or item.get("name") or Path(str(item.get("path") or "project.gpx")).name,
                "stage": item.get("stage") or "Refinement",
                "status": item.get("status") or "Available",
                "path": item.get("path") or item.get("source_path") or "-",
            }
            for item in (payload.get("gpx_index") or {}).get("projects", [])
        ]
        if state.table_options:
            state.selected_table = state.table_options[0]["path"]
            self._table_changed()
        if state.plot_options:
            state.selected_plot = state.plot_options[0]["path"]
            self._plot_changed()
        rapid_tables = [item for item in state.table_options if any(token in item["path"].lower() for token in ("validation_summary", "reranked_512", "hypothesis"))]
        if rapid_tables:
            state.solution_rows = read_table(rapid_tables[0]["path"], limit=100)
        state.flush()

    def _table_changed(self, **_: Any) -> None:
        path = self.server.state.selected_table
        rows = read_table(path) if path else []
        self.server.state.table_rows = rows
        keys = list(rows[0].keys()) if rows else []
        self.server.state.table_headers = [{"title": key.replace("_", " ").title(), "key": key} for key in keys]
        self.server.state.flush()

    def _plot_changed(self, **_: Any) -> None:
        path = self.server.state.selected_plot
        if not path or self._plot_widget is None:
            return
        figure = figure_for_payload(read_json(path))
        self._plot_widget.update(figure)

    def download_artifact(self, **_: Any) -> None:
        path = Path(str(self.server.state.selected_artifact or ""))
        if not path.is_file():
            self.server.state.error_message = "The selected artifact is no longer available in this NOVA session."
            return
        self.download_file(path.name, "application/octet-stream", path.read_bytes())

    @staticmethod
    def _css() -> str:
        return """
        :root { color-scheme: light; }
        body { background: #f5f8f7; }
        .radar-shell { max-width: 1480px; margin: 0 auto; padding: 8px 24px 48px; color: #17251f; }
        .radar-hero { padding: 34px 4px 22px; border-bottom: 1px solid #d7e3de; }
        .radar-kicker { display: inline-block; color: #0b5d46; border: 1px solid #b8d4c9; border-radius: 999px; padding: 5px 10px; font-size: 12px; font-weight: 700; }
        .radar-hero h1 { font-size: clamp(32px, 4vw, 52px); line-height: 1.05; margin: 18px 0 12px; letter-spacing: 0; }
        .radar-subtitle { max-width: 900px; color: #52645c; font-size: 18px; }
        .radar-status-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
        .radar-card, .review-card { background: #fff; border: 1px solid #d7e3de; border-left: 4px solid #0b5d46; border-radius: 7px; box-shadow: 0 4px 14px rgba(21,54,43,.06); margin-bottom: 16px; }
        .radar-card-title { display: flex; align-items: center; gap: 10px; font-size: 18px; font-weight: 700; padding-bottom: 4px; }
        .radar-card-subtitle { white-space: normal; color: #65766f; padding-left: 58px; }
        .step-number { display: grid; place-items: center; width: 28px; height: 28px; border-radius: 50%; background: #d8f3e6; color: #0b5d46; font-weight: 800; }
        .subsection-title { font-size: 16px; margin: 0 0 14px; }
        .field-help { color: #687972; line-height: 1.55; }
        .upload-grid { display: grid; grid-template-columns: 1fr; gap: 12px; margin-bottom: 12px; }
        .sticky-review { position: sticky; top: 12px; }
        .review-summary { display: grid; grid-template-columns: 130px 1fr; gap: 8px 14px; background: #f1f7f4; border-radius: 6px; padding: 16px; margin: 12px 0 20px; }
        .review-label { color: #6a7973; font-size: 13px; }
        .review-value { color: #183c30; font-weight: 650; overflow-wrap: anywhere; }
        .page-section { padding-top: 32px; }
        .page-section h2 { font-size: 30px; margin-bottom: 4px; }
        .section-heading-row { display: flex; justify-content: space-between; align-items: end; gap: 16px; margin-bottom: 18px; }
        .stage-label { font-weight: 700; color: #0b5d46; margin-bottom: 8px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 18px; }
        .metric-card { padding: 15px 17px; border: 1px solid #d7e3de; border-left: 4px solid #0b5d46; border-radius: 6px; background: #fff; }
        .metric-label { color: #66776f; font-size: 12px; font-weight: 700; text-transform: uppercase; }
        .metric-value { color: #17251f; font-size: 19px; font-weight: 750; margin-top: 5px; overflow-wrap: anywhere; }
        .result-panel { background: #fff; border: 1px solid #d7e3de; border-radius: 7px; padding: 18px; margin-top: 10px; }
        @media (max-width: 800px) { .radar-shell { padding: 4px 12px 32px; } .review-summary { grid-template-columns: 1fr; } .sticky-review { position: static; } }
        """
