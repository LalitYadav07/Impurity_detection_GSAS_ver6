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

from .configuration import config_from_contract, load_configuration
from .galaxy_service import GalaxyService
from .models import AnalysisConfig, AnalysisMode, InputSelection, InputSource, RunRecord, RunStatus, selected_run_uid
from .results import discover_plot_payloads, discover_tables, figure_for_payload, phase_fraction_rows, read_json, read_table, total_elapsed_seconds


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
        self._monitored_uids: set[str] = set()
        self._plot_widget: Any | None = None
        self._initialize_state()
        self.server.state.change("run_selection")(self._run_selection_changed)
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
        state.config_import_path = ""
        state.main_cif_source = "none"
        state.database_source = "builtin"
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
        state.reference_window_mode = "auto"
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
        state.history_data_datasets = []
        state.history_instrument_datasets = []
        state.history_cif_datasets = []
        state.history_archive_datasets = []
        state.history_data_id = ""
        state.history_instrument_id = ""
        state.history_main_cif_id = ""
        state.history_database_id = ""
        state.run_rows = []
        state.run_selection = []
        state.selected_run_uid = ""
        state.selected_run_name = "No run selected"
        state.selected_run_status = "-"
        state.selected_run_stage = "-"
        state.selected_run_progress = 0
        state.selected_run_message = ""
        state.selected_run_console = ""
        state.summary_cards = []
        state.viewed_run_mode = ""
        state.viewed_run_name = ""
        state.viewed_configuration = ""
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
        state.solution_headers = []
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
        state.main_cif_source_options = [
            {"title": "No supplied main phase", "value": "none"},
            {"title": "Upload/select a CIF file", "value": "upload"},
            {"title": "Reuse a CIF from Galaxy history", "value": "galaxy"},
        ]
        state.database_source_options = [
            {"title": "Built-in MP/COD catalog", "value": "builtin"},
            {"title": "Upload/select a candidate-library ZIP", "value": "upload"},
            {"title": "Reuse a library archive from Galaxy history", "value": "galaxy"},
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
                        vuetify.VSelect(label="Diffraction data", v_model=("history_data_id",), items=("history_data_datasets",), item_title="name", item_value="id", variant="outlined", no_data_text="No compatible diffraction datasets are in this history")
                        vuetify.VSelect(label="Instrument profile", v_model=("history_instrument_id",), items=("history_instrument_datasets",), item_title="name", item_value="id", variant="outlined", no_data_text="No GSAS-II instrument profiles are in this history")
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
                    html.P("The main-phase and library sources are independent of the diffraction-data source, so uploaded and saved inputs can be mixed.", classes="field-help")
                    vuetify.VSelect(label="Known/main phase", v_model=("main_cif_source",), items=("main_cif_source_options",), item_title="title", item_value="value", variant="outlined")
                    with html.Div(v_if="main_cif_source === 'upload'", classes="upload-grid"):
                        FileUpload("main_cif_path", label="Choose known/main phase CIF (optional)", extensions=[".cif"], return_contents=False, show_server_files=True, color="#0b5d46")
                    vuetify.VSelect(v_if="main_cif_source === 'galaxy'", label="Known/main phase CIF", v_model=("history_main_cif_id",), items=("history_cif_datasets",), item_title="name", item_value="id", clearable=True, variant="outlined", no_data_text="No CIF datasets are in this history")
                    vuetify.VSelect(label="Candidate library", v_model=("database_source",), items=("database_source_options",), item_title="title", item_value="value", variant="outlined")
                    with html.Div(v_if="database_source === 'upload'", classes="upload-grid"):
                        FileUpload("database_archive_path", label="Choose candidate-library ZIP (optional)", extensions=[".zip"], return_contents=False, show_server_files=True, color="#0b5d46")
                    vuetify.VSelect(v_if="database_source === 'galaxy'", label="Candidate-library archive", v_model=("history_database_id",), items=("history_archive_datasets",), item_title="name", item_value="id", clearable=True, variant="outlined", no_data_text="No library archives are in this history")
                    html.A("Build a reusable CIF library in Galaxy", href="/?tool_id=neutrons_radar_pd_library_builder_prototype&version=latest", target="_blank", classes="handoff-link")
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
                    vuetify.VSelect(v_if="reference_masks_enabled", label="Reference-mask window", v_model=("reference_window_mode",), items=("[{title:'Automatic from instrument resolution',value:'auto'},{title:'Fixed window',value:'fixed'}]",), item_title="title", item_value="value", variant="outlined")
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
                    vuetify.VAlert(v_if="main_cif_source === 'none'", text="Main-phase anchoring, main-shadow filtering, internal-coordinate cleanup, and magnetic indexing require a supplied main-phase CIF.", type="info", variant="tonal", classes="mb-3")
                    vuetify.VSwitch(v_if="main_cif_source !== 'none'", v_model=("main_prenudge",), label="Automatically anchor a supplied main-phase cell", color="#0b5d46", inset=True)
                    vuetify.VSwitch(v_if="main_cif_source !== 'none'", v_model=("main_shadow_filter",), label="Filter candidates supported only by strong main-phase peaks", color="#0b5d46", inset=True)
                    vuetify.VSwitch(v_if="main_cif_source !== 'none' && radiation === 'neutron'", v_model=("magnetic_precheck",), label="Check whether residual peaks can be indexed by a commensurate magnetic propagation vector", color="#0b5d46", inset=True)
                    with html.Div(v_if="main_cif_source !== 'none' && radiation === 'neutron' && magnetic_precheck"):
                        vuetify.VTextField(label="Magnetic precheck Q maximum", v_model=("magnetic_q_max",), type="number", variant="outlined")
                    vuetify.VSwitch(v_if="main_cif_source !== 'none'", v_model=("cleanup_enabled",), label="Refine supplied main-CIF internal parameters after lattice anchoring", color="#0b5d46", inset=True)
                    with vuetify.VRow(v_if="main_cif_source !== 'none' && cleanup_enabled"):
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
                        vuetify.VAlert(
                            v_if="full_profile !== 'custom'",
                            text="Quick, Balanced, and Thorough apply complete, reproducible search-breadth presets. Choose Custom to edit every numeric control.",
                            type="info",
                            variant="tonal",
                            classes="mb-3",
                        )
                        with vuetify.VRow(v_if="full_profile === 'custom'"):
                            with vuetify.VCol(cols=12, md=6):
                                vuetify.VTextField(label="Impurity discovery rounds", v_model=("full_max_passes",), type="number", min=1, variant="outlined")
                            with vuetify.VCol(cols=12, md=6):
                                vuetify.VTextField(label="Stop below phase fraction (wt%)", v_model=("full_min_phase_percent",), type="number", min=0, variant="outlined")
                        with vuetify.VExpansionPanels(v_if="full_profile === 'custom'", variant="accordion", classes="mt-2"):
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
                                    with vuetify.VRow():
                                        with vuetify.VCol(cols=12, md=6):
                                            vuetify.VTextField(label="Cell-length tolerance (%)", v_model=("full_cell_length_tolerance_pct",), type="number", min=0.01, variant="outlined", density="compact")
                                        with vuetify.VCol(cols=12, md=6):
                                            vuetify.VTextField(label="Cell-angle tolerance (degrees)", v_model=("full_cell_angle_tolerance_deg",), type="number", min=0.01, variant="outlined", density="compact")
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
                        with vuetify.VExpansionPanels(variant="accordion", classes="mb-3"):
                            with vuetify.VExpansionPanel(title="Import a saved RADAR-PD configuration"):
                                with vuetify.VExpansionPanelText():
                                    FileUpload("config_import_path", label="Choose configuration YAML", extensions=[".yaml", ".yml"], return_contents=False, show_server_files=True, color="#0b5d46")
                                    vuetify.VBtn("Apply configuration", click=self.apply_uploaded_configuration, variant="outlined", prepend_icon="mdi-file-import-outline", disabled=("!config_import_path",), classes="mt-2")
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
                    v_model=("run_selection", []),
                    show_select=True,
                    select_strategy="single",
                    hover=True,
                    density="comfortable",
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
                        vuetify.VBtn("Use configuration", click=self.use_selected_configuration, prepend_icon="mdi-file-restore-outline", disabled=("!selected_run_uid",))
                        vuetify.VBtn("Stop", click=self.cancel_selected_run, prepend_icon="mdi-stop-circle-outline", color="#a33131", disabled=("selected_run_status !== 'Running' && selected_run_status !== 'Queued'",))
                    with vuetify.VExpansionPanels(v_if="viewed_configuration", variant="accordion", classes="mt-4"):
                        with vuetify.VExpansionPanel(title="Saved run configuration"):
                            with vuetify.VExpansionPanelText():
                                html.Pre("{{ viewed_configuration }}", classes="config-preview")
                    with vuetify.VExpansionPanels(v_if="selected_run_console", variant="accordion", classes="mt-3"):
                        with vuetify.VExpansionPanel(title="Live analysis log"):
                            with vuetify.VExpansionPanelText():
                                html.Pre("{{ selected_run_console }}", classes="config-preview live-console")

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
                    vuetify.VTab(v_if="viewed_run_mode === 'rapid'", text="Solution Inspector", value="inspector")
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
                    vuetify.VDataTable(headers=("solution_headers",), items=("solution_rows",), density="compact", no_data_text="No rapid hypothesis table was published for this run.")
                    with html.Div(classes="handoff-actions mt-3"):
                        html.A("Open targeted RADAR-PD analysis", href="/?tool_id=neutrons_radar_pd_analyze_prototype&version=latest", target="_blank", classes="handoff-link")
                        html.A("Open GSAS-II project handoff", href="/?tool_id=neutrons_radar_pd_gpx_handoff_prototype&version=latest", target="_blank", classes="handoff-link")
                    vuetify.VAlert(text="Follow-up jobs are separate Galaxy actions, so this result remains reproducible. Select the source hypothesis or checkpoint in Galaxy when the tool opens.", type="info", variant="tonal", classes="mt-3")

    def _artifacts_page(self) -> None:
        with html.Div(classes="page-section"):
            html.H2("Artifacts and downstream handoffs")
            html.P("Download the complete archive or report, and pass an indexed GPX checkpoint to the hosted GSAS-II workflow.")
            with vuetify.VCard(classes="radar-card", variant="flat"):
                with vuetify.VCardText():
                    vuetify.VSelect(label="Artifact", v_model=("selected_artifact",), items=("artifact_options",), item_title="title", item_value="path", variant="outlined")
                    vuetify.VBtn("Download selected artifact", click=self.download_artifact, color="#0b5d46", prepend_icon="mdi-download", disabled=("!selected_artifact",))
                    html.A("Open GSAS-II project handoff", href="/?tool_id=neutrons_radar_pd_gpx_handoff_prototype&version=latest", target="_blank", classes="handoff-link ml-3")
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
        has_main_phase = (
            (state.main_cif_source == "upload" and bool(state.main_cif_path))
            or (state.main_cif_source == "galaxy" and bool(state.history_main_cif_id))
        )
        presets: dict[str, dict[str, Any]] = {
            "quick": {
                "full_max_passes": 1,
                "full_min_phase_percent": 0.75,
                "full_top_n_ml": 20,
                "full_nudge_candidates": 5,
                "full_nudge_samples": 2500,
                "full_nudge_representatives": 30,
                "full_compare_candidates": 2,
                "full_compare_cycles": 4,
            },
            "balanced": {
                "full_max_passes": 2,
                "full_min_phase_percent": 0.5,
                "full_top_n_ml": 35,
                "full_nudge_candidates": 7,
                "full_nudge_samples": 5000,
                "full_nudge_representatives": 50,
                "full_compare_candidates": 2,
                "full_compare_cycles": 6,
            },
            "thorough": {
                "full_max_passes": 3,
                "full_min_phase_percent": 0.25,
                "full_top_n_ml": 70,
                "full_nudge_candidates": 14,
                "full_nudge_samples": 10000,
                "full_nudge_representatives": 75,
                "full_compare_candidates": 4,
                "full_compare_cycles": 8,
            },
        }
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
            "reference_window_mode": state.reference_window_mode,
            "include_cu_kbeta": bool(state.include_cu_kbeta) if state.radiation == "xray" else False,
            "background_mode": state.background_mode,
            "background_type": state.background_type,
            "background_terms": int(state.background_terms),
            "main_prenudge": bool(state.main_prenudge) if has_main_phase else False,
            "main_shadow_filter": bool(state.main_shadow_filter) if has_main_phase else False,
            "cleanup_enabled": bool(state.cleanup_enabled) if has_main_phase else False,
            "refine_u_iso": bool(state.refine_u_iso) if has_main_phase and state.cleanup_enabled else False,
            "refine_positions": bool(state.refine_positions) if has_main_phase and state.cleanup_enabled else False,
            "magnetic_precheck": bool(state.magnetic_precheck) if has_main_phase and state.radiation == "neutron" else False,
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
        if state.analysis_mode == "full" and state.full_profile in presets:
            values.update(presets[state.full_profile])
        if run_name:
            values["run_name"] = run_name
        return AnalysisConfig(**values)

    def _inputs(self) -> InputSelection:
        state = self.server.state
        source = InputSource(state.input_source)
        if state.radiation == "xray" and source in {InputSource.IPTS_EVENT, InputSource.IPTS_MANUAL}:
            raise ValueError("SNS IPTS resolution is available only for neutron data")
        main_cif_path = state.main_cif_path or None if state.main_cif_source == "upload" else None
        main_cif_dataset_id = state.history_main_cif_id or None if state.main_cif_source == "galaxy" else None
        database_archive_path = state.database_archive_path or None if state.database_source == "upload" else None
        database_dataset_id = state.history_database_id or None if state.database_source == "galaxy" else None
        return InputSelection(
            source=source,
            data_path=(state.data_path or None) if source == InputSource.UPLOAD else None,
            data_dataset_id=(state.history_data_id or None) if source == InputSource.GALAXY else None,
            instrument_path=(state.instrument_path or None) if source == InputSource.UPLOAD else None,
            instrument_dataset_id=(state.history_instrument_id or None) if source == InputSource.GALAXY else None,
            main_cif_path=main_cif_path,
            main_cif_dataset_id=main_cif_dataset_id,
            database_archive_path=database_archive_path,
            database_dataset_id=database_dataset_id,
            use_builtin_cuka=bool(state.use_builtin_cuka) if state.radiation == "xray" else False,
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
            self._start_monitor(record)
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

    def _start_monitor(self, record: RunRecord) -> None:
        if record.uid in self._monitored_uids or record.status not in {RunStatus.NEW, RunStatus.UPLOADING, RunStatus.QUEUED, RunStatus.RUNNING}:
            return
        self._monitored_uids.add(record.uid)

        def monitor() -> None:
            try:
                self.service.monitor(record, self._monitor_update)
            finally:
                self._monitored_uids.discard(record.uid)

        threading.Thread(target=monitor, daemon=True, name=f"radar-monitor-{record.uid[:8]}").start()

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
                if current is None:
                    record = recovered
                else:
                    record = current.model_copy(
                        update={
                            "name": recovered.name,
                            "mode": recovered.mode,
                            "status": recovered.status,
                            "stage": recovered.stage,
                            "progress": recovered.progress,
                            "updated_utc": recovered.updated_utc,
                            "message": recovered.message or current.message,
                            "console_tail": recovered.console_tail or current.console_tail,
                            "config": getattr(recovered, "config", None) or getattr(current, "config", None),
                            "inputs": getattr(recovered, "inputs", None) or getattr(current, "inputs", None),
                        }
                    )
                try:
                    record = self.service.refresh(record)
                except Exception as exc:
                    record.message = f"Recovered from Galaxy; live refresh will retry: {exc}"
                self.records[record.uid] = record
                self._start_monitor(record)
            self._sync_runs()
        except Exception as exc:
            state.connection_ok = False
            state.connection_status = "NDIP connection unavailable"
            state.error_message = str(exc)
        state.flush()

    def refresh_history(self, **_: Any) -> None:
        state = self.server.state
        try:
            datasets = self.service.list_history_datasets()
            state.history_datasets = datasets
            data_extensions = {"dat", "xye", "xy", "csv", "txt", "fxye", "xrdml", "xml", "tabular"}
            instrument_extensions = {"instprm", "prm", "inst", "ins"}
            cif_extensions = {"cif"}
            archive_extensions = {"zip"}

            def compatible(extensions: set[str]) -> list[dict[str, str]]:
                result: list[dict[str, str]] = []
                for item in datasets:
                    extension = str(item.get("extension") or "").lower().lstrip(".")
                    name = str(item.get("name") or "").lower()
                    if extension in extensions or any(name.endswith(f".{suffix}") for suffix in extensions):
                        result.append(item)
                return result

            state.history_data_datasets = compatible(data_extensions)
            state.history_instrument_datasets = compatible(instrument_extensions)
            state.history_cif_datasets = compatible(cif_extensions)
            state.history_archive_datasets = compatible(archive_extensions)
            state.notice = f"Loaded {len(state.history_datasets)} reusable datasets from this history."
        except Exception as exc:
            state.error_message = f"Could not load Galaxy datasets: {exc}"
        state.flush()

    def _run_selection_changed(self, run_selection: Any = None, **__: Any) -> None:
        uid = selected_run_uid(run_selection)
        if uid and uid in self.records:
            self._select_record(self.records[uid])
            self.server.state.run_selection = [uid]
            self.server.state.flush()

    def _select_record(self, record: RunRecord) -> None:
        state = self.server.state
        state.selected_run_uid = record.uid
        state.run_selection = [record.uid]
        state.selected_run_name = record.name
        state.selected_run_status = record.status.value.title()
        state.selected_run_stage = record.stage
        state.selected_run_progress = record.progress
        state.selected_run_message = record.message
        state.selected_run_console = record.console_tail
        saved_config = getattr(record, "config", None)
        configuration = saved_config.portable_contract() if isinstance(saved_config, AnalysisConfig) else (getattr(record, "configuration", {}) or {})
        state.viewed_configuration = json.dumps(configuration, indent=2, sort_keys=False) if configuration else ""

    def use_selected_configuration(self, **_: Any) -> None:
        uid = self.server.state.selected_run_uid
        record = self.records.get(uid)
        if record is None:
            return
        try:
            saved_config = getattr(record, "config", None)
            configuration = saved_config.portable_contract() if isinstance(saved_config, AnalysisConfig) else (getattr(record, "configuration", {}) or {})
            if not configuration and record.local_output_dir:
                path = record.local_output_dir / "resolved_config.yaml"
                if path.is_file():
                    import yaml

                    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
                    configuration = loaded if isinstance(loaded, dict) else {}
            if not configuration:
                raise ValueError("This recovered run does not expose a reusable configuration yet")
            config = saved_config if isinstance(saved_config, AnalysisConfig) else config_from_contract(configuration)
            saved_inputs = getattr(record, "inputs", None)
            selection = saved_inputs.model_dump(mode="json") if isinstance(saved_inputs, InputSelection) else (getattr(record, "input_selection", {}) or {})
            self._apply_configuration(config, selection)
            state = self.server.state
            for key in ("data_dataset_id", "instrument_dataset_id", "main_cif_dataset_id", "database_dataset_id"):
                dataset_id = selection.get(key)
                if not dataset_id:
                    continue
                state_name = {
                    "data_dataset_id": "history_data_id",
                    "instrument_dataset_id": "history_instrument_id",
                    "main_cif_dataset_id": "history_main_cif_id",
                    "database_dataset_id": "history_database_id",
                }[key]
                setattr(state, state_name, dataset_id)
            if selection.get("data_dataset_id"):
                state.input_source = "galaxy"
            if selection.get("main_cif_dataset_id"):
                state.main_cif_source = "galaxy"
            if selection.get("database_dataset_id"):
                state.database_source = "galaxy"
            state.run_name = ""
            state.notice = f"Loaded the scientific configuration from {record.name}. Choose or replace any inputs, then submit a new run."
            state.active_page = "setup"
        except Exception as exc:
            self.server.state.error_message = str(exc)
        self.server.state.flush()

    def apply_uploaded_configuration(self, **_: Any) -> None:
        try:
            path = str(self.server.state.config_import_path or "")
            if not path:
                raise ValueError("Choose a RADAR-PD configuration YAML first")
            config = load_configuration(path)
            self._apply_configuration(config)
            self.server.state.notice = "Loaded the saved scientific configuration. Choose inputs, review the setup, and submit a new run."
            self.server.state.active_page = "setup"
        except Exception as exc:
            self.server.state.error_message = str(exc)
        self.server.state.flush()

    def _apply_configuration(self, config: AnalysisConfig, selection: dict[str, Any] | None = None) -> None:
        state = self.server.state
        for field in AnalysisConfig.model_fields:
            if field == "run_name":
                continue
            value = getattr(config, field)
            state_field = field
            if field in {"sample_elements", "environment_elements"}:
                value = ", ".join(value)
            elif field == "exclude_regions":
                value = "\n".join(f"{start}, {end}" for start, end in value)
                state_field = "ignore_regions"
            elif field == "limits":
                state.fit_start = value[0] if value else None
                state.fit_end = value[1] if value else None
                continue
            setattr(state, state_field, value.value if hasattr(value, "value") else value)
        selection = selection or {}
        state.main_cif_source = "none"
        state.database_source = "builtin"
        state.use_builtin_cuka = bool(selection.get("use_builtin_cuka", False)) if config.radiation.value == "xray" else False
        state.input_source = str(selection.get("source") or state.input_source)
        for path_key, state_name in (
            ("data_path", "data_path"),
            ("instrument_path", "instrument_path"),
            ("main_cif_path", "main_cif_path"),
            ("database_archive_path", "database_archive_path"),
        ):
            if selection.get(path_key):
                setattr(state, state_name, selection[path_key])
        if selection.get("main_cif_path"):
            state.main_cif_source = "upload"
        if selection.get("database_archive_path"):
            state.database_source = "upload"
        state.run_name = ""

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
        self._reset_result_state()
        payload = self.service.result_payload(record)
        summary = payload.get("summary") or {}
        state.viewed_run_mode = record.mode.value
        state.viewed_run_name = record.name
        state.phase_rows = phase_fraction_rows(summary)
        total_seconds = total_elapsed_seconds(summary)
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
                "path": item.get("collection_path") or item.get("path") or item.get("source_path") or "-",
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
            keys = list(state.solution_rows[0].keys()) if state.solution_rows else []
            state.solution_headers = [{"title": key.replace("_", " ").title(), "key": key} for key in keys]
        state.flush()

    def _reset_result_state(self) -> None:
        state = self.server.state
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
        state.solution_headers = []
        state.result_tab = "overview"
        if self._plot_widget is not None:
            self._plot_widget.update(figure_for_payload({}))

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
        mime = {
            ".html": "text/html",
            ".json": "application/json",
            ".csv": "text/csv",
            ".txt": "text/plain",
            ".yaml": "application/yaml",
            ".yml": "application/yaml",
            ".zip": "application/zip",
            ".gpx": "application/octet-stream",
        }.get(path.suffix.lower(), "application/octet-stream")
        self.download_file(path.name, mime, path.read_bytes())

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
        .config-preview { max-height: 360px; overflow: auto; padding: 14px; background: #f5f8f7; border: 1px solid #d7e3de; border-radius: 5px; font-size: 12px; white-space: pre-wrap; }
        .live-console { max-height: 440px; background: #17251f; color: #e7f2ed; border-color: #29483c; font-family: Consolas, "Courier New", monospace; }
        .handoff-actions { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
        .handoff-link { display: inline-flex; align-items: center; min-height: 38px; padding: 8px 14px; color: #0b5d46; border: 1px solid #0b5d46; border-radius: 5px; font-weight: 700; text-decoration: none; }
        .handoff-link:hover { background: #e8f5ef; }
        .result-panel { background: #fff; border: 1px solid #d7e3de; border-radius: 7px; padding: 18px; margin-top: 10px; }
        @media (max-width: 800px) { .radar-shell { padding: 4px 12px 32px; } .review-summary { grid-template-columns: 1fr; } .sticky-review { position: static; } }
        """
