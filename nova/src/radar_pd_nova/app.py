"""RADAR-PD interactive NOVA/Trame application."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nova.trame import ThemedApp
from trame.app import get_server
from trame.widgets import client, html, plotly, vuetify3 as vuetify

from .configuration import config_from_contract, load_configuration
from .galaxy_service import GalaxyService
from .models import AnalysisConfig, AnalysisMode, InputSelection, InputSource, RunRecord, RunStatus, selected_run_uid
from .results import build_result_view, figure_for_payload, read_plot_payload, read_table
from .uploads import NamedFileUpload


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
        self._monitor_tasks: set[asyncio.Task[None]] = set()
        self._opening_run_uid: str | None = None
        self._plot_widget: Any | None = None
        self._primary_plot_widget: Any | None = None
        self._auto_opened_uids: set[str] = set()
        self._initialize_state()
        self.server.state.change("run_selection")(self._run_selection_changed)
        self.create_ui()
        self.server.controller.on_server_ready.add(self._recover_runs)

    def _initialize_state(self) -> None:
        state = self.server.state
        state.trame__title = "RADAR-PD Interactive"
        state.active_page = "setup"
        state.setup_collapsed = False
        state.setup_panels = ["measurement", "data"]
        state.workspace_view = "monitor"
        state.workspace_options = [
            {"title": "Run Monitor", "value": "monitor", "icon": "mdi-progress-clock"},
            {"title": "Rapid Results", "value": "results", "icon": "mdi-chart-line"},
            {"title": "Run File Browser", "value": "files", "icon": "mdi-folder-outline"},
        ]
        state.run_search = ""
        state.cancel_dialog = False
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
        state.selected_run_loading = False
        state.selected_run_elapsed = "-"
        state.monitor_stages = []
        state.summary_cards = []
        state.result_metrics = []
        state.result_warnings = []
        state.phase_total = "-"
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
        state.gallery_selected_plot = ""
        state.primary_plot_path = ""
        state.plot_groups = []
        state.artifact_options = []
        state.selected_artifact = ""
        state.file_groups = []
        state.gpx_rows = []
        state.checkpoint_rows = []
        state.selected_checkpoint = ""
        state.selected_hypothesis = None
        state.comparison_hypothesis = None
        state.solution_rows = []
        state.solution_headers = []
        state.rapid_stage = "final_refinement"
        state.rapid_coarse_rows = []
        state.rapid_nudge_rows = []
        state.rapid_pattern_rows = []
        state.rapid_final_rows = []
        state.top_refinements = []
        state.full_progression = []
        state.full_model_rows = []
        state.last_log = ""
        state.result_tab = "overview"
        state.mode_options = [
            {"title": "Rapid Hypothesis Mode", "value": "rapid"},
            {"title": "Full RADAR-PD", "value": "full"},
        ]
        state.source_options = [
            {"title": "Computer", "value": "upload"},
            {"title": "Galaxy History", "value": "galaxy"},
            {"title": "SNS IPTS / NeXus event file", "value": "ipts_event"},
            {"title": "SNS IPTS / run lookup", "value": "ipts_manual"},
        ]
        state.radiation_options = [
            {"title": "Neutron powder diffraction", "value": "neutron"},
            {"title": "X-ray powder diffraction", "value": "xray"},
        ]
        state.main_cif_source_options = [
            {"title": "No supplied main phase", "value": "none"},
            {"title": "Choose CIF from computer", "value": "upload"},
            {"title": "Choose CIF from Galaxy History", "value": "galaxy"},
        ]
        state.database_source_options = [
            {"title": "Built-in MP/COD catalog", "value": "builtin"},
            {"title": "Candidate-library ZIP from computer", "value": "upload"},
            {"title": "Library archive from Galaxy History", "value": "galaxy"},
        ]

    def create_ui(self) -> None:
        self.set_theme("CompactTheme")
        with super().create_ui() as layout:
            layout.toolbar_title.set_text("RADAR-PD Interactive")
            # Vue deliberately ignores <style> elements embedded in a component
            # template. Client.Style registers the sheet in the document head,
            # which keeps the two-pane layout intact after reactive updates.
            client.Style(self._css())
            with layout.content:
                with html.Div(classes="radar-app-shell"):
                    self._header()
                    with html.Div(
                        classes=("setup_collapsed ? 'radar-layout is-collapsed' : 'radar-layout'",),
                    ):
                        with html.Aside(classes="radar-setup-rail", v_show="!setup_collapsed"):
                            self._setup_page()
                        with html.Main(classes="radar-workspace"):
                            self._workspace_page()
            return layout

    def _header(self) -> None:
        with html.Header(classes="radar-context-header"):
            vuetify.VBtn(
                icon=("setup_collapsed ? 'mdi-menu-open' : 'mdi-menu'",),
                title=("setup_collapsed ? 'Show setup' : 'Hide setup'",),
                variant="text",
                size="small",
                click="setup_collapsed = !setup_collapsed",
            )
            with html.Div(classes="radar-context-title"):
                html.Div("RADAR-PD Interactive", classes="radar-product-name")
                html.Div("Phase detection for powder diffraction", classes="radar-product-subtitle")
            vuetify.VSpacer()
            vuetify.VChip(
                text=("connection_status",),
                color=("connection_ok ? '#dff2e8' : '#fff0d4'",),
                prepend_icon=("connection_ok ? 'mdi-cloud-check-outline' : 'mdi-cloud-alert-outline'",),
                variant="flat",
                size="small",
            )
            vuetify.VChip(
                v_if="!!selected_run_uid",
                text=("selected_run_name",),
                prepend_icon="mdi-flask-outline",
                variant="outlined",
                size="small",
                classes="radar-run-chip",
            )
            vuetify.VChip(
                v_if="!!selected_run_uid",
                text=("selected_run_status",),
                color=("selected_run_status === 'Ok' ? '#dff2e8' : selected_run_status === 'Error' ? '#fde7e7' : '#fff0d4'",),
                variant="flat",
                size="small",
            )

    @contextmanager
    def _setup_section(self, number: int, title: str, value: str, status_expression: str) -> Any:
        with vuetify.VExpansionPanel(value=value, key=f"'setup-{value}'", classes="radar-setup-panel"):
            with vuetify.VExpansionPanelTitle(classes="radar-setup-title"):
                html.Span(str(number), classes="radar-step-number")
                html.Span(title, classes="radar-step-label")
                vuetify.VSpacer()
                vuetify.VIcon(
                    icon=(f"{status_expression} ? 'mdi-check-circle' : 'mdi-circle-outline'",),
                    color=(f"{status_expression} ? '#1f6b4b' : '#91a099'",),
                    size="small",
                )
            # Stateful controls must survive panel collapse. Vuetify otherwise
            # lazily unmounts expansion content, which recreates file inputs
            # and was the root of the sibling-reset/duplicate-upload defect.
            with vuetify.VExpansionPanelText(eager=True):
                yield

    def _setup_page(self) -> None:
        html.Div("SETUP", classes="radar-rail-kicker")
        html.H2("Configure analysis", classes="radar-rail-heading")
        html.P("Work from top to bottom. Earlier choices control the options that follow.", classes="radar-rail-help")

        with vuetify.VExpansionPanels(multiple=True, variant="accordion", classes="radar-history-panel mb-3"):
            with vuetify.VExpansionPanel(value="history"):
                with vuetify.VExpansionPanelTitle():
                    vuetify.VIcon("mdi-history", size="small", classes="mr-2")
                    html.Span("Open Previous Run")
                with vuetify.VExpansionPanelText():
                    vuetify.VTextField(
                        v_model=("run_search",),
                        label="Search runs",
                        prepend_inner_icon="mdi-magnify",
                        density="compact",
                        variant="outlined",
                        hide_details=True,
                        clearable=True,
                    )
                    with html.Div(classes="radar-run-list"):
                        with vuetify.VListItem(
                            v_for="run in run_rows.filter(item => !run_search || item.name.toLowerCase().includes(run_search.toLowerCase()))",
                            key="run.uid",
                            click=(self._run_selection_changed, "[run.uid]"),
                            classes="radar-run-list-item",
                        ):
                            html.Div("{{ run.name }}", classes="radar-run-list-name")
                            html.Div("{{ run.mode }} / {{ run.status }}", classes="radar-run-list-meta")
                            html.Div("{{ run.stage }}", classes="radar-run-list-stage")
                        html.Div("No RADAR-PD runs are in this Galaxy history.", v_if="!run_rows.length", classes="radar-empty-compact")
                    vuetify.VBtn(
                        "Refresh from Galaxy",
                        click=self._recover_runs,
                        prepend_icon="mdi-refresh",
                        variant="text",
                        size="small",
                        block=True,
                        classes="mt-2",
                    )

        with vuetify.VExpansionPanels(
            v_model=("setup_panels",),
            multiple=True,
            variant="accordion",
            classes="radar-setup-panels",
        ):
            with self._setup_section(1, "Measurement Type", "measurement", "!!radiation && !!instrument_mode"):
                vuetify.VSelect(
                    label="Radiation source",
                    v_model=("radiation",),
                    items=("radiation_options",),
                    item_title="title",
                    item_value="value",
                    density="compact",
                    variant="outlined",
                )
                vuetify.VSelect(
                    label="Pattern geometry",
                    v_model=("instrument_mode",),
                    items=("[{title:'Auto detect',value:'auto'},{title:'Constant wavelength',value:'cw'},{title:'Time of flight',value:'tof'}]",),
                    item_title="title",
                    item_value="value",
                    density="compact",
                    variant="outlined",
                    hint="Auto detect is recommended unless the instrument requires an override.",
                    persistent_hint=True,
                )
            with self._setup_section(
                2,
                "Candidate Library",
                "library",
                "database_source === 'builtin' || (database_source === 'upload' && !!database_archive_path) || (database_source === 'galaxy' && !!history_database_id)",
            ):
                vuetify.VSelect(
                    label="Candidate library",
                    v_model=("database_source",),
                    items=("database_source_options",),
                    item_title="title",
                    item_value="value",
                    density="compact",
                    variant="outlined",
                )
                with html.Div(v_show="database_source === 'upload'", key="'radar-library-upload-panel'"):
                    NamedFileUpload(
                        "database_archive_path",
                        label="Candidate-library ZIP",
                        help_text="A reusable RADAR-PD CIF library archive (.zip)",
                        extensions=[".zip"],
                        optional=True,
                        key="radar-database-upload",
                    )
                vuetify.VSelect(
                    v_show="database_source === 'galaxy'",
                    label="Galaxy library archive",
                    v_model=("history_database_id",),
                    items=("history_archive_datasets",),
                    item_title="name",
                    item_value="id",
                    density="compact",
                    variant="outlined",
                    no_data_text="No library archives are in this history",
                )
                html.A(
                    "Build a reusable CIF library",
                    href="/?tool_id=neutrons_radar_pd_library_builder_prototype&version=latest",
                    target="_blank",
                    classes="radar-secondary-link",
                )
            data_ready = "(input_source === 'upload' && !!data_path && (!!instrument_path || (radiation === 'xray' && use_builtin_cuka))) || (input_source === 'galaxy' && !!history_data_id && (!!history_instrument_id || (radiation === 'xray' && use_builtin_cuka))) || (input_source === 'ipts_event' && !!event_file_path && !!bank) || (input_source === 'ipts_manual' && !!ipts_instrument && !!ipts && !!run_number && !!bank)"
            with self._setup_section(3, "Data Collection", "data", data_ready):
                vuetify.VSelect(
                    label="Data source",
                    v_model=("input_source",),
                    items=("source_options",),
                    item_title="title",
                    item_value="value",
                    density="compact",
                    variant="outlined",
                )
                with html.Div(v_show="input_source === 'upload'", key="'radar-computer-inputs'"):
                    NamedFileUpload(
                        "data_path",
                        label="Diffraction data",
                        help_text="Required measurement pattern",
                        extensions=[".dat", ".xye", ".xy", ".csv", ".txt", ".fxye", ".xrdml", ".xml"],
                        key="radar-diffraction-upload",
                    )
                    NamedFileUpload(
                        "instrument_path",
                        label="GSAS-II instrument profile",
                        help_text="Required unless the built-in X-ray profile is used",
                        extensions=[".instprm", ".prm", ".inst", ".ins"],
                        key="radar-instrument-upload",
                    )
                with html.Div(v_show="input_source === 'galaxy'", key="'radar-history-inputs'"):
                    vuetify.VSelect(
                        label="Diffraction data from History",
                        v_model=("history_data_id",),
                        items=("history_data_datasets",),
                        item_title="name",
                        item_value="id",
                        density="compact",
                        variant="outlined",
                        no_data_text="No compatible diffraction datasets",
                        key="radar-history-diffraction",
                    )
                    vuetify.VSelect(
                        label="Instrument profile from History",
                        v_model=("history_instrument_id",),
                        items=("history_instrument_datasets",),
                        item_title="name",
                        item_value="id",
                        density="compact",
                        variant="outlined",
                        no_data_text="No GSAS-II instrument profiles",
                        key="radar-history-instrument",
                    )
                vuetify.VSwitch(
                    v_show="radiation === 'xray' && instrument_mode !== 'tof' && (input_source === 'upload' || input_source === 'galaxy')",
                    v_model=("use_builtin_cuka",),
                    label="Use built-in Cu K-alpha profile",
                    color="#15543c",
                    density="compact",
                    inset=True,
                )
                with html.Div(v_show="input_source === 'ipts_event'", key="'radar-event-inputs'"):
                    NamedFileUpload(
                        "event_file_path",
                        label="NeXus event file",
                        extensions=[".nxs", ".h5", ".hdf5"],
                        key="radar-event-upload",
                    )
                    vuetify.VTextField(label="Detector bank", v_model=("bank",), density="compact", variant="outlined", placeholder="bank1")
                with html.Div(v_show="input_source === 'ipts_manual'", key="'radar-ipts-inputs'"):
                    vuetify.VTextField(label="SNS instrument", v_model=("ipts_instrument",), density="compact", variant="outlined", placeholder="HB2A")
                    vuetify.VTextField(label="IPTS", v_model=("ipts",), density="compact", variant="outlined", placeholder="IPTS-12345")
                    with html.Div(classes="radar-field-pair"):
                        vuetify.VTextField(label="Run number", v_model=("run_number",), type="number", density="compact", variant="outlined")
                        vuetify.VTextField(label="Detector bank", v_model=("bank",), density="compact", variant="outlined")
                vuetify.VDivider(classes="my-3")
                vuetify.VSelect(
                    label="Known/main phase",
                    v_model=("main_cif_source",),
                    items=("main_cif_source_options",),
                    item_title="title",
                    item_value="value",
                    density="compact",
                    variant="outlined",
                    hint="Optional and independent of the diffraction-data source.",
                    persistent_hint=True,
                )
                with html.Div(v_show="main_cif_source === 'upload'", key="'radar-main-cif-upload-panel'"):
                    NamedFileUpload(
                        "main_cif_path",
                        label="Known/main-phase CIF",
                        help_text="Optional crystallographic model (.cif)",
                        extensions=[".cif"],
                        optional=True,
                        key="radar-main-cif-upload",
                    )
                vuetify.VSelect(
                    v_show="main_cif_source === 'galaxy'",
                    label="Main-phase CIF from History",
                    v_model=("history_main_cif_id",),
                    items=("history_cif_datasets",),
                    item_title="name",
                    item_value="id",
                    clearable=True,
                    density="compact",
                    variant="outlined",
                    no_data_text="No CIF datasets are in this history",
                    key="'radar-history-main-cif'",
                )
            with self._setup_section(4, "Chemistry Policy", "chemistry", "!!sample_elements"):
                vuetify.VTextField(
                    label="Sample elements",
                    v_model=("sample_elements",),
                    placeholder="Tb, Be, Ge, O",
                    density="compact",
                    variant="outlined",
                    hint="Comma- or space-separated symbols",
                    persistent_hint=True,
                )
                vuetify.VTextField(
                    label="Sample can / environment",
                    v_model=("environment_elements",),
                    placeholder="Al, V",
                    density="compact",
                    variant="outlined",
                    hint="Allowed as environment phases, not mixed freely into sample chemistry",
                    persistent_hint=True,
                )
            with self._setup_section(5, "Pattern Regions", "pattern", "(!fit_start && !fit_end) || (!!fit_start && !!fit_end)"):
                with html.Div(classes="radar-field-pair"):
                    vuetify.VTextField(label="Fit start", v_model=("fit_start",), type="number", density="compact", variant="outlined", clearable=True)
                    vuetify.VTextField(label="Fit end", v_model=("fit_end",), type="number", density="compact", variant="outlined", clearable=True)
                vuetify.VTextarea(
                    label="Ignored regions",
                    v_model=("ignore_regions",),
                    placeholder="One start,end pair per line\n2.0, 3.2",
                    density="compact",
                    variant="outlined",
                    rows=2,
                    auto_grow=True,
                )
            with self._setup_section(6, "Background Correction", "background", "!!background_mode && !!background_type && background_terms > 0"):
                vuetify.VSelect(
                    label="Background correction",
                    v_model=("background_mode",),
                    items=("[{title:'Automatic fixed points',value:'auto_fixed_points'},{title:'Manual / pipeline default',value:'manual'}]",),
                    item_title="title",
                    item_value="value",
                    density="compact",
                    variant="outlined",
                )
                with html.Div(classes="radar-field-pair"):
                    vuetify.VSelect(label="Function", v_model=("background_type",), items=("['chebyschev-1','chebyschev','cosine','Q^2 power series']",), density="compact", variant="outlined")
                    vuetify.VTextField(label="Terms", v_model=("background_terms",), type="number", min=1, max=36, density="compact", variant="outlined")
            with self._setup_section(7, "Magnetic Ordering Precheck", "magnetic", "true"):
                vuetify.VAlert(
                    v_show="main_cif_source === 'none' || radiation !== 'neutron'",
                    text="Available when a neutron run includes a known main-phase CIF.",
                    type="info",
                    variant="tonal",
                    density="compact",
                )
                vuetify.VSwitch(
                    v_show="main_cif_source !== 'none' && radiation === 'neutron'",
                    v_model=("magnetic_precheck",),
                    label="Check residual peaks for commensurate magnetic indexing",
                    color="#15543c",
                    density="compact",
                    inset=True,
                )
                vuetify.VTextField(
                    v_show="main_cif_source !== 'none' && radiation === 'neutron' && magnetic_precheck",
                    label="Q maximum",
                    v_model=("magnetic_q_max",),
                    type="number",
                    density="compact",
                    variant="outlined",
                )
            with self._setup_section(8, "Analysis Mode", "mode", "!!analysis_mode"):
                vuetify.VSelect(
                    label="Analysis path",
                    v_model=("analysis_mode",),
                    items=("mode_options",),
                    item_title="title",
                    item_value="value",
                    density="compact",
                    variant="outlined",
                )
                vuetify.VAlert(
                    text=("analysis_mode === 'rapid' ? 'Fast staged hypothesis search followed by focused refinements.' : 'Rigorous residual-aware multi-pass phase discovery and refinement.'",),
                    type="info",
                    variant="tonal",
                    density="compact",
                )
            with self._setup_section(9, "Runtime Budget", "budget", "true"):
                with html.Div(v_show="analysis_mode === 'rapid'"):
                    with html.Div(classes="radar-field-pair"):
                        vuetify.VTextField(label="Phases / hypothesis", v_model=("rapid_phases_per_hypothesis",), type="number", min=1, max=5, density="compact", variant="outlined")
                        vuetify.VTextField(label="Retained / stage", v_model=("rapid_stage_output_limit",), type="number", min=3, max=50, density="compact", variant="outlined")
                        vuetify.VTextField(label="Final refinements", v_model=("rapid_gsas_validation_limit",), type="number", min=0, density="compact", variant="outlined")
                        vuetify.VTextField(label="Parallel workers", v_model=("rapid_parallel_workers",), type="number", min=1, max=16, density="compact", variant="outlined")
                with html.Div(v_show="analysis_mode === 'full'"):
                    vuetify.VSelect(
                        label="Search profile",
                        v_model=("full_profile",),
                        items=("[{title:'Quick',value:'quick'},{title:'Balanced',value:'balanced'},{title:'Thorough',value:'thorough'},{title:'Custom',value:'custom'}]",),
                        item_title="title",
                        item_value="value",
                        density="compact",
                        variant="outlined",
                    )
                    vuetify.VAlert(v_show="full_profile !== 'custom'", text="The profile applies a complete reproducible search budget.", type="info", variant="tonal", density="compact")
            with self._setup_section(10, "Expert Tuning", "expert", "true"):
                vuetify.VSwitch(v_model=("reference_masks_enabled",), label="Mask reference/can peaks", color="#15543c", density="compact", inset=True)
                vuetify.VSelect(v_show="reference_masks_enabled", label="Reference structures", v_model=("reference_mask_presets",), items=("['Al_fcc','Cu_fcc','V_bcc']",), multiple=True, chips=True, density="compact", variant="outlined")
                vuetify.VSelect(v_show="reference_masks_enabled", label="Reference-mask window", v_model=("reference_window_mode",), items=("[{title:'Automatic from resolution',value:'auto'},{title:'Fixed window',value:'fixed'}]",), item_title="title", item_value="value", density="compact", variant="outlined")
                vuetify.VSwitch(v_show="radiation === 'xray' && reference_masks_enabled", v_model=("include_cu_kbeta",), label="Mask Cu K-beta companions", color="#15543c", density="compact", inset=True)
                vuetify.VDivider(classes="my-3")
                vuetify.VAlert(v_show="main_cif_source === 'none'", text="Main-phase safeguards become available when a CIF is supplied.", type="info", variant="tonal", density="compact")
                vuetify.VSwitch(v_show="main_cif_source !== 'none'", v_model=("main_prenudge",), label="Anchor supplied main-phase cell", color="#15543c", density="compact", inset=True)
                vuetify.VSwitch(v_show="main_cif_source !== 'none'", v_model=("main_shadow_filter",), label="Filter main-shadow-only candidates", color="#15543c", density="compact", inset=True)
                vuetify.VSwitch(v_show="main_cif_source !== 'none'", v_model=("cleanup_enabled",), label="Clean up main-CIF internal parameters", color="#15543c", density="compact", inset=True)
                vuetify.VCheckbox(v_show="main_cif_source !== 'none' && cleanup_enabled", v_model=("refine_u_iso",), label="Refine isotropic displacement parameters", density="compact")
                vuetify.VCheckbox(v_show="main_cif_source !== 'none' && cleanup_enabled", v_model=("refine_positions",), label="Refine atomic positions", density="compact")
                with html.Div(v_show="analysis_mode === 'rapid'"):
                    vuetify.VDivider(classes="my-3")
                    vuetify.VSwitch(v_model=("rapid_show_family_variants",), label="Keep family variants in Solution Inspector", color="#15543c", density="compact", inset=True)
                    vuetify.VSwitch(v_model=("rapid_final_polish_enabled",), label="Run final polish after ranking", color="#15543c", density="compact", inset=True)
                with html.Div(v_show="analysis_mode === 'full' && full_profile === 'custom'"):
                    vuetify.VDivider(classes="my-3")
                    with html.Div(classes="radar-field-pair"):
                        for label, model in (
                            ("Discovery rounds", "full_max_passes"),
                            ("Minimum phase wt%", "full_min_phase_percent"),
                            ("ML candidates", "full_top_n_ml"),
                            ("Nudge candidates", "full_nudge_candidates"),
                            ("Nudge samples", "full_nudge_samples"),
                            ("Nudge representatives", "full_nudge_representatives"),
                            ("Comparison candidates", "full_compare_candidates"),
                            ("Comparison cycles", "full_compare_cycles"),
                            ("Cell length tolerance %", "full_cell_length_tolerance_pct"),
                            ("Cell angle tolerance (deg)", "full_cell_angle_tolerance_deg"),
                        ):
                            vuetify.VTextField(label=label, v_model=(model,), type="number", min=0, density="compact", variant="outlined")
            ready_expression = f"connection_ok && !!sample_elements && ({data_ready}) && (database_source === 'builtin' || (database_source === 'upload' && !!database_archive_path) || (database_source === 'galaxy' && !!history_database_id))"
            with self._setup_section(11, "Review Run Plan", "review", ready_expression):
                vuetify.VTextField(label="Run name", v_model=("run_name",), density="compact", variant="outlined", placeholder="Generated automatically if blank")
                with html.Div(classes="radar-checklist"):
                    for label, expression in (
                        ("Connected to NDIP", "connection_ok"),
                        ("Measurement selected", "!!radiation && !!instrument_mode"),
                        ("Data and instrument ready", data_ready),
                        ("Sample chemistry entered", "!!sample_elements"),
                        ("Candidate library ready", "database_source === 'builtin' || !!database_archive_path || !!history_database_id"),
                    ):
                        with html.Div(classes="radar-check-row"):
                            vuetify.VIcon(icon=(f"{expression} ? 'mdi-check-circle' : 'mdi-alert-circle-outline'",), color=(f"{expression} ? '#1f6b4b' : '#a66a00'",), size="small")
                            html.Span(label)
                with html.Div(classes="radar-review-summary"):
                    html.Div("{{ analysis_mode === 'rapid' ? 'Rapid Hypothesis Mode' : 'Full RADAR-PD' }}", classes="radar-review-primary")
                    html.Div("{{ radiation === 'neutron' ? 'Neutron' : 'X-ray' }} / {{ instrument_mode.toUpperCase() }}", classes="radar-review-secondary")
                    html.Div("Sample: {{ sample_elements || 'not entered' }}", classes="radar-review-secondary")
                    html.Div("Main phase: {{ main_cif_source === 'none' ? 'not supplied' : 'supplied' }}", classes="radar-review-secondary")
                vuetify.VAlert(v_show="!!error_message", text=("error_message",), type="error", variant="tonal", density="compact", classes="mb-2")
                vuetify.VAlert(v_show="!!notice", text=("notice",), type="success", variant="tonal", density="compact", classes="mb-2")
                with vuetify.VExpansionPanels(variant="accordion", classes="radar-config-import mb-2"):
                    with vuetify.VExpansionPanel(title="Import saved configuration"):
                        with vuetify.VExpansionPanelText():
                            NamedFileUpload(
                                "config_import_path",
                                label="RADAR-PD configuration",
                                extensions=[".yaml", ".yml"],
                                optional=True,
                                key="radar-config-upload",
                            )
                            vuetify.VBtn("Apply configuration", click=self.apply_uploaded_configuration, variant="outlined", size="small", block=True, disabled=("!config_import_path",), classes="mt-2")
                vuetify.VBtn(
                    "Run analysis on NDIP",
                    click=self.submit_run,
                    loading=("busy",),
                    disabled=(f"busy || !({ready_expression})",),
                    color="#15543c",
                    size="large",
                    block=True,
                    prepend_icon="mdi-rocket-launch-outline",
                    classes="radar-primary-action",
                )
                vuetify.VBtn("Refresh Galaxy inputs", click=self.refresh_history, variant="text", block=True, size="small", prepend_icon="mdi-refresh", classes="mt-1")

    def _workspace_page(self) -> None:
        with html.Div(classes="radar-workspace-inner"):
            with html.Div(v_if="!selected_run_uid", classes="radar-workspace-empty"):
                html.Div("RADAR-PD SCIENTIFIC AI WORKSPACE", classes="radar-kicker")
                html.H1("Phase detection for powder diffraction")
                html.P(
                    "Configure inputs in the setup rail, then monitor the Galaxy-backed run and inspect its scientific result here.",
                    classes="radar-empty-lede",
                )
                with html.Div(classes="radar-empty-features"):
                    html.Div("Neutron and X-ray", classes="radar-feature-chip")
                    html.Div("Full and Rapid", classes="radar-feature-chip")
                    html.Div("Recoverable from Galaxy", classes="radar-feature-chip")
                with html.Div(classes="radar-empty-grid"):
                    with html.Div(classes="radar-empty-card"):
                        vuetify.VIcon("mdi-progress-clock", color="#15543c")
                        html.H3("Run Monitor")
                        html.P("Live stage, progress, messages, and bounded console output.")
                    with html.Div(classes="radar-empty-card"):
                        vuetify.VIcon("mdi-chart-line", color="#15543c")
                        html.H3("Scientific Results")
                        html.P("Best refinement, phase fractions, rankings, and diagnostics.")
                    with html.Div(classes="radar-empty-card"):
                        vuetify.VIcon("mdi-folder-outline", color="#15543c")
                        html.H3("Reproducibility")
                        html.P("Reports, tables, CIFs, GPX checkpoints, and configurations.")

            with html.Div(v_if="!!selected_run_uid"):
                with html.Div(classes="radar-run-context"):
                    with html.Div():
                        html.Div("CURRENT GALAXY RUN", classes="radar-context-kicker")
                        html.H1("{{ selected_run_name }}", classes="radar-run-title")
                        html.P("{{ viewed_run_mode === 'rapid' ? 'Rapid Hypothesis Mode' : 'Full RADAR-PD' }} / {{ selected_run_stage }}", classes="radar-run-subtitle")
                    vuetify.VChip(text=("selected_run_status",), color=("selected_run_status === 'Ok' ? '#dff2e8' : selected_run_status === 'Error' ? '#fde7e7' : '#fff0d4'",), variant="flat")
                with vuetify.VBtnToggle(
                    v_model=("workspace_view",),
                    mandatory=True,
                    divided=True,
                    color="#15543c",
                    classes="radar-workspace-nav",
                ):
                    with vuetify.VBtn(v_for="item in workspace_options", key="item.value", value=("item.value",), size="small"):
                        vuetify.VIcon(icon=("item.icon",), size="small", classes="mr-2")
                        html.Span("{{ item.title }}")

                with html.Div(v_show="workspace_view === 'monitor'", classes="radar-workspace-view"):
                    self._run_monitor_view()
                with html.Div(v_show="workspace_view === 'results'", classes="radar-workspace-view"):
                    self._results_dashboard()
                with html.Div(v_show="workspace_view === 'plots'", classes="radar-workspace-view"):
                    self._interactive_plots_view()
                with html.Div(v_show="workspace_view === 'files'", classes="radar-workspace-view"):
                    self._file_browser_view()

    def _run_monitor_view(self) -> None:
        with html.Div(classes="radar-section-heading"):
            with html.Div():
                html.H2("Run Monitor")
                html.P("The analysis remains in Galaxy even if this interactive session is closed.")
            with html.Div(classes="radar-button-row"):
                vuetify.VBtn("Refresh", click=self.refresh_selected_run, prepend_icon="mdi-refresh", variant="outlined", size="small")
                vuetify.VBtn("Use configuration", click=self.use_selected_configuration, prepend_icon="mdi-file-restore-outline", variant="outlined", size="small")
                vuetify.VBtn(
                    "Stop",
                    click="cancel_dialog = true",
                    prepend_icon="mdi-stop-circle-outline",
                    color="#9b2c2c",
                    variant="outlined",
                    size="small",
                    disabled=("selected_run_status !== 'Running' && selected_run_status !== 'Queued'",),
                )
        with html.Div(classes="radar-monitor-summary"):
            with html.Div(classes="radar-monitor-stage"):
                html.Div("CURRENT STAGE", classes="radar-micro-label")
                html.H3("{{ selected_run_stage }}")
                html.P("{{ selected_run_message || 'Waiting for the next Galaxy update.' }}")
                html.Span("{{ selected_run_elapsed }}", classes="radar-elapsed")
            with html.Div(classes="radar-progress-number"):
                html.Strong("{{ selected_run_progress }}%")
                html.Span("complete")
        vuetify.VProgressLinear(model_value=("selected_run_progress",), color="#1f6b4b", height=10, rounded=True, classes="mb-4")
        vuetify.VAlert(v_show="selected_run_loading", text="Loading the completed run and reconstructing its scientific results...", type="info", variant="tonal", classes="mb-3")
        vuetify.VAlert(v_show="selected_run_status === 'Error' && !!selected_run_message", text=("selected_run_message",), type="error", variant="tonal", classes="mb-3")
        with html.Div(classes="radar-stage-timeline"):
            with html.Div(
                v_for="stage in monitor_stages",
                key="stage.name",
                classes=("'radar-stage-item is-' + stage.state",),
            ):
                vuetify.VIcon(icon=("stage.state === 'complete' ? 'mdi-check-circle' : stage.state === 'active' ? 'mdi-progress-clock' : 'mdi-circle-outline'",), size="small")
                with html.Div():
                    html.Strong("{{ stage.name }}")
                    html.Span("{{ stage.state === 'active' ? 'In progress' : stage.state === 'complete' ? 'Complete' : 'Pending' }}")
        with vuetify.VExpansionPanels(variant="accordion", classes="radar-detail-panels mt-4"):
            with vuetify.VExpansionPanel(v_show="!!selected_run_console", title="Live analysis log"):
                with vuetify.VExpansionPanelText():
                    html.Pre("{{ selected_run_console }}", classes="radar-console")
            with vuetify.VExpansionPanel(v_show="!!viewed_configuration", title="Saved run configuration"):
                with vuetify.VExpansionPanelText():
                    html.Pre("{{ viewed_configuration }}", classes="radar-config-preview")
        with vuetify.VDialog(v_model=("cancel_dialog",), max_width=440):
            with vuetify.VCard():
                vuetify.VCardTitle("Stop this Galaxy run?")
                vuetify.VCardText("The current RADAR-PD job will be cancelled. Completed Galaxy outputs will remain in History.")
                with vuetify.VCardActions():
                    vuetify.VSpacer()
                    vuetify.VBtn("Keep running", click="cancel_dialog = false", variant="text")
                    vuetify.VBtn("Stop run", click=self.confirm_cancel_selected_run, color="#9b2c2c", variant="flat")

    def _phase_fraction_panel(self) -> None:
        with html.Section(classes="radar-result-card radar-phase-card"):
            with html.Div(classes="radar-card-heading"):
                with html.Div():
                    html.Div("QUANTITATIVE SUMMARY", classes="radar-micro-label")
                    html.H3("Phase fractions")
                vuetify.VChip(text=("phase_total",), size="small", variant="tonal", color="#15543c")
            with html.Div(v_if="phase_rows.length", classes="radar-phase-list"):
                with html.Div(v_for="phase in phase_rows", key="phase.phase + phase.space_group", classes="radar-phase-row"):
                    with html.Div(classes="radar-phase-copy"):
                        html.Strong("{{ phase.phase }}")
                        html.Span("Space group {{ phase.space_group }}")
                    with html.Div(classes="radar-phase-weight"):
                        html.Strong("{{ phase.weight_display }}")
                        vuetify.VProgressLinear(model_value=("phase.weight_value",), max=100, height=7, rounded=True, color="#1f6b4b")
            html.Div("Phase fractions are not available in this result.", v_if="!phase_rows.length", classes="radar-empty-compact")

    def _results_dashboard(self) -> None:
        with html.Div(classes="radar-section-heading"):
            with html.Div():
                html.H2("{{ viewed_run_mode === 'rapid' ? 'Rapid Results' : 'Scientific Results' }}")
                html.P("Curated from the normalized Galaxy result; technical fields remain available below.")
            vuetify.VBtn("Back to monitor", click="workspace_view = 'monitor'", prepend_icon="mdi-arrow-left", variant="text", size="small")
        with html.Div(v_if="selected_run_status !== 'Ok'", classes="radar-result-not-ready"):
            vuetify.VAlert(text="Results will appear here after the selected Galaxy run completes.", type="info", variant="tonal")
        with html.Div(v_show="selected_run_status === 'Ok'"):
            with html.Div(classes="radar-metric-grid"):
                with html.Div(v_for="metric in result_metrics", key="metric.label", classes="radar-metric-card"):
                    html.Div("{{ metric.label }}", classes="radar-metric-label")
                    html.Div("{{ metric.value }}", classes="radar-metric-value")
            with vuetify.VAlert(v_for="warning in result_warnings", key="warning", text=("warning",), type="warning", variant="tonal", density="compact", classes="mb-2"):
                pass
            with html.Div(classes="radar-result-overview-grid"):
                self._phase_fraction_panel()
                with html.Section(classes="radar-result-card radar-primary-plot-card"):
                    with html.Div(classes="radar-card-heading"):
                        with html.Div():
                            html.Div("PRIMARY SCIENTIFIC VIEW", classes="radar-micro-label")
                            html.H3("Best refinement fit")
                    vuetify.VSelect(
                        v_show="plot_options.length > 1",
                        label="Inspect another published plot",
                        v_model=("selected_plot",),
                        items=("plot_options",),
                        item_title="name",
                        item_value="path",
                        density="compact",
                        variant="outlined",
                        hide_details=True,
                        update_modelValue=(self._primary_plot_changed, "[$event]"),
                        classes="mb-2",
                    )
                    self._primary_plot_widget = plotly.Figure(display_mode_bar=True)
                    html.Div("No interactive refinement payload was published for this run.", v_if="!plot_options.length", classes="radar-empty-compact")

            with html.Section(v_show="viewed_run_mode === 'rapid'", classes="radar-stage-results"):
                html.H3("Rapid hypothesis path")
                html.P("Coarse search / lattice nudge / pattern scoring / final refinement ranking / solution inspector", classes="radar-section-help")
                with vuetify.VTabs(v_model=("rapid_stage",), color="#15543c", density="compact", classes="radar-stage-tabs"):
                    vuetify.VTab("Coarse Search", value="coarse_search")
                    vuetify.VTab("Lattice Nudge", value="lattice_nudge")
                    vuetify.VTab("Pattern Scoring", value="pattern_scoring")
                    vuetify.VTab("Final Refinement", value="final_refinement")
                    vuetify.VTab("Solution Inspector", value="inspector")
                with html.Div(v_show="rapid_stage === 'coarse_search'", classes="radar-result-card"):
                    vuetify.VDataTable(headers=("[{title:'Rank',key:'rank'},{title:'Hypothesis',key:'hypothesis'},{title:'Coarse match',key:'coarse_match'},{title:'Unexplained signal',key:'unexplained_signal'}]",), items=("rapid_coarse_rows",), density="compact", no_data_text="No coarse-search table was published.")
                with html.Div(v_show="rapid_stage === 'lattice_nudge'", classes="radar-result-card"):
                    vuetify.VDataTable(headers=("[{title:'Phase',key:'phase'},{title:'Space group',key:'space_group'},{title:'Nudge match',key:'nudge_match'},{title:'Cell adjustment',key:'cell_adjustment'},{title:'Best cell',key:'best_cell'},{title:'Time',key:'time'},{title:'Status',key:'status'}]",), items=("rapid_nudge_rows",), density="compact", no_data_text="No lattice-nudge table was published.")
                with html.Div(v_show="rapid_stage === 'pattern_scoring'", classes="radar-result-card"):
                    vuetify.VDataTable(headers=("[{title:'Rank',key:'rank'},{title:'Hypothesis',key:'hypothesis'},{title:'Key peak support',key:'key_peak_support'},{title:'Coarse rank',key:'coarse_rank'},{title:'Pattern match',key:'pattern_match'},{title:'Explained signal',key:'explained_signal'},{title:'Unexplained signal',key:'unexplained_signal'}]",), items=("rapid_pattern_rows",), density="compact", no_data_text="No pattern-scoring table was published.")
                with html.Div(v_show="rapid_stage === 'final_refinement'", classes="radar-stage-content"):
                    with html.Div(classes="radar-refinement-card-grid"):
                        with html.Div(v_for="row in top_refinements", key="row.rank", classes="radar-refinement-card"):
                            html.Div("Final rank {{ row.rank }}", classes="radar-micro-label")
                            html.H4("{{ row.hypothesis }}")
                            html.Div("Rwp {{ row.rwp }}%", classes="radar-refinement-rwp")
                            html.P("{{ row.phase_fractions }}")
                            html.Span("{{ row.status }} / {{ row.time }}")
                    with html.Div(classes="radar-result-card"):
                        vuetify.VDataTable(headers=("[{title:'Rank',key:'rank'},{title:'Hypothesis',key:'hypothesis'},{title:'Rwp',key:'rwp'},{title:'Phase fractions',key:'phase_fractions'},{title:'Pattern rank',key:'pattern_rank'},{title:'Status',key:'status'},{title:'Time',key:'time'}]",), items=("rapid_final_rows",), density="compact", no_data_text="No final-refinement ranking was published.")
                with html.Div(v_show="rapid_stage === 'inspector'", classes="radar-result-card"):
                    html.H3("Solution Inspector")
                    html.P("Compare published refined hypotheses and carry an existing GSAS-II checkpoint into the downstream handoff workflow.", classes="radar-section-help")
                    with html.Div(classes="radar-field-pair"):
                        vuetify.VSelect(label="Primary hypothesis", v_model=("selected_hypothesis",), items=("rapid_final_rows",), item_title="hypothesis", item_value="rank", density="compact", variant="outlined", no_data_text="No refined hypotheses are available")
                        vuetify.VSelect(label="Compare with", v_model=("comparison_hypothesis",), items=("rapid_final_rows",), item_title="hypothesis", item_value="rank", density="compact", variant="outlined", clearable=True, no_data_text="No alternate hypothesis is available")
                    with html.Div(classes="radar-refinement-card-grid"):
                        with html.Div(v_for="row in rapid_final_rows.filter(item => String(item.rank) === String(selected_hypothesis) || String(item.rank) === String(comparison_hypothesis))", key="'compare-' + row.rank", classes="radar-refinement-card"):
                            html.Div("Published rank {{ row.rank }}", classes="radar-micro-label")
                            html.H4("{{ row.hypothesis }}")
                            html.Div("Rwp {{ row.rwp }}%", classes="radar-refinement-rwp")
                            html.P("{{ row.phase_fractions }}")
                            html.Span("{{ row.status }} / {{ row.time }}")
                    vuetify.VSelect(label="GSAS-II checkpoint", v_model=("selected_checkpoint",), items=("checkpoint_rows",), item_title="name", item_value="path", density="compact", variant="outlined", no_data_text="No published GPX checkpoint is available")
                    with html.Div(classes="radar-button-row"):
                        vuetify.VBtn("Download selected checkpoint", click=self.download_checkpoint, disabled=("!selected_checkpoint",), prepend_icon="mdi-download", color="#15543c", variant="outlined")
                        html.A("Open GSAS-II project handoff", href="/?tool_id=neutrons_radar_pd_gpx_handoff_prototype&version=latest", target="_blank", classes="radar-secondary-link as-button")
                    vuetify.VAlert(text="This release reviews and hands off existing checkpoints. Editing a phase combination and launching a targeted refinement requires a future Galaxy backend action.", type="info", variant="tonal", density="compact", classes="mt-3")

            with html.Section(v_show="viewed_run_mode === 'full'", classes="radar-full-results"):
                html.H3("Full refinement progression")
                with html.Div(v_if="full_progression.length", classes="radar-full-progression"):
                    with html.Div(v_for="item in full_progression", key="item.stage + item.status", classes="radar-progression-item"):
                        vuetify.VIcon(icon=("item.status.toLowerCase().includes('accept') ? 'mdi-check-circle' : 'mdi-checkbox-blank-circle-outline'",), color="#15543c", size="small")
                        with html.Div():
                            html.Strong("{{ item.stage }}")
                            html.Span("{{ item.status }}")
                html.Div("No GPX progression was published for this run.", v_if="!full_progression.length", classes="radar-empty-compact")
                with html.Div(classes="radar-result-card mt-3"):
                    html.H3("Accepted and reviewed models")
                    html.P("Published model decisions from the Full pipeline, without internal filenames or paths.", classes="radar-section-help")
                    vuetify.VDataTable(
                        headers=("[{title:'Model',key:'model'},{title:'Stage',key:'stage'},{title:'Rwp',key:'rwp'},{title:'Decision',key:'decision'},{title:'Note',key:'note'}]",),
                        items=("full_model_rows",),
                        density="compact",
                        no_data_text="No model-decision table was published.",
                    )

            with vuetify.VExpansionPanels(variant="accordion", classes="radar-technical-panel mt-4"):
                with vuetify.VExpansionPanel(title="Technical tables and raw fields"):
                    with vuetify.VExpansionPanelText():
                        vuetify.VSelect(label="Published result table", v_model=("selected_table",), items=("table_options",), item_title="name", item_value="path", density="compact", variant="outlined", update_modelValue=(self._table_changed, "[$event]"))
                        vuetify.VDataTable(headers=("table_headers",), items=("table_rows",), density="compact", fixed_header=True, height=460, no_data_text="Select a result table.")

    def _interactive_plots_view(self) -> None:
        with html.Div(classes="radar-section-heading"):
            with html.Div():
                html.H2("Interactive Plots")
                html.P("Accepted Full-mode fits and diagnostic plots, grouped by scientific role.")
        with html.Section(classes="radar-result-card"):
            with html.Div(classes="radar-plot-category-row"):
                with vuetify.VChip(v_for="group in plot_groups", key="group.name", size="small", variant="outlined", color="#15543c"):
                    html.Span("{{ group.name }} / {{ group.count }}")
            vuetify.VSelect(label="Published interactive plot", v_model=("gallery_selected_plot",), items=("plot_options",), item_title="name", item_value="path", density="compact", variant="outlined", update_modelValue=(self._gallery_plot_changed, "[$event]"))
            self._plot_widget = plotly.Figure(display_mode_bar=True)
            html.Div("No interactive plots were published.", v_if="!plot_options.length", classes="radar-empty-compact")

    def _file_browser_view(self) -> None:
        with html.Div(classes="radar-section-heading"):
            with html.Div():
                html.H2("Run File Browser")
                html.P("Published results grouped by scientific purpose; local container paths remain hidden.")
            html.A("Open GSAS-II project handoff", href="/?tool_id=neutrons_radar_pd_gpx_handoff_prototype&version=latest", target="_blank", classes="radar-secondary-link as-button")
        with vuetify.VExpansionPanels(multiple=True, variant="accordion", classes="radar-file-groups"):
            with vuetify.VExpansionPanel(v_for="group in file_groups", key="group.name", value=("group.name",)):
                with vuetify.VExpansionPanelTitle():
                    html.Strong("{{ group.name }}")
                    vuetify.VSpacer()
                    vuetify.VChip(text=("String(group.files.length)",), size="x-small", variant="tonal", color="#15543c")
                with vuetify.VExpansionPanelText():
                    with html.Div(v_for="file in group.files", key="file.id", classes="radar-file-row"):
                        with html.Div(classes="radar-file-copy"):
                            html.Strong("{{ file.name }}")
                            html.Span("{{ file.filename }} / {{ file.size }}")
                        vuetify.VBtn(icon="mdi-download", title="Download", size="small", variant="text", color="#15543c", click=(self.download_artifact, "[file.path]"))
        html.Div("No downloadable files are available for this run.", v_if="!file_groups.length", classes="radar-empty-compact")

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
            state.workspace_view = "monitor"
            state.setup_collapsed = True
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
            if record.status == RunStatus.OK and record.output_dir:
                self._load_results(record)
                if record.uid not in self._auto_opened_uids:
                    self._auto_opened_uids.add(record.uid)
                    self.server.state.workspace_view = "results"
            elif record.status == RunStatus.ERROR:
                self.server.state.workspace_view = "monitor"
        self.server.state.flush()

    def _start_monitor(self, record: RunRecord) -> None:
        if record.uid in self._monitored_uids or record.status not in {RunStatus.NEW, RunStatus.UPLOADING, RunStatus.QUEUED, RunStatus.RUNNING}:
            return
        self._monitored_uids.add(record.uid)
        task = asyncio.create_task(self._monitor_record(record), name=f"radar-monitor-{record.uid[:8]}")
        self._monitor_tasks.add(task)
        task.add_done_callback(self._monitor_tasks.discard)

    async def _monitor_record(self, record: RunRecord, *, poll_seconds: float = 5.0) -> None:
        """Poll Galaxy off-thread and publish UI state on Trame's event loop."""

        terminal = {RunStatus.OK, RunStatus.ERROR, RunStatus.CANCELLED}
        try:
            while record.status not in terminal:
                try:
                    record = await asyncio.to_thread(self.service.refresh, record)
                except Exception as exc:
                    record.message = str(exc)
                self._monitor_update(record)
                if record.status in terminal:
                    break
                await asyncio.sleep(poll_seconds)
            if record.status == RunStatus.OK:
                try:
                    record = await asyncio.to_thread(self.service.collect_results, record)
                except Exception as exc:
                    record.status = RunStatus.ERROR
                    record.stage = "Result download failed"
                    record.progress = 100
                    record.message = f"Analysis finished, but result download failed: {exc}"
                self._monitor_update(record)
        finally:
            self._monitored_uids.discard(record.uid)

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
        if not uid or uid not in self.records:
            return
        record = self.records[uid]
        self._select_record(record)
        if record.status == RunStatus.OK:
            self._open_record_results(record)
            return
        self.server.state.workspace_view = "monitor"
        self.server.state.setup_collapsed = True
        self.server.state.flush()

    def _sync_workspace_options(self, mode: AnalysisMode | str) -> None:
        state = self.server.state
        value = mode.value if isinstance(mode, AnalysisMode) else str(mode)
        if value == AnalysisMode.RAPID.value:
            options = [
                {"title": "Run Monitor", "value": "monitor", "icon": "mdi-progress-clock"},
                {"title": "Rapid Results", "value": "results", "icon": "mdi-chart-line"},
                {"title": "Run File Browser", "value": "files", "icon": "mdi-folder-outline"},
            ]
        else:
            options = [
                {"title": "Run Monitor", "value": "monitor", "icon": "mdi-progress-clock"},
                {"title": "Results", "value": "results", "icon": "mdi-chart-line"},
                {"title": "Interactive Plots", "value": "plots", "icon": "mdi-chart-scatter-plot"},
                {"title": "Run File Browser", "value": "files", "icon": "mdi-folder-outline"},
            ]
        state.workspace_options = options
        if getattr(state, "workspace_view", "monitor") not in {item["value"] for item in options}:
            state.workspace_view = "monitor"

    @staticmethod
    def _monitor_stage_rows(record: RunRecord) -> list[dict[str, str]]:
        if record.mode == AnalysisMode.RAPID:
            names = [
                "Signal preparation",
                "Coarse search",
                "Lattice nudge",
                "Pattern scoring",
                "Final refinement",
                "Result collection",
            ]
            tokens = [
                ("prepar", "upload", "queue", "wait"),
                ("coarse", "search64", "hypothesis search"),
                ("nudge", "lattice"),
                ("pattern", "512", "scoring"),
                ("gsas", "refin", "validation"),
                ("result", "download", "collect", "ready"),
            ]
        else:
            names = [
                "Input preparation",
                "Main-phase model",
                "Candidate search",
                "Refinement passes",
                "Finalization",
                "Result collection",
            ]
            tokens = [
                ("prepar", "upload", "queue", "wait"),
                ("main", "anchor"),
                ("candidate", "search", "screen"),
                ("pass", "refin", "compare"),
                ("final", "polish"),
                ("result", "download", "collect", "ready"),
            ]
        if record.status == RunStatus.OK:
            active_index = len(names)
        else:
            lowered = str(record.stage or "").lower()
            active_index = next(
                (index for index, stage_tokens in enumerate(tokens) if any(token in lowered for token in stage_tokens)),
                min(len(names) - 1, max(0, int(record.progress) * len(names) // 101)),
            )
        return [
            {
                "name": name,
                "state": "complete" if index < active_index or record.status == RunStatus.OK else "active" if index == active_index else "pending",
            }
            for index, name in enumerate(names)
        ]

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
        state.viewed_run_mode = record.mode.value
        state.monitor_stages = self._monitor_stage_rows(record)
        try:
            created = datetime.fromisoformat(record.created_utc.replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            elapsed = max(0.0, (datetime.now(timezone.utc) - created).total_seconds())
            state.selected_run_elapsed = f"{elapsed:.0f} s elapsed"
        except (TypeError, ValueError):
            state.selected_run_elapsed = "-"
        self._sync_workspace_options(record.mode)
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
            state.setup_collapsed = False
            state.setup_panels = ["data", "review"]
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
            self.server.state.setup_collapsed = False
            self.server.state.setup_panels = ["data", "review"]
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
            if record.status == RunStatus.OK and record.output_dir:
                self._load_results(record)
                self.server.state.workspace_view = "results"
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

    def confirm_cancel_selected_run(self, **_: Any) -> None:
        self.server.state.cancel_dialog = False
        self.cancel_selected_run()

    def open_selected_results(self, **_: Any) -> None:
        uid = self.server.state.selected_run_uid
        if not uid or uid not in self.records:
            return
        self._open_record_results(self.records[uid])

    def _open_record_results(self, record: RunRecord) -> None:
        """Collect and display one completed run without allowing duplicate loads."""

        if record.status != RunStatus.OK or self._opening_run_uid == record.uid:
            return
        state = self.server.state
        self._opening_run_uid = record.uid
        state.selected_run_loading = True
        state.selected_run_message = ""
        state.error_message = ""
        state.notice = f"Loading results for {record.name}..."
        state.flush()
        try:
            if not record.output_dir:
                record = self.service.collect_results(record)
                self.records[record.uid] = record
                self._select_record(record)
                self._sync_runs()
            self._load_results(record)
            state.active_page = "results"
            state.workspace_view = "results"
            state.setup_collapsed = True
            state.notice = f"Showing results for {record.name}."
        except Exception as exc:
            message = f"Could not load results for {record.name}: {exc}"
            state.active_page = "runs"
            state.workspace_view = "monitor"
            state.error_message = message
            state.selected_run_message = message
        finally:
            state.selected_run_loading = False
            self._opening_run_uid = None
            state.flush()

    def _load_results(self, record: RunRecord) -> None:
        state = self.server.state
        self._reset_result_state()
        payload = self.service.result_payload(record)
        result_document = payload.get("summary") or {}
        state.viewed_run_mode = record.mode.value
        state.viewed_run_name = record.name
        root = record.local_output_dir
        if root is None:
            return
        view = build_result_view(result_document, root).to_state()
        state.result_metrics = view["metrics"]
        state.summary_cards = view["metrics"]
        state.result_warnings = view["warnings"]
        state.phase_rows = view["phases"]
        state.phase_total = view["phase_total"]
        state.table_options = view["tables"]
        state.plot_options = view["plots"]
        state.primary_plot_path = view["primary_plot_path"]
        state.file_groups = view["file_groups"]
        state.checkpoint_rows = view["checkpoints"]
        state.gpx_rows = view["checkpoints"]
        state.rapid_coarse_rows = view["rapid_stages"]["coarse_search"]
        state.rapid_nudge_rows = view["rapid_stages"]["lattice_nudge"]
        state.rapid_pattern_rows = view["rapid_stages"]["pattern_scoring"]
        state.rapid_final_rows = view["rapid_stages"]["final_refinement"]
        state.top_refinements = view["top_refinements"]
        state.solution_rows = view["rapid_stages"]["final_refinement"]
        state.solution_headers = [
            {"title": "Rank", "key": "rank"},
            {"title": "Hypothesis", "key": "hypothesis"},
            {"title": "Rwp", "key": "rwp"},
            {"title": "Phase fractions", "key": "phase_fractions"},
        ]
        state.full_progression = view["full_progression"]
        state.full_model_rows = view["full_models"]
        state.artifact_options = [
            {"title": f"{group['name']} / {item['name']} ({item['size']})", "path": item["path"]}
            for group in view["file_groups"]
            for item in group["files"]
        ]
        counts: dict[str, int] = {}
        for item in state.plot_options:
            counts[item["category"]] = counts.get(item["category"], 0) + 1
        state.plot_groups = [{"name": name, "count": count} for name, count in counts.items()]
        if state.table_options:
            primary_table = next((item for item in state.table_options if item.get("primary")), state.table_options[0])
            state.selected_table = primary_table["path"]
            self._table_changed()
        if state.plot_options:
            state.selected_plot = state.primary_plot_path or state.plot_options[0]["path"]
            state.gallery_selected_plot = state.selected_plot
            self._primary_plot_changed()
            self._gallery_plot_changed()
        if state.artifact_options:
            state.selected_artifact = state.artifact_options[0]["path"]
        if state.checkpoint_rows:
            state.selected_checkpoint = next(
                (item["path"] for item in state.checkpoint_rows if item.get("handoff_available")),
                "",
            )
        if state.rapid_final_rows:
            state.selected_hypothesis = state.rapid_final_rows[0]["rank"]
            state.comparison_hypothesis = (
                state.rapid_final_rows[1]["rank"] if len(state.rapid_final_rows) > 1 else None
            )
        self._sync_workspace_options(record.mode)
        state.flush()

    def _reset_result_state(self) -> None:
        state = self.server.state
        state.summary_cards = []
        state.result_metrics = []
        state.result_warnings = []
        state.phase_rows = []
        state.phase_total = "-"
        state.table_options = []
        state.selected_table = ""
        state.table_rows = []
        state.table_headers = []
        state.plot_options = []
        state.selected_plot = ""
        state.gallery_selected_plot = ""
        state.primary_plot_path = ""
        state.plot_groups = []
        state.artifact_options = []
        state.selected_artifact = ""
        state.file_groups = []
        state.gpx_rows = []
        state.checkpoint_rows = []
        state.selected_checkpoint = ""
        state.solution_rows = []
        state.solution_headers = []
        state.rapid_coarse_rows = []
        state.rapid_nudge_rows = []
        state.rapid_pattern_rows = []
        state.rapid_final_rows = []
        state.top_refinements = []
        state.full_progression = []
        state.full_model_rows = []
        state.selected_hypothesis = None
        state.comparison_hypothesis = None
        state.result_tab = "overview"
        if self._plot_widget is not None:
            self._plot_widget.update(figure_for_payload({}))
        if self._primary_plot_widget is not None:
            self._primary_plot_widget.update(figure_for_payload({}))

    def _table_changed(self, path: str | None = None, **_: Any) -> None:
        path = path or self.server.state.selected_table
        if path:
            self.server.state.selected_table = path
        rows = read_table(path) if path else []
        self.server.state.table_rows = rows
        keys = list(rows[0].keys()) if rows else []
        self.server.state.table_headers = [{"title": key.replace("_", " ").title(), "key": key} for key in keys]
        self.server.state.flush()

    def _primary_plot_changed(self, path: str | None = None, **_: Any) -> None:
        path = path or self.server.state.selected_plot
        if not path or self._primary_plot_widget is None:
            return
        self.server.state.selected_plot = path
        figure = figure_for_payload(read_plot_payload(path))
        self._primary_plot_widget.update(figure)

    def _gallery_plot_changed(self, path: str | None = None, **_: Any) -> None:
        path = path or self.server.state.gallery_selected_plot
        if not path or self._plot_widget is None:
            return
        self.server.state.gallery_selected_plot = path
        figure = figure_for_payload(read_plot_payload(path))
        self._plot_widget.update(figure)

    def _plot_changed(self, path: str | None = None, **_: Any) -> None:
        """Backward-compatible alias for existing callbacks and tests."""

        self._primary_plot_changed(path)

    def download_artifact(self, path: str | None = None, **_: Any) -> None:
        path = Path(str(path or self.server.state.selected_artifact or ""))
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
            ".cif": "chemical/x-cif",
            ".png": "image/png",
            ".svg": "image/svg+xml",
        }.get(path.suffix.lower(), "application/octet-stream")
        self.download_file(path.name, mime, path.read_bytes())

    def download_checkpoint(self, **_: Any) -> None:
        selected = str(self.server.state.selected_checkpoint or "")
        self.download_artifact(selected)

    @staticmethod
    def _css() -> str:
        return """
        :root {
          color-scheme: light;
          --radar-brand-900: #0d3428;
          --radar-brand-800: #124331;
          --radar-brand-700: #15543c;
          --radar-brand-600: #1f6b4b;
          --radar-ink: #18231f;
          --radar-muted: #64746e;
          --radar-surface: #ffffff;
          --radar-surface-muted: #f6faf8;
          --radar-line: #d9e5df;
          --radar-line-strong: #b8cec2;
          --radar-warn: #a66a00;
          --radar-danger: #9b2c2c;
        }
        * { box-sizing: border-box; }
        body { margin: 0; background: #f2f7f4; color: var(--radar-ink); font-family: Inter, "Segoe UI", Arial, sans-serif; }
        .radar-app-shell { min-height: calc(100vh - 64px); background: #f2f7f4; }
        .radar-context-header {
          min-height: 58px; display: flex; align-items: center; gap: 10px; padding: 8px 18px;
          background: var(--radar-surface); border-bottom: 1px solid var(--radar-line); position: sticky; top: 0; z-index: 8;
        }
        .radar-context-title { min-width: 220px; }
        .radar-product-name { color: var(--radar-brand-900); font-size: 16px; font-weight: 800; line-height: 1.2; }
        .radar-product-subtitle { color: var(--radar-muted); font-size: 12px; margin-top: 2px; }
        .radar-run-chip { max-width: 310px; }
        .radar-run-chip .v-chip__content { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .radar-layout { display: grid; grid-template-columns: 360px minmax(0, 1fr); min-height: calc(100vh - 122px); }
        .radar-layout.is-collapsed { grid-template-columns: minmax(0, 1fr); }
        .radar-setup-rail {
          width: 360px; max-height: calc(100vh - 122px); overflow-y: auto; padding: 18px 14px 36px;
          background: #eef5f1; border-right: 1px solid var(--radar-line-strong); scrollbar-gutter: stable;
        }
        .radar-rail-kicker, .radar-kicker, .radar-context-kicker, .radar-micro-label {
          color: var(--radar-brand-700); font-size: 11px; line-height: 1.2; font-weight: 800; letter-spacing: .075em; text-transform: uppercase;
        }
        .radar-rail-heading { margin: 5px 0 4px; color: var(--radar-brand-900); font-size: 23px; line-height: 1.2; }
        .radar-rail-help { margin: 0 0 15px; color: var(--radar-muted); font-size: 13px; line-height: 1.45; }
        .radar-history-panel, .radar-setup-panels { border: 1px solid var(--radar-line-strong); border-radius: 8px; overflow: hidden; box-shadow: none; }
        .radar-setup-panel { border-left: 3px solid var(--radar-brand-700); }
        .radar-setup-panel + .radar-setup-panel { border-top: 1px solid var(--radar-line); }
        .radar-setup-title { min-height: 48px !important; padding: 8px 11px !important; font-size: 13px; }
        .radar-step-number {
          display: inline-grid; place-items: center; flex: 0 0 25px; width: 25px; height: 25px; margin-right: 9px;
          border-radius: 50%; background: #dceee5; color: var(--radar-brand-700); font-size: 12px; font-weight: 850;
        }
        .radar-step-label { color: var(--radar-brand-900); font-weight: 750; }
        .radar-setup-panels .v-expansion-panel-text__wrapper, .radar-history-panel .v-expansion-panel-text__wrapper { padding: 8px 11px 13px; }
        .radar-field-pair { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
        .radar-secondary-link {
          display: inline-flex; align-items: center; color: var(--radar-brand-700); font-size: 13px; font-weight: 750;
          text-decoration: none; border-bottom: 1px solid currentColor; margin-top: 5px;
        }
        .radar-secondary-link.as-button { min-height: 34px; padding: 6px 11px; border: 1px solid var(--radar-brand-700); border-radius: 7px; margin: 0; }
        .radar-secondary-link:hover { color: var(--radar-brand-900); background: #e7f3ed; }
        .radar-hidden-file-input { position: absolute !important; width: 1px !important; height: 1px !important; overflow: hidden !important; opacity: 0 !important; pointer-events: none !important; }
        .radar-upload-card {
          position: relative; display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; align-items: center;
          margin: 8px 0; padding: 10px; background: #fff; border: 1px solid var(--radar-line); border-radius: 8px;
        }
        .radar-upload-copy { min-width: 0; }
        .radar-upload-label { color: var(--radar-ink); font-size: 13px; font-weight: 760; }
        .radar-upload-help { color: var(--radar-muted); font-size: 11px; line-height: 1.35; margin-top: 2px; }
        .radar-upload-filename { color: #7a8982; font-size: 12px; line-height: 1.35; overflow-wrap: anywhere; margin-top: 4px; }
        .radar-upload-filename.is-ready { color: var(--radar-brand-600); font-weight: 720; }
        .radar-upload-actions { display: flex; align-items: center; gap: 2px; }
        .radar-checklist { display: grid; gap: 6px; margin: 7px 0 12px; }
        .radar-check-row { display: flex; align-items: center; gap: 7px; color: #344740; font-size: 12px; }
        .radar-review-summary { padding: 10px; margin-bottom: 11px; background: var(--radar-surface-muted); border: 1px solid var(--radar-line); border-radius: 7px; }
        .radar-review-primary { color: var(--radar-brand-900); font-size: 13px; font-weight: 780; }
        .radar-review-secondary { color: var(--radar-muted); font-size: 12px; margin-top: 3px; overflow-wrap: anywhere; }
        .radar-primary-action { font-weight: 800; letter-spacing: 0; }
        .radar-run-list { max-height: 245px; overflow-y: auto; margin-top: 8px; border: 1px solid var(--radar-line); border-radius: 7px; background: #fff; }
        .radar-run-list-item { min-height: 65px !important; border-bottom: 1px solid var(--radar-line); cursor: pointer; }
        .radar-run-list-name { color: var(--radar-brand-900); font-size: 12px; font-weight: 760; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .radar-run-list-meta, .radar-run-list-stage { color: var(--radar-muted); font-size: 11px; margin-top: 2px; }
        .radar-workspace { min-width: 0; background: #f7faf8; }
        .radar-workspace-inner { max-width: 1520px; margin: 0 auto; padding: 22px clamp(16px, 2.2vw, 34px) 56px; }
        .radar-workspace-empty { min-height: 620px; display: flex; flex-direction: column; justify-content: center; align-items: flex-start; max-width: 1040px; margin: 0 auto; }
        .radar-workspace-empty h1 { max-width: 760px; margin: 16px 0 12px; color: var(--radar-brand-900); font-size: clamp(36px, 5vw, 62px); line-height: 1.02; letter-spacing: -.025em; }
        .radar-empty-lede { max-width: 760px; color: #52645c; font-size: 18px; line-height: 1.55; }
        .radar-empty-features { display: flex; flex-wrap: wrap; gap: 8px; margin: 9px 0 26px; }
        .radar-feature-chip { padding: 5px 10px; color: var(--radar-brand-700); border: 1px solid var(--radar-line-strong); border-radius: 999px; background: #fff; font-size: 12px; font-weight: 700; }
        .radar-empty-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; width: 100%; }
        .radar-empty-card { min-height: 145px; padding: 18px; background: #fff; border: 1px solid var(--radar-line); border-left: 4px solid var(--radar-brand-700); border-radius: 8px; }
        .radar-empty-card h3 { margin: 10px 0 4px; font-size: 16px; }
        .radar-empty-card p { margin: 0; color: var(--radar-muted); font-size: 13px; line-height: 1.45; }
        .radar-run-context { display: flex; justify-content: space-between; align-items: center; gap: 18px; margin-bottom: 13px; }
        .radar-run-title { margin: 3px 0 2px; color: var(--radar-brand-900); font-size: clamp(25px, 3vw, 38px); line-height: 1.12; overflow-wrap: anywhere; }
        .radar-run-subtitle { margin: 0; color: var(--radar-muted); font-size: 13px; }
        .radar-workspace-nav { display: flex !important; width: 100%; min-height: 40px; margin-bottom: 20px; border: 1px solid var(--radar-line-strong); border-radius: 8px; background: #fff; overflow: hidden; }
        .radar-workspace-nav .v-btn { flex: 1 1 160px; text-transform: none; font-weight: 730; }
        .radar-workspace-view { min-width: 0; }
        .radar-section-heading { display: flex; justify-content: space-between; align-items: flex-end; gap: 18px; margin: 0 0 15px; }
        .radar-section-heading h2 { margin: 0; color: var(--radar-brand-900); font-size: 27px; }
        .radar-section-heading p, .radar-section-help { margin: 4px 0 0; color: var(--radar-muted); font-size: 13px; line-height: 1.5; }
        .radar-button-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
        .radar-monitor-summary { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 18px; padding: 19px; background: #fff; border: 1px solid var(--radar-line); border-left: 4px solid var(--radar-brand-700); border-radius: 8px; }
        .radar-monitor-stage h3 { margin: 5px 0 3px; color: var(--radar-brand-900); font-size: 21px; }
        .radar-monitor-stage p { margin: 0; color: var(--radar-muted); }
        .radar-elapsed { display: block; color: var(--radar-muted); font-size: 12px; margin-top: 6px; }
        .radar-progress-number { display: flex; flex-direction: column; align-items: flex-end; justify-content: center; }
        .radar-progress-number strong { color: var(--radar-brand-700); font-size: 30px; }
        .radar-progress-number span { color: var(--radar-muted); font-size: 11px; text-transform: uppercase; }
        .radar-stage-timeline { display: grid; grid-template-columns: repeat(6, minmax(105px, 1fr)); gap: 8px; overflow-x: auto; }
        .radar-stage-item { display: flex; gap: 7px; min-width: 110px; padding: 10px; border: 1px solid var(--radar-line); border-radius: 7px; background: #fff; color: #91a099; }
        .radar-stage-item strong, .radar-stage-item span { display: block; }
        .radar-stage-item strong { font-size: 12px; color: inherit; }
        .radar-stage-item span { margin-top: 2px; font-size: 10px; }
        .radar-stage-item.is-active { color: var(--radar-warn); border-color: #e2c87e; background: #fffaf0; }
        .radar-stage-item.is-complete { color: var(--radar-brand-600); background: #f0f8f4; }
        .radar-console, .radar-config-preview { max-height: 440px; overflow: auto; margin: 0; padding: 14px; border-radius: 7px; font: 12px/1.45 Consolas, "Courier New", monospace; white-space: pre-wrap; }
        .radar-console { background: #17251f; color: #e7f2ed; }
        .radar-config-preview { background: #f5f8f7; color: #263a32; border: 1px solid var(--radar-line); }
        .radar-metric-grid { display: grid; grid-template-columns: repeat(6, minmax(118px, 1fr)); gap: 9px; margin-bottom: 14px; }
        .radar-metric-card { min-width: 0; min-height: 78px; padding: 12px 13px; background: #fff; border: 1px solid var(--radar-line); border-top: 3px solid var(--radar-brand-700); border-radius: 8px; }
        .radar-metric-label { color: var(--radar-muted); font-size: 10px; font-weight: 800; letter-spacing: .055em; text-transform: uppercase; }
        .radar-metric-value { margin-top: 6px; color: var(--radar-ink); font-size: 16px; font-weight: 780; overflow-wrap: anywhere; }
        .radar-result-overview-grid { display: grid; grid-template-columns: minmax(290px, .78fr) minmax(520px, 1.7fr); gap: 13px; align-items: start; }
        .radar-result-card { min-width: 0; padding: 16px; background: #fff; border: 1px solid var(--radar-line); border-radius: 8px; box-shadow: 0 3px 10px rgba(13,52,40,.04); }
        .radar-card-heading { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; margin-bottom: 12px; }
        .radar-card-heading h3 { margin: 3px 0 0; color: var(--radar-brand-900); font-size: 19px; }
        .radar-phase-list { display: grid; gap: 8px; }
        .radar-phase-row { padding: 9px 0; border-bottom: 1px solid var(--radar-line); }
        .radar-phase-row:last-child { border-bottom: 0; }
        .radar-phase-copy, .radar-phase-weight { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; align-items: center; }
        .radar-phase-copy strong { color: var(--radar-ink); font-size: 13px; overflow-wrap: anywhere; }
        .radar-phase-copy span { color: var(--radar-muted); font-size: 11px; }
        .radar-phase-weight { grid-template-columns: 58px minmax(0, 1fr); margin-top: 7px; }
        .radar-phase-weight strong { color: var(--radar-brand-700); font-size: 12px; }
        .radar-primary-plot-card { overflow: hidden; }
        .radar-stage-results, .radar-full-results { margin-top: 18px; }
        .radar-stage-results > h3, .radar-full-results > h3 { margin: 0; color: var(--radar-brand-900); font-size: 20px; }
        .radar-stage-tabs { margin: 12px 0 10px; border-bottom: 1px solid var(--radar-line); }
        .radar-stage-content { min-width: 0; }
        .radar-refinement-card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(215px, 1fr)); gap: 9px; margin-bottom: 10px; }
        .radar-refinement-card { min-width: 0; padding: 13px; background: #fff; border: 1px solid var(--radar-line); border-left: 4px solid var(--radar-brand-700); border-radius: 8px; }
        .radar-refinement-card h4 { margin: 5px 0 8px; color: var(--radar-ink); font-size: 13px; overflow-wrap: anywhere; }
        .radar-refinement-card p { margin: 6px 0; color: #344740; font-size: 11px; line-height: 1.45; }
        .radar-refinement-card span { color: var(--radar-muted); font-size: 11px; }
        .radar-refinement-rwp { color: var(--radar-brand-700); font-size: 18px; font-weight: 800; }
        .radar-full-progression { display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 8px; margin: 11px 0; }
        .radar-progression-item { display: flex; gap: 8px; padding: 11px; background: #fff; border: 1px solid var(--radar-line); border-radius: 7px; }
        .radar-progression-item strong, .radar-progression-item span { display: block; }
        .radar-progression-item strong { font-size: 12px; color: var(--radar-ink); }
        .radar-progression-item span { margin-top: 3px; font-size: 11px; color: var(--radar-muted); }
        .radar-plot-category-row { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 10px; }
        .radar-file-groups { border: 1px solid var(--radar-line); border-radius: 8px; overflow: hidden; }
        .radar-file-row { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 9px 2px; border-bottom: 1px solid var(--radar-line); }
        .radar-file-row:last-child { border-bottom: 0; }
        .radar-file-copy { min-width: 0; }
        .radar-file-copy strong, .radar-file-copy span { display: block; overflow-wrap: anywhere; }
        .radar-file-copy strong { color: var(--radar-ink); font-size: 13px; }
        .radar-file-copy span { margin-top: 2px; color: var(--radar-muted); font-size: 11px; }
        .radar-empty-compact { padding: 12px; color: var(--radar-muted); font-size: 12px; line-height: 1.45; text-align: center; }
        .radar-technical-panel { border: 1px solid var(--radar-line); border-radius: 8px; overflow: hidden; }
        .radar-result-card .v-data-table, .radar-technical-panel .v-data-table { max-width: 100%; overflow-x: auto; }
        @media (max-width: 1280px) {
          .radar-metric-grid { grid-template-columns: repeat(3, minmax(120px, 1fr)); }
          .radar-result-overview-grid { grid-template-columns: 1fr; }
        }
        @media (max-width: 1050px) {
          .radar-context-header { position: static; flex-wrap: wrap; }
          .radar-layout, .radar-layout.is-collapsed { display: block; }
          .radar-setup-rail { width: 100%; max-height: none; border-right: 0; border-bottom: 1px solid var(--radar-line-strong); }
          .radar-workspace-empty { min-height: 430px; }
          .radar-empty-grid { grid-template-columns: 1fr; }
        }
        @media (max-width: 700px) {
          .radar-context-header { padding: 8px 10px; }
          .radar-product-subtitle, .radar-run-chip { display: none; }
          .radar-workspace-inner { padding: 15px 10px 38px; }
          .radar-run-context, .radar-section-heading { align-items: flex-start; flex-direction: column; }
          .radar-workspace-nav { overflow-x: auto; }
          .radar-workspace-nav .v-btn { flex: 0 0 auto; }
          .radar-metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          .radar-field-pair { grid-template-columns: 1fr; }
          .radar-stage-timeline { grid-template-columns: repeat(6, 130px); }
          .radar-monitor-summary { grid-template-columns: 1fr; }
          .radar-progress-number { align-items: flex-start; }
          .radar-upload-card { grid-template-columns: 1fr; }
          .radar-upload-actions { justify-content: flex-start; }
        }
        """
