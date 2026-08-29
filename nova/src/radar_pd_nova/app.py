"""RADAR-PD interactive NOVA/Trame application."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dateutil.tz import tzstr
from nova.trame import ThemedApp
from trame.app import get_server
from trame.widgets import client, html, plotly, vuetify3 as vuetify
import yaml

from .configuration import config_from_contract, load_configuration
from .facility import FacilityBrowser, FacilityPathError, WatchRecipe
from .galaxy_service import (
    COMPARE_SERIES_TOOL_ID,
    GPX_HANDOFF_TOOL_ID,
    GSASII_INTERACTIVE_TOOL_ID,
    LIBRARY_BUILDER_TOOL_ID,
    RESULT_EXPLORER_TOOL_ID,
    SNS_RESOLVER_TOOL_ID,
    GalaxyService,
)
from .models import (
    AnalysisConfig,
    AnalysisMode,
    InputSelection,
    InputSource,
    ResultStatus,
    RunRecord,
    RunStatus,
    SubmissionSnapshot,
    UtilityActionRecord,
    selected_run_uid,
)
from .powgen_controller import (
    PowgenExperimentSettings,
    PowgenWatchController,
    preflight_powgen_experiment,
)
from .results import (
    build_result_view,
    complete_experiment_space_groups,
    experiment_axis_options,
    experiment_fit_diagnostics,
    experiment_fit_quality_figure,
    experiment_phase_labels,
    experiment_phase_fraction_figure,
    experiment_phase_heatmap_figure,
    experiment_sample_identity,
    figure_for_payload,
    load_plot_with_fallback,
    phase_fraction_rows,
    read_plot_payload,
    read_table,
)
from .uploads import NamedFileUpload, NamedMultiCifUpload, build_cif_source_archive


_EASTERN_TIME = tzstr("EST5EDT,M3.2.0/2,M11.1.0/2")


def _format_eastern_time(value: Any, format_string: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
    """Format an ISO timestamp or datetime explicitly in US Eastern time."""

    if isinstance(value, datetime):
        timestamp = value
    else:
        text = str(value or "").strip()
        if not text:
            return "Not reported"
        # NeXus timestamps may carry nanoseconds while Python datetime accepts
        # at most microseconds. Preserve the offset and trim only extra digits.
        text = re.sub(r"(\.\d{6})\d+(?=(?:Z|[+-]\d{2}:?\d{2})?$)", r"\1", text)
        try:
            timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return text
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(_EASTERN_TIME).strftime(format_string)


def _application_version() -> str:
    """Read the source version shipped in this NOVA image."""
    override = os.environ.get("RADAR_PD_NOVA_VERSION", "").strip()
    if override:
        return override
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        match = re.search(
            r'^version\s*=\s*["\']([^"\']+)["\']',
            pyproject.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
    except OSError:
        match = None
    return match.group(1) if match else "unknown"


_SUBMISSION_FIELDS = (
    "analysis_mode", "radiation", "instrument_mode", "input_source", "instrument_source", "data_path", "instrument_path",
    "main_cif_path", "database_archive_path", "main_cif_source", "database_source", "library_archive_source", "event_file_path",
    "use_builtin_cuka", "ipts_instrument", "ipts", "run_number", "bank", "sample_elements",
    "environment_elements", "run_name", "fit_start", "fit_end", "ignore_regions", "reference_masks_enabled",
    "reference_mask_presets", "reference_window_mode", "reference_fixed_half_width", "reference_fwhm_factor",
    "reference_fractional_d_tolerance", "reference_zero_tolerance", "reference_min_half_width",
    "reference_max_half_width", "include_cu_kbeta", "background_mode", "background_type",
    "background_terms", "main_prenudge", "main_shadow_filter", "cleanup_enabled", "refine_u_iso",
    "refine_positions", "light_calibration_enabled", "magnetic_precheck", "magnetic_q_max",
    "magnetic_denominators", "full_profile",
    "full_max_passes", "full_min_phase_percent", "full_top_n_ml", "full_nudge_candidates", "full_nudge_samples",
    "full_nudge_representatives", "full_compare_candidates", "full_compare_cycles",
    "full_cell_length_tolerance_pct", "full_cell_angle_tolerance_deg", "full_rwp_improvement_threshold",
    "full_dedup_threshold", "full_score_q_max", "full_pearson_cell_min_r",
    "full_lattice_tiebreak_score_tol", "full_candidate_pruning", "full_knee_min_points_hist",
    "full_knee_min_relative_span", "full_knee_keep_if_no_knee", "full_knee_keep_at_most",
    "excluded_space_groups",
    "rapid_phases_per_hypothesis", "rapid_stage_output_limit", "rapid_gsas_validation_limit",
    "rapid_parallel_workers", "rapid_show_family_variants", "rapid_final_polish_enabled", "history_data_id",
    "history_instrument_id", "history_main_cif_id", "history_database_id",
    "remote_source", "remote_data_uri", "remote_instrument_uri", "remote_main_cif_uri",
    "use_facility_workspace", "facility_site", "facility_instrument", "facility_ipts",
    "facility_working_directory", "facility_data_path", "facility_data_relative_path",
    "facility_instrument_path", "facility_instrument_relative_path", "facility_main_cif_path",
    "facility_main_cif_relative_path", "publish_results_to_ipts", "facility_output_subfolder",
    "submission_token", "form_revision",
)


# Galaxy's upload and tool-submission endpoints become unreliable when a NOVA
# session opens a full five-scan batch simultaneously. Keep five analysis slots
# available, but feed them through two bounded submission lanes.
_POWGEN_SUBMISSION_CONCURRENCY = max(
    1,
    min(2, int(os.getenv("RADAR_PD_POWGEN_SUBMISSION_CONCURRENCY", "2"))),
)

_POWGEN_BACKFILL_LIMITS: dict[str, int | None] = {
    "new_only": 0,
    "latest_1": 1,
    "latest_5": 5,
    "latest_20": 20,
    "all": None,
}

_GENERAL_SOURCE_OPTIONS = [
    {"title": "Upload from this computer", "value": "upload"},
    {"title": "Choose from Galaxy History", "value": "galaxy"},
    {"title": "Browse an SNS/HFIR experiment folder", "value": "ipts_browser"},
]
_XRAY_SOURCE_OPTIONS = _GENERAL_SOURCE_OPTIONS[:2]
_GENERAL_INSTRUMENT_SOURCE_OPTIONS = [
    {"title": "Upload from this computer", "value": "upload"},
    {"title": "Choose from Galaxy History", "value": "galaxy"},
    {"title": "Choose from the experiment folder", "value": "ipts"},
]
_XRAY_INSTRUMENT_SOURCE_OPTIONS = _GENERAL_INSTRUMENT_SOURCE_OPTIONS[:2]
_GENERAL_MAIN_CIF_SOURCE_OPTIONS = [
    {"title": "No supplied main phase", "value": "none"},
    {"title": "Upload CIF from this computer", "value": "upload"},
    {"title": "Choose CIF from Galaxy History", "value": "galaxy"},
    {"title": "Choose CIF from the experiment folder", "value": "ipts"},
]
_XRAY_MAIN_CIF_SOURCE_OPTIONS = _GENERAL_MAIN_CIF_SOURCE_OPTIONS[:3]
_NEUTRON_GEOMETRY_OPTIONS = [
    {"title": "Auto detect", "value": "auto"},
    {"title": "Constant wavelength", "value": "cw"},
    {"title": "Time of flight", "value": "tof"},
]
_XRAY_GEOMETRY_OPTIONS = _NEUTRON_GEOMETRY_OPTIONS[:2]


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


def _powgen_submission_error_summary(value: Any) -> str:
    """Return a concise, non-HTML explanation for a watcher submission error."""

    text = str(value or "").strip()
    lowered = text.lower()
    if "503" in lowered or "service temporarily unavailable" in lowered:
        return "NDIP/Galaxy was temporarily unavailable (HTTP 503)."
    if "galaxy was restarted" in lowered or "killed when galaxy was restarted" in lowered:
        return "Galaxy restarted while the scan inputs were being submitted."
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "The submission service did not accept the scan."
    return text if len(text) <= 180 else f"{text[:177].rstrip()}..."


def _powgen_retry_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        retry_at = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return retry_at.astimezone(timezone.utc)


class RadarPdNovaApp(ThemedApp):
    """Interactive orchestration and results workspace backed by Galaxy."""

    def __init__(self) -> None:
        self.server = get_server(None, client_type="vue3")
        super().__init__(server=self.server)
        self.service = GalaxyService()
        self.facility = FacilityBrowser(facility="SNS")
        self.records: dict[str, RunRecord] = {}
        self.utility_actions: dict[str, UtilityActionRecord] = {}
        self._monitored_uids: set[str] = set()
        self._monitor_tasks: set[asyncio.Task[None]] = set()
        self._submission_tasks: set[asyncio.Task[None]] = set()
        self._utility_tasks: set[asyncio.Task[None]] = set()
        self._pending_submission_tokens: set[str] = set()
        self._history_search_task: asyncio.Task[None] | None = None
        self._powgen_controller: PowgenWatchController | None = None
        self._powgen_monitor_task: asyncio.Task[None] | None = None
        self._powgen_wake_event = asyncio.Event()
        self._powgen_settings_signature: tuple[str, ...] | None = None
        self._opening_run_uid: str | None = None
        self._plot_widget: Any | None = None
        self._primary_plot_widget: Any | None = None
        self._powgen_phase_widget: Any | None = None
        self._powgen_heatmap_widget: Any | None = None
        self._powgen_quality_widget: Any | None = None
        self._published_checkpoint_ids: dict[tuple[str, str], str] = {}
        self._dataset_label_cache: dict[str, str] = {}
        self._auto_opened_uids: set[str] = set()
        self._initialize_state()
        self.server.state.change("run_selection")(self._run_selection_changed)
        self.server.state.change(*_SUBMISSION_FIELDS[:-2])(self._submission_form_changed)
        self.server.state.change("workflow_mode")(self._workflow_mode_changed)
        self.server.state.change("radiation")(self._radiation_changed)
        self.server.state.change("facility_site")(self._facility_site_changed)
        self.server.state.change("facility_instrument")(self._facility_instrument_changed)
        self.server.state.change("facility_ipts")(self._facility_ipts_changed)
        self.server.state.change("remote_source")(self._remote_source_changed)
        self.create_ui()
        self.server.controller.on_server_ready.add(self._recover_runs)

    def _initialize_state(self) -> None:
        state = self.server.state
        state.trame__title = "RADAR-PD Interactive"
        state.application_version = f"v{_application_version()}"
        state.active_page = "setup"
        state.setup_collapsed = False
        # Vuetify expansion groups reconcile numeric panel values reliably
        # across Trame state flushes. Semantic string values can be replaced by
        # positional values in the browser, leaving a clicked panel stuck shut.
        state.setup_panels = [0, 2]
        state.workflow_mode = "single"
        state.workflow_options = [
            {"title": "Single pattern", "value": "single", "icon": "mdi-file-chart-outline"},
            {"title": "POWGEN live", "value": "powgen", "icon": "mdi-access-point"},
        ]
        state.workspace_view = "monitor"
        state.workspace_options = [
            {"title": "Run Monitor", "value": "monitor", "icon": "mdi-progress-clock"},
            {"title": "Rapid Results", "value": "results", "icon": "mdi-chart-line"},
            {"title": "Run File Browser", "value": "files", "icon": "mdi-folder-outline"},
        ]
        state.run_search = ""
        state.history_panels = []
        state.cancel_dialog = False
        state.connection_status = "Checking NDIP connection"
        state.connection_ok = False
        state.busy = False
        state.form_revision = 0
        state.submission_token = uuid.uuid4().hex
        state.notice = ""
        state.error_message = ""
        state.analysis_mode = "rapid"
        state.radiation = "neutron"
        state.last_radiation = "neutron"
        state.instrument_mode = "auto"
        state.instrument_mode_options = list(_NEUTRON_GEOMETRY_OPTIONS)
        state.input_source = "upload"
        state.instrument_source = "upload"
        state.data_path = ""
        state.instrument_path = ""
        state.main_cif_path = ""
        state.database_archive_path = ""
        state.config_import_path = ""
        state.main_cif_source = "none"
        state.database_source = "builtin"
        state.library_archive_source = "computer"
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
        state.reference_fixed_half_width = None
        state.reference_fwhm_factor = 6.0
        state.reference_fractional_d_tolerance = 0.003
        state.reference_zero_tolerance = None
        state.reference_min_half_width = None
        state.reference_max_half_width = None
        state.include_cu_kbeta = False
        state.background_mode = "auto_fixed_points"
        state.background_type = "chebyschev-1"
        state.background_terms = 6
        state.main_prenudge = True
        state.main_shadow_filter = True
        state.cleanup_enabled = False
        state.refine_u_iso = False
        state.refine_positions = False
        state.light_calibration_enabled = False
        state.magnetic_precheck = False
        state.magnetic_q_max = 4.0
        state.magnetic_denominators = "2, 3, 4"
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
        state.full_rwp_improvement_threshold = 0.06
        state.full_dedup_threshold = 0.95
        state.full_score_q_max = 8.0
        state.full_pearson_cell_min_r = 0.50
        state.full_lattice_tiebreak_score_tol = 0.0005
        state.full_candidate_pruning = True
        state.full_knee_min_points_hist = 5
        state.full_knee_min_relative_span = 0.03
        state.full_knee_keep_if_no_knee = 2
        state.full_knee_keep_at_most = 5
        state.excluded_space_groups = "1, 2"
        state.rapid_phases_per_hypothesis = 3
        state.rapid_stage_output_limit = 10
        state.rapid_gsas_validation_limit = 5
        state.rapid_parallel_workers = 4
        state.rapid_show_family_variants = True
        state.rapid_final_polish_enabled = False
        state.history_datasets = []
        state.history_search = ""
        state.history_offset = 0
        state.history_has_more = False
        state.history_show_all = False
        state.history_data_datasets = []
        state.history_instrument_datasets = []
        state.history_cif_datasets = []
        state.history_archive_datasets = []
        state.history_configuration_datasets = []
        state.history_data_id = ""
        state.history_instrument_id = ""
        state.history_main_cif_id = ""
        state.history_database_id = ""
        state.history_configuration_id = ""
        state.history_configuration_title = "No reusable configuration selected"
        state.history_configuration_summary = []
        state.history_configuration_yaml = ""
        state.history_configuration_error = ""
        state.run_rows = []
        state.run_selection = []
        state.selected_run_uid = ""
        state.selected_run_name = "No run selected"
        state.selected_run_status = "-"
        state.selected_analysis_status = "-"
        state.selected_result_status = "-"
        state.selected_publication_status = "-"
        state.selected_publication_target = ""
        state.selected_publication_job_id = ""
        state.selected_publish_message = ""
        state.result_explorer_available = False
        state.gsasii_launch_url = ""
        state.gsasii_session_status = ""
        state.gsasii_status_message = ""
        state.selected_galaxy_job_id = "Pending"
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
        state.viewed_output_profile = ""
        state.viewed_run_name = ""
        state.viewed_configuration = ""
        state.run_provenance_rows = []
        state.phase_rows = []
        state.table_options = []
        state.selected_table = ""
        state.table_rows = []
        state.table_headers = []
        state.table_preview_notice = ""
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
        state.file_search = ""
        state.show_technical_files = False
        state.activity_rows = []
        state.comparison_run_uids = []
        state.library_builder_cif_ids = []
        state.library_builder_local_paths = []
        state.library_builder_file_rows = []
        state.library_builder_mode = "mini"
        state.library_builder_name = "my_phase_library"
        state.library_builder_active = False
        state.library_builder_status = "idle"
        state.library_builder_message = "Add one or more CIF structures to begin."
        state.library_builder_progress = 0
        state.library_builder_built_count = 0
        state.library_builder_skipped_count = 0
        state.library_builder_failure_rows = []
        state.sns_resolution = {}
        state.use_facility_workspace = False
        state.facility_site = "SNS"
        state.facility_options = [
            {"title": "Spallation Neutron Source (SNS)", "value": "SNS"},
            {"title": "High Flux Isotope Reactor (HFIR)", "value": "HFIR"},
        ]
        state.facility_root = str(self.facility.root)
        state.facility_available = self.facility.available
        state.facility_instruments = self.facility.list_instruments()
        state.facility_ipts_options = []
        state.facility_instrument = ""
        state.facility_ipts = ""
        state.facility_working_directory = "."
        state.facility_working_subdirectory = ""
        state.facility_working_directories = []
        state.facility_working_readable = False
        state.facility_working_writable = False
        state.facility_data_directory = "."
        state.facility_data_subdirectory = ""
        state.facility_data_directories = []
        state.facility_data_files = []
        state.facility_data_path = ""
        state.facility_data_relative_path = ""
        state.facility_instrument_directory = "."
        state.facility_instrument_subdirectory = ""
        state.facility_instrument_directories = []
        state.facility_instrument_files = []
        state.facility_cif_directory = "."
        state.facility_cif_subdirectory = ""
        state.facility_cif_directories = []
        state.facility_cif_files = []
        state.facility_instrument_path = ""
        state.facility_instrument_relative_path = ""
        state.facility_main_cif_path = ""
        state.facility_main_cif_relative_path = ""
        state.publish_results_to_ipts = False
        state.facility_output_directory = "."
        state.facility_output_subdirectory = ""
        state.facility_output_directories = []
        state.facility_new_output_folder = "radar-pd-results"
        state.facility_output_subfolder = "radar-pd-results"
        state.facility_browser_message = ""
        state.remote_sources = []
        state.remote_source = ""
        state.remote_available = False
        state.remote_message = "Select Discover SNS sources to load file locations authorized for your NDIP account."
        state.remote_data_directory = ""
        state.remote_data_subdirectory = ""
        state.remote_data_directories = []
        state.remote_data_files = []
        state.remote_data_uri = ""
        state.remote_instrument_directory = ""
        state.remote_instrument_subdirectory = ""
        state.remote_instrument_directories = []
        state.remote_instrument_files = []
        state.remote_instrument_uri = ""
        state.remote_cif_directory = ""
        state.remote_cif_subdirectory = ""
        state.remote_cif_directories = []
        state.remote_cif_files = []
        state.remote_main_cif_uri = ""
        state.watch_enabled = False
        state.watch_patterns = "*.dat, *.xye, *.fxye, *.gsa, *.xrdml"
        state.watch_settle_seconds = 60
        state.watch_process_existing = False
        state.watch_max_attempts = 3
        state.watch_retry_delay_seconds = 120
        state.watch_recipe_message = ""
        state.powgen_monitoring = False
        state.powgen_ipts = ""
        state.powgen_backfill_mode = "latest_5"
        state.powgen_backfill_options = [
            {"title": "Latest 5 existing scans, then new scans", "value": "latest_5"},
            {"title": "Latest scan only, then new scans", "value": "latest_1"},
            {"title": "Latest 20 existing scans, then new scans", "value": "latest_20"},
            {"title": "New scans only", "value": "new_only"},
            {"title": "All existing scans, then new scans", "value": "all"},
        ]
        state.powgen_wavelength = ""
        state.powgen_wavelength_options = [
            {"title": "Auto-detect from latest scan (recommended)", "value": ""},
            {"title": "0.8 A (Bank 1)", "value": "0.8"},
            {"title": "1.5 A (Bank 2)", "value": "1.5"},
            {"title": "2.665 A (Bank 3)", "value": "2.665"},
        ]
        state.powgen_instrument_source = "automatic"
        state.powgen_instrument_source_options = [
            {"title": "Use the official cycle profile automatically", "value": "automatic"},
            {"title": "Upload a GSAS-II profile from this computer", "value": "computer"},
            {"title": "Choose a GSAS-II profile from Galaxy History", "value": "galaxy"},
        ]
        state.powgen_instrument_path = ""
        state.powgen_instrument_dataset_id = ""
        state.powgen_instrument_label = "Automatic POWGEN cycle profile"
        state.powgen_profile_error = ""
        state.powgen_configuration_dataset_id = ""
        state.powgen_configuration_title = "No reusable configuration selected"
        state.powgen_configuration_summary = []
        state.powgen_configuration_yaml = ""
        state.powgen_configuration_error = ""
        state.powgen_database_dataset_id = ""
        state.powgen_configuration_label = "Not selected"
        state.powgen_database_label = "Built-in MP/COD catalog"
        state.powgen_main_cif_source = "none"
        state.powgen_main_cif_path = ""
        state.powgen_main_cif_dataset_id = ""
        state.powgen_main_cif_source_options = [
            {"title": "No supplied main phase", "value": "none"},
            {"title": "Upload CIF from this computer", "value": "computer"},
            {"title": "Choose CIF from Galaxy History", "value": "galaxy"},
        ]
        state.powgen_source_directory = "/SNS/PG3/<IPTS>/shared/autoreduce"
        state.powgen_rows = []
        state.powgen_scientific_rows = []
        state.powgen_all_scientific_rows = []
        state.powgen_dashboard_metrics = []
        state.powgen_latest_phases = []
        state.powgen_latest_run_id = "-"
        state.powgen_scan_options = []
        state.powgen_selected_run_id = ""
        state.powgen_selected_scan = {}
        state.powgen_selected_quality_label = "Collecting baseline"
        state.powgen_selected_quality_color = "#eef2ef"
        state.powgen_selected_phases = []
        state.powgen_phase_options = []
        state.powgen_selected_phase_labels = []
        state.powgen_sample_options = []
        state.powgen_selected_samples = []
        state.powgen_x_axis = "run_number"
        state.powgen_x_axis_options = [{"title": "Scan number", "value": "run_number"}]
        state.powgen_tracked_phase_count = 0
        state.powgen_attention_count = 0
        state.powgen_dashboard_notice = "Completed scans will populate this experiment view."
        state.powgen_message = (
            "Select an IPTS and reusable Galaxy configuration. "
            "The wavelength is auto-detected unless you choose an override."
        )
        state.powgen_last_checked = "Not started"
        state.powgen_preflight_checked = "Not checked"
        state.powgen_next_check = "Not scheduled"
        state.powgen_preflight_status = "idle"
        state.powgen_preflight_ready = False
        state.powgen_preflight_message = "Check the experiment before starting monitoring."
        state.powgen_preflight_details = []
        state.powgen_poll_seconds = 15
        state.result_tab = "overview"
        state.mode_options = [
            {"title": "Rapid Hypothesis Mode", "value": "rapid"},
            {"title": "Full RADAR-PD", "value": "full"},
        ]
        state.source_options = list(_GENERAL_SOURCE_OPTIONS)
        state.instrument_source_options = list(_GENERAL_INSTRUMENT_SOURCE_OPTIONS)
        state.radiation_options = [
            {"title": "Neutron powder diffraction", "value": "neutron"},
            {"title": "X-ray powder diffraction", "value": "xray"},
        ]
        state.main_cif_source_options = list(_GENERAL_MAIN_CIF_SOURCE_OPTIONS)
        state.database_source_options = [
            {"title": "Built-in MP/COD catalog", "value": "builtin"},
            {"title": "Create a mini-library from my CIFs", "value": "custom_mini"},
            {"title": "Add my CIFs to the built-in MP/COD catalog", "value": "custom_augmented"},
            {"title": "Use an existing RADAR-PD library archive", "value": "archive"},
        ]
        state.library_archive_source_options = [
            {"title": "Upload archive from this computer", "value": "computer"},
            {"title": "Choose archive from Galaxy History", "value": "galaxy"},
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
                text=("application_version",),
                prepend_icon="mdi-source-branch",
                variant="outlined",
                size="small",
                title="Running RADAR-PD Interactive version",
            )
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

    def _facility_picker(
        self,
        *,
        title: str,
        directory_model: str,
        subdirectory_model: str,
        directory_items: str,
        file_model: str,
        file_items: str,
        open_action: Any,
        up_action: Any,
        select_action: Any,
        file_label: str,
    ) -> None:
        """Render one role-filtered, progressive IPTS shared-folder picker."""

        with html.Div(classes="radar-facility-picker"):
            html.Strong(title, classes="radar-facility-picker-title")
            with html.Div(classes="radar-facility-path-row"):
                vuetify.VTextField(
                    v_model=(directory_model,),
                    label="Current folder",
                    readonly=True,
                    density="compact",
                    variant="outlined",
                    hide_details=True,
                )
                vuetify.VBtn(
                    icon="mdi-arrow-up",
                    title="Go to parent folder",
                    click=up_action,
                    variant="outlined",
                    size="small",
                )
            with html.Div(classes="radar-facility-path-row mt-2"):
                vuetify.VSelect(
                    v_model=(subdirectory_model,),
                    items=(directory_items,),
                    item_title="title",
                    item_value="value",
                    label="Subfolder",
                    density="compact",
                    variant="outlined",
                    hide_details=True,
                    clearable=True,
                    no_data_text="No readable subfolders",
                )
                vuetify.VBtn(
                    icon="mdi-folder-open-outline",
                    title="Open selected folder",
                    click=open_action,
                    disabled=(f"!{subdirectory_model}",),
                    variant="outlined",
                    size="small",
                )
            vuetify.VSelect(
                v_model=(file_model,),
                items=(file_items,),
                item_title="title",
                item_value="value",
                label=file_label,
                density="compact",
                variant="outlined",
                clearable=True,
                no_data_text="No compatible files in this folder",
                update_modelValue=(select_action, "[$event]"),
                classes="mt-2",
            )

    def _facility_workspace_controls(self) -> None:
        """Render one lazy, IPTS-confined experiment working-folder browser."""

        with html.Div(
            v_show="input_source === 'ipts_browser' || instrument_source === 'ipts' || main_cif_source === 'ipts' || publish_results_to_ipts",
            classes="radar-facility-picker",
        ):
            html.Strong("Experiment working folder", classes="radar-facility-picker-title")
            html.P(
                "Select a neutron facility, a supported diffraction beamline, an IPTS, and the folder containing this analysis.",
                classes="radar-help-copy",
            )
            with html.Div(classes="radar-field-pair"):
                vuetify.VSelect(
                    label="Neutron facility",
                    v_model=("facility_site",),
                    items=("facility_options",),
                    item_title="title",
                    item_value="value",
                    density="compact",
                    variant="outlined",
                )
                vuetify.VSelect(
                    label="Diffraction beamline",
                    v_model=("facility_instrument",),
                    items=("facility_instruments",),
                    item_title="title",
                    item_value="value",
                    density="compact",
                    variant="outlined",
                    no_data_text="No supported diffraction beamlines are readable",
                )
            html.P(
                "Only powder and engineering diffraction beamlines supported by RADAR-PD are listed.",
                classes="radar-help-copy",
            )
            vuetify.VAlert(
                v_show="facility_site === 'SNS'",
                text=(
                    "POWGEN (PG3) live experiments use the dedicated POWGEN Live Experiment panel above. "
                    "This folder browser is for supported one-pattern neutron analyses."
                ),
                type="info",
                variant="tonal",
                density="compact",
                classes="mb-2",
            )
            vuetify.VCombobox(
                label="Experiment (IPTS)",
                v_model=("facility_ipts",),
                items=("facility_ipts_options",),
                item_title="title",
                item_value="value",
                density="compact",
                variant="outlined",
                hint="Select a recent IPTS or enter any IPTS number, for example IPTS-34537.",
                persistent_hint=True,
                no_data_text="Choose an instrument or enter an IPTS",
            )
            vuetify.VAlert(
                v_show="!facility_available",
                text=("'The ' + facility_site + ' filesystem is not mounted in this interactive session.'",),
                type="warning",
                variant="tonal",
                density="compact",
            )
            with html.Div(classes="radar-facility-path-row"):
                vuetify.VTextField(
                    v_model=("facility_working_directory",),
                    label="Current working folder",
                    readonly=True,
                    density="compact",
                    variant="outlined",
                    hide_details=True,
                )
                vuetify.VBtn(
                    icon="mdi-arrow-up",
                    title="Go to parent folder",
                    click=self.up_facility_working_directory,
                    disabled=("facility_working_directory === '.'",),
                    variant="outlined",
                    size="small",
                )
            with html.Div(classes="radar-facility-path-row mt-2"):
                vuetify.VSelect(
                    v_model=("facility_working_subdirectory",),
                    items=("facility_working_directories",),
                    item_title="title",
                    item_value="value",
                    label="Enter subfolder",
                    density="compact",
                    variant="outlined",
                    hide_details=True,
                    clearable=True,
                    no_data_text="No readable subfolders",
                    update_modelValue=(self.open_facility_working_directory, "[$event]"),
                )
            vuetify.VBtn(
                "Refresh folder",
                click=self.refresh_facility_browser,
                prepend_icon="mdi-refresh",
                variant="text",
                size="small",
                block=True,
            )
            vuetify.VAlert(
                v_show="!!facility_browser_message",
                text=("facility_browser_message",),
                type=("facility_working_readable ? 'info' : 'warning'",),
                variant="tonal",
                density="compact",
            )
            vuetify.VSwitch(
                v_model=("publish_results_to_ipts",),
                label="Export completed results into this working folder",
                hint="Galaxy History remains authoritative. NDIP exports the result archive under your authenticated facility identity.",
                persistent_hint=True,
                color="#15543c",
                density="compact",
                inset=True,
                disabled=("!facility_working_readable",),
            )
            vuetify.VAlert(
                v_show="publish_results_to_ipts",
                text="A uniquely named results ZIP will be written directly into the current working folder.",
                type="info",
                variant="tonal",
                density="compact",
            )

    def _remote_picker(
        self,
        *,
        title: str,
        directory_model: str,
        subdirectory_model: str,
        directory_items: str,
        file_model: str,
        file_items: str,
        open_action: Any,
        up_action: Any,
        select_action: Any,
        file_label: str,
    ) -> None:
        """Render a Galaxy-authorized remote-file picker."""

        with html.Div(classes="radar-facility-picker"):
            html.Strong(title, classes="radar-facility-picker-title")
            with html.Div(classes="radar-facility-path-row"):
                vuetify.VTextField(
                    v_model=(directory_model,),
                    label="Current remote folder",
                    readonly=True,
                    density="compact",
                    variant="outlined",
                    hide_details=True,
                )
                vuetify.VBtn(
                    icon="mdi-arrow-up",
                    title="Go to parent folder",
                    click=up_action,
                    variant="outlined",
                    size="small",
                )
            with html.Div(classes="radar-facility-path-row mt-2"):
                vuetify.VSelect(
                    v_model=(subdirectory_model,),
                    items=(directory_items,),
                    item_title="title",
                    item_value="value",
                    label="Subfolder",
                    density="compact",
                    variant="outlined",
                    hide_details=True,
                    clearable=True,
                    no_data_text="No readable subfolders",
                )
                vuetify.VBtn(
                    icon="mdi-folder-open-outline",
                    title="Open selected folder",
                    click=open_action,
                    disabled=(f"!{subdirectory_model}",),
                    variant="outlined",
                    size="small",
                )
            vuetify.VSelect(
                v_model=(file_model,),
                items=(file_items,),
                item_title="title",
                item_value="value",
                label=file_label,
                density="compact",
                variant="outlined",
                clearable=True,
                no_data_text="No compatible files in this folder",
                update_modelValue=(select_action, "[$event]"),
                classes="mt-2",
            )

    def _remote_source_controls(self) -> None:
        with html.Div(classes="radar-field-pair"):
            vuetify.VSelect(
                label="Authorized file source",
                v_model=("remote_source",),
                items=("remote_sources",),
                item_title="title",
                item_value="value",
                density="compact",
                variant="outlined",
                no_data_text="Discover sources first",
            )
            vuetify.VBtn(
                "Discover SNS sources",
                click=self.discover_remote_sources,
                prepend_icon="mdi-cloud-search-outline",
                variant="outlined",
                color="#15543c",
            )
        vuetify.VAlert(
            v_show="!!remote_message",
            text=("remote_message",),
            type=("remote_available ? 'info' : 'warning'",),
            variant="tonal",
            density="compact",
            classes="mb-2",
        )

    @contextmanager
    def _setup_section(
        self,
        number: int,
        title: str,
        value: int,
        status_expression: str,
        *,
        v_show: str | None = None,
        display_number: str | None = None,
    ) -> Any:
        panel_options: dict[str, Any] = {
            "value": (str(value),),
            "key": f"'setup-{number}'",
            "classes": "radar-setup-panel",
        }
        if v_show:
            panel_options["v_show"] = v_show
        with vuetify.VExpansionPanel(**panel_options):
            with vuetify.VExpansionPanelTitle(classes="radar-setup-title"):
                if display_number:
                    html.Span(f"{{{{ {display_number} }}}}", classes="radar-step-number")
                else:
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

    @staticmethod
    def _number_field(*children: Any, **kwargs: Any) -> Any:
        """Create a keyboard-editable numeric field without browser wheel stepping."""

        kwargs.setdefault("type", "text")
        kwargs.setdefault("inputmode", "decimal")
        return vuetify.VTextField(*children, **kwargs)

    def _setup_page(self) -> None:
        html.Div("SETUP", classes="radar-rail-kicker")
        html.H2("Configure analysis", classes="radar-rail-heading")
        html.P("Work from top to bottom. Earlier choices control the options that follow.", classes="radar-rail-help")

        html.Div("WORKFLOW", classes="radar-rail-kicker mt-2 mb-1")
        with vuetify.VBtnToggle(
            v_model=("workflow_mode",),
            mandatory=True,
            divided=True,
            color="#15543c",
            classes="radar-workflow-toggle mb-3",
            disabled=("powgen_monitoring",),
        ):
            with vuetify.VBtn(
                v_for="item in workflow_options",
                key="item.value",
                value=("item.value",),
                size="small",
                click="history_panels = []; run_search = ''",
            ):
                vuetify.VIcon(icon=("item.icon",), size="small", classes="mr-1")
                html.Span("{{ item.title }}")

        with vuetify.VExpansionPanels(
            v_model=("history_panels",),
            multiple=True,
            variant="accordion",
            classes="radar-history-panel mb-3",
        ):
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
                            v_for="run in run_rows.filter(item => !run_search || (item.name + ' ' + item.mode + ' ' + item.status + ' ' + item.created_display + ' ' + item.job_id).toLowerCase().includes(run_search.toLowerCase()))",
                            key="run.uid",
                            click=(self._run_selection_changed, "[run.uid]"),
                            classes="radar-run-list-item",
                        ):
                            html.Div("{{ run.name }}", classes="radar-run-list-name")
                            html.Div("{{ run.mode }} / {{ run.status }} · {{ run.created_display }} · job {{ run.job_short }}", classes="radar-run-list-meta")
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
                    vuetify.VSelect(
                        label="Compare completed runs",
                        v_model=("comparison_run_uids",),
                        items=("run_rows.filter(item => item.status === 'Ok')",),
                        item_title="display_name",
                        item_value="uid",
                        multiple=True,
                        chips=True,
                        closable_chips=True,
                        density="compact",
                        variant="outlined",
                        hint="Select 2–50 runs for a Galaxy series comparison.",
                        persistent_hint=True,
                        classes="mt-3",
                    )
                    vuetify.VBtn(
                        "Compare selected series",
                        click=self.compare_selected_runs,
                        disabled=("comparison_run_uids.length < 2 || comparison_run_uids.length > 50",),
                        prepend_icon="mdi-chart-timeline-variant",
                        variant="outlined",
                        size="small",
                        block=True,
                    )

        with vuetify.VExpansionPanels(
            model_value="powgen-live",
            variant="accordion",
            classes="radar-history-panel mb-3",
            v_show="workflow_mode === 'powgen'",
        ):
            with vuetify.VExpansionPanel(value="powgen-live"):
                with vuetify.VExpansionPanelTitle():
                    vuetify.VIcon("mdi-access-point", size="small", classes="mr-2")
                    html.Span("POWGEN Live Experiment")
                    vuetify.VSpacer()
                    vuetify.VChip(
                        text=("powgen_monitoring ? 'Monitoring' : 'Stopped'",),
                        color=("powgen_monitoring ? '#dff2e8' : '#eef2ef'",),
                        size="x-small",
                        variant="flat",
                    )
                with vuetify.VExpansionPanelText():
                    vuetify.VAlert(
                        text=(
                            "Dedicated neutron time-of-flight workflow. Only the inputs in this panel control "
                            "live monitoring. Use the workflow selector above to return to one-pattern analysis."
                        ),
                        type="info",
                        variant="tonal",
                        density="compact",
                        classes="mb-2",
                    )
                    html.P(
                        "Analyze every completed POWGEN reduction already present, then continue with each new scan through the existing RADAR-PD Analyze tool.",
                        classes="radar-help-copy",
                    )
                    vuetify.VTextField(
                        label="POWGEN experiment (IPTS)",
                        v_model=("powgen_ipts",),
                        placeholder="IPTS-38000",
                        density="compact",
                        variant="outlined",
                        disabled=("powgen_monitoring",),
                        hint="Enter the IPTS that owns the PG3 experiment.",
                        persistent_hint=True,
                    )
                    vuetify.VSelect(
                        label="Scans to process when monitoring starts",
                        v_model=("powgen_backfill_mode",),
                        items=("powgen_backfill_options",),
                        item_title="title",
                        item_value="value",
                        density="compact",
                        variant="outlined",
                        disabled=("powgen_monitoring",),
                        hint="Latest 5 is the safe default for a live demonstration. Every later scan is still monitored.",
                        persistent_hint=True,
                    )
                    vuetify.VSelect(
                        label="POWGEN wavelength",
                        v_model=("powgen_wavelength",),
                        items=("powgen_wavelength_options",),
                        item_title="title",
                        item_value="value",
                        density="compact",
                        variant="outlined",
                        disabled=("powgen_monitoring",),
                        hint="Auto-detect is recommended. The three explicit choices are the supported POWGEN bank/wavelength modes.",
                        persistent_hint=True,
                    )
                    with html.Div(
                        v_show="!!powgen_profile_error || powgen_instrument_source !== 'automatic'",
                        classes="radar-soft-panel mb-2",
                    ):
                        vuetify.VAlert(
                            v_show="!!powgen_profile_error",
                            text=("'Automatic profile unavailable: ' + powgen_profile_error",),
                            type="warning",
                            variant="tonal",
                            density="compact",
                            classes="mb-2",
                        )
                        vuetify.VSelect(
                            label="POWGEN instrument profile fallback",
                            v_model=("powgen_instrument_source",),
                            items=("powgen_instrument_source_options",),
                            item_title="title",
                            item_value="value",
                            density="compact",
                            variant="outlined",
                            disabled=("powgen_monitoring",),
                            hint=(
                                "Use the GSAS-II parameter file for the experiment cycle. "
                                "It can be downloaded from POWGEN or found under "
                                "/SNS/PG3/run_cycle_<cycle>_CAL on an analysis system."
                            ),
                            persistent_hint=True,
                        )
                        with html.Div(
                            v_show="powgen_instrument_source === 'computer'",
                            key="'powgen-instrument-upload-panel'",
                        ):
                            NamedFileUpload(
                                "powgen_instrument_path",
                                label="GSAS-II instrument profile",
                                help_text="This file is saved to Galaxy History and reused for every monitored scan.",
                                extensions=[".instprm", ".prm", ".inst", ".ins"],
                                key="powgen-instrument-upload",
                            )
                        vuetify.VAutocomplete(
                            v_show="powgen_instrument_source === 'galaxy'",
                            label="Instrument profile from Galaxy History",
                            v_model=("powgen_instrument_dataset_id",),
                            items=("history_instrument_datasets",),
                            item_title="display_name",
                            item_value="id",
                            density="compact",
                            variant="outlined",
                            disabled=("powgen_monitoring",),
                            clearable=True,
                            no_data_text="No GSAS-II instrument profiles are in this Galaxy History",
                        )
                    vuetify.VAutocomplete(
                        label="Reusable RADAR-PD configuration",
                        v_model=("powgen_configuration_dataset_id",),
                        items=("history_configuration_datasets",),
                        item_title="display_name",
                        item_value="id",
                        density="compact",
                        variant="outlined",
                        disabled=("powgen_monitoring",),
                        no_data_text="No reusable configurations are in this Galaxy History",
                        hint=(
                            "Applies analysis settings only. POWGEN takes scans, its instrument profile, "
                            "candidate library, and optional main phase from this panel."
                        ),
                        persistent_hint=True,
                        update_modelValue=(self._powgen_configuration_changed, "[$event]"),
                    )
                    with vuetify.VExpansionPanels(
                        v_show="!!powgen_configuration_dataset_id",
                        variant="accordion",
                        classes="radar-config-import mb-2",
                    ):
                        with vuetify.VExpansionPanel(title=("powgen_configuration_title",)):
                            with vuetify.VExpansionPanelText():
                                vuetify.VAlert(
                                    v_show="!!powgen_configuration_error",
                                    text=("powgen_configuration_error",),
                                    type="warning",
                                    variant="tonal",
                                    density="compact",
                                    classes="mb-2",
                                )
                                with html.Div(classes="radar-preflight-summary"):
                                    with html.Div(
                                        v_for="item in powgen_configuration_summary",
                                        key="item.label",
                                        classes="radar-preflight-row",
                                    ):
                                        html.Span("{{ item.label }}", classes="radar-preflight-label")
                                        html.Span("{{ item.value }}", classes="radar-preflight-value")
                                html.Pre(
                                    "{{ powgen_configuration_yaml }}",
                                    v_show="!!powgen_configuration_yaml",
                                    classes="radar-config-preview mt-2",
                                )
                    vuetify.VAutocomplete(
                        label="Candidate library for monitored scans",
                        v_model=("powgen_database_dataset_id",),
                        items=("history_archive_datasets",),
                        item_title="display_name",
                        item_value="id",
                        density="compact",
                        variant="outlined",
                        disabled=("powgen_monitoring",),
                        clearable=True,
                        no_data_text="No custom RADAR-PD libraries are in this Galaxy History",
                        hint="Leave empty for the built-in catalog. To add CIFs, build or upload a reusable library under Single pattern first, then select it here for every monitored scan.",
                        persistent_hint=True,
                    )
                    vuetify.VSelect(
                        label="Known/main phase (optional)",
                        v_model=("powgen_main_cif_source",),
                        items=("powgen_main_cif_source_options",),
                        item_title="title",
                        item_value="value",
                        density="compact",
                        variant="outlined",
                        disabled=("powgen_monitoring",),
                        hint="Supply one dominant structural phase when it is already known.",
                        persistent_hint=True,
                    )
                    with html.Div(
                        v_show="powgen_main_cif_source === 'computer'",
                        key="'powgen-main-cif-upload-panel'",
                    ):
                        NamedFileUpload(
                            "powgen_main_cif_path",
                            label="Known/main-phase CIF",
                            help_text="The CIF will be saved to Galaxy History and reused for every monitored scan.",
                            extensions=[".cif"],
                            optional=True,
                            key="powgen-main-cif-upload",
                        )
                    vuetify.VAutocomplete(
                        v_show="powgen_main_cif_source === 'galaxy'",
                        label="Main-phase CIF from Galaxy History",
                        v_model=("powgen_main_cif_dataset_id",),
                        items=("history_cif_datasets",),
                        item_title="display_name",
                        item_value="id",
                        density="compact",
                        variant="outlined",
                        disabled=("powgen_monitoring",),
                        clearable=True,
                        no_data_text="No CIF datasets are in this Galaxy History",
                    )
                    vuetify.VBtn(
                        "Check experiment and inputs",
                        click=self.check_powgen_experiment,
                        prepend_icon="mdi-clipboard-check-outline",
                        variant="outlined",
                        size="small",
                        block=True,
                        classes="mb-2",
                        disabled=(
                            "powgen_monitoring || !powgen_ipts || "
                            "!powgen_configuration_dataset_id || powgen_preflight_status === 'checking'",
                        ),
                    )
                    vuetify.VAlert(
                        v_show="powgen_preflight_status !== 'idle'",
                        text=("powgen_preflight_message",),
                        type=("powgen_preflight_ready ? 'success' : powgen_preflight_status === 'checking' ? 'info' : 'warning'",),
                        variant="tonal",
                        density="compact",
                        classes="mb-2",
                    )
                    with html.Div(
                        v_if="powgen_preflight_details.length",
                        classes="radar-preflight-summary mb-2",
                    ):
                        with html.Div(
                            v_for="item in powgen_preflight_details",
                            key="item.label",
                            classes="radar-preflight-row",
                        ):
                            html.Span("{{ item.label }}", classes="radar-preflight-label")
                            html.Span("{{ item.value }}", classes="radar-preflight-value")
                    with html.Div(classes="radar-soft-panel mb-2"):
                        html.Div("READ-ONLY SOURCE", classes="radar-micro-label")
                        html.Code("{{ powgen_source_directory }}")
                        html.P(
                            "Only the selected initial scan range is backfilled. Every later completed .gsa is monitored. Up to five analyses stay active while new submissions use two protected lanes. RADAR-PD does not write into the IPTS.",
                            classes="radar-help-copy mb-0",
                        )
                    vuetify.VAlert(
                        text="This live monitor runs only while this NOVA session is open. Use NDIP Ingress for permanent or unattended background triggering.",
                        type="info",
                        variant="tonal",
                        density="compact",
                        classes="mb-2",
                    )
                    with html.Div(classes="radar-button-row"):
                        vuetify.VBtn(
                            "Start monitoring",
                            click=self.start_powgen_monitoring,
                            prepend_icon="mdi-play-circle-outline",
                            color="#15543c",
                            variant="flat",
                            size="small",
                            disabled=("powgen_monitoring || !powgen_preflight_ready",),
                        )
                        vuetify.VBtn(
                            "Stop",
                            click=self.stop_powgen_monitoring,
                            prepend_icon="mdi-stop-circle-outline",
                            color="#9b2c2c",
                            variant="outlined",
                            size="small",
                            disabled=("!powgen_monitoring",),
                        )
                        vuetify.VBtn(
                            "Refresh inputs",
                            click=self.refresh_history,
                            prepend_icon="mdi-refresh",
                            variant="text",
                            size="small",
                            disabled=("powgen_monitoring",),
                        )
                        vuetify.VBtn(
                            "Refresh now",
                            click=self.refresh_powgen_monitor_now,
                            prepend_icon="mdi-refresh",
                            variant="text",
                            size="small",
                            disabled=("!powgen_monitoring",),
                        )
                    vuetify.VBtn(
                        "Open experiment dashboard",
                        v_show="powgen_rows.length",
                        click=self.show_powgen_dashboard,
                        prepend_icon="mdi-chart-timeline-variant",
                        variant="outlined",
                        size="small",
                        block=True,
                        classes="mt-2",
                    )
                    html.P("{{ powgen_message }}", classes="radar-help-copy mt-2 mb-1")
                    html.Div("Inputs checked: {{ powgen_preflight_checked }}", classes="radar-run-list-meta")
                    html.Div("Monitor last polled: {{ powgen_last_checked }}", classes="radar-run-list-meta")
                    html.Div("Next check: {{ powgen_next_check }}", classes="radar-run-list-meta")
                    with html.Div(classes="radar-run-list mt-2", v_if="powgen_rows.length"):
                        with html.Div(
                            v_for="row in powgen_rows",
                            key="row.run_id",
                            classes="radar-run-list-item",
                        ):
                            with html.Div(classes="d-flex align-center ga-2"):
                                html.Div("{{ row.run_id }}", classes="radar-run-list-name")
                                vuetify.VChip(
                                    text=("row.status",),
                                    color=("row.color",),
                                    size="x-small",
                                    variant="flat",
                                )
                            html.Div("{{ row.file }}", classes="radar-run-list-meta")
                            html.Div("{{ row.detail }}", classes="radar-run-list-stage")

        html.Div(
            "SINGLE-PATTERN ANALYSIS",
            classes="radar-rail-kicker mb-1",
            v_show="workflow_mode === 'single'",
        )
        with vuetify.VExpansionPanels(
            v_model=("setup_panels",),
            multiple=True,
            variant="accordion",
            classes="radar-setup-panels",
            v_show="workflow_mode === 'single'",
        ):
            with self._setup_section(1, "Measurement Type", 0, "!!radiation && !!instrument_mode"):
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
                    items=("instrument_mode_options",),
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
                1,
                "database_source === 'builtin' || (database_source === 'archive' && ((library_archive_source === 'computer' && !!database_archive_path) || (library_archive_source === 'galaxy' && !!history_database_id)))",
            ):
                vuetify.VSelect(
                    label="Which candidate library should RADAR-PD search?",
                    v_model=("database_source",),
                    items=("database_source_options",),
                    item_title="title",
                    item_value="value",
                    density="compact",
                    variant="outlined",
                    hint="Choose the scientific search scope first. CIF and ZIP uploads are equivalent ways to supply custom structures.",
                    persistent_hint=True,
                )
                with html.Div(v_show="database_source === 'archive'", key="'radar-library-archive-panel'", classes="radar-library-panel"):
                    vuetify.VSelect(
                        label="Where is the reusable library archive?",
                        v_model=("library_archive_source",),
                        items=("library_archive_source_options",),
                        item_title="title",
                        item_value="value",
                        density="compact",
                        variant="outlined",
                    )
                    with html.Div(v_show="library_archive_source === 'computer'"):
                        NamedFileUpload(
                            "database_archive_path",
                            label="RADAR-PD library archive",
                            help_text="A library previously produced by the RADAR-PD library builder (.zip)",
                            extensions=[".zip"],
                            optional=False,
                            key="radar-database-upload",
                        )
                    vuetify.VAutocomplete(
                        v_show="library_archive_source === 'galaxy'",
                        label="Galaxy library archive",
                        v_model=("history_database_id",),
                        items=("history_archive_datasets",),
                        item_title="display_name",
                        item_value="id",
                        density="compact",
                        variant="outlined",
                        no_data_text="No library archives are in this history",
                    )
                with html.Div(
                    v_show="database_source === 'custom_mini' || database_source === 'custom_augmented'",
                    key="'radar-library-builder-panel'",
                    classes="radar-library-panel",
                ):
                    with html.Div(classes="radar-library-scope-callout"):
                        html.Strong("{{ database_source === 'custom_augmented' ? 'Built-in MP/COD + my structures' : 'Only my structures' }}")
                        html.Span(
                            "{{ database_source === 'custom_augmented' "
                            "? 'Your CIFs will augment the full built-in catalog.' "
                            ": 'RADAR-PD will search only the compact library built from your CIFs.' }}"
                        )
                    vuetify.VTextField(
                        label="Library name",
                        v_model=("library_builder_name",),
                        density="compact",
                        variant="outlined",
                        hint="A short scientific name, for example Y_Fe_Si_candidate_phases.",
                        persistent_hint=True,
                    )
                    NamedMultiCifUpload("library_builder_local_paths", "library_builder_file_rows")
                    html.Div("Optional: add CIF datasets already saved in Galaxy", classes="radar-inline-divider")
                    vuetify.VAutocomplete(
                        label="CIFs from Galaxy History",
                        v_model=("library_builder_cif_ids",),
                        items=("history_cif_datasets",),
                        item_title="display_name",
                        item_value="id",
                        multiple=True,
                        chips=True,
                        closable_chips=True,
                        density="compact",
                        variant="outlined",
                    )
                    with html.Div(classes="radar-library-build-status", v_show="library_builder_status !== 'idle' || library_builder_file_rows.length > 0"):
                        html.Strong(
                            "{{ library_builder_status === 'idle' "
                            "? library_builder_file_rows.filter(item => item.status === 'Ready').reduce((sum,item) => sum + (item.cif_count || 0), 0) "
                            "+ library_builder_cif_ids.length + ' structure source(s) ready to build.' "
                            ": library_builder_message }}"
                        )
                        vuetify.VProgressLinear(model_value=("library_builder_progress",), color="#15543c", height=6, rounded=True, classes="mt-2")
                        html.Span(
                            "{{ library_builder_built_count }} usable / {{ library_builder_skipped_count }} skipped",
                            v_show="library_builder_status === 'ready' || library_builder_status === 'partial'",
                        )
                        with html.Div(
                            v_for="failure in library_builder_failure_rows.slice(0, 8)",
                            key="failure.name + failure.reason",
                            classes="radar-library-failure-row",
                        ):
                            html.Strong("{{ failure.name }}")
                            html.Span("{{ failure.reason }}")
                    vuetify.VBtn(
                        text=("library_builder_active ? 'Building library...' : 'Build and use this library'",),
                        click=self.build_candidate_library,
                        disabled=("library_builder_active || (!library_builder_cif_ids.length && !library_builder_local_paths.length) || !library_builder_name.trim()",),
                        loading=("library_builder_active",),
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
            instrument_ready = "((radiation === 'xray' && use_builtin_cuka) || (instrument_source === 'upload' && !!instrument_path) || (instrument_source === 'galaxy' && !!history_instrument_id) || (instrument_source === 'galaxy_remote' && !!remote_instrument_uri) || (instrument_source === 'ipts' && !!facility_instrument_path))"
            data_ready = f"((input_source === 'upload' && !!data_path) || (input_source === 'galaxy' && !!history_data_id) || (input_source === 'galaxy_remote' && !!remote_data_uri) || (input_source === 'ipts_browser' && !!facility_data_path)) && {instrument_ready} || (input_source === 'ipts_event' && !!event_file_path && !!bank) || (input_source === 'ipts_manual' && !!ipts_instrument && !!ipts && !!run_number && !!bank)"
            main_phase_ready = "((main_cif_source === 'upload' && !!main_cif_path) || (main_cif_source === 'galaxy' && !!history_main_cif_id) || (main_cif_source === 'galaxy_remote' && !!remote_main_cif_uri) || (main_cif_source === 'ipts' && !!facility_main_cif_path))"
            with self._setup_section(3, "Data Collection", 2, data_ready):
                vuetify.VSelect(
                    label="Where is the diffraction pattern?",
                    v_model=("input_source",),
                    items=("source_options",),
                    item_title="title",
                    item_value="value",
                    density="compact",
                    variant="outlined",
                    hint="Upload a file, reuse a Galaxy dataset, or browse an experiment folder available to your NDIP account.",
                    persistent_hint=True,
                )
                self._facility_workspace_controls()
                with html.Div(v_show="input_source === 'upload'", key="'radar-computer-inputs'"):
                    NamedFileUpload(
                        "data_path",
                        label="Diffraction data",
                        help_text="Required measurement pattern",
                        extensions=[
                            ".dat",
                            ".xye",
                            ".xy",
                            ".csv",
                            ".txt",
                            ".fxye",
                            ".gsa",
                            ".gsas",
                            ".gss",
                            ".xrdml",
                            ".xml",
                        ],
                        key="radar-diffraction-upload",
                    )
                with html.Div(v_show="input_source === 'galaxy'", key="'radar-history-inputs'"):
                    vuetify.VTextField(
                        label="Search Galaxy History",
                        v_model=("history_search",),
                        prepend_inner_icon="mdi-magnify",
                        density="compact",
                        variant="outlined",
                        clearable=True,
                        hint="Search is performed by Galaxy; the newest 100 relevant inputs load initially.",
                        persistent_hint=True,
                        update_modelValue=(self.search_history, "[$event]"),
                    )
                    vuetify.VSwitch(
                        v_model=("history_show_all",),
                        label="Include RADAR-PD input copies",
                        density="compact",
                        color="#15543c",
                        hint="Includes reusable diffraction inputs produced by earlier RADAR-PD runs; logs and result artifacts remain excluded.",
                        persistent_hint=True,
                        update_modelValue=(self.search_history, "[history_search]"),
                    )
                    vuetify.VAutocomplete(
                        label="Diffraction data from History",
                        v_model=("history_data_id",),
                        items=("history_data_datasets",),
                        item_title="display_name",
                        item_value="id",
                        density="compact",
                        variant="outlined",
                        no_data_text="No compatible diffraction datasets",
                        key="radar-history-diffraction",
                    )
                    vuetify.VBtn(
                        "Load more",
                        v_show="history_has_more",
                        click=self.load_more_history,
                        prepend_icon="mdi-chevron-down",
                        variant="text",
                        size="small",
                        block=True,
                    )
                with html.Div(v_show="input_source === 'galaxy_remote'", key="'radar-remote-inputs'"):
                    self._remote_source_controls()
                    self._remote_picker(
                        title="Diffraction data in the experiment filesystem",
                        directory_model="remote_data_directory",
                        subdirectory_model="remote_data_subdirectory",
                        directory_items="remote_data_directories",
                        file_model="remote_data_uri",
                        file_items="remote_data_files",
                        open_action=self.open_remote_data_directory,
                        up_action=self.up_remote_data_directory,
                        select_action=self.select_remote_data,
                        file_label="Measured powder pattern",
                    )
                with html.Div(v_show="input_source === 'ipts_browser'", key="'radar-facility-inputs'"):
                    vuetify.VSelect(
                        label="Diffraction data in working folder",
                        v_model=("facility_data_relative_path",),
                        items=("facility_data_files",),
                        item_title="title",
                        item_value="value",
                        density="compact",
                        variant="outlined",
                        clearable=True,
                        no_data_text="No compatible diffraction files in this folder",
                        update_modelValue=(self.select_facility_data, "[$event]"),
                    )
                vuetify.VSwitch(
                    v_show="radiation === 'xray' && instrument_mode !== 'tof' && (input_source === 'upload' || input_source === 'galaxy' || input_source === 'galaxy_remote' || input_source === 'ipts_browser')",
                    v_model=("use_builtin_cuka",),
                    label="Use built-in Cu K-alpha profile",
                    color="#15543c",
                    density="compact",
                    inset=True,
                )
                vuetify.VAlert(
                    v_show="radiation === 'xray' && use_builtin_cuka",
                    text="The packaged Cu K-alpha GSAS-II profile will be used; no instrument file is required.",
                    type="info",
                    variant="tonal",
                    density="compact",
                    classes="mb-2",
                )
                with html.Div(
                    v_show="(input_source === 'upload' || input_source === 'galaxy' || input_source === 'galaxy_remote' || input_source === 'ipts_browser') && !(radiation === 'xray' && use_builtin_cuka)",
                    key="'radar-independent-instrument-profile'",
                ):
                    vuetify.VSelect(
                        label="Where is the GSAS-II instrument profile?",
                        v_model=("instrument_source",),
                        items=("instrument_source_options",),
                        item_title="title",
                        item_value="value",
                        density="compact",
                        variant="outlined",
                        hint="This file can come from a different source than the diffraction pattern.",
                        persistent_hint=True,
                    )
                    with html.Div(v_show="instrument_source === 'upload'", key="'radar-independent-instrument-upload'"):
                        NamedFileUpload(
                            "instrument_path",
                            label="GSAS-II instrument profile",
                            help_text="Instrument geometry and peak-profile parameters",
                            extensions=[".instprm", ".prm", ".inst", ".ins"],
                            key="radar-instrument-upload",
                        )
                    vuetify.VAutocomplete(
                        v_show="instrument_source === 'galaxy'",
                        label="Instrument profile from History",
                        v_model=("history_instrument_id",),
                        items=("history_instrument_datasets",),
                        item_title="display_name",
                        item_value="id",
                        density="compact",
                        variant="outlined",
                        no_data_text="No GSAS-II instrument profiles",
                        key="radar-history-instrument",
                    )
                    with html.Div(v_show="instrument_source === 'galaxy_remote'", key="'radar-remote-instrument'"):
                        with html.Div(v_show="input_source !== 'galaxy_remote'"):
                            self._remote_source_controls()
                        self._remote_picker(
                            title="Instrument profile in the experiment filesystem",
                            directory_model="remote_instrument_directory",
                            subdirectory_model="remote_instrument_subdirectory",
                            directory_items="remote_instrument_directories",
                            file_model="remote_instrument_uri",
                            file_items="remote_instrument_files",
                            open_action=self.open_remote_instrument_directory,
                            up_action=self.up_remote_instrument_directory,
                            select_action=self.select_remote_instrument,
                            file_label="GSAS-II instrument profile",
                        )
                    with html.Div(v_show="instrument_source === 'ipts'"):
                        vuetify.VSelect(
                            label="Instrument profile in working folder",
                            v_model=("facility_instrument_relative_path",),
                            items=("facility_instrument_files",),
                            item_title="title",
                            item_value="value",
                            density="compact",
                            variant="outlined",
                            clearable=True,
                            no_data_text="No compatible instrument profiles in this folder",
                            update_modelValue=(self.select_facility_instrument, "[$event]"),
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
                        self._number_field(label="Run number", v_model=("run_number",), density="compact", variant="outlined")
                        vuetify.VTextField(label="Detector bank", v_model=("bank",), density="compact", variant="outlined")
                vuetify.VBtn(
                    "Resolve and verify SNS input",
                    v_show="input_source === 'ipts_event' || input_source === 'ipts_manual'",
                    click=self.resolve_sns_input,
                    prepend_icon="mdi-database-search-outline",
                    color="#15543c",
                    variant="outlined",
                    block=True,
                    classes="mb-2",
                )
                vuetify.VAlert(
                    v_show="sns_resolution && sns_resolution.status === 'ready'",
                    text=("'Resolved ' + sns_resolution.pattern + ' / ' + sns_resolution.profile + ' / bank ' + sns_resolution.bank",),
                    type="success",
                    variant="tonal",
                    density="compact",
                    classes="mb-2",
                )
                vuetify.VDivider(classes="my-3")
                vuetify.VSelect(
                    label="Do you have a known/main-phase CIF?",
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
                with html.Div(
                    v_show="main_cif_source === 'ipts'",
                    key="'radar-independent-facility-cif'",
                ):
                    vuetify.VSelect(
                        label="Known/main-phase CIF in working folder",
                        v_model=("facility_main_cif_relative_path",),
                        items=("facility_cif_files",),
                        item_title="title",
                        item_value="value",
                        density="compact",
                        variant="outlined",
                        clearable=True,
                        no_data_text="No CIF files in this folder",
                        update_modelValue=(self.select_facility_main_cif, "[$event]"),
                    )
                vuetify.VAutocomplete(
                    v_show="main_cif_source === 'galaxy'",
                    label="Main-phase CIF from History",
                    v_model=("history_main_cif_id",),
                    items=("history_cif_datasets",),
                    item_title="display_name",
                    item_value="id",
                    clearable=True,
                    density="compact",
                    variant="outlined",
                    no_data_text="No CIF datasets are in this history",
                    key="'radar-history-main-cif'",
                )
                with html.Div(v_show="main_cif_source === 'galaxy_remote'", key="'radar-remote-main-cif'"):
                    with html.Div(v_show="input_source !== 'galaxy_remote' && instrument_source !== 'galaxy_remote'"):
                        self._remote_source_controls()
                    self._remote_picker(
                        title="Known-phase CIF in the experiment filesystem",
                        directory_model="remote_cif_directory",
                        subdirectory_model="remote_cif_subdirectory",
                        directory_items="remote_cif_directories",
                        file_model="remote_main_cif_uri",
                        file_items="remote_cif_files",
                        open_action=self.open_remote_cif_directory,
                        up_action=self.up_remote_cif_directory,
                        select_action=self.select_remote_main_cif,
                        file_label="Known/main-phase CIF",
                    )
                vuetify.VAlert(
                    text=(
                        "These files stay with this single-pattern session if you switch back. "
                        "A reusable configuration saves analysis settings only; POWGEN uses the inputs "
                        "selected in its own panel."
                    ),
                    type="info",
                    variant="tonal",
                    density="compact",
                    classes="mt-3",
                )
                with vuetify.VExpansionPanels(
                    v_show="false",
                    variant="accordion",
                    classes="mt-3",
                ):
                    with vuetify.VExpansionPanel(title="IPTS results and folder automation"):
                        with vuetify.VExpansionPanelText():
                            vuetify.VSwitch(
                                v_model=("publish_results_to_ipts",),
                                label="Export completed results to this IPTS through NDIP",
                                hint="The export runs as a separate Galaxy job using your authenticated facility access.",
                                persistent_hint=True,
                                color="#15543c",
                                density="compact",
                                inset=True,
                            )
                            with html.Div(v_show="publish_results_to_ipts || watch_enabled"):
                                with html.Div(classes="radar-facility-path-row"):
                                    vuetify.VTextField(
                                        v_model=("facility_output_directory",),
                                        label="Results parent folder",
                                        readonly=True,
                                        density="compact",
                                        variant="outlined",
                                        hide_details=True,
                                    )
                                    vuetify.VBtn(
                                        icon="mdi-arrow-up",
                                        title="Go to parent folder",
                                        click=self.up_facility_output_directory,
                                        variant="outlined",
                                        size="small",
                                    )
                                with html.Div(classes="radar-facility-path-row mt-2"):
                                    vuetify.VSelect(
                                        v_model=("facility_output_subdirectory",),
                                        items=("facility_output_directories",),
                                        item_title="title",
                                        item_value="value",
                                        label="Open existing folder",
                                        density="compact",
                                        variant="outlined",
                                        hide_details=True,
                                        clearable=True,
                                    )
                                    vuetify.VBtn(
                                        icon="mdi-folder-open-outline",
                                        title="Open selected output folder",
                                        click=self.open_facility_output_directory,
                                        disabled=("!facility_output_subdirectory",),
                                        variant="outlined",
                                        size="small",
                                    )
                                with html.Div(classes="radar-facility-path-row mt-2"):
                                    vuetify.VTextField(
                                        v_model=("facility_new_output_folder",),
                                        label="Create results folder",
                                        density="compact",
                                        variant="outlined",
                                        hide_details=True,
                                    )
                                    vuetify.VBtn(
                                        icon="mdi-folder-plus-outline",
                                        title="Create and open folder",
                                        click=self.create_facility_output_directory,
                                        variant="outlined",
                                        size="small",
                                    )
                            vuetify.VSwitch(
                                v_model=("watch_enabled",),
                                label="Prepare this data folder for automatic processing",
                                color="#15543c",
                                density="compact",
                                inset=True,
                            )
                            with html.Div(v_show="watch_enabled"):
                                vuetify.VTextField(
                                    v_model=("watch_patterns",),
                                    label="New-file patterns",
                                    density="compact",
                                    variant="outlined",
                                    hint="Comma-separated filename patterns",
                                    persistent_hint=True,
                                )
                                self._number_field(
                                    v_model=("watch_settle_seconds",),
                                    label="Wait after last file change (seconds)",
                                    min=10,
                                    density="compact",
                                    variant="outlined",
                                )
                                vuetify.VSwitch(
                                    v_model=("watch_process_existing",),
                                    label="Include files already present",
                                    color="#15543c",
                                    density="compact",
                                    inset=True,
                                )
                                with html.Div(classes="radar-field-pair"):
                                    self._number_field(
                                        v_model=("watch_max_attempts",),
                                        label="Maximum attempts per file",
                                        min=1,
                                        max=10,
                                        density="compact",
                                        variant="outlined",
                                    )
                                    self._number_field(
                                        v_model=("watch_retry_delay_seconds",),
                                        label="Retry delay (seconds)",
                                        min=10,
                                        density="compact",
                                        variant="outlined",
                                    )
                                vuetify.VBtn(
                                    "Save watch recipe in source folder",
                                    click=self.save_watch_recipe,
                                    prepend_icon="mdi-content-save-cog-outline",
                                    color="#15543c",
                                    variant="outlined",
                                    block=True,
                                )
                                vuetify.VAlert(
                                    v_show="!!watch_recipe_message",
                                    text=("watch_recipe_message",),
                                    type="info",
                                    variant="tonal",
                                    density="compact",
                                    classes="mt-2",
                                )
            with self._setup_section(4, "Chemistry Policy", 3, "!!sample_elements"):
                vuetify.VTextField(
                    label="Sample elements",
                    v_model=("sample_elements",),
                    placeholder="Tb, Be, Ge, O",
                    density="compact",
                    variant="outlined",
                    hint="Comma- or space-separated symbols",
                    persistent_hint=True,
                )
            with self._setup_section(
                5,
                "Pattern Regions",
                4,
                "((fit_start == null || fit_start === '') && (fit_end == null || fit_end === '')) || ((fit_start != null && fit_start !== '') && (fit_end != null && fit_end !== ''))",
            ):
                with html.Div(classes="radar-field-pair"):
                    self._number_field(label="Fit start (pattern x-axis)", v_model=("fit_start",), density="compact", variant="outlined", clearable=True)
                    self._number_field(label="Fit end (pattern x-axis)", v_model=("fit_end",), density="compact", variant="outlined", clearable=True)
                vuetify.VTextarea(
                    label="Ignored regions",
                    v_model=("ignore_regions",),
                    placeholder="One start,end pair per line\n2.0, 3.2",
                    density="compact",
                    variant="outlined",
                    rows=2,
                    auto_grow=True,
                )
                html.P(
                    "Use the same coordinate and units as the measured pattern: TOF in microseconds for GSAS/POWGEN data, or 2-theta in degrees for constant-wavelength data.",
                    classes="radar-field-note",
                )
            with self._setup_section(6, "Background Correction", 5, "!!background_mode && !!background_type && background_terms > 0"):
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
                    self._number_field(label="Terms", v_model=("background_terms",), min=1, max=36, density="compact", variant="outlined")
            with self._setup_section(
                7,
                "Magnetic Ordering Precheck",
                6,
                "true",
                v_show="radiation === 'neutron'",
            ):
                vuetify.VAlert(
                    v_show=f"!({main_phase_ready}) || radiation !== 'neutron'",
                    text="Available when a neutron run includes a known main-phase CIF.",
                    type="info",
                    variant="tonal",
                    density="compact",
                )
                vuetify.VSwitch(
                    v_show=f"({main_phase_ready}) && radiation === 'neutron'",
                    v_model=("magnetic_precheck",),
                    label="Check residual peaks for commensurate magnetic indexing",
                    color="#15543c",
                    density="compact",
                    inset=True,
                )
                self._number_field(
                    v_show=f"({main_phase_ready}) && radiation === 'neutron' && magnetic_precheck",
                    label="Q maximum",
                    v_model=("magnetic_q_max",),
                    density="compact",
                    variant="outlined",
                )
                vuetify.VTextField(
                    v_show=f"({main_phase_ready}) && radiation === 'neutron' && magnetic_precheck",
                    label="Commensurate denominators",
                    v_model=("magnetic_denominators",),
                    density="compact",
                    variant="outlined",
                    hint="Comma-separated integers, for example 2, 3, 4",
                    persistent_hint=True,
                )
            with self._setup_section(
                8,
                "Analysis Mode",
                7,
                "!!analysis_mode",
                display_number="radiation === 'neutron' ? '8' : '7'",
            ):
                with html.Div(classes="radar-mode-cards"):
                    with html.Div(
                        classes=("analysis_mode === 'rapid' ? 'radar-mode-card is-selected' : 'radar-mode-card'",),
                        role="button",
                        tabindex="0",
                        click="analysis_mode = 'rapid'",
                        keydown_enter="analysis_mode = 'rapid'",
                    ):
                        vuetify.VIcon("mdi-flash-outline", size="small")
                        html.Strong("Rapid")
                        html.Span("Staged hypothesis search and focused final refinements")
                    with html.Div(
                        classes=("analysis_mode === 'full' ? 'radar-mode-card is-selected' : 'radar-mode-card'",),
                        role="button",
                        tabindex="0",
                        click="analysis_mode = 'full'",
                        keydown_enter="analysis_mode = 'full'",
                    ):
                        vuetify.VIcon("mdi-layers-triple-outline", size="small")
                        html.Strong("Full")
                        html.Span("Residual-aware multi-pass discovery and refinement")
            with self._setup_section(
                9,
                "Runtime Budget",
                8,
                "true",
                display_number="radiation === 'neutron' ? '9' : '8'",
            ):
                with html.Div(v_show="analysis_mode === 'rapid'"):
                    with html.Div(classes="radar-field-pair"):
                        self._number_field(label="Phases / hypothesis", v_model=("rapid_phases_per_hypothesis",), min=1, max=5, density="compact", variant="outlined")
                        self._number_field(label="Retained / stage", v_model=("rapid_stage_output_limit",), min=3, max=50, density="compact", variant="outlined")
                        self._number_field(label="Final refinements", v_model=("rapid_gsas_validation_limit",), min=0, density="compact", variant="outlined")
                        self._number_field(label="Parallel workers", v_model=("rapid_parallel_workers",), min=1, max=16, density="compact", variant="outlined")
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
                    vuetify.VAlert(v_show="full_profile === 'quick'", text="One discovery pass for a fast first assessment.", type="info", variant="tonal", density="compact")
                    vuetify.VAlert(v_show="full_profile === 'balanced'", text="Up to two discovery passes; the search may stop early when the accepted model no longer improves.", type="info", variant="tonal", density="compact")
                    vuetify.VAlert(v_show="full_profile === 'thorough'", text="Up to three discovery passes. Small Rwp changes alone do not stop the residual search, but scientific safety checks still apply.", type="info", variant="tonal", density="compact")
                    with html.Div(
                        v_show="full_profile === 'custom'",
                        classes="radar-custom-budget-controls",
                    ):
                        with html.Div(classes="radar-custom-budget-grid"):
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
                                ("Minimum Rwp improvement", "full_rwp_improvement_threshold"),
                                ("Duplicate-candidate threshold", "full_dedup_threshold"),
                                ("Nudge scoring Q max", "full_score_q_max"),
                                ("Pearson cell-refine cutoff", "full_pearson_cell_min_r"),
                                ("Nudge near-tie score tolerance", "full_lattice_tiebreak_score_tol"),
                            ):
                                self._number_field(label=label, v_model=(model,), min=0, density="compact", variant="outlined")
                        vuetify.VSwitch(v_model=("full_candidate_pruning",), label="Use automatic candidate pruning", color="#15543c", density="compact", inset=True)
                        with html.Div(v_show="full_candidate_pruning", classes="radar-custom-budget-grid"):
                            self._number_field(label="Knee minimum points", v_model=("full_knee_min_points_hist",), min=1, density="compact", variant="outlined")
                            self._number_field(label="Knee minimum relative span", v_model=("full_knee_min_relative_span",), min=0, max=1, step="0.01", density="compact", variant="outlined")
                            self._number_field(label="Keep when no knee is found", v_model=("full_knee_keep_if_no_knee",), min=0, density="compact", variant="outlined", hint="0 keeps every candidate", persistent_hint=True)
                            self._number_field(label="Maximum candidates after pruning", v_model=("full_knee_keep_at_most",), min=0, density="compact", variant="outlined", hint="0 removes the cap", persistent_hint=True)
            with self._setup_section(
                10,
                "Expert Tuning",
                9,
                "true",
                display_number="radiation === 'neutron' ? '10' : '9'",
            ):
                vuetify.VSwitch(v_model=("reference_masks_enabled",), label="Mask reference/can peaks", color="#15543c", density="compact", inset=True)
                vuetify.VSelect(v_show="reference_masks_enabled", label="Reference structures", v_model=("reference_mask_presets",), items=("['Al_fcc','Cu_fcc','V_bcc']",), multiple=True, chips=True, density="compact", variant="outlined")
                vuetify.VSelect(v_show="reference_masks_enabled", label="Reference-mask window", v_model=("reference_window_mode",), items=("[{title:'Automatic from resolution',value:'auto'},{title:'Fixed window',value:'fixed'}]",), item_title="title", item_value="value", density="compact", variant="outlined")
                self._number_field(
                    v_show="reference_masks_enabled && reference_window_mode === 'fixed'",
                    label="Fixed half-width (pattern x-axis units)",
                    v_model=("reference_fixed_half_width",),
                    min=0,
                    clearable=True,
                    density="compact",
                    variant="outlined",
                    hint="Blank uses the instrument-aware default",
                    persistent_hint=True,
                )
                with html.Div(v_show="reference_masks_enabled && reference_window_mode === 'auto'", classes="radar-field-pair"):
                    self._number_field(label="FWHM multiplier", v_model=("reference_fwhm_factor",), min=0, step="0.1", density="compact", variant="outlined")
                    self._number_field(label="Fractional d tolerance", v_model=("reference_fractional_d_tolerance",), min=0, step="0.001", density="compact", variant="outlined")
                    self._number_field(label="Zero-offset tolerance", v_model=("reference_zero_tolerance",), min=0, clearable=True, density="compact", variant="outlined", hint="Pattern x-axis units; blank uses the instrument default", persistent_hint=True)
                    self._number_field(label="Minimum half-width", v_model=("reference_min_half_width",), min=0, clearable=True, density="compact", variant="outlined", hint="Pattern x-axis units", persistent_hint=True)
                    self._number_field(label="Maximum half-width", v_model=("reference_max_half_width",), min=0, clearable=True, density="compact", variant="outlined", hint="Pattern x-axis units", persistent_hint=True)
                vuetify.VSwitch(v_show="radiation === 'xray' && reference_masks_enabled", v_model=("include_cu_kbeta",), label="Mask Cu K-beta companions", color="#15543c", density="compact", inset=True)
                vuetify.VDivider(classes="my-3")
                vuetify.VAlert(v_show=f"!({main_phase_ready})", text="Main-phase safeguards become available after a CIF is actually selected.", type="info", variant="tonal", density="compact")
                vuetify.VSwitch(v_show=main_phase_ready, v_model=("main_prenudge",), label="Anchor supplied main-phase cell", color="#15543c", density="compact", inset=True)
                vuetify.VSwitch(v_show=main_phase_ready, v_model=("main_shadow_filter",), label="Filter main-shadow-only candidates", color="#15543c", density="compact", inset=True)
                vuetify.VSwitch(v_show=main_phase_ready, v_model=("cleanup_enabled",), label="Clean up main-CIF internal parameters", color="#15543c", density="compact", inset=True)
                vuetify.VCheckbox(v_show=f"({main_phase_ready}) && cleanup_enabled", v_model=("refine_u_iso",), label="Refine isotropic displacement parameters", density="compact")
                vuetify.VCheckbox(v_show=f"({main_phase_ready}) && cleanup_enabled", v_model=("refine_positions",), label="Refine atomic positions", density="compact")
                vuetify.VSwitch(
                    v_show=f"radiation === 'xray' && ({main_phase_ready})",
                    v_model=("light_calibration_enabled",),
                    label="Calibrate Zero and U/V/W from the supplied main phase",
                    color="#15543c",
                    density="compact",
                    inset=True,
                )
                with html.Div(v_show="analysis_mode === 'rapid'"):
                    vuetify.VDivider(classes="my-3")
                    vuetify.VSwitch(v_model=("rapid_show_family_variants",), label="Keep family variants in Solution Inspector", color="#15543c", density="compact", inset=True)
                    vuetify.VSwitch(v_model=("rapid_final_polish_enabled",), label="Run final polish after ranking", color="#15543c", density="compact", inset=True)
                vuetify.VTextField(
                    label="Excluded space groups",
                    v_model=("excluded_space_groups",),
                    density="compact",
                    variant="outlined",
                    hint="Comma-separated space-group numbers; leave blank to allow all",
                    persistent_hint=True,
                )
            ready_expression = f"connection_ok && !!sample_elements && ({data_ready}) && (database_source === 'builtin' || (database_source === 'archive' && ((library_archive_source === 'computer' && !!database_archive_path) || (library_archive_source === 'galaxy' && !!history_database_id))))"
            with self._setup_section(
                11,
                "Review Run Plan",
                10,
                ready_expression,
                display_number="radiation === 'neutron' ? '11' : '10'",
            ):
                vuetify.VTextField(label="Run name", v_model=("run_name",), density="compact", variant="outlined", placeholder="Generated automatically if blank")
                with html.Div(classes="radar-checklist"):
                    for label, expression in (
                        ("Connected to NDIP", "connection_ok"),
                        ("Measurement selected", "!!radiation && !!instrument_mode"),
                        ("Data and instrument ready", data_ready),
                        ("Sample chemistry entered", "!!sample_elements"),
                        ("Candidate library ready", "database_source === 'builtin' || (database_source === 'archive' && (!!database_archive_path || !!history_database_id))"),
                    ):
                        with html.Div(classes="radar-check-row"):
                            vuetify.VIcon(icon=(f"{expression} ? 'mdi-check-circle' : 'mdi-alert-circle-outline'",), color=(f"{expression} ? '#1f6b4b' : '#a66a00'",), size="small")
                            html.Span(label)
                with html.Div(classes="radar-review-summary"):
                    html.Div("{{ analysis_mode === 'rapid' ? 'Rapid Hypothesis Mode' : 'Full RADAR-PD' }}", classes="radar-review-primary")
                    html.Div("{{ radiation === 'neutron' ? 'Neutron' : 'X-ray' }} / {{ instrument_mode.toUpperCase() }}", classes="radar-review-secondary")
                    html.Div("Sample: {{ sample_elements || 'not entered' }}", classes="radar-review-secondary")
                    html.Div(
                        f"Main phase: {{{{ ({main_phase_ready}) ? 'selected' : 'not supplied' }}}}",
                        classes="radar-review-secondary",
                    )
                    html.Div(
                        "{{ analysis_mode === 'full' "
                        "? 'Server request: FULL / ' + full_profile + ' search profile' "
                        ": 'Server request: RAPID / ' + rapid_phases_per_hypothesis + ' phases per hypothesis / ' + rapid_gsas_validation_limit + ' final refinements' }}",
                        classes="radar-review-secondary radar-server-summary",
                    )
                vuetify.VAlert(
                    v_show="busy && !!selected_run_uid",
                    text=("selected_run_stage + ' — ' + selected_run_elapsed",),
                    type="info",
                    variant="tonal",
                    density="compact",
                    classes="mb-2",
                )
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
                            vuetify.VAutocomplete(
                                label="Reusable configuration from Galaxy History",
                                v_model=("history_configuration_id",),
                                items=("history_configuration_datasets",),
                                item_title="display_name",
                                item_value="id",
                                density="compact",
                                variant="outlined",
                                classes="mt-3",
                                no_data_text="No reusable configurations are in this Galaxy History",
                                update_modelValue=(self._history_configuration_changed, "[$event]"),
                            )
                            with vuetify.VExpansionPanels(
                                v_show="!!history_configuration_id",
                                variant="accordion",
                                classes="radar-config-import mt-2",
                            ):
                                with vuetify.VExpansionPanel(title=("history_configuration_title",)):
                                    with vuetify.VExpansionPanelText():
                                        vuetify.VAlert(
                                            v_show="!!history_configuration_error",
                                            text=("history_configuration_error",),
                                            type="warning",
                                            variant="tonal",
                                            density="compact",
                                            classes="mb-2",
                                        )
                                        with html.Div(classes="radar-preflight-summary"):
                                            with html.Div(
                                                v_for="item in history_configuration_summary",
                                                key="item.label",
                                                classes="radar-preflight-row",
                                            ):
                                                html.Span("{{ item.label }}", classes="radar-preflight-label")
                                                html.Span("{{ item.value }}", classes="radar-preflight-value")
                                        html.Pre(
                                            "{{ history_configuration_yaml }}",
                                            v_show="!!history_configuration_yaml",
                                            classes="radar-config-preview mt-2",
                                        )
                            vuetify.VBtn(
                                "Apply History configuration",
                                click=self.apply_history_configuration,
                                variant="outlined",
                                size="small",
                                block=True,
                                disabled=("!history_configuration_id",),
                            )
                vuetify.VBtn(
                    "Save reusable configuration to History",
                    click=(self.save_current_configuration, f"[{self._submission_payload_expression()}]"),
                    prepend_icon="mdi-content-save-outline",
                    variant="outlined",
                    block=True,
                    classes="mb-2",
                )
                vuetify.VBtn(
                    "Run analysis on NDIP",
                    click=(self.submit_run, f"[{self._submission_payload_expression()}]"),
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
            with html.Div(v_if="!selected_run_uid && workspace_view !== 'experiment'", classes="radar-workspace-empty"):
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

            with html.Div(v_show="workspace_view === 'experiment'", classes="radar-workspace-view"):
                self._powgen_experiment_dashboard()

            with html.Div(v_if="!!selected_run_uid && workspace_view !== 'experiment'"):
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
                    with vuetify.VBtn(
                        v_for="item in workspace_options",
                        key="item.value",
                        value=("item.value",),
                        size="small",
                        click="workspace_view = item.value; $nextTick(() => window.setTimeout(() => window.document.querySelectorAll('.js-plotly-plot').forEach(el => window.Plotly && window.Plotly.Plots.resize(el)), 80))",
                    ):
                        # Plotly measures a hidden v-show panel as zero width.
                        # Resize after the selected workspace becomes visible.
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
                self._activity_panel()

    def _powgen_experiment_dashboard(self) -> None:
        with html.Div(classes="radar-section-heading"):
            with html.Div():
                html.Div("POWGEN LIVE EXPERIMENT", classes="radar-micro-label")
                html.H2("{{ powgen_ipts || 'Experiment dashboard' }}")
                html.P("Scan-by-scan phase evolution and refinement quality from completed RADAR-PD analyses.")
            with html.Div(classes="radar-button-row"):
                vuetify.VChip(
                    text=("powgen_monitoring ? 'Monitoring' : 'Stopped'",),
                    color=("powgen_monitoring ? '#dff2e8' : '#eef2ef'",),
                    variant="flat",
                    prepend_icon="mdi-access-point",
                )
                vuetify.VBtn(
                    "Back to current run",
                    v_show="!!selected_run_uid",
                    click="workspace_view = 'monitor'",
                    prepend_icon="mdi-arrow-left",
                    variant="text",
                    size="small",
                )
        with html.Div(classes="radar-experiment-provenance mb-3"):
            html.Span("Configuration: {{ powgen_configuration_label }}")
            html.Span("Candidate library: {{ powgen_database_label }}")
            html.Span("Wavelength: {{ powgen_wavelength }} A")
        vuetify.VAlert(
            text=("powgen_dashboard_notice",),
            type="info",
            variant="tonal",
            density="compact",
            classes="mb-3",
        )
        with html.Div(classes="radar-metric-grid"):
            with html.Div(v_for="metric in powgen_dashboard_metrics", key="metric.label", classes="radar-metric-card"):
                html.Div("{{ metric.label }}", classes="radar-metric-label")
                html.Div("{{ metric.value }}", classes="radar-metric-value")
                html.Div("{{ metric.detail }}", classes="radar-run-list-meta")
        with html.Div(v_if="powgen_scientific_rows.length", classes="radar-experiment-grid"):
            with html.Section(classes="radar-result-card radar-experiment-phase-card"):
                with html.Div(classes="radar-card-heading"):
                    with html.Div():
                        html.Div("PHASE EVOLUTION", classes="radar-micro-label")
                        html.H3("Refined phase fractions")
                with html.Div(classes="radar-experiment-controls"):
                    vuetify.VSelect(
                        label="Samples shown",
                        v_model=("powgen_selected_samples",),
                        items=("powgen_sample_options",),
                        multiple=True,
                        chips=True,
                        closable_chips=True,
                        clearable=True,
                        placeholder="All samples",
                        persistent_placeholder=True,
                        density="compact",
                        variant="outlined",
                        hide_details=True,
                        update_modelValue=(self.update_powgen_sample_selection, "[$event]"),
                    )
                    vuetify.VSelect(
                        label="Horizontal axis",
                        v_model=("powgen_x_axis",),
                        items=("powgen_x_axis_options",),
                        density="compact",
                        variant="outlined",
                        hide_details=True,
                        update_modelValue=(self.update_powgen_dashboard_axis, "[$event]"),
                    )
                    vuetify.VSelect(
                        label="Phases shown",
                        v_model=("powgen_selected_phase_labels",),
                        items=("powgen_phase_options",),
                        multiple=True,
                        chips=True,
                        closable_chips=True,
                        density="compact",
                        variant="outlined",
                        hide_details=True,
                        update_modelValue=(self.update_powgen_phase_selection, "[$event]"),
                        classes="radar-phase-selector",
                    )
                with html.Div(classes="radar-plot-frame"):
                    self._powgen_phase_widget = plotly.Figure(display_mode_bar=True)
            with html.Section(classes="radar-result-card radar-experiment-quality-card"):
                with html.Div(classes="radar-card-heading"):
                    with html.Div():
                        html.Div("FIT STABILITY", classes="radar-micro-label")
                        html.H3("Refinement quality")
                with html.Div(classes="radar-plot-frame"):
                    self._powgen_quality_widget = plotly.Figure(display_mode_bar=True)
            with html.Section(classes="radar-result-card radar-experiment-heatmap-card"):
                with html.Div(classes="radar-card-heading"):
                    with html.Div():
                        html.Div("COMPOSITION MAP", classes="radar-micro-label")
                        html.H3("All reported phases across scans")
                with html.Div(classes="radar-plot-frame radar-experiment-heatmap-frame"):
                    self._powgen_heatmap_widget = plotly.Figure(display_mode_bar=True)
            with html.Section(classes="radar-result-card radar-experiment-table-card"):
                with html.Div(classes="radar-card-heading"):
                    with html.Div():
                        html.Div("SCAN RECORD", classes="radar-micro-label")
                        html.H3("Completed analyses")
                vuetify.VDataTable(
                    headers=("[{title:'Scan',key:'run_number'},{title:'Sample',key:'sample_label'},{title:'Conditions',key:'conditions_display'},{title:'Rwp',key:'rwp_display'},{title:'Trend check',key:'quality_label'},{title:'Dominant refined phases',key:'phase_summary'},{title:'Runtime',key:'elapsed_display'},{title:'Mode',key:'analysis_mode'}]",),
                    items=("powgen_scientific_rows",),
                    density="compact",
                    items_per_page=15,
                    no_data_text="No completed scan summaries are available yet.",
                )
            with html.Section(classes="radar-result-card radar-experiment-latest-card"):
                with html.Div(classes="radar-card-heading"):
                    with html.Div():
                        html.Div("SCAN INSPECTOR", classes="radar-micro-label")
                        html.H3("Inspect one completed scan")
                vuetify.VSelect(
                    label="Completed scan",
                    v_model=("powgen_selected_run_id",),
                    items=("powgen_scan_options",),
                    density="compact",
                    variant="outlined",
                    hide_details=True,
                    update_modelValue=(self.select_powgen_scan, "[$event]"),
                )
                with html.Div(v_if="powgen_selected_scan.run_id", classes="radar-scan-inspector-summary"):
                    html.Strong("{{ powgen_selected_scan.run_id }}")
                    html.Span("Rwp {{ powgen_selected_scan.rwp_display }} | {{ powgen_selected_scan.elapsed_display }} | {{ powgen_selected_scan.analysis_mode }}")
                    html.Span("{{ powgen_selected_scan.conditions_display }}")
                    html.Span(
                        "{{ powgen_selected_scan.temperature_provenance }}",
                        v_if="powgen_selected_scan.temperature_provenance",
                        classes="radar-section-help",
                    )
                    html.Span("{{ powgen_selected_scan.run_title }}", v_if="powgen_selected_scan.run_title")
                    html.Span(
                        "Sample: {{ powgen_selected_scan.sample_display }}",
                        v_if="powgen_selected_scan.sample_display",
                    )
                    vuetify.VChip(
                        text=("powgen_selected_quality_label",),
                        color=("powgen_selected_quality_color",),
                        variant="tonal",
                        size="small",
                    )
                with html.Div(v_if="powgen_selected_phases.length", classes="radar-phase-list"):
                    with html.Div(v_for="phase in powgen_selected_phases", key="phase.label", classes="radar-phase-row"):
                        with html.Div(classes="radar-phase-copy"):
                            html.Strong("{{ phase.phase }}")
                            html.Span("Space group {{ phase.space_group }}")
                        html.Strong("{{ phase.weight_display }}", classes="radar-phase-weight")
                vuetify.VBtn(
                    "Open full scan result",
                    click=self.open_powgen_selected_run,
                    prepend_icon="mdi-open-in-new",
                    variant="outlined",
                    size="small",
                    classes="mt-3",
                    disabled=("!powgen_selected_run_id",),
                )
        with html.Div(v_if="!powgen_scientific_rows.length", classes="radar-result-not-ready"):
            vuetify.VAlert(
                text="Scientific trends will appear after the first scan finishes. Live discovery and processing status remain visible below.",
                type="info",
                variant="tonal",
            )
        with html.Section(classes="radar-result-card radar-experiment-queue-card"):
            with html.Div(classes="radar-card-heading"):
                with html.Div():
                    html.Div("LIVE QUEUE", classes="radar-micro-label")
                    html.H3("Scan processing status")
                    html.P("Updated {{ powgen_last_checked }}", classes="radar-section-help")
            vuetify.VDataTable(
                headers=("[{title:'Scan',key:'run_number'},{title:'File',key:'file'},{title:'Status',key:'status'},{title:'Current activity',key:'detail'}]",),
                items=("powgen_rows",),
                density="compact",
                items_per_page=10,
                no_data_text="No POWGEN scans have been discovered yet.",
            )

    def _run_monitor_view(self) -> None:
        with html.Div(classes="radar-section-heading"):
            with html.Div():
                html.H2("Run Monitor")
                html.P("The analysis remains in Galaxy even if this interactive session is closed.")
            with html.Div(classes="radar-button-row"):
                vuetify.VBtn("Refresh", click=self.refresh_selected_run, prepend_icon="mdi-refresh", variant="outlined", size="small")
                vuetify.VBtn(
                    "Reload results from Galaxy",
                    v_show="selected_analysis_status === 'Ok'",
                    click=self.reload_selected_results,
                    prepend_icon="mdi-cloud-download-outline",
                    variant="outlined",
                    size="small",
                )
                vuetify.VBtn("Use configuration", click=self.use_selected_configuration, prepend_icon="mdi-file-restore-outline", variant="outlined", size="small")
                vuetify.VBtn("Diagnostics", click=self.download_diagnostics, prepend_icon="mdi-bug-outline", variant="text", size="small")
                vuetify.VBtn(
                    "Stop",
                    click="cancel_dialog = true",
                    prepend_icon="mdi-stop-circle-outline",
                    color="#9b2c2c",
                    variant="outlined",
                    size="small",
                    disabled=("selected_run_status !== 'Running' && selected_run_status !== 'Queued' && selected_run_status !== 'Uploading'",),
                )
        with html.Div(classes="radar-monitor-summary"):
            with html.Div(classes="radar-monitor-stage"):
                html.Div("CURRENT STAGE", classes="radar-micro-label")
                html.H3("{{ selected_run_stage }}")
                html.P("{{ selected_run_message || 'Waiting for the next Galaxy update.' }}")
                html.Span("{{ selected_run_elapsed }}", classes="radar-elapsed")
                with html.Div(classes="radar-job-context"):
                    html.Code("Galaxy job {{ selected_galaxy_job_id }}")
                    vuetify.VBtn(
                        icon="mdi-content-copy",
                        title="Copy Galaxy job ID",
                        size="x-small",
                        variant="text",
                        disabled=("selected_galaxy_job_id === 'Pending'",),
                        click="navigator.clipboard.writeText(selected_galaxy_job_id)",
                    )
                    vuetify.VChip(text=("'Analysis: ' + selected_analysis_status",), size="x-small", variant="tonal")
                    vuetify.VChip(text=("'Results: ' + selected_result_status",), size="x-small", variant="tonal")
                    vuetify.VChip(
                        v_show="selected_publication_status !== '-'",
                        text=("'IPTS export: ' + selected_publication_status",),
                        size="x-small",
                        variant="tonal",
                    )
            with html.Div(classes="radar-progress-number"):
                html.Strong("{{ selected_run_progress }}%")
                html.Span("complete")
        vuetify.VProgressLinear(model_value=("selected_run_progress",), color="#1f6b4b", height=10, rounded=True, classes="mb-4")
        vuetify.VAlert(v_show="selected_run_loading", text="Loading the completed run and reconstructing its scientific results...", type="info", variant="tonal", classes="mb-3")
        vuetify.VAlert(v_show="selected_run_status === 'Error' && !!selected_run_message", text=("selected_run_message",), type="error", variant="tonal", classes="mb-3")
        vuetify.VAlert(
            v_show="selected_analysis_status === 'Ok' && selected_result_status === 'Error'",
            text="The scientific analysis completed, but NOVA could not collect its archive. Use Reload results from Galaxy; the Galaxy job is not failed.",
            type="warning",
            variant="tonal",
            classes="mb-3",
        )
        vuetify.VAlert(
            v_show="selected_publication_status === 'Queued' || selected_publication_status === 'Running'",
            text=("selected_publish_message",),
            type="info",
            variant="tonal",
            classes="mb-3",
        )
        vuetify.VAlert(
            v_show="selected_publication_status === 'Ok'",
            text=("'Result archive exported to ' + selected_publication_target",),
            type="success",
            variant="tonal",
            classes="mb-3",
        )
        vuetify.VAlert(
            v_show="selected_publication_status === 'Error' || selected_publication_status === 'Cancelled'",
            text=("selected_publish_message",),
            type="warning",
            variant="tonal",
            classes="mb-3",
        )
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
            with html.Div(classes="radar-button-row"):
                vuetify.VBtn(
                    text=(
                        "gsasii_session_status === 'starting' ? 'Starting GSAS-II...' : 'Open GPX in GSAS-II'",
                    ),
                    click=self.open_selected_checkpoint_in_gsasii,
                    prepend_icon="mdi-tune-vertical",
                    variant="outlined",
                    size="small",
                    loading=("gsasii_session_status === 'starting'",),
                    disabled=("!selected_checkpoint || gsasii_session_status === 'starting'",),
                    v_if="checkpoint_rows.length > 0 && gsasii_session_status !== 'ready'",
                )
                with html.A(
                    href=("gsasii_launch_url",),
                    target="_blank",
                    rel="noopener noreferrer",
                    classes="radar-secondary-link as-button radar-gsasii-launch",
                    v_if="gsasii_session_status === 'ready' && !!gsasii_launch_url",
                ):
                    vuetify.VIcon(icon="mdi-open-in-new", size="small", classes="mr-1")
                    html.Span("Open GSAS-II")
                vuetify.VBtn(
                    "Open Result Explorer",
                    click=self.launch_result_explorer,
                    prepend_icon="mdi-open-in-new",
                    variant="outlined",
                    size="small",
                    v_if="result_explorer_available",
                )
                vuetify.VBtn("Back to monitor", click="workspace_view = 'monitor'", prepend_icon="mdi-arrow-left", variant="text", size="small")
        vuetify.VAlert(
            v_if="gsasii_session_status === 'starting' || gsasii_session_status === 'error'",
            text=("gsasii_status_message",),
            type=("gsasii_session_status === 'error' ? 'error' : 'info'",),
            variant="tonal",
            density="compact",
            classes="mb-3",
        )
        vuetify.VAlert(
            v_if="viewed_output_profile === 'monitor'",
            text=(
                "POWGEN live uses a low-latency publication profile: the normalized dashboard, fit plots, "
                "and any usable GPX checkpoints are published immediately. Run a standard Full "
                "single-pattern analysis when you need the complete table, phase-CIF, configuration, "
                "and results-archive collections."
            ),
            type="info",
            variant="tonal",
            density="compact",
            classes="mb-3",
        )
        with html.Div(v_if="selected_run_status !== 'Ok'", classes="radar-result-not-ready"):
            vuetify.VAlert(text="Results will appear here after the selected Galaxy run completes.", type="info", variant="tonal")
        with html.Div(v_show="selected_run_status === 'Ok'"):
            vuetify.VAlert(
                v_show="selected_publication_status === 'Queued' || selected_publication_status === 'Running'",
                text=("selected_publish_message",),
                type="info",
                variant="tonal",
                classes="mb-3",
            )
            vuetify.VAlert(
                v_show="selected_publication_status === 'Ok'",
                text=("'Result archive exported to ' + selected_publication_target",),
                type="success",
                variant="tonal",
                classes="mb-3",
            )
            vuetify.VAlert(
                v_show="selected_publication_status === 'Error' || selected_publication_status === 'Cancelled'",
                text=("selected_publish_message",),
                type="warning",
                variant="tonal",
                classes="mb-3",
            )
            with html.Section(classes="radar-run-provenance mb-3", v_if="run_provenance_rows.length"):
                with html.Div(classes="radar-run-provenance-grid"):
                    with html.Div(
                        v_for="item in run_provenance_rows",
                        key="item.label",
                        classes="radar-run-provenance-item",
                    ):
                        html.Span("{{ item.label }}")
                        html.Strong("{{ item.value }}")
                with vuetify.VExpansionPanels(
                    v_if="!!viewed_configuration",
                    variant="accordion",
                    classes="radar-detail-panels mt-2",
                ):
                    with vuetify.VExpansionPanel(title="Resolved settings used for this run"):
                        with vuetify.VExpansionPanelText():
                            html.Pre("{{ viewed_configuration }}", classes="radar-config-preview")
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
                            html.H3("Refinement fit")
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
                    with html.Div(classes="radar-plot-frame"):
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
                    vuetify.VSelect(label="GSAS-II checkpoint", v_model=("selected_checkpoint",), items=("checkpoint_rows",), item_title="name", item_value="id", density="compact", variant="outlined", no_data_text="No published GPX checkpoint is available")
                    with html.Div(classes="radar-button-row"):
                        vuetify.VBtn("Download selected checkpoint", click=self.download_checkpoint, disabled=("!checkpoint_rows.some(item => item.id === selected_checkpoint && item.local_available)",), prepend_icon="mdi-download", color="#15543c", variant="outlined")

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
                        vuetify.VAlert(v_show="!!table_preview_notice", text=("table_preview_notice",), type="info", variant="tonal", density="compact", classes="mb-2")
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
            with html.Div(classes="radar-plot-frame"):
                self._plot_widget = plotly.Figure(display_mode_bar=True)
            html.Div("No interactive plots were published.", v_if="!plot_options.length", classes="radar-empty-compact")

    def _file_browser_view(self) -> None:
        visible_file_filter = (
            "(show_technical_files || !item.technical) && "
            "(!file_search || (item.name + ' ' + item.filename).toLowerCase().includes(file_search.toLowerCase()))"
        )
        with html.Div(classes="radar-section-heading"):
            with html.Div():
                html.H2("Run File Browser")
                html.P("Published results grouped by scientific purpose; local container paths remain hidden.")
            vuetify.VBtn(
                "Open GPX in GSAS-II",
                click=self.open_selected_checkpoint_in_gsasii,
                disabled=("!selected_checkpoint || gsasii_session_status === 'starting'",),
                loading=("gsasii_session_status === 'starting'",),
                prepend_icon="mdi-tune-vertical",
                variant="outlined",
                size="small",
                classes="radar-gsasii-launch",
                v_if="checkpoint_rows.length > 0 && gsasii_session_status !== 'ready'",
            )
            with html.A(
                href=("gsasii_launch_url",),
                target="_blank",
                rel="noopener noreferrer",
                classes="radar-secondary-link as-button radar-gsasii-launch",
                v_if="gsasii_session_status === 'ready' && !!gsasii_launch_url",
            ):
                vuetify.VIcon(icon="mdi-open-in-new", size="small", classes="mr-1")
                html.Span("Open GSAS-II")
        vuetify.VAlert(
            v_if="gsasii_session_status === 'starting' || gsasii_session_status === 'error'",
            text=("gsasii_status_message",),
            type=("gsasii_session_status === 'error' ? 'error' : 'info'",),
            variant="tonal",
            density="compact",
            classes="mb-3",
        )
        html.P(
            "Choose a published GPX checkpoint, then open it in the hosted GSAS-II desktop. The RADAR-PD result remains unchanged; saved edits return to Galaxy History.",
            v_show="checkpoint_rows.length > 0",
            classes="radar-section-help mb-3",
        )
        html.P(
            "No GPX checkpoint was published for this run. The GSAS-II action appears only when the analysis produced a usable checkpoint.",
            v_if="!checkpoint_rows.length",
            classes="radar-section-help mb-3",
        )
        with html.Div(classes="radar-file-toolbar"):
            vuetify.VSelect(
                label="GSAS-II checkpoint",
                v_model=("selected_checkpoint",),
                items=("checkpoint_rows",),
                item_title="name",
                item_value="id",
                density="compact",
                variant="outlined",
                hide_details=True,
                v_show="checkpoint_rows.length > 0",
            )
            vuetify.VTextField(
                label="Search published files",
                v_model=("file_search",),
                prepend_inner_icon="mdi-magnify",
                clearable=True,
                density="compact",
                variant="outlined",
                hide_details=True,
            )
            vuetify.VSwitch(
                label="Technical files",
                v_model=("show_technical_files",),
                color="#15543c",
                density="compact",
                hide_details=True,
            )
        with vuetify.VExpansionPanels(multiple=True, variant="accordion", classes="radar-file-groups"):
            with vuetify.VExpansionPanel(
                v_for="group in file_groups",
                key="group.name",
                value=("group.name",),
                v_show=f"group.files.some(item => {visible_file_filter})",
            ):
                with vuetify.VExpansionPanelTitle():
                    html.Strong("{{ group.name }}")
                    vuetify.VSpacer()
                    vuetify.VChip(
                        text=(f"String(group.files.filter(item => {visible_file_filter}).length)",),
                        size="x-small",
                        variant="tonal",
                        color="#15543c",
                    )
                with vuetify.VExpansionPanelText():
                    with html.Div(
                        v_for=f"file in group.files.filter(item => {visible_file_filter})",
                        key="file.id",
                        classes="radar-file-row",
                    ):
                        with html.Div(classes="radar-file-copy"):
                            html.Strong("{{ file.name }}")
                            html.Span("{{ file.filename }} / {{ file.size }}")
                        vuetify.VBtn(icon="mdi-download", title="Download", size="small", variant="text", color="#15543c", click=(self.download_artifact, "[file.path]"))
        html.Div("No downloadable files are available for this run.", v_if="!file_groups.length", classes="radar-empty-compact")
        html.Div(
            "No published files match these filters.",
            v_if=f"file_groups.length && !file_groups.some(group => group.files.some(item => {visible_file_filter}))",
            classes="radar-empty-compact",
        )

    def _activity_panel(self) -> None:
        with vuetify.VExpansionPanels(variant="accordion", classes="radar-activity-panel mt-4"):
            with vuetify.VExpansionPanel(value="activity"):
                with vuetify.VExpansionPanelTitle():
                    vuetify.VIcon("mdi-pulse", size="small", classes="mr-2")
                    html.Strong("Companion-tool activity")
                    vuetify.VSpacer()
                    vuetify.VChip(text=("String(activity_rows.length)",), size="x-small", variant="tonal")
                with vuetify.VExpansionPanelText():
                    with html.Div(v_for="activity in activity_rows", key="activity.uid", classes="radar-activity-row"):
                        with html.Div(classes="radar-file-copy"):
                            html.Strong("{{ activity.name }}")
                            html.Span("{{ activity.status }} / {{ activity.run_name || 'Workbench' }} / {{ activity.job_id }}")
                            html.Span("{{ activity.message }}", v_if="!!activity.message")
                        with html.Div(classes="radar-button-row"):
                            html.A(
                                "Open",
                                v_if="!!activity.launch_url",
                                href=("activity.launch_url",),
                                target="_blank",
                                classes="radar-secondary-link",
                            )
                            vuetify.VBtn(
                                "Retry",
                                v_if="activity.status === 'Error'",
                                click=(self.retry_utility, "[activity.uid]"),
                                size="x-small",
                                variant="text",
                                prepend_icon="mdi-refresh",
                            )
                    html.Div("No companion actions have run in this session.", v_if="!activity_rows.length", classes="radar-empty-compact")

    def _parse_regions(self, value: Any = None) -> list[tuple[float, float]]:
        regions: list[tuple[float, float]] = []
        source = self.server.state.ignore_regions if value is None else value
        for line in str(source or "").splitlines():
            if not line.strip():
                continue
            parts = [part.strip() for part in line.replace(";", ",").split(",")]
            if len(parts) != 2:
                raise ValueError(f"Ignored region must be 'start,end': {line}")
            regions.append((float(parts[0]), float(parts[1])))
        return regions

    @staticmethod
    def _selected_value(value: Any) -> str:
        if isinstance(value, dict):
            value = value.get("value", value.get("id", ""))
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        text = str(value or "").strip()
        if text[:1] in {"{", "[", "("} and text[-1:] in {"}", "]", ")"}:
            try:
                decoded = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                pass
            else:
                if isinstance(decoded, (dict, list, tuple)):
                    return RadarPdNovaApp._selected_value(decoded)
        return text

    def discover_remote_sources(self, **_: Any) -> None:
        state = self.server.state
        try:
            sources = self.service.list_remote_file_sources()
            state.remote_sources = sources
            state.remote_available = bool(sources)
            if not sources:
                state.remote_message = (
                    "Galaxy did not expose an authenticated remote file source to this session. "
                    "Use Galaxy's Upload > Choose remote files, then select the imported file from Galaxy History."
                )
            else:
                current = str(state.remote_source or "")
                allowed = {str(item["value"]) for item in sources}
                if current not in allowed:
                    state.remote_source = str(sources[0]["value"])
                state.remote_message = (
                    "These folders are provided by Galaxy using your NDIP identity. "
                    "Selected files are imported into this History before RADAR-PD starts."
                )
                self._remote_source_changed(state.remote_source)
        except Exception as exc:
            state.remote_available = False
            state.remote_sources = []
            state.remote_message = f"Could not discover Galaxy remote sources: {exc}"
        state.flush()

    def _remote_source_changed(self, remote_source: Any = None, **_: Any) -> None:
        state = self.server.state
        root = self._selected_value(remote_source or state.remote_source)
        if not root:
            return
        for field in ("remote_data_directory", "remote_instrument_directory", "remote_cif_directory"):
            setattr(state, field, root)
        for field in (
            "remote_data_subdirectory",
            "remote_instrument_subdirectory",
            "remote_cif_subdirectory",
            "remote_data_uri",
            "remote_instrument_uri",
            "remote_main_cif_uri",
        ):
            setattr(state, field, "")
        self.refresh_remote_browser()

    def refresh_remote_browser(self, **_: Any) -> None:
        state = self.server.state
        root = str(state.remote_source or "")
        if not root:
            return
        try:
            for prefix, role in (("data", "data"), ("instrument", "instrument"), ("cif", "cif")):
                directory_field = f"remote_{prefix}_directory"
                directory = str(getattr(state, directory_field, "") or root)
                entries = self.service.list_remote_files(directory, role=role)
                setattr(
                    state,
                    f"remote_{prefix}_directories",
                    [entry for entry in entries if entry["kind"] == "directory"],
                )
                setattr(
                    state,
                    f"remote_{prefix}_files",
                    [entry for entry in entries if entry["kind"] == "file"],
                )
            state.remote_available = True
        except Exception as exc:
            state.remote_message = f"Could not browse the selected Galaxy source: {exc}"
        state.flush()

    def _open_remote_directory(self, prefix: str) -> None:
        state = self.server.state
        selection = str(getattr(state, f"remote_{prefix}_subdirectory", "") or "")
        if not selection:
            return
        setattr(state, f"remote_{prefix}_directory", selection)
        setattr(state, f"remote_{prefix}_subdirectory", "")
        selected_file = "remote_main_cif_uri" if prefix == "cif" else f"remote_{prefix}_uri"
        setattr(state, selected_file, "")
        self.refresh_remote_browser()

    def _up_remote_directory(self, prefix: str) -> None:
        state = self.server.state
        root = str(state.remote_source or "")
        current = str(getattr(state, f"remote_{prefix}_directory", "") or root)
        setattr(state, f"remote_{prefix}_directory", self.service.remote_parent_uri(current, root))
        self.refresh_remote_browser()

    def open_remote_data_directory(self, **_: Any) -> None:
        self._open_remote_directory("data")

    def up_remote_data_directory(self, **_: Any) -> None:
        self._up_remote_directory("data")

    def open_remote_instrument_directory(self, **_: Any) -> None:
        self._open_remote_directory("instrument")

    def up_remote_instrument_directory(self, **_: Any) -> None:
        self._up_remote_directory("instrument")

    def open_remote_cif_directory(self, **_: Any) -> None:
        self._open_remote_directory("cif")

    def up_remote_cif_directory(self, **_: Any) -> None:
        self._up_remote_directory("cif")

    def select_remote_data(self, value: Any = None, **_: Any) -> None:
        self.server.state.remote_data_uri = self._selected_value(value)
        self.server.state.flush()

    def select_remote_instrument(self, value: Any = None, **_: Any) -> None:
        self.server.state.remote_instrument_uri = self._selected_value(value)
        self.server.state.flush()

    def select_remote_main_cif(self, value: Any = None, **_: Any) -> None:
        self.server.state.remote_main_cif_uri = self._selected_value(value)
        self.server.state.flush()

    def _facility_site_changed(self, facility_site: Any = None, **_: Any) -> None:
        state = self.server.state
        site = self._selected_value(facility_site or state.facility_site).upper() or "SNS"
        try:
            self.facility = FacilityBrowser(facility=site)
            state.facility_site = self.facility.facility
            state.facility_root = str(self.facility.root)
            state.facility_available = self.facility.available
            state.facility_instruments = self.facility.list_instruments()
            state.facility_instrument = ""
            state.facility_ipts = ""
            state.facility_ipts_options = []
            state.facility_browser_message = f"Select a {site} instrument and IPTS."
        except Exception as exc:
            state.facility_available = False
            state.facility_instruments = []
            state.facility_browser_message = str(exc)
        self._reset_facility_browser()
        state.flush()

    def _facility_instrument_changed(self, facility_instrument: Any = None, **_: Any) -> None:
        state = self.server.state
        instrument = self._selected_value(facility_instrument or state.facility_instrument)
        state.facility_instrument = instrument
        state.facility_ipts = ""
        state.facility_ipts_options = self.facility.list_ipts(instrument) if instrument else []
        self._reset_facility_browser()
        state.flush()

    def _facility_ipts_changed(self, facility_ipts: Any = None, **_: Any) -> None:
        state = self.server.state
        ipts = self._selected_value(facility_ipts or state.facility_ipts)
        state.facility_ipts = ipts
        if ipts:
            self.refresh_facility_browser()
        else:
            self._reset_facility_browser()
            state.flush()

    def _reset_facility_browser(self) -> None:
        state = self.server.state
        for field, value in (
            ("facility_working_directory", "."),
            ("facility_working_subdirectory", ""),
            ("facility_working_directories", []),
            ("facility_working_readable", False),
            ("facility_working_writable", False),
            ("facility_data_directory", "."),
            ("facility_data_subdirectory", ""),
            ("facility_data_directories", []),
            ("facility_data_files", []),
            ("facility_data_path", ""),
            ("facility_data_relative_path", ""),
            ("facility_instrument_directory", "."),
            ("facility_instrument_subdirectory", ""),
            ("facility_instrument_directories", []),
            ("facility_instrument_files", []),
            ("facility_cif_directory", "."),
            ("facility_cif_subdirectory", ""),
            ("facility_cif_directories", []),
            ("facility_cif_files", []),
            ("facility_instrument_path", ""),
            ("facility_instrument_relative_path", ""),
            ("facility_main_cif_path", ""),
            ("facility_main_cif_relative_path", ""),
            ("facility_output_directory", "."),
            ("facility_output_subdirectory", ""),
            ("facility_output_directories", []),
        ):
            setattr(state, field, value)

    def refresh_facility_browser(self, **_: Any) -> None:
        state = self.server.state
        state.facility_browser_message = ""
        try:
            instrument = str(state.facility_instrument or "")
            ipts = str(state.facility_ipts or "")
            if not instrument or not ipts:
                raise FacilityPathError(f"Choose a {self.facility.facility} instrument and IPTS")
            directory = str(state.facility_working_directory or ".")
            data_entries = self.facility.list_directory(
                instrument, ipts, directory, role="data"
            )
            instrument_entries = self.facility.list_directory(
                instrument, ipts, directory, role="instrument"
            )
            cif_entries = self.facility.list_directory(
                instrument, ipts, directory, role="cif"
            )
            directories = [item.as_item() for item in data_entries if item.kind == "directory"]
            state.facility_working_directories = directories
            state.facility_data_directories = directories
            state.facility_data_files = [item.as_item() for item in data_entries if item.kind == "file"]
            state.facility_instrument_directories = directories
            state.facility_instrument_files = [item.as_item() for item in instrument_entries if item.kind == "file"]
            state.facility_cif_directories = directories
            state.facility_cif_files = [item.as_item() for item in cif_entries if item.kind == "file"]
            state.facility_output_directories = directories
            state.facility_data_directory = directory
            state.facility_instrument_directory = directory
            state.facility_cif_directory = directory
            state.facility_output_directory = directory
            access = self.facility.directory_access(instrument, ipts, directory)
            state.facility_working_readable = bool(access["readable"] and access["searchable"])
            state.facility_working_writable = bool(access["writable"])
            access_label = "read/write" if state.facility_working_writable else "read only in this session"
            state.facility_browser_message = (
                f"Browsing {self.facility.root}/{instrument}/{ipts}/{directory} ({access_label}). "
                "Only this folder's compatible files and immediate subfolders are shown. "
                "Result export is delegated to NDIP and does not require this NOVA pod to have write access."
            )
        except Exception as exc:
            state.facility_working_readable = False
            state.facility_working_writable = False
            state.publish_results_to_ipts = False
            state.facility_working_directories = []
            state.facility_data_files = []
            state.facility_instrument_files = []
            state.facility_cif_files = []
            state.facility_browser_message = str(exc)
        state.flush()

    def _clear_facility_file_selections(self) -> None:
        state = self.server.state
        for field in (
            "facility_data_path",
            "facility_data_relative_path",
            "facility_instrument_path",
            "facility_instrument_relative_path",
            "facility_main_cif_path",
            "facility_main_cif_relative_path",
        ):
            setattr(state, field, "")

    def open_facility_working_directory(self, value: Any = None, **_: Any) -> None:
        state = self.server.state
        selected = self._selected_value(value) or str(state.facility_working_subdirectory or "")
        if not selected:
            return
        state.facility_working_directory = selected
        state.facility_working_subdirectory = ""
        self._clear_facility_file_selections()
        self.refresh_facility_browser()

    def up_facility_working_directory(self, **_: Any) -> None:
        state = self.server.state
        current = str(state.facility_working_directory or ".")
        state.facility_working_directory = self.facility.parent_directory(current)
        state.facility_working_subdirectory = ""
        self._clear_facility_file_selections()
        self.refresh_facility_browser()

    # Kept for the currently hidden legacy automation panel while saved NOVA
    # sessions transition to the single working-folder state model.
    def open_facility_output_directory(self, **_: Any) -> None:
        self.open_facility_working_directory()

    def up_facility_output_directory(self, **_: Any) -> None:
        self.up_facility_working_directory()

    def _select_facility_file(self, value: Any, *, role: str, path_field: str, relative_field: str) -> None:
        state = self.server.state
        relative = self._selected_value(value)
        setattr(state, relative_field, relative)
        setattr(state, path_field, "")
        if not relative:
            state.flush()
            return
        try:
            selection = self.facility.select_file(
                str(state.facility_instrument or ""),
                str(state.facility_ipts or ""),
                relative,
                role=role,
            )
            setattr(state, path_field, selection.absolute_path)
            state.facility_browser_message = f"Selected {selection.relative_path}"
        except Exception as exc:
            state.facility_browser_message = str(exc)
            setattr(state, relative_field, "")
        state.flush()

    def select_facility_data(self, value: Any = None, **_: Any) -> None:
        self._select_facility_file(
            value, role="data", path_field="facility_data_path", relative_field="facility_data_relative_path"
        )

    def select_facility_instrument(self, value: Any = None, **_: Any) -> None:
        self._select_facility_file(
            value,
            role="instrument",
            path_field="facility_instrument_path",
            relative_field="facility_instrument_relative_path",
        )

    def select_facility_main_cif(self, value: Any = None, **_: Any) -> None:
        self._select_facility_file(
            value,
            role="cif",
            path_field="facility_main_cif_path",
            relative_field="facility_main_cif_relative_path",
        )

    def create_facility_output_directory(self, **_: Any) -> None:
        state = self.server.state
        try:
            created = self.facility.create_directory(
                str(state.facility_instrument or ""),
                str(state.facility_ipts or ""),
                str(state.facility_output_directory or "shared"),
                str(state.facility_new_output_folder or ""),
            )
            state.facility_output_directory = created
            state.facility_new_output_folder = "radar-pd-results"
            state.facility_browser_message = f"Created {created}"
            self.refresh_facility_browser()
        except Exception as exc:
            state.facility_browser_message = str(exc)
            state.flush()

    def save_watch_recipe(self, **_: Any) -> None:
        state = self.server.state
        state.watch_recipe_message = ""
        try:
            if state.input_source != InputSource.IPTS_BROWSER.value:
                raise ValueError("Choose Browse SNS IPTS shared files first")
            if not state.use_builtin_cuka and state.instrument_source != "ipts":
                raise ValueError("A folder watcher must use an instrument profile stored in this IPTS")
            if state.main_cif_source not in {"none", "ipts"}:
                raise ValueError("A folder watcher must use a main CIF stored in this IPTS or no main CIF")
            config = self._configuration()
            config_path = self.facility.write_configuration(
                str(state.facility_instrument or ""),
                str(state.facility_ipts or ""),
                str(state.facility_data_directory or "shared"),
                config.portable_contract(),
            )
            ipts_root = (self.facility.root / str(state.facility_instrument) / str(state.facility_ipts)).resolve()
            config_relative = config_path.resolve().relative_to(ipts_root).as_posix()
            recipe = WatchRecipe(
                instrument=str(state.facility_instrument or ""),
                ipts=str(state.facility_ipts or ""),
                source_directory=str(state.facility_data_directory or "shared"),
                output_directory=str(state.facility_output_directory or "shared"),
                configuration=config_relative,
                instrument_profile=str(state.facility_instrument_relative_path or "") or None,
                use_builtin_cuka=bool(state.use_builtin_cuka),
                main_cif=(str(state.facility_main_cif_relative_path or "") or None)
                if state.main_cif_source == "ipts"
                else None,
                include_patterns=[item.strip() for item in str(state.watch_patterns or "").split(",") if item.strip()],
                settle_seconds=int(state.watch_settle_seconds),
                process_existing=bool(state.watch_process_existing),
                max_attempts=int(state.watch_max_attempts),
                retry_delay_seconds=int(state.watch_retry_delay_seconds),
                analysis_mode=str(state.analysis_mode),
            )
            recipe_path = self.facility.write_watch_recipe(recipe)
            state.watch_recipe_message = (
                f"Saved {recipe_path.name} and {config_path.name} in {recipe.source_directory}. "
                "The persistent NDIP watcher can now consume this recipe."
            )
        except Exception as exc:
            state.watch_recipe_message = str(exc)
        state.flush()

    @staticmethod
    def _submission_payload_expression() -> str:
        return "{" + ",".join(f"{name}:{name}" for name in _SUBMISSION_FIELDS) + "}"

    def _workflow_mode_changed(self, **_: Any) -> None:
        """Keep the setup form visible when users leave the run-history drawer."""

        state = self.server.state
        state.history_panels = []
        state.run_search = ""
        state.flush()

    def _radiation_changed(self, radiation: Any = None, **_: Any) -> None:
        """Keep single-pattern controls scientifically compatible with the source."""

        state = self.server.state
        selected = self._selected_value(radiation if radiation is not None else state.radiation)
        if selected not in {"neutron", "xray"}:
            return
        previous = str(getattr(state, "last_radiation", selected) or selected)
        measurement_changed = previous in {"neutron", "xray"} and previous != selected
        state.radiation = selected
        if measurement_changed:
            # A diffraction pattern, instrument profile, and simulated candidate
            # library are radiation-specific. Requiring an explicit reselection
            # prevents a restored POWGEN neutron run from appearing valid after
            # the user changes only the measurement type to X-ray (or vice versa).
            for field in (
                "data_path",
                "history_data_id",
                "remote_data_uri",
                "facility_data_path",
                "facility_data_relative_path",
                "event_file_path",
                "instrument_path",
                "history_instrument_id",
                "remote_instrument_uri",
                "facility_instrument_path",
                "facility_instrument_relative_path",
                "database_archive_path",
                "history_database_id",
            ):
                setattr(state, field, "")
            state.instrument_source = "upload"
            state.database_source = "builtin"
            state.library_archive_source = "computer"
            state.use_builtin_cuka = False
        if selected == "xray":
            state.instrument_mode_options = list(_XRAY_GEOMETRY_OPTIONS)
            state.source_options = list(_XRAY_SOURCE_OPTIONS)
            state.instrument_source_options = list(_XRAY_INSTRUMENT_SOURCE_OPTIONS)
            state.main_cif_source_options = list(_XRAY_MAIN_CIF_SOURCE_OPTIONS)
            incompatible = False
            if state.instrument_mode == "tof":
                state.instrument_mode = "auto"
                incompatible = True
            if state.input_source in {"ipts_browser", "ipts_event", "ipts_manual"}:
                state.input_source = "upload"
                incompatible = True
            if state.instrument_source == "ipts":
                state.instrument_source = "upload"
                incompatible = True
            if state.main_cif_source == "ipts":
                state.main_cif_source = "none"
                incompatible = True
            state.use_facility_workspace = False
            state.magnetic_precheck = False
            if (incompatible or measurement_changed) and not getattr(state, "busy", False):
                state.notice = (
                    "X-ray mode uses upload or Galaxy History inputs and constant-wavelength geometry. "
                    "Reselect the diffraction pattern, instrument profile, and candidate library for X-ray. "
                    "Chemistry and the optional main-phase CIF were preserved."
                )
        else:
            state.instrument_mode_options = list(_NEUTRON_GEOMETRY_OPTIONS)
            state.source_options = list(_GENERAL_SOURCE_OPTIONS)
            state.instrument_source_options = list(_GENERAL_INSTRUMENT_SOURCE_OPTIONS)
            state.main_cif_source_options = list(_GENERAL_MAIN_CIF_SOURCE_OPTIONS)
            state.use_builtin_cuka = False
            state.light_calibration_enabled = False
            if measurement_changed and not getattr(state, "busy", False):
                state.notice = (
                    "Measurement type changed to neutron. Reselect the diffraction pattern, instrument "
                    "profile, and candidate library. Chemistry and the optional main-phase CIF were preserved."
                )
        state.last_radiation = selected
        state.flush()

    @staticmethod
    def _configuration_summary_rows(config: AnalysisConfig) -> list[dict[str, str]]:
        limits = "Full pattern"
        if config.limits:
            limits = f"{config.limits[0]:g} to {config.limits[1]:g} (pattern x-axis units)"
        sample = ", ".join(config.sample_elements) or "Not constrained"
        environment = ", ".join(config.environment_elements) or "None"
        radiation = "X-ray" if config.radiation.value == "xray" else "Neutron"
        rows = [
            {"label": "Mode", "value": config.mode.value.title()},
            {
                "label": "Measurement",
                "value": f"{radiation} / {config.instrument_mode.upper()}",
            },
            {"label": "Sample elements", "value": sample},
            {"label": "Can / environment", "value": environment},
            {"label": "Fit region", "value": limits},
            {
                "label": "Background",
                "value": f"{config.background_mode.replace('_', ' ')} / {config.background_type} / {config.background_terms} terms",
            },
        ]
        if config.mode == AnalysisMode.FULL:
            rows.append(
                {
                    "label": "Full search",
                    "value": f"{config.full_profile.title()} / up to {config.full_max_passes} discovery pass(es)",
                }
            )
        else:
            rows.append(
                {
                    "label": "Rapid budget",
                    "value": (
                        f"{config.rapid_phases_per_hypothesis} phase(s) per hypothesis / "
                        f"{config.rapid_gsas_validation_limit} final refinements"
                    ),
                }
            )
        if config.reference_masks_enabled:
            width = config.reference_window_mode.title()
            if config.reference_window_mode == "fixed" and config.reference_fixed_half_width is not None:
                width = f"Fixed half-width {config.reference_fixed_half_width:g}"
            rows.append(
                {
                    "label": "Reference masks",
                    "value": f"{', '.join(config.reference_mask_presets) or 'No structures selected'} / {width}",
                }
            )
        rows.append(
            {
                "label": "Candidate controls",
                "value": (
                    f"dedup {config.full_dedup_threshold:g} / Q max {config.full_score_q_max:g} / "
                    f"Pearson cutoff {config.full_pearson_cell_min_r:g} / "
                    f"pruning {'on' if config.full_candidate_pruning else 'off'}"
                ),
            }
        )
        rows.append(
            {
                "label": "Excluded space groups",
                "value": ", ".join(str(value) for value in config.excluded_space_groups) or "None",
            }
        )
        if config.radiation.value == "xray":
            rows.append(
                {
                    "label": "Light PXRD calibration",
                    "value": "Enabled" if config.light_calibration_enabled else "Disabled",
                }
            )
        rows.append(
            {
                "label": "Main-phase refinement",
                "value": (
                    "Cell anchor enabled"
                    if config.main_prenudge
                    else "No main-phase cell anchor in this configuration"
                ),
            }
        )
        return rows

    def _set_powgen_configuration_preview(
        self,
        dataset_id: str,
        config: AnalysisConfig | None = None,
    ) -> None:
        state = self.server.state
        state.powgen_configuration_dataset_id = dataset_id
        state.powgen_configuration_summary = []
        state.powgen_configuration_yaml = ""
        state.powgen_configuration_error = ""
        state.powgen_configuration_title = "No reusable configuration selected"
        state.powgen_configuration_label = "Not selected"
        if not dataset_id:
            return
        try:
            selected = config or self.service.load_configuration_dataset(dataset_id)
            label = self._powgen_dataset_label(
                state.history_configuration_datasets,
                dataset_id,
                "Selected Galaxy configuration",
            )
            state.powgen_configuration_label = label
            state.powgen_configuration_title = f"{selected.mode.value.title()} configuration details"
            state.powgen_configuration_summary = self._configuration_summary_rows(selected)
            try:
                contract = self.service.load_json_document(dataset_id)
            except Exception:
                contract = selected.portable_contract()
                contract.pop("created_utc", None)
            state.powgen_configuration_yaml = yaml.safe_dump(
                contract,
                sort_keys=False,
                allow_unicode=False,
            )
        except Exception as exc:
            state.powgen_configuration_title = "Configuration could not be read"
            state.powgen_configuration_error = str(exc)

    def _powgen_configuration_changed(self, value: Any = None, **_: Any) -> None:
        state = self.server.state
        dataset_id = self._selected_value(value)
        self._set_powgen_configuration_preview(dataset_id)
        state.powgen_preflight_ready = False
        state.powgen_preflight_status = "idle"
        state.powgen_preflight_message = "Check the experiment after changing its configuration."
        state.powgen_preflight_details = []
        state.flush()

    def _set_history_configuration_preview(
        self,
        dataset_id: str,
        config: AnalysisConfig | None = None,
    ) -> None:
        state = self.server.state
        state.history_configuration_id = dataset_id
        state.history_configuration_summary = []
        state.history_configuration_yaml = ""
        state.history_configuration_error = ""
        state.history_configuration_title = "No reusable configuration selected"
        if not dataset_id:
            return
        try:
            selected = config or self.service.load_configuration_dataset(dataset_id)
            state.history_configuration_title = f"{selected.mode.value.title()} configuration details"
            state.history_configuration_summary = self._configuration_summary_rows(selected)
            try:
                contract = self.service.load_json_document(dataset_id)
            except Exception:
                contract = selected.portable_contract()
                contract.pop("created_utc", None)
            state.history_configuration_yaml = yaml.safe_dump(
                contract,
                sort_keys=False,
                allow_unicode=False,
            )
        except Exception as exc:
            state.history_configuration_title = "Configuration could not be read"
            state.history_configuration_error = str(exc)

    def _history_configuration_changed(self, value: Any = None, **_: Any) -> None:
        self._set_history_configuration_preview(self._selected_value(value))
        self.server.state.flush()

    def _submission_form_changed(self, **_: Any) -> None:
        """Invalidate stale validation feedback and rotate form idempotency."""

        state = self.server.state
        if getattr(state, "busy", False):
            return
        state.error_message = ""
        state.form_revision = int(getattr(state, "form_revision", 0) or 0) + 1
        state.submission_token = uuid.uuid4().hex

    def _form_state(self, payload: dict[str, Any] | None = None) -> Any:
        state = self.server.state
        values = {
            name: (payload[name] if isinstance(payload, dict) and name in payload else getattr(state, name))
            for name in _SUBMISSION_FIELDS
        }
        return SimpleNamespace(**values)

    def _configuration(self, payload: dict[str, Any] | None = None) -> AnalysisConfig:
        state = self._form_state(payload)
        run_name = str(state.run_name or "").strip()
        fit_start = _optional_float(state.fit_start)
        fit_end = _optional_float(state.fit_end)
        if (fit_start is None) != (fit_end is None):
            raise ValueError("Fit range requires both a start and an end value")
        has_main_phase = (
            (state.main_cif_source == "upload" and bool(state.main_cif_path))
            or (state.main_cif_source == "galaxy" and bool(state.history_main_cif_id))
            or (state.main_cif_source == "galaxy_remote" and bool(state.remote_main_cif_uri))
            or (state.main_cif_source == "ipts" and bool(state.facility_main_cif_path))
        )
        presets: dict[str, dict[str, Any]] = {
            "quick": {
                "full_max_passes": 1,
                "full_min_phase_percent": 1.0,
                "full_top_n_ml": 12,
                "full_nudge_candidates": 3,
                "full_nudge_samples": 800,
                "full_nudge_representatives": 10,
                "full_compare_candidates": 1,
                "full_compare_cycles": 4,
                "full_cell_length_tolerance_pct": 0.6,
                "full_cell_angle_tolerance_deg": 1.5,
                "full_rwp_improvement_threshold": 0.12,
                "full_dedup_threshold": 0.95,
                "full_score_q_max": 8.0,
                "full_pearson_cell_min_r": 0.50,
                "full_lattice_tiebreak_score_tol": 0.0005,
                "full_candidate_pruning": True,
                "full_knee_min_points_hist": 5,
                "full_knee_min_relative_span": 0.03,
                "full_knee_keep_if_no_knee": 1,
                "full_knee_keep_at_most": 3,
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
                "full_cell_length_tolerance_pct": 1.0,
                "full_cell_angle_tolerance_deg": 3.0,
                "full_rwp_improvement_threshold": 0.06,
                "full_dedup_threshold": 0.95,
                "full_score_q_max": 8.0,
                "full_pearson_cell_min_r": 0.50,
                "full_lattice_tiebreak_score_tol": 0.0005,
                "full_candidate_pruning": True,
                "full_knee_min_points_hist": 5,
                "full_knee_min_relative_span": 0.03,
                "full_knee_keep_if_no_knee": 2,
                "full_knee_keep_at_most": 5,
            },
            "thorough": {
                "full_max_passes": 3,
                "full_min_phase_percent": 0.25,
                "full_top_n_ml": 75,
                "full_nudge_candidates": 12,
                "full_nudge_samples": 20000,
                "full_nudge_representatives": 150,
                "full_compare_candidates": 3,
                "full_compare_cycles": 8,
                "full_cell_length_tolerance_pct": 2.0,
                "full_cell_angle_tolerance_deg": 5.0,
                "full_rwp_improvement_threshold": 0.0,
                "full_dedup_threshold": 0.95,
                "full_score_q_max": 8.0,
                "full_pearson_cell_min_r": 0.10,
                "full_lattice_tiebreak_score_tol": 0.0005,
                "full_candidate_pruning": True,
                "full_knee_min_points_hist": 5,
                "full_knee_min_relative_span": 0.03,
                "full_knee_keep_if_no_knee": 0,
                "full_knee_keep_at_most": 10,
            },
        }
        values: dict[str, Any] = {
            "mode": state.analysis_mode,
            "radiation": state.radiation,
            "instrument_mode": state.instrument_mode,
            "sample_elements": state.sample_elements,
            "environment_elements": state.environment_elements,
            "limits": None if fit_start is None else (fit_start, fit_end),
            "exclude_regions": self._parse_regions(state.ignore_regions),
            "reference_masks_enabled": bool(state.reference_masks_enabled),
            "reference_mask_presets": _list_value(state.reference_mask_presets),
            "reference_window_mode": state.reference_window_mode,
            "reference_fixed_half_width": _optional_float(state.reference_fixed_half_width),
            "reference_fwhm_factor": float(state.reference_fwhm_factor),
            "reference_fractional_d_tolerance": float(state.reference_fractional_d_tolerance),
            "reference_zero_tolerance": _optional_float(state.reference_zero_tolerance),
            "reference_min_half_width": _optional_float(state.reference_min_half_width),
            "reference_max_half_width": _optional_float(state.reference_max_half_width),
            "include_cu_kbeta": bool(state.include_cu_kbeta) if state.radiation == "xray" else False,
            "background_mode": state.background_mode,
            "background_type": state.background_type,
            "background_terms": int(state.background_terms),
            "main_prenudge": bool(state.main_prenudge) if has_main_phase else False,
            "main_shadow_filter": bool(state.main_shadow_filter) if has_main_phase else False,
            "cleanup_enabled": bool(state.cleanup_enabled) if has_main_phase else False,
            "refine_u_iso": bool(state.refine_u_iso) if has_main_phase and state.cleanup_enabled else False,
            "refine_positions": bool(state.refine_positions) if has_main_phase and state.cleanup_enabled else False,
            "light_calibration_enabled": bool(state.light_calibration_enabled) if has_main_phase and state.radiation == "xray" else False,
            "magnetic_precheck": bool(state.magnetic_precheck) if has_main_phase and state.radiation == "neutron" else False,
            "magnetic_q_max": float(state.magnetic_q_max),
            "magnetic_denominators": [
                int(value)
                for value in str(state.magnetic_denominators or "").replace(";", ",").split(",")
                if value.strip()
            ],
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
            "full_rwp_improvement_threshold": float(state.full_rwp_improvement_threshold),
            "full_dedup_threshold": float(state.full_dedup_threshold),
            "full_score_q_max": float(state.full_score_q_max),
            "full_pearson_cell_min_r": float(state.full_pearson_cell_min_r),
            "full_lattice_tiebreak_score_tol": float(state.full_lattice_tiebreak_score_tol),
            "full_candidate_pruning": bool(state.full_candidate_pruning),
            "full_knee_min_points_hist": int(state.full_knee_min_points_hist),
            "full_knee_min_relative_span": float(state.full_knee_min_relative_span),
            "full_knee_keep_if_no_knee": int(state.full_knee_keep_if_no_knee),
            "full_knee_keep_at_most": int(state.full_knee_keep_at_most),
            "excluded_space_groups": [
                int(value)
                for value in re.split(r"[,;\s]+", str(state.excluded_space_groups or "").strip())
                if value
            ],
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

    def _inputs(self, payload: dict[str, Any] | None = None) -> InputSelection:
        state = self._form_state(payload)
        source = InputSource(state.input_source)
        instrument_source = "builtin" if bool(state.use_builtin_cuka) else str(state.instrument_source or "upload")
        if state.radiation == "xray" and source in {InputSource.IPTS_EVENT, InputSource.IPTS_MANUAL}:
            raise ValueError("SNS IPTS resolution is available only for neutron data")
        if state.main_cif_source == "upload":
            main_cif_path = state.main_cif_path or None
        elif state.main_cif_source == "ipts":
            main_cif_path = state.facility_main_cif_path or None
        else:
            main_cif_path = None
        main_cif_dataset_id = state.history_main_cif_id or None if state.main_cif_source == "galaxy" else None
        main_cif_remote_uri = (
            state.remote_main_cif_uri or None if state.main_cif_source == "galaxy_remote" else None
        )
        database_archive_path = (
            state.database_archive_path or None
            if state.database_source == "archive" and state.library_archive_source == "computer"
            else None
        )
        database_dataset_id = (
            state.history_database_id or None
            if state.database_source == "archive" and state.library_archive_source == "galaxy"
            else None
        )
        uses_facility_scope = (
            bool(state.use_facility_workspace)
            or source == InputSource.IPTS_BROWSER
            or instrument_source == "ipts"
            or state.main_cif_source == "ipts"
            or bool(state.publish_results_to_ipts)
        )
        return InputSelection(
            source=source,
            instrument_source=instrument_source,
            data_path=(state.data_path or None)
            if source == InputSource.UPLOAD
            else ((state.facility_data_path or None) if source == InputSource.IPTS_BROWSER else None),
            data_dataset_id=(state.history_data_id or None) if source == InputSource.GALAXY else None,
            data_remote_uri=(state.remote_data_uri or None) if source == InputSource.GALAXY_REMOTE else None,
            instrument_path=(state.instrument_path or None)
            if instrument_source == "upload"
            else ((state.facility_instrument_path or None) if instrument_source == "ipts" else None),
            instrument_dataset_id=(state.history_instrument_id or None) if instrument_source == "galaxy" else None,
            instrument_remote_uri=(state.remote_instrument_uri or None)
            if instrument_source == "galaxy_remote"
            else None,
            main_cif_path=main_cif_path,
            main_cif_dataset_id=main_cif_dataset_id,
            main_cif_remote_uri=main_cif_remote_uri,
            database_archive_path=database_archive_path,
            database_dataset_id=database_dataset_id,
            use_builtin_cuka=bool(state.use_builtin_cuka) if state.radiation == "xray" else False,
            event_file_path=state.event_file_path or None,
            facility_root=str(self.facility.root) if uses_facility_scope else "/SNS",
            instrument=(state.facility_instrument or None)
            if uses_facility_scope
            else (state.ipts_instrument or None),
            ipts=(state.facility_ipts or None) if uses_facility_scope else (state.ipts or None),
            run_number=int(state.run_number) if state.run_number not in (None, "") else None,
            bank=state.bank or None,
            data_relative_path=(state.facility_data_relative_path or None)
            if source == InputSource.IPTS_BROWSER
            else None,
            instrument_relative_path=(state.facility_instrument_relative_path or None)
            if instrument_source == "ipts"
            else None,
            main_cif_relative_path=(state.facility_main_cif_relative_path or None)
            if state.main_cif_source == "ipts"
            else None,
            publish_results_to_ipts=bool(state.publish_results_to_ipts) if uses_facility_scope else False,
            publish_directory=(state.facility_working_directory or None)
            if uses_facility_scope and state.publish_results_to_ipts
            else None,
            # Retained in the model only for recovery of older sessions. New
            # exports write one uniquely named archive into publish_directory.
            publish_subfolder=None,
        )

    @staticmethod
    def _normalize_powgen_ipts(value: Any) -> str:
        text = str(value or "").strip().upper()
        if text.isdigit():
            text = f"IPTS-{text}"
        if not re.fullmatch(r"IPTS-[1-9][0-9]*", text):
            raise ValueError("POWGEN experiment must have the form IPTS-<number>")
        return text

    def _sync_powgen_rows(self) -> None:
        """Project controller lifecycle state into compact user-facing rows."""

        controller = self._powgen_controller
        if controller is None:
            self.server.state.powgen_rows = []
            self.server.state.powgen_scientific_rows = []
            self.server.state.powgen_all_scientific_rows = []
            self.server.state.powgen_sample_options = []
            return

        rows: list[dict[str, Any]] = []
        active_count = len(controller.state.submitted)
        active_limit = controller.settings.max_active_jobs
        now_utc = datetime.now(timezone.utc)
        phases = (
            ("Failed", controller.state.failed),
            ("Completed", controller.state.completed),
            ("Submitted", controller.state.submitted),
            ("Discovered", controller.state.discovered),
        )
        for lifecycle, runs in phases:
            for run_id, run in runs.items():
                record = controller.records.get(run_id)
                status = lifecycle
                color = "#eef2ef"
                if lifecycle == "Failed":
                    color = "#fde7e7"
                    detail = run.error or "RADAR-PD submission or analysis failed."
                elif lifecycle == "Completed":
                    color = "#dff2e8"
                    count = len(run.galaxy_result_ids)
                    detail = f"{count} Galaxy result dataset{'s' if count != 1 else ''} ready"
                elif lifecycle == "Submitted":
                    record_status = str(getattr(getattr(record, "status", None), "value", "") or "")
                    if record_status == "running":
                        status = "Running"
                        color = "#dcebf8"
                    elif record_status in {"queued", "uploading", "new"}:
                        status = "Queued"
                        color = "#fff0d4"
                    else:
                        color = "#fff0d4"
                    stage = str(record.stage or "Waiting for Galaxy") if record is not None else "Waiting for Galaxy"
                    progress = int(record.progress or 0) if record is not None else 0
                    job_id = run.galaxy_job_id or (record.galaxy_job_id if record is not None else "")
                    detail = f"{stage} · {progress}%{f' · job {job_id[:8]}' if job_id else ''}"
                else:
                    if run.submission_attempts:
                        color = "#fff0d4"
                        retry_at = _powgen_retry_time(run.next_retry_utc)
                        reason = _powgen_submission_error_summary(run.error)
                        attempt = (
                            f"Attempt {run.submission_attempts} of "
                            f"{controller.settings.max_submission_attempts} did not start."
                        )
                        if retry_at is not None and retry_at > now_utc:
                            status = "Waiting to retry"
                            detail = (
                                f"{reason} {attempt} Automatic retry after "
                                f"{_format_eastern_time(retry_at, '%H:%M:%S %Z')}."
                            )
                        elif active_count >= active_limit:
                            status = "Waiting for slot"
                            detail = (
                                f"{reason} {attempt} Retry is ready and will start automatically "
                                f"when one of the {active_limit} active analysis slots is available."
                            )
                        else:
                            status = "Retry queued"
                            detail = (
                                f"{reason} {attempt} The monitor will retry automatically on its "
                                "next submission check."
                            )
                    else:
                        detail = "Discovered in the read-only autoreduce folder; preparing submission."
                rows.append(
                    {
                        "run_id": run_id,
                        "run_number": int(run.run_number),
                        "file": Path(run.source_path).name,
                        "status": status,
                        "color": color,
                        "detail": detail,
                    }
                )
        state = self.server.state
        state.powgen_rows = sorted(rows, key=lambda item: item["run_number"], reverse=True)

        scientific_rows: list[dict[str, Any]] = []
        for run_id, run in controller.state.completed.items():
            record = controller.records.get(run_id)
            summary = dict(run.scientific_summary or {})
            if not summary:
                continue
            projected_phases: list[dict[str, Any]] = []
            for phase in phase_fraction_rows({"phases": summary.get("phases") or []}):
                phase_name = str(phase.get("phase") or "Unknown")
                space_group = str(phase.get("space_group") or "-")
                weight = float(phase.get("weight_percent") or 0.0)
                projected_phases.append(
                    {
                        "phase": phase_name,
                        "space_group": space_group,
                        "label": f"{phase_name} (SG {space_group})" if space_group != "-" else phase_name,
                        "weight_percent": weight,
                        "weight_display": f"{weight:.2f}%",
                    }
                )
            elapsed = summary.get("elapsed_seconds")
            elapsed_value = float(elapsed) if elapsed is not None else None
            elapsed_display = (
                f"{elapsed_value / 60:.1f} min" if elapsed_value is not None and elapsed_value >= 60
                else f"{elapsed_value:.1f} s" if elapsed_value is not None
                else "-"
            )
            rwp = summary.get("rwp")
            rwp_value = float(rwp) if rwp is not None else None
            metadata = dict(run.scan_metadata or {})

            def metric_display(key: str, fallback_unit: str, *, digits: int = 3) -> str:
                metric = metadata.get(key)
                if not isinstance(metric, dict) or metric.get("value") is None:
                    return ""
                value = float(metric["value"])
                unit = str(metric.get("unit") or fallback_unit)
                return f"{value:.{digits}g} {unit}".strip()

            temperature_display = metric_display("temperature", "", digits=6)
            field_display = metric_display("magnetic_field", "", digits=4)
            wavelength_display = metric_display("wavelength", "", digits=5)
            temperature_metric = metadata.get("temperature")
            temperature_provenance = ""
            if isinstance(temperature_metric, dict):
                source = str(temperature_metric.get("source") or "").strip()
                source_unit = str(temperature_metric.get("source_unit") or "").strip()
                display_unit = str(temperature_metric.get("unit") or "").strip()
                if source:
                    unit_note = ""
                    if source_unit and display_unit and source_unit != display_unit:
                        unit_note = f"; recorded in {source_unit}, displayed in {display_unit}"
                    elif display_unit:
                        unit_note = f"; unit {display_unit}"
                    else:
                        unit_note = "; unit not reported"
                    temperature_provenance = f"Temperature source: {source}{unit_note}"
            conditions = [
                value
                for value in (
                    f"T {temperature_display}" if temperature_display else "",
                    f"Field {field_display}" if field_display else "",
                    f"lambda {wavelength_display}" if wavelength_display else "",
                )
                if value
            ]
            sample_name = str(metadata.get("sample_name") or "").strip()
            sample_formula = str(metadata.get("sample_formula") or "").strip()
            sample_id = str(metadata.get("sample_id") or "").strip()
            row = {
                "run_id": run_id,
                "record_uid": record.uid if record is not None else "",
                "run_number": int(run.run_number),
                "rwp": rwp_value,
                "rwp_display": f"{rwp_value:.2f}%" if rwp_value is not None else "-",
                "phases": projected_phases,
                "phase_summary": " + ".join(
                    f"{phase['label']} {phase['weight_percent']:.1f}%"
                    for phase in projected_phases[:3]
                ) or "No refined phase fractions",
                "hypothesis": str(summary.get("hypothesis") or ""),
                "elapsed_seconds": elapsed_value,
                "elapsed_display": elapsed_display,
                "analysis_mode": str(summary.get("analysis_mode") or "-").title(),
                "metadata": metadata,
                "conditions_display": " | ".join(conditions) or "Metadata unavailable",
                "temperature_display": temperature_display or "-",
                "temperature_provenance": temperature_provenance,
                "field_display": field_display or "-",
                "wavelength_display": wavelength_display or "-",
                "run_title": str(metadata.get("run_title") or ""),
                "sample_display": " | ".join(
                    value
                    for value in (
                        f"ID {sample_id}" if sample_id else "",
                        sample_name,
                        sample_formula,
                    )
                    if value
                ),
                "warning_count": int(summary.get("warning_count") or 0),
                "error_count": int(summary.get("error_count") or 0),
            }
            sample_identity = experiment_sample_identity(row)
            row["sample_key"] = sample_identity["key"]
            row["sample_label"] = sample_identity["label"]
            scientific_rows.append(row)
        scientific_rows.sort(key=lambda item: item["run_number"], reverse=True)
        complete_experiment_space_groups(scientific_rows)
        for row in scientific_rows:
            for phase in row["phases"]:
                phase_name = str(phase.get("phase") or "Unknown")
                space_group = str(phase.get("space_group") or "-")
                phase["label"] = (
                    f"{phase_name} (SG {space_group})" if space_group != "-" else phase_name
                )
            row["phase_summary"] = " + ".join(
                f"{phase['label']} {phase['weight_percent']:.1f}%"
                for phase in row["phases"][:3]
            ) or "No refined phase fractions"
        state.powgen_all_scientific_rows = scientific_rows
        sample_labels = {
            str(row["sample_key"]): str(row["sample_label"])
            for row in scientific_rows
        }
        state.powgen_sample_options = [
            {"title": sample_labels[key], "value": key}
            for key in sorted(sample_labels, key=lambda key: sample_labels[key].lower())
        ]
        valid_sample_keys = set(sample_labels)
        selected_samples = [
            str(value)
            for value in (state.powgen_selected_samples or [])
            if str(value) in valid_sample_keys
        ]
        state.powgen_selected_samples = selected_samples
        all_scientific_rows = scientific_rows
        if selected_samples:
            selected_sample_keys = set(selected_samples)
            scientific_rows = [
                row for row in all_scientific_rows
                if str(row.get("sample_key") or "unassigned") in selected_sample_keys
            ]
        diagnostics = experiment_fit_diagnostics(scientific_rows)
        for row in scientific_rows:
            diagnostic = diagnostics.get(str(row["run_id"]), {})
            row["quality_label"] = str(diagnostic.get("label") or "No Rwp reported")
            row["quality_color"] = str(diagnostic.get("color") or "#64748b")

        state.powgen_scientific_rows = scientific_rows
        ranked_phase_labels = experiment_phase_labels(scientific_rows)
        state.powgen_phase_options = [
            {"title": label, "value": label}
            for label in ranked_phase_labels
        ]
        selected_phase_labels = [
            str(label)
            for label in (state.powgen_selected_phase_labels or [])
            if str(label) in ranked_phase_labels
        ]
        if not selected_phase_labels:
            selected_phase_labels = ranked_phase_labels[:5]
        state.powgen_selected_phase_labels = selected_phase_labels
        state.powgen_x_axis_options = experiment_axis_options(scientific_rows)
        valid_axes = {str(option["value"]) for option in state.powgen_x_axis_options}
        if str(state.powgen_x_axis or "run_number") not in valid_axes:
            state.powgen_x_axis = "run_number"
        state.powgen_latest_phases = scientific_rows[0]["phases"] if scientific_rows else []
        state.powgen_latest_run_id = scientific_rows[0]["run_id"] if scientific_rows else "-"
        state.powgen_scan_options = [
            {
                "title": f"{row['run_id']} · {row['rwp_display']} Rwp · {row['quality_label']}",
                "value": row["run_id"],
            }
            for row in scientific_rows
        ]
        selected_run_id = str(state.powgen_selected_run_id or "")
        selected = next((row for row in scientific_rows if row["run_id"] == selected_run_id), None)
        if selected is None and scientific_rows:
            selected = scientific_rows[0]
            selected_run_id = str(selected["run_id"])
        state.powgen_selected_run_id = selected_run_id if selected is not None else ""
        state.powgen_selected_scan = selected or {}
        state.powgen_selected_quality_label = str((selected or {}).get("quality_label") or "Collecting baseline")
        state.powgen_selected_quality_color = str((selected or {}).get("quality_color") or "#eef2ef")
        state.powgen_selected_phases = list(selected.get("phases") or []) if selected is not None else []
        state.powgen_tracked_phase_count = len(
            {
                str(phase.get("label") or "Unknown")
                for row in scientific_rows
                for phase in (row.get("phases") or [])
                if isinstance(phase, dict)
            }
        )
        state.powgen_attention_count = sum(
            1
            for row in scientific_rows
            if row.get("quality_label") in {"Elevated Rwp", "Rwp outlier"}
            or int(row.get("warning_count") or 0) > 0
            or int(row.get("error_count") or 0) > 0
        )

        rwp_values = sorted(row["rwp"] for row in scientific_rows if row["rwp"] is not None)
        median_rwp = None
        if rwp_values:
            middle = len(rwp_values) // 2
            median_rwp = (
                rwp_values[middle]
                if len(rwp_values) % 2
                else (rwp_values[middle - 1] + rwp_values[middle]) / 2
            )
        state.powgen_dashboard_metrics = [
            {
                "label": "Completed scans",
                "value": str(len(controller.state.completed)),
                "detail": (
                    f"Showing {len(scientific_rows)} of {len(all_scientific_rows)} scientific summaries"
                    if selected_samples
                    else f"{len(scientific_rows)} with scientific summaries"
                ),
            },
            {
                "label": "Active analyses",
                "value": str(len(controller.state.submitted)),
                "detail": f"{len(controller.state.discovered)} awaiting submission",
            },
            {
                "label": "Median Rwp",
                "value": f"{median_rwp:.2f}%" if median_rwp is not None else "-",
                "detail": "Across summarized scans",
            },
            {
                "label": "Phases tracked",
                "value": str(state.powgen_tracked_phase_count),
                "detail": "Distinct refined phase labels",
            },
            {
                "label": "Needs review",
                "value": str(state.powgen_attention_count + len(controller.state.failed)),
                "detail": (
                    f"{len(controller.state.failed)} failed, {state.powgen_attention_count} trend/warning flags"
                    if controller.state.failed or state.powgen_attention_count
                    else "No current processing or trend flags"
                ),
            },
        ]
        state.powgen_dashboard_notice = (
            f"Showing {len(scientific_rows)} of {len(all_scientific_rows)} completed scan summaries "
            f"from {controller.source_directory}. "
            "Conditions are read from each matching PG3 NeXus file. Missing phase points mean that a phase "
            "was not reported for that scan, not a forced zero fraction."
            if scientific_rows
            else "Completed scans will populate this experiment view as their normalized Galaxy summaries become available."
        )
        self._update_powgen_dashboard_figures(scientific_rows)

    def _update_powgen_dashboard_figures(self, rows: list[dict[str, Any]]) -> None:
        x_key = str(self.server.state.powgen_x_axis or "run_number")
        selected_labels = [str(label) for label in (self.server.state.powgen_selected_phase_labels or [])]
        if self._powgen_phase_widget is not None:
            self._powgen_phase_widget.update(
                experiment_phase_fraction_figure(
                    rows,
                    selected_labels=selected_labels,
                    x_key=x_key,
                )
            )
        if self._powgen_heatmap_widget is not None:
            self._powgen_heatmap_widget.update(experiment_phase_heatmap_figure(rows, x_key=x_key))
        if self._powgen_quality_widget is not None:
            self._powgen_quality_widget.update(experiment_fit_quality_figure(rows, x_key=x_key))

    def update_powgen_dashboard_axis(self, value: Any = None, **_: Any) -> None:
        """Redraw experiment trends against scan number or one NeXus condition."""

        selected = str(value or self.server.state.powgen_x_axis or "run_number")
        valid = {str(option["value"]) for option in (self.server.state.powgen_x_axis_options or [])}
        self.server.state.powgen_x_axis = selected if selected in valid else "run_number"
        self._update_powgen_dashboard_figures(list(self.server.state.powgen_scientific_rows or []))
        self.server.state.flush()

    def update_powgen_phase_selection(self, values: Any = None, **_: Any) -> None:
        """Keep the line chart readable while retaining every phase in the selector."""

        if isinstance(values, (list, tuple)):
            requested = [str(value) for value in values]
        else:
            requested = [str(value) for value in (self.server.state.powgen_selected_phase_labels or [])]
        ordered_valid = [str(option["value"]) for option in (self.server.state.powgen_phase_options or [])]
        valid = set(ordered_valid)
        selected = [value for value in requested if value in valid]
        self.server.state.powgen_selected_phase_labels = selected or ordered_valid[:5]
        self._update_powgen_dashboard_figures(list(self.server.state.powgen_scientific_rows or []))
        self.server.state.flush()

    def update_powgen_sample_selection(self, values: Any = None, **_: Any) -> None:
        """Filter the experiment dashboard without changing the monitored scans."""

        if isinstance(values, (list, tuple)):
            requested = [str(value) for value in values]
        else:
            requested = [str(value) for value in (self.server.state.powgen_selected_samples or [])]
        valid = {str(option["value"]) for option in (self.server.state.powgen_sample_options or [])}
        self.server.state.powgen_selected_samples = [value for value in requested if value in valid]
        self._sync_powgen_rows()
        self.server.state.flush()

    def select_powgen_scan(self, run_id: Any = None, **_: Any) -> None:
        """Keep the scan inspector synchronized with its experiment row."""

        selected_id = str(run_id or self.server.state.powgen_selected_run_id or "")
        selected = next(
            (row for row in self.server.state.powgen_scientific_rows or [] if row.get("run_id") == selected_id),
            None,
        )
        self.server.state.powgen_selected_run_id = selected_id if selected is not None else ""
        self.server.state.powgen_selected_scan = selected or {}
        self.server.state.powgen_selected_quality_label = str((selected or {}).get("quality_label") or "Collecting baseline")
        self.server.state.powgen_selected_quality_color = str((selected or {}).get("quality_color") or "#eef2ef")
        self.server.state.powgen_selected_phases = list(selected.get("phases") or []) if selected is not None else []
        self.server.state.flush()

    def open_powgen_selected_run(self, **_: Any) -> None:
        """Open the selected experiment scan in the standard result workspace."""

        run_id = str(self.server.state.powgen_selected_run_id or "")
        controller = self._powgen_controller
        record = controller.records.get(run_id) if controller is not None else None
        if record is None:
            self.server.state.error_message = "The selected scan is not available in this monitor session."
            self.server.state.flush()
            return
        completed = controller.state.completed.get(run_id)
        if completed is not None and (
            record.status != RunStatus.OK or not record.output_dataset_ids
        ):
            try:
                record = self.service.refresh(record)
                controller.records[run_id] = record
            except Exception as exc:
                self.server.state.error_message = (
                    f"Could not recover the published result for {run_id}: {exc}"
                )
                self.server.state.flush()
                return
        if record.status != RunStatus.OK:
            self.server.state.error_message = (
                f"The published result for {run_id} is not ready in Galaxy "
                f"(current state: {record.status.value})."
            )
            self.server.state.flush()
            return
        self.records[record.uid] = record
        self._select_record(record)
        self._open_record_results(record)

    def show_powgen_dashboard(self, **_: Any) -> None:
        self.server.state.workspace_view = "experiment"
        self.server.state.setup_collapsed = True
        self._sync_powgen_rows()
        self.server.state.flush()

    @staticmethod
    def _powgen_dataset_label(items: Any, dataset_id: str, fallback: str) -> str:
        selected = next(
            (
                item
                for item in (items or [])
                if str(item.get("id") or "") == str(dataset_id or "")
            ),
            None,
        )
        return str((selected or {}).get("display_name") or (selected or {}).get("name") or fallback)

    def check_powgen_experiment(self, **_: Any) -> bool:
        """Validate a live-monitor setup without writing to the IPTS or submitting jobs."""

        state = self.server.state
        state.powgen_preflight_status = "checking"
        state.powgen_preflight_ready = False
        state.powgen_preflight_checked = "Checking now"
        state.powgen_preflight_message = "Checking the read-only POWGEN source and instrument profile..."
        state.powgen_preflight_details = []
        state.powgen_profile_error = ""
        state.flush()
        try:
            ipts = self._normalize_powgen_ipts(state.powgen_ipts)
            configuration_id = str(state.powgen_configuration_dataset_id or "").strip()
            if not configuration_id:
                raise ValueError("Choose a reusable RADAR-PD configuration")
            configuration = self.service.load_configuration_dataset(configuration_id)
            self._set_powgen_configuration_preview(configuration_id, configuration)

            backfill_mode = str(state.powgen_backfill_mode or "latest_5")
            if backfill_mode not in _POWGEN_BACKFILL_LIMITS:
                raise ValueError("Choose how many existing scans should be processed")

            instrument_source = str(state.powgen_instrument_source or "automatic")
            instrument_profile_name: str | None = None
            if instrument_source == "computer":
                instrument_path = Path(str(state.powgen_instrument_path or ""))
                if not instrument_path.is_file():
                    raise ValueError("Choose a GSAS-II instrument profile from this computer")
                if instrument_path.suffix.lower() not in {".instprm", ".prm", ".inst", ".ins"}:
                    raise ValueError(
                        "The POWGEN instrument profile must use .instprm, .prm, .inst, or .ins"
                    )
                instrument_profile_name = instrument_path.name
            elif instrument_source == "galaxy":
                instrument_id = str(state.powgen_instrument_dataset_id or "").strip()
                if not instrument_id:
                    raise ValueError("Choose a GSAS-II instrument profile from Galaxy History")
                instrument_profile_name = self._powgen_dataset_label(
                    state.history_instrument_datasets,
                    instrument_id,
                    "Selected Galaxy instrument profile",
                ).split(" · ", 1)[0]
            elif instrument_source != "automatic":
                raise ValueError(f"Unsupported POWGEN instrument-profile source: {instrument_source}")

            main_source = str(state.powgen_main_cif_source or "none")
            if main_source == "computer":
                main_path = Path(str(state.powgen_main_cif_path or ""))
                if not main_path.is_file() or main_path.suffix.lower() != ".cif":
                    raise ValueError("Choose a valid .cif main-phase file from this computer")
                main_label = main_path.name
            elif main_source == "galaxy":
                main_id = str(state.powgen_main_cif_dataset_id or "").strip()
                if not main_id:
                    raise ValueError("Choose a main-phase CIF from Galaxy History")
                main_label = self._powgen_dataset_label(
                    state.history_cif_datasets,
                    main_id,
                    "Selected Galaxy CIF",
                )
            else:
                main_label = "Not supplied"

            result = preflight_powgen_experiment(
                ipts,
                requested_wavelength=str(state.powgen_wavelength or ""),
                instrument_profile_name=instrument_profile_name,
            )
            state.powgen_ipts = ipts
            state.powgen_source_directory = str(result.get("source_directory") or "")
            state.powgen_profile_error = str(result.get("profile_error") or "")
            state.powgen_instrument_label = str(
                result.get("profile_filename")
                or instrument_profile_name
                or "Automatic POWGEN cycle profile"
            )
            detected = str(result.get("detected_wavelength") or "")
            if detected:
                state.powgen_wavelength = detected

            configuration_label = self._powgen_dataset_label(
                state.history_configuration_datasets,
                configuration_id,
                "Selected Galaxy configuration",
            )
            database_id = str(state.powgen_database_dataset_id or "").strip()
            database_label = (
                self._powgen_dataset_label(
                    state.history_archive_datasets,
                    database_id,
                    "Selected custom Galaxy library",
                )
                if database_id
                else "Built-in MP/COD catalog"
            )
            state.powgen_configuration_label = configuration_label
            state.powgen_database_label = database_label
            backfill_label = next(
                (
                    str(option["title"])
                    for option in state.powgen_backfill_options
                    if option["value"] == backfill_mode
                ),
                backfill_mode,
            )
            scan_range = (
                f"{result.get('first_run')} to {result.get('latest_run')}"
                if result.get("scan_count")
                else "No completed scans"
            )
            available_scan_count = int(result.get("scan_count") or 0)
            backfill_limit = _POWGEN_BACKFILL_LIMITS[backfill_mode]
            selected_scan_count = (
                available_scan_count
                if backfill_limit is None
                else min(available_scan_count, backfill_limit)
            )
            turnaround = (
                "No existing scans; each new scan will be processed when it appears."
                if selected_scan_count == 0
                else (
                    f"{selected_scan_count} scan(s); first dashboard result typically 1-3 min. "
                    "Bulk backfills finish in batches of five."
                )
            )
            state.powgen_preflight_details = [
                {"label": "Read-only source", "value": state.powgen_source_directory},
                {
                    "label": "Available reductions",
                    "value": f"{result.get('scan_count', 0)} ({scan_range})",
                },
                {
                    "label": "Latest scan",
                    "value": str(result.get("latest_file") or "Not available"),
                },
                {
                    "label": "Latest acquisition",
                    "value": _format_eastern_time(result.get("latest_start_time")),
                },
                {
                    "label": "Latest sample",
                    "value": str(result.get("sample_display") or "Not reported"),
                },
                {
                    "label": "Latest temperature",
                    "value": str(result.get("temperature_display") or "Not reported"),
                },
                {
                    "label": "Wavelength",
                    "value": (
                        f"{result.get('selected_wavelength')} A - {result.get('wavelength_source')}"
                    ),
                },
                {
                    "label": "Instrument profile",
                    "value": (
                        f"{result.get('profile_filename')} - {result.get('profile_source')}"
                        if result.get("profile_filename")
                        else str(result.get("profile_error") or "Unavailable")
                    ),
                },
                {"label": "Initial processing", "value": backfill_label},
                {
                    "label": "Expected turnaround",
                    "value": turnaround,
                },
                {
                    "label": "RADAR-PD configuration",
                    "value": f"{configuration_label} ({configuration.mode.value.title()})",
                },
                {"label": "Candidate library", "value": database_label},
                {"label": "Known/main phase", "value": main_label},
                {"label": "Application", "value": str(state.application_version)},
            ]
            state.powgen_preflight_ready = bool(result.get("ready"))
            state.powgen_preflight_status = "ready" if state.powgen_preflight_ready else "warning"
            state.powgen_preflight_message = str(result.get("message") or "Experiment check completed.")
            state.powgen_message = (
                "Inputs verified. Start monitoring when ready."
                if state.powgen_preflight_ready
                else state.powgen_preflight_message
            )
            state.error_message = "" if state.powgen_preflight_ready else state.powgen_preflight_message
        except Exception as exc:
            state.powgen_preflight_status = "error"
            state.powgen_preflight_ready = False
            state.powgen_preflight_message = _powgen_submission_error_summary(exc)
            state.error_message = state.powgen_preflight_message
        state.powgen_preflight_checked = _format_eastern_time(datetime.now(timezone.utc))
        state.flush()
        return bool(state.powgen_preflight_ready)

    def start_powgen_monitoring(self, **_: Any) -> None:
        """Start one session-scoped read-only POWGEN monitor."""

        state = self.server.state
        if state.powgen_monitoring:
            return
        try:
            if not self.check_powgen_experiment():
                return
            ipts = self._normalize_powgen_ipts(state.powgen_ipts)
            configuration_id = str(state.powgen_configuration_dataset_id or "").strip()
            wavelength = str(state.powgen_wavelength or "").strip()
            if not configuration_id:
                raise ValueError("Choose a reusable Galaxy RADAR-PD configuration")
            if wavelength not in {"0.8", "1.5", "2.665"}:
                raise ValueError("Choose one of the supported POWGEN wavelengths")
            if not self.service.history_id:
                raise RuntimeError("The active Galaxy History is not available to this NOVA session")
            main_cif_id = self._resolve_powgen_main_cif_id()
            database_id = self._resolve_powgen_database_id()
            instrument_profile_id = self._resolve_powgen_instrument_profile_id()
            backfill_mode = str(state.powgen_backfill_mode or "latest_5")
            initial_backfill_limit = _POWGEN_BACKFILL_LIMITS[backfill_mode]

            settings = PowgenExperimentSettings(
                ipts=ipts,
                history_id=self.service.history_id,
                configuration_dataset_id=configuration_id,
                wavelength_angstrom=wavelength,
                instrument_profile_dataset_id=instrument_profile_id,
                instrument_profile_name=str(state.powgen_instrument_label or "") or None,
                main_cif_dataset_id=main_cif_id,
                database_dataset_id=database_id,
                initial_backfill_limit=initial_backfill_limit,
            )
            signature = (
                settings.ipts,
                settings.history_id,
                settings.configuration_dataset_id,
                settings.wavelength_angstrom,
                settings.instrument_profile_dataset_id or "",
                settings.main_cif_dataset_id or "",
                settings.database_dataset_id or "",
                str(settings.initial_backfill_limit),
            )
            if self._powgen_controller is None or self._powgen_settings_signature != signature:
                self._powgen_controller = PowgenWatchController(self.service, settings)
                restored = self._powgen_controller.restore_latest_state()
                self._powgen_settings_signature = signature

            else:
                restored = False

            state.powgen_ipts = ipts
            state.powgen_source_directory = self._powgen_controller.source_directory
            state.workflow_mode = "powgen"
            state.powgen_monitoring = True
            state.workspace_view = "experiment"
            state.powgen_next_check = "Now"
            recovery_note = " Previous Galaxy watch state was restored." if restored else ""
            state.powgen_message = (
                f"Processing the selected initial scan range in {state.powgen_source_directory}, then monitoring for new scans. "
                f"Instrument profile: {state.powgen_instrument_label}. "
                f"Candidate library: {'custom Galaxy archive' if database_id else 'built-in catalog'}. "
                f"The IPTS remains read-only.{recovery_note}"
            )
            state.error_message = ""
            self._sync_powgen_rows()
            self._sync_workspace_options(state.analysis_mode)
            state.flush()
            self._powgen_wake_event.clear()
            task = asyncio.create_task(
                self._powgen_monitor_loop(),
                name=f"radar-powgen-watch-{ipts.lower()}",
            )
            self._powgen_monitor_task = task
        except Exception as exc:
            state.powgen_monitoring = False
            state.powgen_message = f"POWGEN monitor could not start: {exc}"
            state.error_message = str(exc)
            state.flush()

    def _resolve_powgen_instrument_profile_id(self) -> str | None:
        """Return a durable fallback profile, or ``None`` for cycle resolution."""

        state = self.server.state
        source = str(getattr(state, "powgen_instrument_source", "automatic") or "automatic")
        if source == "automatic":
            return None
        if source == "galaxy":
            dataset_id = str(getattr(state, "powgen_instrument_dataset_id", "") or "").strip()
            if not dataset_id:
                raise ValueError("Choose a GSAS-II instrument profile from Galaxy History")
            return dataset_id
        if source != "computer":
            raise ValueError(f"Unsupported POWGEN instrument-profile source: {source}")

        path = Path(str(getattr(state, "powgen_instrument_path", "") or ""))
        if not path.is_file():
            raise ValueError("Choose a GSAS-II instrument profile from this computer")
        if path.suffix.lower() not in {".instprm", ".prm", ".inst", ".ins"}:
            raise ValueError(
                "The POWGEN instrument profile must use .instprm, .prm, .inst, or .ins"
            )

        dataset_id = self.service.upload_document(path, label="POWGEN instrument profile fallback")
        state.powgen_instrument_dataset_id = dataset_id
        state.powgen_instrument_source = "galaxy"
        state.powgen_instrument_label = path.name
        self.refresh_history()
        state.notice = f"Saved {path.name} to Galaxy History for POWGEN monitoring."
        return dataset_id

    def _resolve_powgen_database_id(self) -> str | None:
        """Return the immutable Galaxy library archive for monitored scans.

        The POWGEN panel owns an explicit selection. For compatibility with
        sessions created before that control existed, inherit the custom
        Galaxy archive currently selected in the main Candidate Library panel.
        Local archives are not accepted here because a monitor must retain a
        durable Galaxy dataset after the browser upload object disappears.
        """

        state = self.server.state
        dataset_id = str(getattr(state, "powgen_database_dataset_id", "") or "").strip()
        if not dataset_id:
            uses_selected_archive = (
                str(getattr(state, "database_source", "") or "") == "archive"
                and str(getattr(state, "library_archive_source", "") or "") == "galaxy"
            )
            if uses_selected_archive:
                dataset_id = str(getattr(state, "history_database_id", "") or "").strip()
        if dataset_id:
            state.powgen_database_dataset_id = dataset_id
            return dataset_id
        return None

    def _resolve_powgen_main_cif_id(self) -> str | None:
        """Return the durable Galaxy dataset used as the main-phase anchor."""

        state = self.server.state
        source = str(getattr(state, "powgen_main_cif_source", "none") or "none")
        if source == "none":
            return None
        if source == "galaxy":
            dataset_id = str(state.powgen_main_cif_dataset_id or "").strip()
            if not dataset_id:
                raise ValueError("Choose a main-phase CIF from Galaxy History")
            return dataset_id
        if source != "computer":
            raise ValueError(f"Unsupported POWGEN main-phase source: {source}")

        path = Path(str(getattr(state, "powgen_main_cif_path", "") or ""))
        if not path.is_file():
            raise ValueError("Choose a main-phase CIF from this computer")
        if path.suffix.lower() != ".cif":
            raise ValueError("The supplied main-phase file must use the .cif extension")

        dataset_id = self.service.upload_document(path, label="POWGEN main phase CIF")
        state.powgen_main_cif_dataset_id = dataset_id
        state.powgen_main_cif_source = "galaxy"
        self.refresh_history()
        state.notice = f"Saved {path.name} to Galaxy History for POWGEN monitoring."
        return dataset_id

    def stop_powgen_monitoring(self, **_: Any) -> None:
        """Stop discovery without cancelling Galaxy jobs already submitted."""

        state = self.server.state
        state.powgen_monitoring = False
        task = self._powgen_monitor_task
        self._powgen_monitor_task = None
        if task is not None and not task.done():
            task.cancel()
        self._powgen_wake_event.set()
        state.powgen_next_check = "Stopped"
        state.powgen_message = (
            "POWGEN monitoring stopped. Already-submitted RADAR-PD jobs continue in Galaxy and remain recoverable. "
            "Use NDIP Ingress for unattended triggering."
        )
        state.flush()

    def refresh_powgen_monitor_now(self, **_: Any) -> None:
        """Wake the monitor without restarting it or duplicating its checkpoint."""

        state = self.server.state
        if not state.powgen_monitoring:
            return
        state.powgen_next_check = "Now"
        state.powgen_message = "Refresh requested; checking the POWGEN source and Galaxy jobs now."
        self._powgen_wake_event.set()
        state.flush()

    async def _powgen_monitor_loop(self) -> None:
        """Discover, submit, and refresh POWGEN scans while NOVA stays open."""

        state = self.server.state
        controller = self._powgen_controller
        if controller is None:
            state.powgen_monitoring = False
            return
        try:
            while state.powgen_monitoring and controller is self._powgen_controller:
                state.powgen_last_checked = _format_eastern_time(datetime.now(timezone.utc))
                warnings: list[str] = []
                try:
                    await asyncio.to_thread(controller.discover)
                except Exception as exc:
                    warnings.append(f"Source check failed: {_powgen_submission_error_summary(exc)}")

                try:
                    recovered = await asyncio.to_thread(controller.reconcile_recent_runs)
                    if recovered:
                        warnings.append(f"recovered {recovered} acknowledged Galaxy job(s) from recent History")
                except Exception as exc:
                    warnings.append(
                        "Galaxy job reconciliation will be retried: "
                        f"{_powgen_submission_error_summary(exc)}"
                    )

                self._sync_powgen_rows()
                state.flush()

                # Refresh first so jobs that finished since the last poll free
                # their worker slots before this cycle chooses more scans.
                if controller.state.submitted:
                    try:
                        refreshed = await asyncio.to_thread(controller.refresh)
                        for record in refreshed.values():
                            self.records[record.uid] = record
                        self._sync_runs()
                        if controller.refresh_errors:
                            warnings.append(
                                f"{len(controller.refresh_errors)} Galaxy status update(s) will be retried"
                            )
                    except Exception as exc:
                        warnings.append(
                            f"Galaxy status refresh failed: {_powgen_submission_error_summary(exc)}"
                        )

                due_runs = controller.due_submissions()
                if due_runs and state.powgen_monitoring:
                    submission_lanes = asyncio.Semaphore(_POWGEN_SUBMISSION_CONCURRENCY)

                    async def launch(
                        position: int,
                        run: Any,
                    ) -> tuple[Any, RunRecord | None, Exception | None]:
                        try:
                            # A short offset prevents simultaneous multipart
                            # uploads from arriving at the NDIP gateway in the
                            # same instant. The semaphore remains the actual
                            # concurrency boundary.
                            await asyncio.sleep(min(position, 4) * 0.75)
                            async with submission_lanes:
                                record = await asyncio.to_thread(controller.launch_submission, run)
                            return run, record, None
                        except Exception as exc:
                            return run, None, exc

                    pending = [
                        asyncio.create_task(launch(position, run))
                        for position, run in enumerate(due_runs)
                    ]
                    for future in asyncio.as_completed(pending):
                        # A cancelled asyncio task cannot stop work already
                        # running in ``to_thread``. Always acknowledge this
                        # batch so every Galaxy job remains recoverable; the
                        # monitoring flag prevents the next batch from starting.
                        run, launched, launch_error = await future
                        try:
                            if launch_error is not None:
                                raise launch_error
                            if launched is None:
                                raise RuntimeError(f"Galaxy did not return a record for {run.run_id}")
                            record = await asyncio.to_thread(
                                controller.acknowledge_submission,
                                run,
                                launched,
                                persist=False,
                            )
                            self.records[record.uid] = record
                            self._sync_runs()
                        except Exception as exc:
                            warnings.append(f"{run.run_id} submission failed and will be retried: {exc}")
                            if run.run_id in controller.state.discovered:
                                try:
                                    await asyncio.to_thread(
                                        controller.defer_submission,
                                        run,
                                        str(exc),
                                        persist=False,
                                    )
                                except Exception as persist_exc:
                                    warnings.append(
                                        f"{run.run_id} retry checkpoint also failed: {persist_exc}"
                                    )
                            else:
                                warnings.append(f"{run.run_id} submission status uncertain: {exc}")
                        self._sync_powgen_rows()
                        state.flush()

                    try:
                        await asyncio.to_thread(controller.persist_state)
                    except Exception as exc:
                        warnings.append(
                            "The batch ran, but its recovery checkpoint will be retried: "
                            f"{_powgen_submission_error_summary(exc)}"
                        )

                self._sync_powgen_rows()
                counts = {
                    "discovered": len(controller.state.discovered),
                    "submitted": len(controller.state.submitted),
                    "completed": len(controller.state.completed),
                    "failed": len(controller.state.failed),
                }
                state.powgen_message = (
                    f"Monitoring {controller.source_directory}: "
                    f"{counts['discovered']} awaiting submission, {counts['submitted']} active, "
                    f"{counts['completed']} completed, {counts['failed']} failed."
                )
                if warnings:
                    state.powgen_message += " " + " ".join(warnings) + ". Monitoring remains active."
                delay = max(5, int(state.powgen_poll_seconds or 15))
                state.powgen_next_check = (
                    _format_eastern_time(
                        datetime.now(timezone.utc) + timedelta(seconds=delay),
                        "%H:%M:%S %Z",
                    )
                )
                state.flush()
                try:
                    await asyncio.wait_for(self._powgen_wake_event.wait(), timeout=delay)
                    self._powgen_wake_event.clear()
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass
        finally:
            if controller is self._powgen_controller:
                state.powgen_monitoring = False
                if self._powgen_monitor_task is asyncio.current_task():
                    self._powgen_monitor_task = None
                state.flush()

    def submit_run(self, submission_payload: dict[str, Any] | None = None, **_: Any) -> None:
        state = self.server.state
        state.error_message = ""
        state.notice = ""
        try:
            config = self._configuration(submission_payload)
            inputs = self._inputs(submission_payload)
            token = str(
                (submission_payload or {}).get("submission_token")
                or getattr(state, "submission_token", "")
                or uuid.uuid4().hex
            )
            snapshot = self.service.create_submission_snapshot(
                config,
                inputs,
                client_revision=int((submission_payload or {}).get("form_revision") or state.form_revision or 0),
                idempotency_token=token,
            )
            record = self.service.pending_record(snapshot)
            self.records[record.uid] = record
            state.run_name = config.run_name
            state.selected_run_uid = record.uid
            state.notice = f"Preparing {config.run_name} as {config.mode.value.title()} mode."
            state.active_page = "runs"
            state.workspace_view = "monitor"
            state.setup_collapsed = True
            state.busy = True
            self._sync_runs()
            self._select_record(record)
            state.flush()
            if token in self._pending_submission_tokens:
                state.notice = "This submission is already being prepared; the existing pending run remains active."
                return
            self._pending_submission_tokens.add(token)
            task = asyncio.create_task(self._submit_pending(snapshot), name=f"radar-submit-{token[:8]}")
            self._submission_tasks.add(task)
            task.add_done_callback(self._submission_tasks.discard)
        except Exception as exc:
            state.error_message = str(exc)
            state.busy = False
            state.flush()

    def _submission_update(self, record: RunRecord) -> None:
        """Publish pending progress on Trame's event loop."""

        self.records[record.uid] = record
        self._sync_runs()
        state = self.server.state
        if state.selected_run_uid == record.uid:
            self._select_record(record)
        state.flush()

    async def _submit_pending(self, snapshot: SubmissionSnapshot) -> None:
        loop = asyncio.get_running_loop()

        def publish(record: RunRecord) -> None:
            # Upload callbacks run on isolated worker connections. Copy their
            # state and marshal UI work back to Trame's event loop.
            copied = record.model_copy(deep=True)
            loop.call_soon_threadsafe(self._submission_update, copied)

        try:
            record = await asyncio.to_thread(self.service.submit_snapshot, snapshot, callback=publish)
            self._submission_update(record)
            if record.galaxy_job_id:
                self.server.state.notice = (
                    f"Galaxy accepted {record.name} as {record.mode.value.title()} mode "
                    f"(job {record.galaxy_job_id[:8]})."
                )
                self.server.state.submission_token = uuid.uuid4().hex
                self._start_monitor(record)
            elif record.status == RunStatus.CANCELLED:
                self.server.state.notice = "Submission cancelled before the Analyze job was created."
        except Exception as exc:
            record = self.service.pending_record(snapshot)
            record.message = str(exc)
            self._submission_update(record)
            self.server.state.error_message = f"Could not submit {snapshot.config.run_name}: {exc}"
        finally:
            self._pending_submission_tokens.discard(snapshot.idempotency_token)
            self.server.state.busy = bool(self._pending_submission_tokens)
            self.server.state.flush()

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
                    record = await asyncio.to_thread(self._publish_results_if_requested, record)
                except Exception as exc:
                    record.analysis_status = RunStatus.OK
                    record.status = RunStatus.OK
                    record.result_status = ResultStatus.ERROR
                    record.stage = "Analysis complete; results unavailable"
                    record.progress = 100
                    record.message = f"Analysis finished, but result download failed: {exc}"
                self._monitor_update(record)
                if record.publication_job_id and record.publication_status not in {
                    RunStatus.OK,
                    RunStatus.ERROR,
                    RunStatus.CANCELLED,
                }:
                    self._schedule_utility(
                        self._monitor_results_publication(record),
                        f"radar-publish-{record.uid}",
                    )
        finally:
            self._monitored_uids.discard(record.uid)

    def _publish_results_if_requested(self, record: RunRecord) -> RunRecord:
        """Submit or refresh NDIP's authenticated export of a result archive."""

        inputs = record.inputs
        if (
            inputs is None
            or not inputs.publish_results_to_ipts
            or record.published_output_dir
        ):
            return record

        def set_message(message: str) -> None:
            previous = record.publish_message
            if previous and previous in record.message:
                record.message = record.message.replace(previous, "").strip()
            record.publish_message = message
            record.message = "\n".join(value for value in (record.message, message) if value)

        try:
            if record.publication_job_id:
                action = UtilityActionRecord(
                    uid=f"publication-{record.uid}",
                    tool_id="neutrons_export",
                    name="Publish RADAR-PD results to IPTS",
                    associated_run_uid=record.uid,
                    galaxy_job_id=record.publication_job_id,
                    status=record.publication_status or RunStatus.QUEUED,
                )
                action = self.service.refresh_utility(action)
                record.publication_status = action.status
                if action.status == RunStatus.OK:
                    record.published_output_dir = record.publication_target
                    set_message(f"Published the complete result archive to {record.publication_target}")
                elif action.status in {RunStatus.ERROR, RunStatus.CANCELLED}:
                    detail = action.message or f"NDIP export ended with status {action.status.value}"
                    job_label = f" (Galaxy export job {record.publication_job_id})" if record.publication_job_id else ""
                    set_message(
                        "Analysis completed, but the IPTS copy"
                        f"{job_label} failed. The complete archive remains available in Galaxy's "
                        f"Run File Browser for manual download. {detail}"
                    )
                else:
                    set_message(f"Publishing the complete result archive to {record.publication_target}")
                return record

            archive_dataset_id = record.output_dataset_ids.get("results_archive")
            if not archive_dataset_id:
                raise RuntimeError("Galaxy did not expose the completed results archive for export")
            action, destination = self.service.submit_results_export(
                record,
                archive_dataset_id=archive_dataset_id,
            )
            record.publication_job_id = action.galaxy_job_id
            record.publication_status = action.status
            record.publication_target = destination
            set_message(f"Publishing the complete result archive to {destination}")
        except Exception as exc:
            # Galaxy remains the authoritative result store. Publication is a
            # separate provenance-tracked job and never invalidates analysis.
            record.publication_status = RunStatus.ERROR
            set_message(f"Analysis completed, but authenticated IPTS export failed: {exc}")
        return record

    async def _monitor_results_publication(self, record: RunRecord) -> None:
        terminal = {RunStatus.OK, RunStatus.ERROR, RunStatus.CANCELLED}
        while record.publication_status not in terminal:
            await asyncio.sleep(2.0)
            record = await asyncio.to_thread(self._publish_results_if_requested, record)
            self.records[record.uid] = record
            self._monitor_update(record)

    def _sync_runs(self) -> None:
        rows: list[dict[str, Any]] = []
        for record in sorted(self.records.values(), key=lambda item: item.created_utc, reverse=True):
            row = record.as_row()
            row["created_display"] = _format_eastern_time(record.created_utc, "%Y-%m-%d %H:%M %Z")
            row["job_short"] = (record.galaxy_job_id or "pending")[:8]
            rows.append(row)
        self.server.state.run_rows = rows

    def _recover_runs(self, **_: Any) -> None:
        state = self.server.state
        try:
            self.service.validate_connection()
            state.connection_ok = True
            state.connection_status = "Connected to NDIP"
            self.refresh_history()
            for recovered in self.service.recent_runs():
                current = self.records.get(recovered.uid) or next(
                    (
                        candidate
                        for candidate in self.records.values()
                        if candidate.galaxy_job_id and candidate.galaxy_job_id == recovered.galaxy_job_id
                    ),
                    None,
                )
                if current is None:
                    record = recovered
                else:
                    record = current.model_copy(
                        update={
                            "galaxy_job_id": recovered.galaxy_job_id,
                            "name": recovered.name,
                            "mode": recovered.mode,
                            "status": recovered.status,
                            "analysis_status": recovered.analysis_status,
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
            state.history_offset = 0
            datasets = self.service.search_history_datasets(
                query=str(state.history_search or ""),
                limit=100,
                offset=0,
                include_generated=True,
            )
            merged = list(datasets)
            known_ids = {str(item.get("id")) for item in merged}
            # Large uploads and repeated analyses can fill the newest page.
            # Query each durable input class independently so POWGEN setup does
            # not lose an older configuration, library, CIF, or profile.
            for targeted_query in (
                "radar_pd_config",
                "portable custom library",
                ".cif",
                ".instprm",
            ):
                targeted = self.service.search_history_datasets(
                    query=targeted_query,
                    limit=100,
                    offset=0,
                    include_generated=True,
                )
                for item in targeted:
                    identifier = str(item.get("id"))
                    if identifier in known_ids:
                        continue
                    merged.append(item)
                    known_ids.add(identifier)
            self._apply_history_page(merged, append=False)
            state.history_has_more = len(datasets) == 100
            state.notice = f"Loaded the newest {len(state.history_datasets)} relevant Galaxy inputs."
        except Exception as exc:
            state.error_message = f"Could not load Galaxy datasets: {exc}"
        state.flush()

    def _apply_history_page(self, datasets: list[dict[str, Any]], *, append: bool) -> None:
        state = self.server.state
        selected_ids = {
            str(identifier)
            for identifier in (
                state.history_data_id,
                state.history_instrument_id,
                state.history_main_cif_id,
                state.history_database_id,
                state.history_configuration_id,
                getattr(state, "powgen_configuration_dataset_id", ""),
                getattr(state, "powgen_database_dataset_id", ""),
                getattr(state, "powgen_main_cif_dataset_id", ""),
                *(state.library_builder_cif_ids or []),
            )
            if identifier
        }
        visible = [
            item
            for item in datasets
            if item.get("role") in {"diffraction", "instrument", "cif", "candidate_library", "event", "configuration"}
            and (
                state.history_show_all
                or not item.get("generated")
                or item.get("role") in {"configuration", "candidate_library"}
                or str(item.get("id") or "") in selected_ids
            )
        ]
        previous = list(state.history_datasets)
        if append:
            combined = previous
        else:
            # Retain selected records while the user searches for a sibling
            # input. Otherwise Vuetify can only render the opaque Galaxy ID.
            combined = [item for item in previous if str(item.get("id")) in selected_ids]
        known = {str(item.get("id")) for item in combined}
        combined.extend(item for item in visible if str(item.get("id")) not in known)
        state.history_datasets = combined
        state.history_data_datasets = [item for item in combined if item.get("role") == "diffraction"]
        state.history_instrument_datasets = [item for item in combined if item.get("role") == "instrument"]
        state.history_cif_datasets = [item for item in combined if item.get("role") == "cif"]
        state.history_archive_datasets = [item for item in combined if item.get("role") == "candidate_library"]
        state.history_configuration_datasets = [item for item in combined if item.get("role") == "configuration"]
        cache = getattr(self, "_dataset_label_cache", {})
        for item in combined:
            identifier = str(item.get("id") or "").strip()
            label = str(item.get("display_name") or item.get("name") or "").strip()
            if identifier and label:
                cache[identifier] = label
        self._dataset_label_cache = cache
        state.history_has_more = len(datasets) == 100

    def _restore_selected_history_labels(self) -> None:
        """Resolve exact reused inputs even when generated copies are hidden."""

        state = self.server.state
        selected_ids = [
            str(identifier)
            for identifier in (
                state.history_data_id,
                state.history_instrument_id,
                state.history_main_cif_id,
                state.history_database_id,
            )
            if identifier
        ]
        known_ids = {str(item.get("id") or "") for item in (state.history_datasets or [])}
        restored: list[dict[str, Any]] = []
        for dataset_id in selected_ids:
            if dataset_id in known_ids:
                continue
            try:
                item = self.service.history_dataset_item(dataset_id)
            except Exception:
                item = None
            if item is not None:
                restored.append(item)
                known_ids.add(dataset_id)
        if restored:
            self._apply_history_page(restored, append=True)

    def search_history(self, query: str | None = None, **_: Any) -> None:
        state = self.server.state
        state.history_search = str(query or "")
        if self._history_search_task is not None and not self._history_search_task.done():
            self._history_search_task.cancel()
        try:
            self._history_search_task = asyncio.create_task(self._debounced_history_search())
        except RuntimeError:
            self.refresh_history()

    async def _debounced_history_search(self) -> None:
        try:
            await asyncio.sleep(0.35)
            state = self.server.state
            query = str(state.history_search or "")
            datasets = await asyncio.to_thread(
                self.service.search_history_datasets,
                query=query,
                limit=100,
                offset=0,
                include_generated=True,
            )
            state.history_offset = 0
            self._apply_history_page(datasets, append=False)
            state.flush()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self.server.state.error_message = f"Could not search Galaxy History: {exc}"
            self.server.state.flush()

    def load_more_history(self, **_: Any) -> None:
        state = self.server.state
        try:
            next_offset = int(state.history_offset or 0) + 100
            datasets = self.service.search_history_datasets(
                query=str(state.history_search or ""),
                limit=100,
                offset=next_offset,
                include_generated=True,
            )
            state.history_offset = next_offset
            self._apply_history_page(datasets, append=True)
        except Exception as exc:
            state.error_message = f"Could not load more Galaxy inputs: {exc}"
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
        if getattr(self, "_powgen_controller", None) is not None:
            options.append(
                {
                    "title": "Experiment Dashboard",
                    "value": "experiment",
                    "icon": "mdi-chart-timeline-variant",
                }
            )
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

    def _dataset_display_label(self, dataset_id: str, fallback: str) -> str:
        identifier = str(dataset_id or "").strip()
        if not identifier:
            return fallback
        state = self.server.state
        selected = next(
            (
                item
                for item in (getattr(state, "history_datasets", []) or [])
                if str(item.get("id") or "") == identifier
            ),
            None,
        )
        if selected:
            return str(selected.get("display_name") or selected.get("name") or fallback)
        cache = getattr(self, "_dataset_label_cache", {})
        if identifier in cache:
            return cache[identifier]
        return f"{fallback} ({identifier[:8]})"

    @staticmethod
    def _input_value(inputs: InputSelection | dict[str, Any] | None, name: str) -> Any:
        if isinstance(inputs, InputSelection):
            return getattr(inputs, name, None)
        if isinstance(inputs, dict):
            return inputs.get(name)
        return None

    def _run_provenance(self, record: RunRecord) -> list[dict[str, str]]:
        inputs = getattr(record, "inputs", None) or getattr(record, "input_selection", None)
        config = getattr(record, "config", None)
        data_id = str(self._input_value(inputs, "data_dataset_id") or "")
        instrument_id = str(self._input_value(inputs, "instrument_dataset_id") or "")
        main_id = str(self._input_value(inputs, "main_cif_dataset_id") or "")
        database_id = str(self._input_value(inputs, "database_dataset_id") or "")
        configuration_id = str(
            (record.input_dataset_ids or {}).get("configuration")
            or (record.prepared_dataset_ids or {}).get("configuration")
            or ""
        )

        data_path = str(self._input_value(inputs, "data_path") or "")
        instrument_path = str(self._input_value(inputs, "instrument_path") or "")
        main_path = str(self._input_value(inputs, "main_cif_path") or "")
        use_builtin_cuka = bool(self._input_value(inputs, "use_builtin_cuka"))
        configuration_label = (
            f"{config.mode.value.title()} / {config.radiation.value.title()} / {config.instrument_mode.upper()}"
            if isinstance(config, AnalysisConfig)
            else record.mode.value.title()
        )
        return [
            {
                "label": "Saved configuration",
                "value": self._dataset_display_label(configuration_id, "Galaxy configuration")
                if configuration_id
                else "Inline run configuration",
            },
            {"label": "Analysis profile", "value": configuration_label},
            {
                "label": "Candidate library",
                "value": self._dataset_display_label(database_id, "Custom Galaxy library")
                if database_id
                else "Built-in MP/COD catalog",
            },
            {
                "label": "Diffraction data",
                "value": self._dataset_display_label(data_id, "Galaxy diffraction dataset")
                if data_id
                else (Path(data_path).name if data_path else "Facility-resolved input"),
            },
            {
                "label": "Instrument profile",
                "value": (
                    "Built-in Cu K-alpha profile"
                    if use_builtin_cuka
                    else self._dataset_display_label(instrument_id, "Galaxy instrument profile")
                    if instrument_id
                    else (Path(instrument_path).name if instrument_path else "Facility-resolved profile")
                ),
            },
            {
                "label": "Known/main phase",
                "value": self._dataset_display_label(main_id, "Galaxy main-phase CIF")
                if main_id
                else (Path(main_path).name if main_path else "Not supplied"),
            },
            {"label": "Application", "value": str(getattr(self.server.state, "application_version", "RADAR-PD"))},
        ]

    @staticmethod
    def _record_configuration_yaml(record: RunRecord) -> str:
        root = record.local_output_dir
        if root is not None and root.is_dir():
            resolved = next(root.rglob("resolved_config.yaml"), None)
            if resolved is not None:
                try:
                    raw_document = resolved.read_text(encoding="utf-8")
                    document = yaml.safe_load(raw_document)
                    if isinstance(document, dict):
                        return raw_document
                except Exception:
                    pass
        saved = getattr(record, "config", None)
        if isinstance(saved, AnalysisConfig):
            contract = saved.portable_contract()
            # ``portable_contract`` timestamps newly serialized files. A
            # reconstructed run record does not know that original timestamp,
            # so do not present a synthetic time as saved provenance.
            contract.pop("created_utc", None)
            return yaml.safe_dump(contract, sort_keys=False, allow_unicode=False)
        configuration = getattr(record, "configuration", {}) or {}
        if isinstance(configuration, dict) and configuration:
            return yaml.safe_dump(configuration, sort_keys=False, allow_unicode=False)
        return ""

    def _select_record(self, record: RunRecord) -> None:
        state = self.server.state
        previous_run_uid = str(getattr(state, "selected_run_uid", "") or "")
        state.selected_run_uid = record.uid
        state.run_selection = [record.uid]
        state.selected_run_name = record.name
        state.selected_run_status = record.status.value.title()
        state.selected_analysis_status = (record.analysis_status or record.status).value.title()
        state.selected_result_status = record.result_status.value.replace("_", " ").title()
        state.selected_publication_status = record.publication_status.value.title() if record.publication_status else "-"
        state.selected_publication_target = record.publication_target or ""
        state.selected_publication_job_id = record.publication_job_id or ""
        state.selected_publish_message = record.publish_message
        state.result_explorer_available = bool(record.output_dataset_ids.get("results_archive"))
        state.selected_galaxy_job_id = record.galaxy_job_id or "Pending"
        state.selected_run_stage = record.stage
        state.selected_run_progress = record.progress
        state.selected_run_message = record.message
        state.selected_run_console = record.console_tail
        state.viewed_run_mode = record.mode.value
        state.viewed_output_profile = record.output_profile
        state.monitor_stages = self._monitor_stage_rows(record)
        if previous_run_uid != record.uid:
            state.gsasii_launch_url = ""
            state.gsasii_session_status = ""
            state.gsasii_status_message = ""
        try:
            created = datetime.fromisoformat(record.created_utc.replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            elapsed = max(0.0, (datetime.now(timezone.utc) - created).total_seconds())
            state.selected_run_elapsed = f"{elapsed:.0f} s elapsed"
        except (TypeError, ValueError):
            state.selected_run_elapsed = "-"
        self._sync_workspace_options(record.mode)
        state.viewed_configuration = self._record_configuration_yaml(record)
        state.run_provenance_rows = self._run_provenance(record)

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
                state.database_source = "archive"
                state.library_archive_source = "galaxy"
                state.powgen_database_dataset_id = str(selection["database_dataset_id"])
            self._restore_selected_history_labels()
            state.run_name = ""
            state.notice = (
                f"Loaded the scientific configuration and durable Galaxy inputs from {record.name}. "
                "Review or replace any input, then submit a new run."
            )
            state.active_page = "setup"
            state.setup_collapsed = False
            state.setup_panels = [2, 10]
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
            self.server.state.setup_panels = [2, 10]
        except Exception as exc:
            self.server.state.error_message = str(exc)
        self.server.state.flush()

    def apply_history_configuration(self, **_: Any) -> None:
        dataset_id = str(self.server.state.history_configuration_id or "")
        if not dataset_id:
            return
        try:
            config = self.service.load_configuration_dataset(dataset_id)
            self._set_history_configuration_preview(dataset_id, config)
            self._apply_configuration(config)
            self.server.state.notice = "Loaded the reusable Galaxy configuration. Input selections and run name remain independent."
            self.server.state.setup_panels = [2, 10]
        except Exception as exc:
            self.server.state.error_message = f"Could not load History configuration: {exc}"
        self.server.state.flush()

    def _apply_configuration(self, config: AnalysisConfig, selection: dict[str, Any] | None = None) -> None:
        state = self.server.state
        for field in AnalysisConfig.model_fields:
            if field == "run_name":
                continue
            value = getattr(config, field)
            state_field = "analysis_mode" if field == "mode" else field
            if field in {"sample_elements", "environment_elements"}:
                value = ", ".join(value)
            elif field == "magnetic_denominators":
                value = ", ".join(str(item) for item in value)
            elif field == "excluded_space_groups":
                value = ", ".join(str(item) for item in value)
            elif field == "exclude_regions":
                value = "\n".join(f"{start}, {end}" for start, end in value)
                state_field = "ignore_regions"
            elif field == "limits":
                state.fit_start = value[0] if value else None
                state.fit_end = value[1] if value else None
                continue
            setattr(state, state_field, value.value if hasattr(value, "value") else value)
        self._radiation_changed(config.radiation.value)
        if selection is None:
            # A reusable configuration contains scientific settings, not data
            # provenance. Keep all compatible input choices independent.
            state.run_name = ""
            return
        selection = selection or {}
        state.main_cif_source = "none"
        state.database_source = "builtin"
        state.library_archive_source = "computer"
        state.use_builtin_cuka = bool(selection.get("use_builtin_cuka", False)) if config.radiation.value == "xray" else False
        raw_source = selection.get("source") or state.input_source
        if selection.get("data_dataset_id"):
            state.input_source = InputSource.GALAXY.value
        else:
            state.input_source = raw_source.value if hasattr(raw_source, "value") else str(raw_source)
        raw_instrument_source = selection.get("instrument_source")
        if state.use_builtin_cuka:
            state.instrument_source = "upload"
        elif selection.get("instrument_dataset_id"):
            state.instrument_source = "galaxy"
        elif raw_instrument_source:
            state.instrument_source = (
                raw_instrument_source.value if hasattr(raw_instrument_source, "value") else str(raw_instrument_source)
            )
        elif selection.get("instrument_relative_path"):
            state.instrument_source = "ipts"
        else:
            state.instrument_source = "upload"

        uses_facility_scope = bool(
            state.input_source == InputSource.IPTS_BROWSER.value
            or state.instrument_source == "ipts"
            or selection.get("main_cif_relative_path")
            or selection.get("publish_results_to_ipts")
        )
        if uses_facility_scope:
            self.facility = FacilityBrowser.for_root(str(selection.get("facility_root") or "/SNS"))
            state.use_facility_workspace = True
            state.facility_site = self.facility.facility
            state.facility_root = str(self.facility.root)
            state.facility_available = self.facility.available
            state.facility_instruments = self.facility.list_instruments()
            state.facility_instrument = str(selection.get("instrument") or "")
            state.facility_ipts = str(selection.get("ipts") or "")
            state.facility_ipts_options = (
                self.facility.list_ipts(state.facility_instrument) if state.facility_instrument else []
            )

        if state.input_source == InputSource.IPTS_BROWSER.value:
            state.facility_data_path = str(selection.get("data_path") or "")
            state.facility_data_relative_path = str(selection.get("data_relative_path") or "")
            if state.facility_data_relative_path:
                state.facility_working_directory = Path(state.facility_data_relative_path).parent.as_posix()

        if state.instrument_source == "ipts":
            state.facility_instrument_path = str(selection.get("instrument_path") or "")
            state.facility_instrument_relative_path = str(selection.get("instrument_relative_path") or "")
            if state.facility_instrument_relative_path:
                state.facility_working_directory = Path(state.facility_instrument_relative_path).parent.as_posix()

        if selection.get("main_cif_relative_path"):
            state.facility_main_cif_path = str(selection.get("main_cif_path") or "")
            state.facility_main_cif_relative_path = str(selection.get("main_cif_relative_path") or "")
            state.facility_working_directory = Path(state.facility_main_cif_relative_path).parent.as_posix()
            state.main_cif_source = "ipts"

        if uses_facility_scope:
            state.publish_results_to_ipts = bool(selection.get("publish_results_to_ipts", False))
            if selection.get("publish_directory"):
                state.facility_working_directory = str(selection["publish_directory"])
            state.facility_output_subfolder = str(selection.get("publish_subfolder") or "radar-pd-results")
            self.refresh_facility_browser()
        else:
            state.use_facility_workspace = False
            state.publish_results_to_ipts = False

        state.data_path = str(selection.get("data_path") or "") if state.input_source == InputSource.UPLOAD.value else ""
        state.history_data_id = str(selection.get("data_dataset_id") or "") if state.input_source == InputSource.GALAXY.value else ""
        state.remote_data_uri = (
            str(selection.get("data_remote_uri") or "")
            if state.input_source == InputSource.GALAXY_REMOTE.value
            else ""
        )
        state.instrument_path = (
            str(selection.get("instrument_path") or "") if state.instrument_source == "upload" else ""
        )
        state.history_instrument_id = (
            str(selection.get("instrument_dataset_id") or "") if state.instrument_source == "galaxy" else ""
        )
        state.remote_instrument_uri = (
            str(selection.get("instrument_remote_uri") or "")
            if state.instrument_source == "galaxy_remote"
            else ""
        )
        state.main_cif_path = ""
        state.history_main_cif_id = ""
        state.remote_main_cif_uri = ""
        state.database_archive_path = ""
        state.history_database_id = ""
        if selection.get("main_cif_dataset_id"):
            state.main_cif_source = "galaxy"
            state.history_main_cif_id = str(selection["main_cif_dataset_id"])
        elif selection.get("main_cif_remote_uri"):
            state.main_cif_source = "galaxy_remote"
            state.remote_main_cif_uri = str(selection["main_cif_remote_uri"])
        elif selection.get("main_cif_path") and not selection.get("main_cif_relative_path"):
            state.main_cif_source = "upload"
            state.main_cif_path = str(selection["main_cif_path"])
        if selection.get("database_dataset_id"):
            state.database_source = "archive"
            state.library_archive_source = "galaxy"
            state.history_database_id = str(selection["database_dataset_id"])
            state.powgen_database_dataset_id = str(selection["database_dataset_id"])
        elif selection.get("database_archive_path"):
            state.database_source = "archive"
            state.library_archive_source = "computer"
            state.database_archive_path = str(selection["database_archive_path"])
        state.run_name = ""

    def refresh_selected_run(self, **_: Any) -> None:
        uid = self.server.state.selected_run_uid
        if not uid or uid not in self.records:
            return
        try:
            record = self.service.refresh(self.records[uid])
            if record.status == RunStatus.OK and not record.output_dir:
                record = self.service.collect_results(record)
            if record.status == RunStatus.OK and record.output_dir:
                record = self._publish_results_if_requested(record)
            self.records[uid] = record
            self._select_record(record)
            if record.status == RunStatus.OK and record.output_dir:
                self._load_results(record)
                self.server.state.workspace_view = "results"
            self._sync_runs()
        except Exception as exc:
            self.server.state.selected_run_message = str(exc)
        self.server.state.flush()

    def reload_selected_results(self, **_: Any) -> None:
        uid = self.server.state.selected_run_uid
        record = self.records.get(uid)
        if record is None or (record.analysis_status or record.status) != RunStatus.OK:
            return
        try:
            self.server.state.selected_run_loading = True
            record = self.service.collect_results(record, force=True)
            record = self._publish_results_if_requested(record)
            self.records[uid] = record
            self._select_record(record)
            if record.result_status == ResultStatus.READY and record.output_dir:
                self._load_results(record)
                self.server.state.workspace_view = "results"
                self.server.state.notice = f"Reloaded the canonical result archive for {record.name}."
            else:
                self.server.state.workspace_view = "monitor"
        except Exception as exc:
            self.server.state.selected_run_message = str(exc)
        finally:
            self.server.state.selected_run_loading = False
            self.server.state.flush()

    def download_diagnostics(self, **_: Any) -> None:
        record = self.records.get(self.server.state.selected_run_uid)
        if record is None:
            return
        payload = {
            "schema": "radar-pd-nova-diagnostics/v1",
            "nova_version": _application_version(),
            "run": {
                "name": record.name,
                "mode_submitted": record.mode.value,
                "galaxy_job_id": record.galaxy_job_id,
                "analysis_status": (record.analysis_status or record.status).value,
                "result_status": record.result_status.value,
                "stage": record.stage,
                "progress": record.progress,
                "created_utc": record.created_utc,
                "updated_utc": record.updated_utc,
            },
            "input_dataset_ids": record.input_dataset_ids,
            "output_dataset_ids": record.output_dataset_ids,
            "cache_manifest": record.cache_manifest.model_dump(mode="json") if record.cache_manifest else None,
            "recovery": record.recovery_diagnostics,
            "message": self._safe_diagnostic_text(record.message),
            "console_tail": self._safe_diagnostic_text(record.console_tail[-12000:]),
        }
        self.download_file(
            f"{record.name}_nova_diagnostics.json",
            "application/json",
            json.dumps(payload, indent=2).encode("utf-8"),
        )

    @staticmethod
    def _safe_diagnostic_text(value: str) -> str:
        text = str(value or "")
        text = re.sub(r"(?i)(x-api-key|api[_ -]?key|authorization)\s*[:=]\s*\S+", r"\1=<redacted>", text)
        text = re.sub(r"(?<!https:)(?<!http:)(?:[A-Za-z]:\\|/)[^\s\"']+", "<redacted-path>", text)
        return text

    def cancel_selected_run(self, **_: Any) -> None:
        uid = self.server.state.selected_run_uid
        if not uid:
            return
        try:
            record = self.records[uid]
            if not record.galaxy_job_id and record.idempotency_token:
                self.service.cancel_pending_submission(record.idempotency_token)
                record.cancel_requested = True
                record.stage = "Cancelling after current uploads"
            else:
                self.service.cancel(record.galaxy_job_id or uid)
                record.status = RunStatus.CANCELLED
                record.analysis_status = RunStatus.CANCELLED
                record.stage = "Stopped by user"
            self._select_record(record)
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
                record = self._publish_results_if_requested(record)
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
        state.viewed_output_profile = record.output_profile
        state.viewed_run_name = record.name
        root = record.local_output_dir
        if root is None:
            return
        view = build_result_view(result_document, root, submitted_mode=record.mode.value).to_state()
        state.viewed_run_mode = view["mode"]
        state.result_metrics = view["metrics"]
        state.summary_cards = view["metrics"]
        state.result_warnings = view["warnings"]
        state.phase_rows = view["phases"]
        state.phase_total = view["phase_total"]
        state.table_options = view["tables"]
        state.plot_options = view["plots"]
        state.primary_plot_path = view["primary_plot_path"]
        state.file_groups = view["file_groups"]
        state.checkpoint_rows = self._launchable_checkpoint_rows(record, view["checkpoints"])
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
        else:
            # Avoid an empty-then-final update while the hidden Plotly panels
            # become visible. That race can leave the empty figure mounted
            # until the user visits another tab.
            if self._plot_widget is not None:
                self._plot_widget.update(figure_for_payload({}))
            if self._primary_plot_widget is not None:
                self._primary_plot_widget.update(figure_for_payload({}))
        if state.artifact_options:
            state.selected_artifact = state.artifact_options[0]["path"]
        if state.checkpoint_rows:
            state.selected_checkpoint = next(
                (item["id"] for item in state.checkpoint_rows if item.get("handoff_available")),
                "",
            )
        if state.rapid_final_rows:
            state.selected_hypothesis = state.rapid_final_rows[0]["rank"]
            state.comparison_hypothesis = (
                state.rapid_final_rows[1]["rank"] if len(state.rapid_final_rows) > 1 else None
            )
        self._sync_workspace_options(view["mode"])
        state.flush()

    def _launchable_checkpoint_rows(
        self,
        record: RunRecord,
        checkpoints: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Expose a GSAS-II action only when a real GPX source is reachable."""

        rows = list(checkpoints or [])
        if not rows:
            return []
        collection_id = str(record.output_dataset_ids.get("gpx_projects") or "").strip()

        launchable: list[dict[str, Any]] = []
        for item in rows:
            local_path = Path(str(item.get("path") or ""))
            local_available = (
                bool(item.get("local_available"))
                and local_path.suffix.casefold() == ".gpx"
                and local_path.is_file()
            )
            if not local_available and not collection_id:
                continue
            launchable.append(
                {
                    **item,
                    "local_available": local_available,
                    "handoff_available": True,
                }
            )
        return launchable

    def _reset_result_state(self) -> None:
        state = self.server.state
        state.summary_cards = []
        state.result_metrics = []
        state.result_warnings = []
        state.phase_rows = []
        state.phase_total = "-"
        state.viewed_output_profile = ""
        state.table_options = []
        state.selected_table = ""
        state.table_rows = []
        state.table_headers = []
        state.table_preview_notice = ""
        state.plot_options = []
        state.selected_plot = ""
        state.gallery_selected_plot = ""
        state.primary_plot_path = ""
        state.plot_groups = []
        state.artifact_options = []
        state.selected_artifact = ""
        state.file_groups = []
        state.file_search = ""
        state.show_technical_files = False
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

    def _table_changed(self, path: str | None = None, **_: Any) -> None:
        path = path or self.server.state.selected_table
        if path:
            self.server.state.selected_table = path
        rows = read_table(path, limit=501) if path else []
        truncated = len(rows) > 500
        self.server.state.table_rows = rows[:500]
        self.server.state.table_preview_notice = (
            "Preview limited to 500 rows. Download the published table from Run File Browser for the complete file."
            if truncated
            else ""
        )
        keys = list(rows[0].keys()) if rows else []
        self.server.state.table_headers = [{"title": key.replace("_", " ").title(), "key": key} for key in keys]
        self.server.state.flush()

    def _primary_plot_changed(self, path: str | None = None, **_: Any) -> None:
        path = path or self.server.state.selected_plot
        if not path or self._primary_plot_widget is None:
            return
        selected = load_plot_with_fallback(self.server.state.plot_options, path)
        if selected is None:
            self._primary_plot_widget.update(figure_for_payload({}))
            self.server.state.result_warnings = [
                *list(self.server.state.result_warnings),
                "NOVA found plot metadata but none of the ranked payloads could be rendered.",
            ]
            return
        selected_path, _, figure = selected
        self.server.state.selected_plot = selected_path
        self._primary_plot_widget.update(figure)
        if selected_path != path:
            self.server.state.result_warnings = [
                *list(self.server.state.result_warnings),
                "The preferred plot payload was incomplete; NOVA displayed the next valid ranked fit.",
            ]

    def _gallery_plot_changed(self, path: str | None = None, **_: Any) -> None:
        path = path or self.server.state.gallery_selected_plot
        if not path or self._plot_widget is None:
            return
        selected = load_plot_with_fallback(self.server.state.plot_options, path)
        if selected is None:
            self._plot_widget.update(figure_for_payload({}))
            return
        selected_path, _, figure = selected
        self.server.state.gallery_selected_plot = selected_path
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
        checkpoint = next(
            (
                item
                for item in self.server.state.checkpoint_rows or []
                if str(item.get("id") or "") == selected or str(item.get("path") or "") == selected
            ),
            {},
        )
        path = str(checkpoint.get("path") or "")
        if not path:
            self.server.state.error_message = (
                "This GPX is available in Galaxy for GSAS-II, but no downloadable copy was included in the local result bundle."
            )
            self.server.state.flush()
            return
        self.download_artifact(path)

    def _absolute_launch_url(self, launch_url: str) -> str:
        if launch_url.startswith("/"):
            return f"{self.service.galaxy_url}{launch_url}"
        return launch_url

    def _activity_row(self, action: UtilityActionRecord) -> dict[str, str]:
        run = self.records.get(action.associated_run_uid or "")
        launch_url = self._absolute_launch_url(action.outputs.get("launch_url", ""))
        if not launch_url and action.status == RunStatus.OK:
            preferred = next(
                (
                    action.outputs[key]
                    for key in (
                        "comparison_report",
                        "handoff_report",
                        "library_archive",
                        "config_output",
                        "resolution_metadata",
                    )
                    if action.outputs.get(key)
                ),
                "",
            )
            if preferred:
                launch_url = f"{self.service.galaxy_url}/datasets/{preferred}/display"
        return {
            "uid": action.uid,
            "name": action.name,
            "status": action.status.value.title(),
            "run_name": run.name if run else "",
            "job_id": (action.galaxy_job_id or "pending")[:8],
            "message": action.message,
            "launch_url": launch_url,
        }

    def _sync_activities(self) -> None:
        self.server.state.activity_rows = [
            self._activity_row(action)
            for action in sorted(self.utility_actions.values(), key=lambda item: item.created_utc, reverse=True)
        ]

    def _register_utility(self, action: UtilityActionRecord) -> None:
        self.utility_actions[action.uid] = action
        self._sync_activities()
        self.server.state.flush()

    def _schedule_utility(self, coroutine: Any, name: str) -> None:
        try:
            task = asyncio.create_task(coroutine, name=name)
        except RuntimeError as exc:
            self.server.state.error_message = str(exc)
            return
        self._utility_tasks.add(task)
        task.add_done_callback(self._utility_tasks.discard)

    async def _submit_utility_action(
        self,
        *,
        tool_id: str,
        name: str,
        inputs: dict[str, Any],
        associated_run_uid: str | None = None,
    ) -> UtilityActionRecord | None:
        try:
            action = await asyncio.to_thread(
                self.service.submit_utility,
                tool_id=tool_id,
                name=name,
                inputs=inputs,
                associated_run_uid=associated_run_uid,
            )
            self._register_utility(action)
            await self._monitor_utility(action)
            return action
        except Exception as exc:
            failed = UtilityActionRecord(
                uid=f"utility-{uuid.uuid4().hex}",
                tool_id=tool_id,
                name=name,
                associated_run_uid=associated_run_uid,
                inputs=inputs,
                status=RunStatus.ERROR,
                message=str(exc),
            )
            self._register_utility(failed)
            self.server.state.error_message = f"{name} failed: {exc}"
            return None

    async def _monitor_utility(self, action: UtilityActionRecord) -> None:
        terminal = {RunStatus.OK, RunStatus.ERROR, RunStatus.CANCELLED}
        entrypoint_announced = False
        while action.status not in terminal:
            await asyncio.sleep(2.0)
            try:
                action = await asyncio.to_thread(self.service.refresh_utility, action)
            except Exception as exc:
                action.message = str(exc)
            if action.tool_id == GSASII_INTERACTIVE_TOOL_ID:
                self.server.state.gsasii_status_message = (
                    action.message or "Waiting for NDIP to allocate the GSAS-II desktop."
                )
            self._register_utility(action)
            if action.outputs.get("launch_url") and not entrypoint_announced:
                entrypoint_announced = True
                await self._utility_ready(action)
        if action.status == RunStatus.OK:
            await self._utility_completed(action)
        elif action.tool_id == GSASII_INTERACTIVE_TOOL_ID:
            state = self.server.state
            state.gsasii_launch_url = ""
            state.gsasii_session_status = "closed" if action.status == RunStatus.CANCELLED else "error"
            state.gsasii_status_message = action.message or "The GSAS-II session failed to start."
            if action.status == RunStatus.ERROR:
                state.error_message = action.message or "The GSAS-II session failed to start."
            else:
                state.notice = "GSAS-II session stopped; the last stable project snapshot remains in Galaxy History."
            state.flush()

    async def _utility_ready(self, action: UtilityActionRecord) -> None:
        state = self.server.state
        if action.tool_id == GSASII_INTERACTIVE_TOOL_ID:
            state.gsasii_launch_url = self._absolute_launch_url(action.outputs.get("launch_url", ""))
            state.gsasii_session_status = "ready"
            state.gsasii_status_message = "GSAS-II is ready. Open the desktop session to continue refinement."
            state.notice = "GSAS-II is ready. Open the desktop session to continue refinement."
        elif action.tool_id == RESULT_EXPLORER_TOOL_ID:
            state.notice = "Result Explorer is ready; open it from Companion-tool activity."
        self._sync_activities()
        state.flush()

    async def _utility_completed(self, action: UtilityActionRecord) -> None:
        state = self.server.state
        if action.tool_id == LIBRARY_BUILDER_TOOL_ID and action.outputs.get("library_archive"):
            state.database_source = "archive"
            state.library_archive_source = "galaxy"
            state.history_database_id = action.outputs["library_archive"]
            state.powgen_database_dataset_id = action.outputs["library_archive"]
            manifest: dict[str, Any] = {}
            manifest_id = action.outputs.get("library_manifest")
            if manifest_id:
                try:
                    manifest = await asyncio.to_thread(self.service._dataset_document, manifest_id) or {}
                except Exception:
                    manifest = {}
            built = int(manifest.get("phase_count") or manifest.get("n_phases") or 0)
            failures = list(manifest.get("failures") or [])
            preflight_failures = list(state.library_builder_failure_rows or [])
            preflight_skipped = int(state.library_builder_skipped_count or 0)
            builder_failure_rows = [
                {
                    "name": str(item.get("source_name") or item.get("id") or "CIF"),
                    "reason": str(item.get("error") or "Could not be added"),
                }
                for item in failures
                if isinstance(item, dict)
            ]
            state.library_builder_built_count = built
            total_skipped = preflight_skipped + len(failures)
            state.library_builder_skipped_count = total_skipped
            state.library_builder_failure_rows = [*preflight_failures, *builder_failure_rows][:8]
            state.library_builder_progress = 100
            state.library_builder_status = "partial" if total_skipped else "ready"
            if total_skipped:
                state.library_builder_message = (
                    f"Library ready with {built} usable phase(s); {total_skipped} source item(s) were skipped. "
                    "Open the build log for details."
                )
            else:
                state.library_builder_message = f"Library ready with {built} usable phase(s) and selected for the next analysis."
            state.notice = state.library_builder_message
        elif action.tool_id == SNS_RESOLVER_TOOL_ID:
            pattern_id = action.outputs.get("pattern")
            profile_id = action.outputs.get("instrument_profile")
            if pattern_id and profile_id:
                state.input_source = "galaxy"
                state.history_data_id = pattern_id
                state.history_instrument_id = profile_id
                metadata: dict[str, Any] = {}
                metadata_id = action.outputs.get("resolution_metadata")
                if metadata_id:
                    try:
                        metadata = await asyncio.to_thread(self.service._dataset_document, metadata_id)
                        metadata = metadata or {}
                    except Exception:
                        metadata = {}
                state.sns_resolution = {
                    "status": "ready",
                    "pattern": Path(str(metadata.get("resolved_pattern") or "diffraction pattern")).name,
                    "profile": Path(str(metadata.get("resolved_instrument") or "instrument profile")).name,
                    "bank": str(metadata.get("bank") or action.inputs.get("source|bank") or state.bank or "resolved"),
                    "provenance": " / ".join(
                        str(value)
                        for value in (metadata.get("instrument"), metadata.get("ipts"), metadata.get("run"))
                        if value
                    )
                    or "SNS archive",
                }
                state.notice = "SNS input resolved and selected from Galaxy History."
        elif action.tool_id == GPX_HANDOFF_TOOL_ID:
            state.notice = "The selected GSAS-II checkpoint is ready in Galaxy History."
        elif action.tool_id == GSASII_INTERACTIVE_TOOL_ID:
            state.gsasii_launch_url = ""
            state.gsasii_session_status = "closed"
            state.gsasii_status_message = "GSAS-II session closed; the saved project is available in Galaxy History."
            state.notice = "GSAS-II session closed; the saved project is available in Galaxy History."
        elif action.tool_id == COMPARE_SERIES_TOOL_ID:
            state.notice = "Series comparison completed; open its report from Companion-tool activity."
        elif action.tool_id == RESULT_EXPLORER_TOOL_ID:
            state.notice = "Result Explorer is ready; open it from Companion-tool activity."
        self.refresh_history()
        self._sync_activities()
        state.flush()

    def save_current_configuration(self, submission_payload: dict[str, Any] | None = None, **_: Any) -> None:
        state = self.server.state
        try:
            config = self._configuration(submission_payload)
        except Exception as exc:
            state.error_message = str(exc)
            state.flush()
            return

        state.error_message = ""
        state.flush()

        async def save() -> None:
            try:
                action = await asyncio.to_thread(self.service.save_configuration, config)
                self._register_utility(action)
                state.error_message = ""
                state.notice = "Reusable configuration saved to Galaxy History."
                self.refresh_history()
                state.flush()
            except Exception as exc:
                state.error_message = f"Could not save configuration: {exc}"
                state.flush()

        self._schedule_utility(save(), "radar-save-configuration")

    def build_candidate_library(self, **_: Any) -> None:
        state = self.server.state
        cif_ids = [str(item) for item in state.library_builder_cif_ids or []]
        local_paths = [str(item) for item in state.library_builder_local_paths or []]
        if not cif_ids and not local_paths:
            return

        library_name = re.sub(r"[^A-Za-z0-9_. -]+", "_", str(state.library_builder_name or "")).strip(" ._")
        if not library_name:
            state.error_message = "Enter a library name before building."
            return
        library_mode = "augmented" if state.database_source == "custom_augmented" else "mini"
        state.library_builder_mode = library_mode
        state.library_builder_active = True
        state.library_builder_status = "uploading"
        state.library_builder_progress = 5
        state.library_builder_built_count = 0
        state.library_builder_skipped_count = 0
        state.library_builder_failure_rows = []
        state.library_builder_message = "Preparing CIF structures for Galaxy…"
        state.flush()

        async def build() -> None:
            try:
                state.library_builder_message = "Validating and deduplicating selected CIF and ZIP sources..."
                state.library_builder_progress = 12
                state.flush()

                def prepare_bundle() -> tuple[Path, dict[str, Any]]:
                    with tempfile.TemporaryDirectory(prefix="radar_pd_history_cifs_") as temporary:
                        combined_paths = list(local_paths)
                        for index, dataset_id in enumerate(dict.fromkeys(cif_ids), start=1):
                            metadata = self.service._dataset_metadata(dataset_id)
                            original_name = Path(str(metadata.get("name") or f"history_{index}.cif").replace("\\", "/")).name
                            if not original_name.lower().endswith(".cif"):
                                original_name = f"{original_name}.cif"
                            target = Path(temporary) / f"{index:05d}_{original_name}"
                            self.service._download_dataset(dataset_id, target)
                            combined_paths.append(str(target))
                        return build_cif_source_archive(combined_paths, library_name)

                bundle_path, bundle_stats = await asyncio.to_thread(prepare_bundle)
                state.library_builder_skipped_count = int(bundle_stats.get("skipped_count") or 0)
                state.library_builder_failure_rows = [
                    {"name": "Source preflight", "reason": str(reason)}
                    for reason in list(bundle_stats.get("failures") or [])[:8]
                ]
                state.library_builder_message = (
                    f"Uploading one validated bundle containing {int(bundle_stats.get('cif_count') or 0):,} unique CIF(s)..."
                )
                state.library_builder_progress = 24
                state.flush()
                _, archive_id = await asyncio.to_thread(
                    self.service._upload_one,
                    "library_cif_bundle",
                    str(bundle_path),
                    "CIF source bundle",
                )
                source_inputs = {"cif_archive": {"dataset_id": archive_id}}
                state.library_builder_status = "building"
                state.library_builder_progress = 35
                state.library_builder_message = f"Building {library_name} in Galaxy..."
                state.flush()
            except Exception as exc:
                state.library_builder_active = False
                state.library_builder_status = "error"
                state.library_builder_message = f"Could not prepare the CIF inputs: {exc}"
                state.error_message = state.library_builder_message
                state.flush()
                return
            action = await self._submit_utility_action(
                tool_id=LIBRARY_BUILDER_TOOL_ID,
                name=f"Build custom library: {library_name}",
                inputs={
                    **source_inputs,
                    "library_name": library_name,
                    "library_mode": library_mode,
                    "radiation": str(state.radiation or "neutron"),
                    "overwrite": "",
                },
            )
            state.library_builder_active = False
            if action is None or action.status != RunStatus.OK:
                state.library_builder_status = "error"
                state.library_builder_message = (
                    action.message if action is not None and action.message else "Galaxy could not build the custom library."
                )
                state.library_builder_progress = 0
            state.flush()

        self._schedule_utility(build(), "radar-build-library")

    def resolve_sns_input(self, **_: Any) -> None:
        try:
            selection = self._inputs()
        except Exception as exc:
            self.server.state.error_message = str(exc)
            return

        async def resolve() -> None:
            utility_inputs: dict[str, Any]
            if selection.source == InputSource.IPTS_EVENT:
                event_id = selection.event_dataset_id
                if not event_id and selection.event_file_path:
                    _, event_id = await asyncio.to_thread(
                        self.service._upload_one, "event_file", selection.event_file_path, "NeXus event file"
                    )
                utility_inputs = {
                    "source|source_kind": "event_file",
                    "source|event_file": {"dataset_id": event_id},
                    "source|bank": selection.bank,
                }
            else:
                utility_inputs = {
                    "source|source_kind": "ipts_manual",
                    "source|instrument": selection.instrument,
                    "source|ipts": selection.ipts,
                    "source|run_number": selection.run_number,
                    "source|bank": selection.bank,
                    "source|facility_root": selection.facility_root,
                }
            await self._submit_utility_action(
                tool_id=SNS_RESOLVER_TOOL_ID,
                name="Resolve SNS input",
                inputs=utility_inputs,
            )

        self._schedule_utility(resolve(), "radar-resolve-sns")

    def launch_result_explorer(self, **_: Any) -> None:
        record = self.records.get(self.server.state.selected_run_uid)
        archive_id = record.output_dataset_ids.get("results_archive") if record else None
        if record is None or not archive_id:
            self.server.state.error_message = "This run does not expose a Complete Results Archive dataset."
            return
        self._schedule_utility(
            self._submit_utility_action(
                tool_id=RESULT_EXPLORER_TOOL_ID,
                name="Open Result Explorer",
                inputs={
                    "result_source|source_kind": "archive",
                    "result_source|results_archive": {"dataset_id": archive_id},
                },
                associated_run_uid=record.uid,
            ),
            "radar-result-explorer",
        )

    async def _selected_checkpoint_element(
        self,
        record: RunRecord,
        selected: str,
        checkpoint_rows: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        checkpoint = next(
            (
                item
                for item in checkpoint_rows or self.server.state.checkpoint_rows or []
                if str(item.get("id") or "") == selected or str(item.get("path") or "") == selected
            ),
            {},
        )
        collection_id = record.output_dataset_ids.get("gpx_projects")
        if not collection_id:
            job_id = str(record.galaxy_job_id or record.uid or "").strip()
            if job_id:
                try:
                    output_ids = await asyncio.to_thread(self.service.job_output_ids, job_id)
                except Exception:
                    output_ids = {}
                record.output_dataset_ids.update(output_ids)
                collection_id = record.output_dataset_ids.get("gpx_projects")
        if not collection_id:
            local_path = Path(str(checkpoint.get("path") or ""))
            if local_path.suffix.casefold() != ".gpx" or not local_path.is_file():
                return None
            cache = getattr(self, "_published_checkpoint_ids", None)
            if cache is None:
                cache = {}
                self._published_checkpoint_ids = cache
            cache_key = (record.uid, selected)
            dataset_id = cache.get(cache_key)
            if not dataset_id:
                dataset_id = await asyncio.to_thread(
                    self.service.upload_document,
                    local_path,
                    label=f"GSAS-II checkpoint {record.name}",
                )
                cache[cache_key] = dataset_id
            return {"id": dataset_id, "name": local_path.name}
        try:
            elements = await asyncio.to_thread(self.service.collection_elements, collection_id)
        except Exception:
            # The hosted GSAS-II tool can consume the complete HDCA directly,
            # so a transient element-inspection failure must not block launch.
            elements = []
        def aliases(value: Any) -> set[str]:
            stem = Path(str(value or "")).stem.casefold()
            key = re.sub(r"[^a-z0-9]+", "", stem)
            if not key:
                return set()
            values = {key}
            if "seqfinalmainpolished" in key or "02mainphaseanchor" in key:
                values.update({"seqfinalmainpolished", "02mainphaseanchor"})
            pass_match = re.search(r"seqpass(\d+)keptpolished", key)
            accepted_match = re.search(r"acceptedmodelafterpass(\d+)", key)
            pass_number = pass_match or accepted_match
            if pass_number:
                number = pass_number.group(1)
                values.update({f"seqpass{number}keptpolished", f"acceptedmodelafterpass{number}"})
            if "patternproject" in key or "01importedpatternandmainphase" in key:
                values.update({"patternproject", "01importedpatternandmainphase"})
            return values

        target_aliases: set[str] = set()
        for value in (
            selected,
            checkpoint.get("path"),
            checkpoint.get("name"),
            checkpoint.get("galaxy_element_name"),
        ):
            target_aliases.update(aliases(value))
        match = next(
            (
                item
                for item in elements
                if aliases(item.get("name")) & target_aliases
            ),
            None,
        )
        if match is None and len(elements) == 1:
            match = elements[0]
        return match

    def open_selected_checkpoint_in_gsasii(self, **_: Any) -> None:
        state = self.server.state
        record = self.records.get(state.selected_run_uid)
        selected = str(state.selected_checkpoint or "")
        checkpoint_rows = list(state.checkpoint_rows or [])
        if record is None or not selected:
            return
        if state.gsasii_session_status in {"starting", "ready"}:
            state.notice = "A GSAS-II session is already starting or ready."
            state.flush()
            return
        state.gsasii_launch_url = ""
        state.gsasii_session_status = "starting"
        state.error_message = ""
        state.gsasii_status_message = (
            "Submitting the selected checkpoint and waiting for NDIP to allocate the GSAS-II desktop."
        )
        state.flush()

        async def launch() -> None:
            try:
                match = await self._selected_checkpoint_element(record, selected, checkpoint_rows)
            except Exception as exc:
                state.gsasii_session_status = "error"
                state.gsasii_status_message = f"Could not publish the selected GPX checkpoint to Galaxy: {exc}"
                state.error_message = state.gsasii_status_message
                state.flush()
                return
            collection_id = record.output_dataset_ids.get("gpx_projects")
            if match is not None:
                launch_name = Path(match["name"]).stem
                launch_inputs: dict[str, Any] = {
                    "project_source|source_kind": "single",
                    "project_source|gpx_project": {"dataset_id": match["id"]},
                }
            elif collection_id:
                launch_name = f"preferred checkpoint from {record.name}"
                launch_inputs = {
                    "project_source|source_kind": "collection",
                    "project_source|gpx_projects": {"collection_id": collection_id},
                }
                state.notice = (
                    "Galaxy renamed the selected checkpoint. GSAS-II will open the preferred accepted "
                    "checkpoint from this run's published GPX collection."
                )
                state.error_message = ""
            else:
                state.gsasii_session_status = "error"
                state.gsasii_status_message = (
                    "This run has neither a published Galaxy GPX nor an available GPX checkpoint in its results archive."
                )
                state.error_message = state.gsasii_status_message
                state.flush()
                return
            action = await self._submit_utility_action(
                tool_id=GSASII_INTERACTIVE_TOOL_ID,
                name=f"Refine in GSAS-II: {launch_name}",
                inputs=launch_inputs,
                associated_run_uid=record.uid,
            )
            if action is None:
                state.gsasii_session_status = "error"
                state.gsasii_status_message = "Galaxy did not accept the GSAS-II launch request."
                state.flush()

        self._schedule_utility(launch(), "radar-gsasii-interactive")

    def handoff_selected_checkpoint(self, **_: Any) -> None:
        record = self.records.get(self.server.state.selected_run_uid)
        selected = str(self.server.state.selected_checkpoint or "")
        if record is None or not selected:
            return

        async def handoff() -> None:
            match = await self._selected_checkpoint_element(record, selected)
            if match is None:
                return
            inputs: dict[str, Any] = {"gpx_project": {"dataset_id": match["id"]}}
            if record.output_dataset_ids.get("gpx_index"):
                inputs["gpx_index"] = {"dataset_id": record.output_dataset_ids["gpx_index"]}
            await self._submit_utility_action(
                tool_id=GPX_HANDOFF_TOOL_ID,
                name="GSAS-II checkpoint handoff",
                inputs=inputs,
                associated_run_uid=record.uid,
            )

        self._schedule_utility(handoff(), "radar-gpx-handoff")

    def compare_selected_runs(self, **_: Any) -> None:
        records = [self.records[uid] for uid in self.server.state.comparison_run_uids or [] if uid in self.records]
        summary_ids = [record.output_dataset_ids.get("summary") for record in records]
        summary_ids = [str(identifier) for identifier in summary_ids if identifier]
        if len(summary_ids) < 2:
            self.server.state.error_message = "Select at least two completed runs with published summary datasets."
            return

        async def compare() -> None:
            collection_id = await asyncio.to_thread(
                self.service.create_dataset_collection,
                f"RADAR-PD series summaries {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                summary_ids,
            )
            await self._submit_utility_action(
                tool_id=COMPARE_SERIES_TOOL_ID,
                name="Compare scan series",
                inputs={"summaries": {"collection_id": collection_id}},
            )

        self._schedule_utility(compare(), "radar-compare-series")

    def retry_utility(self, uid: str | None = None, **_: Any) -> None:
        action = self.utility_actions.get(str(uid or ""))
        if action is None:
            return
        self._schedule_utility(
            self._submit_utility_action(
                tool_id=action.tool_id,
                name=f"Retry: {action.name}",
                inputs=action.inputs,
                associated_run_uid=action.associated_run_uid,
            ),
            f"radar-retry-{action.uid[-8:]}",
        )

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
        .radar-workflow-toggle { display: flex !important; width: 100%; border: 1px solid var(--radar-line-strong); border-radius: 6px; overflow: hidden; }
        .radar-workflow-toggle .v-btn { flex: 1 1 50%; min-width: 0; text-transform: none; font-weight: 730; }
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
        .radar-custom-budget-controls { display: grid; gap: 8px; margin-top: 10px; }
        .radar-custom-budget-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(300px, 100%), 1fr)); gap: 8px; }
        .radar-custom-budget-grid .v-input { min-width: 0; }
        .radar-facility-picker { margin: 10px 0; padding: 10px; background: #fff; border: 1px solid var(--radar-line); border-radius: 8px; }
        .radar-facility-picker-title { display: block; margin-bottom: 8px; color: var(--radar-brand-900); font-size: 12px; }
        .radar-facility-path-row { display: grid; grid-template-columns: minmax(0, 1fr) 38px; gap: 7px; align-items: center; }
        .radar-soft-panel { padding: 10px; background: var(--radar-surface-muted); border: 1px solid var(--radar-line); border-radius: 8px; }
        .radar-soft-panel code { display: block; margin: 5px 0 7px; color: var(--radar-brand-900); font-size: 11px; overflow-wrap: anywhere; }
        .radar-preflight-summary { border-top: 1px solid var(--radar-line); border-bottom: 1px solid var(--radar-line); }
        .radar-preflight-row { display: grid; grid-template-columns: 112px minmax(0, 1fr); gap: 8px; padding: 6px 2px; border-bottom: 1px solid #e7efeb; }
        .radar-preflight-row:last-child { border-bottom: 0; }
        .radar-preflight-label { color: var(--radar-muted); font-size: 10px; font-weight: 750; text-transform: uppercase; }
        .radar-preflight-value { min-width: 0; color: var(--radar-ink); font-size: 11px; line-height: 1.35; overflow-wrap: anywhere; }
        .radar-mode-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        .radar-mode-card { display: grid; grid-template-columns: auto 1fr; gap: 4px 7px; align-items: center; padding: 11px; border: 1px solid var(--radar-line-strong); border-radius: 8px; background: #fff; cursor: pointer; }
        .radar-mode-card strong { color: var(--radar-brand-900); font-size: 14px; }
        .radar-mode-card span { grid-column: 1 / -1; color: var(--radar-muted); font-size: 11px; line-height: 1.35; }
        .radar-mode-card.is-selected { border: 2px solid var(--radar-brand-600); padding: 10px; background: #edf7f2; box-shadow: 0 0 0 2px rgba(31,107,75,.08); }
        .radar-mode-card:focus-visible { outline: 3px solid rgba(31,107,75,.28); outline-offset: 2px; }
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
        .radar-multi-upload { display: grid; gap: 8px; margin: 4px 0 12px; }
        .radar-library-panel { display: grid; gap: 8px; margin-top: 10px; }
        .radar-library-scope-callout, .radar-library-source-summary {
          display: grid; gap: 3px; padding: 10px 11px; border: 1px solid var(--radar-line); border-radius: 7px; background: var(--radar-brand-050);
        }
        .radar-library-scope-callout strong, .radar-library-source-summary strong { color: var(--radar-brand-900); font-size: 12px; }
        .radar-library-scope-callout span, .radar-library-source-summary span { color: var(--radar-muted); font-size: 11px; line-height: 1.4; }
        .radar-library-source-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
        .radar-library-source-card { display: grid; align-content: start; gap: 6px; padding: 10px; border: 1px solid var(--radar-line); border-radius: 7px; background: #fff; }
        .radar-library-source-card strong { color: var(--radar-brand-900); font-size: 12px; }
        .radar-library-source-card > span { min-height: 31px; color: var(--radar-muted); font-size: 11px; line-height: 1.35; }
        .radar-library-source-action { align-self: end; width: 100%; min-width: 0; height: auto !important; min-height: 36px; padding-block: 6px !important; }
        .radar-library-source-action .v-btn__content { min-width: 0; max-width: 100%; white-space: normal; overflow-wrap: anywhere; line-height: 1.25; text-align: center; }
        .radar-library-source-action .v-btn__prepend { flex: 0 0 auto; margin-inline-end: 6px; }
        .radar-library-upload-pending { padding: 9px 11px; border: 1px solid #b8d2c7; border-radius: 6px; background: #edf7f2; color: var(--radar-brand-900); font-size: 12px; font-weight: 650; }
        .radar-source-review .v-expansion-panel-title { min-height: 38px !important; font-size: 12px; }
        .radar-source-review .v-expansion-panel-text__wrapper { display: grid; gap: 6px; padding: 7px !important; }
        .radar-cif-file-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 8px 10px; border: 1px solid var(--radar-line); border-radius: 6px; background: #fff; }
        .radar-cif-file-row .radar-file-copy { min-width: 0; }
        .radar-cif-file-row .radar-file-copy strong { display: block; max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12px; }
        .radar-cif-file-row .radar-file-copy span { display: block; color: var(--radar-muted); font-size: 11px; }
        .radar-inline-divider { color: var(--radar-muted); font-size: 11px; font-weight: 700; margin: 10px 0 6px; text-transform: uppercase; }
        .radar-library-build-status { padding: 10px 12px; margin: 8px 0 10px; border: 1px solid var(--radar-line); border-radius: 6px; background: var(--radar-brand-050); color: var(--radar-ink); font-size: 12px; }
        .radar-library-build-status span { display: block; color: var(--radar-muted); margin-top: 6px; }
        .radar-library-failure-row { margin-top: 7px; padding-top: 7px; border-top: 1px solid var(--radar-line); }
        .radar-library-failure-row strong, .radar-library-failure-row span { display: block; overflow-wrap: anywhere; }
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
        .radar-experiment-grid { display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(320px, .75fr); gap: 16px; align-items: start; }
        .radar-experiment-phase-card, .radar-experiment-heatmap-card { grid-column: 1; }
        .radar-experiment-quality-card, .radar-experiment-latest-card { grid-column: 2; }
        .radar-experiment-quality-card { grid-row: 1; }
        .radar-experiment-heatmap-card { grid-row: 2; }
        .radar-experiment-latest-card { grid-row: 2; }
        .radar-experiment-table-card, .radar-experiment-queue-card { grid-column: 1 / -1; }
        .radar-experiment-table-card { grid-row: 3; }
        .radar-experiment-queue-card { grid-row: 4; }
        .radar-experiment-phase-card .radar-plot-frame { height: 430px; min-height: 430px; }
        .radar-experiment-quality-card .radar-plot-frame { height: 340px; min-height: 340px; }
        .radar-experiment-heatmap-frame { height: 430px; min-height: 320px; }
        .radar-experiment-controls { display: grid; grid-template-columns: minmax(220px, 1fr) minmax(180px, .7fr); gap: 10px; margin: 2px 0 10px; }
        .radar-experiment-controls .radar-phase-selector { grid-column: 1 / -1; }
        .radar-experiment-controls .v-field { background: #fff; }
        .radar-experiment-controls .v-chip { max-width: 240px; }
        .radar-experiment-controls .v-chip__content { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .radar-experiment-provenance { display: flex; flex-wrap: wrap; gap: 7px 16px; color: var(--radar-muted); font-size: 12px; }
        .radar-experiment-provenance span { overflow-wrap: anywhere; }
        .radar-scan-inspector-summary { display: grid; gap: 4px; margin: 12px 0; padding: 10px 12px; border: 1px solid var(--radar-line); border-radius: 7px; background: var(--radar-soft); }
        .radar-scan-inspector-summary strong { color: var(--radar-brand-900); font-size: 14px; }
        .radar-scan-inspector-summary span { color: var(--radar-muted); font-size: 12px; }
        .radar-section-heading { display: flex; justify-content: space-between; align-items: flex-end; gap: 18px; margin: 0 0 15px; }
        .radar-section-heading h2 { margin: 0; color: var(--radar-brand-900); font-size: 27px; }
        .radar-section-heading p, .radar-section-help { margin: 4px 0 0; color: var(--radar-muted); font-size: 13px; line-height: 1.5; }
        .radar-button-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
        .radar-monitor-summary { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 18px; padding: 19px; background: #fff; border: 1px solid var(--radar-line); border-left: 4px solid var(--radar-brand-700); border-radius: 8px; }
        .radar-monitor-stage h3 { margin: 5px 0 3px; color: var(--radar-brand-900); font-size: 21px; }
        .radar-monitor-stage p { margin: 0; color: var(--radar-muted); }
        .radar-elapsed { display: block; color: var(--radar-muted); font-size: 12px; margin-top: 6px; }
        .radar-job-context { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-top: 9px; }
        .radar-job-context code { color: var(--radar-brand-900); font-size: 11px; }
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
        .radar-run-provenance { padding: 12px; background: #fff; border: 1px solid var(--radar-line); border-radius: 8px; }
        .radar-run-provenance-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(185px, 1fr)); gap: 8px; }
        .radar-run-provenance-item { min-width: 0; padding: 8px 9px; background: var(--radar-surface-muted); border-radius: 6px; }
        .radar-run-provenance-item span, .radar-run-provenance-item strong { display: block; overflow-wrap: anywhere; }
        .radar-run-provenance-item span { color: var(--radar-muted); font-size: 10px; font-weight: 760; text-transform: uppercase; }
        .radar-run-provenance-item strong { margin-top: 3px; color: var(--radar-ink); font-size: 12px; }
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
        .radar-plot-frame { position: relative; width: 100%; height: 720px; min-height: 560px; overflow: hidden; }
        .radar-plot-frame > div { width: 100% !important; height: 100% !important; }
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
        .radar-gsasii-launch { flex: 0 0 auto; }
        .radar-file-toolbar { display: grid; grid-template-columns: minmax(240px, 1fr) minmax(210px, 280px); gap: 12px; align-items: center; margin-bottom: 11px; }
        .radar-file-row { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 9px 2px; border-bottom: 1px solid var(--radar-line); }
        .radar-file-row:last-child { border-bottom: 0; }
        .radar-file-copy { min-width: 0; }
        .radar-file-copy strong, .radar-file-copy span { display: block; overflow-wrap: anywhere; }
        .radar-file-copy strong { color: var(--radar-ink); font-size: 13px; }
        .radar-file-copy span { margin-top: 2px; color: var(--radar-muted); font-size: 11px; }
        .radar-activity-panel { border: 1px solid var(--radar-line); border-radius: 8px; overflow: hidden; }
        .radar-activity-row { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 9px 0; border-bottom: 1px solid var(--radar-line); }
        .radar-activity-row:last-child { border-bottom: 0; }
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
          .radar-library-source-grid { grid-template-columns: 1fr; }
          .radar-context-header { padding: 8px 10px; }
          .radar-product-subtitle, .radar-run-chip { display: none; }
          .radar-workspace-inner { padding: 15px 10px 38px; }
          .radar-run-context, .radar-section-heading { align-items: flex-start; flex-direction: column; }
          .radar-workspace-nav { overflow-x: auto; }
          .radar-workspace-nav .v-btn { flex: 0 0 auto; }
          .radar-experiment-grid { grid-template-columns: minmax(0, 1fr); }
          .radar-experiment-phase-card, .radar-experiment-quality-card,
          .radar-experiment-heatmap-card, .radar-experiment-table-card,
          .radar-experiment-latest-card, .radar-experiment-queue-card { grid-column: 1; grid-row: auto; }
          .radar-experiment-controls { grid-template-columns: minmax(0, 1fr); }
          .radar-metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          .radar-field-pair { grid-template-columns: 1fr; }
          .radar-mode-cards, .radar-file-toolbar { grid-template-columns: 1fr; }
          .radar-stage-timeline { grid-template-columns: repeat(6, 130px); }
          .radar-monitor-summary { grid-template-columns: 1fr; }
          .radar-progress-number { align-items: flex-start; }
          .radar-upload-card { grid-template-columns: 1fr; }
          .radar-upload-actions { justify-content: flex-start; }
        }
        """
