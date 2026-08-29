"""Structural regressions for the stateful NOVA interface tree."""

from radar_pd_nova.app import (
    RadarPdNovaApp,
    _format_eastern_time,
    _powgen_retry_time,
    _powgen_submission_error_summary,
)


def test_powgen_retry_messages_hide_proxy_html_and_parse_utc() -> None:
    proxy_error = "GET: error 503: <html><h1>503 Service Temporarily Unavailable</h1></html>"

    assert _powgen_submission_error_summary(proxy_error) == (
        "NDIP/Galaxy was temporarily unavailable (HTTP 503)."
    )
    assert _powgen_submission_error_summary(
        "This job was killed when Galaxy was restarted. Please retry the job."
    ) == "Galaxy restarted while the scan inputs were being submitted."
    assert _powgen_retry_time("2026-08-24T17:48:35.680757+00:00").isoformat() == (
        "2026-08-24T17:48:35.680757+00:00"
    )


def test_powgen_user_timestamps_are_explicitly_eastern_and_dst_aware() -> None:
    assert _format_eastern_time("2026-01-15T18:00:00Z") == "2026-01-15 13:00:00 EST"
    assert _format_eastern_time("2026-08-25T18:00:00Z") == "2026-08-25 14:00:00 EDT"
    assert (
        _format_eastern_time("2026-08-24T04:21:26.974623667-04:00")
        == "2026-08-24 04:21:26 EDT"
    )


def test_setup_panels_and_uploads_have_single_stable_instances() -> None:
    app = RadarPdNovaApp()
    template = app.layout.html

    assert app.server.state.setup_panels == [0, 2]
    assert app.server.state.workflow_mode == "single"
    assert [item["value"] for item in app.server.state.workflow_options] == ["single", "powgen"]
    assert 'v-model="workflow_mode"' in template
    assert 'v-show="workflow_mode === \'powgen\'"' in template
    assert template.count('v-show="workflow_mode === \'single\'"') == 2

    for title in (
        "Measurement Type",
        "Candidate Library",
        "Data Collection",
        "Chemistry Policy",
        "Pattern Regions",
        "Background Correction",
        "Magnetic Ordering Precheck",
        "Analysis Mode",
        "Runtime Budget",
        "Expert Tuning",
        "Review Run Plan",
    ):
        assert template.count(title) == 1

    # Every numbered panel is eager-mounted: collapsing a panel must not
    # destroy and recreate its stateful file inputs.
    assert template.count("<VExpansionPanelText eager") == 11
    assert 'v-show="radiation === \'neutron\'"' in template
    assert "{{ radiation === 'neutron' ? '8' : '7' }}" in template
    assert "{{ radiation === 'neutron' ? '11' : '10' }}" in template
    assert app.server.state.powgen_wavelength == ""
    assert app.server.state.powgen_wavelength_options[0] == {
        "title": "Auto-detect from latest scan (recommended)",
        "value": "",
    }
    assert "!powgen_wavelength" not in template
    assert "The wavelength is auto-detected unless you choose an override." in (
        app.server.state.powgen_message
    )
    assert "The packaged Cu K-alpha GSAS-II profile will be used" in template
    for panel_value in range(11):
        assert template.count(f':value="{panel_value}"') >= 1

    # Each browser File is decoded independently. A rejected file must not
    # discard the library panel or the sources already accepted by the server.
    assert "Promise.all" not in template
    assert ".arrayBuffer().then" in template
    # Vuetify updates one bound File model per event. Users add more sources
    # by repeating the action, avoiding browser-dependent File[] payloads.
    assert 'v-model="radar_cif_library_upload_browser_file"' in template
    assert 'v-model="radar_cif_library_upload_browser_archive"' in template
    assert "decode_radar_cif_library_upload" in template
    assert "decode_archive_radar_cif_library_upload" in template
    assert "Array.isArray(radar_cif_library_upload" not in template

    for stable_key in (
        "radar-diffraction-upload-native",
        "radar-instrument-upload-native",
        "radar-main-cif-upload-native",
        "powgen-main-cif-upload-native",
    ):
        assert template.count(f":key=\"'{stable_key}'\"") == 1

    assert "From Server" not in template
    assert 'activator="parent"' not in template


def test_advanced_scientific_controls_are_visible_and_reach_configuration() -> None:
    app = RadarPdNovaApp()
    template = app.layout.html
    state = app.server.state

    for label in (
        "Fixed half-width (pattern x-axis units)",
        "FWHM multiplier",
        "Fractional d tolerance",
        "Calibrate Zero and U/V/W from the supplied main phase",
        "Duplicate-candidate threshold",
        "Nudge scoring Q max",
        "Pearson cell-refine cutoff",
        "Use automatic candidate pruning",
        "Excluded space groups",
    ):
        assert label in template

    state.sample_elements = "Fe, O"
    state.radiation = "xray"
    state.main_cif_source = "upload"
    state.main_cif_path = "main.cif"
    state.reference_masks_enabled = True
    state.reference_mask_presets = ["Al_fcc"]
    state.reference_window_mode = "fixed"
    state.reference_fixed_half_width = 0.4
    state.light_calibration_enabled = True
    state.analysis_mode = "full"
    state.full_profile = "custom"
    state.full_dedup_threshold = 0.9
    state.full_score_q_max = 9.0
    state.full_pearson_cell_min_r = 0.4
    state.full_lattice_tiebreak_score_tol = 0.001
    state.full_candidate_pruning = False
    state.excluded_space_groups = "1, 2, 15"

    config = app._configuration()
    assert config.reference_fixed_half_width == 0.4
    assert config.light_calibration_enabled is True
    assert config.full_dedup_threshold == 0.9
    assert config.full_score_q_max == 9.0
    assert config.full_candidate_pruning is False
    assert config.excluded_space_groups == [1, 2, 15]


def test_full_custom_controls_belong_to_runtime_budget_and_use_responsive_columns() -> None:
    app = RadarPdNovaApp()
    template = app.layout.html
    css = app._css()

    custom_controls = template.index('class="radar-custom-budget-controls"')
    assert template.index("Runtime Budget") < custom_controls < template.index("Expert Tuning")
    assert template.count('class="radar-custom-budget-controls"') == 1
    assert template.count("Minimum phase wt%") == 1
    assert ".radar-custom-budget-grid" in css
    assert "repeat(auto-fit, minmax(min(300px, 100%), 1fr))" in css


def test_plotly_canvases_have_nonzero_layout_frames() -> None:
    app = RadarPdNovaApp()
    template = app.layout.html

    assert template.count('class="radar-plot-frame"') == 4
    assert ".radar-plot-frame { position: relative; width: 100%; height: 720px;" in app._css()
    assert "Plotly.Plots.resize" in template
    assert "window.Plotly && window.Plotly.Plots.resize" in template


def test_powgen_experiment_dashboard_exposes_scientific_trends() -> None:
    app = RadarPdNovaApp()
    template = app.layout.html

    for text in (
        "Open experiment dashboard",
        "POWGEN LIVE EXPERIMENT",
        "Refined phase fractions",
        "Refinement quality",
        "All reported phases across scans",
        "Completed analyses",
        "Inspect one completed scan",
        "Scan processing status",
    ):
        assert text in template

    assert "powgen_scientific_rows" in template
    assert "powgen_dashboard_metrics" in template
    assert "powgen_selected_run_id" in template
    assert "powgen_selected_phases" in template


def test_workbench_exposes_atomic_pending_and_companion_actions() -> None:
    app = RadarPdNovaApp()
    template = app.layout.html

    assert "Run analysis on NDIP" in template
    assert "Server request:" in template
    assert "selected_run_stage" in template
    assert "Reload results from Galaxy" in template
    assert "Resolve and verify SNS input" in template
    assert "Build and use this library" in template
    assert "full_profile + ' search profile'" in template
    assert "rapid_gsas_validation_limit + ' final refinements'" in template


def test_powgen_monitor_requires_preflight_and_exposes_safe_backfill_controls() -> None:
    app = RadarPdNovaApp()
    template = app.layout.html

    assert app.server.state.powgen_backfill_mode == "latest_5"
    assert app.server.state.history_panels == []
    assert app.server.state.powgen_preflight_ready is False
    assert app.server.state.result_explorer_available is False
    assert app.server.state.powgen_backfill_options[0]["title"] == "Latest 5 existing scans, then new scans"
    assert any(option["value"] == "new_only" for option in app.server.state.powgen_backfill_options)
    assert "Check experiment and inputs" in template
    assert 'v-model="history_panels"' in template
    assert "history_panels = []; run_search = ''" in template
    assert "powgen_monitoring || !powgen_preflight_ready" in template
    assert "Refresh now" in template
    assert "Next check:" in template
    assert "Save reusable configuration to History" in template
    assert "history_configuration_summary" in template
    assert "history_configuration_yaml" in template
    assert "No reusable configurations are in this Galaxy History" in template
    assert "POWGEN instrument profile fallback" in template
    assert app.server.state.powgen_instrument_source == "automatic"
    assert any(
        option["value"] == "computer" and "Upload a GSAS-II profile" in option["title"]
        for option in app.server.state.powgen_instrument_source_options
    )
    assert "/SNS/PG3/run_cycle_" in template
    assert "Open Result Explorer" in template
    assert 'v-if="result_explorer_available"' in template
    assert "Open GPX in GSAS-II" in template
    assert "Open GSAS-II" in template
    assert 'target="_blank"' in template
    assert 'rel="noopener noreferrer"' in template
    assert "window.setTimeout" in template
    assert "() => setTimeout" not in template
    assert "No GPX checkpoint was published for this run." in template
    assert "checkpoint_rows.length > 0" in template
    assert "POWGEN live uses a low-latency publication profile" in template
    assert "item.created_display" in template
    assert "job {{ run.job_short }}" in template
    assert "gsasii_session_status !== 'ready'" in template
    assert 'itemValue="id"' in template
    assert "Compare selected series" in template
    assert "Companion-tool activity" in template
    assert "Technical files" in template
    assert "No published files match these filters." in template
    assert "group.files.filter(item =>" in template
    assert template.count("radar-mode-card") >= 2


def test_facility_workspace_exposes_one_folder_and_independent_inputs() -> None:
    app = RadarPdNovaApp()
    template = app.layout.html

    assert any(
        option["value"] == "ipts_browser" and "SNS/HFIR" in option["title"]
        for option in app.server.state.source_options
    )
    for text in (
        "Experiment working folder",
        "Neutron facility",
        "Diffraction beamline",
        "Experiment (IPTS)",
        "Current working folder",
        "Enter subfolder",
        "Export completed results into this working folder",
        "A uniquely named results ZIP will be written directly into the current working folder.",
        "Where is the GSAS-II instrument profile?",
        "Diffraction data in working folder",
        "Instrument profile in working folder",
        "Known/main-phase CIF in working folder",
    ):
        assert text in template

    assert template.count("radar-instrument-upload-native") == 1
    assert "facility_working_directory" in template
    assert "facility_working_subdirectory" in template
    assert "Main-phase CIF from current IPTS auxiliary folder" not in template
    assert app.server.state.facility_options == [
        {"title": "Spallation Neutron Source (SNS)", "value": "SNS"},
        {"title": "High Flux Isotope Reactor (HFIR)", "value": "HFIR"},
    ]


def test_data_collection_exposes_three_clear_standard_sources() -> None:
    app = RadarPdNovaApp()
    template = app.layout.html

    assert app.server.state.source_options == [
        {"title": "Upload from this computer", "value": "upload"},
        {"title": "Choose from Galaxy History", "value": "galaxy"},
        {"title": "Browse an SNS/HFIR experiment folder", "value": "ipts_browser"},
    ]
    assert "Where is the diffraction pattern?" in template
    assert "Where is the GSAS-II instrument profile?" in template
    assert "Do you have a known/main-phase CIF?" in template
    assert "These files stay with this single-pattern session if you switch back." in template
    assert "A reusable configuration saves analysis settings only" in template
    assert "POWGEN uses the inputs selected in its own panel" in template


def test_chemistry_policy_uses_sample_elements_without_legacy_environment_input() -> None:
    app = RadarPdNovaApp()
    template = app.layout.html

    assert "Sample elements" in template
    assert "Sample can / environment" not in template
    assert 'v-model="environment_elements"' not in template
    assert "Ignored regions" in template


def test_candidate_library_upload_actions_wrap_without_styling_button_internals() -> None:
    app = RadarPdNovaApp()
    template = app.layout.html
    css = app._css()

    assert template.count('class="radar-library-source-action"') == 2
    assert ".radar-library-source-card > span" in css
    assert ".radar-library-source-card span {" not in css
    assert ".radar-library-source-action .v-btn__content" in css
    assert "white-space: normal" in css
    assert "overflow-wrap: anywhere" in css


def test_powgen_configuration_copy_keeps_run_inputs_independent() -> None:
    app = RadarPdNovaApp()
    template = app.layout.html

    assert "Applies analysis settings only." in template
    assert "POWGEN takes scans, its instrument profile" in template
    assert "candidate library, and optional main phase from this panel" in template


def test_galaxy_remote_browser_remains_available_for_restored_legacy_runs() -> None:
    app = RadarPdNovaApp()
    template = app.layout.html

    assert all(option["value"] != "galaxy_remote" for option in app.server.state.source_options)
    for text in (
        "Discover SNS sources",
        "Authorized file source",
        "Current remote folder",
    ):
        assert text in template
    handler_messages = " ".join(
        str(constant) for constant in app.discover_remote_sources.__func__.__code__.co_consts
    )
    assert "Selected files are imported into this History before RADAR-PD starts." in handler_messages
    for model in ("remote_data_uri", "remote_instrument_uri", "remote_main_cif_uri"):
        assert model in template
