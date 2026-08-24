"""Structural regressions for the stateful NOVA interface tree."""

from radar_pd_nova.app import (
    RadarPdNovaApp,
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


def test_setup_panels_and_uploads_have_single_stable_instances() -> None:
    app = RadarPdNovaApp()
    template = app.layout.html

    assert app.server.state.setup_panels == [0, 2]

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
    for panel_value in range(11):
        assert template.count(f':value="{panel_value}"') >= 1

    # Vue's template compiler treats bare Promise as component scope. Using
    # Promise.all in the multi-CIF handler raises while the library panel is
    # first rendered and causes Vuetify to discard that panel body.
    assert "Promise.all" not in template
    assert "for (const file of files)" in template
    # Vuetify may update the v-model without forwarding the selected File[]
    # as the custom handler's $event. Both local source pickers therefore read
    # their bound models, matching the stable single-file upload contract.
    assert "Array.isArray(radar_cif_library_upload_browser_files)" in template
    assert "Array.isArray(radar_cif_library_upload_browser_archives)" in template
    assert "Array.isArray($event) ? $event" not in template
    assert "new window.TextEncoder()" in template
    # Loose CIFs are batched; ZIP archives use the proven direct
    # ArrayBuffer + filename contract so each archive completes inspection.
    assert template.count("new window.Uint8Array") == 2
    assert "new window.DataView" in template
    assert "[contents, file.name]" in template
    assert "new TextEncoder()" not in template
    assert "new Uint8Array" not in template
    assert "new DataView" not in template
    assert "ZIP archive(s) selected; inspecting CIF contents..." in template
    assert "CIF file(s) selected; reading contents..." in template

    for stable_key in (
        "radar-diffraction-upload-native",
        "radar-instrument-upload-native",
        "radar-main-cif-upload-native",
        "powgen-main-cif-upload-native",
    ):
        assert template.count(f":key=\"'{stable_key}'\"") == 1

    assert "From Server" not in template
    assert 'activator="parent"' not in template


def test_plotly_canvases_have_nonzero_layout_frames() -> None:
    app = RadarPdNovaApp()
    template = app.layout.html

    assert template.count('class="radar-plot-frame"') == 4
    assert ".radar-plot-frame { position: relative; width: 100%; height: 720px;" in app._css()
    assert "Plotly.Plots.resize" in template


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
    assert "Build and select library" in template
    assert "Save reusable configuration to History" in template
    assert "Open Result Explorer" in template
    assert "Send checkpoint to GSAS-II handoff" in template
    assert "Compare selected series" in template
    assert "Companion-tool activity" in template
    assert "Technical files" in template
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

    assert app.server.state.source_options == [
        {"title": "Upload from this computer", "value": "upload"},
        {"title": "Choose from Galaxy History", "value": "galaxy"},
        {"title": "Browse an SNS/HFIR experiment folder", "value": "ipts_browser"},
    ]
    assert "Where is the diffraction pattern?" in app.layout.html
    assert "Where is the GSAS-II instrument profile?" in app.layout.html
    assert "Do you have a known/main-phase CIF?" in app.layout.html


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
