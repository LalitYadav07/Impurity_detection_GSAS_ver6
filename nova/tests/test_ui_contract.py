"""Structural regressions for the stateful NOVA interface tree."""

from radar_pd_nova.app import RadarPdNovaApp


def test_setup_panels_and_uploads_have_single_stable_instances() -> None:
    app = RadarPdNovaApp()
    template = app.layout.html

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

    for stable_key in (
        "radar-diffraction-upload-native",
        "radar-instrument-upload-native",
        "radar-main-cif-upload-native",
    ):
        assert template.count(f":key=\"'{stable_key}'\"") == 1

    assert "From Server" not in template
    assert 'activator="parent"' not in template


def test_plotly_canvases_have_nonzero_layout_frames() -> None:
    app = RadarPdNovaApp()
    template = app.layout.html

    assert template.count('class="radar-plot-frame"') == 2
    assert ".radar-plot-frame { position: relative; width: 100%; height: 720px;" in app._css()
    assert "Plotly.Plots.resize" in template


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
