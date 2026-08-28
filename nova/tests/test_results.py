import json
import zipfile
from pathlib import Path

import numpy as np

from radar_pd_nova.results import (
    build_result_view,
    complete_experiment_space_groups,
    curate_rapid_rows,
    discover_plot_payloads,
    discover_tables,
    experiment_fit_diagnostics,
    experiment_fit_quality_figure,
    experiment_axis_options,
    experiment_phase_fraction_figure,
    experiment_phase_heatmap_figure,
    experiment_sample_identity,
    experiment_scan_summary,
    figure_for_payload,
    load_plot_with_fallback,
    phase_fraction_rows,
    read_plot_payload,
    total_elapsed_seconds,
)


def test_experiment_scan_summary_uses_best_refined_rapid_hypothesis() -> None:
    payload = {
        "status": "complete",
        "analysis_mode": "rapid",
        "run_name": "IPTS-38000_PG3_63764",
        "summary": {"live_run": {"timings": {"total_seconds": 123.4}}},
        "hypotheses": [
            {
                "gsas_rwp_rank": "2",
                "rwp": "18.0",
                "status": "ok",
                "weights_json": '{"Wrong (SG 1)": 100}',
            },
            {
                "gsas_rwp_rank": "1",
                "rwp": "11.25",
                "status": "ok",
                "weights_json": '{"YFeSi (SG 62)": 70, "Ga (SG 225)": 30}',
            },
        ],
        "warnings": ["review"],
    }

    summary = experiment_scan_summary(payload)

    assert summary["rwp"] == 11.25
    assert summary["elapsed_seconds"] == 123.4
    assert [row["phase"] for row in summary["phases"]] == ["YFeSi", "Ga"]
    assert summary["warning_count"] == 1


def test_experiment_figures_preserve_missing_phase_as_gap() -> None:
    scans = [
        {
            "run_number": 10,
            "rwp": 12.0,
            "elapsed_display": "2 min",
            "hypothesis": "A",
            "phases": [{"label": "A (SG 1)", "weight_percent": 100.0}],
        },
        {
            "run_number": 11,
            "rwp": 10.0,
            "elapsed_display": "2 min",
            "hypothesis": "B",
            "phases": [{"label": "B (SG 2)", "weight_percent": 100.0}],
        },
    ]

    phase_figure = experiment_phase_fraction_figure(scans)
    quality_figure = experiment_fit_quality_figure(scans)

    assert list(phase_figure.data[0].y) in ([100.0, None], [None, 100.0])
    assert list(quality_figure.data[0].x) == ["10", "11"]
    assert quality_figure.layout.xaxis.type == "category"


def test_experiment_figures_use_metadata_axis_and_limit_default_phase_traces() -> None:
    scans = [
        {
            "run_number": 10 + scan_index,
            "rwp": 10.0 + scan_index,
            "metadata": {
                "start_time": f"2026-08-21T0{scan_index}:00:00-04:00",
                "temperature": {"value": 300.0 + 50 * scan_index, "unit": "K"},
            },
            "phases": [
                {"label": f"Phase {phase_index} (SG 1)", "weight_percent": 20.0}
                for phase_index in range(7)
            ],
        }
        for scan_index in range(2)
    ]

    options = experiment_axis_options(scans)
    phase_figure = experiment_phase_fraction_figure(scans, x_key="temperature")
    quality_figure = experiment_fit_quality_figure(scans, x_key="start_time")

    assert {row["value"] for row in options} >= {"run_number", "start_time", "temperature"}
    assert len(phase_figure.data) == 5
    assert list(phase_figure.data[0].x) == [300.0, 350.0]
    assert phase_figure.layout.xaxis.title.text == "Sample temperature (K)"
    assert phase_figure.data[0].mode == "markers"
    assert quality_figure.layout.xaxis.type == "date"
    assert quality_figure.layout.xaxis.title.text == "Acquisition time (Eastern)"
    assert list(quality_figure.data[0].x) == [
        "2026-08-21T00:00:00-04:00",
        "2026-08-21T01:00:00-04:00",
    ]


def test_experiment_temperature_axis_does_not_invent_missing_units() -> None:
    scans = [
        {
            "run_number": 10,
            "metadata": {"temperature": {"value": 25.0}},
            "phases": [{"label": "Phase (SG 1)", "weight_percent": 100.0}],
        }
    ]

    options = experiment_axis_options(scans)
    figure = experiment_phase_fraction_figure(scans, x_key="temperature")

    temperature_option = next(row for row in options if row["value"] == "temperature")
    assert temperature_option["title"] == "Sample temperature (unit not reported)"
    assert figure.layout.xaxis.title.text == "Sample temperature (unit not reported)"


def test_experiment_sample_identity_prefers_sample_id_and_labels_fallbacks() -> None:
    identified = experiment_sample_identity(
        {
            "metadata": {
                "sample_id": "118950",
                "sample_name": "Ga flux",
                "sample_formula": "YFeSiGa",
            }
        }
    )
    named = experiment_sample_identity({"metadata": {"sample_name": "PAC blank"}})
    unknown = experiment_sample_identity({})

    assert identified == {
        "key": "id:118950",
        "label": "Sample 118950 - Ga flux | YFeSiGa",
        "id": "118950",
    }
    assert named["key"] == "name:PAC blank"
    assert unknown["key"] == "unassigned"


def test_phase_trend_breaks_chronological_lines_between_samples() -> None:
    scans = [
        {
            "run_number": 10,
            "metadata": {"sample_id": "A"},
            "phases": [{"label": "Phase (SG 1)", "weight_percent": 40.0}],
        },
        {
            "run_number": 11,
            "metadata": {"sample_id": "B"},
            "phases": [{"label": "Phase (SG 1)", "weight_percent": 60.0}],
        },
    ]

    figure = experiment_phase_fraction_figure(scans)

    assert figure.data[0].mode == "lines+markers"
    assert list(figure.data[0].x) == ["10", None, "11"]
    assert list(figure.data[0].y) == [40.0, None, 60.0]


def test_experiment_heatmap_preserves_missing_phases_and_orders_by_abundance() -> None:
    scans = [
        {
            "run_number": 10,
            "phases": [
                {"label": "Minor (SG 2)", "weight_percent": 8.0},
                {"label": "Major (SG 1)", "weight_percent": 92.0},
            ],
        },
        {
            "run_number": 11,
            "phases": [{"label": "Major (SG 1)", "weight_percent": 100.0}],
        },
    ]

    figure = experiment_phase_heatmap_figure(scans)

    assert list(figure.data[0].y) == ["Major (SG 1)", "Minor (SG 2)"]
    assert list(figure.data[0].z[0]) == [92.0, 100.0]
    assert list(figure.data[0].z[1]) == [8.0, None]


def test_experiment_fit_diagnostics_uses_relative_baseline_and_flags_outlier() -> None:
    scans = [
        {"run_id": "PG3_1", "run_number": 1, "rwp": 10.0},
        {"run_id": "PG3_2", "run_number": 2, "rwp": 10.2},
        {"run_id": "PG3_3", "run_number": 3, "rwp": 9.8},
        {"run_id": "PG3_4", "run_number": 4, "rwp": 10.1},
        {"run_id": "PG3_5", "run_number": 5, "rwp": 25.0},
    ]

    diagnostics = experiment_fit_diagnostics(scans)
    figure = experiment_fit_quality_figure(scans)

    assert diagnostics["PG3_1"]["label"] == "Within experiment trend"
    assert diagnostics["PG3_5"]["label"] == "Rwp outlier"
    assert len(figure.layout.shapes) == 1


def test_discovers_and_renders_component_results(tmp_path: Path) -> None:
    (tmp_path / "ranking.csv").write_text("rank,hypothesis\n1,Cu + Cu2S\n", encoding="utf-8")
    payload = {
        "plot_kind": "rapid_component_fit_v1",
        "q": [1.0, 2.0, 3.0],
        "target": [0.0, 1.0, 0.0],
        "total_fit": [0.0, 0.8, 0.0],
        "residual": [0.0, 0.2, 0.0],
        "background": [0.0, 0.0, 0.0],
        "components": [{"label": "Cu (SG 225)", "scaled": [0.0, 0.8, 0.0]}],
    }
    plot_path = tmp_path / "fit.plotdata.json"
    plot_path.write_text(json.dumps(payload), encoding="utf-8")
    assert discover_tables(tmp_path)[0]["name"] == "Ranking"
    assert discover_plot_payloads(tmp_path)[0]["kind"] == "rapid_component_fit_v1"
    figure = figure_for_payload(payload)
    assert [trace.name for trace in figure.data] == ["Measured", "Background", "Cu (SG 225)", "Total hypothesis fit", "Difference"]


def test_loads_companion_npz_arrays_for_gsas_plot(tmp_path: Path) -> None:
    plot_path = tmp_path / "accepted.png.plotdata.json"
    arrays_path = tmp_path / "accepted.png.plotdata.npz"
    plot_path.write_text(
        json.dumps(
            {
                "plot_kind": "gsas_fit_with_ticks_v1",
                "arrays_npz": arrays_path.name,
                "phase_order": ["main"],
                "phase_ticks": {"main": [20.0]},
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        arrays_path,
        x=np.asarray([10.0, 20.0, 30.0]),
        yobs=np.asarray([1.0, 2.0, 1.0]),
        ycalc=np.asarray([1.0, 1.8, 1.0]),
        resid=np.asarray([0.0, 0.2, 0.0]),
    )

    payload = read_plot_payload(plot_path)
    figure = figure_for_payload(payload)

    assert payload["arrays"]["yobs"] == [1.0, 2.0, 1.0]
    assert [trace.name for trace in figure.data[:3]] == ["Observed", "Calculated", "Difference"]
    assert list(figure.data[0].y) == [1.0, 2.0, 1.0]


def test_discover_plot_payloads_excludes_duplicate_missing_its_arrays(tmp_path: Path) -> None:
    """Galaxy's per-output collection downloads can duplicate a plot's JSON

    under an unrelated filename, without the paired .npz archive that only
    the results-archive extraction keeps alongside it. Such a duplicate must
    never be offered, even when it sorts before the complete copy.
    """

    good_dir = tmp_path / "rapid_results" / "live_run" / "gsas" / "live_rank512_01_rank64_39"
    bad_dir = tmp_path / "diagnostics"
    good_dir.mkdir(parents=True)
    bad_dir.mkdir(parents=True)

    good_json = good_dir / "curve.png.plotdata.json"
    good_json.write_text(
        json.dumps({"plot_kind": "gsas_fit_with_ticks_v1", "arrays_npz": "curve.png.plotdata.npz"}),
        encoding="utf-8",
    )
    np.savez_compressed(
        good_dir / "curve.png.plotdata.npz",
        x=np.asarray([10.0, 20.0]),
        yobs=np.asarray([1.0, 2.0]),
        ycalc=np.asarray([1.0, 1.8]),
        resid=np.asarray([0.0, 0.2]),
    )

    # "diagnostics" sorts before "rapid_results", so without the fix this
    # incomplete duplicate would be picked as the default plot.
    (bad_dir / "rapid_results__live_run__gsas__live_rank512_01_rank64_39__curve.png.plotdata.json").write_text(
        json.dumps({"plot_kind": "gsas_fit_with_ticks_v1", "arrays_npz": "curve.png.plotdata.npz"}),
        encoding="utf-8",
    )

    options = discover_plot_payloads(tmp_path)

    assert len(options) == 1
    assert Path(options[0]["path"]) == good_json


def test_discover_plot_payloads_keeps_distinct_valid_hypothesis_fits_with_the_same_basename(tmp_path: Path) -> None:
    first = tmp_path / "gsas" / "live_rank512_01_rank64_08" / "curve.png.plotdata.json"
    second = tmp_path / "gsas" / "live_rank512_02_rank64_12" / "curve.png.plotdata.json"
    _write_gsas_payload(first, rwp=8.0)
    _write_gsas_payload(second, rwp=9.0)

    options = discover_plot_payloads(tmp_path)

    assert len(options) == 2
    assert {Path(option["path"]).parent.name for option in options} == {
        "live_rank512_01_rank64_08",
        "live_rank512_02_rank64_12",
    }


def test_plot_index_does_not_load_npz_arrays_until_selection(tmp_path: Path, monkeypatch: object) -> None:
    plot_path = tmp_path / "rank01" / "curve.plotdata.json"
    plot_path.parent.mkdir()
    plot_path.write_text(
        json.dumps({"plot_kind": "gsas_fit_with_ticks_v1", "arrays_npz": "curve.npz", "rwp": 4.2}),
        encoding="utf-8",
    )
    (plot_path.parent / "curve.npz").write_bytes(b"present-but-not-read-during-index")

    def fail_if_loaded(*args: object, **kwargs: object) -> object:
        raise AssertionError("np.load must not run while indexing plot metadata")

    monkeypatch.setattr("radar_pd_nova.results.np.load", fail_if_loaded)  # type: ignore[attr-defined]

    options = discover_plot_payloads(tmp_path)

    assert len(options) == 1
    assert options[0]["rwp"] == 4.2


def test_selected_plot_falls_back_when_ranked_npz_is_corrupt(tmp_path: Path) -> None:
    broken = tmp_path / "rank01" / "curve.plotdata.json"
    broken.parent.mkdir()
    broken.write_text(
        json.dumps({"plot_kind": "gsas_fit_with_ticks_v1", "arrays_npz": "curve.npz", "rwp": 3.0}),
        encoding="utf-8",
    )
    (broken.parent / "curve.npz").write_bytes(b"not-an-npz")
    valid = tmp_path / "rank02" / "curve.plotdata.json"
    _write_gsas_payload(valid, rwp=4.0)
    options = discover_plot_payloads(tmp_path)

    selected = load_plot_with_fallback(options, str(broken))

    assert selected is not None
    path, payload, figure = selected
    assert path == str(valid)
    assert payload["rwp"] == 4.0
    assert [trace.name for trace in figure.data[:3]] == ["Observed", "Calculated", "Difference"]


def test_metadata_only_plot_is_excluded_and_falls_back_to_real_curves(tmp_path: Path) -> None:
    empty = tmp_path / "accepted_model.plotdata.json"
    empty.write_text(
        json.dumps({"plot_kind": "gsas_fit_with_ticks_v1", "rwp": 3.0}),
        encoding="utf-8",
    )
    valid = tmp_path / "final_refinement.plotdata.json"
    _write_gsas_payload(valid, rwp=4.0)

    options = discover_plot_payloads(tmp_path)
    selected = load_plot_with_fallback(options, str(empty))

    assert [Path(option["path"]) for option in options] == [valid]
    assert selected is not None
    assert selected[0] == str(valid)


def test_plot_uses_published_static_image_when_interactive_arrays_are_missing(tmp_path: Path) -> None:
    image = tmp_path / "accepted_model.png"
    image.write_bytes(b"published-refinement-image")
    sidecar = tmp_path / "accepted_model.png.plotdata.json"
    sidecar.write_text(
        json.dumps(
            {
                "plot_kind": "gsas_fit_with_ticks_v1",
                "source_plot": image.name,
                "arrays_npz": "missing.plotdata.npz",
                "rwp": 7.5,
            }
        ),
        encoding="utf-8",
    )

    options = discover_plot_payloads(tmp_path)
    selected = load_plot_with_fallback(options, str(sidecar))

    assert len(options) == 1
    assert selected is not None
    assert selected[0] == str(sidecar)
    assert selected[2].layout.title.text == "Published refinement fit / Rwp 7.50%"
    assert len(selected[2].layout.images) == 1


def test_result_archive_uses_orphan_accepted_fit_image(tmp_path: Path) -> None:
    """Older result archives can contain an accepted PNG without a plot sidecar."""

    source = tmp_path / "source" / "gsas_projects" / "seq_pass1_accepted_model.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c63600000020001e221bc330000000049454e44ae426082"
        )
    )
    archive = tmp_path / "results.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(source, "gsas_projects/seq_pass1_accepted_model.png")
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(extracted)

    plots = discover_plot_payloads(extracted)
    assert [plot["name"] for plot in plots] == ["Pass 1 accepted model"]

    selected = load_plot_with_fallback(plots, plots[0]["path"])
    assert selected is not None
    assert selected[0].endswith("seq_pass1_accepted_model.png")
    assert selected[2].layout.title.text == "Published refinement fit"
    assert len(selected[2].layout.images) == 1


def test_result_archive_uses_galaxy_published_accepted_fit_name(tmp_path: Path) -> None:
    """NDIP renames accepted model plots when publishing the Galaxy collection."""

    image = tmp_path / "ndip" / "plots" / "Accepted_fit_after_pass_1.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c63600000020001e221bc330000000049454e44ae426082"
        )
    )

    plots = discover_plot_payloads(tmp_path)
    assert [plot["name"] for plot in plots] == ["Pass 1 accepted model"]

    selected = load_plot_with_fallback(plots, plots[0]["path"])
    assert selected is not None
    assert selected[0].endswith("Accepted_fit_after_pass_1.png")
    assert len(selected[2].layout.images) == 1


def test_galaxy_flattened_plot_companions_keep_interactive_curves(tmp_path: Path) -> None:
    sidecar = tmp_path / "Technical__Plots__main_phase_fit.png.plotdata.json"
    arrays = tmp_path / "Technical__Plots__main_phase_fit.png.plotdata.npz"
    image = tmp_path / "Technical__Plots__main_phase_fit.png"
    sidecar.write_text(
        json.dumps(
            {
                "plot_kind": "gsas_fit_with_ticks_v1",
                "source_plot": "main_phase_fit.png",
                "arrays_npz": "main_phase_fit.png.plotdata.npz",
                "rwp": 6.25,
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        arrays,
        x=np.array([1.0, 2.0, 3.0]),
        yobs=np.array([10.0, 20.0, 30.0]),
        ycalc=np.array([9.0, 19.0, 29.0]),
        resid=np.array([1.0, 1.0, 1.0]),
    )
    image.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c63600000020001e221bc330000000049454e44ae426082"
        )
    )

    plots = discover_plot_payloads(tmp_path)
    selected = load_plot_with_fallback(plots, str(sidecar))

    assert len(plots) == 1
    assert selected is not None
    assert selected[0] == str(sidecar)
    assert [trace.name for trace in selected[2].data[:3]] == ["Observed", "Calculated", "Difference"]
    assert len(selected[2].layout.images) == 0


def test_result_view_deduplicates_manifest_png_represented_by_interactive_sidecar(tmp_path: Path) -> None:
    image = tmp_path / "ndip" / "plots" / "Rapid_final_fit.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"published-fit")
    sidecar = Path(str(image) + ".plotdata.json")
    arrays = Path(str(image) + ".plotdata_arrays.npz")
    sidecar.write_text(
        json.dumps(
            {
                "plot_kind": "gsas_fit_with_ticks_v1",
                "source_plot": image.name,
                "arrays_npz": arrays.name,
                "rwp": 6.25,
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        arrays,
        x=np.array([1.0, 2.0]),
        yobs=np.array([10.0, 20.0]),
        ycalc=np.array([9.0, 19.0]),
        resid=np.array([1.0, 1.0]),
    )
    result = {
        "$schema": "radar-pd-result/v1",
        "analysis_mode": "rapid",
        "status": "complete",
        "phases": [],
        "hypotheses": [],
        "gpx_projects": [],
        "artifacts": {
            "plots": [
                {
                    "name": image.name,
                    "path": f"plots/{image.name}",
                    "source_path": "rapid_results/live_run/curve.png",
                }
            ]
        },
    }

    view = build_result_view(result, tmp_path)

    assert len(view.plots) == 1
    assert Path(view.plots[0].path) == sidecar
    selected = load_plot_with_fallback(view.plots, view.primary_plot_path)
    assert selected is not None
    assert [trace.name for trace in selected[2].data[:3]] == ["Observed", "Calculated", "Difference"]


def test_result_view_resolves_custom_named_plot_from_normalized_manifest(tmp_path: Path) -> None:
    image = tmp_path / "ndip" / "plots" / "custom_database_output_00017.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c63600000020001e221bc330000000049454e44ae426082"
        )
    )
    result = {
        "$schema": "radar-pd-result/v1",
        "analysis_mode": "full",
        "status": "complete",
        "phases": [],
        "hypotheses": [],
        "gpx_projects": [],
        "artifacts": {
            "plots": [
                {
                    "name": image.name,
                    "path": f"plots/{image.name}",
                    "source_path": "unexpected/location/custom_database_output_00017.png",
                }
            ]
        },
    }

    view = build_result_view(result, tmp_path)

    assert len(view.plots) == 1
    assert Path(view.primary_plot_path) == image
    selected = load_plot_with_fallback(view.plots, view.primary_plot_path)
    assert selected is not None
    assert len(selected[2].layout.images) == 1


def test_empty_plot_has_explicit_unavailable_state() -> None:
    figure = figure_for_payload({"plot_kind": "gsas_fit_with_ticks_v1", "rwp": 3.0})

    assert not figure.data
    assert figure.layout.title.text == "Refinement fit unavailable"
    assert "No interactive refinement-fit curves" in figure.layout.annotations[0].text


def test_symbol_only_space_groups_are_completed_from_experiment_peers() -> None:
    scans = [
        {
            "phases": [
                {
                    "phase": "AlFe2V",
                    "space_group": "F m -3 m",
                    "label": "AlFe2V (SG F m -3 m)",
                }
            ]
        },
        {
            "phases": [
                {
                    "phase": "AlVFe2",
                    "space_group": "Fm-3m (225)",
                    "label": "AlVFe2 (SG Fm-3m (225))",
                }
            ]
        },
    ]

    complete_experiment_space_groups(scans)

    assert scans[0]["phases"][0]["space_group"] == "Fm-3m (225)"
    assert scans[0]["phases"][0]["label"] == "AlFe2V (SG Fm-3m (225))"


def test_result_mode_document_is_authoritative_and_warns_on_mismatch(tmp_path: Path) -> None:
    view = build_result_view(
        {"$schema": "radar-pd-result/v1", "analysis_mode": "rapid", "status": "complete"},
        tmp_path,
        submitted_mode="full",
    )

    assert view.mode == "rapid"
    assert view.warnings[0].startswith("Mode mismatch: NOVA submitted Full")


def test_run112_compact_archive_recognizes_five_refinements_and_real_traces(tmp_path: Path) -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "run112_archive_compact.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    for item in fixture["plots"]:
        path = tmp_path / item["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "plot_kind": "gsas_fit_with_ticks_v1",
            "rwp": item["rwp"],
            "phase_order": item["phase_order"],
            "phase_labels": item["phase_labels"],
            "phase_ticks": item["phase_ticks"],
            "arrays": item["arrays"],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
    duplicate = tmp_path / "ndip" / "diagnostics" / "run112_rank01_duplicate.plotdata.json"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_text(
        json.dumps({"plot_kind": "gsas_fit_with_ticks_v1", "arrays_npz": "missing.npz", "rwp": 15.28}),
        encoding="utf-8",
    )

    view = build_result_view(fixture["summary"], tmp_path)

    refinements = [plot for plot in view.plots if plot.stage == "final_refinement"]
    assert len(refinements) == 5
    assert "live_rank512_01_rank64_36" in view.primary_plot_path
    for descriptor in refinements:
        payload = read_plot_payload(descriptor.path)
        figure = figure_for_payload(payload)
        assert [trace.name for trace in figure.data[:3]] == ["Observed", "Calculated", "Difference"]
        assert all(len(trace.x) == 5 for trace in figure.data[:3])
    assert fixture["source"]["source_points_per_curve"] == 2439


def test_phase_fraction_normalization() -> None:
    rows = phase_fraction_rows(
        {
            "phase_fractions": [
                {"formula": "Fe", "space_group": 225, "weight_percent": 91.2},
                {"formula": "Cr", "space_group": 229, "weight_percent": 0.0},
            ]
        }
    )
    assert rows == [
        {"phase": "Fe", "space_group": 225, "weight_percent": 91.2},
        {"phase": "Cr", "space_group": 229, "weight_percent": 0.0},
    ]


def test_phase_fraction_normalization_compacts_formula_and_space_group_not_names() -> None:
    rows = phase_fraction_rows(
        {
            "phases": [
                {"phase": "Al Fe2 V", "space_group": "F m -3 m (225)", "weight_percent": 72.6},
                {"phase": "alpha iron", "space_group": "I m -3 m", "weight_percent": 27.4},
                {"phase": "Al0.5 Fe V0.5 (SG P m -3 m (221))", "weight_percent": 0.0},
            ]
        }
    )

    assert rows == [
        {"phase": "AlFe2V", "space_group": "Fm-3m (225)", "weight_percent": 72.6},
        {"phase": "alpha iron", "space_group": "Im-3m", "weight_percent": 27.4},
        {"phase": "Al0.5FeV0.5", "space_group": "Pm-3m (221)", "weight_percent": 0.0},
    ]


def test_gsas_phase_tick_labels_compact_formula_and_space_group() -> None:
    figure = figure_for_payload(
        {
            "plot_kind": "gsas_fit_with_ticks_v1",
            "phase_order": ["catalog_phase"],
            "phase_labels": {"catalog_phase": "Al Fe2 V (SG F m -3 m (225))"},
            "phase_ticks": {"catalog_phase": [1.5]},
            "arrays": {
                "x": [1.0, 2.0],
                "yobs": [1.0, 2.0],
                "ycalc": [1.0, 2.0],
                "resid": [0.0, 0.0],
            },
        }
    )

    assert figure.layout.yaxis3.ticktext == ("AlFe2V (SG Fm-3m (225))",)


def test_gsas_phase_tick_labels_parse_pipeline_dash_separator() -> None:
    figure = figure_for_payload(
        {
            "plot_kind": "gsas_fit_with_ticks_v1",
            "phase_order": ["catalog_phase"],
            "phase_labels": {"catalog_phase": "Al Fe2 V \u2014 F m -3 m (225)"},
            "phase_ticks": {"catalog_phase": [1.5]},
            "arrays": {
                "x": [1.0, 2.0],
                "yobs": [1.0, 2.0],
                "ycalc": [1.0, 2.0],
                "resid": [0.0, 0.0],
            },
        }
    )

    assert figure.layout.yaxis3.ticktext == ("AlFe2V (SG Fm-3m (225))",)
    assert figure.data[3].name == "AlFe2V (SG Fm-3m (225))"


def test_renders_real_rapid_phase_components_and_zero_contribution() -> None:
    payload = {
        "plot_kind": "rapid_refined_pattern_match",
        "q": [1.0, 2.0, 3.0],
        "target": [0.0, 1.0, 0.0],
        "total_fit": [0.0, 0.9, 0.0],
        "residual": [0.0, 0.1, 0.0],
        "background": [0.0, 0.0, 0.0],
        "phases": [
            {"label": "Cu (SG 225)", "component": [0.0, 0.9, 0.0], "relative_scale": 1.0},
            {"label": "Cu2S (SG 14)", "component": [0.0, 0.0, 0.0], "contribution": 0.0},
        ],
    }

    figure = figure_for_payload(payload)

    assert [trace.name for trace in figure.data] == [
        "Measured",
        "Background",
        "Cu (SG 225) (100.0%)",
        "Cu2S (SG 14) (0.0%)",
        "Total hypothesis fit",
        "Difference",
    ]
    assert list(figure.data[2].y) == [0.0, 0.9, 0.0]


def test_rapid_final_hypothesis_fractions_are_normalized() -> None:
    rows = phase_fraction_rows(
        {
            "analysis_mode": "rapid",
            "phases": [],
            "hypothesis_stage": "final_refinement",
            "hypotheses": [
                {
                    "gsas_rwp_rank": "1",
                    "status": "ok",
                    "weights_json": '{"Cu (SG 225)": 88.5, "Cu2S (SG 14)": 11.5, "CuO (SG 15)": 0.0}',
                }
            ],
        }
    )

    assert rows == [
        {"phase": "Cu", "space_group": "225", "weight_percent": 88.5},
        {"phase": "Cu2S", "space_group": "14", "weight_percent": 11.5},
        {"phase": "CuO", "space_group": "15", "weight_percent": 0.0},
    ]


def test_total_elapsed_seconds_reads_nested_rapid_timing_and_zero() -> None:
    assert total_elapsed_seconds({"summary": {"live_run": {"timings": {"total_seconds": 245.5}}}}) == 245.5
    assert total_elapsed_seconds({"total_seconds": 0.0}) == 0.0


def test_full_result_falls_back_to_manifest_timing_and_presents_known_main_phase(tmp_path: Path) -> None:
    result = {
        "$schema": "radar-pd-result/v1",
        "created_utc": "2026-08-15T03:42:00Z",
        "analysis_mode": "full",
        "status": "complete",
        "hypothesis_stage": "none",
        "phases": [
            {
                "phase_id": "main",
                "compound_name": "main",
                "space_group": "P -4 21 m (113)",
                "weight_fraction_pct": "90.37",
                "is_main": "1",
            },
            {
                "phase_id": "cod-9017775",
                "compound_name": "Al",
                "space_group": "Im-3m (229)",
                "weight_fraction_pct": "9.63",
                "is_main": "0",
            },
        ],
        "provenance": {"source_manifest": {"start_time": "2026-08-15T03:41:37.8"}},
    }

    view = build_result_view(result, tmp_path)

    assert view.phases[0]["phase"] == "Known main phase"
    assert view.phases[1]["phase"] == "Al"
    assert view.metrics[4] == {"label": "Total time", "value": "22.2 s"}
    assert view.metrics[5] == {"label": "Result stage", "value": "Final refinement"}


def test_gsas_plot_renders_ranked_strongest_bragg_ticks() -> None:
    payload = {
        "plot_kind": "gsas_fit_with_ticks_v1",
        "arrays": {
            "x": [10.0, 20.0, 30.0],
            "yobs": [1.0, 2.0, 1.0],
            "ycalc": [1.0, 1.8, 1.0],
            "resid": [0.0, 0.2, 0.0],
        },
        "phase_order": ["phase_1"],
        "phase_labels": {"phase_1": "Fe (SG 225)"},
        "phase_ticks": {"phase_1": [12.0, 20.0, 28.0]},
        "phase_major_ticks": {"phase_1": [20.0]},
        "phase_major_tick_details": {
            "phase_1": [{"x": 20.0, "rank": 1, "hkl": "1 1 0", "relative_strength": 0.0}]
        },
        "rwp": 0.0,
    }

    figure = figure_for_payload(payload)

    assert [trace.name for trace in figure.data] == [
        "Observed",
        "Calculated",
        "Difference",
        "Fe (SG 225)",
        "Key peaks: Fe (SG 225)",
    ]
    strongest = figure.data[-1]
    assert list(strongest.x) == [20.0]
    assert strongest.marker.line.width == 3
    assert "Relative strength=0.000" in strongest.text[0]
    assert figure.layout.title.text.endswith("Rwp 0.00%")
    assert figure.data[3].showlegend is False


def test_gsas_plot_hides_internal_custom_catalog_ids() -> None:
    phase_id = "user_00011_yourcustomfilename_collcode258024_bbac164417"
    payload = {
        "plot_kind": "gsas_fit_with_ticks_v1",
        "arrays": {
            "x": [10.0, 20.0],
            "yobs": [1.0, 2.0],
            "ycalc": [1.0, 1.8],
            "resid": [0.0, 0.2],
        },
        "phase_order": [phase_id],
        "phase_labels": {phase_id: f"{phase_id} (Al0.5V0.5Fe1 - Pm-3m (221))"},
        "phase_ticks": {phase_id: [12.0]},
    }

    figure = figure_for_payload(payload)

    assert figure.layout.yaxis3.ticktext == ("Al0.5V0.5Fe1 (SG 221)",)
    assert figure.data[-1].name == "Al0.5V0.5Fe1 - Pm-3m (221)"
    assert figure.data[-1].showlegend is False
    assert figure.layout.margin.l >= 180


def _write_gsas_payload(path: Path, *, rwp: float, phase: str = "TbSSL") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "plot_kind": "gsas_fit_with_ticks_v1",
                "rwp": rwp,
                "arrays": {
                    "x": [1.0, 2.0, 3.0],
                    "yobs": [2.0, 3.0, 2.0],
                    "ycalc": [2.0, 2.8, 2.0],
                    "resid": [0.0, 0.2, 0.0],
                },
                "phase_order": [phase],
                "phase_ticks": {phase: [2.0]},
            }
        ),
        encoding="utf-8",
    )


def test_builds_curated_rapid_result_and_selects_ranked_refinement(tmp_path: Path) -> None:
    best = tmp_path / "rapid_results" / "live_run" / "gsas" / "live_rank512_01_rank64_39" / "curve.png.plotdata.json"
    other = tmp_path / "rapid_results" / "live_run" / "gsas" / "live_rank512_02_rank64_12" / "curve.png.plotdata.json"
    _write_gsas_payload(best, rwp=15.283)
    _write_gsas_payload(other, rwp=22.0)
    (tmp_path / "results.zip").write_bytes(b"zip")
    checkpoint = tmp_path / "gpx" / "rapid_final.gpx"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"gpx")
    nudge = tmp_path / "nudge_results.csv"
    nudge.write_text("formula,space_group,best_score,distance_from_start,a,b,c,alpha,beta,gamma,seconds\nTbSSL,14,0.9,0.01,4,5,6,90,90,90,2\n", encoding="utf-8")
    pattern = tmp_path / "reranked_512_after_radar_nudge.csv"
    pattern.write_text("rank512,rank64,formulas,space_groups,score512,r2_512,sse512,peak_support_summary\n1,39,TbSSL,14,0.95,0.94,0.06,3+1/5\n", encoding="utf-8")
    result = {
        "$schema": "radar-pd-result/v1",
        "analysis_mode": "rapid",
        "status": "complete",
        "hypothesis_stage": "final_refinement",
        "summary": {"live_run": {"timings": {"total_seconds": 61.2}}},
        "phases": [],
        "hypotheses": [
            {
                "gsas_rwp_rank": "1",
                "rank512": "1",
                "rank64": "39",
                "formulas": "TbSSL|Al",
                "space_groups": "14|225",
                "status": "ok",
                "rwp": "15.283",
                "weights_json": '{"TbSSL (SG 14)": 91.0, "Al (SG 225)": 9.0}',
                "curve_png": "/run/rapid_results/live_run/gsas/live_rank512_01_rank64_39/curve.png",
            },
            {
                "gsas_rwp_rank": "2",
                "rank512": "2",
                "formulas": "TbSSL|Be",
                "status": "ok",
                "rwp": "22.0",
                "curve_png": "/run/rapid_results/live_run/gsas/live_rank512_02_rank64_12/curve.png",
            },
        ],
        "gpx_projects": [
            {"label": "rapid_final", "path": "gpx/rapid_final.gpx", "stage": "hypothesis_refinement", "status": "accepted"}
        ],
        "warnings": [],
        "errors": [],
    }

    view = build_result_view(result, tmp_path)

    assert Path(view.primary_plot_path) == best
    assert view.metrics[2] == {"label": "Best Rwp", "value": "15.283%"}
    assert view.phase_total == "100.00% total"
    assert view.phases[0]["weight_display"] == "91.00%"
    assert view.top_refinements[0]["hypothesis"] == "TbSSL (SG 14) + Al (SG 225)"
    assert view.rapid_stages["pattern_scoring"][0]["key_peak_support"] == "3 supported, 1 weak, 1 missing/review"
    assert view.checkpoints[0].handoff_available is True
    assert len(view.plots) == 2
    assert all("/tmp/" not in plot.name for plot in view.plots)


def test_builds_full_result_and_prioritizes_latest_accepted_fit(tmp_path: Path) -> None:
    accepted = tmp_path / "gsas_projects" / "seq_pass2_accepted_model.png.plotdata.json"
    trial = tmp_path / "gsas_projects" / "seq_pass2_trial_blend.png.plotdata.json"
    main = tmp_path / "main_phase_fit.png.plotdata.json"
    _write_gsas_payload(accepted, rwp=8.5, phase="Fe")
    _write_gsas_payload(trial, rwp=10.2, phase="Fe")
    _write_gsas_payload(main, rwp=12.0, phase="Fe")
    result = {
        "$schema": "radar-pd-result/v1",
        "analysis_mode": "full",
        "status": "complete",
        "hypothesis_stage": "phase_refinement",
        "summary": {"final": {"final_rwp": 8.5}, "elapsed_seconds": 120.0},
        "phases": [{"formula": "Fe", "space_group": 225, "weight_percent": 100.0}],
        "hypotheses": [],
        "gpx_projects": [
            {"label": "accepted model", "path": "accepted.gpx", "stage": "full_pipeline_pass", "status": "accepted"}
        ],
    }

    view = build_result_view(result, tmp_path)

    assert Path(view.primary_plot_path) == accepted
    assert view.mode == "full"
    assert view.metrics[0]["value"] == "Full RADAR-PD"
    assert view.phases[0]["phase"] == "Fe"
    assert {plot.name for plot in view.plots} == {
        "Main-phase refinement fit",
        "Best refinement - Pass 2 accepted model / Rwp 8.50%",
        "Pass 2 trial model / Rwp 10.20%",
    }
    assert view.full_progression == [{"stage": "Full pipeline pass", "status": "Accepted"}]
    assert view.full_models == [
        {
            "model": "Accepted model",
            "stage": "Full pipeline pass",
            "rwp": "-",
            "decision": "Accepted",
            "note": "-",
        }
    ]


def test_checkpoint_prefers_galaxy_collection_alias_over_technical_copy(tmp_path: Path) -> None:
    technical = tmp_path / "Technical" / "GSAS_Projects" / "seq_final_main_polished.gpx"
    published = tmp_path / "gpx" / "02_Main_phase_anchor.gpx"
    technical.parent.mkdir(parents=True)
    published.parent.mkdir(parents=True)
    technical.write_bytes(b"technical")
    published.write_bytes(b"published")
    result = {
        "$schema": "radar-pd-result/v1",
        "analysis_mode": "full",
        "status": "complete",
        "phases": [],
        "hypotheses": [],
        "gpx_projects": [
            {
                "label": "seq_final_main_polished",
                "path": "Technical/GSAS_Projects/seq_final_main_polished.gpx",
                "source_path": "Technical/GSAS_Projects/seq_final_main_polished.gpx",
                "collection_path": "gpx/02_Main_phase_anchor.gpx",
                "collection_name": "02_Main_phase_anchor.gpx",
                "stage": "hypothesis_refinement",
                "status": "accepted",
            }
        ],
    }

    view = build_result_view(result, tmp_path)

    assert Path(view.checkpoints[0].path) == published
    assert view.checkpoints[0].galaxy_element_name == "02_Main_phase_anchor"
    assert view.checkpoints[0].handoff_available is True
    assert view.checkpoints[0].local_available is True


def test_checkpoint_can_launch_from_galaxy_without_local_gpx_copy(tmp_path: Path) -> None:
    result = {
        "$schema": "radar-pd-result/v1",
        "analysis_mode": "full",
        "status": "complete",
        "phases": [],
        "hypotheses": [],
        "gpx_projects": [
            {
                "label": "seq_final_main_polished",
                "path": "Technical/GSAS_Projects/seq_final_main_polished.gpx",
                "collection_path": "gpx/02_Main_phase_anchor.gpx",
                "collection_name": "02_Main_phase_anchor.gpx",
                "stage": "final_refinement",
                "status": "accepted",
            }
        ],
    }

    view = build_result_view(result, tmp_path)

    assert view.checkpoints[0].id == "checkpoint-0"
    assert view.checkpoints[0].path == ""
    assert view.checkpoints[0].handoff_available is True
    assert view.checkpoints[0].local_available is False
    assert view.checkpoints[0].name == "Seq final main polished (GPX)"


def test_pattern_only_result_warns_that_coefficients_are_not_phase_fractions(tmp_path: Path) -> None:
    result = {
        "analysis_mode": "rapid",
        "status": "complete",
        "hypothesis_stage": "pattern_scoring",
        "phases": [],
        "hypotheses": [{"rank512": 1, "formulas": "Cu|Cu2S", "phase_coefs512": "1.0|0.2"}],
    }

    view = build_result_view(result, tmp_path)

    assert "not quantitative phase fractions" in view.warnings[0]


def test_curated_rapid_rows_hide_internal_paths_and_json_fields() -> None:
    rows = curate_rapid_rows(
        "final_refinement",
        [
            {
                "gsas_rwp_rank": 1,
                "formulas": "Al Fe2 V|Cu2S",
                "space_groups": "F m -3 m (225)|P 21/c (14)",
                "rwp": 9.25,
                "weights_json": '{"Al Fe2 V": 80, "Cu2S": 20}',
                "cif_paths": "/internal/a.cif|/internal/b.cif",
                "stdout_tail": "technical",
                "status": "ok",
            }
        ],
    )

    assert list(rows[0]) == ["rank", "hypothesis", "rwp", "phase_fractions", "pattern_rank", "status", "time"]
    assert "/internal" not in str(rows[0])
    assert rows[0]["hypothesis"] == "AlFe2V (SG Fm-3m (225)) + Cu2S (SG P21/c (14))"
    assert rows[0]["phase_fractions"] == "AlFe2V: 80.0%; Cu2S: 20.0%"


def test_fixed_result_contract_fixtures_cover_rapid_full_partial_and_failure(tmp_path: Path) -> None:
    fixture_root = Path(__file__).parent / "fixtures" / "results"
    rapid = build_result_view(json.loads((fixture_root / "rapid_final.json").read_text(encoding="utf-8")), tmp_path)
    full = build_result_view(json.loads((fixture_root / "full_final.json").read_text(encoding="utf-8")), tmp_path)
    partial = build_result_view(
        json.loads((fixture_root / "rapid_pattern_only.json").read_text(encoding="utf-8")), tmp_path
    )
    failed = build_result_view(json.loads((fixture_root / "failed.json").read_text(encoding="utf-8")), tmp_path)

    assert rapid.phase_total == "100.00% total"
    assert full.phase_total == "100.00% total"
    assert [row["decision"] for row in full.full_models] == ["Accepted", "Rejected"]
    assert "not quantitative phase fractions" in partial.warnings[0]
    assert failed.status == "Failed"
    assert failed.warnings == ["No candidate model converged"]


def test_run112_archive_fixture_recovers_five_refinements_and_scientific_traces(tmp_path: Path) -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "run112_archive_compact.json").read_text(encoding="utf-8")
    )
    for plot in fixture["plots"]:
        plot_path = tmp_path / plot["relative_path"]
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path.write_text(
            json.dumps(
                {
                    "plot_kind": "gsas_fit_with_ticks_v1",
                    "rwp": plot["rwp"],
                    "phase_order": plot["phase_order"],
                    "phase_labels": plot["phase_labels"],
                    "phase_ticks": plot["phase_ticks"],
                    "arrays": plot["arrays"],
                }
            ),
            encoding="utf-8",
        )

    # Reproduce the incomplete duplicate that caused the pre-0.1.36 live failure.
    duplicate = tmp_path / "named_collection_copy" / "run112_best.plotdata.json"
    duplicate.parent.mkdir()
    duplicate.write_text(
        json.dumps(
            {
                "plot_kind": "gsas_fit_with_ticks_v1",
                "arrays_npz": "missing.plotdata.npz",
                "rwp": fixture["plots"][0]["rwp"],
            }
        ),
        encoding="utf-8",
    )

    plots = discover_plot_payloads(tmp_path)
    view = build_result_view(fixture["summary"], tmp_path)

    assert fixture["source"]["source_points_per_curve"] == 2439
    assert len(plots) == 5
    assert "rank512_01_rank64_36" in view.primary_plot_path
    assert view.metrics[2]["value"] == "15.283%"
    for plot in plots:
        loaded = load_plot_with_fallback([plot])
        assert loaded is not None
        names = {trace.name: trace for trace in loaded[2].data}
        assert len(names["Observed"].x) == 5
        assert len(names["Calculated"].x) == 5
        assert len(names["Difference"].x) == 5
        assert any(name not in {"Observed", "Calculated", "Difference"} for name in names)
