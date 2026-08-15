import json
from pathlib import Path

import numpy as np

from radar_pd_nova.results import (
    build_result_view,
    curate_rapid_rows,
    discover_plot_payloads,
    discover_tables,
    figure_for_payload,
    phase_fraction_rows,
    read_plot_payload,
    total_elapsed_seconds,
)


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
    main = tmp_path / "main_phase_fit.png.plotdata.json"
    _write_gsas_payload(accepted, rwp=8.5, phase="Fe")
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
                "formulas": "Cu|Cu2S",
                "space_groups": "225|14",
                "rwp": 9.25,
                "weights_json": '{"Cu": 80, "Cu2S": 20}',
                "cif_paths": "/internal/a.cif|/internal/b.cif",
                "stdout_tail": "technical",
                "status": "ok",
            }
        ],
    )

    assert list(rows[0]) == ["rank", "hypothesis", "rwp", "phase_fractions", "pattern_rank", "status", "time"]
    assert "/internal" not in str(rows[0])


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
