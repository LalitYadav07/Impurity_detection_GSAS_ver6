import json
from pathlib import Path

import numpy as np

from radar_pd_nova.results import (
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
