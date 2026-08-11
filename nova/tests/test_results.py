import json
from pathlib import Path

from radar_pd_nova.results import (
    discover_plot_payloads,
    discover_tables,
    figure_for_payload,
    phase_fraction_rows,
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


def test_phase_fraction_normalization() -> None:
    rows = phase_fraction_rows(
        {"phase_fractions": [{"formula": "Fe", "space_group": 225, "weight_percent": 91.2}]}
    )
    assert rows == [{"phase": "Fe", "space_group": 225, "weight_percent": 91.2}]
