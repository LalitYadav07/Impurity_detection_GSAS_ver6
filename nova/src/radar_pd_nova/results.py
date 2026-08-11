"""Result discovery and Plotly rendering for normalized RADAR-PD outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def read_table(path: str | Path, *, limit: int = 500) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for _, row in zip(range(limit), csv.DictReader(handle))]


def discover_tables(root: str | Path) -> list[dict[str, str]]:
    base = Path(root)
    return [
        {"name": path.stem.replace("_", " ").title(), "path": str(path)}
        for path in sorted(base.rglob("*.csv"))
    ]


def discover_plot_payloads(root: str | Path) -> list[dict[str, str]]:
    base = Path(root)
    return [
        {
            "name": path.name.replace(".plotdata.json", "").replace("_", " ").title(),
            "path": str(path),
            "kind": str(read_json(path).get("plot_kind") or "interactive plot"),
        }
        for path in sorted(base.rglob("*.plotdata.json"))
    ]


def _array(payload: dict[str, Any], *names: str) -> list[float]:
    for name in names:
        value = payload.get(name)
        if isinstance(value, list):
            try:
                return [float(item) for item in value]
            except (TypeError, ValueError):
                continue
    return []


def _nested_array(payload: dict[str, Any], section: str, *names: str) -> list[float]:
    nested = payload.get(section)
    return _array(nested, *names) if isinstance(nested, dict) else []


def component_figure(payload: dict[str, Any]) -> go.Figure:
    """Build the compact rapid 512-bin component-fit inspection plot."""

    q = _array(payload, "q", "x") or _nested_array(payload, "data", "q", "x")
    target = _array(payload, "target", "measured", "y") or _nested_array(payload, "data", "target", "measured")
    total = _array(payload, "total_fit", "fit", "ycalc") or _nested_array(payload, "data", "total_fit", "fit")
    residual = _array(payload, "residual", "difference") or _nested_array(payload, "data", "residual", "difference")
    background = _array(payload, "background", "baseline") or _nested_array(payload, "data", "background", "baseline")
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26], vertical_spacing=0.04)
    figure.add_trace(
        go.Scatter(x=q, y=target, mode="markers", name="Measured", marker={"size": 4, "color": "#27313d", "opacity": 0.58}),
        row=1,
        col=1,
    )
    if background:
        figure.add_trace(go.Scatter(x=q, y=background, mode="lines", name="Background", line={"color": "#9aa7b2", "dash": "dash"}), row=1, col=1)
    components = payload.get("components") or payload.get("phase_components") or []
    colors = ["#0f766e", "#2563eb", "#c2410c", "#7c3aed", "#a16207"]
    if isinstance(components, dict):
        components = [{"label": key, "values": value} for key, value in components.items()]
    for index, component in enumerate(components if isinstance(components, list) else []):
        if not isinstance(component, dict):
            continue
        values = _array(component, "scaled", "values", "intensity", "y")
        label = str(component.get("label") or component.get("phase") or component.get("formula") or f"Phase {index + 1}")
        contribution = component.get("relative_contribution_pct") or component.get("weight_pct")
        if contribution is not None:
            label = f"{label} ({float(contribution):.1f}%)"
        figure.add_trace(
            go.Scatter(x=q[: len(values)], y=values, mode="lines", name=label, line={"width": 1.7, "color": colors[index % len(colors)]}),
            row=1,
            col=1,
        )
    figure.add_trace(go.Scatter(x=q, y=total, mode="lines", name="Total hypothesis fit", line={"color": "#dc2626", "width": 2.5}), row=1, col=1)
    figure.add_trace(go.Scatter(x=q, y=residual, mode="lines", name="Difference", line={"color": "#0f766e", "width": 1.3}), row=2, col=1)
    figure.add_hline(y=0, line={"color": "#94a3b8", "width": 1}, row=2, col=1)
    figure.update_yaxes(title_text="Scaled intensity", row=1, col=1)
    figure.update_yaxes(title_text="Difference", row=2, col=1)
    figure.update_xaxes(title_text="Q (1/Angstrom)", row=2, col=1)
    return _finish_figure(figure, str(payload.get("title") or "Pattern contribution fit"), height=560)


def gsas_figure(payload: dict[str, Any]) -> go.Figure:
    """Build an observed/calculated/difference plot with separated phase tick rows."""

    arrays = payload.get("arrays") if isinstance(payload.get("arrays"), dict) else payload
    x = _array(arrays, "x", "two_theta", "tof")
    yobs = _array(arrays, "yobs", "observed")
    ycalc = _array(arrays, "ycalc", "calculated")
    residual = _array(arrays, "resid", "difference")
    phase_labels = payload.get("phase_labels") or payload.get("phase_order") or []
    ticks = payload.get("phase_ticks") or payload.get("ticks") or {}
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.67, 0.20, 0.13],
        vertical_spacing=0.035,
    )
    figure.add_trace(go.Scatter(x=x, y=yobs, mode="markers", name="Observed", marker={"size": 3, "color": "#27313d", "opacity": 0.55}), row=1, col=1)
    figure.add_trace(go.Scatter(x=x, y=ycalc, mode="lines", name="Calculated", line={"color": "#dc2626", "width": 2}), row=1, col=1)
    figure.add_trace(go.Scatter(x=x, y=residual, mode="lines", name="Difference", line={"color": "#0f766e", "width": 1.2}), row=2, col=1)
    figure.add_hline(y=0, line={"color": "#94a3b8", "width": 1}, row=2, col=1)
    colors = ["#0f766e", "#2563eb", "#c2410c", "#7c3aed", "#a16207"]
    if isinstance(ticks, list):
        ticks = {str(label): values for label, values in zip(phase_labels, ticks)}
    for index, label in enumerate(phase_labels):
        positions = ticks.get(label, []) if isinstance(ticks, dict) else []
        if isinstance(positions, dict):
            positions = positions.get("all") or positions.get("positions") or []
        figure.add_trace(
            go.Scatter(
                x=positions,
                y=[index] * len(positions),
                mode="markers",
                name=str(label),
                marker={"symbol": "line-ns", "size": 11, "line": {"width": 2, "color": colors[index % len(colors)]}},
                hovertemplate=f"{label}<br>%{{x:.5g}}<extra></extra>",
            ),
            row=3,
            col=1,
        )
    axis_title = str(payload.get("x_label") or payload.get("xlabel") or "Diffraction coordinate")
    figure.update_yaxes(title_text="Intensity", row=1, col=1)
    figure.update_yaxes(title_text="Difference", row=2, col=1)
    figure.update_yaxes(title_text="Phases", tickmode="array", tickvals=list(range(len(phase_labels))), ticktext=phase_labels, row=3, col=1)
    figure.update_xaxes(title_text=axis_title, row=3, col=1)
    rwp = payload.get("rwp")
    title = "Refinement fit" + (f" / Rwp {float(rwp):.2f}%" if rwp is not None else "")
    return _finish_figure(figure, title, height=720)


def figure_for_payload(payload: dict[str, Any]) -> go.Figure:
    kind = str(payload.get("plot_kind") or "").lower()
    if "gsas" in kind or any(key in payload for key in ("phase_ticks", "phase_weights", "rwp")):
        return gsas_figure(payload)
    return component_figure(payload)


def _finish_figure(figure: go.Figure, title: str, *, height: int) -> go.Figure:
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        height=height,
        margin={"l": 70, "r": 25, "t": 60, "b": 55},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "right", "x": 1},
        font={"family": "Inter, Source Sans 3, Arial, sans-serif", "color": "#17251f"},
    )
    figure.update_xaxes(showgrid=False, zeroline=False, linecolor="#cbd5e1")
    figure.update_yaxes(gridcolor="#e8eef0", zeroline=False, linecolor="#cbd5e1")
    return figure


def phase_fraction_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize phase fractions from full or rapid result summaries."""

    candidates: Iterable[Any] = (
        summary.get("phase_fractions")
        or summary.get("weighted_fractions")
        or summary.get("phases")
        or []
    )
    rows: list[dict[str, Any]] = []
    if isinstance(candidates, dict):
        candidates = [{"phase": key, "weight_percent": value} for key, value in candidates.items()]
    for item in candidates if isinstance(candidates, list) else []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "phase": str(item.get("phase") or item.get("formula") or item.get("label") or "Unknown"),
                "space_group": item.get("space_group") or item.get("sg") or "-",
                "weight_percent": item.get("weight_percent") or item.get("weight_pct") or item.get("fraction") or "-",
            }
        )
    return rows
