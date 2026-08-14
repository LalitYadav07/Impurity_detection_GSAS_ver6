"""Result discovery and Plotly rendering for normalized RADAR-PD outputs."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def read_plot_payload(path: str | Path) -> dict[str, Any]:
    """Load a plot sidecar and its optional compressed scientific arrays."""

    payload_path = Path(path)
    payload = read_json(payload_path)
    arrays_name = payload.get("arrays_npz")
    if not isinstance(arrays_name, str) or not arrays_name.strip():
        return payload

    parent = payload_path.resolve().parent
    arrays_path = (parent / arrays_name).resolve()
    try:
        arrays_path.relative_to(parent)
    except ValueError:
        return payload
    if not arrays_path.is_file():
        return payload

    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            payload["arrays"] = {name: archive[name].tolist() for name in archive.files}
    except Exception:
        pass
    return payload


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


def _plot_payload_has_arrays(payload: dict[str, Any]) -> bool:
    """Check that a plot payload actually loaded its declared numeric arrays.

    Galaxy's per-output collection downloads can duplicate a plot's JSON
    under an unrelated filename and without its paired .npz archive (the
    results-archive extraction is the only source that keeps them together),
    producing a payload with no numeric arrays. Checking the same loaded
    result that rendering uses -- rather than re-deriving file-existence
    logic here -- catches that regardless of what the duplicate is named or
    why the array failed to load.
    """

    arrays_name = payload.get("arrays_npz")
    if not isinstance(arrays_name, str) or not arrays_name.strip():
        return True
    arrays = payload.get("arrays")
    if not isinstance(arrays, dict):
        return False
    return any(isinstance(value, list) and value for value in arrays.values())


def discover_plot_payloads(root: str | Path) -> list[dict[str, str]]:
    base = Path(root)
    seen_names: set[str] = set()
    options: list[dict[str, str]] = []
    for path in sorted(base.rglob("*.plotdata.json")):
        payload = read_plot_payload(path)
        if not _plot_payload_has_arrays(payload):
            continue
        if path.name in seen_names:
            continue
        seen_names.add(path.name)
        options.append(
            {
                "name": path.name.replace(".plotdata.json", "").replace("_", " ").title(),
                "path": str(path),
                "kind": str(payload.get("plot_kind") or "interactive plot"),
            }
        )
    return options


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


def _first_value(payload: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _component_contribution_percent(component: dict[str, Any]) -> float | None:
    percent = _as_float(
        _first_value(
            component,
            "relative_contribution_pct",
            "contribution_pct",
            "weight_percent",
            "weight_pct",
        )
    )
    if percent is not None:
        return percent
    relative_scale = _as_float(_first_value(component, "relative_scale"))
    if relative_scale is not None:
        return relative_scale * 100.0
    contribution = _as_float(_first_value(component, "contribution"))
    if contribution is None:
        return None
    return contribution * 100.0 if abs(contribution) <= 1.0 else contribution


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
    components = payload.get("components") or payload.get("phase_components") or payload.get("phases") or []
    colors = ["#0f766e", "#2563eb", "#c2410c", "#7c3aed", "#a16207"]
    if isinstance(components, dict):
        components = [{"label": key, "values": value} for key, value in components.items()]
    for index, component in enumerate(components if isinstance(components, list) else []):
        if not isinstance(component, dict):
            continue
        values = _array(component, "scaled", "component", "values", "intensity", "y")
        label = str(component.get("label") or component.get("phase") or component.get("formula") or f"Phase {index + 1}")
        contribution = _component_contribution_percent(component)
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
    phase_order = payload.get("phase_order") or []
    phase_labels = payload.get("phase_labels") or {}
    ticks = payload.get("phase_ticks") or payload.get("ticks") or {}
    major_ticks = payload.get("phase_major_ticks") or {}
    major_tick_details = payload.get("phase_major_tick_details") or {}
    if not phase_order:
        if isinstance(ticks, dict):
            phase_order = list(ticks)
        elif isinstance(phase_labels, dict):
            phase_order = list(phase_labels)
        else:
            phase_order = list(phase_labels)
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
        ticks = {str(label): values for label, values in zip(phase_order, ticks)}
    axis_labels: list[str] = []
    for index, phase_name in enumerate(phase_order):
        label = str(phase_labels.get(phase_name) or phase_name) if isinstance(phase_labels, dict) else str(phase_name)
        axis_labels.append(label)
        positions = ticks.get(phase_name, []) if isinstance(ticks, dict) else []
        if isinstance(positions, dict):
            positions = positions.get("all") or positions.get("positions") or []
        color = colors[index % len(colors)]
        figure.add_trace(
            go.Scatter(
                x=positions,
                y=[index] * len(positions),
                mode="markers",
                name=str(label),
                marker={"symbol": "line-ns", "size": 11, "line": {"width": 2, "color": color}},
                hovertemplate=f"{label}<br>%{{x:.5g}}<extra></extra>",
            ),
            row=3,
            col=1,
        )
        strongest: list[tuple[float, str]] = []
        details = major_tick_details.get(phase_name, []) if isinstance(major_tick_details, dict) else []
        for fallback_rank, item in enumerate(details if isinstance(details, list) else [], start=1):
            if not isinstance(item, dict):
                continue
            x_value = _as_float(item.get("x"))
            if x_value is None:
                continue
            rank = item.get("rank", fallback_rank)
            hkl = str(item.get("hkl") or "").strip()
            strength = _as_float(item.get("relative_strength"))
            hover = f"{label}<br>Top Bragg peak {rank}<br>x={x_value:.5g}"
            if hkl:
                hover += f"<br>HKL={hkl}"
            if strength is not None:
                hover += f"<br>Relative strength={strength:.3f}"
            strongest.append((x_value, hover))
        if not strongest:
            fallback = major_ticks.get(phase_name, []) if isinstance(major_ticks, dict) else []
            for rank, value in enumerate(fallback if isinstance(fallback, list) else [], start=1):
                x_value = _as_float(value)
                if x_value is not None:
                    strongest.append((x_value, f"{label}<br>Top Bragg peak {rank}<br>x={x_value:.5g}"))
        if strongest:
            figure.add_trace(
                go.Scatter(
                    x=[item[0] for item in strongest],
                    y=[index] * len(strongest),
                    mode="markers",
                    name=f"Key peaks: {label}",
                    marker={"symbol": "line-ns", "size": 17, "line": {"width": 3, "color": color}},
                    text=[item[1] for item in strongest],
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
    axis_title = str(payload.get("x_label") or payload.get("xlabel") or "Diffraction coordinate")
    figure.update_yaxes(title_text="Intensity", row=1, col=1)
    figure.update_yaxes(title_text="Difference", row=2, col=1)
    figure.update_yaxes(title_text="Phases", tickmode="array", tickvals=list(range(len(axis_labels))), ticktext=axis_labels, row=3, col=1)
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

    candidates: Iterable[Any] = []
    for key in ("phase_fractions", "weighted_fractions", "phases"):
        value = summary.get(key)
        if value:
            candidates = value
            break
    if not candidates:
        candidates = _rapid_final_fractions(summary)
    rows: list[dict[str, Any]] = []
    if isinstance(candidates, dict):
        candidates = [
            {"phase": key, **value} if isinstance(value, dict) else {"phase": key, "weight_percent": value}
            for key, value in candidates.items()
        ]
    for item in candidates if isinstance(candidates, list) else []:
        if not isinstance(item, dict):
            continue
        phase, label_space_group = _split_phase_label(
            str(_first_value(item, "phase", "formula", "label") or "Unknown")
        )
        space_group = _first_value(item, "space_group", "sg")
        weight = _first_value(
            item,
            "weight_percent",
            "weight_pct",
            "weight_fraction_pct",
            "fraction",
        )
        rows.append(
            {
                "phase": phase,
                "space_group": space_group if space_group is not None else (label_space_group or "-"),
                "weight_percent": weight if weight is not None else "-",
            }
        )
    return rows


def _split_phase_label(label: str) -> tuple[str, str | None]:
    match = re.match(r"^(.*?)\s*\(SG\s+([^()]+)\)\s*$", label, flags=re.IGNORECASE)
    if not match:
        return label, None
    return match.group(1).strip() or label, match.group(2).strip()


def _rapid_final_fractions(summary: dict[str, Any]) -> list[dict[str, Any]]:
    hypotheses = summary.get("hypotheses")
    if not isinstance(hypotheses, list):
        return []
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, dict):
            continue
        raw = _first_value(hypothesis, "weights_json", "phase_fractions", "weights")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        if isinstance(raw, dict) and raw:
            return [{"phase": phase, "weight_percent": weight} for phase, weight in raw.items()]
        if isinstance(raw, list) and raw:
            return [item for item in raw if isinstance(item, dict)]
    return []


def total_elapsed_seconds(summary: dict[str, Any]) -> float | None:
    """Read total time from a direct result summary or nested Rapid summary."""

    direct = _first_value(summary, "elapsed_seconds", "total_seconds")
    value = _as_float(direct)
    if value is not None:
        return value
    timing = summary.get("timing")
    if isinstance(timing, dict):
        value = _as_float(_first_value(timing, "total", "total_seconds"))
        if value is not None:
            return value
    nested = summary.get("summary")
    if not isinstance(nested, dict):
        nested = summary
    live_run = nested.get("live_run") if isinstance(nested, dict) else None
    timings = live_run.get("timings") if isinstance(live_run, dict) else None
    return _as_float(_first_value(timings, "total_seconds")) if isinstance(timings, dict) else None
