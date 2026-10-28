"""Render and export registered post-analysis results without rerunning fits."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import re
import zipfile

from .post_analysis import ViewSettings


def _legend_phase(value):
    """Keep the formula and group visible; retain the full description in hover."""
    text = str(value)
    compact = re.match(r"^(.*?)\s+-\s+.+?(\s+\(SG\s+.+\))$", text)
    return compact[1] + compact[2] if compact else text


def _legend_branch(value):
    # Acquisition segments stay separate traces. Their internal sequence
    # numbers do not need separate legend entries for each reversal.
    return re.sub(r"(increasing|decreasing|unclassified):\d+$", r"\1", value)


def axis_title(view):
    axis = view.get("axis", "run_number")
    unit = next((s.get("axis_unit") for s in view.get("series", []) if s.get("axis_unit")), "")
    name = {
        "run_number": "Scan number",
        "start_time": "Acquisition time",
        "elapsed_time": "Elapsed time",
        "sample_state": "Verified composition / preparation state",
    }.get(axis, axis.replace("_", " ").capitalize())
    if axis == "temperature":
        unit = "°C" if view.get("temperature_display_unit") == "C" else "K"
    return name + (f" ({unit})" if unit else "")


def display_x(view, value):
    if value is None:
        return None
    if view.get("axis") == "temperature" and view.get("temperature_display_unit") == "C":
        return value - 273.15
    if view.get("axis") == "start_time":
        return datetime.fromtimestamp(value, timezone.utc).isoformat()
    if view.get("axis") == "run_number":
        return str(value)
    return value


def y_title(view):
    metric = view.get("metric", "")
    rep = view.get("representation", "")
    output = view.get("reference_output")
    if output == "temperature":
        return "Inferred sample temperature (K)"
    if output == "temperature_offset":
        return "Inferred − logged temperature (K)"
    prefix = "Excess over reference · " if output == "residual" else ""
    signal = "Lattice" if metric == "multiple" else "Lattice ratio" if metric == "phase_ratio" else metric
    if rep == "relative":
        return prefix + f"Relative {signal} change (%)"
    if rep == "permille":
        return prefix + f"{signal} expansion · 1000(L/L₀ − 1) (‰)"
    if rep == "normalized":
        return prefix + f"Normalized {signal} · L/L₀"
    if rep == "microstrain":
        return f"Reference {metric} change (µε)"
    if rep == "progress":
        return "Defined progress proxy"
    if metric == "weight_percent":
        return (
            "Crystalline phase fraction (wt%)"
            if rep == "absolute"
            else "Phase-fraction change (percentage points)"
        )
    unit = (
        "Å³"
        if metric == "volume"
        else (
            "°"
            if metric in {"alpha", "beta", "gamma"}
            else "Å" if metric in {"a", "b", "c", "d_hkl"} or metric == "multiple" and view.get("value_unit") == "angstrom" else ""
        )
    )
    if metric == "multiple" and view.get("value_unit") == "angstrom^3":
        unit = "Å³"
    return ("Δ" if rep == "difference" else "") + signal + (f" ({unit})" if unit else "")


def view_figure(view, settings: ViewSettings):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    banks = list(dict.fromkeys(s.get("bank") for s in view.get("series", []))) or [None]
    figure = make_subplots(
        rows=len(banks),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[
            f"Source bank {bank if bank is not None else 'not recorded'}" for bank in banks
        ],
    )
    colors = (
        "#176b48",
        "#6554a4",
        "#b05c23",
        "#1c718c",
        "#a13d67",
        "#596538",
        "#4b56aa",
        "#9b552c",
        "#476963",
        "#845b80",
    )
    samples = {s["sample"] for s in view.get("series", [])}
    all_branches = {p.get("branch") for s in view.get("series", []) for p in s["points"]}
    legend_keys = set()
    for series in view.get("series", []):
        branches = list(dict.fromkeys(p.get("branch") for p in series["points"]))
        color = colors[
            series.get(
                "color_index",
                int(hashlib.sha256(series["phase_key"].encode()).hexdigest()[:6], 16) % len(colors),
            )
        ]
        for branch in branches:
            points = [
                p
                for p in series["points"]
                if p.get("branch") == branch and not p.get("display_hidden")
            ]
            if not points:
                continue
            xs = [display_x(view, p["x"]) for p in points]
            name = _legend_phase(series["phase"])
            metric = series.get("metric", view.get("metric"))
            if view.get("metric") == "multiple":
                name += f" · {metric}"
            if len(samples) > 1:
                name += f" · sample {series['sample']}"
            if len(all_branches) > 1:
                name += f" · {_legend_branch(branch)}"
            legend_key = (series["phase_key"], metric, series["sample"], _legend_branch(branch))
            figure.add_trace(
                go.Scatter(
                    x=xs,
                    y=[p["value"] for p in points],
                    mode="lines+markers" if settings.connect else "markers",
                    connectgaps=False,
                    name=name,
                    legendgroup=series["phase_key"] + ":" + str(metric),
                    showlegend=legend_key not in legend_keys,
                    line={
                        "color": color,
                        "dash": (
                            "dash"
                            if "decreas" in branch.lower() or "cool" in branch.lower()
                            else "solid"
                        ),
                    },
                    marker={
                        "color": color,
                        "symbol": (
                            "diamond"
                            if "decreas" in branch.lower() or "cool" in branch.lower()
                            else {"a": "circle", "b": "square", "c": "triangle-up", "volume": "diamond"}.get(metric, "circle")
                        ),
                    },
                    error_y={
                        "type": "data",
                        "array": [
                            (math.sqrt(p["variance"]) if p.get("variance") is not None else None)
                            for p in points
                        ],
                        "visible": settings.show_uncertainty,
                    },
                    error_x={
                        "type": "data", "symmetric": False,
                        "array": [
                            max(0., p["axis_maximum"] - p["x"])
                            if p.get("axis_maximum") is not None and p.get("x") is not None else None
                            for p in points
                        ],
                        "arrayminus": [
                            max(0., p["x"] - p["axis_minimum"])
                            if p.get("axis_minimum") is not None and p.get("x") is not None else None
                            for p in points
                        ],
                        "visible": settings.show_temperature_span and view.get("axis") == "temperature",
                        "color": "#82918a", "thickness": 1,
                    },
                    customdata=[
                        [
                            p["run_id"],
                            p.get("axis_source"),
                            p.get("state"),
                            p.get("reference_run") or "none",
                            p.get("reason") or "",
                            p.get("source_gpx_sha256") or "legacy",
                            series["phase_key"],
                            series["bank"],
                            series["sample"],
                            branch,
                            series["phase"],
                            metric,
                            math.sqrt(p["variance"]) if p.get("variance") is not None else "unavailable",
                            p.get("uncertainty_note") or "Conditional refinement standard uncertainty (1σ)",
                        ]
                        for p in points
                    ],
                    hovertemplate="%{customdata[10]} · %{customdata[11]}<br>%{customdata[0]} · bank %{customdata[7]} · sample %{customdata[8]}<br>x=%{x}<br>value=%{y:.7g}<br>conditional 1σ: %{customdata[12]}<br>%{customdata[2]}<br>reference: %{customdata[3]}<br>%{customdata[1]}<br>%{customdata[13]}<extra>%{fullData.name}</extra>",
                ),
                row=banks.index(series.get("bank")) + 1,
                col=1,
            )
            legend_keys.add(legend_key)
        for reference in view.get("reference_traces", []):
            if reference["series"] != series["key"]:
                continue
            for branch in dict.fromkeys(p["branch"] for p in reference["points"]):
                points = [p for p in reference["points"] if p["branch"] == branch]
                curve = view.get("reference_curve", {})
                figure.add_trace(go.Scatter(
                    x=[display_x(view, p["x"]) for p in points],
                    y=[p["value"] for p in points],
                    name="Reference · " + _legend_phase(series["phase"]),
                    legendgroup="reference:" + series["key"],
                    showlegend=branch == reference["points"][0]["branch"],
                    mode="lines", line={"color": color, "dash": "dot"},
                    hovertemplate="Reference curve<br>%{x}<br>%{y:.7g}<extra>" + curve.get("label", "Reviewed reference") + "</extra>",
                ), row=banks.index(series.get("bank"))+1, col=1)
    figure.update_layout(
        template="plotly_white",
        height=max(460, len(banks) * 300 + 180),
        margin={"l": 75, "r": 20, "t": 45, "b": 155},
        # Plotly 5 remains supported; it omits the newer legend height cap.
        legend=go.layout.Legend(
            orientation="h", y=-0.18, yanchor="top", maxheight=120, skip_invalid=True
        ),
        uirevision=f"{view['id']}:{view.get('axis')}:{view.get('metric')}:{view.get('representation')}:{view.get('temperature_display_unit')}:{view.get('reference_output')}",
    )
    if view.get("review_required"):
        figure.update_layout(
            title={
                "text": "Scientific review required",
                "font": {"size": 14},
                "x": 0.02,
            },
            margin={"l": 75, "r": 20, "t": 85, "b": 155},
        )
    figure.update_xaxes(
        title_text=axis_title(view),
        row=len(banks),
        col=1,
        type=(
            "category"
            if view.get("axis") == "run_number"
            else "date" if view.get("axis") == "start_time" else "-"
        ),
        nticks=6,
        automargin=True,
    )
    figure.update_yaxes(title_text=y_title(view), zeroline=view.get("representation") != "absolute")
    if settings.x_range is not None:
        figure.update_xaxes(range=list(settings.x_range), autorange=False)
    if settings.y_range is not None:
        figure.update_yaxes(range=list(settings.y_range), autorange=False)
    for crossing in view.get("crossings", []):
        if view.get("axis") not in {"run_number", "sample_state", "start_time"}:
            lo, hi = sorted(crossing["interval"])
            figure.add_vrect(
                x0=display_x(view, lo),
                x1=display_x(view, hi),
                fillcolor="rgba(120,120,120,0.1)",
                line_width=0,
                annotation_text="Candidate crossing interval",
                row="all",
                col=1,
            )
    if not any(p.get("value") is not None for s in view.get("series", []) for p in s["points"]):
        figure.add_annotation(
            text="No eligible points yet. Review the requirements above.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    return figure


def export_bundle(result, settings, scans):
    """An offline, session-independent record of exactly the calculated views."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("post-analysis.json", json.dumps(result, indent=2, allow_nan=False))
        archive.writestr("observations.json", json.dumps(scans, indent=2, allow_nan=False))
        by_id = {s.id: s for s in settings.views}
        for index, view in enumerate(result["views"]):
            # Fixed filenames avoid user-controlled path traversal or duplicate ZIP entries.
            stem = f"view-{index+1:02d}"
            table = io.StringIO(newline="")
            fields = [
                "phase",
                "sample",
                "bank",
                "run_id",
                "run_number",
                "metric",
                "representation",
                "value_unit",
                "axis",
                "x",
                "axis_unit",
                "axis_source",
                "axis_kind",
                "value",
                "variance",
                "raw_value",
                "raw_variance",
                "state",
                "scientific_status",
                "branch",
                "reference_run",
                "excluded",
                "display_hidden",
                "reason",
                "source_gpx_sha256",
                "conditional_standard_uncertainty",
                "uncertainty_note",
                "axis_minimum", "axis_maximum",
                "display_x", "display_axis_unit",
                "reference_curve_id", "reference_value",
                "inferred_temperature_K", "logged_temperature_K", "reference_sensitivity_per_K",
            ]
            writer = csv.DictWriter(table, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for series in view.get("series", []):
                for point in series["points"]:
                    writer.writerow(
                        {
                            **series,
                            **point,
                            **{
                                key: view.get(key)
                                for key in [
                                    "representation",
                                    "value_unit",
                                    "axis",
                                ]
                            },
                            "metric": point.get("metric", series.get("metric", view.get("metric"))),
                            "conditional_standard_uncertainty": math.sqrt(point["variance"]) if point.get("variance") is not None else None,
                            "display_x": display_x(view, point.get("x")),
                            "display_axis_unit": "C" if view.get("axis") == "temperature" and view.get("temperature_display_unit") == "C" else point.get("axis_unit"),
                        }
                    )
            archive.writestr(stem + ".csv", table.getvalue())
            if view["status"] == "available":
                archive.writestr(
                    stem + ".html",
                    view_figure(view, by_id[view["id"]]).to_html(
                        include_plotlyjs=True, full_html=True
                    ),
                )
    return output.getvalue()
