"""Render and export registered post-analysis results without rerunning fits."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import zipfile

from .post_analysis import ViewSettings


def axis_title(view):
    axis = view.get("axis", "run_number")
    unit = next((s.get("axis_unit") for s in view.get("series", []) if s.get("axis_unit")), "")
    name = {
        "run_number": "Scan number",
        "start_time": "Acquisition time",
        "elapsed_time": "Elapsed time",
        "sample_state": "Verified composition / preparation state",
    }.get(axis, axis.replace("_", " ").capitalize())
    return name + (f" ({unit})" if unit else "")


def y_title(view):
    metric = view.get("metric", "")
    rep = view.get("representation", "")
    if rep == "relative":
        return f"Relative {metric} change (%)"
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
            else "Å" if metric in {"a", "b", "c", "d_hkl"} else ""
        )
    )
    return ("Δ" if rep == "difference" else "") + metric + (f" ({unit})" if unit else "")


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
            xs = [
                (
                    datetime.fromtimestamp(p["x"], timezone.utc).isoformat()
                    if view.get("axis") == "start_time" and p["x"] is not None
                    else (
                        str(p["x"])
                        if view.get("axis") == "run_number" and p["x"] is not None
                        else p["x"]
                    )
                )
                for p in points
            ]
            name = f"{series['phase']} · sample {series['sample']} · {branch}"
            figure.add_trace(
                go.Scatter(
                    x=xs,
                    y=[p["value"] for p in points],
                    mode="lines+markers" if settings.connect else "markers",
                    connectgaps=False,
                    name=name,
                    legendgroup=series["phase_key"],
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
                            else "circle"
                        ),
                    },
                    error_y={
                        "type": "data",
                        "array": [
                            math.sqrt(p["variance"]) if p.get("variance") is not None else None
                            for p in points
                        ],
                        "visible": settings.show_uncertainty,
                    },
                    customdata=[
                        [
                            p["run_id"],
                            p.get("axis_source"),
                            p.get("state"),
                            p.get("reference_run") or "none",
                            p.get("reason") or "",
                            p.get("source_gpx_sha256") or "legacy",
                        ]
                        for p in points
                    ],
                    hovertemplate="%{customdata[0]}<br>x=%{x}<br>value=%{y:.7g}<br>%{customdata[2]}<br>reference: %{customdata[3]}<br>%{customdata[1]}<extra>%{fullData.name}</extra>",
                ),
                row=banks.index(series.get("bank")) + 1,
                col=1,
            )
    figure.update_layout(
        template="plotly_white",
        height=max(340, len(banks) * 300),
        margin={"l": 75, "r": 20, "t": 45, "b": 65},
        legend={"orientation": "h"},
        uirevision=f"{view['id']}:{view.get('axis')}:{view.get('metric')}:{view.get('representation')}",
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
                x0=lo,
                x1=hi,
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
                "branch",
                "reference_run",
                "excluded",
                "display_hidden",
                "reason",
                "source_gpx_sha256",
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
                                for key in ["metric", "representation", "value_unit", "axis"]
                            },
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
