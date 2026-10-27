"""Linked observed-pattern views; display binning/contrast never changes inference."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COORDINATES = {
    "tof_us": "Time of flight (μs)",
    "d_angstrom": "d-spacing (Å)",
    "q_inv_angstrom": "Q (Å⁻¹)",
}
COLORS = ["#087b44", "#a15e0d", "#3867b0", "#8b529e"]
ICONS = [
    "mdi-circle-outline",
    "mdi-rhombus-outline",
    "mdi-square-outline",
    "mdi-triangle-outline",
    "mdi-hexagon-outline",
    "mdi-star-outline",
    "mdi-plus",
    "mdi-triangle-down-outline",
]


def marker_style(index):
    if index is None:
        return {"icon": "mdi-help-circle-outline", "color": "#718077"}
    return {"icon": ICONS[index % 8], "color": COLORS[(index // 8 + index % 4) % 4]}


def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
    )
    fig.update_layout(
        template="plotly_white",
        margin={"l": 25, "r": 20, "t": 25, "b": 25},
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=280,
    )
    return fig


def locked_limits(pattern, background=None):
    measured = (
        pattern.intensity if background is None else pattern.intensity - background
    )
    values = measured[pattern.valid & np.isfinite(measured)]
    if not len(values):
        raise ValueError("A color reference needs valid measured intensities")
    low, high = float(min(0, np.min(values))), float(np.quantile(values, 0.999))
    high = max(high, low + 1e-12)
    errors = pattern.sigma[pattern.valid]
    errors = errors[np.isfinite(errors) & (errors > 0)]
    soft = max(
        float(np.median(errors)) if len(errors) else 0, (high - low) * 0.002, 1e-12
    )
    return {"low": low, "high": high, "soft": soft, "source": pattern.observation_id}


def transform(values, limits, *, scale="asinh", contrast=50):
    values = np.asarray(values, dtype=float)
    soft = limits["soft"] * 10 ** ((50 - float(contrast)) / 25)
    if scale == "asinh":
        return np.arcsinh(values / soft)
    if scale == "linear":
        return values.copy()
    if scale == "log":
        output = np.full(values.shape, np.nan)
        positive = values > 0
        output[positive] = np.log10(values[positive])
        return output
    raise ValueError("Unsupported intensity display scale")


def display_grid(pattern, columns, region=None):
    """Keep full-domain coverage, adding native detail within the visible range.

    The browser may expand an axis before a new server frame arrives. Retaining
    coarse bins outside the detail window prevents that interaction from showing
    artificial blank regions. All bins still use original measured samples.
    """
    indices = np.linspace(
        0, len(pattern.x), min(columns, len(pattern.x)) + 1, dtype=int
    )
    if region is not None and region[1]>=pattern.x[0] and region[0]<=pattern.x[-1]:
        start=max(0,int(np.searchsorted(pattern.x,region[0]))-1)
        end=min(len(pattern.x),int(np.searchsorted(pattern.x,region[1],side="right"))+1)
        detail=np.linspace(start,end,min(columns,end-start)+1,dtype=int)
        indices=np.union1d(indices,detail)
    centers = np.array([np.mean(pattern.x[a:b]) for a, b in pairwise(indices)])
    boundaries = np.r_[
        pattern.x[0],
        (pattern.x[indices[1:-1] - 1] + pattern.x[indices[1:-1]]) / 2,
        np.nextafter(pattern.x[-1], np.inf),
    ]
    return centers, boundaries


def binned_intensity(pattern, boundaries, background=None):
    output = np.full(len(boundaries) - 1, -np.inf)
    indices = np.searchsorted(boundaries, pattern.x, side="right") - 1
    valid = pattern.valid & (indices >= 0) & (indices < len(output))
    measured = (
        pattern.intensity if background is None else pattern.intensity - background
    )
    valid &= np.isfinite(measured)
    np.maximum.at(output, indices[valid], measured[valid])
    output[~np.isfinite(output)] = np.nan
    return output


def ribbon_trace(pattern,background,limits,scale,contrast,label):
    raw=pattern.intensity if background is None else pattern.intensity-background
    values=np.where(pattern.valid,transform(raw,limits,scale=scale,contrast=contrast),np.nan)
    return {"type":"scatter","x":pattern.x.tolist(),"y":[float(v) if np.isfinite(v) else None for v in values],
        "customdata":[float(v) if np.isfinite(v) else None for v in raw],"mode":"lines","name":label,"line":{"color":"#133d76","width":1.1},
        "connectgaps":False,"showlegend":False,
        "hovertemplate":label+"<br>%{x:.6g}<br>Intensity %{customdata:.6g}<extra></extra>"}


def selection_presentation(run_ids,labels,selected,*,ribbon=True,comparison_run=None,change_regions=(),overlay="off"):
    axis="y2" if ribbon else "y"
    indices=set(range(0,len(run_ids),max(1,len(run_ids)//25)))|{len(run_ids)-1}
    annotations=[];shapes=[]
    if selected in run_ids:
        i=run_ids.index(selected);indices.add(i)
        annotations=[dict(x=0,xref="paper",xshift=-27,y=selected,yref=axis,text="<b>"+labels[i]+"</b>",
            showarrow=False,xanchor="right",bgcolor="#15543c",bordercolor="#15543c",borderpad=3,font={"color":"white","size":11}),
            dict(x=0,xref="paper",y=selected,yref=axis,ax=-22,ay=0,text="",showarrow=True,arrowhead=2,
                arrowsize=1.2,arrowwidth=2.5,arrowcolor="#15543c")]
        if ribbon:
            annotations.insert(0,dict(text=labels[i],x=0,xref="paper",y=1.03,yref="paper",showarrow=False,xanchor="left"))
        if overlay in {"ribbon","both"}:
            for region in change_regions:
                lo,hi=region["interval"];recognized=region.get("status")=="recognized_split"
                if ribbon:
                    shapes.append(dict(type="rect",xref="x",x0=lo,x1=hi,yref="y domain",y0=0,y1=1,
                        line={"color":"#702d92","width":1.2,"dash":"dot" if recognized else "solid"},
                        fillcolor="rgba(112,45,146,.03)" if recognized else "rgba(112,45,146,.10)"))
                if overlay=="both":
                    pair=[i,run_ids.index(comparison_run)] if comparison_run in run_ids else [i]
                    spans=[(min(pair),max(pair))] if abs(pair[-1]-pair[0])<=1 else [(j,j) for j in pair]
                    for low,high in spans:
                        shapes.append(dict(type="rect",xref="x2" if ribbon else "x",x0=lo,x1=hi,yref=axis,
                            y0=low-.43,y1=high+.43,
                            line={"color":"#382344","width":1.5,"dash":"dot" if recognized else "solid"},
                            fillcolor="rgba(255,255,255,.03)" if recognized else "rgba(255,255,255,.12)"))
    ordered=sorted(i for i in indices if 0<=i<len(run_ids))
    return {"annotations":annotations,"shapes":shapes,"axis":{
        "tickvals":[run_ids[i] for i in ordered],"ticktext":["" if run_ids[i]==selected else labels[i] for i in ordered]}}


def heatmap_figure(
    patterns,
    *,
    labels,
    run_ids,
    limits,
    scale="asinh",
    contrast=50,
    selected=None,
    region=None,
    grid_reference=None,
    backgrounds=None,
    boundaries=(),
    palette="Turbo",
    ribbon=None,
    ribbon_background=None,
    ribbon_label="Selected scan",
    change_regions=(),
    overlay="off",
    comparison_run=None,
    view_revision=0,
):
    if not patterns:
        return empty_figure(
            "Measured curves will populate the map as they become available"
        ), {}
    # Keep the same display columns as rows arrive. Changing aggregation with
    # the row count would silently change earlier colors even with fixed limits.
    if len(patterns) > 100:
        raise ValueError(
            "The map window is bounded to 100 scans; inspect older scans through history"
        )
    x, edges = display_grid(grid_reference or patterns[0], 512, region=region)
    backgrounds = backgrounds if backgrounds is not None else [None] * len(patterns)
    raw = np.vstack(
        [
            binned_intensity(p, edges, background=b)
            for p, b in zip(patterns, backgrounds)
        ]
    )
    z = transform(raw, limits, scale=scale, contrast=contrast)
    low = limits["low"] if scale != "log" else max(limits["soft"] * 0.25, 1e-12)
    high = max(limits["high"], low * 1.01 if scale == "log" else low + 1e-12)
    zlow, zhigh = transform([low, high], limits, scale=scale, contrast=contrast)
    tickvalues = (
        np.geomspace(low, high, 3) if scale == "log" else np.linspace(zlow, zhigh, 3)
    )
    if scale == "log":
        ticks = transform(tickvalues, limits, scale=scale, contrast=contrast)
        texts = [f"{v:.2g}" for v in tickvalues]
    else:
        ticks = tickvalues
        values = (
            np.sinh(ticks) * (limits["soft"] * 10 ** ((50 - float(contrast)) / 25))
            if scale == "asinh"
            else ticks
        )
        texts = [f"{v:.2g}" for v in values]
    # At most 102,500 display cells (coarse full domain + zoom detail). Native inference arrays
    # remain untouched, and absent/masked bins are transparent, never zero.
    clean = lambda a: [[float(v) if np.isfinite(v) else None for v in row] for row in a]
    height = max(330, min(760, 140 + len(patterns) * 22)) + (155 if ribbon is not None else 0)
    if palette not in {"Turbo","Cividis"}:
        raise ValueError("Unsupported diffraction map palette")
    heat = go.Heatmap(
            x=x.tolist(),
            y=run_ids,
            z=clean(z),
            customdata=clean(raw),
            zmin=float(zlow),
            zmax=float(zhigh),
            colorscale=palette,
            hoverongaps=False,
            colorbar={
                "title": {"text": "Intensity above local background" if any(b is not None for b in backgrounds) else "Measured intensity", "side": "top"},
                "orientation": "h",
                "y": -58 / max(100, height - 145),
                "yanchor": "top",
                "tickangle": 0,
                "tickfont": {"size": 11},
                "len": 0.9,
                "thickness": 12,
                "tickvals": ticks.tolist(),
                "ticktext": texts,
            },
            hovertemplate="%{y}<br>Coordinate %{x:.6g}<br>"
            + (
                "Above estimated background "
                if any(b is not None for b in backgrounds)
                else "Measured intensity "
            )
            + "%{customdata:.6g}<extra></extra>",
    )
    heat_axis="y2" if ribbon is not None else "y"
    if ribbon is None:
        fig=go.Figure(heat)
    else:
        fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.045,
            row_heights=[.30,.70])
        fig.add_trace(heat,row=2,col=1)
        fig.add_trace(go.Scatter(ribbon_trace(ribbon,ribbon_background,limits,scale,contrast,ribbon_label)),row=1,col=1)
        fig.update_yaxes(title="Selected scan",tickvals=ticks.tolist(),ticktext=texts,nticks=3,row=1,col=1)
    tick_indices = set(range(0, len(patterns), max(1, len(patterns) // 25))) | {
        len(patterns) - 1
    }
    if selected in run_ids:
        tick_indices.add(run_ids.index(selected))
    tick_indices = sorted(tick_indices)
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        height=height,
        margin={"l": 130, "r": 14, "t": 15, "b": 130},
        uirevision="tracking-map:" + str(patterns[0].scope) + ":view:" + str(view_revision),
        meta={
            "run_ids": run_ids,
            "display_binning": "maximum of valid native samples per cell",
            "limits_locked_to": limits["source"],
            "scale": scale,
            "palette":palette,
            "heatmap_yaxis":heat_axis,
            "coordinate":patterns[0].scope.coordinate,
            "full_range":[float((grid_reference or patterns[0]).x[0]),
                          float((grid_reference or patterns[0]).x[-1])],
        },
        font={"family": "Arial, sans-serif", "size": 12},
    )
    fig.update_xaxes(title_text=COORDINATES[patterns[0].scope.coordinate],nticks=5,
        **({"row":2,"col":1} if ribbon is not None else {}))
    fig.update_yaxes(**{
            "tickmode": "array",
            "tickvals": [run_ids[i] for i in tick_indices],
            "ticktext": ["" if run_ids[i]==selected else labels[i] for i in tick_indices],
            "type": "category",
            "categoryorder": "array",
            "categoryarray": run_ids,
            "autorange": "reversed",
            "fixedrange": True,
            "title": {"text":"Acquisition order","standoff":38},
        },**({"row":2,"col":1} if ribbon is not None else {}))
    by_id = {r["run_id"]: r for r in boundaries}
    for i, run_id in enumerate(run_ids):
        row = by_id.get(run_id, {})
        if i and row.get("boundary"):
            # A dark dashed halo keeps the white dashes visible over bright
            # rainbow cells. Selection uses only a left-margin pointer.
            for color,width in (("rgba(30,40,50,.65)",4.5),("white",2.2)):
                fig.add_shape(type="line",xref="paper",x0=0,x1=1,yref=heat_axis,
                    y0=i-.5,y1=i-.5,line={"color":color,"width":width,"dash":"dash"})
    presentation=selection_presentation(run_ids,labels,selected,ribbon=ribbon is not None,
        comparison_run=comparison_run,change_regions=change_regions,overlay=overlay)
    for annotation in presentation["annotations"]:
        fig.add_annotation(**annotation)
    for shape in presentation["shapes"]:
        fig.add_shape(**shape)
    if region:
        fig.update_xaxes(range=list(region))
    finite = np.isfinite(raw)
    stats = {
        "saturated_cells": int(np.sum(finite & (raw > high))),
        "log_masked_cells": int(np.sum(finite & (raw <= 0))) if scale == "log" else 0,
        "reference_range_exceeded": any(
            p.x[0] < (grid_reference or patterns[0]).x[0]
            or p.x[-1] > (grid_reference or patterns[0]).x[-1] for p in patterns
        ),
        "display_cells": int(raw.size),
    }
    return fig, stats


def boundary_figure(records, values, axis_title):
    """Show the actual selected-rule count, not an invented similarity probability."""
    if not records:
        return empty_figure("The first received pattern starts the map")
    fig = go.Figure()
    symbols = ["circle-open", "diamond-open", "square-open", "triangle-up-open",
               "hexagon-open", "star-open", "cross-open", "triangle-down-open"]
    selected_count=0;required=1
    for row,x in zip(records,values):
        if x is None:
            continue
        decision=row.get("boundary_comparison") or {}
        index=row.get("style_index")
        style=marker_style(index)
        if decision:
            selected_count=max(selected_count,len(decision.get("selected",[])))
            required=decision.get("required",1)
            unknown=sum(v.get("selected",False) and not v.get("available",False)
                for v in decision.get("indicators",{}).values())
            value=decision.get("votes",0)
            label="Rule met" if decision.get("status")=="transition_evidence" else "Incomplete comparison" if decision.get("status")=="uncertain" else "Rule not met"
        else:
            value=-.55;unknown=0;label="Initial reference / no comparison"
        fig.add_trace(go.Scatter(x=[x],y=[value],mode="markers",
            marker={"color":style["color"] if decision.get("status")!="uncertain" and decision else "#718077",
                "size":12,"symbol":symbols[index % len(symbols)] if index is not None and decision else "x-open","line":{"width":2}},
            error_y={"type":"data","symmetric":False,"array":[unknown],"arrayminus":[0],"color":"#718077"},
            customdata=[[row["run_id"],label,value if decision else "not compared",decision.get("before","—"),unknown]],
            hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>Indicators triggered: %{customdata[2]}<br>Reference: %{customdata[3]}<br>Unavailable selected indicators: %{customdata[4]}<extra></extra>",
            showlegend=False))
    if selected_count:
        fig.add_hline(y=required,line_dash="dash",line_color="#56635b",line_width=1,
            annotation_text=f"Rule: at least {required} of {selected_count}",annotation_position="top left")
    fig.update_layout(template="plotly_white", height=230,margin={"l":85,"r":20,"t":15,"b":55},
        xaxis_title=axis_title, yaxis={"title":"Indicators triggered", "tickmode":"array",
            "tickvals":[-.55,*range(selected_count+1)],"ticktext":["No comparison",*[str(i) for i in range(selected_count+1)]],
            "range":[-.9,max(1,selected_count)+.5]},
        meta={"run_ids":[r["run_id"] for r in records],"axis_title":axis_title},
        font={"family":"Arial, sans-serif","size":12})
    return fig


def profile_figure(
    current,
    *,
    comparison=None,
    comparison_label="Earlier observation",
    pair=None,
    region=None,
    fit=None,
    mode="observed",
):
    if mode == "fit":
        if fit is None:
            return empty_figure(
                "Refinement is pending or unavailable; observed tracking remains available"
            )
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.75, 0.25],
            vertical_spacing=0.07,
        )
        for name, values, mask, color in (
            ("Observed", fit.y, fit.valid, "#087b44"),
            ("Calculated", fit.calculated, fit.fitted, "#3867b0"),
            ("Observed − calculated", fit.y - fit.calculated, fit.fitted, "#718077"),
        ):
            values = [
                float(v) if ok and np.isfinite(v) else None
                for v, ok in zip(values, mask)
            ]
            fig.add_trace(
                go.Scatter(
                    x=fit.x.tolist(),
                    y=values,
                    name=name,
                    mode="lines",
                    line={"color": color, "width": 1.2},
                ),
                row=2 if name.startswith("Observed −") else 1,
                col=1,
            )
        coordinate = fit.packet.get("coordinate") or {}
        xlabel = {
            "tof": "Time of flight (μs)",
            "d": "d-spacing (Å)",
            "d_spacing": "d-spacing (Å)",
        }.get(coordinate.get("kind"), "Native histogram coordinate")
        fig.update_yaxes(title_text="Fit difference", row=2, col=1)
        fig.update_xaxes(title_text=xlabel, row=2, col=1)
    else:
        fig = go.Figure()
        gain = float(
            np.exp((pair or {}).get("metrics", {}).get("gain_intercept_log", 0))
        )
        for name, p, color, multiplier in (
            ("Selected observed", current, "#087b44", 1),
            (comparison_label, comparison, "#89948e", gain),
        ):
            if p is None:
                continue
            values = [
                float(v * multiplier) if ok and np.isfinite(v) else None
                for v, ok in zip(p.intensity, p.valid)
            ]
            fig.add_trace(
                go.Scatter(
                    x=p.x.tolist(),
                    y=values,
                    name=name,
                    mode="lines",
                    line={"color": color, "width": 1.3},
                )
            )
        fig.update_xaxes(title_text=COORDINATES[current.scope.coordinate])
        for event in (pair or {}).get("regions", []):
            lo, hi = event.get("interval", (None, None))
            if lo is not None and hi is not None:
                fig.add_vrect(
                    x0=lo,
                    x1=hi,
                    line_width=0,
                    fillcolor="rgba(194,131,46,.12)",
                    layer="below",
                )
    fig.update_layout(
        template="plotly_white",
        height=390,
        autosize=True,
        margin={"l": 60, "r": 14, "t": 52, "b": 52},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.16, "font": {"size": 11}},
        yaxis_title="Intensity",
        font={"family": "Arial, sans-serif", "size": 12},
        uirevision=current.observation_id + ":" + str(region) + ":" + mode,
    )
    if region:
        lo, hi = region
        fig.update_xaxes(range=[lo, hi])
        if mode != "fit":
            values = np.concatenate(
                [
                    p.intensity[p.valid & (p.x >= lo) & (p.x <= hi)]
                    for p in [current, comparison]
                    if p is not None
                ]
            )
            if len(values):
                low, high = float(np.min(values)), float(np.max(values))
                pad = max((high - low) * 0.1, 1e-12)
                fig.update_yaxes(range=[low - pad, high + pad])
    return fig
