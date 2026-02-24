from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plot_payload import load_plot_payload


def discover_plot_payload_files(run_dir: Path) -> List[Path]:
    roots = [
        run_dir / "Results" / "Plots",
        run_dir / "Diagnostics",
        run_dir / "plots",
        run_dir / run_dir.name / "plots",
    ]
    out: List[Path] = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.plotdata.json"):
            rp = str(p.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            out.append(p)
    out.sort(key=lambda p: str(p).lower())
    return out


def _as_array(payload: Dict[str, Any], key: str, default=None):
    arrays = payload.get("arrays", {})
    if key in arrays:
        return np.asarray(arrays[key])
    if key in payload:
        return np.asarray(payload[key])
    return default


def _fig_gsas_fit(payload: Dict[str, Any]) -> Optional[go.Figure]:
    x = _as_array(payload, "x")
    yobs = _as_array(payload, "yobs")
    ycalc = _as_array(payload, "ycalc")
    resid = _as_array(payload, "resid")
    resid_scaled = _as_array(payload, "resid_scaled")
    resid_base = payload.get("resid_base", 0.0)
    inst = str(payload.get("instrument_type", "")).upper()

    if x is None or yobs is None or ycalc is None:
        return None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.72, 0.28], subplot_titles=("Fit", "Bragg ticks"))

    fig.add_trace(go.Scattergl(x=x, y=yobs, mode="markers", name="Observed",
                               marker=dict(size=3, color="#222")), row=1, col=1)
    fig.add_trace(go.Scattergl(x=x, y=ycalc, mode="lines", name="Calculated",
                               line=dict(color="#d62728", width=1.5)), row=1, col=1)

    if resid_scaled is not None:
        fig.add_trace(go.Scattergl(x=x, y=resid_scaled, mode="lines", name="Difference (scaled)",
                                   line=dict(color="#2ca02c", width=1.0)), row=1, col=1)
        fig.add_hline(y=float(resid_base), line_width=1, line_color="#8a8a8a", row=1, col=1)
    elif resid is not None:
        fig.add_trace(go.Scattergl(x=x, y=resid, mode="lines", name="Difference",
                                   line=dict(color="#2ca02c", width=1.0)), row=1, col=1)

    phase_ticks = payload.get("phase_ticks", {}) or {}
    phase_order = payload.get("phase_order", list(phase_ticks.keys()))
    phase_labels = payload.get("phase_labels", {}) or {}
    phase_weights = payload.get("phase_weights", {}) or {}
    y_tick_vals = []
    y_tick_text = []
    for idx, phase_name in enumerate(phase_order):
        ticks = np.asarray(phase_ticks.get(phase_name, []), dtype=float)
        y_tick_vals.append(float(idx))
        lbl = str(phase_labels.get(phase_name) or phase_name)
        wt = phase_weights.get(phase_name, None)
        if wt is not None:
            try:
                lbl = f"{lbl} ({float(wt):.1f}%)"
            except Exception:
                pass
        y_tick_text.append(lbl)
        if ticks.size == 0:
            continue
        y = np.full_like(ticks, fill_value=float(idx), dtype=float)
        fig.add_trace(
            go.Scattergl(
                x=ticks,
                y=y,
                mode="markers",
                name=f"Ticks: {phase_name}",
                marker=dict(symbol="line-ns-open", size=12, color="#d1495b", line=dict(width=1, color="#8b1e2d")),
                hovertemplate=f"{phase_name}<br>x=%{{x:.4f}}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Axis labels (avoid raw TeX artifacts such as "$2\\theta$")
    if inst == "CW":
        x_label = "2θ (degrees)"
    elif inst == "TOF":
        x_label = "Time-of-Flight (μs)"
    else:
        x_label = payload.get("x_label", "x")

    fig.update_yaxes(
        title_text="Intensity",
        title_font=dict(size=16, color="#000000"),
        row=1,
        col=1,
        showline=True,
        linewidth=1.2,
        linecolor="#2b2b2b",
        gridcolor="#d6dce5",
        zeroline=False,
        tickfont=dict(color="#000000"),
    )
    fig.update_yaxes(
        title_text="Phases",
        title_font=dict(size=16, color="#000000"),
        row=2,
        col=1,
        tickmode="array",
        tickvals=y_tick_vals,
        ticktext=y_tick_text,
        showline=True,
        linewidth=1.2,
        linecolor="#2b2b2b",
        gridcolor="#e2e6ec",
        zeroline=False,
        tickfont=dict(color="#000000"),
    )
    fig.update_xaxes(
        title_text=x_label,
        title_font=dict(size=16, color="#000000"),
        row=2,
        col=1,
        showline=True,
        linewidth=1.2,
        linecolor="#2b2b2b",
        gridcolor="#d6dce5",
        zeroline=False,
        tickfont=dict(color="#000000"),
    )

    title = payload.get("title") or payload.get("source_plot") or "Interactive fit"
    fig.update_layout(
        height=800,
        title=dict(text=title, font=dict(size=24, color="#1d2433")),
        template="plotly_white",
        font=dict(size=14, color="#1f2937", family="Arial, Helvetica, sans-serif"),
        legend=dict(font=dict(size=14), bgcolor="rgba(255,255,255,0.92)", bordercolor="#c7ced9", borderwidth=1),
        margin=dict(l=120, r=30, t=70, b=60),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )

    fig.update_traces(selector=dict(name="Observed"), marker=dict(size=4, color="#111827"))
    fig.update_traces(selector=dict(name="Calculated"), line=dict(color="#d62728", width=2.0))
    fig.update_traces(selector=dict(name="Difference (scaled)"), line=dict(color="#2ca02c", width=1.8))

    return fig


def _fig_ml_hist(payload: Dict[str, Any]) -> Optional[go.Figure]:
    x = _as_array(payload, "x_active")
    y_res = _as_array(payload, "y_residual")
    candidates = payload.get("candidates", [])
    if x is None or y_res is None or not candidates:
        return None

    n = len(candidates)
    cols = 2 if n <= 4 else 3
    rows = int(np.ceil(n / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[c.get("label", c.get("phase_id", "candidate")) for c in candidates])

    for i, cand in enumerate(candidates):
        r = i // cols + 1
        c = i % cols + 1
        y_c = np.asarray(cand.get("y_candidate", []), dtype=float)
        if y_c.size == 0:
            continue
        overlap = np.minimum(y_res, y_c)

        # Use bars for histogram visualization
        fig.add_trace(go.Bar(x=x, y=y_res, name="Residual", 
                             marker=dict(color="#222", line=dict(width=0)),
                             opacity=0.7, showlegend=(i == 0)), row=r, col=c)
        fig.add_trace(go.Bar(x=x, y=y_c, name="Scaled candidate",
                             marker=dict(color="#1f77b4", line=dict(width=0)),
                             opacity=0.6, showlegend=(i == 0)), row=r, col=c)
        fig.add_trace(go.Bar(x=x, y=overlap, name="Overlap",
                             marker=dict(color="#2ca02c", line=dict(width=0)),
                             opacity=0.5, showlegend=(i == 0)), row=r, col=c)

        fig.update_xaxes(title_text="Q (Å⁻¹)", title_font=dict(size=14, color="#000000"), tickfont=dict(color="#000000"), row=r, col=c)
        fig.update_yaxes(title_text="Norm. Intensity", title_font=dict(size=14, color="#000000"), tickfont=dict(color="#000000"), row=r, col=c)

    fig.update_layout(
        height=max(420, 280 * rows),
        title=dict(text=payload.get("title", "Interactive ML histogram diagnostics"), font=dict(size=20, color="#000000")),
        template="plotly_white",
        font=dict(size=14, color="#000000"),
        barmode='overlay',
    )
    return fig


def _fig_resid_debug(payload: Dict[str, Any]) -> Optional[go.Figure]:
    q = _as_array(payload, "Q")
    r = _as_array(payload, "R")
    centers = _as_array(payload, "centers")
    h = _as_array(payload, "H")
    c = _as_array(payload, "C")
    q_peaks = _as_array(payload, "Q_main_peaks")
    peak_mask_width = float(payload.get("peak_mask_width", 0.0))

    if q is None or r is None or centers is None or h is None:
        return None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Residual with peak masking", "64-bin histogram"))

    fig.add_trace(go.Scattergl(x=q, y=r, mode="lines", name="Residual R(Q)", line=dict(color="#111")), row=1, col=1)

    if q_peaks is not None and peak_mask_width > 0:
        for q0 in q_peaks:
            ql = float(q0) - peak_mask_width
            qh = float(q0) + peak_mask_width
            fig.add_vrect(x0=ql, x1=qh, fillcolor="rgba(255,179,179,0.25)", line_width=0, row=1, col=1)

    fig.add_trace(go.Scatter(x=centers, y=h, mode="lines", name="H_final",
                             line=dict(color="#2ca02c", width=2, shape="hv")), row=2, col=1)

    if c is not None and len(c) == len(centers):
        fig.add_trace(go.Bar(x=centers, y=c, name="Samples per bin", marker_opacity=0.2,
                             marker_color="#7f7f7f"), row=2, col=1)

    fig.update_xaxes(title_text="Q (Å⁻¹)", title_font=dict(size=16, color="#000000"), tickfont=dict(color="#000000"), row=2, col=1)
    fig.update_yaxes(title_text="Residual", title_font=dict(size=16, color="#000000"), tickfont=dict(color="#000000"), row=1, col=1)
    fig.update_yaxes(title_text="Bin mass / count", title_font=dict(size=16, color="#000000"), tickfont=dict(color="#000000"), row=2, col=1)
    fig.update_layout(height=720, title=payload.get("title", "Interactive residual debug"), template="plotly_white")
    return fig


def build_plotly_figure_from_payload(payload: Dict[str, Any]) -> Optional[go.Figure]:
    kind = payload.get("plot_kind", "")
    if kind == "gsas_fit_with_ticks_v1":
        return _fig_gsas_fit(payload)
    if kind == "ml_hist_grid_v1":
        return _fig_ml_hist(payload)
    if kind == "residual_bin_debug_v1":
        return _fig_resid_debug(payload)
    return None


def load_interactive_payload(json_path: Path) -> Dict[str, Any]:
    return load_plot_payload(str(json_path))
