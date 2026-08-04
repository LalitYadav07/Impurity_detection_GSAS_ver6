from pathlib import Path
from typing import Any, Dict, List, Optional
import re

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


def _short_label(value: Any, limit: int = 72) -> str:
    text = str(value or "").strip()
    if not text:
        return "candidate"
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def _axis_label(payload: Dict[str, Any], inst: str) -> str:
    if inst == "CW":
        return "2-theta (degrees)"
    if inst == "TOF":
        return "Time-of-flight (us)"
    raw = str(payload.get("x_label") or "x")
    return raw.replace("$2\\theta$", "2-theta")


def _phase_color(index: int) -> str:
    palette = [
        "#0f766e",
        "#b91c1c",
        "#2563eb",
        "#9333ea",
        "#c2410c",
        "#047857",
        "#be185d",
        "#4f46e5",
    ]
    return palette[index % len(palette)]


def _phase_formula(label: str, fallback: str) -> str:
    text = str(label or fallback or "phase").strip()
    if re.search(r"\(SG\s*[^)]*\)", text, flags=re.IGNORECASE):
        return text
    match = re.search(r"\((.*?)\)", text)
    if match:
        inside = match.group(1).strip()
        pieces = re.split(r"\s+(?:-|\u2013|\u2014)\s+", inside, maxsplit=1)
        formula = pieces[0].strip() if pieces else ""
        if len(pieces) > 1:
            sg_match = re.search(r"(?:SG\s*)?([0-9]+)", pieces[1], flags=re.IGNORECASE)
            if formula and sg_match:
                return f"{formula} (SG {sg_match.group(1)})"
        if formula and not re.fullmatch(r"SG\s*[0-9]+", formula, flags=re.IGNORECASE):
            return formula
    return text if text else str(fallback or "phase")


def _phase_axis_label(label: str, phase_id: str, weight: Any) -> str:
    formula = _phase_formula(label, phase_id)
    try:
        return f"{formula}  {float(weight):.2g}%"
    except Exception:
        return formula


def _candidate_formula(candidate: Dict[str, Any]) -> str:
    label = str(candidate.get("label") or "").strip()
    if label:
        formula = re.split(r"\s+\[SG\s+", label, maxsplit=1)[0].strip()
        if formula:
            return formula
    return str(candidate.get("phase_id") or "candidate")


def _candidate_title(candidate: Dict[str, Any]) -> str:
    label = _short_label(_candidate_formula(candidate), 24)
    bits = []
    try:
        bits.append(f"score {float(candidate.get('score')):.3f}")
    except Exception:
        pass
    try:
        bits.append(f"cos {float(candidate.get('cosine')):.2f}")
    except Exception:
        pass
    return label if not bits else f"{label} | {' | '.join(bits)}"


def _clean_plot_title(value: Any, fallback: str = "Interactive plot") -> str:
    text = str(value or fallback).strip()
    lower = text.lower().replace("_", " ")
    pass_match = re.search(r"pass\s*(\d+)", lower)
    if "main phase" in lower:
        return "Main phase fit"
    if "accepted" in lower:
        return f"Accepted model - pass {pass_match.group(1)}" if pass_match else "Accepted model"
    if "trial" in lower or "blend" in lower:
        return f"Trial blend - pass {pass_match.group(1)}" if pass_match else "Trial blend"
    if "histogram" in lower or "ml" in lower:
        if "stage0" in lower:
            return "ML candidate histogram - stage 0"
        return f"ML candidate histogram - pass {pass_match.group(1)}" if pass_match else "ML candidate histogram"
    return _short_label(text.replace("_", " "), 96)


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

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=[0.58, 0.24, 0.18],
        subplot_titles=("Fit", "Difference", "Phase peak positions"),
    )

    fig.add_trace(
        go.Scattergl(
            x=x,
            y=yobs,
            mode="markers",
            name="Observed",
            marker=dict(size=3, color="#111827", opacity=0.66),
            hovertemplate="x=%{x:.5g}<br>Observed=%{y:.5g}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=ycalc,
            mode="lines",
            name="Calculated",
            line=dict(color="#b91c1c", width=1.9),
            hovertemplate="x=%{x:.5g}<br>Calculated=%{y:.5g}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if resid is not None:
        residual_y = resid
        residual_name = "Difference"
        residual_zero = 0.0
    elif resid_scaled is not None:
        residual_y = resid_scaled
        residual_name = "Difference (scaled)"
        residual_zero = float(resid_base)
    else:
        residual_y = None
        residual_name = "Difference"
        residual_zero = 0.0
    if residual_y is not None:
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=residual_y,
                mode="lines",
                name=residual_name,
                line=dict(color="#0f766e", width=1.45),
                hovertemplate="x=%{x:.5g}<br>Difference=%{y:.5g}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=float(residual_zero), line_width=1, line_color="#64748b", row=2, col=1)

    phase_ticks = payload.get("phase_ticks", {}) or {}
    phase_major_ticks = payload.get("phase_major_ticks", {}) or {}
    phase_major_tick_details = payload.get("phase_major_tick_details", {}) or {}
    phase_order = payload.get("phase_order", list(phase_ticks.keys()))
    phase_labels = payload.get("phase_labels", {}) or {}
    phase_weights = payload.get("phase_weights", {}) or {}
    y_tick_vals = []
    y_tick_text = []
    for idx, phase_name in enumerate(phase_order):
        ticks = np.asarray(phase_ticks.get(phase_name, []), dtype=float)
        label = str(phase_labels.get(phase_name) or phase_name)
        weight = phase_weights.get(phase_name, None)
        y_tick_vals.append(float(idx))
        y_tick_text.append(_short_label(_phase_axis_label(label, phase_name, weight), 30))
        if ticks.size == 0:
            continue
        color = _phase_color(idx)
        y = np.full_like(ticks, fill_value=float(idx), dtype=float)
        fig.add_trace(
            go.Scattergl(
                x=ticks,
                y=y,
                mode="markers",
                name=f"Peaks: {_phase_formula(label, phase_name)}",
                marker=dict(symbol="line-ns-open", size=11, color=color, line=dict(width=1.2, color=color)),
                hovertemplate=f"{_short_label(label, 95)}<br>x=%{{x:.5g}}<extra></extra>",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        major_points = []
        for rank, item in enumerate(phase_major_tick_details.get(phase_name, []) or [], start=1):
            try:
                x_val = float(item.get("x"))
            except Exception:
                continue
            if not np.isfinite(x_val):
                continue
            try:
                rel_strength = float(item.get("relative_strength"))
            except Exception:
                rel_strength = 0.0
            major_points.append(
                [
                    x_val,
                    item.get("rank", rank),
                    item.get("hkl", ""),
                    rel_strength if np.isfinite(rel_strength) else 0.0,
                ]
            )
        if not major_points:
            for rank, x_val in enumerate(np.asarray(phase_major_ticks.get(phase_name, []), dtype=float), start=1):
                if np.isfinite(x_val):
                    major_points.append([float(x_val), rank, "", 0.0])
        major_x = np.asarray([item[0] for item in major_points], dtype=float)
        if major_x.size:
            major_y = np.full_like(major_x, fill_value=float(idx), dtype=float)
            customdata = np.asarray([[item[1], item[2], item[3]] for item in major_points], dtype=object)
            fig.add_trace(
                go.Scattergl(
                    x=major_x,
                    y=major_y,
                    mode="markers",
                    name=f"Key peaks: {_phase_formula(label, phase_name)}",
                    marker=dict(
                        symbol="line-ns-open",
                        size=17,
                        color=color,
                        line=dict(width=3.0, color=color),
                    ),
                    customdata=customdata,
                    hovertemplate=(
                        f"{_short_label(label, 95)}<br>"
                        "Top Bragg peak %{customdata[0]}<br>"
                        "x=%{x:.5g}<br>"
                        "HKL=%{customdata[1]}<br>"
                        "Relative strength=%{customdata[2]:.3f}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=3,
                col=1,
            )

    axis_common = dict(
        showline=True,
        linewidth=1.1,
        linecolor="#334155",
        gridcolor="#e2e8f0",
        zeroline=False,
        tickfont=dict(color="#111827", size=11),
        exponentformat="SI",
        automargin=True,
    )
    fig.update_yaxes(title_text="Intensity", title_font=dict(size=13, color="#111827"), row=1, col=1, **axis_common)
    fig.update_yaxes(title_text="Difference", title_font=dict(size=13, color="#111827"), row=2, col=1, **axis_common)
    fig.update_yaxes(
        title_text="Phases",
        title_font=dict(size=13, color="#111827"),
        row=3,
        col=1,
        tickmode="array",
        tickvals=y_tick_vals,
        ticktext=y_tick_text,
        **axis_common,
    )
    fig.update_xaxes(
        title_text=_axis_label(payload, inst),
        title_font=dict(size=13, color="#111827"),
        row=3,
        col=1,
        showline=True,
        linewidth=1.1,
        linecolor="#334155",
        gridcolor="#e2e8f0",
        zeroline=False,
        tickfont=dict(color="#111827", size=11),
        automargin=True,
        showspikes=True,
        spikemode="across",
        spikecolor="#64748b",
        spikethickness=1,
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1, showspikes=True, spikemode="across", spikecolor="#64748b")
    fig.update_xaxes(showticklabels=False, row=2, col=1, showspikes=True, spikemode="across", spikecolor="#64748b")

    title = _clean_plot_title(payload.get("title") or payload.get("source_plot"), "Interactive fit")
    fig.update_layout(
        height=max(700, 630 + 30 * max(0, len(phase_order) - 2)),
        title=dict(text=_short_label(title, 96), font=dict(size=17, color="#111827"), x=0.01),
        template="plotly_white",
        font=dict(size=11, color="#111827", family="Arial, Helvetica, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#cbd5e1",
            borderwidth=1,
        ),
        hovermode="x unified",
        dragmode="zoom",
        margin=dict(l=82, r=18, t=84, b=52),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        uirevision=str(payload.get("source_plot") or title),
    )
    fig.update_annotations(font_size=12, font_color="#334155")
    return fig


def _fig_ml_hist(payload: Dict[str, Any], max_candidates: Optional[int] = None) -> Optional[go.Figure]:
    x = _as_array(payload, "x_active")
    y_res = _as_array(payload, "y_residual")
    candidates = payload.get("candidates", [])
    if x is None or y_res is None or not candidates:
        return None

    if max_candidates is not None:
        try:
            candidates = candidates[: max(1, int(max_candidates))]
        except Exception:
            pass

    n = len(candidates)
    cols = 1 if n == 1 else (2 if n <= 8 else 3)
    rows = int(np.ceil(n / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[_candidate_title(c) for c in candidates])

    for i, cand in enumerate(candidates):
        r = i // cols + 1
        c = i % cols + 1
        y_c = np.asarray(cand.get("y_candidate", []), dtype=float)
        if y_c.size == 0:
            continue
        overlap = np.minimum(y_res, y_c)

        fig.add_trace(
            go.Bar(
                x=x,
                y=y_c,
                name="Candidate",
                marker=dict(color="#2563eb", line=dict(width=0)),
                opacity=0.28,
                showlegend=(i == 0),
                hovertemplate="Q=%{x:.3f}<br>Candidate=%{y:.4f}<extra></extra>",
            ),
            row=r,
            col=c,
        )
        fig.add_trace(
            go.Bar(
                x=x,
                y=overlap,
                name="Overlap",
                marker=dict(color="#16a34a", line=dict(width=0)),
                opacity=0.62,
                showlegend=(i == 0),
                hovertemplate="Q=%{x:.3f}<br>Overlap=%{y:.4f}<extra></extra>",
            ),
            row=r,
            col=c,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_res,
                mode="lines+markers",
                name="Residual",
                line=dict(color="#111827", width=1.5),
                marker=dict(size=3, color="#111827"),
                showlegend=(i == 0),
                hovertemplate="Q=%{x:.3f}<br>Residual=%{y:.4f}<extra></extra>",
            ),
            row=r,
            col=c,
        )

        fig.update_xaxes(
            title_text="Q (A^-1)" if r == rows else "",
            title_font=dict(size=10, color="#111827"),
            tickfont=dict(size=9, color="#111827"),
            gridcolor="#e2e8f0",
            showline=True,
            linecolor="#334155",
            row=r,
            col=c,
        )
        fig.update_yaxes(
            title_text="Norm. mass" if c == 1 else "",
            title_font=dict(size=10, color="#111827"),
            tickfont=dict(size=9, color="#111827"),
            gridcolor="#e2e8f0",
            showline=True,
            linecolor="#334155",
            row=r,
            col=c,
        )

    fig.update_layout(
        height=max(460, 255 * rows),
        title=dict(text=_clean_plot_title(payload.get("title"), "ML histogram diagnostics"), font=dict(size=16, color="#111827"), x=0.01),
        template="plotly_white",
        font=dict(size=10, color="#111827", family="Arial, Helvetica, sans-serif"),
        barmode="overlay",
        bargap=0.08,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.01, x=0, font=dict(size=10), bgcolor="rgba(255,255,255,0.88)"),
        margin=dict(l=54, r=18, t=92, b=42),
        uirevision=str(payload.get("source_plot") or payload.get("title") or "ml_hist"),
    )
    fig.update_annotations(font_size=10, font_color="#334155")
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

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Residual with peak masking", "64-bin histogram"),
    )

    fig.add_trace(go.Scattergl(x=q, y=r, mode="lines", name="Residual R(Q)", line=dict(color="#111827", width=1.4)), row=1, col=1)

    if q_peaks is not None and peak_mask_width > 0:
        for q0 in q_peaks:
            ql = float(q0) - peak_mask_width
            qh = float(q0) + peak_mask_width
            fig.add_vrect(x0=ql, x1=qh, fillcolor="rgba(248,113,113,0.20)", line_width=0, row=1, col=1)

    fig.add_trace(go.Scatter(x=centers, y=h, mode="lines", name="H final", line=dict(color="#16a34a", width=2, shape="hv")), row=2, col=1)

    if c is not None and len(c) == len(centers):
        fig.add_trace(go.Bar(x=centers, y=c, name="Samples per bin", marker_opacity=0.2, marker_color="#64748b"), row=2, col=1)

    fig.update_xaxes(title_text="Q (A^-1)", title_font=dict(size=13, color="#111827"), tickfont=dict(color="#111827"), row=2, col=1)
    fig.update_yaxes(title_text="Residual", title_font=dict(size=13, color="#111827"), tickfont=dict(color="#111827"), row=1, col=1)
    fig.update_yaxes(title_text="Bin mass / count", title_font=dict(size=13, color="#111827"), tickfont=dict(color="#111827"), row=2, col=1)
    fig.update_layout(
        height=700,
        title=dict(text=_clean_plot_title(payload.get("title"), "Interactive residual debug"), font=dict(size=16, color="#111827"), x=0.01),
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=76, r=18, t=72, b=52),
        uirevision=str(payload.get("source_plot") or payload.get("title") or "residual_debug"),
    )
    fig.update_annotations(font_size=11, font_color="#334155")
    return fig


def build_plotly_figure_from_payload(
    payload: Dict[str, Any],
    *,
    max_hist_candidates: Optional[int] = None,
) -> Optional[go.Figure]:
    kind = payload.get("plot_kind", "")
    if kind == "gsas_fit_with_ticks_v1":
        return _fig_gsas_fit(payload)
    if kind == "ml_hist_grid_v1":
        return _fig_ml_hist(payload, max_candidates=max_hist_candidates)
    if kind == "residual_bin_debug_v1":
        return _fig_resid_debug(payload)
    return None


def load_interactive_payload(json_path: Path) -> Dict[str, Any]:
    return load_plot_payload(str(json_path))
