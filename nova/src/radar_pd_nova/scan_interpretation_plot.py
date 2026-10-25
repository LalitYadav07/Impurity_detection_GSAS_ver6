"""Evidence overlays retain native bins and gaps; zoom never changes inference."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def observation_figure(pattern, *, label, show_fit=True):
    """A first scan remains inspectable before a compatible pair exists."""
    fig = go.Figure()
    for name, values, mask, color in [
        ("Current observed", pattern.y, pattern.valid, "#006b4f"),
        ("Current calculated", pattern.calculated, pattern.fitted, "#488ee0"),
    ]:
        if name == "Current calculated" and not show_fit:
            continue
        fig.add_trace(go.Scatter(
            x=pattern.x,
            y=[float(v) if ok and np.isfinite(v) else None for v, ok in zip(values, mask)],
            name=name, mode="lines", line={"color": color, "width": 1.3},
        ))
    coordinate = pattern.packet["coordinate"]
    kind = {"tof": "Time of flight", "d": "d-spacing", "d_spacing": "d-spacing"}.get(
        coordinate["kind"], coordinate["kind"].replace("_", " ")
    )
    unit = {"us": "μs", "angstrom": "Å", "A": "Å"}.get(coordinate["unit"], coordinate["unit"])
    fig.update_layout(
        template="plotly_white", autosize=True,
        margin={"l": 58, "r": 12, "t": 62, "b": 48},
        legend={"orientation": "h", "y": 1.18, "x": 0, "font": {"size": 11}},
        xaxis_title=f"{kind} ({unit})", yaxis_title="Intensity", hovermode="x unified",
        font={"family": "Arial, sans-serif", "size": 12},
        meta={"current_run_id": label, "reference_run_id": None}, uirevision=pattern.key,
    )
    return fig


def comparison_figure(
    before, after, comparison, *, before_label, after_label, region=None, show_fit=True
):
    features = comparison.get("features") or {}
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3]
    )
    if features.get("status") != "compared":
        fig.add_annotation(
            text=features.get("reason", "Comparable measured curves are unavailable"),
            showarrow=False,
        )
        return fig
    scale, offset = features["scale"], features["offset"]

    def masked(values, valid):
        return [float(v) if ok and np.isfinite(v) else None for v, ok in zip(values, valid)]

    fig.add_trace(
        go.Scatter(
            x=after.x,
            y=masked(after.y, after.valid),
            name="Current observed",
            hovertemplate=after_label + "<br>x=%{x:.6g}<br>Observed=%{y:.6g}<extra></extra>",
            mode="lines",
            line={"color": "#006b4f", "width": 1.3},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=before.x,
            y=masked(scale * before.y + offset, before.valid),
            name="Reference (adjusted)",
            hovertemplate=before_label
            + "<br>x=%{x:.6g}<br>Adjusted reference=%{y:.6g}<extra></extra>",
            mode="lines",
            line={"color": "#c56e20", "width": 1.1},
        ),
        row=1,
        col=1,
    )
    if show_fit:
        fig.add_trace(
            go.Scatter(
                x=after.x,
                y=masked(after.calculated, after.fitted),
                name="Current calculated",
                hovertemplate=after_label + "<br>x=%{x:.6g}<br>Calculated=%{y:.6g}<extra></extra>",
                mode="lines",
                line={"color": "#488ee0", "width": 1},
            ),
            row=1,
            col=1,
        )
    positions = np.searchsorted(after.x, before.x)
    safe = np.minimum(positions, len(after.x) - 1)
    valid = (positions < len(after.x)) & np.isclose(before.x, after.x[safe], rtol=1e-9, atol=1e-8)
    valid &= before.valid & after.valid[safe]
    difference = after.y[safe] - scale * before.y - offset
    fig.add_trace(
        go.Scatter(
            x=before.x,
            y=masked(difference, valid),
            name="Difference",
            showlegend=False,
            mode="lines",
            line={"color": "#5b526d", "width": 1},
        ),
        row=2,
        col=1,
    )
    coordinate = after.packet["coordinate"]
    coordinate_label = {"tof": "Time of flight", "d": "d-spacing", "d_spacing": "d-spacing", "two_theta": "2θ"}.get(
        coordinate["kind"], coordinate["kind"].replace("_", " ")
    )
    unit_label = {"us": "μs", "A": "Å", "angstrom": "Å"}.get(coordinate["unit"], coordinate["unit"])
    fig.update_xaxes(title_text=f"{coordinate_label} ({unit_label})", row=2, col=1)
    fig.update_yaxes(title_text="Intensity", row=1, col=1)
    fig.update_yaxes(title_text="Difference", row=2, col=1)
    fig.add_hline(y=0, line_width=1, line_color="#c3ccc7", row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin={"l": 58, "r": 12, "t": 62, "b": 48},
        legend={"orientation": "h", "y": 1.18, "x": 0, "font": {"size": 11}},
        font={"family": "Arial, sans-serif", "size": 12},
        meta={"current_run_id": after_label, "reference_run_id": before_label},
        hovermode="x unified",
        uirevision=f"{before.key}:{after.key}:{region}",
    )
    if region:
        lo, hi = region
        pad = max((hi - lo) * 1.5, np.median(np.diff(after.x)) * 10)
        fig.update_xaxes(range=[max(after.x[0], lo - pad), min(after.x[-1], hi + pad)])
        # Local intensity limits make a weak feature inspectable even when
        # a very strong reflection elsewhere dominates the full pattern.
        current = (after.x >= lo - pad) & (after.x <= hi + pad) & after.valid
        old = (before.x >= lo - pad) & (before.x <= hi + pad) & before.valid
        values = np.r_[after.y[current], (scale * before.y + offset)[old]]
        if len(values):
            lower, upper = float(np.min(values)), float(np.max(values))
            padding = max((upper - lower) * 0.1, 1e-6)
            fig.update_yaxes(range=[lower - padding, upper + padding], row=1, col=1)
        local_difference = difference[valid & (before.x >= lo - pad) & (before.x <= hi + pad)]
        if len(local_difference):
            extent = max(float(np.max(np.abs(local_difference))) * 1.12, 1e-6)
            fig.update_yaxes(range=[-extent, extent], row=2, col=1)
    return fig
