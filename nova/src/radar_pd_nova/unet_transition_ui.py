"""Measured-curve transition tracker powered by the pinned full-Q U-Net."""

from __future__ import annotations

import asyncio
from collections import Counter
import math
from pathlib import Path
import re

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from trame.widgets import html, plotly, vuetify3 as vuetify

from .scan_interpretation_ui import scan_label
from .scan_interpretation import Pattern, load_pattern
from .scan_conditions import qualified_metadata
from .sample_labels import recorded_sample_title
from .facility_metadata import parse_nexus_timestamp
from .results import discover_plot_payloads, gsas_figure, read_plot_payload
from .transition_coordinate_mapping import mapped_calibration
from .transition_input import digest
from .transition_tracking_plot import COORDINATES, binned_intensity, display_grid, empty_figure
from .unet_transition_engine import (ComparisonSuperseded, FullQTransitionEngine, acquisition_key,
                                     available_orders, condition_order_value, source_signature)


def _display_points(pattern, limit=6000):
    """Peak-preserving min/max display decimation; inference uses all bins."""
    x = pattern.x
    y = np.where(pattern.valid, pattern.intensity, np.nan)
    if len(x) <= limit:
        return x.tolist(), y.tolist()
    edges = np.linspace(0, len(x), limit // 2 + 1, dtype=int)
    selected = {0, len(x) - 1}
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue
        values = y[left:right]
        missing = np.flatnonzero(~np.isfinite(values))
        if len(missing):
            selected.add(left + int(missing[0]))
        if np.isfinite(values).any():
            selected.add(left + int(np.nanargmin(values)))
            selected.add(left + int(np.nanargmax(values)))
    index = sorted(selected)
    return x[index].tolist(), y[index].tolist()


_FEATURE_COLORS = {
    "peak_emergence": "#137a55",
    "peak_disappearance": "#9d3747",
    "relative_intensity_change": "#925a12",
    "peak_shift": "#315fa4",
    "peak_splitting": "#704ca0",
    "peak_merging": "#704ca0",
    "peak_broadening": "#4b6b9c",
    "peak_narrowing": "#4b6b9c",
    "peak_asymmetry_change": "#4b6b9c",
    "background_change": "#73634c",
    "diffuse_scattering_change": "#73634c",
}

_DEFAULT_OVERLAYS = ["peak_emergence", "relative_intensity_change",
                     "peak_broadening", "peak_shift"]
_OVERLAY_LABELS = {
    "relative_intensity_change": "Intensity",
    "peak_broadening": "Broadening",
    "peak_shift": "Shift",
}

_CAUSE_LABELS = {
    "lattice_change_candidate": "Lattice change",
    "magnetic_order_candidate": "Magnetic order",
    "new_phase_candidate": "New phase",
    "structural_change_candidate": "Structural change",
}

# Resolve a heat-map row using the figure that emitted the browser event.
# Trace metadata avoids repeating long acquisition IDs in every raster cell.
_MAP_CLICK_ARGUMENTS = (
    "((point) => point ? [point.data.name, point.x, point.y, "
    "point.data.name === 'Measured heat map' ? "
    "((point.data.meta || {}).run_ids || [])[Math.round(point.y)] : point.customdata, "
    "point.data.meta] : ['', null, null, null, null])(($event.points || [])[0])"
)


def _event_visible(event, selected_types):
    names = event.get("feature_types") or []
    return "uncertain" in selected_types if not names else any(
        name in selected_types for name in names
    )


def _series_color_limits(patterns, mode):
    """One native-intensity range for every scan in the compatible series."""
    low, high = np.inf, -np.inf
    for pattern in patterns:
        values = pattern.intensity[pattern.valid]
        values = values[np.isfinite(values)]
        if mode == "log":
            values = values[values > 0]
        if len(values):
            low = min(low, float(np.min(values)))
            high = max(high, float(np.max(values)))
    if not np.isfinite(low):
        return (1.0, 10.0) if mode == "log" else (0.0, 1.0)
    return low, max(high, low * (1 + 1e-9) if mode == "log" else low + 1e-9)


def _reported_color_limits(rows, mode):
    if any("color_limits" not in row for row in rows):
        return None
    bounds = [(row.get("color_limits") or {}).get(mode) for row in rows]
    bounds = [pair for pair in bounds if pair is not None]
    if not bounds:
        return None
    low = min(pair[0] for pair in bounds)
    high = max(pair[1] for pair in bounds)
    return low, max(high, low * (1 + 1e-9) if mode == "log" else low + 1e-9)


def _map_display_values(raw, mode, limits=None):
    """Apply the same declared global transform and limits to every map row."""
    if limits is None:
        finite = raw[np.isfinite(raw)]
        if mode == "log":
            finite = finite[finite > 0]
        limits = ((float(np.min(finite)), float(np.max(finite))) if len(finite)
                  else ((1.0, 10.0) if mode == "log" else (0.0, 1.0)))
    low, high = limits
    if mode == "log":
        color = np.where(raw > 0, np.log10(np.maximum(raw, np.finfo(float).tiny)), np.nan)
        ticks = np.linspace(np.log10(low), np.log10(high), 4)
        return color, float(ticks[0]), float(ticks[-1]), ticks.tolist(), \
            [f"{10 ** value:.3g}" for value in ticks]
    ticks = np.linspace(low, high, 4)
    return raw, low, high, ticks.tolist(), [f"{value:.3g}" for value in ticks]


def _map_grid(patterns):
    reference = max(patterns, key=lambda pattern: float(pattern.x[-1] - pattern.x[0]))
    centers, edges = display_grid(reference, min(1200, len(reference.x)))
    low = min(float(pattern.x[0]) for pattern in patterns)
    high = max(float(pattern.x[-1]) for pattern in patterns)
    if low < edges[0] or high >= edges[-1]:
        # Compatible scans can still have different cropped coordinate ranges.
        # Keep every scan visible rather than clipping the shorter scan's tail.
        edges = np.linspace(low, np.nextafter(high, np.inf), len(edges))
        centers = (edges[:-1] + edges[1:]) / 2
    return centers, edges


def _visible_ribbon_range(traces, limits):
    """Display bounds from the measured curves in the visible x window only."""
    values = []
    for trace in traces:
        if trace.get('name') not in {'Reference scan', 'Selected scan'}:
            continue
        x = np.asarray(trace['x'], dtype=float)
        y = np.asarray(trace['y'], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & (x >= limits[0]) & (x <= limits[1])
        if valid.any():
            values.extend((float(np.min(y[valid])), float(np.max(y[valid]))))
    if not values:
        return None
    low, high = min(values), max(values)
    padding = .07 * max(high - low, abs(high) * .01, 1e-9)
    return [low - padding, high + padding]


def _inline_map_height(scan_count):
    if scan_count <= 8:
        return 470
    if scan_count <= 30:
        return 550
    if scan_count <= 60:
        return 650
    return 730


def _evolution_figure(view_rows, patterns, scans, selected_id, current, earlier, *,
                      selected_types, selected_event=-1, scale="linear",
                      palette="Turbo", zoom=None, revision=0, series="",
                      color_limits=None, order_label="Acquisition order",
                      order_by="acquired", identity=None, reference_id=None,
                      height=690):
    """One shared-x view: native pair curves over a globally scaled heat map."""
    if not patterns:
        return plotly.Figure.to_data(empty_figure("Waiting for measured scans")), 0, 0
    centers, edges = _map_grid(patterns)
    raw = np.vstack([binned_intensity(pattern, edges) for pattern in patterns])
    color, low, high, tickvals, ticktext = _map_display_values(
        raw, scale, color_limits)
    reference_high = color_limits[1] if color_limits is not None else np.inf
    saturated = int(np.count_nonzero(np.isfinite(raw) & (raw > reference_high)))
    raw_hover = [[f"{value:.5g}" if np.isfinite(value) else ""
                  for value in row] for row in raw]
    count = len(view_rows)
    row_heights = [.50, .50] if count <= 8 else [.38, .62] if count <= 30 else [.30, .70]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=row_heights, vertical_spacing=.045)
    identity = identity or {}
    for pattern, label, tone in ((earlier, "Reference scan", "#87958d"),
                                 (current, "Selected scan", "#183f78")):
        if pattern is None:
            continue
        x, y = _display_points(pattern)
        fig.add_trace(go.Scattergl(
            x=x, y=y, mode="lines", name=label, connectgaps=False,
            meta={**identity, "run_id": selected_id if label == "Selected scan" else reference_id},
            line={"color": tone, "width": 1.6 if label == "Selected scan" else 1.2},
            hovertemplate=f"{label}<br>x=%{{x:.5g}}<br>Measured intensity=%{{y:.5g}}<extra></extra>"
        ), row=1, col=1)
    fig.add_trace(go.Heatmap(
        x=centers.tolist(), y=list(range(count)),
        z=np.where(np.isfinite(color), color, np.nan),
        text=raw_hover,
        zmin=low, zmax=high, colorscale=palette, name="Measured heat map",
        hoverongaps=False, showscale=True,
        meta={**identity, "run_ids": [row["run_id"] for row in view_rows]},
        hovertemplate="Sorted row %{y}<br>x=%{x:.5g}<br>Measured intensity=%{text}<extra></extra>",
        colorbar={"orientation": "h", "x": .53, "y": -.18, "len": .77,
                  "thickness": 12, "tickvals": tickvals,
                  "ticktext": ticktext, "tickfont": {"size": 10}},
    ), row=2, col=1)
    selected_position = next((i for i, row in enumerate(view_rows)
                              if row["run_id"] == selected_id), None)
    bars = {"x": [], "base": [], "y": [], "width": [], "colors": [],
            "tokens": [], "hover": []}
    for row_index, row in enumerate(view_rows):
        for event_index, event in enumerate(row.get("events") or []):
            if not _event_visible(event, selected_types):
                continue
            names = event.get("feature_types") or []
            color_name = next((name for name in names if name in selected_types), "")
            color_name = color_name or (names[0] if names else "")
            color_value = _FEATURE_COLORS.get(color_name, "#677a71")
            bars["x"].append(float(event["x1"] - event["x0"]))
            bars["base"].append(float(event["x0"]))
            bars["y"].append(row_index - .40)
            bars["width"].append(.22 if row["run_id"] == selected_id else .14)
            bars["colors"].append(color_value)
            bars["tokens"].append(f"{row['run_id']}|{event_index}")
            bars["hover"].append(
                f"{scan_label(scans[row['run_id']])}<br>{_feature_label(event)}"
                f"<br>{event['x0']:.5g}–{event['x1']:.5g}"
                + (f"<br>Q {event['q0']:.4g}–{event['q1']:.4g} Å⁻¹"
                   if "q0" in event else "")
                + f"<br>Model attention {_score_label(event['attention_score'])} (not calibrated)"
            )
            if row["run_id"] == selected_id:
                fig.add_vrect(x0=event["x0"], x1=event["x1"],
                              fillcolor=color_value,
                              opacity=.16 if event_index == selected_event else .075,
                              line_color=color_value,
                              line_width=1.6 if event_index == selected_event else .4,
                              row=1, col=1)
    if bars["x"]:
        fig.add_trace(go.Bar(
            x=bars["x"], base=bars["base"], y=bars["y"], width=bars["width"],
            orientation="h", marker={"color": bars["colors"]},
            customdata=bars["tokens"], hovertext=bars["hover"],
            meta=identity,
            name="U-Net overlays", showlegend=False,
            hovertemplate="%{hovertext}<extra></extra>"
        ), row=2, col=1)
    if selected_position is not None:
        fig.add_shape(type="rect", xref="x2", yref="y2",
                      x0=float(edges[0]), x1=float(edges[-1]),
                      y0=selected_position - .5, y1=selected_position + .5,
                      fillcolor="rgba(0,0,0,0)",
                      line={"color": "#15543c", "width": 2})
    tick_indices = set(range(count)) if count <= 12 else set(
        np.linspace(0, count - 1, 12, dtype=int).tolist()
    )
    if selected_position is not None:
        tick_indices.add(selected_position)
    tick_indices = sorted(tick_indices)
    tick_labels = []
    for index in tick_indices:
        scan = scans[view_rows[index]["run_id"]]
        condition = _condition_label(scan, order_by)
        label = re.sub(r"^Exp\s+\d+\s*·\s*", "", scan_label(scan))
        tick_labels.append(f"{label} · {condition.replace(' measured', '')}")
    fig.update_yaxes(title_text="Measured intensity", row=1, col=1)
    row_axis = 'Temperature order' if order_by == 'temperature' else order_label
    fig.update_yaxes(title_text=row_axis, tickmode="array",
                     tickvals=tick_indices, ticktext=tick_labels,
                     range=[count - .5, -.5], fixedrange=True, row=2, col=1)
    fig.update_xaxes(title_text=COORDINATES.get(current.scope.coordinate,
                                               current.scope.coordinate), row=2, col=1)
    fig.update_layout(template="plotly_white", height=height,
                      margin={"l": 170, "r": 24, "t": 25, "b": 82},
                      hovermode="closest", barmode="overlay", dragmode="zoom",
                      legend={"orientation": "h", "y": 1.12},
                      uirevision=f"{series}:{revision}")
    if zoom is not None:
        fig.update_xaxes(range=list(zoom))
        bounds = _visible_ribbon_range([trace.to_plotly_json() for trace in fig.data], zoom)
        if bounds is not None:
            fig.update_yaxes(range=bounds, autorange=False, row=1, col=1)
    return plotly.Figure.to_data(fig), len(bars["x"]), saturated


def _feature_label(event):
    names = event.get("feature_types") or []
    return ", ".join(name.replace("_", " ") for name in names) if names else "Type uncertain"


def _feature_tone(event):
    name = (event.get("feature_types") or [""])[0]
    if name == "peak_emergence":
        return "emergence"
    if name == "peak_disappearance":
        return "disappearance"
    if name == "relative_intensity_change":
        return "intensity"
    if name in {"peak_shift", "peak_splitting", "peak_merging"}:
        return "position"
    if name in {"peak_broadening", "peak_narrowing", "peak_asymmetry_change"}:
        return "shape"
    if name in {"background_change", "diffuse_scattering_change"}:
        return "background"
    return "uncertain"


def _score_label(value):
    # The model stores six decimals. Preserve them so 0.999862 never reads 1.000.
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


_CONDITION_LABELS = {
    "temperature": ("", "K"),
    "magnetic_field": ("Field ", "T"),
    "pressure": ("Pressure ", "GPa"),
    "wavelength": ("Wavelength ", "Å"),
    "voltage": ("Voltage ", None),
    "gas_flow": ("Gas flow ", None),
    "proton_charge": ("Proton charge ", None),
}


def _condition_label(scan, preferred=None):
    fields = ([preferred] if preferred in _CONDITION_LABELS else []) + [
        field for field in _CONDITION_LABELS if field != preferred]
    for field in fields:
        value = condition_order_value(scan, field)
        if value is None:
            continue
        number, unit_key, source = value
        prefix, canonical_unit = _CONDITION_LABELS[field]
        unit = canonical_unit or unit_key
        provenance = "in run title" if source == "nominal title" else source
        return f"{prefix}{number:g} {unit} {provenance}"
    return "Condition unavailable"


def _acquired_label(scan):
    text = str((scan.get("metadata") or {}).get("start_time") or "")
    try:
        return parse_nexus_timestamp(text).strftime("%Y-%m-%d %H:%M %z")
    except (ValueError, TypeError):
        return "Acquisition time unavailable"


def _fit_directory(scan):
    root_value = scan.get("unet_transition_fit_root")
    if not root_value:
        return None
    try:
        root = Path(root_value).resolve()
        fit_dir = root / "Results" / "Plots"
        return fit_dir if fit_dir.is_dir() and fit_dir.resolve().is_relative_to(root) else None
    except (OSError, RuntimeError):
        return None


def _fit_output_stamp(scan):
    """Detect a replaced fit sidecar as well as a newly created fit folder."""
    fit_dir = _fit_directory(scan)
    if fit_dir is None:
        return None
    try:
        entries = []
        for path in fit_dir.iterdir():
            if not (path.name.endswith(".plotdata.json") or path.suffix == ".npz"):
                continue
            if not path.resolve().is_relative_to(fit_dir):
                continue
            stat = path.stat()
            entries.append((path.name, stat.st_size, stat.st_mtime_ns))
        return tuple(sorted(entries))
    except (OSError, RuntimeError):
        return "unavailable"


def _fit_observation_stamp(scan):
    """Refresh a pending compact fit even if the new container reuses a path."""
    path = scan.get("pattern_path") or (scan.get("pattern_observation") or {}).get("local_path")
    if not path:
        return None
    try:
        stat = Path(path).stat()
        return (stat.st_size, stat.st_mtime_ns, stat.st_ino)
    except (OSError, RuntimeError):
        return "unavailable"


def _fit_cache_key(scan, measured_signature=None):
    packet = scan.get("pattern")
    fields = ("ycalc", "fit_valid", "phase_ticks", "phase_order")
    fit_packet = ({key: packet[key] for key in fields if key in packet}
                  if isinstance(packet, dict) else None)
    return str(scan["run_id"]), digest([
        measured_signature or source_signature(scan), scan.get("unet_transition_fit_root"),
        scan.get("fit_payload"), fit_packet, _fit_observation_stamp(scan),
        _fit_output_stamp(scan)])


def _usable_fit_payload(payload, measured=None):
    """Reject empty, malformed, or disjoint fits before attaching them to a scan."""
    if not isinstance(payload, dict):
        return False
    arrays = payload.get("arrays") if isinstance(payload.get("arrays"), dict) else payload
    try:
        x = np.asarray(arrays.get("x"), dtype=float)
        observed = np.asarray(arrays.get("yobs"), dtype=float)
        calculated = np.asarray(arrays.get("ycalc"), dtype=float)
    except (TypeError, ValueError):
        return False
    if (x.ndim != 1 or not 2 <= len(x) <= 100_000 or
            observed.shape != x.shape or calculated.shape != x.shape or
            not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0) or
            np.count_nonzero(np.isfinite(observed) & np.isfinite(calculated)) < 2):
        return False
    if measured is not None:
        left = max(float(x[0]), float(measured.x[0]))
        right = min(float(x[-1]), float(measured.x[-1]))
        if right <= left:
            return False
    return True


def _fit_priority(option):
    name = Path(option.get("path") or "").name.lower()
    accepted = "accepted" in name
    final_joint = "seq_final_joint" in name
    final = "seq_final" in name
    match = re.search(r"pass[_ -]?(\d+)", name)
    pass_number = int(match.group(1)) if match else 0
    tier = (0 if final_joint and accepted else 1 if final and accepted else
            2 if accepted else 3 if "main_phase" in name else 4)
    rwp = option.get("rwp")
    return (tier, -pass_number, float(rwp) if rwp is not None else np.inf, name)


def _fit_figure_for_scan(scan, measured=None):
    """Render published fit evidence only; the U-Net never consumes this data."""
    payload = scan.get("fit_payload")
    source = "published fit"
    if not _usable_fit_payload(payload, measured):
        payload = None
        fit_dir = _fit_directory(scan)
        if fit_dir is not None:
            options = [item for item in discover_plot_payloads(fit_dir)
                       if item.get("category") == "Refinement fits"
                       and str(item.get("path") or "").endswith(".plotdata.json")
                       and Path(item["path"]).resolve().is_relative_to(fit_dir)]
            options.sort(key=_fit_priority)
            for option in options:
                candidate = read_plot_payload(option["path"])
                if _usable_fit_payload(candidate, measured):
                    payload = candidate
                    break
    if payload is None:
        packet = scan.get("pattern")
        if not isinstance(packet, dict) and scan.get("pattern_path"):
            path = Path(scan["pattern_path"])
            if path.suffix.lower() == ".json":
                packet = load_pattern(
                    path, expected_sha256=(scan.get("pattern_observation") or {}).get("sha256"))
        if isinstance(packet, dict):
            fit = Pattern.parse(packet)
            supported = fit.fitted & np.isfinite(fit.calculated)
            if np.count_nonzero(supported) >= 2:
                kind = (packet.get("coordinate") or {}).get("kind")
                axis = {"tof": "Time of flight (µs)", "two_theta": "2θ (degrees)",
                        "d": "d-spacing (Å)", "d_spacing": "d-spacing (Å)",
                        "q": "Q (Å⁻¹)"}.get(kind, "Native diffraction coordinate")
                payload = {
                    "plot_kind": "gsas_fit_with_ticks_v1",
                    "arrays": {"x": fit.x.tolist(), "yobs": fit.y.tolist(),
                               "ycalc": np.where(supported, fit.calculated, np.nan).tolist(),
                               "resid": np.where(supported, fit.y - fit.calculated, np.nan).tolist(),
                               "valid": fit.valid.tolist(), "fit_valid": fit.fitted.tolist()},
                    "phase_ticks": packet.get("phase_ticks") or {},
                    "phase_order": packet.get("phase_order") or [],
                    "x_label": axis,
                }
                source = "published measured-curve receipt"
    if not _usable_fit_payload(payload, measured):
        fig = empty_figure("No RADAR-PD fit available in this view")
        fig.update_layout(height=440)
        return plotly.Figure.to_data(fig), "Measured curve only; calculated fit and Bragg ticks unavailable in this compact view."
    fig = gsas_figure(payload)
    fig.update_layout(height=440, margin={"l": 85, "r": 20, "t": 38, "b": 58},
                      legend={"orientation": "h", "y": 1.17, "font": {"size": 10}})
    ticks = payload.get("phase_ticks") or payload.get("ticks") or {}
    tick_count = 0
    for value in (ticks.values() if isinstance(ticks, dict) else
                  ticks if isinstance(ticks, list) else []):
        positions = (value.get("all") or value.get("positions") or []) if isinstance(value, dict) else value
        if isinstance(positions, (list, tuple)):
            tick_count += len(positions)
    if not tick_count:
        # GSAS plots put the shared x labels on the phase-tick row. Without
        # published ticks that row has no traces, so Plotly hides its axes.
        # Reuse the two populated rows and leave every scientific array intact.
        axis_title = fig.layout.xaxis3.title.text
        fig.data = tuple(trace for trace in fig.data if trace.xaxis != "x3")
        fig.layout.xaxis3 = None
        fig.layout.yaxis3 = None
        fig.layout.yaxis.domain = [.28, 1.]
        fig.layout.yaxis2.domain = [0., .22]
        fig.layout.xaxis.matches = "x2"
        fig.layout.xaxis2.update(matches=None, showticklabels=True,
                                title={"text": axis_title}, automargin=True)
        fig.update_layout(margin={"l": 85, "r": 20, "t": 38, "b": 65})
    note = f"{source}; {tick_count} Bragg tick{'s' if tick_count != 1 else ''} published."
    if not tick_count:
        note = f"{source}; Bragg ticks are not included in this compact view."
    return plotly.Figure.to_data(fig), note


def _display_scale(before, after, mode):
    return lambda y: y, "Measured intensity"


def _pair_figure(before, after, events, selected=None, scale="linear"):
    transform, ylabel = _display_scale(before, after, scale)
    fig = go.Figure()
    for pattern, name, color in ((before, "Earlier", "#2864b4"),
                                 (after, "Current", "#df542d")):
        x, y = _display_points(pattern)
        raw = np.asarray(y, dtype=float)
        fig.add_trace(go.Scattergl(x=x, y=transform(raw), customdata=raw,
                                   mode="lines", name=name,
                                   line={"color": color, "width": 1.6},
                                   connectgaps=False,
                                   hovertemplate=f"{name}<br>x=%{{x:.4g}}<br>raw=%{{customdata:.5g}}<extra></extra>"))
    for index, event in enumerate(events[:200]):
        color = _FEATURE_COLORS.get((event.get("feature_types") or [""])[0], "#697c76")
        active = index == selected
        fig.add_vrect(x0=event["x0"], x1=event["x1"],
                      fillcolor=color, opacity=.20 if active else .10,
                      line_width=2 if active else .5,
                      line_color="#172c25" if active else color)
    fig.update_layout(template="plotly_white", height=300,
                      margin={"l": 60, "r": 20, "t": 28, "b": 52},
                      legend={"orientation": "h", "y": 1.12},
                      xaxis_title=COORDINATES.get(before.scope.coordinate,
                                                  before.scope.coordinate),
                      yaxis_title=ylabel, hovermode="x unified",
                      dragmode="zoom")
    return plotly.Figure.to_data(fig)


def _baseline_figure(pattern):
    x, y = _display_points(pattern)
    fig = go.Figure(go.Scattergl(x=x, y=y, mode="lines", name="Baseline",
                                 line={"color": "#2864b4", "width": 1.7},
                                 connectgaps=False))
    fig.update_layout(template="plotly_white", height=300,
                      margin={"l": 60, "r": 20, "t": 28, "b": 52},
                      xaxis_title=COORDINATES.get(pattern.scope.coordinate,
                                                  pattern.scope.coordinate),
                      yaxis_title="Measured intensity", hovermode="x unified")
    return plotly.Figure.to_data(fig)


def _matched_display(before, after, lo, hi):
    """Display-only interpolation; inference and native evidence remain untouched."""
    x = after.x[(after.x >= lo) & (after.x <= hi)]
    if len(x) < 2:
        return None
    index = np.searchsorted(before.x, x, side="left")
    right = np.clip(index, 0, len(before.x) - 1)
    left = np.clip(right - 1, 0, len(before.x) - 1)
    left[before.x[right] == x] = right[before.x[right] == x]
    dx = np.diff(before.x)
    typical = float(np.median(dx[dx > 0])) if np.any(dx > 0) else np.inf
    valid = (before.valid[left] & before.valid[right] &
             (x >= before.x[0]) & (x <= before.x[-1]) &
             np.isfinite(before.intensity[left]) & np.isfinite(before.intensity[right]) &
             np.isfinite(before.sigma[left]) & np.isfinite(before.sigma[right]) &
             (before.x[right] - before.x[left] <= 2.5 * typical))
    current_index = np.searchsorted(after.x, x)
    valid &= (after.valid[current_index] & np.isfinite(after.intensity[current_index]) &
              np.isfinite(after.sigma[current_index]))
    if np.count_nonzero(valid) < 2:
        return None
    earlier = np.interp(x, before.x, before.intensity)
    earlier_sigma = np.interp(x, before.x, before.sigma)
    delta = after.intensity[current_index] - earlier
    error = np.sqrt(np.maximum(0, earlier_sigma ** 2 + after.sigma[current_index] ** 2))
    # Keep missing coordinates as NaNs. Dropping them would make Plotly join
    # the residual across an excluded region and invent visible evidence.
    return x, np.where(valid, delta, np.nan), np.where(valid, error, np.nan)


def _focus_figure(before, after, event):
    if event is None:
        return plotly.Figure.to_data(empty_figure("Choose a candidate region to inspect"))
    width = float(event["x1"] - event["x0"])
    margin = max(width * 1.5, .7)
    lo = max(float(max(before.x[0], after.x[0])), float(event["x0"] - margin))
    hi = min(float(min(before.x[-1], after.x[-1])), float(event["x1"] + margin))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.13,
                        row_heights=[.7, .3])
    for pattern, name, color in ((before, "Earlier", "#2864b4"),
                                 (after, "Current", "#df542d")):
        inside = (pattern.x >= lo) & (pattern.x <= hi)
        y = np.where(pattern.valid[inside], pattern.intensity[inside], np.nan)
        fig.add_trace(go.Scatter(x=pattern.x[inside], y=y, mode="lines+markers",
                                 name=name, line={"color": color, "width": 2},
                                 marker={"size": 3}, connectgaps=False,
                                 hovertemplate=f"{name}<br>x=%{{x:.4g}}<br>raw=%{{y:.5g}}<extra></extra>"),
                      row=1, col=1)
    paired = _matched_display(before, after, lo, hi)
    if paired is not None:
        x, delta, error = paired
        fig.add_trace(go.Scatter(x=x, y=2 * error, mode="lines", line={"width": 0},
                                 hoverinfo="skip", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=-2 * error, mode="lines", line={"width": 0},
                                 fill="tonexty", fillcolor="rgba(111,127,119,.17)",
                                 name="±2 reported point errors", hoverinfo="skip"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=delta, mode="lines", name="Current − earlier",
                                 line={"color": "#5d3f92", "width": 1.8},
                                 hovertemplate="x=%{x:.4g}<br>difference=%{y:.5g}<extra></extra>"),
                      row=2, col=1)
    fig.add_vrect(x0=event["x0"], x1=event["x1"], fillcolor="#d7b35b",
                  opacity=.14, line_width=1.5, line_color="#8d671a", row="all", col=1)
    fig.add_hline(y=0, line={"color": "#71817a", "width": 1, "dash": "dot"}, row=2, col=1)
    fig.update_layout(template="plotly_white", height=440,
                      margin={"l": 62, "r": 18, "t": 32, "b": 55},
                      legend={"orientation": "h", "y": 1.10},
                      hovermode="x unified", dragmode="zoom")
    fig.update_yaxes(title_text="Raw intensity", row=1, col=1)
    fig.update_yaxes(title_text="Difference", row=2, col=1)
    fig.update_xaxes(title_text=COORDINATES.get(before.scope.coordinate,
                                               before.scope.coordinate), row=2, col=1)
    return plotly.Figure.to_data(fig)


def _history_figure(rows, scans):
    if not rows:
        return plotly.Figure.to_data(empty_figure("Waiting for measured scans"))
    counts = [len(row.get("events") or []) for row in rows]
    colors = ["#9aa6a0" if row["status"] == "initial" else
              "#bd4656" if row["attention"] == "yes" else
              "#31815e" if row["attention"] == "no" else "#c19b41"
              for row in rows]
    labels = [scan_label(scans.get(row["run_id"], {})) for row in rows]
    fig = go.Figure(go.Bar(x=list(range(1, len(rows) + 1)), y=counts,
                           marker_color=colors, customdata=labels,
                           hovertemplate="%{customdata}<br>%{y} local boxes<extra></extra>"))
    fig.update_layout(template="plotly_white", height=190,
                      margin={"l": 46, "r": 14, "t": 15, "b": 42},
                      xaxis_title="Acquisition order", yaxis_title="Local boxes")
    return plotly.Figure.to_data(fig)


class UnetTransitionMixin:
    def _unet_q_calibration(self, scan):
        return mapped_calibration(scan, getattr(self, "_powgen_controller", None),
                                  getattr(self, "records", {}),
                                  service=getattr(self, "service", None))

    def _new_unet_transition_engine(self):
        return FullQTransitionEngine(calibration_resolver=self._unet_q_calibration)

    def _init_unet_transition_tracking(self):
        previous_generation = getattr(self, "_unet_transition_generation", -1)
        previous_task = getattr(self, "_unet_transition_task", None)
        if previous_task is not None and not previous_task.done():
            previous_task.cancel()
        self._optional_unet_sources = ([], [])
        self._unet_transition_engine = self._new_unet_transition_engine()
        self._unet_transition_report = {"records": [], "groups": []}
        self._unet_transition_scans = {}
        self._unet_transition_signature = None
        self._unet_transition_scope = None
        self._unet_transition_generation = previous_generation + 1
        self._unet_transition_request = None
        self._unet_transition_task = None
        self._unet_transition_sources_cache = []
        self._unet_transition_map_ids = []
        self._unet_transition_map_revision = getattr(self, "_unet_transition_map_revision", 0) + 1
        self._unet_transition_map_identity = None
        self._unet_transition_view_identity = None
        self._unet_transition_x_range = None
        self._unet_transition_zoom_revision = getattr(self, "_unet_transition_zoom_revision", -1) + 1
        self._unet_transition_color_limits = None
        self._unet_transition_color_signature = None
        self._unet_transition_fit_cache = {}
        self._unet_transition_fit_signature = None
        state = self.server.state
        for name, value in {
            "transition_evidence_mode": "unet",
            "unet_transition_series": "", "unet_transition_series_options": [],
            "unet_transition_selected": "", "unet_transition_scans": [],
            "unet_transition_scan_cards": [], "unet_transition_series_summary": "No measured series yet",
            "unet_transition_prev_disabled": True, "unet_transition_next_disabled": True,
            "unet_transition_follow": True, "unet_transition_loading": False,
            "unet_transition_loading_message": "", "unet_transition_error": "",
            "unet_transition_order": "acquired",
            "unet_transition_order_options": [{"title": "Acquisition time", "value": "acquired"},
                                         {"title": "Run number", "value": "run_number"}],
            "unet_transition_direction": "ascending",
            "unet_transition_direction_options": [
                {"title": "Low → high / early → late", "value": "ascending"},
                {"title": "High → low / late → early", "value": "descending"},
            ],
            "unet_transition_reference_gap": 1,
            "unet_transition_reference_gap_options": [
                {"title": "Adjacent scans", "value": 1},
                {"title": "Up to 5 scans back", "value": 5},
                {"title": "Up to 20 scans back", "value": 20},
                {"title": "Up to 100 scans back", "value": 100},
                {"title": "First scan as reference", "value": "first"},
            ],
            "unet_transition_summary": "Waiting for measured patterns",
            "unet_transition_context": "The first scan in each compatible series establishes the comparison reference.",
            "unet_transition_notice": "", "unet_transition_events": [],
            "unet_transition_pair_label": "Waiting for measured scans",
            "unet_transition_pair_context": "", "unet_transition_attention_label": "No comparison",
            "unet_transition_cause_summary": "", "unet_transition_causes": [],
            "unet_transition_coverage_note": "", "unet_transition_profile_note": "",
            "unet_transition_attention_tone": "pending", "unet_transition_feature_count": 0,
            "unet_transition_selected_event": -1, "unet_transition_selected_event_number": 0,
            "unet_transition_selected_feature": "", "unet_transition_selected_interval": "",
            "unet_transition_selected_support": "", "unet_transition_event_items": [],
            "unet_transition_empty_features": "No candidate regions in this comparison.",
            "unet_transition_event_prev_disabled": True, "unet_transition_event_next_disabled": True,
            "unet_transition_feature_filter": "all",
            "unet_transition_feature_filter_options": [{"title": "All feature types", "value": "all"}],
            "unet_transition_overview_scale": "linear",
            "unet_transition_color_scale": "linear",
            "unet_transition_color_scale_options": [
                {"title": "Global linear intensity", "value": "linear"},
                {"title": "Global log₁₀ intensity", "value": "log"},
            ],
            "unet_transition_palette": "Turbo",
            "unet_transition_palette_options": [
                {"title": "Turbo", "value": "Turbo"},
                {"title": "Cividis", "value": "Cividis"},
            ],
            "unet_transition_map_window": 30,
            "unet_transition_map_window_options": [30, 60, 100],
            "unet_transition_map_height": 470,
            "unet_transition_map_scroll_padding": 175,
            "unet_transition_map_start": 1,
            "unet_transition_map_scroll_max": 1,
            "unet_transition_map_scan_position": 1,
            "unet_transition_map_scan_max": 1,
            "unet_transition_map_scroll_label": "All scans visible",
            "unet_transition_map_scroll_compact": "0 scans",
            "unet_transition_map_manual_scroll": False,
            "unet_transition_overlay_types": list(_DEFAULT_OVERLAYS),
            "unet_transition_overlay_options": [],
            "unet_transition_overlay_summary": "Select feature types to display model suggestions.",
            "unet_transition_map": plotly.Figure.to_data(empty_figure("Waiting for measured scans")),
            "unet_transition_scale_options": [
                {"title": "Raw intensity", "value": "linear"},
            ],
            "unet_transition_detail": plotly.Figure.to_data(empty_figure("Choose a candidate region to inspect")),
            "unet_transition_scope_summary": "", "unet_transition_model_sha": self._unet_transition_engine.predictor.metadata["checkpoint_sha256"],
            "unet_transition_event_headers": [
                {"title": "Feature", "key": "feature"},
                {"title": "Start", "key": "x0"},
                {"title": "End", "key": "x1"},
                {"title": "Model score", "key": "score"},
            ],
            "unet_transition_history": plotly.Figure.to_data(empty_figure("Waiting for measured scans")),
            "unet_transition_profile": plotly.Figure.to_data(empty_figure("Select a comparison")),
            "unet_transition_reference_fit": plotly.Figure.to_data(empty_figure("No reference scan selected")),
            "unet_transition_selected_fit": plotly.Figure.to_data(empty_figure("No scan selected")),
            "unet_transition_reference_fit_label": "Reference scan fit",
            "unet_transition_selected_fit_label": "Selected scan fit",
            "unet_transition_reference_fit_note": "No reference scan selected.",
            "unet_transition_selected_fit_note": "No scan selected.",
            "unet_transition_reference_has_fit": False,
            "unet_transition_selected_has_fit": False,
            "unet_transition_dialog": False,
            "unet_transition_dialog_figure": plotly.Figure.to_data(empty_figure("Select a comparison")),
            "unet_transition_unavailable": [],
            "unet_transition_model_name": self._unet_transition_engine.predictor.metadata["name"],
            "unet_transition_model_version": self._unet_transition_engine.predictor.metadata["version"],
        }.items():
            setattr(state, name, value)

    def _optional_unet_view_changed(self, **_):
        if not hasattr(self, "_unet_transition_engine"):
            return
        state = self.server.state
        active = (state.transition_evidence_mode == "unet"
                  and state.workspace_view == "experiment"
                  and state.experiment_result_view == "transitions")
        if not active:
            if self._unet_transition_request is not None:
                self._unet_transition_generation += 1
                self._unet_transition_request = None
            state.unet_transition_loading = False
            state.unet_transition_loading_message = ""
            return
        scans, pending = self._optional_unet_sources
        self._refresh_unet_transition_tracking(scans, pending)

    def _unet_transition_tracking_dashboard(self):
        with html.Section(classes="radar-result-card radar-transition-workspace my-3"):
            with html.Div(classes="radar-tracker-header"):
                html.H3("U-Net feature review", classes="radar-card-title")
                vuetify.VBtn(text=("setup_collapsed ? 'Show setup' : 'Focus plots'",),
                             click="setup_collapsed = !setup_collapsed; $nextTick(() => window.setTimeout(() => window.document.querySelectorAll('.js-plotly-plot').forEach(el => el.getBoundingClientRect().width > 0 && window.Plotly && window.Plotly.Plots.resize(el)), 80))",
                             variant="text", size="small")
                vuetify.VBtn(text=("unet_transition_follow ? 'Following latest' : 'Return to latest'",),
                             click=self.follow_unet_transition_latest, variant="tonal", size="small",
                             v_if="!unet_transition_follow")
            vuetify.VProgressLinear(v_if="unet_transition_loading", indeterminate=True,
                                    color="#15543c", height=3)
            html.P("{{ unet_transition_loading_message }}", v_if="unet_transition_loading",
                   classes="radar-section-help")
            vuetify.VAlert(v_if="!!unet_transition_error", text=("unet_transition_error",),
                           type="error", variant="tonal", density="compact")
            with html.Div(classes="radar-tracker-selectors"):
                vuetify.VSelect(label="Sample / bank / reduction series",
                                v_model=("unet_transition_series",),
                                items=("unet_transition_series_options",),
                                update_modelValue=(self.select_unet_transition_series, "[$event]"),
                                density="compact", variant="outlined", hide_details=True)
                vuetify.VAutocomplete(label="Inspect scan", v_model=("unet_transition_selected",),
                                      items=("unet_transition_scans",),
                                      update_modelValue=(self.select_unet_transition_scan, "[$event]"),
                                      density="compact", variant="outlined", hide_details=True)
                with html.Div(classes="radar-tracker-scan-arrows"):
                    vuetify.VBtn("↑ Previous", click=self.previous_unet_transition_scan,
                                 disabled=("unet_transition_prev_disabled",),
                                 variant="outlined", size="small")
                    vuetify.VBtn("↓ Next", click=self.next_unet_transition_scan,
                                 disabled=("unet_transition_next_disabled",),
                                 variant="outlined", size="small")
            with html.Div(classes="radar-tracker-order-controls"):
                vuetify.VSelect(label="Arrange scans by", v_model=("unet_transition_order",),
                                items=("unet_transition_order_options",),
                                update_modelValue=(self.select_unet_transition_order, "[$event]"),
                                density="compact", variant="outlined", hide_details=True)
                vuetify.VSelect(label="Direction", v_model=("unet_transition_direction",),
                                items=("unet_transition_direction_options",),
                                update_modelValue=(self.select_unet_transition_direction, "[$event]"),
                                density="compact", variant="outlined", hide_details=True)
                vuetify.VSelect(label="Compare each scan with", v_model=("unet_transition_reference_gap",),
                                items=("unet_transition_reference_gap_options",),
                                update_modelValue=(self.select_unet_transition_reference_gap, "[$event]"),
                                density="compact", variant="outlined", hide_details=True)
            html.P("{{ unet_transition_series_summary }}", v_show="!unet_transition_loading",
                   classes="radar-tracker-series-summary")
            with html.Div(classes="radar-tracker-pair-heading", v_show="!unet_transition_loading"):
                with html.Div():
                    html.H4("{{ unet_transition_pair_label }}")
                    html.P("{{ unet_transition_pair_context }}", classes="radar-section-help")
                html.Span("{{ unet_transition_attention_label }}",
                          classes=("'radar-tracker-attention ' + unet_transition_attention_tone",))
            with html.Div(classes="radar-tracker-causes", v_if="unet_transition_causes.length && !unet_transition_loading"):
                html.Span("{{ item.label }} · {{ item.score }}{{ item.selected ? ' · suggested' : '' }}",
                          v_for="item in unet_transition_causes",
                          classes=("'radar-tracker-cause ' + (item.selected ? 'selected' : 'below')",))
            html.P("{{ unet_transition_cause_summary }} {{ unet_transition_coverage_note }}",
                   v_if="!!unet_transition_cause_summary && !unet_transition_loading",
                   classes="radar-section-help")
            html.P("{{ unet_transition_notice }}", v_if="!!unet_transition_notice && !unet_transition_loading",
                   classes="radar-tracker-notice")
            with html.Div(classes="radar-tracker-map-controls", v_show="!unet_transition_loading"):
                vuetify.VSelect(label="U-Net feature overlays", v_model=("unet_transition_overlay_types",),
                                items=("unet_transition_overlay_options",), multiple=True,
                                chips=True, closable_chips=True, clearable=True,
                                update_modelValue=(self.select_unet_transition_overlay_types, "[$event]"),
                                density="compact", variant="outlined", hide_details=True,
                                classes="radar-tracker-overlay-select")
                vuetify.VSelect(label="Heat-map color scale", v_model=("unet_transition_color_scale",),
                                items=("unet_transition_color_scale_options",),
                                update_modelValue=(self.select_unet_transition_color_scale, "[$event]"),
                                density="compact", variant="outlined", hide_details=True)
                vuetify.VSelect(label="Palette", v_model=("unet_transition_palette",),
                                items=("unet_transition_palette_options",),
                                update_modelValue=(self.select_unet_transition_palette, "[$event]"),
                                density="compact", variant="outlined", hide_details=True)
                vuetify.VSelect(label="Scans in map", v_model=("unet_transition_map_window",),
                                items=("unet_transition_map_window_options",),
                                update_modelValue=(self.select_unet_transition_map_window, "[$event]"),
                                density="compact", variant="outlined", hide_details=True)
            with html.Div(classes="radar-tracker-map-actions", v_show="!unet_transition_loading"):
                html.Span("{{ unet_transition_overlay_summary }}", classes="radar-section-help")
                with html.Div(classes="radar-tracker-scan-arrows"):
                    vuetify.VBtn("All overlays", click=self.show_all_unet_transition_overlays,
                                 variant="text", size="small")
                    vuetify.VBtn("Clear", click=self.clear_unet_transition_overlays,
                                 variant="text", size="small")
                    vuetify.VBtn("Reset zoom", click=self.reset_unet_transition_map_zoom,
                                 variant="text", size="small")
                    vuetify.VBtn("Expand map", click=self.expand_unet_transition_plot,
                                 variant="text", size="small")
            html.P("{{ unet_transition_selected_feature }} · {{ unet_transition_selected_interval }} · {{ unet_transition_selected_support }}",
                   v_if="unet_transition_selected_event >= 0 && !unet_transition_loading",
                   classes="radar-tracker-selected-inline")
            with html.Div(classes="radar-tracker-map-shell", v_show="!unet_transition_loading"):
                with html.Div(classes="radar-tracker-map-scroll",
                              style=("'height:' + unet_transition_map_height + 'px;padding-top:' + unet_transition_map_scroll_padding + 'px'",)):
                    vuetify.VBtn("↑ Page", click=self.previous_unet_transition_map_page,
                                 disabled=("unet_transition_map_start <= 1",),
                                 variant="text", size="x-small")
                    vuetify.VSlider(label="Inspect scan", v_model=("unet_transition_map_scan_position",),
                                    direction="vertical", reverse=True,
                                    min=1, max=("unet_transition_map_scan_max",), step=1,
                                    disabled=("unet_transition_map_scan_max <= 1",),
                                    update_modelValue=(self.select_unet_transition_map_scan, "[$event]"),
                                    hide_details=True, density="compact", thumb_label=True)
                    vuetify.VBtn("↓ Page", click=self.next_unet_transition_map_page,
                                 disabled=("unet_transition_map_start >= unet_transition_map_scroll_max",),
                                 variant="text", size="x-small")
                    html.Span("{{ unet_transition_map_scroll_compact }}",
                              title=("unet_transition_map_scroll_label",))
                with html.Div(classes="radar-tracker-map-frame"):
                    figure = plotly.Figure(display_mode_bar=True, responsive=True,
                                           style=("'height:' + unet_transition_map_height + 'px;width:100%'",),
                                           relayout=(self.relayout_unet_transition_map, "[$event]"),
                                           click=(self.select_unet_transition_map_click,
                                                  _MAP_CLICK_ARGUMENTS))
                    figure.data = ("unet_transition_map.data",)
                    figure.layout = ("unet_transition_map.layout",)
            with html.Details(classes="radar-tracker-map-help", v_show="!unet_transition_loading"):
                html.Summary("How to read this map")
                html.P("Top: raw measured reference and selected curves. Bottom: compatible scans in the chosen order. Colored marks are model suggestions mapped to the measured axis. The vertical slider selects a scan; Page browses rows. Wider comparisons can reveal cumulative changes; adjacent scans help locate onset. Heat-map colors share one global intensity scale for the series.",
                       classes="radar-section-help")
            with html.Details(classes="radar-tracker-map-help", v_if="!!powgen_convergence_notice && !unet_transition_loading"):
                html.Summary("Fit quality note")
                html.P("{{ powgen_convergence_notice }}", classes="radar-section-help")
            with html.Div(classes="radar-tracker-fit-grid", v_show="!unet_transition_loading"):
                with html.Div(classes="radar-tracker-fit-card"):
                    html.H4("{{ unet_transition_reference_fit_label }}")
                    fit = plotly.Figure(display_mode_bar=True, responsive=True,
                                        style="height:440px;width:100%",
                                        v_if="unet_transition_reference_has_fit")
                    fit.data = ("unet_transition_reference_fit.data",)
                    fit.layout = ("unet_transition_reference_fit.layout",)
                    html.P("{{ unet_transition_reference_fit_note }}", classes="radar-section-help")
                with html.Div(classes="radar-tracker-fit-card"):
                    html.H4("{{ unet_transition_selected_fit_label }}")
                    fit = plotly.Figure(display_mode_bar=True, responsive=True,
                                        style="height:440px;width:100%",
                                        v_if="unet_transition_selected_has_fit")
                    fit.data = ("unet_transition_selected_fit.data",)
                    fit.layout = ("unet_transition_selected_fit.layout",)
                    html.P("{{ unet_transition_selected_fit_note }}", classes="radar-section-help")
            with html.Details(classes="radar-tracker-provenance", v_show="!unet_transition_loading"):
                html.Summary("Model and source details")
                html.P("Full-Q U-Net {{ unet_transition_model_version }} · one 1024-bin pass over Q = 0.25–12 Å⁻¹ · full checkpoint {{ unet_transition_model_sha }}")
                html.P("{{ unet_transition_scope_summary }}")
                html.P("{{ unet_transition_profile_note }}", v_if="!!unet_transition_profile_note")
                html.P("Experimental model trained on deterministic synthetic changes. Local and whole-pair cause scores are uncalibrated suggestions. Missing Q coverage is not evidence of no change.")
            with html.Div(v_if="unet_transition_unavailable.length > 0 && !unet_transition_loading"):
                html.H4("Scans awaiting a usable comparison")
                html.P("{{ item }}", v_for="item in unet_transition_unavailable",
                       classes="radar-section-help")

    def _unet_transition_tracking_dialog(self):
        with vuetify.VDialog(v_model=("unet_transition_dialog",), fullscreen=True):
            with vuetify.VCard(classes="radar-inspector-card"):
                with html.Div(classes="radar-inspector-toolbar"):
                    html.H3("Diffraction evolution · {{ unet_transition_pair_label }}")
                    vuetify.VBtn("Close", click="unet_transition_dialog = false", variant="outlined")
                with html.Div(classes="radar-inspector-frame"):
                    figure = plotly.Figure(display_mode_bar=True, responsive=True,
                                           style="height:100%;width:100%",
                                           relayout=(self.relayout_unet_transition_map, "[$event]"),
                                           click=(self.select_unet_transition_map_click,
                                                  _MAP_CLICK_ARGUMENTS))
                    figure.data = ("unet_transition_dialog_figure.data",)
                    figure.layout = ("unet_transition_dialog_figure.layout",)

    def _unet_transition_sources(self, scans, pending):
        sources = {str(row["run_id"]): dict(row)
                   for row in [*pending, *scans] if row.get("run_id")}
        seen = Counter(str(row.get("run_id")) for row in scans if row.get("run_id"))
        controller = getattr(self, "_powgen_controller", None)
        if controller is not None and str(getattr(self, "_post_analysis_scope", "")).startswith("watch:"):
            settings = getattr(controller, "settings", None)
            root = getattr(controller, "source_directory", None)
            for stage in ("discovered", "submitted", "completed", "failed"):
                for run in getattr(controller.state, stage, {}).values():
                    row = sources.setdefault(run.run_id, {"run_id": run.run_id})
                    meta = qualified_metadata(run.scan_metadata or {})
                    row.update(run_number=run.run_number, instrument=run.instrument,
                               ipts=getattr(settings, "ipts", row.get("ipts")),
                               bank=run.bank or meta.get("source_bank"), metadata=meta,
                               source_path=run.source_path, transition_source_root=root,
                               analysis_status=stage)
                    record = getattr(controller, "records", {}).get(run.run_id)
                    if record is not None and record.local_output_dir is not None:
                        row["unet_transition_fit_root"] = str(record.local_output_dir)
                    if record is not None:
                        row["transition_profile_dataset_id"] = (
                            (getattr(record, "input_dataset_ids", {}) or {}).get("instrument")
                            or row.get("transition_profile_dataset_id"))
                        row["galaxy_job_id"] = getattr(record, "galaxy_job_id", None)
                    summary = run.scientific_summary or {}
                    if not row.get("pattern_observation"):
                        row["pattern_observation"] = summary.get("pattern_observation")
                    if not row.get("pattern_path"):
                        row["pattern_path"] = (row.get("pattern_observation") or {}).get("local_path")
        return sorted([*sources.values(),
                       *(dict(sources[run_id]) for run_id, count in seen.items()
                         if count > 1 and run_id in sources)], key=acquisition_key)

    def _refresh_unet_transition_tracking(self, scans, pending=()):
        if not hasattr(self, "_unet_transition_engine"):
            return
        scope = str(getattr(self, "_post_analysis_scope", ""))
        if scope != self._unet_transition_scope:
            retain = getattr(self._unet_transition_engine, "retain_sources", None)
            if retain is not None:
                retain([])
            self._unet_transition_scope = scope
            self._unet_transition_generation += 1
            self._unet_transition_request = None
            self._unet_transition_signature = None
            self._unet_transition_engine = self._new_unet_transition_engine()
            self.server.state.unet_transition_follow = True
            self._unet_transition_x_range = None
            self._unet_transition_zoom_revision += 1
            self._unet_transition_view_identity = None
            self._unet_transition_color_limits = None
            self._unet_transition_fit_cache.clear()
            self._unet_transition_fit_signature = None
            self._clear_unet_transition_report()
        sources = self._unet_transition_sources(scans, pending)
        self._unet_transition_sources_cache = sources
        self._schedule_unet_transition_tracking(sources)

    def _schedule_unet_transition_tracking(self, sources):
        scope = str(getattr(self, "_post_analysis_scope", ""))
        state = self.server.state
        order = state.unet_transition_order
        descending = state.unet_transition_direction == "descending"
        gap = state.unet_transition_reference_gap
        signatures = []
        fit_keys = {}
        for row in sources:
            measured_signature = source_signature(row)
            # Scientific-result rows gain summaries, artifacts and convergence
            # fields as jobs finish. Those are not observed-pattern inputs.
            # Keep ordering overrides, but let the adapter own measured-input
            # identity rather than hashing the entire evolving result row.
            ordering = {key: row.get(key) for key in (
                "run_id", "run_number", "axis_overrides", "nominal_temperature_K")}
            signatures.append((ordering, measured_signature))
            run_id = str(row["run_id"])
            fit_keys[run_id] = _fit_cache_key(row, measured_signature)
        signature = digest([scope, signatures, order, descending, gap])
        fit_signature = digest(fit_keys)
        fit_changed = fit_signature != self._unet_transition_fit_signature
        if fit_changed:
            valid_fit_keys = set(fit_keys.values())
            self._unet_transition_fit_cache = {
                key: value for key, value in self._unet_transition_fit_cache.items()
                if key in valid_fit_keys}
            self._unet_transition_fit_signature = fit_signature
        if signature == self._unet_transition_signature:
            if self._unet_transition_task is not None:
                # A user can change the order and then change it back while
                # inference is running. Supersede that pending request.
                self._unet_transition_request = None
                self._update_unet_transition_loading()
            if fit_changed:
                self._unet_transition_scans = {str(row["run_id"]): row for row in sources}
                self._render_unet_transition_tracking()
                state.flush()
            return
        self._unet_transition_color_limits = None
        self._unet_transition_color_signature = None
        self._unet_transition_request = (signature, sources, self._unet_transition_generation,
                                    self._unet_transition_engine, order, descending, gap)
        self._update_unet_transition_loading()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            try:
                report = self._unet_transition_engine.evaluate(
                    sources, order_by=order, descending=descending, reference_gap=gap)
            except Exception as error:
                self._unet_transition_request = None
                self._unet_transition_signature = None
                self._clear_unet_transition_report()
                self._update_unet_transition_loading()
                state.unet_transition_error = str(error)
                state.flush()
                return
            self._apply_unet_transition_tracking(report, sources, signature)
            if self._unet_transition_request is not None and self._unet_transition_request[0] == signature:
                self._unet_transition_request = None
            self._update_unet_transition_loading()
            state.flush()
        elif self._unet_transition_task is None:
            self._unet_transition_task = loop.create_task(self._run_unet_transition_tracking())

    def _update_unet_transition_loading(self):
        state = self.server.state
        request = self._unet_transition_request
        state.unet_transition_loading = bool(
            request is not None and request[2] == self._unet_transition_generation)
        state.unet_transition_loading_message = (
            "Recomputing ordered scan pairs with full-Q U-Net…"
            if state.unet_transition_loading else "")

    def _clear_unet_transition_report(self):
        """Remove displayed scientific data before a new scope or failed request."""
        self._unet_transition_report = {"records": [], "groups": []}
        self._unet_transition_scans = {}
        state = self.server.state
        state.unet_transition_series = ""
        state.unet_transition_series_options = []
        state.unet_transition_selected = ""
        state.unet_transition_selected_event = -1
        state.unet_transition_unavailable = []
        state.unet_transition_error = ""
        self._render_unet_transition_tracking()

    async def _run_unet_transition_tracking(self):
        loop = asyncio.get_running_loop()
        owner = asyncio.current_task()
        try:
            while self._unet_transition_request is not None:
                signature, sources, generation, engine, order, descending, gap = self._unet_transition_request
                def report_progress(done, total, label):
                    stride = max(1, total // 40)
                    if done in {1, total} or done % stride == 0:
                        loop.call_soon_threadsafe(self._set_unet_transition_progress,
                                                  signature, done, total, label, generation)
                try:
                    report = await asyncio.to_thread(
                        engine.evaluate, sources, order_by=order,
                        descending=descending, reference_gap=gap,
                        progress=report_progress,
                        cancelled=lambda: (
                            generation != self._unet_transition_generation
                            or self._unet_transition_request is None
                            or signature != self._unet_transition_request[0]))
                except ComparisonSuperseded:
                    continue
                except Exception as error:
                    if (self._unet_transition_request is None or
                            signature != self._unet_transition_request[0] or
                            generation != self._unet_transition_request[2]):
                        continue
                    if generation == self._unet_transition_generation:
                        self._unet_transition_request = None
                        self._unet_transition_signature = None
                        self._clear_unet_transition_report()
                        self.server.state.unet_transition_error = str(error)
                    break
                if (self._unet_transition_request is None or
                        signature != self._unet_transition_request[0] or
                        generation != self._unet_transition_request[2]):
                    continue
                if generation == self._unet_transition_generation:
                    self._apply_unet_transition_tracking(report, self._unet_transition_request[1], signature)
                if self._unet_transition_request is not None and self._unet_transition_request[0] != signature:
                    continue
                self._unet_transition_request = None
                break
        finally:
            # A reset can start a fresh task before the cancelled task reaches
            # finally. Only the current owner may clear the new task's state.
            if self._unet_transition_task is owner:
                self._unet_transition_task = None
                self._update_unet_transition_loading()
                self.server.state.flush()

    def _set_unet_transition_progress(self, signature, done, total, label, generation=None):
        if (self._unet_transition_request is None or
                self._unet_transition_request[0] != signature or
                (generation is not None and generation != self._unet_transition_request[2]) or
                not self.server.state.unet_transition_loading):
            return
        self.server.state.unet_transition_loading_message = f"{label} · {done}/{total} steps"
        self.server.state.flush()

    def _apply_unet_transition_tracking(self, report, scans, signature):
        state = self.server.state
        self._unet_transition_report = report
        self._unet_transition_scans = {str(row["run_id"]): row for row in scans}
        self._unet_transition_signature = signature
        options = []
        for group in report["groups"]:
            scope = group["scope"]
            first = self._unet_transition_scans[group["run_ids"][0]]
            meta = first.get("metadata") or {}
            title = (f"{recorded_sample_title(meta)} · "
                     f"{scope['instrument']} bank {scope['bank']} · {len(group['run_ids'])} scans")
            options.append({"title": title, "value": group["id"]})
        state.unet_transition_series_options = options
        if state.unet_transition_series not in {row["value"] for row in options}:
            state.unet_transition_series = options[0]["value"] if options else ""
        selected_group = next((group for group in report["groups"]
                               if group["id"] == state.unet_transition_series), None)
        state.unet_transition_order_options = available_orders([
            self._unet_transition_scans[run_id] for run_id in (selected_group or {}).get("run_ids", [])
        ])
        if state.unet_transition_order not in {item["value"] for item in state.unet_transition_order_options}:
            # A newly arrived scan may lack the condition that previously
            # ordered this series. Never show boxes from a hybrid order.
            state.unet_transition_order = "acquired"
            self._schedule_unet_transition_tracking(scans)
            return
        state.unet_transition_unavailable = [
            f"{scan_label(self._unet_transition_scans.get(row['run_id'], {}))}: {row['reason']}"
            for row in report["records"] if row["group"] is None]
        state.unet_transition_error = ""
        self._render_unet_transition_tracking()
        state.flush()

    def _render_unet_transition_tracking(self):
        state = self.server.state
        group = next((item for item in self._unet_transition_report["groups"]
                      if item["id"] == state.unet_transition_series), None)
        if group is not None:
            identity = (self._unet_transition_scope, state.unet_transition_series,
                        digest(group.get("scope")))
            if identity != self._unet_transition_view_identity:
                self._unet_transition_view_identity = identity
                self._unet_transition_x_range = None
                self._unet_transition_zoom_revision += 1
        rows = [row for row in self._unet_transition_report["records"]
                if row["group"] == state.unet_transition_series]
        state.unet_transition_scans = [
            {"title": (f"{scan_label(self._unet_transition_scans.get(row['run_id'], {}))} · "
                       f"{_condition_label(self._unet_transition_scans.get(row['run_id'], {}), state.unet_transition_order)}"),
             "value": row["run_id"]} for row in rows]
        ids = [row["run_id"] for row in rows]
        if state.unet_transition_follow or state.unet_transition_selected not in ids:
            timed = [run_id for run_id in ids
                     if not acquisition_key(self._unet_transition_scans[run_id])[0]]
            latest = timed or ids
            state.unet_transition_selected = max(
                latest, key=lambda run_id: acquisition_key(self._unet_transition_scans[run_id])
            ) if latest else ""
        position = ids.index(state.unet_transition_selected) if state.unet_transition_selected in ids else -1
        state.unet_transition_prev_disabled = position <= 0
        state.unet_transition_next_disabled = position < 0 or position >= len(ids) - 1
        rail_start = max(0, min(max(position - 3, 0), max(len(rows) - 7, 0)))
        rail_stop = min(len(rows), rail_start + 7)
        state.unet_transition_scan_cards = []
        for ordinal, item in enumerate(rows[rail_start:rail_stop], rail_start + 1):
            scan = self._unet_transition_scans.get(item["run_id"], {})
            count = len(item.get("events") or [])
            result = ("Baseline" if item["status"] == "initial" else
                      f"Review needed · {count} candidate{'s' if count != 1 else ''}"
                      if item["status"] == "uncertain" else
                      f"{count} candidate region{'s' if count != 1 else ''}"
                      if item["status"] == "valid" and count else
                      "No supported region" if item["status"] == "valid" else
                      "Unavailable")
            tone = ("baseline" if item["status"] == "initial" else
                    "pending" if item["status"] in {"uncertain", "unsupported"} else
                    "attention" if count else "clear")
            state.unet_transition_scan_cards.append({
                "value": item["run_id"], "ordinal": f"{ordinal:02d}",
                "label": scan_label(scan), "condition": _condition_label(scan, state.unet_transition_order),
                "result": result, "tone": tone,
            })
        compared = sum(row["previous"] is not None for row in rows)
        total_boxes = sum(len(row.get("events") or []) for row in rows)
        state.unet_transition_series_summary = (
            f"{len(rows)} measured scans · {compared} comparisons · "
            f"{total_boxes} model candidate regions."
            if rows else "No compatible measured series yet."
        )
        group = next((item for item in self._unet_transition_report["groups"]
                      if item["id"] == state.unet_transition_series), None)
        scope = (group or {}).get("scope") or {}
        state.unet_transition_scope_summary = (
            f"{scope.get('instrument', 'Unknown instrument')} · bank {scope.get('bank', '?')} · "
            f"{COORDINATES.get(scope.get('coordinate'), scope.get('coordinate', 'unknown coordinate'))} · "
            f"{scope.get('normalization', 'unknown normalization')} · "
            f"{scope.get('intensity_semantics', 'unknown intensity basis')}"
        ) if scope else "No compatible source scope."
        state.unet_transition_history = _history_figure(rows, self._unet_transition_scans)
        row = next((item for item in rows if item["run_id"] == state.unet_transition_selected), None)
        state.unet_transition_events = []
        state.unet_transition_notice = ""
        state.unet_transition_pair_context = ""
        state.unet_transition_pair_label = "Waiting for measured scans"
        state.unet_transition_attention_label = "No comparison"
        state.unet_transition_attention_tone = "pending"
        state.unet_transition_feature_count = 0
        state.unet_transition_event_items = []
        state.unet_transition_empty_features = "No candidate regions in this comparison."
        state.unet_transition_selected_event_number = 0
        state.unet_transition_selected_feature = ""
        state.unet_transition_selected_interval = ""
        state.unet_transition_selected_support = ""
        state.unet_transition_causes = []
        state.unet_transition_cause_summary = ""
        state.unet_transition_coverage_note = ""
        state.unet_transition_profile_note = ""
        state.unet_transition_event_prev_disabled = True
        state.unet_transition_event_next_disabled = True
        state.unet_transition_profile = plotly.Figure.to_data(empty_figure("Select a comparison"))
        state.unet_transition_detail = plotly.Figure.to_data(empty_figure("Choose a candidate region to inspect"))
        state.unet_transition_reference_fit = plotly.Figure.to_data(empty_figure("No reference scan selected"))
        state.unet_transition_reference_fit_note = "No reference scan selected."
        state.unet_transition_reference_fit_label = "Reference scan fit"
        state.unet_transition_reference_has_fit = False
        state.unet_transition_selected_fit = plotly.Figure.to_data(empty_figure("No scan selected"))
        state.unet_transition_selected_fit_note = "No scan selected."
        state.unet_transition_selected_fit_label = "Selected scan fit"
        state.unet_transition_selected_has_fit = False
        if row is None:
            state.unet_transition_summary = "Waiting for compatible measured scans"
            self._update_unet_transition_map(rows, position)
            return
        scan = self._unet_transition_scans[row["run_id"]]
        label = scan_label(scan)
        state.unet_transition_selected_fit_label = f"{label} · RADAR-PD fit"
        try:
            state.unet_transition_selected_fit, state.unet_transition_selected_fit_note = self._cached_unet_transition_fit(scan)
            state.unet_transition_selected_has_fit = any(
                trace.get("name") == "Calculated"
                for trace in state.unet_transition_selected_fit["data"])
        except (OSError, ValueError, TypeError) as error:
            state.unet_transition_selected_fit_note = f"Fit display unavailable: {error}"
        if row["status"] == "initial":
            state.unet_transition_summary = f"{label} establishes this series"
            state.unet_transition_pair_label = f"{label} · baseline scan"
            state.unet_transition_pair_context = f"{_condition_label(scan, state.unet_transition_order)} · {_acquired_label(scan)}. The next scan begins pairwise comparison."
            state.unet_transition_attention_label = "Baseline"
            state.unet_transition_attention_tone = "baseline"
            try:
                state.unet_transition_profile = _baseline_figure(self._unet_transition_engine.load(scan).pattern)
            except (OSError, ValueError, TypeError) as error:
                state.unet_transition_notice = f"Baseline curve display unavailable: {error}"
            self._update_unet_transition_map(rows, position)
            return
        if not row["previous"]:
            state.unet_transition_summary = f"{label}: {row['reason'] or 'comparison unavailable'}"
            state.unet_transition_pair_label = f"{label} · comparison unavailable"
            state.unet_transition_pair_context = row["reason"] or "Waiting for a compatible measured predecessor."
            self._update_unet_transition_map(rows, position)
            return
        before = self._unet_transition_scans[row["previous"]]
        after = scan
        state.unet_transition_reference_fit_label = f"{scan_label(before)} · RADAR-PD fit"
        try:
            state.unet_transition_reference_fit, state.unet_transition_reference_fit_note = self._cached_unet_transition_fit(before)
            state.unet_transition_reference_has_fit = any(
                trace.get("name") == "Calculated"
                for trace in state.unet_transition_reference_fit["data"])
        except (OSError, ValueError, TypeError) as error:
            state.unet_transition_reference_fit_note = f"Fit display unavailable: {error}"
        events = row.get("events") or []
        causes = (row.get("comparison") or {}).get("causes") or []
        state.unet_transition_causes = [{"label": _CAUSE_LABELS.get(item["name"], item["name"]),
                                          "score": _score_label(item["score"]),
                                          "selected": bool(item["selected"])} for item in causes]
        selected_causes = [item for item in causes if item["selected"]]
        state.unet_transition_cause_summary = (
            "Whole-pair cause suggestions are uncalibrated model scores."
            + (" Selected cause has no supported local box." if selected_causes and not events else "")
        ) if causes else ""
        covered = (row.get("comparison") or {}).get("coverage_fraction")
        state.unet_transition_coverage_note = (
            f"Joint Q coverage: {100 * covered:.1f}% of 0.25–12 Å⁻¹."
            if isinstance(covered, (int, float)) else "")
        profiles = (row.get("comparison") or {}).get("profiles") or []
        state.unet_transition_profile_note = "; ".join(
            f"Physical bank {profile['source_bank']} → profile Bank {profile['profile_bank']} · "
            f"SHA-256 {profile['sha256']}" for profile in profiles)
        state.unet_transition_summary = (f"{scan_label(before)} → {label}: " +
                                    (f"{len(events)} local feature box(es)" if events else
                                     "no supported local box" if row["status"] == "valid" else
                                     "comparison uncertain"))
        state.unet_transition_pair_label = f"{scan_label(before)} → {label}"
        state.unet_transition_pair_context = (
            f"{_condition_label(before, state.unet_transition_order)} → {_condition_label(after, state.unet_transition_order)} · "
            f"{row.get('reference_steps', 1)} scan "
            f"{'step' if row.get('reference_steps', 1) == 1 else 'steps'} in chosen order · "
            f"{_acquired_label(before)} → {_acquired_label(after)}"
            + (" · Directional labels follow reference → selected; reverse order is not separately validated."
               if state.unet_transition_direction == "descending" else "")
        )
        state.unet_transition_attention_label = (
            f"Review needed · {len(events)} candidate{'s' if len(events) != 1 else ''}"
            if row["status"] in {"uncertain", "unsupported"} else
            f"{len(events)} candidate region{'s' if len(events) != 1 else ''}"
            if events else "No supported region"
        )
        state.unet_transition_attention_tone = (
            "pending" if row["status"] in {"uncertain", "unsupported"} else
            "attention" if events else "clear"
        )
        state.unet_transition_notice = ("Comparison incomplete: " + row["reason"]
                                   if row["status"] in {"uncertain", "unsupported"} else
                                   "" if len(events) <= 200 else
                                   "Showing the first 200 regions on the full curve; the list includes all candidates.")
        state.unet_transition_events = [
            {"feature": ", ".join(event.get("feature_types") or ["type uncertain"]).replace("_", " "),
             "x0": f"{event['x0']:.6g}", "x1": f"{event['x1']:.6g}",
             "score": _score_label(event["attention_score"])} for event in events]
        state.unet_transition_feature_count = len(events)
        types = sorted({name for event in events for name in event.get("feature_types", [])})
        state.unet_transition_feature_filter_options = [
            {"title": "All feature types", "value": "all"},
            *({"title": name.replace("_", " ").capitalize(), "value": name} for name in types),
            *([{"title": "Type uncertain", "value": "uncertain"}]
              if any(not event.get("feature_types") for event in events) else []),
        ] if events else [{"title": "All feature types", "value": "all"}]
        valid_filters = {option["value"] for option in state.unet_transition_feature_filter_options}
        if state.unet_transition_feature_filter not in valid_filters:
            state.unet_transition_feature_filter = "all"
        filter_name = state.unet_transition_feature_filter
        visible = [index for index, event in enumerate(events)
                   if filter_name == "all" or
                   (not event.get("feature_types") if filter_name == "uncertain"
                    else filter_name in event.get("feature_types", []))]
        state.unet_transition_event_items = [{
            "index": index, "number": f"{index + 1:02d}",
            "feature": _feature_label(events[index]),
            "interval": (f"{events[index]['x0']:.4g}–{events[index]['x1']:.4g} "
                         f"{COORDINATES.get(scope.get('coordinate'), '')} · "
                         f"Q {events[index]['q0']:.4g}–{events[index]['q1']:.4g} Å⁻¹"
                         if "q0" in events[index] else
                         f"{events[index]['x0']:.4g}–{events[index]['x1']:.4g} "
                         f"{COORDINATES.get(scope.get('coordinate'), '')}"),
            "score": _score_label(events[index]["attention_score"]),
            "tone": _feature_tone(events[index]),
        } for index in visible]
        state.unet_transition_empty_features = (
            "No candidate regions in this comparison." if not events else
            "No regions match this feature type." if not visible else ""
        )
        if state.unet_transition_selected_event not in visible:
            state.unet_transition_selected_event = -1
        selected = state.unet_transition_selected_event
        if selected >= 0:
            event = events[selected]
            state.unet_transition_selected_event_number = selected + 1
            state.unet_transition_selected_feature = _feature_label(event).capitalize()
            state.unet_transition_selected_interval = (
                f"{event['x0']:.5g}–{event['x1']:.5g} "
                f"{COORDINATES.get(scope.get('coordinate'), '')} · "
                f"Q {event['q0']:.4g}–{event['q1']:.4g} Å⁻¹"
                if "q0" in event else
                f"{event['x0']:.5g}–{event['x1']:.5g} "
                f"{COORDINATES.get(scope.get('coordinate'), '')}"
            )
            state.unet_transition_selected_support = (
                f"model attention {_score_label(event['attention_score'])} · one full-Q pass"
            )
            selected_position = visible.index(selected)
            state.unet_transition_event_prev_disabled = selected_position == 0
            state.unet_transition_event_next_disabled = selected_position == len(visible) - 1
        try:
            earlier = self._unet_transition_engine.load(before).pattern
            current = self._unet_transition_engine.load(after).pattern
            state.unet_transition_profile = _pair_figure(
                earlier, current, events, selected, state.unet_transition_overview_scale)
            state.unet_transition_detail = _focus_figure(
                earlier, current, events[selected] if selected >= 0 else None)
        except (OSError, ValueError, TypeError) as error:
            state.unet_transition_notice = f"Curve display unavailable: {error}"
        self._update_unet_transition_map(rows, position)

    def _cached_unet_transition_fit(self, scan):
        # A delayed redraw may still hold the previous scan while its revised
        # comparison is pending. Cache under the actual scan being rendered.
        key = _fit_cache_key(scan)
        if key not in self._unet_transition_fit_cache:
            try:
                measured = self._unet_transition_engine.load(scan).pattern
            except (OSError, ValueError, TypeError):
                measured = None
            self._unet_transition_fit_cache[key] = _fit_figure_for_scan(scan, measured)
        return self._unet_transition_fit_cache[key]

    def _update_unet_transition_map(self, rows, position):
        state = self.server.state
        if not rows or position < 0:
            self._clear_unet_transition_map("Waiting for measured scans")
            return
        color_key = (self._unet_transition_signature, state.unet_transition_series,
                     state.unet_transition_color_scale)
        if self._unet_transition_color_signature != color_key:
            self._unet_transition_color_limits = None
            self._unet_transition_color_signature = color_key
        if self._unet_transition_color_limits is None:
            try:
                self._unet_transition_color_limits = _reported_color_limits(
                    rows, state.unet_transition_color_scale)
                if self._unet_transition_color_limits is None:
                    self._unet_transition_color_limits = _series_color_limits(
                        (self._unet_transition_engine.load(self._unet_transition_scans[row["run_id"]]).pattern
                         for row in rows), state.unet_transition_color_scale)
            except (OSError, ValueError, TypeError, KeyError) as error:
                self._clear_unet_transition_map("Measured heat map unavailable")
                state.unet_transition_notice = f"Heat-map display unavailable: {error}"
                return
        counts = Counter(
            name for row in rows for event in (row.get("events") or [])
            for name in (event.get("feature_types") or ["uncertain"])
        )
        state.unet_transition_overlay_options = [
            {"title": f"{_OVERLAY_LABELS.get(name, name.replace('_', ' ').capitalize())} ({counts[name]})",
             "value": name}
            for name in [*_FEATURE_COLORS, "uncertain"]
        ]
        width = min(max(1, int(state.unet_transition_map_window)), 100)
        max_start = max(0, len(rows) - width)
        if state.unet_transition_map_manual_scroll:
            start = max(0, min(int(state.unet_transition_map_start) - 1, max_start))
        else:
            start = max(0, min(position - width // 2, max_start))
        view_rows = rows[start:start + width]
        map_height = _inline_map_height(len(view_rows))
        state.unet_transition_map_height = map_height
        state.unet_transition_map_scroll_padding = round(
            25 + (.5 if len(view_rows) <= 8 else .38 if len(view_rows) <= 30 else .3)
            * (map_height - 107) + .045 * (map_height - 107)
        )
        state.unet_transition_map_start = start + 1
        state.unet_transition_map_scroll_max = max_start + 1
        state.unet_transition_map_scan_position = position + 1
        state.unet_transition_map_scan_max = len(rows)
        state.unet_transition_map_scroll_label = (
            f"Rows {start + 1}–{start + len(view_rows)} of {len(rows)}"
            if max_start else f"All {len(rows)} scans visible")
        state.unet_transition_map_scroll_compact = (
            f"{start + 1}–{start + len(view_rows)} / {len(rows)}"
            if max_start else f"{len(rows)} scans")
        try:
            selected = rows[position]
            retain = getattr(self._unet_transition_engine, "retain_sources", None)
            if retain is not None:
                retained_ids = [selected["run_id"]]
                if selected.get("previous") in self._unet_transition_scans:
                    retained_ids.append(selected["previous"])
                retained_ids.extend(row["run_id"] for row in view_rows)
                retain([self._unet_transition_scans[run_id] for run_id in dict.fromkeys(retained_ids)])
            patterns = [self._unet_transition_engine.load(self._unet_transition_scans[row["run_id"]]).pattern
                        for row in view_rows]
            current = self._unet_transition_engine.load(
                self._unet_transition_scans[selected["run_id"]]).pattern
            earlier = (self._unet_transition_engine.load(
                self._unet_transition_scans[selected["previous"]]).pattern
                if selected.get("previous") in self._unet_transition_scans else None)
            order_title = next((item["title"] for item in state.unet_transition_order_options
                                if item["value"] == state.unet_transition_order), "Scan order")
            self._unet_transition_map_revision += 1
            identity = {"scope": self._unet_transition_scope,
                        "series": state.unet_transition_series,
                        "revision": self._unet_transition_map_revision}
            figure, shown, saturated = _evolution_figure(
                view_rows, patterns, self._unet_transition_scans,
                selected["run_id"], current, earlier,
                selected_types=set(state.unet_transition_overlay_types or []),
                selected_event=state.unet_transition_selected_event,
                scale=state.unet_transition_color_scale, palette=state.unet_transition_palette,
                zoom=self._unet_transition_x_range,
                revision=self._unet_transition_zoom_revision,
                series=state.unet_transition_series,
                color_limits=self._unet_transition_color_limits,
                order_label=order_title,
                order_by=state.unet_transition_order,
                identity=identity, reference_id=selected.get("previous"),
                height=map_height,
            )
            self._unet_transition_map_ids = [row["run_id"] for row in view_rows]
            self._unet_transition_map_identity = identity
            state.unet_transition_map = figure
            state.unet_transition_dialog_figure = {
                **figure, "layout": {**figure["layout"], "height": max(690, map_height)}
            }
            total = sum(len(row.get("events") or []) for row in rows)
            span = (f" · map rows {start + 1}–{start + len(view_rows)} of {len(rows)}"
                    if len(view_rows) < len(rows) else "")
            state.unet_transition_overlay_summary = (
                f"Showing {shown} of {total} model regions{span}."
                + (f" {saturated} heat-map cells exceed the locked color range."
                   if saturated else "")
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            self._clear_unet_transition_map("Measured heat map unavailable")
            state.unet_transition_notice = f"Heat-map display unavailable: {error}"

    def _clear_unet_transition_map(self, message):
        state = self.server.state
        self._unet_transition_map_revision += 1
        self._unet_transition_map_identity = None
        self._unet_transition_map_ids = []
        retain = getattr(self._unet_transition_engine, "retain_sources", None)
        if retain is not None:
            retain([])
        figure = plotly.Figure.to_data(empty_figure(message))
        state.unet_transition_map = figure
        state.unet_transition_dialog_figure = figure
        state.unet_transition_overlay_summary = "No compatible measured series yet."
        state.unet_transition_overlay_options = []
        state.unet_transition_map_start = 1
        state.unet_transition_map_scroll_max = 1
        state.unet_transition_map_scan_position = 1
        state.unet_transition_map_scan_max = 1
        state.unet_transition_map_scroll_label = "No scans available"
        state.unet_transition_map_scroll_compact = "0 scans"
        state.unet_transition_map_manual_scroll = False

    def select_unet_transition_series(self, value, **_):
        state = self.server.state
        state.unet_transition_series = value
        state.unet_transition_follow = True
        state.unet_transition_selected_event = -1
        state.unet_transition_feature_filter = "all"
        state.unet_transition_map_manual_scroll = False
        group = next((item for item in self._unet_transition_report["groups"]
                      if item["id"] == value), None)
        state.unet_transition_order_options = available_orders([
            self._unet_transition_scans[run_id] for run_id in (group or {}).get("run_ids", [])
        ])
        if state.unet_transition_order not in {item["value"] for item in state.unet_transition_order_options}:
            state.unet_transition_order = "acquired"
            self._schedule_unet_transition_tracking(self._unet_transition_sources_cache)
            return
        self._unet_transition_x_range = None
        self._unet_transition_zoom_revision += 1
        self._render_unet_transition_tracking()

    def select_unet_transition_order(self, value, **_):
        state = self.server.state
        if value not in {item["value"] for item in state.unet_transition_order_options}:
            return
        state.unet_transition_order = value
        state.unet_transition_follow = False
        state.unet_transition_selected_event = -1
        state.unet_transition_map_manual_scroll = False
        self._schedule_unet_transition_tracking(self._unet_transition_sources_cache)

    def select_unet_transition_direction(self, value, **_):
        if value not in {"ascending", "descending"}:
            return
        state = self.server.state
        state.unet_transition_direction = value
        state.unet_transition_follow = False
        state.unet_transition_selected_event = -1
        state.unet_transition_map_manual_scroll = False
        self._schedule_unet_transition_tracking(self._unet_transition_sources_cache)

    def select_unet_transition_reference_gap(self, value, **_):
        try:
            gap = "first" if value == "first" else int(value)
        except (TypeError, ValueError):
            return
        if gap not in {1, 5, 20, 100, "first"}:
            return
        state = self.server.state
        state.unet_transition_reference_gap = gap
        state.unet_transition_selected_event = -1
        self._schedule_unet_transition_tracking(self._unet_transition_sources_cache)

    def select_unet_transition_scan(self, value, **_):
        self.server.state.unet_transition_selected = value
        self.server.state.unet_transition_follow = False
        self.server.state.unet_transition_selected_event = -1
        self.server.state.unet_transition_feature_filter = "all"
        self.server.state.unet_transition_map_manual_scroll = False
        self._render_unet_transition_tracking()

    def previous_unet_transition_scan(self, **_):
        ids = [row["value"] for row in self.server.state.unet_transition_scans]
        current = self.server.state.unet_transition_selected
        if current in ids and ids.index(current) > 0:
            self.select_unet_transition_scan(ids[ids.index(current) - 1])

    def next_unet_transition_scan(self, **_):
        ids = [row["value"] for row in self.server.state.unet_transition_scans]
        current = self.server.state.unet_transition_selected
        if current in ids and ids.index(current) < len(ids) - 1:
            self.select_unet_transition_scan(ids[ids.index(current) + 1])

    def select_unet_transition_event(self, index, **_):
        try:
            self.server.state.unet_transition_selected_event = int(index)
        except (TypeError, ValueError):
            return
        self._render_unet_transition_tracking()

    def select_unet_transition_plot_point(self, event, **_):
        try:
            x = float(event["points"][0]["x"] if isinstance(event, dict) else event)
        except (TypeError, ValueError, KeyError, IndexError):
            return
        row = next((item for item in self._unet_transition_report["records"]
                    if item["run_id"] == self.server.state.unet_transition_selected), None)
        events = (row or {}).get("events") or []
        inside = [index for index, item in enumerate(events)
                  if item["x0"] <= x <= item["x1"]]
        if inside:
            selected = min(inside, key=lambda index: events[index]["x1"] - events[index]["x0"])
            self.select_unet_transition_event(selected)

    def select_unet_transition_map_click(self, name, x, y, token, identity=None, **_):
        current_identity = self._unet_transition_map_identity
        if (self.server.state.unet_transition_loading or
                not isinstance(identity, dict) or current_identity is None or
                any(identity.get(key) != value for key, value in current_identity.items())):
            return
        if name == "Measured heat map":
            try:
                index = round(float(y))
            except (TypeError, ValueError, OverflowError):
                return
            if (0 <= index < len(self._unet_transition_map_ids) and
                    token == self._unet_transition_map_ids[index]):
                self.select_unet_transition_scan(token)
            return
        if name == "U-Net overlays":
            try:
                run_id, raw_index = str(token).rsplit("|", 1)
                event_index = int(raw_index)
            except (TypeError, ValueError):
                return
            row = next((item for item in self._unet_transition_report["records"]
                        if item["run_id"] == run_id and
                        item["group"] == self.server.state.unet_transition_series), None)
            events = (row or {}).get("events") or []
            if run_id not in self._unet_transition_map_ids or not 0 <= event_index < len(events):
                return
            event = events[event_index]
            self.server.state.unet_transition_selected = run_id
            self.server.state.unet_transition_follow = False
            self.server.state.unet_transition_selected_event = event_index
            self.server.state.unet_transition_feature_filter = "all"
            self.server.state.unet_transition_map_manual_scroll = False
            width = max(float(event["x1"] - event["x0"]), 1e-9)
            self._unet_transition_x_range = [float(event["x0"] - 1.4 * width),
                                        float(event["x1"] + 1.4 * width)]
            self._unet_transition_zoom_revision += 1
            self._render_unet_transition_tracking()
            return
        if name in {"Selected scan", "Reference scan"}:
            selected = self.server.state.unet_transition_selected
            row = next((item for item in self._unet_transition_report["records"]
                        if item["run_id"] == selected and
                        item["group"] == self.server.state.unet_transition_series), None)
            expected = selected if name == "Selected scan" else (row or {}).get("previous")
            if expected is not None and identity.get("run_id") == expected:
                self.select_unet_transition_plot_point(x)

    def select_unet_transition_overlay_types(self, values, **_):
        allowed = {*_FEATURE_COLORS, "uncertain"}
        self.server.state.unet_transition_overlay_types = [
            name for name in (values or []) if name in allowed
        ]
        self.server.state.unet_transition_selected_event = -1
        self._render_unet_transition_tracking()

    def show_all_unet_transition_overlays(self, **_):
        self.select_unet_transition_overlay_types([*_FEATURE_COLORS, "uncertain"])

    def clear_unet_transition_overlays(self, **_):
        self.select_unet_transition_overlay_types([])

    def select_unet_transition_color_scale(self, value, **_):
        if value in {"linear", "log"}:
            self.server.state.unet_transition_color_scale = value
            self._render_unet_transition_tracking()

    def select_unet_transition_palette(self, value, **_):
        if value in {"Turbo", "Cividis"}:
            self.server.state.unet_transition_palette = value
            self._render_unet_transition_tracking()

    def select_unet_transition_map_window(self, value, **_):
        try:
            width = int(value)
        except (TypeError, ValueError):
            return
        if width in {30, 60, 100}:
            self.server.state.unet_transition_map_window = width
            self.server.state.unet_transition_map_manual_scroll = False
            self._render_unet_transition_tracking()

    def select_unet_transition_map_start(self, value, **_):
        try:
            start = int(value)
        except (TypeError, ValueError):
            return
        state = self.server.state
        state.unet_transition_map_start = max(1, min(start, state.unet_transition_map_scroll_max))
        state.unet_transition_map_manual_scroll = True
        rows = [row for row in self._unet_transition_report["records"]
                if row["group"] == state.unet_transition_series]
        ids = [row["run_id"] for row in rows]
        position = ids.index(state.unet_transition_selected) if state.unet_transition_selected in ids else -1
        self._update_unet_transition_map(rows, position)

    def select_unet_transition_map_scan(self, value, **_):
        try:
            position = int(value) - 1
        except (TypeError, ValueError, OverflowError):
            return
        rows = [row for row in self._unet_transition_report["records"]
                if row["group"] == self.server.state.unet_transition_series]
        if rows:
            position = max(0, min(position, len(rows) - 1))
            self.select_unet_transition_scan(rows[position]["run_id"])

    def previous_unet_transition_map_page(self, **_):
        self.select_unet_transition_map_start(
            self.server.state.unet_transition_map_start - self.server.state.unet_transition_map_window)

    def next_unet_transition_map_page(self, **_):
        self.select_unet_transition_map_start(
            self.server.state.unet_transition_map_start + self.server.state.unet_transition_map_window)

    def relayout_unet_transition_map(self, event, **_):
        """Retain shared x zoom and rescale the ribbon without inference or raster work."""
        if not isinstance(event, dict) or self.server.state.unet_transition_loading:
            return
        for axis in ('xaxis2', 'xaxis'):
            if event.get(axis + '.autorange') is True:
                self.reset_unet_transition_map_zoom()
                return
            limits = event.get(axis + '.range')
            if axis + '.range[0]' in event and axis + '.range[1]' in event:
                limits = [event[axis + '.range[0]'], event[axis + '.range[1]']]
            if not isinstance(limits, (list, tuple)) or len(limits) != 2:
                continue
            try:
                limits = sorted(float(v) for v in limits)
            except (TypeError, ValueError, OverflowError):
                return
            if not all(math.isfinite(v) for v in limits) or limits[0] == limits[1]:
                return
            if limits == self._unet_transition_x_range:
                return
            self._unet_transition_x_range = limits
            state = self.server.state
            figure = dict(state.unet_transition_map)
            figure['layout'] = dict(figure['layout'])
            for name in ('xaxis', 'xaxis2'):
                figure['layout'][name] = {**figure['layout'].get(name, {}), 'range': limits, 'autorange': False}
            bounds = _visible_ribbon_range(figure.get('data', []), limits)
            if bounds is not None:
                figure['layout']['yaxis'] = {**figure['layout'].get('yaxis', {}), 'range': bounds, 'autorange': False}
            state.unet_transition_map = figure
            state.unet_transition_dialog_figure = figure
            state.flush()
            return

    def reset_unet_transition_map_zoom(self, **_):
        self._unet_transition_x_range = None
        self._unet_transition_zoom_revision += 1
        self._render_unet_transition_tracking()

    def previous_unet_transition_event(self, **_):
        visible = [item["index"] for item in self.server.state.unet_transition_event_items]
        selected = self.server.state.unet_transition_selected_event
        if selected in visible and visible.index(selected) > 0:
            self.select_unet_transition_event(visible[visible.index(selected) - 1])

    def next_unet_transition_event(self, **_):
        visible = [item["index"] for item in self.server.state.unet_transition_event_items]
        selected = self.server.state.unet_transition_selected_event
        if selected in visible and visible.index(selected) < len(visible) - 1:
            self.select_unet_transition_event(visible[visible.index(selected) + 1])

    def select_unet_transition_feature_filter(self, value, **_):
        self.server.state.unet_transition_feature_filter = value
        self._render_unet_transition_tracking()

    def select_unet_transition_scale(self, value, **_):
        if value != "linear":
            return
        self.server.state.unet_transition_overview_scale = value
        self._render_unet_transition_tracking()

    def follow_unet_transition_latest(self, **_):
        self.server.state.unet_transition_follow = True
        self.server.state.unet_transition_map_manual_scroll = False
        self.server.state.unet_transition_selected_event = -1
        self.server.state.unet_transition_feature_filter = "all"
        self._render_unet_transition_tracking()

    def expand_unet_transition_plot(self, **_):
        self.server.state.unet_transition_dialog_figure = self.server.state.unet_transition_map
        self.server.state.unet_transition_dialog = True
