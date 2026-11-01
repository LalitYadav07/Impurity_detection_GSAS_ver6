"""Bounded electrical timeline; gaps stay visible and exposure means stay distinct."""
from datetime import datetime, timezone
import math


def _utc(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def timeline_figure(series, joins, *, max_points=6000):
    if max_points < 12:
        raise ValueError("Timeline budget must be at least twelve points")
    points = series.points
    # Retain both electrical extrema and block endpoints. No averaging of the
    # displayed raw record; quantitative joins always use the complete series.
    stride = max(1, math.ceil(len(points) * 6 / max_points))
    chosen = set()
    for start in range(0, len(points), stride):
        indices = range(start, min(start + stride, len(points)))
        chosen.update((indices.start, indices.stop-1))
        chosen.add(min(indices, key=lambda i: points[i].voltage_v))
        chosen.add(max(indices, key=lambda i: points[i].voltage_v))
        measured = [i for i in indices if points[i].current_a is not None]
        if measured:
            chosen.add(min(measured, key=lambda i: points[i].current_a))
            chosen.add(max(measured, key=lambda i: points[i].current_a))
    # A prefix sum detects any excluded interval between displayed samples.
    breaks = [0]
    for a, b in zip(points, points[1:]):
        blocked = (b.timestamp-a.timestamp > series.binding.maximum_gap_seconds
            or (a.cycle, a.step) != (b.cycle, b.step)
            or series.branch(a) != series.branch(b))
        breaks.append(breaks[-1] + int(blocked))
    x, voltage, current = [], [], []
    previous = None
    for index in sorted(chosen):
        if previous is not None and breaks[index] != breaks[previous]:
            x.append(None); voltage.append(None); current.append(None)
        point = points[index]
        x.append(_utc(point.timestamp)); voltage.append(point.voltage_v)
        current.append(point.current_a * 1000 if point.current_a is not None else None)
        previous = index
    data = [dict(type="scattergl", mode="lines", name="Recorded voltage", x=x, y=voltage,
                 connectgaps=False, line=dict(color="#205d49"), yaxis="y")]
    if any(value is not None for value in current):
        data.append(dict(type="scattergl", mode="lines", name="Recorded current", x=x, y=current,
                         connectgaps=False, line=dict(color="#3869a8"), yaxis="y2"))
    means, exposure_x, half_width, hover = [], [], [], []
    for record in joins:
        if record.get("voltage_v") is None:
            continue
        lo = datetime.fromisoformat(record["start_utc"]).timestamp()
        hi = datetime.fromisoformat(record["end_utc"]).timestamp()
        exposure_x.append(_utc((lo+hi)/2)); half_width.append((hi-lo)*500)
        means.append(record["voltage_v"]["mean"])
        hover.append(f"Run {record['run']} · voltage coverage {record['coverage_fraction']:.1%}")
    if means:
        data.append(dict(type="scatter", mode="markers", name="Exposure mean (covered time)",
            x=exposure_x, y=means, text=hover, hovertemplate="%{text}<br>%{y:.4f} V<extra></extra>",
            marker=dict(color="#a75523", size=9, symbol="diamond"),
            error_x=dict(type="data", array=half_width, visible=True, thickness=2), yaxis="y"))
    return dict(data=data, layout=dict(
        title=dict(text="Electrical record and inspected diffraction exposures", font=dict(size=15)),
        margin=dict(l=65, r=55, t=80, b=50), hovermode="closest",
        xaxis=dict(title="Time (UTC)", type="date", anchor="y2"),
        yaxis=dict(title="Voltage (V)", domain=[.4, 1]),
        yaxis2=dict(title="Current (mA)", domain=[0, .25]),
        legend=dict(orientation="h", y=1.17),
        annotations=[] if any(value is not None for value in current) else [dict(
            x=.5, y=.13, xref="paper", yref="paper", showarrow=False,
            text="No measured current column · charge and branch remain unresolved")],
        uirevision=series.source_sha256 + str(series.binding)))
