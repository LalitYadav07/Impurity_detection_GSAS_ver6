"""Explicit cycler import and exposure-interval synchronization.

No SOC or electrode stoichiometry is inferred. The recorded current sign,
clock mapping and cell/channel binding are reviewed inputs. Piecewise linear
integration excludes gaps and protocol boundaries; no endpoint extrapolation.
"""
from __future__ import annotations

from bisect import bisect_right
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import cached_property
from hashlib import sha256
import io
import math
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..facility_metadata import parse_nexus_timestamp

MAX_BYTES = 64 * 1024 * 1024
MAX_ROWS = 1_000_000
LOCAL_FORMATS = {"month/day/year": "%m/%d/%Y %H:%M:%S", "day/month/year": "%d/%m/%Y %H:%M:%S",
                 "year-month-day": "%Y-%m-%d %H:%M:%S"}


def utc_time(value):
    parsed = parse_nexus_timestamp(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("A timestamp needs an explicit UTC offset (for example 2019-12-09T12:00:26-05:00)")
    return parsed.astimezone(timezone.utc)


def finite_number(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def local_calendar_time(value, format_name, timezone_name):
    """Require an explicit date convention and reject ambiguous DST wall times."""
    pattern = LOCAL_FORMATS[format_name]
    text = str(value).strip()
    parsed = None
    for suffix in (".%f", ""):
        try:
            parsed = datetime.strptime(text, pattern+suffix); break
        except ValueError:
            pass
    if parsed is None:
        raise ValueError("Calendar timestamp does not match the selected date format")
    zone = ZoneInfo(timezone_name)
    candidates = {date.timestamp() for fold in (0,1)
                  if (date := parsed.replace(tzinfo=zone, fold=fold)).astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) == parsed}
    if len(candidates) != 1:
        raise ValueError("Calendar time is ambiguous or nonexistent at a daylight-saving transition; supply explicit UTC timestamps")
    return candidates.pop()


@dataclass(frozen=True)
class CyclerBinding:
    sample_id: str
    cell_id: str
    channel: str
    positive_current: str
    clock_origin: str = ""
    clock_offset_seconds: float = 0.0
    clock_uncertainty_seconds: float | None = None
    maximum_gap_seconds: float = 30.0
    rest_current_threshold_a: float = 1e-6
    local_time_format: str = ""
    timezone_name: str = ""

    def validate(self):
        for name in ("sample_id", "cell_id", "channel"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 128:
                raise ValueError(f"Bind the cycler {name.replace('_', ' ')} explicitly")
        if self.positive_current not in {"charge", "discharge"}:
            raise ValueError("Specify whether positive recorded current means charge or discharge")
        if self.clock_origin:
            utc_time(self.clock_origin)
        if self.local_time_format:
            if self.local_time_format not in LOCAL_FORMATS or not self.timezone_name:
                raise ValueError("Select the recorded calendar date format and its IANA timezone")
            if self.clock_origin:
                raise ValueError("Calendar timestamps use their timezone; clear the elapsed-time clock origin")
            try: ZoneInfo(self.timezone_name)
            except (ZoneInfoNotFoundError, ValueError, TypeError):
                raise ValueError("Use a known IANA timezone, for example America/New_York") from None
        elif self.timezone_name:
            raise ValueError("Select a calendar date format when supplying a timezone")
        for key in ("clock_offset_seconds", "clock_uncertainty_seconds", "maximum_gap_seconds", "rest_current_threshold_a"):
            if key == "clock_uncertainty_seconds" and self.clock_uncertainty_seconds is None:
                continue
            if isinstance(getattr(self, key), bool) or not isinstance(getattr(self, key), (int, float)):
                raise ValueError(f"{key.replace('_', ' ')} must be a number")
        finite_number(self.clock_offset_seconds, "Clock offset")
        if self.clock_uncertainty_seconds is not None and finite_number(self.clock_uncertainty_seconds, "Clock uncertainty") < 0:
            raise ValueError("Clock uncertainty cannot be negative")
        if not 0 < finite_number(self.maximum_gap_seconds, "Maximum gap") <= 86400:
            raise ValueError("Maximum gap must be positive and at most one day")
        if finite_number(self.rest_current_threshold_a, "Rest threshold") < 0:
            raise ValueError("Rest threshold cannot be negative")


@dataclass(frozen=True)
class CyclerPoint:
    timestamp: float
    voltage_v: float
    current_a: float | None
    cycle: str = ""
    step: str = ""


@dataclass(frozen=True)
class ElectrochemistrySeries:
    source_name: str
    source_sha256: str
    format: str
    binding: CyclerBinding
    points: tuple[CyclerPoint, ...]
    duplicate_rows: int = 0
    recorded_origin: str = ""

    @cached_property
    def timestamps(self):
        return tuple(point.timestamp for point in self.points)

    def branch(self, point):
        if point.current_a is None:
            return "unresolved"
        if abs(point.current_a) <= self.binding.rest_current_threshold_a:
            return "rest"
        if point.current_a > 0:
            return self.binding.positive_current
        return "discharge" if self.binding.positive_current == "charge" else "charge"

    def summary(self):
        intervals = sorted(b-a for a,b in zip(self.timestamps, self.timestamps[1:]))
        return {"schema": "radar-battery-electrochemistry/v1", "source_name": self.source_name,
                "source_sha256": self.source_sha256, "format": self.format,
                "binding": asdict(self.binding), "point_count": len(self.points),
                "first_utc": datetime.fromtimestamp(self.points[0].timestamp, timezone.utc).isoformat(),
                "last_utc": datetime.fromtimestamp(self.points[-1].timestamp, timezone.utc).isoformat(),
                "duplicate_rows": self.duplicate_rows, "recorded_origin": self.recorded_origin,
                "current_status": "available" if all(p.current_a is not None for p in self.points) else "partial" if any(p.current_a is not None for p in self.points) else "unavailable",
                "median_recording_interval_seconds": intervals[len(intervals)//2],
                "largest_recording_interval_seconds": intervals[-1],
                "composition_status": "not_evaluated"}


def _column(fields, candidates, label, required=True):
    found = [key for key in candidates if key in fields]
    if len(found) > 1:
        raise ValueError(f"Several {label} columns exist; select one explicitly before import")
    if not found and required:
        raise ValueError(f"No supported {label} column. Expected one of: {', '.join(candidates)}")
    return found[0] if found else None


def _text_rows(raw):
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        # EC-Lab exports use Latin-1. Generic tables must use UTF-8.
        if not raw.startswith(b"EC-Lab ASCII FILE"):
            raise ValueError("Cycler tables must be UTF-8; export a CSV with explicit units") from None
        text = raw.decode("latin-1")
    format_name, origin = "explicit_csv", ""
    if text.startswith("EC-Lab ASCII FILE"):
        match = re.search(r"Nb header lines\s*:\s*(\d+)", text[:16384])
        if not match or not 2 <= int(match[1]) <= 10000:
            raise ValueError("EC-Lab text header length is unavailable or invalid")
        lines = text.splitlines()
        header_index = int(match[1])-1
        if header_index >= len(lines):
            raise ValueError("EC-Lab table is incomplete")
        start = re.search(r"Acquisition started on\s*:\s*([^\r\n]+)", text[:16384])
        origin = start[1].strip() if start else ""
        text = "\n".join(lines[header_index:]); format_name = "biologic_text"
        delimiter = "\t"
    else:
        try:
            delimiter = csv.Sniffer().sniff(text[:16384], delimiters=",;\t").delimiter
        except csv.Error:
            raise ValueError("Cannot read a delimited cycler table with named columns") from None
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    fields = [key.strip() for key in (reader.fieldnames or [])]
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("Cycler column names must be present and unique")
    reader.fieldnames = fields
    return reader, fields, format_name, origin


def _mpr_rows(raw):
    try:
        from galvani import BioLogic
    except ImportError:
        raise ValueError("BioLogic binary import requires the battery extra (galvani); an EC-Lab MPT or explicit CSV export is also supported") from None
    # A byte stream prevents the reader from opening inferred companion files.
    try:
        mpr = BioLogic.MPRfile(io.BytesIO(raw))
    except Exception as exc:
        raise ValueError(f"BioLogic binary reader could not parse this file: {exc}. Use a verified EC-Lab text export.") from exc
    fields = list(mpr.data.dtype.names or [])
    if len(mpr.data) > MAX_ROWS:
        raise ValueError("Cycler row budget exceeded; select a bounded time range")
    origin = str(getattr(mpr, "timestamp", "") or "")
    return ({key: row[key] for key in fields} for row in mpr.data), fields, "biologic_binary", origin


def read_cycler(raw: bytes, binding: CyclerBinding, *, source_name="cycler.csv"):
    binding.validate()
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_BYTES:
        raise ValueError("Cycler input must contain at most 64 MiB")
    rows, fields, format_name, recorded_origin = _mpr_rows(raw) if raw.startswith(b"BIO-LOGIC MODULAR FILE") else _text_rows(raw)
    absolute = _column(fields, ("timestamp", "date_time"), "absolute timestamp", required=False)
    elapsed = _column(fields, ("elapsed_seconds", "time_s", "time/s"), "elapsed time", required=False)
    if bool(absolute) == bool(elapsed):
        raise ValueError("Supply either timezone-qualified timestamps or elapsed seconds with an explicit clock origin")
    if elapsed and not binding.clock_origin and not binding.local_time_format:
        hint = f" The file records local start '{recorded_origin}', which does not establish a UTC offset." if recorded_origin else ""
        raise ValueError("Set the cycler clock origin including its UTC offset; for calendar dates select the recorded date format and timezone instead." + hint)
    voltage = _column(fields, ("voltage_v", "Ewe/V", "<Ewe>/V", "Voltage(V)"), "voltage in volts")
    current = _column(fields, ("current_a", "current_ma", "I/mA", "<I>/mA", "Current(A)", "Current(mA)"), "current with units", required=False)
    factor = .001 if current in {"current_ma", "I/mA", "<I>/mA", "Current(mA)"} else 1.0
    cycle = _column(fields, ("cycle", "cycle number", "Cycle_Index"), "cycle", required=False)
    step = _column(fields, ("step", "Ns", "Step_Index"), "step", required=False)
    channel = _column(fields, ("channel", "channel_id", "Channel"), "channel", required=False)
    cell = _column(fields, ("cell_id",), "cell", required=False)
    origin = utc_time(binding.clock_origin).timestamp() if elapsed and not binding.local_time_format else None
    half_cycle = "half cycle" if "half cycle" in fields else None
    points, duplicates, selected = [], 0, 0
    for index, row in enumerate(rows, 1):
        if index > MAX_ROWS:
            raise ValueError("Cycler row budget exceeded; select a bounded time range")
        if None in row:
            raise ValueError(f"Cycler row {index} has more values than named columns")
        if row.get("") not in (None, ""):
            raise ValueError(f"Cycler row {index} has data in an unnamed column")
        if channel and str(row[channel]).strip() != binding.channel:
            continue
        if cell and str(row[cell]).strip() != binding.cell_id:
            continue
        selected += 1
        try:
            if binding.local_time_format:
                seconds = local_calendar_time(row[absolute or elapsed], binding.local_time_format, binding.timezone_name)
            else:
                seconds = utc_time(str(row[absolute])).timestamp() if absolute else origin + finite_number(row[elapsed], "Elapsed time; calendar exports require a date format and timezone")
            protocol = str(row[step]).strip() if step else ""
            if half_cycle: protocol += " | half_cycle=" + str(row[half_cycle])
            point = CyclerPoint(seconds + binding.clock_offset_seconds,
                finite_number(row[voltage], "Voltage"), finite_number(row[current], "Current") * factor if current and row[current] not in (None, "") else None,
                str(row[cycle]).strip() if cycle else "", protocol)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Cycler row {index}: {exc}") from exc
        if points and point.timestamp <= points[-1].timestamp:
            if point == points[-1]:
                duplicates += 1; continue
            raise ValueError(f"Cycler row {index}: timestamps reset, decrease or have conflicting duplicates; split or resolve the series")
        points.append(point)
    if len(points) < 2:
        raise ValueError(f"Need at least two distinct samples for the bound cell/channel; found {selected}")
    return ElectrochemistrySeries(source_name, sha256(raw).hexdigest(), format_name, binding,
                                 tuple(points), duplicates, recorded_origin)


def summarize_exposure(series: ElectrochemistrySeries, start, end, *, sample_id):
    """Integrate electrical evidence over exactly one diffraction exposure."""
    if str(sample_id) != series.binding.sample_id:
        raise ValueError("Diffraction sample does not match the reviewed cycler binding")
    lo, hi = utc_time(start).timestamp(), utc_time(end).timestamp()
    if hi <= lo:
        raise ValueError("Exposure end must follow its start")
    points = series.points
    first = max(0, bisect_right(series.timestamps, lo)-1)
    coverage, current_coverage, sum_v, sum_i, extrema, branches, cycles, steps = 0., 0., 0., 0., [], set(), set(), set()
    excluded = []
    for index in range(first, len(points)-1):
        a, b = points[index], points[index+1]
        if a.timestamp >= hi:
            break
        left, right = max(lo, a.timestamp), min(hi, b.timestamp)
        if right <= left:
            continue
        reason = ""
        if b.timestamp-a.timestamp > series.binding.maximum_gap_seconds:
            reason = "log_gap"
        elif (a.cycle, a.step) != (b.cycle, b.step) or (a.current_a is not None and b.current_a is not None and series.branch(a) != series.branch(b)):
            reason = "protocol_boundary"
        if reason:
            excluded.append({"start_utc": datetime.fromtimestamp(left, timezone.utc).isoformat(),
                             "end_utc": datetime.fromtimestamp(right, timezone.utc).isoformat(), "reason": reason})
            continue
        fractions = [(left-a.timestamp)/(b.timestamp-a.timestamp), (right-a.timestamp)/(b.timestamp-a.timestamp)]
        volts = [a.voltage_v + f*(b.voltage_v-a.voltage_v) for f in fractions]
        dt = right-left
        coverage += dt; sum_v += sum(volts)/2*dt
        if a.current_a is not None and b.current_a is not None:
            currents = [a.current_a + f*(b.current_a-a.current_a) for f in fractions]
            sum_i += sum(currents)/2*dt; current_coverage += dt
        extrema.extend(volts); branches.add(series.branch(a))
        if a.cycle: cycles.add(a.cycle)
        if a.step: steps.add(a.step)
    duration = hi-lo
    voltage_status = "available" if math.isclose(coverage, duration, abs_tol=1e-6) else "partial" if coverage else "unavailable"
    current_status = "available" if math.isclose(current_coverage, duration, abs_tol=1e-6) else "partial" if current_coverage else "unavailable"
    status = "available" if voltage_status == current_status == "available" else "partial" if coverage else "unavailable"
    return {"schema": "radar-battery-exposure-electrochemistry/v1", "status": status,
            "sample_id": str(sample_id), "cell_id": series.binding.cell_id, "channel": series.binding.channel,
            "source_sha256": series.source_sha256, "start_utc": utc_time(start).isoformat(),
            "end_utc": utc_time(end).isoformat(), "duration_seconds": duration,
            "covered_seconds": coverage, "coverage_fraction": coverage/duration,
            "voltage_status": voltage_status, "current_status": current_status,
            "current_covered_seconds": current_coverage, "current_coverage_fraction": current_coverage/duration,
            "voltage_v": {"mean": sum_v/coverage, "minimum": min(extrema), "maximum": max(extrema)} if coverage else None,
            "current_a": {"mean": sum_i/current_coverage} if current_coverage else None,
            "signed_charge_c": sum_i if current_coverage else None,
            "branches": sorted(branches), "cycles": sorted(cycles), "steps": sorted(steps),
            "excluded_intervals": excluded, "clock_uncertainty_seconds": series.binding.clock_uncertainty_seconds,
            "method": "piecewise_linear_within_same_protocol_segment_no_extrapolation",
            "composition_status": "not_evaluated"}
