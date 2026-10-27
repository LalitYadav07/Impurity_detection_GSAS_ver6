"""Versioned, bounded condition observations from named facility NXlogs.

Recorded readbacks, requests and candidate devices remain distinct. This module
does not infer units from titles, choose a pressure gauge from its magnitude,
or interpret beam charge as electrochemical charge. No event groups are read.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np

from .facility_metadata import parse_nexus_timestamp, qualify_temperature

SCHEMA = "radar-pd-conditions/v1"
REGISTRY = "SNS-PG3-NOM-conditions/2"
SNAP_REGISTRY = "SNS-SNAP-conditions/1"
MAX_VALUES = 200_000
MAX_TOTAL_VALUES = 500_000
MAX_POINTS = 512
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")
_QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')


def condition_display(metric, label):
    """Describe an acquisition summary without implying an equilibrium point."""
    if not isinstance(metric, dict):
        return ""
    value = metric.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return ""
    unit = str(metric.get("unit") or "[unit not reported]")
    statistic = "time-weighted mean " if metric.get("statistic") == "time_weighted_step_hold" else ""
    text = f"{label}: {statistic}{value:.6g} {unit}"
    low, high = metric.get("minimum"), metric.get("maximum")
    if (all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (low, high))
            and low <= high):
        text += f"; recorded scan range {low:.6g}–{high:.6g} {unit}"
    return text


@dataclass(frozen=True)
class Source:
    quantity: str
    channel: str
    role: str = "measured"
    needs_binding: bool = False
    family: str = ""


def sources(instrument):
    if instrument == "SNAP":
        # These are facility-wide channels, not proof of the device connected
        # to a specimen. Hydraulic/load pressure is not automatically the
        # sample pressure inside a DAC. Explicit reviewed binding is required.
        return [Source("temperature", name, needs_binding=True, family=name) for name in (
            "LakeshoreSampleTemp", "Lakeshore2SampleTemp", "Cryo100SampleTemp", "BL3:SE:CFurnace:Temp")]
    if instrument not in {"PG3", "NOM"}:
        raise ValueError("Condition registry supports registered neutron beamlines")
    p = "BL11A" if instrument == "PG3" else "BL1B"
    result = [Source("temperature", f"{p}:SE:SampleTemp", family="selected_temperature")]
    if instrument == "PG3":
        result += [Source("temperature", "SampleTemp", family="selected_temperature")]
        result += [
            Source("magnetic_field", name, needs_binding=True, family=family)
            for name, family in [
                ("BL11A:SE:MAG08:GetMGField", "MAG08"),
                ("Mag08ActualField", "MAG08"),
                ("BL11A:SE:MAG08:Field", "MAG08"),
                ("BL11A:SE:SS:ActualField", "SS"),
                ("BL11A:SE:SS:Field", "SS"),
            ]
        ]
        result += [
            Source("magnetic_field", "BL11A:SE:MAG08:GetFieldSet", "requested", family="MAG08")
        ]
    result += [Source("voltage", f"{p}:SE:SampleVolt", family="selected_voltage")]
    result += [
        Source("pressure", f"{p}:SE:MKS660:Pressure{i}", needs_binding=True, family="MKS660")
        for i in range(1, 5)
    ]
    for device in ["TD1"] if instrument == "PG3" else ["TD1", "TeledyneDNF", "TeledyneH"]:
        result += [
            Source("pressure", f"{p}:SE:{device}:Pressure", needs_binding=True, family=device)
        ]
    if instrument == "NOM":
        result += [
            Source(
                "gas_flow",
                f"{p}:SE:Alicat:{letter}VolFlow",
                needs_binding=True,
                family=f"Alicat:{letter}",
            )
            for letter in "ABC"
        ]
    return result


def _unit(quantity, unit):
    unit = str(unit).strip()
    if quantity == "temperature":
        if unit in {"K", "kelvin", "Kelvin"}:
            return "K", 1.0, 0.0
        if unit in {"C", "degC", "°C", "Celsius", "celsius"}:
            return "K", 1.0, 273.15
    maps = {
        "magnetic_field": {"T": ("T", 1), "tesla": ("T", 1), "mT": ("T", 0.001)},
        "voltage": {"V": ("V", 1), "kV": ("V", 1000), "mV": ("V", 0.001)},
        "pressure": {
            "Pa": ("Pa", 1),
            "kPa": ("Pa", 1000),
            "MPa": ("Pa", 1e6),
            "bar": ("Pa", 1e5),
            "Bar": ("Pa", 1e5),
            "mbar": ("Pa", 100),
            "mBar": ("Pa", 100),
            "PSIA": ("Pa", 6894.757293168),
            "psi": ("Pa", 6894.757293168),
        },
        # Reference gas conditions are retained in the unit; no conversion to
        # physical mass flow or between CCM and SCCM is invented.
        "gas_flow": {u: (u, 1) for u in ["SCCM", "sccm", "ml/min", "mL/min", "SLPM"]},
    }
    match = maps.get(quantity, {}).get(unit)
    return (*match, 0.0) if match else None


def _value(raw):
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace").strip()
    if hasattr(raw, "item"):
        raw = raw.item()
    return raw


def _sections(output):
    parts = re.split(r'\bDATASET\s+"([^"\n]+)"\s*\{', output)
    return dict(zip(parts[1::2], parts[2::2]))


def _shape(body):
    match = re.search(r"DATASPACE\s+SIMPLE\s*\{\s*\(([^)]*)\)", body)
    if match:
        return tuple(int(v) for v in re.findall(r"\d+", match[1]))
    return () if "DATASPACE  SCALAR" in body or re.search(r"DATASPACE\s+SCALAR", body) else None


def _native_parse(body):
    shape = _shape(body)
    if shape is None:
        return None
    data = re.search(r"\bDATA\s*\{(.*?)\}", body, re.S)
    if not data:
        return None
    quoted = _QUOTED.findall(data[1])
    attrs = {}
    from .powgen_metadata import _decode_h5dump_text

    for name in ["units", "unit", "start", "offset", "scaling_factor"]:
        match = re.search(r'ATTRIBUTE\s+"' + name + r'".*?\bDATA\s*\{(.*?)\}', body, re.S)
        if match:
            strings = _QUOTED.findall(match[1])
            numbers = _NUMBER.findall(match[1])
            attrs[name] = (
                _decode_h5dump_text(strings[0])
                if strings
                else float(numbers[0]) if len(numbers) == 1 else None
            )
    if quoted:
        return {"text": _decode_h5dump_text(quoted[0]), "attrs": attrs, "shape": shape}
    values = [float(v) for v in _NUMBER.findall(data[1])]
    size = math.prod(shape) if shape else 1
    if len(values) != size or not all(math.isfinite(v) for v in values):
        return {"error": "Nonfinite, incomplete or nonnumeric logged values"}
    return {"values": values, "attrs": attrs, "shape": shape}


def _read_native(path, datasets):
    def run(arguments, limit):
        with tempfile.TemporaryFile() as output:
            subprocess.run(
                ["h5dump", "-m", "%.17g", "-y", *arguments, str(path)],
                stdout=output,
                stderr=subprocess.DEVNULL,
                timeout=40,
                check=False,
            )
            output.seek(0)
            raw = output.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("Condition metadata exceeds its bounded size")
        return raw.decode("utf-8", errors="replace")

    args = [item for name in datasets for item in ["-d", name]]
    headers = _sections(run(["-H", *args], 4_000_000))
    allowed, result, budget = [], {}, MAX_TOTAL_VALUES
    for name in datasets:
        if name not in headers:
            continue
        shape = _shape(headers[name])
        size = math.prod(shape) if shape else 1
        if shape is None or len(shape) > 1 or size < 1 or size > MAX_VALUES or size > budget:
            result[name] = {"error": "Unsupported shape or bounded log size"}
            continue
        allowed.append(name)
        budget -= size
    if allowed:
        output = _sections(run([item for name in allowed for item in ["-d", name]], 32_000_000))
        result.update({name: _native_parse(output[name]) for name in allowed if name in output})
    return result


def _read_h5py(path, datasets):
    import h5py

    result, budget = {}, MAX_TOTAL_VALUES
    with h5py.File(path, "r") as handle:
        for name in datasets:
            dataset = handle.get(name)
            if not isinstance(dataset, h5py.Dataset):
                continue
            # Variable-length strings may point to arbitrarily large payloads;
            # only bounded fixed strings and numeric metadata are supported.
            if (
                dataset.ndim > 1
                or dataset.size < 1
                or dataset.size > MAX_VALUES
                or dataset.size > budget
                or dataset.nbytes > 8_000_000
                or (
                    dataset.dtype.kind not in "iufS"
                    and not (
                        dataset.dtype.kind == "O"
                        and dataset.size == 1
                        and name in {"/entry/title", "/entry/start_time", "/entry/end_time"}
                    )
                )
            ):
                result[name] = {"error": "Unsupported shape or bounded log size"}
                continue
            budget -= dataset.size
            attrs = {
                key: _value(value)
                for key, value in dataset.attrs.items()
                if key in {"units", "unit", "start", "offset", "scaling_factor"}
                and np.size(value) == 1
            }
            raw = np.asarray(dataset[()]).reshape(-1)
            if raw.dtype.kind in "iuf":
                values = raw.astype(float)
                result[name] = (
                    {"values": values.tolist(), "attrs": attrs, "shape": dataset.shape}
                    if np.all(np.isfinite(values))
                    else {"error": "Nonfinite logged values"}
                )
            else:
                text = str(_value(raw[0]))
                result[name] = (
                    {"text": text, "attrs": attrs, "shape": dataset.shape}
                    if len(text) <= 16384
                    else {"error": "Metadata text exceeds its supported size"}
                )
    return result


def _thin(times, values):
    if len(values) <= MAX_POINTS:
        return list(range(len(values)))
    # Retain local extrema as well as endpoints; uniform decimation alone can
    # hide a short field/gas/temperature excursion.
    result = {0, len(values) - 1}
    for chunk in np.array_split(np.arange(len(values)), (MAX_POINTS - 2) // 2):
        result.add(int(chunk[np.argmin(values[chunk])]))
        result.add(int(chunk[np.argmax(values[chunk])]))
    return sorted(result)


def summarize_log(spec, value, time, window, *, title="", binding=None):
    out = {
        "quantity": spec.quantity,
        "role": spec.role,
        "channel": spec.channel,
        "device_family": spec.family,
        "status": "unavailable",
        "notices": [],
        "requires_binding": spec.needs_binding,
        "binding_verified": bool(binding and binding.get("verified") is True
                                 and binding.get("channel") == spec.channel),
    }
    if not value:
        return out
    if value.get("error"):
        return {**out, "status": "invalid", "notices": [value["error"]]}
    source_unit = value.get("attrs", {}).get("units", value.get("attrs", {}).get("unit", ""))
    out["source_unit"] = source_unit
    conversion = _unit(spec.quantity, source_unit)
    raw = np.asarray(value.get("values") or [], dtype=float)
    if not conversion or not len(raw):
        return {
            **out,
            "status": "unqualified_units",
            "notices": ["Channel meaning/units require review; no unit was guessed"],
        }
    unit, factor, offset = conversion
    values = raw * factor + offset
    if not np.all(np.isfinite(values)):
        return {**out, "status": "invalid", "notices": ["Converted channel values are nonfinite"]}
    out.update(
        unit=unit, minimum=float(np.min(values)), maximum=float(np.max(values)), samples=len(values)
    )
    if spec.quantity == "temperature":
        metric = {
            "value": float(np.median(values)),
            "unit": "K",
            "minimum": out["minimum"],
            "maximum": out["maximum"],
        }
        if offset:
            metric["converted_from"] = source_unit
        qualified, notices = qualify_temperature(metric, title)
        if not qualified:
            return {**out, "status": "invalid", "notices": notices}
    if spec.needs_binding and not (
        binding and binding.get("verified") is True and binding.get("channel") == spec.channel
    ):
        out["notices"].append(
            "Select the device connected to this specimen before using this quantity"
        )
        out["status"] = "needs_mapping"
    if not time or time.get("error") or not window.get("start") or not window.get("end"):
        out["status"] = (
            "needs_mapping" if out["status"] == "needs_mapping" else "needs_time_alignment"
        )
        out["update_median"] = float(np.median(values))
        return out
    attrs = time.get("attrs", {})
    if attrs.get("units", attrs.get("unit")) not in {"s", "second", "seconds"}:
        return {
            **out,
            "status": "needs_time_alignment",
            "notices": [*out["notices"], "Unsupported log time units"],
        }
    try:
        start, end = (parse_nexus_timestamp(window[key]) for key in ["start", "end"])
        origin = parse_nexus_timestamp(attrs.get("start") or attrs.get("offset") or "")
        if any(v.tzinfo is None for v in [start, end, origin]):
            raise ValueError("Timezone missing")
        duration = (end - start).total_seconds()
        if duration <= 0:
            raise ValueError("Acquisition interval is empty")
        scaling = float(attrs.get("scaling_factor", 1))
        if not math.isfinite(scaling) or scaling <= 0:
            raise ValueError("Log time scaling must be finite and positive")
        times = (
            np.asarray(time.get("values") or [], dtype=float) * scaling
            + (origin - start).total_seconds()
        )
        if (
            times.shape != values.shape
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError("Time/value lengths or log ordering are inconsistent")
        edges = np.clip(np.r_[times, duration], 0, duration)
        weights = np.maximum(np.diff(edges), 0)
        valid = weights > 0
        covered = float(np.sum(weights))
        if covered <= 0:
            raise ValueError("Log does not overlap acquisition")
        kept_times, kept_values = times[valid], values[valid]
        thin = _thin(kept_times, kept_values)
        out.update(
            value=float(np.dot(weights, values) / covered),
            minimum=float(np.min(kept_values)),
            maximum=float(np.max(kept_values)),
            coverage_fraction=covered / duration,
            statistic="time_weighted_step_hold",
            coverage_policy="NXlog held state; no backfill before first observation",
            tail_hold_seconds=max(0, duration - float(times[-1])),
            trajectory={
                "seconds": kept_times[thin].tolist(),
                "values": kept_values[thin].tolist(),
                "points_are_extrema_preserving_subset": len(thin) != len(kept_values),
            },
            acquisition_window=window,
        )
        if out["status"] != "needs_mapping":
            out["status"] = "recorded" if covered / duration >= 0.95 else "incomplete_coverage"
        if spec.role == "requested":
            out["status"] = "requested"
    except (ValueError, TypeError, KeyError) as exc:
        out.update(status="needs_time_alignment", notices=[*out["notices"], str(exc)])
    return out


def read_conditions(path, instrument, *, window=None, bindings=None, backend=None):
    """Read a condition packet; every quantity has its own qualification state."""
    path = Path(path)
    try:
        stat = path.stat()
    except OSError as exc:
        return {
            "$schema": SCHEMA,
            "registry": SNAP_REGISTRY if instrument == "SNAP" else REGISTRY,
            "status": "unavailable",
            "channels": [],
            "notices": [str(exc)],
        }
    packet = _cached_read(
        str(path),
        stat.st_size,
        stat.st_mtime_ns,
        instrument,
        json.dumps(window, sort_keys=True),
        json.dumps(bindings or {}, sort_keys=True),
        backend,
    )
    # The cache owns its receipt; a caller must not mutate another watch's data.
    return json.loads(json.dumps(packet, allow_nan=False))


def qualified_metadata(metadata):
    """Prevent a legacy scalar from bypassing an explicit device/unit conflict.

    This also qualifies already-saved checkpoints. Historic summaries without
    registry observations retain their existing contract. Rejected scalars
    remain observations for review, rather than selected plotting quantities.
    """
    result = dict(metadata) if isinstance(metadata, dict) else {}
    packet = result.get("conditions") or {}
    if not isinstance(packet, dict):
        return result
    if packet.get("registry") == "SNS-PG3-NOM-conditions/1":
        # Earlier receipts treated an isolated magnet channel as selected.
        # Such channels are facility-wide logs, not a specimen selector. Keep
        # the original receipt hash and expose an explicit policy projection.
        packet = {**packet, "channels": [dict(c) for c in (packet.get("channels") or []) if isinstance(c, dict)]}
        changed = False
        for channel in packet["channels"]:
            if (channel.get("quantity") == "magnetic_field" and channel.get("role") == "measured"
                    and channel.get("status") == "recorded" and not channel.get("binding_verified")):
                channel.update(status="needs_mapping", requires_binding=True,
                               notices=[*channel.get("notices", []), "Verify that this device measured the specimen"])
                changed = True
        if changed:
            packet["qualified_from_registry"] = packet["registry"]
            packet["registry"] = REGISTRY
            packet["source_metadata_sha256"] = packet.pop("metadata_sha256", None)
            packet["metadata_sha256"] = sha256(json.dumps(packet, sort_keys=True, allow_nan=False).encode()).hexdigest()
            result["conditions"] = packet
    channels = [c for c in (packet.get("channels") or [])
                if isinstance(c, dict) and c.get("role") == "measured"]
    for quantity in {c.get("quantity") for c in channels if c.get("quantity")}:
        matching = [c for c in channels if c.get("quantity") == quantity]
        recorded = [c for c in matching if c.get("status") == "recorded"]
        if recorded:
            channel = recorded[0]
            result[quantity] = {
                **channel,
                "quality": "changing" if channel.get("minimum") != channel.get("maximum") else "recorded",
            }
        elif any(c.get("status") in {"needs_mapping", "conflict", "unqualified_units"} for c in matching):
            previous = result.pop(quantity, None)
            if previous is not None:
                result.setdefault(quantity + "_observation", previous)
    return result


def attach_conditions(metadata, path, instrument):
    """Prefer qualified readbacks; never resurrect a rejected legacy metric."""
    return qualified_metadata({**metadata, "conditions": read_conditions(path, instrument)})


@lru_cache(maxsize=128)
def _cached_read(path, size, mtime, instrument, window_json, bindings_json, backend):
    specs = sources(instrument)
    keys = ["/entry/start_time", "/entry/end_time", "/entry/title"]
    keys += [
        f"/entry/DASlogs/{spec.channel}/{field}" for spec in specs for field in ["value", "time"]
    ]
    keys = list(dict.fromkeys(keys))
    packet = {
        "$schema": SCHEMA,
        "registry": SNAP_REGISTRY if instrument == "SNAP" else REGISTRY,
        "source_path": path,
        "source_stat": {"bytes": size, "mtime_ns": mtime},
        "channels": [],
        "notices": [],
    }
    try:
        use_native = backend == "h5dump" or (backend is None and shutil.which("h5dump"))
        data = (_read_native if use_native else _read_h5py)(path, keys)
        requested_window = json.loads(window_json)
        window = requested_window or {
            "start": (data.get("/entry/start_time") or {}).get("text"),
            "end": (data.get("/entry/end_time") or {}).get("text"),
        }
        packet["acquisition_window"] = window
        packet["window_scope"] = (
            "explicit_product_window"
            if requested_window
            else "parent_run; verify reduction does not select a narrower time interval"
        )
        bindings = json.loads(bindings_json)
        title = (data.get("/entry/title") or {}).get("text", "")
        for spec in specs:
            prefix = f"/entry/DASlogs/{spec.channel}/"
            if prefix + "value" not in data:
                continue
            channel = summarize_log(
                spec,
                data.get(prefix + "value"),
                data.get(prefix + "time"),
                window,
                title=title,
                binding=bindings.get(spec.quantity),
            )
            channel["source"] = prefix + "value"
            packet["channels"].append(channel)
        # Distinct devices are never selected by largest or nonzero value.
        for quantity in {s.quantity for s in specs}:
            measured = [
                c
                for c in packet["channels"]
                if c["quantity"] == quantity
                and c["role"] == "measured"
                and c["status"] == "recorded"
            ]
            binding = bindings.get(quantity)
            families = {c["device_family"] for c in measured}
            if binding and binding.get("verified") is True:
                for channel in measured:
                    if channel["channel"] != binding.get("channel"):
                        channel["status"] = "not_selected"
            elif len(families) > 1:
                for channel in measured:
                    channel.update(
                        status="needs_mapping",
                        notices=[
                            *channel["notices"],
                            "Multiple recorded devices; select the specimen-linked readback",
                        ],
                    )
            elif len(measured) > 1:
                values = [c["value"] for c in measured]
                if not np.allclose(values, values[0], rtol=1e-9, atol=1e-8):
                    for channel in measured:
                        channel.update(
                            status="conflict",
                            notices=[*channel["notices"], "Recorded aliases disagree"],
                        )
                else:
                    for channel in measured[1:]:
                        channel["status"] = "duplicate_alias"
        after = Path(path).stat()
        if (after.st_size, after.st_mtime_ns) != (size, mtime):
            raise ValueError("Condition source changed while reading")
        packet["status"] = "read"
    except (OSError, ValueError, TypeError, ImportError, subprocess.TimeoutExpired) as exc:
        packet.update(status="unavailable", channels=[], notices=[str(exc)])
    packet["metadata_sha256"] = sha256(
        json.dumps(packet, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    # Cached packets are returned by value, so UI annotation cannot mutate the
    # original measurement receipt or another watch's condition record.
    return json.loads(json.dumps(packet, allow_nan=False))
