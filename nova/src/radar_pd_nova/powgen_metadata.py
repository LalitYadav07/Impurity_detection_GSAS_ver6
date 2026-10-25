"""Read a bounded scientific summary from one POWGEN NeXus file.

The live monitor remains read-only with respect to the IPTS. Only small,
explicit metadata datasets are read; detector event arrays are never visited.
Deployed NOVA images use the native ``h5dump`` command so their Python/NumPy
environment is not changed. Tests and developer environments may use h5py as
a compatible fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import math
import re
import shutil
import statistics
import subprocess
from typing import Any, Iterable

from .facility_metadata import resolve_nexus_path, resolve_identity, qualify_temperature, parse_nexus_timestamp


_MAX_LOG_VALUES = 200_000
_METADATA_SCHEMA_VERSION = 5
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")
_DATA = re.compile(r"\bDATA\s*\{(.*?)\}", re.DOTALL)
_DATASPACE_SIZE = re.compile(r"DATASPACE\s+SIMPLE\s*\{\s*\(\s*(\d+)")
_UNIT = re.compile(r'ATTRIBUTE\s+"units?".*?\bDATA\s*\{\s*"([^"]*)"', re.DOTALL)
_QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')
_SAMPLE_ID_PATHS = (
    "/entry/sample/id", "/entry/sample/identifier", "/entry/sample/sample_id",
    "/entry/sample/sample_identifier", "/entry/DASlogs/SampleId/value",
    "/entry/DASlogs/BL11A:SE:SampleId/value",
)


def _decode_h5dump_text(text: str) -> str:
    # Some HDF5 builds print UTF-8 bytes as sign-extended octal escapes.
    # Decode those as bytes, rather than interpreting them as Unicode points.
    data = bytearray()
    for part in re.split(r"(\\[0-7]{3,11}|\\.)", text):
        if re.fullmatch(r"\\[0-7]{3,11}", part):
            data.append(int(part[1:], 8) & 255)
        elif part.startswith("\\") and len(part) == 2:
            data.extend({"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}.get(part[1], part).encode("utf-8"))
        else:
            data.extend(part.encode("utf-8"))
    return data.decode("utf-8", errors="replace").strip()


def _sample_identifier_number(raw: str) -> str:
    value = Decimal(raw)
    if not value.is_finite():
        return ""
    return str(int(value)) if value == value.to_integral_value() else format(value.normalize(), "f")


def read_powgen_sample_identity(ipts: str, run_number: int, *, nexus_path=None, instrument="PG3") -> dict[str, Any]:
    """Read bounded sample identity fields, never detector event data."""

    path = Path(nexus_path) if nexus_path is not None else powgen_nexus_path(ipts, run_number)
    fields = (*_SAMPLE_ID_PATHS, "/entry/sample/name", "/entry/run_number", "/entry/experiment_identifier", "/entry/instrument/name")
    numeric_fields = (*_SAMPLE_ID_PATHS, "/entry/run_number")
    before = path.stat()
    values: dict[str, str] = {}
    identity_observations: dict[str, list[str]] = {}
    executable = shutil.which("h5dump")
    if executable:
        # Request only named sample fields. Missing aliases do not invalidate
        # fields which h5dump successfully returned in the same invocation.
        command = [executable, "-y", "-w", "0", "-m", "%.17g"]
        for field in fields:
            command.extend(("-d", field))
        # Check declared shapes before requesting values, including identity
        # time series. A corrupt/oversized field must not allocate huge output.
        header = subprocess.run(
            [*command, '-H', str(path)], capture_output=True, text=True,
            encoding='utf-8', errors='replace', timeout=20, check=False,
        )
        if header.returncode and not re.search(r'\bDATASET\s+"', header.stdout):
            raise OSError('Cannot read raw HDF5 identity metadata')
        for dimensions in re.findall(r'DATASPACE\s+SIMPLE\s*\{\s*\(([^)]*)\)', header.stdout):
            sizes = [int(n) for n in re.findall(r'\d+', dimensions)]
            if math.prod(sizes)>1000:
                raise ValueError('Sample identity log exceeds the bounded identity limit')
        completed = subprocess.run(
            [*command, str(path)], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=20, check=False,
        )
        sections = re.split(r'(?m)^\s*DATASET "([^"]+)" \{', completed.stdout)
        for field, body in zip(sections[1::2], sections[2::2]):
            match = _DATA.search(body)
            if not match:
                continue
            quoted = _QUOTED.findall(match.group(1))
            if quoted:
                values[field] = _decode_h5dump_text(quoted[0])
                if field in _SAMPLE_ID_PATHS:
                    if len(quoted)>1000:raise ValueError('Sample identity log exceeds the bounded identity limit')
                    identity_observations[field] = [_decode_h5dump_text(v) for v in quoted]
            elif field in numeric_fields:
                numeric = _NUMBER.search(match.group(1))
                if numeric:
                    values[field] = _sample_identifier_number(numeric.group())
                if field in _SAMPLE_ID_PATHS:
                    tokens = _NUMBER.findall(match.group(1))
                    if len(tokens)>1000:raise ValueError('Sample identity log exceeds the bounded identity limit')
                    identity_observations[field] = [_sample_identifier_number(v) for v in tokens]
    else:
        import h5py  # type: ignore

        with h5py.File(path, "r") as handle:
            for field in fields:
                dataset = handle.get(field)
                if not isinstance(dataset, h5py.Dataset) or not dataset.size:
                    continue
                if dataset.size>1000:
                    raise ValueError('Sample identity log exceeds the bounded identity limit')
                # Read all bounded ID observations below to detect changes
                # within a run, even when the first and last values agree.
                raw = dataset[tuple(0 for _ in dataset.shape)] if dataset.shape else dataset[()]
                if isinstance(raw, bytes):
                    values[field] = raw.decode("utf-8", errors="replace").strip()
                elif field in numeric_fields and (
                    isinstance(raw, (int, float)) or getattr(getattr(raw, "dtype", None), "kind", "") in {"i", "u", "f"}
                ):
                    values[field] = _sample_identifier_number(str(raw))
                else:
                    values[field] = str(raw).strip()
                if field in _SAMPLE_ID_PATHS:
                    if dataset.size>1000:raise ValueError('Sample identity log exceeds the bounded identity limit')
                    observations=[]
                    for item in dataset[()].reshape(-1) if dataset.shape else [raw]:
                        if isinstance(item,bytes):observations.append(item.decode('utf-8',errors='replace').strip())
                        elif getattr(getattr(item,'dtype',None),'kind','') in {'i','u','f'} or isinstance(item,(int,float)):
                            observations.append(_sample_identifier_number(str(item)))
                        else:observations.append(str(item).strip())
                    identity_observations[field]=observations
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Raw identity metadata changed during inspection; retry")
    if values.get("/entry/run_number") and values["/entry/run_number"] != str(run_number):
        raise ValueError("NeXus run identifier disagrees with the selected run")
    if values.get("/entry/experiment_identifier") and values["/entry/experiment_identifier"].upper() != str(ipts).upper():
        raise ValueError("NeXus proposal disagrees with the selected IPTS")
    stored_instrument = values.get('/entry/instrument/name', '').upper()
    aliases = {'POWGEN':'PG3','PG3':'PG3','NOMAD':'NOM','NOM':'NOM'}
    if stored_instrument in aliases and aliases[stored_instrument] != instrument:
        raise ValueError('NeXus instrument disagrees with the selected beamline')
    result = resolve_identity([{"value": observed, "field": field, "path": str(path), "kind": "nexus"}
                               for field in _SAMPLE_ID_PATHS if field in values
                               for observed in dict.fromkeys(identity_observations.get(field,[values[field]]))])
    result["sample_name"] = values.get("/entry/sample/name", "")
    return result


@dataclass(frozen=True)
class _Snapshot:
    source: str
    values: tuple[float, ...] = ()
    text: str = ""
    unit: str = ""


def powgen_nexus_path(ipts: str, run_number: int) -> Path:
    """Return the canonical read-only NeXus path for a PG3 run."""

    return resolve_nexus_path("PG3", ipts, run_number)


def _h5dump_snapshot(path: Path, dataset: str) -> _Snapshot | None:
    executable = shutil.which("h5dump")
    if not executable:
        return None
    completed = subprocess.run(
        [executable, "-d", dataset, "-y", "-w", "0", "-m", "%.17g", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
        check=False,
    )
    if completed.returncode != 0:
        return None
    output = completed.stdout
    size_match = _DATASPACE_SIZE.search(output)
    if size_match and int(size_match.group(1)) > _MAX_LOG_VALUES:
        return None
    data_match = _DATA.search(output)
    if not data_match:
        return None
    body = data_match.group(1)
    quoted = _QUOTED.findall(body)
    text_value = _decode_h5dump_text(quoted[0]) if quoted else ""
    values: list[float] = []
    if not quoted:
        for token in _NUMBER.findall(body):
            try:
                value = float(token)
            except ValueError:
                continue
            if math.isfinite(value):
                values.append(value)
            if len(values) > _MAX_LOG_VALUES:
                return None
    unit_match = _UNIT.search(output)
    return _Snapshot(
        source=dataset,
        values=tuple(values),
        text=text_value,
        unit=unit_match.group(1).strip() if unit_match else "",
    )


def _h5py_snapshot(path: Path, dataset: str) -> _Snapshot | None:
    """Developer fallback used only when h5dump is unavailable."""

    try:
        import h5py  # type: ignore
        import numpy as np
    except (ImportError, ValueError):
        return None
    try:
        with h5py.File(path, "r") as handle:
            value = handle.get(dataset)
            if not isinstance(value, h5py.Dataset):
                return None
            size = int(np.prod(value.shape)) if value.shape else 1
            if size < 1 or size > _MAX_LOG_VALUES:
                return None
            raw = np.asarray(value[()]).reshape(-1)
            unit_value = value.attrs.get("units", value.attrs.get("unit", ""))
            if isinstance(unit_value, bytes):
                unit = unit_value.decode("utf-8", errors="replace").strip()
            else:
                unit = str(unit_value or "").strip()
            if raw.dtype.kind in "iuf":
                values = tuple(float(item) for item in raw if math.isfinite(float(item)))
                return _Snapshot(source=dataset, values=values, unit=unit)
            first = raw[0]
            text_value = (
                first.decode("utf-8", errors="replace").strip()
                if isinstance(first, bytes)
                else str(first).strip()
            )
            return _Snapshot(source=dataset, text=text_value, unit=unit)
    except (OSError, TypeError, ValueError):
        return None


def _first_snapshot(path: Path, candidates: Iterable[str]) -> _Snapshot | None:
    reader = _h5dump_snapshot if shutil.which("h5dump") else _h5py_snapshot
    for dataset in candidates:
        snapshot = reader(path, dataset)
        if snapshot is not None:
            return snapshot
    return None


def _numeric_log(snapshot: _Snapshot | None) -> dict[str, Any] | None:
    if snapshot is None or not snapshot.values:
        return None
    values = snapshot.values
    return {
        "value": round(float(statistics.median(values)), 8),
        "unit": snapshot.unit,
        "first": round(values[0], 8),
        "last": round(values[-1], 8),
        "minimum": round(min(values), 8),
        "maximum": round(max(values), 8),
        "samples": len(values),
        "source": snapshot.source,
    }


def _temperature_in_kelvin(metric: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize an explicitly labelled temperature while retaining provenance."""

    if metric is None:
        return None
    unit = str(metric.get("unit") or "").strip()
    normalized = re.sub(r"[\s_.-]+", "", unit.lower().replace("°", "deg"))
    if normalized in {"k", "kelvin", "degk", "degreek", "degreeskelvin"}:
        metric["unit"] = "K"
        metric["source_unit"] = unit or "K"
        return metric
    if normalized not in {
        "c",
        "degc",
        "celsius",
        "degreec",
        "degreesc",
        "degreecelsius",
        "degreescelsius",
    }:
        # Never infer a temperature unit from magnitude or a run-title suffix.
        return metric

    converted = dict(metric)
    converted["source_unit"] = unit
    converted["source_value"] = metric.get("value")
    for key in ("value", "first", "last", "minimum", "maximum"):
        value = metric.get(key)
        if value is not None:
            converted[key] = round(float(value) + 273.15, 8)
    converted["unit"] = "K"
    converted["converted_from"] = unit
    return converted


def _text_value(path: Path, candidates: Iterable[str]) -> str:
    snapshot = _first_snapshot(path, candidates)
    return snapshot.text if snapshot is not None else ""


def _identifier_value(path: Path, candidates: Iterable[str]) -> str:
    """Read a textual or integer NeXus identifier without loading large arrays."""

    snapshot = _first_snapshot(path, candidates)
    if snapshot is None:
        return ""
    if snapshot.text:
        return snapshot.text
    if snapshot.values:
        value = snapshot.values[0]
        return str(int(value)) if float(value).is_integer() else f"{value:g}"
    return ""


def _duration_seconds(start_time: str, end_time: str) -> float | None:
    if not start_time or not end_time:
        return None
    try:
        start = parse_nexus_timestamp(start_time)
        end = parse_nexus_timestamp(end_time)
    except ValueError:
        return None
    if (start.tzinfo is None) != (end.tzinfo is None):
        return None
    return round(max(0.0, (end - start).total_seconds()), 3)


def read_powgen_scan_metadata(
    ipts: str,
    run_number: int,
    *,
    nexus_path: str | Path | None = None,
) -> dict[str, Any]:
    """Extract display-ready conditions without loading detector event data."""

    path = Path(nexus_path) if nexus_path is not None else powgen_nexus_path(ipts, run_number)
    if not path.is_file():
        return {"nexus_path": str(path).replace("\\", "/"), "available": False}

    # Prefer the facility's bounded selected-channel mean. A median of
    # irregular updates is a different statistic, retained only as fallback.
    prefix = '/entry/DASlogs/BL11A:SE:SampleTemp/'
    mean = _numeric_log(_first_snapshot(path, (prefix+'average_value', '/entry/DASlogs/SampleTemp/average_value')))
    if mean:
        mean['statistic'] = 'mean'
        for key, name in [('minimum','minimum_value'),('maximum','maximum_value')]:
            bound = _numeric_log(_first_snapshot(path, (prefix+name, '/entry/DASlogs/SampleTemp/'+name)))
            if bound and bound['unit'] == mean['unit']:
                mean[key] = bound['value']
        temperature = _temperature_in_kelvin(mean)
    else:
        temperature = _temperature_in_kelvin(_numeric_log(_first_snapshot(path,
            (prefix+'value', '/entry/sample/temperature'))))
        if temperature:temperature['statistic'] = 'median_of_recorded_updates'
    magnetic_field = _numeric_log(
        _first_snapshot(
            path,
            (
                "/entry/DASlogs/BL11A:SE:SS:Field/value",
                "/entry/DASlogs/BL11A:SE:MAG08:Field/value",
            ),
        )
    )
    wavelength = _numeric_log(
        _first_snapshot(
            path,
            (
                "/entry/DASlogs/BL11A:Chop:Skf1:WavelengthUserReq/value",
                "/entry/instrument/monochromator/wavelength",
            ),
        )
    )
    proton_charge = _numeric_log(_first_snapshot(path, ("/entry/proton_charge",)))
    if proton_charge and proton_charge.get("unit", "").lower() in {"picocoulombs", "pc"}:
        for key in ("value", "first", "last", "minimum", "maximum"):
            proton_charge[key] = round(float(proton_charge[key]) * 1.0e-12, 8)
        proton_charge["unit"] = "C"

    start_time = _text_value(path, ("/entry/start_time",))
    end_time = _text_value(path, ("/entry/end_time",))
    metadata: dict[str, Any] = {
        "schema_version": _METADATA_SCHEMA_VERSION,
        "available": True,
        "nexus_path": str(path).replace("\\", "/"),
        "run_title": _text_value(path, ("/entry/title",)),
        "experiment_title": _text_value(path, ("/entry/experiment_title",)),
        "sample_id": _identifier_value(
            path,
            (
                "/entry/sample/id",
                "/entry/sample/identifier",
                "/entry/sample/sample_id",
                "/entry/sample/sample_identifier",
                "/entry/DASlogs/SampleId/value",
                "/entry/DASlogs/BL11A:SE:SampleId/value",
            ),
        ),
        "sample_name": _text_value(path, ("/entry/sample/name",)),
        "sample_formula": _text_value(
            path,
            ("/entry/sample/chemical_formula", "/entry/sample/formula"),
        ),
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": _duration_seconds(start_time, end_time),
    }
    metadata.update(read_powgen_sample_identity(ipts, run_number, nexus_path=path))
    metadata['temperature_observation'] = temperature
    temperature, notices = qualify_temperature(temperature, metadata["run_title"])
    metadata["metadata_notices"] = notices
    metadata["temperature_sensor"] = _text_value(path, ("/entry/DASlogs/BL11A:SE:SelectSampleTemp/value_strings",))
    for key, fields in [
        ('chopper_frequency_request', ('/entry/DASlogs/BL11A:Chop:Skf1:SpeedUserReq/value',)),
        ('chopper_speed_readback', ('/entry/DASlogs/BL11A:Chop:Skf1:MotorSpeed/average_value', '/entry/DASlogs/Speed1/average_value')),
        ('beam_frequency', ('/entry/DASlogs/frequency/average_value',))]:
        observation = _numeric_log(_first_snapshot(path, fields))
        if observation:metadata[key] = observation
    for key, value in (
        ("temperature", temperature),
        ("magnetic_field", magnetic_field),
        ("wavelength", wavelength),
        ("proton_charge", proton_charge),
    ):
        if value is not None:
            metadata[key] = value
    from .scan_conditions import attach_conditions
    return attach_conditions(metadata, path, "PG3")


__all__ = ["powgen_nexus_path", "read_powgen_scan_metadata"]
