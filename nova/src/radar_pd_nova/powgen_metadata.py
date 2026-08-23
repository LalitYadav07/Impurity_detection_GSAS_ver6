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
from pathlib import Path
import math
import re
import shutil
import statistics
import subprocess
from typing import Any, Iterable


_MAX_LOG_VALUES = 200_000
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")
_DATA = re.compile(r"\bDATA\s*\{(.*?)\}", re.DOTALL)
_DATASPACE_SIZE = re.compile(r"DATASPACE\s+SIMPLE\s*\{\s*\(\s*(\d+)")
_UNIT = re.compile(r'ATTRIBUTE\s+"units?".*?\bDATA\s*\{\s*"([^"]*)"', re.DOTALL)
_QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class _Snapshot:
    source: str
    values: tuple[float, ...] = ()
    text: str = ""
    unit: str = ""


def powgen_nexus_path(ipts: str, run_number: int) -> Path:
    """Return the canonical read-only NeXus path for a PG3 run."""

    return Path("/SNS") / "PG3" / str(ipts).upper() / "nexus" / f"PG3_{int(run_number)}.nxs.h5"


def _h5dump_snapshot(path: Path, dataset: str) -> _Snapshot | None:
    executable = shutil.which("h5dump")
    if not executable:
        return None
    completed = subprocess.run(
        [executable, "-d", dataset, "-y", "-w", "0", str(path)],
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
    text_value = bytes(quoted[0], "utf-8").decode("unicode_escape").strip() if quoted else ""
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


def _text_value(path: Path, candidates: Iterable[str]) -> str:
    snapshot = _first_snapshot(path, candidates)
    return snapshot.text if snapshot is not None else ""


def _duration_seconds(start_time: str, end_time: str) -> float | None:
    if not start_time or not end_time:
        return None
    try:
        start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
    except ValueError:
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

    temperature = _numeric_log(
        _first_snapshot(
            path,
            (
                "/entry/DASlogs/BL11A:SE:SampleTemp/value",
                "/entry/sample/temperature",
            ),
        )
    )
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
        "available": True,
        "nexus_path": str(path).replace("\\", "/"),
        "run_title": _text_value(path, ("/entry/title",)),
        "experiment_title": _text_value(path, ("/entry/experiment_title",)),
        "sample_name": _text_value(path, ("/entry/sample/name",)),
        "sample_formula": _text_value(
            path,
            ("/entry/sample/chemical_formula", "/entry/sample/formula"),
        ),
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": _duration_seconds(start_time, end_time),
    }
    for key, value in (
        ("temperature", temperature),
        ("magnetic_field", magnetic_field),
        ("wavelength", wavelength),
        ("proton_charge", proton_charge),
    ):
        if value is not None:
            metadata[key] = value
    return metadata


__all__ = ["powgen_nexus_path", "read_powgen_scan_metadata"]
