"""Public ONCat metadata for browsing samples, never calibration authority.

Catalog identities are previews. Selected runs still use the independent,
revision-checked NeXus reader before they can be submitted for analysis.
No user credential is sent to this optional discovery endpoint.
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

import requests

from .facility_metadata import context, resolve_identity

MAX_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_ID_VALUES = 1000
TTL_SECONDS = 300
_CACHE: dict[tuple[str, str], tuple[float, dict[int, dict[str, Any]]]] = {}
_LOCK = threading.Lock()


def _values(value):
    if isinstance(value, list):
        if len(value) > MAX_ID_VALUES:
            raise ValueError("Catalog sample identity exceeds the bounded limit")
        result = []
        for item in value:
            if isinstance(item, (list, dict)):
                raise ValueError("Unsupported catalog sample identity shape")
            result.append(item)
        return result
    if isinstance(value, dict):
        raise ValueError("Unsupported catalog sample identity shape")
    return [] if value is None else [value]


def parse_sample_catalog(rows, instrument: str, ipts: str) -> dict[int, dict[str, Any]]:
    instrument, ipts = context(instrument, ipts)
    if not isinstance(rows, list):
        raise ValueError("Unexpected ONCat metadata response")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        path = str(row.get("location") or "")
        prefix = f"/SNS/{instrument}/{ipts}/"
        if not path.startswith(prefix):
            continue
        relative = path[len(prefix):]
        match = re.fullmatch(rf"(?:nexus/|data/|0/(\d+)/NeXus/){instrument}_(\d+)(?:_event)?\.(?:nxs\.h5|nxs)", relative)
        if not match or (match[1] and match[1] != match[2]):
            continue
        run = int(match[2])
        if run < 1:
            continue
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue
        entry = metadata.get("entry") or {}
        if not isinstance(entry, dict):
            continue
        if entry.get("run_number") is not None and str(entry["run_number"]).strip() != str(run):
            continue
        if entry.get("experiment_identifier") and str(entry["experiment_identifier"]).upper() != ipts:
            continue
        stored_instrument = entry.get("instrument") or {}
        if not isinstance(stored_instrument, dict):
            continue
        name = str(stored_instrument.get("name") or "").upper()
        aliases = {"PG3": {"PG3", "POWGEN", "POWGEN3"}, "NOM": {"NOM", "NOMAD"}, "SNAP": {"SNAP"}}
        if name and name not in aliases[instrument]:
            continue
        sample = entry.get("sample") or {}
        if not isinstance(sample, dict):
            continue
        evidence = []
        try:
            for field in ("id", "identifier", "sample_id", "sample_identifier"):
                evidence.extend({"value": v, "path": path, "kind": "oncat_catalog", "field": "/entry/sample/" + field}
                                for v in _values(sample.get(field)))
            logs = entry.get("DASlogs") or {}
            for field in ("SampleId", "BL11A:SE:SampleId"):
                log = logs.get(field) or {}
                evidence.extend({"value": v, "path": path, "kind": "oncat_catalog", "field": "/entry/DASlogs/" + field + "/value"}
                                for v in _values(log.get("value")))
        except (ValueError, AttributeError):
            continue  # This run needs the bounded source-file reader instead.
        if instrument == "SNAP":
            evidence = [e for e in evidence if str(e.get("value")) not in {"4294967295", "18446744073709551615"}]
        grouped.setdefault(run, []).append({"evidence": evidence,
            "sample_name": sample.get("name") if isinstance(sample.get("name"), str) else "",
            "run_title": entry.get("title") if isinstance(entry.get("title"), str) else ""})
    result = {}
    for run, records in grouped.items():
        identity = resolve_identity(e for r in records for e in r["evidence"])
        names = sorted({r["sample_name"].strip() for r in records if r["sample_name"].strip()})
        titles = sorted({r["run_title"].strip() for r in records if r["run_title"].strip()})
        result[run] = {**identity, "sample_name": " / ".join(names),
            "run_title": titles[0] if len(titles) == 1 else "",
            "identity_verification": "catalog_preview", "catalog_recorded": True}
    return result


def load_sample_catalog(instrument: str, ipts: str) -> dict[int, dict[str, Any]]:
    instrument, ipts = context(instrument, ipts)
    key = instrument, ipts
    with _LOCK:
        cached = _CACHE.get(key)
        if cached and time.monotonic() - cached[0] < TTL_SECONDS:
            return cached[1]
    projection = ["location", "metadata.entry.sample", "metadata.entry.title", "metadata.entry.run_number",
                  "metadata.entry.experiment_identifier", "metadata.entry.instrument.name",
                  "metadata.entry.DASlogs.SampleId.value", "metadata.entry.DASlogs.BL11A:SE:SampleId.value"]
    params = [("facility", "SNS"), ("instrument", instrument), ("experiment", ipts)]
    params.extend(("projection", field) for field in projection)
    with requests.get("https://oncat.ornl.gov/api/datafiles", params=params,
                      timeout=(5, 25), stream=True) as response:
        response.raise_for_status()
        content = bytearray()
        for chunk in response.iter_content(65536):
            content.extend(chunk)
            if len(content) > MAX_RESPONSE_BYTES:
                raise ValueError("ONCat sample metadata exceeds the bounded response limit")
        if response.headers.get("Link"):
            raise ValueError("ONCat sample metadata response is paginated; use source-file discovery")
    result = parse_sample_catalog(json.loads(content), instrument, ipts)
    with _LOCK:
        if len(_CACHE) >= 4:
            del _CACHE[min(_CACHE, key=lambda k: _CACHE[k][0])]
        _CACHE[key] = (time.monotonic(), result)
    return result
