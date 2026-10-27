"""Bounded metadata sources shared by NOMAD and POWGEN.

Scan tables are display/identity evidence, never calibration authority. Each
field is resolved independently across duplicate rows; conflicts are retained.
No detector events, facility scripts, database writes or guessed units.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from datetime import datetime
from hashlib import sha256
import io
from pathlib import Path
import re
from typing import Any, Iterable
import xml.etree.ElementTree as ET

MAX_TABLE_BYTES = 16 * 1024 * 1024
MAX_TABLE_ROWS = 250_000
MAX_XML_BYTES = 1024 * 1024
VERSION = "facility-metadata/v1"
_MISSING = {"", "none", "null", "unknown", "n/a", "na", "nan", "(unset)", "-1", "0"}
_FIELDS = {"ITEMS-ID": "sample_id", "SAMPLE": "sample_name", "TITLE": "run_title",
           "FORMULA": "registered_formula", "TEMP": "table_temperature",
           "SE": "recorded_selector", "WAVELENGTH": "chopper_wavelength_request",
           "FREQUENCY": "chopper_speed_readback", "GUIDE": "guide",
           "VAN RUN": "vanadium_run", "VAN BACKGROUND RUN": "vanadium_background_run",
           "EMPTY CAN RUN": "empty_container_run", "CONTAINER": "registered_container"}


def parse_nexus_timestamp(value: Any) -> datetime:
    """Parse facility nanoseconds on Python 3.10, preserving the UTC offset.

    datetime resolution is microseconds. Trim only excess fractional digits;
    original timestamp strings remain unchanged in the metadata record.
    """
    text = str(value).strip()
    text = re.sub(r"(\.\d{6})\d+(?=(?:Z|[+-]\d{2}:?\d{2})?$)", r"\1", text)
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def sample_identifier(value: Any) -> str:
    """Preserve genuine text/leading zeros; reject missing and scan-note IDs."""
    text = str(value if value is not None else "").strip()
    if text.casefold() in _MISSING or len(text) > 128 or '\ufffd' in text or any(0xD800 <= ord(c) <= 0xDFFF for c in text):
        return ""
    if re.search(r"(?:^parameters?\s*:|[=\n\r])", text, re.I):
        return ""
    try:
        number = Decimal(text)
        if not number.is_finite() or number <= 0:
            return ""
    except InvalidOperation:
        pass
    return text


def context(instrument: str, ipts: str, run: int | None = None) -> tuple[str, str]:
    instrument, ipts = str(instrument).upper(), str(ipts).upper()
    if instrument not in {"PG3", "NOM", "SNAP"} or not re.fullmatch(r"IPTS-[1-9]\d*", ipts):
        raise ValueError("Choose a supported neutron beamline and a positive IPTS identifier")
    if run is not None and (isinstance(run, bool) or int(run) != run or run < 1):
        raise ValueError("Run must be a positive integer")
    return instrument, ipts


def nexus_candidates(instrument: str, ipts: str, run: int, *, facility_root=Path("/SNS")) -> tuple[Path, ...]:
    instrument, ipts = context(instrument, ipts, run)
    root = Path(facility_root) / instrument / ipts
    name = f"{instrument}_{run}"
    return (root / "nexus" / (name + ".nxs.h5"),
            root / "data" / (name + "_event.nxs"),
            root / "0" / str(run) / "NeXus" / (name + "_event.nxs"))


def resolve_nexus_path(instrument: str, ipts: str, run: int, *, facility_root=Path("/SNS")) -> Path:
    candidates = nexus_candidates(instrument, ipts, run, facility_root=facility_root)
    found = {}
    for path in candidates:
        try:
            stat = path.stat()
            if path.is_file():
                found.setdefault((stat.st_dev, stat.st_ino), path)
        except FileNotFoundError:
            continue
    if len(found) > 1:
        raise ValueError("Several distinct raw files match this run; review the source before selecting metadata")
    return next(iter(found.values()), candidates[0])


def resolve_identity(evidence: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in evidence]
    valid = [(sample_identifier(r.get("value")), r) for r in rows]
    ids = sorted({value for value, _ in valid if value})
    status = "resolved" if len(ids) == 1 else "conflict" if ids else "unidentified"
    return {"sample_id": ids[0] if status == "resolved" else "", "identity_status": status,
            "identity_sources": rows,
            "identity_warning": ("Conflicting recorded sample identifiers; review sources before selecting this sample"
                                 if status == "conflict" else "")}


@dataclass
class ScanTable:
    source: str
    sha256: str
    schema: str
    rows: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    row_count: int = 0
    rejected_count: int = 0
    foreign_count: int = 0
    issues: list[dict[str, Any]] = field(default_factory=list)

    def issue(self, line: int, code: str, run: int | None = None) -> None:
        if len(self.issues) < 100:
            self.issues.append({"line": line, "code": code, "run": run})

    def fields(self, run: int) -> dict[str, Any]:
        rows = self.rows.get(run, [])
        result = {"fields": {}, "conflicts": [], "sources": []}
        for name in set().union(*(row["fields"] for row in rows)) if rows else ():
            observations = [{"value": row["fields"][name], "path": self.source,
                             "line": row["line"], "sha256": self.sha256,
                             "kind": self.schema, "field": name}
                            for row in rows if row["fields"].get(name, "") != ""]
            if name == "sample_id":
                observations = [o for o in observations if sample_identifier(o["value"])]
            values = {o["value"] for o in observations}
            result["sources"].extend(observations)
            if len(values) == 1:
                result["fields"][name] = next(iter(values))
            elif len(values) > 1:
                result["conflicts"].append(name)
        return result


def parse_scan_table(raw: bytes, *, source: str, instrument: str, ipts: str) -> ScanTable:
    instrument, ipts = context(instrument, ipts)
    if len(raw) > MAX_TABLE_BYTES:
        raise ValueError("Scan table exceeds the bounded metadata limit")
    table = ScanTable(source, sha256(raw).hexdigest(), "unsupported")
    # Preserve ASCII fields when a historical name contains non-UTF8 bytes.
    # Such text is suppressed field-by-field, never guessed as Latin-1/CP1252.
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig", errors="surrogateescape"), newline=""), strict=True)
    try:
        header = [s.strip().lstrip("#").upper() for s in next(reader, [])]
        duplicates = {key for key in header if header.count(key)>1}
        if duplicates & {'RUN','SCAN','IPTS#','IPTS'}:
            raise ValueError("Duplicate scan-table identity key columns")
        if duplicates & set(_FIELDS):
            table.issue(1, 'duplicate_field_columns')
        if instrument == "PG3" and {"RUN", "IPTS#", "ITEMS-ID", "SAMPLE"}.issubset(header):
            table.schema = "powgen_runsummary"
            run_key, proposal_key = "RUN", "IPTS#"
        elif instrument == "NOM" and {"SCAN", "IPTS", "TITLE"}.issubset(header):
            table.schema = "nomad_los"
            run_key, proposal_key = "SCAN", "IPTS"
        elif instrument == "SNAP" and {"RUN", "IPTS", "TITLE", "SAMPLE"}.issubset(header):
            # SNAP run summaries contain useful names/titles but often no
            # acquisition sample identifier or qualified pressure units.
            table.schema = "snap_runsummary"
            run_key, proposal_key = "RUN", "IPTS"
        else:
            return table
        for row in reader:
            if not any(v.strip() for v in row):
                continue
            table.row_count += 1
            if table.row_count > MAX_TABLE_ROWS:
                raise ValueError("Scan table exceeds the row limit")
            if len(row) != len(header):
                table.rejected_count += 1
                table.issue(reader.line_num, "row_width")
                continue
            data = dict(zip(header, (v.strip() for v in row)))
            try:
                run = int(data[run_key])
                if run < 1:
                    raise ValueError
            except ValueError:
                table.rejected_count += 1
                table.issue(reader.line_num, "invalid_run")
                continue
            recorded = data[proposal_key].upper()
            if recorded != ipts and "IPTS-" + recorded != ipts:
                table.foreign_count += 1
                table.issue(reader.line_num, "foreign_proposal", run)
                continue
            fields = {target: data[key] for key, target in _FIELDS.items() if key in data and key not in duplicates}
            for key in list(fields):
                if any(0xDC80 <= ord(c) <= 0xDCFF for c in fields[key]):
                    fields.pop(key)
                    table.issue(reader.line_num, 'unqualified_text_encoding', run)
            if fields.get("sample_id") and not sample_identifier(fields["sample_id"]):
                table.issue(reader.line_num, "invalid_sample_id", run)
                fields.pop("sample_id")
            table.rows.setdefault(run, []).append({"line": reader.line_num, "fields": fields})
    except csv.Error as exc:
        raise ValueError(f"Malformed CSV scan table: {exc}") from exc
    return table


def read_scan_table(path: Path, instrument: str, ipts: str) -> ScanTable:
    before = path.stat()
    if before.st_size > MAX_TABLE_BYTES:
        raise ValueError("Scan table exceeds the bounded metadata limit")
    with path.open("rb") as stream:
        raw = stream.read(MAX_TABLE_BYTES + 1)
    result = parse_scan_table(raw, source=str(path), instrument=instrument, ipts=ipts)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Scan table changed while reading; retry after its writer finishes")
    return result


def read_runinfo(path: Path, instrument: str, run: int) -> dict[str, Any]:
    before = path.stat()
    if before.st_size > MAX_XML_BYTES:
        raise ValueError("Run-info XML exceeds the metadata limit")
    with path.open("rb") as stream:
        raw = stream.read(MAX_XML_BYTES + 1)
    if len(raw) > MAX_XML_BYTES or b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("Unsupported or oversized run-info XML")
    root = ET.fromstring(raw)
    if (root.tag.rsplit("}", 1)[-1] != "RunInfo" or root.get("instrument") != instrument
            or root.get("runnumber") != str(run)):
        raise ValueError("Run-info XML instrument/run disagrees with the selected run")
    samples = [e for e in root.iter() if e.tag.rsplit("}", 1)[-1] == "SampleInfo"]
    if len(samples) != 1:
        raise ValueError("Run-info XML sample identity is ambiguous")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Run-info XML changed while reading")
    return {"sample_id": samples[0].get("Identifier", ""), "sample_name": samples[0].get("Name", ""),
            "path": str(path), "sha256": sha256(raw).hexdigest(), "kind": "runinfo_xml"}


def reduction_xml(path: Path, instrument: str, run: int) -> tuple[dict[str, str], tuple[int, int], str]:
    """Named per-run receipt, not a proposal's mutable default configuration."""
    before = path.stat()
    with path.open('rb') as stream:
        raw = stream.read(MAX_XML_BYTES + 1)
    if len(raw) > MAX_XML_BYTES or b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
        raise ValueError('Unsupported or oversized reduction XML')
    fields = {}
    try:
        document = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError('Malformed reduction XML') from exc
    for element in document.iter():
        if len(element):
            continue
        key = element.tag.rsplit('}', 1)[-1].lower()
        if key in {'instrument_name','runnumber','calibrationfile','characterizationrunsfile','groupingfile','saveas'}:
            if key in fields:
                raise ValueError('Ambiguous duplicate fields in reduction XML')
            fields[key] = (element.text or '').strip()
    if fields.get('instrument_name') != instrument or fields.get('runnumber') != str(run):
        raise ValueError('Reduction XML instrument/run disagrees with selected data')
    if 'gsas' not in fields.get('saveas', '').lower():
        raise ValueError('Reduction XML does not identify a GSAS-producing reduction')
    after = path.stat()
    signature = (before.st_size, before.st_mtime_ns)
    if signature != (after.st_size, after.st_mtime_ns):
        raise ValueError('Reduction XML changed during inspection')
    names = {'calibrationfile':'CalibrationFile','characterizationrunsfile':'CharacterizationRunsFile','groupingfile':'GroupingFile','saveas':'SaveAs'}
    return {target:fields[source] for source,target in names.items() if fields.get(source)}, signature, sha256(raw).hexdigest()


def title_role(title: str) -> str:
    text = re.sub(r"[_-]+", " ", title.casefold())
    if re.search(r"\b(empty|background|bkgd|no sample|without sample)\b", text):
        return "background"
    if re.search(r"\b(ramp(?:ing)?|cool(?:ing)?(?: down| to)?|heat(?:ing)? up)\b", text):
        return "ramp"
    return "unclassified"


def qualify_temperature(metric: dict[str, Any] | None, title: str = "") -> tuple[dict[str, Any] | None, list[str]]:
    """A suspect axis must not prevent a scientifically independent fit."""
    if not metric:
        return None, ["Selected sample temperature is unavailable"]
    import math

    value = metric.get("value")
    unit = str(metric.get("unit", ""))
    notices = []
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None, ["Selected sample temperature is nonfinite or missing"]
    low, high = metric.get('minimum',value), metric.get('maximum',value)
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (low,high)) or not low <= value <= high:
        return None, ["Temperature summary bounds are nonfinite or inconsistent"]
    if unit == "K" and (value <= 0 or metric.get("minimum", value) <= 0):
        return None, ["Selected sample temperature is nonpositive Kelvin; no inactive sensor was substituted"]
    if unit not in {"K"}:
        return None, ["Temperature units are unqualified; verify the selected sensor"]
    if "converted_from" not in metric and re.search(r"temperature\s+[-+\d.eE]+\s*(?:°?C|Celsius)\b", title, re.I):
        return None, ["Recorded temperature units conflict with the run title; verify historical sensor routing"]
    spread = metric.get("maximum", value) - metric.get("minimum", value)
    if spread > 2:
        notices.append(f"Sample temperature changes by {spread:.3g} K within this run; review before using an equilibrium temperature axis")
    return {**metric, "quality": "changing" if notices else "recorded"}, notices


def metadata_actions(identity: dict[str, Any], *, notices=()) -> list[dict[str, str]]:
    actions = []
    status = identity.get("identity_status", "resolved" if identity.get("sample_id") else "unidentified")
    if status != "resolved":
        actions.append({"code": "sample_identity", "category": "review" if status == "conflict" else "choice",
                        "message": identity.get("identity_warning") or "Choose All samples or explicitly exclude unidentified runs; no sample ID was guessed"})
    actions.extend({"code": "condition_metadata", "category": "review", "message": str(n)} for n in notices)
    return actions


class ExperimentMetadataIndex:
    """Revision-aware bulk table cache; raw/XML identities remain independent."""

    def __init__(self, instrument: str, ipts: str, *, root: Path | None = None):
        self.instrument, self.ipts = context(instrument, ipts)
        self.root = Path(root) if root is not None else Path("/SNS") / self.instrument / self.ipts
        self.tables: list[ScanTable] = []
        self.notices: list[str] = []
        self.revision: tuple = ()
        self.generation = 0
        self._retry_read = False
        self._last_refresh = -float("inf")
        import threading

        self._lock = threading.RLock()

    def refresh(self, *, force=False) -> None:
        import time

        with self._lock:
            if not force and time.monotonic() - self._last_refresh < 2:
                return
            paths = []
            notices = []
            if self.instrument in {"PG3", "SNAP"}:
                paths = [self.root / "shared/autoreduce" / f"{self.instrument}_{self.ipts}_runsummary.csv"]
                if self.instrument == "PG3":
                    paths.append(self.root / "shared/autoreduce" / f"{self.ipts[5:]}_run-summary.csv")
            else:
                shared = self.root / "shared"
                paths = [shared / "los.csv"]
                try:
                    import os

                    with os.scandir(shared) as entries:
                        families = sorted(e.name for e in entries
                                          if re.match(r"autoNOM|autoMTS|manuNOM", e.name, re.I)
                                          and e.is_dir(follow_symlinks=False))
                    if len(families) > 25:
                        notices.append("More than 25 reduction families; narrow the source for complete scan-list coverage")
                    paths.extend(shared / name / "los.csv" for name in families[:25])
                except FileNotFoundError:
                    pass
                except OSError:
                    notices.append("Scan-list directory is unreadable; raw identity checks remain available")
            signatures = []
            for path in paths:
                try:
                    stat = path.stat()
                    signatures.append((str(path), stat.st_size, stat.st_mtime_ns))
                except FileNotFoundError:
                    continue
                except OSError:
                    notices.append(f"Scan table is unreadable: {path.name}")
            revision = tuple(signatures)
            if revision != self.revision or self._retry_read or force:
                tables, hashes = [], set()
                retry_read = False
                for rawpath, _, _ in signatures:
                    try:
                        table = read_scan_table(Path(rawpath), self.instrument, self.ipts)
                        if table.sha256 in hashes:
                            continue
                        hashes.add(table.sha256)
                        tables.append(table)
                        if table.rejected_count:
                            notices.append(f"{Path(rawpath).name}: {table.rejected_count} unqualified scan-table rows; raw/XML fallback is required")
                        if any(i['code'] in {'unqualified_text_encoding','duplicate_field_columns'} for i in table.issues):
                            notices.append(f"{Path(rawpath).name}: some text/duplicate columns are unqualified; only independent valid fields were used")
                    except (OSError, ValueError) as exc:
                        notices.append(f"{Path(rawpath).name}: {exc}")
                        retry_read |= isinstance(exc, OSError) or "changed while reading" in str(exc)
                old_content = ([(t.source, t.sha256) for t in self.tables], self.notices)
                if old_content != ([(t.source, t.sha256) for t in tables], notices):
                    self.generation += 1
                self.tables, self.revision, self.notices = tables, revision, notices
                self._retry_read = retry_read
            elif notices:
                self.notices = list(dict.fromkeys([*self.notices, *notices]))
            self._last_refresh = time.monotonic()

    def xml_path(self, run: int) -> Path:
        return self.root / "0" / str(run) / "preNeXus" / f"{self.instrument}_{run}_runinfo.xml"

    def enrich(self, run: int, raw_identity: dict[str, Any], *, include_xml: bool = True) -> dict[str, Any]:
        self.refresh()
        evidence = list(raw_identity.get("identity_sources") or [])
        if not evidence and raw_identity.get("sample_id"):
            evidence.append({"value": raw_identity["sample_id"], "kind": "nexus", "field": "sample_id"})
        name = raw_identity.get("sample_name", "")
        notices = list(raw_identity.get("metadata_notices") or [])
        path = self.xml_path(run)
        try:
            if not include_xml:
                raise FileNotFoundError(path)
            xml = read_runinfo(path, self.instrument, run)
            evidence.append({"value": xml["sample_id"], "kind": xml["kind"], "path": xml["path"],
                             "sha256": xml["sha256"], "field": "sample_id"})
            name = name or xml["sample_name"]
        except FileNotFoundError:
            pass
        except (OSError, ValueError, ET.ParseError) as exc:
            notices.append(f"Run-info metadata needs review: {exc}")
        table_sources, table_fields, conflicts = [], {}, []
        with self._lock:
            for table in self.tables:
                values = table.fields(run)
                table_sources.extend(values["sources"])
                conflicts.extend(values["conflicts"])
            for key in {o["field"] for o in table_sources}:
                values = {o["value"] for o in table_sources if o["field"] == key}
                if len(values) == 1:
                    table_fields[key] = next(iter(values))
                elif len(values) > 1:
                    conflicts.append(key)
        evidence.extend(o for o in table_sources if o["field"] == "sample_id")
        result = resolve_identity(evidence)
        # A raw-source conflict must never be erased by a fallback.
        if raw_identity.get("identity_status") == "conflict":
            result.update(sample_id="", identity_status="conflict", identity_warning=raw_identity.get("identity_warning", "Raw identity conflicts"))
        if conflicts:
            notices.append("Scan-table fields disagree across recorded rows: " + ", ".join(sorted(set(conflicts))))
        result.update(sample_name=name or table_fields.get("sample_name", ""), table_fields=table_fields,
                      table_sources=table_sources, metadata_notices=notices)
        result["metadata_actions"] = metadata_actions(result, notices=notices)
        return result
