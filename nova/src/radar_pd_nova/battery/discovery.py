"""Bounded read-only discovery inside one selected proposal.

Names locate candidate sources; they never establish chemistry, cell/channel
binding or calibration validity. No recursive walk, event-array read, script
execution, archive extraction or writes to the facility are performed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import zipfile
import time

from ..facility_metadata import context, parse_nexus_timestamp, resolve_nexus_path, resolve_identity

SCHEMA = "radar-battery-discovery/v1"
_PRODUCT = re.compile(r"(?:NOM_?(\d+)(?:tof)?|PG3_(\d+))(?:[-_]?(?:bank)?(\d+))?\.(gsa|gss|xye|fxye|dat)$", re.I)
_RAW = re.compile(r"(?:NOM|PG3)_(\d+)(?:_event)?\.nxs(?:\.h5)?$", re.I)
_LOG = re.compile(r"voltage|current|samplevolt|electro|cycler|biologic|arbin|neware|sampletemp|temperature", re.I)
_CYCLER = {".mpr": "biologic_binary", ".mpt": "biologic_text", ".nda": "neware_binary", ".ndax": "neware_archive", ".res": "cycler_database_candidate"}
_TABLE_NAMES = {"los.csv", "scan_table.txt", "scan_table.csv", "scantable.txt", "scantable.csv"}


def read_idl_setup(path, relative, ipts, limits):
    """Read explicit IDL reduction references as evidence, never as commands.

    Copied setup comments can name another proposal. They must not override
    the selected archive root or the acquisition's NeXus identity.
    """
    before=_signature(path)
    with path.open('rb') as stream:raw=stream.read(limits.header_bytes+1)
    if len(raw)>limits.header_bytes:raise ValueError('IDL setup exceeds its bounded metadata read')
    if _signature(path)!=before:raise ValueError('IDL setup changed during metadata capture')
    fields={};comments=[]
    allowed={'Dia','DiaBg','Vana','VanaBg','MTc','recali','renorm','autotemp','scan1','scanl','Hz','Scans','Templates'}
    for line in raw.decode('utf-8-sig').splitlines():
        comment=re.fullmatch(r'\s*#\s*IPTS[-\s]+(\d+)\s*',line,re.I)
        if comment:comments.append('IPTS-'+comment[1]);continue
        parts=line.strip().split(None,1)
        if len(parts)!=2 or parts[0] not in allowed:continue
        if parts[0] in fields:raise ValueError('Duplicate IDL setup field: '+parts[0])
        fields[parts[0]]=parts[1]
    roles={'Dia':'diamond','DiaBg':'diamond_background','Vana':'vanadium','VanaBg':'vanadium_background','MTc':'empty_container'}
    references=[]
    for key,role in roles.items():
        if key in fields:
            if not fields[key].isdigit() or int(fields[key])<1:raise ValueError('Ambiguous IDL reference run: '+key)
            references.append(dict(role=role,run=int(fields[key]),qualification='unverified_reference'))
    mismatch=any(value!=ipts for value in comments)
    summary='; '.join(f"{r['role'].replace('_',' ')} {r['run']}" for r in references)
    if mismatch:summary+='; setup comment names '+', '.join(comments)+' (shared or copied setup; acquisition identity unchanged)'
    return dict(path=relative,**before,sha256=sha256(raw).hexdigest(),fields=fields,
                reference_runs=references,proposal_comments=comments,proposal_comment_differs=mismatch,
                qualification='setup_evidence_only',summary=summary or 'Reduction selection metadata; not acquisition identity')


@dataclass(frozen=True)
class DiscoveryLimits:
    directories: int = 96
    entries_per_directory: int = 50_000
    returned_files: int = 512
    zip_members: int = 4096
    zip_bytes: int = 64 * 1024 * 1024
    header_bytes: int = 16_384
    metadata_values: int = 32
    log_names: int = 2048
    elapsed_seconds: float = 45.0


def _issue(code, message, path=""):
    return {"code": code, "message": message, "path": str(path)}


def _signature(path):
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _inside(path, root):
    """The instrument root may be a facility mount symlink; children cannot escape."""
    path = Path(path)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Selected source resolves outside this IPTS")
    return path


def _entries(path, limits, issues):
    rows = []
    try:
        with os.scandir(path) as stream:
            for index, entry in enumerate(stream):
                if index >= limits.entries_per_directory:
                    issues.append(_issue("directory_limit", "Directory inventory is incomplete; choose a narrower folder", path))
                    break
                # No child symlink traversal. Its target may be outside the proposal.
                if entry.is_symlink():
                    continue
                try:
                    # A stat for every unrelated XYE/PDF product makes large
                    # NFS folders prohibitively slow. Stat only selected sources.
                    rows.append((entry.name, entry.is_dir(follow_symlinks=False), None))
                except OSError as exc:
                    issues.append(_issue("entry_unavailable", str(exc), entry.path))
    except OSError as exc:
        issues.append(_issue("directory_unavailable", str(exc), path))
    return sorted(rows)


def _header_kind(raw):
    """Recognize electrical column evidence, not a filename or voltage magnitude."""
    text = raw.decode("utf-8-sig", errors="replace")
    if text.startswith("EC-Lab ASCII FILE"):
        return "biologic_text"
    for line in text.splitlines()[:80]:
        fields = {s.strip().strip('"').lower() for s in re.split(r"[,;\t]", line)}
        time = bool(fields & {"time/s", "time_s", "elapsed_seconds", "timestamp", "date_time", "test_time(s)", "test time(s)"})
        voltage = bool(fields & {"ewe/v", "voltage_v", "voltage(v)", "voltage (v)"})
        current = bool(fields & {"i/ma", "<i>/ma", "current_a", "current_ma", "current(a)", "current(ma)"})
        if time and voltage and current:
            return "electrochemistry_table"
        if time and voltage:
            return "voltage_table"
    return ""


def _candidate(path, relative, limits):
    suffix = path.suffix.lower()
    kind = _CYCLER.get(suffix, "")
    if not kind and suffix in {".csv", ".txt", ".tsv"}:
        if path.stat().st_size > 128 * 1024 * 1024:
            return None
        with path.open("rb") as stream:
            kind = _header_kind(stream.read(limits.header_bytes))
    if not kind:
        return None
    return {"path": relative, "format": kind, "binding": "unbound", **_signature(path)}


def _zip_candidates(path, relative, limits, issues):
    if path.stat().st_size > limits.zip_bytes:
        issues.append(_issue("archive_limit", "ZIP exceeds discovery budget; select a cycler file explicitly", relative))
        return []
    result = []
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > limits.zip_members:
                issues.append(_issue("archive_limit", "ZIP has too many members for automatic discovery", relative))
                return []
            for member in members:
                name = Path(member.filename)
                if name.is_absolute() or ".." in name.parts or "\\" in member.filename or member.is_dir():
                    continue
                kind = _CYCLER.get(name.suffix.lower())
                if kind:
                    result.append({"path": relative, "member": member.filename, "format": kind,
                                   "binding": "unbound", "member_size_bytes": member.file_size,
                                   "member_crc32": f"{member.CRC:08x}", **_signature(path)})
    except (OSError, zipfile.BadZipFile) as exc:
        issues.append(_issue("archive_unreadable", str(exc), relative))
    return result


def discover_proposal(instrument, ipts, *, facility_root=Path("/SNS"), limits=None, progress=None):
    """Inventory known layouts and shallow auxiliary folders for one proposal.

    Large product families are summarized by count/range. This operation never
    opens every raw file. Exact selected-run metadata is read separately.
    """
    instrument, ipts = context(instrument, ipts)
    if instrument not in {"NOM", "PG3"}:
        raise ValueError("Battery archive discovery currently supports NOMAD and POWGEN")
    limits = limits or DiscoveryLimits()
    started = time.monotonic()
    root = Path(facility_root) / instrument / ipts
    issues, products, tables, cyclers, profiles, auxiliary, calibrations, setups = [], [], [], [], [], [], [], []
    result = {"schema": SCHEMA, "instrument": instrument, "ipts": ipts, "root": str(root),
              "observed_at": datetime.now(timezone.utc).isoformat(), "status": "unavailable",
              "product_families": products, "metadata_tables": tables, "cycler_candidates": cyclers,
              "profile_candidates": profiles, "calibration_candidates": calibrations,
              "reduction_setups": setups,
              "auxiliary_directories": auxiliary, "issues": issues,
              "limits": asdict(limits)}
    try:
        available = root.is_dir()
    except OSError as exc:
        issues.append(_issue("proposal_unavailable", str(exc), root))
        return result
    if not available:
        issues.append(_issue("proposal_unavailable", "Proposal directory is absent or inaccessible", root))
        return result
    root_entries = _entries(root, limits, issues)
    result["root_entries"] = [name for name, _, _ in root_entries]
    # Visit only named layout nodes. One level of auxiliary folders catches
    # common Echem/LNO/channel collections without crawling arbitrary results.
    queue = [(root / "shared", 0)]
    seen, returned = set(), 0
    while queue and len(seen) < limits.directories:
        if time.monotonic() - started >= limits.elapsed_seconds:
            issues.append(_issue("time_budget", "Discovery time budget reached; select a source folder to continue", root))
            break
        folder, depth = queue.pop(0)
        if folder in seen:
            continue
        _inside(folder, root)
        seen.add(folder)
        if progress:
            progress({"folder": folder.relative_to(root).as_posix(), "directories_inspected": len(seen)})
        entries = _entries(folder, limits, issues)
        runs, formats, count = set(), set(), 0
        for name, directory, size in entries:
            path = folder / name
            relative = path.relative_to(root).as_posix()
            if directory:
                if depth == 0:
                    auxiliary.append(relative)
                    queue.append((path, 1))
                elif depth < 3 and (name in {"GSAS", "Input", "multi_banks", "single_banks", "TOPAS", "logs", "electrochemistry", "Echem"}
                        or instrument == 'PG3' and name in {'HR','HI'}):
                    # Observed POWGEN calibration/experiment collections keep
                    # high-resolution and high-intensity products separately.
                    # These names locate sources; they do not qualify a profile.
                    queue.append((path, depth + 1))
                continue
            match = _PRODUCT.fullmatch(name)
            if match and name.upper().startswith(instrument):
                runs.add(int(match[1] or match[2])); formats.add(match[4].lower()); count += 1
                continue
            if returned >= limits.returned_files:
                continue
            try:
                if name.lower() in _TABLE_NAMES or re.fullmatch(r"(?:PG3_IPTS-\d+_runsummary|\d+_run-summary)\.csv", name, re.I):
                    tables.append({"path": relative, **_signature(path)}); returned += 1
                elif path.suffix.lower() in {".instprm", ".prm", ".inst", ".ins"}:
                    profiles.append({"path": relative, "qualification": "unreviewed", **_signature(path)}); returned += 1
                elif path.suffix.lower() in {".h5", ".hdf5"} and re.search(r"calib|detcal|mask|group", name, re.I):
                    calibrations.append({"path": relative, "kind": "detector_calibration_candidate",
                                         "qualification": "not_a_gsas_peak_profile", **_signature(path)}); returned += 1
                elif path.suffix.lower() in {'.calfile','.extcalfile'}:
                    calibrations.append({'path':relative,'kind':'idl_detector_calibration_candidate',
                        'qualification':'not_a_gsas_peak_profile',**_signature(path)});returned+=1
                elif name.lower() in {'exp.ini','los.ini'}:
                    try:setups.append(read_idl_setup(path,relative,ipts,limits));returned+=1
                    except (OSError,ValueError,UnicodeError) as exc:issues.append(_issue('reduction_setup_unavailable',str(exc),relative))
                elif path.suffix.lower() == ".zip":
                    found = _zip_candidates(path, relative, limits, issues)
                    cyclers.extend(found[:limits.returned_files - returned]); returned += min(len(found), limits.returned_files - returned)
                else:
                    try:
                        candidate = _candidate(path, relative, limits)
                        if candidate:
                            cyclers.append(candidate); returned += 1
                    except OSError as exc:
                        issues.append(_issue("candidate_unavailable", str(exc), relative))
            except OSError as exc:
                issues.append(_issue('candidate_unavailable',str(exc),relative))
        if runs:
            products.append({"folder": folder.relative_to(root).as_posix(), "run_count": len(runs),
                             "file_count": count, "first_run": min(runs), "last_run": max(runs),
                             "formats": sorted(formats), "sample_runs": sorted(runs)[::max(1, len(runs)//4)][:4] + [max(runs)]})
    if queue and not any(i["code"] == "time_budget" for i in issues):
        issues.append(_issue("directory_budget", "More auxiliary folders exist; select a folder for further inspection", root))
    if returned >= limits.returned_files:
        issues.append(_issue("file_budget", "Candidate list reached its limit; inspect a narrower folder", root))
    result["directories_inspected"] = len(seen)
    result["elapsed_seconds"] = time.monotonic() - started
    result["status"] = "partial" if issues else "available"
    return result


def family_runs(inventory, folder):
    """Resolve only a discovered family; no browser-supplied arbitrary path."""
    if folder not in {row["folder"] for row in inventory.get("product_families", [])}:
        raise ValueError("Select a diffraction family from the current proposal inventory")
    root = Path(inventory["root"])
    path = _inside(root / folder, root)
    issues = []
    entries = _entries(path, DiscoveryLimits(), issues)
    if issues:
        raise ValueError(issues[0]["message"])
    runs = set()
    for name, directory, _ in entries:
        match = _PRODUCT.fullmatch(name)
        if not directory and match and name.upper().startswith(inventory["instrument"]):
            runs.add(int(match[1] or match[2]))
    return sorted(runs)


def _small_value(node, limit):
    """Read only a bounded dataset; external/soft HDF5 links are never followed."""
    import h5py
    import numpy as np
    if not isinstance(node, h5py.Dataset) or node.size > limit:
        return None
    if node.dtype.kind == "O" and node.size > 1:
        return None  # Variable-length arrays can defeat a fixed byte budget.
    value = node[()]
    def scalar(item):
        if isinstance(item, bytes):
            return item[:4096].decode("utf-8", errors="replace").strip()
        if hasattr(item, "item"):
            item = item.item()
        if isinstance(item, str):
            return item[:4096]
        if isinstance(item, float) and not __import__("math").isfinite(item):
            return None
        return item
    values = [scalar(item) for item in np.asarray(value).reshape(-1)]
    return values[0] if len(values) == 1 else values


def _hard_node(handle, path):
    import h5py
    node = handle
    for part in path.strip("/").split("/"):
        if not isinstance(node, h5py.Group) or not isinstance(node.get(part, getlink=True), h5py.HardLink):
            return None
        node = node[part]
    return node


def inspect_run(instrument, ipts, run, *, facility_root=Path("/SNS"), limits=None):
    """Read exact run metadata, preserving source fields and uncertainty in meaning."""
    instrument, ipts = context(instrument, ipts, run)
    limits = limits or DiscoveryLimits()
    path = resolve_nexus_path(instrument, ipts, run, facility_root=facility_root)
    root = Path(facility_root) / instrument / ipts
    _inside(path, root)
    result = {"schema": "radar-battery-acquisition-metadata/v1", "instrument": instrument,
              "ipts": ipts, "run": run, "path": str(path), "status": "unavailable",
              "fields": {}, "condition_channels": [], "issues": []}
    if not path.is_file():
        result["issues"].append(_issue("raw_metadata_unavailable", "Raw metadata file is unavailable", path))
        return result
    try:
        import h5py
    except ImportError:
        result["issues"].append(_issue("metadata_reader_unavailable", "Install the declared h5py metadata reader; no event data is needed", path))
        return result
    before = _signature(path)
    fields = {"run_number": "/entry/run_number", "ipts": "/entry/experiment_identifier",
              "run_title": "/entry/title", "proposal_title": "/entry/experiment_title",
              "start_time": "/entry/start_time", "end_time": "/entry/end_time",
              "duration": "/entry/duration", "sample_name": "/entry/sample/name",
              "sample_id": "/entry/sample/id", "sample_identifier": "/entry/sample/identifier",
              "registered_formula": "/entry/sample/chemical_formula"}
    try:
        with h5py.File(path, "r") as handle:
            for name, field in fields.items():
                node = _hard_node(handle, field)
                value = _small_value(node, limits.metadata_values)
                if value is not None:
                    result["fields"][name] = {"value": value, "field": field}
            logs = _hard_node(handle, "/entry/DASlogs")
            if isinstance(logs, h5py.Group):
                for index, name in enumerate(logs):
                    if index >= limits.log_names:
                        result["issues"].append(_issue("log_inventory_limit", "Condition channel inventory is incomplete", path))
                        break
                    if not _LOG.search(name):
                        continue
                    base = f"/entry/DASlogs/{name}"
                    channel = {"name": name, "field": base, "binding": "unreviewed"}
                    for leaf in ("value", "time", "average_value", "minimum_value", "maximum_value"):
                        node = _hard_node(handle, base + "/" + leaf)
                        if not isinstance(node, h5py.Dataset):
                            continue
                        item = {"shape": list(node.shape)}
                        for key in ("units", "start", "offset"):
                            attr = node.attrs.get(key)
                            if isinstance(attr, bytes): attr = attr.decode("utf-8", errors="replace")
                            if isinstance(attr, (str, int, float)): item[key] = attr
                        value = _small_value(node, limits.metadata_values)
                        if value is not None: item["values"] = value
                        channel[leaf] = item
                    result["condition_channels"].append(channel)
    except (OSError, ValueError) as exc:
        result["issues"].append(_issue("metadata_read_failed", str(exc), path))
        return result
    if _signature(path) != before:
        result["issues"].append(_issue("source_changed", "Raw file changed during metadata inspection; retry after completion", path))
        result["fields"] = {}; result["condition_channels"] = []
        return result
    result["source_signature"] = before
    identity_evidence = []
    for key in ("sample_id", "sample_identifier"):
        record = result["fields"].get(key)
        if record:
            values = record["value"] if isinstance(record["value"], list) else [record["value"]]
            identity_evidence.extend({"value": value, "path": str(path), "field": record["field"], "kind": "nexus"} for value in values)
    result["identity"] = resolve_identity(identity_evidence)
    if result["identity"]["identity_status"] == "conflict":
        result["issues"].append(_issue("sample_conflict", "Recorded sample identifiers disagree", path))
    get = lambda key: result["fields"].get(key, {}).get("value")
    if get("run_number") is not None and str(get("run_number")) != str(run):
        result["issues"].append(_issue("run_conflict", "Filename and stored run number disagree", path))
    if get("ipts") and str(get("ipts")).upper() != ipts:
        result["issues"].append(_issue("proposal_conflict", "Stored proposal differs from the selected IPTS", path))
    try:
        start, end = (parse_nexus_timestamp(get(k)) for k in ("start_time", "end_time"))
        if start.tzinfo is None or end.tzinfo is None or end <= start:
            raise ValueError("Exposure needs increasing timestamps with explicit UTC offsets")
        result["exposure"] = {"start_utc": start.astimezone(timezone.utc).isoformat(),
                              "end_utc": end.astimezone(timezone.utc).isoformat(),
                              "duration_seconds": (end-start).total_seconds()}
    except (TypeError, ValueError) as exc:
        result["issues"].append(_issue("exposure_unresolved", str(exc), path))
    result["metadata_sha256"] = sha256(json.dumps({k:result[k] for k in ("fields", "condition_channels")}, sort_keys=True, allow_nan=False).encode()).hexdigest()
    result["status"] = "conflict" if any(i["code"].endswith("conflict") for i in result["issues"]) else "partial" if result["issues"] else "available"
    return result
