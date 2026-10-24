"""Cached sample discovery for the completed reductions of one experiment."""

from concurrent.futures import ThreadPoolExecutor
import threading
import subprocess
from pathlib import Path
import re
from typing import Any, Iterable

from .powgen_metadata import powgen_nexus_path, read_powgen_sample_identity
from .powgen_watch import WatchedRun
from .facility_metadata import ExperimentMetadataIndex, title_role, metadata_actions


def run_role(run: WatchedRun) -> dict[str, str]:
    """Conservative title hints, shown for review rather than inferred chemistry."""
    title = ""
    try:
        with Path(run.source_path).open(encoding="utf-8", errors="replace") as handle:
            for _ in range(24):
                line = handle.readline(4096)
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
    except OSError:
        pass
    role = title_role(title)
    return {"run_title": title, "run_role": role}


def include_run(run: WatchedRun, *, excluded_run_numbers=(), include_background_runs=False,
                include_ramp_runs=False) -> bool:
    if run.run_number in excluded_run_numbers:
        return False
    role = run_role(run)["run_role"]
    return ((role != "background" or include_background_runs)
            and (role != "ramp" or include_ramp_runs))


class PowgenSampleIndex:
    def __init__(self, ipts: str, *, nexus_path=None, identity_reader=None, instrument='PG3', metadata_index=None) -> None:
        self.ipts = ipts
        self.nexus_path = nexus_path or powgen_nexus_path
        self.identity_reader = identity_reader or read_powgen_sample_identity
        self.metadata_index = metadata_index or ExperimentMetadataIndex(instrument, ipts)
        self._cache: dict[int, tuple[tuple[Any, ...], dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def identity(self, run: WatchedRun) -> dict[str, Any]:
        """Retry missing/unreadable NeXus files; refresh identities when they change."""

        self.metadata_index.refresh()
        try:
            path = self.nexus_path(self.ipts, run.run_number)
            try:
                stat = path.stat()
                raw_signature = (str(path), stat.st_size, stat.st_mtime_ns)
            except FileNotFoundError:
                raw_signature = None
            xml = self.metadata_index.xml_path(run.run_number)
            try:
                xml_stat = xml.stat()
                xml_signature = (xml_stat.st_size, xml_stat.st_mtime_ns)
            except FileNotFoundError:
                xml_signature = None
            signature = (raw_signature, xml_signature, self.metadata_index.revision, self.metadata_index.generation)
            with self._lock:
                cached = self._cache.get(run.run_number)
            if cached and cached[0] == signature:
                return dict(cached[1])
            try:
                raw_metadata = self.identity_reader(self.ipts, run.run_number) if raw_signature is not None else {}
            except (OSError, ImportError, TimeoutError, subprocess.TimeoutExpired) as exc:
                # Unavailable raw metadata is different from contradictory raw
                # metadata. Keep the notice while checking independent sources.
                raw_metadata = {'metadata_notices': [f'Raw sample identity is unavailable: {exc}']}
            metadata = self.metadata_index.enrich(run.run_number, raw_metadata)
            # Empty metadata may belong to a file still being written.
            if metadata.get("sample_id") and not raw_metadata.get('metadata_notices'):
                with self._lock:
                    self._cache[run.run_number] = (signature, dict(metadata))
            return metadata
        except (OSError, ValueError, TypeError, ImportError, TimeoutError, subprocess.TimeoutExpired) as exc:
            metadata = {"sample_id": "", "sample_name": "", "identity_status": "conflict" if isinstance(exc, ValueError) else "unidentified",
                        "identity_warning": f"Identity metadata needs review: {exc}"}
            metadata['metadata_actions'] = metadata_actions(metadata)
            return metadata

    def inventory(self, runs: Iterable[WatchedRun]) -> dict[str, Any]:
        runs = list(runs)
        self.metadata_index.refresh(force=True)
        # Limit concurrent facility reads. Read scalar identity fields only;
        # full scan conditions are collected later for runs actually processed.
        with ThreadPoolExecutor(max_workers=4) as pool:
            identities = list(pool.map(self.identity, runs))
        grouped: dict[str, dict[str, Any]] = {}
        unidentified = 0
        for run, metadata in zip(runs, identities):
            sample_id = metadata.get("sample_id", "")
            if not sample_id:
                unidentified += 1
                continue
            group = grouped.setdefault(sample_id, {"names": set(), "count": 0})
            if metadata.get("sample_name"):
                group["names"].add(metadata["sample_name"])
            group["count"] += 1
        options = []
        for sample_id, group in sorted(grouped.items(), key=lambda item: item[0].casefold()):
            names = " / ".join(sorted(group["names"])) or "Name not reported"
            options.append({"value": sample_id, "title": f"{sample_id} — {names} ({group['count']} scans)",
                            "scan_count": group["count"]})
        run_options = []
        for run, metadata in zip(runs, identities):
            role = run_role(run)
            title = role["run_title"] or "Title not reported"
            run_options.append({"value": run.run_number, "sample_id": metadata.get("sample_id", ""),
                                "title": f"{run.run_id} — {role['run_role']}: {title}",
                                "identity_status": metadata.get('identity_status','unidentified'),
                                "metadata_actions": metadata.get('metadata_actions',[]), **role})
        return {"options": options, "run_options": run_options,
                "scan_count": len(runs), "unidentified_count": unidentified,
                "metadata_notices": self.metadata_index.notices,
                "identity_conflict_count": sum(m.get('identity_status')=='conflict' for m in identities),
                "metadata_resolution": [{"run_number": r.run_number, "status": m.get('identity_status','unidentified'),
                   "sample_id": m.get('sample_id',''), "actions": m.get('metadata_actions',[]),
                   "sources": m.get('identity_sources',[])} for r,m in zip(runs,identities)]}


def identity_readiness(index, runs):
    """A visible disposition per selected run; does not certify a profile."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        identities = list(pool.map(index.identity, runs))
    return [{"run_number": r.run_number, "status": m.get('identity_status','unidentified'),
             "sample_id": m.get('sample_id',''), "actions": m.get('metadata_actions',[]),
             "sources": m.get('identity_sources',[])} for r,m in zip(runs,identities)]
