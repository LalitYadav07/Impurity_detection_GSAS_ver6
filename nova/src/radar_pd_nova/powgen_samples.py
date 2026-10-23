"""Cached sample discovery for the completed reductions of one experiment."""

from concurrent.futures import ThreadPoolExecutor
import threading
import subprocess
from pathlib import Path
import re
from typing import Any, Iterable

from .powgen_metadata import powgen_nexus_path, read_powgen_sample_identity
from .powgen_watch import WatchedRun


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
    text = title.casefold()
    if re.search(r"\b(empty|background|bkgd|no sample|without sample)\b", text):
        role = "background"
    elif re.search(r"\b(ramp(?:ing)?|cool(?:ing)?(?: down| to)?|heat(?:ing)? up)\b", text):
        role = "ramp"
    else:
        role = "unclassified"
    return {"run_title": title, "run_role": role}


def include_run(run: WatchedRun, *, excluded_run_numbers=(), include_background_runs=False,
                include_ramp_runs=False) -> bool:
    if run.run_number in excluded_run_numbers:
        return False
    role = run_role(run)["run_role"]
    return ((role != "background" or include_background_runs)
            and (role != "ramp" or include_ramp_runs))


class PowgenSampleIndex:
    def __init__(self, ipts: str) -> None:
        self.ipts = ipts
        self._cache: dict[int, tuple[tuple[int, int], dict[str, str]]] = {}
        self._lock = threading.Lock()

    def identity(self, run: WatchedRun) -> dict[str, str]:
        """Retry missing/unreadable NeXus files; refresh identities when they change."""

        try:
            stat = powgen_nexus_path(self.ipts, run.run_number).stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            with self._lock:
                cached = self._cache.get(run.run_number)
            if cached and cached[0] == signature:
                return dict(cached[1])
            metadata = read_powgen_sample_identity(self.ipts, run.run_number)
            # Empty metadata may belong to a file still being written.
            if metadata.get("sample_id"):
                with self._lock:
                    self._cache[run.run_number] = (signature, dict(metadata))
            return metadata
        except (OSError, ValueError, TypeError, ImportError, TimeoutError, subprocess.TimeoutExpired):
            return {"sample_id": "", "sample_name": ""}

    def inventory(self, runs: Iterable[WatchedRun]) -> dict[str, Any]:
        runs = list(runs)
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
                                "title": f"{run.run_id} — {role['run_role']}: {title}", **role})
        return {"options": options, "run_options": run_options,
                "scan_count": len(runs), "unidentified_count": unidentified}
