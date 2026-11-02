"""Cached sample discovery for the completed reductions of one experiment."""

from concurrent.futures import ThreadPoolExecutor
import threading
import subprocess
from pathlib import Path
import re
import time
from typing import Any, Iterable

from .powgen_metadata import powgen_nexus_path, read_powgen_sample_identity
from .powgen_watch import WatchedRun
from .facility_metadata import ExperimentMetadataIndex, title_role, metadata_actions
from .sample_catalog import catalog_created_at, load_sample_catalog, TTL_SECONDS

MAX_CATALOG_NAME_REPRESENTATIVES = 128


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


def explicitly_excluded(run, exclusions):
    if run.run_number in exclusions or run.run_id in exclusions:return True
    if run.product_id and f'NOM_{run.run_number}_{run.product_id}' in exclusions:return True
    if run.instrument=='NOM' and run.product_id and run.product_id.startswith('SUM_') and exclusions:
        from .nomad_products import product_from_path
        product=product_from_path(run.source_path)
        if product is None or set(product.members).intersection(exclusions):return True
    return False


def include_run(run: WatchedRun, *, excluded_run_numbers=(), include_background_runs=False,
                include_ramp_runs=False) -> bool:
    if explicitly_excluded(run,excluded_run_numbers):return False
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
        self._preview_cache: dict[int, tuple[tuple[Any, ...], dict[str, Any]]] = {}
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

    def candidate_runs(self, runs: Iterable[WatchedRun], selected_ids) -> list[WatchedRun]:
        """Use a recent browsing index to avoid deep reads of other samples.

        This only narrows candidates. Every returned candidate is checked by
        ``identity`` before selection/submission; previews never enter its
        verified cache. Missing/stale previews retain the source-file fallback.
        """
        runs = list(runs)
        self.metadata_index.refresh()
        if self._preview_cache and any(time.monotonic() - row[0][3] >= TTL_SECONDS
                                       for row in self._preview_cache.values()):
            self.inventory(runs, catalog=True)
        selected = set(selected_ids)
        result = []
        for run in runs:
            cached = self._preview_cache.get(run.run_number)
            if cached:
                signature, metadata = cached
                fingerprint, revision, generation, created = signature
                if (fingerprint == run.fingerprint and revision == self.metadata_index.revision
                        and generation == self.metadata_index.generation and time.monotonic() - created < TTL_SECONDS):
                    candidate_id = metadata.get("sample_id")
                    if (candidate_id and metadata.get("identity_status") == "resolved"
                            and candidate_id not in selected):
                        continue
            result.append(run)
        return result

    def inventory(self, runs: Iterable[WatchedRun], *, catalog: bool = False, progress=None,
                  representative_names: bool = False) -> dict[str, Any]:
        runs = list(runs)
        self.metadata_index.refresh(force=True)
        catalog_rows, catalog_notice = {}, ""
        created = time.monotonic()
        if catalog:
            try:
                catalog_rows = load_sample_catalog(self.metadata_index.instrument, self.ipts)
                created = catalog_created_at(self.metadata_index.instrument, self.ipts, catalog_rows)
            except (OSError, ValueError):
                catalog_notice = "The sample catalog is unavailable; reading source-file identities."
        representatives, disputed_groups = {}, set()
        groups = {}
        if catalog and representative_names:
            for run in runs:
                preview = catalog_rows.get(run.run_number) or {}
                sample_id = preview.get("sample_id")
                if (sample_id and preview.get("identity_status") == "resolved"
                        and sample_id not in {"4294967295", "18446744073709551615"}):
                    groups.setdefault(sample_id, []).append(run)
            groups = {key: sorted(value, key=lambda r: (r.run_number, r.source_path))
                      for key, value in groups.items()
                      if any(not catalog_rows[r.run_number].get("sample_name") for r in value)}
            chosen, paths = [], set()
            # One name per group before spending the remaining bound on a
            # second endpoint. Repeated source paths are read only once.
            for endpoint in (0, -1):
                for sample_id in sorted(groups):
                    run = groups[sample_id][endpoint]
                    if run.source_path not in paths and len(chosen) < MAX_CATALOG_NAME_REPRESENTATIVES:
                        chosen.append((run, sample_id))
                        paths.add(run.source_path)
            if chosen:
                if progress is not None:
                    progress(0, len(runs))
                with ThreadPoolExecutor(max_workers=4) as pool:
                    for (run, expected), metadata in zip(chosen, pool.map(self.identity, [r for r, _ in chosen])):
                        representatives[run.run_number] = metadata
                        if metadata.get("identity_status") != "resolved" or metadata.get("sample_id") != expected:
                            disputed_groups.add(expected)
        def read_identity(run):
            if run.run_number in representatives:
                return representatives[run.run_number]
            if run.run_number in self._cache:
                return self.identity(run)  # Retain already-observed source conflicts.
            preview = catalog_rows.get(run.run_number)
            if (preview is None or preview.get("identity_status") != "resolved"
                    or not preview.get("sample_id")
                    or preview.get("sample_id") in disputed_groups
                    or preview.get("sample_id") in {"4294967295", "18446744073709551615"}):
                # Missing/conflicting catalog evidence and unset facility IDs
                # require source inspection. In particular, retain a recorded
                # NoSample name rather than inventing one from an index token.
                return self.identity(run)
            metadata = self.metadata_index.enrich(run.run_number, preview, include_xml=False)
            metadata.update(identity_verification="catalog_preview", run_title=preview.get("run_title", ""))
            return metadata
        # Limit concurrent facility reads. Read scalar identity fields only;
        # full scan conditions are collected later for runs actually processed.
        identities=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for count, identity in enumerate(pool.map(read_identity if catalog else self.identity, runs),1):
                identities.append(identity)
                if progress is not None and (count==1 or count%25==0 or count==len(runs)):
                    progress(count,len(runs))
        if catalog:
            observed = time.monotonic()
            self._preview_cache = {r.run_number: ((r.fingerprint, self.metadata_index.revision,
                self.metadata_index.generation,
                created if m.get("identity_verification") == "catalog_preview" else observed), dict(m))
                for r, m in zip(runs, identities)}
        grouped: dict[str, dict[str, Any]] = {}
        unidentified = 0
        for run, metadata in zip(runs, identities):
            sample_id = metadata.get("sample_id", "")
            if not sample_id:
                unidentified += 1
                continue
            group = grouped.setdefault(sample_id, {"names": set(), "count": 0, "name_runs": set(), "previews": 0})
            if metadata.get("sample_name"):
                group["names"].add(metadata["sample_name"])
                if metadata.get("identity_verification") != "catalog_preview":
                    group["name_runs"].add(run.run_number)
            group["previews"] += metadata.get("identity_verification") == "catalog_preview"
            group["count"] += 1
        options = []
        for sample_id, group in sorted(grouped.items(), key=lambda item: item[0].casefold()):
            names = " / ".join(sorted(group["names"])) or "Name not reported"
            options.append({"value": sample_id, "title": f"{sample_id} — {names} ({group['count']} scans)",
                            "scan_count": group["count"]})
            if groups:
                scope = ("source-checked representatives" if group["previews"] and group["name_runs"] else
                         "source files" if group["name_runs"] else "catalog metadata" if group["names"] else "not reported")
                options[-1].update(name_scope=scope, name_run_numbers=sorted(group["name_runs"]),
                                   name_variation_observed=len(group["names"]) > 1)
                if scope == "source-checked representatives":
                    detail = "sampled names differ" if len(group["names"]) > 1 else "representative name"
                    options[-1]["title"] = f"{sample_id} — {names} ({group['count']} scans; {detail})"
        run_options = []
        for run, metadata in zip(runs, identities):
            if metadata.get("identity_verification") == "catalog_preview":
                title = metadata.get("run_title") or metadata.get("table_fields", {}).get("run_title", "")
                role = {"run_title": title, "run_role": title_role(title)}
            else:
                role = run_role(run)
            title = role["run_title"] or "Title not reported"
            run_options.append({"value": run.run_number, "sample_id": metadata.get("sample_id", ""),
                                "title": f"{run.run_id} — {role['run_role']}: {title}",
                                "identity_status": metadata.get('identity_status','unidentified'),
                                "metadata_actions": metadata.get('metadata_actions',[]), **role})
        notices = [*self.metadata_index.notices, *([catalog_notice] if catalog_notice else [])]
        if groups:
            notices.append("Up to two representative runs per ID are checked; cached checked names are retained. Other scan names may differ. Every selected run is source-checked before analysis.")
            if len(representatives) >= MAX_CATALOG_NAME_REPRESENTATIVES:
                notices.append(f"Representative name checks are bounded to {MAX_CATALOG_NAME_REPRESENTATIVES} files; the sample-ID inventory still includes every discovered scan.")
        if disputed_groups:
            notices.append(f"Catalog identities disagreed with source metadata for {len(disputed_groups)} sample group(s); those groups use per-run source checks.")
        return {"options": options, "run_options": run_options,
                "scan_count": len(runs), "unidentified_count": unidentified,
                "metadata_notices": notices,
                "catalog_preview_count": sum(m.get("identity_verification") == "catalog_preview" for m in identities),
                "identity_conflict_count": sum(m.get('identity_status')=='conflict' for m in identities),
                "metadata_resolution": [{"run_number": r.run_number, "status": m.get('identity_status','unidentified'),
                   "sample_id": m.get('sample_id',''), "actions": m.get('metadata_actions',[]),
                   "sources": m.get('identity_sources',[]),
                   "verification": m.get("identity_verification", "source_file")} for r,m in zip(runs,identities)]}


def select_runs(runs, index, selected_ids, *, limit=None, excluded_run_numbers=(),
                include_background_runs=False, include_ramp_runs=False, selection_stats=None):
    """Choose newest eligible runs before expensive per-run scientific checks."""
    runs = list(runs)
    if selected_ids is not None and hasattr(index, "candidate_runs"):
        runs = index.candidate_runs(runs, selected_ids)
    excluded = set()
    identity_review = {}
    def eligible(run):
        if explicitly_excluded(run,excluded_run_numbers):
            excluded.add(run.run_id if run.product_id else run.run_number)
            return False
        if selected_ids is not None:
            identity = index.identity(run)
            if identity.get("sample_id") not in selected_ids:
                if identity.get("identity_status") in {"conflict", "unidentified"} or (
                        hasattr(index, "_preview_cache") and run.run_number in index._preview_cache):
                    identity_review[run.run_number] = identity.get("identity_warning") or "Source identity does not match the selected sample"
                return False
        ok = include_run(run, excluded_run_numbers=excluded_run_numbers,
            include_background_runs=include_background_runs, include_ramp_runs=include_ramp_runs)
        if not ok:
            excluded.add(run.run_id if run.product_id else run.run_number)
        return ok
    if limit is not None:
        selected = []
        for run in reversed(runs):
            if eligible(run):
                selected.append(run)
                if len(selected) >= max(1, limit):
                    break
        result = list(reversed(selected))
        if selection_stats is not None:
            selection_stats["excluded_count"] = len(excluded)
            selection_stats["identity_review"] = identity_review
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        accepted = list(pool.map(eligible, runs))
    if selection_stats is not None:
        selection_stats["excluded_count"] = len(excluded)
        selection_stats["identity_review"] = identity_review
    return [r for r, ok in zip(runs, accepted) if ok]


def identity_readiness(index, runs):
    """A visible disposition per selected run; does not certify a profile."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        identities = list(pool.map(index.identity, runs))
    return [{"run_number": r.run_number, "status": m.get('identity_status','unidentified'),
             "sample_id": m.get('sample_id',''), "actions": m.get('metadata_actions',[]),
             "sources": m.get('identity_sources',[])} for r,m in zip(runs,identities)]
