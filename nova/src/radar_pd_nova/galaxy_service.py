"""Galaxy-backed execution and recovery for the RADAR-PD NOVA client."""

from __future__ import annotations

import json
import os
import posixpath
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlsplit, urlunsplit

import requests
import yaml

from .configuration import config_from_contract, delivery_from_contract, dump_configuration
from .facility import build_facility_export_path
from .models import (
    AnalysisConfig,
    AnalysisMode,
    CacheManifest,
    InputSelection,
    InputSource,
    ResultStatus,
    RunRecord,
    RunStatus,
    SubmissionPhase,
    SubmissionProgress,
    SubmissionSnapshot,
    UtilityActionRecord,
)

ANALYZE_TOOL_ID = os.getenv("RADAR_PD_ANALYZE_TOOL_ID", "neutrons_radar_pd_analyze_prototype")
SNS_RESOLVER_TOOL_ID = "neutrons_radar_pd_resolve_sns_input_prototype"
LIBRARY_BUILDER_TOOL_ID = "neutrons_radar_pd_library_builder_prototype"
GPX_HANDOFF_TOOL_ID = "neutrons_radar_pd_gpx_handoff_prototype"
COMPARE_SERIES_TOOL_ID = "neutrons_radar_pd_compare_series_prototype"
RESULT_EXPLORER_TOOL_ID = "neutrons_radar_pd_result_explorer_prototype"
EXPORT_DATASETS_TOOL_ID = "neutrons_export"
RUN_NAME_PREFIX = "RADAR-PD NOVA"
SUBMISSION_ACK_TIMEOUT_SECONDS = float(os.getenv("RADAR_PD_SUBMISSION_ACK_TIMEOUT", "60"))

_DIFFRACTION_SUFFIXES = frozenset(
    {".dat", ".xye", ".xy", ".csv", ".txt", ".fxye", ".gsa", ".gsas", ".gss", ".xrdml", ".xml"}
)
_DIFFRACTION_EXTENSIONS = frozenset(suffix.lstrip(".") for suffix in _DIFFRACTION_SUFFIXES)

_UPLOAD_SUFFIX_POLICIES: dict[str, tuple[frozenset[str], str]] = {
    "diffraction data": (
        _DIFFRACTION_SUFFIXES,
        ".dat",
    ),
    "instrument profile": (frozenset({".instprm", ".prm", ".inst", ".ins"}), ".instprm"),
    "main phase CIF": (frozenset({".cif"}), ".cif"),
    "candidate library": (frozenset({".zip"}), ".zip"),
    "NeXus event file": (frozenset({".nxs", ".h5", ".hdf5"}), ".nxs"),
}

_REMOTE_ROLE_SUFFIXES: dict[str, frozenset[str]] = {
    "data": _DIFFRACTION_SUFFIXES,
    "instrument": frozenset({".instprm", ".prm", ".inst", ".ins"}),
    "cif": frozenset({".cif"}),
    "event": frozenset({".nxs", ".h5", ".hdf5"}),
}


def _upload_filename(source: Path, label: str) -> str:
    """Return a Galaxy filename that retains a meaningful scientific suffix.

    nova-trame's browser upload currently writes laptop files to an
    extensionless ``NamedTemporaryFile`` and exposes only that server path.
    Galaxy uses the supplied display filename while creating the HDA, so add
    a conservative type-specific suffix whenever the temporary basename no
    longer carries one of the extensions accepted by the corresponding form
    control. Server-selected files keep their original names unchanged.
    """

    name = source.name
    policy = _UPLOAD_SUFFIX_POLICIES.get(label)
    if policy is not None:
        accepted_suffixes, fallback_suffix = policy
        if source.suffix.lower() not in accepted_suffixes:
            name = f"{name}{fallback_suffix}"
    return f"RADAR-PD {label} | {name}"


def _galaxy_upload_file_type(dataset_name: str) -> str:
    """Return the Galaxy datatype needed by strict downstream tool inputs.

    Galaxy cannot reliably sniff a ZIP archive from NOVA's extensionless
    temporary upload paths.  The display filename has already recovered the
    scientific suffix, so use it to declare archive uploads explicitly.
    Other inputs retain Galaxy's normal automatic datatype detection.
    """

    return "zip" if Path(dataset_name).suffix.lower() == ".zip" else "auto"


def _extract_results_archive(archive: Path, destination: Path) -> None:
    """Extract a RADAR-PD archive without allowing paths outside destination."""

    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise ValueError(f"Unsafe path in RADAR-PD results archive: {member.filename}")
        handle.extractall(destination)


def _decode_parameter(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{\"":
        return value
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return value


def _parameter_items(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten Galaxy's several job-parameter encodings without losing parent values."""

    decoded = _decode_parameter(value)
    items: list[tuple[str, Any]] = []
    if prefix:
        items.append((prefix, decoded))
    if isinstance(decoded, dict):
        if "name" in decoded and "value" in decoded:
            child = str(decoded["name"])
            path = f"{prefix}.{child}" if prefix else child
            items.extend(_parameter_items(decoded["value"], path))
        else:
            for key, child in decoded.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                items.extend(_parameter_items(child, path))
    elif isinstance(decoded, list):
        for index, child in enumerate(decoded):
            path = f"{prefix}.{index}" if prefix else str(index)
            items.extend(_parameter_items(child, path))
    return items


def _normalized_parameter_path(value: str) -> str:
    return re.sub(r"[|/]+", ".", value).strip(".").lower()


def _parameter_value(parameters: Any, *suffixes: str) -> Any:
    normalized_suffixes = tuple(_normalized_parameter_path(item) for item in suffixes)
    for path, value in _parameter_items(parameters):
        normalized = _normalized_parameter_path(path)
        if any(normalized == suffix or normalized.endswith(f".{suffix}") for suffix in normalized_suffixes):
            return value
    return None


def _scalar(value: Any) -> Any:
    decoded = _decode_parameter(value)
    if isinstance(decoded, dict):
        for key in ("value", "values"):
            if key in decoded:
                return _scalar(decoded[key])
        return None
    if isinstance(decoded, list):
        return _scalar(decoded[0]) if decoded else None
    return decoded


def _text(value: Any) -> str | None:
    scalar = _scalar(value)
    if scalar is None:
        return None
    text = str(scalar).strip()
    return text if text and text not in {"None", "Not available."} else None


def _boolean(value: Any) -> bool | None:
    scalar = _scalar(value)
    if isinstance(scalar, bool):
        return scalar
    text = str(scalar).strip().lower() if scalar is not None else ""
    if text.startswith("--no-") or text in {"false", "no", "0", "off", "none", ""}:
        return False if text else None
    if text.startswith("--") or text in {"true", "yes", "1", "on"}:
        return True
    return None


def _number(value: Any, kind: type[int] | type[float]) -> int | float | None:
    scalar = _scalar(value)
    if scalar is None:
        return None
    try:
        return kind(scalar)
    except (TypeError, ValueError):
        return None


def _dataset_id(value: Any) -> str | None:
    decoded = _decode_parameter(value)
    if isinstance(decoded, dict):
        if decoded.get("id"):
            return str(decoded["id"])
        for key in ("value", "values", "dataset"):
            if key in decoded:
                found = _dataset_id(decoded[key])
                if found:
                    return found
    elif isinstance(decoded, list):
        for child in decoded:
            found = _dataset_id(child)
            if found:
                return found
    return None


def stage_from_console(text: str, status: RunStatus) -> tuple[str, int]:
    """Infer a stable user-facing stage from the existing batch console log."""

    lowered = text.lower()
    if status == RunStatus.OK:
        return "Results ready", 100
    if status in {RunStatus.ERROR, RunStatus.CANCELLED}:
        return "Stopped", 100
    stages = [
        (("final refinement", "gsas validation", "polish"), "Final refinement ranking", 88),
        (("512", "pattern scoring", "rerank"), "Pattern scoring", 72),
        (("pearson", "lattice nudge", "nudg"), "Lattice nudging", 55),
        (("coarse", "histogram", "ml rank"), "Candidate search", 35),
        (("main phase", "background", "signal"), "Signal and main-phase preparation", 18),
    ]
    for needles, label, progress in stages:
        if any(needle in lowered for needle in needles):
            return label, progress
    if status == RunStatus.RUNNING:
        return "Starting RADAR-PD", 8
    return "Waiting for compute", 3


def normalize_status(value: Any) -> RunStatus:
    text = str(getattr(value, "value", value)).lower()
    if any(token in text for token in ("finish", "ok", "complete")):
        return RunStatus.OK
    if any(token in text for token in ("error", "fail")):
        return RunStatus.ERROR
    if any(token in text for token in ("cancel", "delete", "stop")):
        return RunStatus.CANCELLED
    if "run" in text:
        return RunStatus.RUNNING
    return RunStatus.QUEUED


class GalaxyService:
    """Thin service around nova-galaxy plus read-only Galaxy REST endpoints."""

    def __init__(
        self,
        galaxy_url: str | None = None,
        galaxy_key: str | None = None,
        history_id: str | None = None,
        *,
        output_root: str | Path | None = None,
    ) -> None:
        self.galaxy_url = (galaxy_url or os.getenv("GALAXY_URL", "")).rstrip("/")
        self.galaxy_key = galaxy_key or os.getenv("GALAXY_API_KEY", "")
        self.history_id = history_id or os.getenv("HISTORY_ID", "")
        runtime_uid = getattr(os, "getuid", lambda: "user")()
        self.output_root = Path(output_root) if output_root is not None else (
            Path(tempfile.gettempdir()) / f"radar-pd-nova-{runtime_uid}-{os.getpid()}"
        )
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._tools: dict[str, Any] = {}
        self._submissions: dict[str, RunRecord] = {}
        self._submission_cancel_events: dict[str, threading.Event] = {}
        self._active_submission_tokens: set[str] = set()
        self._lock = threading.RLock()

    def validate_connection(self) -> None:
        if not self.galaxy_url or not self.galaxy_key or not self.history_id:
            raise RuntimeError("GALAXY_URL, GALAXY_API_KEY, and HISTORY_ID are required")
        response = requests.get(
            f"{self.galaxy_url}/api/version",
            headers=self._headers,
            timeout=20,
        )
        response.raise_for_status()

    @property
    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.galaxy_key, "accept": "application/json"}

    @contextmanager
    def _store(self) -> Iterator[Any]:
        from nova.galaxy import Connection
        from nova.galaxy.data_store import Datastore

        connection = Connection(galaxy_url=self.galaxy_url, galaxy_key=self.galaxy_key)
        helper = connection.connect()
        store = Datastore(RUN_NAME_PREFIX, helper, self.history_id)
        helper.datastores.append(store)
        store.persist()
        try:
            yield store
        finally:
            helper.close()

    @staticmethod
    def _existing_dataset(dataset_id: str, store: Any, name: str) -> Any:
        try:
            from nova.galaxy import Dataset

            dataset = Dataset(name=name, force_upload=False)
        except ImportError:  # Lightweight test doubles may omit Dataset.
            from types import SimpleNamespace

            dataset = SimpleNamespace(name=name)
        dataset.id = dataset_id
        dataset.store = store
        return dataset

    @staticmethod
    def _upload_dataset(path: str, store: Any, label: str) -> Any:
        from nova.galaxy import Dataset

        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"{label} does not exist: {source}")
        dataset_name = _upload_filename(source, label)
        galaxy_instance = store.nova_connection.galaxy_instance
        upload_result = galaxy_instance.tools.upload_file(
            path=str(source),
            history_id=store.history_id,
            file_name=dataset_name,
            file_type=_galaxy_upload_file_type(dataset_name),
        )
        outputs = upload_result.get("outputs", []) if isinstance(upload_result, dict) else []
        if not outputs or not outputs[0].get("id"):
            raise RuntimeError(f"Galaxy did not return a dataset identifier after uploading {label}")

        dataset = Dataset(name=dataset_name, force_upload=False)
        dataset.id = str(outputs[0]["id"])
        dataset.store = store
        galaxy_instance.datasets.wait_for_dataset(dataset.id)
        return dataset

    def upload_document(self, path: str | Path, *, label: str) -> str:
        """Upload one provenance document into the active Galaxy history.

        This is intentionally a small public wrapper around the same upload
        path used by Analyze submissions.  Watch state therefore remains a
        Galaxy-owned dataset instead of being written into an experiment IPTS.
        """

        with self._store() as store:
            dataset = self._upload_dataset(str(path), store, label)
            return str(dataset.id)

    def upload_json_document(self, payload: dict[str, Any], *, name: str, label: str) -> str:
        """Serialize and upload a JSON provenance document via owned staging."""

        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name)).strip(".-") or "document.json"
        if not safe_name.lower().endswith(".json"):
            safe_name += ".json"
        staging = self.output_root / "provenance"
        staging.mkdir(parents=True, exist_ok=True)
        path = staging / safe_name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return self.upload_document(path, label=label)

    def load_configuration_dataset(self, dataset_id: str) -> AnalysisConfig:
        """Load a reusable RADAR-PD configuration from Galaxy."""

        contract = self._dataset_document(str(dataset_id))
        if not contract:
            raise ValueError(f"Galaxy configuration dataset {dataset_id!r} is empty")
        return config_from_contract(contract)

    def load_json_document(self, dataset_id: str) -> dict[str, Any]:
        """Load one JSON/YAML provenance document from Galaxy History."""

        document = self._dataset_document(str(dataset_id))
        if not document:
            raise ValueError(f"Galaxy provenance dataset {dataset_id!r} is empty")
        return document

    @staticmethod
    def _dataset_scientific_role(row: dict[str, Any]) -> tuple[str, bool]:
        name = str(row.get("name") or "").lower()
        extension = str(row.get("extension") or row.get("file_ext") or "").lower().lstrip(".")
        # Galaxy may sniff an uploaded ``.instprm`` or ``.dat`` as the
        # generic ``txt`` datatype.  The preserved scientific filename and
        # NOVA upload label are therefore more authoritative than datatype
        # alone for these input roles.
        preserved_name = name.rsplit("|", 1)[-1].strip()
        preserved_suffix = Path(preserved_name).suffix.lower().lstrip(".")
        generated_tokens = (
            "results archive",
            "radar-pd summary",
            "resolved config",
            "input manifest",
            "console output",
            "plot payload",
            "phase fractions",
            "gpx index",
        )
        generated = any(token in name for token in generated_tokens)
        if preserved_suffix in {"instprm", "prm", "inst", "ins"} or "instrument profile" in name:
            return "instrument", generated
        if preserved_suffix == "cif" or extension == "cif":
            return "cif", generated
        if (preserved_suffix == "zip" or extension == "zip") and not generated:
            return "candidate_library", False
        if preserved_suffix in {"nxs", "h5", "hdf5"} or extension in {"nxs", "h5", "hdf5"}:
            return "event", generated
        if preserved_suffix in _DIFFRACTION_EXTENSIONS or extension in _DIFFRACTION_EXTENSIONS:
            return "diffraction", generated
        if (preserved_suffix in {"yaml", "yml"} or extension in {"yaml", "yml"}) and (
            "config" in name or "radar" in name
        ):
            return "configuration", generated
        return "other", True

    def search_history_datasets(
        self,
        *,
        query: str = "",
        limit: int = 100,
        offset: int = 0,
        include_generated: bool = False,
    ) -> list[dict[str, Any]]:
        """Page and search the active Galaxy history on the server."""

        if limit < 1 or offset < 0:
            return []
        params: dict[str, Any] = {
            "v": "dev",
            "limit": min(limit, 500),
            "offset": offset,
            "order": "update_time-dsc",
        }
        if query.strip():
            params.update({"q": "name-contains", "qv": query.strip()})
        response = requests.get(
            f"{self.galaxy_url}/api/histories/{self.history_id}/contents",
            headers=self._headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        page = response.json()
        if not isinstance(page, list):
            raise RuntimeError("Galaxy returned an invalid history-dataset response")
        result: list[dict[str, Any]] = []
        for row in page:
            if not isinstance(row, dict):
                continue
            if row.get("history_content_type", "dataset") != "dataset" or row.get("deleted"):
                continue
            if str(row.get("state") or "ok") not in {"ok", "deferred"}:
                continue
            role, generated = self._dataset_scientific_role(row)
            if generated and not include_generated:
                continue
            dataset_id = str(row.get("id"))
            dataset_name = str(row.get("name") or row.get("hid") or "dataset")
            original_name = dataset_name.rsplit("|", 1)[-1].strip()
            update_time = str(row.get("update_time") or "")[:16].replace("T", " ")
            display_suffix = " · ".join(value for value in (update_time, dataset_id[:8]) if value)
            result.append(
                {
                    "id": dataset_id,
                    "name": dataset_name,
                    "display_name": f"{original_name} · {display_suffix}" if display_suffix else original_name,
                    "extension": str(row.get("extension") or row.get("file_ext") or "data"),
                    "state": str(row.get("state") or ""),
                    "update_time": str(row.get("update_time") or ""),
                    "role": role,
                    "generated": generated,
                }
            )
        return result

    def list_remote_file_sources(self) -> list[dict[str, Any]]:
        """Return Galaxy file sources available to the current NDIP user.

        Galaxy applies the user's file-source authorization when this endpoint
        is called.  RADAR-PD never accepts a native server path from the
        browser; it retains the opaque ``gxfiles://`` URI until Galaxy imports
        the selected object into the active History.
        """

        response = requests.get(
            f"{self.galaxy_url}/api/remote_files/plugins",
            headers=self._headers,
            params={"browsable_only": "true"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError("Galaxy returned an invalid remote-file-source response")
        sources: list[dict[str, Any]] = []
        for raw in payload:
            if not isinstance(raw, dict) or raw.get("browsable") is False:
                continue
            source_id = str(raw.get("id") or raw.get("plugin_id") or "").strip()
            uri_root = str(raw.get("uri_root") or raw.get("uri") or "").strip()
            if not uri_root and source_id:
                uri_root = f"gxfiles://{source_id}/"
            if not source_id or not uri_root.startswith("gxfiles://"):
                continue
            sources.append(
                {
                    "id": source_id,
                    "title": str(raw.get("label") or raw.get("name") or source_id),
                    "value": uri_root.rstrip("/") + "/",
                    "description": str(raw.get("doc") or raw.get("description") or ""),
                    "writable": bool(raw.get("writable", False)),
                }
            )
        return sorted(sources, key=lambda item: item["title"].lower())

    @staticmethod
    def remote_parent_uri(uri: str, root_uri: str) -> str:
        """Move one directory up without escaping the selected file source."""

        current = urlsplit(uri)
        root = urlsplit(root_uri)
        if current.scheme != "gxfiles" or (current.scheme, current.netloc) != (root.scheme, root.netloc):
            return root_uri.rstrip("/") + "/"
        root_path = root.path.rstrip("/") or "/"
        current_path = current.path.rstrip("/") or "/"
        parent = posixpath.dirname(current_path)
        if root_path != "/" and not (parent == root_path or parent.startswith(root_path + "/")):
            parent = root_path
        return urlunsplit((current.scheme, current.netloc, parent.rstrip("/") + "/", "", ""))

    def list_remote_files(self, target: str, *, role: str = "any") -> list[dict[str, Any]]:
        """List one Galaxy remote directory and apply scientific file filters."""

        if not target.startswith("gxfiles://"):
            raise ValueError("Remote file target must be a gxfiles:// URI")
        response = requests.get(
            f"{self.galaxy_url}/api/remote_files",
            headers=self._headers,
            params={"target": target, "format": "uri", "recursive": "false"},
            timeout=45,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            payload = payload.get("entries") or payload.get("items") or []
        if not isinstance(payload, list):
            raise RuntimeError("Galaxy returned an invalid remote-directory response")
        accepted = _REMOTE_ROLE_SUFFIXES.get(role)
        entries: list[dict[str, Any]] = []
        for raw in payload:
            if not isinstance(raw, dict):
                continue
            kind_text = str(raw.get("class") or raw.get("kind") or raw.get("type") or "").lower()
            is_directory = kind_text in {"directory", "folder", "dir"} or bool(raw.get("is_directory"))
            uri = str(raw.get("uri") or raw.get("url") or raw.get("path") or "").strip()
            if not uri.startswith("gxfiles://"):
                continue
            name = str(raw.get("name") or Path(urlsplit(uri).path.rstrip("/")).name or uri)
            if not is_directory and accepted is not None and Path(name).suffix.lower() not in accepted:
                continue
            entries.append(
                {
                    "title": name,
                    "value": uri.rstrip("/") + "/" if is_directory else uri,
                    "kind": "directory" if is_directory else "file",
                    "size": raw.get("size"),
                }
            )
        return sorted(entries, key=lambda item: (item["kind"] != "directory", item["title"].lower()))

    def _import_remote_dataset(self, uri: str, label: str) -> str:
        """Import an authorized Galaxy file-source URI into this History."""

        if not uri.startswith("gxfiles://"):
            raise ValueError(f"{label} is not a Galaxy remote-file URI")
        filename = Path(urlsplit(uri).path).name or label.replace(" ", "_")
        payload = {
            "history_id": self.history_id,
            "targets": [
                {
                    "destination": {"type": "hdas"},
                    "items": [
                        {
                            "src": "url",
                            "url": uri,
                            "name": f"RADAR-PD {label} | {filename}",
                            "ext": "auto",
                        }
                    ],
                }
            ],
        }
        response = requests.post(
            f"{self.galaxy_url}/api/tools/fetch",
            headers={**self._headers, "content-type": "application/json"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        outputs = result.get("outputs", []) if isinstance(result, dict) else []
        if not outputs or not outputs[0].get("id"):
            raise RuntimeError(f"Galaxy did not return a dataset after importing {label}")
        dataset_id = str(outputs[0]["id"])
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            status_response = requests.get(
                f"{self.galaxy_url}/api/datasets/{dataset_id}",
                headers=self._headers,
                timeout=30,
            )
            status_response.raise_for_status()
            state = str(status_response.json().get("state") or "").lower()
            if state in {"ok", "deferred"}:
                return dataset_id
            if state in {"error", "discarded", "failed"}:
                raise RuntimeError(f"Galaxy could not import {label} from the selected remote file")
            time.sleep(1)
        raise TimeoutError(f"Galaxy did not finish importing {label} within 10 minutes")

    def list_history_datasets(self, *, limit: int = 5000, page_size: int = 500) -> list[dict[str, Any]]:
        """Compatibility pagination API; new views should call search_history_datasets."""

        rows: list[dict[str, Any]] = []
        offset = 0
        while offset < limit:
            requested = min(page_size, limit - offset)
            page = self.search_history_datasets(
                limit=requested,
                offset=offset,
                include_generated=True,
            )
            rows.extend(page)
            if len(page) < requested:
                break
            offset += requested
        return rows

    def _job_details(self, uid: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fetch the durable Galaxy job record used to recover a NOVA session."""

        response = requests.get(
            f"{self.galaxy_url}/api/jobs/{uid}",
            headers=self._headers,
            params={"full": "true"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        details = dict(fallback or {})
        if isinstance(payload, dict):
            details.update(payload)
        return details

    @staticmethod
    def _job_parameters(job: dict[str, Any]) -> Any:
        candidates = [job.get(key) for key in ("params", "parameters", "inputs", "job_inputs") if job.get(key)]
        if not candidates:
            return {}
        return candidates[0] if len(candidates) == 1 else candidates

    @staticmethod
    def _named_output_ids(value: Any) -> dict[str, str]:
        output_ids: dict[str, str] = {}
        decoded = _decode_parameter(value)
        if isinstance(decoded, dict):
            for name, item in decoded.items():
                identifier = _dataset_id(item)
                if identifier:
                    output_ids[str(name)] = identifier
        elif isinstance(decoded, list):
            for item in decoded:
                if not isinstance(item, dict):
                    continue
                name = item.get("name") or item.get("output_name") or item.get("label")
                identifier = _dataset_id(item)
                if name and identifier:
                    output_ids[str(name)] = identifier
        return output_ids

    def _job_output_ids(self, job: dict[str, Any]) -> dict[str, str]:
        output_ids: dict[str, str] = {}
        for key in ("outputs", "output_datasets", "output_collections"):
            output_ids.update(self._named_output_ids(job.get(key)))
        return output_ids

    @staticmethod
    def _public_input_dataset_ids(job: dict[str, Any]) -> dict[str, str]:
        """Read encoded HDA IDs from Galaxy's full job ``inputs`` map.

        Tool parameter JSON stores Galaxy's internal integer dataset IDs,
        while the public REST API requires encoded IDs. Full job details
        expose the encoded equivalents in ``inputs``; recovered runs must use
        those values when a saved configuration is reused.
        """

        inputs = _decode_parameter(job.get("inputs"))
        if not isinstance(inputs, dict):
            return {}
        aliases = (
            ("configuration", ("config_file",)),
            ("database_archive", ("database_archive",)),
            ("instrument", ("instrument_file",)),
            ("event_file", ("event_file",)),
            ("main_cif", ("main_cif",)),
            ("data", ("diffraction_pattern",)),
        )
        result: dict[str, str] = {}
        for parameter_name, item in inputs.items():
            normalized = str(parameter_name).replace("|", ".").lower()
            identifier = _dataset_id(item)
            if not identifier:
                continue
            for name, suffixes in aliases:
                if any(normalized.endswith(suffix) for suffix in suffixes):
                    result[name] = identifier
                    break
        return result

    def _dataset_document(self, dataset_id: str) -> dict[str, Any] | None:
        response = requests.get(
            f"{self.galaxy_url}/api/datasets/{dataset_id}/display",
            headers=self._headers,
            params={"raw": "true"},
            timeout=30,
        )
        response.raise_for_status()
        payload = yaml.safe_load(response.text)
        return payload if isinstance(payload, dict) else None

    def _download_dataset(self, dataset_id: str, destination: Path) -> None:
        """Download a durable Galaxy dataset without relying on a live Tool object."""

        response = requests.get(
            f"{self.galaxy_url}/api/datasets/{dataset_id}/display",
            headers=self._headers,
            params={"raw": "true"},
            timeout=120,
            stream=True,
        )
        response.raise_for_status()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    def _dataset_metadata(self, dataset_id: str) -> dict[str, Any]:
        response = requests.get(
            f"{self.galaxy_url}/api/datasets/{dataset_id}",
            headers=self._headers,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _job_metric(job: dict[str, Any], name: str) -> str | None:
        metrics = job.get("job_metrics") or {}
        if isinstance(metrics, dict):
            return _text(metrics.get(name))
        if isinstance(metrics, list):
            for metric in metrics:
                if not isinstance(metric, dict):
                    continue
                metric_name = str(metric.get("name") or metric.get("metric_name") or "")
                if metric_name == name:
                    return _text(metric.get("value"))
        return None

    @staticmethod
    def _elements(value: Any) -> list[str]:
        decoded = _decode_parameter(value)
        if isinstance(decoded, list):
            return [str(item).strip() for item in decoded if str(item).strip()]
        text = _text(decoded)
        return re.split(r"[\s,]+", text) if text else []

    @staticmethod
    def _regions(value: Any) -> list[tuple[float, float]]:
        decoded = _decode_parameter(value)
        if isinstance(decoded, dict):
            decoded = decoded.get("regions") or decoded.get("values") or []
        if not isinstance(decoded, list):
            return []
        regions: list[tuple[float, float]] = []
        for item in decoded:
            if isinstance(item, dict):
                start = _number(item.get("start"), float)
                end = _number(item.get("end"), float)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                start = _number(item[0], float)
                end = _number(item[1], float)
            else:
                continue
            if start is not None and end is not None and start < end:
                regions.append((float(start), float(end)))
        return regions

    def _config_from_parameters(self, parameters: Any) -> AnalysisConfig | None:
        sample_elements = self._elements(
            _parameter_value(parameters, "chemistry.sample_elements", "sample_elements")
        )
        if not sample_elements:
            return None

        kwargs: dict[str, Any] = {"sample_elements": sample_elements}
        text_fields = {
            "run_name": ("reproducibility.run_name", "run_name"),
            "mode": ("analysis.strategy.analysis_mode", "analysis.analysis_mode", "analysis_mode", "mode"),
            "radiation": ("measurement.radiation", "radiation"),
            "instrument_mode": ("measurement.instrument_mode", "instrument_mode"),
            "background_mode": ("background.background_mode", "background.mode", "background_mode"),
            "background_type": ("background.background_type", "background.type", "background_type"),
            "full_profile": ("analysis.strategy.full_profile", "full_profile"),
        }
        for field, aliases in text_fields.items():
            value = _text(_parameter_value(parameters, *aliases))
            if value:
                kwargs[field] = value.lower() if field in {"mode", "radiation", "instrument_mode"} else value

        environment = self._elements(
            _parameter_value(parameters, "chemistry.environment_elements", "environment_elements")
        )
        kwargs["environment_elements"] = environment

        start = _number(_parameter_value(parameters, "pattern.fit_start", "fit_start"), float)
        end = _number(_parameter_value(parameters, "pattern.fit_end", "fit_end"), float)
        if start is not None and end is not None and start < end:
            kwargs["limits"] = (float(start), float(end))
        regions = self._regions(_parameter_value(parameters, "pattern.exclude_regions", "exclude_regions"))
        if regions:
            kwargs["exclude_regions"] = regions

        boolean_fields = {
            "reference_masks_enabled": ("pattern.reference_masks_enabled", "reference_masks_enabled"),
            "include_cu_kbeta": ("pattern.include_cu_kbeta", "include_cu_kbeta"),
            "main_prenudge": ("safeguards.main_prenudge", "main_prenudge"),
            "main_shadow_filter": ("safeguards.main_shadow_filter", "main_shadow_filter"),
            "cleanup_enabled": ("cleanup.cleanup_enabled", "cleanup_enabled"),
            "refine_u_iso": ("cleanup.refine_u_iso", "refine_u_iso"),
            "refine_positions": ("cleanup.refine_positions", "refine_positions"),
            "magnetic_precheck": ("magnetic.magnetic_precheck", "magnetic_precheck"),
            "rapid_show_family_variants": ("rapid.show_family_variants", "show_family_variants"),
            "rapid_final_polish_enabled": ("rapid.final_polish_enabled", "final_polish_enabled"),
        }
        for field, aliases in boolean_fields.items():
            value = _boolean(_parameter_value(parameters, *aliases))
            if value is not None:
                kwargs[field] = value

        number_fields: dict[str, tuple[type[int] | type[float], tuple[str, ...]]] = {
            "background_terms": (int, ("background.background_terms", "background.terms", "background_terms")),
            "magnetic_q_max": (float, ("magnetic.magnetic_q_max", "magnetic.q_max", "magnetic_q_max")),
            "full_max_passes": (int, ("full.max_passes", "full_max_passes")),
            "full_min_phase_percent": (float, ("full.min_phase_percent", "full_min_phase_percent")),
            "full_top_n_ml": (int, ("full.top_n_ml", "full_top_n_ml")),
            "full_nudge_candidates": (int, ("full.nudge_candidates", "full_nudge_candidates")),
            "full_nudge_samples": (int, ("full.nudge_samples", "full_nudge_samples")),
            "full_nudge_representatives": (int, ("full.nudge_representatives", "full_nudge_representatives")),
            "full_compare_candidates": (int, ("full.compare_candidates", "full_compare_candidates")),
            "full_compare_cycles": (int, ("full.compare_cycles", "full_compare_cycles")),
            "rapid_phases_per_hypothesis": (int, ("rapid.phases_per_hypothesis", "phases_per_hypothesis")),
            "rapid_stage_output_limit": (int, ("rapid.stage_output_limit", "stage_output_limit")),
            "rapid_gsas_validation_limit": (int, ("rapid.gsas_validation_limit", "gsas_validation_limit")),
            "rapid_parallel_workers": (int, ("rapid.parallel_workers", "parallel_workers")),
        }
        for field, (kind, aliases) in number_fields.items():
            value = _number(_parameter_value(parameters, *aliases), kind)
            if value is not None:
                kwargs[field] = value
        try:
            config = AnalysisConfig(**kwargs)
            if "run_name" not in kwargs:
                config.run_name = ""
            return config
        except (TypeError, ValueError):
            return None

    def _config_from_job(
        self,
        job: dict[str, Any],
        parameters: Any,
        output_ids: dict[str, str],
    ) -> AnalysisConfig | None:
        dataset_ids = [output_ids.get("resolved_config")]
        dataset_ids.append(_dataset_id(_parameter_value(parameters, "configuration.config_file", "config_file")))
        for dataset_id in dataset_ids:
            if not dataset_id:
                continue
            try:
                payload = self._dataset_document(dataset_id)
                if payload and payload.get("$schema") == "radar-pd-config/v1":
                    config = config_from_contract(payload)
                    run_name = _text(_parameter_value(parameters, "reproducibility.run_name", "run_name"))
                    config.run_name = run_name or self._job_metric(job, "run_name") or ""
                    return config
            except Exception:
                continue
        return self._config_from_parameters(parameters)

    @staticmethod
    def _input_dataset_ids(parameters: Any) -> dict[str, str]:
        aliases = {
            "data": ("data_inputs.input_source.diffraction_pattern", "input_source.data", "diffraction_pattern", "data"),
            "instrument": (
                "data_inputs.input_source.instrument_source.instrument_file",
                "input_source.instrument_source.instrument",
                "instrument_file",
            ),
            "main_cif": ("data_inputs.main_cif", "main_cif"),
            "database_archive": ("library.database.database_archive", "database.database_archive", "database_archive"),
            "event_file": ("data_inputs.input_source.event_file", "input_source.ipts_lookup.event_file", "event_file"),
            "configuration": ("reproducibility.configuration_override.config_file", "configuration.config_file", "config_file"),
        }
        result: dict[str, str] = {}
        for name, paths in aliases.items():
            identifier = _dataset_id(_parameter_value(parameters, *paths))
            if identifier:
                result[name] = identifier
        return result

    def _inputs_from_parameters(
        self,
        parameters: Any,
        public_dataset_ids: dict[str, str] | None = None,
    ) -> InputSelection | None:
        dataset_ids = self._input_dataset_ids(parameters)
        dataset_ids.update(public_dataset_ids or {})
        source_kind = (
            _text(
                _parameter_value(
                    parameters,
                    "data_inputs.input_source.source_kind",
                    "input_source.source_kind",
                    "source_kind",
                )
            )
            or "history"
        ).lower()
        instrument_kind = (
            _text(
                _parameter_value(
                    parameters,
                    "data_inputs.input_source.instrument_source.kind",
                    "input_source.instrument_source.instrument_kind",
                    "instrument_kind",
                )
            )
            or ""
        ).lower()
        kwargs: dict[str, Any] = {
            "main_cif_dataset_id": dataset_ids.get("main_cif"),
            "database_dataset_id": dataset_ids.get("database_archive"),
        }
        config_dataset_id = dataset_ids.get("configuration")
        if config_dataset_id:
            try:
                contract = self._dataset_document(config_dataset_id)
                if contract:
                    kwargs.update(delivery_from_contract(contract))
            except Exception:
                # Scientific run recovery must remain available even if an old
                # configuration dataset cannot be read.
                pass
        if source_kind in {"history", "upload", "galaxy"}:
            kwargs.update(
                source=InputSource.GALAXY,
                data_dataset_id=dataset_ids.get("data"),
                instrument_dataset_id=dataset_ids.get("instrument"),
                use_builtin_cuka=instrument_kind in {"builtin", "builtin_cuka", "cuka"},
            )
        elif source_kind in {"event", "ipts_event"} or dataset_ids.get("event_file"):
            kwargs.update(
                source=InputSource.IPTS_EVENT,
                event_dataset_id=dataset_ids.get("event_file"),
                bank=_text(_parameter_value(parameters, "data_inputs.input_source.bank", "input_source.ipts_lookup.bank", "bank")),
            )
        else:
            kwargs.update(
                source=InputSource.IPTS_MANUAL,
                facility_root=_text(_parameter_value(parameters, "facility_root")) or "/SNS",
                instrument=_text(
                    _parameter_value(
                        parameters,
                        "data_inputs.input_source.instrument",
                        "input_source.ipts_lookup.instrument",
                        "ipts_lookup.instrument",
                    )
                ),
                ipts=_text(_parameter_value(parameters, "ipts_lookup.ipts", "ipts")),
                run_number=_number(_parameter_value(parameters, "run_number"), int),
                bank=_text(_parameter_value(parameters, "bank")),
            )
        try:
            return InputSelection(**kwargs)
        except (TypeError, ValueError):
            return None

    def _dataset_for_input(
        self,
        *,
        path: str | None,
        dataset_id: str | None,
        store: Any,
        label: str,
    ) -> Any | None:
        if dataset_id:
            return self._existing_dataset(dataset_id, store, label)
        if path:
            return self._upload_dataset(path, store, label)
        return None

    def create_submission_snapshot(
        self,
        config: AnalysisConfig,
        inputs: InputSelection,
        *,
        client_revision: int = 0,
        idempotency_token: str | None = None,
    ) -> SubmissionSnapshot:
        """Capture the exact validated form values accepted for one click."""

        return SubmissionSnapshot(
            config=config.model_copy(deep=True),
            inputs=inputs.model_copy(deep=True),
            client_revision=client_revision,
            idempotency_token=idempotency_token or uuid.uuid4().hex,
            display_summary={
                "run_name": config.run_name,
                "mode": config.mode.value,
                "runtime_profile": config.full_profile if config.mode == AnalysisMode.FULL else "rapid",
                "final_refinements": str(config.rapid_gsas_validation_limit),
            },
        )

    def pending_record(self, snapshot: SubmissionSnapshot) -> RunRecord:
        """Return one stable pending record for an idempotency token."""

        with self._lock:
            existing = self._submissions.get(snapshot.idempotency_token)
            if existing is not None:
                return existing
            record = RunRecord(
                uid=f"pending-{snapshot.idempotency_token}",
                name=snapshot.config.run_name,
                mode=snapshot.config.mode,
                history_id=self.history_id,
                status=RunStatus.UPLOADING,
                analysis_status=RunStatus.UPLOADING,
                result_status=ResultStatus.NOT_REQUESTED,
                stage="Validating",
                progress=1,
                config=snapshot.config.model_copy(deep=True),
                inputs=snapshot.inputs.model_copy(deep=True),
                idempotency_token=snapshot.idempotency_token,
                submission=SubmissionProgress(),
            )
            self._submissions[snapshot.idempotency_token] = record
            self._submission_cancel_events[snapshot.idempotency_token] = threading.Event()
            return record

    @staticmethod
    def _submission_label(key: str) -> tuple[SubmissionPhase, str]:
        return {
            "configuration": (SubmissionPhase.UPLOADING_CONFIGURATION, "Uploading configuration"),
            "data": (SubmissionPhase.UPLOADING_DATA, "Uploading diffraction data"),
            "instrument": (SubmissionPhase.UPLOADING_INSTRUMENT, "Uploading instrument profile"),
            "event_file": (SubmissionPhase.UPLOADING_DATA, "Uploading NeXus event data"),
            "main_cif": (SubmissionPhase.UPLOADING_OPTIONAL, "Uploading main-phase CIF"),
            "database_archive": (SubmissionPhase.UPLOADING_OPTIONAL, "Uploading candidate library"),
        }[key]

    def _emit_submission_progress(
        self,
        record: RunRecord,
        phase: SubmissionPhase,
        label: str,
        completed: int,
        total: int,
        callback: Callable[[RunRecord], None] | None,
    ) -> None:
        started = record.submission.started_utc if record.submission else datetime.now(timezone.utc).isoformat()
        try:
            elapsed = max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(started)).total_seconds())
        except ValueError:
            elapsed = 0.0
        record.submission = SubmissionProgress(
            phase=phase,
            label=label,
            completed_items=completed,
            total_items=total,
            started_utc=started,
            elapsed_seconds=elapsed,
            galaxy_job_id=record.galaxy_job_id,
        )
        record.stage = label
        record.progress = min(5 + int(85 * completed / max(total, 1)), 90)
        record.updated_utc = datetime.now(timezone.utc).isoformat()
        if callback is not None:
            callback(record)

    def _upload_one(self, key: str, path: str, label: str) -> tuple[str, str]:
        # nova-galaxy connections are not thread safe. Each upload owns a
        # short-lived connection, allowing independent files to proceed in
        # parallel without sharing a Bioblend client or Datastore.
        with self._store() as store:
            dataset = self._upload_dataset(path, store, label)
            return key, str(dataset.id)

    def _import_one(self, key: str, uri: str, label: str) -> tuple[str, str]:
        return key, self._import_remote_dataset(uri, label)

    def _prepare_datasets(
        self,
        snapshot: SubmissionSnapshot,
        record: RunRecord,
        config_path: Path,
        callback: Callable[[RunRecord], None] | None,
    ) -> dict[str, str]:
        inputs = snapshot.inputs
        prepared = dict(record.prepared_dataset_ids)
        existing = {
            "data": inputs.data_dataset_id,
            "instrument": inputs.instrument_dataset_id,
            "main_cif": inputs.main_cif_dataset_id,
            "database_archive": inputs.database_dataset_id,
            "event_file": inputs.event_dataset_id,
        }
        prepared.update({key: str(value) for key, value in existing.items() if value})
        uploads: dict[str, tuple[str, str]] = {"configuration": (str(config_path), "configuration")}
        imports: dict[str, tuple[str, str]] = {}
        for key, path, label in (
            ("data", inputs.data_path, "diffraction data"),
            ("instrument", inputs.instrument_path, "instrument profile"),
            ("main_cif", inputs.main_cif_path, "main phase CIF"),
            ("database_archive", inputs.database_archive_path, "candidate library"),
            ("event_file", inputs.event_file_path, "NeXus event file"),
        ):
            if path and key not in prepared:
                uploads[key] = (path, label)
        for key, uri, label in (
            ("data", inputs.data_remote_uri, "diffraction data"),
            ("instrument", inputs.instrument_remote_uri, "instrument profile"),
            ("main_cif", inputs.main_cif_remote_uri, "main phase CIF"),
        ):
            if uri and key not in prepared:
                imports[key] = (uri, label)
        if "configuration" in prepared:
            uploads.pop("configuration", None)

        total = len(uploads) + len(imports) + len(prepared)
        completed = len(prepared)
        progress_lock = threading.Lock()
        if not uploads and not imports:
            return prepared
        first_key = next(
            (
                key
                for key in ("configuration", "data", "event_file", "instrument", "main_cif", "database_archive")
                if key in uploads or key in imports
            ),
            next(iter(uploads or imports)),
        )
        phase, label = self._submission_label(first_key)
        self._emit_submission_progress(record, phase, label, completed, total, callback)
        with ThreadPoolExecutor(max_workers=min(4, len(uploads) + len(imports)), thread_name_prefix="radar-input") as pool:
            futures = {
                pool.submit(self._upload_one, key, path, label): key
                for key, (path, label) in uploads.items()
            }
            futures.update(
                {
                    pool.submit(self._import_one, key, uri, label): key
                    for key, (uri, label) in imports.items()
                }
            )
            failures: list[Exception] = []
            for future in as_completed(futures):
                try:
                    key, dataset_id = future.result()
                except Exception as exc:
                    failures.append(exc)
                    continue
                with progress_lock:
                    prepared[key] = dataset_id
                    record.prepared_dataset_ids[key] = dataset_id
                    completed += 1
                    phase, label = self._submission_label(key)
                    self._emit_submission_progress(record, phase, label, completed, total, callback)
            if failures:
                raise failures[0]
        return prepared

    def _prepared_parameters(
        self,
        store: Any,
        snapshot: SubmissionSnapshot,
        prepared: dict[str, str],
    ) -> tuple[Any, InputSelection]:
        from nova.galaxy import Parameters

        config = snapshot.config
        inputs = snapshot.inputs.model_copy(deep=True)
        params = Parameters()
        params.add_input(name="measurement|radiation", value=config.radiation.value)
        params.add_input(name="measurement|instrument_mode", value=config.instrument_mode)
        params.add_input(name="chemistry|sample_elements", value=", ".join(config.sample_elements))
        params.add_input(name="chemistry|environment_elements", value=", ".join(config.environment_elements))
        params.add_input(name="analysis|strategy|analysis_mode", value=config.mode.value)
        params.add_input(name="reproducibility|configuration_override|config_kind", value="existing")
        config_ds = self._existing_dataset(prepared["configuration"], store, "configuration")
        params.add_input(name="reproducibility|configuration_override|config_file", value=config_ds)

        if inputs.source in {
            InputSource.UPLOAD,
            InputSource.GALAXY,
            InputSource.GALAXY_REMOTE,
            InputSource.IPTS_BROWSER,
        }:
            params.add_input(name="data_inputs|input_source|source_kind", value="history")
            if "data" not in prepared:
                raise ValueError("A diffraction pattern is required")
            data = self._existing_dataset(prepared["data"], store, "diffraction data")
            inputs.data_dataset_id = prepared["data"]
            params.add_input(name="data_inputs|input_source|diffraction_pattern", value=data)
            if inputs.use_builtin_cuka:
                params.add_input(name="data_inputs|input_source|instrument_source|kind", value="builtin_cuka")
            else:
                if "instrument" not in prepared:
                    raise ValueError("An instrument profile is required")
                instrument = self._existing_dataset(prepared["instrument"], store, "instrument profile")
                inputs.instrument_dataset_id = prepared["instrument"]
                params.add_input(name="data_inputs|input_source|instrument_source|kind", value="uploaded")
                params.add_input(name="data_inputs|input_source|instrument_source|instrument_file", value=instrument)
        else:
            params.add_input(name="data_inputs|input_source|source_kind", value=inputs.source.value)
            if inputs.source == InputSource.IPTS_EVENT:
                if "event_file" not in prepared:
                    raise ValueError("A NeXus event file is required")
                event = self._existing_dataset(prepared["event_file"], store, "NeXus event file")
                inputs.event_dataset_id = prepared["event_file"]
                params.add_input(name="data_inputs|input_source|event_file", value=event)
                params.add_input(name="data_inputs|input_source|bank", value=inputs.bank)
            else:
                params.add_input(name="data_inputs|input_source|instrument", value=inputs.instrument)
                params.add_input(name="data_inputs|input_source|ipts", value=inputs.ipts)
                params.add_input(name="data_inputs|input_source|run_number", value=inputs.run_number)
                params.add_input(name="data_inputs|input_source|bank", value=inputs.bank)

        if "main_cif" in prepared:
            main_cif = self._existing_dataset(prepared["main_cif"], store, "main phase CIF")
            inputs.main_cif_dataset_id = prepared["main_cif"]
            params.add_input(name="data_inputs|main_cif", value=main_cif)
        if "database_archive" in prepared:
            database = self._existing_dataset(prepared["database_archive"], store, "candidate library")
            inputs.database_dataset_id = prepared["database_archive"]
            params.add_input(name="library|database|database_kind", value="custom")
            params.add_input(name="library|database|database_archive", value=database)
        else:
            params.add_input(name="library|database|database_kind", value="builtin")
        params.add_input(name="reproducibility|run_name", value=config.run_name)
        return params, inputs

    def submit(self, config: AnalysisConfig, inputs: InputSelection) -> RunRecord:
        """Compatibility wrapper for a fully tracked immutable submission."""

        return self.submit_snapshot(self.create_submission_snapshot(config, inputs))

    def submit_snapshot(
        self,
        snapshot: SubmissionSnapshot,
        *,
        callback: Callable[[RunRecord], None] | None = None,
    ) -> RunRecord:
        """Prepare inputs concurrently and launch Analyze exactly once."""

        record = self.pending_record(snapshot)
        with self._lock:
            if record.galaxy_job_id or record.analysis_status in {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.OK}:
                return record
            if snapshot.idempotency_token in self._active_submission_tokens:
                return record
            self._active_submission_tokens.add(snapshot.idempotency_token)
        cancel_event = self._submission_cancel_events[snapshot.idempotency_token]
        config = snapshot.config
        inputs = snapshot.inputs
        record.status = RunStatus.UPLOADING
        record.analysis_status = RunStatus.UPLOADING
        record.message = ""
        self._emit_submission_progress(record, SubmissionPhase.VALIDATING, "Validating", 0, 1, callback)
        try:
            if inputs.use_builtin_cuka and (config.radiation.value != "xray" or config.instrument_mode == "tof"):
                raise ValueError("The built-in Cu K-alpha profile is only valid for X-ray CW data")
            work_dir = self.output_root / "submissions" / snapshot.idempotency_token
            work_dir.mkdir(parents=True, exist_ok=True)
            config_path = dump_configuration(
                config,
                work_dir / "radar_pd_config.yaml",
                inputs=snapshot.inputs,
            )
            prepared = self._prepare_datasets(snapshot, record, config_path, callback)
            if cancel_event.is_set() or record.cancel_requested:
                record.status = RunStatus.CANCELLED
                record.analysis_status = RunStatus.CANCELLED
                self._emit_submission_progress(
                    record, SubmissionPhase.CANCELLED, "Submission cancelled before Analyze", len(prepared), len(prepared), callback
                )
                return record
            self._emit_submission_progress(
                record, SubmissionPhase.WAITING_FOR_GALAXY, "Waiting for Galaxy", len(prepared), len(prepared), callback
            )
            with self._store() as store:
                params, submitted_inputs = self._prepared_parameters(store, snapshot, prepared)
                uid, tool = self._submit_with_retry(
                    store,
                    params,
                    run_name=config.run_name,
                    config_dataset_id=prepared.get("configuration"),
                )
            with self._lock:
                self._tools[uid] = tool
            record.galaxy_job_id = uid
            record.status = RunStatus.QUEUED
            record.analysis_status = RunStatus.QUEUED
            record.stage = "Waiting for compute"
            record.progress = 3
            record.input_dataset_ids = dict(prepared)
            record.prepared_dataset_ids = dict(prepared)
            record.config = config.model_copy(deep=True)
            record.inputs = submitted_inputs
            record.submission = SubmissionProgress(
                phase=SubmissionPhase.ACKNOWLEDGED,
                label="Galaxy job created",
                completed_items=len(prepared),
                total_items=len(prepared),
                started_utc=record.submission.started_utc if record.submission else datetime.now(timezone.utc).isoformat(),
                galaxy_job_id=uid,
            )
            if callback is not None:
                callback(record)
            return record
        except Exception as exc:
            record.status = RunStatus.ERROR
            record.analysis_status = RunStatus.ERROR
            record.message = str(exc)
            self._emit_submission_progress(record, SubmissionPhase.ERROR, "Submission failed", 0, 1, callback)
            raise
        finally:
            with self._lock:
                self._active_submission_tokens.discard(snapshot.idempotency_token)

    def cancel_pending_submission(self, token: str) -> RunRecord | None:
        """Prevent a prepared pending submission from creating an Analyze job."""

        with self._lock:
            event = self._submission_cancel_events.get(token)
            record = self._submissions.get(token)
            if event is not None:
                event.set()
            if record is not None and not record.galaxy_job_id:
                record.cancel_requested = True
                record.stage = "Cancelling after current uploads"
            return record

    def create_dataset_collection(self, name: str, dataset_ids: list[str]) -> str:
        """Create a reusable Galaxy list collection from existing HDAs."""

        if not dataset_ids:
            raise ValueError("Select at least one Galaxy dataset")
        response = requests.post(
            f"{self.galaxy_url}/api/dataset_collections",
            headers={**self._headers, "content-type": "application/json"},
            json={
                "history_id": self.history_id,
                "collection_type": "list",
                "name": name,
                "element_identifiers": [
                    {"id": dataset_id, "src": "hda", "name": f"item_{index:03d}"}
                    for index, dataset_id in enumerate(dataset_ids, start=1)
                ],
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        identifier = payload.get("id") if isinstance(payload, dict) else None
        if not identifier:
            raise RuntimeError("Galaxy did not return a dataset collection identifier")
        return str(identifier)

    def submit_utility(
        self,
        *,
        tool_id: str,
        name: str,
        inputs: dict[str, Any],
        associated_run_uid: str | None = None,
    ) -> UtilityActionRecord:
        """Submit one deployed RADAR-PD companion tool with typed references."""

        from nova.galaxy import Parameters, Tool

        with self._store() as store:
            params = Parameters()
            serialized_inputs: dict[str, Any] = {}
            for parameter_name, value in inputs.items():
                resolved = value
                if (
                    isinstance(value, list)
                    and value
                    and all(isinstance(item, dict) and item.get("dataset_id") for item in value)
                ):
                    dataset_ids = [str(item["dataset_id"]) for item in value]
                    # Galaxy multiple-data parameters require a list of native
                    # HDA references. Keep this as an ordinary Parameters value;
                    # nova-galaxy's Dataset wrapper is only suitable for uploads.
                    resolved = [{"src": "hda", "id": dataset_id} for dataset_id in dataset_ids]
                    serialized_inputs[parameter_name] = [
                        {"dataset_id": dataset_id} for dataset_id in dataset_ids
                    ]
                elif isinstance(value, dict) and value.get("dataset_id"):
                    dataset_id = str(value["dataset_id"])
                    # Utility tools receive existing History datasets, not new
                    # uploads. Use Galaxy's native HDA reference here. The
                    # nova-galaxy Dataset wrapper serializes an existing dataset
                    # as a ``values`` list, which breaks scalar data parameters
                    # nested inside conditionals (for example cif_archive).
                    resolved = {"src": "hda", "id": dataset_id}
                    serialized_inputs[parameter_name] = {"dataset_id": dataset_id}
                elif isinstance(value, dict) and value.get("collection_id"):
                    collection_id = str(value["collection_id"])
                    # nova-galaxy only special-cases Dataset inputs. Passing its
                    # DatasetCollection object through Parameters leaves that object
                    # in the run_tool JSON payload and fails at serialization time.
                    # Galaxy's public tool API accepts the native HDCA reference.
                    resolved = {"src": "hdca", "id": collection_id}
                    serialized_inputs[parameter_name] = {"collection_id": collection_id}
                else:
                    serialized_inputs[parameter_name] = value
                params.add_input(name=parameter_name, value=resolved)
            tool = Tool(id=tool_id)
            tool.run(store, params, wait=False)
            uid = self._wait_for_submission(tool)
            if not uid:
                raise RuntimeError(f"Galaxy did not acknowledge {name}")
            with self._lock:
                self._tools[uid] = tool
        return UtilityActionRecord(
            uid=f"utility-{uuid.uuid4().hex}",
            tool_id=tool_id,
            name=name,
            associated_run_uid=associated_run_uid,
            inputs=serialized_inputs,
            galaxy_job_id=uid,
            status=RunStatus.QUEUED,
        )

    def submit_results_export(
        self,
        record: RunRecord,
        *,
        archive_dataset_id: str,
    ) -> tuple[UtilityActionRecord, str]:
        """Delegate one completed result archive to NDIP's authenticated exporter.

        The export tool is routed by NDIP to its UCAMS-aware analysis-cluster
        execution destination. RADAR-PD supplies only an HDA reference and a
        validated facility path; it never handles a UCAMS password, OIDC token,
        SSH key, or host credential.
        """

        inputs = record.inputs
        if inputs is None:
            raise ValueError("The run has no persisted input selection")
        run_token = re.sub(r"[^A-Za-z0-9_. -]+", "-", record.name).strip(" .-") or "radar-pd-run"
        suffix = (record.galaxy_job_id or record.uid).replace("pending-", "")[:8]
        destination = build_facility_export_path(
            inputs.facility_root,
            str(inputs.instrument or ""),
            str(inputs.ipts or ""),
            str(inputs.publish_directory or ""),
            f"{run_token}-{suffix}",
        )
        action = self.submit_utility(
            tool_id=EXPORT_DATASETS_TOOL_ID,
            name="Publish RADAR-PD results to IPTS",
            inputs={
                "series_0|input_mode|input_mode_collection": False,
                "series_0|input_mode|input": {"dataset_id": archive_dataset_id},
                "series_0|input_mode|export_path": destination,
            },
            associated_run_uid=record.uid,
        )
        return action, destination

    def refresh_utility(self, action: UtilityActionRecord) -> UtilityActionRecord:
        if not action.galaxy_job_id:
            return action
        job = self._job_details(action.galaxy_job_id)
        action.status = normalize_status(job.get("state"))
        action.outputs.update(self._job_output_ids(job))
        stderr = str(job.get("stderr") or job.get("tool_stderr") or job.get("job_stderr") or "")
        if action.status == RunStatus.ERROR:
            action.message = stderr[-2000:] or "Galaxy reported a utility-tool error"
        if action.tool_id == RESULT_EXPLORER_TOOL_ID and action.status in {RunStatus.RUNNING, RunStatus.OK}:
            try:
                response = requests.get(
                    f"{self.galaxy_url}/api/entry_points",
                    headers=self._headers,
                    params={"job_id": action.galaxy_job_id},
                    timeout=30,
                )
                response.raise_for_status()
                entries = response.json()
                if isinstance(entries, list) and entries:
                    entry = entries[0]
                    action.entrypoint_id = str(entry.get("id") or entry.get("entry_point_id") or "") or None
                    target = entry.get("target") or entry.get("url") or entry.get("access_url")
                    active = bool(entry.get("active"))
                    if active and target:
                        action.outputs["launch_url"] = str(target)
                        action.status = RunStatus.OK
                        action.message = "Result Explorer is ready"
                    elif action.status == RunStatus.OK and not active:
                        action.status = RunStatus.ERROR
                        action.message = stderr[-2000:] or "Result Explorer stopped before its NDIP entry point became active"
            except Exception:
                pass
        action.updated_utc = datetime.now(timezone.utc).isoformat()
        return action

    def save_configuration(self, config: AnalysisConfig, *, associated_run_uid: str | None = None) -> UtilityActionRecord:
        """Persist the exact validated radar-pd-config/v1 document in History."""

        work_dir = self.output_root / "utilities" / uuid.uuid4().hex
        config_path = dump_configuration(config, work_dir / f"{config.run_name}_radar_pd_config.yaml")
        with self._store() as store:
            dataset = self._upload_dataset(str(config_path), store, "configuration")
        return UtilityActionRecord(
            uid=f"utility-{uuid.uuid4().hex}",
            tool_id="neutrons_radar_pd_configure_prototype",
            name="Save reusable configuration",
            associated_run_uid=associated_run_uid,
            inputs={"schema": "radar-pd-config/v1"},
            status=RunStatus.OK,
            outputs={"config_output": str(dataset.id)},
            message="Validated configuration saved to Galaxy History",
        )

    def collection_elements(self, collection_id: str) -> list[dict[str, str]]:
        response = requests.get(
            f"{self.galaxy_url}/api/dataset_collections/{collection_id}",
            headers=self._headers,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        elements = payload.get("elements") if isinstance(payload, dict) else []
        result: list[dict[str, str]] = []
        for element in elements or []:
            obj = element.get("object") or {}
            identifier = obj.get("id") or element.get("id")
            if identifier:
                result.append(
                    {
                        "id": str(identifier),
                        "name": str(element.get("element_identifier") or obj.get("name") or "artifact"),
                    }
                )
        return result

    def _find_acknowledged_job(self, run_name: str, config_dataset_id: str | None) -> str | None:
        """Find a just-created job after the client lost its acknowledgement."""

        response = requests.get(
            f"{self.galaxy_url}/api/jobs",
            headers=self._headers,
            params={
                "history_id": self.history_id,
                "tool_id": ANALYZE_TOOL_ID,
                "order": "update_time-dsc",
                "limit": 30,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        for summary in payload if isinstance(payload, list) else []:
            uid = str(summary.get("id") or "")
            if not uid:
                continue
            try:
                job = self._job_details(uid, summary)
            except Exception:
                continue
            parameters = self._job_parameters(job)
            candidate_name = _text(_parameter_value(parameters, "reproducibility.run_name", "run_name"))
            candidate_config = self._public_input_dataset_ids(job).get("configuration") or _dataset_id(
                _parameter_value(
                    parameters,
                    "reproducibility.configuration_override.config_file",
                    "configuration_override.config_file",
                    "config_file",
                )
            )
            if candidate_name == run_name and (not config_dataset_id or candidate_config == config_dataset_id):
                return uid
        return None

    def _submit_with_retry(
        self,
        store: Any,
        params: Any,
        *,
        attempts: int = 3,
        run_name: str | None = None,
        config_dataset_id: str | None = None,
    ) -> tuple[str, Any]:
        """Start the Analyze tool run, retrying on transient Galaxy submission failures.

        Galaxy submissions can fail before creating a job because of a
        transient transport or service error. A failed attempt never creates
        a Galaxy job, so retrying is safe and does not risk duplicate
        submissions. Deterministic input failures still surface after the
        bounded retry count.
        """

        from nova.galaxy import Tool

        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            tool = Tool(id=ANALYZE_TOOL_ID)
            try:
                tool.run(store, params, wait=False)
                uid = self._wait_for_submission(tool)
                if not uid:
                    raise RuntimeError("Galaxy accepted the request but did not return a job identifier")
            except Exception as exc:
                last_error = exc
                try:
                    recovered_uid = self._find_acknowledged_job(run_name, config_dataset_id) if run_name else None
                except Exception:
                    recovered_uid = None
                if recovered_uid:
                    recovered = Tool(id=ANALYZE_TOOL_ID)
                    recovered.assign_id(recovered_uid, store)
                    return recovered_uid, recovered
                if attempt < attempts:
                    time.sleep(2.0 * attempt)
                    continue
                raise
            return uid, tool
        raise last_error or RuntimeError("Galaxy submission failed for an unknown reason")

    @staticmethod
    def _wait_for_submission(tool: Any) -> str:
        """Wait only for Galaxy to acknowledge an async NOVA submission.

        NOVA starts ``Tool.run(wait=False)`` in a background thread. The tool
        UID is assigned after Galaxy accepts ``run_tool``, so reading it on the
        next line races that thread. This wait ends as soon as the UID exists;
        it does not wait for the scientific job to finish.
        """

        deadline = time.monotonic() + SUBMISSION_ACK_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            uid = tool.get_uid()
            if uid:
                return str(uid)
            try:
                status = tool.get_full_status()
            except Exception:
                status = None
            state = getattr(status, "state", None)
            state_value = str(getattr(state, "value", state) or "").lower()
            if state_value in {"error", "canceled", "cancelled"}:
                details = getattr(status, "details", None) or {}
                message = details.get("message") if isinstance(details, dict) else None
                raise RuntimeError(message or f"Galaxy submission failed with status {state_value}")
            time.sleep(0.1)
        return ""

    def _recover_tool(self, uid: str) -> Any:
        from nova.galaxy import Tool

        with self._lock:
            if uid in self._tools:
                return self._tools[uid]
        with self._store() as store:
            tool = Tool(id=ANALYZE_TOOL_ID)
            tool.assign_id(uid, store)
            with self._lock:
                self._tools[uid] = tool
            return tool

    def refresh(self, record: RunRecord) -> RunRecord:
        # Galaxy's REST job record is authoritative for both recovered and
        # newly submitted runs. Some nova-galaxy releases leave their cached
        # Job at QUEUED after Galaxy reports the terminal ``ok`` state. Never
        # fall back to that stale object: a temporary REST error must leave the
        # last known state intact instead of regressing an OK run to QUEUED.
        if not record.galaxy_job_id:
            return record
        job = self._job_details(record.galaxy_job_id)
        status = normalize_status(job.get("state"))
        stdout = str(job.get("stdout") or job.get("tool_stdout") or job.get("job_stdout") or "")
        stderr = str(job.get("stderr") or job.get("tool_stderr") or job.get("job_stderr") or "")
        parameters = self._job_parameters(job)
        output_ids = self._job_output_ids(job)
        public_input_ids = self._public_input_dataset_ids(job)
        record.output_dataset_ids.update(output_ids)
        record.input_dataset_ids.update(self._input_dataset_ids(parameters))
        record.input_dataset_ids.update(public_input_ids)
        recovered_config = self._config_from_job(job, parameters, output_ids)
        if recovered_config is not None:
            record.config = recovered_config
            record.mode = recovered_config.mode
            if recovered_config.run_name:
                record.name = recovered_config.run_name
        recovered_inputs = self._inputs_from_parameters(parameters, public_input_ids)
        if recovered_inputs is not None:
            record.inputs = self._merge_recovered_inputs(record.inputs, recovered_inputs)
        stage, progress = stage_from_console(stdout, status)
        record.status = status
        record.analysis_status = status
        record.stage = stage
        record.progress = progress
        record.console_tail = stdout[-12000:]
        record.updated_utc = datetime.now(timezone.utc).isoformat()
        if status == RunStatus.ERROR:
            record.message = (stderr or stdout or "Galaxy reported an error")[-4000:]
        return record

    @staticmethod
    def _merge_recovered_inputs(
        current: InputSelection | None,
        recovered: InputSelection,
    ) -> InputSelection:
        """Merge runtime Galaxy IDs without losing click-time facility intent."""

        if current is None:
            return recovered
        merged = current.model_copy(deep=True)
        for field in (
            "data_dataset_id",
            "instrument_dataset_id",
            "main_cif_dataset_id",
            "database_dataset_id",
            "event_dataset_id",
        ):
            value = getattr(recovered, field)
            if value:
                setattr(merged, field, value)
        return merged

    def cancel(self, uid: str) -> None:
        self._recover_tool(uid).stop()

    def recent_runs(self, *, limit: int = 50) -> list[RunRecord]:
        """Recover RADAR-PD Analyze jobs directly from the active Galaxy history."""

        response = requests.get(
            f"{self.galaxy_url}/api/jobs",
            headers=self._headers,
            params={"history_id": self.history_id, "tool_id": ANALYZE_TOOL_ID, "order": "update_time-dsc", "limit": limit},
            timeout=30,
        )
        response.raise_for_status()
        records: list[RunRecord] = []
        for summary in response.json():
            uid = str(summary.get("id") or "")
            if not uid:
                continue
            try:
                job = self._job_details(uid, summary)
            except Exception:
                job = dict(summary)
            state = normalize_status(job.get("state"))
            parameters = self._job_parameters(job)
            output_ids = self._job_output_ids(job)
            public_input_ids = self._public_input_dataset_ids(job)
            config = self._config_from_job(job, parameters, output_ids)
            inputs = self._inputs_from_parameters(parameters, public_input_ids)
            parameter_name = _text(_parameter_value(parameters, "reproducibility.run_name", "run_name"))
            run_name = config.run_name if config is not None and config.run_name else parameter_name
            run_name = run_name or self._job_metric(job, "run_name")
            command = str(job.get("command_line") or job.get("command") or "")
            if not run_name:
                match = re.search(r"--run-name\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))", command)
                run_name = next((group for group in match.groups() if group), uid) if match else uid
            mode_text = _text(
                _parameter_value(parameters, "analysis.strategy.analysis_mode", "analysis.analysis_mode", "analysis_mode")
            )
            if config is not None:
                mode = config.mode
            elif mode_text and mode_text.lower() in {item.value for item in AnalysisMode}:
                mode = AnalysisMode(mode_text.lower())
            else:
                mode = AnalysisMode.RAPID if "rapid" in command.lower() else AnalysisMode.FULL
            created = str(job.get("create_time") or job.get("created_utc") or datetime.now(timezone.utc).isoformat())
            updated = str(job.get("update_time") or job.get("updated_utc") or created)
            records.append(
                RunRecord(
                    uid=uid,
                    galaxy_job_id=uid,
                    name=run_name,
                    mode=mode,
                    history_id=self.history_id,
                    status=state,
                    analysis_status=state,
                    stage=stage_from_console("", state)[0],
                    progress=stage_from_console("", state)[1],
                    created_utc=created,
                    updated_utc=updated,
                    input_dataset_ids={**self._input_dataset_ids(parameters), **public_input_ids},
                    output_dataset_ids=output_ids,
                    config=config,
                    inputs=inputs,
                )
            )
        return records

    @staticmethod
    def _read_cache_manifest(destination: Path) -> CacheManifest | None:
        path = destination / ".radar-pd-cache.json"
        if not path.is_file():
            return None
        try:
            return CacheManifest.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _current_cache_manifest(self, record: RunRecord) -> CacheManifest | None:
        job_id = record.galaxy_job_id
        archive_id = record.output_dataset_ids.get("results_archive")
        if not job_id or not archive_id:
            return None
        try:
            metadata = self._dataset_metadata(archive_id)
        except Exception:
            metadata = {}
        raw_size = metadata.get("file_size", metadata.get("size"))
        try:
            size = int(raw_size) if raw_size is not None else None
        except (TypeError, ValueError):
            size = None
        return CacheManifest(
            job_id=job_id,
            archive_dataset_id=archive_id,
            archive_size=size,
            archive_update_time=str(metadata.get("update_time") or metadata.get("updated_utc") or "") or None,
        )

    @staticmethod
    def _cache_is_current(cached: CacheManifest | None, current: CacheManifest | None, destination: Path) -> bool:
        if cached is None or current is None or not destination.is_dir():
            return False
        return (
            cached.adapter_version == current.adapter_version
            and cached.job_id == current.job_id
            and cached.archive_dataset_id == current.archive_dataset_id
            and (current.archive_size is None or cached.archive_size == current.archive_size)
            and (not current.archive_update_time or cached.archive_update_time == current.archive_update_time)
            and any(destination.rglob("*.plotdata.json"))
        )

    def collect_results(self, record: RunRecord, *, force: bool = False) -> RunRecord:
        """Collect the canonical archive without changing scientific job state on failure."""

        job_id = record.galaxy_job_id or record.uid
        destination = self.output_root / "runs" / job_id
        staging = destination.parent / f".{job_id}-{threading.get_ident()}.partial"
        # Result collection only begins after Galaxy has completed Analyze.
        # Old recovered records may not carry the new analysis_status field.
        record.analysis_status = RunStatus.OK
        record.status = RunStatus.OK
        record.result_status = ResultStatus.COLLECTING
        try:
            try:
                job = self._job_details(job_id)
                record.output_dataset_ids.update(self._job_output_ids(job))
            except Exception:
                pass

            current_manifest = self._current_cache_manifest(record)
            cached_manifest = self._read_cache_manifest(destination)
            if not force and self._cache_is_current(cached_manifest, current_manifest, destination):
                record.cache_manifest = cached_manifest
                record.output_dir = str(destination)
                record.result_status = ResultStatus.READY
                record.status = record.analysis_status or RunStatus.OK
                record.stage = "Results ready"
                record.progress = 100
                record.message = ""
                return record

            outputs = None
            try:
                tool = self._recover_tool(job_id)
                outputs = tool.get_results()
            except Exception:
                # Recovered sessions should use durable dataset IDs. The Tool
                # object is retained only as a fallback for older Galaxy jobs.
                outputs = None
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            output_ids: dict[str, str] = {}
            downloaded = 0
            usable_downloads = 0
            failures: list[str] = []
            archive_path: Path | None = None
            archive_extracted = False
            for name in (
                "report",
                "summary",
                "state",
                "resolved_config",
                "input_manifest",
                "gpx_index",
                "results_archive",
                "input_resolution_metadata",
                "console_output",
            ):
                suffix = {
                    "report": ".html",
                    "summary": ".json",
                    "state": ".json",
                    "resolved_config": ".yaml",
                    "input_manifest": ".json",
                    "gpx_index": ".json",
                    "results_archive": ".zip",
                    "input_resolution_metadata": ".json",
                    "console_output": ".txt",
                }[name]
                target = staging / f"{name}{suffix}"
                dataset_id = record.output_dataset_ids.get(name)
                dataset = None
                if not dataset_id and outputs is not None:
                    try:
                        dataset = outputs.get_dataset(name)
                    except Exception:
                        continue
                try:
                    if dataset_id:
                        self._download_dataset(dataset_id, target)
                        output_ids[name] = dataset_id
                    elif dataset is not None:
                        dataset.download(str(target))
                        output_ids[name] = str(getattr(dataset, "id", ""))
                    else:
                        continue
                except Exception as exc:
                    target.unlink(missing_ok=True)
                    failures.append(f"{name}: {exc}")
                    continue
                downloaded += 1
                if name == "results_archive":
                    archive_path = target
                else:
                    usable_downloads += 1
            if archive_path is not None:
                try:
                    _extract_results_archive(archive_path, staging)
                    usable_downloads += 1
                    archive_extracted = True
                except Exception as exc:
                    failures.append(f"results_archive extraction: {exc}")
            # The archive is canonical and preserves plot JSON/NPZ pairs. Only
            # fall back to named collection downloads for legacy jobs that did
            # not publish an extractable archive.
            for name in (() if archive_extracted else ("plots", "tables", "phases", "gpx_projects", "diagnostics")):
                if outputs is None:
                    continue
                try:
                    collection = outputs.get_collection(name)
                except Exception:
                    continue
                target = staging / name
                target.mkdir(exist_ok=True)
                try:
                    collection.download(str(target))
                except Exception as exc:
                    shutil.rmtree(target, ignore_errors=True)
                    failures.append(f"{name}: {exc}")
                    continue
                downloaded += 1
                usable_downloads += 1
                output_ids[name] = str(getattr(collection, "id", ""))
            if not downloaded or not usable_downloads:
                details = "; ".join(failures)
                message = (
                    "Galaxy did not return the declared result outputs"
                    if outputs is None and not record.output_dataset_ids
                    else "Galaxy returned no downloadable RADAR-PD result artifacts"
                )
                raise RuntimeError(f"{message}: {details}" if details else message)
            if destination.exists():
                shutil.rmtree(destination)
            manifest = current_manifest or CacheManifest(
                job_id=job_id,
                archive_dataset_id=record.output_dataset_ids.get("results_archive", "legacy-no-archive"),
            )
            (staging / ".radar-pd-cache.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            staging.replace(destination)
            record.output_dataset_ids.update(output_ids)
            record.output_dir = str(destination)
            record.status = record.analysis_status or RunStatus.OK
            record.analysis_status = record.status
            record.result_status = ResultStatus.READY
            record.cache_manifest = manifest
            record.stage = "Results ready"
            record.progress = 100
            record.message = (
                f"Results loaded; {len(failures)} duplicate or optional Galaxy output(s) were unavailable."
                if failures
                else ""
            )
        except Exception as exc:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            record.status = record.analysis_status or RunStatus.OK
            record.result_status = ResultStatus.ERROR
            record.stage = "Analysis complete; results unavailable"
            record.progress = 100
            record.message = f"Analysis finished, but result download failed: {exc}"
        record.updated_utc = datetime.now(timezone.utc).isoformat()
        return record

    def result_payload(self, record: RunRecord) -> dict[str, Any]:
        root = record.local_output_dir
        if root is None:
            return {}
        payload: dict[str, Any] = {"artifacts": []}
        for key in ("summary", "state", "gpx_index", "input_manifest"):
            candidates = (root / f"{key}.json", root / "ndip" / f"{key}.json")
            path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
            if path.is_file():
                try:
                    payload[key] = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    payload[key] = {}
        for path in sorted(root.rglob("*")):
            if path.is_file():
                payload["artifacts"].append(
                    {
                        "name": path.name,
                        "kind": path.parent.name if path.parent != root else "result",
                        "path": str(path),
                        "size": path.stat().st_size,
                    }
                )
        return payload

    def monitor(
        self,
        record: RunRecord,
        callback: Callable[[RunRecord], None],
        *,
        poll_seconds: float = 5.0,
    ) -> None:
        """Blocking monitor intended for a background thread."""

        import time

        while record.status not in {RunStatus.OK, RunStatus.ERROR, RunStatus.CANCELLED}:
            try:
                record = self.refresh(record)
            except Exception as exc:
                record.message = str(exc)
            callback(record)
            if record.status in {RunStatus.OK, RunStatus.ERROR, RunStatus.CANCELLED}:
                break
            time.sleep(poll_seconds)
        if record.status == RunStatus.OK:
            try:
                record = self.collect_results(record)
            except Exception as exc:
                record.status = RunStatus.ERROR
                record.stage = "Result download failed"
                record.progress = 100
                record.message = f"Analysis finished, but result download failed: {exc}"
            callback(record)
