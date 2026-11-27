"""Bounded, read-only fit previews kept separate from complete result caches."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import stat
import threading
from typing import Any, Mapping
import uuid
import zipfile

import numpy as np
import requests

from .models import RunStatus
from .planning_bundle import read_json
from .results import (
    _as_float, _plot_payload_has_arrays, _plot_stage, _primary_plot_index,
    load_plot_with_fallback,
)
from .scan_interpretation import MAX_BYTES as MAX_OBSERVATION_BYTES
from .tool_ids import tool_id
from .result_loading import measure, subject, submit_context, transfer_slot


PREVIEW_RECEIPT = ".radar-pd-preview.json"
MAX_SUMMARY_BYTES = 8 * 1024**2
MAX_SIDECAR_BYTES = 2 * 1024**2
MAX_SIDECAR_TOTAL = 16 * 1024**2
MAX_SIDECARS = 64
MAX_ARRAY_BYTES = 64 * 1024**2
MAX_EXPANDED_ARRAY_BYTES = 128 * 1024**2
MAX_IMAGE_BYTES = 8 * 1024**2
MAX_PREVIEW_BYTES = 96 * 1024**2
_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


@dataclass(frozen=True)
class ResultPreview:
    """A published immutable preview, never a complete or native result."""

    root: Path
    output_dataset_ids: tuple[tuple[str, str], ...]


class _Unavailable(Exception):
    """An old, oversized or incomplete publication can use the full loader."""


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("Invalid Galaxy preview identifier")
    return value


def _filename(value: Any) -> str:
    if (not isinstance(value, str) or not value or len(value) > 240
            or value in {".", ".."} or any(c in value for c in "/\\:")
            or any(ord(c) < 32 for c in value)):
        raise ValueError("Unsafe Galaxy preview filename")
    return value


def _ready_dataset(metadata: Mapping[str, Any], identifier: str, history: str) -> int:
    if (metadata.get("id") != identifier or metadata.get("history_id") != history):
        raise ValueError("Galaxy preview dataset identity changed")
    if metadata.get("deleted") or metadata.get("purged") or metadata.get("state") != "ok":
        raise _Unavailable("Galaxy preview dataset is not ready")
    size = metadata.get("file_size")
    if type(size) is not int or size < 0:
        raise _Unavailable("Galaxy preview dataset has no bounded size")
    return size


class _Transfers:
    def __init__(self, service: Any, history: str):
        self.service, self.history = service, history
        self.bytes = 0
        self.lock = threading.Lock()
        self.files: dict[str, dict[str, Any]] = {}

    def download(self, identifier: str, target: Path, maximum: int,
                 metadata: Mapping[str, Any] | None = None) -> None:
        identifier = _identifier(identifier)
        if metadata is None or metadata.get("history_id") is None or metadata.get("file_size") is None:
            with transfer_slot(self.service._result_download_slots):
                current = self.service._dataset_metadata(identifier)
            if metadata is not None and current.get("id") != metadata.get("id"):
                raise ValueError("Galaxy preview collection member identity changed")
            metadata = current
        size = _ready_dataset(metadata, identifier, self.history)
        with self.lock:
            if size > maximum or self.bytes + size > MAX_PREVIEW_BYTES:
                raise _Unavailable("Galaxy preview exceeds its transfer budget")
            self.bytes += size
        with transfer_slot(self.service._result_download_slots):
            self.service._download_dataset(identifier, target, maximum_bytes=maximum, metadata=metadata)
        if target.stat().st_size != size:
            raise ValueError("Galaxy preview dataset size changed during transfer")
        with self.lock:
            self.files[str(target)] = dict(dataset_id=identifier, bytes=size,
                sha256=hashlib.sha256(target.read_bytes()).hexdigest())


def _source(service: Any, record: Any) -> tuple[str, str, dict[str, str]] | None:
    if record.status != RunStatus.OK:
        return None
    job_id = _identifier(record.galaxy_job_id or record.uid)
    history = _identifier(record.history_id)
    if str(getattr(service, "history_id", "")) != history:
        raise ValueError("Galaxy preview belongs to a different History")
    with transfer_slot(service._result_download_slots):
        job = service._job_details(job_id)
    if job.get("id") != job_id or job.get("history_id") != history:
        raise ValueError("Galaxy preview job identity changed")
    if job.get("tool_id") != tool_id("analyze"):
        raise ValueError("Galaxy preview job is not an Analyze result")
    if job.get("state") != "ok":
        return None
    output_ids = dict(record.output_dataset_ids)

    def merge(values: Mapping[str, Any]) -> None:
        for name, identifier in values.items():
            identifier = _identifier(identifier)
            if name in output_ids and output_ids[name] != identifier:
                raise ValueError("Galaxy preview output identity changed")
            output_ids[name] = identifier

    authoritative = dict(service._job_output_ids(job))
    merge(authoritative)
    expected = {"summary", "plots"} | ({"pattern_observation"} if output_ids.get("pattern_observation") else set())
    if not expected.issubset(authoritative):
        if not callable(getattr(service, "job_output_ids", None)):
            return None
        with transfer_slot(service._result_download_slots):
            resolved = service.job_output_ids(job_id)
        merge(resolved)
        authoritative.update(resolved)
    if not {"summary", "plots"}.issubset(authoritative):
        return None
    if "pattern_observation" not in authoritative:
        output_ids.pop("pattern_observation", None)
    return job_id, history, output_ids


def _plot_members(service: Any, identifier: str, history: str) -> dict[str, dict[str, Any]]:
    with transfer_slot(service._result_download_slots), measure("collection_metadata"):
        response = requests.get(f"{service.galaxy_url}/api/dataset_collections/{_identifier(identifier)}",
            headers=service._headers, params={"view": "element"}, timeout=30,
            allow_redirects=False, stream=True)
        try:
            response.raise_for_status()
            raw = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                raw.extend(chunk)
                if len(raw) > MAX_SUMMARY_BYTES:
                    raise _Unavailable("Galaxy preview collection metadata exceeds its limit")
            collection = read_json(bytes(raw))
        finally:
            response.close()
    if not isinstance(collection, dict) or collection.get("id") != identifier:
        raise ValueError("Galaxy preview collection identity changed")
    elements = collection.get("elements")
    if (collection.get("collection_type") != "list" or not isinstance(elements, list)
            or len(elements) > 2048 or collection.get("element_count") != len(elements)
            or collection.get("populated_state") != "ok"):
        raise _Unavailable("Galaxy preview collection is not completely populated")
    members: dict[str, dict[str, Any]] = {}
    names, identifiers = set(), set()
    for element in elements:
        if not isinstance(element, dict) or element.get("element_type") != "hda":
            raise _Unavailable("Galaxy preview needs a flat dataset collection")
        metadata = element.get("object")
        if not isinstance(metadata, dict):
            raise ValueError("Missing Galaxy preview collection member identity")
        dataset_id = _identifier(metadata.get("id"))
        if metadata.get("history_id") not in (None, history):
            raise ValueError("Galaxy preview collection member belongs to another History")
        if metadata.get("deleted") or metadata.get("purged") or metadata.get("state") != "ok":
            raise _Unavailable("Galaxy preview collection member is not ready")
        name = _filename(element.get("element_identifier"))
        extension = metadata.get("file_ext") or metadata.get("extension")
        if not isinstance(extension, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", extension):
            raise ValueError("Unsafe Galaxy preview collection extension")
        filename = name if name.lower().endswith("." + extension.lower()) else name + "." + extension
        filename = _filename(filename)
        if filename.casefold() in names or dataset_id in identifiers:
            raise ValueError("Ambiguous Galaxy preview collection member")
        names.add(filename.casefold());identifiers.add(dataset_id)
        members[filename] = dict(metadata)
    return members


def _bounded_arrays(path: Path) -> None:
    """Check expansion and NPY headers before the existing plot loader allocates."""
    total = 0
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > 128 or len({e.filename for e in entries}) != len(entries):
            raise _Unavailable("Preview numeric archive has too many or duplicate arrays")
        for entry in entries:
            if (not re.fullmatch(r"[A-Za-z0-9_-]+\.npy", entry.filename)
                    or stat.S_ISLNK(entry.external_attr >> 16)
                    or entry.file_size > MAX_EXPANDED_ARRAY_BYTES):
                raise _Unavailable("Unsupported preview numeric array")
            total += entry.file_size
            if total > MAX_EXPANDED_ARRAY_BYTES:
                raise _Unavailable("Preview numeric arrays exceed their expansion limit")
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise _Unavailable("Unsupported preview numeric array format")
                expected = math.prod(shape) * dtype.itemsize
                if (dtype.hasobject or dtype.fields or dtype.kind not in "biufcUS" or len(shape) > 4
                        or expected > MAX_EXPANDED_ARRAY_BYTES or expected != entry.file_size - stream.tell()):
                    raise _Unavailable("Unsafe preview numeric array shape")


def collect_result_preview(service: Any, record: Any) -> ResultPreview | None:
    """Read a small same-job fit without publishing a canonical/native cache.

    Unsupported or oversized publications fall back to the complete loader.
    Identity, ambiguous-member and unsafe-path failures remain explicit errors.
    """
    if (getattr(service, "_result_download_slots", None) is None
            or any(not callable(getattr(service, name, None)) for name in (
                "_job_details", "_job_output_ids", "_dataset_metadata",
                "_download_dataset", "_ensure_plot_observation"))):
        return None
    source = _source(service, record)
    if source is None:
        return None
    job_id, history, output_ids = source
    root = Path(service.output_root) / "result-previews" / job_id / uuid.uuid4().hex
    root.mkdir(parents=True)
    transfers = _Transfers(service, history)
    keep = False
    try:
        reads = [("summary", MAX_SUMMARY_BYTES)]
        if output_ids.get("pattern_observation"):
            reads.append(("pattern_observation", MAX_OBSERVATION_BYTES))
        def download(name, maximum):
            with subject(name):
                transfers.download(output_ids[name], root / (name + ".json"), maximum)
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="radar-preview") as pool:
            futures = [submit_context(pool, download, name, maximum)
                       for name, maximum in reads]
            for future in futures:
                future.result()
        with measure("preview_parse", "summary"):
            summary = read_json((root / "summary.json").read_bytes())
        if not isinstance(summary, dict) or summary.get("$schema") != "radar-pd-result/v1":
            raise _Unavailable("This result has no supported preview summary")
        members = _plot_members(service, output_ids["plots"], history)
        sidecars = [(name, metadata) for name, metadata in members.items() if name.endswith(".plotdata.json")]
        if not sidecars or len(sidecars) > MAX_SIDECARS:
            raise _Unavailable("This result has no bounded set of interactive plot metadata")
        if sum(metadata.get("file_size", MAX_SIDECAR_BYTES) for _, metadata in sidecars
               if type(metadata.get("file_size", MAX_SIDECAR_BYTES)) is int) > MAX_SIDECAR_TOTAL:
            raise _Unavailable("Preview plot metadata exceeds its aggregate limit")
        plots = root / "plots"
        plots.mkdir()
        def download_sidecar(name, metadata):
            with subject("plot_metadata"):
                transfers.download(metadata["id"], plots / name, MAX_SIDECAR_BYTES, metadata)
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="radar-preview-plots") as pool:
            futures = [submit_context(pool, download_sidecar, name, metadata)
                       for name, metadata in sidecars]
            for future in futures:
                future.result()
        candidates = []
        payloads = {}
        for name, _ in sidecars:
            path = plots / name
            try:
                with measure("preview_parse", "plot_metadata"):
                    payload = read_json(path.read_bytes())
            except ValueError:
                continue
            if not isinstance(payload, dict) or not payload:
                continue
            arrays = payload.get("arrays_npz")
            if arrays is not None:
                arrays = _filename(arrays)
                if not arrays.endswith(".npz"):
                    raise ValueError("Unsafe preview numeric companion")
            image = payload.get("source_plot")
            if image is not None:
                _filename(image)
            else:
                image = name.removesuffix(".plotdata.json")
            if not (arrays in members or _plot_payload_has_arrays(payload) or image in members):
                continue
            _, stage, rank = _plot_stage(path, payload)
            candidates.append(dict(path=str(path), stage=stage, rank=rank, rwp=_as_float(payload.get("rwp"))))
            payloads[str(path)] = (payload, arrays, image)
        mode = str(summary.get("analysis_mode") or record.mode.value)
        if not candidates:
            raise _Unavailable("No published fit can be previewed")
        order, remaining = [], list(candidates)
        while remaining:
            selected = _primary_plot_index(remaining, summary, mode)
            order.append(remaining.pop(selected))
        private = record.model_copy(deep=True, update={"output_dataset_ids": output_ids})
        service._ensure_plot_observation(private, root, allow_download=False)
        chosen = None
        for candidate in order:
            payload, arrays, image = payloads[candidate["path"]]
            if arrays in members:
                target = plots / arrays
                try:
                    with subject("selected_plot_arrays"):
                        transfers.download(members[arrays]["id"], target, MAX_ARRAY_BYTES, members[arrays])
                except _Unavailable:
                    target.unlink(missing_ok=True)
                else:
                    try:
                        with measure("preview_parse", "selected_plot_arrays"):
                            _bounded_arrays(target)
                    except (_Unavailable, OSError, ValueError, zipfile.BadZipFile):
                        target.unlink(missing_ok=True)
            # Let the existing renderer decide whether the arrays are useful.
            with measure("preview_plot"):
                rendered = load_plot_with_fallback([candidate], candidate["path"])
            if rendered is not None:
                chosen = candidate
                break
            if image in members and Path(image).suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
                with subject("selected_plot_image"):
                    transfers.download(members[image]["id"], plots / image, MAX_IMAGE_BYTES, members[image])
                with measure("preview_plot"):
                    rendered = load_plot_with_fallback([candidate], candidate["path"])
                if rendered is not None:
                    chosen = candidate
                    break
        if chosen is None:
            raise _Unavailable("No published fit can be rendered")
        # The renderer supports legacy suffix-based companion discovery. Keep
        # only this exact plot's files so unselected metadata cannot borrow its
        # companion and appear to be another complete fit.
        _, arrays, image = payloads[chosen["path"]]
        retained = {Path(chosen["path"]).name, arrays, image}
        for path in plots.iterdir():
            if path.name not in retained:
                path.unlink()
        receipt = dict(schema="radar-pd-result-preview/v1", job_id=job_id, history_id=history,
            output_dataset_ids=output_ids, selected_plot=Path(chosen["path"]).relative_to(root).as_posix(),
            files={Path(path).relative_to(root).as_posix(): row for path, row in transfers.files.items()
                   if Path(path).is_file()})
        (root / PREVIEW_RECEIPT).write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
        keep = True
        return ResultPreview(root=root, output_dataset_ids=tuple(sorted(output_ids.items())))
    except _Unavailable:
        return None
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)
