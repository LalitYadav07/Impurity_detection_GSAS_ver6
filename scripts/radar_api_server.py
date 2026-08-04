#!/usr/bin/env python3
"""
Small HTTP API for submitting RADAR-PD jobs from scripts.

This server intentionally uses only the Python standard library plus PyYAML,
which RADAR-PD already requires. It accepts a frontend-generated
``pipeline_config.yaml`` plus uploaded input files, rewrites the config to an
ephemeral API job folder, runs either the rapid or full pipeline, and exposes
status, intermediate artifacts, and a result ZIP.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import posixpath
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default as email_default_policy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import yaml

try:
    from xrdml_converter import prepare_powder_data_file
except Exception:  # pragma: no cover - keeps API importable during partial installs
    def prepare_powder_data_file(data_path):
        return Path(data_path)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JOBS_ROOT = PROJECT_ROOT / "api_runs"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8503
DEFAULT_TTL_HOURS = 24.0
MAX_UPLOAD_BYTES = int(os.environ.get("RADAR_PD_API_MAX_UPLOAD_MB", "800")) * 1024 * 1024

JOBS_ROOT = Path(os.environ.get("RADAR_PD_API_JOBS_ROOT", DEFAULT_JOBS_ROOT)).resolve()
API_KEY = os.environ.get("RADAR_PD_API_KEY", "").strip()
STATUS_LOCK = threading.RLock()


@dataclass
class FormPart:
    name: str
    filename: str | None
    data: bytes
    content_type: str | None = None


def _now() -> float:
    return time.time()


def _iso(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or _now()))


def _safe_name(name: str, fallback: str = "file") -> str:
    name = Path(str(name or fallback)).name
    name = re.sub(r"[^A-Za-z0-9._+-]+", "_", name).strip("._")
    return name or fallback


def _new_job_id(prefix: str = "radar") -> str:
    import secrets

    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{secrets.token_hex(4)}"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(default or {})


def _job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id or ""):
        raise ValueError("Invalid job id")
    return JOBS_ROOT / job_id


def _status_path(job_dir: Path) -> Path:
    return job_dir / "status.json"


def _load_status(job_dir: Path) -> dict[str, Any]:
    return _read_json(_status_path(job_dir), {})


def _update_status(job_dir: Path, **updates: Any) -> dict[str, Any]:
    with STATUS_LOCK:
        status = _load_status(job_dir)
        status.update(updates)
        status["updated_at"] = _iso()
        _write_json(_status_path(job_dir), status)
        return status


def _parse_multipart_form(content_type: str, body: bytes) -> dict[str, list[FormPart]]:
    if "multipart/form-data" not in content_type:
        raise ValueError("Use multipart/form-data")
    message_bytes = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
        + body
    )
    message = BytesParser(policy=email_default_policy).parsebytes(message_bytes)
    if not message.is_multipart():
        raise ValueError("Request body is not multipart")

    form: dict[str, list[FormPart]] = {}
    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        if not disposition:
            continue
        params = dict(part.get_params(header="content-disposition") or [])
        name = params.get("name")
        if not name:
            continue
        filename = params.get("filename")
        payload = part.get_payload(decode=True) or b""
        form.setdefault(str(name), []).append(
            FormPart(
                name=str(name),
                filename=str(filename) if filename else None,
                data=payload,
                content_type=part.get_content_type(),
            )
        )
    return form


def _field_items(form: dict[str, list[FormPart]], name: str) -> list[FormPart]:
    return form.get(name, [])


def _text_field(form: dict[str, list[FormPart]], name: str, default: str = "") -> str:
    items = _field_items(form, name)
    if not items:
        return default
    item = items[0]
    if item.filename:
        return default
    return item.data.decode("utf-8", errors="replace")


def _first_file_field(form: dict[str, list[FormPart]], *names: str) -> FormPart | None:
    for name in names:
        for item in _field_items(form, name):
            if item.filename:
                return item
    return None


def _save_upload(item: FormPart, dest_dir: Path, fallback_name: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = _safe_name(item.filename or fallback_name, fallback_name)
    dest = dest_dir / filename
    with dest.open("wb") as handle:
        handle.write(item.data)
    if dest.stat().st_size <= 0:
        raise ValueError(f"Uploaded file is empty: {filename}")
    return dest


def _select_dataset(cfg: dict[str, Any], requested: str | None) -> dict[str, Any]:
    datasets = cfg.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Config must contain at least one dataset entry")
    if requested:
        for dataset in datasets:
            if str(dataset.get("name")) == str(requested):
                return dataset
        raise ValueError(f"Dataset not found in config: {requested}")
    dataset = datasets[0]
    if not isinstance(dataset, dict):
        raise ValueError("First dataset entry is not a mapping")
    return dataset


def _path_exists(value: Any) -> bool:
    try:
        return bool(value) and Path(str(value)).exists()
    except Exception:
        return False


def _mode_from_config(cfg: dict[str, Any], override: str = "auto") -> str:
    override = str(override or "auto").strip().lower()
    if override in {"rapid", "rapid_hypothesis", "rapid-hypothesis"}:
        return "rapid"
    if override in {"full", "full_radar_pd", "radar", "radar-pd"}:
        return "full"
    rapid_cfg = cfg.get("rapid_hypothesis") or {}
    if cfg.get("analysis_mode") == "rapid_hypothesis" or rapid_cfg.get("enabled"):
        return "rapid"
    return "full"


def _rewrite_config_for_job(
    cfg: dict[str, Any],
    *,
    job_dir: Path,
    run_dir: Path,
    dataset_name: str | None,
    run_name: str | None,
    data_path: Path | None,
    instprm_path: Path | None,
    main_cif_path: Path | None,
    mode: str,
) -> tuple[dict[str, Any], str]:
    cfg = dict(cfg)
    dataset = _select_dataset(cfg, dataset_name)
    dataset = dict(dataset)

    final_name = _safe_name(run_name or dataset.get("name") or job_dir.name, job_dir.name)
    dataset["name"] = final_name

    if data_path is not None:
        dataset["data_path"] = str(data_path.resolve())
    if instprm_path is not None:
        dataset["instprm_path"] = str(instprm_path.resolve())
    if main_cif_path is not None:
        dataset["main_cif"] = str(main_cif_path.resolve())

    if not _path_exists(dataset.get("data_path")):
        raise ValueError("No usable diffraction data file was provided or found in the config")
    if not _path_exists(dataset.get("instprm_path")):
        raise ValueError("No usable instrument parameter file was provided or found in the config")
    if dataset.get("main_cif") and not _path_exists(dataset.get("main_cif")):
        raise ValueError("Config references a main_cif path that does not exist; upload main_cif or remove it")

    cfg["PROJECT_ROOT"] = str(PROJECT_ROOT)
    cfg["DATA_ROOT"] = str(PROJECT_ROOT / "data")
    cfg["WORK_ROOT"] = str(job_dir)
    cfg["work_root"] = str(run_dir)
    cfg["ml_components_dir"] = str(PROJECT_ROOT / "ML_components")
    cfg["api_job"] = {
        "job_id": job_dir.name,
        "created_at": _iso(),
        "ephemeral": True,
    }

    if mode == "rapid":
        cfg["analysis_mode"] = "rapid_hypothesis"
        rapid_cfg = dict(cfg.get("rapid_hypothesis") or {})
        rapid_cfg["enabled"] = True
        cfg["rapid_hypothesis"] = rapid_cfg
    elif mode == "full":
        if cfg.get("analysis_mode") == "rapid_hypothesis":
            cfg["analysis_mode"] = "full_radar_pd"
        rapid_cfg = dict(cfg.get("rapid_hypothesis") or {})
        rapid_cfg["enabled"] = False
        cfg["rapid_hypothesis"] = rapid_cfg

    datasets = cfg.get("datasets") or []
    original_name = dataset_name or str((datasets[0] or {}).get("name"))
    replaced = False
    new_datasets = []
    for idx, item in enumerate(datasets):
        should_replace = str(item.get("name")) == str(original_name) if dataset_name else idx == 0
        if not replaced and should_replace:
            new_datasets.append(dataset)
            replaced = True
        else:
            new_datasets.append(item)
    cfg["datasets"] = new_datasets
    return cfg, final_name


def _pipeline_command(config_path: Path, dataset: str, mode: str) -> list[str]:
    script = "rapid_hypothesis_pipeline.py" if mode == "rapid" else "gsas_complete_pipeline_nomain.py"
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / script),
        "--config",
        str(config_path),
        "--dataset",
        dataset,
    ]


def _run_job(job_dir: Path, config_path: Path, dataset: str, mode: str) -> None:
    run_dir = job_dir / "run"
    log_path = run_dir / "pipeline.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = _pipeline_command(config_path, dataset, mode)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    scripts_path = str((PROJECT_ROOT / "scripts").resolve())
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [scripts_path, env.get("PYTHONPATH", "")]))

    _update_status(
        job_dir,
        state="running",
        started_at=_iso(),
        command=cmd,
        run_dir=str(run_dir),
        log_path=str(log_path),
    )
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            _update_status(job_dir, pid=process.pid)
            assert process.stdout is not None
            last_status_write = 0.0
            for line in process.stdout:
                log.write(line)
                log.flush()
                now = _now()
                if line.strip() and (now - last_status_write) >= 2.0:
                    last_status_write = now
                    _update_status(job_dir, last_log_line=line.rstrip()[-500:])
            returncode = process.wait()

        state = "complete" if returncode == 0 else "failed"
        _update_status(
            job_dir,
            state=state,
            returncode=returncode,
            finished_at=_iso(),
            message="Pipeline completed" if returncode == 0 else f"Pipeline failed with exit code {returncode}",
        )
    except Exception as exc:
        trace_path = job_dir / "api_exception.txt"
        trace_path.write_text(traceback.format_exc(), encoding="utf-8")
        _update_status(
            job_dir,
            state="failed",
            returncode=None,
            finished_at=_iso(),
            error=str(exc),
            message=f"API runner failed: {exc}",
        )


def _latest_event(run_dir: Path) -> dict[str, Any] | None:
    candidates = [
        run_dir / "Technical" / "Logs" / "run_events.jsonl",
        run_dir / "pipeline_events.jsonl",
    ]
    for path in candidates:
        if not path.exists():
            continue
        last = None
        try:
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        last = json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue
        if last:
            last["event_file"] = str(path)
            return last
    return None


def _tail_text(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max(1, min(int(max_lines), 500)):])
    except Exception:
        return ""


def _artifact_rows(run_dir: Path, limit: int = 500) -> list[dict[str, Any]]:
    if not run_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    hidden = {"pipeline.log"}
    for path in run_dir.rglob("*"):
        if not path.is_file() or path.name in hidden:
            continue
        try:
            rel = path.relative_to(run_dir).as_posix()
            rows.append(
                {
                    "path": rel,
                    "name": path.name,
                    "size": path.stat().st_size,
                    "modified_at": _iso(path.stat().st_mtime),
                }
            )
        except Exception:
            continue
    rows.sort(key=lambda item: item["modified_at"], reverse=True)
    return rows[:limit]


def _safe_artifact_path(run_dir: Path, rel_path: str) -> Path:
    rel_path = unquote(str(rel_path or "")).replace("\\", "/")
    rel_path = posixpath.normpath(rel_path).lstrip("/")
    if rel_path.startswith("../") or rel_path == "..":
        raise ValueError("Invalid artifact path")
    path = (run_dir / rel_path).resolve()
    run_resolved = run_dir.resolve()
    if run_resolved not in path.parents and path != run_resolved:
        raise ValueError("Artifact path escapes run directory")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(rel_path)
    return path


def _query_int(
    query: dict[str, list[str]],
    key: str,
    default: int,
    *,
    min_value: int = 1,
    max_value: int | None = None,
) -> int:
    raw = query.get(key, [str(default)])[0]
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = int(default)
    value = max(int(min_value), value)
    if max_value is not None:
        value = min(int(max_value), value)
    return value


def _zip_results(job_dir: Path) -> Path:
    run_dir = job_dir / "run"
    zip_path = job_dir / "results.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for root in [run_dir, job_dir / "inputs"]:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(job_dir).as_posix())
        for path in [job_dir / "pipeline_config.resolved.yaml", job_dir / "pipeline_config.original.yaml", job_dir / "status.json"]:
            if path.exists():
                archive.write(path, path.name)
    return zip_path


def _cleanup_stale_jobs(ttl_hours: float) -> None:
    if ttl_hours <= 0 or not JOBS_ROOT.exists():
        return
    cutoff = _now() - ttl_hours * 3600.0
    for child in JOBS_ROOT.iterdir():
        if not child.is_dir():
            continue
        try:
            status = _load_status(child)
            state = status.get("state")
            updated = child.stat().st_mtime
            if status.get("updated_at"):
                try:
                    updated = time.mktime(time.strptime(str(status["updated_at"]).replace("Z", ""), "%Y-%m-%dT%H:%M:%S"))
                except Exception:
                    pass
            if state in {"complete", "failed", "deleted"} and updated < cutoff:
                shutil.rmtree(child, ignore_errors=True)
        except Exception:
            continue


class RadarAPIHandler(BaseHTTPRequestHandler):
    server_version = "RADARPDAPI/0.1"

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, indent=2, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _send_error_json(self, status: int, message: str, **extra: Any) -> None:
        payload = {"error": message, **extra}
        self._send_json(payload, status=status)

    def _check_auth(self) -> bool:
        if not API_KEY:
            return True
        supplied = self.headers.get("X-API-Key", "")
        if supplied == API_KEY:
            return True
        self._send_error_json(HTTPStatus.UNAUTHORIZED, "Missing or invalid X-API-Key")
        return False

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (_iso(), fmt % args))

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        if not self._check_auth():
            return
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]
        query = parse_qs(parsed.query)

        try:
            if parsed.path in {"/", "/health", "/api/v1/health"}:
                self._send_json(
                    {
                        "status": "ok",
                        "service": "radar-pd-api",
                        "project_root": str(PROJECT_ROOT),
                        "jobs_root": str(JOBS_ROOT),
                        "time": _iso(),
                    }
                )
                return

            if len(parts) >= 4 and parts[:3] == ["api", "v1", "jobs"]:
                job_id = parts[3]
                job_dir = _job_dir(job_id)
                if not job_dir.exists():
                    self._send_error_json(HTTPStatus.NOT_FOUND, "Job not found", job_id=job_id)
                    return

                if len(parts) == 4:
                    self._send_json(self._job_status_payload(job_dir, query))
                    return
                if len(parts) == 5 and parts[4] == "log":
                    status = _load_status(job_dir)
                    log_path = Path(status.get("log_path") or job_dir / "run" / "pipeline.log")
                    tail = _query_int(query, "tail", 120, min_value=1, max_value=500)
                    self._send_json({"job_id": job_id, "log_tail": _tail_text(log_path, tail)})
                    return
                if len(parts) == 5 and parts[4] == "artifacts":
                    run_dir = job_dir / "run"
                    limit = _query_int(query, "limit", 500, min_value=1, max_value=10000)
                    self._send_json({"job_id": job_id, "artifacts": _artifact_rows(run_dir, limit=limit)})
                    return
                if len(parts) == 5 and parts[4] == "artifact":
                    rel = query.get("path", [""])[0]
                    self._send_file(_safe_artifact_path(job_dir / "run", rel), download_name=Path(rel).name)
                    return
                if len(parts) == 5 and parts[4] == "results.zip":
                    zip_path = _zip_results(job_dir)
                    cleanup = query.get("cleanup", ["1"])[0].lower() not in {"0", "false", "no"}
                    self._send_file(zip_path, download_name=f"{job_id}_results.zip")
                    if cleanup:
                        shutil.rmtree(job_dir, ignore_errors=True)
                    return

            self._send_error_json(HTTPStatus.NOT_FOUND, "Unknown endpoint")
        except Exception as exc:
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_DELETE(self) -> None:
        if not self._check_auth():
            return
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) == 4 and parts[:3] == ["api", "v1", "jobs"]:
            try:
                job_dir = _job_dir(parts[3])
                shutil.rmtree(job_dir, ignore_errors=True)
                self._send_json({"job_id": parts[3], "state": "deleted"})
            except Exception as exc:
                self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def do_POST(self) -> None:
        if not self._check_auth():
            return
        parsed = urlparse(self.path)
        if parsed.path != "/api/v1/jobs":
            self._send_error_json(HTTPStatus.NOT_FOUND, "Unknown endpoint")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0") or "0")
        except (TypeError, ValueError):
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
            return
        if content_length <= 0:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Empty request body")
            return
        if content_length > MAX_UPLOAD_BYTES:
            self._send_error_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Upload too large")
            return

        try:
            body = self.rfile.read(content_length)
            form = _parse_multipart_form(self.headers.get("Content-Type", ""), body)
            job_id = _safe_name(_text_field(form, "job_id") or _new_job_id(), "job")
            job_dir = _job_dir(job_id)
            if job_dir.exists():
                job_id = _new_job_id()
                job_dir = _job_dir(job_id)
            inputs_dir = job_dir / "inputs"
            run_dir = job_dir / "run"
            inputs_dir.mkdir(parents=True, exist_ok=True)
            run_dir.mkdir(parents=True, exist_ok=True)

            config_item = _first_file_field(form, "config", "pipeline_config")
            if config_item is None:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "Upload a pipeline_config.yaml file as field 'config'")
                return
            original_config_path = _save_upload(config_item, job_dir, "pipeline_config.original.yaml")
            original_config_path = original_config_path.rename(job_dir / "pipeline_config.original.yaml")
            cfg = yaml.safe_load(original_config_path.read_text(encoding="utf-8")) or {}

            data_item = _first_file_field(form, "data", "diffraction_data")
            inst_item = _first_file_field(form, "instrument", "instprm", "instrument_profile")
            cif_item = _first_file_field(form, "main_cif", "cif")
            data_path = _save_upload(data_item, inputs_dir, "data.dat") if data_item is not None else None
            if data_path is not None:
                data_path = prepare_powder_data_file(data_path)
            inst_path = _save_upload(inst_item, inputs_dir, "instrument.instprm") if inst_item is not None else None
            main_cif_path = _save_upload(cif_item, inputs_dir, "main.cif") if cif_item is not None else None

            requested_dataset = _text_field(form, "dataset", "").strip() or None
            requested_run_name = _text_field(form, "run_name", "").strip() or None
            mode = _mode_from_config(cfg, _text_field(form, "mode", "auto"))
            resolved_cfg, dataset_name = _rewrite_config_for_job(
                cfg,
                job_dir=job_dir,
                run_dir=run_dir,
                dataset_name=requested_dataset,
                run_name=requested_run_name,
                data_path=data_path,
                instprm_path=inst_path,
                main_cif_path=main_cif_path,
                mode=mode,
            )
            resolved_config_path = job_dir / "pipeline_config.resolved.yaml"
            resolved_config_path.write_text(yaml.dump(resolved_cfg, sort_keys=False), encoding="utf-8")

            status = {
                "job_id": job_id,
                "state": "queued",
                "created_at": _iso(),
                "updated_at": _iso(),
                "mode": mode,
                "dataset": dataset_name,
                "job_dir": str(job_dir),
                "run_dir": str(run_dir),
                "config_path": str(resolved_config_path),
                "results_url": f"/api/v1/jobs/{job_id}/results.zip",
                "status_url": f"/api/v1/jobs/{job_id}",
                "artifacts_url": f"/api/v1/jobs/{job_id}/artifacts",
            }
            _write_json(_status_path(job_dir), status)
            thread = threading.Thread(
                target=_run_job,
                args=(job_dir, resolved_config_path, dataset_name, mode),
                daemon=True,
            )
            thread.start()
            self._send_json(status, status=HTTPStatus.ACCEPTED)
        except Exception as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

    def _job_status_payload(self, job_dir: Path, query: dict[str, list[str]]) -> dict[str, Any]:
        status = _load_status(job_dir)
        run_dir = Path(status.get("run_dir") or job_dir / "run")
        event = _latest_event(run_dir)
        if event:
            status["progress"] = {
                "stage": event.get("stage"),
                "message": event.get("message"),
                "percent": event.get("percent"),
                "level": event.get("level", "INFO"),
                "metrics": event.get("metrics", {}),
            }
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            status["manifest"] = _read_json(manifest_path, {})
        status["artifact_count"] = len(_artifact_rows(run_dir, limit=10000))
        if "tail" in query:
            tail = _query_int(query, "tail", 80, min_value=1, max_value=500)
            status["log_tail"] = _tail_text(Path(status.get("log_path") or run_dir / "pipeline.log"), tail)
        return status

    def _send_file(self, path: Path, *, download_name: str | None = None) -> None:
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Access-Control-Allow-Origin", "*")
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{_safe_name(download_name)}"')
        self.end_headers()
        with path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)


def serve(host: str, port: int, ttl_hours: float) -> None:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_jobs(ttl_hours)
    server = ThreadingHTTPServer((host, port), RadarAPIHandler)
    print(f"RADAR-PD API listening on http://{host}:{port}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Jobs root: {JOBS_ROOT}")
    if API_KEY:
        print("API key protection: enabled")
    else:
        print("API key protection: disabled")
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("RADAR_PD_API_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("RADAR_PD_API_PORT", DEFAULT_PORT)))
    parser.add_argument("--ttl-hours", type=float, default=float(os.environ.get("RADAR_PD_API_TTL_HOURS", DEFAULT_TTL_HOURS)))
    args = parser.parse_args(argv)
    serve(args.host, args.port, args.ttl_hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
