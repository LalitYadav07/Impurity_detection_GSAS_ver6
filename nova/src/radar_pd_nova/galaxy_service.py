"""Galaxy-backed execution and recovery for the RADAR-PD NOVA client."""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import requests

from .configuration import dump_configuration
from .models import AnalysisConfig, AnalysisMode, InputSelection, InputSource, RunRecord, RunStatus

ANALYZE_TOOL_ID = os.getenv("RADAR_PD_ANALYZE_TOOL_ID", "neutrons_radar_pd_analyze_prototype")
RUN_NAME_PREFIX = "RADAR-PD NOVA"


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
        output_root: str | Path = "/tmp/radar-pd-nova",
    ) -> None:
        self.galaxy_url = (galaxy_url or os.getenv("GALAXY_URL", "")).rstrip("/")
        self.galaxy_key = galaxy_key or os.getenv("GALAXY_API_KEY", "")
        self.history_id = history_id or os.getenv("HISTORY_ID", "")
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._tools: dict[str, Any] = {}
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
        from nova.galaxy import Dataset

        dataset = Dataset(name=name, force_upload=False)
        dataset.id = dataset_id
        dataset.store = store
        return dataset

    @staticmethod
    def _upload_dataset(path: str, store: Any, label: str) -> Any:
        from nova.galaxy import Dataset

        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"{label} does not exist: {source}")
        dataset = Dataset(path=str(source), name=f"RADAR-PD {label} | {source.name}")
        dataset.upload(store)
        return dataset

    def list_history_datasets(self, *, limit: int = 500) -> list[dict[str, str]]:
        """Return selectable, non-deleted datasets from the active history."""

        response = requests.get(
            f"{self.galaxy_url}/api/histories/{self.history_id}/contents",
            headers=self._headers,
            params={"v": "dev", "limit": limit, "order": "update_time-dsc"},
            timeout=30,
        )
        response.raise_for_status()
        rows = response.json()
        return [
            {
                "id": str(row.get("id")),
                "name": str(row.get("name") or row.get("hid") or "dataset"),
                "extension": str(row.get("extension") or row.get("file_ext") or "data"),
                "state": str(row.get("state") or ""),
            }
            for row in rows
            if row.get("history_content_type", "dataset") == "dataset"
            and not row.get("deleted")
            and str(row.get("state") or "ok") in {"ok", "deferred"}
        ]

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

    def submit(self, config: AnalysisConfig, inputs: InputSelection) -> RunRecord:
        """Upload inputs and asynchronously start the existing Analyze tool."""

        from nova.galaxy import Parameters, Tool

        if inputs.use_builtin_cuka and (config.radiation.value != "xray" or config.instrument_mode == "tof"):
            raise ValueError("The built-in Cu K-alpha profile is only valid for X-ray CW data")
        work_dir = self.output_root / "submissions" / config.run_name
        work_dir.mkdir(parents=True, exist_ok=True)
        config_path = dump_configuration(config, work_dir / "radar_pd_config.yaml")

        with self._store() as store:
            params = Parameters()
            uploaded: dict[str, str] = {}
            config_ds = self._upload_dataset(str(config_path), store, "configuration")
            uploaded["configuration"] = config_ds.id
            params.add_input(name="configuration|config_kind", value="existing")
            params.add_input(name="configuration|config_file", value=config_ds)

            if inputs.source in {InputSource.UPLOAD, InputSource.GALAXY}:
                params.add_input(name="input_source|source_kind", value="history")
                data = self._dataset_for_input(
                    path=inputs.data_path,
                    dataset_id=inputs.data_dataset_id,
                    store=store,
                    label="diffraction data",
                )
                if data is None:
                    raise ValueError("A diffraction pattern is required")
                uploaded["data"] = data.id
                params.add_input(name="input_source|data", value=data)
                if inputs.use_builtin_cuka:
                    params.add_input(name="input_source|instrument_source|instrument_kind", value="builtin_cuka")
                else:
                    instrument = self._dataset_for_input(
                        path=inputs.instrument_path,
                        dataset_id=inputs.instrument_dataset_id,
                        store=store,
                        label="instrument profile",
                    )
                    if instrument is None:
                        raise ValueError("An instrument profile is required")
                    uploaded["instrument"] = instrument.id
                    params.add_input(name="input_source|instrument_source|instrument_kind", value="upload")
                    params.add_input(name="input_source|instrument_source|instrument", value=instrument)
            else:
                params.add_input(name="input_source|source_kind", value="ipts")
                if inputs.source == InputSource.IPTS_EVENT:
                    params.add_input(name="input_source|ipts_lookup|lookup_kind", value="event")
                    event = self._dataset_for_input(
                        path=inputs.event_file_path,
                        dataset_id=inputs.event_dataset_id,
                        store=store,
                        label="NeXus event file",
                    )
                    if event is None:
                        raise ValueError("A NeXus event file is required")
                    uploaded["event_file"] = event.id
                    params.add_input(name="input_source|ipts_lookup|event_file", value=event)
                    params.add_input(name="input_source|ipts_lookup|bank", value=inputs.bank)
                else:
                    params.add_input(name="input_source|ipts_lookup|lookup_kind", value="manual")
                    params.add_input(name="input_source|ipts_lookup|instrument", value=inputs.instrument)
                    params.add_input(name="input_source|ipts_lookup|ipts", value=inputs.ipts)
                    params.add_input(name="input_source|ipts_lookup|run_number", value=inputs.run_number)
                    params.add_input(name="input_source|ipts_lookup|bank", value=inputs.bank)
                params.add_input(name="input_source|facility_root", value=inputs.facility_root)

            main_cif = self._dataset_for_input(
                path=inputs.main_cif_path,
                dataset_id=inputs.main_cif_dataset_id,
                store=store,
                label="main phase CIF",
            )
            if main_cif is not None:
                uploaded["main_cif"] = main_cif.id
                params.add_input(name="main_cif", value=main_cif)

            database = self._dataset_for_input(
                path=inputs.database_archive_path,
                dataset_id=inputs.database_dataset_id,
                store=store,
                label="candidate library",
            )
            if database is None:
                params.add_input(name="database|database_kind", value="builtin")
            else:
                uploaded["database_archive"] = database.id
                params.add_input(name="database|database_kind", value="custom")
                params.add_input(name="database|database_archive", value=database)
            params.add_input(name="run_name", value=config.run_name)

            tool = Tool(id=ANALYZE_TOOL_ID)
            tool.run(store, params, wait=False)
            uid = tool.get_uid()
            if not uid:
                raise RuntimeError("Galaxy accepted the request but did not return a job identifier")
            with self._lock:
                self._tools[uid] = tool
            return RunRecord(
                uid=uid,
                name=config.run_name,
                mode=config.mode,
                history_id=self.history_id,
                status=RunStatus.QUEUED,
                stage="Waiting for compute",
                progress=3,
                input_dataset_ids=uploaded,
            )

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
        tool = self._recover_tool(record.uid)
        status = normalize_status(tool.get_status())
        stdout = tool.get_stdout() or ""
        stderr = tool.get_stderr() or ""
        stage, progress = stage_from_console(stdout, status)
        record.status = status
        record.stage = stage
        record.progress = progress
        record.updated_utc = datetime.now(timezone.utc).isoformat()
        if status == RunStatus.ERROR:
            record.message = (stderr or stdout or "Galaxy reported an error")[-4000:]
        return record

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
        for job in response.json():
            uid = str(job.get("id") or "")
            if not uid:
                continue
            state = normalize_status(job.get("state"))
            name = str(job.get("job_metrics", {}).get("run_name") or job.get("command_line") or uid)
            match = re.search(r"--run-name\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))", name)
            run_name = next((group for group in match.groups() if group), uid) if match else uid
            records.append(
                RunRecord(
                    uid=uid,
                    name=run_name,
                    mode=AnalysisMode.RAPID if "rapid" in name.lower() else AnalysisMode.FULL,
                    history_id=self.history_id,
                    status=state,
                    stage=stage_from_console("", state)[0],
                    progress=stage_from_console("", state)[1],
                )
            )
        return records

    def collect_results(self, record: RunRecord) -> RunRecord:
        """Download stable scalar outputs and collections when the job completes."""

        tool = self._recover_tool(record.uid)
        outputs = tool.get_results()
        if outputs is None:
            return record
        destination = self.output_root / "runs" / record.uid
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        output_ids: dict[str, str] = {}
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
            try:
                dataset = outputs.get_dataset(name)
            except Exception:
                continue
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
            target = destination / f"{name}{suffix}"
            dataset.download(str(target))
            output_ids[name] = str(getattr(dataset, "id", ""))
        for name in ("plots", "tables", "phases", "gpx_projects", "diagnostics"):
            try:
                collection = outputs.get_collection(name)
            except Exception:
                continue
            target = destination / name
            target.mkdir(exist_ok=True)
            collection.download(str(target))
            output_ids[name] = str(getattr(collection, "id", ""))
        record.output_dataset_ids = output_ids
        record.output_dir = str(destination)
        record.status = RunStatus.OK
        record.stage = "Results ready"
        record.progress = 100
        return record

    def result_payload(self, record: RunRecord) -> dict[str, Any]:
        root = record.local_output_dir
        if root is None:
            return {}
        payload: dict[str, Any] = {"artifacts": []}
        for key in ("summary", "state", "gpx_index", "input_manifest"):
            path = root / f"{key}.json"
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
                record.message = f"Analysis finished, but result download failed: {exc}"
            callback(record)
